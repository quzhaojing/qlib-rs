//! Concrete single-stream replay storage for Qlib's full-history Candle PPO.
//! Tensor copying/broadcasting/batching belongs to Candle; slot and episode
//! semantics belong to the shared replay index. Vector/stacked stores build on it.

use crate::rl_candle_network::RecurrentObservation;
use crate::rl_candle_policy::{CandlePpoReplay, CandlePpoRollout};
use crate::rl_replay_index::{ReplayEpisode, ReplayIndex, ReplayIndexError};
use candle_core::Tensor;
use ndarray::Array1;
use rand::RngCore;
use std::sync::Arc;
use thiserror::Error;

mod vector;
pub use vector::{CandleReplayBatch, CandleVectorReplayBuffer, CandleVectorReplayError};

#[derive(Clone)]
pub struct CandleReplayTransition {
    /// Each observation field has one leading batch row.
    pub observation: RecurrentObservation,
    pub next_observation: RecurrentObservation,
    /// One batched action, preserving its source dtype.
    pub action: Tensor,
    pub reward: f64,
    pub terminated: bool,
    pub truncated: bool,
}

#[derive(Debug, Error)]
pub enum CandleReplayError {
    #[error("replay observation storage is not initialized")]
    MissingStorage,
    #[error("replay insertion requires one leading batch row")]
    BatchSize,
    #[error("sample size exceeds the native signed index range")]
    SampleSize,
    #[error(transparent)]
    Index(#[from] ReplayIndexError),
    #[error(transparent)]
    Tensor(#[from] candle_core::Error),
}

fn fields(obs: &RecurrentObservation) -> [&Tensor; 7] {
    [
        &obs.data_processed,
        &obs.cur_tick,
        &obs.cur_step,
        &obs.position_history,
        &obs.target,
        &obs.num_step,
        &obs.acquiring,
    ]
}
fn map_fields(
    obs: &RecurrentObservation,
    mut f: impl FnMut(&Tensor) -> candle_core::Result<Tensor>,
) -> candle_core::Result<RecurrentObservation> {
    Ok(RecurrentObservation {
        data_processed: f(&obs.data_processed)?,
        cur_tick: f(&obs.cur_tick)?,
        cur_step: f(&obs.cur_step)?,
        position_history: f(&obs.position_history)?,
        target: f(&obs.target)?,
        num_step: f(&obs.num_step)?,
        acquiring: f(&obs.acquiring)?,
    })
}
fn assign(target: &mut Tensor, source: &Tensor) -> candle_core::Result<()> {
    if target.dtype() != source.dtype() {
        return Err(candle_core::Error::Msg(
            "replay tensor assignment requires matching dtypes".into(),
        ));
    }
    *target = source
        .broadcast_as(target.shape())?
        .to_device(target.device())?
        .detach()
        .copy()?;
    Ok(())
}
fn assign_observation(
    target: &mut RecurrentObservation,
    source: &RecurrentObservation,
) -> candle_core::Result<()> {
    assign(&mut target.data_processed, &source.data_processed)?;
    assign(&mut target.cur_tick, &source.cur_tick)?;
    assign(&mut target.cur_step, &source.cur_step)?;
    assign(&mut target.position_history, &source.position_history)?;
    assign(&mut target.target, &source.target)?;
    assign(&mut target.num_step, &source.num_step)?;
    assign(&mut target.acquiring, &source.acquiring)
}

pub(crate) fn concatenate(
    observations: &[&RecurrentObservation],
    template: &RecurrentObservation,
) -> candle_core::Result<RecurrentObservation> {
    if observations.is_empty() {
        return map_fields(template, |tensor| tensor.narrow(0, 0, 0));
    }
    let mut field = 0;
    map_fields(template, |_| {
        let tensors: Vec<_> = observations.iter().map(|obs| fields(obs)[field]).collect();
        field += 1;
        Tensor::cat(&tensors, 0)
    })
}

/// Default (`stack_num=1`, stored `obs_next`) replay for scalar rewards. Physical
/// slots retain an initialized zero schema across reset, as in Tianshou. Zero
/// templates are shared until a slot is written; writes copy only that row,
/// never the whole capacity. Stored tensors are detached owned copies.
pub struct CandleReplayBuffer {
    index: ReplayIndex,
    storage: ReplayStorage,
}
impl CandleReplayBuffer {
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            index: ReplayIndex::new(capacity),
            storage: ReplayStorage::default(),
        }
    }
    #[must_use]
    pub fn index(&self) -> &ReplayIndex {
        &self.index
    }
    pub fn reset(&mut self, keep_statistics: bool) {
        self.index.reset(keep_statistics);
    }

    /// Inserts a typed single-row record. Shape validation of the leading row
    /// precedes mutation; source-specific index/episode advancement precedes slot
    /// assignment. Dtypes must match, like Torch indexed assignment; failed
    /// assignments retain preceding writes and index state.
    /// Assignment order is observation, next observation, action, reward and flags.
    /// # Errors
    /// Returns invalid row metadata, zero capacity, index or tensor errors.
    pub fn add(
        &mut self,
        value: &CandleReplayTransition,
    ) -> Result<ReplayEpisode, CandleReplayError> {
        if fields(&value.observation)
            .into_iter()
            .chain(fields(&value.next_observation))
            .chain([&value.action])
            .any(|tensor| tensor.dims().first() != Some(&1))
        {
            return Err(CandleReplayError::BatchSize);
        }
        let result = self
            .index
            .advance(value.reward, value.terminated || value.truncated)?;
        if self.storage.initialize(
            &value.observation,
            &value.next_observation,
            &value.action,
            self.index.capacity(),
        )? {
            self.index.initialize_done();
        }
        let target = Arc::make_mut(&mut self.storage.slots[result.index]);
        assign_observation(&mut target.observation, &value.observation)?;
        assign_observation(&mut target.next_observation, &value.next_observation)?;
        assign(&mut target.action, &value.action)?;
        target.reward = value.reward;
        target.terminated = value.terminated;
        target.truncated = value.truncated;
        self.index
            .commit_done(result.index, value.terminated || value.truncated)?;
        Ok(result)
    }

    /// Batched owned read preserving requested order/duplicates and physical slot
    /// semantics. Unwritten allocated slots are zero values; an empty read after
    /// reset keeps observation/action shapes and dtype. A fresh read has no schema.
    /// # Errors
    /// Returns missing storage, invalid positions and native tensor errors.
    pub fn get(&self, indices: &[usize]) -> Result<CandlePpoRollout, CandleReplayError> {
        let mut batch = self.storage.get(indices)?;
        batch.unfinished_indices = self.index.unfinished_indices()?;
        Ok(batch)
    }

    /// # Errors
    /// Returns invalid sampling population, missing schema or tensor read errors.
    pub fn sample_batch(
        &self,
        size: i64,
        rng: &mut dyn RngCore,
    ) -> Result<CandlePpoRollout, CandleReplayError> {
        self.get(&self.index.sample_indices(size, 1, false, rng)?)
    }
}

#[derive(Default)]
struct ReplayStorage {
    template: Option<Arc<CandleReplayTransition>>,
    slots: Vec<Arc<CandleReplayTransition>>,
}

impl ReplayStorage {
    fn scalar_column<T>(
        &self,
        indices: Option<&[usize]>,
        select: impl Fn(&CandleReplayTransition) -> T,
    ) -> Result<Array1<T>, CandleReplayError> {
        if self.template.is_none() {
            return Err(CandleReplayError::MissingStorage);
        }
        if let Some(indices) = indices {
            indices
                .iter()
                .map(|&index| {
                    self.slots
                        .get(index)
                        .map(|slot| select(slot))
                        .ok_or_else(|| ReplayIndexError::Index.into())
                })
                .collect()
        } else {
            Ok(Array1::from_iter(
                self.slots.iter().map(|slot| select(slot)),
            ))
        }
    }

    fn initialize(
        &mut self,
        observation: &RecurrentObservation,
        next_observation: &RecurrentObservation,
        action: &Tensor,
        capacity: usize,
    ) -> candle_core::Result<bool> {
        if self.template.is_some() {
            return Ok(false);
        }
        let zero_row = |tensor: &Tensor| {
            let mut shape = vec![1];
            shape.extend_from_slice(tensor.dims().get(1..).unwrap_or_default());
            Tensor::zeros(shape, tensor.dtype(), tensor.device())
        };
        let zero = Arc::new(CandleReplayTransition {
            observation: map_fields(observation, zero_row)?,
            next_observation: map_fields(next_observation, zero_row)?,
            action: zero_row(action)?,
            reward: 0.,
            terminated: false,
            truncated: false,
        });
        self.slots = vec![Arc::clone(&zero); capacity];
        self.template = Some(zero);
        Ok(true)
    }

    fn get(&self, indices: &[usize]) -> Result<CandlePpoRollout, CandleReplayError> {
        let template = self
            .template
            .as_ref()
            .ok_or(CandleReplayError::MissingStorage)?;
        let selected = indices
            .iter()
            .map(|&index| self.slots.get(index).ok_or(ReplayIndexError::Index))
            .collect::<Result<Vec<_>, _>>()?;
        let observations = selected
            .iter()
            .map(|value| &value.observation)
            .collect::<Vec<_>>();
        let next = selected
            .iter()
            .map(|value| &value.next_observation)
            .collect::<Vec<_>>();
        let actions = if selected.is_empty() {
            template.action.narrow(0, 0, 0)?
        } else {
            Tensor::cat(
                &selected
                    .iter()
                    .map(|value| &value.action)
                    .collect::<Vec<_>>(),
                0,
            )?
        };
        Ok(CandlePpoRollout {
            observations: concatenate(&observations, &template.observation)?,
            next_observations: concatenate(&next, &template.next_observation)?,
            actions,
            rewards: Array1::from_iter(selected.iter().map(|value| value.reward)),
            terminated: Array1::from_iter(selected.iter().map(|value| value.terminated)),
            truncated: Array1::from_iter(selected.iter().map(|value| value.truncated)),
            bootstrap_valid: Array1::from_iter(selected.iter().map(|value| !value.terminated)),
            indices: indices.to_vec(),
            unfinished_indices: Vec::new(),
            replay_weights: None,
        })
    }
}
impl CandlePpoReplay for CandleReplayBuffer {
    fn sample(&mut self, size: u64, rng: &mut dyn RngCore) -> Result<CandlePpoRollout, String> {
        let size = i64::try_from(size).map_err(|_| CandleReplayError::SampleSize.to_string())?;
        self.sample_batch(size, rng)
            .map_err(|error| error.to_string())
    }
}

impl crate::rl_candle_nstep::CandleNStepReplay for CandleReplayBuffer {
    fn rewards(&self) -> Result<Array1<f64>, String> {
        self.storage
            .scalar_column(None, |row| row.reward)
            .map_err(|error| error.to_string())
    }
    fn next_indices(&self, indices: &[usize]) -> Result<Vec<usize>, String> {
        self.index.next(indices).map_err(|error| error.to_string())
    }
    fn bootstrap_mask(&self, indices: &[usize]) -> Result<Array1<bool>, String> {
        self.storage
            .scalar_column(Some(indices), |row| !row.terminated)
            .map_err(|error| error.to_string())
    }
    fn end_flags(&self) -> Result<Array1<bool>, String> {
        self.storage
            .scalar_column(None, |row| row.terminated || row.truncated)
            .map_err(|error| error.to_string())
    }
    fn unfinished(&self) -> Result<Vec<usize>, String> {
        self.index
            .unfinished_indices()
            .map_err(|error| error.to_string())
    }
}

#[cfg(test)]
#[path = "../tests/support/rl_candle_replay.rs"]
mod tests;
