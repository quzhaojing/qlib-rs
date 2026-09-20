//! Multi-environment replay with shared schema and source two-stage sampling.

use super::{
    CandlePpoReplay, CandlePpoRollout, CandleReplayError, CandleReplayTransition,
    RecurrentObservation, ReplayEpisode, ReplayIndex, ReplayIndexError, ReplayStorage, fields,
};
use candle_core::Tensor;
use ndarray::Array1;
use rand::{
    RngCore,
    distr::{Distribution, weighted::WeightedIndex},
};
use std::{borrow::Cow, sync::Arc};
use thiserror::Error;

/// Borrowed batch columns. Reward and flag rows correspond to `buffer_ids`.
/// Observation/action tensor columns may broadcast to the selected storage rows.
pub struct CandleReplayBatch<'a> {
    pub observations: &'a RecurrentObservation,
    pub next_observations: &'a RecurrentObservation,
    pub actions: &'a Tensor,
    pub rewards: &'a [f64],
    pub terminated: &'a [bool],
    pub truncated: &'a [bool],
}

#[derive(Debug, Error)]
pub enum CandleVectorReplayError {
    #[error("vector replay requires at least one environment")]
    EnvironmentCount,
    #[error("rounded vector replay capacity exceeds native index range")]
    CapacityOverflow,
    #[error("vector replay environment id is outside the child buffers")]
    EnvironmentIndex,
    #[error("vector replay reward/flag rows do not match environment ids")]
    BatchSize,
    #[error("empty vector insertion has no integer physical indices")]
    EmptyIndices,
    #[error(transparent)]
    Storage(#[from] CandleReplayError),
    #[error(transparent)]
    Tensor(#[from] candle_core::Error),
    #[error(transparent)]
    Index(#[from] ReplayIndexError),
}

/// Equal-sized child rings over one initialized observation/action schema.
/// Uses stored next observations and one full-history observation per transition.
pub struct CandleVectorReplayBuffer {
    children: Vec<ReplayIndex>,
    child_capacity: usize,
    capacity: usize,
    storage: ReplayStorage,
}

impl CandleVectorReplayBuffer {
    /// Actual capacity rounds upward to an equal number of slots per child.
    /// # Errors
    /// Zero environments or overflow of the rounded capacity.
    pub fn new(total_size: usize, environments: usize) -> Result<Self, CandleVectorReplayError> {
        if environments == 0 {
            return Err(CandleVectorReplayError::EnvironmentCount);
        }
        let child_capacity = total_size.div_ceil(environments);
        let capacity = child_capacity
            .checked_mul(environments)
            .ok_or(CandleVectorReplayError::CapacityOverflow)?;
        Ok(Self {
            children: (0..environments)
                .map(|_| ReplayIndex::new(child_capacity))
                .collect(),
            child_capacity,
            capacity,
            storage: ReplayStorage::default(),
        })
    }

    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[must_use]
    pub fn children(&self) -> &[ReplayIndex] {
        &self.children
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.children.iter().map(ReplayIndex::len).sum()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.children.iter().all(ReplayIndex::is_empty)
    }

    pub fn reset(&mut self, keep_statistics: bool) {
        for child in &mut self.children {
            child.reset(keep_statistics);
        }
    }

    /// Advances all selected episode/index states before assigning tensor fields.
    /// Duplicate environment ids and physical slots preserve input order.
    /// Native field order is current observation, next observation, action, scalars.
    /// Each tensor field validates/broadcasts the complete batch before any row write.
    /// # Errors
    /// Returns row/id/capacity/storage errors without rolling back preceding mutations.
    pub fn add(
        &mut self,
        batch: &CandleReplayBatch<'_>,
        buffer_ids: Option<&[usize]>,
    ) -> Result<Vec<ReplayEpisode>, CandleVectorReplayError> {
        let ids: Cow<'_, [usize]> = buffer_ids.map_or_else(
            || Cow::Owned((0..self.children.len()).collect()),
            Cow::Borrowed,
        );
        let mut episodes = Vec::with_capacity(ids.len());
        for (row, &id) in ids.iter().enumerate() {
            let child = self
                .children
                .get_mut(id)
                .ok_or(CandleVectorReplayError::EnvironmentIndex)?;
            let reward = *batch
                .rewards
                .get(row)
                .ok_or(CandleVectorReplayError::BatchSize)?;
            let terminated = *batch
                .terminated
                .get(row)
                .ok_or(CandleVectorReplayError::BatchSize)?;
            let truncated = *batch
                .truncated
                .get(row)
                .ok_or(CandleVectorReplayError::BatchSize)?;
            let mut episode = child.advance(reward, terminated || truncated)?;
            let offset = id * self.child_capacity;
            episode.index += offset;
            episode.start += offset;
            episodes.push(episode);
        }
        if self.storage.initialize(
            batch.observations,
            batch.next_observations,
            batch.actions,
            self.capacity,
        )? {
            for child in &mut self.children {
                child.initialize_done();
            }
        }
        if ids.is_empty() {
            // Source allocates schema before rejecting its empty float-index array.
            return Err(CandleVectorReplayError::EmptyIndices);
        }
        self.assign_batch(batch, &episodes)?;
        for (episode, &id) in episodes.iter().zip(ids.iter()) {
            let value = &self.storage.slots[episode.index];
            self.children[id].commit_done(
                episode.index - id * self.child_capacity,
                value.terminated || value.truncated,
            )?;
        }
        Ok(episodes)
    }

    fn assign_batch(
        &mut self,
        batch: &CandleReplayBatch<'_>,
        episodes: &[ReplayEpisode],
    ) -> Result<(), CandleVectorReplayError> {
        for (field, source) in fields(batch.observations)
            .into_iter()
            .chain(fields(batch.next_observations))
            .chain([batch.actions])
            .enumerate()
        {
            let template = tensor_fields(&self.storage.slots[0])[field];
            if template.dtype() != source.dtype() {
                return Err(candle_core::Error::Msg(
                    "replay tensor assignment requires matching dtypes".into(),
                )
                .into());
            }
            let mut shape = template.dims().to_vec();
            shape[0] = episodes.len();
            let prepared = source
                .broadcast_as(shape)?
                .to_device(template.device())?
                .detach();
            // Prepare all copied rows first: a failing field must not partly commit.
            let rows = (0..episodes.len())
                .map(|row| prepared.narrow(0, row, 1)?.copy())
                .collect::<candle_core::Result<Vec<_>>>()?;
            for (episode, value) in episodes.iter().zip(rows) {
                let slot = Arc::make_mut(&mut self.storage.slots[episode.index]);
                *tensor_fields_mut(slot)[field] = value;
            }
        }
        if batch.rewards.len() != episodes.len()
            || batch.terminated.len() != episodes.len()
            || batch.truncated.len() != episodes.len()
        {
            return Err(CandleVectorReplayError::BatchSize);
        }
        for (row, episode) in episodes.iter().enumerate() {
            let slot = Arc::make_mut(&mut self.storage.slots[episode.index]);
            slot.reward = batch.rewards[row];
            slot.terminated = batch.terminated[row];
            slot.truncated = batch.truncated[row];
        }
        Ok(())
    }

    /// # Errors
    /// A fresh vector has no initialized done column, including in empty children.
    pub fn unfinished_indices(&self) -> Result<Vec<usize>, ReplayIndexError> {
        let mut result = Vec::new();
        for (id, child) in self.children.iter().enumerate() {
            result.extend(
                child
                    .unfinished_indices()?
                    .into_iter()
                    .map(|index| index + id * self.child_capacity),
            );
        }
        Ok(result)
    }

    /// # Errors
    /// Missing schema, invalid physical slots or tensor batching failures.
    pub fn get(&self, indices: &[usize]) -> Result<CandlePpoRollout, CandleVectorReplayError> {
        let mut batch = self.storage.get(indices)?;
        batch.unfinished_indices = self.unfinished_indices()?;
        Ok(batch)
    }

    /// Positions wrap over global capacity, but never cross a child episode/ring boundary.
    /// # Errors
    /// Missing done schema or zero-capacity indexing.
    pub fn previous(&self, indices: &[usize]) -> Result<Vec<usize>, ReplayIndexError> {
        self.navigate(indices, false)
    }

    /// # Errors
    /// Missing done schema or zero-capacity indexing.
    pub fn next(&self, indices: &[usize]) -> Result<Vec<usize>, ReplayIndexError> {
        self.navigate(indices, true)
    }

    fn navigate(&self, indices: &[usize], forward: bool) -> Result<Vec<usize>, ReplayIndexError> {
        self.children[0].previous(&[])?;
        indices
            .iter()
            .map(|&index| {
                let index = index
                    .checked_rem(self.capacity)
                    .ok_or(ReplayIndexError::ZeroCapacity)?;
                let id = index / self.child_capacity;
                let offset = id * self.child_capacity;
                let child = &self.children[id];
                let local = [index - offset];
                let positions = if forward {
                    child.next(&local)?
                } else {
                    child.previous(&local)?
                };
                Ok(positions[0] + offset)
            })
            .collect()
    }

    /// Positive samples first select children proportionally to active lengths,
    /// then draw replacement positions grouped by child. Zero returns each child's
    /// chronological contents; negative sizes return empty without consuming RNG.
    /// # Errors
    /// Positive sampling from an empty population.
    pub fn sample_indices(
        &self,
        size: i64,
        rng: &mut dyn RngCore,
    ) -> Result<Vec<usize>, ReplayIndexError> {
        if size < 0 {
            return Ok(Vec::new());
        }
        let mut counts = vec![0_i64; self.children.len()];
        if size > 0 {
            let distribution = WeightedIndex::new(self.children.iter().map(ReplayIndex::len))
                .map_err(|_| ReplayIndexError::EmptyPopulation)?;
            for _ in 0..size {
                counts[distribution.sample(rng)] += 1;
            }
            for count in &mut counts {
                if *count == 0 {
                    *count = -1;
                }
            }
        }
        let mut result = Vec::new();
        for (id, (child, count)) in self.children.iter().zip(counts).enumerate() {
            result.extend(
                child
                    .sample_indices(count, 1, false, rng)?
                    .into_iter()
                    .map(|index| index + id * self.child_capacity),
            );
        }
        Ok(result)
    }

    /// # Errors
    /// Sampling population, storage schema or tensor errors.
    pub fn sample_batch(
        &self,
        size: i64,
        rng: &mut dyn RngCore,
    ) -> Result<CandlePpoRollout, CandleVectorReplayError> {
        self.get(&self.sample_indices(size, rng)?)
    }
}

impl CandlePpoReplay for CandleVectorReplayBuffer {
    fn sample(&mut self, size: u64, rng: &mut dyn RngCore) -> Result<CandlePpoRollout, String> {
        let size = i64::try_from(size).map_err(|_| CandleReplayError::SampleSize.to_string())?;
        self.sample_batch(size, rng)
            .map_err(|error| error.to_string())
    }
}

impl crate::rl_candle_nstep::CandleNStepReplay for CandleVectorReplayBuffer {
    fn rewards(&self) -> Result<Array1<f64>, String> {
        self.storage
            .scalar_column(None, |row| row.reward)
            .map_err(|error| error.to_string())
    }
    fn next_indices(&self, indices: &[usize]) -> Result<Vec<usize>, String> {
        self.next(indices).map_err(|error| error.to_string())
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
        self.unfinished_indices().map_err(|error| error.to_string())
    }
}

fn tensor_fields(value: &CandleReplayTransition) -> [&Tensor; 15] {
    let obs = fields(&value.observation);
    let next = fields(&value.next_observation);
    [
        obs[0],
        obs[1],
        obs[2],
        obs[3],
        obs[4],
        obs[5],
        obs[6],
        next[0],
        next[1],
        next[2],
        next[3],
        next[4],
        next[5],
        next[6],
        &value.action,
    ]
}

fn tensor_fields_mut(value: &mut CandleReplayTransition) -> [&mut Tensor; 15] {
    let obs = &mut value.observation;
    let next = &mut value.next_observation;
    [
        &mut obs.data_processed,
        &mut obs.cur_tick,
        &mut obs.cur_step,
        &mut obs.position_history,
        &mut obs.target,
        &mut obs.num_step,
        &mut obs.acquiring,
        &mut next.data_processed,
        &mut next.cur_tick,
        &mut next.cur_step,
        &mut next.position_history,
        &mut next.target,
        &mut next.num_step,
        &mut next.acquiring,
        &mut value.action,
    ]
}

#[cfg(test)]
#[path = "../../tests/support/rl_candle_vector_replay.rs"]
mod tests;
