//! Stateful central RL log aggregation and the Trainer's numeric metric buffer.

use arrow_arith::aggregate::sum;
use arrow_array::Float64Array;
use indexmap::{IndexMap, IndexSet};
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{EnvironmentPluginError, FiniteBackendStep, FiniteVectorLogger};

/// Float means Python `isinstance(value, float)`, not merely a convertible numeric value.
/// Other values (including integers and booleans) retain their adapter-owned representation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum RlLogValue<V> {
    Float(f64),
    Other(V),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RlLogEntry<V> {
    pub level: i64,
    pub value: RlLogValue<V>,
}

pub type RlLogContents<V> = IndexMap<String, RlLogValue<V>>;
pub type RlLeveledLogs<V> = IndexMap<String, RlLogEntry<V>>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RlLogInfo<V> {
    pub log: Option<RlLeveledLogs<V>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RlLogWriterState<V> {
    pub episode_count: BigInt,
    pub step_count: BigInt,
    pub global_step: BigInt,
    pub global_episode: BigInt,
    pub active_env_ids: IndexSet<usize>,
    pub episode_lengths: IndexMap<usize, BigInt>,
    pub episode_rewards: IndexMap<usize, Vec<f64>>,
    pub episode_logs: IndexMap<usize, Vec<RlLogContents<V>>>,
}

impl<V> Default for RlLogWriterState<V> {
    fn default() -> Self {
        Self {
            episode_count: BigInt::default(),
            step_count: BigInt::default(),
            global_step: BigInt::default(),
            global_episode: BigInt::default(),
            active_env_ids: IndexSet::new(),
            episode_lengths: IndexMap::new(),
            episode_rewards: IndexMap::new(),
            episode_logs: IndexMap::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RlEpisodeField {
    Length,
    Rewards,
    Logs,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RlLogBufferEvent {
    Episode,
    Collect,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum RlLogError {
    #[error("episode {field:?} is not initialized for environment {id}")]
    Uninitialized { id: usize, field: RlEpisodeField },
    #[error("environment info has no log field")]
    MissingLog,
    #[error("finite environment supplied no reward")]
    MissingReward,
    #[error("the aggregated array must not be empty")]
    EmptyAggregation,
    #[error("no episode metrics available yet")]
    NoEpisodeMetrics,
    #[error("episode count cannot be represented as a finite float: {0}")]
    EpisodeCountNotRepresentable(BigInt),
    #[error("RL log hook failed: {0}")]
    Hook(String),
    #[error("RL log buffer {event:?} callback failed: {message}")]
    Callback {
        event: RlLogBufferEvent,
        message: String,
    },
}

pub trait RlLogWriterHooks<V> {
    fn clear(&mut self) {}

    /// # Errors
    /// Returns a sink failure after step history has been updated.
    fn log_step(
        &mut self,
        _reward: f64,
        _contents: &RlLogContents<V>,
        _state: &RlLogWriterState<V>,
    ) -> Result<(), RlLogError> {
        Ok(())
    }

    /// # Errors
    /// Returns a sink failure after episode counters have been updated.
    fn log_episode(
        &mut self,
        _length: &BigInt,
        _rewards: &[f64],
        _contents: &[RlLogContents<V>],
        _state: &RlLogWriterState<V>,
    ) -> Result<(), RlLogError> {
        Ok(())
    }

    /// # Errors
    /// Returns a finalization failure without clearing current state.
    fn on_all_done(&mut self, _state: &RlLogWriterState<V>) -> Result<(), RlLogError> {
        Ok(())
    }
}

#[derive(Default)]
pub struct NoopRlLogWriterHooks;
impl<V> RlLogWriterHooks<V> for NoopRlLogWriterHooks {}

pub struct RlLogWriter<V, H = NoopRlLogWriterHooks> {
    pub loglevel: i64,
    state: RlLogWriterState<V>,
    hooks: H,
}

impl<V: Clone, H: RlLogWriterHooks<V>> RlLogWriter<V, H> {
    #[must_use]
    pub fn new(loglevel: i64, hooks: H) -> Self {
        let mut writer = Self {
            loglevel,
            state: RlLogWriterState::default(),
            hooks,
        };
        writer.clear();
        writer
    }

    /// Clears collect-local counters and active ids, not global counters or episode histories.
    pub fn clear(&mut self) {
        self.state.episode_count = BigInt::default();
        self.state.step_count = BigInt::default();
        self.state.active_env_ids.clear();
        self.hooks.clear();
    }

    #[must_use]
    pub const fn state_dict(&self) -> &RlLogWriterState<V> {
        &self.state
    }

    /// Takes an owned, typed checkpoint. Shape/type validation belongs to its deserializer.
    pub fn load_state_dict(&mut self, state: RlLogWriterState<V>) {
        self.state = state;
    }

    pub fn on_env_reset(&mut self, id: usize) {
        self.state.episode_lengths.insert(id, BigInt::default());
        self.state.episode_rewards.insert(id, Vec::new());
        self.state.episode_logs.insert(id, Vec::new());
    }

    /// # Errors
    /// Returns missing-state/log or hook errors, retaining mutations preceding the failure.
    pub fn on_env_step(
        &mut self,
        id: usize,
        reward: f64,
        done: bool,
        logs: Option<&RlLeveledLogs<V>>,
    ) -> Result<(), RlLogError> {
        self.state.global_step += 1;
        self.state.step_count += 1;
        self.state.active_env_ids.insert(id);
        *self
            .state
            .episode_lengths
            .get_mut(&id)
            .ok_or(RlLogError::Uninitialized {
                id,
                field: RlEpisodeField::Length,
            })? += 1;
        self.state
            .episode_rewards
            .get_mut(&id)
            .ok_or(RlLogError::Uninitialized {
                id,
                field: RlEpisodeField::Rewards,
            })?
            .push(reward);
        let values: RlLogContents<V> = logs
            .ok_or(RlLogError::MissingLog)?
            .iter()
            .filter(|(_, entry)| entry.level >= self.loglevel)
            .map(|(key, entry)| (key.clone(), entry.value.clone()))
            .collect();
        self.state
            .episode_logs
            .get_mut(&id)
            .ok_or(RlLogError::Uninitialized {
                id,
                field: RlEpisodeField::Logs,
            })?
            .push(values.clone());
        self.hooks.log_step(reward, &values, &self.state)?;
        if done {
            self.state.global_episode += 1;
            self.state.episode_count += 1;
            self.hooks.log_episode(
                &self.state.episode_lengths[&id],
                &self.state.episode_rewards[&id],
                &self.state.episode_logs[&id],
                &self.state,
            )?;
        }
        Ok(())
    }

    /// # Errors
    /// Returns the sink's collect-finalization failure.
    pub fn on_env_all_done(&mut self) -> Result<(), RlLogError> {
        self.hooks.on_all_done(&self.state)
    }
}

/// Float-only sequences reduce by mean (or sum for `reward`); mixed sequences return the first
/// value. Use shared handles in `V` when nonnumeric object identity matters.
///
/// # Errors
/// Returns an error for empty sequences.
pub fn aggregate_rl_logs<V: Clone>(
    values: &[RlLogValue<V>],
    name: Option<&str>,
) -> Result<RlLogValue<V>, RlLogError> {
    let first = values.first().ok_or(RlLogError::EmptyAggregation)?;
    let floats: Option<Vec<f64>> = values
        .iter()
        .map(|value| match value {
            RlLogValue::Float(value) => Some(*value),
            RlLogValue::Other(_) => None,
        })
        .collect();
    Ok(floats.map_or_else(
        || first.clone(),
        |values| RlLogValue::Float(aggregate_floats(values, name)),
    ))
}

fn aggregate_floats(values: Vec<f64>, name: Option<&str>) -> f64 {
    let count = values.len().to_f64().expect("array length fits a float");
    let total = sum(&Float64Array::from(values)).expect("float aggregation is nonempty");
    if name == Some("reward") {
        total
    } else {
        total / count
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RlLogBufferState {
    #[serde(deserialize_with = "Option::deserialize")]
    pub latest_metrics: Option<IndexMap<String, f64>>,
    pub aggregated_metrics: IndexMap<String, f64>,
}

impl RlLogBufferState {
    /// # Errors
    /// Returns an error if no episode has completed since clear.
    pub fn episode_metrics(&self) -> Result<&IndexMap<String, f64>, RlLogError> {
        self.latest_metrics
            .as_ref()
            .ok_or(RlLogError::NoEpisodeMetrics)
    }

    /// Uses the total episode count, not the number of episodes that emitted each metric.
    ///
    /// # Errors
    /// Returns a conversion error if a nonempty buffer's count exceeds finite Float64 range.
    pub fn collect_metrics(
        &self,
        episode_count: &BigInt,
    ) -> Result<IndexMap<String, f64>, RlLogError> {
        self.aggregated_metrics
            .iter()
            .map(|(name, value)| {
                let count = episode_count
                    .to_f64()
                    .filter(|count| count.is_finite())
                    .ok_or_else(|| {
                        RlLogError::EpisodeCountNotRepresentable(episode_count.clone())
                    })?;
                Ok((name.clone(), value / count))
            })
            .collect()
    }
}

pub struct RlLogBufferHooks<C> {
    state: RlLogBufferState,
    callback: C,
}

impl<V, C> RlLogWriterHooks<V> for RlLogBufferHooks<C>
where
    C: FnMut(RlLogBufferEvent, &RlLogWriterState<V>, &RlLogBufferState) -> Result<(), String>,
{
    fn clear(&mut self) {
        self.state = RlLogBufferState::default();
    }

    fn log_episode(
        &mut self,
        _: &BigInt,
        _: &[f64],
        contents: &[RlLogContents<V>],
        writer: &RlLogWriterState<V>,
    ) -> Result<(), RlLogError> {
        let mut grouped: IndexMap<String, Vec<f64>> = IndexMap::new();
        for step in contents {
            for (name, value) in step {
                if let RlLogValue::Float(value) = value {
                    grouped.entry(name.clone()).or_default().push(*value);
                }
            }
        }
        let mut latest = IndexMap::new();
        for (name, values) in grouped {
            let value = aggregate_floats(values, Some(&name));
            *self
                .state
                .aggregated_metrics
                .entry(name.clone())
                .or_default() += value;
            latest.insert(name, value);
        }
        self.state.latest_metrics = Some(latest);
        (self.callback)(RlLogBufferEvent::Episode, writer, &self.state).map_err(|message| {
            RlLogError::Callback {
                event: RlLogBufferEvent::Episode,
                message,
            }
        })
    }

    fn on_all_done(&mut self, writer: &RlLogWriterState<V>) -> Result<(), RlLogError> {
        (self.callback)(RlLogBufferEvent::Collect, writer, &self.state).map_err(|message| {
            RlLogError::Callback {
                event: RlLogBufferEvent::Collect,
                message,
            }
        })
    }
}

pub type RlLogBuffer<V, C> = RlLogWriter<V, RlLogBufferHooks<C>>;

impl<V: Clone, C> RlLogBuffer<V, C>
where
    C: FnMut(RlLogBufferEvent, &RlLogWriterState<V>, &RlLogBufferState) -> Result<(), String>,
{
    #[must_use]
    pub fn new_buffer(loglevel: i64, callback: C) -> Self {
        Self::new(
            loglevel,
            RlLogBufferHooks {
                state: RlLogBufferState::default(),
                callback,
            },
        )
    }

    #[must_use]
    pub const fn buffer_state(&self) -> &RlLogBufferState {
        &self.hooks.state
    }

    pub fn load_buffer_state(&mut self, buffer: RlLogBufferState, writer: RlLogWriterState<V>) {
        self.hooks.state = buffer;
        self.load_state_dict(writer);
    }
}

impl<O, V: Clone + Send, H: RlLogWriterHooks<V> + Send> FiniteVectorLogger<O, f64, RlLogInfo<V>>
    for RlLogWriter<V, H>
{
    fn on_all_ready(&mut self) -> Result<(), EnvironmentPluginError> {
        self.clear();
        Ok(())
    }
    fn on_all_done(&mut self) -> Result<(), EnvironmentPluginError> {
        self.on_env_all_done()
            .map_err(|error| EnvironmentPluginError::new(error.to_string()))
    }
    fn on_reset(&mut self, id: usize, _: &[Option<O>]) -> Result<(), EnvironmentPluginError> {
        self.on_env_reset(id);
        Ok(())
    }
    fn on_step(
        &mut self,
        id: usize,
        step: &FiniteBackendStep<O, f64, RlLogInfo<V>>,
    ) -> Result<(), EnvironmentPluginError> {
        let reward = step
            .reward
            .ok_or_else(|| EnvironmentPluginError::new(RlLogError::MissingReward.to_string()))?;
        self.on_env_step(
            id,
            reward,
            step.done,
            step.info.as_ref().and_then(|info| info.log.as_ref()),
        )
        .map_err(|error| EnvironmentPluginError::new(error.to_string()))
    }
}

/// Preserve the wrapper's typed debug payloads without converting them to numbers
/// or consuming its auxiliary information. Numeric aggregation stays in the writer.
impl<O, A, I, H> FiniteVectorLogger<O, f64, crate::EnvironmentStepInfo<O, A, I>>
    for RlLogWriter<crate::EnvironmentLogValue<O, A>, H>
where
    O: Clone + Send,
    A: Clone + Send,
    H: RlLogWriterHooks<crate::EnvironmentLogValue<O, A>> + Send,
{
    fn on_all_ready(&mut self) -> Result<(), EnvironmentPluginError> {
        self.clear();
        Ok(())
    }

    fn on_all_done(&mut self) -> Result<(), EnvironmentPluginError> {
        self.on_env_all_done()
            .map_err(|error| EnvironmentPluginError::new(error.to_string()))
    }

    fn on_reset(&mut self, id: usize, _: &[Option<O>]) -> Result<(), EnvironmentPluginError> {
        self.on_env_reset(id);
        Ok(())
    }

    fn on_step(
        &mut self,
        id: usize,
        step: &FiniteBackendStep<O, f64, crate::EnvironmentStepInfo<O, A, I>>,
    ) -> Result<(), EnvironmentPluginError> {
        let reward = step
            .reward
            .ok_or_else(|| EnvironmentPluginError::new(RlLogError::MissingReward.to_string()))?;
        let logs = step.info.as_ref().map(|info| {
            info.logs
                .iter()
                .map(|(name, entry)| {
                    let value = match &entry.value {
                        crate::EnvironmentLogValue::Scalar(value) => RlLogValue::Float(*value),
                        value => RlLogValue::Other(value.clone()),
                    };
                    (
                        name.clone(),
                        RlLogEntry {
                            level: i64::from(entry.level as i32),
                            value,
                        },
                    )
                })
                .collect()
        });
        self.on_env_step(id, reward, step.done, logs.as_ref())
            .map_err(|error| EnvironmentPluginError::new(error.to_string()))
    }
}

#[cfg(test)]
#[path = "../tests/support/rl_log_writer.rs"]
mod tests;
