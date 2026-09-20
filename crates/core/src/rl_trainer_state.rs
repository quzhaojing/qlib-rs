//! Live Trainer state and the concrete bridge from RL log buffers to vessel-visible metrics.

use std::sync::{Arc, Mutex, MutexGuard};

use indexmap::IndexMap;
use num_bigint::BigInt;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    EnvironmentLogLevel, RlLogBufferEvent, RlLogBufferState, RlLogError, RlLogWriterState,
    TrainingTrainerView,
};

pub type RlTrainerMetrics<M = f64> = IndexMap<String, M>;

/// None represents an attribute not yet initialized by the Python lifecycle. In particular,
/// construction does not initialize iteration counters, and initialize does not clear metrics.
/// This is runtime metadata, not the complete vessel/callback/logger checkpoint graph.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RlTrainerState<M = f64> {
    pub should_stop: Option<bool>,
    pub current_iter: Option<BigInt>,
    pub current_episode: Option<BigInt>,
    pub current_stage: String,
    pub metrics: Option<RlTrainerMetrics<M>>,
}

impl<M> Default for RlTrainerState<M> {
    fn default() -> Self {
        Self {
            should_stop: None,
            current_iter: None,
            current_episode: None,
            current_stage: "train".into(),
            metrics: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RlTrainerStateField {
    CurrentIteration,
    Metrics,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum RlTrainerStateError {
    #[error("trainer {0:?} has not been initialized")]
    Uninitialized(RlTrainerStateField),
    #[error("metric callback selected neither episode nor collect metrics")]
    NoMetricEvent,
    #[error("trainer metric provider failed at {field}: {message}")]
    Provider { field: String, message: String },
    #[error(transparent)]
    Buffer(#[from] RlLogError),
    #[error("trainer runtime state mutex is poisoned")]
    Poisoned,
}

/// A metric producer controls the payload type; values are moved into the trainer without
/// requiring Clone or serializing arbitrary model-specific data.
pub trait RlTrainerMetricSource<M> {
    /// # Errors
    /// Returns a failure before the trainer's episode counter is updated.
    fn global_episode(&mut self) -> Result<BigInt, RlTrainerStateError>;
    /// # Errors
    /// Returns a metric-read failure after the episode counter has been updated.
    fn episode_metrics(&mut self) -> Result<RlTrainerMetrics<M>, RlTrainerStateError>;
    /// # Errors
    /// Returns a collect-read failure without changing the episode counter.
    fn collect_metrics(&mut self) -> Result<RlTrainerMetrics<M>, RlTrainerStateError>;
}

impl<M> RlTrainerState<M> {
    pub fn initialize(&mut self) {
        self.should_stop = Some(false);
        self.current_iter = Some(BigInt::default());
        self.current_episode = Some(BigInt::default());
        self.current_stage = "train".into();
    }

    pub fn initialize_iter(&mut self) {
        self.metrics = Some(IndexMap::new());
    }

    /// Episode wins if both flags are set. Only the exact stage `val` adds a `val/` prefix;
    /// existing prefixes are not stripped. Existing keys retain their insertion positions.
    ///
    /// # Errors
    /// Returns provider, missing-metrics, or missing-event failures in source evaluation order.
    pub fn metrics_callback(
        &mut self,
        on_episode: bool,
        on_collect: bool,
        source: &mut dyn RlTrainerMetricSource<M>,
    ) -> Result<(), RlTrainerStateError> {
        let mut metrics = if on_episode {
            self.current_episode = Some(source.global_episode()?);
            Some(source.episode_metrics()?)
        } else if on_collect {
            Some(source.collect_metrics()?)
        } else {
            None
        };
        if self.current_stage == "val" {
            metrics = Some(
                metrics
                    .ok_or(RlTrainerStateError::NoMetricEvent)?
                    .into_iter()
                    .map(|(name, value)| (format!("val/{name}"), value))
                    .collect(),
            );
        }
        let target = self
            .metrics
            .as_mut()
            .ok_or(RlTrainerStateError::Uninitialized(
                RlTrainerStateField::Metrics,
            ))?;
        target.extend(metrics.ok_or(RlTrainerStateError::NoMetricEvent)?);
        Ok(())
    }
}

/// Computes the live minimum over the supplied writers; only an empty collection defaults
/// to PERIODIC. Arbitrary integral levels and changes between calls are preserved.
#[must_use]
pub fn minimum_rl_log_level(levels: impl IntoIterator<Item = i64>) -> i64 {
    levels
        .into_iter()
        .min()
        .unwrap_or(EnvironmentLogLevel::Periodic as i64)
}

pub struct RlLogBufferMetricSource<'a, V> {
    pub writer: &'a RlLogWriterState<V>,
    pub buffer: &'a RlLogBufferState,
}

impl<V, M: From<f64>> RlTrainerMetricSource<M> for RlLogBufferMetricSource<'_, V> {
    fn global_episode(&mut self) -> Result<BigInt, RlTrainerStateError> {
        Ok(self.writer.global_episode.clone())
    }
    fn episode_metrics(&mut self) -> Result<RlTrainerMetrics<M>, RlTrainerStateError> {
        Ok(self
            .buffer
            .episode_metrics()?
            .iter()
            .map(|(name, value)| (name.clone(), M::from(*value)))
            .collect())
    }
    fn collect_metrics(&mut self) -> Result<RlTrainerMetrics<M>, RlTrainerStateError> {
        Ok(self
            .buffer
            .collect_metrics(&self.writer.episode_count)?
            .into_iter()
            .map(|(name, value)| (name, M::from(value)))
            .collect())
    }
}

struct RuntimeData<M> {
    state: RlTrainerState<M>,
    fast_dev_run: Option<i64>,
}

/// Shared state for the synchronous Trainer, log callbacks, and weak vessel bindings.
/// Read/update closures execute while the mutex is held and must not reenter this runtime.
pub struct RlTrainerRuntime<M = f64> {
    data: Mutex<RuntimeData<M>>,
}

impl<M> RlTrainerRuntime<M> {
    #[must_use]
    pub fn new(fast_dev_run: Option<i64>) -> Self {
        Self {
            data: Mutex::new(RuntimeData {
                state: RlTrainerState::default(),
                fast_dev_run,
            }),
        }
    }

    fn lock(&self) -> Result<MutexGuard<'_, RuntimeData<M>>, RlTrainerStateError> {
        self.data.lock().map_err(|_| RlTrainerStateError::Poisoned)
    }

    /// # Errors
    /// Returns a poisoned-state error without silently recovering partially mutated state.
    pub fn read<T>(
        &self,
        read: impl FnOnce(&RlTrainerState<M>) -> T,
    ) -> Result<T, RlTrainerStateError> {
        Ok(read(&self.lock()?.state))
    }

    /// # Errors
    /// Returns a poisoned-state error. A panic inside update poisons future runtime access.
    pub fn update<T>(
        &self,
        update: impl FnOnce(&mut RlTrainerState<M>) -> T,
    ) -> Result<T, RlTrainerStateError> {
        Ok(update(&mut self.lock()?.state))
    }

    /// # Errors
    /// Returns a poisoned-state error.
    pub fn set_fast_dev_run(&self, value: Option<i64>) -> Result<(), RlTrainerStateError> {
        self.lock()?.fast_dev_run = value;
        Ok(())
    }
}

impl<M: Send> TrainingTrainerView for RlTrainerRuntime<M> {
    fn current_iteration(&self) -> Result<BigInt, String> {
        self.read(|state| {
            state
                .current_iter
                .clone()
                .ok_or(RlTrainerStateError::Uninitialized(
                    RlTrainerStateField::CurrentIteration,
                ))
        })
        .and_then(std::convert::identity)
        .map_err(|error| error.to_string())
    }
    fn fast_dev_run(&self) -> Result<Option<i64>, String> {
        self.lock()
            .map(|data| data.fast_dev_run)
            .map_err(|error| error.to_string())
    }
}

pub type RlTrainerBufferCallback<V> = Box<
    dyn FnMut(RlLogBufferEvent, &RlLogWriterState<V>, &RlLogBufferState) -> Result<(), String>
        + Send,
>;

impl<M: From<f64> + Send + 'static> RlTrainerRuntime<M> {
    /// Supplies the concrete callback accepted by `RlLogBuffer::new_buffer`. Like Python's
    /// bound method, it keeps runtime state alive. The runtime owns no callbacks or loggers,
    /// so this does not create a reference cycle. Counter updates before failures are retained.
    #[must_use]
    pub fn buffer_callback<V>(self: &Arc<Self>) -> RlTrainerBufferCallback<V> {
        let runtime = Arc::clone(self);
        Box::new(move |event, writer, buffer| {
            let mut source = RlLogBufferMetricSource { writer, buffer };
            runtime
                .update(|state| {
                    state.metrics_callback(
                        event == RlLogBufferEvent::Episode,
                        event == RlLogBufferEvent::Collect,
                        &mut source,
                    )
                })
                .and_then(std::convert::identity)
                .map_err(|error| error.to_string())
        })
    }
}

#[cfg(test)]
#[path = "../tests/support/rl_trainer_state.rs"]
mod tests;
