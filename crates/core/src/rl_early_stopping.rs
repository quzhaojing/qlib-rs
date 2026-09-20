//! Qlib early stopping with ordered snapshots, diagnostics and partial checkpoint restore.

use crate::{
    RlCheckpointField, RlCheckpointState, RlTrainerCallback, RlTrainerCheckpointError,
    RlTrainerControl, RlTrainerDriverError, RlTrainerHook, RlTrainerRuntime, RlTrainerStateError,
};
use num_bigint::BigInt;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use thiserror::Error;

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum RlEarlyStoppingError {
    #[error("Unsupported earlystopping mode: {0}")]
    UnsupportedMode(String),
    #[error(transparent)]
    Runtime(#[from] RlTrainerStateError),
    #[error(transparent)]
    Checkpoint(#[from] RlTrainerCheckpointError),
    #[error("trainer {0} has not been initialized")]
    MissingTrainerField(&'static str),
    #[error("early stopping {stage} failed: {message}")]
    Plugin {
        stage: &'static str,
        message: String,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct RlEarlyStoppingConfig {
    pub monitor: String,
    pub min_delta: f64,
    pub patience: BigInt,
    pub mode: String,
    pub baseline: Option<f64>,
    pub restore_best_weights: bool,
}
impl Default for RlEarlyStoppingConfig {
    fn default() -> Self {
        Self {
            monitor: "reward".into(),
            min_delta: 0.0,
            patience: BigInt::default(),
            mode: "max".into(),
            baseline: None,
            restore_best_weights: false,
        }
    }
}

/// Partial map documents preserve sequential missing-key failures. Positional formats require
/// every field; nonfinite best values require a codec that supports nonfinite floats.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound(deserialize = "S: Deserialize<'de>"))]
pub struct RlEarlyStoppingState<S> {
    #[serde(default, skip_serializing_if = "RlCheckpointField::is_missing")]
    pub wait: RlCheckpointField<BigInt>,
    #[serde(default, skip_serializing_if = "RlCheckpointField::is_missing")]
    pub best: RlCheckpointField<f64>,
    #[serde(default, skip_serializing_if = "RlCheckpointField::is_missing")]
    pub best_weights: RlCheckpointField<Option<S>>,
    #[serde(default, skip_serializing_if = "RlCheckpointField::is_missing")]
    pub best_iter: RlCheckpointField<BigInt>,
}

/// Model adapters must detach mutable handles, not just shallow-clone them.
pub trait RlCheckpointSnapshot<S> {
    /// None represents a Python snapshot whose value is None; it must not be restored.
    /// # Errors
    /// Returns deep-copy failures after vessel save has completed.
    fn snapshot(&mut self, state: &S) -> Result<Option<S>, String>;
}

/// Detached owned DTO snapshot, not an arbitrary tensor/Python object graph codec.
pub struct BincodeRlCheckpointSnapshot;
impl<S: Serialize + DeserializeOwned> RlCheckpointSnapshot<S> for BincodeRlCheckpointSnapshot {
    fn snapshot(&mut self, state: &S) -> Result<Option<S>, String> {
        let bytes = bincode::serialize(state).map_err(|error| error.to_string())?;
        bincode::deserialize(&bytes)
            .map(Some)
            .map_err(|error| error.to_string())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RlEarlyStoppingLogLevel {
    Info,
    Warning,
}
pub trait RlEarlyStoppingLogger {
    /// # Errors
    /// Returns a diagnostic failure without rolling back previous mutations.
    fn log(&mut self, level: RlEarlyStoppingLogLevel, message: &str) -> Result<(), String>;
}
pub struct TracingRlEarlyStoppingLogger;
impl RlEarlyStoppingLogger for TracingRlEarlyStoppingLogger {
    fn log(&mut self, level: RlEarlyStoppingLogLevel, message: &str) -> Result<(), String> {
        match level {
            RlEarlyStoppingLogLevel::Info => tracing::info!("{message}"),
            RlEarlyStoppingLogLevel::Warning => tracing::warn!("{message}"),
        }
        Ok(())
    }
}

pub struct RlEarlyStopping<S, D = BincodeRlCheckpointSnapshot, L = TracingRlEarlyStoppingLogger> {
    /// `min_delta` is normalized to its signed threshold at construction; mode selects the
    /// comparator only at construction, matching Qlib's stored `monitor_op`.
    pub config: RlEarlyStoppingConfig,
    pub state: RlEarlyStoppingState<S>,
    pub snapshot: D,
    pub logger: L,
    minimize: bool,
}

impl<S, D, L: RlEarlyStoppingLogger> RlEarlyStopping<S, D, L> {
    /// # Errors
    /// Rejects mode strings other than min/max.
    pub fn new(
        mut config: RlEarlyStoppingConfig,
        snapshot: D,
        logger: L,
    ) -> Result<Self, RlEarlyStoppingError> {
        let minimize = match config.mode.as_str() {
            "min" => true,
            "max" => false,
            _ => return Err(RlEarlyStoppingError::UnsupportedMode(config.mode)),
        };
        config.min_delta = config.min_delta.abs() * if minimize { -1.0 } else { 1.0 };
        Ok(Self {
            config,
            snapshot,
            logger,
            minimize,
            state: RlEarlyStoppingState {
                wait: RlCheckpointField::Missing,
                best: RlCheckpointField::Missing,
                best_weights: RlCheckpointField::Present(None),
                best_iter: RlCheckpointField::Missing,
            },
        })
    }
    pub fn on_fit_start(&mut self) {
        self.state = RlEarlyStoppingState {
            wait: RlCheckpointField::Present(BigInt::default()),
            best: RlCheckpointField::Present(if self.minimize {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            }),
            best_weights: RlCheckpointField::Present(None),
            best_iter: RlCheckpointField::Present(BigInt::default()),
        };
    }
    #[must_use]
    pub fn is_improvement(&self, value: f64, reference: f64) -> bool {
        let adjusted = value - self.config.min_delta;
        if self.minimize {
            adjusted < reference
        } else {
            adjusted > reference
        }
    }
    /// Clone follows payload-defined checkpoint sharing; validation uses the explicit
    /// detached snapshot plugin instead.
    /// # Errors
    /// Reports the first uninitialized field in source order.
    pub fn state_dict(&self) -> Result<RlEarlyStoppingState<S>, RlEarlyStoppingError>
    where
        S: Clone,
    {
        Ok(RlEarlyStoppingState {
            wait: RlCheckpointField::Present(self.state.wait.require("wait")?.clone()),
            best: RlCheckpointField::Present(*self.state.best.require("best")?),
            best_weights: RlCheckpointField::Present(
                self.state.best_weights.require("best_weights")?.clone(),
            ),
            best_iter: RlCheckpointField::Present(
                self.state.best_iter.require("best_iter")?.clone(),
            ),
        })
    }
    /// # Errors
    /// Retains earlier assignments when a later field is missing.
    pub fn load_state_dict(
        &mut self,
        state: &RlEarlyStoppingState<S>,
    ) -> Result<(), RlEarlyStoppingError>
    where
        S: Clone,
    {
        self.state.wait = RlCheckpointField::Present(state.wait.require("wait")?.clone());
        self.state.best = RlCheckpointField::Present(*state.best.require("best")?);
        self.state.best_weights =
            RlCheckpointField::Present(state.best_weights.require("best_weights")?.clone());
        self.state.best_iter =
            RlCheckpointField::Present(state.best_iter.require("best_iter")?.clone());
        Ok(())
    }
    /// # Errors
    /// Reports missing/poisoned metrics or a warning-sink failure.
    pub fn get_monitor_value<M: Copy + Into<Option<f64>>>(
        &mut self,
        runtime: &RlTrainerRuntime<M>,
    ) -> Result<Option<f64>, RlEarlyStoppingError> {
        let (value, available) = runtime.read(|state| {
            let metrics = state
                .metrics
                .as_ref()
                .ok_or(RlEarlyStoppingError::MissingTrainerField("metrics"))?;
            let value = metrics
                .get(&self.config.monitor)
                .and_then(|value| (*value).into());
            let available = if value.is_none() {
                metrics.keys().cloned().collect::<Vec<_>>().join(",")
            } else {
                String::new()
            };
            Ok::<_, RlEarlyStoppingError>((value, available))
        })??;
        if value.is_none() {
            emit(
                &mut self.logger,
                RlEarlyStoppingLogLevel::Warning,
                &format!(
                    "Early stopping conditioned on metric `{}` which is not available. Available metrics are: {available}",
                    self.config.monitor
                ),
            )?;
        }
        Ok(value)
    }
}

fn emit(
    logger: &mut impl RlEarlyStoppingLogger,
    level: RlEarlyStoppingLogLevel,
    message: &str,
) -> Result<(), RlEarlyStoppingError> {
    logger
        .log(level, message)
        .map_err(|message| RlEarlyStoppingError::Plugin {
            stage: "log",
            message,
        })
}
fn iteration<M>(runtime: &RlTrainerRuntime<M>) -> Result<BigInt, RlEarlyStoppingError> {
    runtime
        .read(|state| state.current_iter.clone())?
        .ok_or(RlEarlyStoppingError::MissingTrainerField("current_iter"))
}

impl<S, D: RlCheckpointSnapshot<S>, L: RlEarlyStoppingLogger> RlEarlyStopping<S, D, L> {
    fn capture(
        &mut self,
        vessel: &mut impl RlCheckpointState<S>,
    ) -> Result<Option<S>, RlEarlyStoppingError> {
        let state = vessel
            .save_checkpoint()
            .map_err(|message| RlEarlyStoppingError::Plugin {
                stage: "save",
                message,
            })?;
        self.snapshot
            .snapshot(&state)
            .map_err(|message| RlEarlyStoppingError::Plugin {
                stage: "snapshot",
                message,
            })
    }
    /// # Errors
    /// Preserves ordered partial callback state and component side effects on every failure.
    pub fn on_validate_end<M: Copy + Into<Option<f64>>>(
        &mut self,
        runtime: &RlTrainerRuntime<M>,
        vessel: &mut impl RlCheckpointState<S>,
    ) -> Result<(), RlEarlyStoppingError> {
        let Some(current) = self.get_monitor_value(runtime)? else {
            return Ok(());
        };
        if self.config.restore_best_weights
            && self.state.best_weights.require("best_weights")?.is_none()
        {
            self.state.best_weights = RlCheckpointField::Present(self.capture(vessel)?);
        }
        let mut wait: BigInt = self.state.wait.require("wait")? + 1;
        self.state.wait = RlCheckpointField::Present(wait.clone());
        let mut best = *self.state.best.require("best")?;
        if self.is_improvement(current, best) {
            best = current;
            self.state.best = RlCheckpointField::Present(best);
            self.state.best_iter = RlCheckpointField::Present(iteration(runtime)?);
            if self.config.restore_best_weights {
                self.state.best_weights = RlCheckpointField::Present(self.capture(vessel)?);
            }
            if self
                .config
                .baseline
                .is_none_or(|baseline| self.is_improvement(current, baseline))
            {
                wait = BigInt::default();
                self.state.wait = RlCheckpointField::Present(wait.clone());
            }
        }
        // No plugin can mutate the borrowed callback between these reads. Retain the local
        // scalar values instead of introducing fallible rereads of already initialized state.
        let current_iter = iteration(runtime)?;
        let best_iter = self.state.best_iter.require("best_iter")?;
        let message = format!(
            "#{current_iter} current reward: {}, best reward: {} in #{best_iter}",
            format!("{current:.4}").to_lowercase(),
            format!("{best:.4}").to_lowercase()
        );
        emit(&mut self.logger, RlEarlyStoppingLogLevel::Info, &message)?;
        let stop_iteration = if wait >= self.config.patience {
            // Eligibility, stop assignment and message iteration have no intervening plugin
            // call. Keep them under one lock to avoid check/update races.
            runtime.update(|state| {
                let current = state
                    .current_iter
                    .as_ref()
                    .ok_or(RlEarlyStoppingError::MissingTrainerField("current_iter"))?;
                if *current > BigInt::default() {
                    state.should_stop = Some(true);
                    Ok::<_, RlEarlyStoppingError>(Some(current + 1))
                } else {
                    Ok(None)
                }
            })??
        } else {
            None
        };
        if let Some(next) = stop_iteration {
            emit(
                &mut self.logger,
                RlEarlyStoppingLogLevel::Info,
                &format!("On iteration {next}: early stopping"),
            )?;
            if let (true, RlCheckpointField::Present(Some(weights))) =
                (self.config.restore_best_weights, &self.state.best_weights)
            {
                emit(
                    &mut self.logger,
                    RlEarlyStoppingLogLevel::Info,
                    &format!(
                        "Restoring model weights from the end of the best iteration: {}",
                        best_iter + 1
                    ),
                )?;
                vessel.load_checkpoint(weights).map_err(|message| {
                    RlEarlyStoppingError::Plugin {
                        stage: "load",
                        message,
                    }
                })?;
            }
        }
        Ok(())
    }
}

impl<S: Clone, D, L: RlEarlyStoppingLogger> RlCheckpointState<RlEarlyStoppingState<S>>
    for RlEarlyStopping<S, D, L>
{
    fn save_checkpoint(&mut self) -> Result<RlEarlyStoppingState<S>, String> {
        self.state_dict().map_err(|error| error.to_string())
    }
    fn load_checkpoint(&mut self, state: &RlEarlyStoppingState<S>) -> Result<(), String> {
        self.load_state_dict(state)
            .map_err(|error| error.to_string())
    }
}
impl<
    S,
    D: RlCheckpointSnapshot<S>,
    L: RlEarlyStoppingLogger,
    V: RlCheckpointState<S>,
    M: Copy + Into<Option<f64>>,
> RlTrainerCallback<V, M> for RlEarlyStopping<S, D, L>
{
    fn call(
        &mut self,
        hook: RlTrainerHook,
        control: &mut RlTrainerControl<M>,
        vessel: &mut V,
    ) -> Result<(), RlTrainerDriverError> {
        match hook {
            RlTrainerHook::FitStart => {
                self.on_fit_start();
                Ok(())
            }
            RlTrainerHook::ValidateEnd => self.on_validate_end(&control.runtime, vessel),
            _ => Ok(()),
        }
        .map_err(|error| RlTrainerDriverError::Plugin {
            stage: "early_stopping".into(),
            message: error.to_string(),
        })
    }
}

#[cfg(test)]
#[path = "../tests/support/rl_early_stopping.rs"]
mod tests;
