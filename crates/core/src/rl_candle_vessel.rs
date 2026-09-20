//! Connect the default training-vessel protocol to the native Candle PPO policy.
//! Arrow owns metric interchange; `serde_json` owns the default keyword value model.

use crate::TrainingPolicyState;
use crate::rl_candle_categorical::CategoricalForward;
use crate::rl_candle_checkpoint::CandlePolicySnapshot;
use crate::rl_candle_network::RecurrentObservation;
use crate::rl_candle_policy::{CandlePpo, CandlePpoError, CandlePpoReplay, CandlePpoScheduler};
use crate::rl_policy_batch::minibatch_indices;
use crate::{
    TrainingMetricValue, TrainingPolicyMode, TrainingRunPolicy, TrainingUpdateOptions,
    TrainingVesselMetrics, TrainingVesselRunError,
};
use arrow_array::Float64Array;
use rand::RngCore;
use serde_json::Value;
use std::sync::Arc;

fn option_error(message: impl Into<String>) -> CandlePpoError {
    CandlePpoError::Adapter {
        stage: "learn options",
        message: message.into(),
    }
}

// Python's range accepts integers (including bool), not integer-valued floats.
// Negative repeats do nothing. Larger-than-native integers are explicit boundary
// errors, not silently narrowed or rounded JSON numbers.
fn repeat_count(value: &Value) -> Result<usize, CandlePpoError> {
    if let Some(value) = value.as_bool() {
        return Ok(usize::from(value));
    }
    if value.as_i64().is_some_and(|value| value < 0) {
        return Ok(0);
    }
    let value = value
        .as_u64()
        .ok_or_else(|| option_error("repeat must be an integer"))?;
    usize::try_from(value).map_err(|_| option_error("repeat exceeds native index range"))
}

fn learn_options(
    options: &TrainingUpdateOptions,
    length: usize,
    rng: &mut dyn RngCore,
) -> Result<(usize, usize), CandlePpoError> {
    let batch = options
        .get("batch_size")
        .ok_or_else(|| option_error("missing batch_size"))?;
    let repeat = options
        .get("repeat")
        .ok_or_else(|| option_error("missing repeat"))?;
    let repeat = repeat_count(repeat)?;
    if repeat == 0 {
        return Ok((0, 0));
    }
    let positive = match batch {
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_some_and(|value| value >= 1.),
        _ => return Err(option_error("batch_size must be comparable to an integer")),
    };
    if !positive {
        return Err(option_error("batch_size must be at least one"));
    }
    if let Some(value) = batch.as_bool() {
        return Ok((usize::from(value), repeat));
    }
    if let Some(value) = batch.as_u64() {
        return Ok((
            usize::try_from(value)
                .map_err(|_| option_error("batch_size exceeds native index range"))?,
            repeat,
        ));
    }
    // Source Batch.split checks 1 <= size, shuffles, then range rejects a float.
    // Reuse the same native permutation operation to preserve that RNG side effect.
    minibatch_indices(length, 1, true, true, rng)?;
    Err(option_error("batch_size must be an integer"))
}

/// A reusable native policy runtime with caller-supplied RNG and optional scheduler.
/// Forward and vessel updates share the same policy, optimizer and RNG instance.
/// This adapter implements the default JSON keyword boundary; custom opaque
/// `TrainingUpdateOptions` value types can use a separate typed adapter.
pub struct CandlePpoVessel<R> {
    pub policy: CandlePpo,
    pub rng: R,
    pub scheduler: Option<Box<dyn CandlePpoScheduler + Send>>,
}

impl<R: RngCore> CandlePpoVessel<R> {
    #[must_use]
    pub fn new(policy: CandlePpo, rng: R) -> Self {
        Self {
            policy,
            rng,
            scheduler: None,
        }
    }

    /// # Errors
    /// Propagates observation/model/distribution errors without replacing caller state.
    pub fn forward<State>(
        &mut self,
        obs: &RecurrentObservation,
        state: State,
    ) -> Result<CategoricalForward<State>, CandlePpoError> {
        self.policy.forward(obs, state, &mut self.rng)
    }
}

impl<B: CandlePpoReplay, R: RngCore + Send> TrainingRunPolicy<B> for CandlePpoVessel<R> {
    fn set_mode(&mut self, mode: TrainingPolicyMode) -> Result<(), TrainingVesselRunError> {
        self.policy.set_mode(mode);
        Ok(())
    }

    fn update(
        &mut self,
        sample_size: u64,
        buffer: Option<&mut B>,
        options: &TrainingUpdateOptions,
    ) -> Result<TrainingVesselMetrics, TrainingVesselRunError> {
        for key in options.keys() {
            if matches!(key.as_str(), "sample_size" | "buffer") {
                return Err(TrainingVesselRunError::DuplicateKeyword(key.clone()));
            }
        }
        let scheduler = self
            .scheduler
            .as_mut()
            .map(|scheduler| scheduler.as_mut() as &mut dyn CandlePpoScheduler);
        let metrics = self
            .policy
            .update_with_options(
                buffer.map(|buffer| buffer as &mut dyn CandlePpoReplay),
                scheduler,
                sample_size,
                &mut self.rng,
                |length, rng| learn_options(options, length, rng),
            )
            .map_err(|error| TrainingVesselRunError::Plugin {
                stage: "PPO update".into(),
                message: error.to_string(),
            })?;
        Ok(metrics
            .into_iter()
            .map(|(name, values)| {
                (
                    name,
                    TrainingMetricValue::Numeric(Arc::new(Float64Array::from(values))),
                )
            })
            .collect())
    }
}

impl<R: Send> TrainingPolicyState<CandlePolicySnapshot<()>> for CandlePpoVessel<R> {
    fn state_dict(&mut self) -> Result<CandlePolicySnapshot<()>, String> {
        self.policy.state_dict()
    }

    fn load_state_dict(&mut self, state: &CandlePolicySnapshot<()>) -> Result<(), String> {
        self.policy.load_state_dict(state)
    }
}

#[cfg(test)]
#[path = "../tests/support/rl_candle_vessel.rs"]
mod tests;
