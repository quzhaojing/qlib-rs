//! Qlib's one-retry policy-weight loading protocol, independent of an ML runtime.

use indexmap::IndexMap;
use std::sync::Arc;
use thiserror::Error;

/// Classify failures by behavior, not by matching a backend's diagnostic text.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum PolicyWeightLoadError {
    #[error("{0}")]
    Runtime(String),
    #[error("{0}")]
    Other(String),
}

/// Ordered in-memory weights with shared values and caller-owned module metadata.
/// Neither the values nor metadata need `Clone`; a retry aliases existing values.
/// This is a logical plugin boundary, not a file format or a native dynamic ABI.
pub struct PolicyWeights<Weight, Metadata = ()> {
    pub weights: IndexMap<String, Arc<Weight>>,
    pub metadata: Metadata,
}

/// A policy loader may retain partial model, mapping and metadata mutations on error.
pub trait PolicyWeightLoader<Weight, Metadata = ()> {
    /// # Errors
    /// Distinguishes retryable runtime failures from errors that must propagate.
    fn load_weights(
        &mut self,
        state: &mut PolicyWeights<Weight, Metadata>,
    ) -> Result<(), PolicyWeightLoadError>;
}

/// Run `policy.set_weight`: one load, and exactly one converted retry on runtime error.
///
/// Conversion snapshots keys after the failed call, but reads each value from the
/// live map. Existing prefixed keys are overwritten in place, including values
/// that a later iteration will read. Original keys and shared identities remain.
/// Metadata is neither cloned nor interpreted. A second failure does not roll back
/// the first load, the mapping conversion, or the second load's partial effects.
///
/// # Errors
/// Returns the first non-runtime error or the second load's error unchanged.
pub fn set_policy_weights<Weight, Metadata>(
    policy: &mut dyn PolicyWeightLoader<Weight, Metadata>,
    state: &mut PolicyWeights<Weight, Metadata>,
) -> Result<(), PolicyWeightLoadError> {
    match policy.load_weights(state) {
        Err(PolicyWeightLoadError::Runtime(_)) => {
            let keys = state.weights.keys().cloned().collect::<Vec<_>>();
            for key in keys {
                let value = Arc::clone(&state.weights[&key]);
                state.weights.insert(format!("_actor_critic.{key}"), value);
            }
            policy.load_weights(state)
        }
        result => result,
    }
}

#[cfg(test)]
#[path = "../tests/support/rl_policy_weight.rs"]
mod tests;
