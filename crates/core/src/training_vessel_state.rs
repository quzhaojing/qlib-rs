//! Non-owning trainer attachment and policy checkpoint delegation for training vessels.

use std::sync::{Arc, Weak};

use num_bigint::BigInt;
use serde::{Deserialize, Serialize};
use strum::Display;
use thiserror::Error;

/// Read-only view into the live trainer. Implementations own their synchronization strategy.
pub trait TrainingTrainerView: Send + Sync {
    /// # Errors
    /// Returns a trainer-specific access failure.
    fn current_iteration(&self) -> Result<BigInt, String>;

    /// # Errors
    /// Returns a trainer-specific access failure.
    fn fast_dev_run(&self) -> Result<Option<i64>, String>;
}

#[derive(Clone, Copy, Debug, Display, Eq, PartialEq)]
#[strum(serialize_all = "snake_case")]
pub enum TrainingTrainerField {
    CurrentIteration,
    FastDevRun,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum TrainingVesselBindingError {
    #[error("training vessel has no assigned trainer")]
    Unassigned,
    #[error("training vessel's assigned trainer has been dropped")]
    Expired,
    #[error("failed to read trainer {field}: {message}")]
    Access {
        field: TrainingTrainerField,
        message: String,
    },
}

/// A binding does not keep its trainer alive, matching Python's `weakref.proxy`.
#[derive(Default)]
pub struct TrainingVesselBinding {
    trainer: Option<Weak<dyn TrainingTrainerView>>,
}

impl TrainingVesselBinding {
    pub fn assign_trainer(&mut self, trainer: &Arc<dyn TrainingTrainerView>) {
        self.trainer = Some(Arc::downgrade(trainer));
    }

    /// Temporarily pins the trainer during an operation; the binding itself retains no ownership.
    ///
    /// # Errors
    /// Returns an error if no trainer was assigned or the assigned trainer has been dropped.
    pub fn trainer(&self) -> Result<Arc<dyn TrainingTrainerView>, TrainingVesselBindingError> {
        self.trainer
            .as_ref()
            .ok_or(TrainingVesselBindingError::Unassigned)?
            .upgrade()
            .ok_or(TrainingVesselBindingError::Expired)
    }

    /// # Errors
    /// Returns binding or live-trainer access failures.
    pub fn current_iteration(&self) -> Result<BigInt, TrainingVesselBindingError> {
        self.trainer()?
            .current_iteration()
            .map_err(|message| TrainingVesselBindingError::Access {
                field: TrainingTrainerField::CurrentIteration,
                message,
            })
    }

    /// # Errors
    /// Returns binding or live-trainer access failures.
    pub fn fast_dev_run(&self) -> Result<Option<i64>, TrainingVesselBindingError> {
        self.trainer()?
            .fast_dev_run()
            .map_err(|message| TrainingVesselBindingError::Access {
                field: TrainingTrainerField::FastDevRun,
                message,
            })
    }
}

/// Policy-owned state format: tensors, optimizer state, and serialization belong to the adapter.
pub trait TrainingPolicyState<State>: Send {
    /// # Errors
    /// Returns a policy-specific checkpoint failure.
    fn state_dict(&mut self) -> Result<State, String>;

    /// # Errors
    /// Returns a policy-specific restoration failure; partial mutation remains policy-owned.
    fn load_state_dict(&mut self, state: &State) -> Result<(), String>;
}

/// The exact vessel-level checkpoint envelope. No trainer or seed state is added here.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(bound(deserialize = "State: Deserialize<'de>"))]
pub struct TrainingVesselCheckpoint<State> {
    #[serde(deserialize_with = "State::deserialize")]
    pub policy: State,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum TrainingVesselStateError {
    #[error("training vessel policy checkpoint failed: {0}")]
    Save(String),
    #[error("training vessel policy restoration failed: {0}")]
    Load(String),
}

/// Delegates policy state without cloning, coercing, or interpreting its payload.
pub struct TrainingVesselState<State> {
    policy: Box<dyn TrainingPolicyState<State>>,
}

impl<State> TrainingVesselState<State> {
    #[must_use]
    pub fn new(policy: Box<dyn TrainingPolicyState<State>>) -> Self {
        Self { policy }
    }

    /// # Errors
    /// Returns the policy's save failure.
    pub fn state_dict(
        &mut self,
    ) -> Result<TrainingVesselCheckpoint<State>, TrainingVesselStateError> {
        let policy = self
            .policy
            .state_dict()
            .map_err(TrainingVesselStateError::Save)?;
        Ok(TrainingVesselCheckpoint { policy })
    }

    /// # Errors
    /// Returns the policy's load failure without rolling back its partial mutation.
    pub fn load_state_dict(
        &mut self,
        checkpoint: &TrainingVesselCheckpoint<State>,
    ) -> Result<(), TrainingVesselStateError> {
        self.policy
            .load_state_dict(&checkpoint.policy)
            .map_err(TrainingVesselStateError::Load)
    }
}

#[cfg(test)]
#[path = "../tests/support/training_vessel_state.rs"]
mod tests;
