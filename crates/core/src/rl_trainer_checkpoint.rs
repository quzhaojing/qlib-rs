//! Typed, ordered Trainer checkpoint graph traversal, independent of a file/model codec.

use indexmap::IndexMap;
use num_bigint::BigInt;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

use crate::{
    RlTrainerControl, RlTrainerDriverError, RlTrainerRestore, RlTrainerRuntime, RlTrainerStateError,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RlCheckpointOperation {
    Save,
    Load,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum RlTrainerCheckpointError {
    #[error(transparent)]
    Runtime(#[from] RlTrainerStateError),
    #[error("checkpoint field is missing: {0}")]
    MissingField(String),
    #[error("trainer {0} has not been initialized")]
    Uninitialized(&'static str),
    #[error("{operation:?} checkpoint component {component} failed: {message}")]
    Component {
        operation: RlCheckpointOperation,
        component: String,
        message: String,
    },
}

/// Missing differs from a present null/optional payload. JSON missing fields stay missing
/// until traversal reaches them, preserving Qlib's partial-restore failure order.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum RlCheckpointField<T> {
    #[default]
    Missing,
    Present(T),
}

impl<T> RlCheckpointField<T> {
    #[must_use]
    pub const fn is_missing(&self) -> bool {
        matches!(self, Self::Missing)
    }

    /// # Errors
    /// Returns the requested missing-field path without interpreting a present payload.
    pub fn require(&self, path: &str) -> Result<&T, RlTrainerCheckpointError> {
        match self {
            Self::Missing => Err(RlTrainerCheckpointError::MissingField(path.into())),
            Self::Present(value) => Ok(value),
        }
    }
}

impl<T: Serialize> Serialize for RlCheckpointField<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Missing => Err(serde::ser::Error::custom(
                "a missing checkpoint field has no standalone value",
            )),
            Self::Present(value) => value.serialize(serializer),
        }
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for RlCheckpointField<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        T::deserialize(deserializer).map(Self::Present)
    }
}

/// Generic payload types preserve model-owned values. No Clone/Serialize requirement is
/// imposed on component state by traversal. Runtime metric snapshots use Clone explicitly.
/// JSON supports partial documents; positional formats such as Bincode require all fields.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound(
    deserialize = "V: Deserialize<'de>, C: Deserialize<'de>, L: Deserialize<'de>, M: Deserialize<'de>"
))]
pub struct RlTrainerCheckpoint<V, C, L, M = f64> {
    #[serde(default, skip_serializing_if = "RlCheckpointField::is_missing")]
    pub vessel: RlCheckpointField<V>,
    #[serde(default, skip_serializing_if = "RlCheckpointField::is_missing")]
    pub callbacks: RlCheckpointField<IndexMap<String, C>>,
    #[serde(default, skip_serializing_if = "RlCheckpointField::is_missing")]
    pub loggers: RlCheckpointField<IndexMap<String, L>>,
    #[serde(default, skip_serializing_if = "RlCheckpointField::is_missing")]
    pub should_stop: RlCheckpointField<bool>,
    #[serde(default, skip_serializing_if = "RlCheckpointField::is_missing")]
    pub current_iter: RlCheckpointField<BigInt>,
    #[serde(default, skip_serializing_if = "RlCheckpointField::is_missing")]
    pub current_episode: RlCheckpointField<BigInt>,
    #[serde(default, skip_serializing_if = "RlCheckpointField::is_missing")]
    pub current_stage: RlCheckpointField<String>,
    #[serde(default, skip_serializing_if = "RlCheckpointField::is_missing")]
    pub metrics: RlCheckpointField<IndexMap<String, M>>,
}

impl<V, C, L, M> Default for RlTrainerCheckpoint<V, C, L, M> {
    fn default() -> Self {
        Self {
            vessel: RlCheckpointField::Missing,
            callbacks: RlCheckpointField::Missing,
            loggers: RlCheckpointField::Missing,
            should_stop: RlCheckpointField::Missing,
            current_iter: RlCheckpointField::Missing,
            current_episode: RlCheckpointField::Missing,
            current_stage: RlCheckpointField::Missing,
            metrics: RlCheckpointField::Missing,
        }
    }
}

/// A stateful component owns checkpoint representation and interior sharing semantics.
pub trait RlCheckpointState<S> {
    /// # Errors
    /// Returns save failures, potentially after component-owned side effects.
    fn save_checkpoint(&mut self) -> Result<S, String>;
    /// # Errors
    /// Returns restore failures without rolling back component-owned mutations.
    fn load_checkpoint(&mut self, state: &S) -> Result<(), String>;
}

/// Borrow existing live components; the graph does not construct duplicate plugin instances.
pub struct RlNamedCheckpointComponent<'a, S> {
    pub type_name: String,
    pub state: &'a mut dyn RlCheckpointState<S>,
}

/// Reproduces `_named_collection`, including collisions between numeric suffixes and real
/// type names: the last item wins but the first insertion position remains unchanged.
#[must_use]
pub fn named_rl_checkpoint_indices<'a>(
    names: impl IntoIterator<Item = &'a str>,
) -> IndexMap<String, usize> {
    let mut counts = IndexMap::<String, usize>::new();
    let mut result = IndexMap::new();
    for (index, name) in names.into_iter().enumerate() {
        let name = name.to_lowercase();
        let count = counts.entry(name.clone()).or_default();
        let key = if *count == 0 {
            name
        } else {
            format!("{name}{count}")
        };
        *count += 1;
        result.insert(key, index);
    }
    result
}

fn save_named<S>(
    components: &mut [RlNamedCheckpointComponent<'_, S>],
    group: &str,
) -> Result<IndexMap<String, S>, RlTrainerCheckpointError> {
    let indices =
        named_rl_checkpoint_indices(components.iter().map(|item| item.type_name.as_str()));
    let mut output = IndexMap::new();
    for (name, index) in indices {
        let value = components[index]
            .state
            .save_checkpoint()
            .map_err(|message| RlTrainerCheckpointError::Component {
                operation: RlCheckpointOperation::Save,
                component: format!("{group}.{name}"),
                message,
            })?;
        output.insert(name, value);
    }
    Ok(output)
}

fn load_named<S>(
    components: &mut [RlNamedCheckpointComponent<'_, S>],
    state: &RlCheckpointField<IndexMap<String, S>>,
    group: &str,
) -> Result<(), RlTrainerCheckpointError> {
    let indices =
        named_rl_checkpoint_indices(components.iter().map(|item| item.type_name.as_str()));
    for (name, index) in indices {
        let path = format!("{group}.{name}");
        let value = state
            .require(group)?
            .get(&name)
            .ok_or_else(|| RlTrainerCheckpointError::MissingField(path.clone()))?;
        components[index]
            .state
            .load_checkpoint(value)
            .map_err(|message| RlTrainerCheckpointError::Component {
                operation: RlCheckpointOperation::Load,
                component: path,
                message,
            })?;
    }
    Ok(())
}

/// Save graph components before checking Trainer initialization, preserving side effects.
/// # Errors
/// Returns the first component, runtime-lock, or uninitialized metadata error.
pub fn save_rl_trainer_checkpoint<V, C, L, M: Clone>(
    runtime: &RlTrainerRuntime<M>,
    vessel: &mut dyn RlCheckpointState<V>,
    callbacks: &mut [RlNamedCheckpointComponent<'_, C>],
    loggers: &mut [RlNamedCheckpointComponent<'_, L>],
) -> Result<RlTrainerCheckpoint<V, C, L, M>, RlTrainerCheckpointError> {
    let vessel =
        vessel
            .save_checkpoint()
            .map_err(|message| RlTrainerCheckpointError::Component {
                operation: RlCheckpointOperation::Save,
                component: "vessel".into(),
                message,
            })?;
    let callbacks = save_named(callbacks, "callbacks")?;
    let loggers = save_named(loggers, "loggers")?;
    runtime.read(|state| {
        Ok(RlTrainerCheckpoint {
            vessel: RlCheckpointField::Present(vessel),
            callbacks: RlCheckpointField::Present(callbacks),
            loggers: RlCheckpointField::Present(loggers),
            should_stop: RlCheckpointField::Present(
                state
                    .should_stop
                    .ok_or(RlTrainerCheckpointError::Uninitialized("should_stop"))?,
            ),
            current_iter: RlCheckpointField::Present(
                state
                    .current_iter
                    .clone()
                    .ok_or(RlTrainerCheckpointError::Uninitialized("current_iter"))?,
            ),
            current_episode: RlCheckpointField::Present(
                state
                    .current_episode
                    .clone()
                    .ok_or(RlTrainerCheckpointError::Uninitialized("current_episode"))?,
            ),
            current_stage: RlCheckpointField::Present(state.current_stage.clone()),
            metrics: RlCheckpointField::Present(
                state
                    .metrics
                    .clone()
                    .ok_or(RlTrainerCheckpointError::Uninitialized("metrics"))?,
            ),
        })
    })?
}

/// Restore graph components, then assign metadata in source order. A later missing field
/// retains earlier assignments; this deliberately is not an all-or-nothing transaction.
/// # Errors
/// Returns the first missing field, component failure, or poisoned runtime error.
pub fn load_rl_trainer_checkpoint<V, C, L, M: Clone>(
    runtime: &RlTrainerRuntime<M>,
    vessel: &mut dyn RlCheckpointState<V>,
    callbacks: &mut [RlNamedCheckpointComponent<'_, C>],
    loggers: &mut [RlNamedCheckpointComponent<'_, L>],
    checkpoint: &RlTrainerCheckpoint<V, C, L, M>,
) -> Result<(), RlTrainerCheckpointError> {
    vessel
        .load_checkpoint(checkpoint.vessel.require("vessel")?)
        .map_err(|message| RlTrainerCheckpointError::Component {
            operation: RlCheckpointOperation::Load,
            component: "vessel".into(),
            message,
        })?;
    load_named(callbacks, &checkpoint.callbacks, "callbacks")?;
    load_named(loggers, &checkpoint.loggers, "loggers")?;
    runtime.update(|state| {
        state.should_stop = Some(*checkpoint.should_stop.require("should_stop")?);
        state.current_iter = Some(checkpoint.current_iter.require("current_iter")?.clone());
        state.current_episode = Some(
            checkpoint
                .current_episode
                .require("current_episode")?
                .clone(),
        );
        state
            .current_stage
            .clone_from(checkpoint.current_stage.require("current_stage")?);
        state.metrics = Some(checkpoint.metrics.require("metrics")?.clone());
        Ok(())
    })?
}

/// Use live component handles also registered with the driver's callbacks/loggers. File I/O
/// belongs upstream; this adapter restores an already decoded typed document after attachment.
pub struct RlTrainerCheckpointRestore<'a, 'state, V, C, L, M = f64> {
    pub checkpoint: &'a RlTrainerCheckpoint<V, C, L, M>,
    pub callbacks: &'a mut [RlNamedCheckpointComponent<'state, C>],
    pub loggers: &'a mut [RlNamedCheckpointComponent<'state, L>],
}

impl<W, V, C, L, M: Clone> RlTrainerRestore<W, M> for RlTrainerCheckpointRestore<'_, '_, V, C, L, M>
where
    W: RlCheckpointState<V>,
{
    fn restore(
        &mut self,
        control: &mut RlTrainerControl<M>,
        vessel: &mut W,
    ) -> Result<(), RlTrainerDriverError> {
        load_rl_trainer_checkpoint(
            &control.runtime,
            vessel,
            self.callbacks,
            self.loggers,
            self.checkpoint,
        )
        .map_err(|error| RlTrainerDriverError::Plugin {
            stage: "checkpoint_restore".into(),
            message: error.to_string(),
        })
    }
}

impl<S> RlCheckpointState<crate::TrainingVesselCheckpoint<S>> for crate::TrainingVesselState<S> {
    fn save_checkpoint(&mut self) -> Result<crate::TrainingVesselCheckpoint<S>, String> {
        self.state_dict().map_err(|error| error.to_string())
    }
    fn load_checkpoint(
        &mut self,
        state: &crate::TrainingVesselCheckpoint<S>,
    ) -> Result<(), String> {
        self.load_state_dict(state)
            .map_err(|error| error.to_string())
    }
}

impl<V: Clone, H: crate::RlLogWriterHooks<V>> RlCheckpointState<crate::RlLogWriterState<V>>
    for crate::RlLogWriter<V, H>
{
    fn save_checkpoint(&mut self) -> Result<crate::RlLogWriterState<V>, String> {
        Ok(self.state_dict().clone())
    }
    fn load_checkpoint(&mut self, state: &crate::RlLogWriterState<V>) -> Result<(), String> {
        self.load_state_dict(state.clone());
        Ok(())
    }
}

/// Explicit writer+buffer envelope, not a byte-compatible Python pickle representation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RlLogBufferCheckpoint<V> {
    pub writer: crate::RlLogWriterState<V>,
    pub buffer: crate::RlLogBufferState,
}

impl<V: Clone, C> RlCheckpointState<RlLogBufferCheckpoint<V>> for crate::RlLogBuffer<V, C>
where
    C: FnMut(
        crate::RlLogBufferEvent,
        &crate::RlLogWriterState<V>,
        &crate::RlLogBufferState,
    ) -> Result<(), String>,
{
    fn save_checkpoint(&mut self) -> Result<RlLogBufferCheckpoint<V>, String> {
        Ok(RlLogBufferCheckpoint {
            writer: self.state_dict().clone(),
            buffer: self.buffer_state().clone(),
        })
    }
    fn load_checkpoint(&mut self, state: &RlLogBufferCheckpoint<V>) -> Result<(), String> {
        self.load_buffer_state(state.buffer.clone(), state.writer.clone());
        Ok(())
    }
}

#[cfg(test)]
#[path = "../tests/support/rl_trainer_checkpoint.rs"]
mod tests;
