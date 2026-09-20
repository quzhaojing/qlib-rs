//! Concrete wall clocks, filesystem codecs, and live Trainer checkpoint graph adapters.

use std::{
    fs::{self, File},
    io::{Read, Write},
    marker::PhantomData,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Serialize, de::DeserializeOwned};

use crate::rl_checkpoint_callback::{RlCheckpointClock, RlCheckpointGraph, RlCheckpointStorage};
use crate::{
    RlCheckpointState, RlNamedCheckpointComponent, RlTrainerCheckpoint, RlTrainerControl,
    RlTrainerDriverError, RlTrainerRestore, load_rl_trainer_checkpoint, save_rl_trainer_checkpoint,
};

pub struct SystemRlCheckpointClock;

fn timestamp(time: SystemTime) -> f64 {
    match time.duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_secs_f64(),
        Err(error) => -error.duration().as_secs_f64(),
    }
}

impl RlCheckpointClock for SystemRlCheckpointClock {
    fn timestamp(&mut self) -> Result<f64, String> {
        Ok(timestamp(SystemTime::now()))
    }
    fn local_time(&mut self) -> Result<String, String> {
        Ok(chrono::Local::now().format("%Y%m%d%H%M%S").to_string())
    }
}

pub trait RlCheckpointFileCodec<S> {
    /// # Errors
    /// Returns serialization or output errors. The caller has already opened the file.
    fn encode(&mut self, state: &S, output: &mut dyn Write) -> Result<(), String>;
    /// # Errors
    /// Returns I/O, corruption or incompatible-schema errors before graph restoration.
    fn decode(&mut self, input: &mut dyn Read) -> Result<S, String>;
}

/// Explicitly a Bincode 1.x Rust DTO codec, NOT a Torch/pickle codec. Complete graph DTOs
/// are required because this is a positional format. Only load trusted, bounded files.
pub struct BincodeRlCheckpointCodec;
impl<S: Serialize + DeserializeOwned> RlCheckpointFileCodec<S> for BincodeRlCheckpointCodec {
    fn encode(&mut self, state: &S, output: &mut dyn Write) -> Result<(), String> {
        bincode::serialize_into(output, state).map_err(|error| error.to_string())
    }
    fn decode(&mut self, input: &mut dyn Read) -> Result<S, String> {
        bincode::deserialize_from(input).map_err(|error| error.to_string())
    }
}

pub struct FileRlCheckpointStorage<C> {
    pub codec: C,
}

impl<C> FileRlCheckpointStorage<C> {
    #[must_use]
    pub const fn new(codec: C) -> Self {
        Self { codec }
    }

    /// # Errors
    /// Returns open/decode errors. Does not mutate a Trainer or inspect the file suffix.
    pub fn load<S>(&mut self, path: &Path) -> Result<S, String>
    where
        C: RlCheckpointFileCodec<S>,
    {
        let mut file = File::open(path).map_err(|error| error.to_string())?;
        self.codec.decode(&mut file)
    }
}

impl<S, C: RlCheckpointFileCodec<S>> RlCheckpointStorage<S> for FileRlCheckpointStorage<C> {
    fn create_directory(&mut self, path: &Path) -> Result<(), String> {
        fs::create_dir_all(path).map_err(|error| error.to_string())
    }
    fn save(&mut self, state: &S, path: &Path) -> Result<(), String> {
        let mut file = File::create(path).map_err(|error| error.to_string())?;
        self.codec.encode(state, &mut file)
    }
    fn exists(&mut self, path: &Path) -> Result<bool, String> {
        Ok(path.exists())
    }
    fn is_link(&mut self, path: &Path) -> Result<bool, String> {
        Ok(path.is_symlink())
    }
    fn remove(&mut self, path: &Path) -> Result<(), String> {
        fs::remove_file(path).map_err(|error| error.to_string())
    }
    fn link(&mut self, target: &Path, link: &Path) -> Result<(), String> {
        #[cfg(unix)]
        use std::os::unix::fs::symlink;
        #[cfg(windows)]
        use std::os::windows::fs::symlink_file as symlink;
        symlink(target, link).map_err(|error| error.to_string())
    }
    fn copy(&mut self, source: &Path, destination: &Path) -> Result<(), String> {
        fs::copy(source, destination)
            .map(|_| ())
            .map_err(|error| error.to_string())
    }
}

/// Owns a state adapter, which must access the same live component registered with the
/// Trainer. A shared-handle adapter is appropriate; an independent cloned model is not.
pub struct RlOwnedCheckpointComponent<S> {
    pub type_name: String,
    pub state: Box<dyn RlCheckpointState<S>>,
}

fn borrowed<S>(
    components: &mut [RlOwnedCheckpointComponent<S>],
) -> Vec<RlNamedCheckpointComponent<'_, S>> {
    components
        .iter_mut()
        .map(|component| RlNamedCheckpointComponent {
            type_name: component.type_name.clone(),
            state: component.state.as_mut(),
        })
        .collect()
}

/// Owns adapters for live callbacks/loggers; the vessel is borrowed from hook dispatch.
/// Use the executing Checkpoint's unit-state adapter rather than recursively locking its
/// callback object. Heterogeneous states can use a caller-defined Serde enum.
pub struct RlTrainerGraph<V, C, L> {
    pub callbacks: Vec<RlOwnedCheckpointComponent<C>>,
    pub loggers: Vec<RlOwnedCheckpointComponent<L>>,
    vessel_state: PhantomData<fn() -> V>,
}

impl<V, C, L> RlTrainerGraph<V, C, L> {
    #[must_use]
    pub const fn new(
        callbacks: Vec<RlOwnedCheckpointComponent<C>>,
        loggers: Vec<RlOwnedCheckpointComponent<L>>,
    ) -> Self {
        Self {
            callbacks,
            loggers,
            vessel_state: PhantomData,
        }
    }

    /// # Errors
    /// Returns ordered component/runtime failures without undoing earlier assignments.
    pub fn restore<W: RlCheckpointState<V>, M: Clone>(
        &mut self,
        control: &mut RlTrainerControl<M>,
        vessel: &mut W,
        state: &RlTrainerCheckpoint<V, C, L, M>,
    ) -> Result<(), String> {
        load_rl_trainer_checkpoint(
            &control.runtime,
            vessel,
            &mut borrowed(&mut self.callbacks),
            &mut borrowed(&mut self.loggers),
            state,
        )
        .map_err(|error| error.to_string())
    }
}

impl<W: RlCheckpointState<V>, M: Clone, V, C, L> RlCheckpointGraph<W, M>
    for RlTrainerGraph<V, C, L>
{
    type State = RlTrainerCheckpoint<V, C, L, M>;

    fn collect(
        &mut self,
        control: &mut RlTrainerControl<M>,
        vessel: &mut W,
    ) -> Result<Self::State, String> {
        save_rl_trainer_checkpoint(
            &control.runtime,
            vessel,
            &mut borrowed(&mut self.callbacks),
            &mut borrowed(&mut self.loggers),
        )
        .map_err(|error| error.to_string())
    }
}

/// Decode the selected file after vessel attachment, then restore all live graph groups
/// before the driver calls `FitStart`. No inferred Torch decoding or format auto-detection.
pub struct RlCheckpointFileRestore<'a, F, V, C, L> {
    pub path: PathBuf,
    pub storage: &'a mut FileRlCheckpointStorage<F>,
    pub graph: &'a mut RlTrainerGraph<V, C, L>,
}

impl<W: RlCheckpointState<V>, M: Clone, F, V, C, L> RlTrainerRestore<W, M>
    for RlCheckpointFileRestore<'_, F, V, C, L>
where
    F: RlCheckpointFileCodec<RlTrainerCheckpoint<V, C, L, M>>,
{
    fn restore(
        &mut self,
        control: &mut RlTrainerControl<M>,
        vessel: &mut W,
    ) -> Result<(), RlTrainerDriverError> {
        let mut restore = || {
            let state = self.storage.load(&self.path)?;
            self.graph.restore(control, vessel, &state)
        };
        // The closure sequences decode before any component mutation.
        restore().map_err(|message| RlTrainerDriverError::Plugin {
            stage: "checkpoint_file_restore".into(),
            message,
        })
    }
}

#[cfg(test)]
#[path = "../tests/support/rl_checkpoint_file.rs"]
mod tests;
