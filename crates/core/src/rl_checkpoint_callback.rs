//! Ordered Qlib Checkpoint scheduling over clock, naming, graph and storage plugins.

use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use num_bigint::BigInt;
use num_traits::FromPrimitive;
use thiserror::Error;

use crate::{
    RlCheckpointState, RlTrainerCallback, RlTrainerControl, RlTrainerDriverError, RlTrainerHook,
    RlTrainerRuntime, RlTrainerStateError,
};

/// Python accepts integer intervals without narrowing them to Float64; float inputs are
/// also accepted at runtime. Keep the distinction for comparisons near rounding boundaries.
#[derive(Clone, Debug, PartialEq)]
pub enum RlCheckpointTimeInterval {
    Integer(BigInt),
    Float(f64),
}

impl RlCheckpointTimeInterval {
    pub(crate) fn elapsed(&self, seconds: f64) -> bool {
        match self {
            Self::Float(interval) => seconds >= *interval,
            Self::Integer(interval) => match BigInt::from_f64(seconds.floor()) {
                Some(seconds) => seconds >= *interval,
                None => seconds == f64::INFINITY,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RlCheckpointConfig {
    pub dirpath: PathBuf,
    pub filename: String,
    /// None/empty disable latest updates. Other unrecognized strings retain Python's
    /// remove-without-replacement behavior instead of silently selecting a valid mode.
    pub save_latest: Option<String>,
    pub every_n_iters: Option<BigInt>,
    pub time_interval: Option<RlCheckpointTimeInterval>,
    pub save_on_fit_end: bool,
}

impl RlCheckpointConfig {
    #[must_use]
    pub fn new(dirpath: impl Into<PathBuf>) -> Self {
        Self {
            dirpath: dirpath.into(),
            filename: "{iter:03d}.pth".into(),
            save_latest: Some("link".into()),
            every_n_iters: None,
            time_interval: None,
            save_on_fit_end: true,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RlCheckpointCallbackState {
    pub last_name: Option<String>,
    pub last_iter: Option<BigInt>,
    pub last_time: Option<f64>,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum RlCheckpointCallbackError {
    #[error(transparent)]
    Runtime(#[from] RlTrainerStateError),
    #[error("trainer {0} has not been initialized")]
    MissingTrainerField(&'static str),
    #[error("checkpoint iteration interval is zero")]
    ZeroIterationInterval,
    #[error("checkpoint filename has a duplicate keyword: {0}")]
    DuplicateKeyword(&'static str),
    #[error("checkpoint {stage} failed: {message}")]
    Plugin {
        stage: &'static str,
        message: String,
    },
}

fn plugin<T>(
    stage: &'static str,
    result: Result<T, String>,
) -> Result<T, RlCheckpointCallbackError> {
    result.map_err(|message| RlCheckpointCallbackError::Plugin { stage, message })
}

pub trait RlCheckpointClock {
    /// Wall-clock seconds, not monotonic elapsed time.
    /// # Errors
    /// Returns a clock failure at the original source call site.
    fn timestamp(&mut self) -> Result<f64, String>;
    /// Local calendar time rendered as `YYYYmmddHHMMSS`. This is a separate clock read.
    /// # Errors
    /// Returns calendar lookup/formatting failures before filename assignment.
    fn local_time(&mut self) -> Result<String, String>;
}

pub trait RlCheckpointName<M> {
    /// Implements the caller's filename language and model-owned formatting semantics.
    /// # Errors
    /// Returns missing field, malformed format or value formatting failures.
    fn render(
        &mut self,
        template: &str,
        iteration: &BigInt,
        local_time: &str,
        metrics: &IndexMap<String, M>,
    ) -> Result<String, String>;
}

pub trait RlCheckpointGraph<V, M> {
    type State;
    /// Collect the complete live Trainer graph, not just model weights. Invoked after
    /// last-save metadata assignment and before the storage adapter opens the file.
    /// # Errors
    /// Returns component failures with preceding component effects retained.
    fn collect(
        &mut self,
        control: &mut RlTrainerControl<M>,
        vessel: &mut V,
    ) -> Result<Self::State, String>;
}

/// Storage owns the actual file/model codec. A Rust DTO encoding must not be advertised
/// as a Torch-compatible .pth encoding solely because the source filename has that suffix.
pub trait RlCheckpointStorage<S> {
    /// # Errors
    /// Returns directory creation errors; construction itself does not create directories.
    fn create_directory(&mut self, path: &Path) -> Result<(), String>;
    /// # Errors
    /// Returns codec/write errors, retaining actual partial effects.
    fn save(&mut self, state: &S, path: &Path) -> Result<(), String>;
    /// Follows symlinks as Python Path.exists does.
    /// # Errors
    /// Returns adapter failures during the latest-file existence check.
    fn exists(&mut self, path: &Path) -> Result<bool, String>;
    /// Recognizes dangling links without following them.
    /// # Errors
    /// Returns adapter failures; this check is skipped when exists returned true.
    fn is_link(&mut self, path: &Path) -> Result<bool, String>;
    /// # Errors
    /// Returns unlink failures before creating the replacement.
    fn remove(&mut self, path: &Path) -> Result<(), String>;
    /// Uses target text as given without canonicalizing relative paths.
    /// # Errors
    /// Returns symlink failures without falling back to copy.
    fn link(&mut self, target: &Path, link: &Path) -> Result<(), String>;
    /// # Errors
    /// Returns copy failures without restoring a previously removed latest file.
    fn copy(&mut self, source: &Path, destination: &Path) -> Result<(), String>;
}

pub struct RlCheckpointCallback<C, N, G, F> {
    pub config: RlCheckpointConfig,
    pub state: RlCheckpointCallbackState,
    pub clock: C,
    pub name: N,
    pub graph: G,
    pub storage: F,
}

impl<C, N, G, F> RlCheckpointCallback<C, N, G, F> {
    /// Retains unvalidated configuration; source failures occur at the relevant hook.
    #[must_use]
    pub fn new(config: RlCheckpointConfig, clock: C, name: N, graph: G, storage: F) -> Self {
        Self {
            config,
            state: RlCheckpointCallbackState::default(),
            clock,
            name,
            graph,
            storage,
        }
    }
}

fn iteration<M>(runtime: &RlTrainerRuntime<M>) -> Result<BigInt, RlCheckpointCallbackError> {
    runtime.read(|state| state.current_iter.clone())?.ok_or(
        RlCheckpointCallbackError::MissingTrainerField("current_iter"),
    )
}

impl<C: RlCheckpointClock, N, G, F> RlCheckpointCallback<C, N, G, F> {
    /// # Errors
    /// Returns the first scheduler, runtime, naming, graph or storage failure.
    pub fn on_hook<V, M: Clone>(
        &mut self,
        hook: RlTrainerHook,
        control: &mut RlTrainerControl<M>,
        vessel: &mut V,
    ) -> Result<(), RlCheckpointCallbackError>
    where
        N: RlCheckpointName<M>,
        G: RlCheckpointGraph<V, M>,
        F: RlCheckpointStorage<G::State>,
    {
        let due = match hook {
            RlTrainerHook::FitEnd => {
                self.config.save_on_fit_end
                    && Some(iteration(&control.runtime)?) != self.state.last_iter
            }
            RlTrainerHook::IterEnd => self.iteration_due(&control.runtime)?,
            _ => false,
        };
        if due {
            self.save(control, vessel)?;
        }
        Ok(())
    }

    fn iteration_due<M>(
        &mut self,
        runtime: &RlTrainerRuntime<M>,
    ) -> Result<bool, RlCheckpointCallbackError> {
        let mut due = false;
        if let Some(interval) = &self.config.every_n_iters {
            let next = iteration(runtime)? + 1;
            if *interval == BigInt::default() {
                return Err(RlCheckpointCallbackError::ZeroIterationInterval);
            }
            due = next % interval == BigInt::default();
        }
        if let Some(interval) = &self.config.time_interval {
            // Do not short-circuit this condition merely because iteration scheduling fired.
            if match self.state.last_time {
                None => true,
                Some(last) => interval.elapsed(plugin("time", self.clock.timestamp())? - last),
            } {
                due = true;
            }
        }
        Ok(due)
    }

    /// Direct save preserves Qlib's non-transactional state and file mutation order.
    /// # Errors
    /// Returns the first failure, retaining previous state assignments and plugin effects.
    pub fn save<V, M: Clone>(
        &mut self,
        control: &mut RlTrainerControl<M>,
        vessel: &mut V,
    ) -> Result<(), RlCheckpointCallbackError>
    where
        N: RlCheckpointName<M>,
        G: RlCheckpointGraph<V, M>,
        F: RlCheckpointStorage<G::State>,
    {
        plugin("mkdir", self.storage.create_directory(&self.config.dirpath))?;
        let name = self.new_name(&control.runtime)?;
        self.state.last_name = Some(name.clone());
        self.state.last_iter = Some(iteration(&control.runtime)?);
        self.state.last_time = Some(plugin("time", self.clock.timestamp())?);
        let graph = plugin("graph", self.graph.collect(control, vessel))?;
        let path = self.config.dirpath.join(name);
        plugin("save", self.storage.save(&graph, &path))?;
        let latest = self.config.dirpath.join("latest.pth");
        if self
            .config
            .save_latest
            .as_ref()
            .is_some_and(|s| !s.is_empty())
            && (plugin("exists", self.storage.exists(&latest))?
                || plugin("is_link", self.storage.is_link(&latest))?)
        {
            plugin("remove", self.storage.remove(&latest))?;
        }
        match self.config.save_latest.as_deref() {
            Some("link") => plugin("link", self.storage.link(&path, &latest))?,
            Some("copy") => plugin("copy", self.storage.copy(&path, &latest))?,
            _ => (),
        }
        Ok(())
    }

    /// # Errors
    /// Returns runtime, clock, duplicate-keyword or formatter failures without updating state.
    pub fn new_name<M: Clone>(
        &mut self,
        runtime: &RlTrainerRuntime<M>,
    ) -> Result<String, RlCheckpointCallbackError>
    where
        N: RlCheckpointName<M>,
    {
        let current = iteration(runtime)?;
        let time = plugin("local_time", self.clock.local_time())?;
        let metrics = runtime
            .read(|state| state.metrics.clone())?
            .ok_or(RlCheckpointCallbackError::MissingTrainerField("metrics"))?;
        for reserved in ["iter", "time"] {
            if metrics.contains_key(reserved) {
                return Err(RlCheckpointCallbackError::DuplicateKeyword(reserved));
            }
        }
        plugin(
            "format",
            self.name
                .render(&self.config.filename, &current, &time, &metrics),
        )
    }
}

impl<V, M: Clone, C, N, G, F> RlTrainerCallback<V, M> for RlCheckpointCallback<C, N, G, F>
where
    C: RlCheckpointClock,
    N: RlCheckpointName<M>,
    G: RlCheckpointGraph<V, M>,
    F: RlCheckpointStorage<G::State>,
{
    fn call(
        &mut self,
        hook: RlTrainerHook,
        control: &mut RlTrainerControl<M>,
        vessel: &mut V,
    ) -> Result<(), RlTrainerDriverError> {
        self.on_hook(hook, control, vessel)
            .map_err(|error| RlTrainerDriverError::Plugin {
                stage: "checkpoint".into(),
                message: error.to_string(),
            })
    }
}

// Inherited Callback state_dict/load_state_dict deliberately do not persist or reset
// scheduling metadata. A new object and a reused object therefore behave differently.
impl<C, N, G, F> RlCheckpointState<()> for RlCheckpointCallback<C, N, G, F> {
    fn save_checkpoint(&mut self) -> Result<(), String> {
        Ok(())
    }
    fn load_checkpoint(&mut self, (): &()) -> Result<(), String> {
        Ok(())
    }
}

#[cfg(test)]
#[path = "../tests/support/rl_checkpoint_callback.rs"]
mod tests;
