//! Additive lossless-text Checkpoint callback; the established UTF-8 callback is unchanged.

use indexmap::IndexMap;
use num_bigint::BigInt;
use std::path::PathBuf;

use crate::{
    RlCheckpointCallbackError, RlCheckpointClock, RlCheckpointGraph, RlCheckpointState,
    RlCheckpointStorage, RlCheckpointText, RlCheckpointTimeInterval, RlLosslessCheckpointName,
    RlTrainerCallback, RlTrainerControl, RlTrainerDriverError, RlTrainerHook, RlTrainerRuntime,
};

#[derive(Clone, Debug, PartialEq)]
pub struct RlLosslessCheckpointConfig {
    pub dirpath: PathBuf,
    pub filename: RlCheckpointText,
    pub save_latest: Option<RlCheckpointText>,
    pub every_n_iters: Option<BigInt>,
    pub time_interval: Option<RlCheckpointTimeInterval>,
    pub save_on_fit_end: bool,
}
impl RlLosslessCheckpointConfig {
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
pub struct RlLosslessCheckpointCallbackState {
    pub last_name: Option<RlCheckpointText>,
    pub last_iter: Option<BigInt>,
    pub last_time: Option<f64>,
}

pub struct RlLosslessCheckpointCallback<C, N, G, F> {
    pub config: RlLosslessCheckpointConfig,
    pub state: RlLosslessCheckpointCallbackState,
    pub clock: C,
    pub name: N,
    pub graph: G,
    pub storage: F,
}
impl<C, N, G, F> RlLosslessCheckpointCallback<C, N, G, F> {
    #[must_use]
    pub fn new(
        config: RlLosslessCheckpointConfig,
        clock: C,
        name: N,
        graph: G,
        storage: F,
    ) -> Self {
        Self {
            config,
            state: RlLosslessCheckpointCallbackState::default(),
            clock,
            name,
            graph,
            storage,
        }
    }
}

fn plugin<T>(
    stage: &'static str,
    result: Result<T, String>,
) -> Result<T, RlCheckpointCallbackError> {
    result.map_err(|message| RlCheckpointCallbackError::Plugin { stage, message })
}
fn iteration<M>(runtime: &RlTrainerRuntime<M>) -> Result<BigInt, RlCheckpointCallbackError> {
    runtime.read(|state| state.current_iter.clone())?.ok_or(
        RlCheckpointCallbackError::MissingTrainerField("current_iter"),
    )
}

impl<C: RlCheckpointClock, N, G, F> RlLosslessCheckpointCallback<C, N, G, F> {
    /// Runs the same scheduling and partial-failure contract as the UTF-8 callback.
    /// # Errors
    /// Returns the first scheduler, runtime, naming, graph, or storage failure.
    pub fn on_hook<V, M: Clone>(
        &mut self,
        hook: RlTrainerHook,
        control: &mut RlTrainerControl<M>,
        vessel: &mut V,
    ) -> Result<(), RlCheckpointCallbackError>
    where
        N: RlLosslessCheckpointName<M>,
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
            if match self.state.last_time {
                None => true,
                Some(last) => interval.elapsed(plugin("time", self.clock.timestamp())? - last),
            } {
                due = true;
            }
        }
        Ok(due)
    }

    /// Saves after assigning raw lossless callback state and collecting the live graph.
    /// # Errors
    /// Returns the first naming, graph, path, or storage failure with prior effects retained.
    pub fn save<V, M: Clone>(
        &mut self,
        control: &mut RlTrainerControl<M>,
        vessel: &mut V,
    ) -> Result<(), RlCheckpointCallbackError>
    where
        N: RlLosslessCheckpointName<M>,
        G: RlCheckpointGraph<V, M>,
        F: RlCheckpointStorage<G::State>,
    {
        plugin("mkdir", self.storage.create_directory(&self.config.dirpath))?;
        let name = self.new_name(&control.runtime)?;
        self.state.last_name = Some(name.clone());
        self.state.last_iter = Some(iteration(&control.runtime)?);
        self.state.last_time = Some(plugin("time", self.clock.timestamp())?);
        let graph = plugin("graph", self.graph.collect(control, vessel))?;
        let path = plugin(
            "save",
            name.join_to(&self.config.dirpath)
                .and_then(|path| self.storage.save(&graph, &path).map(|()| path)),
        )?;
        let latest = self.config.dirpath.join("latest.pth");
        if self
            .config
            .save_latest
            .as_ref()
            .is_some_and(|mode| !mode.is_empty())
            && (plugin("exists", self.storage.exists(&latest))?
                || plugin("is_link", self.storage.is_link(&latest))?)
        {
            plugin("remove", self.storage.remove(&latest))?;
        }
        match self.config.save_latest.as_ref() {
            Some(mode) if mode == &RlCheckpointText::from("link") => {
                plugin("link", self.storage.link(&path, &latest))?;
            }
            Some(mode) if mode == &RlCheckpointText::from("copy") => {
                plugin("copy", self.storage.copy(&path, &latest))?;
            }
            _ => {}
        }
        Ok(())
    }

    /// Renders after cloning metrics outside the runtime lock.
    /// # Errors
    /// Returns runtime, clock, duplicate-keyword, or lossless formatter failures.
    pub fn new_name<M: Clone>(
        &mut self,
        runtime: &RlTrainerRuntime<M>,
    ) -> Result<RlCheckpointText, RlCheckpointCallbackError>
    where
        N: RlLosslessCheckpointName<M>,
    {
        let current = iteration(runtime)?;
        let time = RlCheckpointText::from(plugin("local_time", self.clock.local_time())?);
        let metrics = runtime
            .read(|state| state.metrics.clone())?
            .ok_or(RlCheckpointCallbackError::MissingTrainerField("metrics"))?;
        for reserved in ["iter", "time"] {
            if metrics.contains_key(reserved) {
                return Err(RlCheckpointCallbackError::DuplicateKeyword(reserved));
            }
        }
        let metrics: IndexMap<_, _> = metrics
            .into_iter()
            .map(|(name, value)| (RlCheckpointText::from(name), value))
            .collect();
        plugin(
            "format",
            self.name
                .render(&self.config.filename, &current, &time, &metrics),
        )
    }
}

impl<V, M: Clone, C, N, G, F> RlTrainerCallback<V, M> for RlLosslessCheckpointCallback<C, N, G, F>
where
    C: RlCheckpointClock,
    N: RlLosslessCheckpointName<M>,
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
impl<C, N, G, F> RlCheckpointState<()> for RlLosslessCheckpointCallback<C, N, G, F> {
    fn save_checkpoint(&mut self) -> Result<(), String> {
        Ok(())
    }
    fn load_checkpoint(&mut self, (): &()) -> Result<(), String> {
        Ok(())
    }
}

#[cfg(test)]
#[path = "../tests/support/rl_checkpoint_lossless_callback.rs"]
mod tests;
