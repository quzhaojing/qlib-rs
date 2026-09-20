//! Synchronous Qlib Trainer scheduling over typed vessel, callback, and seed-context plugins.

use std::sync::{Arc, Mutex};

use num_bigint::BigInt;
use thiserror::Error;

use crate::{RlTrainerRuntime, RlTrainerStateError};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RlTrainerConfig {
    pub max_iters: Option<BigInt>,
    pub val_every_n_iters: Option<BigInt>,
}

/// Callbacks may update configuration and shared runtime state. No runtime lock is held
/// while calling plugins, allowing environment metric callbacks to use the same runtime.
pub struct RlTrainerControl<M = f64> {
    pub runtime: Arc<RlTrainerRuntime<M>>,
    pub config: RlTrainerConfig,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RlTrainerPhase {
    Train,
    Validation,
    Test,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RlTrainerHook {
    FitStart,
    FitEnd,
    IterStart,
    IterEnd,
    TrainStart,
    TrainEnd,
    ValidateStart,
    ValidateEnd,
    TestStart,
    TestEnd,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum RlTrainerDriverError {
    #[error(transparent)]
    State(#[from] RlTrainerStateError),
    #[error("trainer {0} has not been initialized")]
    MissingState(&'static str),
    #[error("validation interval is zero")]
    ZeroValidationInterval,
    #[error("trainer plugin failed at {stage}: {message}")]
    Plugin { stage: String, message: String },
}

/// Stores the input iterable and, for context-managed seeds, the possibly different value
/// yielded by enter. Vessel environment construction consumes that value through this type.
/// Plain iterables can use the default no-op enter/exit implementations.
pub trait RlTrainerSeedContext {
    /// # Errors
    /// Entry failures do not trigger exit.
    fn enter(&mut self) -> Result<(), RlTrainerDriverError> {
        Ok(())
    }

    /// Returns true to suppress a phase error. Exit failure replaces the phase error.
    /// Exit is invoked after successful entry, including when environment construction fails.
    ///
    /// # Errors
    /// Returns cleanup failures, without running the phase-end hook.
    fn exit(
        &mut self,
        _error: Option<&RlTrainerDriverError>,
    ) -> Result<bool, RlTrainerDriverError> {
        Ok(false)
    }
}

/// Qlib's default seed queue activates on context entry and cleans up on every
/// successful entry's exit, without suppressing the phase error. The queue keeps
/// its existing single-use, backpressure and producer-failure semantics.
impl<T: Send + 'static> RlTrainerSeedContext for crate::DataQueue<T> {
    fn enter(&mut self) -> Result<(), RlTrainerDriverError> {
        self.activate()
            .map(|_| ())
            .map_err(|error| RlTrainerDriverError::Plugin {
                stage: "seed_enter".into(),
                message: error.to_string(),
            })
    }

    fn exit(
        &mut self,
        _error: Option<&RlTrainerDriverError>,
    ) -> Result<bool, RlTrainerDriverError> {
        self.cleanup();
        Ok(false)
    }
}

/// Shared queue ownership lets independently owned in-process environments
/// consume one seed stream while the Trainer retains context cleanup authority.
/// This is a linked in-process handle, not a subprocess transport or plugin ABI.
impl<T: Send + 'static> RlTrainerSeedContext for Arc<Mutex<crate::DataQueue<T>>> {
    fn enter(&mut self) -> Result<(), RlTrainerDriverError> {
        let mut queue = self.lock().map_err(|error| RlTrainerDriverError::Plugin {
            stage: "seed_enter".into(),
            message: error.to_string(),
        })?;
        RlTrainerSeedContext::enter(&mut *queue)
    }

    fn exit(&mut self, error: Option<&RlTrainerDriverError>) -> Result<bool, RlTrainerDriverError> {
        let mut queue = self.lock().map_err(|error| RlTrainerDriverError::Plugin {
            stage: "seed_exit".into(),
            message: error.to_string(),
        })?;
        RlTrainerSeedContext::exit(&mut *queue, error)
    }
}

pub trait RlTrainerVessel<M = f64> {
    type Seed: RlTrainerSeedContext;
    type Environment;

    /// # Errors
    /// Returns attachment errors before initialization or checkpoint restoration.
    fn assign_trainer(
        &mut self,
        runtime: &Arc<RlTrainerRuntime<M>>,
    ) -> Result<(), RlTrainerDriverError>;
    /// # Errors
    /// Returns a seed-iterator factory failure before entry.
    fn seeds(&mut self, phase: RlTrainerPhase) -> Result<Self::Seed, RlTrainerDriverError>;
    /// # Errors
    /// Returns environment construction errors inside the seed context.
    fn environment(
        &mut self,
        seed: &mut Self::Seed,
        control: &mut RlTrainerControl<M>,
    ) -> Result<Self::Environment, RlTrainerDriverError>;
    /// # Errors
    /// Returns training/evaluation errors inside the seed context.
    fn run(
        &mut self,
        phase: RlTrainerPhase,
        environment: &mut Self::Environment,
        control: &mut RlTrainerControl<M>,
    ) -> Result<(), RlTrainerDriverError>;
}

/// One typed dispatch replaces Python's getattr-based callback invocation. Registered
/// callbacks run in vector order and share the same live control and vessel references.
pub trait RlTrainerCallback<V, M = f64> {
    /// # Errors
    /// Returns the first hook failure; later callbacks and later lifecycle hooks are skipped.
    fn call(
        &mut self,
        hook: RlTrainerHook,
        control: &mut RlTrainerControl<M>,
        vessel: &mut V,
    ) -> Result<(), RlTrainerDriverError>;
}

/// Owns checkpoint I/O and graph restoration, including any resume diagnostic. The driver
/// invokes this only after vessel attachment, instead of initialize. It does not claim a
/// Torch/pickle codec or deep-copy arbitrary model objects.
pub trait RlTrainerRestore<V, M = f64> {
    /// # Errors
    /// Returns read/restore errors, retaining prior partial state changes.
    fn restore(
        &mut self,
        control: &mut RlTrainerControl<M>,
        vessel: &mut V,
    ) -> Result<(), RlTrainerDriverError>;
}

pub trait RlTrainerProgress {
    /// Called before metrics are cleared and before iteration-start callbacks.
    /// # Errors
    /// Returns a diagnostic failure before iteration initialization.
    fn iteration(
        &mut self,
        next: &BigInt,
        maximum: Option<&BigInt>,
    ) -> Result<(), RlTrainerDriverError>;
}

pub struct TracingRlTrainerProgress;

impl RlTrainerProgress for TracingRlTrainerProgress {
    fn iteration(
        &mut self,
        next: &BigInt,
        maximum: Option<&BigInt>,
    ) -> Result<(), RlTrainerDriverError> {
        tracing::info!(iteration = %next, max_iters = ?maximum, "Train iteration");
        Ok(())
    }
}

impl<M> RlTrainerControl<M> {
    fn iteration(&self) -> Result<BigInt, RlTrainerDriverError> {
        self.runtime
            .read(|state| state.current_iter.clone())?
            .ok_or(RlTrainerDriverError::MissingState("current_iter"))
    }

    fn stopped(&self) -> Result<bool, RlTrainerDriverError> {
        self.runtime
            .read(|state| state.should_stop)?
            .ok_or(RlTrainerDriverError::MissingState("should_stop"))
    }

    // The interval check and stage transition have no intervening plugin calls. Keep them
    // under one lock so concurrent runtime mutation cannot invalidate the checked iteration.
    fn begin_validation(&self) -> Result<bool, RlTrainerDriverError> {
        match &self.config.val_every_n_iters {
            None => Ok(false),
            Some(interval) => self.runtime.update(|state| {
                let next = state
                    .current_iter
                    .as_ref()
                    .ok_or(RlTrainerDriverError::MissingState("current_iter"))?
                    + 1;
                if *interval == BigInt::default() {
                    return Err(RlTrainerDriverError::ZeroValidationInterval);
                }
                let due = next % interval == BigInt::default();
                if due {
                    state.current_stage = "val".into();
                }
                Ok(due)
            })?,
        }
    }

    fn finish_iteration(&self) -> Result<(), RlTrainerDriverError> {
        self.runtime.update(|state| {
            let current = state
                .current_iter
                .as_mut()
                .ok_or(RlTrainerDriverError::MissingState("current_iter"))?;
            *current += 1;
            if self
                .config
                .max_iters
                .as_ref()
                .is_some_and(|maximum| &*current >= maximum)
            {
                state.should_stop = Some(true);
            }
            Ok(())
        })?
    }
}

/// Owns the vessel for reuse after fit/test, including after errors. Callbacks can mutate
/// vessel/control state, but callback-list mutation during dispatch and arbitrary Python
/// dynamic attribute replacement are not represented by this linked Rust interface.
pub struct RlTrainerDriver<V, M = f64> {
    pub control: RlTrainerControl<M>,
    pub vessel: V,
    pub callbacks: Vec<Box<dyn RlTrainerCallback<V, M>>>,
    pub progress: Box<dyn RlTrainerProgress>,
}

impl<V: RlTrainerVessel<M>, M> RlTrainerDriver<V, M> {
    #[must_use]
    pub fn new(vessel: V, runtime: Arc<RlTrainerRuntime<M>>, config: RlTrainerConfig) -> Self {
        Self {
            control: RlTrainerControl { runtime, config },
            vessel,
            callbacks: Vec::new(),
            progress: Box::new(TracingRlTrainerProgress),
        }
    }

    /// # Errors
    /// Returns the first non-suppressed plugin/state error. End hooks are not finally blocks.
    pub fn fit(
        &mut self,
        restore: Option<&mut dyn RlTrainerRestore<V, M>>,
    ) -> Result<(), RlTrainerDriverError> {
        self.vessel.assign_trainer(&self.control.runtime)?;
        if let Some(restore) = restore {
            restore.restore(&mut self.control, &mut self.vessel)?;
        } else {
            self.control
                .runtime
                .update(crate::RlTrainerState::initialize)?;
        }
        self.call(RlTrainerHook::FitStart)?;
        // Python retains the local binding when execution fails and __exit__ suppresses it.
        // Keep that environment until successful replacement or method return, too.
        let mut environment = None;
        while !self.control.stopped()? {
            let next = self.control.iteration()? + 1;
            self.progress
                .iteration(&next, self.control.config.max_iters.as_ref())?;
            self.control
                .runtime
                .update(crate::RlTrainerState::initialize_iter)?;
            self.call(RlTrainerHook::IterStart)?;
            self.control
                .runtime
                .update(|state| state.current_stage = "train".into())?;
            self.call(RlTrainerHook::TrainStart)?;
            self.run_phase(RlTrainerPhase::Train, &mut environment)?;
            self.call(RlTrainerHook::TrainEnd)?;
            if self.control.begin_validation()? {
                self.call(RlTrainerHook::ValidateStart)?;
                self.run_phase(RlTrainerPhase::Validation, &mut environment)?;
                self.call(RlTrainerHook::ValidateEnd)?;
            }
            self.control.finish_iteration()?;
            self.call(RlTrainerHook::IterEnd)?;
        }
        self.call(RlTrainerHook::FitEnd)
    }

    /// Test clears only metrics, leaving iteration/episode/stop metadata unchanged or absent.
    /// # Errors
    /// Returns the first non-suppressed error, without calling test-end on failure.
    pub fn test(&mut self) -> Result<(), RlTrainerDriverError> {
        self.vessel.assign_trainer(&self.control.runtime)?;
        self.control.runtime.update(|state| {
            state.initialize_iter();
            state.current_stage = "test".into();
        })?;
        self.call(RlTrainerHook::TestStart)?;
        let mut environment = None;
        self.run_phase(RlTrainerPhase::Test, &mut environment)?;
        self.call(RlTrainerHook::TestEnd)
    }

    fn call(&mut self, hook: RlTrainerHook) -> Result<(), RlTrainerDriverError> {
        for callback in &mut self.callbacks {
            callback.call(hook, &mut self.control, &mut self.vessel)?;
        }
        Ok(())
    }

    fn run_phase(
        &mut self,
        phase: RlTrainerPhase,
        environment: &mut Option<V::Environment>,
    ) -> Result<(), RlTrainerDriverError> {
        let mut seed = self.vessel.seeds(phase)?;
        seed.enter()?;
        let result = match self.vessel.environment(&mut seed, &mut self.control) {
            Ok(mut next) => {
                // Assignment releases any previously suppressed failed environment only after
                // the replacement has been constructed, matching Python evaluation order.
                drop(environment.take());
                let result = self.vessel.run(phase, &mut next, &mut self.control);
                if result.is_ok() {
                    drop(next);
                } else {
                    *environment = Some(next);
                }
                result
            }
            Err(error) => Err(error),
        };
        if seed.exit(result.as_ref().err())? {
            Ok(())
        } else {
            result
        }
    }
}

#[cfg(test)]
#[path = "../tests/support/rl_trainer_driver.rs"]
mod tests;
