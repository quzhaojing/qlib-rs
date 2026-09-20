//! Transport-neutral finite vector-environment coordination.

use indexmap::IndexSet;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::warn;

use crate::{EnvironmentPluginError, FiniteObservation, is_invalid};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FiniteBackendStep<Observation, Reward, Info> {
    pub observation: Option<Observation>,
    pub reward: Option<Reward>,
    pub done: bool,
    pub info: Option<Info>,
}

impl<Observation, Reward, Info> Default for FiniteBackendStep<Observation, Reward, Info> {
    fn default() -> Self {
        Self {
            observation: None,
            reward: None,
            done: false,
            info: None,
        }
    }
}

pub trait FiniteVectorBackend<Observation, Action, Reward, Info>: Send {
    fn environment_count(&self) -> usize;

    /// # Errors
    /// Returns backend reset or transport failures.
    fn reset(
        &mut self,
        environment_ids: &[usize],
    ) -> Result<Vec<Option<Observation>>, EnvironmentPluginError>;

    /// # Errors
    /// Returns backend step or transport failures.
    fn step(
        &mut self,
        actions: &[Action],
        environment_ids: &[usize],
    ) -> Result<Vec<FiniteBackendStep<Observation, Reward, Info>>, EnvironmentPluginError>;
}

pub trait FiniteObservationPredicate<Observation>: Send {
    /// # Errors
    /// Returns observation decoding or dtype failures.
    fn is_invalid(&mut self, observation: &Observation) -> Result<bool, EnvironmentPluginError>;
}

pub struct RecursiveFiniteObservationPredicate;

impl FiniteObservationPredicate<FiniteObservation> for RecursiveFiniteObservationPredicate {
    fn is_invalid(
        &mut self,
        observation: &FiniteObservation,
    ) -> Result<bool, EnvironmentPluginError> {
        is_invalid(observation).map_err(|error| EnvironmentPluginError::new(error.to_string()))
    }
}

pub trait FiniteVectorLogger<Observation, Reward, Info>: Send {
    /// # Errors
    /// Returns logger initialization failures.
    fn on_all_ready(&mut self) -> Result<(), EnvironmentPluginError> {
        Ok(())
    }

    /// # Errors
    /// Returns logger finalization failures.
    fn on_all_done(&mut self) -> Result<(), EnvironmentPluginError> {
        Ok(())
    }

    /// # Errors
    /// Returns per-environment reset logging failures.
    fn on_reset(
        &mut self,
        _environment_id: usize,
        _all_observations: &[Option<Observation>],
    ) -> Result<(), EnvironmentPluginError> {
        Ok(())
    }

    /// # Errors
    /// Returns per-environment step logging failures.
    fn on_step(
        &mut self,
        _environment_id: usize,
        _step: &FiniteBackendStep<Observation, Reward, Info>,
    ) -> Result<(), EnvironmentPluginError> {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FiniteVectorStage {
    BackendReset,
    BackendStep,
    ResetObservation,
    StepObservation,
    LoggerAllReady,
    LoggerAllDone,
    LoggerReset,
    LoggerStep,
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum FiniteVectorError {
    #[error("finite vector environment is a zombie")]
    Zombie,
    #[error("finite vector environment is exhausted")]
    Exhausted,
    #[error("environment id {id} is outside 0..{environment_count}")]
    InvalidEnvironmentId { id: usize, environment_count: usize },
    #[error("action batch has {actual} entries but at least {required} are required")]
    ActionBatchTooShort { required: usize, actual: usize },
    #[error("cannot stack an empty finite-vector selection")]
    EmptySelection,
    #[error(
        "finite vector component failed at {stage:?} for environment {environment_id:?}: {source}"
    )]
    Component {
        stage: FiniteVectorStage,
        environment_id: Option<usize>,
        #[source]
        source: EnvironmentPluginError,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct FiniteVectorReset<Observation> {
    pub observations: Vec<Option<Observation>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FiniteVectorStep<Observation, Reward, Info> {
    pub transitions: Vec<FiniteBackendStep<Observation, Reward, Info>>,
}

pub struct FiniteVectorEnv<Observation, Action, Reward, Info> {
    backend: Box<dyn FiniteVectorBackend<Observation, Action, Reward, Info>>,
    predicate: Box<dyn FiniteObservationPredicate<Observation>>,
    loggers: Vec<Box<dyn FiniteVectorLogger<Observation, Reward, Info>>>,
    alive_environment_ids: IndexSet<usize>,
    default_observation: Option<Observation>,
    default_reward: Option<Reward>,
    default_info: Option<Info>,
    zombie: bool,
    collector_guarded: bool,
    unguarded_reset_warnings: usize,
}

impl<Observation, Action, Reward, Info> FiniteVectorEnv<Observation, Action, Reward, Info>
where
    Observation: Clone + 'static,
    Action: Clone + 'static,
    Reward: Clone + 'static,
    Info: Clone + 'static,
{
    #[must_use]
    pub fn new(
        backend: Box<dyn FiniteVectorBackend<Observation, Action, Reward, Info>>,
        predicate: Box<dyn FiniteObservationPredicate<Observation>>,
        loggers: Vec<Box<dyn FiniteVectorLogger<Observation, Reward, Info>>>,
    ) -> Self {
        let mut environment = Self {
            backend,
            predicate,
            loggers,
            alive_environment_ids: IndexSet::new(),
            default_observation: None,
            default_reward: None,
            default_info: None,
            zombie: false,
            collector_guarded: false,
            unguarded_reset_warnings: 0,
        };
        environment.reset_alive_environments();
        environment
    }

    #[must_use]
    pub fn environment_count(&self) -> usize {
        self.backend.environment_count()
    }

    #[must_use]
    pub const fn is_zombie(&self) -> bool {
        self.zombie
    }

    #[must_use]
    pub const fn is_collector_guarded(&self) -> bool {
        self.collector_guarded
    }

    #[must_use]
    pub const fn unguarded_reset_warnings(&self) -> usize {
        self.unguarded_reset_warnings
    }

    #[must_use]
    pub const fn alive_environment_ids(&self) -> &IndexSet<usize> {
        &self.alive_environment_ids
    }

    /// Runs one collector operation with Python-compatible exhaustion and logger lifecycle.
    ///
    /// # Errors
    /// Returns logger failures or non-exhaustion errors from `collect`.
    pub fn collect_guarded<T>(
        &mut self,
        collect: impl FnOnce(&mut Self) -> Result<T, FiniteVectorError>,
    ) -> Result<Option<T>, FiniteVectorError> {
        self.collect_guarded_with(collect, |error| {
            matches!(error, FiniteVectorError::Exhausted)
        })
    }

    /// Preserves the collector lifecycle while retaining a caller's typed operation errors.
    /// Only errors recognized by `is_exhausted` are suppressed. As in the source context
    /// manager, ready-hook failures leave the guard set, and body failures skip done hooks.
    /// Rust panic unwinding is not an exhaustion signal.
    ///
    /// # Errors
    /// Returns converted logger failures or unrecognized errors from `collect`.
    pub fn collect_guarded_with<T, E>(
        &mut self,
        collect: impl FnOnce(&mut Self) -> Result<T, E>,
        is_exhausted: impl FnOnce(&E) -> bool,
    ) -> Result<Option<T>, E>
    where
        E: From<FiniteVectorError>,
    {
        self.collector_guarded = true;
        for logger in &mut self.loggers {
            logger
                .on_all_ready()
                .map_err(|source| FiniteVectorError::Component {
                    stage: FiniteVectorStage::LoggerAllReady,
                    environment_id: None,
                    source,
                })?;
        }

        let result = collect(self);
        self.collector_guarded = false;
        let output = match result {
            Ok(value) => Some(value),
            Err(error) if is_exhausted(&error) => None,
            Err(error) => return Err(error),
        };
        for logger in &mut self.loggers {
            logger
                .on_all_done()
                .map_err(|source| FiniteVectorError::Component {
                    stage: FiniteVectorStage::LoggerAllDone,
                    environment_id: None,
                    source,
                })?;
        }
        Ok(output)
    }

    /// # Errors
    /// Returns zombie, selection, backend, observation, logger, or exhaustion errors.
    ///
    /// # Panics
    /// Panics only if an internally derived request id no longer maps to its validated wrapped
    /// selection, which would violate the coordinator's construction invariant.
    pub fn reset(
        &mut self,
        environment_ids: Option<&[usize]>,
    ) -> Result<FiniteVectorReset<Observation>, FiniteVectorError> {
        if self.zombie {
            return Err(FiniteVectorError::Zombie);
        }
        if !self.collector_guarded {
            self.unguarded_reset_warnings += 1;
            warn!("Collector is not guarded by FiniteVectorEnv");
        }
        let wrapped_ids = self.wrap_ids(environment_ids)?;
        self.reset_alive_environments();
        let request_ids = wrapped_ids
            .iter()
            .copied()
            .filter(|id| self.alive_environment_ids.contains(id))
            .collect::<Vec<_>>();
        let id_to_index = self.last_id_indices(&wrapped_ids);
        let mut observations = vec![None; wrapped_ids.len()];
        if !request_ids.is_empty() {
            let backend_observations = self.backend.reset(&request_ids).map_err(|source| {
                FiniteVectorError::Component {
                    stage: FiniteVectorStage::BackendReset,
                    environment_id: None,
                    source,
                }
            })?;
            for (environment_id, observation) in
                request_ids.iter().copied().zip(backend_observations)
            {
                let target = id_to_index[environment_id]
                    .expect("requested environment id came from wrapped ids");
                observations[target] = self.postprocess_observation(
                    observation,
                    FiniteVectorStage::ResetObservation,
                    environment_id,
                )?;
            }
        }

        for (environment_id, observation) in wrapped_ids.iter().copied().zip(&observations) {
            if observation.is_none() && self.alive_environment_ids.contains(&environment_id) {
                self.alive_environment_ids.shift_remove(&environment_id);
            }
        }
        for environment_id in wrapped_ids.iter().copied() {
            if self.alive_environment_ids.contains(&environment_id) {
                for logger in &mut self.loggers {
                    logger
                        .on_reset(environment_id, &observations)
                        .map_err(|source| FiniteVectorError::Component {
                            stage: FiniteVectorStage::LoggerReset,
                            environment_id: Some(environment_id),
                            source,
                        })?;
                }
            }
        }

        for observation in &observations {
            if self.default_observation.is_none() {
                if let Some(observation) = observation {
                    self.default_observation = Some(observation.clone());
                }
            }
        }
        for observation in &mut observations {
            if observation.is_none() {
                observation.clone_from(&self.default_observation);
            }
        }
        if self.alive_environment_ids.is_empty() {
            self.zombie = true;
            return Err(FiniteVectorError::Exhausted);
        }
        if observations.is_empty() {
            return Err(FiniteVectorError::EmptySelection);
        }
        Ok(FiniteVectorReset { observations })
    }

    /// # Errors
    /// Returns zombie, selection, action, backend, observation, or logger errors.
    ///
    /// # Panics
    /// Panics only if an internally derived request id no longer maps to its validated wrapped
    /// selection, which would violate the coordinator's construction invariant.
    pub fn step(
        &mut self,
        actions: &[Action],
        environment_ids: Option<&[usize]>,
    ) -> Result<FiniteVectorStep<Observation, Reward, Info>, FiniteVectorError> {
        if self.zombie {
            return Err(FiniteVectorError::Zombie);
        }
        let wrapped_ids = self.wrap_ids(environment_ids)?;
        if actions.len() < wrapped_ids.len() {
            return Err(FiniteVectorError::ActionBatchTooShort {
                required: wrapped_ids.len(),
                actual: actions.len(),
            });
        }
        let id_to_index = self.last_id_indices(&wrapped_ids);
        let request_ids = wrapped_ids
            .iter()
            .copied()
            .filter(|id| self.alive_environment_ids.contains(id))
            .collect::<Vec<_>>();
        let mut transitions = (0..wrapped_ids.len())
            .map(|_| FiniteBackendStep::default())
            .collect::<Vec<_>>();
        if !request_ids.is_empty() {
            let valid_actions = request_ids
                .iter()
                .map(|id| {
                    actions
                        [id_to_index[*id].expect("requested environment id came from wrapped ids")]
                    .clone()
                })
                .collect::<Vec<_>>();
            let backend_transitions =
                self.backend
                    .step(&valid_actions, &request_ids)
                    .map_err(|source| FiniteVectorError::Component {
                        stage: FiniteVectorStage::BackendStep,
                        environment_id: None,
                        source,
                    })?;
            for (environment_id, mut transition) in
                request_ids.iter().copied().zip(backend_transitions)
            {
                transition.observation = self.postprocess_observation(
                    transition.observation,
                    FiniteVectorStage::StepObservation,
                    environment_id,
                )?;
                let target = id_to_index[environment_id]
                    .expect("requested environment id came from wrapped ids");
                transitions[target] = transition;
            }
        }

        for (environment_id, transition) in wrapped_ids.iter().copied().zip(&transitions) {
            if self.alive_environment_ids.contains(&environment_id) {
                for logger in &mut self.loggers {
                    logger
                        .on_step(environment_id, transition)
                        .map_err(|source| FiniteVectorError::Component {
                            stage: FiniteVectorStage::LoggerStep,
                            environment_id: Some(environment_id),
                            source,
                        })?;
                }
            }
        }
        for transition in &transitions {
            if self.default_info.is_none() {
                if let Some(info) = &transition.info {
                    self.default_info = Some(info.clone());
                }
            }
            if self.default_reward.is_none() {
                if let Some(reward) = &transition.reward {
                    self.default_reward = Some(reward.clone());
                }
            }
        }
        for transition in &mut transitions {
            if transition.observation.is_none() {
                transition.observation.clone_from(&self.default_observation);
            }
            if transition.reward.is_none() {
                transition.reward.clone_from(&self.default_reward);
            }
            if transition.info.is_none() {
                transition.info.clone_from(&self.default_info);
            }
        }
        if transitions.is_empty() {
            return Err(FiniteVectorError::EmptySelection);
        }
        Ok(FiniteVectorStep { transitions })
    }

    fn wrap_ids(&self, environment_ids: Option<&[usize]>) -> Result<Vec<usize>, FiniteVectorError> {
        let ids = environment_ids.map_or_else(
            || (0..self.environment_count()).collect(),
            <[usize]>::to_vec,
        );
        for id in &ids {
            if *id >= self.environment_count() {
                return Err(FiniteVectorError::InvalidEnvironmentId {
                    id: *id,
                    environment_count: self.environment_count(),
                });
            }
        }
        Ok(ids)
    }

    fn reset_alive_environments(&mut self) {
        if self.alive_environment_ids.is_empty() {
            self.alive_environment_ids
                .extend(0..self.backend.environment_count());
        }
    }

    fn last_id_indices(&self, ids: &[usize]) -> Vec<Option<usize>> {
        let mut indices = vec![None; self.environment_count()];
        for (index, id) in ids.iter().copied().enumerate() {
            indices[id] = Some(index);
        }
        indices
    }

    fn postprocess_observation(
        &mut self,
        observation: Option<Observation>,
        stage: FiniteVectorStage,
        environment_id: usize,
    ) -> Result<Option<Observation>, FiniteVectorError> {
        let Some(observation) = observation else {
            return Ok(None);
        };
        if self.predicate.is_invalid(&observation).map_err(|source| {
            FiniteVectorError::Component {
                stage,
                environment_id: Some(environment_id),
                source,
            }
        })? {
            Ok(None)
        } else {
            Ok(Some(observation))
        }
    }
}
