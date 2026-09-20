//! Sequential in-process backend for finite vector environments.

use thiserror::Error;

use crate::{
    EnvironmentPluginError, EnvironmentResetRunner, EnvironmentStepInfo, FiniteBackendStep,
    FiniteVectorBackend,
};

pub type BoxedFiniteEnvironment<Observation, Action, Reward, Info> =
    Box<dyn FiniteEnvironment<Observation, Action, Reward, Info>>;

pub trait FiniteEnvironment<Observation, Action, Reward, Info>: Send {
    /// # Errors
    /// Returns the underlying environment reset failure.
    fn reset(&mut self) -> Result<Observation, EnvironmentPluginError>;

    /// # Errors
    /// Returns the underlying environment step failure.
    fn step(
        &mut self,
        action: &Action,
    ) -> Result<FiniteBackendStep<Observation, Reward, Info>, EnvironmentPluginError>;
}

impl<InitialState, Observation, PolicyAction, SimulatorAction, AuxiliaryInfo>
    FiniteEnvironment<
        Observation,
        PolicyAction,
        f64,
        EnvironmentStepInfo<Observation, PolicyAction, AuxiliaryInfo>,
    >
    for EnvironmentResetRunner<
        InitialState,
        Observation,
        PolicyAction,
        SimulatorAction,
        AuxiliaryInfo,
    >
where
    InitialState: Clone + Send + 'static,
    Observation: Clone + Send + Sync + 'static,
    PolicyAction: Clone + Send + Sync + 'static,
    SimulatorAction: Send + 'static,
    AuxiliaryInfo: Default + Send + 'static,
{
    fn reset(&mut self) -> Result<Observation, EnvironmentPluginError> {
        EnvironmentResetRunner::reset(self)
            .map_err(|error| EnvironmentPluginError::new(error.to_string()))
    }

    fn step(
        &mut self,
        action: &PolicyAction,
    ) -> Result<
        FiniteBackendStep<
            Observation,
            f64,
            EnvironmentStepInfo<Observation, PolicyAction, AuxiliaryInfo>,
        >,
        EnvironmentPluginError,
    > {
        let output = EnvironmentResetRunner::step(self, action.clone())
            .map_err(|error| EnvironmentPluginError::new(error.to_string()))?;
        Ok(FiniteBackendStep {
            observation: Some(output.observation),
            reward: Some(output.reward),
            done: output.done,
            info: Some(output.info),
        })
    }
}

pub trait FiniteEnvironmentFactory<Observation, Action, Reward, Info>: Send {
    /// # Errors
    /// Returns an environment-construction failure.
    fn create(
        &mut self,
    ) -> Result<BoxedFiniteEnvironment<Observation, Action, Reward, Info>, EnvironmentPluginError>;
}

impl<Observation, Action, Reward, Info, Factory>
    FiniteEnvironmentFactory<Observation, Action, Reward, Info> for Factory
where
    Factory: FnMut() -> Result<
            BoxedFiniteEnvironment<Observation, Action, Reward, Info>,
            EnvironmentPluginError,
        > + Send,
{
    fn create(
        &mut self,
    ) -> Result<BoxedFiniteEnvironment<Observation, Action, Reward, Info>, EnvironmentPluginError>
    {
        self()
    }
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum FiniteDummyBuildError {
    #[error("a finite dummy backend requires at least one environment")]
    Empty,
    #[error("finite dummy environment {environment_id} construction failed: {source}")]
    Environment {
        environment_id: usize,
        #[source]
        source: EnvironmentPluginError,
    },
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum FiniteDummyError {
    #[error("environment id {id} is outside 0..{environment_count}")]
    InvalidEnvironmentId { id: usize, environment_count: usize },
    #[error("action batch has {actual} entries but exactly {required} are required")]
    ActionCount { required: usize, actual: usize },
    #[error("finite dummy environment {environment_id} reset failed: {source}")]
    Reset {
        environment_id: usize,
        #[source]
        source: EnvironmentPluginError,
    },
    #[error("finite dummy environment {environment_id} step failed: {source}")]
    Step {
        environment_id: usize,
        #[source]
        source: EnvironmentPluginError,
    },
}

pub struct FiniteDummyBackend<Observation, Action, Reward, Info> {
    environments: Vec<BoxedFiniteEnvironment<Observation, Action, Reward, Info>>,
}

impl<Observation, Action, Reward, Info> FiniteDummyBackend<Observation, Action, Reward, Info> {
    /// # Errors
    /// Returns `Empty` because Tianshou's synchronous vector environment requires at least one
    /// worker.
    pub fn new(
        environments: Vec<BoxedFiniteEnvironment<Observation, Action, Reward, Info>>,
    ) -> Result<Self, FiniteDummyBuildError> {
        if environments.is_empty() {
            return Err(FiniteDummyBuildError::Empty);
        }
        Ok(Self { environments })
    }

    /// Calls the same factory once per worker, in ascending worker-id order.
    ///
    /// # Errors
    /// Returns an empty-pool error or the first environment construction failure.
    pub fn from_factory(
        concurrency: usize,
        factory: &mut dyn FiniteEnvironmentFactory<Observation, Action, Reward, Info>,
    ) -> Result<Self, FiniteDummyBuildError> {
        if concurrency == 0 {
            return Err(FiniteDummyBuildError::Empty);
        }
        let environments = (0..concurrency)
            .map(|environment_id| {
                factory
                    .create()
                    .map_err(|source| FiniteDummyBuildError::Environment {
                        environment_id,
                        source,
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { environments })
    }

    #[must_use]
    pub fn environment_count(&self) -> usize {
        self.environments.len()
    }

    /// Runs the synchronous Dummy `send` phase before reproducing its ordered `recv` phase.
    ///
    /// Repeated ids execute the same environment repeatedly; each corresponding output slot
    /// receives a clone of the final result retained by that worker.
    ///
    /// # Errors
    /// Returns the first invalid id or worker reset failure after retaining earlier mutations.
    pub fn reset_environments(
        &mut self,
        environment_ids: &[usize],
    ) -> Result<Vec<Option<Observation>>, FiniteDummyError>
    where
        Observation: Clone,
    {
        let mut pending = (0..self.environments.len())
            .map(|_| None)
            .collect::<Vec<_>>();
        for environment_id in environment_ids.iter().copied() {
            let environment_count = self.environments.len();
            let environment = self.environments.get_mut(environment_id).ok_or(
                FiniteDummyError::InvalidEnvironmentId {
                    id: environment_id,
                    environment_count,
                },
            )?;
            let observation = environment
                .reset()
                .map_err(|source| FiniteDummyError::Reset {
                    environment_id,
                    source,
                })?;
            pending[environment_id] = Some(observation);
        }
        Ok(environment_ids
            .iter()
            .map(|environment_id| pending[*environment_id].clone())
            .collect())
    }

    /// Runs the synchronous Dummy `send` phase before reproducing its ordered `recv` phase.
    ///
    /// # Errors
    /// Returns an action-count mismatch before mutation, or the first invalid id or worker step
    /// failure after retaining earlier mutations.
    ///
    /// # Panics
    /// Panics only if an environment that successfully completed the send phase loses its
    /// retained result before the immediately following receive phase, violating the backend's
    /// internal invariant.
    pub fn step_environments(
        &mut self,
        actions: &[Action],
        environment_ids: &[usize],
    ) -> Result<Vec<FiniteBackendStep<Observation, Reward, Info>>, FiniteDummyError>
    where
        Observation: Clone,
        Reward: Clone,
        Info: Clone,
    {
        if actions.len() != environment_ids.len() {
            return Err(FiniteDummyError::ActionCount {
                required: environment_ids.len(),
                actual: actions.len(),
            });
        }
        let mut pending = (0..self.environments.len())
            .map(|_| None)
            .collect::<Vec<_>>();
        for (action, environment_id) in actions.iter().zip(environment_ids.iter().copied()) {
            let environment_count = self.environments.len();
            let environment = self.environments.get_mut(environment_id).ok_or(
                FiniteDummyError::InvalidEnvironmentId {
                    id: environment_id,
                    environment_count,
                },
            )?;
            let transition = environment
                .step(action)
                .map_err(|source| FiniteDummyError::Step {
                    environment_id,
                    source,
                })?;
            pending[environment_id] = Some(transition);
        }
        Ok(environment_ids
            .iter()
            .map(|environment_id| {
                pending[*environment_id]
                    .as_ref()
                    .expect("each requested environment retained a result")
                    .clone()
            })
            .collect())
    }
}

impl<Observation, Action, Reward, Info> FiniteVectorBackend<Observation, Action, Reward, Info>
    for FiniteDummyBackend<Observation, Action, Reward, Info>
where
    Observation: Clone + Send,
    Action: Send,
    Reward: Clone + Send,
    Info: Clone + Send,
{
    fn environment_count(&self) -> usize {
        self.environment_count()
    }

    fn reset(
        &mut self,
        environment_ids: &[usize],
    ) -> Result<Vec<Option<Observation>>, EnvironmentPluginError> {
        self.reset_environments(environment_ids)
            .map_err(|error| EnvironmentPluginError::new(error.to_string()))
    }

    fn step(
        &mut self,
        actions: &[Action],
        environment_ids: &[usize],
    ) -> Result<Vec<FiniteBackendStep<Observation, Reward, Info>>, EnvironmentPluginError> {
        self.step_environments(actions, environment_ids)
            .map_err(|error| EnvironmentPluginError::new(error.to_string()))
    }
}
