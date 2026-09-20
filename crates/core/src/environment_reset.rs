//! Reset and finite-seed lifecycle for reinforcement-learning environments.

use thiserror::Error;

use crate::{
    EnvironmentPluginError, EnvironmentSimulator, EnvironmentStatus, EnvironmentStepError,
    EnvironmentStepOutput, EnvironmentStepRunner,
};

pub trait EnvironmentSimulatorFactory<InitialState, Observation, PolicyAction, SimulatorAction>:
    Send
{
    /// # Errors
    /// Returns construction failures. `EnvironmentPluginError::stop_iteration` has the same
    /// exhaustion meaning as Python's `StopIteration` raised anywhere inside `reset`.
    fn create(
        &mut self,
        initial_state: Option<&InitialState>,
    ) -> Result<
        Box<dyn EnvironmentSimulator<InitialState, Observation, PolicyAction, SimulatorAction>>,
        EnvironmentPluginError,
    >;
}

pub trait EnvironmentObservationSpace<Observation>: Send {
    /// Produces the sentinel observation returned after finite seeds are exhausted.
    ///
    /// Implementations can sample their native space and recursively replace floating values
    /// with NaN and integer values with their dtype maximum, matching Qlib's `fill_invalid`.
    ///
    /// # Errors
    /// Returns failures from sampling or invalid-value conversion.
    fn invalid_observation(&mut self) -> Result<Observation, EnvironmentPluginError>;
}

pub enum EnvironmentSeedSource<InitialState> {
    Unseeded,
    Seeded(Box<dyn Iterator<Item = InitialState> + Send>),
    TrySeeded(Box<dyn Iterator<Item = Result<InitialState, EnvironmentPluginError>> + Send>),
    Exhausted,
}

impl<InitialState> EnvironmentSeedSource<InitialState> {
    #[must_use]
    pub const fn unseeded() -> Self {
        Self::Unseeded
    }

    #[must_use]
    pub fn seeded(iterator: impl Iterator<Item = InitialState> + Send + 'static) -> Self {
        Self::Seeded(Box::new(iterator))
    }

    /// A fallible queue/iterator can report seed-read errors without inventing an
    /// initial state or silently converting every failure into exhaustion.
    #[must_use]
    pub fn try_seeded(
        iterator: impl Iterator<Item = Result<InitialState, EnvironmentPluginError>> + Send + 'static,
    ) -> Self {
        Self::TrySeeded(Box::new(iterator))
    }

    #[must_use]
    pub const fn is_exhausted(&self) -> bool {
        matches!(self, Self::Exhausted)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnvironmentResetStage {
    SeedIterator,
    SimulatorFactory,
    InitialObservation,
    InvalidObservation,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum EnvironmentResetError {
    #[error("cannot reset a dead environment wrapper")]
    Dead,
    #[error("environment reset failed at {stage:?}: {source}")]
    Component {
        stage: EnvironmentResetStage,
        #[source]
        source: EnvironmentPluginError,
    },
}

pub struct EnvironmentResetRunner<
    InitialState,
    Observation,
    PolicyAction,
    SimulatorAction,
    AuxiliaryInfo,
> {
    seed_source: EnvironmentSeedSource<InitialState>,
    simulator_factory: Box<
        dyn EnvironmentSimulatorFactory<InitialState, Observation, PolicyAction, SimulatorAction>,
    >,
    observation_space: Box<dyn EnvironmentObservationSpace<Observation>>,
    step_runner: EnvironmentStepRunner<
        InitialState,
        Observation,
        PolicyAction,
        SimulatorAction,
        AuxiliaryInfo,
    >,
}

impl<InitialState, Observation, PolicyAction, SimulatorAction, AuxiliaryInfo>
    EnvironmentResetRunner<InitialState, Observation, PolicyAction, SimulatorAction, AuxiliaryInfo>
where
    InitialState: Clone + 'static,
    Observation: Clone + Send + Sync + 'static,
    PolicyAction: Clone + Send + Sync + 'static,
    SimulatorAction: 'static,
    AuxiliaryInfo: Default + 'static,
{
    #[must_use]
    pub fn new(
        seed_source: EnvironmentSeedSource<InitialState>,
        simulator_factory: Box<
            dyn EnvironmentSimulatorFactory<
                    InitialState,
                    Observation,
                    PolicyAction,
                    SimulatorAction,
                >,
        >,
        observation_space: Box<dyn EnvironmentObservationSpace<Observation>>,
        step_runner: EnvironmentStepRunner<
            InitialState,
            Observation,
            PolicyAction,
            SimulatorAction,
            AuxiliaryInfo,
        >,
    ) -> Self {
        Self {
            seed_source,
            simulator_factory,
            observation_space,
            step_runner,
        }
    }

    #[must_use]
    pub const fn is_exhausted(&self) -> bool {
        self.seed_source.is_exhausted()
    }

    #[must_use]
    pub fn status(&self) -> Option<&EnvironmentStatus<InitialState, Observation, PolicyAction>> {
        self.step_runner.try_status()
    }

    #[must_use]
    pub const fn step_runner(
        &self,
    ) -> &EnvironmentStepRunner<
        InitialState,
        Observation,
        PolicyAction,
        SimulatorAction,
        AuxiliaryInfo,
    > {
        &self.step_runner
    }

    /// # Errors
    /// Returns `Dead` after permanent exhaustion, or the exact reset plugin stage that failed.
    pub fn reset(&mut self) -> Result<Observation, EnvironmentResetError> {
        let initial_state = match &mut self.seed_source {
            EnvironmentSeedSource::Exhausted => return Err(EnvironmentResetError::Dead),
            EnvironmentSeedSource::Unseeded => {
                self.step_runner.logger().reset();
                None
            }
            EnvironmentSeedSource::Seeded(iterator) => {
                self.step_runner.logger().reset();
                match iterator.next() {
                    Some(initial_state) => Some(initial_state),
                    None => return self.exhaust(),
                }
            }
            EnvironmentSeedSource::TrySeeded(iterator) => {
                self.step_runner.logger().reset();
                match iterator.next() {
                    Some(Ok(initial_state)) => Some(initial_state),
                    Some(Err(source)) if source.is_stop_iteration() => return self.exhaust(),
                    Some(Err(source)) => {
                        return Err(EnvironmentResetError::Component {
                            stage: EnvironmentResetStage::SeedIterator,
                            source,
                        });
                    }
                    None => return self.exhaust(),
                }
            }
        };
        let simulator = match self.simulator_factory.create(initial_state.as_ref()) {
            Ok(simulator) => simulator,
            Err(source) if source.is_stop_iteration() => return self.exhaust(),
            Err(source) => {
                return Err(EnvironmentResetError::Component {
                    stage: EnvironmentResetStage::SimulatorFactory,
                    source,
                });
            }
        };
        match self.step_runner.reset_episode(simulator, initial_state) {
            Ok(observation) => Ok(observation),
            Err(source) if source.is_stop_iteration() => self.exhaust(),
            Err(source) => Err(EnvironmentResetError::Component {
                stage: EnvironmentResetStage::InitialObservation,
                source,
            }),
        }
    }

    /// # Errors
    /// Delegates to the ordered step runner.
    pub fn step(
        &mut self,
        policy_action: PolicyAction,
    ) -> Result<EnvironmentStepOutput<Observation, PolicyAction, AuxiliaryInfo>, EnvironmentStepError>
    {
        self.step_runner.step(policy_action)
    }

    fn exhaust(&mut self) -> Result<Observation, EnvironmentResetError> {
        self.seed_source = EnvironmentSeedSource::Exhausted;
        self.step_runner.mark_exhausted();
        self.observation_space
            .invalid_observation()
            .map_err(|source| EnvironmentResetError::Component {
                stage: EnvironmentResetStage::InvalidObservation,
                source,
            })
    }
}
