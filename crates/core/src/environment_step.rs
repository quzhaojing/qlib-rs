//! Ordered reinforcement-learning environment step orchestration.

use std::sync::{Arc, Mutex};

use indexmap::IndexMap;
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::{
    SaoeActionInterpreter, SaoeObservation, SaoePolicyAction, SaoeReward, SaoeRewardError,
    SaoeRewardLogError, SaoeRewardLogSink, SaoeState, SaoeStateInterpreter,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(i32)]
pub enum EnvironmentLogLevel {
    Debug = 10,
    Periodic = 20,
    Info = 30,
    Critical = 40,
}

#[derive(Clone, Debug, PartialEq)]
pub enum EnvironmentLogValue<Observation, PolicyAction> {
    Scalar(f64),
    Observation(Observation),
    PolicyAction(PolicyAction),
}

#[derive(Clone, Debug, PartialEq)]
pub struct EnvironmentLogEntry<Observation, PolicyAction> {
    pub level: EnvironmentLogLevel,
    pub value: EnvironmentLogValue<Observation, PolicyAction>,
}

pub type EnvironmentLogs<Observation, PolicyAction> =
    IndexMap<String, EnvironmentLogEntry<Observation, PolicyAction>>;

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum EnvironmentLogError {
    #[error("environment metric is already logged: {0}")]
    DuplicateMetric(String),
    #[error("environment step count cannot be converted to Python-compatible float: {0}")]
    StepCountNotRepresentable(BigInt),
}

pub struct EnvironmentLogCollector<Observation, PolicyAction> {
    minimum_level: EnvironmentLogLevel,
    logs: Mutex<EnvironmentLogs<Observation, PolicyAction>>,
}

impl<Observation: Clone, PolicyAction: Clone> EnvironmentLogCollector<Observation, PolicyAction> {
    #[must_use]
    pub fn new(minimum_level: EnvironmentLogLevel) -> Self {
        Self {
            minimum_level,
            logs: Mutex::new(IndexMap::new()),
        }
    }

    /// # Panics
    /// Panics only if another thread previously panicked while holding the collector lock.
    pub fn reset(&self) {
        self.logs
            .lock()
            .expect("environment log mutex poisoned")
            .clear();
    }

    /// # Errors
    /// Returns a duplicate-name error for metrics retained at this log level.
    pub fn add_scalar(
        &self,
        name: &str,
        value: f64,
        level: EnvironmentLogLevel,
    ) -> Result<(), EnvironmentLogError> {
        self.insert(name, EnvironmentLogValue::Scalar(value), level)
    }

    /// # Errors
    /// Returns conversion or duplicate-name errors.
    pub fn add_step_count(
        &self,
        name: &str,
        value: &BigInt,
        level: EnvironmentLogLevel,
    ) -> Result<(), EnvironmentLogError> {
        if level < self.minimum_level {
            return Ok(());
        }
        let scalar = value
            .to_f64()
            .filter(|scalar| scalar.is_finite())
            .ok_or_else(|| EnvironmentLogError::StepCountNotRepresentable(value.clone()))?;
        self.insert(name, EnvironmentLogValue::Scalar(scalar), level)
    }

    /// # Errors
    /// Returns a duplicate-name error for metrics retained at this log level.
    pub fn add_observation(
        &self,
        name: &str,
        value: &Observation,
        level: EnvironmentLogLevel,
    ) -> Result<(), EnvironmentLogError> {
        self.insert(name, EnvironmentLogValue::Observation(value.clone()), level)
    }

    /// # Errors
    /// Returns a duplicate-name error for metrics retained at this log level.
    pub fn add_policy_action(
        &self,
        name: &str,
        value: &PolicyAction,
        level: EnvironmentLogLevel,
    ) -> Result<(), EnvironmentLogError> {
        self.insert(
            name,
            EnvironmentLogValue::PolicyAction(value.clone()),
            level,
        )
    }

    #[must_use]
    /// # Panics
    /// Panics only if another thread previously panicked while holding the collector lock.
    pub fn snapshot(&self) -> EnvironmentLogs<Observation, PolicyAction> {
        self.logs
            .lock()
            .expect("environment log mutex poisoned")
            .clone()
    }

    fn insert(
        &self,
        name: &str,
        value: EnvironmentLogValue<Observation, PolicyAction>,
        level: EnvironmentLogLevel,
    ) -> Result<(), EnvironmentLogError> {
        if level < self.minimum_level {
            return Ok(());
        }
        let mut logs = self.logs.lock().expect("environment log mutex poisoned");
        if logs.contains_key(name) {
            return Err(EnvironmentLogError::DuplicateMetric(name.to_owned()));
        }
        logs.insert(name.to_owned(), EnvironmentLogEntry { level, value });
        Ok(())
    }
}

impl<Observation, PolicyAction> SaoeRewardLogSink
    for EnvironmentLogCollector<Observation, PolicyAction>
where
    Observation: Clone + Send + Sync,
    PolicyAction: Clone + Send + Sync,
{
    fn log_scalar(&self, name: &str, value: f64) -> Result<(), SaoeRewardLogError> {
        self.add_scalar(name, value, EnvironmentLogLevel::Periodic)
            .map_err(|error| SaoeRewardLogError {
                message: error.to_string(),
            })
    }
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("environment plugin error: {message}")]
pub struct EnvironmentPluginError {
    pub message: String,
    stop_iteration: bool,
}

impl EnvironmentPluginError {
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            stop_iteration: false,
        }
    }

    #[must_use]
    pub fn stop_iteration() -> Self {
        Self {
            message: "iteration exhausted".to_owned(),
            stop_iteration: true,
        }
    }

    #[must_use]
    pub const fn is_stop_iteration(&self) -> bool {
        self.stop_iteration
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnvironmentStepStage {
    PreActionState,
    ActionInterpreter,
    SimulatorStep,
    SimulatorDone,
    PostStepState,
    StateInterpreter,
    AuxiliaryInfo,
    StepsPerEpisodeLog,
    RewardLog,
    ObservationLog,
    PolicyActionLog,
}

#[derive(Debug, Error)]
pub enum EnvironmentRewardError {
    #[error(transparent)]
    Plugin(#[from] EnvironmentPluginError),
    #[error(transparent)]
    Saoe(#[from] SaoeRewardError),
}

#[derive(Debug, Error)]
pub enum EnvironmentStepError {
    #[error("environment seed iterator is exhausted")]
    Exhausted,
    #[error("environment has not been reset")]
    NotReset,
    #[error("environment component failed at {stage:?}: {source}")]
    Component {
        stage: EnvironmentStepStage,
        #[source]
        source: EnvironmentPluginError,
    },
    #[error(transparent)]
    Reward(#[from] EnvironmentRewardError),
    #[error("environment logging failed at {stage:?}: {source}")]
    Log {
        stage: EnvironmentStepStage,
        #[source]
        source: EnvironmentLogError,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EnvironmentStatus<InitialState, Observation, PolicyAction> {
    pub cur_step: BigInt,
    pub done: bool,
    pub initial_state: Option<InitialState>,
    pub observation_history: Vec<Observation>,
    pub action_history: Vec<PolicyAction>,
    pub reward_history: Vec<f64>,
}

impl<InitialState, Observation, PolicyAction>
    EnvironmentStatus<InitialState, Observation, PolicyAction>
{
    #[must_use]
    pub fn empty(initial_state: Option<InitialState>) -> Self {
        Self {
            cur_step: BigInt::from(0_u8),
            done: false,
            initial_state,
            observation_history: Vec::new(),
            action_history: Vec::new(),
            reward_history: Vec::new(),
        }
    }

    #[must_use]
    pub fn new(initial_state: Option<InitialState>, initial_observation: Observation) -> Self {
        let mut status = Self::empty(initial_state);
        status.observation_history.push(initial_observation);
        status
    }
}

pub type EnvironmentAuxInfo = IndexMap<String, Value>;

#[derive(Clone, Debug, PartialEq)]
pub struct EnvironmentStepInfo<Observation, PolicyAction, AuxiliaryInfo> {
    pub logs: EnvironmentLogs<Observation, PolicyAction>,
    pub auxiliary_info: AuxiliaryInfo,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EnvironmentStepOutput<Observation, PolicyAction, AuxiliaryInfo> {
    pub observation: Observation,
    pub reward: f64,
    pub done: bool,
    pub info: EnvironmentStepInfo<Observation, PolicyAction, AuxiliaryInfo>,
}

pub trait EnvironmentSimulator<InitialState, Observation, PolicyAction, SimulatorAction>:
    Send
{
    fn set_logger(&mut self, _logger: Option<Arc<dyn SaoeRewardLogSink>>) {}

    /// # Errors
    /// Returns simulator state retrieval failures.
    fn state(
        &self,
        status: &EnvironmentStatus<InitialState, Observation, PolicyAction>,
    ) -> Result<SaoeState, EnvironmentPluginError>;

    /// # Errors
    /// Returns simulator transition failures.
    fn step(
        &mut self,
        action: SimulatorAction,
        status: &EnvironmentStatus<InitialState, Observation, PolicyAction>,
    ) -> Result<(), EnvironmentPluginError>;

    /// # Errors
    /// Returns terminal-state lookup failures.
    fn done(
        &self,
        status: &EnvironmentStatus<InitialState, Observation, PolicyAction>,
    ) -> Result<bool, EnvironmentPluginError>;
}

pub trait EnvironmentActionInterpreter<InitialState, Observation, PolicyAction, SimulatorAction>:
    Send
{
    fn set_logger(&mut self, _logger: Option<Arc<dyn SaoeRewardLogSink>>) {}

    /// # Errors
    /// Returns validation or interpretation failures.
    fn interpret(
        &mut self,
        state: &SaoeState,
        action: &PolicyAction,
        status: &EnvironmentStatus<InitialState, Observation, PolicyAction>,
    ) -> Result<SimulatorAction, EnvironmentPluginError>;
}

pub trait EnvironmentStateInterpreter<InitialState, Observation, PolicyAction>: Send {
    fn set_logger(&mut self, _logger: Option<Arc<dyn SaoeRewardLogSink>>) {}

    /// # Errors
    /// Returns validation or interpretation failures.
    fn interpret(
        &mut self,
        state: &SaoeState,
        status: &EnvironmentStatus<InitialState, Observation, PolicyAction>,
    ) -> Result<Observation, EnvironmentPluginError>;
}

pub trait EnvironmentReward<InitialState, Observation, PolicyAction>: Send {
    fn set_logger(&mut self, _logger: Option<Arc<dyn SaoeRewardLogSink>>) {}

    /// # Errors
    /// Returns reward plugin failures.
    fn reward(
        &mut self,
        state: &SaoeState,
        status: &EnvironmentStatus<InitialState, Observation, PolicyAction>,
    ) -> Result<f64, EnvironmentRewardError>;
}

pub trait EnvironmentAuxiliaryInfo<InitialState, Observation, PolicyAction, AuxiliaryInfo>:
    Send
{
    fn set_logger(&mut self, _logger: Option<Arc<dyn SaoeRewardLogSink>>) {}

    /// # Errors
    /// Returns auxiliary-info plugin failures.
    fn collect(
        &mut self,
        state: &SaoeState,
        status: &EnvironmentStatus<InitialState, Observation, PolicyAction>,
    ) -> Result<AuxiliaryInfo, EnvironmentPluginError>;
}

/// Connects an existing stateless SAOE interpreter without projecting away observation fields.
pub struct SaoeEnvironmentStateInterpreter {
    interpreter: Box<dyn SaoeStateInterpreter>,
}

impl SaoeEnvironmentStateInterpreter {
    #[must_use]
    pub fn new(interpreter: Box<dyn SaoeStateInterpreter>) -> Self {
        Self { interpreter }
    }
}

impl<InitialState, PolicyAction>
    EnvironmentStateInterpreter<InitialState, SaoeObservation, PolicyAction>
    for SaoeEnvironmentStateInterpreter
{
    fn interpret(
        &mut self,
        state: &SaoeState,
        _status: &EnvironmentStatus<InitialState, SaoeObservation, PolicyAction>,
    ) -> Result<SaoeObservation, EnvironmentPluginError> {
        self.interpreter
            .interpret(state)
            .map_err(|error| EnvironmentPluginError::new(error.to_string()))
    }
}

/// Connects discrete or continuous SAOE actions to the environment's execution-volume boundary.
pub struct SaoeEnvironmentActionInterpreter {
    interpreter: Box<dyn SaoeActionInterpreter>,
}

impl SaoeEnvironmentActionInterpreter {
    #[must_use]
    pub fn new(interpreter: Box<dyn SaoeActionInterpreter>) -> Self {
        Self { interpreter }
    }
}

impl<InitialState, Observation>
    EnvironmentActionInterpreter<InitialState, Observation, SaoePolicyAction, f64>
    for SaoeEnvironmentActionInterpreter
{
    fn interpret(
        &mut self,
        state: &SaoeState,
        action: &SaoePolicyAction,
        _status: &EnvironmentStatus<InitialState, Observation, SaoePolicyAction>,
    ) -> Result<f64, EnvironmentPluginError> {
        self.interpreter
            .interpret(state, *action)
            .map_err(|error| EnvironmentPluginError::new(error.to_string()))
    }
}

pub struct SaoeEnvironmentReward {
    reward: Box<dyn SaoeReward>,
}

impl SaoeEnvironmentReward {
    #[must_use]
    pub fn new(reward: Box<dyn SaoeReward>) -> Self {
        Self { reward }
    }
}

impl<InitialState, Observation, PolicyAction>
    EnvironmentReward<InitialState, Observation, PolicyAction> for SaoeEnvironmentReward
{
    fn set_logger(&mut self, logger: Option<Arc<dyn SaoeRewardLogSink>>) {
        self.reward.set_logger(logger);
    }

    fn reward(
        &mut self,
        state: &SaoeState,
        _status: &EnvironmentStatus<InitialState, Observation, PolicyAction>,
    ) -> Result<f64, EnvironmentRewardError> {
        self.reward.reward(state).map_err(Into::into)
    }
}

pub struct EnvironmentStepRunner<
    InitialState,
    Observation,
    PolicyAction,
    SimulatorAction,
    AuxiliaryInfo,
> {
    episode: Option<EnvironmentEpisode<InitialState, Observation, PolicyAction, SimulatorAction>>,
    state_interpreter:
        Box<dyn EnvironmentStateInterpreter<InitialState, Observation, PolicyAction>>,
    action_interpreter: Box<
        dyn EnvironmentActionInterpreter<InitialState, Observation, PolicyAction, SimulatorAction>,
    >,
    reward: Option<Box<dyn EnvironmentReward<InitialState, Observation, PolicyAction>>>,
    auxiliary_info: Option<
        Box<dyn EnvironmentAuxiliaryInfo<InitialState, Observation, PolicyAction, AuxiliaryInfo>>,
    >,
    logger: Arc<EnvironmentLogCollector<Observation, PolicyAction>>,
    active: bool,
}

struct EnvironmentEpisode<InitialState, Observation, PolicyAction, SimulatorAction> {
    simulator:
        Box<dyn EnvironmentSimulator<InitialState, Observation, PolicyAction, SimulatorAction>>,
    status: EnvironmentStatus<InitialState, Observation, PolicyAction>,
}

impl<InitialState, Observation, PolicyAction, SimulatorAction, AuxiliaryInfo>
    EnvironmentStepRunner<InitialState, Observation, PolicyAction, SimulatorAction, AuxiliaryInfo>
where
    Observation: Clone + Send + Sync + 'static,
    PolicyAction: Clone + Send + Sync + 'static,
    InitialState: 'static,
    SimulatorAction: 'static,
    AuxiliaryInfo: Default + 'static,
{
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors Qlib EnvWrapper component assembly"
    )]
    #[must_use]
    pub fn new(
        mut simulator: Box<
            dyn EnvironmentSimulator<InitialState, Observation, PolicyAction, SimulatorAction>,
        >,
        mut state_interpreter: Box<
            dyn EnvironmentStateInterpreter<InitialState, Observation, PolicyAction>,
        >,
        mut action_interpreter: Box<
            dyn EnvironmentActionInterpreter<
                    InitialState,
                    Observation,
                    PolicyAction,
                    SimulatorAction,
                >,
        >,
        mut reward: Option<Box<dyn EnvironmentReward<InitialState, Observation, PolicyAction>>>,
        mut auxiliary_info: Option<
            Box<
                dyn EnvironmentAuxiliaryInfo<InitialState, Observation, PolicyAction, AuxiliaryInfo>,
            >,
        >,
        logger: Arc<EnvironmentLogCollector<Observation, PolicyAction>>,
        initial_state: Option<InitialState>,
        initial_observation: Observation,
    ) -> Self {
        let sink: Arc<dyn SaoeRewardLogSink> = logger.clone();
        simulator.set_logger(Some(Arc::clone(&sink)));
        state_interpreter.set_logger(Some(Arc::clone(&sink)));
        action_interpreter.set_logger(Some(Arc::clone(&sink)));
        if let Some(reward) = &mut reward {
            reward.set_logger(Some(Arc::clone(&sink)));
        }
        if let Some(auxiliary_info) = &mut auxiliary_info {
            auxiliary_info.set_logger(Some(sink));
        }
        Self {
            episode: Some(EnvironmentEpisode {
                simulator,
                status: EnvironmentStatus::new(initial_state, initial_observation),
            }),
            state_interpreter,
            action_interpreter,
            reward,
            auxiliary_info,
            logger,
            active: true,
        }
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors Qlib EnvWrapper component assembly"
    )]
    #[must_use]
    pub fn uninitialized(
        mut state_interpreter: Box<
            dyn EnvironmentStateInterpreter<InitialState, Observation, PolicyAction>,
        >,
        mut action_interpreter: Box<
            dyn EnvironmentActionInterpreter<
                    InitialState,
                    Observation,
                    PolicyAction,
                    SimulatorAction,
                >,
        >,
        mut reward: Option<Box<dyn EnvironmentReward<InitialState, Observation, PolicyAction>>>,
        mut auxiliary_info: Option<
            Box<
                dyn EnvironmentAuxiliaryInfo<InitialState, Observation, PolicyAction, AuxiliaryInfo>,
            >,
        >,
        logger: Arc<EnvironmentLogCollector<Observation, PolicyAction>>,
    ) -> Self {
        let sink: Arc<dyn SaoeRewardLogSink> = logger.clone();
        state_interpreter.set_logger(Some(Arc::clone(&sink)));
        action_interpreter.set_logger(Some(Arc::clone(&sink)));
        if let Some(reward) = &mut reward {
            reward.set_logger(Some(Arc::clone(&sink)));
        }
        if let Some(auxiliary_info) = &mut auxiliary_info {
            auxiliary_info.set_logger(Some(sink));
        }
        Self {
            episode: None,
            state_interpreter,
            action_interpreter,
            reward,
            auxiliary_info,
            logger,
            active: true,
        }
    }

    #[must_use]
    /// # Panics
    /// Panics when called before the first successful or partially initialized reset. Use
    /// [`Self::try_status`] when the lifecycle state is not already known.
    pub fn status(&self) -> &EnvironmentStatus<InitialState, Observation, PolicyAction> {
        self.episode
            .as_ref()
            .map(|episode| &episode.status)
            .expect("environment status is unavailable before reset")
    }

    #[must_use]
    pub const fn try_status(
        &self,
    ) -> Option<&EnvironmentStatus<InitialState, Observation, PolicyAction>> {
        match &self.episode {
            Some(episode) => Some(&episode.status),
            None => None,
        }
    }

    #[must_use]
    pub fn logger(&self) -> Arc<EnvironmentLogCollector<Observation, PolicyAction>> {
        Arc::clone(&self.logger)
    }

    pub fn mark_exhausted(&mut self) {
        self.active = false;
    }

    /// Installs a fresh simulator and builds the initial observation.
    ///
    /// The empty status is installed before simulator/state-interpreter calls, matching
    /// Python's partially observable reset mutation order.
    ///
    /// # Errors
    /// Returns the plugin failure from initial state retrieval or interpretation.
    ///
    /// # Panics
    /// Internal invariant checks panic only if the simulator or status assigned immediately
    /// beforehand becomes unavailable without this method mutating it.
    pub fn reset_episode(
        &mut self,
        mut simulator: Box<
            dyn EnvironmentSimulator<InitialState, Observation, PolicyAction, SimulatorAction>,
        >,
        initial_state: Option<InitialState>,
    ) -> Result<Observation, EnvironmentPluginError> {
        let sink: Arc<dyn SaoeRewardLogSink> = self.logger.clone();
        simulator.set_logger(Some(sink));
        self.episode = Some(EnvironmentEpisode {
            simulator,
            status: EnvironmentStatus::empty(initial_state),
        });
        self.active = true;
        let episode = self.episode.as_mut().expect("episode was just installed");
        let simulator_state = episode.simulator.state(&episode.status)?;
        let observation = self
            .state_interpreter
            .interpret(&simulator_state, &episode.status)?;
        episode.status.observation_history.push(observation.clone());
        Ok(observation)
    }

    /// # Errors
    /// Returns the exact component, reward, or logging stage that failed.
    pub fn step(
        &mut self,
        policy_action: PolicyAction,
    ) -> Result<EnvironmentStepOutput<Observation, PolicyAction, AuxiliaryInfo>, EnvironmentStepError>
    {
        if !self.active {
            return Err(EnvironmentStepError::Exhausted);
        }
        let episode = self
            .episode
            .as_mut()
            .ok_or(EnvironmentStepError::NotReset)?;
        let simulator = &mut episode.simulator;
        let status = &mut episode.status;
        self.logger.reset();
        status.action_history.push(policy_action.clone());
        let pre_state =
            simulator
                .state(status)
                .map_err(|source| EnvironmentStepError::Component {
                    stage: EnvironmentStepStage::PreActionState,
                    source,
                })?;
        let action = self
            .action_interpreter
            .interpret(&pre_state, &policy_action, status)
            .map_err(|source| EnvironmentStepError::Component {
                stage: EnvironmentStepStage::ActionInterpreter,
                source,
            })?;
        status.cur_step += 1_u8;
        simulator
            .step(action, status)
            .map_err(|source| EnvironmentStepError::Component {
                stage: EnvironmentStepStage::SimulatorStep,
                source,
            })?;
        let done = simulator
            .done(status)
            .map_err(|source| EnvironmentStepError::Component {
                stage: EnvironmentStepStage::SimulatorDone,
                source,
            })?;
        status.done = done;
        let simulator_state =
            simulator
                .state(status)
                .map_err(|source| EnvironmentStepError::Component {
                    stage: EnvironmentStepStage::PostStepState,
                    source,
                })?;
        let observation = self
            .state_interpreter
            .interpret(&simulator_state, status)
            .map_err(|source| EnvironmentStepError::Component {
                stage: EnvironmentStepStage::StateInterpreter,
                source,
            })?;
        status.observation_history.push(observation.clone());
        let reward = if let Some(reward) = &mut self.reward {
            reward.reward(&simulator_state, status)?
        } else {
            0.0
        };
        status.reward_history.push(reward);
        let auxiliary_info = if let Some(auxiliary_info) = &mut self.auxiliary_info {
            auxiliary_info
                .collect(&simulator_state, status)
                .map_err(|source| EnvironmentStepError::Component {
                    stage: EnvironmentStepStage::AuxiliaryInfo,
                    source,
                })?
        } else {
            AuxiliaryInfo::default()
        };
        let logs = self.finalize_logs(done, reward, &observation, &policy_action)?;
        Ok(EnvironmentStepOutput {
            observation,
            reward,
            done,
            info: EnvironmentStepInfo {
                logs,
                auxiliary_info,
            },
        })
    }

    fn finalize_logs(
        &self,
        done: bool,
        reward: f64,
        observation: &Observation,
        policy_action: &PolicyAction,
    ) -> Result<EnvironmentLogs<Observation, PolicyAction>, EnvironmentStepError> {
        if done {
            self.logger
                .add_step_count(
                    "steps_per_episode",
                    &self.status().cur_step,
                    EnvironmentLogLevel::Periodic,
                )
                .map_err(|source| EnvironmentStepError::Log {
                    stage: EnvironmentStepStage::StepsPerEpisodeLog,
                    source,
                })?;
        }
        self.logger
            .add_scalar("reward", reward, EnvironmentLogLevel::Periodic)
            .map_err(|source| EnvironmentStepError::Log {
                stage: EnvironmentStepStage::RewardLog,
                source,
            })?;
        self.logger
            .add_observation("obs", observation, EnvironmentLogLevel::Debug)
            .map_err(|source| EnvironmentStepError::Log {
                stage: EnvironmentStepStage::ObservationLog,
                source,
            })?;
        self.logger
            .add_policy_action("policy_act", policy_action, EnvironmentLogLevel::Debug)
            .map_err(|source| EnvironmentStepError::Log {
                stage: EnvironmentStepStage::PolicyActionLog,
                source,
            })?;
        Ok(self.logger.snapshot())
    }
}
