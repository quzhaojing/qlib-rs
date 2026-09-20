//! Policy and interpreter boundaries for single-asset order execution.

use std::sync::Arc;

use arrow_array::{Float32Array, Float64Array};
use chrono::{NaiveDate, NaiveDateTime};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{LiveSaoeState, SaoeState};

/// Owned observation accepted by SAOE policies.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SaoeObservation {
    Dummy { dummy: i32 },
    CurrentStep(CurrentStepObservation),
    FullHistory(FullHistoryObservation),
}

/// Current-step observation produced by Qlib's lightweight interpreter.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CurrentStepObservation {
    pub acquiring: bool,
    pub cur_step: i64,
    pub num_step: i64,
    pub target: f64,
    pub position: f64,
}

/// Fixed-shape observation consumed by Qlib's recurrent and attention policies.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FullHistoryObservation {
    pub data_processed: Array2<f32>,
    pub data_processed_prev: Array2<f32>,
    pub acquiring: i32,
    pub cur_tick: i32,
    pub cur_step: i32,
    pub num_step: i32,
    pub target: f32,
    pub position: f32,
    pub position_history: Array1<f32>,
}

/// Owned result from the processed-feature plugin.
#[derive(Clone, Debug, PartialEq)]
pub struct ProcessedSaoeData {
    pub today: Array2<f32>,
    pub yesterday: Array2<f32>,
    pub today_index: Vec<NaiveDateTime>,
}

/// Processed feature provider used by the full-history interpreter.
pub trait SaoeProcessedDataProvider: Send + Sync {
    /// Load today's and yesterday's fixed-width features.
    ///
    /// # Errors
    /// Returns data-source, schema, or transport failures.
    fn get_data(
        &self,
        stock_id: &str,
        date: NaiveDate,
        feature_dim: usize,
        time_index: &[NaiveDateTime],
    ) -> Result<ProcessedSaoeData, SaoeInterpreterError>;
}

/// Typed scalar action emitted by a policy.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum SaoePolicyAction {
    Discrete(i64),
    Continuous(f64),
}

/// One policy action paired with the execution volume produced from it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SaoePolicyDecision {
    pub action: SaoePolicyAction,
    pub execution_volume: f64,
}

/// Serializable action-space description used at plugin boundaries.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum SaoeActionSpace {
    Discrete { size: usize },
    NonNegativeContinuous,
}

/// Interpreter and baseline-policy diagnostics.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum SaoeInterpreterError {
    #[error("max_step must be positive, got {0}")]
    InvalidMaxStep(i64),
    #[error("categorical action count must be positive")]
    EmptyCategoricalCount,
    #[error("categorical action values cannot be empty")]
    EmptyCategoricalValues,
    #[error("expected a discrete action")]
    ExpectedDiscrete,
    #[error("expected a continuous action")]
    ExpectedContinuous,
    #[error("discrete action {action} is outside 0..{size}")]
    DiscreteOutOfRange { action: i64, size: usize },
    #[error("continuous action must be non-negative")]
    NegativeContinuous,
    #[error("ticks_per_step must be positive")]
    ZeroTicksPerStep,
    #[error("TWAP has no remaining execution steps")]
    NoRemainingSteps,
    #[error("{name} must be positive and fit in int32, got {value}")]
    InvalidObservationDimension { name: &'static str, value: usize },
    #[error("processed {dataset} shape is {actual:?}, expected {expected:?}")]
    ProcessedShape {
        dataset: &'static str,
        expected: (usize, usize),
        actual: (usize, usize),
    },
    #[error("processed today index has {actual} rows, expected {expected}")]
    ProcessedIndexLength { expected: usize, actual: usize },
    #[error("state history_steps has no position column")]
    MissingHistoryPosition,
    #[error("state history_steps position must be Float32 or Float64, got {0}")]
    InvalidHistoryPositionType(String),
    #[error("state history_steps position contains null values")]
    NullHistoryPosition,
    #[error("state has {actual} history steps, exceeding max_step {max_step}")]
    HistoryTooLong { actual: usize, max_step: usize },
    #[error("full-history observation requires an order start time")]
    MissingOrderStartTime,
    #[error("processed-data plugin failed: {0}")]
    ProcessedDataPlugin(String),
    #[error("policy returned {actual} actions for {expected} observations")]
    PolicyBatchLength { expected: usize, actual: usize },
    #[error("live SAOE state materialization failed: {0}")]
    LiveState(String),
}

/// Stateless state-to-observation plugin.
pub trait SaoeStateInterpreter: Send + Sync {
    /// Convert one simulator snapshot into an owned policy observation.
    ///
    /// # Errors
    /// Returns invalid interpreter configuration or state bounds.
    fn interpret(&self, state: &SaoeState) -> Result<SaoeObservation, SaoeInterpreterError>;

    /// Interpret the original live aliases. Implementations may override this to observe or
    /// mutate identities directly; the default materializes them at this callback boundary.
    ///
    /// # Errors
    /// Returns alias materialization or ordinary interpretation failures.
    fn interpret_live(
        &self,
        state: &LiveSaoeState,
    ) -> Result<SaoeObservation, SaoeInterpreterError> {
        self.interpret(
            &state
                .snapshot()
                .map_err(|error| SaoeInterpreterError::LiveState(error.to_string()))?,
        )
    }
}

/// Stateless policy-action-to-volume plugin.
pub trait SaoeActionInterpreter: Send + Sync {
    fn action_space(&self) -> SaoeActionSpace;

    /// Convert one validated policy action into an execution volume.
    ///
    /// # Errors
    /// Returns action type, range, or state-shape failures.
    fn interpret(
        &self,
        state: &SaoeState,
        action: SaoePolicyAction,
    ) -> Result<f64, SaoeInterpreterError>;

    /// Interpret an action against the retained live aliases. The default takes a fresh snapshot
    /// after policy inference, matching the source's reuse of the same state object.
    ///
    /// # Errors
    /// Returns alias materialization or ordinary action interpretation failures.
    fn interpret_live(
        &self,
        state: &LiveSaoeState,
        action: SaoePolicyAction,
    ) -> Result<f64, SaoeInterpreterError> {
        self.interpret(
            &state
                .snapshot()
                .map_err(|error| SaoeInterpreterError::LiveState(error.to_string()))?,
            action,
        )
    }
}

/// Batched policy plugin used by configured SAOE strategies.
pub trait SaoePolicy: Send {
    /// Produce policy actions for a batch of observations.
    /// The strict pipeline checks cardinality; the source-compatible pull pipeline
    /// preserves Python's zip truncation for shorter or longer outputs.
    ///
    /// # Errors
    /// Returns inference or output-shape failures.
    fn actions(
        &mut self,
        observations: &[SaoeObservation],
    ) -> Result<Vec<SaoePolicyAction>, SaoeInterpreterError>;
}

/// Stable-order state → observation → policy action → execution-volume pipeline.
pub struct SaoePolicyPipeline {
    state_interpreter: Box<dyn SaoeStateInterpreter>,
    policy: Box<dyn SaoePolicy>,
    action_interpreter: Box<dyn SaoeActionInterpreter>,
}

impl SaoePolicyPipeline {
    /// Pull and immediately interpret each state before requesting the next one.
    /// This preserves callback-visible changes to a live order iterator. The caller
    /// owns that iterator and must release its list/order guards before returning.
    /// States are retained for action interpretation after one batched policy call,
    /// including for an empty input. Action/state pairing truncates like Python zip.
    ///
    /// This boundary accepts existing typed state values; it does not itself obtain
    /// live orders, rebuild adapters, create decisions or preserve arbitrary Python
    /// object aliases within a state value.
    ///
    /// # Errors
    /// Returns the first state-source or interpreter/policy failure in source order.
    /// No later callback runs after failure; prior callback effects are retained.
    pub fn decisions_from<E>(
        &mut self,
        mut next_state: impl FnMut() -> Result<Option<SaoeState>, E>,
    ) -> Result<Vec<SaoePolicyDecision>, E>
    where
        E: From<SaoeInterpreterError>,
    {
        let mut states = Vec::new();
        let mut observations = Vec::new();
        while let Some(state) = next_state()? {
            observations.push(self.state_interpreter.interpret(&state)?);
            states.push(state);
        }
        let actions = self.policy.actions(&observations)?;
        states
            .iter()
            .zip(actions)
            .map(|(state, action)| {
                let execution_volume = self.action_interpreter.interpret(state, action)?;
                Ok(SaoePolicyDecision {
                    action,
                    execution_volume,
                })
            })
            .collect()
    }

    /// Pull and immediately interpret live alias states, preserving their identities through the
    /// later action callback. Default interpreters materialize at each reached callback; alias-
    /// aware plugins can override the live methods.
    ///
    /// # Errors
    /// Returns the first state-source, alias, interpreter, policy, or action failure.
    pub fn decisions_from_live<E>(
        &mut self,
        mut next_state: impl FnMut() -> Result<Option<LiveSaoeState>, E>,
    ) -> Result<Vec<SaoePolicyDecision>, E>
    where
        E: From<SaoeInterpreterError>,
    {
        let mut states = Vec::new();
        let mut observations = Vec::new();
        while let Some(state) = next_state()? {
            observations.push(self.state_interpreter.interpret_live(&state)?);
            states.push(state);
        }
        let actions = self.policy.actions(&observations)?;
        states
            .iter()
            .zip(actions)
            .map(|(state, action)| {
                let execution_volume = self.action_interpreter.interpret_live(state, action)?;
                Ok(SaoePolicyDecision {
                    action,
                    execution_volume,
                })
            })
            .collect()
    }

    #[must_use]
    pub fn new(
        state_interpreter: Box<dyn SaoeStateInterpreter>,
        policy: Box<dyn SaoePolicy>,
        action_interpreter: Box<dyn SaoeActionInterpreter>,
    ) -> Self {
        Self {
            state_interpreter,
            policy,
            action_interpreter,
        }
    }

    /// Interpret a batch and retain both the action and volume for every input state.
    ///
    /// # Errors
    /// Returns the first state, policy, output-shape, or action interpretation failure.
    pub fn decisions(
        &mut self,
        states: &[SaoeState],
    ) -> Result<Vec<SaoePolicyDecision>, SaoeInterpreterError> {
        let observations = states
            .iter()
            .map(|state| self.state_interpreter.interpret(state))
            .collect::<Result<Vec<_>, _>>()?;
        let actions = self.policy.actions(&observations)?;
        if actions.len() != states.len() {
            return Err(SaoeInterpreterError::PolicyBatchLength {
                expected: states.len(),
                actual: actions.len(),
            });
        }
        states
            .iter()
            .zip(actions)
            .map(|(state, action)| {
                self.action_interpreter
                    .interpret(state, action)
                    .map(|execution_volume| SaoePolicyDecision {
                        action,
                        execution_volume,
                    })
            })
            .collect()
    }

    /// Interpret a batch and return exactly one execution volume per input state.
    ///
    /// # Errors
    /// Returns the first state, policy, output-shape, or action interpretation failure.
    pub fn execution_volumes(
        &mut self,
        states: &[SaoeState],
    ) -> Result<Vec<f64>, SaoeInterpreterError> {
        Ok(self
            .decisions(states)?
            .into_iter()
            .map(|decision| decision.execution_volume)
            .collect())
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DummyStateInterpreter;

impl SaoeStateInterpreter for DummyStateInterpreter {
    fn interpret(&self, _state: &SaoeState) -> Result<SaoeObservation, SaoeInterpreterError> {
        Ok(SaoeObservation::Dummy { dummy: 1 })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CurrentStepStateInterpreter {
    max_step: i64,
}

impl CurrentStepStateInterpreter {
    /// # Errors
    /// Returns when `max_step` cannot describe a non-empty episode.
    pub fn new(max_step: i64) -> Result<Self, SaoeInterpreterError> {
        if max_step <= 0 {
            return Err(SaoeInterpreterError::InvalidMaxStep(max_step));
        }
        Ok(Self { max_step })
    }
}

impl SaoeStateInterpreter for CurrentStepStateInterpreter {
    fn interpret(&self, state: &SaoeState) -> Result<SaoeObservation, SaoeInterpreterError> {
        let parts = state.parts();
        if parts.cur_step > self.max_step {
            return Err(SaoeInterpreterError::InvalidMaxStep(self.max_step));
        }
        Ok(SaoeObservation::CurrentStep(CurrentStepObservation {
            acquiring: parts.order.direction() == crate::OrderDir::Buy,
            cur_step: parts.cur_step,
            num_step: self.max_step,
            target: parts.order.amount(),
            position: parts.position,
        }))
    }
}

pub struct FullHistoryStateInterpreter {
    max_step: usize,
    data_ticks: usize,
    data_dim: usize,
    provider: Arc<dyn SaoeProcessedDataProvider>,
}

impl FullHistoryStateInterpreter {
    /// # Errors
    /// Returns when a fixed observation dimension is zero or cannot fit `NumPy`'s int32 fields.
    pub fn new(
        max_step: usize,
        data_ticks: usize,
        data_dim: usize,
        provider: Arc<dyn SaoeProcessedDataProvider>,
    ) -> Result<Self, SaoeInterpreterError> {
        validate_observation_dimension("max_step", max_step)?;
        validate_observation_dimension("data_ticks", data_ticks)?;
        validate_observation_dimension("data_dim", data_dim)?;
        Ok(Self {
            max_step,
            data_ticks,
            data_dim,
            provider,
        })
    }
}

impl SaoeStateInterpreter for FullHistoryStateInterpreter {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_precision_loss
    )]
    fn interpret(&self, state: &SaoeState) -> Result<SaoeObservation, SaoeInterpreterError> {
        let parts = state.parts();
        let start_time = parts
            .order
            .start_time()
            .ok_or(SaoeInterpreterError::MissingOrderStartTime)?;
        let processed = self.provider.get_data(
            parts.order.stock_id(),
            start_time.date(),
            self.data_dim,
            &parts.ticks_index,
        )?;
        validate_processed_shape(
            "today",
            processed.today.dim(),
            self.data_ticks,
            self.data_dim,
        )?;
        validate_processed_shape(
            "yesterday",
            processed.yesterday.dim(),
            self.data_ticks,
            self.data_dim,
        )?;
        if processed.today_index.len() != self.data_ticks {
            return Err(SaoeInterpreterError::ProcessedIndexLength {
                expected: self.data_ticks,
                actual: processed.today_index.len(),
            });
        }
        let mut today = processed.today;
        for (row, timestamp) in processed.today_index.iter().enumerate() {
            if *timestamp >= parts.cur_time {
                today.row_mut(row).fill(0.0);
            }
        }
        let positions = history_positions(state)?;
        if positions.len() > self.max_step {
            return Err(SaoeInterpreterError::HistoryTooLong {
                actual: positions.len(),
                max_step: self.max_step,
            });
        }
        let mut position_history = Array1::zeros(self.max_step + 1);
        position_history[0] = parts.order.amount() as f32;
        for (index, position) in positions.iter().enumerate() {
            position_history[index + 1] = *position as f32;
        }
        let current_tick = parts
            .ticks_index
            .iter()
            .filter(|timestamp| **timestamp < parts.cur_time)
            .count()
            .min(self.data_ticks - 1);
        let max_observation_step = self.max_step - 1;
        let current_step = parts.cur_step.min(max_observation_step as i64);
        Ok(SaoeObservation::FullHistory(FullHistoryObservation {
            data_processed: today,
            data_processed_prev: processed.yesterday,
            acquiring: i32::from(parts.order.direction() == crate::OrderDir::Buy),
            cur_tick: current_tick as i32,
            cur_step: current_step as i32,
            num_step: self.max_step as i32,
            target: parts.order.amount() as f32,
            position: parts.position as f32,
            position_history: Array1::from_iter(
                position_history.iter().take(self.max_step).copied(),
            ),
        }))
    }
}

fn validate_observation_dimension(
    name: &'static str,
    value: usize,
) -> Result<(), SaoeInterpreterError> {
    if value == 0 || value > i32::MAX as usize {
        return Err(SaoeInterpreterError::InvalidObservationDimension { name, value });
    }
    Ok(())
}

fn validate_processed_shape(
    dataset: &'static str,
    actual: (usize, usize),
    rows: usize,
    columns: usize,
) -> Result<(), SaoeInterpreterError> {
    let expected = (rows, columns);
    if actual != expected {
        return Err(SaoeInterpreterError::ProcessedShape {
            dataset,
            expected,
            actual,
        });
    }
    Ok(())
}

fn history_positions(state: &SaoeState) -> Result<Vec<f64>, SaoeInterpreterError> {
    let batch = &state.parts().history_steps;
    let column = batch
        .column_by_name("position")
        .ok_or(SaoeInterpreterError::MissingHistoryPosition)?;
    if column.null_count() != 0 {
        return Err(SaoeInterpreterError::NullHistoryPosition);
    }
    if let Some(values) = column.as_any().downcast_ref::<Float64Array>() {
        return Ok(values.values().to_vec());
    }
    if let Some(values) = column.as_any().downcast_ref::<Float32Array>() {
        return Ok(values
            .values()
            .iter()
            .map(|value| f64::from(*value))
            .collect());
    }
    Err(SaoeInterpreterError::InvalidHistoryPositionType(
        column.data_type().to_string(),
    ))
}

#[derive(Clone, Debug)]
pub struct CategoricalActionInterpreter {
    action_values: Vec<f64>,
    max_step: Option<i64>,
}

impl CategoricalActionInterpreter {
    /// Reproduce Python's generated `[0/n, ..., n/n]` action grid.
    ///
    /// # Errors
    /// Returns when `count` is zero or `max_step` is not positive.
    #[allow(clippy::cast_precision_loss)]
    pub fn from_count(count: usize, max_step: Option<i64>) -> Result<Self, SaoeInterpreterError> {
        if count == 0 {
            return Err(SaoeInterpreterError::EmptyCategoricalCount);
        }
        let denominator = count as f64;
        Self::from_values(
            (0..=count)
                .map(|value| value as f64 / denominator)
                .collect(),
            max_step,
        )
    }

    /// # Errors
    /// Returns for an empty action list or non-positive `max_step`.
    pub fn from_values(
        action_values: Vec<f64>,
        max_step: Option<i64>,
    ) -> Result<Self, SaoeInterpreterError> {
        if action_values.is_empty() {
            return Err(SaoeInterpreterError::EmptyCategoricalValues);
        }
        if let Some(step) = max_step
            && step <= 0
        {
            return Err(SaoeInterpreterError::InvalidMaxStep(step));
        }
        Ok(Self {
            action_values,
            max_step,
        })
    }
}

impl SaoeActionInterpreter for CategoricalActionInterpreter {
    fn action_space(&self) -> SaoeActionSpace {
        SaoeActionSpace::Discrete {
            size: self.action_values.len(),
        }
    }

    fn interpret(
        &self,
        state: &SaoeState,
        action: SaoePolicyAction,
    ) -> Result<f64, SaoeInterpreterError> {
        let SaoePolicyAction::Discrete(action) = action else {
            return Err(SaoeInterpreterError::ExpectedDiscrete);
        };
        let index =
            usize::try_from(action).map_err(|_| SaoeInterpreterError::DiscreteOutOfRange {
                action,
                size: self.action_values.len(),
            })?;
        let value =
            *self
                .action_values
                .get(index)
                .ok_or(SaoeInterpreterError::DiscreteOutOfRange {
                    action,
                    size: self.action_values.len(),
                })?;
        let parts = state.parts();
        if self
            .max_step
            .is_some_and(|max_step| parts.cur_step >= max_step - 1)
        {
            return Ok(parts.position);
        }
        Ok(parts.position.min(parts.order.amount() * value))
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TwapRelativeActionInterpreter;

impl SaoeActionInterpreter for TwapRelativeActionInterpreter {
    fn action_space(&self) -> SaoeActionSpace {
        SaoeActionSpace::NonNegativeContinuous
    }

    #[allow(clippy::cast_precision_loss)]
    fn interpret(
        &self,
        state: &SaoeState,
        action: SaoePolicyAction,
    ) -> Result<f64, SaoeInterpreterError> {
        let SaoePolicyAction::Continuous(action) = action else {
            return Err(SaoeInterpreterError::ExpectedContinuous);
        };
        if action < 0.0 {
            return Err(SaoeInterpreterError::NegativeContinuous);
        }
        let parts = state.parts();
        if parts.ticks_per_step == 0 {
            return Err(SaoeInterpreterError::ZeroTicksPerStep);
        }
        let tick_count = parts.ticks_for_order.len();
        let estimated_steps = tick_count.div_ceil(parts.ticks_per_step);
        let remaining = estimated_steps as f64 - parts.cur_step as f64;
        if remaining == 0.0 {
            return Err(SaoeInterpreterError::NoRemainingSteps);
        }
        let twap_volume = parts.position / remaining;
        Ok(parts.position.min(twap_volume * action))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AllOnePolicy {
    fill_value: SaoePolicyAction,
}

impl AllOnePolicy {
    #[must_use]
    pub const fn new(fill_value: SaoePolicyAction) -> Self {
        Self { fill_value }
    }
}

impl Default for AllOnePolicy {
    fn default() -> Self {
        Self::new(SaoePolicyAction::Continuous(1.0))
    }
}

impl SaoePolicy for AllOnePolicy {
    fn actions(
        &mut self,
        observations: &[SaoeObservation],
    ) -> Result<Vec<SaoePolicyAction>, SaoeInterpreterError> {
        Ok(vec![self.fill_value; observations.len()])
    }
}
