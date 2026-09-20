//! Synchronous `NestedExecutor._collect_data` orchestration.

use std::sync::{Arc, Mutex};

use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{BasePriceStep, NumpyOrderIndicator, Order, OrderDecision, OrderExecution};

/// Owned bottom-level execution tuple retained across nested executor steps.
#[derive(Clone, Debug, PartialEq)]
pub struct OwnedOrderExecution {
    pub order: Order,
    pub trade_value: f64,
    pub trade_cost: f64,
    pub trade_price: f64,
}

impl OwnedOrderExecution {
    /// Move an explicitly detached legacy snapshot into the live transport representation.
    /// This does not restore identity with the original borrowed order.
    #[must_use]
    pub fn into_shared(self) -> SharedOrderExecution {
        Arc::new(crate::shared_simulator::SharedSimulatorExecution {
            order: Arc::new(std::sync::RwLock::new(self.order)),
            trade_value: self.trade_value,
            trade_cost: self.trade_cost,
            trade_price: self.trade_price,
        })
    }

    /// Copy one borrowed simulator execution into the nested owned boundary.
    #[must_use]
    pub fn from_execution(execution: OrderExecution<'_>) -> Self {
        Self {
            order: execution.order.clone(),
            trade_value: execution.trade_value,
            trade_cost: execution.trade_cost,
            trade_price: execution.trade_price,
        }
    }
}

/// The same live execution tuple representation used by the shared simulator.
pub type SharedOrderExecution = crate::shared_simulator::SharedSimulatorExecutionHandle;

/// Retainable child-result list containing live order handles, not copied transport DTOs.
pub type SharedNestedResult = crate::shared_simulator::SharedExecutionResult;

pub(crate) fn snapshot_nested_result(
    result: &SharedNestedResult,
) -> Result<Vec<SharedOrderExecution>, NestedStrategyError> {
    result
        .lock()
        .map(|rows| rows.clone())
        .map_err(|_| NestedStrategyError {
            message: "nested execution result list lock poisoned".to_owned(),
        })
}

/// Diagnostic returned by a nested calendar operation.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("nested calendar error: {message}")]
pub struct NestedCalendarError {
    pub message: String,
}

/// Calendar subset observed by outer decisions and driven by the nested loop.
pub trait NestedCalendar: Send {
    /// Report whether all inner steps have completed.
    ///
    /// # Errors
    /// Returns a calendar state or transport failure.
    fn finished(&self) -> Result<bool, NestedCalendarError>;
    /// Return the total number of inner steps.
    ///
    /// # Errors
    /// Returns a calendar state or transport failure.
    fn trade_len(&self) -> Result<i64, NestedCalendarError>;
    /// Return the current zero-based inner step.
    ///
    /// # Errors
    /// Returns a calendar state or transport failure.
    fn trade_step(&self) -> Result<i64, NestedCalendarError>;
    /// Return the current closed interval.
    ///
    /// # Errors
    /// Returns a calendar state or transport failure.
    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), NestedCalendarError>;
    /// Advance one inner step without execution.
    ///
    /// # Errors
    /// Returns a calendar exhaustion or transport failure.
    fn step(&self) -> Result<(), NestedCalendarError>;
}

/// Whether the outer decision update hook installed a replacement decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NestedDecisionUpdate {
    Unchanged,
    Replaced,
}

/// Diagnostic returned by the outer-decision plugin.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("nested outer decision error: {message}")]
pub struct NestedOuterDecisionError {
    pub message: String,
}

/// Mutable outer-decision handle used throughout one nested collection call.
///
/// Implementations that replace the Python decision object keep the replacement behind this
/// stable handle and return [`NestedDecisionUpdate::Replaced`].
pub trait NestedOuterDecision: Send {
    /// Borrow the current concrete decision for tracking and account aggregation.
    fn order_decision(&self) -> &dyn OrderDecision;
    /// Mutably borrow the current concrete decision for the optional tracking yield boundary.
    fn order_decision_mut(&mut self) -> &mut dyn OrderDecision;
    /// Update or replace the outer decision against the inner calendar.
    ///
    /// # Errors
    /// Returns a decision update failure.
    fn update(
        &mut self,
        calendar: &dyn NestedCalendar,
    ) -> Result<NestedDecisionUpdate, NestedOuterDecisionError>;
    /// Apply Qlib's decision emptiness rule.
    ///
    /// # Errors
    /// Returns a decision access failure.
    fn is_empty(&self) -> Result<bool, NestedOuterDecisionError>;
    /// Return `None` for Python's `NotImplementedError` fallback to the full calendar.
    ///
    /// # Errors
    /// Returns a range calculation failure.
    fn range_limit(
        &self,
        calendar: &dyn NestedCalendar,
    ) -> Result<Option<(i64, i64)>, NestedOuterDecisionError>;
    /// Propagate outer metadata into one generated inner decision.
    ///
    /// # Errors
    /// Returns a decision mutation failure.
    fn modify_inner_decision(
        &self,
        decision: &mut dyn OrderDecision,
    ) -> Result<(), NestedOuterDecisionError>;
}

/// Diagnostic returned by a nested strategy hook.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("nested strategy error: {message}")]
pub struct NestedStrategyError {
    pub message: String,
}

/// Owned strategy-state notification yielded across the RL control boundary.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NestedStrategyPrompt {
    pub kind: String,
    pub schema_version: u32,
    pub payload: Vec<u8>,
}

/// Input used to advance a suspended nested executor or child session.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum NestedExecutorResume {
    /// Equivalent to Python `next(generator)` or `send(None)`.
    Continue,
    /// Resume a yielded RL strategy with its requested execution volume.
    Action(Option<f64>),
}

/// Owned pre-execution view corresponding to one decision yielded by `track_data`.
#[derive(Clone, Debug, PartialEq)]
pub struct TrackedOrderDecision {
    pub orders: Vec<Order>,
    pub start_time: NaiveDateTime,
    pub end_time: NaiveDateTime,
    pub has_trade_range: bool,
}

impl TrackedOrderDecision {
    pub(crate) fn capture(decision: &dyn OrderDecision) -> Self {
        Self {
            orders: decision.orders().to_vec(),
            start_time: decision.start_time(),
            end_time: decision.end_time(),
            has_trade_range: decision.trade_range().is_some(),
        }
    }
}

/// Non-terminal control event propagated by a recursively nested child executor.
pub enum NestedControlEvent {
    TrackedDecision(TrackedOrderDecision),
    StrategyPrompt(NestedStrategyPrompt),
}

/// Result of beginning or resuming one owned child collection.
pub enum NestedInnerProgress {
    Suspended(NestedControlEvent),
    Complete {
        decision: Box<dyn OrderDecision>,
        executions: SharedNestedResult,
    },
}

/// Completion retaining the actual decision and shared result list for native child transport.
pub struct LiveNestedInnerCompletion {
    pub decision: crate::decision_update::LiveDecisionHandle,
    pub executions: SharedNestedResult,
}

/// Native control transport retains the original decision across a generator yield.
pub enum LiveNestedControlEvent {
    TrackedDecision(crate::decision_update::LiveDecisionHandle),
    StrategyPrompt(NestedStrategyPrompt),
}

/// Native recursive child progress, without converting decisions into owned snapshots.
pub enum LiveNestedInnerProgress {
    Suspended(LiveNestedControlEvent),
    Complete(LiveNestedInnerCompletion),
}

/// Selects whether the parent framework or the child session emits collection control events.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NestedInnerControlMode {
    #[default]
    Framework,
    Delegated,
}

/// Result of asking a strategy for its next decision.
pub enum NestedStrategyProgress {
    Ready(Box<dyn OrderDecision>),
    Suspended(NestedStrategyPrompt),
}

/// Immediate, synchronous strategy boundary for one nested loop.
pub trait NestedStrategy: Send {
    /// Generate using the actual retainable previous list, without a framework-held guard.
    /// Legacy slice plugins receive an independent list of the same execution handles.
    /// Shared-list plugins must override this method to retain or change list membership.
    ///
    /// # Errors
    /// Returns a poisoned legacy-adapter list or the original strategy failure.
    fn generate_shared_trade_decision(
        &mut self,
        previous: Option<&SharedNestedResult>,
    ) -> Result<Box<dyn OrderDecision>, NestedStrategyError> {
        let snapshot = previous.map(snapshot_nested_result).transpose()?;
        self.generate_trade_decision(snapshot.as_deref())
    }

    /// Start generation with a retainable list, including across a suspended strategy.
    /// Legacy plugins use their existing begin hook with a handle snapshot.
    ///
    /// # Errors
    /// Returns a poisoned legacy-adapter list or the original begin failure.
    fn begin_shared_trade_decision(
        &mut self,
        previous: Option<&SharedNestedResult>,
    ) -> Result<NestedStrategyProgress, NestedStrategyError> {
        let snapshot = previous.map(snapshot_nested_result).transpose()?;
        self.begin_trade_decision(snapshot.as_deref())
    }

    /// Modify or retain the actual child list before it is extended into the outer result.
    /// Legacy plugins receive a handle snapshot, never a guard across their callback.
    ///
    /// # Errors
    /// Returns a poisoned legacy-adapter list or the original post-hook failure.
    fn post_shared_execute(
        &mut self,
        executions: &SharedNestedResult,
    ) -> Result<(), NestedStrategyError> {
        self.post_execute(&snapshot_nested_result(executions)?)
    }

    /// Reset the strategy for a newly initialized sub-level.
    ///
    /// # Errors
    /// Returns a strategy initialization failure.
    fn reset(&mut self, outer: &dyn NestedOuterDecision) -> Result<(), NestedStrategyError>;
    /// React to replacement of the outer decision.
    ///
    /// # Errors
    /// Returns a strategy alteration failure.
    fn alter_outer_decision(
        &mut self,
        outer: &mut dyn NestedOuterDecision,
    ) -> Result<(), NestedStrategyError>;
    /// Generate the next immediate inner decision.
    ///
    /// # Errors
    /// Returns a synchronous strategy generation failure.
    fn generate_trade_decision(
        &mut self,
        previous: Option<&[SharedOrderExecution]>,
    ) -> Result<Box<dyn OrderDecision>, NestedStrategyError>;

    /// Start decision generation, optionally yielding an owned RL prompt.
    ///
    /// Immediate strategies use the default adapter over [`Self::generate_trade_decision`].
    ///
    /// # Errors
    /// Returns a strategy generation or prompt-encoding failure.
    fn begin_trade_decision(
        &mut self,
        previous: Option<&[SharedOrderExecution]>,
    ) -> Result<NestedStrategyProgress, NestedStrategyError> {
        self.generate_trade_decision(previous)
            .map(NestedStrategyProgress::Ready)
    }

    /// Resume a previously suspended strategy with the optional execution volume.
    ///
    /// # Errors
    /// Returns a strategy, action-decoding, or invalid-state failure.
    fn resume_trade_decision(
        &mut self,
        _execution_volume: Option<f64>,
    ) -> Result<NestedStrategyProgress, NestedStrategyError> {
        Err(NestedStrategyError {
            message: "strategy is not suspended".to_owned(),
        })
    }
    /// Close active decision generation without running normal completion hooks.
    /// Suspended implementations must release their active generator even on failure.
    /// The default is for immediate strategies without suspended resources.
    ///
    /// # Errors
    /// Returns a generator cleanup failure.
    fn close_trade_decision(&mut self) -> Result<(), NestedStrategyError> {
        Ok(())
    }

    /// Observe the just-completed inner execution list.
    ///
    /// # Errors
    /// Returns a post-step hook failure.
    fn post_execute(
        &mut self,
        executions: &[SharedOrderExecution],
    ) -> Result<(), NestedStrategyError>;
    /// Finalize the strategy after the nested loop exits normally.
    ///
    /// # Errors
    /// Returns a finalization failure.
    fn post_upper_level(&mut self) -> Result<(), NestedStrategyError>;
}

/// Diagnostic returned by inner-executor initialization, collection, or account snapshotting.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("nested inner executor error: {message}")]
pub struct NestedInnerExecutorError {
    pub message: String,
}

/// Inner executor driven by the framework-owned nested loop.
pub trait NestedInnerExecutor: NestedCalendar + Send {
    /// Select who emits native tracking events independently of legacy transport.
    fn live_control_mode(&self) -> NestedInnerControlMode {
        NestedInnerControlMode::Framework
    }

    /// Begin native collection, retaining original identities across optional suspension.
    ///
    /// # Errors
    /// Returns an unsupported capability or collection failure.
    fn begin_live_collect_data(
        &mut self,
        decision: crate::decision_update::LiveDecisionHandle,
        level: usize,
    ) -> Result<LiveNestedInnerProgress, NestedInnerExecutorError> {
        self.collect_live_decision(decision, level)
            .map(LiveNestedInnerProgress::Complete)
    }

    /// Resume a native child generator; synchronous plugins have no suspended session.
    ///
    /// # Errors
    /// Returns an invalid-state or child execution failure.
    fn resume_live_collect_data(
        &mut self,
        _input: NestedExecutorResume,
    ) -> Result<LiveNestedInnerProgress, NestedInnerExecutorError> {
        Err(NestedInnerExecutorError {
            message: "live inner executor is not suspended".to_owned(),
        })
    }

    /// Release a native generator without running normal completion hooks.
    ///
    /// # Errors
    /// Returns a child cleanup failure. Implementations must release even on failure.
    fn close_live_collect_data(&mut self) -> Result<(), NestedInnerExecutorError> {
        Ok(())
    }

    /// Execute the original shared decision with no owned-order compatibility conversion.
    /// Legacy plugins must explicitly implement this capability before accepting live input.
    ///
    /// # Errors
    /// Returns an unsupported-capability or native collection failure.
    fn collect_live_data(
        &mut self,
        _decision: &crate::decision_update::LiveDecisionHandle,
        _level: usize,
    ) -> Result<SharedNestedResult, NestedInnerExecutorError> {
        Err(NestedInnerExecutorError {
            message: "inner executor does not support live decisions".to_owned(),
        })
    }

    /// Complete a synchronous native collection, retaining both original identities.
    ///
    /// # Errors
    /// Returns the native collection failure without rolling back reached mutations.
    fn collect_live_decision(
        &mut self,
        decision: crate::decision_update::LiveDecisionHandle,
        level: usize,
    ) -> Result<LiveNestedInnerCompletion, NestedInnerExecutorError> {
        let executions = self.collect_live_data(&decision, level)?;
        Ok(LiveNestedInnerCompletion {
            decision,
            executions,
        })
    }

    /// Execute and return a retainable mutable child list.
    /// Legacy owned-result plugins are adapted by moving their list into a shared handle.
    ///
    /// # Errors
    /// Returns the inner collection failure.
    fn collect_shared_data(
        &mut self,
        decision: &mut dyn OrderDecision,
        level: usize,
    ) -> Result<SharedNestedResult, NestedInnerExecutorError> {
        self.collect_data(decision, level)
            .map(|rows| Arc::new(Mutex::new(rows)))
    }

    /// Reset the inner executor to the outer bar's closed interval.
    ///
    /// # Errors
    /// Returns an inner initialization failure.
    fn reset_window(
        &mut self,
        start_time: NaiveDateTime,
        end_time: NaiveDateTime,
    ) -> Result<(), NestedInnerExecutorError>;
    /// Execute one generated decision and advance the inner executor.
    ///
    /// # Errors
    /// Returns an inner collection failure.
    fn collect_data(
        &mut self,
        decision: &mut dyn OrderDecision,
        level: usize,
    ) -> Result<Vec<SharedOrderExecution>, NestedInnerExecutorError>;

    /// Report whether this executor delegates tracked decisions and prompts from a child session.
    fn control_mode(&self) -> NestedInnerControlMode {
        NestedInnerControlMode::Framework
    }

    /// Begin collection while transferring ownership of the decision across possible suspension.
    ///
    /// Synchronous executors use this default adapter. Recursive executors override it and may
    /// return [`NestedInnerProgress::Suspended`].
    ///
    /// # Errors
    /// Returns an inner collection or child-session failure.
    fn begin_collect_data(
        &mut self,
        mut decision: Box<dyn OrderDecision>,
        level: usize,
    ) -> Result<NestedInnerProgress, NestedInnerExecutorError> {
        let executions = self.collect_shared_data(&mut *decision, level)?;
        Ok(NestedInnerProgress::Complete {
            decision,
            executions,
        })
    }

    /// Resume collection delegated to a recursively nested child session.
    ///
    /// # Errors
    /// Returns an invalid-state or child-session failure.
    fn resume_collect_data(
        &mut self,
        _input: NestedExecutorResume,
    ) -> Result<NestedInnerProgress, NestedInnerExecutorError> {
        Err(NestedInnerExecutorError {
            message: "inner executor is not suspended".to_owned(),
        })
    }
    /// Close a suspended collection without committing or advancing its calendar.
    /// Suspended implementations must release the active session even on failure.
    /// The default is for synchronous executors without suspended resources.
    ///
    /// # Errors
    /// Returns a child cleanup failure.
    fn close_collect_data(&mut self) -> Result<(), NestedInnerExecutorError> {
        Ok(())
    }

    /// Retain the current raw order store without cloning its contents.
    /// Returned handles must not retain account or engine guards. Repeated calls
    /// return the same identity until the account replaces its current store.
    ///
    /// # Errors
    /// Returns an account or plugin handle-acquisition failure.
    fn order_indicator_handle(
        &self,
    ) -> Result<crate::SharedOrderIndicator<NumpyOrderIndicator>, NestedInnerExecutorError>;

    /// Copy numerical values for independent inspection, not raw-reference transport.
    ///
    /// # Errors
    /// Returns an account or snapshot conversion failure.
    fn order_indicator_snapshot(&self) -> Result<NumpyOrderIndicator, NestedInnerExecutorError>;
}

/// Diagnostic returned while attaching the inner level to its parent infrastructure.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("nested level binding error: {message}")]
pub struct NestedLevelBindingError {
    pub message: String,
}

/// Replaceable infrastructure hook corresponding to `set_sub_level_infra`.
pub trait NestedLevelBinding: Send {
    /// Attach the initialized inner level to its parent infrastructure.
    ///
    /// # Errors
    /// Returns a binding or transport failure.
    fn bind_inner(
        &mut self,
        inner: &dyn NestedInnerExecutor,
    ) -> Result<(), NestedLevelBindingError>;
}

/// One generated inner decision paired with the calendar interval captured before collection.
pub struct NestedDecisionRecord {
    decision: Box<dyn OrderDecision>,
    start_time: NaiveDateTime,
    end_time: NaiveDateTime,
}

impl NestedDecisionRecord {
    #[must_use]
    pub(crate) fn new(
        decision: Box<dyn OrderDecision>,
        start_time: NaiveDateTime,
        end_time: NaiveDateTime,
    ) -> Self {
        Self {
            decision,
            start_time,
            end_time,
        }
    }

    #[must_use]
    pub fn decision(&self) -> &dyn OrderDecision {
        &*self.decision
    }

    #[must_use]
    pub const fn start_time(&self) -> NaiveDateTime {
        self.start_time
    }

    #[must_use]
    pub const fn end_time(&self) -> NaiveDateTime {
        self.end_time
    }

    /// Borrow the record in the form consumed by outer indicator aggregation.
    #[must_use]
    pub fn base_price_step(&self) -> BasePriceStep<'_> {
        BasePriceStep {
            start_time: self.start_time,
            end_time: self.end_time,
            trade_range: self.decision.trade_range(),
        }
    }
}

/// Complete synchronous result of one nested executor call.
pub struct NestedCollection {
    executions: SharedNestedResult,
    inner_order_indicators: Vec<crate::SharedOrderIndicator<NumpyOrderIndicator>>,
    decisions: Vec<NestedDecisionRecord>,
}

impl NestedCollection {
    #[must_use]
    pub(crate) fn from_parts(
        executions: Vec<SharedOrderExecution>,
        inner_order_indicators: Vec<crate::SharedOrderIndicator<NumpyOrderIndicator>>,
        decisions: Vec<NestedDecisionRecord>,
    ) -> Self {
        Self {
            executions: Arc::new(Mutex::new(executions)),
            inner_order_indicators,
            decisions,
        }
    }

    #[must_use]
    pub fn executions(&self) -> &SharedNestedResult {
        &self.executions
    }

    #[must_use]
    pub(crate) fn into_executions(self) -> SharedNestedResult {
        self.executions
    }

    #[must_use]
    pub fn inner_order_indicators(&self) -> &[crate::SharedOrderIndicator<NumpyOrderIndicator>] {
        &self.inner_order_indicators
    }

    /// Replace retained input bindings; ordinary aggregation mutates stores through their handles.
    pub fn inner_order_indicators_mut(
        &mut self,
    ) -> &mut [crate::SharedOrderIndicator<NumpyOrderIndicator>] {
        &mut self.inner_order_indicators
    }

    #[must_use]
    pub fn decisions(&self) -> &[NestedDecisionRecord] {
        &self.decisions
    }

    #[must_use]
    pub fn base_price_steps(&self) -> Vec<BasePriceStep<'_>> {
        self.decisions
            .iter()
            .map(NestedDecisionRecord::base_price_step)
            .collect()
    }

    /// Split the two disjoint inputs consumed together by nested account aggregation.
    #[must_use]
    pub fn aggregation_inputs(
        &self,
    ) -> (
        &[crate::SharedOrderIndicator<NumpyOrderIndicator>],
        Vec<BasePriceStep<'_>>,
    ) {
        let Self {
            inner_order_indicators,
            decisions,
            ..
        } = self;
        let steps = decisions
            .iter()
            .map(NestedDecisionRecord::base_price_step)
            .collect();
        (inner_order_indicators, steps)
    }
}

/// Failure from the first reached nested orchestration stage.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum NestedExecutorError {
    #[error(transparent)]
    Calendar(#[from] NestedCalendarError),
    #[error(transparent)]
    OuterDecision(#[from] NestedOuterDecisionError),
    #[error(transparent)]
    Strategy(#[from] NestedStrategyError),
    #[error(transparent)]
    InnerExecutor(#[from] NestedInnerExecutorError),
    #[error(transparent)]
    LevelBinding(#[from] NestedLevelBindingError),
}

/// Framework-owned synchronous core of `NestedExecutor._collect_data`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NestedExecutorCore {
    skip_empty_decision: bool,
    align_range_limit: bool,
}

impl NestedExecutorCore {
    #[must_use]
    pub const fn new(skip_empty_decision: bool, align_range_limit: bool) -> Self {
        Self {
            skip_empty_decision,
            align_range_limit,
        }
    }

    /// Run one complete synchronous nested loop.
    ///
    /// # Errors
    ///
    /// Returns the first reached calendar, initialization, binding, decision, strategy,
    /// collection, snapshot, or hook failure without rolling back earlier side effects.
    pub fn collect_data(
        &self,
        outer_calendar: &dyn NestedCalendar,
        level_binding: &mut dyn NestedLevelBinding,
        inner: &mut dyn NestedInnerExecutor,
        strategy: &mut dyn NestedStrategy,
        outer: &mut dyn NestedOuterDecision,
        level: usize,
    ) -> Result<NestedCollection, NestedExecutorError> {
        let (trade_start_time, trade_end_time) = outer_calendar.step_time()?;
        inner.reset_window(trade_start_time, trade_end_time)?;
        level_binding.bind_inner(inner)?;
        strategy.reset(outer)?;

        let mut executions = Vec::new();
        let mut inner_order_indicators = Vec::new();
        let mut decisions = Vec::new();
        let mut previous: Option<SharedNestedResult> = None;

        while !inner.finished()? {
            if outer.update(inner)? == NestedDecisionUpdate::Replaced {
                strategy.alter_outer_decision(outer)?;
            }

            if self.skip_empty_decision && outer.is_empty()? {
                break;
            }

            let (start_idx, end_idx) = match outer.range_limit(inner)? {
                Some(range) => range,
                None => (0, inner.trade_len()? - 1),
            };
            let step = inner.trade_step()?;
            if self.align_range_limit && !(start_idx <= step && step <= end_idx) {
                inner.step()?;
                continue;
            }

            let mut inner_decision = strategy.generate_shared_trade_decision(previous.as_ref())?;
            outer.modify_inner_decision(&mut *inner_decision)?;
            let (start_time, end_time) = inner.step_time()?;
            let current = inner.collect_shared_data(&mut *inner_decision, level + 1)?;
            strategy.post_shared_execute(&current)?;
            executions.extend(snapshot_nested_result(&current)?);
            let handle = inner.order_indicator_handle()?;
            inner_order_indicators.push(handle);
            decisions.push(NestedDecisionRecord::new(
                inner_decision,
                start_time,
                end_time,
            ));
            previous = Some(current);
        }

        strategy.post_upper_level()?;
        Ok(NestedCollection::from_parts(
            executions,
            inner_order_indicators,
            decisions,
        ))
    }
}
