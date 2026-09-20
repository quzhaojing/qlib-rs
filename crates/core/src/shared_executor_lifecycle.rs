//! Atomic lifecycle carrying original decisions and a shared mutable execution-result list.

use std::sync::Arc;

use chrono::NaiveDateTime;
use thiserror::Error;

use crate::decision_construction::SharedOrderDecisionConstruction;
use crate::shared_simulator::SharedSimulatorError;
use crate::{
    AtomicExecutorAccount, AtomicExecutorAccountError, ExecutorDecisionTrackerError,
    ExecutorLifecycleCalendar, ExecutorLifecycleCalendarError, ExecutorReturnSinkError,
    IndicatorConfig, RangeLimitDefault, SimulatorCollector, TradeDecision, TradeDecisionError,
};

/// One list identity passed to account reporting, return sinks and the caller.
pub type SharedAtomicResult = crate::shared_simulator::SharedExecutionResult;

/// Account boundary for an original, retainable, dynamically dispatched decision.
pub trait LiveAtomicExecutorAccount: AtomicExecutorAccount {
    /// Report with the same decision and result handles received by execution.
    ///
    /// # Errors
    /// Returns the account failure without undoing reached mutations.
    fn update_live_bar_end(
        &mut self,
        bar: SharedAtomicBarEnd,
        decision: &crate::decision_update::LiveDecisionHandle,
    ) -> Result<(), AtomicExecutorAccountError>;
}

pub trait LiveExecutorDecisionTracker {
    /// Observe the original decision before any validation or settlement.
    ///
    /// # Errors
    /// Returns the observer failure without rolling back its mutations.
    fn track_live(
        &self,
        decision: &crate::decision_update::LiveDecisionHandle,
    ) -> Result<(), ExecutorDecisionTrackerError>;
}

pub struct SharedAtomicBarEnd {
    pub trade_start_time: NaiveDateTime,
    pub trade_end_time: NaiveDateTime,
    pub trade_info: SharedAtomicResult,
    pub indicator_config: IndicatorConfig,
}

/// Shared-account reporting sees the actual decision and result list, without held order locks.
pub trait SharedAtomicExecutorAccount<S, D>: AtomicExecutorAccount {
    /// Apply bar-end updates with the same decision object passed to collection.
    ///
    /// # Errors
    /// Returns an account/reporting failure; reached mutations are retained.
    fn update_shared_bar_end(
        &mut self,
        bar: SharedAtomicBarEnd,
        decision: &mut SharedOrderDecisionConstruction<S, D>,
    ) -> Result<(), AtomicExecutorAccountError>;
}

pub trait SharedExecutorDecisionTracker<S, D> {
    /// Observe or mutate the original decision before range validation.
    ///
    /// # Errors
    /// Returns the first observer failure.
    fn track(
        &self,
        decision: &mut SharedOrderDecisionConstruction<S, D>,
    ) -> Result<(), ExecutorDecisionTrackerError>;
}

pub trait SharedExecutorReturnSink {
    /// Retain or modify the actual result list after calendar advancement and commit.
    ///
    /// # Errors
    /// Returns the sink failure without rolling back completed execution.
    fn store(&mut self, result: SharedAtomicResult) -> Result<(), ExecutorReturnSinkError>;
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SharedAtomicLifecycleError {
    #[error(transparent)]
    LiveDecision(#[from] crate::decision_update::LiveDecisionAccessError),
    #[error("decision base has not been initialized")]
    MissingBase,
    #[error("decision order list has not been initialized")]
    MissingOrders,
    #[error("atomic executor does not support range [{0}, {1}]")]
    UnsupportedRange(i64, i64),
    #[error(transparent)]
    Range(#[from] TradeDecisionError),
    #[error(transparent)]
    Tracker(#[from] ExecutorDecisionTrackerError),
    #[error(transparent)]
    Account(#[from] AtomicExecutorAccountError),
    #[error(transparent)]
    Collection(#[from] SharedSimulatorError),
    #[error(transparent)]
    Calendar(#[from] ExecutorLifecycleCalendarError),
    #[error(transparent)]
    ReturnSink(#[from] ExecutorReturnSinkError),
}

/// Synchronous atomic lifecycle; asynchronous decision yielding remains at the caller's edge.
pub struct SharedAtomicExecutorLifecycle {
    pub calendar: Arc<dyn ExecutorLifecycleCalendar>,
    pub track_data: bool,
    pub settle_type: String,
    pub indicator_config: IndicatorConfig,
}

/// Generator event retaining the actual decision or execution result, never a tracking snapshot.
pub enum LiveAtomicEvent {
    TrackedDecision(crate::decision_update::LiveDecisionHandle),
    Complete(SharedAtomicResult),
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum LiveAtomicSessionError {
    #[error("atomic session has already completed")]
    Complete,
    #[error("atomic session has been closed")]
    Closed,
    #[error("atomic session stopped after an earlier failure")]
    Failed,
    #[error(transparent)]
    Execution(#[from] SharedAtomicLifecycleError),
}

#[derive(Clone, Copy)]
enum LiveAtomicPhase {
    Start,
    AfterTrack,
    Complete,
    Closed,
    Failed,
}

/// Borrowed atomic generator session; plugins remain exclusively borrowed while suspended.
/// Closing or dropping before execution performs no settlement/reporting/calendar work.
pub struct LiveAtomicSession<'a> {
    lifecycle: &'a SharedAtomicExecutorLifecycle,
    collector: &'a mut SimulatorCollector,
    account: &'a mut dyn LiveAtomicExecutorAccount,
    decision: Option<crate::decision_update::LiveDecisionHandle>,
    sink: Option<&'a mut dyn SharedExecutorReturnSink>,
    level: usize,
    phase: LiveAtomicPhase,
}

impl<'a> LiveAtomicSession<'a> {
    #[must_use]
    pub fn new(
        lifecycle: &'a SharedAtomicExecutorLifecycle,
        collector: &'a mut SimulatorCollector,
        account: &'a mut dyn LiveAtomicExecutorAccount,
        decision: crate::decision_update::LiveDecisionHandle,
        sink: Option<&'a mut dyn SharedExecutorReturnSink>,
        level: usize,
    ) -> Self {
        Self {
            lifecycle,
            collector,
            account,
            decision: Some(decision),
            sink,
            level,
            phase: LiveAtomicPhase::Start,
        }
    }

    /// Resume a bare decision yield; its sent value has no meaning in the source generator.
    ///
    /// # Errors
    /// Returns lifecycle failure or a terminal protocol state; execution is never retried.
    ///
    /// # Panics
    /// Panics if the private active-phase invariant is broken and its decision is absent.
    pub fn resume(&mut self) -> Result<LiveAtomicEvent, LiveAtomicSessionError> {
        match self.phase {
            LiveAtomicPhase::Complete => return Err(LiveAtomicSessionError::Complete),
            LiveAtomicPhase::Closed => return Err(LiveAtomicSessionError::Closed),
            LiveAtomicPhase::Failed => return Err(LiveAtomicSessionError::Failed),
            LiveAtomicPhase::Start if self.lifecycle.track_data => {
                self.phase = LiveAtomicPhase::AfterTrack;
                return Ok(LiveAtomicEvent::TrackedDecision(Arc::clone(
                    self.decision.as_ref().expect("active decision"),
                )));
            }
            LiveAtomicPhase::Start | LiveAtomicPhase::AfterTrack => {}
        }
        let decision = self.decision.take().expect("active decision");
        self.phase = LiveAtomicPhase::Failed;
        let result = self.lifecycle.collect_live_data(
            self.collector,
            &decision,
            self.account,
            None,
            self.sink.take(),
            self.level,
        )?;
        self.phase = LiveAtomicPhase::Complete;
        Ok(LiveAtomicEvent::Complete(result))
    }

    /// Cancel without running execution or finalization; repeated cancellation is inert.
    pub fn close(&mut self) {
        if matches!(
            self.phase,
            LiveAtomicPhase::Start | LiveAtomicPhase::AfterTrack
        ) {
            self.phase = LiveAtomicPhase::Closed;
        }
        self.decision = None;
        self.sink = None;
    }
}

impl SharedAtomicExecutorLifecycle {
    /// Execute a retainable decision without converting it to a borrowed or owned-order DTO.
    /// No decision guard spans tracker, range, dealer, account, calendar or sink callbacks.
    /// The simulator's documented individual-order guard restriction still applies.
    ///
    /// # Errors
    /// Returns the first reached lifecycle failure, retaining previous side effects.
    pub fn collect_live_data(
        &self,
        collector: &mut SimulatorCollector,
        decision: &crate::decision_update::LiveDecisionHandle,
        account: &mut dyn LiveAtomicExecutorAccount,
        tracker: Option<&dyn LiveExecutorDecisionTracker>,
        sink: Option<&mut dyn SharedExecutorReturnSink>,
        level: usize,
    ) -> Result<SharedAtomicResult, SharedAtomicLifecycleError> {
        if self.track_data
            && let Some(tracker) = tracker
        {
            tracker.track_live(decision)?;
        }
        if let Some((start, end)) = decision.range_limit(None, RangeLimitDefault::Value(None))? {
            return Err(SharedAtomicLifecycleError::UnsupportedRange(start, end));
        }
        if self.settle_type != "None" {
            account.settle_start(&self.settle_type)?;
        }
        let target = account.execution_target()?;
        let orders = decision.orders()?;
        let collection = collector
            .collect_shared_data(&orders, target, level)?
            .into_shared();
        let (trade_start_time, trade_end_time) = self.calendar.step_time()?;
        account.update_live_bar_end(
            SharedAtomicBarEnd {
                trade_start_time,
                trade_end_time,
                trade_info: Arc::clone(&collection),
                indicator_config: self.indicator_config,
            },
            decision,
        )?;
        self.calendar.step()?;
        if self.settle_type != "None" {
            account.settle_commit()?;
        }
        if let Some(sink) = sink {
            sink.store(Arc::clone(&collection))?;
        }
        Ok(collection)
    }

    /// Preserve source side-effect order while transporting shared orders/results unchanged.
    ///
    /// No collection/order mutex is held while calling tracker, account or return-sink plugins.
    /// The collector's legacy dealer locking restriction remains as documented on that API.
    ///
    /// # Errors
    /// Returns the first reached validation, tracker, settlement, execution, calendar, reporting
    /// or sink error. Earlier state changes and mutations through retained handles are preserved.
    pub fn collect_data<S, D>(
        &self,
        collector: &mut SimulatorCollector,
        decision: &mut SharedOrderDecisionConstruction<S, D>,
        account: &mut dyn SharedAtomicExecutorAccount<S, D>,
        tracker: Option<&dyn SharedExecutorDecisionTracker<S, D>>,
        sink: Option<&mut dyn SharedExecutorReturnSink>,
        level: usize,
    ) -> Result<SharedAtomicResult, SharedAtomicLifecycleError> {
        if self.track_data
            && let Some(tracker) = tracker
        {
            tracker.track(decision)?;
        }
        let base = decision
            .base
            .as_ref()
            .ok_or(SharedAtomicLifecycleError::MissingBase)?;
        let mut range_context = TradeDecision::<()>::from_items(
            Vec::new(),
            base.start_time,
            base.end_time,
            base.trade_range.clone(),
        );
        if let crate::decision_construction::DecisionTotalStep::Value(total_step) =
            decision.total_step
        {
            range_context.set_total_step(total_step);
        }
        if let Some((start, end)) =
            range_context.range_limit(None, RangeLimitDefault::Value(None))?
        {
            return Err(SharedAtomicLifecycleError::UnsupportedRange(start, end));
        }
        if self.settle_type != "None" {
            account.settle_start(&self.settle_type)?;
        }
        let target = account.execution_target()?;
        let orders = decision
            .get_decision()
            .ok_or(SharedAtomicLifecycleError::MissingOrders)?;
        let collection = collector
            .collect_shared_data(orders, target, level)?
            .into_shared();
        let (trade_start_time, trade_end_time) = self.calendar.step_time()?;
        account.update_shared_bar_end(
            SharedAtomicBarEnd {
                trade_start_time,
                trade_end_time,
                trade_info: Arc::clone(&collection),
                indicator_config: self.indicator_config,
            },
            decision,
        )?;
        self.calendar.step()?;
        if self.settle_type != "None" {
            account.settle_commit()?;
        }
        if let Some(sink) = sink {
            sink.store(Arc::clone(&collection))?;
        }
        Ok(collection)
    }
}
