//! Atomic `BaseExecutor.collect_data` lifecycle orchestration.

use std::sync::Arc;

use chrono::NaiveDateTime;
use thiserror::Error;

use crate::{
    ExecutionTarget, IndicatorConfig, NumpyOrderIndicator, OrderDecision, OrderExecution,
    SimulatorCollection, SimulatorCollectionError, SimulatorCollector, TradeDecisionError,
    TradeRangeError,
};

const EXECUTOR_NO_SETTLEMENT: &str = "None";

/// Diagnostic returned by the lifecycle calendar adapter.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("executor lifecycle calendar error: {message}")]
pub struct ExecutorLifecycleCalendarError {
    pub message: String,
}

/// Calendar operations performed after the concrete executor has collected its data.
pub trait ExecutorLifecycleCalendar: Send + Sync {
    /// Return the current closed bar interval.
    ///
    /// # Errors
    ///
    /// Returns a calendar state or retrieval failure.
    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), ExecutorLifecycleCalendarError>;

    /// Advance the calendar by one step.
    ///
    /// # Errors
    ///
    /// Returns a calendar advancement failure.
    fn step(&self) -> Result<(), ExecutorLifecycleCalendarError>;
}

/// Diagnostic returned by a replaceable decision-yield observer.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("executor decision tracker error: {message}")]
pub struct ExecutorDecisionTrackerError {
    pub message: String,
}

/// Rust edge corresponding to consuming the optional decision yielded for RL data collection.
pub trait ExecutorDecisionTracker: Send + Sync {
    /// Observe or mutate the decision before atomic-range validation.
    ///
    /// # Errors
    ///
    /// Returns an observer or transport failure.
    fn track(&self, decision: &mut dyn OrderDecision) -> Result<(), ExecutorDecisionTrackerError>;
}

/// Stable decision metadata forwarded to atomic bar-end integrations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AtomicDecisionSnapshot {
    pub start_time: NaiveDateTime,
    pub end_time: NaiveDateTime,
    pub order_count: usize,
    pub has_trade_range: bool,
}

/// Atomic subset of the keyword arguments passed to `Account.update_bar_end`.
#[derive(Clone, Copy, Debug)]
pub struct AtomicBarEnd<'a> {
    pub trade_start_time: NaiveDateTime,
    pub trade_end_time: NaiveDateTime,
    pub atomic: bool,
    pub outer_decision: AtomicDecisionSnapshot,
    pub trade_info: &'a [OrderExecution<'a>],
    pub indicator_config: IndicatorConfig,
}

/// Diagnostic returned by account lifecycle integration.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("atomic executor account error: {message}")]
pub struct AtomicExecutorAccountError {
    pub message: String,
}

/// Account operations surrounding an atomic concrete collector.
pub trait AtomicExecutorAccount: Send {
    /// Expose the target mutated by the exchange deal executor.
    ///
    /// # Errors
    ///
    /// Returns an account/position access failure.
    fn execution_target(&mut self) -> Result<&mut dyn ExecutionTarget, AtomicExecutorAccountError>;

    /// Begin the configured settlement transaction.
    ///
    /// # Errors
    ///
    /// Returns a settlement initialization failure.
    fn settle_start(&mut self, settle_type: &str) -> Result<(), AtomicExecutorAccountError>;

    /// Perform atomic bar-end accounting and indicator updates.
    ///
    /// # Errors
    ///
    /// Returns an account, market, metric, or adapter failure.
    fn update_bar_end(&mut self, bar: AtomicBarEnd<'_>) -> Result<(), AtomicExecutorAccountError>;

    /// Commit the active settlement transaction.
    ///
    /// # Errors
    ///
    /// Returns a settlement commit failure.
    fn settle_commit(&mut self) -> Result<(), AtomicExecutorAccountError>;

    /// Copy an independent numerical order indicator after one completed atomic step.
    ///
    /// # Errors
    ///
    /// Returns an account, plugin, or snapshot conversion failure.
    fn order_indicator_snapshot(&self) -> Result<NumpyOrderIndicator, AtomicExecutorAccountError>;

    /// Retain the original order store after a completed step, without copying its values.
    ///
    /// # Errors
    /// Returns an account or indicator-plugin handle-acquisition failure.
    fn order_indicator_handle(
        &self,
    ) -> Result<crate::SharedOrderIndicator<NumpyOrderIndicator>, AtomicExecutorAccountError>;
}

/// Diagnostic returned while mirroring Python's optional `return_value` mapping update.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("executor return sink error: {message}")]
pub struct ExecutorReturnSinkError {
    pub message: String,
}

/// Replaceable sink for the final execution result.
pub trait ExecutorReturnSink: Send {
    /// Store the execution result after calendar advancement and settlement commit.
    ///
    /// # Errors
    ///
    /// Returns a storage or transport failure.
    fn store_execute_result(
        &mut self,
        executions: &[OrderExecution<'_>],
    ) -> Result<(), ExecutorReturnSinkError>;
}

/// Failures from the atomic `BaseExecutor.collect_data` lifecycle.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum AtomicExecutorLifecycleError {
    #[error(transparent)]
    Tracker(#[from] ExecutorDecisionTrackerError),
    #[error(transparent)]
    Decision(#[from] TradeDecisionError),
    #[error("atomic executor doesn't support specify `range_limit`: [{start_idx}, {end_idx}]")]
    UnsupportedRange { start_idx: i64, end_idx: i64 },
    #[error(transparent)]
    Account(#[from] AtomicExecutorAccountError),
    #[error(transparent)]
    Collection(#[from] SimulatorCollectionError),
    #[error(transparent)]
    Calendar(#[from] ExecutorLifecycleCalendarError),
    #[error(transparent)]
    ReturnSink(#[from] ExecutorReturnSinkError),
}

/// Framework-owned sequencing for the atomic path of `BaseExecutor.collect_data`.
pub struct AtomicExecutorLifecycle {
    calendar: Arc<dyn ExecutorLifecycleCalendar>,
    track_data: bool,
    settle_type: String,
    indicator_config: IndicatorConfig,
}

impl AtomicExecutorLifecycle {
    #[must_use]
    pub fn new(
        calendar: Arc<dyn ExecutorLifecycleCalendar>,
        track_data: bool,
        settle_type: impl Into<String>,
        indicator_config: IndicatorConfig,
    ) -> Self {
        Self {
            calendar,
            track_data,
            settle_type: settle_type.into(),
            indicator_config,
        }
    }

    /// Run one atomic executor step around the completed simulator collector.
    ///
    /// The optional tracker models a consumer resuming after Python's decision yield. It runs
    /// before range validation and can mutate the typed decision. The optional return sink runs
    /// last, after settlement commit.
    ///
    /// # Errors
    ///
    /// Returns the first reached tracker, decision-range, settlement, collection, calendar,
    /// bar-end, commit, or return-sink failure without rolling back earlier side effects.
    pub fn collect_data<'a>(
        &self,
        collector: &mut SimulatorCollector,
        decision: &'a mut dyn OrderDecision,
        account: &mut dyn AtomicExecutorAccount,
        tracker: Option<&dyn ExecutorDecisionTracker>,
        return_sink: Option<&mut dyn ExecutorReturnSink>,
        level: usize,
    ) -> Result<SimulatorCollection<'a>, AtomicExecutorLifecycleError> {
        if self.track_data
            && let Some(tracker) = tracker
        {
            tracker.track(decision)?;
        }

        if let Some((start_idx, end_idx)) = atomic_range_limit(decision)? {
            return Err(AtomicExecutorLifecycleError::UnsupportedRange { start_idx, end_idx });
        }

        let decision_snapshot = AtomicDecisionSnapshot {
            start_time: decision.start_time(),
            end_time: decision.end_time(),
            order_count: decision.orders().len(),
            has_trade_range: decision.trade_range().is_some(),
        };

        if self.settle_type != EXECUTOR_NO_SETTLEMENT {
            account.settle_start(&self.settle_type)?;
        }

        let collection = collector.collect_data(decision, account.execution_target()?, level)?;
        let (trade_start_time, trade_end_time) = self.calendar.step_time()?;
        account.update_bar_end(AtomicBarEnd {
            trade_start_time,
            trade_end_time,
            atomic: true,
            outer_decision: decision_snapshot,
            trade_info: collection.trade_info(),
            indicator_config: self.indicator_config,
        })?;
        self.calendar.step()?;

        if self.settle_type != EXECUTOR_NO_SETTLEMENT {
            account.settle_commit()?;
        }

        if let Some(return_sink) = return_sink {
            return_sink.store_execute_result(collection.execution_result())?;
        }

        Ok(collection)
    }
}

fn atomic_range_limit(
    decision: &dyn OrderDecision,
) -> Result<Option<(i64, i64)>, TradeDecisionError> {
    let Some(trade_range) = decision.trade_range() else {
        return Ok(None);
    };
    match trade_range.range_indices(None) {
        Ok(range) => Ok(Some(range)),
        Err(TradeRangeError::MissingCalendar) => Ok(None),
        Err(error) => Err(error.into()),
    }
}
