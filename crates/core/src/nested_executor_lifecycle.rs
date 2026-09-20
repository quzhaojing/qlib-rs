//! `BaseExecutor.collect_data` lifecycle around a synchronous nested collection.

use std::sync::Arc;

use chrono::NaiveDateTime;
use thiserror::Error;

use crate::{
    BasePriceStep, ExecutorDecisionTracker, ExecutorDecisionTrackerError, IndicatorConfig,
    NestedCalendar, NestedCalendarError, NestedCollection, NestedExecutorCore, NestedExecutorError,
    NestedInnerExecutor, NestedLevelBinding, NestedOuterDecision, NestedStrategy,
    NumpyOrderIndicator, OrderDecision, OrderIndicatorAggregationConfig,
};

use crate::nested_executor::SharedNestedResult;

const NO_SETTLEMENT: &str = "None";

/// Nested subset of the keyword arguments passed to `Account.update_bar_end`.
pub struct NestedBarEnd<'a> {
    pub trade_start_time: NaiveDateTime,
    pub trade_end_time: NaiveDateTime,
    pub outer_decision: &'a dyn OrderDecision,
    pub inner_order_indicators: &'a [crate::SharedOrderIndicator<NumpyOrderIndicator>],
    pub steps: &'a [BasePriceStep<'a>],
    pub indicator_config: IndicatorConfig,
    pub aggregation_config: OrderIndicatorAggregationConfig,
}

/// Native nested bar request retaining decisions until each reporting stage reads them.
pub struct LiveNestedBarEnd<'a> {
    pub trade_start_time: NaiveDateTime,
    pub trade_end_time: NaiveDateTime,
    pub outer_decision: &'a crate::decision_update::LiveDecisionHandle,
    pub inner_order_indicators: &'a [crate::SharedOrderIndicator<NumpyOrderIndicator>],
    pub steps: &'a [crate::base_price::LiveBasePriceStep],
    pub indicator_config: IndicatorConfig,
    pub aggregation_config: OrderIndicatorAggregationConfig,
}

/// Diagnostic returned by nested account lifecycle integration.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("nested executor account error: {message}")]
pub struct NestedExecutorAccountError {
    pub message: String,
}

/// Account operations surrounding a nested concrete collection.
pub trait NestedExecutorAccount: Send {
    /// Report native nested decisions without converting them to detached order snapshots.
    ///
    /// # Errors
    /// Returns an unsupported-capability or reporting failure.
    fn update_live_bar_end(
        &mut self,
        _bar: LiveNestedBarEnd<'_>,
    ) -> Result<(), NestedExecutorAccountError> {
        Err(NestedExecutorAccountError {
            message: "nested account does not support live decisions".to_owned(),
        })
    }

    /// Begin the configured settlement transaction.
    ///
    /// # Errors
    /// Returns a settlement initialization failure.
    fn settle_start(&mut self, settle_type: &str) -> Result<(), NestedExecutorAccountError>;

    /// Perform nested bar-end accounting and outer indicator aggregation.
    ///
    /// # Errors
    /// Returns an account, market, metric, or adapter failure.
    fn update_bar_end(&mut self, bar: NestedBarEnd<'_>) -> Result<(), NestedExecutorAccountError>;

    /// Commit the configured settlement transaction.
    ///
    /// # Errors
    /// Returns a settlement commit failure.
    fn settle_commit(&mut self) -> Result<(), NestedExecutorAccountError>;
}

/// Diagnostic returned while mirroring Python's optional nested `return_value` mapping update.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("nested executor return sink error: {message}")]
pub struct NestedExecutorReturnSinkError {
    pub message: String,
}

/// Replaceable sink for the final flattened nested execution list.
pub trait NestedExecutorReturnSink: Send {
    /// Store the result after bar-end accounting, calendar advancement, and settlement commit.
    /// The sink may retain or modify the exact list subsequently returned to the caller.
    /// The framework holds no result-list guard across this callback.
    ///
    /// # Errors
    /// Returns a storage or transport failure.
    fn store_execute_result(
        &mut self,
        executions: &SharedNestedResult,
    ) -> Result<(), NestedExecutorReturnSinkError>;
}

/// Mutable collaborators and per-call inputs for one nested lifecycle run.
pub struct NestedExecutorRun<'a> {
    pub level_binding: &'a mut dyn NestedLevelBinding,
    pub inner: &'a mut dyn NestedInnerExecutor,
    pub strategy: &'a mut dyn NestedStrategy,
    pub outer: &'a mut dyn NestedOuterDecision,
    pub account: &'a mut dyn NestedExecutorAccount,
    pub tracker: Option<&'a dyn ExecutorDecisionTracker>,
    pub return_sink: Option<&'a mut dyn NestedExecutorReturnSink>,
    pub level: usize,
}

/// Failure from the first reached nested `BaseExecutor.collect_data` stage.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum NestedExecutorLifecycleError {
    #[error(transparent)]
    Tracker(#[from] ExecutorDecisionTrackerError),
    #[error(transparent)]
    Account(#[from] NestedExecutorAccountError),
    #[error(transparent)]
    Collection(#[from] NestedExecutorError),
    #[error(transparent)]
    Calendar(#[from] NestedCalendarError),
    #[error(transparent)]
    ReturnSink(#[from] NestedExecutorReturnSinkError),
}

/// Framework-owned sequencing around [`NestedExecutorCore`].
pub struct NestedExecutorLifecycle {
    core: NestedExecutorCore,
    calendar: Arc<dyn NestedCalendar>,
    track_data: bool,
    settle_type: String,
    indicator_config: IndicatorConfig,
    aggregation_config: OrderIndicatorAggregationConfig,
}

impl NestedExecutorLifecycle {
    #[must_use]
    pub fn new(
        core: NestedExecutorCore,
        calendar: Arc<dyn NestedCalendar>,
        track_data: bool,
        settle_type: impl Into<String>,
        indicator_config: IndicatorConfig,
        aggregation_config: OrderIndicatorAggregationConfig,
    ) -> Self {
        Self {
            core,
            calendar,
            track_data,
            settle_type: settle_type.into(),
            indicator_config,
            aggregation_config,
        }
    }

    /// Run one complete nested executor step, including the surrounding base lifecycle.
    ///
    /// # Errors
    ///
    /// Returns the first reached tracker, settlement, collection, calendar, account, commit, or
    /// return-sink failure without rolling back earlier side effects.
    pub fn collect_data(
        &self,
        run: NestedExecutorRun<'_>,
    ) -> Result<NestedCollection, NestedExecutorLifecycleError> {
        let NestedExecutorRun {
            level_binding,
            inner,
            strategy,
            outer,
            account,
            tracker,
            return_sink,
            level,
        } = run;
        if self.track_data
            && let Some(tracker) = tracker
        {
            tracker.track(outer.order_decision_mut())?;
        }

        if self.settle_type != NO_SETTLEMENT {
            account.settle_start(&self.settle_type)?;
        }

        let collection = self.core.collect_data(
            &*self.calendar,
            level_binding,
            inner,
            strategy,
            outer,
            level,
        )?;
        let (trade_start_time, trade_end_time) = self.calendar.step_time()?;
        {
            let (inner_order_indicators, steps) = collection.aggregation_inputs();
            account.update_bar_end(NestedBarEnd {
                trade_start_time,
                trade_end_time,
                outer_decision: outer.order_decision(),
                inner_order_indicators,
                steps: &steps,
                indicator_config: self.indicator_config,
                aggregation_config: self.aggregation_config,
            })?;
        }
        self.calendar.step()?;

        if self.settle_type != NO_SETTLEMENT {
            account.settle_commit()?;
        }

        if let Some(return_sink) = return_sink {
            return_sink.store_execute_result(collection.executions())?;
        }
        Ok(collection)
    }
}
