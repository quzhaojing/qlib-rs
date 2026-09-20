//! Concrete synchronous nested-inner adapter over the atomic executor lifecycle.

use std::sync::Arc;

use chrono::NaiveDateTime;

use crate::{
    AtomicExecutorAccount, AtomicExecutorLifecycle, ExecutorLifecycleCalendar, IndicatorConfig,
    NestedCalendar, NestedCalendarError, NestedInnerExecutor, NestedInnerExecutorError,
    NumpyOrderIndicator, OrderDecision, OwnedOrderExecution, SharedOrderExecution,
    SimulatorCollector,
};

/// Shared calendar boundary required by the nested loop and atomic lifecycle.
pub trait ResettableNestedCalendar: NestedCalendar + ExecutorLifecycleCalendar + Sync {
    /// Reset the closed inner execution window.
    ///
    /// # Errors
    ///
    /// Returns a calendar lookup, validation, or transport failure.
    fn reset_window(
        &self,
        start_time: NaiveDateTime,
        end_time: NaiveDateTime,
    ) -> Result<(), NestedCalendarError>;
}

/// Concrete immediate-decision adapter used by [`crate::NestedExecutorCore`].
pub struct AtomicNestedInnerAdapter<'a> {
    calendar: Arc<dyn ResettableNestedCalendar>,
    lifecycle: AtomicExecutorLifecycle,
    collector: &'a mut SimulatorCollector,
    account: &'a mut dyn AtomicExecutorAccount,
}

impl<'a> AtomicNestedInnerAdapter<'a> {
    /// Construct an inner adapter whose atomic lifecycle shares the exact same calendar object.
    #[must_use]
    pub fn new(
        calendar: Arc<dyn ResettableNestedCalendar>,
        collector: &'a mut SimulatorCollector,
        account: &'a mut dyn AtomicExecutorAccount,
        settle_type: impl Into<String>,
        indicator_config: IndicatorConfig,
    ) -> Self {
        let lifecycle_calendar: Arc<dyn ExecutorLifecycleCalendar> = calendar.clone();
        Self {
            calendar,
            lifecycle: AtomicExecutorLifecycle::new(
                lifecycle_calendar,
                false,
                settle_type,
                indicator_config,
            ),
            collector,
            account,
        }
    }
}

impl NestedCalendar for AtomicNestedInnerAdapter<'_> {
    fn finished(&self) -> Result<bool, NestedCalendarError> {
        self.calendar.finished()
    }

    fn trade_len(&self) -> Result<i64, NestedCalendarError> {
        self.calendar.trade_len()
    }

    fn trade_step(&self) -> Result<i64, NestedCalendarError> {
        self.calendar.trade_step()
    }

    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), NestedCalendarError> {
        NestedCalendar::step_time(&*self.calendar)
    }

    fn step(&self) -> Result<(), NestedCalendarError> {
        NestedCalendar::step(&*self.calendar)
    }
}

impl NestedInnerExecutor for AtomicNestedInnerAdapter<'_> {
    fn reset_window(
        &mut self,
        start_time: NaiveDateTime,
        end_time: NaiveDateTime,
    ) -> Result<(), NestedInnerExecutorError> {
        self.calendar
            .reset_window(start_time, end_time)
            .map_err(|error| NestedInnerExecutorError {
                message: error.to_string(),
            })
    }

    fn collect_data(
        &mut self,
        decision: &mut dyn OrderDecision,
        level: usize,
    ) -> Result<Vec<SharedOrderExecution>, NestedInnerExecutorError> {
        let collection = self
            .lifecycle
            .collect_data(self.collector, decision, self.account, None, None, level)
            .map_err(|error| NestedInnerExecutorError {
                message: error.to_string(),
            })?;
        Ok(collection
            .execution_result()
            .iter()
            // Legacy borrowed decisions still require an explicit detached snapshot here.
            .map(|execution| OwnedOrderExecution::from_execution(*execution).into_shared())
            .collect())
    }

    fn order_indicator_handle(
        &self,
    ) -> Result<crate::SharedOrderIndicator<NumpyOrderIndicator>, NestedInnerExecutorError> {
        self.account
            .order_indicator_handle()
            .map_err(|error| NestedInnerExecutorError {
                message: error.to_string(),
            })
    }

    fn order_indicator_snapshot(&self) -> Result<NumpyOrderIndicator, NestedInnerExecutorError> {
        self.account
            .order_indicator_snapshot()
            .map_err(|error| NestedInnerExecutorError {
                message: error.to_string(),
            })
    }
}
