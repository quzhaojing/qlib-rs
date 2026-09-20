//! Owned atomic child execution over the same live account used by observations.

use std::sync::Arc;

use chrono::NaiveDateTime;

use crate::decision_update::LiveDecisionHandle;
use crate::nested_executor::{
    LiveNestedControlEvent, LiveNestedInnerProgress, NestedExecutorResume, NestedInnerControlMode,
};

use crate::{
    AccountBarMarket, AtomicAccountAdapter, AtomicExecutorAccount, AtomicNestedInnerAdapter,
    IndicatorConfig, NestedCalendar, NestedCalendarError, NestedInnerExecutor,
    NestedInnerExecutorError, NumpyOrderIndicator, OrderDecision, ResettableNestedCalendar,
    SharedOrderExecution, SharedSaoeAccount, SimulatorCollector,
};

/// Owns the collector and retains shared account/market/calendar identities across calls.
/// Each synchronous collection holds the account mutex for the complete atomic lifecycle.
/// Market, dealer, reporter and calendar callbacks must not reenter that same account.
/// No guard survives a return or a nested suspension; errors keep reached state mutations.
/// Tracking is framework-owned by default, as for [`AtomicNestedInnerAdapter`]; native
/// child-owned tracking can be enabled explicitly. This adapter does not clone accounts
/// or reset collectors. Dropping a suspended adapter only releases the retained decision.
pub struct OwnedAtomicNestedInnerAdapter {
    calendar: Arc<dyn ResettableNestedCalendar>,
    collector: SimulatorCollector,
    account: SharedSaoeAccount,
    market: Arc<dyn AccountBarMarket>,
    settle_type: String,
    indicator_config: IndicatorConfig,
    show_indicator: bool,
    live_tracking: bool,
    pending_live: Option<(LiveDecisionHandle, usize)>,
}

impl OwnedAtomicNestedInnerAdapter {
    #[must_use]
    pub fn new(
        calendar: Arc<dyn ResettableNestedCalendar>,
        collector: SimulatorCollector,
        account: SharedSaoeAccount,
        market: Arc<dyn AccountBarMarket>,
        settle_type: impl Into<String>,
        indicator_config: IndicatorConfig,
        show_indicator: bool,
    ) -> Self {
        Self {
            calendar,
            collector,
            account,
            market,
            settle_type: settle_type.into(),
            indicator_config,
            show_indicator,
            live_tracking: false,
            pending_live: None,
        }
    }

    /// Configure child-owned native tracking. Legacy tracking remains framework-owned.
    #[must_use]
    pub fn with_live_tracking(mut self, enabled: bool) -> Self {
        self.live_tracking = enabled;
        self
    }
}

impl NestedCalendar for OwnedAtomicNestedInnerAdapter {
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

impl NestedInnerExecutor for OwnedAtomicNestedInnerAdapter {
    fn live_control_mode(&self) -> NestedInnerControlMode {
        if self.live_tracking {
            NestedInnerControlMode::Delegated
        } else {
            NestedInnerControlMode::Framework
        }
    }

    fn begin_live_collect_data(
        &mut self,
        decision: LiveDecisionHandle,
        level: usize,
    ) -> Result<LiveNestedInnerProgress, NestedInnerExecutorError> {
        if self.pending_live.is_some() {
            return Err(NestedInnerExecutorError {
                message: "live inner executor is already suspended".into(),
            });
        }
        if self.live_tracking {
            self.pending_live = Some((decision.clone(), level));
            Ok(LiveNestedInnerProgress::Suspended(
                LiveNestedControlEvent::TrackedDecision(decision),
            ))
        } else {
            self.collect_live_decision(decision, level)
                .map(LiveNestedInnerProgress::Complete)
        }
    }

    fn resume_live_collect_data(
        &mut self,
        _input: NestedExecutorResume,
    ) -> Result<LiveNestedInnerProgress, NestedInnerExecutorError> {
        // The value sent to BaseExecutor's tracking yield is unused by Python too.
        // Take first so failures cannot execute the same suspended collection twice.
        let (decision, level) =
            self.pending_live
                .take()
                .ok_or_else(|| NestedInnerExecutorError {
                    message: "live inner executor is not suspended".into(),
                })?;
        self.collect_live_decision(decision, level)
            .map(LiveNestedInnerProgress::Complete)
    }

    fn close_live_collect_data(&mut self) -> Result<(), NestedInnerExecutorError> {
        self.pending_live = None;
        Ok(())
    }

    fn collect_live_data(
        &mut self,
        decision: &crate::decision_update::LiveDecisionHandle,
        level: usize,
    ) -> Result<crate::nested_executor::SharedNestedResult, NestedInnerExecutorError> {
        let mut account = self.account.lock().map_err(|_| poisoned())?;
        let mut adapter =
            AtomicAccountAdapter::new(&mut account, &*self.market, self.show_indicator);
        let lifecycle = crate::shared_executor_lifecycle::SharedAtomicExecutorLifecycle {
            calendar: self.calendar.clone(),
            track_data: false,
            settle_type: self.settle_type.clone(),
            indicator_config: self.indicator_config,
        };
        lifecycle
            .collect_live_data(
                &mut self.collector,
                decision,
                &mut adapter,
                None,
                None,
                level,
            )
            .map_err(|error| NestedInnerExecutorError {
                message: error.to_string(),
            })
    }

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
        let mut account = self.account.lock().map_err(|_| poisoned())?;
        let mut adapter =
            AtomicAccountAdapter::new(&mut account, &*self.market, self.show_indicator);
        AtomicNestedInnerAdapter::new(
            self.calendar.clone(),
            &mut self.collector,
            &mut adapter,
            &self.settle_type,
            self.indicator_config,
        )
        .collect_data(decision, level)
    }

    fn order_indicator_handle(
        &self,
    ) -> Result<crate::SharedOrderIndicator<NumpyOrderIndicator>, NestedInnerExecutorError> {
        let mut account = self.account.lock().map_err(|_| poisoned())?;
        AtomicAccountAdapter::new(&mut account, &*self.market, self.show_indicator)
            .order_indicator_handle()
            .map_err(|error| NestedInnerExecutorError {
                message: error.to_string(),
            })
    }

    fn order_indicator_snapshot(&self) -> Result<NumpyOrderIndicator, NestedInnerExecutorError> {
        let mut account = self.account.lock().map_err(|_| poisoned())?;
        AtomicAccountAdapter::new(&mut account, &*self.market, self.show_indicator)
            .order_indicator_snapshot()
            .map_err(|error| NestedInnerExecutorError {
                message: error.to_string(),
            })
    }
}

fn poisoned() -> NestedInnerExecutorError {
    NestedInnerExecutorError {
        message: "owned atomic account lock poisoned".into(),
    }
}
