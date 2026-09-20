//! Explicit suspend/resume orchestration for generator-producing nested strategies.

use std::mem;

use crate::nested_executor::{SharedNestedResult, snapshot_nested_result};

use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    BasePriceStep, IndicatorConfig, NestedBarEnd, NestedCalendar, NestedCalendarError,
    NestedCollection, NestedControlEvent, NestedDecisionRecord, NestedDecisionUpdate,
    NestedExecutorAccount, NestedExecutorAccountError, NestedExecutorError, NestedExecutorResume,
    NestedExecutorReturnSink, NestedExecutorReturnSinkError, NestedInnerControlMode,
    NestedInnerExecutor, NestedInnerProgress, NestedLevelBinding, NestedOuterDecision,
    NestedStrategy, NestedStrategyProgress, NumpyOrderIndicator, OrderDecision,
    OrderIndicatorAggregationConfig, SharedOrderExecution, TrackedOrderDecision,
};

const NO_SETTLEMENT: &str = "None";

/// One externally observable generator event.
pub enum ResumableNestedEvent {
    TrackedDecision(TrackedOrderDecision),
    StrategyPrompt(crate::NestedStrategyPrompt),
    Complete(NestedCollection),
}

impl From<NestedControlEvent> for ResumableNestedEvent {
    fn from(event: NestedControlEvent) -> Self {
        match event {
            NestedControlEvent::TrackedDecision(decision) => Self::TrackedDecision(decision),
            NestedControlEvent::StrategyPrompt(prompt) => Self::StrategyPrompt(prompt),
        }
    }
}

/// Selects which Python `track_data` decision yields are exposed to the caller.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NestedDecisionTracking {
    pub outer: bool,
    pub inner: bool,
}

/// Immutable behavior and lifecycle settings for one resumable nested executor.
#[derive(Clone, Debug)]
pub struct ResumableNestedConfig {
    pub skip_empty_decision: bool,
    pub align_range_limit: bool,
    pub decision_tracking: NestedDecisionTracking,
    pub settle_type: String,
    pub indicator_config: IndicatorConfig,
    pub aggregation_config: OrderIndicatorAggregationConfig,
    pub level: usize,
}

/// Owned plugin collaborators retained while execution is suspended.
pub struct ResumableNestedRun {
    pub level_binding: Box<dyn NestedLevelBinding>,
    pub inner: Box<dyn NestedInnerExecutor>,
    pub strategy: Box<dyn NestedStrategy>,
    pub outer: Box<dyn NestedOuterDecision>,
    pub account: Box<dyn NestedExecutorAccount>,
    pub return_sink: Option<Box<dyn NestedExecutorReturnSink>>,
}

/// Typed failure from a resumable nested lifecycle.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ResumableNestedExecutorError {
    #[error("a newly created executor must be advanced with Continue")]
    InvalidInitialAction,
    #[error("resumable nested executor has already completed")]
    AlreadyComplete,
    #[error("resumable nested executor stopped after an earlier failure")]
    Failed,
    #[error("resumable nested executor has been closed")]
    Closed,
    #[error(transparent)]
    Collection(#[from] NestedExecutorError),
    #[error(transparent)]
    Account(#[from] NestedExecutorAccountError),
    #[error(transparent)]
    Calendar(#[from] NestedCalendarError),
    #[error(transparent)]
    ReturnSink(#[from] NestedExecutorReturnSinkError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Start,
    AfterOuterTrack,
    Loop,
    AwaitStrategy,
    AfterInnerTrack,
    AwaitInner,
    Complete,
    Failed,
    Closed,
}

/// Stackless state machine mirroring the nested Python generator at one executor level.
pub struct ResumableNestedExecutor {
    calendar: Box<dyn NestedCalendar>,
    config: ResumableNestedConfig,
    level_binding: Box<dyn NestedLevelBinding>,
    inner: Box<dyn NestedInnerExecutor>,
    strategy: Box<dyn NestedStrategy>,
    outer: Box<dyn NestedOuterDecision>,
    account: Box<dyn NestedExecutorAccount>,
    return_sink: Option<Box<dyn NestedExecutorReturnSink>>,
    phase: Phase,
    executions: Vec<SharedOrderExecution>,
    inner_order_indicators: Vec<crate::SharedOrderIndicator<NumpyOrderIndicator>>,
    decisions: Vec<NestedDecisionRecord>,
    previous: Option<SharedNestedResult>,
    current_decision: Option<Box<dyn OrderDecision>>,
    current_interval: Option<(NaiveDateTime, NaiveDateTime)>,
}

impl Drop for ResumableNestedExecutor {
    fn drop(&mut self) {
        if let Err(error) = self.close() {
            tracing::warn!(%error, "failed to close suspended nested executor");
        }
    }
}

impl ResumableNestedExecutor {
    #[must_use]
    pub fn new(
        calendar: Box<dyn NestedCalendar>,
        config: ResumableNestedConfig,
        run: ResumableNestedRun,
    ) -> Self {
        Self {
            calendar,
            config,
            level_binding: run.level_binding,
            inner: run.inner,
            strategy: run.strategy,
            outer: run.outer,
            account: run.account,
            return_sink: run.return_sink,
            phase: Phase::Start,
            executions: Vec::new(),
            inner_order_indicators: Vec::new(),
            decisions: Vec::new(),
            previous: None,
            current_decision: None,
            current_interval: None,
        }
    }

    /// Advance until the next tracked decision, strategy prompt, or final result.
    ///
    /// An action received at either tracked-decision yield is deliberately ignored, matching
    /// Python's bare `yield trade_decision`. An action received at a strategy prompt is forwarded
    /// to [`NestedStrategy::resume_trade_decision`].
    ///
    /// # Errors
    ///
    /// Returns the first protocol, calendar, decision, strategy, inner-executor, account, or sink
    /// failure. Plugin failures terminate the machine because the Python generator would unwind.
    pub fn resume(
        &mut self,
        input: NestedExecutorResume,
    ) -> Result<ResumableNestedEvent, ResumableNestedExecutorError> {
        if self.phase == Phase::Start && input != NestedExecutorResume::Continue {
            return Err(ResumableNestedExecutorError::InvalidInitialAction);
        }
        let result = self.resume_inner(input);
        if matches!(
            result,
            Err(ref error)
                if !matches!(
                    error,
                    ResumableNestedExecutorError::AlreadyComplete
                        | ResumableNestedExecutorError::Failed
                        | ResumableNestedExecutorError::Closed
                )
        ) {
            self.phase = Phase::Failed;
        }
        result
    }

    /// Cancel only the currently suspended delegate and discard session-local results.
    /// Repeated close is inert, including after a cleanup failure. Successful completion
    /// stays complete; cancellation never runs finalization, bar-end, settlement or sink hooks.
    ///
    /// # Errors
    /// Returns the active strategy or inner executor's typed cleanup failure.
    pub fn close(&mut self) -> Result<(), ResumableNestedExecutorError> {
        let phase = self.phase;
        if !matches!(phase, Phase::Complete | Phase::Failed) {
            self.phase = Phase::Closed;
        }
        let result = match phase {
            Phase::AwaitStrategy => self
                .strategy
                .close_trade_decision()
                .map_err(NestedExecutorError::from),
            Phase::AwaitInner => self
                .inner
                .close_collect_data()
                .map_err(NestedExecutorError::from),
            _ => Ok(()),
        };
        self.executions.clear();
        self.inner_order_indicators.clear();
        self.decisions.clear();
        self.previous = None;
        self.current_decision = None;
        self.current_interval = None;
        result.map_err(ResumableNestedExecutorError::from)
    }

    fn resume_inner(
        &mut self,
        input: NestedExecutorResume,
    ) -> Result<ResumableNestedEvent, ResumableNestedExecutorError> {
        let mut input = Some(input);
        loop {
            match self.phase {
                Phase::Start => {
                    input.take();
                    self.phase = Phase::AfterOuterTrack;
                    if self.config.decision_tracking.outer {
                        return Ok(ResumableNestedEvent::TrackedDecision(
                            TrackedOrderDecision::capture(self.outer.order_decision()),
                        ));
                    }
                }
                Phase::AfterOuterTrack => {
                    input.take();
                    self.initialize()?;
                    self.phase = Phase::Loop;
                }
                Phase::Loop => {
                    if self.inner.finished().map_err(NestedExecutorError::from)? {
                        return self.finish();
                    }
                    if self
                        .outer
                        .update(&*self.inner)
                        .map_err(NestedExecutorError::from)?
                        == NestedDecisionUpdate::Replaced
                    {
                        self.strategy
                            .alter_outer_decision(&mut *self.outer)
                            .map_err(NestedExecutorError::from)?;
                    }
                    if self.config.skip_empty_decision
                        && self.outer.is_empty().map_err(NestedExecutorError::from)?
                    {
                        return self.finish();
                    }
                    let (start_idx, end_idx) = match self
                        .outer
                        .range_limit(&*self.inner)
                        .map_err(NestedExecutorError::from)?
                    {
                        Some(range) => range,
                        None => (
                            0,
                            self.inner.trade_len().map_err(NestedExecutorError::from)? - 1,
                        ),
                    };
                    let step = self.inner.trade_step().map_err(NestedExecutorError::from)?;
                    if self.config.align_range_limit && !(start_idx <= step && step <= end_idx) {
                        self.inner.step().map_err(NestedExecutorError::from)?;
                        continue;
                    }
                    match self
                        .strategy
                        .begin_shared_trade_decision(self.previous.as_ref())
                        .map_err(NestedExecutorError::from)?
                    {
                        NestedStrategyProgress::Ready(decision) => {
                            if let Some(event) = self.prepare_decision(decision)? {
                                return Ok(event);
                            }
                        }
                        NestedStrategyProgress::Suspended(prompt) => {
                            self.phase = Phase::AwaitStrategy;
                            return Ok(ResumableNestedEvent::StrategyPrompt(prompt));
                        }
                    }
                }
                Phase::AwaitStrategy => {
                    let strategy_input = input.take().unwrap_or(NestedExecutorResume::Continue);
                    if let Some(event) = self.resume_strategy(strategy_input)? {
                        return Ok(event);
                    }
                }
                Phase::AfterInnerTrack => {
                    input.take();
                    if let Some(event) = self.start_current()? {
                        return Ok(event);
                    }
                }
                Phase::AwaitInner => {
                    let child_input = input.take().unwrap_or(NestedExecutorResume::Continue);
                    if let Some(event) = self.resume_child(child_input)? {
                        return Ok(event);
                    }
                }
                Phase::Complete => return Err(ResumableNestedExecutorError::AlreadyComplete),
                Phase::Failed => return Err(ResumableNestedExecutorError::Failed),
                Phase::Closed => return Err(ResumableNestedExecutorError::Closed),
            }
        }
    }

    fn initialize(&mut self) -> Result<(), ResumableNestedExecutorError> {
        if self.config.settle_type != NO_SETTLEMENT {
            self.account.settle_start(&self.config.settle_type)?;
        }
        let (start_time, end_time) = self
            .calendar
            .step_time()
            .map_err(NestedExecutorError::from)?;
        self.inner
            .reset_window(start_time, end_time)
            .map_err(NestedExecutorError::from)?;
        self.level_binding
            .bind_inner(&*self.inner)
            .map_err(NestedExecutorError::from)?;
        self.strategy
            .reset(&*self.outer)
            .map_err(NestedExecutorError::from)?;
        Ok(())
    }

    fn prepare_decision(
        &mut self,
        mut decision: Box<dyn OrderDecision>,
    ) -> Result<Option<ResumableNestedEvent>, ResumableNestedExecutorError> {
        self.outer
            .modify_inner_decision(&mut *decision)
            .map_err(NestedExecutorError::from)?;
        let interval = self.inner.step_time().map_err(NestedExecutorError::from)?;
        let tracked = self.config.decision_tracking.inner
            && self.inner.control_mode() == NestedInnerControlMode::Framework;
        let tracked = tracked.then(|| TrackedOrderDecision::capture(&*decision));
        self.current_decision = Some(decision);
        self.current_interval = Some(interval);
        if let Some(tracked) = tracked {
            self.phase = Phase::AfterInnerTrack;
            Ok(Some(ResumableNestedEvent::TrackedDecision(tracked)))
        } else {
            self.start_current()
        }
    }

    fn start_current(
        &mut self,
    ) -> Result<Option<ResumableNestedEvent>, ResumableNestedExecutorError> {
        let decision = self
            .current_decision
            .take()
            .expect("phase guarantees a current decision");
        let progress = self
            .inner
            .begin_collect_data(decision, self.config.level + 1)
            .map_err(NestedExecutorError::from)?;
        self.handle_inner_progress(progress)
    }

    fn resume_strategy(
        &mut self,
        input: NestedExecutorResume,
    ) -> Result<Option<ResumableNestedEvent>, ResumableNestedExecutorError> {
        let volume = match input {
            NestedExecutorResume::Continue => None,
            NestedExecutorResume::Action(volume) => volume,
        };
        match self
            .strategy
            .resume_trade_decision(volume)
            .map_err(NestedExecutorError::from)?
        {
            NestedStrategyProgress::Ready(decision) => self.prepare_decision(decision),
            NestedStrategyProgress::Suspended(prompt) => {
                Ok(Some(ResumableNestedEvent::StrategyPrompt(prompt)))
            }
        }
    }

    fn resume_child(
        &mut self,
        input: NestedExecutorResume,
    ) -> Result<Option<ResumableNestedEvent>, ResumableNestedExecutorError> {
        let progress = self
            .inner
            .resume_collect_data(input)
            .map_err(NestedExecutorError::from)?;
        self.handle_inner_progress(progress)
    }

    fn handle_inner_progress(
        &mut self,
        progress: NestedInnerProgress,
    ) -> Result<Option<ResumableNestedEvent>, ResumableNestedExecutorError> {
        match progress {
            NestedInnerProgress::Suspended(event) => {
                self.phase = Phase::AwaitInner;
                Ok(Some(event.into()))
            }
            NestedInnerProgress::Complete {
                decision,
                executions,
            } => {
                self.finish_current(decision, executions)?;
                self.phase = Phase::Loop;
                Ok(None)
            }
        }
    }

    fn finish_current(
        &mut self,
        decision: Box<dyn OrderDecision>,
        current: SharedNestedResult,
    ) -> Result<(), ResumableNestedExecutorError> {
        let (start_time, end_time) = self
            .current_interval
            .take()
            .expect("phase guarantees a current interval");
        self.strategy
            .post_shared_execute(&current)
            .map_err(NestedExecutorError::from)?;
        self.executions
            .extend(snapshot_nested_result(&current).map_err(NestedExecutorError::from)?);
        self.inner_order_indicators.push(
            self.inner
                .order_indicator_handle()
                .map_err(NestedExecutorError::from)?,
        );
        self.decisions
            .push(NestedDecisionRecord::new(decision, start_time, end_time));
        self.previous = Some(current);
        Ok(())
    }

    fn finish(&mut self) -> Result<ResumableNestedEvent, ResumableNestedExecutorError> {
        self.strategy
            .post_upper_level()
            .map_err(NestedExecutorError::from)?;
        let collection = NestedCollection::from_parts(
            mem::take(&mut self.executions),
            mem::take(&mut self.inner_order_indicators),
            mem::take(&mut self.decisions),
        );
        let (trade_start_time, trade_end_time) = self.calendar.step_time()?;
        {
            let (inner_order_indicators, steps) = collection.aggregation_inputs();
            self.update_account(
                trade_start_time,
                trade_end_time,
                inner_order_indicators,
                &steps,
            )?;
        }
        self.calendar.step()?;
        if self.config.settle_type != NO_SETTLEMENT {
            self.account.settle_commit()?;
        }
        if let Some(return_sink) = self.return_sink.as_deref_mut() {
            return_sink.store_execute_result(collection.executions())?;
        }
        self.phase = Phase::Complete;
        Ok(ResumableNestedEvent::Complete(collection))
    }

    fn update_account(
        &mut self,
        trade_start_time: NaiveDateTime,
        trade_end_time: NaiveDateTime,
        inner_order_indicators: &[crate::SharedOrderIndicator<NumpyOrderIndicator>],
        steps: &[BasePriceStep<'_>],
    ) -> Result<(), NestedExecutorAccountError> {
        self.account.update_bar_end(NestedBarEnd {
            trade_start_time,
            trade_end_time,
            outer_decision: self.outer.order_decision(),
            inner_order_indicators,
            steps,
            indicator_config: self.config.indicator_config,
            aggregation_config: self.config.aggregation_config,
        })
    }
}
