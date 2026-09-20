//! Native nested generator transport: retain decisions rather than tracking snapshots.

use std::sync::{Arc, Mutex};

use thiserror::Error;

use crate::base_price::LiveBasePriceStep;
use crate::decision_update::{
    LiveDecisionAccessError, LiveDecisionHandle, SharedDecisionUpdateError,
};
use crate::nested_executor::{
    LiveNestedControlEvent, LiveNestedInnerProgress, SharedNestedResult, snapshot_nested_result,
};
use crate::nested_executor_lifecycle::LiveNestedBarEnd;
use crate::{
    NestedCalendar, NestedCalendarError, NestedDecisionCalendarAdapter, NestedExecutorAccount,
    NestedExecutorAccountError, NestedExecutorResume, NestedExecutorReturnSink,
    NestedExecutorReturnSinkError, NestedInnerControlMode, NestedInnerExecutor,
    NestedInnerExecutorError, NestedLevelBinding, NestedLevelBindingError, NestedStrategyError,
    NestedStrategyPrompt, NumpyOrderIndicator, RangeLimitDefault, ResumableNestedConfig,
    SharedOrderExecution, SharedOrderIndicator, TradeCalendarRange,
};

/// A live strategy may return a decision immediately or yield repeatedly to an RL policy.
pub enum LiveNestedStrategyProgress {
    Ready(LiveDecisionHandle),
    Suspended(NestedStrategyPrompt),
}

/// In-process plugin boundary using original decisions and retainable previous result lists.
pub trait LiveNestedStrategy: Send {
    /// # Errors
    /// Returns the strategy initialization failure.
    fn reset(&mut self, outer: &LiveDecisionHandle) -> Result<(), NestedStrategyError>;
    /// Return the decision selected by the alteration hook, including a further replacement.
    /// # Errors
    /// Returns the original hook failure, preserving earlier mutations.
    fn alter_outer_decision(
        &mut self,
        outer: LiveDecisionHandle,
    ) -> Result<LiveDecisionHandle, NestedStrategyError>;
    /// # Errors
    /// Returns a generation failure without holding any framework result-list guard.
    fn begin(
        &mut self,
        previous: Option<&SharedNestedResult>,
    ) -> Result<LiveNestedStrategyProgress, NestedStrategyError>;
    /// # Errors
    /// Returns an invalid-state or generation failure.
    fn resume(
        &mut self,
        _volume: Option<f64>,
    ) -> Result<LiveNestedStrategyProgress, NestedStrategyError> {
        Err(NestedStrategyError {
            message: "live strategy is not suspended".into(),
        })
    }
    /// # Errors
    /// Returns a cleanup failure; active implementations must release even on failure.
    fn close(&mut self) -> Result<(), NestedStrategyError> {
        Ok(())
    }
    /// # Errors
    /// Returns the post-step failure; mutations to the actual child list are retained.
    fn post_execute(&mut self, executions: &SharedNestedResult) -> Result<(), NestedStrategyError>;
    /// # Errors
    /// Returns a finalization failure.
    fn post_upper_level(&mut self) -> Result<(), NestedStrategyError>;
}

/// Completed native collection with live decisions and independently flattened result list.
pub struct LiveNestedCollection {
    pub executions: SharedNestedResult,
    pub inner_order_indicators: Vec<SharedOrderIndicator<NumpyOrderIndicator>>,
    pub decisions: Vec<LiveBasePriceStep>,
}

pub enum LiveNestedEvent {
    Suspended(LiveNestedControlEvent),
    Complete(LiveNestedCollection),
}

/// Owned collaborators. The range provider must refer to the same resettable inner calendar.
/// It is required explicitly so time ranges are never silently evaluated without a calendar.
pub struct LiveNestedRun {
    pub level_binding: Box<dyn NestedLevelBinding>,
    pub inner: Box<dyn NestedInnerExecutor>,
    pub inner_range_calendar: Arc<dyn TradeCalendarRange>,
    pub strategy: Box<dyn LiveNestedStrategy>,
    pub outer: LiveDecisionHandle,
    pub account: Box<dyn NestedExecutorAccount>,
    pub return_sink: Option<Box<dyn NestedExecutorReturnSink>>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum LiveNestedError {
    #[error("a newly created executor must be advanced with Continue")]
    InvalidInitialAction,
    #[error("live nested executor has already completed")]
    Complete,
    #[error("live nested executor has been closed")]
    Closed,
    #[error("live nested executor stopped after an earlier failure")]
    Failed,
    #[error(transparent)]
    Calendar(#[from] NestedCalendarError),
    #[error(transparent)]
    Decision(#[from] LiveDecisionAccessError),
    #[error(transparent)]
    Update(#[from] SharedDecisionUpdateError),
    #[error(transparent)]
    Strategy(#[from] NestedStrategyError),
    #[error(transparent)]
    Inner(#[from] NestedInnerExecutorError),
    #[error(transparent)]
    Binding(#[from] NestedLevelBindingError),
    #[error(transparent)]
    Account(#[from] NestedExecutorAccountError),
    #[error(transparent)]
    ReturnSink(#[from] NestedExecutorReturnSinkError),
}

enum Phase {
    Start,
    Initialize,
    Loop,
    Strategy,
    TrackedChild(LiveDecisionHandle),
    Child,
    Complete,
    Closed,
    Failed,
}

/// Native counterpart of the nested Python generator, including owned suspension delegates.
/// Legacy snapshot-based plugins are not implicitly adapted into this path.
pub struct LiveNestedExecutor {
    calendar: Box<dyn NestedCalendar>,
    config: ResumableNestedConfig,
    run: LiveNestedRun,
    current_outer: LiveDecisionHandle,
    phase: Phase,
    executions: Vec<SharedOrderExecution>,
    indicators: Vec<SharedOrderIndicator<NumpyOrderIndicator>>,
    decisions: Vec<LiveBasePriceStep>,
    previous: Option<SharedNestedResult>,
}

impl LiveNestedExecutor {
    #[must_use]
    pub fn new(
        calendar: Box<dyn NestedCalendar>,
        config: ResumableNestedConfig,
        run: LiveNestedRun,
    ) -> Self {
        Self {
            calendar,
            config,
            current_outer: run.outer.clone(),
            run,
            phase: Phase::Start,
            executions: Vec::new(),
            indicators: Vec::new(),
            decisions: Vec::new(),
            previous: None,
        }
    }

    /// Advance to the next original decision, strategy prompt, or completion.
    /// # Errors
    /// Returns the first reached failure. Execution errors terminate this generator.
    pub fn resume(
        &mut self,
        input: NestedExecutorResume,
    ) -> Result<LiveNestedEvent, LiveNestedError> {
        if matches!(self.phase, Phase::Start) && input != NestedExecutorResume::Continue {
            return Err(LiveNestedError::InvalidInitialAction);
        }
        // Every fallible stage runs with Failed installed; success explicitly advances phase.
        let mut input = input;
        loop {
            let phase = std::mem::replace(&mut self.phase, Phase::Failed);
            let event = match phase {
                Phase::Start => {
                    self.phase = Phase::Initialize;
                    if self.config.decision_tracking.outer {
                        Some(LiveNestedEvent::Suspended(
                            LiveNestedControlEvent::TrackedDecision(self.run.outer.clone()),
                        ))
                    } else {
                        None
                    }
                }
                Phase::Initialize => {
                    self.initialize()?;
                    self.phase = Phase::Loop;
                    None
                }
                Phase::Loop => {
                    if self.run.inner.finished()? {
                        return self.finish();
                    }
                    if let Some(updated) = self
                        .current_outer
                        .clone()
                        .update(&NestedDecisionCalendarAdapter::new(&*self.run.inner))?
                    {
                        self.current_outer = self.run.strategy.alter_outer_decision(updated)?;
                    }
                    // Python evaluates empty() before inspecting the skip flag.
                    if self.current_outer.is_empty()? && self.config.skip_empty_decision {
                        return self.finish();
                    }
                    let range = self.current_outer.range_limit(
                        Some(&*self.run.inner_range_calendar),
                        RangeLimitDefault::Value(None),
                    )?;
                    let (start, end) = match range {
                        Some(value) => value,
                        None => (0, self.run.inner.trade_len()? - 1),
                    };
                    // Conversely, Python does not read trade_step when alignment is disabled.
                    if self.config.align_range_limit {
                        let step = self.run.inner.trade_step()?;
                        if !(start <= step && step <= end) {
                            self.run.inner.step()?;
                            self.phase = Phase::Loop;
                            continue;
                        }
                    }
                    let progress = self.run.strategy.begin(self.previous.as_ref())?;
                    self.strategy_progress(progress)?
                }
                Phase::Strategy => {
                    let volume = match input {
                        NestedExecutorResume::Continue => None,
                        NestedExecutorResume::Action(value) => value,
                    };
                    let progress = self.run.strategy.resume(volume)?;
                    self.strategy_progress(progress)?
                }
                Phase::TrackedChild(decision) => self.start_child(decision)?,
                Phase::Child => {
                    let progress = self.run.inner.resume_live_collect_data(input)?;
                    self.child_progress(progress)?
                }
                Phase::Complete => {
                    self.phase = Phase::Complete;
                    return Err(LiveNestedError::Complete);
                }
                Phase::Closed => {
                    self.phase = Phase::Closed;
                    return Err(LiveNestedError::Closed);
                }
                Phase::Failed => return Err(LiveNestedError::Failed),
            };
            if let Some(event) = event {
                return Ok(event);
            }
            input = NestedExecutorResume::Continue;
        }
    }

    fn initialize(&mut self) -> Result<(), LiveNestedError> {
        if self.config.settle_type != "None" {
            self.run.account.settle_start(&self.config.settle_type)?;
        }
        let (start, end) = self.calendar.step_time()?;
        self.run.inner.reset_window(start, end)?;
        self.run.level_binding.bind_inner(&*self.run.inner)?;
        self.run.strategy.reset(&self.run.outer)?;
        Ok(())
    }

    fn strategy_progress(
        &mut self,
        progress: LiveNestedStrategyProgress,
    ) -> Result<Option<LiveNestedEvent>, LiveNestedError> {
        match progress {
            LiveNestedStrategyProgress::Suspended(prompt) => {
                self.phase = Phase::Strategy;
                Ok(Some(LiveNestedEvent::Suspended(
                    LiveNestedControlEvent::StrategyPrompt(prompt),
                )))
            }
            LiveNestedStrategyProgress::Ready(decision) => {
                self.current_outer.modify_inner_decision(&decision)?;
                let (start_time, end_time) = self.run.inner.step_time()?;
                // Record the generated decision before the child runs, not a returned replacement.
                self.decisions.push(LiveBasePriceStep {
                    decision: decision.clone(),
                    start_time,
                    end_time,
                });
                if self.config.decision_tracking.inner
                    && self.run.inner.live_control_mode() == NestedInnerControlMode::Framework
                {
                    self.phase = Phase::TrackedChild(decision.clone());
                    Ok(Some(LiveNestedEvent::Suspended(
                        LiveNestedControlEvent::TrackedDecision(decision),
                    )))
                } else {
                    self.start_child(decision)
                }
            }
        }
    }

    fn start_child(
        &mut self,
        decision: LiveDecisionHandle,
    ) -> Result<Option<LiveNestedEvent>, LiveNestedError> {
        let progress = self
            .run
            .inner
            .begin_live_collect_data(decision, self.config.level + 1)?;
        self.child_progress(progress)
    }

    fn child_progress(
        &mut self,
        progress: LiveNestedInnerProgress,
    ) -> Result<Option<LiveNestedEvent>, LiveNestedError> {
        match progress {
            LiveNestedInnerProgress::Suspended(event) => {
                self.phase = Phase::Child;
                Ok(Some(LiveNestedEvent::Suspended(event)))
            }
            LiveNestedInnerProgress::Complete(done) => {
                self.run.strategy.post_execute(&done.executions)?;
                self.executions
                    .extend(snapshot_nested_result(&done.executions)?);
                self.indicators
                    .push(self.run.inner.order_indicator_handle()?);
                self.previous = Some(done.executions);
                self.phase = Phase::Loop;
                Ok(None)
            }
        }
    }

    fn finish(&mut self) -> Result<LiveNestedEvent, LiveNestedError> {
        self.run.strategy.post_upper_level()?;
        let collection = LiveNestedCollection {
            executions: Arc::new(Mutex::new(std::mem::take(&mut self.executions))),
            inner_order_indicators: std::mem::take(&mut self.indicators),
            decisions: std::mem::take(&mut self.decisions),
        };
        let (trade_start_time, trade_end_time) = self.calendar.step_time()?;
        self.run.account.update_live_bar_end(LiveNestedBarEnd {
            trade_start_time,
            trade_end_time,
            // _collect_data's local replacements do not rebind BaseExecutor's argument.
            outer_decision: &self.run.outer,
            inner_order_indicators: &collection.inner_order_indicators,
            steps: &collection.decisions,
            indicator_config: self.config.indicator_config,
            aggregation_config: self.config.aggregation_config,
        })?;
        self.calendar.step()?;
        if self.config.settle_type != "None" {
            self.run.account.settle_commit()?;
        }
        if let Some(sink) = self.run.return_sink.as_deref_mut() {
            sink.store_execute_result(&collection.executions)?;
        }
        self.phase = Phase::Complete;
        Ok(LiveNestedEvent::Complete(collection))
    }

    /// Cancel the currently suspended delegate without performing normal finalization.
    /// # Errors
    /// Returns delegate cleanup failure; repeated close never calls it again.
    pub fn close(&mut self) -> Result<(), LiveNestedError> {
        let phase = std::mem::replace(&mut self.phase, Phase::Closed);
        let result = match phase {
            Phase::Strategy => self.run.strategy.close().map_err(LiveNestedError::from),
            Phase::Child => self
                .run
                .inner
                .close_live_collect_data()
                .map_err(LiveNestedError::from),
            Phase::Complete => {
                self.phase = Phase::Complete;
                Ok(())
            }
            Phase::Failed => {
                self.phase = Phase::Failed;
                Ok(())
            }
            _ => Ok(()),
        };
        self.executions.clear();
        self.indicators.clear();
        self.decisions.clear();
        self.previous = None;
        result
    }
}

impl Drop for LiveNestedExecutor {
    fn drop(&mut self) {
        if let Err(error) = self.close() {
            tracing::warn!(%error, "failed to close live nested executor");
        }
    }
}
