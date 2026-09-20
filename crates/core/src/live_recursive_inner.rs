//! Recursive native sessions retaining original decisions and live result lists.

use chrono::NaiveDateTime;

use crate::decision_update::LiveDecisionHandle;
use crate::live_nested_executor::{
    LiveNestedCollection, LiveNestedError, LiveNestedEvent, LiveNestedExecutor, LiveNestedRun,
};
use crate::nested_executor::{
    LiveNestedControlEvent, LiveNestedInnerCompletion, LiveNestedInnerProgress,
};
use crate::{
    NestedCalendar, NestedCalendarError, NestedExecutorResume, NestedInnerControlMode,
    NestedInnerExecutor, NestedInnerExecutorError, NestedStrategyPrompt, NumpyOrderIndicator,
    OrderDecision, ResumableNestedConfig, SharedOrderExecution, SharedOrderIndicator,
};

/// A native child assembly. Calendar/account bindings must refer to its lifecycle's objects.
pub struct LiveNestedChildAssembly {
    pub calendar: Box<dyn NestedCalendar>,
    pub config: ResumableNestedConfig,
    pub run: LiveNestedRun,
}

/// Builds owned child graphs; no decision or account guard crosses the factory callback.
pub trait LiveNestedChildAssemblyFactory: Send {
    /// # Errors
    /// Returns invalid configuration or plugin construction failure.
    fn assemble(
        &mut self,
        decision: &LiveDecisionHandle,
        level: usize,
    ) -> Result<LiveNestedChildAssembly, NestedInnerExecutorError>;
}

struct ActiveChild {
    decision: LiveDecisionHandle,
    executor: LiveNestedExecutor,
}

/// Reusable native session, releasing its graph on completion, failure, close or drop.
pub struct LiveConfiguredChildSession {
    factory: Box<dyn LiveNestedChildAssemblyFactory>,
    active: Option<ActiveChild>,
}

impl LiveConfiguredChildSession {
    #[must_use]
    pub fn new(factory: Box<dyn LiveNestedChildAssemblyFactory>) -> Self {
        Self {
            factory,
            active: None,
        }
    }

    /// Begin a new graph with exactly the incoming decision and call depth.
    /// # Errors
    /// Rejects an overlapping session, assembly failure or first execution failure.
    pub fn begin(
        &mut self,
        decision: LiveDecisionHandle,
        level: usize,
    ) -> Result<LiveNestedInnerProgress, NestedInnerExecutorError> {
        if self.active.is_some() {
            return Err(error("live child session is already active"));
        }
        let mut assembly = self.factory.assemble(&decision, level)?;
        // These are call arguments, not independently configurable factory defaults.
        assembly.run.outer = decision.clone();
        assembly.config.level = level;
        self.active = Some(ActiveChild {
            decision,
            executor: LiveNestedExecutor::new(assembly.calendar, assembly.config, assembly.run),
        });
        self.resume(NestedExecutorResume::Continue)
    }

    /// Forward an action/event to the same suspended graph.
    /// # Errors
    /// Returns inactive-session or child failure. A failed graph cannot be resumed again.
    pub fn resume(
        &mut self,
        input: NestedExecutorResume,
    ) -> Result<LiveNestedInnerProgress, NestedInnerExecutorError> {
        let mut active = self
            .active
            .take()
            .ok_or_else(|| error("live child session is not active"))?;
        match active
            .executor
            .resume(input)
            .map_err(|cause| error(&cause.to_string()))?
        {
            LiveNestedEvent::Suspended(event) => {
                self.active = Some(active);
                Ok(LiveNestedInnerProgress::Suspended(event))
            }
            LiveNestedEvent::Complete(collection) => Ok(LiveNestedInnerProgress::Complete(
                LiveNestedInnerCompletion {
                    decision: active.decision.clone(),
                    executions: collection.executions,
                },
            )),
        }
    }

    /// Release the active graph even when a delegate fails to close; repeated close is inert.
    /// # Errors
    /// Returns the delegate's cleanup failure without calling normal finalization hooks.
    pub fn close(&mut self) -> Result<(), NestedInnerExecutorError> {
        if let Some(mut active) = self.active.take() {
            active
                .executor
                .close()
                .map_err(|cause| error(&cause.to_string()))?;
        }
        Ok(())
    }
}

fn error(message: &str) -> NestedInnerExecutorError {
    NestedInnerExecutorError {
        message: message.to_owned(),
    }
}

/// Reuses existing calendar/reset/raw-indicator capabilities while delegating live collection.
pub struct LiveRecursiveInnerAdapter {
    lifecycle: Box<dyn NestedInnerExecutor>,
    child: LiveConfiguredChildSession,
}

impl LiveRecursiveInnerAdapter {
    #[must_use]
    pub fn new(lifecycle: Box<dyn NestedInnerExecutor>, child: LiveConfiguredChildSession) -> Self {
        Self { lifecycle, child }
    }
}

impl NestedCalendar for LiveRecursiveInnerAdapter {
    fn finished(&self) -> Result<bool, NestedCalendarError> {
        self.lifecycle.finished()
    }
    fn trade_len(&self) -> Result<i64, NestedCalendarError> {
        self.lifecycle.trade_len()
    }
    fn trade_step(&self) -> Result<i64, NestedCalendarError> {
        self.lifecycle.trade_step()
    }
    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), NestedCalendarError> {
        self.lifecycle.step_time()
    }
    fn step(&self) -> Result<(), NestedCalendarError> {
        self.lifecycle.step()
    }
}

impl NestedInnerExecutor for LiveRecursiveInnerAdapter {
    fn reset_window(
        &mut self,
        start: NaiveDateTime,
        end: NaiveDateTime,
    ) -> Result<(), NestedInnerExecutorError> {
        self.lifecycle.reset_window(start, end)
    }
    fn collect_data(
        &mut self,
        _decision: &mut dyn OrderDecision,
        _level: usize,
    ) -> Result<Vec<SharedOrderExecution>, NestedInnerExecutorError> {
        Err(error(
            "live recursive child requires the native resumable protocol",
        ))
    }
    fn live_control_mode(&self) -> NestedInnerControlMode {
        NestedInnerControlMode::Delegated
    }
    fn begin_live_collect_data(
        &mut self,
        decision: LiveDecisionHandle,
        level: usize,
    ) -> Result<LiveNestedInnerProgress, NestedInnerExecutorError> {
        self.child.begin(decision, level)
    }
    fn resume_live_collect_data(
        &mut self,
        input: NestedExecutorResume,
    ) -> Result<LiveNestedInnerProgress, NestedInnerExecutorError> {
        self.child.resume(input)
    }
    fn close_live_collect_data(&mut self) -> Result<(), NestedInnerExecutorError> {
        self.child.close()
    }
    fn order_indicator_handle(
        &self,
    ) -> Result<SharedOrderIndicator<NumpyOrderIndicator>, NestedInnerExecutorError> {
        self.lifecycle.order_indicator_handle()
    }
    fn order_indicator_snapshot(&self) -> Result<NumpyOrderIndicator, NestedInnerExecutorError> {
        self.lifecycle.order_indicator_snapshot()
    }
}

/// Terminal event when skipping tracking yields to reach an RL policy prompt.
pub enum LiveRecursiveStrategyEvent {
    StrategyPrompt(NestedStrategyPrompt),
    Complete(LiveNestedCollection),
}

enum DriverExecutor<'a> {
    Borrowed(&'a mut LiveNestedExecutor),
    Owned(Box<LiveNestedExecutor>),
}

/// `_iter_strategy` driver retaining each original observed decision.
pub struct LiveRecursiveStrategyDriver<'a> {
    executor: DriverExecutor<'a>,
    decisions: Vec<LiveDecisionHandle>,
}

impl LiveRecursiveStrategyDriver<'static> {
    #[must_use]
    pub fn new_owned(executor: LiveNestedExecutor) -> Self {
        Self {
            executor: DriverExecutor::Owned(Box::new(executor)),
            decisions: Vec::new(),
        }
    }
}

impl<'a> LiveRecursiveStrategyDriver<'a> {
    #[must_use]
    pub fn new(executor: &'a mut LiveNestedExecutor) -> Self {
        Self {
            executor: DriverExecutor::Borrowed(executor),
            decisions: Vec::new(),
        }
    }
    #[must_use]
    pub fn decisions(&self) -> &[LiveDecisionHandle] {
        &self.decisions
    }

    /// # Errors
    /// Returns the active delegate's cleanup failure.
    pub fn close(&mut self) -> Result<(), LiveNestedError> {
        match &mut self.executor {
            DriverExecutor::Borrowed(executor) => executor.close(),
            DriverExecutor::Owned(executor) => executor.close(),
        }
    }

    /// Repeatedly send the given action through intermediate tracking yields, as Python does.
    /// # Errors
    /// Returns the first protocol or nested execution failure.
    pub fn advance(
        &mut self,
        action: Option<f64>,
    ) -> Result<LiveRecursiveStrategyEvent, LiveNestedError> {
        let input = action.map_or(NestedExecutorResume::Continue, |value| {
            NestedExecutorResume::Action(Some(value))
        });
        let executor = match &mut self.executor {
            DriverExecutor::Borrowed(executor) => &mut **executor,
            DriverExecutor::Owned(executor) => executor.as_mut(),
        };
        loop {
            match executor.resume(input)? {
                LiveNestedEvent::Suspended(LiveNestedControlEvent::TrackedDecision(decision)) => {
                    self.decisions.push(decision);
                }
                LiveNestedEvent::Suspended(LiveNestedControlEvent::StrategyPrompt(prompt)) => {
                    return Ok(LiveRecursiveStrategyEvent::StrategyPrompt(prompt));
                }
                LiveNestedEvent::Complete(collection) => {
                    return Ok(LiveRecursiveStrategyEvent::Complete(collection));
                }
            }
        }
    }
}
