//! Recursive child-session adapter and `_iter_strategy`-compatible driver.

use chrono::NaiveDateTime;

use crate::{
    NestedCalendar, NestedCalendarError, NestedCollection, NestedExecutorResume,
    NestedInnerControlMode, NestedInnerExecutor, NestedInnerExecutorError, NestedInnerProgress,
    NestedStrategyPrompt, NumpyOrderIndicator, OrderDecision, ResumableNestedConfig,
    ResumableNestedEvent, ResumableNestedExecutor, ResumableNestedExecutorError,
    ResumableNestedRun, SharedOrderExecution, TrackedOrderDecision,
};

/// Owned recursively nested collection session retained across child yields.
pub trait NestedChildSession: Send {
    /// Begin collecting one owned child outer decision.
    ///
    /// # Errors
    /// Returns a child lifecycle, control-protocol, or transport failure.
    fn begin(
        &mut self,
        decision: Box<dyn OrderDecision>,
        level: usize,
    ) -> Result<NestedInnerProgress, NestedInnerExecutorError>;

    /// Resume the exact child session suspended by [`Self::begin`] or an earlier resume.
    ///
    /// # Errors
    /// Returns an invalid-state, child lifecycle, or transport failure.
    fn resume(
        &mut self,
        input: NestedExecutorResume,
    ) -> Result<NestedInnerProgress, NestedInnerExecutorError>;

    /// Close and release the active child, including when cleanup fails.
    ///
    /// # Errors
    /// Returns a child cleanup failure; repeated close must be inert.
    fn close(&mut self) -> Result<(), NestedInnerExecutorError>;
}

/// Fully owned collaborators for one freshly configured child collection.
pub struct NestedChildAssembly {
    pub calendar: Box<dyn NestedCalendar>,
    pub config: ResumableNestedConfig,
    pub run: ResumableNestedRun,
}

/// Plugin boundary that assembles a new child executor graph for each outer decision.
pub trait NestedChildAssemblyFactory: Send {
    /// Build a fresh graph without retaining a borrow of `decision`.
    ///
    /// # Errors
    /// Returns invalid configuration or plugin construction failures.
    fn assemble(
        &mut self,
        decision: &dyn OrderDecision,
        level: usize,
    ) -> Result<NestedChildAssembly, NestedInnerExecutorError>;
}

struct ActiveChild {
    decision: Box<dyn OrderDecision>,
    executor: ResumableNestedExecutor,
}

/// Reusable child session that owns each configured executor graph across suspension points.
pub struct ConfiguredNestedChildSession {
    factory: Box<dyn NestedChildAssemblyFactory>,
    active: Option<ActiveChild>,
}

impl ConfiguredNestedChildSession {
    #[must_use]
    pub fn new(factory: Box<dyn NestedChildAssemblyFactory>) -> Self {
        Self {
            factory,
            active: None,
        }
    }

    fn drive(
        &mut self,
        input: NestedExecutorResume,
    ) -> Result<NestedInnerProgress, NestedInnerExecutorError> {
        let event = self
            .active
            .as_mut()
            .ok_or_else(|| child_error("configured child session is not active"))?
            .executor
            .resume(input)
            .map_err(|error| child_error(&error.to_string()))?;
        match event {
            ResumableNestedEvent::TrackedDecision(decision) => Ok(NestedInnerProgress::Suspended(
                crate::NestedControlEvent::TrackedDecision(decision),
            )),
            ResumableNestedEvent::StrategyPrompt(prompt) => Ok(NestedInnerProgress::Suspended(
                crate::NestedControlEvent::StrategyPrompt(prompt),
            )),
            ResumableNestedEvent::Complete(collection) => {
                let active = self
                    .active
                    .take()
                    .expect("a complete event requires an active child");
                Ok(NestedInnerProgress::Complete {
                    decision: active.decision,
                    executions: collection.into_executions(),
                })
            }
        }
    }
}

impl NestedChildSession for ConfiguredNestedChildSession {
    fn begin(
        &mut self,
        decision: Box<dyn OrderDecision>,
        level: usize,
    ) -> Result<NestedInnerProgress, NestedInnerExecutorError> {
        if self.active.is_some() {
            return Err(child_error("configured child session is already active"));
        }
        let assembly = self.factory.assemble(&*decision, level)?;
        self.active = Some(ActiveChild {
            decision,
            executor: ResumableNestedExecutor::new(
                assembly.calendar,
                assembly.config,
                assembly.run,
            ),
        });
        self.drive(NestedExecutorResume::Continue)
    }

    fn resume(
        &mut self,
        input: NestedExecutorResume,
    ) -> Result<NestedInnerProgress, NestedInnerExecutorError> {
        self.drive(input)
    }

    fn close(&mut self) -> Result<(), NestedInnerExecutorError> {
        match self.active.take() {
            Some(mut active) => active
                .executor
                .close()
                .map_err(|error| child_error(&error.to_string())),
            None => Ok(()),
        }
    }
}

fn child_error(message: &str) -> NestedInnerExecutorError {
    NestedInnerExecutorError {
        message: message.to_owned(),
    }
}

/// Adapts a recursively resumable child session to the existing nested-inner boundary.
pub struct RecursiveNestedInnerAdapter {
    lifecycle: Box<dyn NestedInnerExecutor>,
    child: Box<dyn NestedChildSession>,
}

impl RecursiveNestedInnerAdapter {
    #[must_use]
    pub fn new(
        lifecycle: Box<dyn NestedInnerExecutor>,
        child: Box<dyn NestedChildSession>,
    ) -> Self {
        Self { lifecycle, child }
    }
}

impl NestedCalendar for RecursiveNestedInnerAdapter {
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

impl NestedInnerExecutor for RecursiveNestedInnerAdapter {
    fn reset_window(
        &mut self,
        start_time: NaiveDateTime,
        end_time: NaiveDateTime,
    ) -> Result<(), NestedInnerExecutorError> {
        self.lifecycle.reset_window(start_time, end_time)
    }

    fn collect_data(
        &mut self,
        _decision: &mut dyn OrderDecision,
        _level: usize,
    ) -> Result<Vec<SharedOrderExecution>, NestedInnerExecutorError> {
        Err(NestedInnerExecutorError {
            message: "recursive child collection requires the resumable protocol".to_owned(),
        })
    }

    fn control_mode(&self) -> NestedInnerControlMode {
        NestedInnerControlMode::Delegated
    }

    fn begin_collect_data(
        &mut self,
        decision: Box<dyn OrderDecision>,
        level: usize,
    ) -> Result<NestedInnerProgress, NestedInnerExecutorError> {
        self.child.begin(decision, level)
    }

    fn resume_collect_data(
        &mut self,
        input: NestedExecutorResume,
    ) -> Result<NestedInnerProgress, NestedInnerExecutorError> {
        self.child.resume(input)
    }

    fn close_collect_data(&mut self) -> Result<(), NestedInnerExecutorError> {
        self.child.close()
    }

    fn order_indicator_handle(
        &self,
    ) -> Result<crate::SharedOrderIndicator<NumpyOrderIndicator>, NestedInnerExecutorError> {
        self.lifecycle.order_indicator_handle()
    }

    fn order_indicator_snapshot(&self) -> Result<NumpyOrderIndicator, NestedInnerExecutorError> {
        self.lifecycle.order_indicator_snapshot()
    }
}

/// Terminal event returned after driving through intermediate decision yields.
pub enum RecursiveStrategyEvent {
    StrategyPrompt(NestedStrategyPrompt),
    Complete(NestedCollection),
}

/// Driver mirroring `SingleAssetOrderExecution._iter_strategy` send behavior.
pub struct RecursiveStrategyDriver<'executor> {
    executor: DriverExecutor<'executor>,
    decisions: Vec<TrackedOrderDecision>,
}

enum DriverExecutor<'executor> {
    Borrowed(&'executor mut ResumableNestedExecutor),
    Owned(Box<ResumableNestedExecutor>),
}

impl RecursiveStrategyDriver<'static> {
    /// Own the entire suspended executor graph, allowing a simulator to retain the
    /// driver between actions without borrowing a separately stored executor.
    /// Dropping the driver releases that graph; no executor or state is cloned.
    #[must_use]
    pub fn new_owned(executor: ResumableNestedExecutor) -> Self {
        Self {
            executor: DriverExecutor::Owned(Box::new(executor)),
            decisions: Vec::new(),
        }
    }
}

impl<'executor> RecursiveStrategyDriver<'executor> {
    #[must_use]
    pub fn new(executor: &'executor mut ResumableNestedExecutor) -> Self {
        Self {
            executor: DriverExecutor::Borrowed(executor),
            decisions: Vec::new(),
        }
    }

    /// Return every decision observed while searching for the next strategy prompt.
    #[must_use]
    pub fn decisions(&self) -> &[TrackedOrderDecision] {
        &self.decisions
    }

    /// Close the active executor without discarding already observed tracking events.
    ///
    /// # Errors
    /// Returns the active delegate's cleanup failure.
    pub fn close(&mut self) -> Result<(), ResumableNestedExecutorError> {
        match &mut self.executor {
            DriverExecutor::Borrowed(executor) => executor.close(),
            DriverExecutor::Owned(executor) => executor.close(),
        }
    }

    /// Advance until the next strategy prompt or final nested collection.
    ///
    /// As in Python `_iter_strategy`, a supplied action is repeatedly sent through intermediate
    /// decision yields. Bare decision yields ignore it; the exact suspended strategy receives it.
    ///
    /// # Errors
    /// Returns the first executor protocol or plugin failure.
    pub fn advance(
        &mut self,
        action: Option<f64>,
    ) -> Result<RecursiveStrategyEvent, ResumableNestedExecutorError> {
        let input = action.map_or(NestedExecutorResume::Continue, |value| {
            NestedExecutorResume::Action(Some(value))
        });
        let executor = match &mut self.executor {
            DriverExecutor::Borrowed(executor) => &mut **executor,
            DriverExecutor::Owned(executor) => executor.as_mut(),
        };
        loop {
            match executor.resume(input)? {
                ResumableNestedEvent::TrackedDecision(decision) => {
                    self.decisions.push(decision);
                }
                ResumableNestedEvent::StrategyPrompt(prompt) => {
                    return Ok(RecursiveStrategyEvent::StrategyPrompt(prompt));
                }
                ResumableNestedEvent::Complete(collection) => {
                    return Ok(RecursiveStrategyEvent::Complete(collection));
                }
            }
        }
    }
}
