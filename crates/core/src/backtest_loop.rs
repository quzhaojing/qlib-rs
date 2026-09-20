//! Resumable outer backtest orchestration, independent of executor nesting depth.

use chrono::NaiveDateTime;
use thiserror::Error;

use crate::{
    NestedControlEvent, NestedExecutorResume, NestedInnerProgress, OrderDecision,
    nested_executor::SharedNestedResult,
};

/// Bound strategy/executor infrastructure for one outer backtest.
/// Collection hooks reuse the nested session protocol. On a collection error the backend
/// must unwind its own child session, just as a Python `yield from` target does.
pub trait BacktestLoopBackend: Send {
    type Error: std::error::Error + Send + Sync + 'static;
    type Reports;
    /// # Errors
    /// Returns executor-window reset failures.
    fn reset_executor(
        &mut self,
        start: NaiveDateTime,
        end: NaiveDateTime,
    ) -> Result<(), Self::Error>;
    /// Obtain current executor level infrastructure, then reset the strategy with it.
    /// # Errors
    /// Returns infrastructure access or strategy reset failures.
    fn reset_strategy(&mut self) -> Result<(), Self::Error>;
    /// # Errors
    /// Returns calendar length failures.
    fn trade_len(&mut self) -> Result<i64, Self::Error>;
    /// # Errors
    /// Returns progress context construction/entry failures.
    fn enter_progress(&mut self, total: i64) -> Result<(), Self::Error>;
    /// # Errors
    /// Returns calendar completion-query failures.
    fn finished(&mut self) -> Result<bool, Self::Error>;
    /// # Errors
    /// Returns strategy decision failures.
    fn generate(
        &mut self,
        previous: Option<&SharedNestedResult>,
    ) -> Result<Box<dyn OrderDecision>, Self::Error>;
    /// Begin one collection at outer level zero.
    /// # Errors
    /// Returns collection initialization failures.
    fn begin_collection(
        &mut self,
        decision: Box<dyn OrderDecision>,
    ) -> Result<NestedInnerProgress, Self::Error>;
    /// # Errors
    /// Returns suspended collection failures.
    fn resume_collection(
        &mut self,
        input: NestedExecutorResume,
    ) -> Result<NestedInnerProgress, Self::Error>;
    /// Close the suspended child before the outer progress context exits.
    /// # Errors
    /// Returns child generator cleanup failures.
    fn close_collection(&mut self) -> Result<(), Self::Error>;
    /// # Errors
    /// Returns strategy post-execution failures.
    fn post_execute(&mut self, executions: &SharedNestedResult) -> Result<(), Self::Error>;
    /// # Errors
    /// Returns one-step progress update failures.
    fn update_progress(&mut self) -> Result<(), Self::Error>;
    /// # Errors
    /// Returns the strategy's finalization failure.
    fn finalize_strategy(&mut self) -> Result<(), Self::Error>;
    /// # Errors
    /// Returns progress context exit failures. Must not suppress an earlier error.
    fn close_progress(&mut self) -> Result<(), Self::Error>;
    /// Build complete reports without publishing partially collected maps.
    /// # Errors
    /// Returns executor enumeration or report aggregation failures.
    fn collect_reports(&mut self) -> Result<Self::Reports, Self::Error>;
}

pub enum BacktestLoopEvent {
    Suspended(NestedControlEvent),
    Complete,
}

/// Run the non-interactive wrapper to completion, ignoring every yielded control event just as
/// Python iteration over `collect_data_loop` repeatedly sends `None`.
///
/// # Errors
/// Returns the first loop, collaborator, cleanup, or report failure.
///
/// # Panics
/// Panics only if the private report-enabled completion invariant is broken.
pub fn run_backtest_loop<B: BacktestLoopBackend>(
    backend: B,
    start: NaiveDateTime,
    end: NaiveDateTime,
) -> Result<B::Reports, BacktestLoopError<B::Error>> {
    let mut loop_ = BacktestLoop::new(backend, start, end, true);
    loop {
        match loop_.resume(NestedExecutorResume::Continue)? {
            BacktestLoopEvent::Suspended(_) => {}
            BacktestLoopEvent::Complete => {
                return Ok(loop_
                    .reports
                    .take()
                    .expect("report-enabled completion publishes reports"));
            }
        }
    }
}

#[derive(Debug, Error)]
pub enum BacktestLoopError<E: std::error::Error + 'static> {
    #[error("a newly created backtest must be advanced with Continue")]
    InvalidInitialAction,
    #[error("backtest has already completed")]
    AlreadyComplete,
    #[error("backtest stopped after an earlier failure")]
    Failed,
    #[error("backtest has been closed")]
    Closed,
    #[error(transparent)]
    Backend(E),
    #[error("{cleanup} (while handling {original})")]
    Cleanup {
        original: E,
        #[source]
        cleanup: E,
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Start,
    Ready,
    AwaitCollection,
    Complete,
    Failed,
    Closed,
}

/// Owns a multi-round outer generator and publishes reports only on normal completion.
pub struct BacktestLoop<B: BacktestLoopBackend> {
    backend: B,
    start: NaiveDateTime,
    end: NaiveDateTime,
    want_reports: bool,
    reports: Option<B::Reports>,
    previous: Option<SharedNestedResult>,
    phase: Phase,
    progress_open: bool,
}

impl<B: BacktestLoopBackend> BacktestLoop<B> {
    /// Construction is lazy, matching creation of a Python generator.
    #[must_use]
    pub fn new(backend: B, start: NaiveDateTime, end: NaiveDateTime, want_reports: bool) -> Self {
        Self {
            backend,
            start,
            end,
            want_reports,
            reports: None,
            previous: None,
            phase: Phase::Start,
            progress_open: false,
        }
    }

    #[must_use]
    pub fn reports(&self) -> Option<&B::Reports> {
        self.reports.as_ref()
    }

    /// Move completed reports into a public wrapper or caller-owned result target.
    pub fn take_reports(&mut self) -> Option<B::Reports> {
        self.reports.take()
    }

    /// Drive across any number of completed outer decisions until suspension or completion.
    /// # Errors
    /// Returns protocol errors or the first backend failure. A progress-exit failure
    /// takes precedence while retaining the original failure, matching exception context.
    pub fn resume(
        &mut self,
        input: NestedExecutorResume,
    ) -> Result<BacktestLoopEvent, BacktestLoopError<B::Error>> {
        match self.phase {
            Phase::Start if input != NestedExecutorResume::Continue => {
                return Err(BacktestLoopError::InvalidInitialAction);
            }
            Phase::Complete => return Err(BacktestLoopError::AlreadyComplete),
            Phase::Failed => return Err(BacktestLoopError::Failed),
            Phase::Closed => return Err(BacktestLoopError::Closed),
            _ => {}
        }
        match self.drive(input) {
            Ok(event) => Ok(event),
            Err(original) => {
                self.phase = Phase::Failed;
                if self.progress_open {
                    if let Err(cleanup) = self.exit_progress() {
                        return Err(BacktestLoopError::Cleanup { original, cleanup });
                    }
                }
                Err(BacktestLoopError::Backend(original))
            }
        }
    }

    fn drive(&mut self, input: NestedExecutorResume) -> Result<BacktestLoopEvent, B::Error> {
        if self.phase == Phase::Start {
            self.backend.reset_executor(self.start, self.end)?;
            self.backend.reset_strategy()?;
            let total = self.backend.trade_len()?;
            self.backend.enter_progress(total)?;
            self.progress_open = true;
            self.phase = Phase::Ready;
        }
        let mut progress = if self.phase == Phase::AwaitCollection {
            Some(self.backend.resume_collection(input)?)
        } else {
            None
        };
        loop {
            if let Some(current) = progress.take() {
                match current {
                    NestedInnerProgress::Suspended(event) => {
                        self.phase = Phase::AwaitCollection;
                        return Ok(BacktestLoopEvent::Suspended(event));
                    }
                    NestedInnerProgress::Complete { executions, .. } => {
                        self.phase = Phase::Ready;
                        self.backend.post_execute(&executions)?;
                        self.previous = Some(executions);
                        self.backend.update_progress()?;
                    }
                }
            }
            if self.backend.finished()? {
                self.backend.finalize_strategy()?;
                self.exit_progress()?;
                if self.want_reports {
                    self.reports = Some(self.backend.collect_reports()?);
                }
                self.phase = Phase::Complete;
                return Ok(BacktestLoopEvent::Complete);
            }
            let decision = self.backend.generate(self.previous.as_ref())?;
            progress = Some(self.backend.begin_collection(decision)?);
        }
    }

    fn exit_progress(&mut self) -> Result<(), B::Error> {
        self.progress_open = false;
        self.backend.close_progress()
    }

    /// Close a suspended child before closing progress; safe to call repeatedly.
    /// # Errors
    /// Returns child/progress cleanup failures, retaining both when both fail.
    pub fn close(&mut self) -> Result<(), BacktestLoopError<B::Error>> {
        let child = if self.phase == Phase::AwaitCollection {
            self.backend.close_collection().err()
        } else {
            None
        };
        if self.phase != Phase::Complete {
            self.phase = Phase::Closed;
        }
        let progress = if self.progress_open {
            self.exit_progress().err()
        } else {
            None
        };
        match (child, progress) {
            (Some(original), Some(cleanup)) => {
                Err(BacktestLoopError::Cleanup { original, cleanup })
            }
            (Some(error), None) | (None, Some(error)) => Err(BacktestLoopError::Backend(error)),
            (None, None) => Ok(()),
        }
    }
}

impl<B: BacktestLoopBackend> Drop for BacktestLoop<B> {
    fn drop(&mut self) {
        if let Err(error) = self.close() {
            tracing::warn!(%error, "backtest cleanup failed during drop");
        }
    }
}
