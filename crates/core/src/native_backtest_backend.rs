//! Owned production assembly for the outer backtest loop.

use std::sync::{Arc, Mutex};

use chrono::NaiveDateTime;
use indicatif::ProgressBar;
use thiserror::Error;

use crate::{
    BacktestLoopBackend, BacktestReportError, BacktestReports, NestedCalendarError,
    NestedExecutorResume, NestedInnerExecutor, NestedInnerExecutorError, NestedInnerProgress,
    OrderDecision, SaoeCalendar, SaoeOrderFactory, SharedSaoeAccount, SingleOrderStrategy,
    collect_shared_backtest_reports,
};

use crate::nested_executor::SharedNestedResult;

/// Mutable order construction service shared by a reset strategy infrastructure.
pub type SharedOrderFactory = Arc<Mutex<Box<dyn SaoeOrderFactory>>>;

/// Typed level infrastructure installed into an outer strategy on every loop reset.
#[derive(Clone)]
pub struct OuterStrategyInfrastructure {
    calendar: Arc<dyn SaoeCalendar>,
    orders: SharedOrderFactory,
}

impl OuterStrategyInfrastructure {
    #[must_use]
    pub fn new(calendar: Arc<dyn SaoeCalendar>, orders: SharedOrderFactory) -> Self {
        Self { calendar, orders }
    }
}

/// Strategy lifecycle required by the outermost loop.
pub trait OuterBacktestStrategy: Send {
    /// Install the executor's current level infrastructure.
    ///
    /// # Errors
    /// Returns strategy reset failures.
    fn reset(
        &mut self,
        infrastructure: OuterStrategyInfrastructure,
    ) -> Result<(), NativeBacktestBackendError>;

    /// Generate one decision from the prior execution result.
    ///
    /// # Errors
    /// Returns strategy, calendar, or order-construction failures.
    fn generate(
        &mut self,
        previous: Option<&SharedNestedResult>,
    ) -> Result<Box<dyn OrderDecision>, NativeBacktestBackendError>;

    /// Observe one completed outer execution.
    ///
    /// # Errors
    /// Returns post-step hook failures.
    fn post_execute(
        &mut self,
        executions: &SharedNestedResult,
    ) -> Result<(), NativeBacktestBackendError>;

    /// Finalize the strategy after normal executor completion.
    ///
    /// # Errors
    /// Returns finalization failures.
    fn post_upper_level(&mut self) -> Result<(), NativeBacktestBackendError>;
}

/// Production adapter for Qlib's single-order strategy.
pub struct SingleOrderOuterStrategy {
    strategy: SingleOrderStrategy,
    infrastructure: Option<OuterStrategyInfrastructure>,
}

impl SingleOrderOuterStrategy {
    #[must_use]
    pub fn new(strategy: SingleOrderStrategy) -> Self {
        Self {
            strategy,
            infrastructure: None,
        }
    }
}

impl OuterBacktestStrategy for SingleOrderOuterStrategy {
    fn reset(
        &mut self,
        infrastructure: OuterStrategyInfrastructure,
    ) -> Result<(), NativeBacktestBackendError> {
        self.infrastructure = Some(infrastructure);
        Ok(())
    }

    fn generate(
        &mut self,
        _previous: Option<&SharedNestedResult>,
    ) -> Result<Box<dyn OrderDecision>, NativeBacktestBackendError> {
        let infrastructure =
            self.infrastructure
                .as_ref()
                .ok_or_else(|| NativeBacktestBackendError::Strategy {
                    message: "outer strategy infrastructure is not initialized".to_owned(),
                })?;
        let mut orders =
            infrastructure
                .orders
                .lock()
                .map_err(|_| NativeBacktestBackendError::Strategy {
                    message: "outer strategy order factory lock poisoned".to_owned(),
                })?;
        self.strategy
            // The source single-order strategy ignores previous results without reading them.
            .generate_trade_decision(None, &mut **orders, &*infrastructure.calendar)
            .map(|decision| Box::new(decision) as Box<dyn OrderDecision>)
            .map_err(|error| NativeBacktestBackendError::Strategy {
                message: error.to_string(),
            })
    }

    fn post_execute(
        &mut self,
        _executions: &SharedNestedResult,
    ) -> Result<(), NativeBacktestBackendError> {
        Ok(())
    }

    fn post_upper_level(&mut self) -> Result<(), NativeBacktestBackendError> {
        Ok(())
    }
}

/// Replaceable progress lifecycle. The default implementation uses `indicatif`.
pub trait BacktestProgress: Send {
    /// # Errors
    /// Returns invalid totals or progress construction failures.
    fn enter(&mut self, total: i64) -> Result<(), NativeBacktestBackendError>;
    /// # Errors
    /// Returns update failures.
    fn increment(&mut self) -> Result<(), NativeBacktestBackendError>;
    /// # Errors
    /// Returns close failures. Repeated close must be inert.
    fn close(&mut self) -> Result<(), NativeBacktestBackendError>;
}

/// Terminal-aware progress bar equivalent to the source `tqdm` scope.
#[derive(Default)]
pub struct IndicatifBacktestProgress {
    active: Option<ProgressBar>,
}

impl BacktestProgress for IndicatifBacktestProgress {
    fn enter(&mut self, total: i64) -> Result<(), NativeBacktestBackendError> {
        if self.active.is_some() {
            return Err(NativeBacktestBackendError::Progress {
                message: "backtest progress is already active".to_owned(),
            });
        }
        let total = u64::try_from(total).map_err(|_| NativeBacktestBackendError::Progress {
            message: format!("backtest progress total cannot be negative: {total}"),
        })?;
        self.active = Some(ProgressBar::new(total).with_message("backtest loop"));
        Ok(())
    }

    fn increment(&mut self) -> Result<(), NativeBacktestBackendError> {
        let progress =
            self.active
                .as_ref()
                .ok_or_else(|| NativeBacktestBackendError::Progress {
                    message: "backtest progress is not active".to_owned(),
                })?;
        progress.inc(1);
        Ok(())
    }

    fn close(&mut self) -> Result<(), NativeBacktestBackendError> {
        if let Some(progress) = self.active.take() {
            progress.finish();
        }
        Ok(())
    }
}

/// Delayed report source used only after strategy/progress completion.
pub trait BacktestReportSource: Send {
    /// # Errors
    /// Returns account-lock or report-conversion failures.
    fn collect(&mut self) -> Result<BacktestReports, NativeBacktestBackendError>;
}

/// Executor traversal order and shared live accounts used by final report collection.
pub struct SharedAccountReportSource {
    levels: Vec<(String, SharedSaoeAccount)>,
}

impl SharedAccountReportSource {
    #[must_use]
    pub fn new(levels: Vec<(String, SharedSaoeAccount)>) -> Self {
        Self { levels }
    }
}

impl BacktestReportSource for SharedAccountReportSource {
    fn collect(&mut self) -> Result<BacktestReports, NativeBacktestBackendError> {
        collect_shared_backtest_reports(
            self.levels
                .iter()
                .map(|(frequency, account)| (frequency.as_str(), account)),
        )
        .map_err(NativeBacktestBackendError::Report)
    }
}

/// Errors retain the failing component boundary without flattening it into a loop error.
#[derive(Debug, Error)]
pub enum NativeBacktestBackendError {
    #[error(transparent)]
    Calendar(#[from] NestedCalendarError),
    #[error(transparent)]
    Executor(#[from] NestedInnerExecutorError),
    #[error("outer backtest strategy error: {message}")]
    Strategy { message: String },
    #[error("outer backtest progress error: {message}")]
    Progress { message: String },
    #[error(transparent)]
    Report(#[from] BacktestReportError),
}

/// Owns the real outer executor, strategy, level infrastructure, progress and reports.
pub struct NativeBacktestBackend {
    executor: Box<dyn NestedInnerExecutor>,
    strategy: Box<dyn OuterBacktestStrategy>,
    infrastructure: OuterStrategyInfrastructure,
    progress: Box<dyn BacktestProgress>,
    reports: Box<dyn BacktestReportSource>,
}

impl NativeBacktestBackend {
    /// Construct the production backend with an `indicatif` progress bar.
    #[must_use]
    pub fn new(
        executor: Box<dyn NestedInnerExecutor>,
        strategy: Box<dyn OuterBacktestStrategy>,
        infrastructure: OuterStrategyInfrastructure,
        reports: Box<dyn BacktestReportSource>,
    ) -> Self {
        Self::with_progress(
            executor,
            strategy,
            infrastructure,
            Box::<IndicatifBacktestProgress>::default(),
            reports,
        )
    }

    /// Construct with a custom progress plugin for services or tests.
    #[must_use]
    pub fn with_progress(
        executor: Box<dyn NestedInnerExecutor>,
        strategy: Box<dyn OuterBacktestStrategy>,
        infrastructure: OuterStrategyInfrastructure,
        progress: Box<dyn BacktestProgress>,
        reports: Box<dyn BacktestReportSource>,
    ) -> Self {
        Self {
            executor,
            strategy,
            infrastructure,
            progress,
            reports,
        }
    }
}

impl BacktestLoopBackend for NativeBacktestBackend {
    type Error = NativeBacktestBackendError;
    type Reports = BacktestReports;

    fn reset_executor(
        &mut self,
        start: NaiveDateTime,
        end: NaiveDateTime,
    ) -> Result<(), Self::Error> {
        self.executor.reset_window(start, end)?;
        Ok(())
    }

    fn reset_strategy(&mut self) -> Result<(), Self::Error> {
        self.strategy.reset(self.infrastructure.clone())
    }

    fn trade_len(&mut self) -> Result<i64, Self::Error> {
        Ok(self.executor.trade_len()?)
    }

    fn enter_progress(&mut self, total: i64) -> Result<(), Self::Error> {
        self.progress.enter(total)
    }

    fn finished(&mut self) -> Result<bool, Self::Error> {
        Ok(self.executor.finished()?)
    }

    fn generate(
        &mut self,
        previous: Option<&SharedNestedResult>,
    ) -> Result<Box<dyn OrderDecision>, Self::Error> {
        self.strategy.generate(previous)
    }

    fn begin_collection(
        &mut self,
        decision: Box<dyn OrderDecision>,
    ) -> Result<NestedInnerProgress, Self::Error> {
        Ok(self.executor.begin_collect_data(decision, 0)?)
    }

    fn resume_collection(
        &mut self,
        input: NestedExecutorResume,
    ) -> Result<NestedInnerProgress, Self::Error> {
        Ok(self.executor.resume_collect_data(input)?)
    }

    fn close_collection(&mut self) -> Result<(), Self::Error> {
        self.executor.close_collect_data()?;
        Ok(())
    }

    fn post_execute(&mut self, executions: &SharedNestedResult) -> Result<(), Self::Error> {
        self.strategy.post_execute(executions)
    }

    fn update_progress(&mut self) -> Result<(), Self::Error> {
        self.progress.increment()
    }

    fn finalize_strategy(&mut self) -> Result<(), Self::Error> {
        self.strategy.post_upper_level()
    }

    fn close_progress(&mut self) -> Result<(), Self::Error> {
        self.progress.close()
    }

    fn collect_reports(&mut self) -> Result<Self::Reports, Self::Error> {
        self.reports.collect()
    }
}
