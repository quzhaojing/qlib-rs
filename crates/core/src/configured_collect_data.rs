//! Lazy public `collect_data` wrapper over configured construction and the native loop.

use std::error::Error;

use chrono::NaiveDateTime;
use thiserror::Error;

use crate::{
    BacktestLoop, BacktestLoopBackend, BacktestLoopError, BacktestLoopEvent,
    ConfiguredBacktestRequest, GetStrategyExecutorError, NestedExecutorResume,
    StrategyExecutorAssembler, StrategyExecutorPair, get_strategy_executor,
};

/// Replaceable publication edge corresponding to Python's caller-owned `return_value` mapping.
pub trait CollectDataReportTarget<Reports> {
    /// Publish the complete report after normal generator exhaustion.
    fn publish(&mut self, reports: Reports);
}

/// Public configured-collection failures retain their construction or loop stage.
#[derive(Debug, Error)]
pub enum ConfiguredCollectDataError<BackendError: Error + 'static> {
    #[error(transparent)]
    Construction(#[from] GetStrategyExecutorError),
    #[error(transparent)]
    Loop(BacktestLoopError<BackendError>),
}

/// Lazy configured generator corresponding to `qlib.backtest.collect_data`.
pub struct ConfiguredCollectData<
    'a,
    AccountInput,
    StrategyConfig,
    ExecutorConfig,
    Value,
    Assembler,
    Backend: BacktestLoopBackend,
    MakeBackend,
> {
    request:
        Option<ConfiguredBacktestRequest<'a, AccountInput, StrategyConfig, ExecutorConfig, Value>>,
    assembler: &'a mut Assembler,
    make_backend: Option<MakeBackend>,
    report_target: Option<&'a mut dyn CollectDataReportTarget<Backend::Reports>>,
    loop_: Option<BacktestLoop<Backend>>,
    failed: bool,
}

impl<'a, AccountInput, StrategyConfig, ExecutorConfig, Value, Assembler, Backend, MakeBackend>
    ConfiguredCollectData<
        'a,
        AccountInput,
        StrategyConfig,
        ExecutorConfig,
        Value,
        Assembler,
        Backend,
        MakeBackend,
    >
where
    Value: Clone + From<NaiveDateTime>,
    Assembler: StrategyExecutorAssembler<
            NaiveDateTime,
            AccountInput,
            StrategyConfig,
            ExecutorConfig,
            Value,
        >,
    Backend: BacktestLoopBackend,
    MakeBackend: FnOnce(StrategyExecutorPair<Assembler::Strategy, Assembler::Executor>) -> Backend,
{
    /// Create an inert generator. Construction starts only on the first valid resume.
    pub fn new(
        request: ConfiguredBacktestRequest<'a, AccountInput, StrategyConfig, ExecutorConfig, Value>,
        assembler: &'a mut Assembler,
        make_backend: MakeBackend,
        report_target: Option<&'a mut dyn CollectDataReportTarget<Backend::Reports>>,
    ) -> Self {
        Self {
            request: Some(request),
            assembler,
            make_backend: Some(make_backend),
            report_target,
            loop_: None,
            failed: false,
        }
    }

    /// Resume the public generator and forward nested control values unchanged.
    ///
    /// # Errors
    ///
    /// Returns an initial protocol error without starting construction, the first configured
    /// construction failure, or the delegated loop/cleanup failure.
    ///
    /// # Panics
    ///
    /// Panics only if this type's private request, adapter, loop, or report-publication
    /// invariants are broken.
    pub fn resume(
        &mut self,
        input: NestedExecutorResume,
    ) -> Result<BacktestLoopEvent, ConfiguredCollectDataError<Backend::Error>> {
        if self.failed {
            return Err(ConfiguredCollectDataError::Loop(BacktestLoopError::Failed));
        }
        if self.loop_.is_none() {
            if input != NestedExecutorResume::Continue {
                return Err(ConfiguredCollectDataError::Loop(
                    BacktestLoopError::InvalidInitialAction,
                ));
            }
            let request = self
                .request
                .take()
                .expect("unstarted generator has request");
            let start_time = request.start_time;
            let end_time = request.end_time;
            let pair = match get_strategy_executor(request, self.assembler) {
                Ok(pair) => pair,
                Err(error) => {
                    self.failed = true;
                    return Err(ConfiguredCollectDataError::Construction(error));
                }
            };
            let make_backend = self
                .make_backend
                .take()
                .expect("unstarted generator has backend adapter");
            self.loop_ = Some(BacktestLoop::new(
                make_backend(pair),
                start_time,
                end_time,
                self.report_target.is_some(),
            ));
        }

        let loop_ = self.loop_.as_mut().expect("started generator has loop");
        match loop_.resume(input) {
            Ok(BacktestLoopEvent::Complete) => {
                if let Some(target) = self.report_target.as_deref_mut() {
                    target.publish(
                        loop_
                            .take_reports()
                            .expect("report-enabled completion publishes reports"),
                    );
                }
                Ok(BacktestLoopEvent::Complete)
            }
            Ok(event) => Ok(event),
            Err(error) => {
                self.failed = true;
                Err(ConfiguredCollectDataError::Loop(error))
            }
        }
    }
}
