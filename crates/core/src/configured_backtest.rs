//! Public configured `backtest` wrapper over native construction and loop execution.

use std::error::Error;

use chrono::NaiveDateTime;
use thiserror::Error;

use crate::{
    BacktestLoopBackend, BacktestLoopError, GetStrategyExecutorError, StrategyExecutorAssembler,
    StrategyExecutorConstructionRequest, StrategyExecutorPair, get_strategy_executor,
    run_backtest_loop,
};

/// Native-time request accepted by the public configured backtest wrapper.
pub type ConfiguredBacktestRequest<'a, AccountInput, StrategyConfig, ExecutorConfig, Value> =
    StrategyExecutorConstructionRequest<
        'a,
        NaiveDateTime,
        AccountInput,
        StrategyConfig,
        ExecutorConfig,
        Value,
    >;

/// Construction and loop failures remain distinguishable at the public boundary.
#[derive(Debug, Error)]
pub enum ConfiguredBacktestError<BackendError: Error + 'static> {
    #[error(transparent)]
    Construction(#[from] GetStrategyExecutorError),
    #[error(transparent)]
    Loop(BacktestLoopError<BackendError>),
}

/// Configure the strategy/executor graph and run the native backtest to reports.
///
/// `make_backend` is an infallible architecture adapter: it may construct
/// [`crate::NativeBacktestBackend`] or another backend from the already resolved
/// pair, but it does not introduce a failure stage absent from the Python wrapper.
///
/// # Errors
///
/// Returns construction failures before the backend adapter is invoked, or the
/// first loop/cleanup/report failure from [`run_backtest_loop`].
pub fn run_configured_backtest<
    AccountInput,
    StrategyConfig,
    ExecutorConfig,
    Value,
    Assembler,
    Backend,
    MakeBackend,
>(
    request: ConfiguredBacktestRequest<'_, AccountInput, StrategyConfig, ExecutorConfig, Value>,
    assembler: &mut Assembler,
    make_backend: MakeBackend,
) -> Result<Backend::Reports, ConfiguredBacktestError<Backend::Error>>
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
    let start_time = request.start_time;
    let end_time = request.end_time;
    let pair = get_strategy_executor(request, assembler)?;
    run_backtest_loop(make_backend(pair), start_time, end_time)
        .map_err(ConfiguredBacktestError::Loop)
}
