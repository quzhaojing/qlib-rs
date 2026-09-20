//! Configured strategy/executor assembly for `qlib.backtest.get_strategy_executor`.

use std::sync::Arc;

use indexmap::IndexMap;
use thiserror::Error;

/// Account and exchange identity shared by the configured strategy and executor.
pub struct StrategyExecutorInfrastructure<Account, Exchange> {
    pub account: Account,
    pub exchange: Exchange,
}

/// One configured component that accepts the shared backtest infrastructure.
pub trait StrategyExecutorInfrastructureTarget<Account, Exchange> {
    /// Install or update the common account/exchange bindings.
    ///
    /// # Errors
    ///
    /// Returns the component's reset or nested graph initialization failure.
    fn reset_common_infrastructure(
        &mut self,
        infrastructure: Arc<StrategyExecutorInfrastructure<Account, Exchange>>,
    ) -> Result<(), StrategyExecutorPluginError>;
}

/// Inputs whose mutation and shallow-copy boundaries are observable in Python.
pub struct StrategyExecutorConstructionRequest<
    'a,
    Time,
    AccountInput,
    StrategyConfig,
    ExecutorConfig,
    Value,
> {
    pub start_time: Time,
    pub end_time: Time,
    pub strategy: StrategyConfig,
    pub executor: ExecutorConfig,
    pub benchmark: Option<String>,
    pub account: &'a mut AccountInput,
    pub exchange_arguments: &'a IndexMap<String, Value>,
    pub position_type: String,
}

/// Result pair returned in the same strategy-then-executor order as Python.
pub struct StrategyExecutorPair<Strategy, Executor> {
    pub strategy: Strategy,
    pub executor: Executor,
}

/// A replaceable dynamic construction registry for every configured component.
pub trait StrategyExecutorAssembler<Time, AccountInput, StrategyConfig, ExecutorConfig, Value> {
    type Account;
    type Exchange;
    type Strategy: StrategyExecutorInfrastructureTarget<Self::Account, Self::Exchange>;
    type Executor: StrategyExecutorInfrastructureTarget<Self::Account, Self::Exchange>;

    /// Construct the trading account before touching exchange arguments.
    ///
    /// # Errors
    ///
    /// Returns account input, position, data, or report construction failures.
    fn create_account(
        &mut self,
        start_time: &Time,
        end_time: &Time,
        benchmark: Option<&str>,
        account: &mut AccountInput,
        position_type: &str,
    ) -> Result<Self::Account, StrategyExecutorPluginError>;

    /// Construct or resolve the exchange from a private shallow argument copy.
    ///
    /// # Errors
    ///
    /// Returns exchange default, configuration, or constructor failures.
    fn create_exchange(
        &mut self,
        arguments: IndexMap<String, Value>,
    ) -> Result<Self::Exchange, StrategyExecutorPluginError>;

    /// Resolve a strategy configuration with the required runtime type check.
    ///
    /// # Errors
    ///
    /// Returns configuration, import, construction, or type-check failures.
    fn resolve_strategy(
        &mut self,
        configuration: StrategyConfig,
    ) -> Result<Self::Strategy, StrategyExecutorPluginError>;

    /// Resolve an executor configuration with the required runtime type check.
    ///
    /// # Errors
    ///
    /// Returns configuration, import, construction, or type-check failures.
    fn resolve_executor(
        &mut self,
        configuration: ExecutorConfig,
    ) -> Result<Self::Executor, StrategyExecutorPluginError>;
}

/// Failure returned by a configured construction or reset collaborator.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("{message}")]
pub struct StrategyExecutorPluginError {
    pub message: String,
}

/// Typed failure preserving the first source stage that did not complete.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum GetStrategyExecutorError {
    #[error("account construction failed: {0}")]
    Account(#[source] StrategyExecutorPluginError),
    #[error("exchange construction failed: {0}")]
    Exchange(#[source] StrategyExecutorPluginError),
    #[error("strategy resolution failed: {0}")]
    StrategyResolution(#[source] StrategyExecutorPluginError),
    #[error("strategy infrastructure reset failed: {0}")]
    StrategyReset(#[source] StrategyExecutorPluginError),
    #[error("executor resolution failed: {0}")]
    ExecutorResolution(#[source] StrategyExecutorPluginError),
    #[error("executor infrastructure reset failed: {0}")]
    ExecutorReset(#[source] StrategyExecutorPluginError),
}

/// Assemble configured strategy/executor components in exact source order.
///
/// Exchange arguments are cloned only after account construction. Missing
/// `start_time`/`end_time` keys receive the caller values, while present null or
/// other values remain untouched. For opaque Python objects, use an `Arc`-like
/// value representation so cloning the map retains nested identity.
///
/// # Errors
///
/// Returns the first account, exchange, resolution, or reset failure. Completed
/// mutations and component resets are not rolled back.
pub fn get_strategy_executor<Time, AccountInput, StrategyConfig, ExecutorConfig, Value, Assembler>(
    request: StrategyExecutorConstructionRequest<
        '_,
        Time,
        AccountInput,
        StrategyConfig,
        ExecutorConfig,
        Value,
    >,
    assembler: &mut Assembler,
) -> Result<StrategyExecutorPair<Assembler::Strategy, Assembler::Executor>, GetStrategyExecutorError>
where
    Time: Clone + Into<Value>,
    Value: Clone,
    Assembler: StrategyExecutorAssembler<Time, AccountInput, StrategyConfig, ExecutorConfig, Value>,
{
    let account = assembler
        .create_account(
            &request.start_time,
            &request.end_time,
            request.benchmark.as_deref(),
            request.account,
            &request.position_type,
        )
        .map_err(GetStrategyExecutorError::Account)?;

    let mut exchange_arguments = request.exchange_arguments.clone();
    exchange_arguments
        .entry("start_time".to_owned())
        .or_insert_with(|| request.start_time.clone().into());
    exchange_arguments
        .entry("end_time".to_owned())
        .or_insert_with(|| request.end_time.clone().into());
    let exchange = assembler
        .create_exchange(exchange_arguments)
        .map_err(GetStrategyExecutorError::Exchange)?;

    let infrastructure = Arc::new(StrategyExecutorInfrastructure { account, exchange });
    let mut strategy = assembler
        .resolve_strategy(request.strategy)
        .map_err(GetStrategyExecutorError::StrategyResolution)?;
    strategy
        .reset_common_infrastructure(Arc::clone(&infrastructure))
        .map_err(GetStrategyExecutorError::StrategyReset)?;
    let mut executor = assembler
        .resolve_executor(request.executor)
        .map_err(GetStrategyExecutorError::ExecutorResolution)?;
    executor
        .reset_common_infrastructure(infrastructure)
        .map_err(GetStrategyExecutorError::ExecutorReset)?;

    Ok(StrategyExecutorPair { strategy, executor })
}
