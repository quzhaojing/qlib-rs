//! Construction boundary for `qlib.backtest.create_account_instance`.

use std::sync::Arc;

use chrono::NaiveDateTime;
use indexmap::IndexMap;
use thiserror::Error;

use crate::{
    Account, AccountPosition, AccountReportConfig, AccountResetError, BenchmarkReturnSampler,
    InfinitePosition, InitialPositionValue, InitialStockPriceProvider, Position,
};

/// Benchmark used by an empty upstream benchmark configuration.
pub const DEFAULT_ACCOUNT_BENCHMARK: &str = "SH000300";

/// Typed equivalent of the Python numeric-or-dictionary account argument.
#[derive(Clone, Debug, PartialEq)]
pub enum AccountConstructionInput {
    /// A cash-only finite position.
    Cash(f64),
    /// A mutable dictionary whose `cash` member is consumed before construction.
    Dictionary {
        /// `None` represents a missing or already-popped `cash` key.
        cash: Option<f64>,
        /// Remaining insertion-ordered stock entries.
        positions: IndexMap<String, InitialPositionValue>,
    },
    /// An input rejected by the source `isinstance` checks.
    Unsupported,
}

/// Borrowed request passed to a position-construction plugin.
pub struct AccountPositionRequest<'a> {
    pub position_type: &'a str,
    pub initial_cash: f64,
    pub positions: &'a IndexMap<String, InitialPositionValue>,
}

/// Borrowed benchmark query corresponding to `PortfolioMetrics` construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AccountDataRequest<'a> {
    pub benchmark: &'a str,
    pub start_time: Option<NaiveDateTime>,
    pub end_time: Option<NaiveDateTime>,
    pub frequency: &'a str,
}

/// Resolved data plugins retained by the constructed account.
pub struct ResolvedAccountData {
    pub benchmark: Arc<dyn BenchmarkReturnSampler>,
    pub initial_price_provider: Option<Arc<dyn InitialStockPriceProvider>>,
}

/// Failure returned by dynamic construction collaborators.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("{message}")]
pub struct AccountConstructionPluginError {
    pub message: String,
}

/// Replaceable equivalent of the source's dynamic `pos_type` lookup.
pub trait AccountPositionFactory {
    /// Build one owned position implementation.
    ///
    /// # Errors
    ///
    /// Returns a dynamic type lookup or constructor failure.
    fn create(
        &self,
        request: AccountPositionRequest<'_>,
    ) -> Result<Box<dyn AccountPosition>, AccountConstructionPluginError>;
}

/// Resolves benchmark returns and the optional initial-price provider.
pub trait AccountDataResolver {
    /// Resolve all data collaborators needed after position construction.
    ///
    /// # Errors
    ///
    /// Returns a benchmark lookup or provider-construction failure.
    fn resolve(
        &self,
        request: AccountDataRequest<'_>,
    ) -> Result<ResolvedAccountData, AccountConstructionPluginError>;
}

/// Built-in implementation of Qlib's two shipped position classes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeAccountPositionFactory;

impl AccountPositionFactory for NativeAccountPositionFactory {
    fn create(
        &self,
        request: AccountPositionRequest<'_>,
    ) -> Result<Box<dyn AccountPosition>, AccountConstructionPluginError> {
        match request.position_type {
            "Position" => Ok(Box::new(Position::from_initial(
                request.initial_cash,
                request.positions.clone(),
            ))),
            "InfPosition" => Ok(Box::new(InfinitePosition)),
            position_type => Err(AccountConstructionPluginError {
                message: format!("unknown account position type {position_type}"),
            }),
        }
    }
}

/// Typed failure from account argument normalization and construction.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum AccountConstructionError {
    #[error("account must be in (int, float, dict)")]
    UnsupportedInput,
    #[error("account dictionary is missing cash")]
    MissingCash,
    #[error("account position construction failed: {0}")]
    Position(#[source] AccountConstructionPluginError),
    #[error("account data resolution failed: {0}")]
    Data(#[source] AccountConstructionPluginError),
    #[error(transparent)]
    Account(#[from] AccountResetError),
}

/// Construct an account while preserving the source dictionary mutation order.
///
/// The dictionary cash value is removed before any plugin is invoked. A finite
/// position resolves the requested benchmark (or Qlib's default benchmark) and
/// then builds reports; an infinite/custom skip-update position avoids all data
/// resolution just like the source account.
///
/// # Errors
///
/// Returns the first input, position factory, data resolver, or account reset
/// failure, retaining the dictionary's already-consumed cash value.
pub fn create_account_instance(
    start_time: NaiveDateTime,
    end_time: NaiveDateTime,
    benchmark: Option<&str>,
    input: &mut AccountConstructionInput,
    position_type: &str,
    position_factory: &dyn AccountPositionFactory,
    data_resolver: &dyn AccountDataResolver,
) -> Result<Account, AccountConstructionError> {
    let empty_positions = IndexMap::new();
    let (initial_cash, positions) = match input {
        AccountConstructionInput::Cash(cash) => (*cash, &empty_positions),
        AccountConstructionInput::Dictionary { cash, positions } => (
            cash.take().ok_or(AccountConstructionError::MissingCash)?,
            &*positions,
        ),
        AccountConstructionInput::Unsupported => {
            return Err(AccountConstructionError::UnsupportedInput);
        }
    };

    let position = position_factory
        .create(AccountPositionRequest {
            position_type,
            initial_cash,
            positions,
        })
        .map_err(AccountConstructionError::Position)?;

    let metrics_supported = position.portfolio_metrics_supported();
    let data = metrics_supported
        .then(|| {
            data_resolver.resolve(AccountDataRequest {
                benchmark: benchmark.unwrap_or(DEFAULT_ACCOUNT_BENCHMARK),
                start_time: benchmark.map(|_| start_time),
                end_time: benchmark.map(|_| end_time),
                frequency: "day",
            })
        })
        .transpose()
        .map_err(AccountConstructionError::Data)?;

    let report_config = AccountReportConfig {
        benchmark: data.as_ref().map(|data| Arc::clone(&data.benchmark)),
        benchmark_name: benchmark.map(str::to_owned),
        start_time: benchmark.map(|_| start_time),
        end_time: benchmark.map(|_| end_time),
        initial_price_provider: benchmark.and(data.and_then(|data| data.initial_price_provider)),
    };
    Account::try_from_boxed_with_report_config(position, initial_cash, true, "day", report_config)
        .map_err(AccountConstructionError::Account)
}
