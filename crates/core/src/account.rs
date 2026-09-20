//! Order-level account state and accumulated trading metrics.

use std::sync::{Arc, RwLock};

use arrow_array::RecordBatch;
use chrono::{Duration, NaiveDateTime};
use indexmap::IndexMap;
use thiserror::Error;

use crate::{
    AggregateOrderIndicatorsError, BasePriceDataProvider, BasePriceStep, BenchmarkReturnSampler,
    ExchangeQuoteProvider, ExecutionPosition, ExecutionTarget, ExecutionTargetError, Indicator,
    IndicatorConfig, IndicatorError, InfinitePosition, NumpyOrderIndicator, Order, OrderDecision,
    OrderDir, OrderExecution, OrderIndicatorAggregationConfig, PortfolioMetricUpdate,
    PortfolioMetrics, Position, QuoteMethod, TimeRange,
};

/// Accumulated return, cost, and turnover shared by Qlib executor levels.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AccumulatedInfo {
    return_value: f64,
    cost: f64,
    turnover: f64,
}

impl AccumulatedInfo {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            return_value: 0.0,
            cost: 0.0,
            turnover: 0.0,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    pub fn add_return_value(&mut self, value: f64) {
        self.return_value += value;
    }

    pub fn add_cost(&mut self, value: f64) {
        self.cost += value;
    }

    pub fn add_turnover(&mut self, value: f64) {
        self.turnover += value;
    }

    #[must_use]
    pub const fn return_value(self) -> f64 {
        self.return_value
    }

    #[must_use]
    pub const fn cost(self) -> f64 {
        self.cost
    }

    #[must_use]
    pub const fn turnover(self) -> f64 {
        self.turnover
    }
}

/// Diagnostic returned by a replaceable account-position implementation.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("account position error: {message}")]
pub struct AccountPositionError {
    pub message: String,
}

/// One deterministic request for missing initial holding prices.
pub struct InitialStockPriceRequest<'a> {
    pub stocks: &'a [String],
    pub start_time: NaiveDateTime,
    pub end_time: NaiveDateTime,
    pub frequency: &'a str,
    pub disk_cache: bool,
}

/// Diagnostic returned by the initial-price data boundary.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("initial stock price provider error: {message}")]
pub struct InitialStockPriceProviderError {
    pub message: String,
}

/// Replaceable equivalent of Qlib's `D.features` call used by `Position.fill_stock_value`.
pub trait InitialStockPriceProvider: Send + Sync {
    /// Return each requested stock's latest non-null close in the inclusive range.
    ///
    /// # Errors
    ///
    /// Returns a data, schema, type, or plugin transport failure.
    fn latest_close_prices(
        &self,
        request: InitialStockPriceRequest<'_>,
    ) -> Result<IndexMap<String, f64>, InitialStockPriceProviderError>;
}

/// Diagnostic returned by a replaceable bar-market implementation.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("account bar market error: {message}")]
pub struct AccountBarMarketError {
    pub message: String,
}

/// Diagnostic returned by a replaceable account-indicator engine.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("account indicator plugin error: {message}")]
pub struct AccountIndicatorError {
    pub message: String,
}

/// Independently retained indicator engine. Release guards before account updates.
/// Indicator and output callbacks must not reenter this lock during an update.
pub type SharedAccountIndicator = Arc<RwLock<Box<dyn AccountIndicator>>>;

/// Diagnostic returned by a replaceable account-indicator output.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("account indicator output error: {message}")]
pub struct AccountIndicatorOutputError {
    pub message: String,
}

/// Market operations required to mark a position at the end of one trading bar.
pub trait AccountBarMarket: Send + Sync {
    /// Return whether a stock has no trading opportunity in the requested bar.
    ///
    /// # Errors
    ///
    /// Returns a quote or plugin transport failure.
    fn is_suspended(&self, stock: &str, range: TimeRange) -> Result<bool, AccountBarMarketError>;

    /// Return the last valid close in the requested bar.
    ///
    /// # Errors
    ///
    /// Returns an absent close, quote, shape, type, or plugin transport failure.
    fn close(&self, stock: &str, range: TimeRange) -> Result<f64, AccountBarMarketError>;
}

impl AccountBarMarket for ExchangeQuoteProvider {
    fn is_suspended(&self, stock: &str, range: TimeRange) -> Result<bool, AccountBarMarketError> {
        self.stock_is_suspended(stock, range)
            .map_err(|error| AccountBarMarketError {
                message: error.to_string(),
            })
    }

    fn close(&self, stock: &str, range: TimeRange) -> Result<f64, AccountBarMarketError> {
        self.get_aggregated_scalar(stock, range, "$close", QuoteMethod::LastValid)
            .map_err(|error| AccountBarMarketError {
                message: error.to_string(),
            })?
            .ok_or_else(|| AccountBarMarketError {
                message: format!("stock {stock} has no close price in the requested bar"),
            })
    }
}

/// Position operations required by the order-level account core.
pub trait AccountPosition: Send + Sync {
    /// Whether this position can produce finite portfolio metric rows.
    fn portfolio_metrics_supported(&self) -> bool;

    /// Return the cash baseline used for the first portfolio return.
    fn initial_cash(&self) -> f64;

    /// Return current cash plus marked stock value.
    ///
    /// # Errors
    ///
    /// Returns a valuation or plugin transport failure.
    fn total_value(&self) -> Result<f64, AccountPositionError>;

    /// Return current marked stock value.
    ///
    /// # Errors
    ///
    /// Returns a valuation or plugin transport failure.
    fn stock_value(&self) -> Result<f64, AccountPositionError>;

    /// Return immediately available cash, excluding delayed settlement.
    ///
    /// # Errors
    ///
    /// Returns a local state or plugin transport failure.
    fn available_cash(&self) -> Result<f64, AccountPositionError>;

    /// Return the last account value stored for reporting, if present.
    fn stored_account_value(&self) -> Option<f64>;

    /// Return one stored total-asset stock weight.
    ///
    /// # Errors
    ///
    /// Returns a missing stock/weight or plugin transport failure.
    fn stock_weight(&self, stock: &str) -> Result<f64, AccountPositionError>;

    /// Store the account value visible on the current position and future snapshots.
    ///
    /// # Errors
    ///
    /// Returns a local mutation or plugin transport failure.
    fn set_account_value(&mut self, value: f64) -> Result<(), AccountPositionError>;

    /// Recalculate and store every holding weight.
    ///
    /// # Errors
    ///
    /// Returns a valuation, mutation, or plugin transport failure.
    fn update_weights(&mut self) -> Result<(), AccountPositionError>;

    /// Produce an independent owned snapshot of the current plugin state.
    ///
    /// # Errors
    ///
    /// Returns a clone, serialization, or plugin transport failure.
    fn history_snapshot(&self) -> Result<Box<dyn AccountPosition>, AccountPositionError>;

    /// Fill missing initial prices before the first report row.
    ///
    /// The default matches Qlib's no-op `BasePosition.fill_stock_value`; finite `Position`
    /// overrides it with the complete data-provider-backed implementation.
    ///
    /// # Errors
    ///
    /// Returns a data-provider, missing-price, time-range, valuation, or plugin failure.
    fn fill_stock_value(
        &mut self,
        _start_time: NaiveDateTime,
        _frequency: &str,
        _provider: &dyn InitialStockPriceProvider,
    ) -> Result<(), AccountPositionError> {
        Ok(())
    }

    /// Whether every state update should be skipped, as for `InfPosition`.
    fn skip_update(&self) -> bool;

    /// Return the read-only execution view used by an exchange.
    ///
    /// # Errors
    ///
    /// Returns a local state or plugin transport failure.
    fn execution_position(&self) -> Result<&dyn ExecutionPosition, AccountPositionError>;

    /// Return a stable snapshot of the current stock order.
    ///
    /// # Errors
    ///
    /// Returns a local state or plugin transport failure.
    fn stock_ids(&self) -> Result<Vec<String>, AccountPositionError>;

    /// Read the latest price used by account return accounting.
    ///
    /// # Errors
    ///
    /// Returns a missing price or plugin failure.
    fn stock_price(&self, stock: &str) -> Result<f64, AccountPositionError>;

    /// Replace the latest price used for position valuation.
    ///
    /// # Errors
    ///
    /// Returns a missing stock or plugin mutation failure.
    fn update_stock_price(&mut self, stock: &str, price: f64) -> Result<(), AccountPositionError>;

    /// Increment the holding count for every stock.
    ///
    /// # Errors
    ///
    /// Returns a local state or plugin mutation failure.
    fn add_count_all(&mut self, bar: &str) -> Result<(), AccountPositionError>;

    /// Begin one executor settlement transaction.
    ///
    /// # Errors
    ///
    /// Returns a nested-settlement, unsupported-state, or plugin transport failure.
    fn settle_start(&mut self, settlement_type: &str) -> Result<(), AccountPositionError>;

    /// Commit the active executor settlement transaction.
    ///
    /// # Errors
    ///
    /// Returns an unsupported-state, missing delayed-cash, or plugin transport failure.
    fn settle_commit(&mut self) -> Result<(), AccountPositionError>;

    /// Apply a completed execution to the underlying position.
    ///
    /// # Errors
    ///
    /// Returns a position mutation or plugin failure.
    fn update_order(
        &mut self,
        order: &Order,
        trade_value: f64,
        trade_cost: f64,
        trade_price: f64,
    ) -> Result<(), AccountPositionError>;
}

impl AccountPosition for Position {
    fn portfolio_metrics_supported(&self) -> bool {
        true
    }

    fn initial_cash(&self) -> f64 {
        Position::initial_cash(self)
    }

    fn total_value(&self) -> Result<f64, AccountPositionError> {
        Position::calculate_value(self).map_err(|error| AccountPositionError {
            message: error.to_string(),
        })
    }

    fn stock_value(&self) -> Result<f64, AccountPositionError> {
        Position::calculate_stock_value(self).map_err(|error| AccountPositionError {
            message: error.to_string(),
        })
    }

    fn available_cash(&self) -> Result<f64, AccountPositionError> {
        Ok(Position::cash(self, false))
    }

    fn stored_account_value(&self) -> Option<f64> {
        Position::account_value(self)
    }

    fn stock_weight(&self, stock: &str) -> Result<f64, AccountPositionError> {
        Position::stock_weight(self, stock).map_err(|error| AccountPositionError {
            message: error.to_string(),
        })
    }

    fn set_account_value(&mut self, value: f64) -> Result<(), AccountPositionError> {
        Position::set_account_value(self, value);
        Ok(())
    }

    fn update_weights(&mut self) -> Result<(), AccountPositionError> {
        Position::update_weight_all(self).map_err(|error| AccountPositionError {
            message: error.to_string(),
        })
    }

    fn history_snapshot(&self) -> Result<Box<dyn AccountPosition>, AccountPositionError> {
        Ok(Box::new(self.clone()))
    }

    fn fill_stock_value(
        &mut self,
        start_time: NaiveDateTime,
        frequency: &str,
        provider: &dyn InitialStockPriceProvider,
    ) -> Result<(), AccountPositionError> {
        let stocks: Vec<_> = self
            .holdings()
            .iter()
            .filter(|(_, holding)| holding.price().is_none())
            .map(|(stock, _)| stock.clone())
            .collect();
        if stocks.is_empty() {
            return Ok(());
        }
        let price_start_time = start_time
            .checked_sub_signed(Duration::days(30))
            .ok_or_else(|| AccountPositionError {
                message: "initial stock price lookback is outside the supported datetime range"
                    .to_owned(),
            })?;
        let prices = provider
            .latest_close_prices(InitialStockPriceRequest {
                stocks: &stocks,
                start_time: price_start_time,
                end_time: start_time,
                frequency,
                disk_cache: true,
            })
            .map_err(|error| AccountPositionError {
                message: error.to_string(),
            })?;
        Position::fill_missing_stock_prices(self, &prices).map_err(|error| AccountPositionError {
            message: error.to_string(),
        })
    }

    fn skip_update(&self) -> bool {
        false
    }

    fn execution_position(&self) -> Result<&dyn ExecutionPosition, AccountPositionError> {
        Ok(self)
    }

    fn stock_ids(&self) -> Result<Vec<String>, AccountPositionError> {
        Ok(Position::stock_ids(self).map(str::to_owned).collect())
    }

    fn stock_price(&self, stock: &str) -> Result<f64, AccountPositionError> {
        Position::stock_price(self, stock).map_err(|error| AccountPositionError {
            message: error.to_string(),
        })
    }

    fn update_stock_price(&mut self, stock: &str, price: f64) -> Result<(), AccountPositionError> {
        Position::update_stock_price(self, stock, price).map_err(|error| AccountPositionError {
            message: error.to_string(),
        })
    }

    fn add_count_all(&mut self, bar: &str) -> Result<(), AccountPositionError> {
        Position::add_count_all(self, bar);
        Ok(())
    }

    fn settle_start(&mut self, settlement_type: &str) -> Result<(), AccountPositionError> {
        Position::settle_start(self, settlement_type).map_err(|error| AccountPositionError {
            message: error.to_string(),
        })
    }

    fn settle_commit(&mut self) -> Result<(), AccountPositionError> {
        Position::settle_commit(self).map_err(|error| AccountPositionError {
            message: error.to_string(),
        })
    }

    fn update_order(
        &mut self,
        order: &Order,
        trade_value: f64,
        trade_cost: f64,
        trade_price: f64,
    ) -> Result<(), AccountPositionError> {
        Position::update_order(self, order, trade_value, trade_cost, trade_price).map_err(|error| {
            AccountPositionError {
                message: error.to_string(),
            }
        })
    }
}

impl AccountPosition for InfinitePosition {
    fn portfolio_metrics_supported(&self) -> bool {
        false
    }

    fn initial_cash(&self) -> f64 {
        f64::INFINITY
    }

    fn total_value(&self) -> Result<f64, AccountPositionError> {
        InfinitePosition::calculate_value(*self).map_err(|error| AccountPositionError {
            message: error.to_string(),
        })
    }

    fn stock_value(&self) -> Result<f64, AccountPositionError> {
        Ok(InfinitePosition::calculate_stock_value(*self))
    }

    fn available_cash(&self) -> Result<f64, AccountPositionError> {
        Ok(f64::INFINITY)
    }

    fn stored_account_value(&self) -> Option<f64> {
        None
    }

    fn stock_weight(&self, _stock: &str) -> Result<f64, AccountPositionError> {
        Err(AccountPositionError {
            message: "infinite position does not support stock weight snapshot".to_owned(),
        })
    }

    fn set_account_value(&mut self, _value: f64) -> Result<(), AccountPositionError> {
        Err(AccountPositionError {
            message: "infinite position does not support storing account value".to_owned(),
        })
    }

    fn update_weights(&mut self) -> Result<(), AccountPositionError> {
        InfinitePosition::update_weight_all(*self).map_err(|error| AccountPositionError {
            message: error.to_string(),
        })
    }

    fn history_snapshot(&self) -> Result<Box<dyn AccountPosition>, AccountPositionError> {
        Ok(Box::new(*self))
    }

    fn skip_update(&self) -> bool {
        true
    }

    fn execution_position(&self) -> Result<&dyn ExecutionPosition, AccountPositionError> {
        Ok(self)
    }

    fn stock_ids(&self) -> Result<Vec<String>, AccountPositionError> {
        InfinitePosition::stock_list(*self).map_err(|error| AccountPositionError {
            message: error.to_string(),
        })
    }

    fn stock_price(&self, stock: &str) -> Result<f64, AccountPositionError> {
        Ok(InfinitePosition::stock_price(*self, stock))
    }

    fn update_stock_price(&mut self, stock: &str, price: f64) -> Result<(), AccountPositionError> {
        InfinitePosition::update_stock_price(*self, stock, price);
        Ok(())
    }

    fn add_count_all(&mut self, bar: &str) -> Result<(), AccountPositionError> {
        InfinitePosition::add_count_all(*self, bar).map_err(|error| AccountPositionError {
            message: error.to_string(),
        })
    }

    fn settle_start(&mut self, _settlement_type: &str) -> Result<(), AccountPositionError> {
        Ok(())
    }

    fn settle_commit(&mut self) -> Result<(), AccountPositionError> {
        Ok(())
    }

    fn update_order(
        &mut self,
        _order: &Order,
        _trade_value: f64,
        _trade_cost: f64,
        _trade_price: f64,
    ) -> Result<(), AccountPositionError> {
        Ok(())
    }
}

/// Typed failures from order-level account accounting.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum AccountError {
    #[error("historical positions lock poisoned")]
    HistoryPoisoned,
    #[error("trade_info is necessary in atomic executor")]
    MissingAtomicTradeInfo,
    #[error("inner_order_indicators is necessary in un-atomic executor")]
    MissingInnerOrderIndicators,
    #[error("trade price cannot be zero")]
    ZeroTradePrice,
    #[error(transparent)]
    Position(#[from] AccountPositionError),
    #[error(transparent)]
    Market(#[from] AccountBarMarketError),
    #[error("portfolio metrics are disabled for this account")]
    PortfolioMetricsDisabled,
    #[error("generate_portfolio_metrics should be True if you want to generate portfolio_metrics")]
    PortfolioReportDisabled,
    #[error("portfolio metrics error: {message}")]
    PortfolioMetrics { message: String },
    #[error("previous account value cannot be zero")]
    ZeroPreviousAccountValue,
    #[error(transparent)]
    Indicator(#[from] AccountIndicatorError),
    #[error(transparent)]
    IndicatorOutput(#[from] AccountIndicatorOutputError),
}

/// Nested-executor inputs required by the outer indicator aggregation pipeline.
pub struct NestedAccountIndicatorUpdate<'a> {
    pub inner: &'a [crate::SharedOrderIndicator<NumpyOrderIndicator>],
    pub outer_decision: &'a dyn OrderDecision,
    pub steps: &'a [BasePriceStep<'a>],
    pub provider: &'a dyn BasePriceDataProvider,
    pub config: OrderIndicatorAggregationConfig,
}

/// Native nested inputs retaining live decisions until the report stage that reads them.
pub struct LiveNestedAccountIndicatorUpdate<'a> {
    pub inner: &'a [crate::SharedOrderIndicator<NumpyOrderIndicator>],
    pub outer_decision: &'a crate::decision_update::LiveDecisionHandle,
    pub steps: &'a [crate::base_price::LiveBasePriceStep],
    pub provider: &'a dyn BasePriceDataProvider,
    pub config: OrderIndicatorAggregationConfig,
}

/// Mutually exclusive atomic and nested account-indicator update modes.
pub enum AccountIndicatorMode<'a> {
    LiveNested(LiveNestedAccountIndicatorUpdate<'a>),
    Atomic(&'a [OrderExecution<'a>]),
    SharedAtomic(&'a crate::shared_executor_lifecycle::SharedAtomicResult),
    Nested(NestedAccountIndicatorUpdate<'a>),
}

/// One bar's complete indicator update request.
pub struct AccountIndicatorUpdate<'a> {
    pub trade_start_time: NaiveDateTime,
    pub mode: AccountIndicatorMode<'a>,
    pub calculation: IndicatorConfig,
    pub show_indicator: bool,
}

/// Inputs that distinguish Python's missing bar-end values from valid empty slices.
pub enum AccountBarEndMode<'a> {
    LiveNested(Option<LiveNestedAccountIndicatorUpdate<'a>>),
    Atomic(Option<&'a [OrderExecution<'a>]>),
    SharedAtomic(&'a crate::shared_executor_lifecycle::SharedAtomicResult),
    Nested(Option<NestedAccountIndicatorUpdate<'a>>),
}

/// One complete account bar-end request.
pub struct AccountBarEndUpdate<'a> {
    pub trade_start_time: NaiveDateTime,
    pub trade_end_time: NaiveDateTime,
    pub market: &'a dyn AccountBarMarket,
    pub mode: AccountBarEndMode<'a>,
    pub calculation: IndicatorConfig,
    pub show_indicator: bool,
}

/// Object-safe indicator engine used by account orchestration.
pub trait AccountIndicator: Send + Sync {
    /// Aggregate native nested inputs without adapting decisions to owned order snapshots.
    ///
    /// # Errors
    /// Unsupported plugins reject the capability; native backends propagate stage failures.
    fn update_live_nested(
        &mut self,
        _update: LiveNestedAccountIndicatorUpdate<'_>,
    ) -> Result<(), AccountIndicatorError> {
        Err(AccountIndicatorError {
            message: "indicator backend does not support live nested decisions".to_owned(),
        })
    }

    /// Clear only current metrics.
    ///
    /// # Errors
    ///
    /// Returns a local backend or plugin transport failure.
    fn reset(&mut self) -> Result<(), AccountIndicatorError>;

    /// Populate innermost metrics from executions.
    ///
    /// # Errors
    ///
    /// Returns a local backend or plugin transport failure.
    fn update_atomic(
        &mut self,
        executions: &[OrderExecution<'_>],
    ) -> Result<(), AccountIndicatorError>;

    /// Read live shared executions after the account has reset this indicator.
    ///
    /// # Errors
    /// Backends must explicitly support shared inputs; the default rejects them without copying
    /// orders into detached values or silently changing plugin semantics.
    fn update_shared_atomic(
        &mut self,
        _executions: &crate::shared_executor_lifecycle::SharedAtomicResult,
    ) -> Result<(), AccountIndicatorError> {
        Err(AccountIndicatorError {
            message: "indicator backend does not support shared atomic executions".to_owned(),
        })
    }

    /// Aggregate already-computed inner metrics.
    ///
    /// # Errors
    ///
    /// Returns a metric, market-data, or plugin transport failure.
    fn update_nested(
        &mut self,
        update: NestedAccountIndicatorUpdate<'_>,
    ) -> Result<(), AccountIndicatorError>;

    /// Calculate bar-level indicator summaries.
    ///
    /// # Errors
    ///
    /// Returns a missing metric, alignment, configuration, or plugin transport failure.
    fn calculate(&mut self, config: IndicatorConfig) -> Result<(), AccountIndicatorError>;

    /// Store current metrics under one timestamp.
    ///
    /// # Errors
    ///
    /// Returns a local backend or plugin transport failure.
    fn record(&mut self, trade_start_time: NaiveDateTime) -> Result<(), AccountIndicatorError>;

    /// Current bar-level summaries in stable order.
    fn trade_indicator(&self) -> &crate::SharedTradeIndicator;

    /// Produce an independent numerical order-indicator snapshot.
    ///
    /// # Errors
    ///
    /// Returns a clone, serialization, or plugin transport failure.
    fn order_indicator_snapshot(&self) -> Result<NumpyOrderIndicator, AccountIndicatorError>;

    /// Retain the current raw order-store identity for nested execution.
    /// Reset/replacement must not rebind previously returned handles. Do not clone store values.
    ///
    /// # Errors
    /// Returns a plugin transport or handle-acquisition failure.
    fn order_indicator_handle(
        &self,
    ) -> Result<crate::SharedOrderIndicator<NumpyOrderIndicator>, AccountIndicatorError>;

    /// Recorded bar-level summaries for one timestamp.
    fn recorded_trade_indicator(&self, time: NaiveDateTime)
    -> Option<&crate::SharedTradeIndicator>;

    /// Export the complete recorded table without replacing the live indicator.
    ///
    /// # Errors
    /// Returns a backend or plugin transport failure.
    fn trade_indicator_report(&self) -> Result<crate::TradeIndicatorReport, AccountIndicatorError>;
}

impl AccountIndicator for Indicator<NumpyOrderIndicator> {
    fn update_live_nested(
        &mut self,
        update: LiveNestedAccountIndicatorUpdate<'_>,
    ) -> Result<(), AccountIndicatorError> {
        self.aggregate_live_order_indicators(
            update.inner,
            update.outer_decision,
            update.steps,
            update.provider,
            update.config,
        )
        .map_err(|error| AccountIndicatorError {
            message: error.to_string(),
        })
    }

    fn update_shared_atomic(
        &mut self,
        executions: &crate::shared_executor_lifecycle::SharedAtomicResult,
    ) -> Result<(), AccountIndicatorError> {
        self.update_shared_order_indicators(executions)
            .map_err(|error| AccountIndicatorError {
                message: error.to_string(),
            })
    }
    fn reset(&mut self) -> Result<(), AccountIndicatorError> {
        Indicator::reset(self);
        Ok(())
    }

    fn update_atomic(
        &mut self,
        executions: &[OrderExecution<'_>],
    ) -> Result<(), AccountIndicatorError> {
        self.update_order_indicators(executions)
            .map_err(|error| AccountIndicatorError {
                message: error.to_string(),
            })
    }

    fn update_nested(
        &mut self,
        update: NestedAccountIndicatorUpdate<'_>,
    ) -> Result<(), AccountIndicatorError> {
        self.aggregate_shared_order_indicators(
            update.inner,
            update.outer_decision.orders(),
            update.steps,
            update.provider,
            update.config,
        )
        .map_err(
            |error: AggregateOrderIndicatorsError| AccountIndicatorError {
                message: error.to_string(),
            },
        )
    }

    fn calculate(&mut self, config: IndicatorConfig) -> Result<(), AccountIndicatorError> {
        self.calculate_trade_indicators(config)
            .map_err(|error: IndicatorError| AccountIndicatorError {
                message: error.to_string(),
            })
    }

    fn record(&mut self, trade_start_time: NaiveDateTime) -> Result<(), AccountIndicatorError> {
        Indicator::record(self, trade_start_time);
        Ok(())
    }

    fn trade_indicator(&self) -> &crate::SharedTradeIndicator {
        Indicator::trade_indicator(self)
    }

    fn order_indicator_snapshot(&self) -> Result<NumpyOrderIndicator, AccountIndicatorError> {
        self.order_indicator()
            .read()
            .map(|store| store.clone())
            .map_err(|_| AccountIndicatorError {
                message: "order indicator store lock poisoned".to_owned(),
            })
    }

    fn order_indicator_handle(
        &self,
    ) -> Result<crate::SharedOrderIndicator<NumpyOrderIndicator>, AccountIndicatorError> {
        Ok(self.order_indicator().clone())
    }

    fn recorded_trade_indicator(
        &self,
        time: NaiveDateTime,
    ) -> Option<&crate::SharedTradeIndicator> {
        self.trade_indicator_history().get(&time)
    }

    fn trade_indicator_report(&self) -> Result<crate::TradeIndicatorReport, AccountIndicatorError> {
        Indicator::trade_indicator_report(self).map_err(|error| AccountIndicatorError {
            message: error.to_string(),
        })
    }
}

/// Replaceable presentation boundary for Python's optional indicator line.
pub trait AccountIndicatorOutput: Send + Sync {
    /// Emit one formatted indicator summary.
    ///
    /// # Errors
    ///
    /// Returns a formatting, stream, or plugin transport failure.
    fn write(
        &self,
        frequency: &str,
        trade_start_time: NaiveDateTime,
        values: &IndexMap<String, f64>,
    ) -> Result<(), AccountIndicatorOutputError>;
}

/// Default stdout presentation compatible with Qlib's `show_indicator` option.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StdoutAccountIndicatorOutput;

impl AccountIndicatorOutput for StdoutAccountIndicatorOutput {
    fn write(
        &self,
        frequency: &str,
        trade_start_time: NaiveDateTime,
        values: &IndexMap<String, f64>,
    ) -> Result<(), AccountIndicatorOutputError> {
        println!(
            "{}",
            format_account_indicator_output(frequency, trade_start_time, values)?
        );
        Ok(())
    }
}

/// Format the optional Qlib-compatible indicator status line.
///
/// # Errors
///
/// Returns a missing-metric failure when calculation did not produce `ffr`, `pa`, or `pos`.
pub fn format_account_indicator_output(
    frequency: &str,
    trade_start_time: NaiveDateTime,
    values: &IndexMap<String, f64>,
) -> Result<String, AccountIndicatorOutputError> {
    let required = |name: &str| {
        values
            .get(name)
            .copied()
            .ok_or_else(|| AccountIndicatorOutputError {
                message: format!("indicator metric not found: {name}"),
            })
    };
    Ok(format!(
        "[Indicator({frequency}) {}]: FFR: {}, PA: {}, POS: {}",
        trade_start_time.format("%Y-%m-%d %H:%M:%S"),
        python_float(required("ffr")?),
        python_float(required("pa")?),
        python_float(required("pos")?),
    ))
}

fn python_float(value: f64) -> String {
    if value.is_nan() {
        "nan".to_owned()
    } else if value == f64::INFINITY {
        "inf".to_owned()
    } else if value == f64::NEG_INFINITY {
        "-inf".to_owned()
    } else {
        let value = value.to_string();
        if value.contains(['.', 'e', 'E']) {
            value
        } else {
            format!("{value}.0")
        }
    }
}

/// One independently owned historical position and its bar-end account value.
pub struct HistoricalPosition {
    account_value: f64,
    position: Box<dyn AccountPosition>,
}

/// Independently retained history object. Release read/write guards before invoking
/// account methods that update this same history. Replacing the handle preserves old reports.
pub type HistoricalPositions = Arc<RwLock<IndexMap<NaiveDateTime, HistoricalPosition>>>;

/// Owned subset of Python's benchmark configuration needed by account reset.
#[derive(Clone, Default)]
pub struct AccountReportConfig {
    pub benchmark: Option<Arc<dyn BenchmarkReturnSampler>>,
    pub benchmark_name: Option<String>,
    pub start_time: Option<NaiveDateTime>,
    pub end_time: Option<NaiveDateTime>,
    pub initial_price_provider: Option<Arc<dyn InitialStockPriceProvider>>,
}

/// Diagnostic returned by a replaceable account report factory.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("{message}")]
pub struct AccountReportFactoryError {
    pub message: String,
}

/// Plugin boundary for the two constructors monkeypatchable in Python's reset path.
pub trait AccountReportFactory: Send + Sync {
    /// Build an empty portfolio ledger.
    ///
    /// # Errors
    ///
    /// Returns a benchmark-configuration or plugin construction failure.
    fn portfolio_metrics(
        &self,
        frequency: &str,
        config: &AccountReportConfig,
    ) -> Result<PortfolioMetrics, AccountReportFactoryError>;

    /// Build an empty order/trade indicator engine.
    ///
    /// # Errors
    ///
    /// Returns a plugin construction failure.
    fn indicator(&self) -> Result<Box<dyn AccountIndicator>, AccountReportFactoryError>;
}

/// Built-in report constructors used by ordinary reset calls.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DefaultAccountReportFactory;

impl AccountReportFactory for DefaultAccountReportFactory {
    fn portfolio_metrics(
        &self,
        frequency: &str,
        config: &AccountReportConfig,
    ) -> Result<PortfolioMetrics, AccountReportFactoryError> {
        Ok(PortfolioMetrics::new(frequency, config.benchmark.clone()))
    }

    fn indicator(&self) -> Result<Box<dyn AccountIndicator>, AccountReportFactoryError> {
        Ok(Box::new(Indicator::new()))
    }
}

/// Optional account metadata updates applied before rebuilding report objects.
#[derive(Clone, Default)]
pub struct AccountResetUpdate {
    pub frequency: Option<String>,
    pub report_config: Option<AccountReportConfig>,
    pub portfolio_metrics_enabled: Option<bool>,
}

/// Typed failure from the complete account report-reset state machine.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum AccountResetError {
    #[error(transparent)]
    PortfolioMetricsFactory(AccountReportFactoryError),
    #[error("initial stock price provider is required when reset start_time is set")]
    MissingInitialPriceProvider,
    #[error(transparent)]
    Position(#[from] AccountPositionError),
    #[error(transparent)]
    IndicatorFactory(AccountReportFactoryError),
}

/// Frozen Arrow table plus the original independently retained historical-position mapping.
pub struct AccountPortfolioReport {
    pub metrics: RecordBatch,
    pub positions: HistoricalPositions,
}

impl HistoricalPosition {
    #[must_use]
    pub const fn account_value(&self) -> f64 {
        self.account_value
    }

    #[must_use]
    pub fn position(&self) -> &dyn AccountPosition {
        self.position.as_ref()
    }
}

/// Qlib account core that owns one replaceable position and shared accumulated metrics.
pub struct Account {
    current_position: Box<dyn AccountPosition>,
    accumulated_info: AccumulatedInfo,
    portfolio_metrics_enabled: bool,
    frequency: String,
    report_config: AccountReportConfig,
    initial_cash: f64,
    portfolio_metrics: Option<PortfolioMetrics>,
    historical_positions: HistoricalPositions,
    indicator: SharedAccountIndicator,
    indicator_output: Box<dyn AccountIndicatorOutput>,
}

impl Account {
    #[must_use]
    pub fn new(
        current_position: impl AccountPosition + 'static,
        portfolio_metrics_enabled: bool,
    ) -> Self {
        Self::from_boxed(Box::new(current_position), portfolio_metrics_enabled)
    }

    #[must_use]
    pub fn from_boxed(
        current_position: Box<dyn AccountPosition>,
        portfolio_metrics_enabled: bool,
    ) -> Self {
        Self::from_boxed_with_frequency(current_position, portfolio_metrics_enabled, "day")
    }

    /// Construct an account with the normalized bar name used for holding counts.
    #[must_use]
    pub fn with_frequency(
        current_position: impl AccountPosition + 'static,
        portfolio_metrics_enabled: bool,
        frequency: impl Into<String>,
    ) -> Self {
        Self::from_boxed_with_frequency(
            Box::new(current_position),
            portfolio_metrics_enabled,
            frequency,
        )
    }

    /// Construct an account from a boxed position plugin and explicit bar name.
    #[must_use]
    pub fn from_boxed_with_frequency(
        current_position: Box<dyn AccountPosition>,
        portfolio_metrics_enabled: bool,
        frequency: impl Into<String>,
    ) -> Self {
        Self::from_boxed_with_portfolio_metrics(
            current_position,
            portfolio_metrics_enabled,
            frequency,
            None,
        )
    }

    /// Construct an account with an optional benchmark-return plugin.
    #[must_use]
    pub fn with_portfolio_metrics(
        current_position: impl AccountPosition + 'static,
        portfolio_metrics_enabled: bool,
        frequency: impl Into<String>,
        benchmark: Option<Arc<dyn BenchmarkReturnSampler>>,
    ) -> Self {
        Self::from_boxed_with_portfolio_metrics(
            Box::new(current_position),
            portfolio_metrics_enabled,
            frequency,
            benchmark,
        )
    }

    /// Construct an account from boxed position and benchmark plugins.
    #[must_use]
    pub fn from_boxed_with_portfolio_metrics(
        current_position: Box<dyn AccountPosition>,
        portfolio_metrics_enabled: bool,
        frequency: impl Into<String>,
        benchmark: Option<Arc<dyn BenchmarkReturnSampler>>,
    ) -> Self {
        let initial_cash = current_position.initial_cash();
        let frequency = frequency.into();
        let enabled = portfolio_metrics_enabled && current_position.portfolio_metrics_supported();
        let report_config = AccountReportConfig {
            benchmark,
            ..AccountReportConfig::default()
        };
        Self {
            current_position,
            accumulated_info: AccumulatedInfo::new(),
            portfolio_metrics_enabled,
            portfolio_metrics: enabled
                .then(|| PortfolioMetrics::new(frequency.clone(), report_config.benchmark.clone())),
            frequency,
            report_config,
            initial_cash,
            historical_positions: Arc::new(RwLock::new(IndexMap::new())),
            indicator: Arc::new(RwLock::new(Box::new(Indicator::new()))),
            indicator_output: Box::new(StdoutAccountIndicatorOutput),
        }
    }

    /// Construct from already resolved plugins while retaining the caller's
    /// explicit initial cash independently of the position implementation.
    ///
    /// # Errors
    ///
    /// Returns the first report, initial-price, position, or indicator error.
    pub fn try_from_boxed_with_report_config(
        current_position: Box<dyn AccountPosition>,
        initial_cash: f64,
        portfolio_metrics_enabled: bool,
        frequency: impl Into<String>,
        report_config: AccountReportConfig,
    ) -> Result<Self, AccountResetError> {
        let mut account = Self {
            current_position,
            accumulated_info: AccumulatedInfo::new(),
            portfolio_metrics_enabled,
            frequency: frequency.into(),
            report_config: AccountReportConfig::default(),
            initial_cash,
            portfolio_metrics: None,
            historical_positions: Arc::new(RwLock::new(IndexMap::new())),
            indicator: Arc::new(RwLock::new(Box::new(Indicator::new()))),
            indicator_output: Box::new(StdoutAccountIndicatorOutput),
        };
        account.reset(AccountResetUpdate {
            frequency: None,
            report_config: Some(report_config),
            portfolio_metrics_enabled: None,
        })?;
        Ok(account)
    }

    #[must_use]
    pub const fn initial_cash(&self) -> f64 {
        self.initial_cash
    }

    #[must_use]
    pub const fn portfolio_metrics(&self) -> Option<&PortfolioMetrics> {
        self.portfolio_metrics.as_ref()
    }

    #[must_use]
    pub const fn portfolio_metrics_mut(&mut self) -> Option<&mut PortfolioMetrics> {
        self.portfolio_metrics.as_mut()
    }

    #[must_use]
    pub const fn historical_positions(&self) -> &HistoricalPositions {
        &self.historical_positions
    }

    /// Replace the history object without clearing references held by older reports.
    /// This is the history replacement stage used by report reset, not a full account reset.
    pub fn replace_historical_positions(
        &mut self,
        positions: HistoricalPositions,
    ) -> HistoricalPositions {
        std::mem::replace(&mut self.historical_positions, positions)
    }

    #[must_use]
    pub const fn indicator(&self) -> &SharedAccountIndicator {
        &self.indicator
    }

    /// Return an independent numerical order-indicator snapshot.
    ///
    /// # Errors
    ///
    /// Returns a clone, serialization, or indicator-plugin transport failure.
    pub fn order_indicator_snapshot(&self) -> Result<NumpyOrderIndicator, AccountError> {
        self.indicator
            .read()
            .map_err(|_| AccountIndicatorError {
                message: "account indicator lock poisoned".to_owned(),
            })?
            .order_indicator_snapshot()
            .map_err(AccountError::Indicator)
    }

    /// Retain the current raw store without retaining an account/indicator lock.
    /// The store remains the same object after engine reset, replacement or account drop.
    /// Publishing its identity does not require reading the store, even if that store is poisoned.
    ///
    /// # Errors
    /// Returns an indicator-engine lock or plugin handle-acquisition failure.
    pub fn order_indicator_handle(
        &self,
    ) -> Result<crate::SharedOrderIndicator<NumpyOrderIndicator>, AccountError> {
        self.indicator
            .read()
            .map_err(|_| AccountIndicatorError {
                message: "account indicator lock poisoned".to_owned(),
            })?
            .order_indicator_handle()
            .map_err(AccountError::Indicator)
    }

    /// Replace the indicator engine without changing objects retained by old reports.
    pub fn replace_indicator(
        &mut self,
        indicator: Box<dyn AccountIndicator>,
    ) -> SharedAccountIndicator {
        std::mem::replace(&mut self.indicator, Arc::new(RwLock::new(indicator)))
    }

    /// Replace the optional display output and return the previous plugin.
    pub fn replace_indicator_output(
        &mut self,
        output: Box<dyn AccountIndicatorOutput>,
    ) -> Box<dyn AccountIndicatorOutput> {
        std::mem::replace(&mut self.indicator_output, output)
    }

    #[must_use]
    pub fn frequency(&self) -> &str {
        &self.frequency
    }

    #[must_use]
    pub const fn report_config(&self) -> &AccountReportConfig {
        &self.report_config
    }

    /// Apply optional metadata and rebuild report objects with the built-in factory.
    ///
    /// # Errors
    ///
    /// Returns the first portfolio construction, initial-price fill, or indicator failure.
    pub fn reset(&mut self, update: AccountResetUpdate) -> Result<(), AccountResetError> {
        self.reset_with_factory(update, &DefaultAccountReportFactory)
    }

    /// Apply Python's non-transactional reset order using replaceable report constructors.
    /// State reached before an error is deliberately retained, matching upstream Qlib.
    ///
    /// # Errors
    ///
    /// Returns the first portfolio construction, initial-price fill, or indicator failure.
    pub fn reset_with_factory(
        &mut self,
        update: AccountResetUpdate,
        factory: &dyn AccountReportFactory,
    ) -> Result<(), AccountResetError> {
        if let Some(frequency) = update.frequency {
            self.frequency = frequency;
        }
        if let Some(report_config) = update.report_config {
            self.report_config = report_config;
        }
        if let Some(enabled) = update.portfolio_metrics_enabled {
            self.portfolio_metrics_enabled = enabled;
        }

        if self.is_portfolio_metrics_enabled() {
            let metrics = factory
                .portfolio_metrics(&self.frequency, &self.report_config)
                .map_err(AccountResetError::PortfolioMetricsFactory)?;
            self.portfolio_metrics = Some(metrics);
            self.historical_positions = Arc::new(RwLock::new(IndexMap::new()));
            if let Some(start_time) = self.report_config.start_time {
                let provider = self
                    .report_config
                    .initial_price_provider
                    .clone()
                    .ok_or(AccountResetError::MissingInitialPriceProvider)?;
                self.current_position.fill_stock_value(
                    start_time,
                    &self.frequency,
                    provider.as_ref(),
                )?;
            }
        }

        let indicator = factory
            .indicator()
            .map_err(AccountResetError::IndicatorFactory)?;
        self.indicator = Arc::new(RwLock::new(indicator));
        Ok(())
    }

    #[must_use]
    pub const fn accumulated_info(&self) -> &AccumulatedInfo {
        &self.accumulated_info
    }

    #[must_use]
    pub const fn accumulated_info_mut(&mut self) -> &mut AccumulatedInfo {
        &mut self.accumulated_info
    }

    #[must_use]
    pub fn current_position(&self) -> &dyn AccountPosition {
        self.current_position.as_ref()
    }

    #[must_use]
    pub fn current_position_mut(&mut self) -> &mut dyn AccountPosition {
        self.current_position.as_mut()
    }

    #[must_use]
    pub fn is_portfolio_metrics_enabled(&self) -> bool {
        self.portfolio_metrics_enabled && !self.current_position.skip_update()
    }

    /// Export portfolio metrics and the aligned historical-position mapping.
    ///
    /// # Errors
    ///
    /// Returns Python's disabled-report failure or a typed internal invariant failure when an
    /// enabled custom position did not create a portfolio ledger.
    pub fn portfolio_report(&self) -> Result<AccountPortfolioReport, AccountError> {
        if !self.is_portfolio_metrics_enabled() {
            return Err(AccountError::PortfolioReportDisabled);
        }
        let metrics = self
            .portfolio_metrics
            .as_ref()
            .ok_or(AccountError::PortfolioMetricsDisabled)?
            .to_record_batch();
        Ok(AccountPortfolioReport {
            metrics,
            positions: self.historical_positions.clone(),
        })
    }

    /// Append one portfolio row using Qlib's accumulated-total delta accounting.
    ///
    /// # Errors
    ///
    /// Returns before ledger mutation if metrics are disabled, valuation fails, the previous
    /// account value is zero, or benchmark sampling fails.
    pub fn update_portfolio_metrics(
        &mut self,
        trade_start_time: NaiveDateTime,
        trade_end_time: NaiveDateTime,
    ) -> Result<(), AccountError> {
        let Self {
            current_position,
            accumulated_info,
            initial_cash,
            portfolio_metrics,
            ..
        } = self;
        let metrics = portfolio_metrics
            .as_mut()
            .ok_or(AccountError::PortfolioMetricsDisabled)?;
        let (last_account_value, last_total_cost, last_total_turnover) =
            if let Ok(latest) = metrics.latest_record() {
                (
                    latest.account_value,
                    latest.total_cost,
                    latest.total_turnover,
                )
            } else {
                (*initial_cash, 0.0, 0.0)
            };

        let account_value = current_position.total_value()?;
        let stock_value = current_position.stock_value()?;
        let earning = account_value - last_account_value;
        let cost = accumulated_info.cost - last_total_cost;
        let turnover = accumulated_info.turnover - last_total_turnover;
        let cash = current_position.available_cash()?;
        if last_account_value == 0.0 {
            return Err(AccountError::ZeroPreviousAccountValue);
        }

        metrics
            .update_record(PortfolioMetricUpdate {
                trade_start_time: Some(trade_start_time),
                trade_end_time: Some(trade_end_time),
                account_value: Some(account_value),
                cash: Some(cash),
                return_rate: Some((earning + cost) / last_account_value),
                total_turnover: Some(accumulated_info.turnover),
                turnover_rate: Some(turnover / last_account_value),
                total_cost: Some(accumulated_info.cost),
                cost_rate: Some(cost / last_account_value),
                stock_value: Some(stock_value),
                bench_value: None,
            })
            .map_err(|error| AccountError::PortfolioMetrics {
                message: error.to_string(),
            })
    }

    /// Refresh current weights and store an independent position snapshot for one bar.
    ///
    /// # Errors
    ///
    /// Returns at the first valuation, mutation, weight, or snapshot-plugin failure. State
    /// reached before the failure is retained, while the history map changes only after a
    /// successful snapshot.
    pub fn update_historical_positions(
        &mut self,
        trade_start_time: NaiveDateTime,
    ) -> Result<(), AccountError> {
        let account_value = self.current_position.total_value()?;
        self.current_position.set_account_value(account_value)?;
        self.current_position.update_weights()?;
        let position = self.current_position.history_snapshot()?;
        self.historical_positions
            .write()
            .map_err(|_| AccountError::HistoryPoisoned)?
            .insert(
                trade_start_time,
                HistoricalPosition {
                    account_value,
                    position,
                },
            );
        Ok(())
    }

    /// Reset, update, summarize, optionally display, and record one indicator bar.
    ///
    /// # Errors
    ///
    /// Returns the first indicator or output-plugin failure. Earlier mutations are retained and
    /// recording occurs only after every preceding stage succeeds.
    pub fn update_indicator(
        &mut self,
        update: AccountIndicatorUpdate<'_>,
    ) -> Result<(), AccountError> {
        let mut indicator = self.indicator.write().map_err(|_| AccountIndicatorError {
            message: "account indicator lock poisoned".to_owned(),
        })?;
        indicator.reset().map_err(AccountError::Indicator)?;
        match update.mode {
            AccountIndicatorMode::LiveNested(nested) => indicator.update_live_nested(nested),
            AccountIndicatorMode::Atomic(executions) => indicator.update_atomic(executions),
            AccountIndicatorMode::SharedAtomic(executions) => {
                indicator.update_shared_atomic(executions)
            }
            AccountIndicatorMode::Nested(nested) => indicator.update_nested(nested),
        }
        .map_err(AccountError::Indicator)?;
        indicator
            .calculate(update.calculation)
            .map_err(AccountError::Indicator)?;
        if update.show_indicator {
            let trade = indicator
                .trade_indicator()
                .read()
                .map_err(|_| AccountIndicatorError {
                    message: "trade indicator row lock poisoned".to_owned(),
                })?;
            self.indicator_output
                .write(&self.frequency, update.trade_start_time, &trade)
                .map_err(AccountError::IndicatorOutput)?;
        }
        indicator
            .record(update.trade_start_time)
            .map_err(AccountError::Indicator)
    }

    /// Complete Qlib's ordered account update at the end of one trading bar.
    ///
    /// # Errors
    ///
    /// Missing mode-specific inputs fail before any mutation. Otherwise the method returns the
    /// first current-position, portfolio, history, indicator, or output failure and retains state
    /// produced by every earlier completed stage.
    pub fn update_bar_end(&mut self, update: AccountBarEndUpdate<'_>) -> Result<(), AccountError> {
        let indicator_mode = match update.mode {
            AccountBarEndMode::LiveNested(Some(nested)) => AccountIndicatorMode::LiveNested(nested),
            AccountBarEndMode::Atomic(Some(executions)) => AccountIndicatorMode::Atomic(executions),
            AccountBarEndMode::SharedAtomic(executions) => {
                AccountIndicatorMode::SharedAtomic(executions)
            }
            AccountBarEndMode::Atomic(None) => return Err(AccountError::MissingAtomicTradeInfo),
            AccountBarEndMode::Nested(Some(nested)) => AccountIndicatorMode::Nested(nested),
            AccountBarEndMode::Nested(None) | AccountBarEndMode::LiveNested(None) => {
                return Err(AccountError::MissingInnerOrderIndicators);
            }
        };

        self.update_current_position(
            update.trade_start_time,
            update.trade_end_time,
            update.market,
        )?;
        if self.is_portfolio_metrics_enabled() {
            self.update_portfolio_metrics(update.trade_start_time, update.trade_end_time)?;
            self.update_historical_positions(update.trade_start_time)?;
        }
        self.update_indicator(AccountIndicatorUpdate {
            trade_start_time: update.trade_start_time,
            mode: indicator_mode,
            calculation: update.calculation,
            show_indicator: update.show_indicator,
        })
    }

    /// Return the read-only execution position.
    ///
    /// # Errors
    ///
    /// Returns a position access or plugin failure.
    pub fn execution_position(&self) -> Result<&dyn ExecutionPosition, AccountError> {
        self.current_position
            .execution_position()
            .map_err(Into::into)
    }

    /// Mark every non-suspended holding to its bar close, then increment holding counts.
    ///
    /// # Errors
    ///
    /// Returns the first position or market failure. Mutations already completed are retained,
    /// while holding counts are incremented only after every stock succeeds.
    pub fn update_current_position(
        &mut self,
        trade_start_time: NaiveDateTime,
        trade_end_time: NaiveDateTime,
        market: &dyn AccountBarMarket,
    ) -> Result<(), AccountError> {
        if self.current_position.skip_update() {
            return Ok(());
        }

        let stocks = self.current_position.stock_ids()?;
        let range = TimeRange {
            start: Some(trade_start_time),
            end: Some(trade_end_time),
        };
        for stock in stocks {
            if market.is_suspended(&stock, range)? {
                continue;
            }
            let close = market.close(&stock, range)?;
            self.current_position.update_stock_price(&stock, close)?;
        }
        self.current_position.add_count_all(&self.frequency)?;
        Ok(())
    }

    /// Apply one execution using Qlib's direction-specific accounting order.
    ///
    /// # Errors
    ///
    /// Returns the first reached zero-price or position failure without rolling back accumulated
    /// metrics or position changes already reached.
    pub fn update_order(
        &mut self,
        order: &Order,
        trade_value: f64,
        trade_cost: f64,
        trade_price: f64,
    ) -> Result<(), AccountError> {
        if self.current_position.skip_update() {
            return Ok(());
        }

        match order.direction() {
            OrderDir::Sell => {
                self.update_state_from_order(order, trade_value, trade_cost, trade_price)?;
                self.current_position
                    .update_order(order, trade_value, trade_cost, trade_price)?;
            }
            OrderDir::Buy => {
                self.current_position
                    .update_order(order, trade_value, trade_cost, trade_price)?;
                self.update_state_from_order(order, trade_value, trade_cost, trade_price)?;
            }
        }
        Ok(())
    }

    fn update_state_from_order(
        &mut self,
        order: &Order,
        trade_value: f64,
        trade_cost: f64,
        trade_price: f64,
    ) -> Result<(), AccountError> {
        if !self.is_portfolio_metrics_enabled() {
            return Ok(());
        }

        self.accumulated_info.add_turnover(trade_value);
        self.accumulated_info.add_cost(trade_cost);
        if trade_price == 0.0 {
            return Err(AccountError::ZeroTradePrice);
        }
        let trade_amount = trade_value / trade_price;
        let stock_price = self.current_position.stock_price(order.stock_id())?;
        let profit = match order.direction() {
            OrderDir::Sell => trade_value - stock_price * trade_amount,
            OrderDir::Buy => stock_price * trade_amount - trade_value,
        };
        self.accumulated_info.add_return_value(profit);
        Ok(())
    }
}

impl ExecutionTarget for Account {
    fn position(&self) -> Result<&dyn ExecutionPosition, ExecutionTargetError> {
        self.execution_position()
            .map_err(|error| ExecutionTargetError {
                message: error.to_string(),
            })
    }

    fn update_order(
        &mut self,
        order: &Order,
        trade_value: f64,
        trade_cost: f64,
        trade_price: f64,
    ) -> Result<(), ExecutionTargetError> {
        Account::update_order(self, order, trade_value, trade_cost, trade_price).map_err(|error| {
            ExecutionTargetError {
                message: error.to_string(),
            }
        })
    }
}
