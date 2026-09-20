//! Exchange order-level price, amount, value, and cost calculation.

use std::{cmp::Ordering, collections::HashMap, sync::Arc};

use thiserror::Error;

use crate::{
    ExchangeExecutionSizer, ExchangeQuoteProvider, ExchangeSizingError, ExchangeVolumeError,
    ExchangeVolumeLimiter, FactorInput, Order, OrderDir, QuoteMethod, TimeRange,
};

/// Immutable costs and sizing settings used by one Exchange execution calculator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExchangeTradeConfig {
    pub open_cost: f64,
    pub close_cost: f64,
    pub min_cost: f64,
    pub impact_cost: f64,
    pub trade_with_adjusted_price: bool,
    pub trade_unit: Option<f64>,
}

/// Diagnostic returned by the replaceable execution market-data provider.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("execution market-data provider error: {message}")]
pub struct ExecutionMarketProviderError {
    pub message: String,
}

/// Narrow object-safe market-data boundary required by `_calc_trade_info_by_order`.
pub trait ExecutionMarketProvider: Send + Sync {
    /// Return the last-valid direction-specific deal price.
    ///
    /// # Errors
    ///
    /// Returns a provider retrieval, aggregation, shape, or conversion failure.
    fn deal_price(
        &self,
        stock: &str,
        range: TimeRange,
        direction: OrderDir,
    ) -> Result<Option<f64>, ExecutionMarketProviderError>;

    /// Return summed market volume for the execution interval.
    ///
    /// # Errors
    ///
    /// Returns a provider retrieval, aggregation, shape, or conversion failure.
    fn market_volume(
        &self,
        stock: &str,
        range: TimeRange,
    ) -> Result<Option<f64>, ExecutionMarketProviderError>;

    /// Return the last-valid adjustment factor.
    ///
    /// # Errors
    ///
    /// Returns a provider retrieval, aggregation, shape, or conversion failure.
    fn factor(
        &self,
        stock: &str,
        range: TimeRange,
    ) -> Result<Option<f64>, ExecutionMarketProviderError>;
}

impl ExecutionMarketProvider for ExchangeQuoteProvider {
    fn deal_price(
        &self,
        stock: &str,
        range: TimeRange,
        direction: OrderDir,
    ) -> Result<Option<f64>, ExecutionMarketProviderError> {
        self.get_trade_deal_price(stock, range, direction)
            .map_err(|error| market_error(&error))
    }

    fn market_volume(
        &self,
        stock: &str,
        range: TimeRange,
    ) -> Result<Option<f64>, ExecutionMarketProviderError> {
        self.get_aggregated_scalar(
            stock,
            range,
            "$volume",
            QuoteMethod::BuiltIn(crate::BuiltInAggregation::Sum),
        )
        .map_err(|error| market_error(&error))
    }

    fn factor(
        &self,
        stock: &str,
        range: TimeRange,
    ) -> Result<Option<f64>, ExecutionMarketProviderError> {
        if range.start.is_none() || range.end.is_none() {
            return Err(ExecutionMarketProviderError {
                message: "factor lookup requires bounded start and end times".to_owned(),
            });
        }
        if !self.contains_stock(stock) {
            return Ok(None);
        }
        self.get_factor(stock, range)
            .map_err(|error| market_error(&error))
    }
}

fn market_error(error: &crate::ExchangeQuoteError) -> ExecutionMarketProviderError {
    ExecutionMarketProviderError {
        message: error.to_string(),
    }
}

/// Diagnostic returned by a replaceable read-only position view.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("execution position error: {message}")]
pub struct ExecutionPositionError {
    pub message: String,
}

/// Read-only position seam needed before account mutation occurs.
pub trait ExecutionPosition: Send + Sync {
    /// Match `BasePosition.check_stock` without assuming storage layout.
    ///
    /// # Errors
    ///
    /// Returns a position-provider failure.
    fn check_stock(&self, stock: &str) -> Result<bool, ExecutionPositionError>;

    /// Return the currently held amount after a successful stock check.
    ///
    /// # Errors
    ///
    /// Returns a position-provider failure.
    fn stock_amount(&self, stock: &str) -> Result<f64, ExecutionPositionError>;

    /// Return currently tradable cash, excluding delayed settlement.
    ///
    /// # Errors
    ///
    /// Returns a position-provider failure.
    fn cash(&self) -> Result<f64, ExecutionPositionError>;
}

/// Result tuple returned by Python's `_calc_trade_info_by_order`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TradeInfo {
    pub trade_price: f64,
    pub trade_value: f64,
    pub trade_cost: f64,
}

/// Typed failures from the order-level execution calculation.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ExchangeTradeError {
    #[error("execution provider returned no market volume")]
    MissingMarketVolume,
    #[error("execution provider returned no deal price")]
    MissingDealPrice,
    #[error(transparent)]
    MarketProvider(#[from] ExecutionMarketProviderError),
    #[error(transparent)]
    Volume(#[from] ExchangeVolumeError),
    #[error(transparent)]
    Sizing(#[from] ExchangeSizingError),
    #[error(transparent)]
    Position(#[from] ExecutionPositionError),
}

/// Composes the completed Exchange market-data, sizing, and volume-limit slices.
#[derive(Clone)]
pub struct ExchangeTradeCalculator {
    config: ExchangeTradeConfig,
    market_provider: Arc<dyn ExecutionMarketProvider>,
    sizing: ExchangeExecutionSizer,
    volume_limiter: ExchangeVolumeLimiter,
}

impl ExchangeTradeCalculator {
    #[must_use]
    pub fn new(
        config: ExchangeTradeConfig,
        market_provider: Arc<dyn ExecutionMarketProvider>,
        volume_limiter: ExchangeVolumeLimiter,
    ) -> Self {
        let sizing = ExchangeExecutionSizer::new(
            config.trade_with_adjusted_price,
            config.trade_unit,
            config.min_cost,
            None,
        );
        Self {
            config,
            market_provider,
            sizing,
            volume_limiter,
        }
    }

    #[must_use]
    pub const fn config(&self) -> ExchangeTradeConfig {
        self.config
    }

    /// Calculate one order's price, final amount, value, and cost in Python evaluation order.
    ///
    /// The order is first mutated only after price, volume, and factor lookup succeed. Later
    /// volume, sizing, or position failures retain mutations already performed by Python.
    ///
    /// # Errors
    ///
    /// Returns typed market-data, volume-limit, sizing, or position failures.
    pub fn calculate(
        &self,
        order: &mut Order,
        position: Option<&dyn ExecutionPosition>,
        dealt_order_amount: &HashMap<String, f64>,
    ) -> Result<TradeInfo, ExchangeTradeError> {
        let range = TimeRange {
            start: order.start_time(),
            end: order.end_time(),
        };
        let trade_price =
            self.market_provider
                .deal_price(order.stock_id(), range, order.direction())?;
        let market_volume = self
            .market_provider
            .market_volume(order.stock_id(), range)?;
        let market_volume = market_volume.ok_or(ExchangeTradeError::MissingMarketVolume)?;
        let trade_price = trade_price.ok_or(ExchangeTradeError::MissingDealPrice)?;
        let total_trade_value = market_volume * trade_price;
        let factor = self.market_provider.factor(order.stock_id(), range)?;
        order.set_factor(factor);
        order.set_deal_amount(order.amount());
        self.volume_limiter
            .clip_amount_by_volume(order, dealt_order_amount)?;

        let initial_trade_value = order.deal_amount() * trade_price;
        let adjusted_impact = if total_trade_value == 0.0 || total_trade_value.is_nan() {
            self.config.impact_cost
        } else {
            self.config.impact_cost * (initial_trade_value / total_trade_value).powi(2)
        };

        let cost_ratio = match order.direction() {
            OrderDir::Sell => {
                let cost_ratio = self.config.close_cost + adjusted_impact;
                if let Some(position) = position {
                    let current_amount = if position.check_stock(order.stock_id())? {
                        position.stock_amount(order.stock_id())?
                    } else {
                        0.0
                    };
                    if !numpy_is_close(order.deal_amount(), current_amount) {
                        let amount = python_min(current_amount, order.deal_amount());
                        order.set_deal_amount(self.round(order, amount)?);
                    }
                    let trade_value = order.deal_amount() * trade_price;
                    if position.cash()? + trade_value
                        < python_max(trade_value * cost_ratio, self.config.min_cost)
                    {
                        order.set_deal_amount(0.0);
                        tracing::debug!(
                            stock = order.stock_id(),
                            "sell order clipped due to cash limitation"
                        );
                    }
                }
                cost_ratio
            }
            OrderDir::Buy => {
                let cost_ratio = self.config.open_cost + adjusted_impact;
                if let Some(position) = position {
                    let cash = position.cash()?;
                    let trade_value = order.deal_amount() * trade_price;
                    let trade_cost = python_max(trade_value * cost_ratio, self.config.min_cost);
                    if cash < trade_cost {
                        order.set_deal_amount(0.0);
                        tracing::debug!(
                            stock = order.stock_id(),
                            "buy order clipped because cost exceeds cash"
                        );
                    } else if cash < trade_value + trade_cost {
                        let max_amount =
                            self.sizing
                                .buy_amount_by_cash_limit(trade_price, cash, cost_ratio)?;
                        let amount = python_min(max_amount, order.deal_amount());
                        order.set_deal_amount(self.round(order, amount)?);
                        tracing::debug!(
                            stock = order.stock_id(),
                            "buy order clipped due to cash limitation"
                        );
                    } else {
                        order.set_deal_amount(self.round(order, order.deal_amount())?);
                    }
                } else {
                    order.set_deal_amount(self.round(order, order.deal_amount())?);
                }
                cost_ratio
            }
        };

        let trade_value = order.deal_amount() * trade_price;
        let mut trade_cost = python_max(trade_value * cost_ratio, self.config.min_cost);
        if trade_value <= 1e-5 {
            trade_cost = 0.0;
        }
        Ok(TradeInfo {
            trade_price,
            trade_value,
            trade_cost,
        })
    }

    fn round(&self, order: &Order, amount: f64) -> Result<f64, ExchangeSizingError> {
        self.sizing.round_amount_by_trade_unit(
            amount,
            FactorInput {
                factor: order.factor(),
                stock: None,
                start_time: None,
                end_time: None,
            },
        )
    }
}

pub(crate) fn numpy_is_close(left: f64, right: f64) -> bool {
    if left.partial_cmp(&right) == Some(Ordering::Equal) {
        return true;
    }
    if !left.is_finite() || !right.is_finite() {
        return false;
    }
    (left - right).abs() <= 1e-8 + 1e-5 * right.abs()
}

fn python_min(left: f64, right: f64) -> f64 {
    if right < left { right } else { left }
}

fn python_max(left: f64, right: f64) -> f64 {
    if right > left { right } else { left }
}
