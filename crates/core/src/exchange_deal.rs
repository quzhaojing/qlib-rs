//! Exchange tradability checks and order-dealing orchestration.

use std::{collections::HashMap, sync::Arc};

use thiserror::Error;

use crate::{
    ExchangeQuoteProvider, ExchangeTradeCalculator, ExchangeTradeError, ExecutionPosition, Order,
    TimeRange, TradeInfo,
};

/// Diagnostic returned by a replaceable order-tradability provider.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("order tradability provider error: {message}")]
pub struct OrderTradabilityProviderError {
    pub message: String,
}

/// Narrow object-safe boundary behind `Exchange.check_order`.
pub trait OrderTradabilityProvider: Send + Sync {
    /// Determine whether one order can reach execution.
    ///
    /// # Errors
    ///
    /// Returns a market-data, policy, or provider failure.
    fn is_tradable(&self, order: &Order) -> Result<bool, OrderTradabilityProviderError>;
}

impl OrderTradabilityProvider for ExchangeQuoteProvider {
    fn is_tradable(&self, order: &Order) -> Result<bool, OrderTradabilityProviderError> {
        let range = TimeRange {
            start: order.start_time(),
            end: order.end_time(),
        };
        if self
            .stock_is_suspended(order.stock_id(), range)
            .map_err(|error| tradability_error(&error))?
        {
            return Ok(false);
        }
        self.stock_has_trade_limit(order.stock_id(), range, order.direction())
            .map(|limited| !limited)
            .map_err(|error| tradability_error(&error))
    }
}

fn tradability_error(error: &crate::ExchangeQuoteError) -> OrderTradabilityProviderError {
    OrderTradabilityProviderError {
        message: error.to_string(),
    }
}

/// Diagnostic returned while reading or updating an execution target.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("execution target error: {message}")]
pub struct ExecutionTargetError {
    pub message: String,
}

/// Account-or-position boundary required by `Exchange.deal_order`.
pub trait ExecutionTarget: Send + Sync {
    /// Return the position view used during amount and cash clipping.
    ///
    /// # Errors
    ///
    /// Returns an account or position access failure.
    fn position(&self) -> Result<&dyn ExecutionPosition, ExecutionTargetError>;

    /// Apply a completed nonzero execution to the target.
    ///
    /// # Errors
    ///
    /// Returns an account or position update failure.
    fn update_order(
        &mut self,
        order: &Order,
        trade_value: f64,
        trade_cost: f64,
        trade_price: f64,
    ) -> Result<(), ExecutionTargetError>;
}

/// Public result order of Python's `Exchange.deal_order` tuple.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrderDealResult {
    pub trade_value: f64,
    pub trade_cost: f64,
    pub trade_price: f64,
}

impl From<TradeInfo> for OrderDealResult {
    fn from(info: TradeInfo) -> Self {
        Self {
            trade_value: info.trade_value,
            trade_cost: info.trade_cost,
            trade_price: info.trade_price,
        }
    }
}

/// Typed failures from tradability and order-dealing orchestration.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ExchangeDealError {
    #[error(transparent)]
    Tradability(#[from] OrderTradabilityProviderError),
    #[error("trade account and position can only choose one")]
    ConflictingTargets,
    #[error(transparent)]
    Target(#[from] ExecutionTargetError),
    #[error(transparent)]
    Trade(#[from] ExchangeTradeError),
}

/// Composes order tradability, trade calculation, and optional target updates.
#[derive(Clone)]
pub struct ExchangeDealExecutor {
    tradability: Arc<dyn OrderTradabilityProvider>,
    calculator: ExchangeTradeCalculator,
}

impl ExchangeDealExecutor {
    #[must_use]
    pub fn new(
        tradability: Arc<dyn OrderTradabilityProvider>,
        calculator: ExchangeTradeCalculator,
    ) -> Self {
        Self {
            tradability,
            calculator,
        }
    }

    /// Match `Exchange.check_order` through the configured policy provider.
    ///
    /// # Errors
    ///
    /// Returns a typed tradability-provider failure.
    pub fn check_order(&self, order: &Order) -> Result<bool, OrderTradabilityProviderError> {
        self.tradability.is_tradable(order)
    }

    /// Deal one order and optionally update exactly one account or position target.
    ///
    /// # Errors
    ///
    /// Returns typed tradability, target-conflict, calculation, or target-update failures.
    pub fn deal_order<'a>(
        &self,
        order: &mut Order,
        account: Option<&'a mut dyn ExecutionTarget>,
        position: Option<&'a mut dyn ExecutionTarget>,
        dealt_order_amount: &HashMap<String, f64>,
    ) -> Result<OrderDealResult, ExchangeDealError> {
        if !self.check_order(order)? {
            order.set_deal_amount(0.0);
            tracing::debug!(
                stock = order.stock_id(),
                "order failed due to trading limitation"
            );
            return Ok(OrderDealResult {
                trade_value: 0.0,
                trade_cost: 0.0,
                trade_price: f64::NAN,
            });
        }
        match (account, position) {
            (Some(_), Some(_)) => Err(ExchangeDealError::ConflictingTargets),
            (Some(target), None) | (None, Some(target)) => {
                self.deal_with_target(order, target, dealt_order_amount)
            }
            (None, None) => self
                .calculator
                .calculate(order, None, dealt_order_amount)
                .map(Into::into)
                .map_err(Into::into),
        }
    }

    fn deal_with_target(
        &self,
        order: &mut Order,
        target: &mut dyn ExecutionTarget,
        dealt_order_amount: &HashMap<String, f64>,
    ) -> Result<OrderDealResult, ExchangeDealError> {
        let info =
            self.calculator
                .calculate(order, Some(target.position()?), dealt_order_amount)?;
        if info.trade_value > 1e-5 {
            target.update_order(order, info.trade_value, info.trade_cost, info.trade_price)?;
        }
        Ok(info.into())
    }
}
