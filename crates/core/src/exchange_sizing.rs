//! Exchange trade-unit rounding and cash-limited buy sizing.

use std::{cmp::Ordering, sync::Arc};

use chrono::NaiveDateTime;
use thiserror::Error;

use crate::{ExchangeQuoteProvider, TimeRange};

/// Failure emitted by a replaceable factor source.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[error("factor provider error: {message}")]
pub struct FactorProviderError {
    /// Provider diagnostic retained at the compatibility boundary.
    pub message: String,
}

/// Narrow object-safe factor lookup used only when no direct factor is supplied.
pub trait FactorProvider: Send + Sync {
    /// Return the last valid factor for one stock and closed interval.
    ///
    /// # Errors
    ///
    /// Returns a retrieval, shape, or conversion failure from the provider adapter.
    fn factor(&self, stock: &str, range: TimeRange) -> Result<Option<f64>, FactorProviderError>;
}

impl FactorProvider for ExchangeQuoteProvider {
    fn factor(&self, stock: &str, range: TimeRange) -> Result<Option<f64>, FactorProviderError> {
        self.get_factor(stock, range)
            .map_err(|error| FactorProviderError {
                message: error.to_string(),
            })
    }
}

/// Python-compatible direct-or-market factor arguments.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FactorInput<'a> {
    /// Direct factor; when present it has priority over every market lookup argument.
    pub factor: Option<f64>,
    /// Stock identifier used only when `factor` is absent.
    pub stock: Option<&'a str>,
    /// Closed lookup start used only when `factor` is absent.
    pub start_time: Option<NaiveDateTime>,
    /// Closed lookup end used only when `factor` is absent.
    pub end_time: Option<NaiveDateTime>,
}

/// Failures from Exchange execution-sizing arithmetic and factor resolution.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ExchangeSizingError {
    /// Neither a direct factor nor all market lookup arguments were supplied.
    #[error("factor and (stock, start_time, end_time) cannot both be absent")]
    MissingFactorInput,
    /// Market lookup was requested but the Exchange has no factor provider.
    #[error("factor lookup requires a configured factor provider")]
    MissingFactorProvider,
    /// Python's post-lookup assertion rejects an absent factor.
    #[error("factor provider returned no factor")]
    MissingFactor,
    /// The replaceable factor source failed.
    #[error(transparent)]
    FactorProvider(#[from] FactorProviderError),
    /// A zero factor is used as a divisor.
    #[error("factor cannot be zero for trade-unit sizing")]
    ZeroFactor,
    /// A zero trade unit is used by Python's floor-division expression.
    #[error("trade unit cannot be zero for amount rounding")]
    ZeroTradeUnit,
    /// A zero cost ratio is used while computing the critical cash threshold.
    #[error("cost ratio cannot be zero when cash reaches the minimum cost")]
    ZeroCostRatio,
    /// `1 + cost_ratio` became zero on the proportional-fee branch.
    #[error("one plus cost ratio cannot be zero")]
    ZeroCostDenominator,
    /// A zero trade price is used on either cash-limited sizing branch.
    #[error("trade price cannot be zero when sizing a buy")]
    ZeroTradePrice,
}

/// Immutable Exchange configuration required by the three execution-sizing primitives.
#[derive(Clone)]
pub struct ExchangeExecutionSizer {
    trade_with_adjusted_price: bool,
    trade_unit: Option<f64>,
    min_cost: f64,
    factor_provider: Option<Arc<dyn FactorProvider>>,
}

impl ExchangeExecutionSizer {
    /// Construct the sizing façade without imposing validation absent from Python.
    #[must_use]
    pub fn new(
        trade_with_adjusted_price: bool,
        trade_unit: Option<f64>,
        min_cost: f64,
        factor_provider: Option<Arc<dyn FactorProvider>>,
    ) -> Self {
        Self {
            trade_with_adjusted_price,
            trade_unit,
            min_cost,
            factor_provider,
        }
    }

    /// Whether missing factor data forced adjusted-price mode.
    #[must_use]
    pub const fn trade_with_adjusted_price(&self) -> bool {
        self.trade_with_adjusted_price
    }

    /// Configured raw trade unit; `None` disables unit sizing.
    #[must_use]
    pub const fn trade_unit(&self) -> Option<f64> {
        self.trade_unit
    }

    /// Minimum transaction cost used by cash-limited buy sizing.
    #[must_use]
    pub const fn min_cost(&self) -> f64 {
        self.min_cost
    }

    /// Return the adjusted share amount represented by one raw trade unit.
    ///
    /// Adjusted-price mode and a disabled trade unit return `None` without resolving a factor.
    ///
    /// # Errors
    ///
    /// Returns typed factor lookup failures or a zero-factor division failure.
    pub fn amount_of_trade_unit(
        &self,
        input: FactorInput<'_>,
    ) -> Result<Option<f64>, ExchangeSizingError> {
        let Some(trade_unit) = self.active_trade_unit() else {
            return Ok(None);
        };
        let factor = self.resolve_factor(input)?;
        if factor == 0.0 {
            return Err(ExchangeSizingError::ZeroFactor);
        }
        Ok(Some(trade_unit / factor))
    }

    /// Round an adjusted amount down using Qlib's trade-unit expression and `+0.1` correction.
    ///
    /// Adjusted-price mode and a disabled trade unit return the original amount unchanged and do
    /// not resolve a factor.
    ///
    /// # Errors
    ///
    /// Returns typed factor lookup failures or a zero trade-unit/factor division failure.
    pub fn round_amount_by_trade_unit(
        &self,
        deal_amount: f64,
        input: FactorInput<'_>,
    ) -> Result<f64, ExchangeSizingError> {
        let Some(trade_unit) = self.active_trade_unit() else {
            return Ok(deal_amount);
        };
        let factor = self.resolve_factor(input)?;
        let numerator = deal_amount * factor + 0.1;
        let units = python_float_floor_div(numerator, trade_unit)?;
        if factor == 0.0 {
            return Err(ExchangeSizingError::ZeroFactor);
        }
        Ok(units * trade_unit / factor)
    }

    /// Calculate the maximum buy amount affordable under proportional/minimum fees.
    ///
    /// # Errors
    ///
    /// Returns the division failure reached in Python evaluation order.
    pub fn buy_amount_by_cash_limit(
        &self,
        trade_price: f64,
        cash: f64,
        cost_ratio: f64,
    ) -> Result<f64, ExchangeSizingError> {
        if !matches!(
            cash.partial_cmp(&self.min_cost),
            Some(Ordering::Greater | Ordering::Equal)
        ) {
            return Ok(0.0);
        }
        if cost_ratio == 0.0 {
            return Err(ExchangeSizingError::ZeroCostRatio);
        }
        let critical_price = self.min_cost / cost_ratio + self.min_cost;
        if cash >= critical_price {
            let denominator = 1.0 + cost_ratio;
            if denominator == 0.0 {
                return Err(ExchangeSizingError::ZeroCostDenominator);
            }
            if trade_price == 0.0 {
                return Err(ExchangeSizingError::ZeroTradePrice);
            }
            Ok(cash / denominator / trade_price)
        } else {
            if trade_price == 0.0 {
                return Err(ExchangeSizingError::ZeroTradePrice);
            }
            Ok((cash - self.min_cost) / trade_price)
        }
    }

    fn active_trade_unit(&self) -> Option<f64> {
        if self.trade_with_adjusted_price {
            None
        } else {
            self.trade_unit
        }
    }

    fn resolve_factor(&self, input: FactorInput<'_>) -> Result<f64, ExchangeSizingError> {
        if let Some(factor) = input.factor {
            return Ok(factor);
        }
        let (Some(stock), Some(start), Some(end)) = (input.stock, input.start_time, input.end_time)
        else {
            return Err(ExchangeSizingError::MissingFactorInput);
        };
        self.factor_provider
            .as_deref()
            .ok_or(ExchangeSizingError::MissingFactorProvider)?
            .factor(
                stock,
                TimeRange {
                    start: Some(start),
                    end: Some(end),
                },
            )?
            .ok_or(ExchangeSizingError::MissingFactor)
    }
}

/// `CPython`'s float `//` correction, needed because `(lhs / rhs).floor()` differs near exact
/// multiples and for negative/infinite divisors.
fn python_float_floor_div(dividend: f64, divisor: f64) -> Result<f64, ExchangeSizingError> {
    if divisor == 0.0 {
        return Err(ExchangeSizingError::ZeroTradeUnit);
    }
    let modulo = dividend % divisor;
    let mut division = (dividend - modulo) / divisor;
    if modulo != 0.0 && ((divisor < 0.0) != (modulo < 0.0)) {
        division -= 1.0;
    }
    if division == 0.0 {
        Ok(0.0_f64.copysign(dividend / divisor))
    } else {
        let mut floored = division.floor();
        if division - floored > 0.5 {
            floored += 1.0;
        }
        Ok(floored)
    }
}
