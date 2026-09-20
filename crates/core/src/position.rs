//! Finite and infinite execution positions compatible with Qlib order updates.

use indexmap::IndexMap;
use thiserror::Error;

use crate::{
    ExecutionPosition, ExecutionPositionError, ExecutionTarget, ExecutionTargetError, Order,
    OrderDir, PositionCash, PositionCashError, exchange_trade::numpy_is_close,
};

/// One stock dictionary normalized into typed optional fields.
#[derive(Clone, Debug, PartialEq)]
pub struct PositionHolding {
    amount: f64,
    price: Option<f64>,
    weight: Option<f64>,
    counts: IndexMap<String, f64>,
}

impl PositionHolding {
    /// Restore an existing Python holding, where price and weight may be absent.
    #[must_use]
    pub fn restored(amount: f64, price: Option<f64>, weight: Option<f64>) -> Self {
        Self {
            amount,
            price,
            weight,
            counts: IndexMap::new(),
        }
    }

    /// Restore an existing Python holding with normalized `count_<bar>` fields.
    #[must_use]
    pub const fn restored_with_counts(
        amount: f64,
        price: Option<f64>,
        weight: Option<f64>,
        counts: IndexMap<String, f64>,
    ) -> Self {
        Self {
            amount,
            price,
            weight,
            counts,
        }
    }

    #[must_use]
    pub const fn amount(&self) -> f64 {
        self.amount
    }

    #[must_use]
    pub const fn price(&self) -> Option<f64> {
        self.price
    }

    #[must_use]
    pub const fn weight(&self) -> Option<f64> {
        self.weight
    }

    #[must_use]
    pub const fn counts(&self) -> &IndexMap<String, f64> {
        &self.counts
    }

    fn opened(amount: f64, price: f64) -> Self {
        Self {
            amount,
            price: Some(price),
            weight: Some(0.0),
            counts: IndexMap::new(),
        }
    }
}

/// Python constructor input after a binding adapter normalizes a bare numeric amount.
#[derive(Clone, Debug, PartialEq)]
pub enum InitialPositionValue {
    Amount(f64),
    Holding(PositionHolding),
}

/// Typed failures from finite position access and mutation.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum PositionError {
    #[error("trade price cannot be zero")]
    ZeroTradePrice,
    #[error("{stock} not in current position")]
    MissingStock { stock: String },
    #[error("stock {stock} has no price")]
    MissingPrice { stock: String },
    #[error("stock {stock} has no weight")]
    MissingWeight { stock: String },
    #[error("{{{stocks}}} doesn't have close price in qlib in the latest 30 days")]
    MissingInitialPrices { stocks: String },
    #[error("position value cannot be zero while calculating stock weights")]
    ZeroPositionValue,
    #[error("only have {available} {stock}, require {required}")]
    Oversell {
        available: f64,
        stock: String,
        required: f64,
    },
    #[error(transparent)]
    Cash(#[from] PositionCashError),
    #[error("infinite position does not support {operation}")]
    UnsupportedInfiniteOperation { operation: &'static str },
}

/// Typed finite Qlib position used by exchange execution.
#[derive(Clone, Debug, PartialEq)]
pub struct Position {
    init_cash: f64,
    holdings: IndexMap<String, PositionHolding>,
    cash: PositionCash,
    account_value: Option<f64>,
}

impl Position {
    /// Normalize Python's documented integer-or-dictionary initialization surface.
    #[must_use]
    pub fn from_initial(cash: f64, initial: IndexMap<String, InitialPositionValue>) -> Self {
        let holdings = initial
            .into_iter()
            .map(|(stock, value)| {
                let holding = match value {
                    InitialPositionValue::Amount(amount) => {
                        PositionHolding::restored(amount, None, None)
                    }
                    InitialPositionValue::Holding(holding) => holding,
                };
                (stock, holding)
            })
            .collect();
        Self {
            init_cash: cash,
            holdings,
            cash: PositionCash::new(cash),
            account_value: None,
        }
    }

    #[must_use]
    pub const fn initial_cash(&self) -> f64 {
        self.init_cash
    }

    #[must_use]
    pub const fn holdings(&self) -> &IndexMap<String, PositionHolding> {
        &self.holdings
    }

    #[must_use]
    pub fn holding(&self, stock: &str) -> Option<&PositionHolding> {
        self.holdings.get(stock)
    }

    pub fn stock_ids(&self) -> impl Iterator<Item = &str> {
        self.holdings.keys().map(String::as_str)
    }

    #[must_use]
    pub fn check_stock(&self, stock: &str) -> bool {
        self.holdings.contains_key(stock)
    }

    /// Match `Position.get_stock_amount`, including zero for an absent stock.
    #[must_use]
    pub fn stock_amount(&self, stock: &str) -> f64 {
        self.holding(stock).map_or(0.0, PositionHolding::amount)
    }

    /// Return the current stock price.
    ///
    /// # Errors
    ///
    /// Returns a missing-stock or missing-price failure.
    pub fn stock_price(&self, stock: &str) -> Result<f64, PositionError> {
        self.holding(stock)
            .ok_or_else(|| PositionError::MissingStock {
                stock: stock.to_owned(),
            })?
            .price()
            .ok_or_else(|| PositionError::MissingPrice {
                stock: stock.to_owned(),
            })
    }

    /// Replace the latest stock price.
    ///
    /// # Errors
    ///
    /// Returns a missing-stock failure before mutation.
    pub fn update_stock_price(&mut self, stock: &str, price: f64) -> Result<(), PositionError> {
        self.holding_mut(stock)?.price = Some(price);
        Ok(())
    }

    /// Atomically restore every missing initial price and recalculate account value.
    /// Extra provider values are ignored and existing holding prices are retained.
    ///
    /// # Errors
    ///
    /// Returns all missing provider keys in deterministic holding order before mutation.
    pub fn fill_missing_stock_prices(
        &mut self,
        prices: &IndexMap<String, f64>,
    ) -> Result<(), PositionError> {
        let missing: Vec<_> = self
            .holdings
            .iter()
            .filter(|(_, holding)| holding.price.is_none())
            .filter(|(stock, _)| prices.get(*stock).is_none_or(|price| price.is_nan()))
            .map(|(stock, _)| stock.as_str())
            .collect();
        if !missing.is_empty() {
            return Err(PositionError::MissingInitialPrices {
                stocks: missing
                    .into_iter()
                    .map(|stock| format!("'{stock}'"))
                    .collect::<Vec<_>>()
                    .join(", "),
            });
        }
        let resolved_prices: Vec<_> = self
            .holdings
            .iter()
            .map(|(stock, holding)| holding.price.unwrap_or_else(|| prices[stock]))
            .collect();
        let stock_value = self
            .holdings
            .values()
            .zip(&resolved_prices)
            .fold(0.0, |value, (holding, price)| {
                value + holding.amount * price
            });
        let account_value = stock_value + self.cash(true);
        for (holding, price) in self.holdings.values_mut().zip(resolved_prices) {
            if holding.price.is_none() {
                holding.price = Some(price);
            }
        }
        self.account_value = Some(account_value);
        Ok(())
    }

    /// Replace one normalized Python `count_<bar>` value.
    ///
    /// # Errors
    ///
    /// Returns a missing-stock failure before mutation.
    pub fn update_stock_count(
        &mut self,
        stock: &str,
        bar: &str,
        count: f64,
    ) -> Result<(), PositionError> {
        self.holding_mut(stock)?
            .counts
            .insert(bar.to_owned(), count);
        Ok(())
    }

    /// Replace the stored total-asset weight.
    ///
    /// # Errors
    ///
    /// Returns a missing-stock failure before mutation.
    pub fn update_stock_weight(&mut self, stock: &str, weight: f64) -> Result<(), PositionError> {
        self.holding_mut(stock)?.weight = Some(weight);
        Ok(())
    }

    /// Return a holding count, or zero when that bar field is absent.
    ///
    /// # Errors
    ///
    /// Returns a missing-stock failure.
    pub fn stock_count(&self, stock: &str, bar: &str) -> Result<f64, PositionError> {
        Ok(*self
            .holding_required(stock)?
            .counts
            .get(bar)
            .unwrap_or(&0.0))
    }

    /// Return the stored stock weight.
    ///
    /// # Errors
    ///
    /// Returns a missing-stock or missing-weight failure.
    pub fn stock_weight(&self, stock: &str) -> Result<f64, PositionError> {
        self.holding_required(stock)?
            .weight
            .ok_or_else(|| PositionError::MissingWeight {
                stock: stock.to_owned(),
            })
    }

    /// Calculate all non-cash assets in deterministic holding order.
    ///
    /// # Errors
    ///
    /// Returns the first missing-price failure.
    pub fn calculate_stock_value(&self) -> Result<f64, PositionError> {
        Ok(self
            .stock_values()?
            .values()
            .fold(0.0, |value, stock_value| value + stock_value))
    }

    /// Calculate stock value plus available and delayed cash.
    ///
    /// # Errors
    ///
    /// Returns the first missing-price failure.
    pub fn calculate_value(&self) -> Result<f64, PositionError> {
        Ok(self.calculate_stock_value()? + self.cash(true))
    }

    /// Produce an owned, insertion-ordered stock-to-amount snapshot.
    #[must_use]
    pub fn stock_amounts(&self) -> IndexMap<String, f64> {
        self.holdings
            .iter()
            .map(|(stock, holding)| (stock.clone(), holding.amount))
            .collect()
    }

    /// Calculate each stock's value weight against stock-only or total value.
    ///
    /// # Errors
    ///
    /// Returns a missing-price failure or Python-compatible zero-division failure.
    pub fn stock_weights(&self, only_stock: bool) -> Result<IndexMap<String, f64>, PositionError> {
        let stock_values = self.stock_values()?;
        let stock_value = stock_values
            .values()
            .fold(0.0, |value, stock_value| value + stock_value);
        let position_value = if only_stock {
            stock_value
        } else {
            stock_value + self.cash(true)
        };
        stock_values
            .into_iter()
            .map(|(stock, stock_value)| {
                if position_value == 0.0 {
                    return Err(PositionError::ZeroPositionValue);
                }
                Ok((stock, stock_value / position_value))
            })
            .collect()
    }

    /// Increment every holding's normalized `count_<bar>` value.
    pub fn add_count_all(&mut self, bar: &str) {
        for holding in self.holdings.values_mut() {
            *holding.counts.entry(bar.to_owned()).or_insert(0.0) += 1.0;
        }
    }

    /// Recalculate and store total-asset weights atomically.
    ///
    /// # Errors
    ///
    /// Returns a valuation error without changing any weight.
    pub fn update_weight_all(&mut self) -> Result<(), PositionError> {
        let weights = self.stock_weights(false)?;
        for (holding, weight) in self.holdings.values_mut().zip(weights.into_values()) {
            holding.weight = Some(weight);
        }
        Ok(())
    }

    #[must_use]
    pub fn cash(&self, include_settlement: bool) -> f64 {
        self.cash.cash(include_settlement)
    }

    #[must_use]
    pub const fn cash_state(&self) -> &PositionCash {
        &self.cash
    }

    #[must_use]
    pub const fn account_value(&self) -> Option<f64> {
        self.account_value
    }

    pub const fn set_account_value(&mut self, value: f64) {
        self.account_value = Some(value);
    }

    /// Start cash or custom settlement through the completed cash state core.
    ///
    /// # Errors
    ///
    /// Returns a nested-settlement failure.
    pub fn settle_start(&mut self, settlement_type: &str) -> Result<(), PositionCashError> {
        self.cash.settle_start(settlement_type)
    }

    /// Commit the current settlement transaction.
    ///
    /// # Errors
    ///
    /// Returns an unsupported or inconsistent settlement failure.
    pub fn settle_commit(&mut self) -> Result<(), PositionCashError> {
        self.cash.settle_commit()
    }

    /// Apply one successful exchange order in Python's mutation order.
    ///
    /// # Errors
    ///
    /// Returns a zero-price, missing-stock, oversell, or settlement failure. Holdings changes
    /// reached before an oversell or settlement error are deliberately retained.
    pub fn update_order(
        &mut self,
        order: &Order,
        trade_value: f64,
        cost: f64,
        trade_price: f64,
    ) -> Result<(), PositionError> {
        match order.direction() {
            OrderDir::Buy => self.buy_stock(order.stock_id(), trade_value, cost, trade_price),
            OrderDir::Sell => self.sell_stock(order.stock_id(), trade_value, cost, trade_price),
        }
    }

    fn buy_stock(
        &mut self,
        stock: &str,
        trade_value: f64,
        cost: f64,
        trade_price: f64,
    ) -> Result<(), PositionError> {
        let trade_amount = trade_amount(trade_value, trade_price)?;
        if let Some(holding) = self.holdings.get_mut(stock) {
            holding.amount += trade_amount;
        } else {
            self.holdings.insert(
                stock.to_owned(),
                PositionHolding::opened(trade_amount, trade_price),
            );
        }
        self.cash.pay_for_purchase(trade_value, cost);
        Ok(())
    }

    fn sell_stock(
        &mut self,
        stock: &str,
        trade_value: f64,
        cost: f64,
        trade_price: f64,
    ) -> Result<(), PositionError> {
        let trade_amount = trade_amount(trade_value, trade_price)?;
        let sell_all = {
            let holding =
                self.holdings
                    .get_mut(stock)
                    .ok_or_else(|| PositionError::MissingStock {
                        stock: stock.to_owned(),
                    })?;
            if numpy_is_close(holding.amount, trade_amount) {
                true
            } else {
                holding.amount -= trade_amount;
                if holding.amount < -1e-5 {
                    return Err(PositionError::Oversell {
                        available: holding.amount + trade_amount,
                        stock: stock.to_owned(),
                        required: trade_amount,
                    });
                }
                false
            }
        };
        if sell_all {
            self.holdings.shift_remove(stock);
        }
        self.cash.record_sale_proceeds(trade_value, cost)?;
        Ok(())
    }

    fn holding_required(&self, stock: &str) -> Result<&PositionHolding, PositionError> {
        self.holdings
            .get(stock)
            .ok_or_else(|| PositionError::MissingStock {
                stock: stock.to_owned(),
            })
    }

    fn holding_mut(&mut self, stock: &str) -> Result<&mut PositionHolding, PositionError> {
        self.holdings
            .get_mut(stock)
            .ok_or_else(|| PositionError::MissingStock {
                stock: stock.to_owned(),
            })
    }

    fn stock_values(&self) -> Result<IndexMap<String, f64>, PositionError> {
        self.holdings
            .iter()
            .map(|(stock, holding)| {
                let price = holding.price.ok_or_else(|| PositionError::MissingPrice {
                    stock: stock.clone(),
                })?;
                Ok((stock.clone(), holding.amount * price))
            })
            .collect()
    }
}

fn trade_amount(trade_value: f64, trade_price: f64) -> Result<f64, PositionError> {
    if trade_price == 0.0 {
        return Err(PositionError::ZeroTradePrice);
    }
    Ok(trade_value / trade_price)
}

impl ExecutionPosition for Position {
    fn check_stock(&self, stock: &str) -> Result<bool, ExecutionPositionError> {
        Ok(self.check_stock(stock))
    }

    fn stock_amount(&self, stock: &str) -> Result<f64, ExecutionPositionError> {
        Ok(self.stock_amount(stock))
    }

    fn cash(&self) -> Result<f64, ExecutionPositionError> {
        Ok(self.cash(false))
    }
}

impl ExecutionTarget for Position {
    fn position(&self) -> Result<&dyn ExecutionPosition, ExecutionTargetError> {
        Ok(self)
    }

    fn update_order(
        &mut self,
        order: &Order,
        trade_value: f64,
        trade_cost: f64,
        trade_price: f64,
    ) -> Result<(), ExecutionTargetError> {
        self.update_order(order, trade_value, trade_cost, trade_price)
            .map_err(|error| ExecutionTargetError {
                message: error.to_string(),
            })
    }
}

/// Execution-facing behavior of `InfPosition`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InfinitePosition;

impl InfinitePosition {
    #[must_use]
    pub const fn skip_update(self) -> bool {
        true
    }

    pub const fn update_stock_price(self, _stock: &str, _price: f64) {}

    #[must_use]
    pub const fn calculate_stock_value(self) -> f64 {
        f64::INFINITY
    }

    /// Infinite positions deliberately do not define a total account value.
    ///
    /// # Errors
    ///
    /// Always returns an unsupported-operation failure.
    pub const fn calculate_value(self) -> Result<f64, PositionError> {
        Err(PositionError::UnsupportedInfiniteOperation {
            operation: "calculating value",
        })
    }

    /// Infinite positions have no enumerable stock set.
    ///
    /// # Errors
    ///
    /// Always returns an unsupported-operation failure.
    pub const fn stock_list(self) -> Result<Vec<String>, PositionError> {
        Err(PositionError::UnsupportedInfiniteOperation {
            operation: "stock list",
        })
    }

    #[must_use]
    pub const fn stock_price(self, _stock: &str) -> f64 {
        f64::NAN
    }

    /// Infinite positions have no finite amount snapshot.
    ///
    /// # Errors
    ///
    /// Always returns an unsupported-operation failure.
    pub fn stock_amounts(self) -> Result<IndexMap<String, f64>, PositionError> {
        Err(PositionError::UnsupportedInfiniteOperation {
            operation: "stock amount snapshot",
        })
    }

    /// Infinite positions have no finite weight snapshot.
    ///
    /// # Errors
    ///
    /// Always returns an unsupported-operation failure.
    pub fn stock_weights(self, _only_stock: bool) -> Result<IndexMap<String, f64>, PositionError> {
        Err(PositionError::UnsupportedInfiniteOperation {
            operation: "stock weight snapshot",
        })
    }

    /// Infinite positions do not maintain holding counts.
    ///
    /// # Errors
    ///
    /// Always returns an unsupported-operation failure.
    pub const fn add_count_all(self, _bar: &str) -> Result<(), PositionError> {
        Err(PositionError::UnsupportedInfiniteOperation {
            operation: "incrementing holding counts",
        })
    }

    /// Infinite positions do not maintain weights.
    ///
    /// # Errors
    ///
    /// Always returns an unsupported-operation failure.
    pub const fn update_weight_all(self) -> Result<(), PositionError> {
        Err(PositionError::UnsupportedInfiniteOperation {
            operation: "updating weights",
        })
    }
}

impl ExecutionPosition for InfinitePosition {
    fn check_stock(&self, _stock: &str) -> Result<bool, ExecutionPositionError> {
        Ok(true)
    }

    fn stock_amount(&self, _stock: &str) -> Result<f64, ExecutionPositionError> {
        Ok(f64::INFINITY)
    }

    fn cash(&self) -> Result<f64, ExecutionPositionError> {
        Ok(f64::INFINITY)
    }
}

impl ExecutionTarget for InfinitePosition {
    fn position(&self) -> Result<&dyn ExecutionPosition, ExecutionTargetError> {
        Ok(self)
    }

    fn update_order(
        &mut self,
        _order: &Order,
        _trade_value: f64,
        _trade_cost: f64,
        _trade_price: f64,
    ) -> Result<(), ExecutionTargetError> {
        Ok(())
    }
}
