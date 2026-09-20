//! Cash and delayed-cash state used by Qlib positions.

use thiserror::Error;

/// Qlib's exact spelling for cash settlement.
pub const CASH_SETTLEMENT: &str = "cash";
/// Qlib's exact spelling for disabled settlement.
pub const NO_SETTLEMENT: &str = "None";

/// Typed failures from finite-position settlement and sale proceeds.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum PositionCashError {
    /// Qlib rejects beginning a transaction while any non-`None` settlement is active.
    #[error("settlement cannot be nested while {active} is active")]
    NestedSettlement { active: String },
    /// Unknown settlement names are retained at start and rejected only when first used.
    #[error("unsupported settlement type: {settlement_type}")]
    UnsupportedSettlement { settlement_type: String },
    /// A Python-compatible restored state can be missing the `cash_delay` dictionary entry.
    #[error("cash settlement is missing delayed cash")]
    MissingDelayedCash,
}

/// Finite position's available cash, optional delayed cash, and active settlement type.
#[derive(Clone, Debug, PartialEq)]
pub struct PositionCash {
    available_cash: f64,
    delayed_cash: Option<f64>,
    settlement_type: String,
}

impl PositionCash {
    /// Construct the normal initial state used by `Position.__init__`.
    #[must_use]
    pub fn new(cash: f64) -> Self {
        Self {
            available_cash: cash,
            delayed_cash: None,
            settlement_type: NO_SETTLEMENT.to_owned(),
        }
    }

    /// Restore the three observable fields from a Python dictionary or checkpoint adapter.
    #[must_use]
    pub fn restore(
        available_cash: f64,
        delayed_cash: Option<f64>,
        settlement_type: impl Into<String>,
    ) -> Self {
        Self {
            available_cash,
            delayed_cash,
            settlement_type: settlement_type.into(),
        }
    }

    /// Cash available to buy during the current execution step.
    #[must_use]
    pub const fn available_cash(&self) -> f64 {
        self.available_cash
    }

    /// Current `cash_delay` entry, including one restored outside an active cash transaction.
    #[must_use]
    pub const fn delayed_cash(&self) -> Option<f64> {
        self.delayed_cash
    }

    /// Exact active settlement spelling.
    #[must_use]
    pub fn settlement_type(&self) -> &str {
        &self.settlement_type
    }

    /// Match `Position.get_cash(include_settle=...)`.
    #[must_use]
    pub fn cash(&self, include_settlement: bool) -> f64 {
        if include_settlement {
            self.available_cash + self.delayed_cash.unwrap_or(0.0)
        } else {
            self.available_cash
        }
    }

    /// Begin one settlement transaction.
    ///
    /// Cash settlement replaces any pre-existing delayed-cash entry with positive zero. The
    /// special `"None"` value remains inactive and therefore can be started repeatedly.
    ///
    /// # Errors
    ///
    /// Returns [`PositionCashError::NestedSettlement`] when a non-`None` transaction is active.
    pub fn settle_start(&mut self, settlement_type: &str) -> Result<(), PositionCashError> {
        if self.settlement_type != NO_SETTLEMENT {
            return Err(PositionCashError::NestedSettlement {
                active: self.settlement_type.clone(),
            });
        }
        settlement_type.clone_into(&mut self.settlement_type);
        if settlement_type == CASH_SETTLEMENT {
            self.delayed_cash = Some(0.0);
        }
        Ok(())
    }

    /// Commit the current settlement transaction.
    ///
    /// # Errors
    ///
    /// Returns an unsupported-type or missing-delayed-cash failure while retaining the active
    /// transaction, matching Python's mutation boundary.
    pub fn settle_commit(&mut self) -> Result<(), PositionCashError> {
        if self.settlement_type == NO_SETTLEMENT {
            return Ok(());
        }
        if self.settlement_type != CASH_SETTLEMENT {
            return Err(PositionCashError::UnsupportedSettlement {
                settlement_type: self.settlement_type.clone(),
            });
        }
        let delayed_cash = self
            .delayed_cash
            .ok_or(PositionCashError::MissingDelayedCash)?;
        self.available_cash += delayed_cash;
        self.delayed_cash = None;
        NO_SETTLEMENT.clone_into(&mut self.settlement_type);
        Ok(())
    }

    /// Apply the cash portion of a successful sell after holdings have already been mutated.
    ///
    /// # Errors
    ///
    /// Returns an unsupported-type or missing-delayed-cash failure without changing cash.
    pub fn record_sale_proceeds(
        &mut self,
        trade_value: f64,
        cost: f64,
    ) -> Result<(), PositionCashError> {
        let new_cash = trade_value - cost;
        if self.settlement_type == CASH_SETTLEMENT {
            let delayed_cash = self
                .delayed_cash
                .as_mut()
                .ok_or(PositionCashError::MissingDelayedCash)?;
            *delayed_cash += new_cash;
            return Ok(());
        }
        if self.settlement_type == NO_SETTLEMENT {
            self.available_cash += new_cash;
            return Ok(());
        }
        Err(PositionCashError::UnsupportedSettlement {
            settlement_type: self.settlement_type.clone(),
        })
    }

    /// Apply Qlib's purchase cash expression, independent of settlement state.
    pub fn pay_for_purchase(&mut self, trade_value: f64, cost: f64) {
        self.available_cash -= trade_value + cost;
    }
}

/// Cash behavior of `InfPosition`, where all updates and settlement calls are no-ops.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InfinitePositionCash;

impl InfinitePositionCash {
    /// Both available and settlement-inclusive cash are positive infinity.
    #[must_use]
    pub fn cash(self, _include_settlement: bool) -> f64 {
        f64::INFINITY
    }

    /// Ignore every settlement type, including repeated and unsupported values.
    pub const fn settle_start(&mut self, _settlement_type: &str) {}

    /// Commit is intentionally a no-op.
    pub const fn settle_commit(&mut self) {}

    /// Selling cannot change an infinite position.
    pub const fn record_sale_proceeds(&mut self, _trade_value: f64, _cost: f64) {}

    /// Buying cannot change an infinite position.
    pub const fn pay_for_purchase(&mut self, _trade_value: f64, _cost: f64) {}
}
