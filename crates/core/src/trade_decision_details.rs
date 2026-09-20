//! Order decisions carrying an arbitrary execution-report payload.

use chrono::NaiveDateTime;

use crate::{Order, OrderDecision, OrderTradeDecision, SharedTradeRange, TradeRange};

/// Typed counterpart of Python's `TradeDecisionWithDetails`.
///
/// `D` retains the caller's payload representation and ownership. Use `Option<D>` when the
/// Python-compatible `None` state must be represented explicitly; no serialization or cloning is
/// imposed on the payload.
pub struct TradeDecisionWithDetails<D> {
    core: OrderTradeDecision,
    details: D,
}

impl<D> TradeDecisionWithDetails<D> {
    /// Attach `details` after the order decision has been constructed and normalized.
    #[must_use]
    pub const fn new(core: OrderTradeDecision, details: D) -> Self {
        Self { core, details }
    }

    /// Construct the parent order decision, filling only missing endpoints, then attach details.
    #[must_use]
    pub fn from_orders(
        orders: Vec<Order>,
        start_time: NaiveDateTime,
        end_time: NaiveDateTime,
        trade_range: Option<SharedTradeRange>,
        details: D,
    ) -> Self {
        Self::new(
            OrderTradeDecision::from_orders(orders, start_time, end_time, trade_range),
            details,
        )
    }

    /// Borrow the normalized parent order decision.
    #[must_use]
    pub const fn core(&self) -> &OrderTradeDecision {
        &self.core
    }

    /// Mutably borrow the parent decision, matching inherited Python attribute mutability.
    pub const fn core_mut(&mut self) -> &mut OrderTradeDecision {
        &mut self.core
    }

    /// Borrow the exact caller-selected detail payload.
    #[must_use]
    pub const fn details(&self) -> &D {
        &self.details
    }

    /// Mutably borrow the detail payload.
    pub const fn details_mut(&mut self) -> &mut D {
        &mut self.details
    }

    /// Replace the payload while retaining the parent decision.
    pub fn replace_details(&mut self, details: D) -> D {
        std::mem::replace(&mut self.details, details)
    }

    /// Recover both owned values without cloning either one.
    #[must_use]
    pub fn into_parts(self) -> (OrderTradeDecision, D) {
        (self.core, self.details)
    }
}

impl<D: Send + Sync> OrderDecision for TradeDecisionWithDetails<D> {
    fn orders(&self) -> &[Order] {
        self.core.orders()
    }

    fn orders_mut(&mut self) -> &mut [Order] {
        self.core.orders_mut()
    }

    fn start_time(&self) -> NaiveDateTime {
        self.core.start_time()
    }

    fn end_time(&self) -> NaiveDateTime {
        self.core.end_time()
    }

    fn trade_range(&self) -> Option<&dyn TradeRange> {
        self.core.trade_range()
    }
}
