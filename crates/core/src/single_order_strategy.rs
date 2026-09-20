//! Single-order outer decision generation for the SAOE backtest.

use crate::{
    Order, OrderTradeDecision, SaoeCalendar, SaoeOrderFactory, SaoePluginError,
    SharedOrderExecution, SharedTradeRange,
};

/// Owns the requested order and retains the shared trade-range rule.
/// Runtime collaborators are supplied on every call so an outer infrastructure reset
/// can use its current exchange helper and calendar without recreating this strategy.
pub struct SingleOrderStrategy {
    order: Order,
    trade_range: Option<SharedTradeRange>,
}

impl SingleOrderStrategy {
    #[must_use]
    pub fn new(order: Order, trade_range: Option<SharedTradeRange>) -> Self {
        Self { order, trade_range }
    }

    /// Create a fresh order using only the request's instrument, amount and direction.
    /// Prior executions are intentionally ignored, matching the source strategy.
    /// The first calendar read captures the decision interval; the second supplies
    /// only missing order endpoints, matching the two source decision constructors.
    ///
    /// # Errors
    /// Returns the first helper or calendar error without rewriting its diagnostic.
    pub fn generate_trade_decision(
        &self,
        _previous: Option<&[SharedOrderExecution]>,
        orders: &mut dyn SaoeOrderFactory,
        calendar: &dyn SaoeCalendar,
    ) -> Result<OrderTradeDecision, SaoePluginError> {
        let mut order = orders.create(
            self.order.stock_id(),
            Some(self.order.amount()),
            self.order.direction(),
        )?;
        let (start, end) = calendar.step_time()?;
        let (order_start, order_end) = calendar.step_time()?;
        order.fill_missing_interval(order_start, order_end);
        Ok(OrderTradeDecision::from_items(
            vec![order],
            start,
            end,
            self.trade_range.clone(),
        ))
    }
}
