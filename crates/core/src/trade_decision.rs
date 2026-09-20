//! Typed trade-decision containers compatible with the order-bearing Qlib decision path.

use std::{convert::Infallible, sync::Arc};

use chrono::NaiveDateTime;
use thiserror::Error;
use tracing::warn;

use crate::{BasePriceStep, Order, TradeCalendarRange, TradeRange, TradeRangeError};

/// Python treats an order as non-empty only when its amount is strictly above this value.
pub const EMPTY_ORDER_AMOUNT: f64 = 1.0e-6;

/// Owned, shareable in-process trade-range rule.
pub type SharedTradeRange = Arc<dyn TradeRange>;

/// Explicit replacement for Python's distinction between an omitted `default_value` and one
/// supplied as either a range or `None`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RangeLimitDefault {
    /// Raise when the decision cannot provide a range.
    Error,
    /// Return the supplied value; `None` is a meaningful fallback.
    Value(Option<(i64, i64)>),
}

/// Failures from resolving a decision's closed step range.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TradeDecisionError {
    /// Neither a usable range nor an explicit fallback was available.
    #[error("the decision did not provide an index range")]
    MissingRange,
    /// A configured range or its calendar provider failed.
    #[error(transparent)]
    TradeRange(#[from] TradeRangeError),
}

/// Metadata and concrete items produced by one strategy step.
///
/// `T` keeps Qlib's generic base-decision capability without allowing non-order values to cross
/// an order-execution API that requires [`TradeDecision<Order>`].
pub struct TradeDecision<T> {
    items: Vec<T>,
    start_time: NaiveDateTime,
    end_time: NaiveDateTime,
    total_step: Option<i64>,
    trade_range: Option<SharedTradeRange>,
}

impl<T> TradeDecision<T> {
    /// Construct a generic decision whose values are already normalized by its producer.
    #[must_use]
    pub fn from_items(
        items: Vec<T>,
        start_time: NaiveDateTime,
        end_time: NaiveDateTime,
        trade_range: Option<SharedTradeRange>,
    ) -> Self {
        Self {
            items,
            start_time,
            end_time,
            total_step: None,
            trade_range,
        }
    }

    /// Concrete decisions in stable producer order.
    #[must_use]
    pub fn items(&self) -> &[T] {
        &self.items
    }

    /// Mutable concrete decisions, matching Python's returned list mutability.
    pub fn items_mut(&mut self) -> &mut Vec<T> {
        &mut self.items
    }

    /// Replace the concrete decisions without changing interval/range metadata.
    pub fn replace_items(&mut self, items: Vec<T>) -> Vec<T> {
        std::mem::replace(&mut self.items, items)
    }

    /// Closed strategy-step start captured when the decision was created.
    #[must_use]
    pub const fn start_time(&self) -> NaiveDateTime {
        self.start_time
    }

    /// Closed strategy-step end captured when the decision was created.
    #[must_use]
    pub const fn end_time(&self) -> NaiveDateTime {
        self.end_time
    }

    /// Number of inner steps learned during executor initialization/update.
    #[must_use]
    pub const fn total_step(&self) -> Option<i64> {
        self.total_step
    }

    /// Record the inner calendar length used by subsequent range clipping.
    pub const fn set_total_step(&mut self, total_step: i64) {
        self.total_step = Some(total_step);
    }

    /// Clear the learned inner calendar length.
    pub const fn clear_total_step(&mut self) {
        self.total_step = None;
    }

    /// Borrow the configured range as the stable domain trait.
    #[must_use]
    pub fn trade_range(&self) -> Option<&dyn TradeRange> {
        self.trade_range.as_deref()
    }

    /// Borrow the owned range handle for identity-sensitive propagation.
    #[must_use]
    pub const fn shared_trade_range(&self) -> Option<&SharedTradeRange> {
        self.trade_range.as_ref()
    }

    /// Replace or clear the decision-level range.
    pub fn set_trade_range(&mut self, trade_range: Option<SharedTradeRange>) {
        self.trade_range = trade_range;
    }

    /// Propagate this range only when the inner decision has none, matching `mod_inner_decision`.
    pub fn propagate_trade_range_to<U>(&self, inner: &mut TradeDecision<U>) {
        if inner.trade_range.is_none() {
            inner.trade_range.clone_from(&self.trade_range);
        }
    }

    /// Package the captured interval and range for report baseline calculation.
    #[must_use]
    pub fn base_price_step(&self) -> BasePriceStep<'_> {
        BasePriceStep {
            start_time: self.start_time,
            end_time: self.end_time,
            trade_range: self.trade_range(),
        }
    }

    /// Resolve the closed step range, applying Python-compatible fallback and clipping rules.
    ///
    /// # Errors
    ///
    /// Returns [`TradeDecisionError::MissingRange`] when no range can be resolved and the caller
    /// omitted a fallback. Other range/provider failures retain their typed source.
    pub fn range_limit(
        &self,
        calendar: Option<&dyn TradeCalendarRange>,
        default: RangeLimitDefault,
    ) -> Result<Option<(i64, i64)>, TradeDecisionError> {
        let range = match self.trade_range() {
            Some(range) => match range.range_indices(calendar) {
                Ok(range) => range,
                Err(TradeRangeError::MissingCalendar) => return unavailable_range(default),
                Err(error) => return Err(error.into()),
            },
            None => return unavailable_range(default),
        };
        Ok(Some(clip_range_to_total(range, self.total_step)))
    }
}

impl TradeDecision<Order> {
    /// Construct an order decision and fill only missing order interval endpoints.
    #[must_use]
    pub fn from_orders(
        mut orders: Vec<Order>,
        start_time: NaiveDateTime,
        end_time: NaiveDateTime,
        trade_range: Option<SharedTradeRange>,
    ) -> Self {
        fill_missing_order_intervals(&mut orders, start_time, end_time);
        Self::from_items(orders, start_time, end_time, trade_range)
    }

    /// Orders in their original strategy order.
    #[must_use]
    pub fn orders(&self) -> &[Order] {
        self.items()
    }

    /// Mutable order list, matching `TradeDecisionWO.get_decision()` list mutability.
    pub fn orders_mut(&mut self) -> &mut Vec<Order> {
        self.items_mut()
    }

    /// Replace orders and apply the same missing-time normalization used at construction.
    pub fn replace_orders(&mut self, mut orders: Vec<Order>) -> Vec<Order> {
        fill_missing_order_intervals(&mut orders, self.start_time, self.end_time);
        self.replace_items(orders)
    }

    /// Whether every order amount is at most Qlib's strict non-empty threshold.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        !self
            .items
            .iter()
            .any(|order| order.amount() > EMPTY_ORDER_AMOUNT)
    }
}

/// Typed alias for the order-bearing `TradeDecisionWO` surface.
pub type OrderTradeDecision = TradeDecision<Order>;

/// Decision that can never contain a concrete value.
pub struct EmptyTradeDecision {
    core: TradeDecision<Infallible>,
}

impl EmptyTradeDecision {
    /// Construct an always-empty decision while retaining step and range metadata.
    #[must_use]
    pub fn new(
        start_time: NaiveDateTime,
        end_time: NaiveDateTime,
        trade_range: Option<SharedTradeRange>,
    ) -> Self {
        Self {
            core: TradeDecision::from_items(Vec::new(), start_time, end_time, trade_range),
        }
    }

    /// Empty concrete decision slice.
    #[must_use]
    pub fn items(&self) -> &[Infallible] {
        self.core.items()
    }

    /// Empty decisions remain empty regardless of metadata.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        true
    }

    /// Access shared decision metadata and range operations.
    #[must_use]
    pub const fn core(&self) -> &TradeDecision<Infallible> {
        &self.core
    }

    /// Mutably access shared decision metadata for propagation/update integration.
    pub const fn core_mut(&mut self) -> &mut TradeDecision<Infallible> {
        &mut self.core
    }
}

/// Object-safe order-decision boundary for executors and report orchestration plugins.
pub trait OrderDecision: Send + Sync {
    /// Orders in stable strategy order.
    fn orders(&self) -> &[Order];
    /// Mutable orders used by an executor to record deal results in place.
    fn orders_mut(&mut self) -> &mut [Order];
    /// Closed step start.
    fn start_time(&self) -> NaiveDateTime;
    /// Closed step end.
    fn end_time(&self) -> NaiveDateTime;
    /// Optional decision-level trade range.
    fn trade_range(&self) -> Option<&dyn TradeRange>;

    /// Qlib-compatible empty check.
    fn is_empty(&self) -> bool {
        !self
            .orders()
            .iter()
            .any(|order| order.amount() > EMPTY_ORDER_AMOUNT)
    }

    /// Convert the decision metadata to the existing baseline step DTO.
    fn base_price_step(&self) -> BasePriceStep<'_> {
        BasePriceStep {
            start_time: self.start_time(),
            end_time: self.end_time(),
            trade_range: self.trade_range(),
        }
    }
}

impl OrderDecision for OrderTradeDecision {
    fn orders(&self) -> &[Order] {
        self.orders()
    }

    fn orders_mut(&mut self) -> &mut [Order] {
        self.orders_mut()
    }

    fn start_time(&self) -> NaiveDateTime {
        self.start_time
    }

    fn end_time(&self) -> NaiveDateTime {
        self.end_time
    }

    fn trade_range(&self) -> Option<&dyn TradeRange> {
        self.trade_range()
    }
}

impl OrderDecision for EmptyTradeDecision {
    fn orders(&self) -> &[Order] {
        &[]
    }

    fn orders_mut(&mut self) -> &mut [Order] {
        &mut []
    }

    fn start_time(&self) -> NaiveDateTime {
        self.core.start_time
    }

    fn end_time(&self) -> NaiveDateTime {
        self.core.end_time
    }

    fn trade_range(&self) -> Option<&dyn TradeRange> {
        self.core.trade_range()
    }
}

pub(crate) fn clip_range_to_total(
    (mut start_idx, mut end_idx): (i64, i64),
    total_step: Option<i64>,
) -> (i64, i64) {
    if let Some(total_step) = total_step
        && (start_idx < 0 || end_idx >= total_step)
    {
        warn!(
            start_idx,
            end_idx, total_step, "decision range exceeds the total step count and will be clipped"
        );
        start_idx = start_idx.max(0);
        end_idx = end_idx.min(total_step - 1);
    }
    (start_idx, end_idx)
}

pub(crate) fn unavailable_range(
    default: RangeLimitDefault,
) -> Result<Option<(i64, i64)>, TradeDecisionError> {
    match default {
        RangeLimitDefault::Error => Err(TradeDecisionError::MissingRange),
        RangeLimitDefault::Value(value) => Ok(value),
    }
}

fn fill_missing_order_intervals(
    orders: &mut [Order],
    start_time: NaiveDateTime,
    end_time: NaiveDateTime,
) {
    for order in orders {
        order.fill_missing_interval(start_time, end_time);
    }
}
