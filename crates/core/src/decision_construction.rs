//! Shared, observable initialization of source order decisions, including failed initialization.

use std::{
    any::Any,
    sync::{Arc, RwLock},
};

use chrono::NaiveDateTime;
use thiserror::Error;

use crate::{IdxTradeRange, Order, SaoeCalendar, SaoePluginError, SharedTradeRange};

/// Original objects in a source order list. Invalid values are retained until their turn is checked.
pub enum DecisionOrderItem {
    Order(Arc<RwLock<Order>>),
    Other(Arc<dyn Any + Send + Sync>),
}

/// Retains both the caller's list identity and individual mutable order identities.
pub type SharedDecisionOrders = Arc<RwLock<Vec<DecisionOrderItem>>>;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum DecisionAccessError {
    #[error("decision item {0} is not an Order")]
    InvalidOrder(usize),
    #[error("decision order list has not been initialized")]
    MissingOrders,
    #[error("decision order list lock is poisoned")]
    ListPoisoned,
    #[error("decision order {0} lock is poisoned")]
    OrderPoisoned(usize),
}

/// Tuple normalization is performed after the first calendar read, as in the source constructor.
pub enum DecisionRangeInput {
    Indices(i64, i64),
    Rule(SharedTradeRange),
}

/// Strategy capability observed by the decision constructor.
pub trait DecisionConstructionStrategy {
    type Error;

    /// Read this strategy's current calendar interval; calls must not be cached.
    ///
    /// # Errors
    /// Returns the original calendar failure.
    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), Self::Error>;
}

impl<S: DecisionConstructionStrategy + ?Sized> DecisionConstructionStrategy for Arc<S> {
    type Error = S::Error;

    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), Self::Error> {
        (**self).step_time()
    }
}

/// Associates a real strategy identity with its currently supplied SAOE calendar.
pub struct SaoeDecisionOrigin<'a, S: ?Sized> {
    pub strategy: &'a S,
    pub calendar: &'a dyn SaoeCalendar,
}

impl<S: ?Sized> DecisionConstructionStrategy for SaoeDecisionOrigin<'_, S> {
    type Error = SaoePluginError;

    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), Self::Error> {
        self.calendar.step_time()
    }
}

/// Metadata that becomes present only after the first calendar lookup succeeds.
pub struct ConstructedDecisionBase {
    pub start_time: NaiveDateTime,
    pub end_time: NaiveDateTime,
    pub trade_range: Option<SharedTradeRange>,
}

/// Source attribute state: absent, assigned None, or assigned an inner calendar length.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DecisionTotalStep {
    #[default]
    Missing,
    Unset,
    Value(i64),
}

/// Failures retain all changes already made to the decision and caller-owned orders.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum DecisionConstructionError<E> {
    #[error("decision calendar failed: {0}")]
    Calendar(E),
    #[error("decision item {0} is not an Order")]
    InvalidOrder(usize),
    #[error("decision order list lock is poisoned")]
    ListPoisoned,
    #[error("decision order {0} lock is poisoned")]
    OrderPoisoned(usize),
}

/// Source object state, intentionally inspectable after a failed `__init__` equivalent.
///
/// `None` means the corresponding constructor assignment has not occurred. In particular,
/// `Some(None)` for `details: Option<Option<D>>` represents an assigned Python `None` payload.
/// Reinitialization overwrites fields only when the source would assign them, so an old detail
/// payload survives a later failed parent initialization.
pub struct SharedOrderDecisionConstruction<S, D> {
    pub strategy: S,
    pub base: Option<ConstructedDecisionBase>,
    pub total_step: DecisionTotalStep,
    pub orders: Option<SharedDecisionOrders>,
    pub details: Option<D>,
}

impl<S, D> SharedOrderDecisionConstruction<S, D> {
    /// Perform the calendar assignment stage of source update, even before initialization.
    /// The strategy callback must run after this stage; no callback is invoked here.
    ///
    /// # Errors
    /// Returns the calendar failure without changing the existing attribute state.
    pub fn refresh_total_step(
        &mut self,
        calendar: &dyn crate::DecisionUpdateCalendar,
    ) -> Result<(), crate::DecisionUpdateCalendarError> {
        self.total_step = DecisionTotalStep::Value(calendar.trade_len()?);
        Ok(())
    }

    /// Return the original list handle, including after partial initialization.
    /// None represents the source object's still-missing `order_list` attribute.
    #[must_use]
    pub fn get_decision(&self) -> Option<&SharedDecisionOrders> {
        self.orders.as_ref()
    }

    /// Observe current list membership and order amounts in source short-circuit order.
    /// A non-order encountered before any positive order makes the decision empty.
    ///
    /// # Errors
    /// Returns missing-list or reached poisoned-lock errors; skipped items are never locked.
    pub fn is_empty(&self) -> Result<bool, DecisionAccessError> {
        let orders = self
            .get_decision()
            .ok_or(DecisionAccessError::MissingOrders)?;
        let items = orders
            .read()
            .map_err(|_| DecisionAccessError::ListPoisoned)?;
        for (index, item) in items.iter().enumerate() {
            let DecisionOrderItem::Order(order) = item else {
                return Ok(true);
            };
            let amount = order
                .read()
                .map_err(|_| DecisionAccessError::OrderPoisoned(index))?
                .amount();
            if amount > crate::trade_decision::EMPTY_ORDER_AMOUNT {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Store the strategy identity before any calendar access.
    #[must_use]
    pub const fn new(strategy: S) -> Self {
        Self {
            strategy,
            base: None,
            total_step: DecisionTotalStep::Missing,
            orders: None,
            details: None,
        }
    }
}

impl<S: DecisionConstructionStrategy, D> SharedOrderDecisionConstruction<S, D> {
    /// Initialize the actual shared order list, then attach the details payload.
    ///
    /// The list and individual orders remain aliased, never cloned into independent order values.
    /// Every list element is checked immediately before its missing endpoints are filled. The
    /// list read guard serializes this operation against concurrent list replacement; no Python
    /// thread scheduling equivalence is claimed.
    ///
    /// # Errors
    /// Returns the first calendar, invalid-item or poisoned-lock failure. Previously written
    /// metadata, list identity, individual order mutations and old details remain observable.
    pub fn initialize(
        &mut self,
        orders: &SharedDecisionOrders,
        trade_range: Option<DecisionRangeInput>,
        details: D,
    ) -> Result<(), DecisionConstructionError<S::Error>> {
        let (start_time, end_time) = self
            .strategy
            .step_time()
            .map_err(DecisionConstructionError::Calendar)?;
        self.total_step = DecisionTotalStep::Unset;
        let trade_range = trade_range.map(|range| match range {
            DecisionRangeInput::Indices(start, end) => {
                Arc::new(IdxTradeRange::new(start, end)) as SharedTradeRange
            }
            DecisionRangeInput::Rule(rule) => rule,
        });
        self.base = Some(ConstructedDecisionBase {
            start_time,
            end_time,
            trade_range,
        });
        self.orders = Some(Arc::clone(orders));
        let (order_start, order_end) = self
            .strategy
            .step_time()
            .map_err(DecisionConstructionError::Calendar)?;
        let items = orders
            .read()
            .map_err(|_| DecisionConstructionError::ListPoisoned)?;
        for (index, item) in items.iter().enumerate() {
            let DecisionOrderItem::Order(order) = item else {
                return Err(DecisionConstructionError::InvalidOrder(index));
            };
            order
                .write()
                .map_err(|_| DecisionConstructionError::OrderPoisoned(index))?
                .fill_missing_interval(order_start, order_end);
        }
        self.details = Some(details);
        Ok(())
    }
}
