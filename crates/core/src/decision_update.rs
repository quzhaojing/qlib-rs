//! Ordered calendar/strategy update hook for one trade decision.

use std::sync::{Arc, RwLock};
use thiserror::Error;

use crate::{NestedCalendar, TradeDecision};

/// Retainable decision identity; the originating strategy is itself a retained shared object.
pub type SharedLiveDecision<S, D> =
    Arc<RwLock<crate::decision_construction::SharedOrderDecisionConstruction<Arc<S>, D>>>;

/// Type-erased transport retaining the original decision allocation, not a snapshot.
pub type LiveDecisionHandle = Arc<dyn LiveDecision>;

/// Object-safe capabilities for shared decision transport between executor plugins.
/// This is an in-process Rust interface, not a stable dynamic-library ABI.
pub trait LiveDecision: Send + Sync {
    /// Read the current source attribute state, including incomplete initialization.
    ///
    /// # Errors
    /// Returns a poisoned decision access.
    fn total_step(
        &self,
    ) -> Result<crate::decision_construction::DecisionTotalStep, LiveDecisionAccessError>;

    /// Resolve the retained rule without a decision guard; clip using the state after callback.
    ///
    /// # Errors
    /// Returns reached metadata, range-provider, fallback, or post-callback access failures.
    fn range_limit(
        &self,
        calendar: Option<&dyn crate::TradeCalendarRange>,
        default: crate::RangeLimitDefault,
    ) -> Result<Option<(i64, i64)>, LiveDecisionAccessError> {
        let Some(rule) = self.base()?.trade_range else {
            return Ok(crate::trade_decision::unavailable_range(default)?);
        };
        let range = match rule.range_indices(calendar) {
            Ok(range) => range,
            Err(crate::TradeRangeError::MissingCalendar) => {
                return Ok(crate::trade_decision::unavailable_range(default)?);
            }
            Err(error) => return Err(crate::TradeDecisionError::from(error).into()),
        };
        let total = match self.total_step()? {
            crate::decision_construction::DecisionTotalStep::Value(value) => Some(value),
            _ => None,
        };
        Ok(Some(crate::trade_decision::clip_range_to_total(
            range, total,
        )))
    }

    /// Source default outer-to-inner propagation; inspect the inner before reading the outer.
    /// No guard is retained across another decision's plugin method, including self-aliasing.
    ///
    /// # Errors
    /// Returns the first reached inner/outer access or assignment failure.
    fn modify_inner_decision(
        &self,
        inner: &LiveDecisionHandle,
    ) -> Result<(), LiveDecisionAccessError> {
        if inner.base()?.trade_range.is_none() {
            inner.inherit_range(self.base()?.trade_range)?;
        }
        Ok(())
    }

    /// Read the current initialized base metadata, retaining its range object.
    /// This short-lived metadata view is not a replacement decision for transport.
    ///
    /// # Errors
    /// Returns a poisoned decision or missing base attribute.
    fn base(
        &self,
    ) -> Result<crate::decision_construction::ConstructedDecisionBase, LiveDecisionAccessError>;

    /// Apply the source default propagation rule only when the inner range is None.
    ///
    /// # Errors
    /// Returns a poisoned decision or missing base attribute.
    fn inherit_range(
        &self,
        range: Option<crate::SharedTradeRange>,
    ) -> Result<(), LiveDecisionAccessError>;

    /// Acquire the current original list handle without copying its orders.
    ///
    /// # Errors
    /// Returns poisoned decision access or an uninitialized order list.
    fn orders(
        &self,
    ) -> Result<crate::decision_construction::SharedDecisionOrders, LiveDecisionAccessError>;

    /// Read live emptiness with source short-circuit ordering.
    ///
    /// # Errors
    /// Returns poisoned or missing state reached by the source operation.
    fn is_empty(&self) -> Result<bool, LiveDecisionAccessError>;

    /// Dispatch through the originating strategy without holding a decision guard.
    ///
    /// # Errors
    /// Preserves calendar, decision-lock and strategy failures in source order.
    fn update(
        self: Arc<Self>,
        calendar: &dyn DecisionUpdateCalendar,
    ) -> Result<Option<LiveDecisionHandle>, SharedDecisionUpdateError>;
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum LiveDecisionAccessError {
    #[error(transparent)]
    Range(#[from] crate::TradeDecisionError),
    #[error("decision base has not been initialized")]
    MissingBase,
    #[error("decision lock poisoned")]
    DecisionPoisoned,
    #[error(transparent)]
    Access(#[from] crate::decision_construction::DecisionAccessError),
}

impl<S, D> LiveDecision
    for RwLock<crate::decision_construction::SharedOrderDecisionConstruction<Arc<S>, D>>
where
    S: SharedDecisionUpdateStrategy<D> + Send + Sync + 'static,
    D: Send + Sync + 'static,
{
    fn total_step(
        &self,
    ) -> Result<crate::decision_construction::DecisionTotalStep, LiveDecisionAccessError> {
        Ok(self
            .read()
            .map_err(|_| LiveDecisionAccessError::DecisionPoisoned)?
            .total_step)
    }

    fn base(
        &self,
    ) -> Result<crate::decision_construction::ConstructedDecisionBase, LiveDecisionAccessError>
    {
        let decision = self
            .read()
            .map_err(|_| LiveDecisionAccessError::DecisionPoisoned)?;
        let base = decision
            .base
            .as_ref()
            .ok_or(LiveDecisionAccessError::MissingBase)?;
        Ok(crate::decision_construction::ConstructedDecisionBase {
            start_time: base.start_time,
            end_time: base.end_time,
            trade_range: base.trade_range.clone(),
        })
    }

    fn inherit_range(
        &self,
        range: Option<crate::SharedTradeRange>,
    ) -> Result<(), LiveDecisionAccessError> {
        let mut decision = self
            .write()
            .map_err(|_| LiveDecisionAccessError::DecisionPoisoned)?;
        let base = decision
            .base
            .as_mut()
            .ok_or(LiveDecisionAccessError::MissingBase)?;
        if base.trade_range.is_none() {
            base.trade_range = range;
        }
        Ok(())
    }

    fn orders(
        &self,
    ) -> Result<crate::decision_construction::SharedDecisionOrders, LiveDecisionAccessError> {
        let decision = self
            .read()
            .map_err(|_| LiveDecisionAccessError::DecisionPoisoned)?;
        Ok(Arc::clone(decision.get_decision().ok_or(
            crate::decision_construction::DecisionAccessError::MissingOrders,
        )?))
    }

    fn is_empty(&self) -> Result<bool, LiveDecisionAccessError> {
        let decision = self
            .read()
            .map_err(|_| LiveDecisionAccessError::DecisionPoisoned)?;
        Ok(decision.is_empty()?)
    }

    fn update(
        self: Arc<Self>,
        calendar: &dyn DecisionUpdateCalendar,
    ) -> Result<Option<LiveDecisionHandle>, SharedDecisionUpdateError> {
        Ok(update_shared_trade_decision(&self, calendar)?
            .map(|decision| decision as LiveDecisionHandle))
    }
}

/// Source update callback, invoked without a framework-held decision or order guard.
pub trait SharedDecisionUpdateStrategy<D> {
    /// Mutate the original decision and return None, itself, or another retained decision.
    ///
    /// # Errors
    /// Returns the callback failure without undoing mutations already made.
    fn update_trade_decision(
        &self,
        decision: &SharedLiveDecision<Self, D>,
        calendar: &dyn DecisionUpdateCalendar,
    ) -> Result<Option<SharedLiveDecision<Self, D>>, DecisionUpdateStrategyError>;
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SharedDecisionUpdateError {
    #[error(transparent)]
    Calendar(#[from] DecisionUpdateCalendarError),
    #[error("decision lock poisoned after calendar lookup")]
    DecisionPoisoned,
    #[error(transparent)]
    Strategy(#[from] DecisionUpdateStrategyError),
}

/// Assign the inner length, then call the decision's current originating strategy.
/// Calendar and strategy callbacks can reenter the original decision without a held guard.
///
/// # Errors
/// Returns calendar, decision-lock or strategy failure in that order; reached changes persist.
pub fn update_shared_trade_decision<S: SharedDecisionUpdateStrategy<D> + ?Sized, D>(
    decision: &SharedLiveDecision<S, D>,
    calendar: &dyn DecisionUpdateCalendar,
) -> Result<Option<SharedLiveDecision<S, D>>, SharedDecisionUpdateError> {
    let total_step = calendar.trade_len()?;
    let strategy = {
        let mut current = decision
            .write()
            .map_err(|_| SharedDecisionUpdateError::DecisionPoisoned)?;
        current.total_step = crate::decision_construction::DecisionTotalStep::Value(total_step);
        Arc::clone(&current.strategy)
    };
    Ok(strategy.update_trade_decision(decision, calendar)?)
}

/// Narrow calendar capability consumed by `BaseTradeDecision.update`.
pub trait DecisionUpdateCalendar {
    /// Return the total number of executable inner steps.
    ///
    /// # Errors
    ///
    /// Returns a calendar state or transport failure before the strategy hook runs.
    fn trade_len(&self) -> Result<i64, DecisionUpdateCalendarError>;
}

/// Failure while reading the inner calendar length.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("decision update calendar failed: {message}")]
pub struct DecisionUpdateCalendarError {
    pub message: String,
}

/// Adapter that exposes an existing nested calendar through the narrow update boundary.
pub struct NestedDecisionCalendarAdapter<'a> {
    calendar: &'a dyn NestedCalendar,
}

impl<'a> NestedDecisionCalendarAdapter<'a> {
    #[must_use]
    pub const fn new(calendar: &'a dyn NestedCalendar) -> Self {
        Self { calendar }
    }
}

impl DecisionUpdateCalendar for NestedDecisionCalendarAdapter<'_> {
    fn trade_len(&self) -> Result<i64, DecisionUpdateCalendarError> {
        self.calendar
            .trade_len()
            .map_err(|error| DecisionUpdateCalendarError {
                message: error.message,
            })
    }
}

/// Strategy result corresponding to Python's optional returned decision.
pub enum DecisionUpdate<T> {
    /// Python returned `None`; callers keep the current decision without running replacement hooks.
    Unchanged,
    /// Python returned `self`; callers keep the same identity but do run replacement hooks.
    Current,
    /// Python returned another decision object for the caller to install.
    Replacement(TradeDecision<T>),
}

/// Replaceable strategy hook that may mutate, retain, or replace a decision.
pub trait DecisionUpdateStrategy<T> {
    /// Run the strategy callback after `total_step` has been updated.
    ///
    /// # Errors
    ///
    /// Returns the strategy's typed callback failure. Mutations already made to the current
    /// decision remain visible, matching Python object semantics.
    fn update_trade_decision(
        &mut self,
        decision: &mut TradeDecision<T>,
        calendar: &dyn DecisionUpdateCalendar,
    ) -> Result<DecisionUpdate<T>, DecisionUpdateStrategyError>;
}

/// Failure returned by a replaceable strategy update hook.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("decision update strategy failed: {message}")]
pub struct DecisionUpdateStrategyError {
    pub message: String,
}

/// Ordered failures from the decision update operation.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum DecisionUpdateError {
    #[error(transparent)]
    Calendar(#[from] DecisionUpdateCalendarError),
    #[error(transparent)]
    Strategy(#[from] DecisionUpdateStrategyError),
}

/// Update the current decision with the inner calendar, then invoke its originating strategy.
///
/// The calendar length is stored before the strategy is called. A calendar failure therefore
/// leaves both the decision and strategy untouched, while a strategy failure retains the newly
/// stored length and any callback mutations.
///
/// # Errors
///
/// Returns [`DecisionUpdateError::Calendar`] before calling the strategy, or
/// [`DecisionUpdateError::Strategy`] after `total_step` has been written.
pub fn update_trade_decision<T>(
    decision: &mut TradeDecision<T>,
    calendar: &dyn DecisionUpdateCalendar,
    strategy: &mut dyn DecisionUpdateStrategy<T>,
) -> Result<DecisionUpdate<T>, DecisionUpdateError> {
    let total_step = calendar.trade_len()?;
    decision.set_total_step(total_step);
    Ok(strategy.update_trade_decision(decision, calendar)?)
}
