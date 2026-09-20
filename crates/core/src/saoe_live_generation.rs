//! Live order-list reads for SAOE generation, before source decision initialization.
use std::sync::{Arc, RwLock};

use thiserror::Error;

use crate::decision_construction::{DecisionAccessError, DecisionOrderItem, SharedDecisionOrders};
use crate::decision_construction::{
    DecisionConstructionError, DecisionConstructionStrategy, SharedOrderDecisionConstruction,
};
use crate::decision_update::SharedLiveDecision;
use crate::decision_update::{LiveDecisionAccessError, LiveDecisionHandle};
use crate::{
    LiveSaoeState, Order, SaoeDecisionCalendar, SaoeInterpreterError, SaoeOrderFactory,
    SaoePluginError, SaoePolicyDecision, SaoePolicyPipeline, SaoeState, SaoeTradeDetail,
};

#[derive(Debug, Error)]
pub enum LiveSaoeGenerationError {
    #[error(transparent)]
    Decision(#[from] LiveDecisionAccessError),
    #[error(transparent)]
    Orders(#[from] DecisionAccessError),
    #[error(transparent)]
    Pipeline(#[from] SaoeInterpreterError),
    #[error("SAOE state lookup failed: {0}")]
    State(SaoePluginError),
    #[error("SAOE order factory failed: {0}")]
    Factory(SaoePluginError),
    #[error("SAOE detail calendar failed: {0}")]
    Calendar(SaoePluginError),
}

/// Arguments for the existing shared decision initializer. Not a completed decision:
/// its originating strategy and two constructor calendar reads must still be supplied.
pub struct LiveSaoeDecisionParts {
    pub orders: SharedDecisionOrders,
    pub details: Vec<SaoeTradeDetail>,
}

impl LiveSaoeDecisionParts {
    /// Initialize caller-owned state through the existing source-ordered constructor.
    /// The supplied strategy identity is retained; failed initialization remains inspectable.
    ///
    /// # Errors
    /// Returns the first constructor failure, preserving reached state and order mutations.
    pub fn initialize_into<S: DecisionConstructionStrategy + ?Sized>(
        self,
        decision: &mut SharedOrderDecisionConstruction<Arc<S>, Vec<SaoeTradeDetail>>,
    ) -> Result<(), DecisionConstructionError<S::Error>> {
        decision.initialize(&self.orders, None, self.details)
    }

    /// Publish a shared decision only after successful initialization with the original strategy.
    /// No origin proxy or copied order list is substituted. Call `initialize_into` instead
    /// when failed constructor state itself must be inspected. Retained order handles always
    /// observe reached mutations even when this convenience constructor fails.
    ///
    /// # Errors
    /// Returns the first calendar, list or order initialization failure.
    pub fn into_decision<S: DecisionConstructionStrategy + ?Sized>(
        self,
        strategy: Arc<S>,
    ) -> Result<SharedLiveDecision<S, Vec<SaoeTradeDetail>>, DecisionConstructionError<S::Error>>
    {
        let mut decision = SharedOrderDecisionConstruction::new(strategy);
        self.initialize_into(&mut decision)?;
        Ok(Arc::new(RwLock::new(decision)))
    }
}

/// Owns replaceable inference, order construction and calendar plugins.
pub struct LiveSaoeGeneration {
    pipeline: SaoePolicyPipeline,
    orders: Box<dyn SaoeOrderFactory>,
    calendar: Arc<dyn SaoeDecisionCalendar>,
}

impl LiveSaoeGeneration {
    #[must_use]
    pub fn new(
        pipeline: SaoePolicyPipeline,
        orders: Box<dyn SaoeOrderFactory>,
        calendar: Arc<dyn SaoeDecisionCalendar>,
    ) -> Self {
        Self {
            pipeline,
            orders,
            calendar,
        }
    }

    /// Generate constructor arguments using three separate reads of the original outer list.
    /// Each loop retains that list but observes in-place edits before fetching its next item.
    /// State callbacks receive original order handles, without a list or order guard held.
    ///
    /// # Errors
    /// Returns the first reached decision, lock or plugin failure without rolling back effects.
    pub fn generate(
        &mut self,
        outer: &LiveDecisionHandle,
        mut state: impl FnMut(&Arc<RwLock<Order>>) -> Result<SaoeState, SaoePluginError>,
    ) -> Result<LiveSaoeDecisionParts, LiveSaoeGenerationError> {
        let mut cursor = Cursor::new(outer.orders()?);
        let decisions = self.pipeline.decisions_from(|| {
            let Some((index, item)) = cursor.next()? else {
                return Ok(None);
            };
            let order = as_order(index, item)?;
            state(&order)
                .map(Some)
                .map_err(LiveSaoeGenerationError::State)
        })?;

        self.finish(outer, decisions)
    }

    /// Generate using source-compatible live state aliases through both interpreter callbacks.
    ///
    /// # Errors
    /// Returns the first reached decision, lock, alias, or plugin failure without rollback.
    pub fn generate_live(
        &mut self,
        outer: &LiveDecisionHandle,
        mut state: impl FnMut(&Arc<RwLock<Order>>) -> Result<LiveSaoeState, SaoePluginError>,
    ) -> Result<LiveSaoeDecisionParts, LiveSaoeGenerationError> {
        let mut cursor = Cursor::new(outer.orders()?);
        let decisions = self.pipeline.decisions_from_live(|| {
            let Some((index, item)) = cursor.next()? else {
                return Ok(None);
            };
            let order = as_order(index, item)?;
            state(&order)
                .map(Some)
                .map_err(LiveSaoeGenerationError::State)
        })?;

        self.finish(outer, decisions)
    }

    fn finish(
        &mut self,
        outer: &LiveDecisionHandle,
        decisions: Vec<SaoePolicyDecision>,
    ) -> Result<LiveSaoeDecisionParts, LiveSaoeGenerationError> {
        let mut cursor = Cursor::new(outer.orders()?);
        let mut children = Vec::new();
        let mut volumes = decisions.iter();
        while let Some((index, item)) = cursor.next()? {
            // Source zip reads the order iterator first. Zero volumes do not access order fields.
            let Some(decision) = volumes.next() else {
                break;
            };
            if decision.execution_volume != 0.0 {
                let order = as_order(index, item)?;
                let (stock, direction) = {
                    let order = order
                        .read()
                        .map_err(|_| DecisionAccessError::OrderPoisoned(index))?;
                    (order.stock_id().to_owned(), order.direction())
                };
                let child = self
                    .orders
                    .create(&stock, Some(decision.execution_volume), direction)
                    .map_err(LiveSaoeGenerationError::Factory)?;
                children.push(DecisionOrderItem::Order(Arc::new(RwLock::new(child))));
            }
        }

        let mut cursor = Cursor::new(outer.orders()?);
        let mut details = Vec::new();
        for decision in decisions {
            let Some((index, item)) = cursor.next()? else {
                break;
            };
            let order = as_order(index, item)?;
            let instrument = order
                .read()
                .map_err(|_| DecisionAccessError::OrderPoisoned(index))?
                .stock_id()
                .to_owned();
            let datetime = self
                .calendar
                .step_time()
                .map_err(LiveSaoeGenerationError::Calendar)?
                .0;
            let frequency = self
                .calendar
                .frequency()
                .map_err(LiveSaoeGenerationError::Calendar)?;
            details.push(SaoeTradeDetail {
                instrument,
                datetime,
                frequency,
                execution_volume: decision.execution_volume,
                action: Some(decision.action),
            });
        }
        Ok(LiveSaoeDecisionParts {
            orders: Arc::new(RwLock::new(children)),
            details,
        })
    }
}

pub(crate) struct Cursor {
    orders: SharedDecisionOrders,
    index: usize,
}

impl Cursor {
    pub(crate) fn new(orders: SharedDecisionOrders) -> Self {
        Self { orders, index: 0 }
    }

    pub(crate) fn next(
        &mut self,
    ) -> Result<Option<(usize, DecisionOrderItem)>, DecisionAccessError> {
        let items = self
            .orders
            .read()
            .map_err(|_| DecisionAccessError::ListPoisoned)?;
        let Some(item) = items.get(self.index) else {
            return Ok(None);
        };
        let item = match item {
            DecisionOrderItem::Order(order) => DecisionOrderItem::Order(Arc::clone(order)),
            DecisionOrderItem::Other(value) => DecisionOrderItem::Other(Arc::clone(value)),
        };
        let index = self.index;
        self.index += 1;
        Ok(Some((index, item)))
    }
}

pub(crate) fn as_order(
    index: usize,
    item: DecisionOrderItem,
) -> Result<Arc<RwLock<Order>>, DecisionAccessError> {
    match item {
        DecisionOrderItem::Order(order) => Ok(order),
        DecisionOrderItem::Other(_) => Err(DecisionAccessError::InvalidOrder(index)),
    }
}
