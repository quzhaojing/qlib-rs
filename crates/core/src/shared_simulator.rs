//! Simulator order batches and results retaining live, mutable order identities.

use std::{
    cmp::Reverse,
    sync::{Arc, Mutex, RwLock},
};

use thiserror::Error;

use crate::decision_construction::{DecisionOrderItem, SharedDecisionOrders};
use crate::{Order, SimulatorCollectionError, SimulatorExecutorError, SimulatorTradeType};

/// Native shared order object, retained independently of any decision/result list.
pub type SharedSimulatorOrder = Arc<RwLock<Order>>;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SharedSimulatorError {
    #[error("decision order list lock is poisoned")]
    ListPoisoned,
    #[error("decision item {0} is not an Order")]
    InvalidOrder(usize),
    #[error("order {0} lock is poisoned")]
    OrderPoisoned(usize),
    #[error(transparent)]
    Iterator(#[from] SimulatorExecutorError),
    #[error(transparent)]
    Collection(#[from] SimulatorCollectionError),
}

/// Extract handles before validating mode; stable sorting changes only this new list.
///
/// # Errors
/// Returns the first extraction, mode or sort-key access failure in source order.
pub fn shared_simulator_order_iterator(
    decision: &SharedDecisionOrders,
    trade_type: &str,
) -> Result<Vec<SharedSimulatorOrder>, SharedSimulatorError> {
    let mut orders = Vec::new();
    for (index, item) in decision
        .read()
        .map_err(|_| SharedSimulatorError::ListPoisoned)?
        .iter()
        .enumerate()
    {
        let DecisionOrderItem::Order(order) = item else {
            return Err(SharedSimulatorError::InvalidOrder(index));
        };
        orders.push(Arc::clone(order));
    }
    match SimulatorTradeType::parse_text(trade_type)? {
        SimulatorTradeType::Serial => Ok(orders),
        SimulatorTradeType::Parallel => {
            let mut keyed = Vec::with_capacity(orders.len());
            for (index, order) in orders.into_iter().enumerate() {
                let direction = order
                    .read()
                    .map_err(|_| SharedSimulatorError::OrderPoisoned(index))?
                    .direction();
                keyed.push((Reverse(direction.value()), order));
            }
            keyed.sort_by_key(|(key, _)| *key);
            Ok(keyed.into_iter().map(|(_, order)| order).collect())
        }
    }
}

/// Trade amounts/prices are captured at execution, but the order remains the original live object.
#[derive(Debug)]
pub struct SharedSimulatorExecution {
    pub order: SharedSimulatorOrder,
    pub trade_value: f64,
    pub trade_cost: f64,
    pub trade_price: f64,
}

/// A source execution tuple: list copies retain this identity and the original mutable order.
pub type SharedSimulatorExecutionHandle = Arc<SharedSimulatorExecution>;

/// One mutable list shared by atomic reporting, return callbacks and nested parent hooks.
pub type SharedExecutionResult = Arc<Mutex<Vec<SharedSimulatorExecutionHandle>>>;

pub struct SharedSimulatorCollection {
    pub(crate) executions: Vec<SharedSimulatorExecutionHandle>,
}

impl SharedSimulatorCollection {
    /// Move the collector's list into its retained transport before any lifecycle callbacks.
    #[must_use]
    pub fn into_shared(self) -> SharedExecutionResult {
        Arc::new(Mutex::new(self.executions))
    }

    /// Mutate the original result list, for source-compatible account and return-sink callbacks.
    pub fn execution_result_mut(&mut self) -> &mut Vec<SharedSimulatorExecutionHandle> {
        &mut self.executions
    }

    #[must_use]
    pub fn execution_result(&self) -> &[SharedSimulatorExecutionHandle] {
        &self.executions
    }

    /// The same list allocation as `execution_result`, not a separately copied result.
    #[must_use]
    pub fn trade_info(&self) -> &[SharedSimulatorExecutionHandle] {
        &self.executions
    }
}
