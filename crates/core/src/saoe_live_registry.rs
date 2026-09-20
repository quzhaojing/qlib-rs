//! SAOE adapter lifecycle retaining live order, decision and range identities.
use std::sync::{Arc, RwLock};

use chrono::NaiveDateTime;
use indexmap::IndexMap;
use thiserror::Error;

use crate::decision_construction::DecisionAccessError;
use crate::decision_update::{LiveDecisionAccessError, LiveDecisionHandle};
use crate::nested_executor::SharedNestedResult;
use crate::saoe_live_generation::{Cursor, as_order};
use crate::{
    LiveSaoeState, Order, OrderDir, OrderError, SaoePluginError, SaoeState, SharedOrderExecution,
    SharedTradeRange,
};

type Key = (String, NaiveDateTime, OrderDir);

/// An adapter owns its original order handle; lookup does not replace it with a query order.
/// Supports both owned snapshots and source-compatible live order/history aliases.
pub trait LiveSaoeStateAdapter: Send {
    /// Return original live order/history identities when the adapter supports source aliases.
    ///
    /// # Errors
    /// Returns a typed compatibility failure for legacy implementations or an observation error.
    fn live_state(&self) -> Result<LiveSaoeState, SaoePluginError> {
        Err(SaoePluginError {
            message: "live SAOE state aliases are not supported by this adapter".to_owned(),
        })
    }

    /// Materialize current state from the retained order and adapter state.
    ///
    /// # Errors
    /// Returns the first state observation/conversion failure.
    fn state(&self) -> Result<SaoeState, SaoePluginError>;

    /// Apply original execution tuple handles, after all grouping has completed.
    ///
    /// # Errors
    /// Returns the first update failure, without undoing earlier mutations.
    fn update(
        &mut self,
        executions: &[SharedOrderExecution],
        range: (i64, i64),
    ) -> Result<(), SaoePluginError>;

    /// Finalize retained metrics.
    ///
    /// # Errors
    /// Returns a metric finalization failure.
    fn finalize(&mut self) -> Result<(), SaoePluginError>;
}

/// In-process factory boundary: no order snapshots or substitute outer decisions.
pub trait LiveSaoeAdapterFactory: Send {
    /// Construct using original handles and the range captured once by reset.
    /// The registry holds no decision/list/order guard during this callback.
    ///
    /// # Errors
    /// Returns data loading or adapter construction failures.
    fn create(
        &mut self,
        order: &Arc<RwLock<Order>>,
        outer: &LiveDecisionHandle,
        range: &SharedTradeRange,
    ) -> Result<Box<dyn LiveSaoeStateAdapter>, SaoePluginError>;
}

#[derive(Debug, Error)]
pub enum LiveSaoeRegistryError {
    #[error(transparent)]
    Decision(#[from] LiveDecisionAccessError),
    #[error(transparent)]
    Access(#[from] DecisionAccessError),
    #[error(transparent)]
    Key(#[from] OrderError),
    #[error("SAOE order lock poisoned")]
    OrderPoisoned,
    #[error("SAOE execution list lock poisoned")]
    ExecutionsPoisoned,
    #[error("nonempty SAOE outer decision requires a trade range")]
    MissingRange,
    #[error("missing SAOE adapter for {0}/{1}/{2}")]
    MissingAdapter(String, NaiveDateTime, OrderDir),
    #[error("a nonpositive SAOE step length cannot receive executions")]
    UnexpectedExecutions,
    #[error("SAOE adapter creation failed: {0}")]
    Factory(SaoePluginError),
    #[error("SAOE state observation failed: {0}")]
    State(SaoePluginError),
    #[error("SAOE adapter update failed: {0}")]
    Update(SaoePluginError),
    #[error("SAOE adapter finalization failed: {0}")]
    Finalize(SaoePluginError),
}

/// Ordered registry for the live strategy. Infrastructure reset, retained outer identity and
/// last-step-range assignment belong to the owning strategy, before invoking this component.
pub struct LiveSaoeAdapterRegistry {
    factory: Box<dyn LiveSaoeAdapterFactory>,
    adapters: IndexMap<Key, Box<dyn LiveSaoeStateAdapter>>,
}

impl LiveSaoeAdapterRegistry {
    #[must_use]
    pub fn new(factory: Box<dyn LiveSaoeAdapterFactory>) -> Self {
        Self {
            factory,
            adapters: IndexMap::new(),
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.adapters.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.adapters.is_empty()
    }

    /// Replace the registry; iterate the original list with live membership, read each key only
    /// after its factory returns, and preserve earlier insertions on failure.
    ///
    /// # Errors
    /// Returns the first reached decision, range, order, key or factory failure.
    pub fn reset(
        &mut self,
        outer: Option<&LiveDecisionHandle>,
    ) -> Result<(), LiveSaoeRegistryError> {
        self.adapters = IndexMap::new();
        let Some(outer) = outer else {
            return Ok(());
        };
        if outer.is_empty()? {
            return Ok(());
        }
        let range = outer
            .base()?
            .trade_range
            .ok_or(LiveSaoeRegistryError::MissingRange)?;
        let mut cursor = Cursor::new(outer.orders()?);
        while let Some((index, item)) = cursor.next()? {
            let order = as_order(index, item)?;
            let adapter = self
                .factory
                .create(&order, outer, &range)
                .map_err(LiveSaoeRegistryError::Factory)?;
            self.adapters.insert(key(&order)?, adapter);
        }
        Ok(())
    }

    /// Locate by the query's current key; observe the adapter's own original order.
    ///
    /// # Errors
    /// Returns key access, lookup or state plugin failure.
    pub fn state(&self, order: &Arc<RwLock<Order>>) -> Result<SaoeState, LiveSaoeRegistryError> {
        let key = key(order)?;
        self.adapters
            .get(&key)
            .ok_or(LiveSaoeRegistryError::MissingAdapter(key.0, key.1, key.2))?
            .state()
            .map_err(LiveSaoeRegistryError::State)
    }

    /// Locate by the query's current key and return the adapter's retained alias state.
    ///
    /// # Errors
    /// Returns key access, lookup or live-state plugin failure.
    pub fn live_state(
        &self,
        order: &Arc<RwLock<Order>>,
    ) -> Result<LiveSaoeState, LiveSaoeRegistryError> {
        let key = key(order)?;
        self.adapters
            .get(&key)
            .ok_or(LiveSaoeRegistryError::MissingAdapter(key.0, key.1, key.2))?
            .live_state()
            .map_err(LiveSaoeRegistryError::State)
    }

    /// Group original execution handles before invoking adapters in registry order, including
    /// adapters with no matching executions. No result-list/order lock spans a plugin call.
    ///
    /// # Errors
    /// Returns invalid step length, lock/key access or the first adapter update failure.
    pub fn update(
        &mut self,
        executions: Option<&SharedNestedResult>,
        range: (i64, i64),
    ) -> Result<(), LiveSaoeRegistryError> {
        if range.1 <= range.0 {
            if let Some(rows) = executions {
                if !rows
                    .lock()
                    .map_err(|_| LiveSaoeRegistryError::ExecutionsPoisoned)?
                    .is_empty()
                {
                    return Err(LiveSaoeRegistryError::UnexpectedExecutions);
                }
            }
            return Ok(());
        }
        let mut grouped: IndexMap<Key, Vec<SharedOrderExecution>> = IndexMap::new();
        if let Some(rows) = executions {
            let mut index = 0;
            loop {
                let row = rows
                    .lock()
                    .map_err(|_| LiveSaoeRegistryError::ExecutionsPoisoned)?
                    .get(index)
                    .cloned();
                let Some(row) = row else {
                    break;
                };
                grouped.entry(key(&row.order)?).or_default().push(row);
                index += 1;
            }
        }
        for (key, adapter) in &mut self.adapters {
            adapter
                .update(grouped.get(key).map_or(&[], Vec::as_slice), range)
                .map_err(LiveSaoeRegistryError::Update)?;
        }
        Ok(())
    }

    /// Finalize in registry insertion order, stopping at the first failure.
    ///
    /// # Errors
    /// Returns the first finalization plugin failure.
    pub fn finalize(&mut self) -> Result<(), LiveSaoeRegistryError> {
        for adapter in self.adapters.values_mut() {
            adapter
                .finalize()
                .map_err(LiveSaoeRegistryError::Finalize)?;
        }
        Ok(())
    }
}

fn key(order: &Arc<RwLock<Order>>) -> Result<Key, LiveSaoeRegistryError> {
    let order = order
        .read()
        .map_err(|_| LiveSaoeRegistryError::OrderPoisoned)?;
    let (name, day, direction) = order.key_by_day()?;
    Ok((name.to_owned(), day, direction))
}
