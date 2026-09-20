//! Immediate SAOE strategy decision assembly around the policy/interpreter pipeline.

use std::sync::Arc;

use chrono::NaiveDateTime;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    NestedOuterDecision, NestedStrategy, NestedStrategyError, Order, OrderDecision, OrderDir,
    SaoeCalendar, SaoeInterpreterError, SaoeOrderFactory, SaoePluginError, SaoePolicyAction,
    SaoePolicyPipeline, SaoeState, SaoeStateProvider, SharedOrderExecution,
    TradeDecisionWithDetails,
};

type OwnedOrderDayKey = (String, NaiveDateTime, OrderDir);

/// Lifecycle boundary for the per-order adapters owned by an immediate SAOE strategy.
pub trait SaoeIntStateProvider: Send {
    /// Rebuild the state registry for a new outer decision.
    ///
    /// # Errors
    /// Returns adapter construction, key conversion, or transport failures.
    fn reset(&mut self, outer: &dyn NestedOuterDecision) -> Result<(), SaoePluginError>;

    /// Snapshot the current state for one outer order.
    ///
    /// # Errors
    /// Returns adapter lookup, state conversion, or transport failures.
    fn state(&self, order: &Order) -> Result<SaoeState, SaoePluginError>;

    /// Group and apply executions for the last inclusive data-calendar range.
    ///
    /// # Errors
    /// Returns execution-key, adapter update, or transport failures.
    fn update(
        &mut self,
        executions: &[SharedOrderExecution],
        step_range: (i64, i64),
    ) -> Result<(), SaoePluginError>;

    /// Finalize every retained adapter in stable registry order.
    ///
    /// # Errors
    /// Returns the first metric finalization or transport failure.
    fn finalize(&mut self) -> Result<(), SaoePluginError>;
}

/// Factory for one fully configured per-order state adapter.
pub trait SaoeStateAdapterFactory: Send {
    /// Create an adapter for `order` using outer-decision metadata when required.
    ///
    /// # Errors
    /// Returns data loading, configuration, or transport failures.
    fn create(
        &mut self,
        order: &Order,
        outer: &dyn NestedOuterDecision,
    ) -> Result<Box<dyn SaoeStateProvider>, SaoePluginError>;
}

/// Ordered multi-order registry matching Python's `adapter_dict` lifecycle.
pub struct SaoeAdapterRegistry {
    factory: Box<dyn SaoeStateAdapterFactory>,
    adapters: IndexMap<OwnedOrderDayKey, Box<dyn SaoeStateProvider>>,
}

impl SaoeAdapterRegistry {
    #[must_use]
    pub fn new(factory: Box<dyn SaoeStateAdapterFactory>) -> Self {
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
}

/// Calendar metadata read while creating SAOE report details and the concrete decision.
pub trait SaoeDecisionCalendar: SaoeCalendar {
    /// Return Qlib's textual frequency for the current strategy level.
    ///
    /// # Errors
    /// Returns a calendar or transport failure.
    fn frequency(&self) -> Result<String, SaoePluginError>;
}

/// One row from Python's `TradeDecisionWithDetails.details` table.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SaoeTradeDetail {
    pub instrument: String,
    pub datetime: NaiveDateTime,
    pub frequency: String,
    pub execution_volume: f64,
    pub action: Option<SaoePolicyAction>,
}

/// SAOE specialization of the source-level decision-with-details container.
pub type SaoeIntDecision = TradeDecisionWithDetails<Vec<SaoeTradeDetail>>;

/// Typed failure retaining the exact plugin stage that stopped decision assembly.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum SaoeIntStrategyError {
    #[error("SAOE state provider failed: {0}")]
    State(SaoePluginError),
    #[error(transparent)]
    Pipeline(#[from] SaoeInterpreterError),
    #[error("SAOE order factory failed: {0}")]
    OrderFactory(SaoePluginError),
    #[error("SAOE decision calendar failed: {0}")]
    Calendar(SaoePluginError),
    #[error("SAOE immediate strategy has not been reset with an outer decision")]
    MissingOuterDecision,
    #[error("a zero-length SAOE step range cannot receive executions")]
    ExecutionsForZeroRange,
}

/// Owns the replaceable state, policy/interpreter, order, and calendar components.
pub struct SaoeIntDecisionBuilder {
    states: Box<dyn SaoeIntStateProvider>,
    pipeline: SaoePolicyPipeline,
    orders: Box<dyn SaoeOrderFactory>,
    calendar: Arc<dyn SaoeDecisionCalendar>,
}

impl SaoeIntDecisionBuilder {
    #[must_use]
    pub fn new(
        states: Box<dyn SaoeIntStateProvider>,
        pipeline: SaoePolicyPipeline,
        orders: Box<dyn SaoeOrderFactory>,
        calendar: Arc<dyn SaoeDecisionCalendar>,
    ) -> Self {
        Self {
            states,
            pipeline,
            orders,
            calendar,
        }
    }

    /// Rebuild the owned state registry for a new outer decision.
    ///
    /// # Errors
    /// Returns the state plugin's first reset failure.
    pub fn reset_states(
        &mut self,
        outer: &dyn NestedOuterDecision,
    ) -> Result<(), SaoeIntStrategyError> {
        self.states
            .reset(outer)
            .map_err(SaoeIntStrategyError::State)
    }

    /// Return the current inclusive data-calendar step range.
    ///
    /// # Errors
    /// Returns a calendar plugin failure.
    pub fn available_step_range(&self) -> Result<(i64, i64), SaoeIntStrategyError> {
        self.calendar
            .available_step_range()
            .map_err(SaoeIntStrategyError::Calendar)
    }

    /// Apply one completed execution batch to the state registry.
    ///
    /// # Errors
    /// Returns the state plugin's first grouping or update failure.
    pub fn update_states(
        &mut self,
        executions: &[SharedOrderExecution],
        step_range: (i64, i64),
    ) -> Result<(), SaoeIntStrategyError> {
        self.states
            .update(executions, step_range)
            .map_err(SaoeIntStrategyError::State)
    }

    /// Finalize all state adapters.
    ///
    /// # Errors
    /// Returns the state plugin's first finalization failure.
    pub fn finalize_states(&mut self) -> Result<(), SaoeIntStrategyError> {
        self.states.finalize().map_err(SaoeIntStrategyError::State)
    }

    /// Build one immediate SAOE decision in Python-compatible stable order.
    ///
    /// Child orders are created only for volumes unequal to zero. Details retain every outer
    /// order, including zero-volume actions. The typed policy boundary always supplies an action,
    /// so the optional detail field is populated for every row.
    ///
    /// # Errors
    /// Returns the first state, pipeline, order-factory, or calendar failure.
    pub fn generate(
        &mut self,
        outer_orders: &[Order],
    ) -> Result<SaoeIntDecision, SaoeIntStrategyError> {
        let states = outer_orders
            .iter()
            .map(|order| {
                self.states
                    .state(order)
                    .map_err(SaoeIntStrategyError::State)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let decisions = self.pipeline.decisions(&states)?;

        let mut child_orders = Vec::new();
        for (outer, decision) in outer_orders.iter().zip(&decisions) {
            if decision.execution_volume != 0.0 {
                child_orders.push(
                    self.orders
                        .create(
                            outer.stock_id(),
                            Some(decision.execution_volume),
                            outer.direction(),
                        )
                        .map_err(SaoeIntStrategyError::OrderFactory)?,
                );
            }
        }

        let mut details = Vec::with_capacity(outer_orders.len());
        for (outer, decision) in outer_orders.iter().zip(decisions) {
            let datetime = self
                .calendar
                .step_time()
                .map_err(SaoeIntStrategyError::Calendar)?
                .0;
            let frequency = self
                .calendar
                .frequency()
                .map_err(SaoeIntStrategyError::Calendar)?;
            details.push(SaoeTradeDetail {
                instrument: outer.stock_id().to_owned(),
                datetime,
                frequency,
                execution_volume: decision.execution_volume,
                action: Some(decision.action),
            });
        }

        let (start_time, end_time) = self
            .calendar
            .step_time()
            .map_err(SaoeIntStrategyError::Calendar)?;
        Ok(SaoeIntDecision::from_orders(
            child_orders,
            start_time,
            end_time,
            None,
            details,
        ))
    }
}

impl SaoeIntStateProvider for SaoeAdapterRegistry {
    fn reset(&mut self, outer: &dyn NestedOuterDecision) -> Result<(), SaoePluginError> {
        self.adapters.clear();
        for order in outer.order_decision().orders() {
            let adapter = self.factory.create(order, outer)?;
            let key = owned_order_day_key(order)?;
            self.adapters.insert(key, adapter);
        }
        Ok(())
    }

    fn state(&self, order: &Order) -> Result<SaoeState, SaoePluginError> {
        let key = owned_order_day_key(order)?;
        self.adapters
            .get(&key)
            .ok_or_else(|| SaoePluginError {
                message: format!(
                    "missing SAOE state adapter for {}/{}/{}",
                    key.0, key.1, key.2
                ),
            })?
            .state(order)
    }

    fn update(
        &mut self,
        executions: &[SharedOrderExecution],
        step_range: (i64, i64),
    ) -> Result<(), SaoePluginError> {
        let mut grouped: IndexMap<OwnedOrderDayKey, Vec<SharedOrderExecution>> = IndexMap::new();
        for execution in executions {
            let key = {
                let order = execution.order.read().map_err(|_| SaoePluginError {
                    message: "SAOE execution order lock poisoned".to_owned(),
                })?;
                owned_order_day_key(&order)?
            };
            grouped.entry(key).or_default().push(Arc::clone(execution));
        }
        for (key, adapter) in &mut self.adapters {
            adapter.update(grouped.get(key).map_or(&[], Vec::as_slice), step_range)?;
        }
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), SaoePluginError> {
        for adapter in self.adapters.values_mut() {
            adapter.finalize()?;
        }
        Ok(())
    }
}

/// Lifecycle-owning immediate strategy corresponding to Python's `SAOEIntStrategy`.
pub struct SaoeIntStrategy {
    builder: SaoeIntDecisionBuilder,
    outer_orders: Option<Vec<Order>>,
    last_step_range: (i64, i64),
}

impl SaoeIntStrategy {
    #[must_use]
    pub const fn new(builder: SaoeIntDecisionBuilder) -> Self {
        Self {
            builder,
            outer_orders: None,
            last_step_range: (0, 0),
        }
    }

    #[must_use]
    pub const fn last_step_range(&self) -> (i64, i64) {
        self.last_step_range
    }
}

impl NestedStrategy for SaoeIntStrategy {
    fn reset(&mut self, outer: &dyn NestedOuterDecision) -> Result<(), NestedStrategyError> {
        self.outer_orders = Some(outer.order_decision().orders().to_vec());
        self.last_step_range = (0, 0);
        self.builder.reset_states(outer).map_err(strategy_error)
    }

    fn alter_outer_decision(
        &mut self,
        _outer: &mut dyn NestedOuterDecision,
    ) -> Result<(), NestedStrategyError> {
        Ok(())
    }

    fn generate_trade_decision(
        &mut self,
        _previous: Option<&[SharedOrderExecution]>,
    ) -> Result<Box<dyn OrderDecision>, NestedStrategyError> {
        let outer_orders = self
            .outer_orders
            .as_deref()
            .ok_or(SaoeIntStrategyError::MissingOuterDecision)
            .map_err(strategy_error)?;
        let step_range = self
            .builder
            .available_step_range()
            .map_err(strategy_error)?;
        self.last_step_range = step_range;
        self.builder
            .generate(outer_orders)
            .map(|decision| Box::new(decision) as Box<dyn OrderDecision>)
            .map_err(strategy_error)
    }

    fn post_execute(
        &mut self,
        executions: &[SharedOrderExecution],
    ) -> Result<(), NestedStrategyError> {
        if self.last_step_range.1 - self.last_step_range.0 <= 0 {
            if executions.is_empty() {
                return Ok(());
            }
            return Err(strategy_error(SaoeIntStrategyError::ExecutionsForZeroRange));
        }
        self.builder
            .update_states(executions, self.last_step_range)
            .map_err(strategy_error)
    }

    fn post_upper_level(&mut self) -> Result<(), NestedStrategyError> {
        self.builder.finalize_states().map_err(strategy_error)
    }
}

fn owned_order_day_key(order: &Order) -> Result<OwnedOrderDayKey, SaoePluginError> {
    let (stock_id, day, direction) = order.key_by_day().map_err(|error| SaoePluginError {
        message: error.to_string(),
    })?;
    Ok((stock_id.to_owned(), day, direction))
}

fn strategy_error(error: impl std::fmt::Display) -> NestedStrategyError {
    NestedStrategyError {
        message: error.to_string(),
    }
}
