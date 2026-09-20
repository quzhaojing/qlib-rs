//! Production live SAOE strategy composed from the verified shared components.

use std::sync::{Arc, Mutex};

use thiserror::Error;

use crate::decision_construction::{DecisionConstructionError, DecisionConstructionStrategy};
use crate::decision_update::{
    DecisionUpdateStrategyError, LiveDecisionAccessError, LiveDecisionHandle,
    SharedDecisionUpdateStrategy, SharedLiveDecision,
};
use crate::live_nested_executor::{LiveNestedStrategy, LiveNestedStrategyProgress};
use crate::nested_executor::SharedNestedResult;
use crate::saoe_live_generation::{LiveSaoeGeneration, LiveSaoeGenerationError};
use crate::saoe_live_registry::{LiveSaoeAdapterRegistry, LiveSaoeRegistryError};
use crate::{
    NestedStrategyError, SaoeDecisionCalendar, SaoePluginError, SaoeTradeDetail, TradeCalendarRange,
};

#[derive(Debug, Error)]
pub enum LiveSaoeStrategyError {
    #[error("live SAOE strategy state lock poisoned")]
    StatePoisoned,
    #[error("live SAOE strategy has not been reset with an outer decision")]
    MissingOuterDecision,
    #[error("live SAOE outer decision has no executable range")]
    MissingStepRange,
    #[error(transparent)]
    Decision(#[from] LiveDecisionAccessError),
    #[error("live SAOE calendar failed: {0}")]
    Calendar(SaoePluginError),
    #[error(transparent)]
    Registry(#[from] LiveSaoeRegistryError),
    #[error(transparent)]
    Generation(#[from] LiveSaoeGenerationError),
    #[error(transparent)]
    Construction(#[from] DecisionConstructionError<SaoePluginError>),
}

struct LiveSaoeRuntime {
    registry: LiveSaoeAdapterRegistry,
    generation: LiveSaoeGeneration,
    outer: Option<LiveDecisionHandle>,
    last_step_range: (i64, i64),
}

/// Shared strategy identity retained by every generated decision and by the nested executor.
///
/// `Arc<LiveSaoeIntStrategy>` implements the live nested strategy interface. Generated decisions
/// retain that exact `Arc`, so decision update callbacks and executor lifecycle calls address the
/// same strategy object rather than a proxy or snapshot.
pub struct LiveSaoeIntStrategy {
    runtime: Mutex<LiveSaoeRuntime>,
    calendar: Arc<dyn SaoeDecisionCalendar>,
    range_calendar: Arc<dyn TradeCalendarRange>,
}

impl LiveSaoeIntStrategy {
    #[must_use]
    pub fn new(
        registry: LiveSaoeAdapterRegistry,
        generation: LiveSaoeGeneration,
        calendar: Arc<dyn SaoeDecisionCalendar>,
        range_calendar: Arc<dyn TradeCalendarRange>,
    ) -> Arc<Self> {
        Arc::new(Self {
            runtime: Mutex::new(LiveSaoeRuntime {
                registry,
                generation,
                outer: None,
                last_step_range: (0, 0),
            }),
            calendar,
            range_calendar,
        })
    }

    /// Return the last source-visible inclusive data-calendar range.
    ///
    /// # Errors
    /// Returns a poisoned strategy-state lock.
    pub fn last_step_range(&self) -> Result<(i64, i64), LiveSaoeStrategyError> {
        Ok(self
            .runtime
            .lock()
            .map_err(|_| LiveSaoeStrategyError::StatePoisoned)?
            .last_step_range)
    }

    fn reset_live(&self, outer: &LiveDecisionHandle) -> Result<(), LiveSaoeStrategyError> {
        let mut runtime = self
            .runtime
            .lock()
            .map_err(|_| LiveSaoeStrategyError::StatePoisoned)?;
        runtime.outer = Some(outer.clone());
        runtime.last_step_range = (0, 0);
        runtime.registry.reset(Some(outer))?;
        Ok(())
    }

    fn generate_live(self: &Arc<Self>) -> Result<LiveDecisionHandle, LiveSaoeStrategyError> {
        let mut runtime = self
            .runtime
            .lock()
            .map_err(|_| LiveSaoeStrategyError::StatePoisoned)?;
        let outer = runtime
            .outer
            .clone()
            .ok_or(LiveSaoeStrategyError::MissingOuterDecision)?;
        let calendar_range = self
            .calendar
            .available_step_range()
            .map_err(LiveSaoeStrategyError::Calendar)?;
        let base = outer.base()?;
        let rule = base
            .trade_range
            .ok_or(LiveSaoeStrategyError::MissingStepRange)?;
        let outer_range = rule
            .range_indices(Some(self.range_calendar.as_ref()))
            .map_err(crate::TradeDecisionError::from)
            .map_err(LiveDecisionAccessError::from)?;
        let total = match outer.total_step()? {
            crate::decision_construction::DecisionTotalStep::Value(value) => Some(value),
            _ => None,
        };
        let outer_range = crate::trade_decision::clip_range_to_total(outer_range, total);
        runtime.last_step_range = (
            calendar_range.0.max(outer_range.0),
            calendar_range.1.min(outer_range.1),
        );
        let LiveSaoeRuntime {
            registry,
            generation,
            ..
        } = &mut *runtime;
        let parts = generation.generate_live(&outer, |order| {
            registry.live_state(order).map_err(plugin_error)
        })?;
        Ok(parts.into_decision(Arc::clone(self))? as LiveDecisionHandle)
    }

    fn update_live(&self, executions: &SharedNestedResult) -> Result<(), LiveSaoeStrategyError> {
        let mut runtime = self
            .runtime
            .lock()
            .map_err(|_| LiveSaoeStrategyError::StatePoisoned)?;
        let range = runtime.last_step_range;
        runtime.registry.update(Some(executions), range)?;
        Ok(())
    }

    fn finalize_live(&self) -> Result<(), LiveSaoeStrategyError> {
        self.runtime
            .lock()
            .map_err(|_| LiveSaoeStrategyError::StatePoisoned)?
            .registry
            .finalize()?;
        Ok(())
    }
}

impl DecisionConstructionStrategy for LiveSaoeIntStrategy {
    type Error = SaoePluginError;

    fn step_time(&self) -> Result<(chrono::NaiveDateTime, chrono::NaiveDateTime), Self::Error> {
        self.calendar.step_time()
    }
}

impl SharedDecisionUpdateStrategy<Vec<SaoeTradeDetail>> for LiveSaoeIntStrategy {
    fn update_trade_decision(
        &self,
        _decision: &SharedLiveDecision<Self, Vec<SaoeTradeDetail>>,
        _calendar: &dyn crate::DecisionUpdateCalendar,
    ) -> Result<Option<SharedLiveDecision<Self, Vec<SaoeTradeDetail>>>, DecisionUpdateStrategyError>
    {
        Ok(None)
    }
}

impl LiveNestedStrategy for Arc<LiveSaoeIntStrategy> {
    fn reset(&mut self, outer: &LiveDecisionHandle) -> Result<(), NestedStrategyError> {
        self.reset_live(outer).map_err(strategy_error)
    }

    fn alter_outer_decision(
        &mut self,
        outer: LiveDecisionHandle,
    ) -> Result<LiveDecisionHandle, NestedStrategyError> {
        Ok(outer)
    }

    fn begin(
        &mut self,
        _previous: Option<&SharedNestedResult>,
    ) -> Result<LiveNestedStrategyProgress, NestedStrategyError> {
        self.generate_live()
            .map(LiveNestedStrategyProgress::Ready)
            .map_err(strategy_error)
    }

    fn post_execute(&mut self, executions: &SharedNestedResult) -> Result<(), NestedStrategyError> {
        self.update_live(executions).map_err(strategy_error)
    }

    fn post_upper_level(&mut self) -> Result<(), NestedStrategyError> {
        self.finalize_live().map_err(strategy_error)
    }
}

fn plugin_error(error: impl std::fmt::Display) -> SaoePluginError {
    SaoePluginError {
        message: error.to_string(),
    }
}

fn strategy_error(error: impl std::fmt::Display) -> NestedStrategyError {
    NestedStrategyError {
        message: error.to_string(),
    }
}
