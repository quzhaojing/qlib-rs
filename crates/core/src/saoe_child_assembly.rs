//! Configured assembly of a fresh immediate-SAOE nested child graph.

use std::sync::Arc;

use thiserror::Error;

use crate::TradeCalendarRange;
use crate::decision_update::LiveDecisionHandle;
use crate::live_nested_executor::LiveNestedRun;
use crate::live_recursive_inner::{LiveNestedChildAssembly, LiveNestedChildAssemblyFactory};
use crate::saoe_live_generation::LiveSaoeGeneration;
use crate::saoe_live_registry::{LiveSaoeAdapterFactory, LiveSaoeAdapterRegistry};
use crate::{
    LiveSaoeIntStrategy, NestedCalendar, NestedChildAssembly, NestedChildAssemblyFactory,
    NestedExecutorAccount, NestedExecutorReturnSink, NestedInnerExecutor, NestedInnerExecutorError,
    NestedLevelBinding, NestedOuterDecision, NestedStrategy, OrderDecision, ResumableNestedConfig,
    ResumableNestedRun, SaoeDecisionCalendar, SaoeIntDecisionBuilder, SaoeIntStateProvider,
    SaoeIntStrategy, SaoeOrderFactory, SaoePolicyPipeline,
};

/// Typed diagnostic from a configured SAOE component provider.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("SAOE child component assembly failed: {message}")]
pub struct SaoeChildComponentsError {
    pub message: String,
}

/// Fresh owned components required to build one immediate-SAOE child executor graph.
pub struct SaoeChildComponents {
    pub calendar: Box<dyn NestedCalendar>,
    pub strategy_calendar: Arc<dyn SaoeDecisionCalendar>,
    pub level_binding: Box<dyn NestedLevelBinding>,
    pub inner: Box<dyn NestedInnerExecutor>,
    pub states: Box<dyn SaoeIntStateProvider>,
    pub pipeline: SaoePolicyPipeline,
    pub orders: Box<dyn SaoeOrderFactory>,
    pub outer: Box<dyn NestedOuterDecision>,
    pub account: Box<dyn NestedExecutorAccount>,
    pub return_sink: Option<Box<dyn NestedExecutorReturnSink>>,
}

/// Replaceable provider of a fresh configured component set per decision and level.
pub trait SaoeChildComponentsFactory: Send {
    /// Build components without retaining a borrow of `decision`.
    ///
    /// # Errors
    /// Returns configuration, data-source, or plugin-construction failures.
    fn create(
        &mut self,
        decision: &dyn OrderDecision,
        level: usize,
    ) -> Result<SaoeChildComponents, SaoeChildComponentsError>;
}

/// Converts configured SAOE components into the generic owned child-assembly protocol.
pub struct ConfiguredSaoeChildAssemblyFactory {
    config: ResumableNestedConfig,
    components: Box<dyn SaoeChildComponentsFactory>,
}

impl ConfiguredSaoeChildAssemblyFactory {
    #[must_use]
    pub fn new(
        config: ResumableNestedConfig,
        components: Box<dyn SaoeChildComponentsFactory>,
    ) -> Self {
        Self { config, components }
    }
}

impl NestedChildAssemblyFactory for ConfiguredSaoeChildAssemblyFactory {
    fn assemble(
        &mut self,
        decision: &dyn OrderDecision,
        level: usize,
    ) -> Result<NestedChildAssembly, NestedInnerExecutorError> {
        let components =
            self.components
                .create(decision, level)
                .map_err(|error| NestedInnerExecutorError {
                    message: error.to_string(),
                })?;
        let strategy: Box<dyn NestedStrategy> =
            Box::new(SaoeIntStrategy::new(SaoeIntDecisionBuilder::new(
                components.states,
                components.pipeline,
                components.orders,
                components.strategy_calendar,
            )));
        let mut config = self.config.clone();
        config.level = level;
        Ok(NestedChildAssembly {
            calendar: components.calendar,
            config,
            run: ResumableNestedRun {
                level_binding: components.level_binding,
                inner: components.inner,
                strategy,
                outer: components.outer,
                account: components.account,
                return_sink: components.return_sink,
            },
        })
    }
}

/// Fresh shared components required to build one production live-SAOE child graph.
pub struct LiveSaoeChildComponents {
    pub calendar: Box<dyn NestedCalendar>,
    pub strategy_calendar: Arc<dyn SaoeDecisionCalendar>,
    pub range_calendar: Arc<dyn TradeCalendarRange>,
    pub level_binding: Box<dyn NestedLevelBinding>,
    pub inner: Box<dyn NestedInnerExecutor>,
    pub adapters: Box<dyn LiveSaoeAdapterFactory>,
    pub pipeline: SaoePolicyPipeline,
    pub orders: Box<dyn SaoeOrderFactory>,
    pub account: Box<dyn NestedExecutorAccount>,
    pub return_sink: Option<Box<dyn NestedExecutorReturnSink>>,
}

/// Replaceable provider of a fresh shared component set per live decision and level.
pub trait LiveSaoeChildComponentsFactory: Send {
    /// Build components without retaining a borrow of `decision`.
    ///
    /// # Errors
    /// Returns configuration, data-source, or plugin-construction failures.
    fn create(
        &mut self,
        decision: &LiveDecisionHandle,
        level: usize,
    ) -> Result<LiveSaoeChildComponents, SaoeChildComponentsError>;
}

/// Configured bridge from shared SAOE components to the native live nested executor.
pub struct ConfiguredLiveSaoeChildAssemblyFactory {
    config: ResumableNestedConfig,
    components: Box<dyn LiveSaoeChildComponentsFactory>,
}

impl ConfiguredLiveSaoeChildAssemblyFactory {
    #[must_use]
    pub fn new(
        config: ResumableNestedConfig,
        components: Box<dyn LiveSaoeChildComponentsFactory>,
    ) -> Self {
        Self { config, components }
    }
}

impl LiveNestedChildAssemblyFactory for ConfiguredLiveSaoeChildAssemblyFactory {
    fn assemble(
        &mut self,
        decision: &LiveDecisionHandle,
        level: usize,
    ) -> Result<LiveNestedChildAssembly, NestedInnerExecutorError> {
        let components =
            self.components
                .create(decision, level)
                .map_err(|error| NestedInnerExecutorError {
                    message: error.to_string(),
                })?;
        let strategy = LiveSaoeIntStrategy::new(
            LiveSaoeAdapterRegistry::new(components.adapters),
            LiveSaoeGeneration::new(
                components.pipeline,
                components.orders,
                Arc::clone(&components.strategy_calendar),
            ),
            components.strategy_calendar,
            Arc::clone(&components.range_calendar),
        );
        let mut config = self.config.clone();
        config.level = level;
        Ok(LiveNestedChildAssembly {
            calendar: components.calendar,
            config,
            run: LiveNestedRun {
                level_binding: components.level_binding,
                inner: components.inner,
                inner_range_calendar: components.range_calendar,
                strategy: Box::new(Arc::clone(&strategy)),
                outer: decision.clone(),
                account: components.account,
                return_sink: components.return_sink,
            },
        })
    }
}
