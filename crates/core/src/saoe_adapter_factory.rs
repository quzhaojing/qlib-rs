//! Configured construction of concrete per-order SAOE state adapters.

use std::sync::{Arc, RwLock};

use num_traits::ToPrimitive;

use crate::decision_update::LiveDecisionHandle;
use crate::saoe_live_registry::{LiveSaoeAdapterFactory, LiveSaoeStateAdapter};
use crate::{
    ConcreteSaoeStateAdapter, NestedOuterDecision, Order, SaoeAdapterConfig, SaoeAdapterContext,
    SaoeAdapterMarket, SaoeBacktestDataLoader, SaoePluginError, SaoeStateAdapterFactory,
    SaoeStateProvider, SharedTradeRange,
};

/// Fully owned inputs required to construct one concrete state adapter.
pub struct SaoeAdapterInputs {
    pub market: Arc<dyn SaoeAdapterMarket>,
    pub context: Arc<dyn SaoeAdapterContext>,
    pub config: SaoeAdapterConfig,
}

/// Per-order data/configuration loading boundary for concrete SAOE adapters.
pub trait SaoeAdapterInputsProvider: Send {
    /// Load every immutable input for `order` using outer-decision metadata where required.
    ///
    /// # Errors
    /// Returns market-data, calendar, configuration, or transport failures.
    fn load(
        &mut self,
        order: &Order,
        outer: &dyn NestedOuterDecision,
    ) -> Result<SaoeAdapterInputs, SaoePluginError>;
}

/// Runtime calendar metadata observed while constructing a concrete SAOE adapter.
pub trait SaoeAdapterRuntime: Send + Sync {
    /// Convert the current strategy-level frequency to whole minutes per executor step.
    ///
    /// # Errors
    /// Returns calendar, conversion, or transport failures.
    fn ticks_per_step(&self) -> Result<usize, SaoePluginError>;

    /// Resolve Qlib's `get_start_end_idx(...).0` for the outer decision.
    ///
    /// # Errors
    /// Returns decision-range, calendar, or transport failures.
    fn start_step(&self, outer: &dyn NestedOuterDecision) -> Result<i64, SaoePluginError>;
}

/// Concrete input provider composing cached market data with runtime infrastructure.
pub struct ConfiguredSaoeAdapterInputsProvider {
    loader: SaoeBacktestDataLoader,
    market: Arc<dyn SaoeAdapterMarket>,
    context: Arc<dyn SaoeAdapterContext>,
    runtime: Arc<dyn SaoeAdapterRuntime>,
    data_granularity: usize,
}

impl ConfiguredSaoeAdapterInputsProvider {
    #[must_use]
    pub fn new(
        loader: SaoeBacktestDataLoader,
        market: Arc<dyn SaoeAdapterMarket>,
        context: Arc<dyn SaoeAdapterContext>,
        runtime: Arc<dyn SaoeAdapterRuntime>,
        data_granularity: usize,
    ) -> Self {
        Self {
            loader,
            market,
            context,
            runtime,
            data_granularity,
        }
    }

    #[must_use]
    pub fn backtest_cache_len(&self) -> usize {
        self.loader.cache_len()
    }
}

impl SaoeAdapterInputsProvider for ConfiguredSaoeAdapterInputsProvider {
    fn load(
        &mut self,
        order: &Order,
        outer: &dyn NestedOuterDecision,
    ) -> Result<SaoeAdapterInputs, SaoePluginError> {
        let trade_range = outer
            .order_decision()
            .trade_range()
            .ok_or_else(|| SaoePluginError {
                message: "SAOE adapter construction requires an outer trade range".to_owned(),
            })?;
        let backtest_data =
            self.loader
                .load(order, trade_range)
                .map_err(|error| SaoePluginError {
                    message: error.to_string(),
                })?;
        let ticks_per_step = self.runtime.ticks_per_step()?;
        let start_step = self.runtime.start_step(outer)?;
        let deal_prices = backtest_data.deal_prices.clone();
        Ok(SaoeAdapterInputs {
            market: Arc::clone(&self.market),
            context: Arc::clone(&self.context),
            config: SaoeAdapterConfig {
                backtest_data,
                deal_prices,
                ticks_per_step,
                data_granularity: self.data_granularity,
                start_step,
            },
        })
    }
}

/// Factory that turns configured inputs into initialized concrete adapters.
pub struct ConfiguredSaoeStateAdapterFactory {
    inputs: Box<dyn SaoeAdapterInputsProvider>,
}

impl ConfiguredSaoeStateAdapterFactory {
    #[must_use]
    pub fn new(inputs: Box<dyn SaoeAdapterInputsProvider>) -> Self {
        Self { inputs }
    }
}

impl SaoeStateAdapterFactory for ConfiguredSaoeStateAdapterFactory {
    fn create(
        &mut self,
        order: &Order,
        outer: &dyn NestedOuterDecision,
    ) -> Result<Box<dyn SaoeStateProvider>, SaoePluginError> {
        let inputs = self.inputs.load(order, outer)?;
        let mut adapter =
            ConcreteSaoeStateAdapter::new(inputs.market, inputs.context, inputs.config).map_err(
                |error| SaoePluginError {
                    message: error.to_string(),
                },
            )?;
        adapter
            .reset_order(order)
            .map_err(|error| SaoePluginError {
                message: error.to_string(),
            })?;
        Ok(Box::new(adapter))
    }
}

/// Runtime observations reached after live backtest-data loading.
pub trait LiveSaoeAdapterRuntime: Send + Sync {
    /// Convert the current executor frequency to whole minutes per step.
    ///
    /// # Errors
    /// Returns a frequency, conversion, or transport failure.
    fn ticks_per_step(&self) -> Result<usize, SaoePluginError>;

    /// Resolve the source adapter constructor's start index from the original decision.
    ///
    /// # Errors
    /// Returns a decision, calendar, range, or transport failure.
    fn start_step(&self, outer: &LiveDecisionHandle) -> Result<i64, SaoePluginError>;
}

impl LiveSaoeAdapterRuntime for crate::SharedExecutionCalendar {
    fn ticks_per_step(&self) -> Result<usize, SaoePluginError> {
        let frequency = crate::SaoeDecisionCalendar::frequency(self)?;
        let frequency: crate::Frequency = frequency.parse().map_err(plugin_error)?;
        if frequency.unit == crate::FrequencyUnit::Month {
            return Err(SaoePluginError {
                message: "SAOE executor frequency cannot use calendar months".to_owned(),
            });
        }
        frequency
            .approximate_minutes()
            .to_usize()
            .ok_or_else(|| SaoePluginError {
                message: format!("SAOE executor frequency is too large: {frequency}"),
            })
    }

    fn start_step(&self, outer: &LiveDecisionHandle) -> Result<i64, SaoePluginError> {
        outer
            .range_limit(Some(self), crate::RangeLimitDefault::Error)
            .map(|range| range.map_or(0, |(start, _)| start))
            .map_err(plugin_error)
    }
}

/// Configured factory for the live registry and concrete numerical adapter.
pub struct ConfiguredLiveSaoeAdapterFactory {
    loader: SaoeBacktestDataLoader,
    context: Arc<dyn SaoeAdapterContext>,
    runtime: Arc<dyn LiveSaoeAdapterRuntime>,
    data_granularity: usize,
}

impl ConfiguredLiveSaoeAdapterFactory {
    #[must_use]
    pub fn new(
        loader: SaoeBacktestDataLoader,
        context: Arc<dyn SaoeAdapterContext>,
        runtime: Arc<dyn LiveSaoeAdapterRuntime>,
        data_granularity: usize,
    ) -> Self {
        Self {
            loader,
            context,
            runtime,
            data_granularity,
        }
    }

    #[must_use]
    pub fn backtest_cache_len(&self) -> usize {
        self.loader.cache_len()
    }
}

impl LiveSaoeAdapterFactory for ConfiguredLiveSaoeAdapterFactory {
    fn create(
        &mut self,
        order: &Arc<RwLock<Order>>,
        outer: &LiveDecisionHandle,
        range: &SharedTradeRange,
    ) -> Result<Box<dyn LiveSaoeStateAdapter>, SaoePluginError> {
        let backtest_data = self
            .loader
            .load_shared(order, range.as_ref())
            .map_err(plugin_error)?;
        let ticks_per_step = self.runtime.ticks_per_step()?;
        let runtime = Arc::clone(&self.runtime);
        let decision = Arc::clone(outer);
        let mut start_step = move || runtime.start_step(&decision);
        ConcreteSaoeStateAdapter::new_live_cached(
            self.loader.source(),
            Arc::clone(&self.context),
            backtest_data,
            ticks_per_step,
            self.data_granularity,
            order,
            &mut start_step,
        )
        .map(|adapter| Box::new(adapter) as Box<dyn LiveSaoeStateAdapter>)
        .map_err(plugin_error)
    }
}

fn plugin_error(error: impl std::fmt::Display) -> SaoePluginError {
    SaoePluginError {
        message: error.to_string(),
    }
}
