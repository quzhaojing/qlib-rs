//! Concrete state calculations for Qlib's single-asset order-execution adapter.

use std::sync::{Arc, RwLock};

use arrow_array::{
    ArrayRef, Float64Array, Int8Array, RecordBatch, StringArray, TimestampNanosecondArray,
};
use arrow_schema::{ArrowError, DataType, Field, Schema, TimeUnit};
use chrono::{NaiveDateTime, TimeDelta};
use ndarray::Array1;
use num_traits::ToPrimitive;
use thiserror::Error;

use crate::{
    Frequency, FrequencyUnit, Order, OrderDir, OrderError, Region, SaoeBacktestData, SaoeMetrics,
    SaoeNumeric, SaoePluginError, SaoeState, SaoeStateParts, SaoeStateProvider, SaoeTime,
    SharedOrderExecution,
    time_calendar_cache::default_time_calendar_cache,
    time_compat::{TimeCompatError, day_minute_index_range_with_cache, python_clock_precision},
};

const EPS: f64 = 1.0e-8;

#[cfg(test)]
#[path = "saoe_state_adapter/unit_tests.rs"]
mod unit_tests;

enum AdapterMarket {
    Combined(Arc<dyn SaoeAdapterMarket>),
    Live(Arc<dyn crate::SaoeBacktestDataSource>),
}

/// Market vectors returned for one inclusive SAOE step range.
#[derive(Clone, Debug, PartialEq)]
pub struct SaoeMarketSlice {
    pub volume: Array1<f64>,
    pub price: Array1<f64>,
}

/// Exchange-facing market-data boundary used by the concrete adapter.
pub trait SaoeAdapterMarket: Send + Sync {
    /// Return volume and deal-price vectors for the closed interval.
    ///
    /// # Errors
    /// Returns a quote, conversion, or transport failure.
    fn market_slice(
        &self,
        stock_id: &str,
        start: NaiveDateTime,
        end: NaiveDateTime,
        direction: OrderDir,
    ) -> Result<SaoeMarketSlice, SaoePluginError>;
}

/// Executor/account observations needed by the concrete adapter.
pub trait SaoeAdapterContext: Send + Sync {
    /// Current trade-calendar step.
    ///
    /// # Errors
    /// Returns a calendar or transport failure.
    fn current_trade_step(&self) -> Result<i64, SaoePluginError>;

    /// Latest executor-level price-advantage indicator.
    ///
    /// # Errors
    /// Returns an indicator or transport failure.
    fn latest_price_advantage(&self) -> Result<f64, SaoePluginError>;

    /// Report Python's warning emitted for material overfill.
    ///
    /// # Errors
    /// Returns an observer or transport failure.
    fn warn_overfill(&self, execution_volume: f64, position: f64) -> Result<(), SaoePluginError>;
}

/// Immutable data and granularity used for each reset order.
#[derive(Clone, Debug)]
pub struct SaoeAdapterConfig {
    pub backtest_data: SaoeBacktestData,
    pub deal_prices: Array1<f64>,
    pub ticks_per_step: usize,
    pub data_granularity: usize,
    pub start_step: i64,
}

/// One scalar row in the execution or step history.
#[derive(Clone, Debug, PartialEq)]
pub struct SaoeMetricRow {
    pub stock_id: String,
    pub datetime: NaiveDateTime,
    pub direction: OrderDir,
    pub market_volume: f64,
    pub market_price: f64,
    pub amount: f64,
    pub inner_amount: f64,
    pub deal_amount: f64,
    pub trade_price: f64,
    pub trade_value: f64,
    pub position: f64,
    pub ffr: f64,
    pub pa: f64,
}

impl SaoeMetricRow {
    fn as_metrics(&self) -> SaoeMetrics {
        SaoeMetrics {
            stock_id: self.stock_id.clone(),
            datetime: SaoeTime::Scalar(self.datetime),
            direction: self.direction,
            market_volume: SaoeNumeric::Scalar(self.market_volume),
            market_price: SaoeNumeric::Scalar(self.market_price),
            amount: SaoeNumeric::Scalar(self.amount),
            inner_amount: SaoeNumeric::Scalar(self.inner_amount),
            deal_amount: SaoeNumeric::Scalar(self.deal_amount),
            trade_price: SaoeNumeric::Scalar(self.trade_price),
            trade_value: SaoeNumeric::Scalar(self.trade_value),
            position: SaoeNumeric::Scalar(self.position),
            ffr: SaoeNumeric::Scalar(self.ffr),
            pa: SaoeNumeric::Scalar(self.pa),
        }
    }
}

/// Typed failures from the concrete state calculations.
#[derive(Debug, Error)]
pub enum SaoeAdapterError {
    #[error(transparent)]
    TimeCompatibility(#[from] TimeCompatError),
    #[error("SAOE adapter order lock poisoned")]
    AdapterOrderPoisoned,
    #[error(transparent)]
    LiveMarket(#[from] crate::saoe_live_market::LiveSaoeMarketError),
    #[error("SAOE execution order lock poisoned")]
    ExecutionOrderPoisoned,
    #[error("SAOE execution history lock poisoned")]
    HistoryExecPoisoned,
    #[error("SAOE step history lock poisoned")]
    HistoryStepsPoisoned,
    #[error("SAOE final metrics lock poisoned")]
    MetricsPoisoned,
    #[error("SAOE backtest-data object lock poisoned")]
    BacktestDataPoisoned,
    #[error("SAOE backtest-data field {0} lock poisoned")]
    BacktestDataFieldPoisoned(&'static str),
    #[error("SAOE adapter has not been reset with an order")]
    NotInitialized,
    #[error("SAOE backtest ticks are empty")]
    EmptyTicks,
    #[error("SAOE order ticks are empty")]
    EmptyOrderTicks,
    #[error("SAOE data granularity must be positive")]
    ZeroGranularity,
    #[error("SAOE data granularity exceeds Chrono's minute range")]
    GranularityTooLarge,
    #[error(
        "ticks_per_step {ticks_per_step} is not divisible by data granularity {data_granularity}"
    )]
    IncompatibleGranularity {
        ticks_per_step: usize,
        data_granularity: usize,
    },
    #[error("invalid inclusive SAOE step range ({start}, {end}) for {ticks} ticks")]
    InvalidStepRange { start: i64, end: i64, ticks: usize },
    #[error("current SAOE time {0} is absent or ambiguous in the backtest tick index")]
    MissingCurrentTime(NaiveDateTime),
    #[error("execution minute index {index} is outside step range ({start}, {end})")]
    ExecutionOutsideStep { index: usize, start: i64, end: i64 },
    #[error("order end time is required by the SAOE adapter")]
    MissingEndTime,
    #[error("{name} vector length {actual} does not match expected step length {expected}")]
    VectorLength {
        name: &'static str,
        actual: usize,
        expected: usize,
    },
    #[error(transparent)]
    Order(#[from] OrderError),
    #[error(transparent)]
    Arrow(#[from] ArrowError),
    #[error(transparent)]
    Plugin(#[from] SaoePluginError),
}

/// Production calculation engine behind [`SaoeStateProvider`].
pub struct ConcreteSaoeStateAdapter {
    market: AdapterMarket,
    context: Arc<dyn SaoeAdapterContext>,
    config: SaoeAdapterConfig,
    session: Option<SaoeAdapterSession>,
    position: f64,
    twap_price: f64,
    history_exec: SharedSaoeHistory,
    history_steps: SharedSaoeHistory,
    metrics: Option<SharedSaoeMetrics>,
    backtest_data: SharedSaoeBacktestData,
}

#[derive(Clone)]
struct SaoeAdapterSession {
    order: Arc<RwLock<Order>>,
    cur_time: NaiveDateTime,
}

/// One replaceable Python-DataFrame identity represented by shared typed rows.
pub type SharedSaoeHistory = Arc<RwLock<Vec<SaoeMetricRow>>>;

/// One replaceable mutable Python `SAOEMetrics` dictionary identity.
pub type SharedSaoeMetrics = Arc<RwLock<SaoeMetrics>>;

/// One mutable Python attribute value retained independently across attribute rebinding.
pub type SharedSaoeTicks = Arc<RwLock<Vec<NaiveDateTime>>>;
pub type SharedSaoeValues = Arc<RwLock<Array1<f64>>>;
pub type SharedSaoeFeatures = Arc<RwLock<RecordBatch>>;

/// Mutable attributes of one Python `IntradayBacktestData` object.
#[derive(Clone)]
pub struct LiveSaoeBacktestData {
    pub source_order: Option<Arc<RwLock<Order>>>,
    pub source: Option<Arc<dyn crate::SaoeBacktestDataSource>>,
    pub ticks_index: SharedSaoeTicks,
    pub ticks_for_order: SharedSaoeTicks,
    pub deal_prices: SharedSaoeValues,
    pub market_volumes: SharedSaoeValues,
    pub features: SharedSaoeFeatures,
}

impl LiveSaoeBacktestData {
    #[must_use]
    pub fn from_owned(data: SaoeBacktestData) -> Self {
        Self {
            source_order: None,
            source: None,
            ticks_index: Arc::new(RwLock::new(data.ticks_index)),
            ticks_for_order: Arc::new(RwLock::new(data.ticks_for_order)),
            deal_prices: Arc::new(RwLock::new(data.deal_prices)),
            market_volumes: Arc::new(RwLock::new(data.market_volumes)),
            features: Arc::new(RwLock::new(data.features)),
        }
    }

    #[must_use]
    pub fn into_shared(self) -> SharedSaoeBacktestData {
        Arc::new(RwLock::new(self))
    }
}

impl std::fmt::Debug for LiveSaoeBacktestData {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LiveSaoeBacktestData")
            .field("source_order", &self.source_order)
            .field("source", &self.source.as_ref().map(|_| "<data source>"))
            .field("ticks_index", &self.ticks_index)
            .field("ticks_for_order", &self.ticks_for_order)
            .field("deal_prices", &self.deal_prices)
            .field("market_volumes", &self.market_volumes)
            .field("features", &self.features)
            .finish()
    }
}

/// One mutable Python `IntradayBacktestData` object identity.
pub type SharedSaoeBacktestData = Arc<RwLock<LiveSaoeBacktestData>>;

/// Constructor fields for one live `SAOEState` alias view.
#[derive(Clone)]
pub struct LiveSaoeStateParts {
    pub order: Arc<RwLock<Order>>,
    pub cur_time: NaiveDateTime,
    pub cur_step: i64,
    pub position: f64,
    pub history_exec: SharedSaoeHistory,
    pub history_steps: SharedSaoeHistory,
    pub metrics: Option<SharedSaoeMetrics>,
    pub backtest_data: SharedSaoeBacktestData,
    pub ticks_index: SharedSaoeTicks,
    pub ticks_for_order: SharedSaoeTicks,
    pub ticks_per_step: usize,
}

/// Live state aliases returned by the concrete adapter before optional materialization.
#[derive(Clone)]
pub struct LiveSaoeState {
    order: Arc<RwLock<Order>>,
    cur_time: NaiveDateTime,
    cur_step: i64,
    position: f64,
    history_exec: SharedSaoeHistory,
    history_steps: SharedSaoeHistory,
    metrics: Option<SharedSaoeMetrics>,
    backtest_data: SharedSaoeBacktestData,
    ticks_index: SharedSaoeTicks,
    ticks_for_order: SharedSaoeTicks,
    ticks_per_step: usize,
}

impl LiveSaoeState {
    #[must_use]
    pub fn new(parts: LiveSaoeStateParts) -> Self {
        Self {
            order: parts.order,
            cur_time: parts.cur_time,
            cur_step: parts.cur_step,
            position: parts.position,
            history_exec: parts.history_exec,
            history_steps: parts.history_steps,
            metrics: parts.metrics,
            backtest_data: parts.backtest_data,
            ticks_index: parts.ticks_index,
            ticks_for_order: parts.ticks_for_order,
            ticks_per_step: parts.ticks_per_step,
        }
    }

    #[must_use]
    pub fn order(&self) -> &Arc<RwLock<Order>> {
        &self.order
    }

    #[must_use]
    pub fn history_exec(&self) -> &SharedSaoeHistory {
        &self.history_exec
    }

    #[must_use]
    pub fn history_steps(&self) -> &SharedSaoeHistory {
        &self.history_steps
    }

    #[must_use]
    pub fn metrics(&self) -> Option<&SharedSaoeMetrics> {
        self.metrics.as_ref()
    }

    #[must_use]
    pub fn backtest_data(&self) -> &SharedSaoeBacktestData {
        &self.backtest_data
    }

    #[must_use]
    pub fn ticks_index(&self) -> &SharedSaoeTicks {
        &self.ticks_index
    }

    #[must_use]
    pub fn ticks_for_order(&self) -> &SharedSaoeTicks {
        &self.ticks_for_order
    }

    /// Materialize the aliases at the time this method is called.
    ///
    /// # Errors
    /// Returns poisoned order/history access or an Arrow conversion failure.
    pub fn snapshot(&self) -> Result<SaoeState, SaoeAdapterError> {
        let order = self
            .order
            .read()
            .map_err(|_| SaoeAdapterError::AdapterOrderPoisoned)?
            .clone();
        let history_exec = self
            .history_exec
            .read()
            .map_err(|_| SaoeAdapterError::HistoryExecPoisoned)?;
        let history_steps = self
            .history_steps
            .read()
            .map_err(|_| SaoeAdapterError::HistoryStepsPoisoned)?;
        let metrics = self
            .metrics
            .as_ref()
            .map(|metrics| {
                metrics
                    .read()
                    .map(|metrics| metrics.clone())
                    .map_err(|_| SaoeAdapterError::MetricsPoisoned)
            })
            .transpose()?;
        let backtest_data = snapshot_backtest_data(&self.backtest_data)?;
        let ticks_index = clone_backtest_field(&self.ticks_index, "state.ticks_index")?;
        let ticks_for_order = clone_backtest_field(&self.ticks_for_order, "state.ticks_for_order")?;
        Ok(SaoeState::new(SaoeStateParts {
            order,
            cur_time: self.cur_time,
            cur_step: self.cur_step,
            position: self.position,
            history_exec: rows_to_batch(&history_exec)?,
            history_steps: rows_to_batch(&history_steps)?,
            metrics,
            backtest_data,
            ticks_per_step: self.ticks_per_step,
            ticks_index,
            ticks_for_order,
        }))
    }
}

fn clone_backtest_field<T: Clone>(
    field: &Arc<RwLock<T>>,
    name: &'static str,
) -> Result<T, SaoeAdapterError> {
    field
        .read()
        .map(|value| value.clone())
        .map_err(|_| SaoeAdapterError::BacktestDataFieldPoisoned(name))
}

fn backtest_handles(
    data: &SharedSaoeBacktestData,
) -> Result<LiveSaoeBacktestData, SaoeAdapterError> {
    data.read()
        .map(|data| data.clone())
        .map_err(|_| SaoeAdapterError::BacktestDataPoisoned)
}

fn snapshot_backtest_data(
    data: &SharedSaoeBacktestData,
) -> Result<SaoeBacktestData, SaoeAdapterError> {
    let data = backtest_handles(data)?;
    Ok(SaoeBacktestData {
        ticks_index: clone_backtest_field(&data.ticks_index, "ticks_index")?,
        ticks_for_order: clone_backtest_field(&data.ticks_for_order, "ticks_for_order")?,
        deal_prices: clone_backtest_field(&data.deal_prices, "deal_prices")?,
        market_volumes: clone_backtest_field(&data.market_volumes, "market_volumes")?,
        features: clone_backtest_field(&data.features, "features")?,
    })
}

impl ConcreteSaoeStateAdapter {
    /// Validate immutable inputs and create an uninitialized adapter.
    ///
    /// # Errors
    /// Returns an invalid tick or granularity configuration.
    pub fn new(
        market: Arc<dyn SaoeAdapterMarket>,
        context: Arc<dyn SaoeAdapterContext>,
        config: SaoeAdapterConfig,
    ) -> Result<Self, SaoeAdapterError> {
        Self::with_market(AdapterMarket::Combined(market), context, config)
    }

    /// Create the numerical engine using separate live volume/price callbacks.
    /// Inputs are already loaded: source constructor loading/initialization ordering belongs
    /// to the factory and is not performed by this engine constructor.
    ///
    /// # Errors
    /// Returns invalid tick or granularity configuration.
    pub fn new_live(
        market: Arc<dyn crate::SaoeBacktestDataSource>,
        context: Arc<dyn SaoeAdapterContext>,
        config: SaoeAdapterConfig,
    ) -> Result<Self, SaoeAdapterError> {
        Self::with_market(AdapterMarket::Live(market), context, config)
    }

    /// Construct and bind a live adapter in the source constructor's observable order.
    ///
    /// The original amount is captured before the start-step callback. Baseline prices are then
    /// loaded, and only afterwards is the possibly mutated order start read for the initial time.
    /// This entry point consumes already loaded intraday data; outer factory loading order remains
    /// the responsibility of the caller.
    ///
    /// # Errors
    /// Returns the first order access, runtime callback, baseline callback, or configuration
    /// failure. No partially constructed adapter is published.
    pub fn new_live_bound(
        market: Arc<dyn crate::SaoeBacktestDataSource>,
        context: Arc<dyn SaoeAdapterContext>,
        mut config: SaoeAdapterConfig,
        order: &Arc<RwLock<Order>>,
        start_step: &mut dyn FnMut() -> Result<i64, SaoePluginError>,
        baseline: &mut dyn FnMut() -> Result<Array1<f64>, SaoePluginError>,
    ) -> Result<Self, SaoeAdapterError> {
        let position = order
            .read()
            .map_err(|_| SaoeAdapterError::AdapterOrderPoisoned)?
            .amount();
        config.start_step = start_step()?;
        config.deal_prices = baseline()?;
        let twap_price = nan_mean(config.deal_prices.iter().copied());
        let first_tick = *config
            .backtest_data
            .ticks_for_order
            .first()
            .ok_or(SaoeAdapterError::EmptyOrderTicks)?;
        let start = order
            .read()
            .map_err(|_| SaoeAdapterError::AdapterOrderPoisoned)?
            .start_time()
            .ok_or(OrderError::MissingStartTime)?;
        validate_config(&config)?;
        let backtest_data =
            LiveSaoeBacktestData::from_owned(config.backtest_data.clone()).into_shared();
        Ok(Self {
            market: AdapterMarket::Live(market),
            context,
            config,
            session: Some(SaoeAdapterSession {
                order: Arc::clone(order),
                cur_time: first_tick.max(start),
            }),
            position,
            twap_price,
            history_exec: Arc::new(RwLock::new(Vec::new())),
            history_steps: Arc::new(RwLock::new(Vec::new())),
            metrics: None,
            backtest_data,
        })
    }

    /// Construct a live adapter around the exact cached backtest-data object.
    ///
    /// The start-step callback runs before the cached deal-price and tick fields are read, matching
    /// the source constructor after it stores the supplied `IntradayBacktestData` object. The
    /// adapter retains that same parent allocation rather than rebuilding an equivalent clone.
    ///
    /// # Errors
    /// Returns the first order access, runtime callback, cached-field access, or configuration
    /// failure. No partially constructed adapter is published.
    pub fn new_live_cached(
        market: Arc<dyn crate::SaoeBacktestDataSource>,
        context: Arc<dyn SaoeAdapterContext>,
        backtest_data: SharedSaoeBacktestData,
        ticks_per_step: usize,
        data_granularity: usize,
        order: &Arc<RwLock<Order>>,
        start_step: &mut dyn FnMut() -> Result<i64, SaoePluginError>,
    ) -> Result<Self, SaoeAdapterError> {
        let position = order
            .read()
            .map_err(|_| SaoeAdapterError::AdapterOrderPoisoned)?
            .amount();
        let start_step = start_step()?;
        let handles = backtest_handles(&backtest_data)?;
        let deal_prices = clone_backtest_field(&handles.deal_prices, "deal_prices")?;
        let twap_price = nan_mean(deal_prices.iter().copied());
        let first_tick = clone_backtest_field(&handles.ticks_for_order, "ticks_for_order")?
            .into_iter()
            .next()
            .ok_or(SaoeAdapterError::EmptyOrderTicks)?;
        let start = order
            .read()
            .map_err(|_| SaoeAdapterError::AdapterOrderPoisoned)?
            .start_time()
            .ok_or(OrderError::MissingStartTime)?;
        let config = SaoeAdapterConfig {
            backtest_data: snapshot_backtest_data(&backtest_data)?,
            deal_prices,
            ticks_per_step,
            data_granularity,
            start_step,
        };
        validate_config(&config)?;
        Ok(Self {
            market: AdapterMarket::Live(market),
            context,
            config,
            session: Some(SaoeAdapterSession {
                order: Arc::clone(order),
                cur_time: first_tick.max(start),
            }),
            position,
            twap_price,
            history_exec: Arc::new(RwLock::new(Vec::new())),
            history_steps: Arc::new(RwLock::new(Vec::new())),
            metrics: None,
            backtest_data,
        })
    }

    fn with_market(
        market: AdapterMarket,
        context: Arc<dyn SaoeAdapterContext>,
        config: SaoeAdapterConfig,
    ) -> Result<Self, SaoeAdapterError> {
        validate_config(&config)?;
        let twap_price = nan_mean(config.deal_prices.iter().copied());
        let backtest_data =
            LiveSaoeBacktestData::from_owned(config.backtest_data.clone()).into_shared();
        Ok(Self {
            market,
            context,
            config,
            session: None,
            position: 0.0,
            twap_price,
            history_exec: Arc::new(RwLock::new(Vec::new())),
            history_steps: Arc::new(RwLock::new(Vec::new())),
            metrics: None,
            backtest_data,
        })
    }

    /// Reset all mutable history for one outer order.
    ///
    /// # Errors
    /// Returns missing order interval metadata.
    pub fn reset_order(&mut self, order: &Order) -> Result<(), SaoeAdapterError> {
        order.start_time().ok_or(OrderError::MissingStartTime)?;
        order.end_time().ok_or(SaoeAdapterError::MissingEndTime)?;
        self.reset_shared_order(&Arc::new(RwLock::new(order.clone())))
    }

    /// Bind an original order to the already configured engine and clear histories.
    /// End-time access is deferred until time advancement, as in live source updates.
    /// This is not the full source adapter constructor or its partially initialized object.
    ///
    /// # Errors
    /// Returns poisoned order access or a missing start time before changing engine state.
    pub fn reset_shared_order(
        &mut self,
        order: &Arc<RwLock<Order>>,
    ) -> Result<(), SaoeAdapterError> {
        let (start, amount) = {
            let order = order
                .read()
                .map_err(|_| SaoeAdapterError::AdapterOrderPoisoned)?;
            (
                order.start_time().ok_or(OrderError::MissingStartTime)?,
                order.amount(),
            )
        };
        let backtest_data = backtest_handles(&self.backtest_data)?;
        let ticks_for_order =
            clone_backtest_field(&backtest_data.ticks_for_order, "ticks_for_order")?;
        let first_tick = *ticks_for_order
            .first()
            .ok_or(SaoeAdapterError::EmptyOrderTicks)?;
        self.position = amount;
        self.session = Some(SaoeAdapterSession {
            order: Arc::clone(order),
            cur_time: first_tick.max(start),
        });
        self.history_exec = Arc::new(RwLock::new(Vec::new()));
        self.history_steps = Arc::new(RwLock::new(Vec::new()));
        self.metrics = None;
        Ok(())
    }

    /// Snapshot the current execution-history object.
    ///
    /// # Errors
    /// Returns when the current history identity is poisoned.
    pub fn history_exec(&self) -> Result<Vec<SaoeMetricRow>, SaoeAdapterError> {
        self.history_exec
            .read()
            .map(|history| history.clone())
            .map_err(|_| SaoeAdapterError::HistoryExecPoisoned)
    }

    /// Snapshot the current step-history object.
    ///
    /// # Errors
    /// Returns when the current history identity is poisoned.
    pub fn history_steps(&self) -> Result<Vec<SaoeMetricRow>, SaoeAdapterError> {
        self.history_steps
            .read()
            .map(|history| history.clone())
            .map_err(|_| SaoeAdapterError::HistoryStepsPoisoned)
    }

    #[must_use]
    pub const fn position(&self) -> f64 {
        self.position
    }

    #[must_use]
    pub const fn cur_time(&self) -> Option<NaiveDateTime> {
        match &self.session {
            Some(session) => Some(session.cur_time),
            None => None,
        }
    }

    #[must_use]
    pub const fn twap_price(&self) -> f64 {
        self.twap_price
    }

    /// Apply one inclusive execution range using Python's mutation order.
    ///
    /// # Errors
    /// Returns invalid ranges, execution timestamps, vector shapes, or plugin failures.
    ///
    /// # Panics
    ///
    /// Panics if a prior Rust panic poisoned the shared calendar-cache state, or
    /// if range indexing violates its nonnegative, representable list-index invariant.
    #[allow(clippy::too_many_lines)]
    pub fn update_executions(
        &mut self,
        executions: &[SharedOrderExecution],
        step_range: (i64, i64),
    ) -> Result<(), SaoeAdapterError> {
        let session = self
            .session
            .clone()
            .ok_or(SaoeAdapterError::NotInitialized)?;
        let backtest_data = backtest_handles(&self.backtest_data)?;
        let ticks_index = clone_backtest_field(&backtest_data.ticks_index, "ticks_index")?;
        let (start_index, end_index) = validate_range(step_range, ticks_index.len())?;
        let start_time = ticks_index[start_index];
        let end_time = ticks_index[end_index];
        let expected = end_index - start_index + 1;
        let mut execution_volume = Array1::zeros(expected);
        let frequency = Frequency {
            count: self.config.data_granularity.into(),
            unit: FrequencyUnit::Minute,
        };
        for execution in executions {
            let live_order = execution
                .order
                .read()
                .map_err(|_| SaoeAdapterError::ExecutionOrderPoisoned)?;
            let execution_start = live_order
                .start_time()
                .ok_or(OrderError::MissingStartTime)?;
            let execution_end = live_order
                .end_time()
                .ok_or(SaoeAdapterError::MissingEndTime)?;
            let (index, _) = day_minute_index_range_with_cache(
                python_clock_precision(execution_start.time()),
                python_clock_precision(execution_end.time()),
                &frequency,
                Region::Cn.code(),
                default_time_calendar_cache(),
            )?;
            let index = index
                .to_usize()
                .expect("bisect-left returns a nonnegative list index");
            if index < start_index || index > end_index {
                return Err(SaoeAdapterError::ExecutionOutsideStep {
                    index,
                    start: step_range.0,
                    end: step_range.1,
                });
            }
            let offset = index - start_index;
            execution_volume[offset] = live_order.deal_amount();
        }
        let mut execution_sum = execution_volume.sum();
        if execution_sum > 0.0 && execution_sum > self.position {
            if execution_sum > self.position + 1.0 {
                self.context.warn_overfill(execution_sum, self.position)?;
            }
            execution_volume *= self.position / execution_sum;
            execution_sum = execution_volume.sum();
        }
        let market = self.read_market(&session.order, start_time, end_time)?;
        let market_price = fill_missing_data(market.price);
        let market_volume = fill_missing_data(market.volume);
        validate_length("market price", market_price.len(), expected)?;
        validate_length("market volume", market_volume.len(), expected)?;
        let timestamps = closed_timestamps(start_time, end_time, self.config.data_granularity);
        validate_length("timestamp", timestamps.len(), expected)?;
        let latest_pa = self.context.latest_price_advantage()?;
        let order = session
            .order
            .read()
            .map_err(|_| SaoeAdapterError::AdapterOrderPoisoned)?
            .clone();
        let mut next_history_exec = self
            .history_exec
            .read()
            .map_err(|_| SaoeAdapterError::HistoryExecPoisoned)?
            .clone();
        let mut cumulative = 0.0;
        for index in 0..expected {
            cumulative += execution_volume[index];
            next_history_exec.push(SaoeMetricRow {
                stock_id: order.stock_id().to_owned(),
                datetime: timestamps[index],
                direction: order.direction(),
                market_volume: market_volume[index],
                market_price: market_price[index],
                amount: execution_volume[index],
                inner_amount: execution_volume[index],
                deal_amount: execution_volume[index],
                trade_price: market_price[index],
                trade_value: market_price[index] * execution_volume[index],
                position: self.position - cumulative,
                ffr: execution_volume[index] / order.amount(),
                pa: latest_pa,
            });
        }
        self.history_exec = Arc::new(RwLock::new(next_history_exec));
        let mut next_history_steps = self
            .history_steps
            .read()
            .map_err(|_| SaoeAdapterError::HistoryStepsPoisoned)?
            .clone();
        next_history_steps.push(collect_single_metric(SingleMetricInput {
            order: &order,
            datetime: session.cur_time,
            market_volume: market_volume.as_slice().unwrap_or(&[]),
            market_price: market_price.as_slice().unwrap_or(&[]),
            amount: execution_sum,
            execution_volume: execution_volume.as_slice().unwrap_or(&[]),
            position: self.position,
            twap_price: self.twap_price,
        }));
        self.history_steps = Arc::new(RwLock::new(next_history_steps));
        self.position -= execution_sum;
        self.session = Some(SaoeAdapterSession {
            cur_time: self.next_time(session.cur_time, &session.order)?,
            order: session.order,
        });
        Ok(())
    }

    /// Generate the upper-level final metric from accumulated histories.
    ///
    /// # Errors
    /// Returns when the adapter was not initialized.
    pub fn finalize_metrics(&mut self) -> Result<(), SaoeAdapterError> {
        let order = self
            .session
            .as_ref()
            .ok_or(SaoeAdapterError::NotInitialized)?
            .order
            .read()
            .map_err(|_| SaoeAdapterError::AdapterOrderPoisoned)?
            .clone();
        let history_exec = self
            .history_exec
            .read()
            .map_err(|_| SaoeAdapterError::HistoryExecPoisoned)?;
        let history_steps = self
            .history_steps
            .read()
            .map_err(|_| SaoeAdapterError::HistoryStepsPoisoned)?;
        let market_volume: Vec<_> = history_exec.iter().map(|row| row.market_volume).collect();
        let market_price: Vec<_> = history_exec.iter().map(|row| row.market_price).collect();
        let execution_volume: Vec<_> = history_exec.iter().map(|row| row.deal_amount).collect();
        let amount = history_steps.iter().map(|row| row.amount).sum();
        let backtest_data = backtest_handles(&self.backtest_data)?;
        let ticks_index = clone_backtest_field(&backtest_data.ticks_index, "ticks_index")?;
        let first_tick = *ticks_index.first().ok_or(SaoeAdapterError::EmptyTicks)?;
        self.metrics = Some(Arc::new(RwLock::new(
            collect_single_metric(SingleMetricInput {
                order: &order,
                datetime: first_tick,
                market_volume: &market_volume,
                market_price: &market_price,
                amount,
                execution_volume: &execution_volume,
                position: self.position,
                twap_price: self.twap_price,
            })
            .as_metrics(),
        )));
        Ok(())
    }

    /// Materialize a fully owned state snapshot.
    ///
    /// # Errors
    /// Returns when uninitialized, when the runtime step fails, or Arrow rejects a history.
    pub fn state_snapshot(&self) -> Result<SaoeState, SaoeAdapterError> {
        self.live_state()?.snapshot()
    }

    /// Return the source-compatible current order and history identities.
    ///
    /// A subsequent update replaces both history identities, just as `dataframe_append` replaces
    /// the adapter attributes. Existing live states therefore retain their old histories, while
    /// in-place mutations made before the next update are consumed by that update.
    ///
    /// # Errors
    /// Returns when uninitialized or when the runtime step callback fails.
    pub fn live_state(&self) -> Result<LiveSaoeState, SaoeAdapterError> {
        let session = self
            .session
            .clone()
            .ok_or(SaoeAdapterError::NotInitialized)?;
        let cur_step = self.context.current_trade_step()? - self.config.start_step;
        let backtest_data = backtest_handles(&self.backtest_data)?;
        Ok(LiveSaoeState {
            order: session.order,
            cur_time: session.cur_time,
            cur_step,
            position: self.position,
            history_exec: Arc::clone(&self.history_exec),
            history_steps: Arc::clone(&self.history_steps),
            metrics: self.metrics.clone(),
            backtest_data: Arc::clone(&self.backtest_data),
            ticks_index: Arc::clone(&backtest_data.ticks_index),
            ticks_for_order: Arc::clone(&backtest_data.ticks_for_order),
            ticks_per_step: self.config.ticks_per_step,
        })
    }

    fn next_time(
        &self,
        current: NaiveDateTime,
        order: &Arc<RwLock<Order>>,
    ) -> Result<NaiveDateTime, SaoeAdapterError> {
        let backtest_data = backtest_handles(&self.backtest_data)?;
        let ticks_index = clone_backtest_field(&backtest_data.ticks_index, "ticks_index")?;
        let mut locations = ticks_index
            .iter()
            .enumerate()
            .filter_map(|(index, value)| (*value == current).then_some(index));
        let current_index = locations
            .next()
            .ok_or(SaoeAdapterError::MissingCurrentTime(current))?;
        if locations.next().is_some() {
            return Err(SaoeAdapterError::MissingCurrentTime(current));
        }
        let stride = self.config.ticks_per_step / self.config.data_granularity;
        let mut next_index = current_index + stride;
        next_index -= next_index % stride;
        let end = order
            .read()
            .map_err(|_| SaoeAdapterError::AdapterOrderPoisoned)?
            .end_time()
            .ok_or(SaoeAdapterError::MissingEndTime)?;
        Ok(
            if next_index < ticks_index.len() && ticks_index[next_index] < end {
                ticks_index[next_index]
            } else {
                end
            },
        )
    }

    fn read_market(
        &self,
        order: &Arc<RwLock<Order>>,
        start: NaiveDateTime,
        end: NaiveDateTime,
    ) -> Result<SaoeMarketSlice, SaoeAdapterError> {
        match &self.market {
            AdapterMarket::Live(source) => Ok(crate::saoe_live_market::read_live_saoe_market(
                &**source, order, start, end,
            )?),
            AdapterMarket::Combined(source) => {
                let (stock, direction) = {
                    let order = order
                        .read()
                        .map_err(|_| SaoeAdapterError::AdapterOrderPoisoned)?;
                    (order.stock_id().to_owned(), order.direction())
                };
                Ok(source.market_slice(&stock, start, end, direction)?)
            }
        }
    }
}

fn validate_config(config: &SaoeAdapterConfig) -> Result<(), SaoeAdapterError> {
    if config.backtest_data.ticks_index.is_empty() {
        return Err(SaoeAdapterError::EmptyTicks);
    }
    if config.backtest_data.ticks_for_order.is_empty() {
        return Err(SaoeAdapterError::EmptyOrderTicks);
    }
    if config.data_granularity == 0 {
        return Err(SaoeAdapterError::ZeroGranularity);
    }
    if config.ticks_per_step % config.data_granularity != 0 {
        return Err(SaoeAdapterError::IncompatibleGranularity {
            ticks_per_step: config.ticks_per_step,
            data_granularity: config.data_granularity,
        });
    }
    i64::try_from(config.data_granularity).map_err(|_| SaoeAdapterError::GranularityTooLarge)?;
    Ok(())
}

impl SaoeStateProvider for ConcreteSaoeStateAdapter {
    fn reset(&mut self, order: &Order) -> Result<(), SaoePluginError> {
        self.reset_order(order)
            .map_err(|error| plugin_error(&error))
    }

    fn state(&self, _order: &Order) -> Result<SaoeState, SaoePluginError> {
        self.state_snapshot().map_err(|error| plugin_error(&error))
    }

    fn update(
        &mut self,
        executions: &[SharedOrderExecution],
        step_range: (i64, i64),
    ) -> Result<(), SaoePluginError> {
        self.update_executions(executions, step_range)
            .map_err(|error| plugin_error(&error))
    }

    fn finalize(&mut self) -> Result<(), SaoePluginError> {
        self.finalize_metrics()
            .map_err(|error| plugin_error(&error))
    }
}

impl crate::saoe_live_registry::LiveSaoeStateAdapter for ConcreteSaoeStateAdapter {
    fn live_state(&self) -> Result<LiveSaoeState, SaoePluginError> {
        ConcreteSaoeStateAdapter::live_state(self).map_err(|error| plugin_error(&error))
    }

    fn state(&self) -> Result<SaoeState, SaoePluginError> {
        self.state_snapshot().map_err(|error| plugin_error(&error))
    }

    fn update(
        &mut self,
        executions: &[SharedOrderExecution],
        range: (i64, i64),
    ) -> Result<(), SaoePluginError> {
        self.update_executions(executions, range)
            .map_err(|error| plugin_error(&error))
    }

    fn finalize(&mut self) -> Result<(), SaoePluginError> {
        self.finalize_metrics()
            .map_err(|error| plugin_error(&error))
    }
}

fn plugin_error(error: &SaoeAdapterError) -> SaoePluginError {
    SaoePluginError {
        message: error.to_string(),
    }
}

fn validate_range(range: (i64, i64), ticks: usize) -> Result<(usize, usize), SaoeAdapterError> {
    if range.0 < 0 || range.1 < range.0 || usize::try_from(range.1).map_or(true, |end| end >= ticks)
    {
        return Err(SaoeAdapterError::InvalidStepRange {
            start: range.0,
            end: range.1,
            ticks,
        });
    }
    Ok((
        usize::try_from(range.0).expect("the validated range start is nonnegative"),
        usize::try_from(range.1).expect("the validated range end fits the tick vector"),
    ))
}

fn validate_length(
    name: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), SaoeAdapterError> {
    if actual != expected {
        return Err(SaoeAdapterError::VectorLength {
            name,
            actual,
            expected,
        });
    }
    Ok(())
}

fn closed_timestamps(
    start: NaiveDateTime,
    end: NaiveDateTime,
    granularity: usize,
) -> Vec<NaiveDateTime> {
    let mut values = Vec::new();
    let mut current = start;
    let delta = TimeDelta::minutes(
        i64::try_from(granularity).expect("adapter construction validates the granularity"),
    );
    while current <= end {
        values.push(current);
        current += delta;
    }
    values
}

#[allow(clippy::manual_midpoint)]
fn fill_missing_data(mut values: Array1<f64>) -> Array1<f64> {
    let mut present: Vec<_> = values
        .iter()
        .copied()
        .filter(|value| !value.is_nan())
        .collect();
    present.sort_by(f64::total_cmp);
    let median = match present.len() {
        0 => f64::NAN,
        length if length % 2 == 1 => present[length / 2],
        length => (present[length / 2 - 1] + present[length / 2]) / 2.0,
    };
    values.mapv_inplace(|value| if value.is_nan() { median } else { value });
    values
}

#[derive(Clone, Copy)]
struct SingleMetricInput<'a> {
    order: &'a Order,
    datetime: NaiveDateTime,
    market_volume: &'a [f64],
    market_price: &'a [f64],
    amount: f64,
    execution_volume: &'a [f64],
    position: f64,
    twap_price: f64,
}

fn collect_single_metric(input: SingleMetricInput<'_>) -> SaoeMetricRow {
    let execution_sum: f64 = input.execution_volume.iter().sum();
    let trade_price = if execution_sum.abs() < EPS {
        0.0
    } else {
        input
            .market_price
            .iter()
            .zip(input.execution_volume)
            .map(|(price, volume)| price * volume)
            .sum::<f64>()
            / execution_sum
    };
    SaoeMetricRow {
        stock_id: input.order.stock_id().to_owned(),
        datetime: input.datetime,
        direction: input.order.direction(),
        market_volume: input.market_volume.iter().sum(),
        market_price: mean(input.market_price),
        amount: input.amount,
        inner_amount: execution_sum,
        deal_amount: execution_sum,
        trade_price,
        trade_value: input
            .market_price
            .iter()
            .zip(input.execution_volume)
            .map(|(price, volume)| price * volume)
            .sum(),
        position: input.position - execution_sum,
        ffr: execution_sum / input.order.amount(),
        pa: price_advantage(trade_price, input.twap_price, input.order.direction()),
    }
}

fn price_advantage(execution_price: f64, baseline_price: f64, direction: OrderDir) -> f64 {
    crate::price_advantage::price_advantage(
        execution_price,
        baseline_price,
        i64::from(direction.value()),
    )
    .expect("OrderDir contains only the two valid directions")
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        f64::NAN
    } else {
        values.iter().sum::<f64>() / values.len().to_f64().expect("slice length fits f64")
    }
}

fn nan_mean(values: impl Iterator<Item = f64>) -> f64 {
    let values: Vec<_> = values.filter(|value| !value.is_nan()).collect();
    mean(&values)
}

fn rows_to_batch(rows: &[SaoeMetricRow]) -> Result<RecordBatch, ArrowError> {
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "datetime",
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            false,
        ),
        Field::new("stock_id", DataType::Utf8, false),
        Field::new("direction", DataType::Int8, false),
        Field::new("market_volume", DataType::Float64, false),
        Field::new("market_price", DataType::Float64, false),
        Field::new("amount", DataType::Float64, false),
        Field::new("inner_amount", DataType::Float64, false),
        Field::new("deal_amount", DataType::Float64, false),
        Field::new("trade_price", DataType::Float64, false),
        Field::new("trade_value", DataType::Float64, false),
        Field::new("position", DataType::Float64, false),
        Field::new("ffr", DataType::Float64, false),
        Field::new("pa", DataType::Float64, false),
    ]));
    let floats = |field: fn(&SaoeMetricRow) -> f64| -> ArrayRef {
        Arc::new(Float64Array::from_iter_values(rows.iter().map(field)))
    };
    let timestamps = rows
        .iter()
        .map(|row| {
            row.datetime.and_utc().timestamp_nanos_opt().ok_or_else(|| {
                ArrowError::ParseError(format!(
                    "SAOE timestamp {} is outside Arrow nanosecond precision",
                    row.datetime
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(TimestampNanosecondArray::from(timestamps)),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.stock_id.as_str()),
            )),
            Arc::new(Int8Array::from_iter_values(rows.iter().map(
                |row| match row.direction {
                    OrderDir::Sell => 0,
                    OrderDir::Buy => 1,
                },
            ))),
            floats(|row| row.market_volume),
            floats(|row| row.market_price),
            floats(|row| row.amount),
            floats(|row| row.inner_amount),
            floats(|row| row.deal_amount),
            floats(|row| row.trade_price),
            floats(|row| row.trade_value),
            floats(|row| row.position),
            floats(|row| row.ffr),
            floats(|row| row.pa),
        ],
    )
}
