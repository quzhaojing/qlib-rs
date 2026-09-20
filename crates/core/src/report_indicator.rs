use std::{
    collections::{BTreeSet, HashMap, HashSet},
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use arrow_array::{Array, ArrayRef, Float64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use chrono::NaiveDateTime;
use indexmap::{IndexMap, IndexSet};
use strum::{Display, EnumString};
use thiserror::Error;

use crate::{
    BasePriceConfig, BasePriceDataProvider, BasePriceError, BasePriceRequest, BasePriceStep,
    BaseVolumePrice, DenseOrderIndicator, NumpyOrderIndicator, Order, OrderDir,
    OrderIndicator as PandasIndicatorBoundary, PandasOrderIndicator, PandasSingleMetric,
    SingleMetric,
    base_price::{calculate_base_volume_price, nan_sum},
};

#[cfg(test)]
#[path = "report_indicator/unit_tests.rs"]
mod unit_tests;

/// Live trade-summary dictionary retained by current state and recorded history rows.
/// Release guards before recalculating this indicator or exporting its history.
pub type SharedTradeIndicator = Arc<RwLock<IndexMap<String, f64>>>;

/// Order-store identity retained by current state and recorded rows.
/// Release guards before invoking indicator operations; store callbacks must not reenter it.
pub type SharedOrderIndicator<S> = Arc<RwLock<S>>;

/// Owned indicator table with an unnamed, lossless datetime index separate from columns.
#[derive(Clone, Debug)]
pub struct TradeIndicatorReport {
    pub timestamps: Vec<NaiveDateTime>,
    pub metrics: RecordBatch,
}

impl TradeIndicatorReport {
    /// Freeze live rows for numeric export without changing their shared identities.
    ///
    /// # Errors
    /// Returns a poisoned row lock; no partially constructed report is published.
    pub fn from_shared_history(
        history: &IndexMap<NaiveDateTime, SharedTradeIndicator>,
    ) -> Result<Self, IndicatorError> {
        let rows = history
            .iter()
            .map(|(time, row)| {
                row.read()
                    .map(|values| (*time, values.clone()))
                    .map_err(|_| IndicatorError::TradeRowPoisoned)
            })
            .collect::<Result<IndexMap<_, _>, _>>()?;
        Ok(Self::from_history(&rows))
    }

    /// Match pandas `DataFrame.from_dict(history, orient="index")` for numeric metrics.
    /// Empty rows disappear; column and index unions follow first encounter order.
    /// Missing entries are IEEE NaN, not Arrow nulls, matching pandas float columns.
    ///
    /// # Panics
    /// Panics only if internally constructed columns have inconsistent lengths.
    #[must_use]
    pub fn from_history(history: &IndexMap<NaiveDateTime, IndexMap<String, f64>>) -> Self {
        let names: IndexSet<&String> = history.values().flat_map(IndexMap::keys).collect();
        let timestamps: IndexSet<NaiveDateTime> = names
            .iter()
            .flat_map(|name| {
                history
                    .iter()
                    .filter_map(move |(time, row)| row.contains_key(*name).then_some(*time))
            })
            .collect();
        let timestamps: Vec<_> = timestamps.into_iter().collect();
        let fields: Vec<_> = names
            .iter()
            .map(|name| Field::new((*name).clone(), DataType::Float64, false))
            .collect();
        let columns: Vec<ArrayRef> = names
            .iter()
            .map(|name| {
                Arc::new(Float64Array::from_iter_values(timestamps.iter().map(
                    |time| history[time].get(*name).copied().unwrap_or(f64::NAN),
                ))) as ArrayRef
            })
            .collect();
        let schema = Arc::new(Schema::new(fields));
        let metrics = if columns.is_empty() {
            RecordBatch::new_empty(schema)
        } else {
            RecordBatch::try_new(schema, columns)
                .expect("indicator report columns share the same ordered index")
        };
        Self {
            timestamps,
            metrics,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MetricSnapshot {
    index: Vec<String>,
    values: Vec<f64>,
}

impl MetricSnapshot {
    /// Creates a validated backend-neutral numeric metric.
    ///
    /// # Errors
    ///
    /// Returns an error for unequal lengths or duplicate stock identifiers.
    pub fn try_new(index: Vec<String>, values: Vec<f64>) -> Result<Self, IndicatorError> {
        if index.len() != values.len() {
            return Err(IndicatorError::LengthMismatch {
                index: index.len(),
                values: values.len(),
            });
        }
        let mut unique = HashSet::with_capacity(index.len());
        for stock in &index {
            if !unique.insert(stock) {
                return Err(IndicatorError::DuplicateStock(stock.clone()));
            }
        }
        Ok(Self { index, values })
    }

    #[must_use]
    pub fn empty() -> Self {
        Self {
            index: Vec::new(),
            values: Vec::new(),
        }
    }

    #[must_use]
    pub fn index(&self) -> &[String] {
        &self.index
    }

    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    fn from_ordered(values: IndexMap<String, f64>) -> Self {
        let (index, values) = values.into_iter().unzip();
        Self { index, values }
    }

    fn positions(&self) -> HashMap<&str, usize> {
        self.index
            .iter()
            .enumerate()
            .map(|(row, stock)| (stock.as_str(), row))
            .collect()
    }
}

impl Default for MetricSnapshot {
    fn default() -> Self {
        Self::empty()
    }
}

/// Object-safe backend access contract exposing only Qlib-owned DTOs.
pub trait IndicatorStoreAccess: Send + Sync {
    fn metric_names(&self) -> Box<dyn Iterator<Item = &str> + '_>;
    fn metric_snapshot(&self, name: &str) -> Option<MetricSnapshot>;
    /// Assigns or replaces one already-validated metric.
    fn assign_snapshot(&mut self, name: &str, metric: MetricSnapshot);
    /// Selects the missing-value and stock-union behavior of the emulated Python backend.
    fn aggregation_mode(&self) -> IndicatorAggregationMode {
        IndicatorAggregationMode::DenseZeroFill
    }
}

/// Owned backend contract used by `Indicator` state and histories.
pub trait IndicatorStore: IndicatorStoreAccess + Clone + Default {}

impl<T> IndicatorStore for T where T: IndicatorStoreAccess + Clone + Default {}

impl IndicatorStoreAccess for NumpyOrderIndicator {
    fn metric_names(&self) -> Box<dyn Iterator<Item = &str> + '_> {
        DenseOrderIndicator::metric_names(self)
    }

    fn metric_snapshot(&self, name: &str) -> Option<MetricSnapshot> {
        let metric = DenseOrderIndicator::metric(self, name)?;
        Some(MetricSnapshot {
            index: metric.index().to_vec(),
            values: metric.values().to_vec(),
        })
    }

    fn assign_snapshot(&mut self, name: &str, metric: MetricSnapshot) {
        let values = metric
            .index
            .into_iter()
            .zip(metric.values.into_iter().map(Some));
        let metric = crate::SingleData::from_f64(values)
            .expect("MetricSnapshot validation guarantees a valid dense metric");
        self.assign(name, metric);
    }
}

impl IndicatorStoreAccess for PandasOrderIndicator {
    fn metric_names(&self) -> Box<dyn Iterator<Item = &str> + '_> {
        PandasIndicatorBoundary::metric_names(self)
    }

    fn metric_snapshot(&self, name: &str) -> Option<MetricSnapshot> {
        PandasIndicatorBoundary::metric(self, name)?;
        let metric = self.get_index_data(name);
        let values = metric
            .values_ref()
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("PandasOrderIndicator index-data access always returns Float64");
        Some(MetricSnapshot {
            index: metric.index().iter().flatten().map(str::to_owned).collect(),
            values: values
                .iter()
                .map(|value| value.unwrap_or(f64::NAN))
                .collect(),
        })
    }

    fn assign_snapshot(&mut self, name: &str, metric: MetricSnapshot) {
        let values = metric.index.into_iter().zip(
            metric
                .values
                .into_iter()
                .map(|value| (!value.is_nan()).then_some(value)),
        );
        let metric = PandasSingleMetric::from_f64(values)
            .expect("MetricSnapshot validation guarantees a valid Arrow metric");
        self.assign(name, metric);
    }

    fn aggregation_mode(&self) -> IndicatorAggregationMode {
        IndicatorAggregationMode::PandasFillValue
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndicatorAggregationMode {
    DenseZeroFill,
    PandasFillValue,
}

#[derive(Clone, Copy, Debug, Display, EnumString, PartialEq, Eq)]
#[strum(serialize_all = "snake_case")]
pub enum IndicatorWeightMethod {
    Mean,
    AmountWeighted,
    ValueWeighted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndicatorConfig {
    pub fulfill_rate: IndicatorWeightMethod,
    pub price_advantage: IndicatorWeightMethod,
}

impl Default for IndicatorConfig {
    fn default() -> Self {
        Self {
            fulfill_rate: IndicatorWeightMethod::Mean,
            price_advantage: IndicatorWeightMethod::Mean,
        }
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum IndicatorError {
    #[error("shared execution list lock poisoned")]
    ExecutionListPoisoned,
    #[error("shared execution order {0} lock poisoned")]
    ExecutionOrderPoisoned(usize),
    #[error("order indicator store lock poisoned")]
    OrderStorePoisoned,
    #[error("trade indicator row lock poisoned")]
    TradeRowPoisoned,
    #[error("metric index and values must have equal lengths, got {index} and {values}")]
    LengthMismatch { index: usize, values: usize },
    #[error("duplicate metric stock identifier: {0}")]
    DuplicateStock(String),
    #[error("indicator metric not found: {0}")]
    MissingMetric(String),
    #[error("indicator metrics have different stock sets: {metric} and {weight}")]
    IndexMismatch { metric: String, weight: String },
    #[error("unsupported indicator weight method: {0}")]
    UnsupportedWeightMethod(String),
}

/// Failures from cross-step base-price aggregation.
#[derive(Debug, Error, PartialEq)]
pub enum AggregateBasePriceError {
    #[error(transparent)]
    Indicator(#[from] IndicatorError),
    /// A missing price requires provider lookup, but the stored direction is not sell or buy.
    #[error("invalid base-price direction {direction} for stock {stock}")]
    InvalidDirection {
        /// Instrument whose direction could not cross the typed provider boundary.
        stock: String,
        /// Original metric value, including NaN when present.
        direction: f64,
    },
    /// Trade-range clipping or market-data lookup failed.
    #[error(transparent)]
    BasePrice(#[from] BasePriceError),
}

/// Typed subset of Python's `indicator_config` consumed by order aggregation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OrderIndicatorAggregationConfig {
    /// Baseline-price source and aggregation rules from Python's `pa_config`.
    pub base_price: BasePriceConfig,
}

/// Stage failure from full outer-layer order-indicator aggregation.
#[derive(Debug, Error, PartialEq)]
pub enum AggregateOrderIndicatorsError {
    #[error(transparent)]
    Decision(#[from] crate::decision_update::LiveDecisionAccessError),
    /// Trade aggregation or price-advantage calculation failed.
    #[error(transparent)]
    Indicator(#[from] IndicatorError),
    /// Cross-step base-price aggregation failed.
    #[error(transparent)]
    BasePrice(#[from] AggregateBasePriceError),
}

#[derive(Clone, Copy, Debug)]
pub struct OrderExecution<'a> {
    pub order: &'a Order,
    pub trade_value: f64,
    pub trade_cost: f64,
    pub trade_price: f64,
}

#[derive(Clone, Copy)]
struct AtomicIndicatorRow {
    amount: f64,
    deal_amount: f64,
    trade_price: f64,
    trade_value: f64,
    trade_cost: f64,
    trade_dir: f64,
}

impl AtomicIndicatorRow {
    fn read(execution: OrderExecution<'_>) -> Self {
        let order = execution.order;
        Self {
            amount: order.amount_delta(),
            deal_amount: order.deal_amount_delta(),
            trade_price: execution.trade_price,
            trade_value: execution.trade_value * f64::from(order.sign()),
            trade_cost: execution.trade_cost,
            trade_dir: f64::from(order.direction().value()),
        }
    }
}

#[derive(Clone, Debug)]
#[allow(clippy::struct_field_names)]
pub struct Indicator<S = NumpyOrderIndicator>
where
    S: IndicatorStore,
{
    order_indicator_history: IndexMap<NaiveDateTime, SharedOrderIndicator<S>>,
    order_indicator: SharedOrderIndicator<S>,
    trade_indicator_history: IndexMap<NaiveDateTime, SharedTradeIndicator>,
    trade_indicator: SharedTradeIndicator,
}

impl Indicator<NumpyOrderIndicator> {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl<S> Default for Indicator<S>
where
    S: IndicatorStore,
{
    fn default() -> Self {
        Self {
            order_indicator_history: IndexMap::new(),
            order_indicator: Arc::new(RwLock::new(S::default())),
            trade_indicator_history: IndexMap::new(),
            trade_indicator: Arc::new(RwLock::new(IndexMap::new())),
        }
    }
}

impl<S> Indicator<S>
where
    S: IndicatorStore,
{
    #[must_use]
    pub fn with_store(store: S) -> Self {
        Self {
            order_indicator: Arc::new(RwLock::new(store)),
            ..Self::default()
        }
    }

    pub fn reset(&mut self) {
        self.order_indicator = Arc::new(RwLock::new(S::default()));
        self.trade_indicator = Arc::new(RwLock::new(IndexMap::new()));
    }

    pub fn record(&mut self, trade_start_time: NaiveDateTime) {
        self.order_indicator_history
            .insert(trade_start_time, self.order_indicator.clone());
        self.trade_indicator_history
            .insert(trade_start_time, self.trade_indicator.clone());
    }

    #[must_use]
    pub fn order_indicator(&self) -> &SharedOrderIndicator<S> {
        &self.order_indicator
    }

    /// Mutably access the live order store without copying its identity.
    ///
    /// # Errors
    /// Returns a poisoned order-store lock.
    pub fn order_indicator_mut(&mut self) -> Result<RwLockWriteGuard<'_, S>, IndicatorError> {
        self.order_indicator
            .write()
            .map_err(|_| IndicatorError::OrderStorePoisoned)
    }

    fn read_order(&self) -> Result<RwLockReadGuard<'_, S>, IndicatorError> {
        self.order_indicator
            .read()
            .map_err(|_| IndicatorError::OrderStorePoisoned)
    }

    #[must_use]
    pub fn trade_indicator(&self) -> &SharedTradeIndicator {
        &self.trade_indicator
    }

    #[must_use]
    pub fn order_indicator_history(&self) -> &IndexMap<NaiveDateTime, SharedOrderIndicator<S>> {
        &self.order_indicator_history
    }

    #[must_use]
    pub fn trade_indicator_history(&self) -> &IndexMap<NaiveDateTime, SharedTradeIndicator> {
        &self.trade_indicator_history
    }

    /// Export all recorded trade summaries without changing the live indicator.
    ///
    /// # Errors
    /// Returns a poisoned history row lock.
    pub fn trade_indicator_report(&self) -> Result<TradeIndicatorReport, IndicatorError> {
        TradeIndicatorReport::from_shared_history(&self.trade_indicator_history)
    }

    /// Returns every current metric in stable column order.
    ///
    /// # Errors
    ///
    /// Returns an error if a backend advertises a name that it cannot read.
    pub fn order_snapshot(&self) -> Result<IndexMap<String, MetricSnapshot>, IndicatorError> {
        let store = self.read_order()?;
        store
            .metric_names()
            .map(|name| {
                store
                    .metric_snapshot(name)
                    .map(|metric| (name.to_owned(), metric))
                    .ok_or_else(|| IndicatorError::MissingMetric(name.to_owned()))
            })
            .collect()
    }

    /// Assigns the innermost order metrics and computes the fill rate.
    ///
    /// # Errors
    /// Returns a poisoned order-store lock before assigning any metrics.
    pub fn update_order_indicators(
        &mut self,
        executions: &[OrderExecution<'_>],
    ) -> Result<(), IndicatorError> {
        let mut rows = IndexMap::new();
        for execution in executions {
            rows.insert(
                execution.order.stock_id().to_owned(),
                AtomicIndicatorRow::read(*execution),
            );
        }
        self.assign_atomic_rows(&rows)
    }

    /// Read original order objects at indicator-update time, then assign numerical metrics.
    ///
    /// No order or result-list guard survives into metric-store callbacks. Duplicate stock IDs
    /// retain their first position and last row values, just as in the source dictionaries.
    ///
    /// # Errors
    /// Returns a poisoned list/order error before assigning metrics, or a metric-store failure.
    pub fn update_shared_order_indicators(
        &mut self,
        executions: &crate::shared_executor_lifecycle::SharedAtomicResult,
    ) -> Result<(), IndicatorError> {
        let mut rows = IndexMap::new();
        {
            let collection = executions
                .lock()
                .map_err(|_| IndicatorError::ExecutionListPoisoned)?;
            for (index, execution) in collection.iter().enumerate() {
                let order = execution
                    .order
                    .read()
                    .map_err(|_| IndicatorError::ExecutionOrderPoisoned(index))?;
                rows.insert(
                    order.stock_id().to_owned(),
                    AtomicIndicatorRow::read(OrderExecution {
                        order: &order,
                        trade_value: execution.trade_value,
                        trade_cost: execution.trade_cost,
                        trade_price: execution.trade_price,
                    }),
                );
            }
        }
        self.assign_atomic_rows(&rows)
    }

    fn assign_atomic_rows(
        &mut self,
        rows: &IndexMap<String, AtomicIndicatorRow>,
    ) -> Result<(), IndicatorError> {
        let snapshot = |value: fn(AtomicIndicatorRow) -> f64| {
            MetricSnapshot::from_ordered(
                rows.iter()
                    .map(|(stock, row)| (stock.clone(), value(*row)))
                    .collect(),
            )
        };
        let amount = snapshot(|row| row.amount);
        let deal_amount = snapshot(|row| row.deal_amount);
        let base_metrics = [
            ("amount", amount.clone()),
            ("inner_amount", amount.clone()),
            ("deal_amount", deal_amount.clone()),
            ("trade_price", snapshot(|row| row.trade_price)),
            ("trade_value", snapshot(|row| row.trade_value)),
            ("trade_cost", snapshot(|row| row.trade_cost)),
            ("trade_dir", snapshot(|row| row.trade_dir)),
            ("pa", snapshot(|_| 0.0)),
        ];
        let mut store = self.order_indicator_mut()?;
        for (name, metric) in base_metrics {
            store.assign_snapshot(name, metric);
        }

        store.assign_snapshot("ffr", calculate_fulfillment_rate(&amount, &deal_amount));
        Ok(())
    }

    /// Aggregates already-computed inner-layer order metrics into this layer.
    ///
    /// Each inner store's `trade_price` is first replaced by signed traded notional, matching
    /// Qlib's observable mutation. The output then restores the volume-weighted trade price.
    ///
    /// # Errors
    ///
    /// Returns an error when an inner store lacks a required metric or when dense multiplication
    /// receives differently labelled deal-amount and trade-price metrics.
    pub fn aggregate_order_trade_info(&mut self, inner: &mut [S]) -> Result<(), IndicatorError> {
        let mut inner: Vec<&mut dyn IndicatorStoreAccess> = inner
            .iter_mut()
            .map(|store| store as &mut dyn IndicatorStoreAccess)
            .collect();
        let mut store = self.order_indicator_mut()?;
        aggregate_stores(&mut *store, &mut inner)
    }

    /// Aggregate retained inner objects, including repeated handles and this output itself.
    ///
    /// Each occurrence is transformed in list order. Only one store is locked at a time;
    /// no guard is retained between stores or returned to the caller. This preserves aliases
    /// without copying stores or creating a multi-lock deadlock. Concurrent operations may
    /// interleave between accesses; callbacks must not reenter the store they are accessing.
    /// Earlier input mutations and output columns survive a later failure.
    ///
    /// # Errors
    /// Returns the first poisoned lock, missing metric or incompatible dense-label error.
    pub fn aggregate_shared_order_trade_info(
        &mut self,
        inner: &[SharedOrderIndicator<S>],
    ) -> Result<(), IndicatorError> {
        let mode = self.read_order()?.aggregation_mode();
        for indicator in inner {
            let mut store = indicator
                .write()
                .map_err(|_| IndicatorError::OrderStorePoisoned)?;
            transform_trade_notional(&mut *store, mode)?;
        }
        aggregate_columns(
            mode,
            inner.len(),
            |index, name| {
                let store = inner[index]
                    .read()
                    .map_err(|_| IndicatorError::OrderStorePoisoned)?;
                store_metric(&*store, name)
            },
            |name, metric| self.assign(name, metric),
        )
    }

    /// Replaces the outer-layer target amount metric from a typed order slice.
    ///
    /// Duplicate stocks retain their first position and their last signed amount, matching
    /// Python dictionary-comprehension semantics. An empty slice installs an empty metric.
    ///
    /// # Errors
    /// Returns a poisoned order-store lock.
    pub fn update_trade_amount(&mut self, orders: &[Order]) -> Result<(), IndicatorError> {
        self.assign("amount", trade_amount_snapshot(orders))
    }

    /// Recomputes outer-layer fulfillment rate by aligning deals to target-amount labels.
    ///
    /// Missing and NaN deal amounts become zero. The output index and order always follow
    /// `amount`, while division retains IEEE NaN and infinity for zero or NaN targets.
    /// All input reads and the output assignment share one exclusive store guard.
    ///
    /// # Errors
    ///
    /// Returns a poisoned store lock or an error when `deal_amount` or `amount` is absent.
    pub fn update_order_fulfill_rate(&mut self) -> Result<(), IndicatorError> {
        let mut store = self.order_indicator_mut()?;
        let deal_amount = store_metric(&*store, "deal_amount")?;
        let amount = store_metric(&*store, "amount")?;
        store.assign_snapshot("ffr", calculate_fulfillment_rate(&amount, &deal_amount));
        Ok(())
    }

    /// Calculate one stock's base price and effective base volume for an execution interval.
    ///
    /// The optional decision range clips timestamps before the provider is queried. An absent or
    /// fully filtered price slice returns `Ok(None)`; TWAP and VWAP otherwise preserve Qlib's
    /// NaN, infinity, duplicate-index, and volume-alignment behavior.
    ///
    /// # Errors
    ///
    /// Returns a typed configuration, trade-range, provider, or missing-volume failure.
    pub fn get_base_volume_price(
        &self,
        request: BasePriceRequest<'_>,
        provider: &dyn BasePriceDataProvider,
    ) -> Result<Option<BaseVolumePrice>, BasePriceError> {
        calculate_base_volume_price(request, provider)
    }

    /// Aggregate per-step base prices and volumes, backfilling only missing prices.
    ///
    /// Inputs and step contexts are paired with `zip`, so an unmatched tail is deliberately
    /// ignored. An empty direction metric leaves existing outputs unchanged; a non-empty
    /// direction metric with no pairs replaces both outputs with empty metrics. Provider and
    /// direction failures occur before either output is assigned.
    ///
    /// # Errors
    ///
    /// Returns a typed invalid-direction, trade-range, missing-volume, or provider failure.
    pub fn aggregate_base_price(
        &mut self,
        inner: &[&dyn IndicatorStoreAccess],
        steps: &[BasePriceStep<'_>],
        provider: &dyn BasePriceDataProvider,
        config: BasePriceConfig,
    ) -> Result<(), AggregateBasePriceError> {
        self.aggregate_base_price_with(
            inner.len(),
            &mut |index| {
                Ok((
                    inner[index]
                        .metric_snapshot("base_price")
                        .unwrap_or_default(),
                    inner[index]
                        .metric_snapshot("base_volume")
                        .unwrap_or_default(),
                ))
            },
            steps,
            provider,
            config,
        )
    }

    /// Aggregate base prices from retained stores, reading each step only when reached.
    ///
    /// Each input guard is released before provider callbacks. Repeated and output-alias
    /// handles are allowed; later steps observe changes made by earlier callbacks.
    /// Unpaired inputs are not read. Outputs are published only after every paired step.
    ///
    /// # Errors
    /// Returns the first store-lock, direction, range or provider failure.
    pub fn aggregate_shared_base_price(
        &mut self,
        inner: &[SharedOrderIndicator<S>],
        steps: &[BasePriceStep<'_>],
        provider: &dyn BasePriceDataProvider,
        config: BasePriceConfig,
    ) -> Result<(), AggregateBasePriceError> {
        self.aggregate_base_price_with(
            inner.len(),
            &mut |index| {
                let store = inner[index]
                    .read()
                    .map_err(|_| IndicatorError::OrderStorePoisoned)?;
                Ok((
                    store.metric_snapshot("base_price").unwrap_or_default(),
                    store.metric_snapshot("base_volume").unwrap_or_default(),
                ))
            },
            steps,
            provider,
            config,
        )
    }

    /// Aggregate retained stores using original decisions, reading each range only for a
    /// reached missing baseline. Neither decision nor input-store guards span callbacks.
    ///
    /// # Errors
    /// Returns the first reached store, decision, direction, range or provider failure.
    pub fn aggregate_live_base_price(
        &mut self,
        inner: &[SharedOrderIndicator<S>],
        steps: &[crate::base_price::LiveBasePriceStep],
        provider: &dyn BasePriceDataProvider,
        config: BasePriceConfig,
    ) -> Result<(), AggregateBasePriceError> {
        self.aggregate_base_price_with(
            inner.len(),
            &mut |index| {
                let store = inner[index]
                    .read()
                    .map_err(|_| IndicatorError::OrderStorePoisoned)?;
                Ok((
                    store.metric_snapshot("base_price").unwrap_or_default(),
                    store.metric_snapshot("base_volume").unwrap_or_default(),
                ))
            },
            steps,
            provider,
            config,
        )
    }

    fn aggregate_base_price_with<T: crate::base_price::BasePriceStepContext>(
        &mut self,
        inner_len: usize,
        read: &mut dyn FnMut(usize) -> Result<(MetricSnapshot, MetricSnapshot), IndicatorError>,
        steps: &[T],
        provider: &dyn BasePriceDataProvider,
        config: BasePriceConfig,
    ) -> Result<(), AggregateBasePriceError> {
        let trade_dir = self
            .read_order()?
            .metric_snapshot("trade_dir")
            .unwrap_or_default();
        if trade_dir.is_empty() {
            return Ok(());
        }

        let mut price_rows = Vec::new();
        let mut volume_rows = Vec::new();
        for (index, step) in (0..inner_len).zip(steps) {
            let (base_price, base_volume) = read(index)?;
            let prices = reindex_values(&base_price, &trade_dir.index);
            let volumes = reindex_values(&base_volume, &trade_dir.index);
            let mut price_row = IndexMap::new();
            let mut volume_row = IndexMap::new();
            for (((stock, direction), price), volume) in trade_dir
                .index
                .iter()
                .zip(&trade_dir.values)
                .zip(prices)
                .zip(volumes)
            {
                if price.is_nan() {
                    let direction = base_price_direction(stock, *direction)?;
                    if let Some(value) = step.calculate(stock, direction, provider, config)? {
                        price_row.insert(stock.clone(), value.base_price);
                        volume_row.insert(stock.clone(), value.base_volume);
                    }
                } else {
                    price_row.insert(stock.clone(), price);
                    volume_row.insert(stock.clone(), volume);
                }
            }
            price_rows.push(MetricSnapshot::from_ordered(price_row));
            volume_rows.push(MetricSnapshot::from_ordered(volume_row));
        }

        let index = sorted_union(price_rows.iter());
        let price_positions: Vec<_> = price_rows.iter().map(MetricSnapshot::positions).collect();
        let volume_positions: Vec<_> = volume_rows.iter().map(MetricSnapshot::positions).collect();
        let mut total_volumes = Vec::with_capacity(index.len());
        let mut aggregate_prices = Vec::with_capacity(index.len());
        for stock in &index {
            let total_volume = nan_sum(
                volume_rows
                    .iter()
                    .zip(&volume_positions)
                    .map(|(row, positions)| lookup(row, positions, stock)),
            );
            let weighted_price = nan_sum(
                price_rows
                    .iter()
                    .zip(&price_positions)
                    .zip(volume_rows.iter().zip(&volume_positions))
                    .map(|((price, price_positions), (volume, volume_positions))| {
                        lookup(price, price_positions, stock)
                            * lookup(volume, volume_positions, stock)
                    }),
            );
            total_volumes.push(total_volume);
            aggregate_prices.push(weighted_price / total_volume);
        }
        // Provider callbacks run without the order lock. Publish the related outputs
        // under one guard so readers cannot observe a half-published price/volume pair.
        let mut store = self.order_indicator_mut()?;
        store.assign_snapshot(
            "base_volume",
            MetricSnapshot {
                index: index.clone(),
                values: total_volumes,
            },
        );
        store.assign_snapshot(
            "base_price",
            MetricSnapshot {
                index,
                values: aggregate_prices,
            },
        );
        Ok(())
    }

    /// Computes signed execution-price advantage over the externally supplied base price.
    ///
    /// An explicitly empty trade-price metric produces an empty result without requiring the
    /// direction or base-price metrics, matching Qlib's evaluation order.
    /// One exclusive guard prevents interleaved store changes between these reads and assignment.
    ///
    /// # Errors
    ///
    /// Returns a poisoned store lock, missing required metric or incompatible dense stock labels.
    pub fn aggregate_order_price_advantage(&mut self) -> Result<(), IndicatorError> {
        let mut store = self.order_indicator_mut()?;
        let trade_price = store_metric(&*store, "trade_price")?;
        if trade_price.is_empty() {
            store.assign_snapshot("pa", MetricSnapshot::empty());
            return Ok(());
        }
        let trade_dir = store_metric(&*store, "trade_dir")?;
        let base_price = store_metric(&*store, "base_price")?;
        let pa = calculate_price_advantage(
            store.aggregation_mode(),
            &trade_dir,
            &trade_price,
            &base_price,
        )?;
        store.assign_snapshot("pa", pa);
        Ok(())
    }

    /// Runs Qlib's full non-atomic outer-layer order-indicator pipeline in stage order.
    ///
    /// Successful earlier stages remain visible if a later stage fails. Inner `trade_price`
    /// metrics are mutated to signed traded notional by the first stage, exactly as in Python.
    /// The local trade/target/fill stages share an exclusive output guard, released before
    /// external market-data callbacks. Store callbacks must not reenter that output lock.
    ///
    /// # Errors
    ///
    /// Returns the first trade, base-price, or price-advantage stage failure.
    pub fn aggregate_order_indicators(
        &mut self,
        inner: &mut [S],
        outer_orders: &[Order],
        steps: &[BasePriceStep<'_>],
        provider: &dyn BasePriceDataProvider,
        config: OrderIndicatorAggregationConfig,
    ) -> Result<(), AggregateOrderIndicatorsError> {
        {
            // Keep related local reads and writes consistent. This is not a rollback
            // boundary: aggregate_stores retains all inner writes preceding an error.
            let mut store = self.order_indicator_mut()?;
            let mut inner_access: Vec<&mut dyn IndicatorStoreAccess> = inner
                .iter_mut()
                .map(|store| store as &mut dyn IndicatorStoreAccess)
                .collect();
            aggregate_stores(&mut *store, &mut inner_access)?;
            update_outer_target_and_fill(&mut *store, outer_orders);
        }
        // Never retain the local-stage guard across external market-data callbacks.
        let inner_access: Vec<&dyn IndicatorStoreAccess> = inner
            .iter()
            .map(|store| store as &dyn IndicatorStoreAccess)
            .collect();
        self.aggregate_base_price(&inner_access, steps, provider, config.base_price)?;
        self.aggregate_order_price_advantage()?;
        Ok(())
    }

    /// Run the non-atomic pipeline on retained raw inputs without copying or deduplicating.
    ///
    /// Trade aggregation allows output aliases using one store lock at a time. Target/fill
    /// updates then share one output guard, released before base-price provider callbacks.
    /// Completed stages and mutations to retained inner histories survive later failures.
    ///
    /// # Errors
    /// Returns the first trade, lock, base-price or price-advantage failure.
    pub fn aggregate_shared_order_indicators(
        &mut self,
        inner: &[SharedOrderIndicator<S>],
        outer_orders: &[Order],
        steps: &[BasePriceStep<'_>],
        provider: &dyn BasePriceDataProvider,
        config: OrderIndicatorAggregationConfig,
    ) -> Result<(), AggregateOrderIndicatorsError> {
        self.aggregate_shared_order_trade_info(inner)?;
        self.finish_shared_order_indicators(inner, outer_orders, steps, provider, config)
    }

    /// Full nested aggregation over original outer orders and retained inner decisions.
    /// Read target orders only after trade aggregation, and ranges only for reached quotes.
    ///
    /// # Errors
    /// Returns the first reached trade, decision, output-store, baseline or advantage failure.
    /// Completed stages remain visible; no decision/order guard spans provider callbacks.
    pub fn aggregate_live_order_indicators(
        &mut self,
        inner: &[SharedOrderIndicator<S>],
        outer: &crate::decision_update::LiveDecisionHandle,
        steps: &[crate::base_price::LiveBasePriceStep],
        provider: &dyn BasePriceDataProvider,
        config: OrderIndicatorAggregationConfig,
    ) -> Result<(), AggregateOrderIndicatorsError> {
        self.aggregate_shared_order_trade_info(inner)?;
        let amount = live_trade_amount_snapshot(&outer.orders()?)?;
        {
            let mut store = self.order_indicator_mut()?;
            assign_outer_target_and_fill(&mut *store, amount);
        }
        self.aggregate_live_base_price(inner, steps, provider, config.base_price)?;
        self.aggregate_order_price_advantage()?;
        Ok(())
    }

    // Separate the completed trade stage from subsequent locks and external callbacks.
    // Another owner can poison the output between stages; reached trade mutations remain.
    fn finish_shared_order_indicators(
        &mut self,
        inner: &[SharedOrderIndicator<S>],
        outer_orders: &[Order],
        steps: &[BasePriceStep<'_>],
        provider: &dyn BasePriceDataProvider,
        config: OrderIndicatorAggregationConfig,
    ) -> Result<(), AggregateOrderIndicatorsError> {
        {
            let mut store = self.order_indicator_mut()?;
            update_outer_target_and_fill(&mut *store, outer_orders);
        }
        self.aggregate_shared_base_price(inner, steps, provider, config.base_price)?;
        self.aggregate_order_price_advantage()?;
        Ok(())
    }

    /// Parses weight-method strings in Python evaluation order and computes all summaries.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown method, missing metric, misalignment, or backend failure.
    pub fn calculate_trade_indicators_str(
        &mut self,
        fulfill_rate: &str,
        price_advantage: &str,
    ) -> Result<(), IndicatorError> {
        let fulfill_rate = parse_weight_method(fulfill_rate)?;
        let price_advantage = parse_weight_method(price_advantage)?;
        self.calculate_trade_indicators(IndicatorConfig {
            fulfill_rate,
            price_advantage,
        })
    }

    /// Computes fill rate, price advantage, positive rate, volume, value, and count summaries.
    ///
    /// # Errors
    ///
    /// Returns an error for missing or differently labelled metrics.
    pub fn calculate_trade_indicators(
        &mut self,
        config: IndicatorConfig,
    ) -> Result<(), IndicatorError> {
        let ffr = self.required_metric("ffr")?;
        let pa = self.required_metric("pa")?;
        let deal_amount = self.required_metric("deal_amount")?;
        let trade_value = self.required_metric("trade_value")?;
        let amount = self.required_metric("amount")?;

        let fulfill_rate =
            Self::aggregate_weighted("ffr", &ffr, config.fulfill_rate, &deal_amount, &trade_value)?;
        let price_advantage = Self::aggregate_weighted(
            "pa",
            &pa,
            config.price_advantage,
            &deal_amount,
            &trade_value,
        )?;
        let positive_rate = usize_as_f64(pa.values.iter().filter(|value| **value > 0.0).count())
            / usize_as_f64(non_nan_count(&pa.values));

        let values = [
            ("ffr".to_owned(), fulfill_rate),
            ("pa".to_owned(), price_advantage),
            ("pos".to_owned(), positive_rate),
            ("deal_amount".to_owned(), sum_abs(&deal_amount.values)),
            ("value".to_owned(), sum_abs(&trade_value.values)),
            (
                "count".to_owned(),
                usize_as_f64(non_nan_count(&amount.values)),
            ),
        ];
        self.trade_indicator
            .write()
            .map_err(|_| IndicatorError::TradeRowPoisoned)?
            .extend(values);
        Ok(())
    }

    fn assign(&mut self, name: &str, metric: MetricSnapshot) -> Result<(), IndicatorError> {
        self.order_indicator_mut()?.assign_snapshot(name, metric);
        Ok(())
    }

    fn required_metric(&self, name: &str) -> Result<MetricSnapshot, IndicatorError> {
        self.read_order()?
            .metric_snapshot(name)
            .ok_or_else(|| IndicatorError::MissingMetric(name.to_owned()))
    }

    fn aggregate_weighted(
        metric_name: &str,
        metric: &MetricSnapshot,
        method: IndicatorWeightMethod,
        deal_amount: &MetricSnapshot,
        trade_value: &MetricSnapshot,
    ) -> Result<f64, IndicatorError> {
        match method {
            IndicatorWeightMethod::Mean => Ok(nan_mean(&metric.values)),
            IndicatorWeightMethod::AmountWeighted => {
                weighted_mean(metric_name, metric, "deal_amount", deal_amount)
            }
            IndicatorWeightMethod::ValueWeighted => {
                weighted_mean(metric_name, metric, "trade_value", trade_value)
            }
        }
    }
}

fn store_metric<S: IndicatorStoreAccess + ?Sized>(
    store: &S,
    name: &str,
) -> Result<MetricSnapshot, IndicatorError> {
    store
        .metric_snapshot(name)
        .ok_or_else(|| IndicatorError::MissingMetric(name.to_owned()))
}

fn calculate_fulfillment_rate(
    amount: &MetricSnapshot,
    deal_amount: &MetricSnapshot,
) -> MetricSnapshot {
    let deal_positions = deal_amount.positions();
    let values = amount
        .index
        .iter()
        .zip(&amount.values)
        .map(|(stock, amount)| {
            let deal = lookup(deal_amount, &deal_positions, stock);
            let deal = if deal.is_nan() { 0.0 } else { deal };
            deal / amount
        })
        .collect();
    MetricSnapshot {
        index: amount.index.clone(),
        values,
    }
}

fn update_outer_target_and_fill(store: &mut dyn IndicatorStoreAccess, orders: &[Order]) {
    assign_outer_target_and_fill(store, trade_amount_snapshot(orders));
}

fn assign_outer_target_and_fill(store: &mut dyn IndicatorStoreAccess, amount: MetricSnapshot) {
    store.assign_snapshot("amount", amount);
    let deal_amount = store.metric_snapshot("deal_amount").unwrap_or_default();
    let amount = store.metric_snapshot("amount").unwrap_or_default();
    store.assign_snapshot("ffr", calculate_fulfillment_rate(&amount, &deal_amount));
}

fn live_trade_amount_snapshot(
    orders: &crate::decision_construction::SharedDecisionOrders,
) -> Result<MetricSnapshot, crate::decision_update::LiveDecisionAccessError> {
    use crate::decision_construction::{DecisionAccessError, DecisionOrderItem};
    let items = orders
        .read()
        .map_err(|_| DecisionAccessError::ListPoisoned)?;
    let mut amounts = IndexMap::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        let DecisionOrderItem::Order(order) = item else {
            return Err(DecisionAccessError::InvalidOrder(index).into());
        };
        let order = order
            .read()
            .map_err(|_| DecisionAccessError::OrderPoisoned(index))?;
        amounts.insert(order.stock_id().to_owned(), order.amount_delta());
    }
    Ok(MetricSnapshot::from_ordered(amounts))
}

fn trade_amount_snapshot(orders: &[Order]) -> MetricSnapshot {
    let mut amounts = IndexMap::with_capacity(orders.len());
    for order in orders {
        amounts.insert(order.stock_id().to_owned(), order.amount_delta());
    }
    MetricSnapshot::from_ordered(amounts)
}

fn aggregate_stores(
    output: &mut dyn IndicatorStoreAccess,
    inner: &mut [&mut dyn IndicatorStoreAccess],
) -> Result<(), IndicatorError> {
    let mode = output.aggregation_mode();
    for indicator in inner.iter_mut() {
        transform_trade_notional(*indicator, mode)?;
    }
    aggregate_columns(
        mode,
        inner.len(),
        |index, name| store_metric(&*inner[index], name),
        |name, metric| {
            output.assign_snapshot(name, metric);
            Ok(())
        },
    )
}

fn transform_trade_notional(
    store: &mut dyn IndicatorStoreAccess,
    mode: IndicatorAggregationMode,
) -> Result<(), IndicatorError> {
    let deal_amount = store_metric(store, "deal_amount")?;
    let trade_price = store_metric(store, "trade_price")?;
    let traded_notional = multiply_metrics(mode, &deal_amount, &trade_price)?;
    store.assign_snapshot("trade_price", traded_notional);
    Ok(())
}

fn aggregate_columns(
    mode: IndicatorAggregationMode,
    inner_len: usize,
    mut read: impl FnMut(usize, &str) -> Result<MetricSnapshot, IndicatorError>,
    mut assign: impl FnMut(&str, MetricSnapshot) -> Result<(), IndicatorError>,
) -> Result<(), IndicatorError> {
    const METRICS: [&str; 6] = [
        "inner_amount",
        "deal_amount",
        "trade_price",
        "trade_value",
        "trade_cost",
        "trade_dir",
    ];
    // NumPy fixes the stock universe before writing any column; Pandas resolves
    // labels per column. Read failures and completed output columns follow source order.
    let dense_index = if mode == IndicatorAggregationMode::DenseZeroFill {
        let first_metrics = (0..inner_len)
            .map(|index| read(index, METRICS[0]))
            .collect::<Result<Vec<_>, _>>()?;
        Some(sorted_union(first_metrics.iter()))
    } else {
        None
    };
    let mut aggregated = Vec::with_capacity(METRICS.len());
    for name in METRICS {
        let metrics = (0..inner_len)
            .map(|index| read(index, name))
            .collect::<Result<Vec<_>, _>>()?;
        let metric = aggregate_metric(mode, &metrics, dense_index.as_deref());
        assign(name, metric.clone())?;
        aggregated.push(metric);
    }
    let [_, deal_amount, trade_price, _, _, trade_dir]: [MetricSnapshot; 6] = aggregated
        .try_into()
        .expect("six requested metrics produce six aggregation results");
    let trade_price = divide_trade_price(&trade_price, &deal_amount);
    let trade_dir = MetricSnapshot {
        index: trade_dir.index,
        values: trade_dir
            .values
            .into_iter()
            .map(|value| f64::from(OrderDir::parse_number(value).value()))
            .collect(),
    };
    assign("trade_price", trade_price)?;
    assign("trade_dir", trade_dir)
}

fn multiply_metrics(
    mode: IndicatorAggregationMode,
    left: &MetricSnapshot,
    right: &MetricSnapshot,
) -> Result<MetricSnapshot, IndicatorError> {
    let right_positions = right.positions();
    if mode == IndicatorAggregationMode::DenseZeroFill
        && (left.index.len() != right.index.len()
            || !left
                .index
                .iter()
                .all(|stock| right_positions.contains_key(stock.as_str())))
    {
        return Err(IndicatorError::IndexMismatch {
            metric: "deal_amount".to_owned(),
            weight: "trade_price".to_owned(),
        });
    }
    let same_stocks = left.index.len() == right.index.len()
        && left
            .index
            .iter()
            .all(|stock| right_positions.contains_key(stock.as_str()));
    let index = if mode == IndicatorAggregationMode::DenseZeroFill || same_stocks {
        left.index.clone()
    } else {
        sorted_union([left, right])
    };
    let left_positions = left.positions();
    let values = index
        .iter()
        .map(|stock| lookup(left, &left_positions, stock) * lookup(right, &right_positions, stock))
        .collect();
    Ok(MetricSnapshot { index, values })
}

fn aggregate_metric(
    mode: IndicatorAggregationMode,
    metrics: &[MetricSnapshot],
    dense_index: Option<&[String]>,
) -> MetricSnapshot {
    let index = dense_index.map_or_else(|| sorted_union(metrics.iter()), <[String]>::to_vec);
    let positions: Vec<_> = metrics.iter().map(MetricSnapshot::positions).collect();
    let values = index
        .iter()
        .map(|stock| match mode {
            IndicatorAggregationMode::DenseZeroFill => metrics
                .iter()
                .zip(&positions)
                .map(|(metric, positions)| {
                    let value = lookup(metric, positions, stock);
                    if value.is_nan() { 0.0 } else { value }
                })
                .sum(),
            IndicatorAggregationMode::PandasFillValue => {
                let mut sum = 0.0;
                let mut has_value = false;
                for (metric, positions) in metrics.iter().zip(&positions) {
                    let value = lookup(metric, positions, stock);
                    if !value.is_nan() {
                        sum += value;
                        has_value = true;
                    }
                }
                if has_value { sum } else { f64::NAN }
            }
        })
        .collect();
    MetricSnapshot { index, values }
}

fn divide_trade_price(price: &MetricSnapshot, amount: &MetricSnapshot) -> MetricSnapshot {
    let index = sorted_union([price, amount]);
    let price_positions = price.positions();
    let amount_positions = amount.positions();
    let values = index
        .iter()
        .map(|stock| {
            let amount = lookup(amount, &amount_positions, stock);
            let amount = if amount == 0.0 { f64::NAN } else { amount };
            lookup(price, &price_positions, stock) / amount
        })
        .collect();
    MetricSnapshot { index, values }
}

fn calculate_price_advantage(
    mode: IndicatorAggregationMode,
    trade_dir: &MetricSnapshot,
    trade_price: &MetricSnapshot,
    base_price: &MetricSnapshot,
) -> Result<MetricSnapshot, IndicatorError> {
    if mode == IndicatorAggregationMode::DenseZeroFill {
        if !same_stock_set(trade_price, base_price) {
            return Err(IndicatorError::IndexMismatch {
                metric: "trade_price".to_owned(),
                weight: "base_price".to_owned(),
            });
        }
        if !same_stock_set(trade_dir, trade_price) {
            return Err(IndicatorError::IndexMismatch {
                metric: "trade_dir".to_owned(),
                weight: "trade_price".to_owned(),
            });
        }
    }
    let index = if mode == IndicatorAggregationMode::DenseZeroFill {
        trade_dir.index.clone()
    } else {
        sorted_union([trade_dir, trade_price, base_price])
    };
    let direction_positions = trade_dir.positions();
    let price_positions = trade_price.positions();
    let base_positions = base_price.positions();
    let values = index
        .iter()
        .map(|stock| {
            let direction = lookup(trade_dir, &direction_positions, stock);
            let price = lookup(trade_price, &price_positions, stock);
            let base = lookup(base_price, &base_positions, stock);
            (1.0 - direction * 2.0) * (price / base - 1.0)
        })
        .collect();
    Ok(MetricSnapshot { index, values })
}

fn same_stock_set(left: &MetricSnapshot, right: &MetricSnapshot) -> bool {
    if left.index.len() != right.index.len() {
        return false;
    }
    let right_positions = right.positions();
    left.index
        .iter()
        .all(|stock| right_positions.contains_key(stock.as_str()))
}

fn sorted_union<'a>(metrics: impl IntoIterator<Item = &'a MetricSnapshot>) -> Vec<String> {
    metrics
        .into_iter()
        .flat_map(|metric| metric.index.iter().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn lookup(metric: &MetricSnapshot, positions: &HashMap<&str, usize>, stock: &str) -> f64 {
    positions
        .get(stock)
        .map_or(f64::NAN, |row| metric.values[*row])
}

fn reindex_values(metric: &MetricSnapshot, index: &[String]) -> Vec<f64> {
    let positions = metric.positions();
    index
        .iter()
        .map(|stock| lookup(metric, &positions, stock))
        .collect()
}

#[allow(
    clippy::float_cmp,
    reason = "Qlib Exchange accepts exactly the numeric IntEnum values 0.0 and 1.0"
)]
fn base_price_direction(stock: &str, direction: f64) -> Result<OrderDir, AggregateBasePriceError> {
    if direction == f64::from(OrderDir::Sell.value()) {
        Ok(OrderDir::Sell)
    } else if direction == f64::from(OrderDir::Buy.value()) {
        Ok(OrderDir::Buy)
    } else {
        Err(AggregateBasePriceError::InvalidDirection {
            stock: stock.to_owned(),
            direction,
        })
    }
}

fn parse_weight_method(value: &str) -> Result<IndicatorWeightMethod, IndicatorError> {
    value
        .parse()
        .map_err(|_| IndicatorError::UnsupportedWeightMethod(value.to_owned()))
}

fn non_nan_count(values: &[f64]) -> usize {
    values.iter().filter(|value| !value.is_nan()).count()
}

#[allow(clippy::cast_precision_loss)]
fn usize_as_f64(value: usize) -> f64 {
    value as f64
}

fn sum_abs(values: &[f64]) -> f64 {
    values
        .iter()
        .filter(|value| !value.is_nan())
        .map(|value| value.abs())
        .sum()
}

fn nan_mean(values: &[f64]) -> f64 {
    let count = non_nan_count(values);
    if count == 0 {
        f64::NAN
    } else {
        values.iter().filter(|value| !value.is_nan()).sum::<f64>() / usize_as_f64(count)
    }
}

fn weighted_mean(
    metric_name: &str,
    metric: &MetricSnapshot,
    weight_name: &str,
    weights: &MetricSnapshot,
) -> Result<f64, IndicatorError> {
    let positions = weights.positions();
    if positions.len() != metric.index.len()
        || !metric
            .index
            .iter()
            .all(|stock| positions.contains_key(stock.as_str()))
    {
        return Err(IndicatorError::IndexMismatch {
            metric: metric_name.to_owned(),
            weight: weight_name.to_owned(),
        });
    }
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (row, stock) in metric.index.iter().enumerate() {
        let weight = weights.values[positions[stock.as_str()]].abs();
        if !weight.is_nan() {
            denominator += weight;
            let product = metric.values[row] * weight;
            if !product.is_nan() {
                numerator += product;
            }
        }
    }
    Ok(numerator / denominator)
}
