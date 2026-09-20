//! Cached construction of Qlib single-asset intraday backtest data.

use std::{
    num::NonZeroUsize,
    sync::{Arc, RwLock, RwLockReadGuard},
};

use arrow_array::RecordBatch;
use arrow_schema::Schema;
use chrono::{NaiveDate, NaiveDateTime, Timelike};
use lru::LruCache;
use ndarray::Array1;
use thiserror::Error;

use crate::{
    LiveSaoeBacktestData, Order, OrderDir, OrderError, SaoeBacktestData, SaoePluginError,
    SharedSaoeBacktestData, TradeRange,
};

pub const SAOE_BACKTEST_DATA_CACHE_CAPACITY: usize = 100;

type BacktestCacheKey = (String, NaiveDate, u32, OrderDir);

fn shared_order(
    order: &Arc<RwLock<Order>>,
) -> Result<RwLockReadGuard<'_, Order>, SaoeBacktestDataLoadError> {
    order
        .read()
        .map_err(|_| SaoeBacktestDataLoadError::OrderPoisoned)
}

/// Exchange data required eagerly by Python's `IntradayBacktestData` constructor.
pub trait SaoeBacktestDataSource: Send + Sync {
    /// Return the exchange quote timestamps in their source order, including duplicates.
    ///
    /// # Errors
    /// Returns quote storage or conversion failures.
    fn quote_timestamps(&self) -> Result<Vec<NaiveDateTime>, SaoePluginError>;

    /// Return deal prices for the closed order-specific tick interval.
    ///
    /// # Errors
    /// Returns quote storage or conversion failures.
    fn deal_prices(
        &self,
        stock_id: &str,
        start: NaiveDateTime,
        end: NaiveDateTime,
        direction: OrderDir,
    ) -> Result<Array1<f64>, SaoePluginError>;

    /// Return market volumes for the closed order-specific tick interval.
    ///
    /// # Errors
    /// Returns quote storage or conversion failures.
    fn market_volumes(
        &self,
        stock_id: &str,
        start: NaiveDateTime,
        end: NaiveDateTime,
    ) -> Result<Array1<f64>, SaoePluginError>;
}

/// Typed failures from loading and slicing intraday backtest data.
#[derive(Debug, Error)]
pub enum SaoeBacktestDataLoadError {
    #[error("SAOE backtest order lock poisoned")]
    OrderPoisoned,
    #[error(transparent)]
    Order(#[from] OrderError),
    #[error("SAOE backtest loading requires a time-based trade range")]
    UnsupportedTradeRange,
    #[error("order end time is required by the SAOE backtest loader")]
    MissingEndTime,
    #[error("no exchange ticks fall inside the order interval")]
    EmptyOrderInterval,
    #[error("no exchange ticks fall inside the time-based trade range")]
    EmptyTradeRange,
    #[error("cached SAOE backtest data lock poisoned")]
    CachedDataPoisoned,
    #[error("cached SAOE backtest data field `{0}` lock poisoned")]
    CachedFieldPoisoned(&'static str),
    #[error(transparent)]
    Source(#[from] SaoePluginError),
}

/// Per-source 100-entry LRU loader matching Qlib's order-day cache key.
pub struct SaoeBacktestDataLoader {
    source: Arc<dyn SaoeBacktestDataSource>,
    cache: LruCache<BacktestCacheKey, SharedSaoeBacktestData>,
}

fn cached_field<T: Clone>(
    field: &RwLock<T>,
    name: &'static str,
) -> Result<T, SaoeBacktestDataLoadError> {
    field
        .read()
        .map(|value| value.clone())
        .map_err(|_| SaoeBacktestDataLoadError::CachedFieldPoisoned(name))
}

fn materialize_cached(
    shared: &SharedSaoeBacktestData,
) -> Result<SaoeBacktestData, SaoeBacktestDataLoadError> {
    let data = shared
        .read()
        .map_err(|_| SaoeBacktestDataLoadError::CachedDataPoisoned)?;
    Ok(SaoeBacktestData {
        ticks_index: cached_field(&data.ticks_index, "ticks_index")?,
        ticks_for_order: cached_field(&data.ticks_for_order, "ticks_for_order")?,
        deal_prices: cached_field(&data.deal_prices, "deal_prices")?,
        market_volumes: cached_field(&data.market_volumes, "market_volumes")?,
        features: cached_field(&data.features, "features")?,
    })
}

impl SaoeBacktestDataLoader {
    /// # Panics
    /// Cannot panic because the fixed cache capacity is a nonzero constant.
    #[must_use]
    pub fn new(source: Arc<dyn SaoeBacktestDataSource>) -> Self {
        Self {
            source,
            cache: LruCache::new(
                NonZeroUsize::new(SAOE_BACKTEST_DATA_CACHE_CAPACITY)
                    .expect("the SAOE backtest cache capacity is nonzero"),
            ),
        }
    }

    #[must_use]
    pub fn cache_len(&self) -> usize {
        self.cache.len()
    }

    /// Clone the original source for live market reads after construction.
    #[must_use]
    pub fn source(&self) -> Arc<dyn SaoeBacktestDataSource> {
        Arc::clone(&self.source)
    }

    /// Load from an original shared order without retaining its guard across source callbacks.
    ///
    /// This follows the source's observable callback boundaries: cache key, quote timestamps, a
    /// post-quote order view for interval/deal lookup, and finally a fresh post-deal stock read for
    /// volume. Mutations performed by a source callback are therefore visible to later stages.
    ///
    /// # Errors
    /// Returns the first reached order-lock, interval, range, or source failure.
    pub fn load_shared(
        &mut self,
        order: &Arc<RwLock<Order>>,
        trade_range: &dyn TradeRange,
    ) -> Result<SharedSaoeBacktestData, SaoeBacktestDataLoadError> {
        let key = {
            let order = shared_order(order)?;
            let (stock_id, day, direction) = order.key_by_day()?;
            (stock_id.to_owned(), day.date(), day.nanosecond(), direction)
        };
        if let Some(cached) = self.cache.get(&key) {
            return Ok(Arc::clone(cached));
        }

        let quote_timestamps = self.source.quote_timestamps()?;
        let (start, end, deal_stock, direction) = {
            let order = shared_order(order)?;
            (
                order.start_time().ok_or(OrderError::MissingStartTime)?,
                order
                    .end_time()
                    .ok_or(SaoeBacktestDataLoadError::MissingEndTime)?,
                order.stock_id().to_owned(),
                order.direction(),
            )
        };
        let after_start: Vec<_> = quote_timestamps
            .into_iter()
            .filter(|timestamp| start <= *timestamp)
            .collect();
        let ticks_index: Vec<_> = after_start
            .into_iter()
            .filter(|timestamp| *timestamp <= end)
            .collect();
        if ticks_index.is_empty() {
            return Err(SaoeBacktestDataLoadError::EmptyOrderInterval);
        }

        let (range_start, range_end) = trade_range
            .time_bounds()
            .ok_or(SaoeBacktestDataLoadError::UnsupportedTradeRange)?;
        let ticks_for_order: Vec<_> = ticks_index
            .iter()
            .copied()
            .filter(|timestamp| {
                let time = timestamp.time();
                range_start <= time && time <= range_end
            })
            .collect();
        let (first, last) = match ticks_for_order.as_slice() {
            [] => return Err(SaoeBacktestDataLoadError::EmptyTradeRange),
            [only] => (*only, *only),
            [first, .., last] => (*first, *last),
        };

        let deal_prices = self
            .source
            .deal_prices(&deal_stock, first, last, direction)?;
        let volume_stock = shared_order(order)?.stock_id().to_owned();
        let market_volumes = self.source.market_volumes(&volume_stock, first, last)?;
        let loaded = SaoeBacktestData {
            ticks_index,
            ticks_for_order,
            deal_prices,
            market_volumes,
            features: RecordBatch::new_empty(Arc::new(Schema::empty())),
        };
        let mut live = LiveSaoeBacktestData::from_owned(loaded);
        live.source_order = Some(Arc::clone(order));
        live.source = Some(Arc::clone(&self.source));
        let loaded = live.into_shared();
        self.cache.put(key, Arc::clone(&loaded));
        Ok(loaded)
    }

    /// Load or clone cached data for one order-day-direction key.
    ///
    /// # Errors
    /// Returns missing order bounds, unsupported range, empty slice, or source failures.
    pub fn load(
        &mut self,
        order: &Order,
        trade_range: &dyn TradeRange,
    ) -> Result<SaoeBacktestData, SaoeBacktestDataLoadError> {
        let start = order.start_time().ok_or(OrderError::MissingStartTime)?;
        let key = (
            order.stock_id().to_owned(),
            start.date(),
            start.nanosecond(),
            order.direction(),
        );
        if let Some(cached) = self.cache.get(&key) {
            return materialize_cached(cached);
        }

        let end = order
            .end_time()
            .ok_or(SaoeBacktestDataLoadError::MissingEndTime)?;
        let ticks_index: Vec<_> = self
            .source
            .quote_timestamps()?
            .into_iter()
            .filter(|timestamp| start <= *timestamp && *timestamp <= end)
            .collect();
        if ticks_index.is_empty() {
            return Err(SaoeBacktestDataLoadError::EmptyOrderInterval);
        }

        let (range_start, range_end) = trade_range
            .time_bounds()
            .ok_or(SaoeBacktestDataLoadError::UnsupportedTradeRange)?;
        let ticks_for_order: Vec<_> = ticks_index
            .iter()
            .copied()
            .filter(|timestamp| {
                let time = timestamp.time();
                range_start <= time && time <= range_end
            })
            .collect();
        let (first, last) = match ticks_for_order.as_slice() {
            [] => return Err(SaoeBacktestDataLoadError::EmptyTradeRange),
            [only] => (*only, *only),
            [first, .., last] => (*first, *last),
        };
        let deal_prices =
            self.source
                .deal_prices(order.stock_id(), first, last, order.direction())?;
        let market_volumes = self.source.market_volumes(order.stock_id(), first, last)?;
        let loaded = SaoeBacktestData {
            ticks_index,
            ticks_for_order,
            deal_prices,
            market_volumes,
            features: RecordBatch::new_empty(Arc::new(Schema::empty())),
        };
        self.cache.put(
            key,
            LiveSaoeBacktestData::from_owned(loaded.clone()).into_shared(),
        );
        Ok(loaded)
    }
}
