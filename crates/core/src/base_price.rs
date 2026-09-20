//! Base-volume and base-price calculation used by Qlib's execution reports.

use std::{collections::HashMap, str::FromStr};

use chrono::NaiveDateTime;
use strum::EnumString;
use thiserror::Error;

use crate::{OrderDir, TimeRange, TradeRange, TradeRangeError};

/// Smallest positive price retained by the original Qlib report calculation.
pub const MIN_BASE_PRICE: f64 = 1.0e-8;

/// Supported baseline aggregation rules.
#[derive(Clone, Copy, Debug, Default, EnumString, PartialEq, Eq)]
#[strum(serialize_all = "snake_case", ascii_case_insensitive)]
pub enum BasePriceAggregation {
    /// Every retained quote receives unit weight.
    #[default]
    Twap,
    /// Retained quotes are weighted by volume aligned on timestamp.
    Vwap,
}

/// Supported source for baseline prices.
#[derive(Clone, Copy, Debug, Default, EnumString, PartialEq, Eq)]
#[strum(serialize_all = "snake_case", ascii_case_insensitive)]
pub enum BasePriceSource {
    /// Use the Exchange direction-specific deal-price field.
    #[default]
    DealPrice,
}

/// Typed counterpart of Python's `pa_config` subset used by `_get_base_vol_pri`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BasePriceConfig {
    /// Time- or volume-weighted aggregation.
    pub aggregation: BasePriceAggregation,
    /// Direction-aware price source.
    pub source: BasePriceSource,
}

impl BasePriceConfig {
    /// Parse Python configuration spellings, applying the same defaults.
    ///
    /// Source validation intentionally precedes aggregation validation because the Python
    /// calculation rejects an unsupported price source before it evaluates aggregation.
    ///
    /// # Errors
    ///
    /// Returns a typed error for an unsupported price source or aggregation.
    pub fn parse(aggregation: Option<&str>, source: Option<&str>) -> Result<Self, BasePriceError> {
        let source_text = source.unwrap_or("deal_price");
        let source = BasePriceSource::from_str(source_text).map_err(|_| {
            BasePriceError::UnsupportedPriceSource {
                price_source: source_text.to_owned(),
            }
        })?;
        let aggregation_text = aggregation.unwrap_or("twap");
        let aggregation = BasePriceAggregation::from_str(aggregation_text).map_err(|_| {
            BasePriceError::UnsupportedAggregation {
                aggregation: aggregation_text.to_owned(),
            }
        })?;
        Ok(Self {
            aggregation,
            source,
        })
    }
}

/// Timestamp-labelled numeric values returned by a market-data provider.
///
/// Duplicate timestamps are permitted because Qlib's `SingleData` permits them and their
/// exact-order versus reindex behavior is observable in VWAP calculations.
#[derive(Clone, Debug, PartialEq)]
pub struct MarketDataSeries {
    timestamps: Vec<NaiveDateTime>,
    values: Vec<f64>,
}

impl MarketDataSeries {
    /// Construct one validated timestamp/value series.
    ///
    /// # Errors
    ///
    /// Returns a length mismatch when timestamps and values differ in size.
    pub fn try_new(
        timestamps: Vec<NaiveDateTime>,
        values: Vec<f64>,
    ) -> Result<Self, BasePriceError> {
        if timestamps.len() != values.len() {
            return Err(BasePriceError::SeriesLengthMismatch {
                timestamps: timestamps.len(),
                values: values.len(),
            });
        }
        Ok(Self { timestamps, values })
    }

    /// Timestamp labels in provider order.
    #[must_use]
    pub fn timestamps(&self) -> &[NaiveDateTime] {
        &self.timestamps
    }

    /// Numeric values in provider order.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }
}

/// Scalar-or-series market-data union matching the two supported Python result shapes.
#[derive(Clone, Debug, PartialEq)]
pub enum MarketDataValue {
    /// A scalar is labelled with the effective request start time before calculation.
    Scalar(f64),
    /// An already timestamp-labelled series.
    Series(MarketDataSeries),
}

/// Failure emitted by a replaceable base-price market-data provider.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum BasePriceProviderError {
    /// The provider failed to retrieve or decode the requested market data.
    #[error("base-price market-data provider error: {message}")]
    Provider {
        /// Provider diagnostic retained for adapters and logs.
        message: String,
    },
}

/// Narrow object-safe market-data boundary required by baseline calculation.
pub trait BasePriceDataProvider: Send + Sync {
    /// Return direction-specific deal prices without pre-aggregation.
    ///
    /// `Ok(None)` represents an absent stock/time slice.
    ///
    /// # Errors
    ///
    /// Returns a provider-specific retrieval or decoding failure.
    fn deal_price(
        &self,
        stock: &str,
        range: TimeRange,
        direction: OrderDir,
    ) -> Result<Option<MarketDataValue>, BasePriceProviderError>;

    /// Return raw volume values without pre-aggregation.
    ///
    /// `Ok(None)` is a provider-shape failure for VWAP, matching Python's assertion that a
    /// volume result must become `SingleData`.
    ///
    /// # Errors
    ///
    /// Returns a provider-specific retrieval or decoding failure.
    fn volume(
        &self,
        stock: &str,
        range: TimeRange,
    ) -> Result<Option<MarketDataValue>, BasePriceProviderError>;
}

/// Complete typed input for one `_get_base_vol_pri` calculation.
#[derive(Clone, Copy)]
pub struct BasePriceRequest<'a> {
    /// Requested instrument identifier.
    pub stock: &'a str,
    /// Unclipped closed interval start.
    pub start_time: NaiveDateTime,
    /// Unclipped closed interval end.
    pub end_time: NaiveDateTime,
    /// Buy/sell direction used to select the Exchange deal-price field.
    pub direction: OrderDir,
    /// Optional decision-specific timestamp clipping rule.
    pub trade_range: Option<&'a dyn TradeRange>,
    /// Typed price source and aggregation settings.
    pub config: BasePriceConfig,
}

/// Decision context paired with one inner order-indicator step.
#[derive(Clone, Copy)]
pub struct BasePriceStep<'a> {
    /// Unclipped closed interval start for this inner step.
    pub start_time: NaiveDateTime,
    /// Unclipped closed interval end for this inner step.
    pub end_time: NaiveDateTime,
    /// Optional decision-specific timestamp clipping rule.
    pub trade_range: Option<&'a dyn TradeRange>,
}

/// Original decision plus the inner calendar interval captured before its execution.
/// Range metadata is deliberately not cached: each missing baseline observes current state.
pub struct LiveBasePriceStep {
    pub decision: crate::decision_update::LiveDecisionHandle,
    pub start_time: NaiveDateTime,
    pub end_time: NaiveDateTime,
}

pub(crate) trait BasePriceStepContext {
    fn calculate(
        &self,
        stock: &str,
        direction: OrderDir,
        provider: &dyn BasePriceDataProvider,
        config: BasePriceConfig,
    ) -> Result<Option<BaseVolumePrice>, BasePriceError>;
}

impl BasePriceStepContext for BasePriceStep<'_> {
    fn calculate(
        &self,
        stock: &str,
        direction: OrderDir,
        provider: &dyn BasePriceDataProvider,
        config: BasePriceConfig,
    ) -> Result<Option<BaseVolumePrice>, BasePriceError> {
        calculate_base_volume_price(
            BasePriceRequest {
                stock,
                start_time: self.start_time,
                end_time: self.end_time,
                direction,
                trade_range: self.trade_range,
                config,
            },
            provider,
        )
    }
}

impl BasePriceStepContext for LiveBasePriceStep {
    fn calculate(
        &self,
        stock: &str,
        direction: OrderDir,
        provider: &dyn BasePriceDataProvider,
        config: BasePriceConfig,
    ) -> Result<Option<BaseVolumePrice>, BasePriceError> {
        let range = self.decision.base()?.trade_range;
        calculate_base_volume_price(
            BasePriceRequest {
                stock,
                start_time: self.start_time,
                end_time: self.end_time,
                direction,
                trade_range: range.as_deref(),
                config,
            },
            provider,
        )
    }
}

/// Successful baseline pair. Both values are always present together.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BaseVolumePrice {
    /// Weighted baseline price.
    pub base_price: f64,
    /// Sum of effective weights (point count for TWAP).
    pub base_volume: f64,
}

/// Failures from configuring or calculating a baseline pair.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum BasePriceError {
    /// Reading the current decision at the reached baseline lookup failed.
    #[error(transparent)]
    Decision(#[from] crate::decision_update::LiveDecisionAccessError),
    /// Timestamp and numeric buffers must describe the same number of samples.
    #[error(
        "market-data timestamps and values must have equal lengths, got {timestamps} and {values}"
    )]
    SeriesLengthMismatch { timestamps: usize, values: usize },
    /// Only Qlib's current `deal_price` source is supported.
    #[error("unsupported base-price source: {price_source}")]
    UnsupportedPriceSource { price_source: String },
    /// Only TWAP and VWAP are supported.
    #[error("unsupported base-price aggregation: {aggregation}")]
    UnsupportedAggregation { aggregation: String },
    /// VWAP requires a scalar or timestamp-labelled volume result.
    #[error("VWAP market-data provider returned no volume")]
    MissingVolume,
    /// The optional decision range could not clip the requested timestamps.
    #[error(transparent)]
    TradeRange(#[from] TradeRangeError),
    /// The replaceable market-data implementation failed.
    #[error(transparent)]
    Provider(#[from] BasePriceProviderError),
}

pub(crate) fn calculate_base_volume_price(
    request: BasePriceRequest<'_>,
    provider: &dyn BasePriceDataProvider,
) -> Result<Option<BaseVolumePrice>, BasePriceError> {
    let BasePriceRequest {
        stock,
        mut start_time,
        mut end_time,
        direction,
        trade_range,
        config,
    } = request;
    if let Some(trade_range) = trade_range {
        (start_time, end_time) = trade_range.clip_time_range(start_time, end_time)?;
    }
    let range = TimeRange {
        start: Some(start_time),
        end: Some(end_time),
    };

    let price = match config.source {
        BasePriceSource::DealPrice => provider.deal_price(stock, range, direction)?,
    };
    let Some(price) = price else {
        return Ok(None);
    };
    let price = normalize(price, start_time);
    let price = filter_prices(price);
    if price.values.is_empty() {
        return Ok(None);
    }

    let volume = match config.aggregation {
        BasePriceAggregation::Twap => vec![1.0; price.values.len()],
        BasePriceAggregation::Vwap => {
            let volume = provider
                .volume(stock, range)?
                .ok_or(BasePriceError::MissingVolume)?;
            align_volume(&price.timestamps, normalize(volume, start_time))
        }
    };
    let base_volume = nan_sum(volume.iter().copied());
    let weighted_price = nan_sum(
        price
            .values
            .iter()
            .copied()
            .zip(volume)
            .map(|(price, volume)| price * volume),
    );
    Ok(Some(BaseVolumePrice {
        base_price: weighted_price / base_volume,
        base_volume,
    }))
}

fn normalize(value: MarketDataValue, start_time: NaiveDateTime) -> MarketDataSeries {
    match value {
        MarketDataValue::Scalar(value) => MarketDataSeries {
            timestamps: vec![start_time],
            values: vec![value],
        },
        MarketDataValue::Series(series) => series,
    }
}

fn filter_prices(series: MarketDataSeries) -> MarketDataSeries {
    let (timestamps, values) = series
        .timestamps
        .into_iter()
        .zip(series.values)
        .filter(|(_, value)| *value > MIN_BASE_PRICE)
        .unzip();
    MarketDataSeries { timestamps, values }
}

fn align_volume(price_timestamps: &[NaiveDateTime], volume: MarketDataSeries) -> Vec<f64> {
    if price_timestamps == volume.timestamps {
        return volume.values;
    }
    let positions: HashMap<_, _> = volume.timestamps.into_iter().zip(volume.values).collect();
    price_timestamps
        .iter()
        .map(|timestamp| positions.get(timestamp).copied().unwrap_or(f64::NAN))
        .collect()
}

pub(crate) fn nan_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    values.into_iter().filter(|value| !value.is_nan()).sum()
}
