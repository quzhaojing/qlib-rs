//! Direction-aware Exchange quote access and base-price provider adaptation.

use std::{cmp::Ordering, sync::Arc};

use arrow_array::{Array, ArrayRef, Float64Array, PrimitiveArray, types::ArrowPrimitiveType};
use arrow_cast::cast;
use arrow_schema::{DataType, TimeUnit};
use chrono::{DateTime, NaiveDateTime, Utc};
use thiserror::Error;

use crate::{
    BasePriceDataProvider, BasePriceProviderError, MIN_BASE_PRICE, MarketDataSeries,
    MarketDataValue, OrderDir, Quote, QuoteData, QuoteError, QuoteMethod, TimeRange,
};

/// Buy and sell quote expressions configured on Python's `Exchange`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DealPriceFields {
    buy: String,
    sell: String,
}

impl DealPriceFields {
    /// Configure one expression for both directions, prepending `$` when absent.
    ///
    /// # Errors
    ///
    /// Returns an error for the empty string that Python rejects with `IndexError`.
    pub fn shared(field: impl Into<String>) -> Result<Self, ExchangeQuoteError> {
        let mut field = field.into();
        if field.is_empty() {
            return Err(ExchangeQuoteError::EmptySharedDealPrice);
        }
        if !field.starts_with('$') {
            field.insert(0, '$');
        }
        Ok(Self {
            buy: field.clone(),
            sell: field,
        })
    }

    /// Configure separate buy and sell expressions without normalization.
    ///
    /// Python likewise leaves the two entries of a tuple/list unchanged.
    #[must_use]
    pub fn directional(buy: impl Into<String>, sell: impl Into<String>) -> Self {
        Self {
            buy: buy.into(),
            sell: sell.into(),
        }
    }

    #[must_use]
    pub fn buy(&self) -> &str {
        &self.buy
    }

    #[must_use]
    pub fn sell(&self) -> &str {
        &self.sell
    }
}

/// Failures at the typed Exchange-to-Quote compatibility boundary.
#[derive(Debug, Error)]
pub enum ExchangeQuoteError {
    #[error("shared deal-price expression cannot be empty")]
    EmptySharedDealPrice,
    #[error(transparent)]
    Quote(#[from] QuoteError),
    #[error("aggregated quote result must be scalar, not a series")]
    AggregatedSeries,
    #[error("quote scalar must contain exactly one value, got {length}")]
    ScalarLength { length: usize },
    #[error("quote scalar has unsupported Arrow type {data_type}")]
    ScalarType { data_type: DataType },
    #[error("raw quote series must contain exactly two columns, got {columns}")]
    SeriesShape { columns: usize },
    #[error("raw quote series timestamp column has unsupported Arrow type {data_type}")]
    TimestampType { data_type: DataType },
    #[error("raw quote series value column must be Float64, not {data_type}")]
    SeriesValueType { data_type: DataType },
    #[error("raw quote series contains a null timestamp at row {row}")]
    NullTimestamp { row: usize },
    #[error("raw quote timestamp {value} {unit:?} is outside Chrono's range at row {row}")]
    TimestampOutOfRange {
        row: usize,
        value: i64,
        unit: TimeUnit,
    },
}

/// Thin Exchange façade over a replaceable quote implementation.
#[derive(Clone)]
pub struct ExchangeQuoteProvider {
    quote: Arc<dyn Quote>,
    fields: DealPriceFields,
}

impl ExchangeQuoteProvider {
    #[must_use]
    pub fn new(quote: Arc<dyn Quote>, fields: DealPriceFields) -> Self {
        Self { quote, fields }
    }

    #[must_use]
    pub fn fields(&self) -> &DealPriceFields {
        &self.fields
    }

    pub(crate) fn contains_stock(&self, stock: &str) -> bool {
        self.quote
            .get_all_stock()
            .iter()
            .any(|candidate| candidate == stock)
    }

    /// Query Exchange's last valid `$factor` scalar.
    ///
    /// # Errors
    ///
    /// Returns typed quote, scalar-shape, or scalar-dtype failures.
    pub fn get_factor(
        &self,
        stock: &str,
        range: TimeRange,
    ) -> Result<Option<f64>, ExchangeQuoteError> {
        match self
            .quote
            .get_data(stock, range, "$factor", QuoteMethod::LastValid)?
        {
            None => Ok(None),
            Some(QuoteData::Scalar(value)) => scalar_f64(&value).map(Some),
            Some(QuoteData::Series(_)) => Err(ExchangeQuoteError::AggregatedSeries),
        }
    }

    pub(crate) fn get_aggregated_scalar(
        &self,
        stock: &str,
        range: TimeRange,
        field: &str,
        method: QuoteMethod,
    ) -> Result<Option<f64>, ExchangeQuoteError> {
        match self.quote.get_data(stock, range, field, method)? {
            None => Ok(None),
            Some(QuoteData::Scalar(value)) => scalar_f64(&value).map(Some),
            Some(QuoteData::Series(_)) => Err(ExchangeQuoteError::AggregatedSeries),
        }
    }

    pub(crate) fn get_trade_deal_price(
        &self,
        stock: &str,
        range: TimeRange,
        direction: OrderDir,
    ) -> Result<Option<f64>, ExchangeQuoteError> {
        let field = match direction {
            OrderDir::Sell => self.fields.sell(),
            OrderDir::Buy => self.fields.buy(),
        };
        let price = self.get_aggregated_scalar(stock, range, field, QuoteMethod::LastValid)?;
        if price.is_none_or(|price| price.is_nan() || price <= MIN_BASE_PRICE) {
            tracing::warn!(
                stock,
                ?range,
                field,
                "invalid aggregated deal price; falling back to $close"
            );
            return self.get_aggregated_scalar(stock, range, "$close", QuoteMethod::LastValid);
        }
        Ok(price)
    }

    pub(crate) fn stock_is_suspended(
        &self,
        stock: &str,
        range: TimeRange,
    ) -> Result<bool, ExchangeQuoteError> {
        if !self.contains_stock(stock) {
            return Ok(true);
        }
        let Some(close) = self
            .quote
            .get_data(stock, range, "$close", QuoteMethod::Selection)?
        else {
            return Ok(true);
        };
        match close {
            QuoteData::Scalar(value) => scalar_f64(&value).map(f64::is_nan),
            QuoteData::Series(series) => quote_series(&series)
                .map(|series| series.values().iter().all(|value| value.is_nan())),
        }
    }

    pub(crate) fn stock_has_trade_limit(
        &self,
        stock: &str,
        range: TimeRange,
        direction: OrderDir,
    ) -> Result<bool, ExchangeQuoteError> {
        let field = match direction {
            OrderDir::Sell => "limit_sell",
            OrderDir::Buy => "limit_buy",
        };
        self.get_aggregated_scalar(
            stock,
            range,
            field,
            QuoteMethod::BuiltIn(crate::BuiltInAggregation::All),
        )
        .map(|value| value.is_some_and(|value| value.partial_cmp(&0.0) != Some(Ordering::Equal)))
    }

    /// Query the direction-specific price expression and apply Exchange's scalar fallback.
    ///
    /// Selection (`method=None` in Python) returns the raw series unchanged and never falls
    /// back. Aggregated absent, NaN, or `<= 1e-8` values are replaced by `$close` queried with
    /// the same method.
    ///
    /// # Errors
    ///
    /// Returns typed quote, scalar-shape, or scalar-dtype failures.
    pub fn get_deal_price(
        &self,
        stock: &str,
        range: TimeRange,
        direction: OrderDir,
        method: QuoteMethod,
    ) -> Result<Option<QuoteData>, ExchangeQuoteError> {
        let field = match direction {
            OrderDir::Sell => self.fields.sell(),
            OrderDir::Buy => self.fields.buy(),
        };
        let price = self.quote.get_data(stock, range, field, method)?;
        if method != QuoteMethod::Selection && needs_close_fallback(price.as_ref())? {
            tracing::warn!(
                stock,
                ?range,
                field,
                "invalid aggregated deal price; falling back to $close"
            );
            return self
                .quote
                .get_data(stock, range, "$close", method)
                .map_err(Into::into);
        }
        Ok(price)
    }

    /// Query `$volume` with the requested raw or aggregate method.
    ///
    /// # Errors
    ///
    /// Returns the underlying typed quote failure.
    pub fn get_volume(
        &self,
        stock: &str,
        range: TimeRange,
        method: QuoteMethod,
    ) -> Result<Option<QuoteData>, ExchangeQuoteError> {
        self.quote
            .get_data(stock, range, "$volume", method)
            .map_err(Into::into)
    }
}

impl BasePriceDataProvider for ExchangeQuoteProvider {
    fn deal_price(
        &self,
        stock: &str,
        range: TimeRange,
        direction: OrderDir,
    ) -> Result<Option<MarketDataValue>, BasePriceProviderError> {
        self.get_deal_price(stock, range, direction, QuoteMethod::Selection)
            .and_then(quote_data_to_market_data)
            .map_err(|error| provider_error(&error))
    }

    fn volume(
        &self,
        stock: &str,
        range: TimeRange,
    ) -> Result<Option<MarketDataValue>, BasePriceProviderError> {
        self.get_volume(stock, range, QuoteMethod::Selection)
            .and_then(quote_data_to_market_data)
            .map_err(|error| provider_error(&error))
    }
}

fn provider_error(error: &ExchangeQuoteError) -> BasePriceProviderError {
    BasePriceProviderError::Provider {
        message: error.to_string(),
    }
}

fn needs_close_fallback(value: Option<&QuoteData>) -> Result<bool, ExchangeQuoteError> {
    match value {
        None => Ok(true),
        Some(QuoteData::Scalar(value)) => {
            let value = scalar_f64(value)?;
            Ok(value.is_nan() || value <= MIN_BASE_PRICE)
        }
        Some(QuoteData::Series(_)) => Err(ExchangeQuoteError::AggregatedSeries),
    }
}

fn scalar_f64(value: &ArrayRef) -> Result<f64, ExchangeQuoteError> {
    if value.len() != 1 {
        return Err(ExchangeQuoteError::ScalarLength {
            length: value.len(),
        });
    }
    if !matches!(
        value.data_type(),
        DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64
    ) {
        return Err(ExchangeQuoteError::ScalarType {
            data_type: value.data_type().clone(),
        });
    }
    let values = cast(value.as_ref(), &DataType::Float64)
        .expect("every validated quote scalar type casts to Float64");
    let values = values
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("the requested Arrow cast type is Float64");
    Ok(if values.is_null(0) {
        f64::NAN
    } else {
        values.value(0)
    })
}

fn quote_data_to_market_data(
    value: Option<QuoteData>,
) -> Result<Option<MarketDataValue>, ExchangeQuoteError> {
    let Some(value) = value else {
        return Ok(None);
    };
    match value {
        QuoteData::Scalar(value) => scalar_f64(&value).map(MarketDataValue::Scalar).map(Some),
        QuoteData::Series(series) => quote_series(&series).map(MarketDataValue::Series).map(Some),
    }
}

fn quote_series(series: &arrow_array::RecordBatch) -> Result<MarketDataSeries, ExchangeQuoteError> {
    if series.num_columns() != 2 {
        return Err(ExchangeQuoteError::SeriesShape {
            columns: series.num_columns(),
        });
    }
    let timestamps = timestamps(series.column(0))?;
    let values = series.column(1);
    if values.data_type() != &DataType::Float64 {
        return Err(ExchangeQuoteError::SeriesValueType {
            data_type: values.data_type().clone(),
        });
    }
    let values = values
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("the series value type was validated as Float64")
        .iter()
        .map(|value| value.unwrap_or(f64::NAN))
        .collect();
    Ok(MarketDataSeries::try_new(timestamps, values)
        .expect("two columns from one RecordBatch have equal row counts"))
}

fn timestamps(array: &ArrayRef) -> Result<Vec<NaiveDateTime>, ExchangeQuoteError> {
    let (unit, values) = match array.data_type() {
        DataType::Timestamp(TimeUnit::Second, _) => (
            TimeUnit::Second,
            raw_timestamps::<arrow_array::types::TimestampSecondType>(array),
        ),
        DataType::Timestamp(TimeUnit::Millisecond, _) => (
            TimeUnit::Millisecond,
            raw_timestamps::<arrow_array::types::TimestampMillisecondType>(array),
        ),
        DataType::Timestamp(TimeUnit::Microsecond, _) => (
            TimeUnit::Microsecond,
            raw_timestamps::<arrow_array::types::TimestampMicrosecondType>(array),
        ),
        DataType::Timestamp(TimeUnit::Nanosecond, _) => (
            TimeUnit::Nanosecond,
            raw_timestamps::<arrow_array::types::TimestampNanosecondType>(array),
        ),
        data_type => Err(ExchangeQuoteError::TimestampType {
            data_type: data_type.clone(),
        })?,
    };
    values
        .into_iter()
        .enumerate()
        .map(|(row, value)| {
            let value = value.ok_or(ExchangeQuoteError::NullTimestamp { row })?;
            convert_timestamp(value, unit).ok_or(ExchangeQuoteError::TimestampOutOfRange {
                row,
                value,
                unit,
            })
        })
        .collect()
}

fn raw_timestamps<T>(array: &ArrayRef) -> Vec<Option<i64>>
where
    T: ArrowPrimitiveType<Native = i64>,
{
    array
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .expect("the timestamp unit selects its matching Arrow array type")
        .iter()
        .collect()
}

fn convert_timestamp(value: i64, unit: TimeUnit) -> Option<NaiveDateTime> {
    match unit {
        TimeUnit::Second => DateTime::from_timestamp(value, 0).map(|value| value.naive_utc()),
        TimeUnit::Millisecond => {
            DateTime::from_timestamp_millis(value).map(|value| value.naive_utc())
        }
        TimeUnit::Microsecond => {
            DateTime::from_timestamp_micros(value).map(|value| value.naive_utc())
        }
        TimeUnit::Nanosecond => Some(DateTime::<Utc>::from_timestamp_nanos(value).naive_utc()),
    }
}
