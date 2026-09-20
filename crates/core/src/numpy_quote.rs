//! Cached Arrow equivalent of `qlib.backtest.high_performance_ds.NumpyQuote`.

use std::{
    num::NonZeroUsize,
    str::FromStr,
    sync::{Arc, Mutex},
};

use arrow_array::{Array, ArrayRef, Float64Array, RecordBatch};
use arrow_cast::{CastOptions, cast_with_options};
use arrow_schema::{DataType, Schema};
use chrono::TimeDelta;
use lru::LruCache;

use crate::{
    ArrowQuote, BuiltInAggregation, Frequency, FrequencyUnit, ONE_DAY, ONE_MIN, Quote, QuoteData,
    QuoteError, QuoteMethod, Region, TimeRange, is_single_market_value,
};

/// Python's `@lru_cache(maxsize=512)` capacity.
pub const NUMPY_QUOTE_CACHE_CAPACITY: usize = 512;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct QueryKey {
    stock: String,
    range: TimeRange,
    field: String,
    method: Option<String>,
}

/// Float64-normalized, cached quote optimized for repeated backtest reads.
#[derive(Debug)]
pub struct NumpyQuote {
    inner: ArrowQuote,
    frequency: TimeDelta,
    region: String,
    cache: Mutex<LruCache<QueryKey, Option<QuoteData>>>,
}

impl NumpyQuote {
    /// Normalize all value columns to Float64 and construct a cached quote.
    ///
    /// Frequency multipliers are intentionally ignored, matching Python's
    /// `Freq.get_timedelta(1, unit)` call in `NumpyQuote`.
    ///
    /// # Errors
    ///
    /// Returns typed frequency, schema, index, instrument, or Float64 conversion errors.
    ///
    /// # Panics
    ///
    /// Panics only if Arrow violates the declared output type or row-count
    /// invariants of a successful cast, or if the fixed cache capacity is zero.
    pub fn try_new(
        batch: &RecordBatch,
        instrument: impl Into<String>,
        datetime: impl Into<String>,
        frequency: &str,
        region: impl Into<String>,
    ) -> Result<Self, QuoteError> {
        let frequency_value = Frequency::from_str(frequency)?;
        let duration = match frequency_value.unit {
            FrequencyUnit::Minute => ONE_MIN,
            FrequencyUnit::Day => ONE_DAY,
            unit => {
                return Err(QuoteError::UnsupportedNumpyFrequency {
                    input: frequency.to_owned(),
                    unit,
                });
            }
        };
        let instrument = instrument.into();
        let datetime = datetime.into();

        // Validate the index and instrument before attempting value conversion.
        let _ = ArrowQuote::try_new(batch, instrument.clone(), datetime.clone())?;
        let mut fields = Vec::with_capacity(batch.num_columns());
        let mut columns = Vec::with_capacity(batch.num_columns());
        for (field, column) in batch.schema_ref().fields().iter().zip(batch.columns()) {
            if field.name() == &instrument || field.name() == &datetime {
                fields.push(field.clone());
                columns.push(column.clone());
                continue;
            }
            let converted = cast_with_options(
                column.as_ref(),
                &DataType::Float64,
                &CastOptions {
                    safe: false,
                    ..CastOptions::default()
                },
            )
            .map_err(|source| QuoteError::Float64Conversion {
                field: field.name().to_owned(),
                data_type: column.data_type().clone(),
                source,
            })?;
            let converted = converted
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("Arrow Float64 cast returns a Float64Array");
            let values = Float64Array::from_iter((0..converted.len()).map(|row| {
                Some(if converted.is_null(row) {
                    f64::NAN
                } else {
                    converted.value(row)
                })
            }));
            fields.push(Arc::new(
                field
                    .as_ref()
                    .clone()
                    .with_data_type(DataType::Float64)
                    .with_nullable(true),
            ));
            columns.push(Arc::new(values) as ArrayRef);
        }
        let normalized = RecordBatch::try_new(
            Arc::new(Schema::new_with_metadata(
                fields,
                batch.schema_ref().metadata().clone(),
            )),
            columns,
        )
        .expect("value casts retain the source row count");
        Ok(Self {
            inner: ArrowQuote::try_new(&normalized, instrument, datetime)
                .expect("Float64 normalization preserves the validated quote index"),
            frequency: duration,
            region: region.into(),
            cache: Mutex::new(LruCache::new(
                NonZeroUsize::new(NUMPY_QUOTE_CACHE_CAPACITY)
                    .expect("the fixed cache capacity is non-zero"),
            )),
        })
    }

    /// Query using Python's raw optional method string.
    ///
    /// # Errors
    ///
    /// Returns typed bound, region, field, method, selection, or aggregation errors.
    ///
    /// # Panics
    ///
    /// Panics if another thread previously panicked while holding this quote's cache lock.
    pub fn get_data_str(
        &self,
        stock: &str,
        range: TimeRange,
        field: &str,
        method: Option<&str>,
    ) -> Result<Option<QuoteData>, QuoteError> {
        let key = QueryKey {
            stock: stock.to_owned(),
            range,
            field: field.to_owned(),
            method: method.map(str::to_owned),
        };
        if let Some(value) = self
            .cache
            .lock()
            .expect("NumpyQuote cache mutex is not poisoned")
            .get(&key)
            .cloned()
        {
            return Ok(value);
        }
        let value = self.compute(stock, range, field, method)?;
        self.cache
            .lock()
            .expect("NumpyQuote cache mutex is not poisoned")
            .put(key, value.clone());
        Ok(value)
    }

    /// Number of successful queries currently retained by the LRU.
    ///
    /// # Panics
    ///
    /// Panics if another thread previously panicked while holding this quote's cache lock.
    #[must_use]
    pub fn cache_len(&self) -> usize {
        self.cache
            .lock()
            .expect("NumpyQuote cache mutex is not poisoned")
            .len()
    }

    fn compute(
        &self,
        stock: &str,
        range: TimeRange,
        field: &str,
        method: Option<&str>,
    ) -> Result<Option<QuoteData>, QuoteError> {
        if !self.inner.contains_stock(stock) {
            return Ok(None);
        }
        let start = range
            .start
            .ok_or(QuoteError::MissingRangeBound { bound: "start" })?;
        let end = range
            .end
            .ok_or(QuoteError::MissingRangeBound { bound: "end" })?;
        let region = Region::from_str(&self.region).map_err(|_| QuoteError::UnsupportedRegion {
            region: self.region.clone(),
        })?;
        if is_single_market_value(
            start.time(),
            end.signed_duration_since(start),
            self.frequency,
            region,
        ) {
            return match self.inner.get_series(
                stock,
                TimeRange {
                    start: Some(start),
                    end: Some(start),
                },
                field,
            ) {
                Ok(Some(series)) if series.num_rows() == 1 => {
                    Ok(Some(QuoteData::Scalar(series.column(1).slice(0, 1))))
                }
                Ok(Some(series)) => Ok(Some(QuoteData::Series(series))),
                Ok(None) | Err(QuoteError::MissingField { .. }) => Ok(None),
                Err(error) => Err(error),
            };
        }

        match method {
            None => self
                .inner
                .get_data(stock, range, field, QuoteMethod::Selection),
            Some("sum") => self.aggregate(stock, range, field, BuiltInAggregation::Sum),
            Some("mean") => self
                .aggregate(stock, range, field, BuiltInAggregation::Mean)
                .map(normalize_null_mean),
            Some("all") => self.aggregate(stock, range, field, BuiltInAggregation::All),
            Some("last") => self.last(stock, range, field),
            Some("ts_data_last") => self.last_valid(stock, range, field),
            Some(method) => Err(QuoteError::UnsupportedMethod {
                method: method.to_owned(),
            }),
        }
    }

    fn aggregate(
        &self,
        stock: &str,
        range: TimeRange,
        field: &str,
        method: BuiltInAggregation,
    ) -> Result<Option<QuoteData>, QuoteError> {
        self.inner
            .get_data(stock, range, field, QuoteMethod::BuiltIn(method))
    }

    fn last(
        &self,
        stock: &str,
        range: TimeRange,
        field: &str,
    ) -> Result<Option<QuoteData>, QuoteError> {
        let Some(series) = self.inner.get_series(stock, range, field)? else {
            return Ok(None);
        };
        Ok(Some(QuoteData::Scalar(
            series.column(1).slice(series.num_rows() - 1, 1),
        )))
    }

    fn last_valid(
        &self,
        stock: &str,
        range: TimeRange,
        field: &str,
    ) -> Result<Option<QuoteData>, QuoteError> {
        let Some(series) = self.inner.get_series(stock, range, field)? else {
            return Ok(None);
        };
        let value = crate::last_valid_value(series.column(1).as_ref())
            .expect("a selected quote series has at least one row");
        if crate::valid_value::is_missing(value.as_ref(), 0) {
            Ok(None)
        } else {
            Ok(Some(QuoteData::Scalar(value)))
        }
    }
}

impl Quote for NumpyQuote {
    fn get_all_stock(&self) -> Vec<String> {
        self.inner.get_all_stock()
    }

    fn get_data(
        &self,
        stock: &str,
        range: TimeRange,
        field: &str,
        method: QuoteMethod,
    ) -> Result<Option<QuoteData>, QuoteError> {
        self.get_data_str(stock, range, field, method.python_name())
    }
}

fn normalize_null_mean(value: Option<QuoteData>) -> Option<QuoteData> {
    match value {
        Some(QuoteData::Scalar(value)) if value.is_null(0) => {
            Some(QuoteData::Scalar(Arc::new(Float64Array::from(vec![
                f64::NAN,
            ]))))
        }
        value => value,
    }
}
