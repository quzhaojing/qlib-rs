//! Unified dispatch for `qlib.utils.resam.resam_ts_data`.

use std::{fmt, str::FromStr};

use arrow_array::RecordBatch;
use thiserror::Error;

use crate::{
    AggregationArguments, BuiltInAggregation, TimeRange, TimeSeriesAggregationError,
    TimeSeriesAggregator, TimeSeriesCallableError, TimeSeriesIndex, TimeSeriesSelectionError,
    select_time_series, time_series_aggregation::aggregate_selected,
    time_series_callable::aggregate_selected_with,
};

/// Unified equivalent of Python's `None | str | Callable` method union.
#[derive(Clone, Copy)]
pub enum TimeSeriesMethod<'a> {
    /// Return only the stably sorted, inclusively sliced source data.
    Selection,
    /// Apply a supported Pandas-compatible built-in method.
    BuiltIn(BuiltInAggregation),
    /// Apply an injected batch-oriented callable implementation.
    Callable(&'a dyn TimeSeriesAggregator),
}

impl Default for TimeSeriesMethod<'_> {
    fn default() -> Self {
        Self::BuiltIn(BuiltInAggregation::Last)
    }
}

impl fmt::Debug for TimeSeriesMethod<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Selection => formatter.write_str("Selection"),
            Self::BuiltIn(method) => formatter.debug_tuple("BuiltIn").field(method).finish(),
            Self::Callable(aggregator) => formatter
                .debug_tuple("Callable")
                .field(&aggregator.name())
                .finish(),
        }
    }
}

impl TimeSeriesMethod<'_> {
    /// Parse a supported Python string method.
    ///
    /// # Errors
    ///
    /// Returns [`TimeSeriesAggregationError::UnsupportedMethod`] for a name
    /// outside the intentionally supported built-in set.
    pub fn built_in(method: &str) -> Result<Self, TimeSeriesAggregationError> {
        BuiltInAggregation::from_str(method)
            .map(Self::BuiltIn)
            .map_err(|_| TimeSeriesAggregationError::UnsupportedMethod {
                method: method.to_owned(),
            })
    }
}

/// Failures from unified time-series resampling.
#[derive(Debug, Error)]
pub enum TimeSeriesResampleError {
    /// Stable sorting or inclusive slicing rejected the input.
    #[error(transparent)]
    Selection(#[from] TimeSeriesSelectionError),
    /// A built-in method, layout, argument, or value type was rejected.
    #[error(transparent)]
    BuiltIn(#[from] TimeSeriesAggregationError),
    /// A callable lifecycle or output contract was rejected.
    #[error(transparent)]
    Callable(#[from] TimeSeriesCallableError),
}

/// Sort, inclusively slice, and optionally aggregate a time series once.
///
/// The selection stage is shared by all dispatch variants. Arguments are
/// intentionally ignored for [`TimeSeriesMethod::Selection`], matching Python
/// when `method` is `None` or neither a string nor callable.
///
/// # Errors
///
/// Returns a typed selection, built-in aggregation, or callable error.
///
/// # Panics
///
/// Has the same internal same-batch Arrow invariant assertions as the selected
/// aggregation implementation.
pub fn resample_time_series(
    batch: &RecordBatch,
    index: &TimeSeriesIndex,
    range: TimeRange,
    method: TimeSeriesMethod<'_>,
    arguments: &AggregationArguments,
) -> Result<Option<RecordBatch>, TimeSeriesResampleError> {
    let selected = select_time_series(batch, index, range)?;
    let Some(selected) = selected else {
        return Ok(None);
    };
    match method {
        TimeSeriesMethod::Selection => Ok(Some(selected)),
        TimeSeriesMethod::BuiltIn(method) => {
            aggregate_selected(&selected, index, method, arguments).map_err(Into::into)
        }
        TimeSeriesMethod::Callable(aggregator) => {
            aggregate_selected_with(&selected, index, aggregator, arguments).map_err(Into::into)
        }
    }
}
