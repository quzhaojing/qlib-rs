//! Stable index sorting and slicing for the selection stage of `resam_ts_data`.

use std::{cmp::Ordering, sync::Arc};

use arrow_array::{
    Array, ArrayRef, BooleanArray, PrimitiveArray, RecordBatch, UInt64Array,
    types::{
        ArrowPrimitiveType, TimestampMicrosecondType, TimestampMillisecondType,
        TimestampNanosecondType, TimestampSecondType,
    },
};
use arrow_ord::{
    ord::make_comparator,
    sort::{SortColumn, lexsort_to_indices},
};
use arrow_schema::{DataType, SortOptions, TimeUnit};
use arrow_select::{filter::filter_record_batch, take::take_arrays};
use chrono::NaiveDateTime;
use thiserror::Error;

/// Sort order of a Qlib two-level time-series index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeSeriesIndexOrder {
    /// Equivalent to `MultiIndex[instrument, datetime]`.
    InstrumentDatetime,
    /// Equivalent to `MultiIndex[datetime, instrument]`.
    DatetimeInstrument,
}

/// Explicit Arrow columns that represent the Pandas index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimeSeriesIndex {
    /// A single `DatetimeIndex`.
    Datetime {
        /// Timestamp column name.
        datetime: String,
    },
    /// A two-level instrument/datetime index.
    InstrumentDatetime {
        /// Instrument column name.
        instrument: String,
        /// Timestamp column name.
        datetime: String,
        /// Lexicographic index level order.
        order: TimeSeriesIndexOrder,
    },
}

/// Inclusive exact timestamp bounds. `None` means unbounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TimeRange {
    /// Inclusive lower bound.
    pub start: Option<NaiveDateTime>,
    /// Inclusive upper bound.
    pub end: Option<NaiveDateTime>,
}

/// Failures produced while selecting an Arrow time series.
#[derive(Debug, Error)]
pub enum TimeSeriesSelectionError {
    /// A declared index column is absent.
    #[error("time-series index column not found: {column}")]
    MissingIndexColumn {
        /// Missing name.
        column: String,
    },
    /// A declared index name occurs more than once in the Arrow schema.
    #[error("time-series index column is ambiguous: {column}")]
    AmbiguousIndexColumn {
        /// Ambiguous name.
        column: String,
    },
    /// Instrument and datetime levels cannot reference the same column.
    #[error("instrument and datetime index columns must differ: {column}")]
    DuplicateIndexColumns {
        /// Duplicated name.
        column: String,
    },
    /// The datetime index must use an Arrow timestamp type.
    #[error("datetime index column {column} must be a timestamp, got {data_type}")]
    InvalidDatetimeType {
        /// Column name.
        column: String,
        /// Rejected Arrow type.
        data_type: DataType,
    },
    /// The exact bound lies outside Pandas' nanosecond timestamp range.
    #[error("time bound is outside the Pandas nanosecond range: {timestamp}")]
    TimeBoundOutOfRange {
        /// Rejected timestamp.
        timestamp: NaiveDateTime,
    },
}

/// Sort a time series by its declared index and select an inclusive time range.
///
/// Index columns remain explicit in the returned Arrow batch. Fully duplicate
/// index rows retain source order. With no bounds, null timestamps (Pandas `NaT`)
/// remain at the sorted end; with either bound they are excluded. A result with
/// no rows or no non-index value columns is represented as `None`.
///
/// # Errors
///
/// Returns a typed schema or time-bound error.
///
/// # Panics
///
/// Panics only if an Arrow 59 kernel violates an invariant established from the
/// same valid record batch: equal row counts, in-bounds generated indices, a
/// same-length mask, or schema-preserving column selection.
pub fn select_time_series(
    batch: &RecordBatch,
    index: &TimeSeriesIndex,
    range: TimeRange,
) -> Result<Option<RecordBatch>, TimeSeriesSelectionError> {
    let resolved = ResolvedIndex::new(batch, index)?;
    let timestamp_kind = timestamp_kind(batch.column(resolved.datetime), resolved.datetime_name)?;
    let sorted = stable_sort(batch, &resolved.sort_columns);

    if sorted.num_rows() == 0 || sorted.num_columns() == resolved.index_column_count {
        return Ok(None);
    }
    if range.start.is_none() && range.end.is_none() {
        return Ok(Some(sorted));
    }

    let start = range.start.map(pandas_nanoseconds).transpose()?;
    let end = range.end.map(pandas_nanoseconds).transpose()?;
    let datetime = sorted.column(resolved.datetime);
    let mask = time_mask(datetime.as_ref(), timestamp_kind, start, end);
    let selected = filter_record_batch(&sorted, &mask)
        .expect("mask is derived from a column in the same valid record batch");
    if selected.num_rows() == 0 {
        Ok(None)
    } else {
        Ok(Some(selected))
    }
}

struct ResolvedIndex<'a> {
    datetime: usize,
    datetime_name: &'a str,
    sort_columns: Vec<ArrayRef>,
    index_column_count: usize,
}

impl<'a> ResolvedIndex<'a> {
    #[allow(
        clippy::question_mark,
        reason = "explicit result arms avoid uncovered compiler continuation regions"
    )]
    fn new(
        batch: &RecordBatch,
        index: &'a TimeSeriesIndex,
    ) -> Result<Self, TimeSeriesSelectionError> {
        match index {
            TimeSeriesIndex::Datetime { datetime } => {
                let datetime_index = unique_column_index(batch, datetime)?;
                Ok(Self {
                    datetime: datetime_index,
                    datetime_name: datetime,
                    sort_columns: vec![batch.column(datetime_index).clone()],
                    index_column_count: 1,
                })
            }
            TimeSeriesIndex::InstrumentDatetime {
                instrument,
                datetime,
                order,
            } => {
                if instrument == datetime {
                    return Err(TimeSeriesSelectionError::DuplicateIndexColumns {
                        column: datetime.clone(),
                    });
                }
                let instrument_index = match unique_column_index(batch, instrument) {
                    Ok(index) => index,
                    Err(error) => return Err(error),
                };
                let datetime_index = match unique_column_index(batch, datetime) {
                    Ok(index) => index,
                    Err(error) => return Err(error),
                };
                let instrument_column = batch.column(instrument_index).clone();
                let datetime_column = batch.column(datetime_index).clone();
                let sort_columns = match order {
                    TimeSeriesIndexOrder::InstrumentDatetime => {
                        vec![instrument_column, datetime_column]
                    }
                    TimeSeriesIndexOrder::DatetimeInstrument => {
                        vec![datetime_column, instrument_column]
                    }
                };
                Ok(Self {
                    datetime: datetime_index,
                    datetime_name: datetime,
                    sort_columns,
                    index_column_count: 2,
                })
            }
        }
    }
}

fn unique_column_index(batch: &RecordBatch, name: &str) -> Result<usize, TimeSeriesSelectionError> {
    let mut matches = batch
        .schema_ref()
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, field)| field.name() == name)
        .map(|(index, _)| index);
    let Some(index) = matches.next() else {
        return Err(TimeSeriesSelectionError::MissingIndexColumn {
            column: name.to_owned(),
        });
    };
    if matches.next().is_some() {
        return Err(TimeSeriesSelectionError::AmbiguousIndexColumn {
            column: name.to_owned(),
        });
    }
    Ok(index)
}

#[derive(Clone, Copy)]
enum TimestampKind {
    Second,
    Millisecond,
    Microsecond,
    Nanosecond,
}

fn timestamp_kind(array: &ArrayRef, name: &str) -> Result<TimestampKind, TimeSeriesSelectionError> {
    match array.data_type() {
        DataType::Timestamp(TimeUnit::Second, _) => Ok(TimestampKind::Second),
        DataType::Timestamp(TimeUnit::Millisecond, _) => Ok(TimestampKind::Millisecond),
        DataType::Timestamp(TimeUnit::Microsecond, _) => Ok(TimestampKind::Microsecond),
        DataType::Timestamp(TimeUnit::Nanosecond, _) => Ok(TimestampKind::Nanosecond),
        data_type => Err(TimeSeriesSelectionError::InvalidDatetimeType {
            column: name.to_owned(),
            data_type: data_type.clone(),
        }),
    }
}

fn stable_sort(batch: &RecordBatch, index_columns: &[ArrayRef]) -> RecordBatch {
    let options = SortOptions {
        descending: false,
        nulls_first: false,
    };
    if is_lexsorted(index_columns, options) {
        return batch.clone();
    }

    let mut columns: Vec<_> = index_columns
        .iter()
        .map(|values| SortColumn {
            values: values.clone(),
            options: Some(options),
        })
        .collect();
    let ordinal = UInt64Array::from_iter_values((0_u64..).take(batch.num_rows()));
    columns.push(SortColumn {
        values: Arc::new(ordinal),
        options: Some(options),
    });
    let indices = lexsort_to_indices(&columns, None)
        .expect("validated equal-length self-sort columns have a natural Arrow order");
    let arrays = take_arrays(batch.columns(), &indices, None)
        .expect("sort indices were produced for these arrays and are in bounds");
    RecordBatch::try_new(batch.schema(), arrays)
        .expect("taking every column with one index array preserves schema and row counts")
}

fn is_lexsorted(columns: &[ArrayRef], options: SortOptions) -> bool {
    let comparators: Vec<_> = columns
        .iter()
        .map(|column| make_comparator(column.as_ref(), column.as_ref(), options))
        .collect::<Result<_, _>>()
        .expect("Arrow 59 defines a natural self-order for every valid array data type");
    for row in 1..columns[0].len() {
        for comparator in &comparators {
            match comparator(row - 1, row) {
                Ordering::Less => break,
                Ordering::Greater => return false,
                Ordering::Equal => {}
            }
        }
    }
    true
}

fn pandas_nanoseconds(value: NaiveDateTime) -> Result<i64, TimeSeriesSelectionError> {
    value
        .and_utc()
        .timestamp_nanos_opt()
        .ok_or(TimeSeriesSelectionError::TimeBoundOutOfRange { timestamp: value })
}

fn time_mask(
    array: &dyn Array,
    kind: TimestampKind,
    start: Option<i64>,
    end: Option<i64>,
) -> BooleanArray {
    match kind {
        TimestampKind::Second => {
            primitive_time_mask::<TimestampSecondType>(array, 1_000_000_000, start, end)
        }
        TimestampKind::Millisecond => {
            primitive_time_mask::<TimestampMillisecondType>(array, 1_000_000, start, end)
        }
        TimestampKind::Microsecond => {
            primitive_time_mask::<TimestampMicrosecondType>(array, 1_000, start, end)
        }
        TimestampKind::Nanosecond => {
            primitive_time_mask::<TimestampNanosecondType>(array, 1, start, end)
        }
    }
}

fn primitive_time_mask<T>(
    array: &dyn Array,
    nanoseconds_per_unit: i128,
    start: Option<i64>,
    end: Option<i64>,
) -> BooleanArray
where
    T: ArrowPrimitiveType<Native = i64>,
{
    let values = array
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .expect("timestamp logical type uses its matching primitive array");
    BooleanArray::from(
        (0..values.len())
            .map(|index| {
                if values.is_null(index) {
                    return false;
                }
                let value = i128::from(values.value(index)) * nanoseconds_per_unit;
                start.is_none_or(|bound| value >= i128::from(bound))
                    && end.is_none_or(|bound| value <= i128::from(bound))
            })
            .collect::<Vec<_>>(),
    )
}
