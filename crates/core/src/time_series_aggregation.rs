//! Built-in reduction stage for `qlib.utils.resam.resam_ts_data`.

use std::{cmp::Ordering, str::FromStr, sync::Arc};

use arrow_arith::aggregate::{bool_and, product, sum};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float16Array, Float32Array, Float64Array, Int64Array,
    RecordBatch, UInt32Array, UInt64Array,
};
use arrow_cast::cast;
use arrow_ord::{
    ord::make_comparator,
    sort::{SortColumn, lexsort_to_indices},
};
use arrow_schema::{DataType, Schema, SortOptions};
use arrow_select::{concat::concat, filter::filter, take::take_arrays};
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use strum::{Display, EnumString};
use thiserror::Error;

use crate::{
    AggregationArguments, TimeRange, TimeSeriesIndex, TimeSeriesSelectionError, ValidEdge,
    select_time_series, valid_value, valid_value::is_missing,
};

/// Pandas reduction names supported by the built-in Rust aggregation path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Display, EnumString)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum BuiltInAggregation {
    /// Pandas `all` with its default missing-value behavior.
    All,
    /// Pandas `sum` with default `min_count=0`.
    Sum,
    /// Pandas `mean` with default `skipna=True`.
    Mean,
    /// Pandas `prod` with default `min_count=0`.
    #[serde(rename = "prod")]
    #[strum(serialize = "prod")]
    Product,
    /// Pandas `GroupBy.first`.
    First,
    /// Pandas `GroupBy.last`.
    Last,
}

/// Failures from built-in time-series aggregation.
#[derive(Debug, Error)]
pub enum TimeSeriesAggregationError {
    /// Sorting or slicing rejected the input.
    #[error(transparent)]
    Selection(#[from] TimeSeriesSelectionError),
    /// The dynamic Pandas method name is not part of the built-in Rust contract.
    #[error("unsupported built-in time-series aggregation method: {method}")]
    UnsupportedMethod {
        /// Rejected method name.
        method: String,
    },
    /// Pandas single-index `first` and `last` require an offset argument.
    #[error("aggregation method {method} requires a multi-index instrument grouping")]
    RequiresInstrumentGrouping {
        /// Rejected method.
        method: BuiltInAggregation,
    },
    /// The selected Arrow value type has no compatible Pandas-like reduction here.
    #[error("aggregation method {method} does not support Arrow type {data_type}")]
    UnsupportedDataType {
        /// Reduction being applied.
        method: BuiltInAggregation,
        /// Rejected type.
        data_type: DataType,
    },
    /// A keyword is not accepted by this Pandas method/layout pair.
    #[error(
        "aggregation method {method} does not accept argument {argument} for this index layout"
    )]
    UnsupportedArgument {
        /// Reduction being configured.
        method: BuiltInAggregation,
        /// Rejected keyword.
        argument: String,
    },
    /// A supported keyword has a non-wire-compatible value.
    #[error("aggregation method {method} argument {argument} must be {expected}")]
    InvalidArgument {
        /// Reduction being configured.
        method: BuiltInAggregation,
        /// Rejected keyword.
        argument: String,
        /// Stable expected-value description.
        expected: &'static str,
    },
}

/// Select and aggregate a time series with a supported Pandas string method.
///
/// A single datetime index produces one Arrow row containing the reduced value
/// columns. A two-level index produces one row per non-null instrument, sorted
/// by instrument, with the instrument column followed by reduced value columns.
///
/// # Errors
///
/// Returns a typed selection, method-shape, or value-type error.
///
/// # Panics
///
/// Panics only if an Arrow 59 kernel violates invariants established from the
/// same valid batch, such as equal row counts, generated in-bounds indices, or
/// schema-preserving casts and concatenation.
pub fn aggregate_time_series(
    batch: &RecordBatch,
    index: &TimeSeriesIndex,
    range: TimeRange,
    method: BuiltInAggregation,
) -> Result<Option<RecordBatch>, TimeSeriesAggregationError> {
    aggregate_time_series_with_arguments(batch, index, range, method, &AggregationArguments::new())
}

/// Select and aggregate with JSON-compatible Python `method_kwargs`.
///
/// # Errors
///
/// Returns a typed selection, method-shape, argument, or value-type error.
///
/// # Panics
///
/// Has the same internal Arrow invariant assertions as [`aggregate_time_series`].
pub fn aggregate_time_series_with_arguments(
    batch: &RecordBatch,
    index: &TimeSeriesIndex,
    range: TimeRange,
    method: BuiltInAggregation,
    arguments: &AggregationArguments,
) -> Result<Option<RecordBatch>, TimeSeriesAggregationError> {
    let selected = match select_time_series(batch, index, range) {
        Ok(selected) => selected,
        Err(error) => return Err(TimeSeriesAggregationError::Selection(error)),
    };
    let Some(selected) = selected else {
        return Ok(None);
    };
    aggregate_selected(&selected, index, method, arguments)
}

/// String-dispatch adapter matching the Python entry point's method names.
///
/// # Errors
///
/// Returns [`TimeSeriesAggregationError::UnsupportedMethod`] for unknown names,
/// or an error from [`aggregate_time_series`] for a known method.
///
/// # Panics
///
/// Has the same internal Arrow invariant assertions as [`aggregate_time_series`].
pub fn aggregate_time_series_str(
    batch: &RecordBatch,
    index: &TimeSeriesIndex,
    range: TimeRange,
    method: &str,
) -> Result<Option<RecordBatch>, TimeSeriesAggregationError> {
    let Ok(parsed) = BuiltInAggregation::from_str(method) else {
        return Err(TimeSeriesAggregationError::UnsupportedMethod {
            method: method.to_owned(),
        });
    };
    aggregate_time_series(batch, index, range, parsed)
}

pub(crate) fn aggregate_selected(
    batch: &RecordBatch,
    index: &TimeSeriesIndex,
    method: BuiltInAggregation,
    arguments: &AggregationArguments,
) -> Result<Option<RecordBatch>, TimeSeriesAggregationError> {
    let grouped = matches!(index, TimeSeriesIndex::InstrumentDatetime { .. });
    if !grouped && matches!(method, BuiltInAggregation::First | BuiltInAggregation::Last) {
        return Err(TimeSeriesAggregationError::RequiresInstrumentGrouping { method });
    }
    let options = BuiltInAggregationOptions::parse(method, grouped, arguments)?;

    let prepared = prepare_time_series_groups(batch, index);
    if prepared.ranges.is_empty() {
        return Ok(None);
    }

    let value_indices: Vec<_> = prepared
        .value_indices()
        .into_iter()
        .filter(|column| options.includes(prepared.batch.column(*column).data_type()))
        .collect();
    let mut fields = Vec::with_capacity(value_indices.len() + usize::from(grouped));
    let mut columns = Vec::with_capacity(fields.capacity());

    if let Some(instrument_index) = prepared.instrument_index {
        let starts = UInt32Array::from_iter_values(
            prepared
                .ranges
                .iter()
                .map(|range| u32::try_from(range.start).expect("Arrow row counts fit u32")),
        );
        let instruments = arrow_select::take::take(
            prepared.batch.column(instrument_index).as_ref(),
            &starts,
            None,
        )
        .expect("group starts were generated from the same instrument array");
        fields.push(prepared.batch.schema_ref().field(instrument_index).clone());
        columns.push(instruments);
    }

    for value_index in value_indices {
        let source = prepared.batch.column(value_index);
        let mut parts = Vec::with_capacity(prepared.ranges.len());
        for range in &prepared.ranges {
            parts.push(reduce_group(
                source.as_ref(),
                range.clone(),
                method,
                options,
            )?);
        }
        let refs: Vec<_> = parts.iter().map(std::convert::AsRef::as_ref).collect();
        let mut reduced = concat(&refs)
            .expect("one-value reductions of one source column share an Arrow data type");
        reduced = restore_group_integer_width(reduced, source.data_type(), grouped, method);
        fields.push(
            prepared
                .batch
                .schema_ref()
                .field(value_index)
                .clone()
                .with_data_type(reduced.data_type().clone())
                .with_nullable(true),
        );
        columns.push(reduced);
    }

    let schema = Arc::new(Schema::new_with_metadata(
        fields,
        prepared.batch.schema_ref().metadata().clone(),
    ));
    Ok(Some(RecordBatch::try_new(schema, columns).expect(
        "aggregated columns have one row per generated group",
    )))
}

#[derive(Clone, Copy)]
struct BuiltInAggregationOptions {
    skipna: bool,
    numeric_only: bool,
    bool_only: bool,
    min_count: i64,
}

impl BuiltInAggregationOptions {
    fn parse(
        method: BuiltInAggregation,
        grouped: bool,
        arguments: &AggregationArguments,
    ) -> Result<Self, TimeSeriesAggregationError> {
        let allowed: &[&str] = match (method, grouped) {
            (BuiltInAggregation::All, true) => &["skipna"],
            (BuiltInAggregation::All, false) => &["axis", "bool_only", "skipna"],
            (BuiltInAggregation::Sum | BuiltInAggregation::Product, true) => {
                &["min_count", "numeric_only"]
            }
            (BuiltInAggregation::Sum | BuiltInAggregation::Product, false) => {
                &["axis", "min_count", "numeric_only", "skipna"]
            }
            (BuiltInAggregation::Mean, false) => &["axis", "numeric_only", "skipna"],
            (BuiltInAggregation::Mean, true) => &["numeric_only"],
            (BuiltInAggregation::First | BuiltInAggregation::Last, _) => {
                &["min_count", "numeric_only", "skipna"]
            }
        };
        for argument in arguments.keys() {
            if !allowed.contains(&argument.as_str()) {
                return Err(TimeSeriesAggregationError::UnsupportedArgument {
                    method,
                    argument: argument.clone(),
                });
            }
        }
        validate_axis(method, arguments)?;
        let skipna = boolean_argument(method, arguments, "skipna", true)?;
        let numeric_only = boolean_argument(method, arguments, "numeric_only", false)?;
        let bool_only = boolean_argument(method, arguments, "bool_only", false)?;
        let default_min_count =
            if matches!(method, BuiltInAggregation::First | BuiltInAggregation::Last) {
                -1
            } else {
                0
            };
        let min_count = integer_argument(method, arguments, "min_count", default_min_count)?;
        Ok(Self {
            skipna,
            numeric_only,
            bool_only,
            min_count,
        })
    }

    fn includes(self, data_type: &DataType) -> bool {
        if self.bool_only {
            data_type == &DataType::Boolean
        } else if self.numeric_only {
            is_numeric_or_boolean(data_type)
        } else {
            true
        }
    }
}

fn boolean_argument(
    method: BuiltInAggregation,
    arguments: &AggregationArguments,
    argument: &'static str,
    default: bool,
) -> Result<bool, TimeSeriesAggregationError> {
    match arguments.get(argument) {
        None => Ok(default),
        Some(serde_json::Value::Bool(value)) => Ok(*value),
        Some(_) => Err(TimeSeriesAggregationError::InvalidArgument {
            method,
            argument: argument.to_owned(),
            expected: "a JSON boolean",
        }),
    }
}

fn integer_argument(
    method: BuiltInAggregation,
    arguments: &AggregationArguments,
    argument: &'static str,
    default: i64,
) -> Result<i64, TimeSeriesAggregationError> {
    match arguments.get(argument) {
        None => Ok(default),
        Some(serde_json::Value::Number(value)) if value.as_i64().is_some() => {
            Ok(value.as_i64().expect("the guarded JSON number is an i64"))
        }
        Some(_) => Err(TimeSeriesAggregationError::InvalidArgument {
            method,
            argument: argument.to_owned(),
            expected: "a signed JSON integer",
        }),
    }
}

fn validate_axis(
    method: BuiltInAggregation,
    arguments: &AggregationArguments,
) -> Result<(), TimeSeriesAggregationError> {
    let Some(axis) = arguments.get("axis") else {
        return Ok(());
    };
    if axis.is_null()
        || axis.as_i64() == Some(0)
        || axis.as_str().is_some_and(|axis| axis == "index")
    {
        Ok(())
    } else {
        Err(TimeSeriesAggregationError::InvalidArgument {
            method,
            argument: "axis".to_owned(),
            expected: "0, \"index\", or null",
        })
    }
}

pub(crate) struct PreparedTimeSeriesGroups {
    pub(crate) batch: RecordBatch,
    pub(crate) instrument_index: Option<usize>,
    pub(crate) datetime_index: usize,
    pub(crate) ranges: Vec<std::ops::Range<usize>>,
}

impl PreparedTimeSeriesGroups {
    pub(crate) fn value_indices(&self) -> Vec<usize> {
        (0..self.batch.num_columns())
            .filter(|column| {
                Some(*column) != self.instrument_index && *column != self.datetime_index
            })
            .collect()
    }
}

pub(crate) fn prepare_time_series_groups(
    batch: &RecordBatch,
    index: &TimeSeriesIndex,
) -> PreparedTimeSeriesGroups {
    let (instrument_name, datetime_name) = match index {
        TimeSeriesIndex::Datetime { datetime } => (None, datetime.as_str()),
        TimeSeriesIndex::InstrumentDatetime {
            instrument,
            datetime,
            ..
        } => (Some(instrument.as_str()), datetime.as_str()),
    };
    let prepared = if let Some(instrument) = instrument_name {
        sort_for_grouping(batch, instrument, datetime_name)
    } else {
        batch.clone()
    };
    let instrument_index = instrument_name.map(|name| {
        prepared
            .schema_ref()
            .index_of(name)
            .expect("selection already validated the unique instrument column")
    });
    let datetime_index = prepared
        .schema_ref()
        .index_of(datetime_name)
        .expect("selection already validated the unique datetime column");
    let ranges = group_ranges(&prepared, instrument_index);
    PreparedTimeSeriesGroups {
        batch: prepared,
        instrument_index,
        datetime_index,
        ranges,
    }
}

fn sort_for_grouping(batch: &RecordBatch, instrument: &str, datetime: &str) -> RecordBatch {
    let schema = batch.schema_ref();
    let instrument_index = schema
        .index_of(instrument)
        .expect("selection already validated the unique instrument column");
    let datetime_index = schema
        .index_of(datetime)
        .expect("selection already validated the unique datetime column");
    let options = Some(SortOptions {
        descending: false,
        nulls_first: false,
    });
    let ordinal = UInt64Array::from_iter_values((0_u64..).take(batch.num_rows()));
    let columns = [
        SortColumn {
            values: batch.column(instrument_index).clone(),
            options,
        },
        SortColumn {
            values: batch.column(datetime_index).clone(),
            options,
        },
        SortColumn {
            values: Arc::new(ordinal),
            options,
        },
    ];
    let indices = lexsort_to_indices(&columns, None)
        .expect("validated index columns are equal-length and naturally ordered");
    let arrays = take_arrays(batch.columns(), &indices, None)
        .expect("sort indices were generated from these batch columns");
    RecordBatch::try_new(batch.schema(), arrays)
        .expect("taking every column with one index array preserves the schema")
}

#[allow(
    clippy::single_range_in_vec_init,
    reason = "an ungrouped series is deliberately represented as one aggregation group"
)]
fn group_ranges(
    batch: &RecordBatch,
    instrument_index: Option<usize>,
) -> Vec<std::ops::Range<usize>> {
    let Some(instrument_index) = instrument_index else {
        return Vec::from([0..batch.num_rows()]);
    };
    let instrument = batch.column(instrument_index);
    let comparator = make_comparator(
        instrument.as_ref(),
        instrument.as_ref(),
        SortOptions::default(),
    )
    .expect("Arrow 59 defines a natural self-order for valid instrument arrays");
    let valid_rows = (0..batch.num_rows())
        .find(|row| instrument.is_null(*row))
        .unwrap_or(batch.num_rows());
    let mut starts = Vec::new();
    for row in 0..valid_rows {
        if row == 0 || comparator(row - 1, row) != Ordering::Equal {
            starts.push(row);
        }
    }
    starts
        .iter()
        .enumerate()
        .map(|(position, start)| *start..starts.get(position + 1).copied().unwrap_or(valid_rows))
        .collect()
}

fn reduce_group(
    array: &dyn Array,
    range: std::ops::Range<usize>,
    method: BuiltInAggregation,
    options: BuiltInAggregationOptions,
) -> Result<ArrayRef, TimeSeriesAggregationError> {
    match method {
        BuiltInAggregation::First => Ok(reduce_edge(array, range, ValidEdge::First, options)),
        BuiltInAggregation::Last => Ok(reduce_edge(array, range, ValidEdge::Last, options)),
        BuiltInAggregation::All => {
            let values = array.slice(range.start, range.len());
            reduce_all_with_missing(values.as_ref(), options.skipna)
        }
        BuiltInAggregation::Sum => {
            reduce_numeric_group(array, range, NumericAggregation::Sum, options)
        }
        BuiltInAggregation::Mean => {
            reduce_numeric_group(array, range, NumericAggregation::Mean, options)
        }
        BuiltInAggregation::Product => {
            reduce_numeric_group(array, range, NumericAggregation::Product, options)
        }
    }
}

fn reduce_edge(
    array: &dyn Array,
    range: std::ops::Range<usize>,
    edge: ValidEdge,
    options: BuiltInAggregationOptions,
) -> ArrayRef {
    let values = array.slice(range.start, range.len());
    if required_count(options.min_count)
        .is_some_and(|required| non_missing_count(values.as_ref()) < required)
    {
        return arrow_array::new_null_array(array.data_type(), 1);
    }
    if options.skipna {
        valid_value(values.as_ref(), edge).expect("groups are non-empty")
    } else {
        let index = match edge {
            ValidEdge::First => 0,
            ValidEdge::Last => values.len() - 1,
        };
        values.slice(index, 1)
    }
}

fn reduce_numeric_group(
    array: &dyn Array,
    range: std::ops::Range<usize>,
    method: NumericAggregation,
    options: BuiltInAggregationOptions,
) -> Result<ArrayRef, TimeSeriesAggregationError> {
    let values = array.slice(range.start, range.len());
    let valid_count = non_missing_count(values.as_ref());
    let cleaned = clean_missing(values.as_ref(), 0..values.len());
    let reduced = reduce_numeric(cleaned.as_ref(), method)?;
    let below_min_count =
        required_count(options.min_count).is_some_and(|required| valid_count < required);
    let propagate_missing = !options.skipna && valid_count != values.len();
    if below_min_count || propagate_missing {
        Ok(arrow_array::new_null_array(reduced.data_type(), 1))
    } else {
        Ok(reduced)
    }
}

fn required_count(min_count: i64) -> Option<usize> {
    (min_count > 0).then(|| usize::try_from(min_count).unwrap_or(usize::MAX))
}

fn non_missing_count(array: &dyn Array) -> usize {
    (0..array.len())
        .filter(|index| !is_missing(array, *index))
        .count()
}

fn reduce_all_with_missing(
    array: &dyn Array,
    skipna: bool,
) -> Result<ArrayRef, TimeSeriesAggregationError> {
    if skipna {
        let cleaned = clean_missing(array, 0..array.len());
        return reduce_numeric(cleaned.as_ref(), NumericAggregation::All);
    }
    if !is_numeric_or_boolean(array.data_type()) {
        return Err(TimeSeriesAggregationError::UnsupportedDataType {
            method: BuiltInAggregation::All,
            data_type: array.data_type().clone(),
        });
    }
    let floats = cast(array, &DataType::Float64)
        .expect("all supported numeric and boolean arrays cast to Float64");
    let floats = floats
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Float64 cast returns Float64Array");
    let mut saw_null = false;
    for index in 0..floats.len() {
        if floats.is_null(index) {
            saw_null = true;
        } else if floats.value(index) == 0.0 {
            return Ok(Arc::new(BooleanArray::from(vec![Some(false)])));
        }
    }
    Ok(Arc::new(BooleanArray::from(vec![if saw_null {
        None
    } else {
        Some(true)
    }])))
}

pub(crate) fn clean_missing(array: &dyn Array, range: std::ops::Range<usize>) -> ArrayRef {
    let values = array.slice(range.start, range.len());
    let mask = BooleanArray::from_iter(
        (0..values.len()).map(|index| Some(!is_missing(values.as_ref(), index))),
    );
    if mask.values().count_set_bits() == values.len() {
        values
    } else {
        filter(values.as_ref(), &mask)
            .expect("the missing-value mask was generated from this exact array slice")
    }
}

#[derive(Clone, Copy)]
enum NumericAggregation {
    All,
    Sum,
    Mean,
    Product,
}

impl From<NumericAggregation> for BuiltInAggregation {
    fn from(value: NumericAggregation) -> Self {
        match value {
            NumericAggregation::All => Self::All,
            NumericAggregation::Sum => Self::Sum,
            NumericAggregation::Mean => Self::Mean,
            NumericAggregation::Product => Self::Product,
        }
    }
}

fn reduce_numeric(
    array: &dyn Array,
    method: NumericAggregation,
) -> Result<ArrayRef, TimeSeriesAggregationError> {
    let reduced = match method {
        NumericAggregation::All => reduce_all(array),
        NumericAggregation::Sum => reduce_sum(array),
        NumericAggregation::Mean => reduce_mean(array),
        NumericAggregation::Product => reduce_product(array),
    };
    reduced.ok_or_else(|| TimeSeriesAggregationError::UnsupportedDataType {
        method: method.into(),
        data_type: array.data_type().clone(),
    })
}

fn is_numeric_or_boolean(data_type: &DataType) -> bool {
    matches!(
        data_type,
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
    )
}

fn reduce_all(array: &dyn Array) -> Option<ArrayRef> {
    if !is_numeric_or_boolean(array.data_type()) {
        return None;
    }
    let truthy = if array.is_empty() {
        true
    } else {
        let floats = cast(array, &DataType::Float64)
            .expect("all supported numeric and boolean arrays cast to Float64");
        let floats = floats
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("Float64 cast returns Float64Array");
        let values =
            BooleanArray::from_iter(floats.values().iter().map(|value| Some(*value != 0.0)));
        bool_and(&values).expect("the truth-value array is non-empty")
    };
    Some(Arc::new(BooleanArray::from(vec![truthy])))
}

fn reduce_sum(array: &dyn Array) -> Option<ArrayRef> {
    let result: ArrayRef = match array.data_type() {
        DataType::Boolean
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64 => {
            let values = cast(array, &DataType::Int64)
                .expect("signed integers and booleans cast losslessly to Int64");
            let values = values
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("Int64 cast returns Int64Array");
            Arc::new(Int64Array::from(vec![sum(values).unwrap_or(0)]))
        }
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
            let values = cast(array, &DataType::UInt64)
                .expect("unsigned integers cast losslessly to UInt64");
            let values = values
                .as_any()
                .downcast_ref::<UInt64Array>()
                .expect("UInt64 cast returns UInt64Array");
            Arc::new(UInt64Array::from(vec![sum(values).unwrap_or(0)]))
        }
        DataType::Float16 => {
            let values = array
                .as_any()
                .downcast_ref::<Float16Array>()
                .expect("Float16 type uses Float16Array");
            Arc::new(Float16Array::from(vec![sum(values).unwrap_or_default()]))
        }
        DataType::Float32 => {
            let values = array
                .as_any()
                .downcast_ref::<Float32Array>()
                .expect("Float32 type uses Float32Array");
            Arc::new(Float32Array::from(vec![sum(values).unwrap_or(0.0)]))
        }
        DataType::Float64 => {
            let values = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("Float64 type uses Float64Array");
            Arc::new(Float64Array::from(vec![sum(values).unwrap_or(0.0)]))
        }
        _ => return None,
    };
    Some(result)
}

pub(crate) fn reduce_product(array: &dyn Array) -> Option<ArrayRef> {
    let result: ArrayRef = match array.data_type() {
        DataType::Boolean
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64 => {
            let values = cast(array, &DataType::Int64)
                .expect("signed integers and booleans cast losslessly to Int64");
            let values = values
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("Int64 cast returns Int64Array");
            Arc::new(Int64Array::from(vec![product(values).unwrap_or(1)]))
        }
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
            let values = cast(array, &DataType::UInt64)
                .expect("unsigned integers cast losslessly to UInt64");
            let values = values
                .as_any()
                .downcast_ref::<UInt64Array>()
                .expect("UInt64 cast returns UInt64Array");
            Arc::new(UInt64Array::from(vec![product(values).unwrap_or(1)]))
        }
        DataType::Float16 => {
            let values = array
                .as_any()
                .downcast_ref::<Float16Array>()
                .expect("Float16 type uses Float16Array");
            Arc::new(Float16Array::from(vec![
                product(values).unwrap_or_else(|| half::f16::from_f32(1.0)),
            ]))
        }
        DataType::Float32 => {
            let values = array
                .as_any()
                .downcast_ref::<Float32Array>()
                .expect("Float32 type uses Float32Array");
            Arc::new(Float32Array::from(vec![product(values).unwrap_or(1.0)]))
        }
        DataType::Float64 => {
            let values = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("Float64 type uses Float64Array");
            Arc::new(Float64Array::from(vec![product(values).unwrap_or(1.0)]))
        }
        _ => return None,
    };
    Some(result)
}

fn reduce_mean(array: &dyn Array) -> Option<ArrayRef> {
    if !is_numeric_or_boolean(array.data_type()) {
        return None;
    }
    if array.is_empty() {
        return Some(match array.data_type() {
            DataType::Float16 | DataType::Float32 => {
                arrow_array::new_null_array(array.data_type(), 1)
            }
            _ => arrow_array::new_null_array(&DataType::Float64, 1),
        });
    }
    let floats = cast(array, &DataType::Float64)
        .expect("all supported numeric and boolean arrays cast to Float64");
    let floats = floats
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Float64 cast returns Float64Array");
    let count = array
        .len()
        .to_f64()
        .expect("Arrow array lengths fit a floating mean denominator");
    let mean = sum(floats).expect("cleaned non-empty numeric arrays have a sum") / count;
    let result = Arc::new(Float64Array::from(vec![mean])) as ArrayRef;
    Some(match array.data_type() {
        DataType::Float16 | DataType::Float32 => cast(result.as_ref(), array.data_type())
            .expect("a finite-width float mean casts back to its source float type"),
        _ => result,
    })
}

fn restore_group_integer_width(
    reduced: ArrayRef,
    source_type: &DataType,
    grouped: bool,
    method: BuiltInAggregation,
) -> ArrayRef {
    if !grouped
        || !matches!(
            method,
            BuiltInAggregation::Sum | BuiltInAggregation::Product
        )
    {
        return reduced;
    }
    if !matches!(
        source_type,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
    ) {
        return reduced;
    }
    let narrowed = cast(reduced.as_ref(), source_type)
        .expect("safe integer narrowing returns nulls instead of an Arrow error");
    if narrowed.null_count() == reduced.null_count() {
        narrowed
    } else {
        reduced
    }
}
