//! Object-safe callable aggregation boundary for `resam_ts_data`.

use std::{borrow::Cow, collections::BTreeMap, error::Error, sync::Arc};

use arrow_arith::numeric::add_wrapping;
use arrow_array::{ArrayRef, Int64Array, RecordBatch, Scalar, UInt32Array};
use arrow_cast::cast;
use arrow_schema::{DataType, FieldRef, Schema, SchemaRef};
use arrow_select::{concat::concat_batches, take::take};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use strum::Display;
use thiserror::Error;

use crate::{
    TimeRange, TimeSeriesIndex, TimeSeriesSelectionError, ValidEdge, select_time_series,
    time_series_aggregation::{clean_missing, prepare_time_series_groups, reduce_product},
    valid_value,
};

/// JSON-compatible equivalent of Python's `method_kwargs` dictionary.
pub type AggregationArguments = BTreeMap<String, Value>;

/// Error returned by a callable aggregator implementation.
pub type AggregatorError = Box<dyn Error + Send + Sync + 'static>;

/// Object-safe, batch-oriented callable aggregation extension point.
///
/// Implementations declare one stable output schema for the selected input and
/// then return that schema for every non-null instrument group. The input batch
/// retains explicit index columns so an implementation can inspect or return
/// datetime values. For multi-index input the framework prepends the instrument
/// column, so plugin output must not declare that reserved name.
pub trait TimeSeriesAggregator: Send + Sync {
    /// Stable diagnostic/plugin identifier.
    fn name(&self) -> Cow<'_, str>;

    /// Declare the result schema before any group is executed.
    ///
    /// # Errors
    ///
    /// Returns an implementation-defined error when arguments or input schema
    /// cannot produce a stable output schema.
    fn output_schema(
        &self,
        input: &RecordBatch,
        index: &TimeSeriesIndex,
        arguments: &AggregationArguments,
    ) -> Result<SchemaRef, AggregatorError>;

    /// Aggregate one chronologically sorted, non-empty series or instrument group.
    ///
    /// # Errors
    ///
    /// Returns an implementation-defined error when group execution fails.
    fn aggregate_group(
        &self,
        group: &RecordBatch,
        index: &TimeSeriesIndex,
        arguments: &AggregationArguments,
    ) -> Result<RecordBatch, AggregatorError>;
}

/// Stage at which a callable aggregator failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Display)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum AggregatorPhase {
    /// Output-schema declaration.
    Schema,
    /// Per-group execution.
    Group,
}

/// Failures from callable time-series aggregation.
#[derive(Debug, Error)]
pub enum TimeSeriesCallableError {
    /// Sorting or slicing rejected the input.
    #[error(transparent)]
    Selection(#[from] TimeSeriesSelectionError),
    /// The aggregator returned its own typed failure.
    #[error("aggregator {aggregator} failed during {phase}: {source}")]
    Aggregator {
        /// Aggregator identifier.
        aggregator: String,
        /// Failed lifecycle stage.
        phase: AggregatorPhase,
        /// Original implementation error.
        #[source]
        source: AggregatorError,
    },
    /// A grouped plugin attempted to provide the framework-owned key column.
    #[error("aggregator {aggregator} output uses reserved instrument column {column}")]
    ReservedInstrumentColumn {
        /// Aggregator identifier.
        aggregator: String,
        /// Conflicting column name.
        column: String,
    },
    /// A group result did not match the schema declared before execution.
    #[error("aggregator {aggregator} returned a different schema for group {group}")]
    OutputSchemaMismatch {
        /// Aggregator identifier.
        aggregator: String,
        /// Zero-based execution group ordinal.
        group: usize,
        /// Declared schema.
        expected: SchemaRef,
        /// Returned schema.
        actual: SchemaRef,
    },
}

/// Select a time range and invoke an object-safe callable aggregator.
///
/// Single-index input invokes the aggregator once. Multi-index input is sorted
/// by instrument/datetime, null instrument keys are dropped, and the aggregator
/// is invoked once per non-null instrument in sorted key order. Multi-row and
/// empty plugin outputs are supported.
///
/// # Errors
///
/// Returns typed selection, lifecycle, reserved-column, or schema errors.
///
/// # Panics
///
/// Panics only if an Arrow 59 kernel violates same-batch index, row-count,
/// concatenation, or schema invariants established by this function.
pub fn aggregate_time_series_with(
    batch: &RecordBatch,
    index: &TimeSeriesIndex,
    range: TimeRange,
    aggregator: &dyn TimeSeriesAggregator,
    arguments: &AggregationArguments,
) -> Result<Option<RecordBatch>, TimeSeriesCallableError> {
    let selected = match select_time_series(batch, index, range) {
        Ok(selected) => selected,
        Err(error) => return Err(TimeSeriesCallableError::Selection(error)),
    };
    let Some(selected) = selected else {
        return Ok(None);
    };
    aggregate_selected_with(&selected, index, aggregator, arguments)
}

pub(crate) fn aggregate_selected_with(
    selected: &RecordBatch,
    index: &TimeSeriesIndex,
    aggregator: &dyn TimeSeriesAggregator,
    arguments: &AggregationArguments,
) -> Result<Option<RecordBatch>, TimeSeriesCallableError> {
    let prepared = prepare_time_series_groups(selected, index);
    let declared = call_output_schema(aggregator, &prepared.batch, index, arguments)?;
    let Some(instrument_index) = prepared.instrument_index else {
        let output = call_group(aggregator, &prepared.batch, index, arguments)?;
        validate_output_schema(aggregator, 0, &declared, &output)?;
        return Ok(Some(output));
    };

    let instrument_name = prepared.batch.schema_ref().field(instrument_index).name();
    if declared
        .fields()
        .iter()
        .any(|field| field.name() == instrument_name)
    {
        return Err(TimeSeriesCallableError::ReservedInstrumentColumn {
            aggregator: aggregator.name().into_owned(),
            column: instrument_name.clone(),
        });
    }
    let output_schema = grouped_output_schema(&prepared.batch, instrument_index, &declared);
    if prepared.ranges.is_empty() {
        return Ok(Some(RecordBatch::new_empty(output_schema)));
    }

    let mut outputs = Vec::with_capacity(prepared.ranges.len());
    for (group, range) in prepared.ranges.iter().enumerate() {
        let input = prepared.batch.slice(range.start, range.len());
        let output = call_group(aggregator, &input, index, arguments)?;
        validate_output_schema(aggregator, group, &declared, &output)?;
        outputs.push(prepend_instrument(
            &prepared.batch,
            instrument_index,
            range.start,
            &output,
            output_schema.clone(),
        ));
    }
    let output = concat_batches(&output_schema, &outputs)
        .expect("validated plugin batches share the declared grouped schema");
    Ok(Some(output))
}

fn call_output_schema(
    aggregator: &dyn TimeSeriesAggregator,
    input: &RecordBatch,
    index: &TimeSeriesIndex,
    arguments: &AggregationArguments,
) -> Result<SchemaRef, TimeSeriesCallableError> {
    match aggregator.output_schema(input, index, arguments) {
        Ok(schema) => Ok(schema),
        Err(source) => Err(TimeSeriesCallableError::Aggregator {
            aggregator: aggregator.name().into_owned(),
            phase: AggregatorPhase::Schema,
            source,
        }),
    }
}

fn call_group(
    aggregator: &dyn TimeSeriesAggregator,
    group: &RecordBatch,
    index: &TimeSeriesIndex,
    arguments: &AggregationArguments,
) -> Result<RecordBatch, TimeSeriesCallableError> {
    match aggregator.aggregate_group(group, index, arguments) {
        Ok(output) => Ok(output),
        Err(source) => Err(TimeSeriesCallableError::Aggregator {
            aggregator: aggregator.name().into_owned(),
            phase: AggregatorPhase::Group,
            source,
        }),
    }
}

fn validate_output_schema(
    aggregator: &dyn TimeSeriesAggregator,
    group: usize,
    expected: &SchemaRef,
    output: &RecordBatch,
) -> Result<(), TimeSeriesCallableError> {
    if output.schema_ref() == expected {
        Ok(())
    } else {
        Err(TimeSeriesCallableError::OutputSchemaMismatch {
            aggregator: aggregator.name().into_owned(),
            group,
            expected: expected.clone(),
            actual: output.schema(),
        })
    }
}

fn grouped_output_schema(
    input: &RecordBatch,
    instrument_index: usize,
    plugin_schema: &SchemaRef,
) -> SchemaRef {
    let mut fields = Vec::with_capacity(plugin_schema.fields().len() + 1);
    fields.push(Arc::new(input.schema_ref().field(instrument_index).clone()));
    fields.extend(plugin_schema.fields().iter().cloned());
    Arc::new(Schema::new_with_metadata(
        fields,
        plugin_schema.metadata().clone(),
    ))
}

fn prepend_instrument(
    input: &RecordBatch,
    instrument_index: usize,
    group_start: usize,
    output: &RecordBatch,
    schema: SchemaRef,
) -> RecordBatch {
    let row = u32::try_from(group_start).expect("Arrow row counts fit u32");
    let indices = UInt32Array::from(vec![row; output.num_rows()]);
    let instrument = take(input.column(instrument_index).as_ref(), &indices, None)
        .expect("the repeated group index belongs to this instrument array");
    let mut columns = Vec::with_capacity(output.num_columns() + 1);
    columns.push(instrument);
    columns.extend(output.columns().iter().cloned());
    RecordBatch::try_new(schema, columns)
        .expect("the prepended instrument has exactly the plugin output row count")
}

fn value_indices(schema: &Schema, index: &TimeSeriesIndex) -> Vec<usize> {
    let (instrument, datetime) = match index {
        TimeSeriesIndex::Datetime { datetime } => (None, datetime.as_str()),
        TimeSeriesIndex::InstrumentDatetime {
            instrument,
            datetime,
            ..
        } => (Some(instrument.as_str()), datetime.as_str()),
    };
    let instrument = instrument.map(|name| {
        schema
            .index_of(name)
            .expect("selection validated the instrument field")
    });
    let datetime = schema
        .index_of(datetime)
        .expect("selection validated the datetime field");
    (0..schema.fields().len())
        .filter(|column| Some(*column) != instrument && *column != datetime)
        .collect()
}

#[derive(Debug, Error)]
enum AdapterError {
    #[error("unsupported argument {argument}")]
    UnsupportedArgument { argument: String },
    #[error("argument last must be a JSON boolean")]
    InvalidLastArgument,
    #[error("unsupported Arrow value type {data_type}")]
    UnsupportedDataType { data_type: DataType },
}

fn last_argument(arguments: &AggregationArguments) -> Result<bool, AggregatorError> {
    for name in arguments.keys() {
        if name != "last" {
            return Err(Box::new(AdapterError::UnsupportedArgument {
                argument: name.clone(),
            }));
        }
    }
    match arguments.get("last") {
        None => Ok(true),
        Some(Value::Bool(last)) => Ok(*last),
        Some(_) => Err(Box::new(AdapterError::InvalidLastArgument)),
    }
}

fn no_arguments(arguments: &AggregationArguments) -> Result<(), AggregatorError> {
    if let Some(name) = arguments.keys().next() {
        Err(Box::new(AdapterError::UnsupportedArgument {
            argument: name.clone(),
        }))
    } else {
        Ok(())
    }
}

fn value_schema(
    input: &RecordBatch,
    index: &TimeSeriesIndex,
    data_type: impl Fn(&DataType) -> Result<DataType, AggregatorError>,
) -> Result<SchemaRef, AggregatorError> {
    let mut fields: Vec<FieldRef> = Vec::new();
    for column in value_indices(input.schema_ref(), index) {
        let field = input.schema_ref().field(column);
        fields.push(Arc::new(
            field
                .clone()
                .with_data_type(data_type(field.data_type())?)
                .with_nullable(true),
        ));
    }
    Ok(Arc::new(Schema::new_with_metadata(
        fields,
        input.schema_ref().metadata().clone(),
    )))
}

fn identity_value_schema(input: &RecordBatch, index: &TimeSeriesIndex) -> SchemaRef {
    let fields: Vec<FieldRef> = value_indices(input.schema_ref(), index)
        .into_iter()
        .map(|column| Arc::new(input.schema_ref().field(column).clone().with_nullable(true)))
        .collect();
    Arc::new(Schema::new_with_metadata(
        fields,
        input.schema_ref().metadata().clone(),
    ))
}

/// Adapter for Qlib's `ts_data_last`/`_ts_data_valid` callable.
#[derive(Debug, Default, Clone, Copy)]
pub struct LastValidAggregator;

impl TimeSeriesAggregator for LastValidAggregator {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed("ts_data_last")
    }

    fn output_schema(
        &self,
        input: &RecordBatch,
        index: &TimeSeriesIndex,
        arguments: &AggregationArguments,
    ) -> Result<SchemaRef, AggregatorError> {
        let _ = last_argument(arguments)?;
        Ok(identity_value_schema(input, index))
    }

    fn aggregate_group(
        &self,
        group: &RecordBatch,
        index: &TimeSeriesIndex,
        arguments: &AggregationArguments,
    ) -> Result<RecordBatch, AggregatorError> {
        let edge = if last_argument(arguments)? {
            ValidEdge::Last
        } else {
            ValidEdge::First
        };
        let schema = identity_value_schema(group, index);
        let columns = value_indices(group.schema_ref(), index)
            .into_iter()
            .map(|column| {
                valid_value(group.column(column).as_ref(), edge)
                    .expect("callable groups are non-empty")
            })
            .collect();
        Ok(RecordBatch::try_new(schema, columns)
            .expect("one-value slices match the declared last-valid schema"))
    }
}

/// Adapter for Qlib benchmark's `(x + 1).prod()` callable.
#[derive(Debug, Default, Clone, Copy)]
pub struct CompoundedReturnAggregator;

impl TimeSeriesAggregator for CompoundedReturnAggregator {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed("compounded_return")
    }

    fn output_schema(
        &self,
        input: &RecordBatch,
        index: &TimeSeriesIndex,
        arguments: &AggregationArguments,
    ) -> Result<SchemaRef, AggregatorError> {
        no_arguments(arguments)?;
        value_schema(input, index, compounded_output_type)
    }

    fn aggregate_group(
        &self,
        group: &RecordBatch,
        index: &TimeSeriesIndex,
        arguments: &AggregationArguments,
    ) -> Result<RecordBatch, AggregatorError> {
        no_arguments(arguments)?;
        let schema = value_schema(group, index, compounded_output_type)?;
        let mut columns = Vec::with_capacity(schema.fields().len());
        for column in value_indices(group.schema_ref(), index) {
            let output_type = schema.field(columns.len()).data_type();
            columns.push(compounded_product(group.column(column), output_type));
        }
        Ok(RecordBatch::try_new(schema, columns)
            .expect("compounded products match their declared schema"))
    }
}

fn compounded_output_type(data_type: &DataType) -> Result<DataType, AggregatorError> {
    match data_type {
        DataType::Boolean
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64 => Ok(DataType::Int64),
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
            Ok(DataType::UInt64)
        }
        DataType::Float16 | DataType::Float32 | DataType::Float64 => Ok(data_type.clone()),
        _ => Err(Box::new(AdapterError::UnsupportedDataType {
            data_type: data_type.clone(),
        })),
    }
}

fn compounded_product(array: &ArrayRef, output_type: &DataType) -> ArrayRef {
    let normalized = cast(array.as_ref(), output_type)
        .expect("the declared compounded-return type is a lossless numeric widening");
    let one = cast(&Int64Array::from(vec![1]), normalized.data_type())
        .expect("the integer one is representable in every supported numeric type");
    let adjusted = add_wrapping(&normalized.as_ref(), &Scalar::new(one))
        .expect("same-type numeric addition is supported by Arrow");
    let cleaned = clean_missing(adjusted.as_ref(), 0..adjusted.len());
    reduce_product(cleaned.as_ref())
        .expect("the compounded-return adapter validated its numeric type")
}
