//! Ordered column-map construction at the Arrow boundary.
use super::{
    ArrowOperations, BuiltinFrameValue as V, FrameOperations, TemporalFrameValue as T,
    builtin_frame_array,
};
use arrow_array::{
    ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, UInt64Array, new_null_array,
};
use arrow_schema::{ArrowError, DataType, Field, TimeUnit};
use indexmap::IndexMap;
use std::sync::Arc;
use thiserror::Error;

/// Columns in a dictionary-like constructor. Arrays have already undergone dtype inference;
/// they are positional vectors, not indexed Series. Scalar types survive empty broadcasts.
#[derive(Clone, Debug)]
pub enum FrameColumnInput {
    Array(ArrayRef),
    /// A positional list whose supported built-in payloads need dtype inference.
    Untyped(Vec<V>),
    /// A positional list of supported built-in and temporal cells, requiring dtype inference.
    Temporal(Vec<T>),
    /// Infer the dtype of a supported Python built-in scalar.
    Scalar(V),
    /// A one-element, explicitly typed scalar, e.g. a timestamp or `NumPy` float32.
    TypedScalar(ArrayRef),
}

#[derive(Debug, Error)]
pub enum FrameConstructionError {
    #[error("If using all scalar values, you must pass an index")]
    ScalarOnly,
    #[error("All arrays must be of the same length")]
    UnequalLengths,
    #[error("typed scalar must contain exactly one element")]
    ScalarLength,
    #[error(transparent)]
    Arrow(#[from] ArrowError),
}

/// Construct an ordered column mapping, broadcasting scalars to the vector length.
/// Infers supported built-in/temporal list columns but does not align indexed Series.
/// Use the separate record constructors for record-list inputs.
/// Use the returned batch as `other` for [`super::dataframe_append`].
/// # Errors
/// Rejects unequal vector lengths, invalid typed scalars and nonempty all-scalar mappings.
/// Propagates Arrow scalar construction, take and batch errors without modifying inputs.
pub fn frame_from_columns(
    columns: &IndexMap<String, FrameColumnInput>,
) -> Result<RecordBatch, FrameConstructionError> {
    construct(columns, &mut ArrowConstructor)
}

/// Construct records containing the supported built-in values, retaining first-seen key order.
/// An omitted field contributes a floating NaN, not None. Empty records still contribute rows.
/// Typed temporal/`NumPy` cells and arbitrary objects require a richer record-cell adapter;
/// this entry point intentionally accepts only [`super::BuiltinFrameValue`].
/// # Errors
/// Propagates inference and Arrow batch errors without changing the input records.
pub fn frame_from_builtin_records(
    records: &[IndexMap<String, V>],
) -> Result<RecordBatch, FrameConstructionError> {
    construct_records(
        records,
        V::Float(f64::NAN),
        |ops, values| ops.infer(values),
        &mut ArrowConstructor,
    )
}

/// Construct records with built-in and temporal cells, retaining first-seen column order.
/// Missing fields contribute NaN. `NumPy` numeric scalars and arbitrary objects are not represented.
/// # Errors
/// Propagates temporal/built-in inference and Arrow batch errors without changing inputs.
pub fn frame_from_temporal_records(
    records: &[IndexMap<String, T>],
) -> Result<RecordBatch, FrameConstructionError> {
    construct_records(
        records,
        T::Builtin(V::Float(f64::NAN)),
        |ops, values| ops.infer_temporal(values),
        &mut ArrowConstructor,
    )
}

fn construct_records<C: Clone>(
    records: &[IndexMap<String, C>],
    missing: C,
    infer: fn(&mut dyn ConstructorOperations, &[C]) -> Result<ArrayRef, ArrowError>,
    operations: &mut dyn ConstructorOperations,
) -> Result<RecordBatch, FrameConstructionError> {
    let mut names = IndexMap::new();
    for record in records {
        for name in record.keys() {
            names.insert(name, ());
        }
    }
    let mut fields = Vec::with_capacity(names.len());
    let mut arrays = Vec::with_capacity(names.len());
    for name in names.keys() {
        let values = records
            .iter()
            .map(|record| {
                record
                    .get(*name)
                    .cloned()
                    .unwrap_or_else(|| missing.clone())
            })
            .collect::<Vec<_>>();
        let array = infer(operations, &values)?;
        fields.push(Field::new(*name, array.data_type().clone(), true));
        arrays.push(array);
    }
    Ok(operations.batch(fields, arrays, records.len())?)
}

fn rows(columns: &IndexMap<String, FrameColumnInput>) -> Result<usize, FrameConstructionError> {
    let mut rows = None;
    for column in columns.values() {
        let length = match column {
            FrameColumnInput::Array(array) => Some(array.len()),
            FrameColumnInput::Untyped(values) => Some(values.len()),
            FrameColumnInput::Temporal(values) => Some(values.len()),
            _ => None,
        };
        if let Some(length) = length {
            if rows.is_some_and(|rows| rows != length) {
                return Err(FrameConstructionError::UnequalLengths);
            }
            rows = Some(length);
        }
        match column {
            FrameColumnInput::TypedScalar(array) if array.len() != 1 => {
                return Err(FrameConstructionError::ScalarLength);
            }
            _ => {}
        }
    }
    if columns.is_empty() {
        Ok(0)
    } else {
        rows.ok_or(FrameConstructionError::ScalarOnly)
    }
}

trait ConstructorOperations {
    fn infer(&mut self, values: &[V]) -> Result<ArrayRef, ArrowError>;
    fn infer_temporal(&mut self, values: &[T]) -> Result<ArrayRef, ArrowError>;
    fn scalar(&mut self, value: &V) -> Result<ArrayRef, ArrowError>;
    fn repeat(&mut self, value: &ArrayRef, rows: usize) -> Result<ArrayRef, ArrowError>;
    fn batch(
        &mut self,
        fields: Vec<Field>,
        values: Vec<ArrayRef>,
        rows: usize,
    ) -> Result<RecordBatch, ArrowError>;
}
struct ArrowConstructor;
impl ConstructorOperations for ArrowConstructor {
    fn infer(&mut self, values: &[V]) -> Result<ArrayRef, ArrowError> {
        super::infer_frame_values(values)
    }
    fn infer_temporal(&mut self, values: &[T]) -> Result<ArrayRef, ArrowError> {
        super::infer_temporal_frame_values(values)
    }
    fn scalar(&mut self, value: &V) -> Result<ArrayRef, ArrowError> {
        Ok(match value {
            V::Bool(v) => Arc::new(BooleanArray::from(vec![*v])),
            V::Int(v) => Arc::new(Int64Array::from(vec![*v])),
            V::UInt(v) => match i64::try_from(*v) {
                Ok(v) => Arc::new(Int64Array::from(vec![v])),
                Err(_) => Arc::new(UInt64Array::from(vec![*v])),
            },
            V::Float(v) => Arc::new(Float64Array::from(vec![*v])),
            V::NotATime => new_null_array(&DataType::Timestamp(TimeUnit::Nanosecond, None), 1),
            V::None | V::PandasNa | V::Text(_) => {
                return builtin_frame_array(std::slice::from_ref(value));
            }
        })
    }
    fn repeat(&mut self, value: &ArrayRef, rows: usize) -> Result<ArrayRef, ArrowError> {
        arrow_select::take::take(value.as_ref(), &UInt64Array::from(vec![0_u64; rows]), None)
    }
    fn batch(
        &mut self,
        fields: Vec<Field>,
        values: Vec<ArrayRef>,
        rows: usize,
    ) -> Result<RecordBatch, ArrowError> {
        ArrowOperations.batch(fields, values, rows)
    }
}

fn construct(
    columns: &IndexMap<String, FrameColumnInput>,
    operations: &mut dyn ConstructorOperations,
) -> Result<RecordBatch, FrameConstructionError> {
    let rows = rows(columns)?;
    let mut fields = Vec::with_capacity(columns.len());
    let mut values = Vec::with_capacity(columns.len());
    for (name, column) in columns {
        let array = match column {
            FrameColumnInput::Array(array) => array.clone(),
            FrameColumnInput::Untyped(values) => operations.infer(values)?,
            FrameColumnInput::Temporal(values) => operations.infer_temporal(values)?,
            FrameColumnInput::Scalar(value) => {
                let value = operations.scalar(value)?;
                operations.repeat(&value, rows)?
            }
            FrameColumnInput::TypedScalar(array) => operations.repeat(array, rows)?,
        };
        fields.push(Field::new(name, array.data_type().clone(), true));
        values.push(array);
    }
    Ok(operations.batch(fields, values, rows)?)
}

#[cfg(test)]
#[path = "constructor_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "record_tests.rs"]
mod record_tests;
