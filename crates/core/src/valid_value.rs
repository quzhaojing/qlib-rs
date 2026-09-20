//! First/last valid value selection compatible with `qlib.utils.resam`.

use arrow_array::{Array, ArrayRef, Float16Array, Float32Array, Float64Array, RecordBatch};
use arrow_schema::DataType;
use thiserror::Error;

/// Which edge of a series supplies its valid value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidEdge {
    /// Equivalent to Pandas `bfill().iloc[0]`.
    First,
    /// Equivalent to Pandas `ffill().iloc[-1]`.
    Last,
}

/// Failures produced while selecting a value from an Arrow array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ValidValueError {
    /// Pandas raises an index error when a Series has no rows.
    #[error("cannot select a valid value from an empty array")]
    EmptyArray,
}

/// Select the first or last non-missing value as a one-element zero-copy Arrow slice.
///
/// Arrow nulls and IEEE floating-point NaNs are considered missing, matching
/// Pandas `ffill`/`bfill`. If every value is missing, the corresponding edge
/// element is returned so the original dtype and missing representation survive.
///
/// # Errors
///
/// Returns [`ValidValueError::EmptyArray`] when `array` has no rows.
pub fn valid_value(array: &dyn Array, edge: ValidEdge) -> Result<ArrayRef, ValidValueError> {
    if array.is_empty() {
        return Err(ValidValueError::EmptyArray);
    }
    let fallback = match edge {
        ValidEdge::First => 0,
        ValidEdge::Last => array.len() - 1,
    };
    let selected = match edge {
        ValidEdge::First => (0..array.len()).find(|index| !is_missing(array, *index)),
        ValidEdge::Last => (0..array.len())
            .rev()
            .find(|index| !is_missing(array, *index)),
    }
    .unwrap_or(fallback);
    Ok(array.slice(selected, 1))
}

/// Select the first non-missing value from one Arrow array.
///
/// # Errors
///
/// Returns [`ValidValueError::EmptyArray`] when `array` has no rows.
pub fn first_valid_value(array: &dyn Array) -> Result<ArrayRef, ValidValueError> {
    valid_value(array, ValidEdge::First)
}

/// Select the last non-missing value from one Arrow array.
///
/// # Errors
///
/// Returns [`ValidValueError::EmptyArray`] when `array` has no rows.
pub fn last_valid_value(array: &dyn Array) -> Result<ArrayRef, ValidValueError> {
    valid_value(array, ValidEdge::Last)
}

/// Select one valid value independently from every column of a record batch.
///
/// Non-empty tabular input becomes a one-row batch. A batch with no rows or no
/// columns is returned unchanged, matching Pandas `DataFrame.apply` behavior.
///
/// # Panics
///
/// Panics only if an Arrow array's one-element slice changes its declared data
/// type or length, which would violate the [`Array::slice`] contract and the
/// invariants of the already-valid input [`RecordBatch`].
#[must_use]
pub fn valid_values(batch: &RecordBatch, edge: ValidEdge) -> RecordBatch {
    if batch.num_rows() == 0 || batch.num_columns() == 0 {
        return batch.clone();
    }
    let columns = batch
        .columns()
        .iter()
        .map(|column| {
            valid_value(column.as_ref(), edge)
                .expect("non-empty RecordBatch columns have at least one value")
        })
        .collect();
    RecordBatch::try_new(batch.schema(), columns)
        .expect("one-value slices preserve the source RecordBatch schema")
}

/// Select the first non-missing value independently from every column.
#[must_use]
pub fn first_valid_values(batch: &RecordBatch) -> RecordBatch {
    valid_values(batch, ValidEdge::First)
}

/// Select the last non-missing value independently from every column.
#[must_use]
pub fn last_valid_values(batch: &RecordBatch) -> RecordBatch {
    valid_values(batch, ValidEdge::Last)
}

pub(crate) fn is_missing(array: &dyn Array, index: usize) -> bool {
    if array.is_null(index) {
        return true;
    }
    match array.data_type() {
        DataType::Float16 => array
            .as_any()
            .downcast_ref::<Float16Array>()
            .expect("Float16 data type uses Float16Array")
            .value(index)
            .is_nan(),
        DataType::Float32 => array
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("Float32 data type uses Float32Array")
            .value(index)
            .is_nan(),
        DataType::Float64 => array
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("Float64 data type uses Float64Array")
            .value(index)
            .is_nan(),
        _ => false,
    }
}
