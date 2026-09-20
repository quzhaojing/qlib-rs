//! Ordinary object-index inference/boxing when recursive tuple cells participate.
use super::{
    DataframeAppendError, FrameOperations, TupleFrameValue, temporal_cast, tuple_frame_dtype,
    tuple_frame_values,
};
use arrow_array::ArrayRef;
use arrow_schema::{ArrowError, DataType};

pub(super) fn values(array: &ArrayRef) -> Result<Vec<TupleFrameValue>, ArrowError> {
    if array.data_type() == &tuple_frame_dtype() {
        tuple_frame_values(array)
    } else {
        Ok(temporal_cast::values(array)?
            .into_iter()
            .map(TupleFrameValue::Scalar)
            .collect())
    }
}

pub(super) fn infer(
    array: &ArrayRef,
    operations: &mut dyn FrameOperations,
) -> Result<ArrayRef, DataframeAppendError> {
    // Decode the complete array before selecting inference, so a preceding tuple
    // cannot mask an invalid later scalar or malformed row.
    let values = tuple_frame_values(array)?;
    let has_python = values
        .iter()
        .any(|v| matches!(v, TupleFrameValue::PythonTemporal(_)));
    let scalars: Option<Vec<_>> = values
        .into_iter()
        .map(|value| match value {
            TupleFrameValue::Scalar(value) => Some(value),
            TupleFrameValue::PythonTemporal(value) => value.inference_value(),
            TupleFrameValue::Tuple(_) => None,
        })
        .collect();
    if let Some(scalars) = scalars {
        let inferred = operations.infer_index(&scalars)?;
        if has_python && super::blocks::logical_object(inferred.data_type()) {
            Ok(array.clone())
        } else {
            Ok(inferred)
        }
    } else {
        Ok(array.clone())
    }
}

pub(super) fn join(
    left: &ArrayRef,
    right: &ArrayRef,
    dtype: &DataType,
    operations: &mut dyn FrameOperations,
) -> Result<ArrayRef, ArrowError> {
    if dtype == &tuple_frame_dtype() {
        let left = operations.tuple_array(&values(left)?)?;
        let right = operations.tuple_array(&values(right)?)?;
        super::join_arrays(&left, &right, dtype, operations)
    } else {
        super::join_arrays(left, right, dtype, operations)
    }
}
