//! Index construction/concatenation differs from data-block missing coercion.
use super::{DataframeAppendError, FrameOperations, blocks, temporal_cast, temporal_frame_dtype};
use arrow_array::{ArrayRef, new_null_array};
use arrow_schema::DataType;

const INFERENCE_WARNING: &str = "Dtype inference on a pandas object (Series, Index, ExtensionArray) is deprecated. The Index constructor will keep the original dtype in the future. Call `infer_objects` on the result to get the old behavior.";

fn object(dtype: &DataType) -> bool {
    blocks::logical_object(dtype) || dtype == &super::tuple_frame_dtype()
}

fn logical_type_eq(left: &DataType, right: &DataType) -> bool {
    left == right || (object(left) && object(right))
}

pub(super) fn validate(dtype: &DataType) -> Result<(), DataframeAppendError> {
    if dtype == &DataType::Float16 {
        Err(DataframeAppendError::Float16Index)
    } else {
        Ok(())
    }
}

pub(super) fn infer(
    array: &ArrayRef,
    operations: &mut dyn FrameOperations,
) -> Result<ArrayRef, DataframeAppendError> {
    if array.is_empty() || !object(array.data_type()) {
        return Ok(array.clone());
    }
    if array.data_type() == &super::tuple_frame_dtype() {
        return super::tuple_index::infer(array, operations);
    }
    Ok(operations.infer_index(&temporal_cast::values(array)?)?)
}

pub(super) fn from_column(
    array: &ArrayRef,
    operations: &mut dyn FrameOperations,
    warning: &mut dyn FnMut(&str),
) -> Result<ArrayRef, DataframeAppendError> {
    validate(array.data_type())?;
    if object(array.data_type()) {
        let inferred = infer(array, operations)?;
        if temporal_cast::is_temporal(inferred.data_type()) {
            warning(INFERENCE_WARNING);
            return Ok(inferred);
        }
    }
    Ok(array.clone())
}

fn common(left: &DataType, right: &DataType) -> Result<DataType, DataframeAppendError> {
    if left == &super::tuple_frame_dtype() || right == &super::tuple_frame_dtype() {
        return Ok(super::tuple_frame_dtype());
    }
    if blocks::logical_object(left) || blocks::logical_object(right) {
        return Ok(temporal_frame_dtype());
    }
    if left == right {
        return Ok(left.clone());
    }
    if temporal_cast::is_temporal(left) || temporal_cast::is_temporal(right) {
        return Ok(match (left, right) {
            (DataType::Timestamp(_, _), DataType::Timestamp(_, _)) => {
                super::joined_type(left, right).unwrap_or_else(|_| temporal_frame_dtype())
            }
            (DataType::Duration(lu), DataType::Duration(ru)) => DataType::Duration((*lu).max(*ru)),
            _ => temporal_frame_dtype(),
        });
    }
    if let Some(dtype) = super::numeric::common(left, right) {
        return Ok(dtype);
    }
    if left == &DataType::Boolean && right.is_floating() {
        return Ok(right.clone());
    }
    if right == &DataType::Boolean && left.is_floating() {
        return Ok(left.clone());
    }
    super::joined_type(left, right)
}

pub(super) fn join(
    left: &ArrayRef,
    right: &ArrayRef,
    operations: &mut dyn FrameOperations,
    warning: &mut dyn FnMut(&str),
) -> Result<ArrayRef, DataframeAppendError> {
    let ld = left.data_type();
    let rd = right.data_type();
    let mut dtype = common(ld, rd)?;
    if left.is_empty()
        && right.is_empty()
        && ((ld == &DataType::Boolean && (rd.is_floating() || rd.is_integer()))
            || (rd == &DataType::Boolean && (ld.is_floating() || ld.is_integer())))
    {
        dtype = temporal_frame_dtype();
    } else if left.is_empty() != right.is_empty() && !logical_type_eq(ld, rd) {
        let selected = if left.is_empty() { rd } else { ld };
        if !logical_type_eq(&dtype, selected) || temporal_cast::is_temporal(selected) {
            warning(super::EMPTY_INDEX_WARNING);
        }
        dtype = selected.clone();
    }
    // Empty arrays contribute no payload; avoid interpreting their source dtype
    // when casting them to the dtype selected by the surviving index.
    let empty = new_null_array(&dtype, 0);
    let left = if left.is_empty() { &empty } else { left };
    let right = if right.is_empty() { &empty } else { right };
    let joined = super::tuple_index::join(left, right, &dtype, operations)?;
    if !left.is_empty()
        && !right.is_empty()
        && ((ld == &DataType::Boolean && (rd.is_integer() || rd.is_floating()))
            || (rd == &DataType::Boolean && (ld.is_integer() || ld.is_floating())))
    {
        // concat_compat boxes mixed bool/numeric results before Index._with_infer.
        infer(&super::temporal_object_array(&joined)?, operations)
    } else {
        infer(&joined, operations)
    }
}
