//! Lossless Arrow representation of built-in numeric Python object payloads.
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Float64Array, UnionArray, new_null_array};
use arrow_schema::{ArrowError, DataType, Field, UnionFields, UnionMode};

pub(super) fn fields() -> UnionFields {
    [
        (0, "python.bool", DataType::Boolean),
        (1, "python.int64", DataType::Int64),
        (2, "python.uint64", DataType::UInt64),
        (3, "python.float", DataType::Float64),
    ]
    .into_iter()
    .map(|(id, name, dtype)| (id, Arc::new(Field::new(name, dtype, true))))
    .collect()
}

/// Canonical sparse union for built-in bool, signed/unsigned integer and float objects.
/// Integer children are exact storage alternatives for Python integers, not float coercions.
/// Other objects, arbitrary-size integers and Python runtime identity need separate adapters.
#[must_use]
pub fn numeric_object_dtype() -> DataType {
    DataType::Union(fields(), UnionMode::Sparse)
}

/// Box native numeric values without erasing their scalar type or integer precision.
/// Arrow child nulls remain typed missing values; they are not a Python `None` adapter.
/// # Errors
/// Rejects unsupported input dtypes and propagates Arrow conversion errors.
pub fn numeric_object_array(array: &ArrayRef) -> Result<ArrayRef, ArrowError> {
    box_with(array, &mut ArrowObjects)
}

pub(super) trait ObjectOperations {
    fn cast_numeric(&mut self, array: &ArrayRef, dtype: &DataType) -> Result<ArrayRef, ArrowError>;
    fn build(
        &mut self,
        fields: UnionFields,
        ids: Vec<i8>,
        children: Vec<ArrayRef>,
    ) -> Result<ArrayRef, ArrowError>;
}

pub(super) struct ArrowObjects;
impl ObjectOperations for ArrowObjects {
    fn cast_numeric(&mut self, array: &ArrayRef, dtype: &DataType) -> Result<ArrayRef, ArrowError> {
        arrow_cast::cast(array, dtype)
    }
    fn build(
        &mut self,
        fields: UnionFields,
        ids: Vec<i8>,
        children: Vec<ArrayRef>,
    ) -> Result<ArrayRef, ArrowError> {
        UnionArray::try_new(fields, ids.into(), None, children)
            .map(|array| Arc::new(array) as ArrayRef)
    }
}

fn box_with(
    array: &ArrayRef,
    operations: &mut dyn ObjectOperations,
) -> Result<ArrayRef, ArrowError> {
    if is_object(array.data_type()) {
        return Ok(array.clone());
    }
    let id = match array.data_type() {
        DataType::Boolean => 0,
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => 1,
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => 2,
        DataType::Float16 | DataType::Float32 | DataType::Float64 => 3,
        dtype => {
            return Err(ArrowError::CastError(format!(
                "numeric object adapter does not support {dtype}"
            )));
        }
    };
    let fields = fields();
    let children = fields
        .iter()
        .map(|(child_id, field)| {
            if child_id == id {
                operations.cast_numeric(array, field.data_type())
            } else {
                Ok(new_null_array(field.data_type(), array.len()))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    operations.build(fields, vec![id; array.len()], children)
}

#[cfg(test)]
#[path = "object_tests.rs"]
mod tests;

pub(super) fn is_object(dtype: &DataType) -> bool {
    dtype == &numeric_object_dtype()
}

pub(super) fn missing(dtype: &DataType, rows: usize) -> ArrayRef {
    if dtype == &super::temporal_frame_dtype() {
        super::temporal_frame_array(&vec![
            super::TemporalFrameValue::Builtin(
                super::BuiltinFrameValue::Float(f64::NAN)
            );
            rows
        ])
        .expect("canonical temporal missing objects")
    } else if super::extended_plan::is_extended(dtype) {
        super::builtin_frame_array(&vec![super::BuiltinFrameValue::Float(f64::NAN); rows])
            .expect("canonical missing objects")
    } else if is_object(dtype) {
        let nan = Arc::new(Float64Array::from(vec![f64::NAN; rows])) as ArrayRef;
        numeric_object_array(&nan).expect("canonical floating object construction")
    } else {
        new_null_array(dtype, rows)
    }
}

pub(super) fn all_na(array: &ArrayRef) -> bool {
    let array = array
        .as_any()
        .downcast_ref::<UnionArray>()
        .expect("canonical object union");
    (0..array.len()).all(|i| {
        let child = array.child(array.type_id(i));
        let offset = array.value_offset(i);
        child.is_null(offset)
            || child
                .as_any()
                .downcast_ref::<Float64Array>()
                .is_some_and(|floats| floats.value(offset).is_nan())
    })
}

pub(super) fn cast(array: &ArrayRef, dtype: &DataType) -> Result<ArrayRef, ArrowError> {
    if dtype == &super::temporal_frame_dtype() {
        super::temporal_cast::temporal_object_array(array)
    } else if array.data_type() == &super::temporal_frame_dtype()
        || (super::blocks::logical_object(array.data_type())
            && super::temporal_cast::is_temporal(dtype))
    {
        super::temporal_cast::unbox_missing(array, dtype)
    } else if super::extended_plan::is_extended(dtype) {
        super::extended_plan::promote(array)
    } else if super::extended_plan::is_extended(array.data_type()) {
        super::extended_plan::unbox_missing(array, dtype)
    } else if is_object(dtype) {
        numeric_object_array(array)
    } else if is_object(array.data_type()) {
        if dtype.is_floating() && all_na(array) {
            // Pandas replaces a valid all-NA object join unit with floating NaNs.
            Ok(new_null_array(dtype, array.len()))
        } else {
            // Arrow union casts select a matching child, rather than convert all variants.
            Err(ArrowError::CastError(format!(
                "cannot unbox numeric objects as {dtype}"
            )))
        }
    } else if matches!(
        (array.data_type(), dtype),
        (DataType::Timestamp(_, _), DataType::Timestamp(_, _))
            | (DataType::Duration(_), DataType::Duration(_))
    ) {
        arrow_cast::cast_with_options(
            array,
            dtype,
            &arrow_cast::CastOptions {
                safe: false,
                ..Default::default()
            },
        )
    } else {
        arrow_cast::cast(array, dtype)
    }
}
