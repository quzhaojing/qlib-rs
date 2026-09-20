//! Extended object payload storage; source-aware append planning is a separate layer.
use std::sync::Arc;

use arrow_array::builder::{ListBuilder, UInt32Builder};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int64Array, ListArray, NullArray, UInt32Array,
    UInt64Array, UnionArray,
};
use arrow_schema::{ArrowError, DataType, Field, UnionFields, UnionMode};

use super::objects::{ArrowObjects, ObjectOperations};
use crate::RlCheckpointText;

/// Owned built-in payloads, with distinct missing sentinels and lossless Python text.
/// This does not represent arbitrary-size integers, complex values or runtime object identity.
#[derive(Clone, Debug, PartialEq)]
pub enum BuiltinFrameValue {
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    None,
    PandasNa,
    NotATime,
    Text(RlCheckpointText),
}

pub(super) fn fields() -> UnionFields {
    super::objects::fields()
        .iter()
        .map(|(id, field)| (id, field.clone()))
        .chain([
            (4, Arc::new(Field::new("python.none", DataType::Null, true))),
            (5, Arc::new(Field::new("pandas.NA", DataType::Null, true))),
            (6, Arc::new(Field::new("pandas.NaT", DataType::Null, true))),
            (
                7,
                Arc::new(Field::new(
                    "python.codepoints",
                    DataType::List(Arc::new(Field::new("item", DataType::UInt32, true))),
                    true,
                )),
            ),
        ])
        .collect()
}

/// Extended object schema. The existing four-child numeric schema remains unchanged.
#[must_use]
pub fn builtin_frame_dtype() -> DataType {
    DataType::Union(fields(), UnionMode::Sparse)
}

/// Encode payloads without coercing sentinel identities, integer precision or string code points.
/// Storage alone does not apply Pandas block-dependent missing-value replacement.
/// # Errors
/// Propagates Arrow union validation/construction errors.
pub fn builtin_frame_array(values: &[BuiltinFrameValue]) -> Result<ArrayRef, ArrowError> {
    let mut ids = Vec::with_capacity(values.len());
    let mut boolean = Vec::with_capacity(values.len());
    let mut signed = Vec::with_capacity(values.len());
    let mut unsigned = Vec::with_capacity(values.len());
    let mut floating = Vec::with_capacity(values.len());
    let mut text = ListBuilder::new(UInt32Builder::new());
    for value in values {
        boolean.push(if let BuiltinFrameValue::Bool(v) = value {
            Some(*v)
        } else {
            None
        });
        signed.push(if let BuiltinFrameValue::Int(v) = value {
            Some(*v)
        } else {
            None
        });
        unsigned.push(if let BuiltinFrameValue::UInt(v) = value {
            Some(*v)
        } else {
            None
        });
        floating.push(if let BuiltinFrameValue::Float(v) = value {
            Some(*v)
        } else {
            None
        });
        if let BuiltinFrameValue::Text(v) = value {
            text.values().append_slice(v.as_code_points());
            text.append(true);
        } else {
            text.append(false);
        }
        ids.push(match value {
            BuiltinFrameValue::Bool(_) => 0,
            BuiltinFrameValue::Int(_) => 1,
            BuiltinFrameValue::UInt(_) => 2,
            BuiltinFrameValue::Float(_) => 3,
            BuiltinFrameValue::None => 4,
            BuiltinFrameValue::PandasNa => 5,
            BuiltinFrameValue::NotATime => 6,
            BuiltinFrameValue::Text(_) => 7,
        });
    }
    ArrowObjects.build(
        fields(),
        ids,
        vec![
            Arc::new(BooleanArray::from(boolean)),
            Arc::new(Int64Array::from(signed)),
            Arc::new(UInt64Array::from(unsigned)),
            Arc::new(Float64Array::from(floating)),
            Arc::new(NullArray::new(values.len())),
            Arc::new(NullArray::new(values.len())),
            Arc::new(NullArray::new(values.len())),
            Arc::new(text.finish()),
        ],
    )
}

fn text_value(array: &ArrayRef, offset: usize) -> Result<RlCheckpointText, ArrowError> {
    let value = array
        .as_any()
        .downcast_ref::<ListArray>()
        .expect("validated object schema")
        .value(offset);
    let points = value
        .as_any()
        .downcast_ref::<UInt32Array>()
        .expect("validated codepoint schema");
    if points.null_count() != 0 {
        return Err(ArrowError::InvalidArgumentError(
            "object string contains a null code point".into(),
        ));
    }
    RlCheckpointText::try_from_code_points(points.values().iter().copied())
        .map_err(ArrowError::InvalidArgumentError)
}

/// Decode the extended schema, including sliced arrays and values read from Arrow IPC.
/// # Errors
/// Rejects foreign schemas, typed-null active numeric/text children, missing code points,
/// and code points outside Python's range. Sentinel children are deliberately null-typed.
/// # Panics
/// If concrete Arrow arrays or their children violate their declared schema.
pub fn builtin_frame_values(array: &ArrayRef) -> Result<Vec<BuiltinFrameValue>, ArrowError> {
    if array.data_type() != &builtin_frame_dtype() {
        return Err(ArrowError::InvalidArgumentError(
            "expected extended built-in object schema".into(),
        ));
    }
    let array = array
        .as_any()
        .downcast_ref::<UnionArray>()
        .expect("validated object schema");
    (0..array.len())
        .map(|i| {
            let id = array.type_id(i);
            let child = array.child(id);
            let offset = array.value_offset(i);
            if !matches!(id, 4..=6) && child.is_null(offset) {
                return Err(ArrowError::InvalidArgumentError(
                    "typed null is not an explicit object sentinel".into(),
                ));
            }
            Ok(match id {
                0 => BuiltinFrameValue::Bool(
                    child
                        .as_any()
                        .downcast_ref::<BooleanArray>()
                        .expect("bool child")
                        .value(offset),
                ),
                1 => BuiltinFrameValue::Int(
                    child
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .expect("int child")
                        .value(offset),
                ),
                2 => BuiltinFrameValue::UInt(
                    child
                        .as_any()
                        .downcast_ref::<UInt64Array>()
                        .expect("uint child")
                        .value(offset),
                ),
                3 => BuiltinFrameValue::Float(
                    child
                        .as_any()
                        .downcast_ref::<Float64Array>()
                        .expect("float child")
                        .value(offset),
                ),
                4 => BuiltinFrameValue::None,
                5 => BuiltinFrameValue::PandasNa,
                6 => BuiltinFrameValue::NotATime,
                _ => BuiltinFrameValue::Text(text_value(child, offset)?),
            })
        })
        .collect()
}
