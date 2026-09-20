//! Recursive tuple cells carried by a fixed Arrow schema, independently of index factoring.
use super::{
    BuiltinFrameValue, TemporalFrameValue, temporal_frame_array, temporal_frame_dtype,
    temporal_frame_values,
};
use arrow_array::builder::{LargeListBuilder, UInt8Builder};
use arrow_array::{Array, ArrayRef, LargeListArray, StructArray, UInt8Array};
use arrow_schema::{ArrowError, DataType, Field, Fields};
use std::sync::Arc;

/// Object values including recursively nested, possibly empty or ragged Python tuples.
/// Scalar payloads preserve their type; a one-element tuple is not its scalar element.
/// This is value storage, not Python object identity or `MultiIndex` level metadata.
#[derive(Clone, Debug, PartialEq)]
pub enum TupleFrameValue {
    Scalar(TemporalFrameValue),
    PythonTemporal(super::PythonTemporalValue),
    Tuple(Vec<Self>),
}

fn fields() -> Fields {
    vec![
        Field::new("token", DataType::UInt8, true),
        Field::new("scalar", temporal_frame_dtype(), true),
    ]
    .into()
}

fn item() -> Arc<Field> {
    Arc::new(Field::new(
        "python.tuple.tokens",
        DataType::Struct(fields()),
        true,
    ))
}

/// Fixed schema for mixed tuples/scalars at any represented nesting depth.
/// Tokens: 0 scalar, 1 tuple-open, 2 tuple-close, 3 builtin datetime, 4 builtin timedelta.
/// Datetime payloads are i64 microseconds; timedelta payloads are canonical i128
/// decimal text inside the existing scalar union. Older token encodings are unchanged.
/// Marker scalar payloads are inactive, just as inactive Arrow union children are.
#[must_use]
pub fn tuple_frame_dtype() -> DataType {
    DataType::LargeList(item())
}

fn invalid(message: &str) -> ArrowError {
    ArrowError::InvalidArgumentError(message.into())
}

/// Encode recursive cells using Arrow's list/struct/union storage and IPC support.
/// Iterative traversal does not use the Rust call stack for each tuple level.
/// # Errors
/// Propagates scalar encoding errors, including reserved temporal `NaT` ticks.
pub fn tuple_frame_array(values: &[TupleFrameValue]) -> Result<ArrayRef, ArrowError> {
    let mut tags = LargeListBuilder::new(UInt8Builder::new());
    let mut scalars = Vec::new();
    for value in values {
        let mut pending = vec![Some(value)];
        while let Some(next) = pending.pop() {
            let (tag, scalar) = match next {
                Some(TupleFrameValue::Scalar(value)) => (0, value.clone()),
                Some(TupleFrameValue::PythonTemporal(value)) => value.encode()?,
                Some(TupleFrameValue::Tuple(children)) => {
                    pending.push(None);
                    pending.extend(children.iter().rev().map(Some));
                    (1, TemporalFrameValue::Builtin(BuiltinFrameValue::None))
                }
                None => (2, TemporalFrameValue::Builtin(BuiltinFrameValue::None)),
            };
            tags.values().append_value(tag);
            scalars.push(scalar);
        }
        tags.append(true);
    }
    let tags = tags.finish();
    let tokens = StructArray::new(
        fields(),
        vec![tags.values().clone(), temporal_frame_array(&scalars)?],
        None,
    );
    Ok(Arc::new(LargeListArray::new(
        item(),
        tags.offsets().clone(),
        Arc::new(tokens),
        None,
    )))
}

fn decode_cell(tokens: &ArrayRef) -> Result<TupleFrameValue, ArrowError> {
    let tokens = tokens
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("validated tuple token schema");
    let tags = tokens
        .column(0)
        .as_any()
        .downcast_ref::<UInt8Array>()
        .expect("validated tuple tag schema");
    let mut stack: Vec<Vec<TupleFrameValue>> = vec![];
    let mut root = None;
    for position in 0..tokens.len() {
        if tokens.is_null(position) || tags.is_null(position) {
            return Err(invalid("tuple object has null token"));
        }
        let value = match tags.value(position) {
            0 => TupleFrameValue::Scalar(
                temporal_frame_values(&tokens.column(1).slice(position, 1))?.remove(0),
            ),
            1 => {
                stack.push(vec![]);
                continue;
            }
            2 => TupleFrameValue::Tuple(
                stack
                    .pop()
                    .ok_or_else(|| invalid("tuple object has unmatched close"))?,
            ),
            tag @ (3 | 4) => TupleFrameValue::PythonTemporal(super::PythonTemporalValue::decode(
                tag,
                temporal_frame_values(&tokens.column(1).slice(position, 1))?.remove(0),
            )?),
            _ => return Err(invalid("tuple object has unknown token")),
        };
        if let Some(parent) = stack.last_mut() {
            parent.push(value);
        } else if root.replace(value).is_some() {
            return Err(invalid("tuple object has multiple roots"));
        }
    }
    if !stack.is_empty() {
        return Err(invalid("tuple object has unclosed tuple"));
    }
    root.ok_or_else(|| invalid("tuple object has no root"))
}

/// Decode a canonical tuple/scalar array, including slices and Arrow round trips.
/// # Errors
/// Rejects foreign schemas, null rows/tokens, malformed tuple structure, and invalid
/// active scalar payloads. Python None/NA/NaT are explicit scalar values, not null rows.
/// # Panics
/// If concrete Arrow arrays violate their declared schema.
pub fn tuple_frame_values(array: &ArrayRef) -> Result<Vec<TupleFrameValue>, ArrowError> {
    if array.data_type() != &tuple_frame_dtype() {
        return Err(invalid("expected tuple object schema"));
    }
    let array = array
        .as_any()
        .downcast_ref::<LargeListArray>()
        .expect("validated tuple list schema");
    (0..array.len())
        .map(|position| {
            if array.is_null(position) {
                return Err(invalid("tuple object has null row"));
            }
            decode_cell(&array.value(position))
        })
        .collect()
}

#[cfg(test)]
#[path = "tuple_object_tests.rs"]
pub(super) mod tests;
