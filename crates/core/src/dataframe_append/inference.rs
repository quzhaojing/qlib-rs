//! Pandas-compatible inference for the supported built-in list payloads.
use super::{BuiltinFrameValue as V, builtin_frame_array};
use arrow_array::{ArrayRef, BooleanArray, Float64Array, Int64Array, UInt64Array, new_null_array};
use arrow_schema::{ArrowError, DataType, TimeUnit};
use num_traits::ToPrimitive;
use std::sync::Arc;

/// Infer a list column without conflating None, pd.NA and `NaT` or rounding object integers.
/// Empty lists infer float64; unlike explicit object storage this may convert missing values.
/// # Errors
/// Propagates Arrow object construction errors.
/// # Panics
/// Internal integer narrowing panics only if the inferred integer range invariant is violated.
pub fn infer_frame_values(values: &[V]) -> Result<ArrayRef, ArrowError> {
    if values.is_empty() {
        return Ok(Arc::new(Float64Array::from(Vec::<f64>::new())));
    }
    let mut booleans = Vec::new();
    let mut integers = Vec::new();
    let mut floats = Vec::new();
    let mut none = 0;
    let mut before_null = None;
    let mut nat = 0;
    for value in values {
        match value {
            V::Bool(v) => booleans.push(*v),
            V::Int(v) => {
                integers.push(i128::from(*v));
                floats.push(v.to_f64().expect("i64 fits finite f64"));
            }
            V::UInt(v) => {
                integers.push(i128::from(*v));
                floats.push(v.to_f64().expect("u64 fits finite f64"));
            }
            V::Float(v) => floats.push(*v),
            V::None => {
                before_null.get_or_insert(integers.len());
                none += 1;
                floats.push(f64::NAN);
            }
            V::NotATime => nat += 1,
            V::PandasNa | V::Text(_) => {}
        }
    }
    if booleans.len() == values.len() {
        return Ok(Arc::new(BooleanArray::from(booleans)));
    }
    if nat > 0 && nat + floats.len() == values.len() && floats.iter().all(|v| v.is_nan()) {
        return Ok(new_null_array(
            &DataType::Timestamp(TimeUnit::Nanosecond, None),
            values.len(),
        ));
    }
    if floats.len() == values.len() && none != values.len() {
        let prefix = &integers[..before_null.unwrap_or(integers.len())];
        let wide = prefix.iter().any(|v| *v > i128::from(i64::MAX));
        let negative = prefix.iter().any(|v| *v < 0);
        // Pandas checks signed/wide-unsigned conflicts only until the first None.
        if !(wide && negative) {
            if integers.len() != values.len() {
                return Ok(Arc::new(Float64Array::from(floats)));
            }
            return Ok(if wide {
                Arc::new(UInt64Array::from_iter_values(
                    integers
                        .into_iter()
                        .map(|v| u64::try_from(v).expect("inferred u64 range")),
                ))
            } else {
                Arc::new(Int64Array::from_iter_values(
                    integers
                        .into_iter()
                        .map(|v| i64::try_from(v).expect("inferred i64 range")),
                ))
            });
        }
    }
    builtin_frame_array(values)
}

#[cfg(test)]
#[path = "inference_tests.rs"]
pub(super) mod tests;
