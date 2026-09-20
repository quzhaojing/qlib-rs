//! Boxing native Arrow temporal/numeric arrays into lossless object cells.
use super::{
    BuiltinFrameValue as B, TemporalFrameValue as V, builtin_frame_values, temporal_frame_array,
    temporal_frame_dtype, temporal_frame_values,
};
use arrow_array::{ArrayRef, Int64Array, new_null_array};
use arrow_schema::{ArrowError, DataType};

pub(super) fn is_temporal(dtype: &DataType) -> bool {
    matches!(dtype, DataType::Timestamp(_, _) | DataType::Duration(_))
}

/// Box temporal, built-in object and native numeric arrays without changing temporal units.
/// Numeric and missing payloads use the existing lossless object codecs.
/// # Errors
/// Rejects unsupported types, malformed object payloads and failed Arrow conversions.
/// # Panics
/// If Arrow's integer cast returns an array contradicting its requested dtype.
pub fn temporal_object_array(array: &ArrayRef) -> Result<ArrayRef, ArrowError> {
    promote_with(array, &mut |array| {
        arrow_cast::cast(array, &DataType::Int64)
    })
}

fn promote_with(
    array: &ArrayRef,
    cast_ticks: &mut dyn FnMut(&ArrayRef) -> Result<ArrayRef, ArrowError>,
) -> Result<ArrayRef, ArrowError> {
    if array.data_type() == &temporal_frame_dtype() {
        return Ok(array.clone());
    }
    let temporal = match array.data_type() {
        DataType::Timestamp(unit, zone) => Some((*unit, zone.clone(), true)),
        DataType::Duration(unit) => Some((*unit, None, false)),
        _ => None,
    };
    let values: Vec<V> = if let Some((unit, timezone, timestamp)) = temporal {
        let ticks = cast_ticks(array)?;
        let ticks = ticks
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("requested i64 cast");
        ticks
            .iter()
            .map(|tick| {
                if let Some(tick) = tick.filter(|tick| *tick != i64::MIN) {
                    if timestamp {
                        V::Timestamp {
                            ticks: tick,
                            unit,
                            timezone: timezone.clone(),
                        }
                    } else {
                        V::Duration { ticks: tick, unit }
                    }
                } else {
                    V::Builtin(B::NotATime)
                }
            })
            .collect()
    } else {
        let array = super::extended_plan::promote(array)?;
        builtin_frame_values(&array)?
            .into_iter()
            .map(V::Builtin)
            .collect()
    };
    temporal_frame_array(&values)
}

pub(super) fn missing(value: &V) -> bool {
    match value {
        V::Builtin(B::None | B::PandasNa | B::NotATime) => true,
        V::Builtin(B::Float(value)) => value.is_nan(),
        _ => false,
    }
}

pub(super) fn values(array: &ArrayRef) -> Result<Vec<V>, ArrowError> {
    temporal_frame_values(&temporal_object_array(array)?)
}

pub(super) fn unbox_missing(array: &ArrayRef, dtype: &DataType) -> Result<ArrayRef, ArrowError> {
    let values = values(array)?;
    let valid = if is_temporal(dtype) {
        values.iter().all(missing)
    } else {
        dtype.is_floating()
            && values
                .iter()
                .all(|v| missing(v) && !matches!(v, V::Builtin(B::NotATime)))
    };
    if valid {
        Ok(new_null_array(dtype, array.len()))
    } else {
        Err(ArrowError::CastError(format!(
            "cannot unbox temporal objects as {dtype}"
        )))
    }
}

#[cfg(test)]
#[path = "temporal_cast_tests.rs"]
mod tests;
