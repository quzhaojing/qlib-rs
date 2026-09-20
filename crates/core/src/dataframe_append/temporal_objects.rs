//! Lossless temporal object cells; dtype inference and append promotion are separate layers.
use super::{BuiltinFrameValue, builtin_frame_array, builtin_frame_dtype, builtin_frame_values};
use arrow_array::{Array, ArrayRef, Int64Array, StringArray, StructArray, UInt8Array, UnionArray};
use arrow_schema::{ArrowError, DataType, Field, Fields, TimeUnit, UnionFields, UnionMode};
use std::sync::Arc;

/// Owned object cells retaining each timestamp/duration's original resolution.
/// A timezone is a retained label, not a request to reinterpret or convert the ticks.
/// `i64::MIN` is reserved for `NaT`; use `Builtin(NotATime)` instead.
/// This does not yet represent arbitrary Python object identity or `NumPy` numeric scalars.
#[derive(Clone, Debug, PartialEq)]
pub enum TemporalFrameValue {
    Builtin(BuiltinFrameValue),
    Timestamp {
        ticks: i64,
        unit: TimeUnit,
        timezone: Option<Arc<str>>,
    },
    Duration {
        ticks: i64,
        unit: TimeUnit,
    },
}

fn temporal_fields(timestamp: bool) -> Fields {
    let mut fields = vec![
        Field::new("ticks", DataType::Int64, true),
        Field::new("unit", DataType::UInt8, true),
    ];
    if timestamp {
        fields.push(Field::new("timezone", DataType::Utf8, true));
    }
    fields.into()
}

fn fields() -> UnionFields {
    [
        (0, "builtin", builtin_frame_dtype()),
        (
            1,
            "pandas.Timestamp",
            DataType::Struct(temporal_fields(true)),
        ),
        (
            2,
            "pandas.Timedelta",
            DataType::Struct(temporal_fields(false)),
        ),
    ]
    .into_iter()
    .map(|(id, name, dtype)| (id, Arc::new(Field::new(name, dtype, true))))
    .collect()
}

/// A separate schema so existing numeric and built-in object schemas remain unchanged.
#[must_use]
pub fn temporal_frame_dtype() -> DataType {
    DataType::Union(fields(), UnionMode::Sparse)
}

fn unit_code(unit: TimeUnit) -> u8 {
    match unit {
        TimeUnit::Second => 0,
        TimeUnit::Millisecond => 1,
        TimeUnit::Microsecond => 2,
        TimeUnit::Nanosecond => 3,
    }
}

fn invalid(message: &str) -> ArrowError {
    ArrowError::InvalidArgumentError(message.into())
}

fn temporal_array(values: &[TemporalFrameValue], timestamp: bool) -> Result<ArrayRef, ArrowError> {
    let mut ticks = Vec::with_capacity(values.len());
    let mut units = Vec::with_capacity(values.len());
    let mut zones = Vec::with_capacity(values.len());
    for value in values {
        let cell = match (value, timestamp) {
            (
                TemporalFrameValue::Timestamp {
                    ticks,
                    unit,
                    timezone,
                },
                true,
            ) => Some((*ticks, *unit, timezone.as_deref())),
            (TemporalFrameValue::Duration { ticks, unit }, false) => Some((*ticks, *unit, None)),
            _ => None,
        };
        if let Some((value, unit, zone)) = cell {
            if value == i64::MIN {
                return Err(invalid("temporal object uses reserved NaT ticks"));
            }
            ticks.push(Some(value));
            units.push(Some(unit_code(unit)));
            zones.push(zone);
        } else {
            ticks.push(None);
            units.push(None);
            zones.push(None);
        }
    }
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(ticks)),
        Arc::new(UInt8Array::from(units)),
    ];
    if timestamp {
        columns.push(Arc::new(StringArray::from(zones)));
    }
    Ok(Arc::new(StructArray::new(
        temporal_fields(timestamp),
        columns,
        None,
    )))
}

/// Encode object cells without converting units, interpreting zones, or losing sentinel identity.
/// # Errors
/// Rejects reserved `NaT` ticks and propagates Arrow object-construction errors.
pub fn temporal_frame_array(values: &[TemporalFrameValue]) -> Result<ArrayRef, ArrowError> {
    encode_with(values, &mut builtin_frame_array)
}

fn encode_with(
    values: &[TemporalFrameValue],
    encode_builtin: &mut dyn FnMut(&[BuiltinFrameValue]) -> Result<ArrayRef, ArrowError>,
) -> Result<ArrayRef, ArrowError> {
    let builtin = values
        .iter()
        .map(|value| match value {
            TemporalFrameValue::Builtin(value) => value.clone(),
            _ => BuiltinFrameValue::None,
        })
        .collect::<Vec<_>>();
    let ids = values
        .iter()
        .map(|value| match value {
            TemporalFrameValue::Builtin(_) => 0,
            TemporalFrameValue::Timestamp { .. } => 1,
            TemporalFrameValue::Duration { .. } => 2,
        })
        .collect::<Vec<i8>>();
    let children = vec![
        encode_builtin(&builtin)?,
        temporal_array(values, true)?,
        temporal_array(values, false)?,
    ];
    UnionArray::try_new(fields(), ids.into(), None, children)
        .map(|array| Arc::new(array) as ArrayRef)
}

fn temporal_value(
    array: &ArrayRef,
    offset: usize,
    timestamp: bool,
) -> Result<TemporalFrameValue, ArrowError> {
    let array = array
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("validated temporal schema");
    let ticks = array
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("ticks schema");
    let units = array
        .column(1)
        .as_any()
        .downcast_ref::<UInt8Array>()
        .expect("unit schema");
    if array.is_null(offset) || ticks.is_null(offset) || units.is_null(offset) {
        return Err(invalid("temporal object has null active fields"));
    }
    let ticks = ticks.value(offset);
    if ticks == i64::MIN {
        return Err(invalid("temporal object uses reserved NaT ticks"));
    }
    let unit = match units.value(offset) {
        0 => TimeUnit::Second,
        1 => TimeUnit::Millisecond,
        2 => TimeUnit::Microsecond,
        3 => TimeUnit::Nanosecond,
        _ => return Err(invalid("temporal object has unknown unit code")),
    };
    Ok(if timestamp {
        let zones = array
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("timezone schema");
        TemporalFrameValue::Timestamp {
            ticks,
            unit,
            timezone: (!zones.is_null(offset)).then(|| zones.value(offset).into()),
        }
    } else {
        TemporalFrameValue::Duration { ticks, unit }
    })
}

/// Decode canonical temporal object storage, including slices and Arrow IPC round trips.
/// # Errors
/// Rejects foreign schemas, invalid active temporal fields and malformed built-in payloads.
/// # Panics
/// If concrete Arrow arrays violate their declared schema.
pub fn temporal_frame_values(array: &ArrayRef) -> Result<Vec<TemporalFrameValue>, ArrowError> {
    if array.data_type() != &temporal_frame_dtype() {
        return Err(invalid("expected temporal object schema"));
    }
    let array = array
        .as_any()
        .downcast_ref::<UnionArray>()
        .expect("validated temporal union");
    (0..array.len())
        .map(|i| {
            let id = array.type_id(i);
            let child = array.child(id);
            let offset = array.value_offset(i);
            if id == 0 {
                Ok(TemporalFrameValue::Builtin(
                    builtin_frame_values(&child.slice(offset, 1))?.remove(0),
                ))
            } else {
                temporal_value(child, offset, id == 1)
            }
        })
        .collect()
}

#[cfg(test)]
#[path = "temporal_object_tests.rs"]
pub(super) mod tests;
