//! Pandas record/list inference over the supported owned temporal object cells.
use super::{
    BuiltinFrameValue as B, TemporalFrameValue as V, infer_frame_values, temporal_frame_array,
};
use arrow_array::{ArrayRef, DurationNanosecondArray, TimestampNanosecondArray};
use arrow_schema::{ArrowError, TimeUnit};
use chrono::{DateTime, Datelike};
use std::sync::Arc;

#[derive(Clone, PartialEq)]
enum Kind {
    Timestamp(Option<Arc<str>>),
    Duration,
}

fn temporal(value: &V) -> Option<(Kind, i64, TimeUnit)> {
    match value {
        V::Timestamp {
            ticks,
            unit,
            timezone,
        } => Some((Kind::Timestamp(timezone.clone()), *ticks, *unit)),
        V::Duration { ticks, unit } => Some((Kind::Duration, *ticks, *unit)),
        V::Builtin(_) => None,
    }
}

fn nanoseconds(ticks: i64, unit: TimeUnit, kind: &Kind) -> Option<i64> {
    const CYCLE_SECONDS: i64 = 146_097 * 86_400;
    let scale = match unit {
        TimeUnit::Second => 1_000_000_000,
        TimeUnit::Millisecond => 1_000_000,
        TimeUnit::Microsecond => 1_000,
        TimeUnit::Nanosecond => 1,
    };
    if let Some(value) = ticks.checked_mul(scale) {
        return Some(value);
    }
    if !matches!(kind, Kind::Timestamp(Some(zone)) if zone.as_ref() == "UTC") {
        return None;
    }
    // Pandas' aware object constructor reads the datetime base fields of a
    // Timestamp. Outside years 1..9999 that base uses 1970 (non-leap) or 1972
    // (leap), while the exposed Timestamp.year still stores the true year.
    // Reduce by Gregorian 400-year cycles so chrono handles even i64 seconds.
    let nanos = i128::from(ticks) * i128::from(scale);
    let seconds = i64::try_from(nanos.div_euclid(1_000_000_000))
        .expect("supported units cannot exceed i64 seconds");
    let fraction =
        u32::try_from(nanos.rem_euclid(1_000_000_000)).expect("nanosecond remainder fits u32");
    let reduced = DateTime::from_timestamp(seconds.rem_euclid(CYCLE_SECONDS), fraction)
        .expect("reduced date is within 1970..2370");
    let year = i64::from(reduced.year()) + seconds.div_euclid(CYCLE_SECONDS) * 400;
    if (1..=9999).contains(&year) {
        return None;
    }
    let surrogate = if reduced.date_naive().leap_year() {
        1972
    } else {
        1970
    };
    reduced
        .with_year(surrogate)
        .expect("surrogate preserves leap-day validity")
        .timestamp_nanos_opt()
}

/// Infer list/record cells, promoting compatible temporal values to nanoseconds.
/// Incompatible mixtures or out-of-ns-range values retain their original object payloads,
/// except UTC Timestamp base-year reconstruction outside Python's year range.
/// Pure built-in columns use the existing built-in inference and object schema.
/// # Errors
/// Rejects reserved `NaT` ticks and propagates object storage/inference errors.
/// # Panics
/// Only if internal temporal-kind discovery contradicts the preceding built-in count.
pub fn infer_temporal_frame_values(values: &[V]) -> Result<ArrayRef, ArrowError> {
    let builtin = values
        .iter()
        .filter_map(|value| match value {
            V::Builtin(value) => Some(value.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    if builtin.len() == values.len() {
        return infer_frame_values(&builtin);
    }
    let (kind, _, _) = values
        .iter()
        .find_map(temporal)
        .expect("at least one temporal value");
    let mut ticks = Vec::with_capacity(values.len());
    for value in values {
        match value {
            V::Builtin(B::None | B::NotATime) => ticks.push(None),
            V::Builtin(B::Float(value)) if value.is_nan() => ticks.push(None),
            _ => {
                let Some((other_kind, value, unit)) = temporal(value) else {
                    return temporal_frame_array(values);
                };
                if value == i64::MIN {
                    return Err(ArrowError::InvalidArgumentError(
                        "temporal object uses reserved NaT ticks".into(),
                    ));
                }
                if other_kind != kind {
                    return temporal_frame_array(values);
                }
                let Some(value) = nanoseconds(value, unit, &kind) else {
                    return temporal_frame_array(values);
                };
                ticks.push(Some(value));
            }
        }
    }
    Ok(match kind {
        Kind::Timestamp(zone) => {
            Arc::new(TimestampNanosecondArray::from(ticks).with_timezone_opt(zone))
        }
        Kind::Duration => Arc::new(DurationNanosecondArray::from(ticks)),
    })
}

#[cfg(test)]
#[path = "temporal_inference_tests.rs"]
mod tests;
