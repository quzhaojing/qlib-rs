//! Remaining typed compatibility boundaries for `qlib.utils.time`.

use std::str::FromStr;

use arrow_array::timezone::Tz;
use arrow_schema::TimeUnit;
use chrono::{DateTime, Datelike, NaiveDate, NaiveTime, Timelike, Utc};
use num_bigint::Sign;
use thiserror::Error;

use crate::{
    EpsilonDirection, EpsilonError, Frequency, FrequencyError, MinuteAlignmentError, Region,
    align_sampled_minute,
    dataframe_append::{BuiltinFrameValue, TemporalFrameValue},
    epsilon_change,
};

/// A source `Freq.get_recent_freq` input retaining whether it was text or a `Freq` object.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompatibleFrequency {
    /// Original string spelling, including case, aliases, and leading zeroes.
    Text(String),
    /// Parsed `Freq` identity.
    Frequency(Frequency),
}

impl CompatibleFrequency {
    fn parsed(&self) -> Result<Frequency, FrequencyError> {
        match self {
            Self::Text(value) => value.parse(),
            Self::Frequency(value) => Ok(value.clone()),
        }
    }

    fn source_string(&self) -> String {
        match self {
            Self::Text(value) => value.clone(),
            Self::Frequency(value) => value.to_string(),
        }
    }
}

impl From<&str> for CompatibleFrequency {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

impl From<String> for CompatibleFrequency {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<Frequency> for CompatibleFrequency {
    fn from(value: Frequency) -> Self {
        Self::Frequency(value)
    }
}

/// Select the source-compatible recent frequency without erasing its dynamic result kind.
///
/// The first eligible candidate is always returned as text via source `str(candidate)` semantics.
/// A later strictly closer candidate replaces it with that candidate's original string/`Freq`
/// kind. Equal deltas retain the first winner.
///
/// # Errors
///
/// Returns the existing frequency parse error for malformed text in the base or any candidate,
/// including candidates that are too coarse (the source parses before checking their delta).
pub fn recent_frequency_compatible(
    base: &CompatibleFrequency,
    candidates: &[CompatibleFrequency],
) -> Result<Option<CompatibleFrequency>, FrequencyError> {
    let base = base.parsed()?;
    let mut best = None;
    for candidate in candidates {
        let parsed = candidate.parsed()?;
        let delta = base.minute_delta(&parsed);
        if delta.sign() == Sign::Minus {
            continue;
        }
        match &best {
            None => best = Some((delta, CompatibleFrequency::Text(candidate.source_string()))),
            Some((best_delta, _)) if delta < *best_delta => {
                best = Some((delta, candidate.clone()));
            }
            Some(_) => {}
        }
    }
    Ok(best.map(|(_, candidate)| candidate))
}

/// Failures at the typed date/time compatibility boundary.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TimeCompatError {
    /// Rust supplied a date outside Python's `datetime.date` year range.
    #[error("date year is outside Python's 1..=9999 range: {year}")]
    DateYearOutOfRange {
        /// Rejected proleptic Gregorian year.
        year: i32,
    },
    /// Rust supplied sub-microsecond precision unavailable on Python `datetime.time`.
    #[error("time has sub-microsecond precision")]
    SubmicrosecondTime,
    /// A timestamp cannot be represented in the required Chrono or Pandas resolution.
    #[error("timestamp is out of range for {operation}")]
    TimestampOutOfRange {
        /// Compatibility operation being performed.
        operation: &'static str,
    },
    /// The temporal union variant is not accepted by the source operation.
    #[error("{operation} requires a pandas Timestamp or NaT")]
    UnsupportedTemporalValue {
        /// Compatibility operation being performed.
        operation: &'static str,
    },
    /// `NaT.time()` raises before sampled-minute alignment.
    #[error("NaTType does not support time")]
    NaTDoesNotSupportTime,
    /// Arrow could not resolve the retained timezone label.
    #[error("invalid timestamp timezone {timezone:?}: {message}")]
    InvalidTimezone {
        /// Original retained Arrow timezone label.
        timezone: String,
        /// Arrow/chrono-tz parse diagnostic.
        message: String,
    },
    /// Existing one-second epsilon validation or range failure.
    #[error(transparent)]
    Epsilon(#[from] EpsilonError),
    /// Existing sampled-minute alignment failure.
    #[error(transparent)]
    Alignment(#[from] MinuteAlignmentError),
}

/// Combine a Python-range date and microsecond-resolution time into a naive Timestamp value.
///
/// The result uses Arrow's temporal object representation at microsecond resolution, matching
/// `pd.Timestamp(datetime(...)).unit == "us"` and retaining the full Python year range.
///
/// # Errors
///
/// Rejects years outside `1..=9999`, sub-microsecond Rust times, and values outside the retained
/// microsecond timestamp representation.
pub fn concat_date_time_compatible(
    date: NaiveDate,
    time: NaiveTime,
) -> Result<TemporalFrameValue, TimeCompatError> {
    if !(1..=9_999).contains(&date.year()) {
        return Err(TimeCompatError::DateYearOutOfRange { year: date.year() });
    }
    if time.nanosecond() >= 1_000_000_000 || time.nanosecond() % 1_000 != 0 {
        return Err(TimeCompatError::SubmicrosecondTime);
    }
    let ticks = date.and_time(time).and_utc().timestamp_micros();
    Ok(TemporalFrameValue::Timestamp {
        ticks,
        unit: TimeUnit::Microsecond,
        timezone: None,
    })
}

/// Apply the source one-second epsilon shift to a typed Timestamp or `NaT`.
///
/// Timestamp inputs of any Arrow time unit are promoted to nanoseconds because adding Pandas'
/// nanosecond-resolution `Timedelta(seconds=1)` promotes the source result. Retained timezone
/// labels are preserved without reinterpreting the instant. `NaT` propagates unchanged.
///
/// # Errors
///
/// Rejects invalid direction strings, non-Timestamp/non-`NaT` variants, values that cannot be
/// promoted to Pandas nanoseconds, and shifts outside the Pandas nanosecond range.
pub fn epsilon_change_compatible(
    value: &TemporalFrameValue,
    direction: &str,
) -> Result<TemporalFrameValue, TimeCompatError> {
    let direction =
        EpsilonDirection::from_str(direction).map_err(|_| EpsilonError::InvalidDirection {
            direction: direction.to_owned(),
        })?;
    match value {
        TemporalFrameValue::Builtin(BuiltinFrameValue::NotATime) => Ok(value.clone()),
        TemporalFrameValue::Timestamp {
            ticks,
            unit,
            timezone,
        } => {
            let nanos =
                ticks_to_nanos(*ticks, *unit).ok_or(TimeCompatError::TimestampOutOfRange {
                    operation: "epsilon_change nanosecond promotion",
                })?;
            let timestamp = DateTime::<Utc>::from_timestamp_nanos(nanos).naive_utc();
            epsilon_change(timestamp, direction)?;
            // The reused implementation has now validated this exact one-second operation against
            // the non-NaT i64 nanosecond domain, so the corresponding arithmetic cannot overflow.
            let ticks = match direction {
                EpsilonDirection::Backward => nanos - 1_000_000_000,
                EpsilonDirection::Forward => nanos + 1_000_000_000,
            };
            Ok(TemporalFrameValue::Timestamp {
                ticks,
                unit: TimeUnit::Nanosecond,
                timezone: timezone.clone(),
            })
        }
        _ => Err(TimeCompatError::UnsupportedTemporalValue {
            operation: "epsilon_change",
        }),
    }
}

/// Align a typed Timestamp to a sampled local market minute and strip timezone metadata.
///
/// A retained Arrow timezone label is resolved through Arrow's timezone adapter (including IANA
/// DST rules), the UTC ticks are projected to local wall time, and the existing native alignment
/// implementation is reused. The result is timezone-naive microseconds because the source calls
/// `concat_date_time` after selecting the local date and time.
///
/// # Errors
///
/// Rejects `NaT`, non-Timestamp variants, invalid timezone labels, timestamps outside Chrono's
/// range, and errors from the existing sampled-minute alignment implementation.
pub fn align_sampled_minute_compatible(
    value: &TemporalFrameValue,
    sample_minutes: &num_bigint::BigInt,
    minute_shift: &num_bigint::BigInt,
    region: Region,
) -> Result<TemporalFrameValue, TimeCompatError> {
    let TemporalFrameValue::Timestamp {
        ticks,
        unit,
        timezone,
    } = value
    else {
        return if matches!(
            value,
            TemporalFrameValue::Builtin(BuiltinFrameValue::NotATime)
        ) {
            Err(TimeCompatError::NaTDoesNotSupportTime)
        } else {
            Err(TimeCompatError::UnsupportedTemporalValue {
                operation: "cal_sam_minute",
            })
        };
    };
    let utc = ticks_to_utc(*ticks, *unit).ok_or(TimeCompatError::TimestampOutOfRange {
        operation: "cal_sam_minute input",
    })?;
    let local = match timezone {
        None => utc.naive_utc(),
        Some(label) => {
            let zone = label
                .parse::<Tz>()
                .map_err(|error| TimeCompatError::InvalidTimezone {
                    timezone: label.to_string(),
                    message: error.to_string(),
                })?;
            utc.with_timezone(&zone).naive_local()
        }
    };
    let aligned = align_sampled_minute(local, sample_minutes, minute_shift, region)?;
    concat_date_time_compatible(aligned.date(), aligned.time())
}

fn ticks_to_nanos(ticks: i64, unit: TimeUnit) -> Option<i64> {
    match unit {
        TimeUnit::Second => ticks.checked_mul(1_000_000_000),
        TimeUnit::Millisecond => ticks.checked_mul(1_000_000),
        TimeUnit::Microsecond => ticks.checked_mul(1_000),
        TimeUnit::Nanosecond => Some(ticks),
    }
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "remainders are nonnegative and strictly below the at-most-1e9 positive divisor"
)]
fn ticks_to_utc(ticks: i64, unit: TimeUnit) -> Option<DateTime<Utc>> {
    let divisor = match unit {
        TimeUnit::Second => 1,
        TimeUnit::Millisecond => 1_000,
        TimeUnit::Microsecond => 1_000_000,
        TimeUnit::Nanosecond => 1_000_000_000,
    };
    let subsecond_nanos = match unit {
        TimeUnit::Second => 0,
        TimeUnit::Millisecond => (ticks.rem_euclid(divisor) as u32) * 1_000_000,
        TimeUnit::Microsecond => (ticks.rem_euclid(divisor) as u32) * 1_000,
        TimeUnit::Nanosecond => ticks.rem_euclid(divisor) as u32,
    };
    DateTime::from_timestamp(ticks.div_euclid(divisor), subsecond_nanos)
}
