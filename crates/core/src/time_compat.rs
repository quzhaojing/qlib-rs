//! Remaining typed compatibility boundaries for `qlib.utils.time`.

use std::{borrow::Cow, str::FromStr};

use arrow_array::timezone::Tz;
use arrow_schema::TimeUnit;
use chrono::{
    DateTime, Datelike, FixedOffset, NaiveDate, NaiveDateTime, NaiveTime, Offset, Timelike, Utc,
};
use num_bigint::Sign;
use num_traits::ToPrimitive;
use thiserror::Error;

use crate::{
    EpsilonDirection, EpsilonError, Frequency, FrequencyError, MinuteAlignmentError, Region,
    dataframe_append::{BuiltinFrameValue, TemporalFrameValue},
    epsilon_change, is_single_market_value,
    minute_alignment::{
        align_to_sampled_calendar, python_bisect_left, python_bisect_right, sample_calendar,
    },
    time_calendar_cache::{
        TimeCalendarCache, TimeCalendarCacheError, TimeCalendarCall, TimeCalendarKeyword,
        default_time_calendar_cache,
    },
};

/// `Freq.get_timedelta` outcome in the shared temporal compatibility representation.
#[derive(Clone, Debug, PartialEq)]
pub struct CompatibleTimeDelta {
    /// Nanosecond duration, `NaT`, or a conversion error.
    pub result: Result<TemporalFrameValue, FrequencyError>,
    /// Ordered `FutureWarning` messages, including those emitted before errors.
    pub future_warnings: Vec<String>,
}

/// Convert Qlib's concatenated count/suffix while preserving warnings and missingness.
///
/// Unlike the single-unit native helper, this entry point accepts compound text.
/// Callers decide how to display or filter the retained warnings.
///
/// # Panics
///
/// Panics only if the underlying parser violates its internal invariant that every
/// non-missing result fits an `i64` number of nanoseconds.
#[must_use]
pub fn time_delta_compatible(count: &num_bigint::BigInt, suffix: &str) -> CompatibleTimeDelta {
    let conversion = Frequency::compound_time_delta(count, suffix);
    CompatibleTimeDelta {
        result: conversion.result.map(|duration| {
            duration.map_or(
                TemporalFrameValue::Builtin(BuiltinFrameValue::NotATime),
                |duration| TemporalFrameValue::Duration {
                    ticks: duration
                        .num_nanoseconds()
                        .expect("parser returns an i64 nanosecond duration"),
                    unit: TimeUnit::Nanosecond,
                },
            )
        }),
        future_warnings: conversion.future_warnings,
    }
}

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
    /// Source general timestamp parser failure, distinct from bounds overflow.
    #[error("{message}")]
    TimestampTextDateParse { message: String },
    /// Source numeric text fails the date-likeness precheck.
    #[error("{message}")]
    TimestampTextNotDateLike { message: String },
    /// Fast date construction raises a plain `ValueError` without appended input.
    #[error("{message}")]
    TimestampTextInvalidCalendar { message: String },
    /// General timestamp parsing produced an invalid UTC offset.
    #[error("{message}")]
    TimestampTextInvalidOffset { message: String },
    /// General timestamp timezone conversion exceeds the integer tick domain.
    #[error("int too big to convert")]
    TimestampTextOffsetOverflow,
    /// A recognized text constructor exceeds its inferred resolution bounds.
    #[error(transparent)]
    TimestampTextBounds(#[from] crate::timestamp_text::TimestampTextBoundsError),
    /// Source frequency construction rejects malformed text before calendar lookup.
    #[error(
        "freq format is not supported, the freq should be like (n)month/mon, (n)week/w, (n)day/d, (n)minute/min"
    )]
    RangeFrequencyFormat,
    /// Nanosecond Timestamp comparison takes Pandas' awareness-error path.
    #[error("Cannot compare tz-naive and tz-aware timestamps")]
    AwareIntradayTimestamp,
    /// Microsecond-aligned Timestamp comparisons first construct Python datetime.
    #[error("year must be in 1..9999, not {year}")]
    IntradayTimestampYear {
        /// Original timestamp year.
        year: i64,
    },
    /// The pinned Windows Python datetime conversion uses a 32-bit C long.
    #[error("Python int too large to convert to C long")]
    IntradayTimestampYearOverflow,
    /// No prefix matches the complete Python hour-colon-minute pattern.
    #[error("time data {input_repr} does not match format '%H:%M'")]
    MarketClockFormat {
        /// Python repr of the original input, including quote selection.
        input_repr: String,
    },
    /// A complete clock prefix matched, but input remains unconsumed.
    #[error("unconverted data remains: {remainder}")]
    MarketClockRemainder {
        /// Unescaped trailing source text.
        remainder: String,
    },
    /// Intraday dispatch rejects unknown regions before datetime comparisons.
    #[error("{region} is not supported")]
    UnsupportedIntradayRegion {
        /// Original region spelling.
        region: String,
    },
    /// Source session endpoints are naive Python datetimes.
    #[error("can't compare offset-naive and offset-aware datetimes")]
    AwareIntradayDatetime,
    /// The complete datetime lies outside the source's dated sessions.
    #[error("{datetime} is not the opening time of the {region} stock market")]
    OutsideIntradayDatetime {
        /// Python datetime display, including microseconds when nonzero.
        datetime: String,
        /// Selected source region.
        region: String,
    },
    /// Chrono permits leap seconds, but Python datetime does not.
    #[error("second must be in 0..59")]
    DatetimeLeapSecond,
    /// Source region dispatch precedes temporal evaluation.
    #[error("please implement the is_single_value func for {region}")]
    UnsupportedSingleValueRegion {
        /// Unrecognized source region spelling.
        region: String,
    },
    /// Aware and naive timestamps cannot be subtracted.
    #[error("Cannot subtract tz-naive and tz-aware datetime-like objects.")]
    MixedTimestampAwareness,
    /// Timestamp subtraction overflows its common resolution.
    #[error(
        "Result is too large for pandas.Timedelta. Convert inputs to datetime.datetime with 'Timestamp.to_pydatetime()' before subtracting."
    )]
    TimestampSubtractionOverflow,
    /// Pandas asserts when a finite timestamp difference equals its `NaT` sentinel.
    #[error("")]
    TimestampSubtractionSentinel,
    /// An operand cannot be promoted to the subtraction's common resolution.
    #[error("Cannot cast {timestamp} to unit='{unit}' without overflow.")]
    TimestampPromotionOverflow {
        /// Source timestamp display.
        timestamp: String,
        /// Required Pandas resolution.
        unit: &'static str,
    },
    /// The typed single-value boundary requires a duration or missing frequency.
    #[error("is_single_value requires a Timedelta or NaT frequency")]
    UnsupportedSingleValueFrequency,
    /// The mutable source calendar cache could not execute the call.
    #[error(transparent)]
    CalendarCache(#[from] TimeCalendarCacheError),
    /// A previous Rust panic left a caller-owned calendar lock poisoned.
    #[error("mutable minute calendar lock is poisoned")]
    CalendarLockPoisoned,
    /// The source indexes an empty sampled calendar only after reading the timestamp.
    #[error("list index out of range")]
    EmptyCalendar,
    /// Pandas rejects extracting a Python date outside the standard-library year range.
    #[error(
        "date not yet supported on Timestamps which are outside the range of Python's standard library. "
    )]
    TimestampDateNotSupported,
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
    /// `NaT.time()` raises after the calendar and sampling slice have been evaluated.
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

/// Evaluate a Pandas Timestamp or `NaT` input to `time_to_day_index`.
///
/// Preserves region-first dispatch, nanosecond-aware comparison diagnostics and
/// missingness. Uses the shared temporal value representation without Python execution.
///
/// # Errors
///
/// Returns source comparison, conversion and outside-session errors, or rejects
/// variants other than Timestamp/`NaT` at this typed boundary.
pub fn timestamp_to_day_index_compatible(
    value: &TemporalFrameValue,
    region: &str,
) -> Result<i64, TimeCompatError> {
    let market = intraday_region(region)?;
    let display = match value {
        TemporalFrameValue::Builtin(BuiltinFrameValue::NotATime) => "NaT".to_owned(),
        TemporalFrameValue::Timestamp {
            ticks,
            unit,
            timezone,
        } => {
            let CompatibleLocalTimestamp { local, year, .. } =
                compatible_local_timestamp(*ticks, *unit, timezone.as_deref())?;
            let nanos = local.nanosecond();
            if nanos % 1_000 == 0 {
                if i32::try_from(year).is_err() {
                    return Err(TimeCompatError::IntradayTimestampYearOverflow);
                }
                if !(1..=9999).contains(&year) {
                    return Err(TimeCompatError::IntradayTimestampYear { year });
                }
            }
            if timezone.is_some() {
                return Err(if nanos % 1_000 == 0 {
                    TimeCompatError::AwareIntradayDatetime
                } else {
                    TimeCompatError::AwareIntradayTimestamp
                });
            }
            if (year, local.month(), local.day()) == (1900, 1, 1) {
                if let Ok(index) = crate::time_to_day_index(local.time(), market) {
                    return Ok(index);
                }
            }
            let format = if nanos == 0 {
                "-%m-%d %H:%M:%S"
            } else if nanos % 1_000 == 0 {
                "-%m-%d %H:%M:%S%.6f"
            } else {
                "-%m-%d %H:%M:%S%.9f"
            };
            format!("{year:04}{}", local.format(format))
        }
        _ => {
            return Err(TimeCompatError::UnsupportedTemporalValue {
                operation: "time_to_day_index",
            });
        }
    };
    Err(TimeCompatError::OutsideIntradayDatetime {
        datetime: display,
        region: region.to_owned(),
    })
}

fn intraday_region(region: &str) -> Result<Region, TimeCompatError> {
    match region {
        "cn" => Ok(Region::Cn),
        "us" => Ok(Region::Us),
        "tw" => Ok(Region::Tw),
        _ => Err(TimeCompatError::UnsupportedIntradayRegion {
            region: region.to_owned(),
        }),
    }
}

/// Evaluate the string input to Python `time_to_day_index`.
///
/// Parsing and its exact diagnostics precede region dispatch. A valid clock is
/// anchored to 1900-01-01 and enters the same dated-session path as datetime inputs.
///
/// # Errors
///
/// Returns Python-compatible parse, region, or outside-session errors.
///
/// # Panics
///
/// Panics only if Chrono rejects the constant valid date 1900-01-01.
pub fn time_to_day_index_compatible(input: &str, region: &str) -> Result<i64, TimeCompatError> {
    let (clock, remainder) = crate::intraday_index::parse_python_market_clock_prefix(input)
        .ok_or_else(|| TimeCompatError::MarketClockFormat {
            input_repr: crate::rl_checkpoint_name::python_string_repr(input),
        })?;
    if !remainder.is_empty() {
        return Err(TimeCompatError::MarketClockRemainder {
            remainder: remainder.to_owned(),
        });
    }
    let local = NaiveDate::from_ymd_opt(1900, 1, 1)
        .expect("1900-01-01 is a valid date")
        .and_time(clock);
    datetime_to_day_index_compatible(local, None, region)
}

/// Evaluate a Python `datetime` input to `time_to_day_index`.
///
/// `local` retains the complete wall-clock date and time. `offset` represents
/// whether `utcoffset()` is non-None; aware values are rejected, not normalized.
/// Source sessions are anchored to 1900-01-01, so another date is not interchangeable.
///
/// # Errors
///
/// Rejects values Python datetime cannot represent, unsupported regions, aware
/// datetimes, and values outside the selected dated sessions, in that order.
pub fn datetime_to_day_index_compatible(
    local: NaiveDateTime,
    offset: Option<FixedOffset>,
    region: &str,
) -> Result<i64, TimeCompatError> {
    if !(1..=9999).contains(&local.year()) {
        return Err(TimeCompatError::DateYearOutOfRange { year: local.year() });
    }
    if local.nanosecond() >= 1_000_000_000 {
        return Err(TimeCompatError::DatetimeLeapSecond);
    }
    if local.nanosecond() % 1_000 != 0 {
        return Err(TimeCompatError::SubmicrosecondTime);
    }
    let market = intraday_region(region)?;
    if offset.is_some() {
        return Err(TimeCompatError::AwareIntradayDatetime);
    }
    if (local.year(), local.month(), local.day()) == (1900, 1, 1) {
        if let Ok(index) = crate::time_to_day_index(local.time(), market) {
            return Ok(index);
        }
    }
    let format = if local.nanosecond() == 0 {
        "%Y-%m-%d %H:%M:%S"
    } else {
        "%Y-%m-%d %H:%M:%S%.6f"
    };
    let datetime = local.format(format).to_string();
    Err(TimeCompatError::OutsideIntradayDatetime {
        datetime,
        region: region.to_owned(),
    })
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
/// Rejects `NaT`, non-Timestamp variants, invalid timezone labels, local dates outside Python's
/// year range, and errors from the existing sampled-minute alignment implementation.
///
/// Uses [`default_time_calendar_cache`], retaining mutations to the source positional key.
///
/// # Panics
///
/// Panics if an earlier Rust panic poisoned the shared cache's internal state lock.
pub fn align_sampled_minute_compatible(
    value: &TemporalFrameValue,
    sample_minutes: &num_bigint::BigInt,
    minute_shift: &num_bigint::BigInt,
    region: Region,
) -> Result<TemporalFrameValue, TimeCompatError> {
    align_sampled_minute_with_cache(
        value,
        sample_minutes,
        minute_shift,
        region,
        default_time_calendar_cache(),
    )
}

/// Align using an explicit source cache, including mutations to its positional `(shift, region)` key.
///
/// Sampling takes a snapshot before reading the timestamp, matching Python list slicing.
///
/// # Errors
///
/// Returns calendar/sampling errors before timestamp errors; an empty sampled list fails after
/// timestamp access. Also reports a poisoned caller-owned calendar lock.
///
/// # Panics
///
/// Panics if an earlier Rust panic poisoned the cache's internal state lock.
pub fn align_sampled_minute_with_cache(
    value: &TemporalFrameValue,
    sample_minutes: &num_bigint::BigInt,
    minute_shift: &num_bigint::BigInt,
    region: Region,
    cache: &TimeCalendarCache,
) -> Result<TemporalFrameValue, TimeCompatError> {
    let calendar = cache.get(&TimeCalendarCall::new(
        vec![minute_shift.clone().into(), region.code().into()],
        Vec::new(),
    ))?;
    let sampled = sample_calendar(
        &calendar
            .lock()
            .map_err(|_| TimeCompatError::CalendarLockPoisoned)?,
        sample_minutes,
    )?;
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
    let CompatibleLocalTimestamp { local, year, .. } =
        compatible_local_timestamp(*ticks, *unit, timezone.as_deref())?;
    // Source evaluates x.date() before cal[idx], even when the sampled list is empty.
    if !(1..=9_999).contains(&year) {
        return Err(TimeCompatError::TimestampDateNotSupported);
    }
    if sampled.is_empty() {
        return Err(TimeCompatError::EmptyCalendar);
    }
    let aligned = align_to_sampled_calendar(local, &sampled);
    concat_date_time_compatible(aligned.date(), aligned.time())
}

/// Query a closed intraday range using the shared source calendar cache.
///
/// Inputs are already parsed clock values and a frequency. As in the source, only the frequency
/// count determines sampling; its unit is ignored.
///
/// # Errors
///
/// Returns calendar, poisoned-calendar-lock or zero sampling step errors.
///
/// # Panics
///
/// Panics if an earlier Rust panic poisoned the shared cache's internal state lock.
pub fn day_minute_index_range_compatible(
    start: NaiveTime,
    end: NaiveTime,
    frequency: &Frequency,
    region: Region,
) -> Result<(i64, i64), TimeCompatError> {
    day_minute_index_range_with_cache(
        start,
        end,
        frequency,
        region.code(),
        default_time_calendar_cache(),
    )
}

/// Run text range queries through the currently implemented constructor stages.
///
/// `None` means a reached endpoint needs the not-yet-complete general parser.
/// It does not mean invalid input. Recognized early errors are returned even if
/// a later endpoint would need that parser. Local fields, not rounded storage
/// ticks, supply the clocks. No calendar is accessed until both clocks and the
/// raw frequency have been successfully constructed.
///
/// # Errors
///
/// Preserves constructor, missing-clock, frequency, calendar and sampling errors.
///
/// # Panics
///
/// Only on violated parser/time invariants or a poisoned cache state lock.
pub fn day_minute_text_range_with_cache(
    start: &str,
    end: &str,
    frequency: &str,
    region: &str,
    context: impl Into<crate::timestamp_text::TimestampTextContext>,
    cache: &TimeCalendarCache,
) -> Option<Result<(i64, i64), TimeCompatError>> {
    let context = context.into();
    let start = match text_clock(start, context)? {
        Ok(value) => value,
        Err(error) => return Some(Err(error)),
    };
    let end = match text_clock(end, context)? {
        Ok(value) => value,
        Err(error) => return Some(Err(error)),
    };
    Some(
        frequency
            .parse::<Frequency>()
            .map_err(|_| TimeCompatError::RangeFrequencyFormat)
            .and_then(|frequency| {
                day_minute_index_range_with_cache(start, end, &frequency, region, cache)
            }),
    )
}

/// Text range result with constructor warnings retained even on a later failure.
#[derive(Debug)]
pub struct TextRangeOutcome {
    /// Pending parsing remains distinct from a source error.
    pub result: Option<Result<(i64, i64), TimeCompatError>>,
    /// Ordered source `FutureWarning` messages from reached endpoints.
    pub future_warnings: Vec<String>,
}

/// Query text endpoints while preserving their constructor warnings and order.
///
/// # Panics
/// Only on violated parser invariants or a poisoned calendar cache lock.
#[must_use]
pub fn day_minute_text_range_with_warnings(
    start: &str,
    end: &str,
    frequency: &str,
    region: &str,
    context: crate::timestamp_text::TimestampTextZoneContext<'_>,
    cache: &TimeCalendarCache,
) -> TextRangeOutcome {
    let mut future_warnings = Vec::new();
    let mut clock = |text| {
        let outcome = crate::timestamp_text::parse_timestamp_text_with_warnings(text, context);
        future_warnings.extend(outcome.future_warnings);
        outcome.result.map(|result| {
            result
                .map_err(TimeCompatError::from)
                .and_then(|value| value.time())
        })
    };
    let result = (|| {
        let start = match clock(start)? {
            Ok(value) => value,
            Err(error) => return Some(Err(error)),
        };
        let end = match clock(end)? {
            Ok(value) => value,
            Err(error) => return Some(Err(error)),
        };
        Some(
            frequency
                .parse::<Frequency>()
                .map_err(|_| TimeCompatError::RangeFrequencyFormat)
                .and_then(|frequency| {
                    day_minute_index_range_with_cache(start, end, &frequency, region, cache)
                }),
        )
    })();
    TextRangeOutcome {
        result,
        future_warnings,
    }
}

impl From<crate::timestamp_text::TimestampTextError> for TimeCompatError {
    fn from(error: crate::timestamp_text::TimestampTextError) -> Self {
        match error {
            crate::timestamp_text::TimestampTextError::OffsetOverflow => {
                Self::TimestampTextOffsetOverflow
            }
            crate::timestamp_text::TimestampTextError::InvalidOffset { message } => {
                Self::TimestampTextInvalidOffset { message }
            }
            crate::timestamp_text::TimestampTextError::NotDateLike { message } => {
                Self::TimestampTextNotDateLike { message }
            }
            crate::timestamp_text::TimestampTextError::InvalidCalendar { message } => {
                Self::TimestampTextInvalidCalendar { message }
            }
            crate::timestamp_text::TimestampTextError::OutOfBounds(error) => {
                Self::TimestampTextBounds(error)
            }
            crate::timestamp_text::TimestampTextError::DateParse { message } => {
                Self::TimestampTextDateParse { message }
            }
        }
    }
}

fn text_clock(
    text: &str,
    context: crate::timestamp_text::TimestampTextContext,
) -> Option<Result<NaiveTime, TimeCompatError>> {
    crate::timestamp_text::parse_timestamp_text_compatible(text, context).map(|result| {
        result
            .map_err(TimeCompatError::from)
            .and_then(|value| value.time())
    })
}

/// Query a closed range from Timestamp/`NaT` endpoints and raw frequency text.
///
/// Extracts local, timezone-free clocks at Python microsecond precision, in
/// start/end order, before parsing frequency or consulting the supplied cache.
/// Calendar dates are irrelevant, including years outside Python datetime range.
///
/// # Errors
///
/// Preserves endpoint, frequency, calendar and sampling error order. This typed
/// entry rejects non-Timestamp/non-`NaT` variants rather than parsing arbitrary objects.
///
/// # Panics
///
/// Panics only on violated internal time invariants or a poisoned cache state lock.
pub fn day_minute_timestamp_range_with_cache(
    start: &TemporalFrameValue,
    end: &TemporalFrameValue,
    frequency: &str,
    region: &str,
    cache: &TimeCalendarCache,
) -> Result<(i64, i64), TimeCompatError> {
    let start = timestamp_clock(start)?;
    let end = timestamp_clock(end)?;
    let frequency = frequency
        .parse::<Frequency>()
        .map_err(|_| TimeCompatError::RangeFrequencyFormat)?;
    day_minute_index_range_with_cache(start, end, &frequency, region, cache)
}

fn timestamp_clock(value: &TemporalFrameValue) -> Result<NaiveTime, TimeCompatError> {
    match value {
        TemporalFrameValue::Timestamp {
            ticks,
            unit,
            timezone,
        } => {
            let local = compatible_local_timestamp(*ticks, *unit, timezone.as_deref())?.local;
            Ok(python_clock_precision(local.time()))
        }
        TemporalFrameValue::Builtin(BuiltinFrameValue::NotATime) => {
            Err(TimeCompatError::NaTDoesNotSupportTime)
        }
        _ => Err(TimeCompatError::UnsupportedTemporalValue {
            operation: "get_day_min_idx_range",
        }),
    }
}

pub(crate) fn python_clock_precision(clock: NaiveTime) -> NaiveTime {
    clock
        .with_nanosecond(clock.nanosecond() / 1_000 * 1_000)
        .expect("truncating a valid fractional second preserves its range")
}

/// Query a closed range with an explicit cache, retaining the source's keyword-only region key.
///
/// Mutable lists are not sorted: Python bisect's exact control flow is retained even if a caller
/// reorders a cached list. Empty lists return `(0, -1)`.
/// Raw region strings retain source validation and unsupported-region diagnostics.
///
/// # Errors
///
/// Returns calendar, poisoned-calendar-lock or zero sampling step errors.
///
/// # Panics
///
/// Panics if an earlier Rust panic poisoned the cache's internal state lock.
pub fn day_minute_index_range_with_cache(
    start: NaiveTime,
    end: NaiveTime,
    frequency: &Frequency,
    region: &str,
    cache: &TimeCalendarCache,
) -> Result<(i64, i64), TimeCompatError> {
    let calendar = cache.get(&TimeCalendarCall::new(
        Vec::new(),
        vec![TimeCalendarKeyword::new("region", region)],
    ))?;
    let sampled = sample_calendar(
        &calendar
            .lock()
            .map_err(|_| TimeCompatError::CalendarLockPoisoned)?,
        &frequency.count.clone().into(),
    )?;
    Ok((
        python_bisect_left(&sampled, start)
            .to_i64()
            .unwrap_or(i64::MAX),
        python_bisect_right(&sampled, end)
            .to_i64()
            .unwrap_or(i64::MAX)
            - 1,
    ))
}

/// Evaluate the source single-value predicate on typed timestamps and durations.
///
/// Subtraction uses absolute instants and the finer operand resolution, while
/// market closing rules use the starting timestamp's local clock. Missingness
/// disables the duration comparison, not the subsequent market closing rules.
/// Wide dates use Gregorian cycle reduction. Named zones outside the source's
/// finite transition table retain its final offset rather than recurring DST.
///
/// # Errors
///
/// Preserves region-first dispatch, mixed-awareness errors, operand promotion
/// and duration overflow. Other union variants and invalid Arrow timezone labels
/// are rejected at this typed boundary.
///
/// # Panics
///
/// Only if bounded Gregorian-cycle decomposition violates its internal integer
/// or reduced-year invariants.
pub fn is_single_value_compatible(
    start: &TemporalFrameValue,
    end: &TemporalFrameValue,
    frequency: &TemporalFrameValue,
    region: &str,
) -> Result<bool, TimeCompatError> {
    let region = match region {
        "cn" => Region::Cn,
        "tw" => Region::Tw,
        "us" => Region::Us,
        _ => {
            return Err(TimeCompatError::UnsupportedSingleValueRegion {
                region: region.into(),
            });
        }
    };
    let start_parts = single_timestamp_parts(start)?;
    let end_parts = single_timestamp_parts(end)?;
    if let (
        Some(SingleTimestamp {
            ticks: start_ticks,
            unit: start_unit,
            timezone: start_zone,
        }),
        Some(SingleTimestamp {
            ticks: end_ticks,
            unit: end_unit,
            timezone: end_zone,
        }),
    ) = (start_parts, end_parts)
    {
        if start_zone.is_some() != end_zone.is_some() {
            return Err(TimeCompatError::MixedTimestampAwareness);
        }
        let factor = unit_nanos(start_unit).min(unit_nanos(end_unit));
        // Pandas promotes the left operand (end) before the right operand (start).
        let end_ticks = promote_single_timestamp(end_ticks, end_unit, end_zone, factor)?;
        let start_ticks = promote_single_timestamp(start_ticks, start_unit, start_zone, factor)?;
        let elapsed = end_ticks
            .checked_sub(start_ticks)
            .ok_or(TimeCompatError::TimestampSubtractionOverflow)?;
        if elapsed == i64::MIN {
            return Err(TimeCompatError::TimestampSubtractionSentinel);
        }
        match frequency {
            TemporalFrameValue::Duration { ticks, unit } => {
                if i128::from(elapsed) * factor < i128::from(*ticks) * unit_nanos(*unit) {
                    return Ok(true);
                }
            }
            TemporalFrameValue::Builtin(BuiltinFrameValue::NotATime) => {}
            _ => return Err(TimeCompatError::UnsupportedSingleValueFrequency),
        }
    }
    let Some(SingleTimestamp {
        ticks,
        unit,
        timezone,
    }) = start_parts
    else {
        return Ok(false);
    };
    let clock = compatible_local_timestamp(ticks, unit, timezone)?
        .local
        .time();
    Ok(is_single_market_value(
        clock,
        chrono::TimeDelta::zero(),
        chrono::TimeDelta::zero(),
        region,
    ))
}

#[derive(Clone, Copy)]
struct SingleTimestamp<'a> {
    ticks: i64,
    unit: TimeUnit,
    timezone: Option<&'a str>,
}

fn single_timestamp_parts(
    value: &TemporalFrameValue,
) -> Result<Option<SingleTimestamp<'_>>, TimeCompatError> {
    match value {
        TemporalFrameValue::Timestamp {
            ticks,
            unit,
            timezone,
        } => Ok(Some(SingleTimestamp {
            ticks: *ticks,
            unit: *unit,
            timezone: timezone.as_deref(),
        })),
        TemporalFrameValue::Builtin(BuiltinFrameValue::NotATime) => Ok(None),
        _ => Err(TimeCompatError::UnsupportedTemporalValue {
            operation: "is_single_value",
        }),
    }
}

fn unit_nanos(unit: TimeUnit) -> i128 {
    match unit {
        TimeUnit::Second => 1_000_000_000,
        TimeUnit::Millisecond => 1_000_000,
        TimeUnit::Microsecond => 1_000,
        TimeUnit::Nanosecond => 1,
    }
}

struct CompatibleLocalTimestamp {
    local: NaiveDateTime,
    year: i64,
    offset: Option<FixedOffset>,
}

fn arrow_timezone_label(label: &str) -> Cow<'_, str> {
    if label == "tzutc()" {
        return Cow::Borrowed("UTC");
    }
    // Retain dateutil's label in owned values, translating only at the Arrow
    // calculation boundary. The text constructor emits whole-minute offsets.
    if let Some(offset) = label
        .strip_prefix("tzoffset(")
        .and_then(|value| value.strip_suffix(')'))
        .and_then(|value| value.split_once(", "))
        .filter(|(name, _)| {
            *name == "None"
                || name
                    .strip_prefix('\'')
                    .and_then(|value| value.strip_suffix('\''))
                    .is_some_and(|value| {
                        (1..=5).contains(&value.len())
                            && value.bytes().all(|byte| byte.is_ascii_uppercase())
                    })
        })
        .map(|(_, offset)| offset)
        .and_then(|value| value.parse::<i32>().ok())
        .filter(|value| value % 60 == 0)
        .and_then(FixedOffset::east_opt)
    {
        return Cow::Owned(offset.to_string());
    }
    Cow::Borrowed(
        label
            .strip_prefix("UTC")
            .filter(|suffix| suffix.starts_with(['+', '-']))
            .unwrap_or(label),
    )
}

fn compatible_local_timestamp(
    ticks: i64,
    unit: TimeUnit,
    timezone: Option<&str>,
) -> Result<CompatibleLocalTimestamp, TimeCompatError> {
    let zone = timezone
        .map(|label| {
            // Pandas' datetime.timezone display prefixes numeric offsets with UTC;
            // Arrow accepts the equivalent signed offset without that prefix.
            let normalized = arrow_timezone_label(label);
            normalized
                .parse::<Tz>()
                .map_err(|error| TimeCompatError::InvalidTimezone {
                    timezone: label.to_owned(),
                    message: error.to_string(),
                })
        })
        .transpose()?;
    let named_zone = zone.is_some_and(|zone| {
        let name = zone.to_string();
        name != "UTC" && !name.starts_with(['+', '-'])
    });
    // Pandas 2.3.3 Localizer scales its i64-nanosecond sentinel to the input
    // resolution; bisect before that sentinel selects the final array entry.
    // pytz's TZif v1 transition data ends within signed 32-bit epoch seconds.
    // Resolve that last state with the existing timezone database, including
    // southern-hemisphere DST, instead of assuming a universal standard offset.
    let scale = unit_nanos(unit);
    let outside_transitions = i128::from(ticks) < i128::from(i64::MIN + 1).div_euclid(scale)
        || i128::from(ticks) * scale >= i128::from(i32::MAX) * 1_000_000_000;
    let final_offset = if outside_transitions {
        zone.map(|zone| {
            DateTime::from_timestamp(i64::from(i32::MAX), 0)
                .expect("signed 32-bit epoch limit is in Chrono's range")
                .with_timezone(&zone)
                .offset()
                .fix()
        })
    } else if named_zone && i128::from(ticks) * scale < i128::from(i32::MIN) * 1_000_000_000 {
        // pytz's 32-bit table retains its first standard (LMT) entry before
        // the earliest representable transition, not intervening 64-bit history.
        zone.map(|zone| {
            NaiveDate::from_ymd_opt(1, 1, 1)
                .expect("year one is valid")
                .and_hms_opt(0, 0, 0)
                .expect("midnight is valid")
                .and_utc()
                .with_timezone(&zone)
                .offset()
                .fix()
        })
    } else {
        None
    };
    let (utc, year_shift) = if let Some(utc) = ticks_to_utc(ticks, unit) {
        (utc, 0)
    } else {
        // Same Gregorian-cycle reduction used by temporal object inference.
        // Chrono still performs calendar decomposition; only whole 400-year
        // cycles are carried separately to retain the full i64 timestamp range.
        const CYCLE_SECONDS: i128 = 146_097 * 86_400;
        let nanos = i128::from(ticks) * unit_nanos(unit);
        let seconds = nanos.div_euclid(1_000_000_000);
        let reduced_seconds = i64::try_from(seconds.rem_euclid(CYCLE_SECONDS))
            .expect("400-year cycle fits i64 seconds");
        let fraction =
            u32::try_from(nanos.rem_euclid(1_000_000_000)).expect("nanosecond remainder fits u32");
        let reduced = DateTime::from_timestamp(reduced_seconds, fraction)
            .expect("reduced timestamp is in 1970..2370");
        let shift = i64::try_from(seconds.div_euclid(CYCLE_SECONDS) * 400)
            .expect("years represented by i64 seconds fit i64");
        (reduced, shift)
    };
    let (local, offset) = match zone {
        None => (utc.naive_utc(), None),
        Some(zone) => {
            let offset = final_offset.unwrap_or_else(|| utc.with_timezone(&zone).offset().fix());
            let offset = if named_zone {
                // pytz/tzfile.py rounds transition offsets with floor((s+30)/60).
                FixedOffset::east_opt((offset.local_minus_utc() + 30).div_euclid(60) * 60)
                    .expect("rounded IANA offsets remain less than 24 hours")
            } else {
                offset
            };
            let local = utc.with_timezone(&offset);
            (local.naive_local(), Some(offset))
        }
    };
    Ok(CompatibleLocalTimestamp {
        local,
        year: i64::from(local.year()) + year_shift,
        offset,
    })
}

fn promote_single_timestamp(
    ticks: i64,
    unit: TimeUnit,
    timezone: Option<&str>,
    factor: i128,
) -> Result<i64, TimeCompatError> {
    if let Ok(value) = i64::try_from(i128::from(ticks) * (unit_nanos(unit) / factor)) {
        Ok(value)
    } else {
        let CompatibleLocalTimestamp {
            local,
            year,
            offset,
        } = compatible_local_timestamp(ticks, unit, timezone)?;
        // Pandas displays microsecond-aligned fractions with six digits, even
        // when Chrono's default display would shorten them to milliseconds.
        let format = if local.nanosecond() == 0 {
            "-%m-%d %H:%M:%S"
        } else {
            "-%m-%d %H:%M:%S%.6f"
        };
        // A nanosecond timestamp never overflows promotion: there is no finer
        // supported resolution. All operands here are microsecond-aligned.
        let offset = offset.map_or_else(String::new, |offset| offset.to_string());
        let timestamp = format!("{year:04}{}{offset}", local.format(format));
        // A common seconds resolution cannot overflow promotion: seconds are
        // the coarsest supported unit, so both operands would be unchanged.
        // This error therefore only has millisecond-or-finer destinations.
        let unit = match factor {
            1_000_000 => "ms",
            1_000 => "us",
            _ => "ns",
        };
        Err(TimeCompatError::TimestampPromotionOverflow { timestamp, unit })
    }
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
