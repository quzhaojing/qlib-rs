//! Intraday minute-index helpers compatible with `qlib.utils.time`.

use chrono::NaiveTime;
use num_traits::{ToPrimitive, Zero};
use thiserror::Error;

use crate::{Frequency, Region, regular_minute_calendar};

/// Errors produced by intraday time and sampled-range indexing.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum IntradayIndexError {
    /// The string is not in Python's `%H:%M` input format.
    #[error("invalid market time, expected %H:%M: {input}")]
    InvalidTime {
        /// Rejected input.
        input: String,
    },
    /// The time falls outside every regular session for the selected market.
    #[error("{time} is not an opening minute of the {region} stock market")]
    OutsideTradingSessions {
        /// Rejected clock time.
        time: NaiveTime,
        /// Selected market.
        region: Region,
    },
    /// Python list slicing rejects a zero stride.
    #[error("intraday frequency step cannot be zero")]
    ZeroFrequencyStep,
}

/// Parse the exact clock-only string format accepted by `time_to_day_index`.
///
/// # Errors
///
/// Returns [`IntradayIndexError::InvalidTime`] for malformed or trailing input.
pub fn parse_market_time(input: &str) -> Result<NaiveTime, IntradayIndexError> {
    NaiveTime::parse_from_str(input, "%H:%M").map_err(|_| IntradayIndexError::InvalidTime {
        input: input.to_owned(),
    })
}

/// Return the zero-based minute index across a region's concatenated sessions.
///
/// Seconds and subsecond values are truncated to the containing minute, matching
/// Python's `int(total_seconds / 60)` for a time inside a session.
///
/// # Errors
///
/// Returns [`IntradayIndexError::OutsideTradingSessions`] at session ends, during
/// breaks, and outside regular market hours.
pub fn time_to_day_index(time: NaiveTime, region: Region) -> Result<i64, IntradayIndexError> {
    let mut session_offset = 0;
    for session in region.trading_sessions() {
        if session.start <= time && time < session.end {
            return Ok(
                session_offset + time.signed_duration_since(session.start).num_seconds() / 60
            );
        }
        session_offset += session
            .end
            .signed_duration_since(session.start)
            .num_minutes();
    }
    Err(IntradayIndexError::OutsideTradingSessions { time, region })
}

/// Parse `%H:%M` and return its zero-based intraday minute index.
///
/// # Errors
///
/// Returns [`IntradayIndexError::InvalidTime`] for malformed text or
/// [`IntradayIndexError::OutsideTradingSessions`] outside regular sessions.
pub fn time_to_day_index_str(input: &str, region: Region) -> Result<i64, IntradayIndexError> {
    time_to_day_index(parse_market_time(input)?, region)
}

/// Locate a closed clock-time range in a frequency-sampled intraday calendar.
///
/// This intentionally permits reversed and empty results. An end before the first
/// sampled minute is represented as `-1`, exactly like Python's
/// `bisect_right(calendar, end) - 1`.
///
/// # Errors
///
/// Returns [`IntradayIndexError::ZeroFrequencyStep`] because Python rejects a
/// zero list-slice stride.
pub fn day_minute_index_range(
    start: NaiveTime,
    end: NaiveTime,
    frequency: &Frequency,
    region: Region,
) -> Result<(i64, i64), IntradayIndexError> {
    if frequency.count.is_zero() {
        return Err(IntradayIndexError::ZeroFrequencyStep);
    }
    let step = frequency.count.to_usize().unwrap_or(usize::MAX);
    let sampled: Vec<_> = regular_minute_calendar(region)
        .iter()
        .step_by(step)
        .copied()
        .collect();
    let left = sampled.partition_point(|time| *time < start);
    let right_exclusive = sampled.partition_point(|time| *time <= end);
    Ok((
        left.to_i64().unwrap_or(i64::MAX),
        right_exclusive.to_i64().unwrap_or(i64::MAX) - 1,
    ))
}
