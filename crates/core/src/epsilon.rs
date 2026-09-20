//! One-second boundary shifts compatible with `qlib.utils.time.epsilon_change`.

use std::str::FromStr;

use chrono::{DateTime, NaiveDateTime, Utc};
use serde::{Deserialize, Serialize};
use strum::{Display, EnumString};
use thiserror::Error;

const NANOS_PER_SECOND: i64 = 1_000_000_000;

/// Direction of an epsilon timestamp shift.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Display, EnumString)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum EpsilonDirection {
    /// Move one second into history.
    Backward,
    /// Move one second into the future.
    Forward,
}

/// Failures while applying a Qlib epsilon shift.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum EpsilonError {
    /// The compatibility string is not one of the two Python spellings.
    #[error("invalid epsilon direction: {direction}")]
    InvalidDirection {
        /// Rejected direction.
        direction: String,
    },
    /// The input cannot be represented as a non-NaT Pandas nanosecond timestamp.
    #[error("timestamp is outside the pandas nanosecond range: {timestamp}")]
    InputOutOfRange {
        /// Rejected input.
        timestamp: NaiveDateTime,
    },
    /// Moving one second would leave Pandas' nanosecond timestamp range.
    #[error("epsilon shift leaves the pandas nanosecond range: {timestamp} {direction}")]
    ResultOutOfRange {
        /// Original timestamp.
        timestamp: NaiveDateTime,
        /// Requested direction.
        direction: EpsilonDirection,
    },
}

/// Move a timestamp exactly one second in the requested direction.
///
/// # Errors
///
/// Returns [`EpsilonError::InputOutOfRange`] when the input is outside Pandas'
/// non-NaT nanosecond domain, or [`EpsilonError::ResultOutOfRange`] when the result
/// would cross that domain's boundary.
pub fn epsilon_change(
    timestamp: NaiveDateTime,
    direction: EpsilonDirection,
) -> Result<NaiveDateTime, EpsilonError> {
    let Some(nanos) = timestamp
        .and_utc()
        .timestamp_nanos_opt()
        .filter(|value| *value != i64::MIN)
    else {
        return Err(EpsilonError::InputOutOfRange { timestamp });
    };
    let shifted = match direction {
        EpsilonDirection::Backward => nanos.checked_sub(NANOS_PER_SECOND),
        EpsilonDirection::Forward => nanos.checked_add(NANOS_PER_SECOND),
    }
    .filter(|value| *value != i64::MIN)
    .ok_or(EpsilonError::ResultOutOfRange {
        timestamp,
        direction,
    })?;
    Ok(DateTime::<Utc>::from_timestamp_nanos(shifted).naive_utc())
}

/// Apply Python's default backward epsilon shift.
///
/// # Errors
///
/// Returns the same range errors as [`epsilon_change`].
pub fn epsilon_change_backward(timestamp: NaiveDateTime) -> Result<NaiveDateTime, EpsilonError> {
    epsilon_change(timestamp, EpsilonDirection::Backward)
}

/// Parse a Python-compatible direction string and apply the epsilon shift.
///
/// # Errors
///
/// Returns [`EpsilonError::InvalidDirection`] for every spelling other than the
/// lowercase `backward` and `forward`, plus the range errors from [`epsilon_change`].
pub fn epsilon_change_str(
    timestamp: NaiveDateTime,
    direction: &str,
) -> Result<NaiveDateTime, EpsilonError> {
    let parsed =
        EpsilonDirection::from_str(direction).map_err(|_| EpsilonError::InvalidDirection {
            direction: direction.to_owned(),
        })?;
    epsilon_change(timestamp, parsed)
}
