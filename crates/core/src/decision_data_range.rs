//! Data-calendar range projection for one trade decision.

use chrono::{Duration, NaiveDateTime};
use thiserror::Error;

use crate::{TradeDecision, TradeRangeError};

/// Closed index bounds returned by Qlib's global data-calendar locator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DataCalendarLocation {
    pub start_index: i64,
    pub end_index: i64,
}

/// Replaceable global data-calendar boundary used by decision range projection.
pub trait DataCalendarLocator {
    /// Locate the closed timestamps at the requested exchange frequency.
    ///
    /// # Errors
    ///
    /// Returns the provider's acquisition, parsing, or indexing failure.
    fn locate(
        &mut self,
        start_time: NaiveDateTime,
        end_time: NaiveDateTime,
        frequency: &str,
    ) -> Result<DataCalendarLocation, DataCalendarLocatorError>;
}

/// Failure from a replaceable data-calendar locator.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("data calendar locator failed: {message}")]
pub struct DataCalendarLocatorError {
    pub message: String,
}

/// Failures from `BaseTradeDecision.get_data_cal_range_limit` projection.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum DecisionDataRangeError {
    #[error(transparent)]
    Locator(#[from] DataCalendarLocatorError),
    #[error("there is no trade range in this case")]
    MissingRange,
    #[error("data calendar range type is not supported: {0}")]
    UnsupportedRangeType(String),
    #[error(transparent)]
    TradeRange(#[from] TradeRangeError),
    #[error("data calendar index difference is outside the native integer range")]
    IndexOverflow,
}

/// Project a decision's optional time rule into zero-based data-calendar indices for its day.
///
/// The first full-day lookup always occurs before range absence, range-type validation, or
/// clipping, matching the source's observable provider order.
///
/// # Errors
///
/// Returns the first locator, missing-range, range-type, clipping, or index failure.
pub fn data_calendar_range_limit<T, L: DataCalendarLocator>(
    decision: &TradeDecision<T>,
    range_type: &str,
    raise_missing: bool,
    frequency: &str,
    locator: &mut L,
) -> Result<(i64, i64), DecisionDataRangeError> {
    let day_start = decision
        .start_time()
        .date()
        .and_time(chrono::NaiveTime::MIN);
    let day_end = day_start + (Duration::days(1) - Duration::seconds(1));
    let day = locator.locate(day_start, day_end, frequency)?;

    let Some(range) = decision.trade_range() else {
        if raise_missing {
            return Err(DecisionDataRangeError::MissingRange);
        }
        return day
            .end_index
            .checked_sub(day.start_index)
            .map(|end| (0, end))
            .ok_or(DecisionDataRangeError::IndexOverflow);
    };

    let (clip_start, clip_end) = match range_type {
        "full" => range.clip_time_range(day_start, day_end)?,
        "step" => range.clip_time_range(decision.start_time(), decision.end_time())?,
        other => {
            return Err(DecisionDataRangeError::UnsupportedRangeType(
                other.to_owned(),
            ));
        }
    };
    let clipped = locator.locate(clip_start, clip_end, frequency)?;
    Ok((
        clipped
            .start_index
            .checked_sub(day.start_index)
            .ok_or(DecisionDataRangeError::IndexOverflow)?,
        clipped
            .end_index
            .checked_sub(day.start_index)
            .ok_or(DecisionDataRangeError::IndexOverflow)?,
    ))
}
