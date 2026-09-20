//! Sampled-minute alignment compatible with `qlib.utils.time.cal_sam_minute`.

use chrono::{NaiveDateTime, NaiveTime};
use num_bigint::{BigInt, Sign};
use num_traits::ToPrimitive;
use thiserror::Error;

use crate::{MarketCalendarError, Region, minute_calendar};

/// Failures while aligning a timestamp to a sampled intraday calendar.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum MinuteAlignmentError {
    /// Python list slicing rejects a zero stride.
    #[error("sample-minute step cannot be zero")]
    ZeroSamplingStep,
    /// The configured minute shift cannot be represented by the upstream calendar.
    #[error(transparent)]
    Calendar(#[from] MarketCalendarError),
}

/// Align a timestamp backward to Qlib's sampled intraday minute calendar.
///
/// `sample_minutes` deliberately supports negative values because Python list
/// slicing does. `minute_shift` is the explicit equivalent of
/// `QlibConfig.min_data_shift`; a positive shift moves the market calendar backward.
/// The input date is preserved and the result always lands on a whole minute.
///
/// # Errors
///
/// Returns [`MinuteAlignmentError::ZeroSamplingStep`] for a zero slice stride and
/// [`MinuteAlignmentError::Calendar`] when the configured shift exceeds Pandas'
/// representable calendar range.
pub fn align_sampled_minute(
    value: NaiveDateTime,
    sample_minutes: &BigInt,
    minute_shift: &BigInt,
    region: Region,
) -> Result<NaiveDateTime, MinuteAlignmentError> {
    let direction = sample_minutes.sign();
    if direction == Sign::NoSign {
        return Err(MinuteAlignmentError::ZeroSamplingStep);
    }
    let step = sample_minutes.magnitude().to_usize().unwrap_or(usize::MAX);
    let calendar = minute_calendar(minute_shift, region)?;
    let sampled: Vec<NaiveTime> = if direction == Sign::Minus {
        calendar.iter().rev().step_by(step).copied().collect()
    } else {
        calendar.iter().step_by(step).copied().collect()
    };
    let insertion = python_bisect_right(&sampled, value.time());
    let selected = if insertion == 0 {
        sampled.last().copied().unwrap_or(value.time())
    } else {
        sampled.get(insertion - 1).copied().unwrap_or(value.time())
    };
    Ok(value.date().and_time(selected))
}

fn python_bisect_right(values: &[NaiveTime], needle: NaiveTime) -> usize {
    let mut low = 0;
    let mut high = values.len();
    while low < high {
        let middle = low + (high - low) / 2;
        if needle < values[middle] {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    low
}
