//! Calendar resampling compatible with `qlib.utils.resam.resam_calendar`.

use chrono::{Datelike, NaiveDateTime, NaiveTime};
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use thiserror::Error;

use crate::{Frequency, FrequencyUnit, MinuteAlignmentError, Region, align_sampled_minute};

/// Failures produced while resampling a calendar.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CalendarResampleError {
    /// Minute output requires minute input in the upstream implementation.
    #[error("when sampling minute calendar, freq of raw calendar must be minute or min")]
    MinuteSampleFromNonMinuteRaw,
    /// The raw minute frequency cannot be coarser than the requested frequency.
    #[error("raw freq must be higher than sampling freq")]
    RawFrequencyCoarser,
    /// Python list/array slicing rejects a zero stride.
    #[error("calendar sampling step cannot be zero")]
    ZeroSamplingStep,
    /// Minute alignment failed, normally because the configured shift is out of range.
    #[error(transparent)]
    MinuteAlignment(#[from] MinuteAlignmentError),
}

/// Resample a sorted-or-unsorted calendar using Qlib's calendar semantics.
///
/// The result is always sorted and deduplicated, matching `numpy.unique`. Empty
/// calendars return before compatibility validation, as the Python function does.
/// `minute_shift` explicitly replaces the process-global `C.min_data_shift`.
///
/// # Errors
///
/// Returns a typed error for incompatible minute frequencies, a zero sampling
/// stride, or a minute-calendar alignment failure.
pub fn resample_calendar(
    raw_calendar: &[NaiveDateTime],
    raw_frequency: &Frequency,
    sampled_frequency: &Frequency,
    region: Region,
    minute_shift: &BigInt,
) -> Result<Vec<NaiveDateTime>, CalendarResampleError> {
    if raw_calendar.is_empty() {
        return Ok(Vec::new());
    }

    match sampled_frequency.unit {
        FrequencyUnit::Minute => resample_minutes(
            raw_calendar,
            raw_frequency,
            sampled_frequency,
            region,
            minute_shift,
        ),
        FrequencyUnit::Day => resample_period(raw_calendar, sampled_frequency, |days| days),
        FrequencyUnit::Week => resample_period(raw_calendar, sampled_frequency, week_starts),
        FrequencyUnit::Month => resample_period(raw_calendar, sampled_frequency, month_starts),
    }
}

fn resample_minutes(
    raw_calendar: &[NaiveDateTime],
    raw_frequency: &Frequency,
    sampled_frequency: &Frequency,
    region: Region,
    minute_shift: &BigInt,
) -> Result<Vec<NaiveDateTime>, CalendarResampleError> {
    validate_minute_frequencies(raw_frequency, sampled_frequency)?;
    let sample_minutes = BigInt::from(sampled_frequency.count.clone());
    let mut result = raw_calendar
        .iter()
        .copied()
        .map(|value| align_sampled_minute(value, &sample_minutes, minute_shift, region))
        .collect::<Result<Vec<_>, _>>()?;
    sort_unique(&mut result);
    Ok(result)
}

pub(crate) fn validate_minute_frequencies(
    raw_frequency: &Frequency,
    sampled_frequency: &Frequency,
) -> Result<(), CalendarResampleError> {
    if raw_frequency.unit != FrequencyUnit::Minute {
        return Err(CalendarResampleError::MinuteSampleFromNonMinuteRaw);
    }
    if raw_frequency.count > sampled_frequency.count {
        return Err(CalendarResampleError::RawFrequencyCoarser);
    }
    Ok(())
}

fn resample_period(
    raw_calendar: &[NaiveDateTime],
    sampled_frequency: &Frequency,
    select_starts: impl FnOnce(Vec<NaiveDateTime>) -> Vec<NaiveDateTime>,
) -> Result<Vec<NaiveDateTime>, CalendarResampleError> {
    let step = sampling_step(sampled_frequency)?;
    let mut days: Vec<_> = raw_calendar
        .iter()
        .map(|value| value.date().and_time(NaiveTime::MIN))
        .collect();
    sort_unique(&mut days);

    let period_starts = select_starts(days);
    Ok(period_starts.into_iter().step_by(step).collect())
}

fn sampling_step(frequency: &Frequency) -> Result<usize, CalendarResampleError> {
    if frequency.count == 0_u8.into() {
        return Err(CalendarResampleError::ZeroSamplingStep);
    }
    Ok(frequency.count.to_usize().unwrap_or(usize::MAX))
}

pub(crate) fn sort_unique(values: &mut Vec<NaiveDateTime>) {
    values.sort_unstable();
    values.dedup();
}

fn week_starts(days: Vec<NaiveDateTime>) -> Vec<NaiveDateTime> {
    let mut previous = None;
    days.into_iter()
        .filter(|value| {
            let current = value.weekday().num_days_from_monday();
            let keep = previous.is_none_or(|old| current < old);
            previous = Some(current);
            keep
        })
        .collect()
}

fn month_starts(days: Vec<NaiveDateTime>) -> Vec<NaiveDateTime> {
    let mut previous = None;
    days.into_iter()
        .filter(|value| {
            let current = value.day();
            let keep = previous.is_none_or(|old| current < old);
            previous = Some(current);
            keep
        })
        .collect()
}
