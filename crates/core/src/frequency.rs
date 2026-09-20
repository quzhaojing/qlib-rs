//! Frequency parsing and normalization compatible with `qlib.utils.time.Freq`.

use std::{fmt, str::FromStr, sync::LazyLock};

use chrono::TimeDelta;
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, ToPrimitive};
use regex::Regex;
use serde::{Deserialize, Serialize};
use strum::{Display, EnumString};
use thiserror::Error;

static FREQUENCY_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^([0-9]*)([a-z]+)$").expect("the static frequency regular expression is valid")
});

/// Normalized Qlib frequency units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Display, EnumString)]
#[serde(rename_all = "lowercase")]
#[strum(ascii_case_insensitive)]
pub enum FrequencyUnit {
    /// Calendar month.
    #[strum(to_string = "month", serialize = "mon")]
    Month,
    /// Calendar week.
    #[strum(to_string = "week", serialize = "w")]
    Week,
    /// Calendar day.
    #[strum(to_string = "day", serialize = "d")]
    Day,
    /// Minute, normalized to `min` for Qlib filenames.
    #[strum(to_string = "min", serialize = "minute")]
    Minute,
}

impl FrequencyUnit {
    const fn approximate_minutes(self) -> u64 {
        match self {
            Self::Minute => 1,
            Self::Day => 60 * 24,
            Self::Week => 7 * 60 * 24,
            Self::Month => 30 * 7 * 60 * 24,
        }
    }
}

/// Units backed directly by Qlib calendar files.
pub const SUPPORTED_CALENDAR_UNITS: [FrequencyUnit; 2] =
    [FrequencyUnit::Minute, FrequencyUnit::Day];

/// A normalized Qlib frequency.
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Frequency {
    /// Number of normalized units. Zero is accepted for Python compatibility.
    pub count: BigUint,
    /// Normalized unit.
    pub unit: FrequencyUnit,
}

/// Failures produced while parsing or converting frequencies.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum FrequencyError {
    /// The text does not match the Qlib frequency grammar.
    #[error(
        "freq format is not supported, the freq should be like (n)month/mon, (n)week/w, (n)day/d, (n)minute/min: {input}"
    )]
    UnsupportedFormat {
        /// Rejected input.
        input: String,
    },
    /// Pandas Timedelta does not accept this spelling as a fixed-duration unit.
    #[error("invalid unit abbreviation: {unit}")]
    UnsupportedDurationUnit {
        /// Rejected unit.
        unit: String,
    },
    /// The requested fixed duration exceeds Chrono's representable range.
    #[error("duration is out of range: {count}{unit}")]
    DurationOutOfRange {
        /// Requested count.
        count: BigInt,
        /// Requested unit.
        unit: String,
    },
}

impl Frequency {
    /// Construct a frequency from already-normalized parts.
    #[must_use]
    pub fn new(count: u64, unit: FrequencyUnit) -> Self {
        Self {
            count: BigUint::from(count),
            unit,
        }
    }

    /// Approximate this frequency using Qlib's minute weights.
    ///
    /// The month weight intentionally preserves the upstream `30 * 7` day formula.
    #[must_use]
    pub fn approximate_minutes(&self) -> BigUint {
        &self.count * BigUint::from(self.unit.approximate_minutes())
    }

    /// Return `self - other` in Qlib's approximate minutes.
    #[must_use]
    pub fn minute_delta(&self, other: &Self) -> BigInt {
        BigInt::from(self.approximate_minutes()) - BigInt::from(other.approximate_minutes())
    }

    /// Select the nearest candidate that is no coarser than `base`.
    #[must_use]
    pub fn nearest_resample_source(base: &Self, candidates: &[Self]) -> Option<Self> {
        let mut best = None;
        for candidate in candidates {
            let delta = base.minute_delta(candidate);
            if delta.sign() == Sign::Minus {
                continue;
            }
            if best
                .as_ref()
                .is_none_or(|(best_delta, _)| delta < *best_delta)
            {
                best = Some((delta, candidate.clone()));
            }
        }
        best.map(|(_, frequency)| frequency)
    }

    /// Reproduce `Freq.get_timedelta(n, freq)` for the unit spellings Pandas accepts.
    ///
    /// # Errors
    ///
    /// Returns [`FrequencyError::UnsupportedDurationUnit`] for unit spellings rejected by
    /// Pandas Timedelta, or [`FrequencyError::DurationOutOfRange`] when Chrono cannot represent
    /// the requested fixed duration.
    pub fn time_delta(count: BigInt, unit: &str) -> Result<TimeDelta, FrequencyError> {
        let normalized = unit.to_ascii_lowercase();
        let Some(fixed_count) = count.to_i64() else {
            return Err(FrequencyError::DurationOutOfRange {
                count,
                unit: unit.to_owned(),
            });
        };
        let value = match normalized.as_str() {
            "min" | "minute" => TimeDelta::try_minutes(fixed_count),
            "day" | "d" => TimeDelta::try_days(fixed_count),
            "w" => TimeDelta::try_weeks(fixed_count),
            _ => {
                return Err(FrequencyError::UnsupportedDurationUnit {
                    unit: unit.to_owned(),
                });
            }
        };
        value.ok_or_else(|| FrequencyError::DurationOutOfRange {
            count,
            unit: unit.to_owned(),
        })
    }
}

impl FromStr for Frequency {
    type Err = FrequencyError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let normalized = input.to_ascii_lowercase();
        let captures = FREQUENCY_PATTERN.captures(&normalized).ok_or_else(|| {
            FrequencyError::UnsupportedFormat {
                input: input.to_owned(),
            }
        })?;
        let count_text = captures.get(1).map_or("", |value| value.as_str());
        let count = if count_text.is_empty() {
            BigUint::from(1_u8)
        } else {
            count_text
                .parse()
                .expect("the frequency regex captures only decimal digits")
        };
        let unit_text = captures.get(2).map_or("", |value| value.as_str());
        let unit = unit_text
            .parse()
            .map_err(|_| FrequencyError::UnsupportedFormat {
                input: input.to_owned(),
            })?;
        Ok(Self { count, unit })
    }
}

impl TryFrom<String> for Frequency {
    type Error = FrequencyError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.parse()
    }
}

impl From<Frequency> for String {
    fn from(value: Frequency) -> Self {
        value.to_string()
    }
}

impl fmt::Display for Frequency {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.count.is_one() && self.unit == FrequencyUnit::Day {
            formatter.write_str("day")
        } else {
            write!(formatter, "{}{}", self.count, self.unit)
        }
    }
}

impl fmt::Debug for Frequency {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Freq({self})")
    }
}
