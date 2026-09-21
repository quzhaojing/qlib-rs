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
    // Python's `$` also matches immediately before one terminal LF (not CRLF).
    Regex::new(r"^([0-9]*)([a-z]+)\n?$").expect("the static frequency regular expression is valid")
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
    /// Pandas rejects ambiguous month/year duration spellings.
    #[error(
        "Units 'M', 'Y' and 'y' do not represent unambiguous timedelta values and are not supported."
    )]
    AmbiguousDurationUnit,
    /// Invalid duration text grammar.
    #[error("{message}")]
    InvalidDurationText {
        /// Source-compatible parser diagnostic.
        message: &'static str,
    },
    /// A clock separator appeared without a preceding number.
    #[error("expecting hh:mm:ss format, received: {input}")]
    InvalidClockText {
        /// Original Qlib concatenation.
        input: String,
    },
    /// A clock field cannot fit the parser's signed integer representation.
    #[error("int too big to convert")]
    ClockIntegerOverflow,
    /// A parsed decimal token is not a valid floating-point number.
    #[error("could not convert string to float: '{input}'")]
    InvalidDurationNumber {
        /// Rejected decimal token.
        input: String,
    },
    /// A decimal duration component exceeds the supported nanosecond range.
    #[error("{}", decimal_range_message(input))]
    DecimalDurationOutOfRange {
        /// Original decimal component.
        input: String,
    },
    /// The requested fixed duration exceeds Pandas' nanosecond range.
    #[error("{}", duration_range_message(&count.magnitude().to_string(), unit))]
    DurationOutOfRange {
        /// Requested count.
        count: BigInt,
        /// Requested unit.
        unit: String,
    },
}

impl FrequencyError {
    /// Python exception category represented by this frequency/duration failure.
    #[must_use]
    pub const fn python_exception_name(&self) -> &'static str {
        match self {
            Self::ClockIntegerOverflow => "OverflowError",
            Self::DurationOutOfRange { .. } | Self::DecimalDurationOutOfRange { .. } => {
                "OutOfBoundsDatetime"
            }
            _ => "ValueError",
        }
    }
}

fn decimal_range_message(input: &str) -> String {
    let boundary = input
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .unwrap_or(input.len());
    duration_range_message(&input[..boundary], &input[boundary..])
}

fn duration_range_message(number: &str, unit: &str) -> String {
    let value = number.parse::<f64>().unwrap_or(f64::INFINITY);
    let factor = Frequency::time_delta(BigInt::from(1), unit)
        .ok()
        .and_then(|value| value.num_nanoseconds());
    let canonical = match factor {
        Some(604_800_000_000_000) => "W",
        Some(86_400_000_000_000) => "D",
        Some(3_600_000_000_000) => "h",
        Some(60_000_000_000) => "m",
        Some(1_000_000_000) => "s",
        Some(1_000_000) => "ms",
        Some(1_000) => "us",
        Some(1) => "ns",
        _ => unit,
    };
    format!(
        "cannot convert input {} with the unit '{canonical}'",
        rustpython_literal::float::to_string(value)
    )
}

/// Duration conversion with the source's observable warning, including on failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DurationConversion {
    /// Converted duration or conversion failure.
    pub result: Result<TimeDelta, FrequencyError>,
    /// Pandas `FutureWarning` text for deprecated single-unit spellings.
    pub future_warning: Option<String>,
}

/// Compound duration conversion preserving ordered warnings, also on failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompoundDurationConversion {
    /// Converted duration or failure; `None` represents Pandas `NaT`.
    pub result: Result<Option<TimeDelta>, FrequencyError>,
    /// Ordered Pandas `FutureWarning` messages.
    pub future_warnings: Vec<String>,
}

impl Frequency {
    /// Convert integer unit sequences in Qlib's concatenated duration text.
    ///
    /// Supports unit sequences and clock suffixes, including fractional quantities.
    #[must_use]
    pub fn compound_time_delta(count: &BigInt, suffix: &str) -> CompoundDurationConversion {
        let mut warnings = Vec::new();
        let result = parse_integer_duration(count, suffix, &mut warnings);
        CompoundDurationConversion {
            result,
            future_warnings: warnings,
        }
    }

    /// Convert a single-unit duration while retaining Pandas deprecation diagnostics.
    ///
    /// Warning filters are left to the caller. The warning survives range failures.
    #[must_use]
    pub fn time_delta_with_warnings(count: BigInt, unit: &str) -> DurationConversion {
        let replacement = match unit {
            "H" => Some("h"),
            "S" => Some("s"),
            "t" | "T" => Some("min"),
            "l" | "L" => Some("ms"),
            "u" | "U" => Some("us"),
            "n" | "N" => Some("ns"),
            _ => None,
        };
        let future_warning = replacement.map(|replacement| {
            format!("'{unit}' is deprecated and will be removed in a future version. Please use '{replacement}' instead of '{unit}'.")
        });
        DurationConversion {
            result: Self::time_delta(count, unit),
            future_warning,
        }
    }

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
    /// Pandas Timedelta, [`FrequencyError::AmbiguousDurationUnit`] for month/year spellings,
    /// or [`FrequencyError::DurationOutOfRange`] outside Pandas' nanosecond range.
    pub fn time_delta(count: BigInt, unit: &str) -> Result<TimeDelta, FrequencyError> {
        if matches!(unit, "M" | "Y" | "y") {
            return Err(FrequencyError::AmbiguousDurationUnit);
        }
        let normalized = unit.to_ascii_lowercase();
        let nanos_per_unit = match normalized.as_str() {
            "w" => 604_800_000_000_000_i64,
            "day" | "days" | "d" => 86_400_000_000_000,
            "h" | "hr" | "hour" | "hours" => 3_600_000_000_000,
            "m" | "min" | "minute" | "minutes" | "t" => 60_000_000_000,
            "s" | "sec" | "second" | "seconds" => 1_000_000_000,
            "ms" | "millisecond" | "milliseconds" | "milli" | "millis" | "l" => 1_000_000,
            "us" | "µs" | "microsecond" | "microseconds" | "micro" | "micros" | "u" => 1_000,
            "" | "ns" | "nanosecond" | "nanoseconds" | "nano" | "nanos" | "n" => 1,
            _ => {
                return Err(FrequencyError::UnsupportedDurationUnit {
                    unit: unit.to_owned(),
                });
            }
        };
        // Pandas' string-duration parser converts the unsigned number through a double,
        // then applies the leading sign after unit conversion. Preserve its rounding at 2^53.
        let negative = count.sign() == Sign::Minus;
        let nanos = count
            .magnitude()
            .to_f64()
            .and_then(|value| value.to_i64())
            .and_then(|value| value.checked_mul(nanos_per_unit))
            .ok_or_else(|| FrequencyError::DurationOutOfRange {
                count,
                unit: unit.to_owned(),
            })?;
        Ok(TimeDelta::nanoseconds(if negative {
            -nanos
        } else {
            nanos
        }))
    }
}

fn parse_integer_duration(
    count: &BigInt,
    suffix: &str,
    warnings: &mut Vec<String>,
) -> Result<Option<TimeDelta>, FrequencyError> {
    let text = format!("{}{suffix}", count.magnitude());
    let mut number = String::new();
    let mut unit = String::new();
    let mut total = 0_i64;
    let mut have_unit = false;
    let mut negative = count.sign() == Sign::Minus;
    let mut clock_fields = 0_u8;
    let mut fraction = None::<String>;
    let mut have_dot = false;
    for character in text.chars() {
        match character {
            ' ' | ',' | '+' => (),
            '-' => {
                if negative || have_unit || clock_fields != 0 {
                    return Err(FrequencyError::InvalidDurationText {
                        message: "only leading negative signs are allowed",
                    });
                }
                negative = true;
            }
            '0'..='9' => {
                let value = duration_digit(
                    character,
                    &mut number,
                    &mut unit,
                    &mut fraction,
                    &mut have_dot,
                    warnings,
                )?;
                total = total.wrapping_add(signed_duration(value, negative));
            }
            ':' => {
                if have_unit {
                    negative = false;
                }
                if number.is_empty() {
                    return Err(FrequencyError::InvalidClockText {
                        input: format!("{count}{suffix}"),
                    });
                }
                let multiplier = match clock_fields {
                    0 => 3_600_000_000_000,
                    1 => 60_000_000_000,
                    _ => 1_000_000_000,
                };
                let value = clock_component(&number, multiplier)?;
                total = total.wrapping_add(signed_duration(value, negative));
                clock_fields = clock_fields.saturating_add(1);
                number.clear();
                unit.clear();
            }
            '.' if clock_fields != 0 && !number.is_empty() => {
                if clock_fields != 2 {
                    return Err(FrequencyError::InvalidDurationText {
                        message: "expected hh:mm:ss format before .",
                    });
                }
                let value = clock_component(&number, 1_000_000_000)?;
                total = total.wrapping_add(signed_duration(value, negative));
                have_unit = true;
                number.clear();
                unit.clear();
                fraction = Some(String::new());
                have_dot = true;
            }
            '.' => {
                fraction.get_or_insert_with(String::new);
                have_dot = true;
            }
            _ => {
                unit.push(character);
                have_unit = true;
                have_dot = false;
            }
        }
    }
    let value = finish_duration(
        &number,
        fraction.as_deref(),
        &unit,
        clock_fields,
        have_unit,
        have_dot,
        warnings,
    )?;
    total = total.wrapping_add(signed_duration(value, negative));
    Ok((total != i64::MIN).then(|| TimeDelta::nanoseconds(total)))
}

fn finish_duration(
    number: &str,
    fraction: Option<&str>,
    unit: &str,
    clock_fields: u8,
    have_unit: bool,
    have_dot: bool,
    warnings: &mut Vec<String>,
) -> Result<i64, FrequencyError> {
    if have_dot && !unit.is_empty() {
        return duration_component(number, fraction, unit, warnings);
    }
    if have_dot && clock_fields == 0 {
        return Err(FrequencyError::InvalidDurationText {
            message: "no units specified",
        });
    }
    if have_dot {
        return finish_clock("", fraction, 2);
    }
    if clock_fields != 0 {
        return finish_clock(number, None, clock_fields);
    }
    if unit.is_empty() && have_unit {
        return Err(FrequencyError::InvalidDurationText {
            message: "have leftover units",
        });
    }
    duration_component(number, fraction, unit, warnings)
}

fn duration_digit(
    digit: char,
    number: &mut String,
    unit: &mut String,
    fraction: &mut Option<String>,
    have_dot: &mut bool,
    warnings: &mut Vec<String>,
) -> Result<i64, FrequencyError> {
    if *have_dot {
        if unit.is_empty() {
            fraction.get_or_insert_with(String::new).push(digit);
        } else {
            number.push(digit);
            *have_dot = false;
        }
        return Ok(0);
    }
    let value = if unit.is_empty() {
        0
    } else {
        let value = duration_component(number, fraction.as_deref(), unit, warnings)?;
        number.clear();
        unit.clear();
        *fraction = None;
        value
    };
    number.push(digit);
    Ok(value)
}

fn duration_component(
    number: &str,
    fraction: Option<&str>,
    unit: &str,
    warnings: &mut Vec<String>,
) -> Result<i64, FrequencyError> {
    if fraction.is_none() && !number.is_empty() {
        let count = number
            .parse()
            .expect("nonempty token contains only ASCII digits");
        return integer_duration_component(count, unit, warnings);
    }
    let fraction = fraction.unwrap_or("");
    // Validate the unit (and emit its warning) before attempting magnitude conversion.
    let multiplier = integer_duration_component(BigInt::from(1), unit, warnings)?;
    let input = format!("{number}.{fraction}");
    let value: f64 = input
        .parse()
        .map_err(|_| FrequencyError::InvalidDurationNumber {
            input: input.clone(),
        })?;
    let range_error = || FrequencyError::DecimalDurationOutOfRange {
        input: format!("{input}{unit}"),
    };
    let base = value.to_i64().ok_or_else(&range_error)?;
    let mut residual = value - value.trunc();
    let precision = usize::try_from(multiplier.ilog10()).expect("duration precision fits usize");
    if precision != 0 {
        // Decimal formatting rounds the exact binary fraction, rather than a scaled float.
        residual = format!("{residual:.precision$}")
            .parse()
            .expect("formatted float parses");
    }
    // A finite, nonnegative input already passed to_i64. Its rounded residual is
    // in [0, 1], and the largest unit multiplier is 604_800_000_000_000 (< i64::MAX).
    let fractional_nanos = (residual * multiplier.to_f64().expect("i64 fits f64 range"))
        .to_i64()
        .expect("bounded fractional nanoseconds fit i64");
    base.checked_mul(multiplier)
        .map(|whole| whole.wrapping_add(fractional_nanos))
        .ok_or_else(range_error)
}

fn signed_duration(value: i64, negative: bool) -> i64 {
    if negative {
        value.wrapping_neg()
    } else {
        value
    }
}

fn clock_component(number: &str, multiplier: i64) -> Result<i64, FrequencyError> {
    if number.is_empty() {
        return Err(FrequencyError::InvalidDurationText {
            message: "invalid literal for int() with base 10: ''",
        });
    }
    number
        .parse::<i64>()
        .map(|value| value.wrapping_mul(multiplier))
        .map_err(|_| FrequencyError::ClockIntegerOverflow)
}

fn finish_clock(number: &str, fraction: Option<&str>, fields: u8) -> Result<i64, FrequencyError> {
    if fields != 2 {
        return Err(FrequencyError::InvalidDurationText {
            message: "expected hh:mm:ss format",
        });
    }
    let mut value = if number.is_empty() && fraction.is_some() {
        0
    } else {
        clock_component(number, 1_000_000_000)?
    };
    if let Some(fraction) = fraction {
        let digits: String = fraction.chars().take(9).collect();
        let nanos = clock_component(
            &digits,
            10_i64.pow(9 - u32::try_from(digits.len()).expect("at most nine digits")),
        )?;
        value = value.wrapping_add(nanos);
    }
    Ok(value)
}

fn integer_duration_component(
    count: BigInt,
    unit: &str,
    warnings: &mut Vec<String>,
) -> Result<i64, FrequencyError> {
    let outcome = Frequency::time_delta_with_warnings(count, unit);
    warnings.extend(outcome.future_warning);
    outcome.result.map(|value| {
        value
            .num_nanoseconds()
            .expect("duration is bounded to i64 nanoseconds")
    })
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
