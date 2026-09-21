//! Native stages of the Pandas timestamp text constructor used by time utilities.
//!
//! The ISO stage is not a complete constructor: `None` means the general parser
//! must still be tried, not that Pandas rejects the input.

use std::sync::LazyLock;

use arrow_schema::TimeUnit;
use chrono::{Datelike, Months, NaiveDate, NaiveDateTime, NaiveTime, Timelike};
use regex::Regex;
use thiserror::Error;

use crate::dataframe_append::{BuiltinFrameValue, TemporalFrameValue};

static ISO_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"\A(?P<year>[0-9]{4})(?P<ym_sep>[-./\\ ])(?P<month>[0-9]{1,2})",
        r"(?P<md_sep>[-./\\ ])(?P<day>[0-9]{1,2})",
        r"(?:[ T](?P<hour>[0-9]{1,2})(?:(?P<hm_sep>:?)(?P<minute>[0-9]{1,2})",
        r"(?:(?P<ms_sep>:?)(?P<second>[0-9]{1,2})(?:\.(?P<fraction>[0-9]{0,18}))?)?)?",
        r"[ \t\r\n\x0b\x0c]*(?P<zone>Z|[+-](?:[0-9]{1,2}:[0-9]{1,2}|[0-9]{1,4}))?)?\z"
    ))
    .expect("static ISO recognition pattern is valid")
});

static CLOCK_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"\A[ \t\r\n]*(?P<hour>[0-9]{1,2}):(?P<minute>[0-9]{1,2})",
        r"(?::(?P<second>[0-9]{1,2})(?:(?P<separator>[.,])(?P<fraction>[0-9]*))?)?",
        r"[ \t\r\n]*(?P<period>[aApP][mM])?[ \t\r\n]*\z"
    ))
    .expect("static clock recognition pattern is valid")
});

static NUMERIC_DATE_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"\A[ \t\r\n]*(?P<digits>[0-9]{4}|[0-9]{6}|[0-9]{8}|[0-9]{12}|[0-9]{14})",
        r"[ \t\r\n]*(?P<period>[aApP][mM])?[ \t\r\n]*\z"
    ))
    .expect("static numeric date recognition pattern is valid")
});

static DELIMITED_DATE_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"\A(?P<month>[0-9]{1,2})[ /.-](?P<day>[0-9]{1,2})[ /.-](?P<year>[0-9]{4})\z",
        r"|\A(?P<month_only>[0-9]{2})[ /-](?P<year_only>[0-9]{4})\z"
    ))
    .expect("static delimited date recognition pattern is valid")
});

static DATE_CLOCK_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"\A[ \t\r\n]*(?P<month>[0-9]{1,2})(?P<sep1>[ /.-])(?P<day>[0-9]{1,2})(?P<sep2>[ /.-])",
        r"(?P<year>[0-9]{4})(?:[ T](?P<clock>[0-9](?s:.*?)))?[ \t\r\n]*\z"
    ))
    .expect("static general delimited date recognition pattern is valid")
});

static YEAR_FIRST_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"\A[ \t\r\n]*(?P<year>[0-9]{4,})(?P<ym_sep>[-./\\ ])(?P<month>[0-9]{1,2})",
        r"(?P<md_sep>[-./\\ ])(?P<day>[0-9]{1,2})",
        r"(?:[ T](?P<clock>[0-9](?s:.*?)))?[ \t\r\n]*\z"
    ))
    .expect("static year-first date recognition pattern is valid")
});

static YEAR_MONTH_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"\A[ \t\r\n]*(?P<year>[0-9]{4,})(?P<sep>[-./\\ ])(?P<month>[0-9]{1,2})",
        r"(?:[ T](?P<clock>[0-9](?s:.*?)))?[ \t\r\n]*\z"
    ))
    .expect("static year-month recognition pattern is valid")
});

static HOUR_PERIOD_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\A[ \t\r\n]*(?P<hour>[0-9]{1,2})(?P<gap>[ \t]*)(?P<period>[aApP][mM])[ \t\r\n]*\z")
        .expect("static hour-period recognition pattern is valid")
});

static GENERAL_ZONE_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"\A(?P<clock>(?s:.*?))[ \t\r\n\x0b\x0c]*",
        r"(?P<zone>(?:UTC|GMT|Z|z)(?:[ \t]*[+-](?:[0-9]{1,2}:[0-9]{1,2}|[0-9]{1,4}))?",
        r"|[+-](?:[0-9]{1,2}:[0-9]{1,2}|[0-9]{1,4}))",
        r"[ \t\r\n\x0b\x0c]*\z"
    ))
    .expect("static numeric timezone pattern is valid")
});

static EXPLICIT_COMPACT_CLOCK_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"\A[ \t]*(?P<digits>[0-9]{2}(?:[0-9]{2}){0,2})(?:\.(?P<fraction>[0-9]*))?",
        r"[ \t]*(?P<period>[aApP][mM])?[ \t\r\n]*\z"
    ))
    .expect("static explicit compact clock pattern is valid")
});

static MIDDLE_MONTH_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"\A[ \t\r\n]*(?P<first>[0-9]{1,4})(?P<first_suffix>(?i-u:st|nd|rd|th))?(?P<sep1>[-/., ])(?P<month>[A-Za-z]+)",
        r"(?P<sep2>[-/., ])(?P<last>[0-9]{1,4})(?P<last_suffix>(?i-u:st|nd|rd|th))?(?:[ T](?P<clock>[0-9](?s:.*?)))?[ \t\r\n]*\z"
    ))
    .expect("static middle-month date pattern is valid")
});

static FOLLOWING_MONTH_NUMBER_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\A(?P<number>[0-9]{1,4})(?:[ T](?P<clock>[0-9](?s:.*?)))?\z")
        .expect("static following month number pattern is valid")
});

static FIRST_MONTH_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"\A[ \t\r\n]*(?P<month>[A-Za-z]+)(?P<sep1>[-/., ]| [oO][fF] )(?P<first>[0-9]{1,4})(?P<first_suffix>(?i-u:st|nd|rd|th))?",
        r"(?P<sep2>[-/., ]|, )(?P<last>[0-9]{1,4})(?P<last_suffix>(?i-u:st|nd|rd|th))?(?:[ T](?P<clock>[0-9](?s:.*?)))?[ \t\r\n]*\z"
    ))
    .expect("static first-month date pattern is valid")
});

static LAST_MONTH_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"\A[ \t\r\n]*(?P<first>[0-9]{1,4})(?P<first_suffix>(?i-u:st|nd|rd|th))?(?P<sep1>[-/., ])(?P<last>[0-9]{1,4})(?P<last_suffix>(?i-u:st|nd|rd|th))?",
        r"(?P<sep2>[-/., ])(?P<month>[A-Za-z]+)(?:[ T](?P<clock>[0-9](?s:.*?)))?[ \t\r\n]*\z"
    ))
    .expect("static last-month date pattern is valid")
});

static PARTIAL_FIRST_MONTH_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"\A[ \t\r\n]*(?P<month>[A-Za-z]+)(?:(?P<sep>[-/., ]| [oO][fF] )(?P<number>[0-9]{1,4})(?P<number_suffix>(?i-u:st|nd|rd|th))?)?",
        r"(?:[ T](?P<clock>[0-9](?s:.*?)))?[ \t\r\n]*\z"
    ))
    .expect("static partial first-month pattern is valid")
});

static PARTIAL_LAST_MONTH_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"\A[ \t\r\n]*(?P<number>[0-9]{1,4})(?P<number_suffix>(?i-u:st|nd|rd|th))?(?P<sep>[-/., ])(?P<month>[A-Za-z]+)",
        r"(?:[ T](?P<clock>[0-9](?s:.*?)))?[ \t\r\n]*\z"
    ))
    .expect("static partial last-month pattern is valid")
});

static MONTH_DATE_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"\A[ \t\r\n]*(?P<month>[A-Za-z]+)[ \t]+(?P<day>[0-9]{1,2}),?[ \t]+",
        r"(?P<year>[0-9]{4})(?:[ T](?P<clock>[0-9](?s:.*?)))?[ \t\r\n]*\z"
    ))
    .expect("static month-name date recognition pattern is valid")
});

static NAMED_ZONE_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"\A(?P<prefix>(?s:.*?))(?P<name>[A-Za-z]+)",
        r"(?P<offset>[ \t]*[+-](?:[0-9]{1,2}:[0-9]{1,2}|[0-9]{1,4}))?[ \t\r\n]*\z"
    ))
    .expect("static named timezone recognition pattern is valid")
});

/// A recognized timestamp whose requested precision cannot represent the value.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("{message}")]
pub struct TimestampTextBoundsError {
    /// Exact source `OutOfBoundsDatetime` message.
    pub message: String,
}

/// Source error category retained across the implemented constructor stages.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum TimestampTextError {
    /// General timezone conversion cannot fit its resulting integer ticks.
    #[error("int too big to convert")]
    OffsetOverflow,
    /// General parsing produced an offset outside the Python timezone bounds.
    #[error("{message}")]
    InvalidOffset { message: String },
    /// The source's fast calendar constructor raises a plain `ValueError`.
    #[error("{message}")]
    InvalidCalendar { message: String },
    /// Source rejects a numeric spelling before general date parsing.
    #[error("{message}")]
    NotDateLike { message: String },
    /// Timestamp precision or Python datetime argument conversion overflow.
    #[error(transparent)]
    OutOfBounds(#[from] TimestampTextBoundsError),
    /// Recognized general-parser input cannot form a valid datetime.
    #[error("{message}")]
    DateParse { message: String },
}

/// Constructor state whose visible fields need not equal its stored tick value.
///
/// Pandas' general parser can retain microseconds while choosing second storage.
/// Field-based consumers must use `local_datetime`, not reconstruct it from ticks.
#[derive(Clone, Debug, PartialEq)]
pub struct ParsedTimestampText {
    /// Source integer storage value (UTC ticks for an aware timestamp).
    pub ticks: i64,
    /// Source inferred storage resolution.
    pub unit: TimeUnit,
    /// Original source timezone label, absent for naive timestamps.
    pub timezone: Option<String>,
    /// Source local calendar fields, including subsecond data retained separately.
    pub local_datetime: NaiveDateTime,
}

impl ParsedTimestampText {
    /// Materialize the shared temporal value only at a consumer boundary.
    /// A parsed timestamp cannot contain a duration or a missing-value variant.
    #[must_use]
    pub fn to_temporal(&self) -> TemporalFrameValue {
        TemporalFrameValue::Timestamp {
            ticks: self.ticks,
            unit: self.unit,
            timezone: self.timezone.as_deref().map(Into::into),
        }
    }

    /// Source `.time()` result, retaining visible microseconds but not nanoseconds.
    #[must_use]
    pub fn time(&self) -> NaiveTime {
        crate::time_compat::python_clock_precision(self.local_datetime.time())
    }
}

// Internal clock materialization is always a timestamp, never another temporal
// union variant. Keep this invariant in the type until the public boundary.
struct ParsedClock {
    ticks: i64,
    unit: TimeUnit,
    local_datetime: NaiveDateTime,
}

impl ParsedClock {
    fn into_timestamp(self, timezone: Option<String>) -> ParsedTimestampText {
        ParsedTimestampText {
            ticks: self.ticks,
            unit: self.unit,
            timezone,
            local_datetime: self.local_datetime,
        }
    }
}

/// Result of a recognized native timestamp construction, including missingness.
#[derive(Clone, Debug, PartialEq)]
pub enum TimestampTextValue {
    /// Timestamp storage and independently retained calendar fields.
    Timestamp(ParsedTimestampText),
    /// Source `NaT`, without fabricated calendar fields.
    NotATime,
}

impl TimestampTextValue {
    /// Extract the source microsecond clock without rebuilding its calendar fields.
    ///
    /// # Errors
    ///
    /// Returns the source `.time()` failure for `NaT`.
    pub fn time(&self) -> Result<NaiveTime, crate::time_compat::TimeCompatError> {
        match self {
            Self::Timestamp(value) => Ok(value.time()),
            Self::NotATime => Err(crate::time_compat::TimeCompatError::NaTDoesNotSupportTime),
        }
    }

    fn into_temporal(self) -> TemporalFrameValue {
        match self {
            Self::Timestamp(value) => value.to_temporal(),
            Self::NotATime => TemporalFrameValue::Builtin(BuiltinFrameValue::NotATime),
        }
    }
}

/// Independent clock and parser-initialization state for text construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimestampTextContext {
    /// Current local date used by source clock-only prefix handling.
    pub reference_date: NaiveDate,
    /// Only the year is used, matching dateutil's captured parserinfo year.
    pub parser_initialized_date: NaiveDate,
}

impl From<NaiveDate> for TimestampTextContext {
    /// Convenience for a parser initialized in the reference date's year.
    fn from(reference_date: NaiveDate) -> Self {
        Self {
            reference_date,
            parser_initialized_date: reference_date,
        }
    }
}

/// Run the implemented constructor stages in source order.
///
/// Exact missing markers precede ISO, naive clock and single numeric date-token stages.
/// `None` explicitly requests the still-required general parser; it is not an
/// invalid-input result. Pass explicit context for a long-lived parser; a bare
/// date assumes parser initialization and clock reference share the same year.
///
/// # Errors
///
/// Recognized timestamps retain their source bounds or general date-parse failure.
///
/// # Panics
///
/// Only on violated static parser invariants documented by the underlying stages.
#[must_use]
pub fn parse_timestamp_text_compatible(
    text: &str,
    context: impl Into<TimestampTextContext>,
) -> Option<Result<TimestampTextValue, TimestampTextError>> {
    let context = context.into();
    if matches!(text, "" | "NaT" | "nat" | "NAT" | "nan" | "NaN" | "NAN") {
        return Some(Ok(TimestampTextValue::NotATime));
    }
    parse_iso_timestamp_with_fields(text)
        .map(|result| result.map_err(TimestampTextError::from))
        .or_else(|| parse_year_first_date(text, context))
        .or_else(|| parse_year_month(text, context))
        .or_else(|| parse_numeric_date(text, context))
        .or_else(|| parse_delimited_date(text))
        .or_else(|| parse_date_clock(text, context))
        .or_else(|| parse_month_date(text, context))
        .or_else(|| parse_named_month_composition(text, context))
        .or_else(|| parse_partial_month_date(text, context))
        .or_else(|| {
            parse_clock_timestamp_compatible(text, context.reference_date).map(|result| {
                result
                    .map(TimestampTextValue::Timestamp)
                    .map_err(TimestampTextError::from)
            })
        })
        .or_else(|| parse_clock_date_token(text, context))
        .or_else(|| parse_zoned_clock(text, context))
        .or_else(|| parse_clock_failure(text, context.reference_date))
}

/// Caller-owned runtime names used when distinguishing unknown from local zones.
#[derive(Clone, Copy, Debug)]
pub struct TimestampTextZoneContext<'a> {
    /// Date defaults and parser initialization year.
    pub timestamp: TimestampTextContext,
    /// The source runtime's local standard/daylight timezone names.
    pub local_timezone_names: &'a [&'a str],
}

/// A constructor result together with warnings emitted before success or failure.
#[derive(Debug)]
pub struct TimestampTextOutcome {
    /// `None` retains the existing not-yet-implemented parser contract.
    pub result: Option<Result<TimestampTextValue, TimestampTextError>>,
    /// Ordered source `FutureWarning` messages, independent of the result.
    pub future_warnings: Vec<String>,
}

/// Parse timestamps without dropping unknown-timezone warnings.
///
/// Host-local names remain pending until their runtime offset/DST semantics are
/// supplied; they are never silently treated as unknown. Other unsupported syntax
/// retains `None`. The original warning-free entry is unchanged.
///
/// # Panics
/// Only if static parser capture invariants are violated.
#[must_use]
pub fn parse_timestamp_text_with_warnings(
    text: &str,
    context: TimestampTextZoneContext<'_>,
) -> TimestampTextOutcome {
    let mut outcome = TimestampTextOutcome {
        result: parse_timestamp_text_compatible(text, context.timestamp),
        future_warnings: Vec::new(),
    };
    let Some(captures) = NAMED_ZONE_PATTERN.captures(text) else {
        return outcome;
    };
    let name = captures.name("name").expect("required name");
    if context.local_timezone_names.contains(&name.as_str()) {
        outcome.result = None;
        return outcome;
    }
    if outcome.result.is_some() {
        return outcome;
    }
    let normalized = format!("{}UTC{}", &text[..name.start()], &text[name.end()..]);
    let Some(mut result) = parse_timestamp_text_compatible(&normalized, context.timestamp) else {
        return outcome;
    };
    if name.as_str().len() > 5 || !name.as_str().bytes().all(|byte| byte.is_ascii_uppercase()) {
        outcome.result = Some(Err(TimestampTextError::DateParse {
            message: format!("Unknown datetime string format, unable to parse: {text}"),
        }));
        return outcome;
    }
    let explicit_offset = captures.name("offset").is_some();
    if !explicit_offset
        && (result.is_ok()
            || matches!(&result,
        Err(TimestampTextError::OutOfBounds(error)) if error.message == "Out of bounds nanosecond timestamp"))
    {
        outcome.future_warnings.push(format!(
            "Parsed string \"{text}\" included an un-recognized timezone \"{}\". Dropping unrecognized timezones is deprecated; in a future version this will raise. Instead pass the string without the timezone, then use .tz_localize to convert to a recognized timezone.", name.as_str()
        ));
    }
    if let Ok(TimestampTextValue::Timestamp(value)) = &mut result {
        value.timezone = if explicit_offset {
            value
                .timezone
                .take()
                .map(|label| label.replace("None", &format!("'{}'", name.as_str())))
        } else {
            None
        };
    }
    outcome.result =
        Some(result.map_err(|error| restore_named_zone_error(error, &normalized, text)));
    outcome
}

fn restore_named_zone_error(
    mut error: TimestampTextError,
    normalized: &str,
    original: &str,
) -> TimestampTextError {
    match &mut error {
        TimestampTextError::DateParse { message }
        | TimestampTextError::InvalidOffset { message } => {
            *message = message.replace(normalized, original);
        }
        TimestampTextError::OutOfBounds(error) => {
            error.message = error.message.replace(normalized, original);
        }
        _ => {}
    }
    error
}

fn parse_year_first_date(
    text: &str,
    context: TimestampTextContext,
) -> Option<Result<TimestampTextValue, TimestampTextError>> {
    let captures = YEAR_FIRST_PATTERN.captures(text)?;
    if matches!(captures["year"].len(), 6 | 8 | 12 | 14) {
        return Some(finish_compact_following(text, &captures, context));
    }
    if captures["ym_sep"] != captures["md_sep"]
        && !(matches!(&captures["ym_sep"], "-" | "/") && &captures["md_sep"] == " ")
    {
        return None; // Mixed-separator token resolution still needs the general parser.
    }
    let unknown = || TimestampTextError::DateParse {
        message: format!("Unknown datetime string format, unable to parse: {text}"),
    };
    if &captures["ym_sep"] == "\\" {
        return Some(Err(unknown())); // Backslashes are accepted only by the ISO stage.
    }
    let parsed_year = captures["year"].parse::<i32>();
    let overflow = parsed_year.is_err();
    // Keep lexical validation ahead of Python's integer conversion failure.
    // A saturated invalid year retains calendar precedence until that stage.
    let mut year = u32::try_from(parsed_year.unwrap_or(i32::MAX))
        .expect("unsigned decimal year is nonnegative");
    let mut month = numeric_field(&captures, "month");
    let mut day = numeric_field(&captures, "day");
    let spaced = &captures["ym_sep"] == " ";
    if spaced && year <= 31 {
        if year > 12 {
            std::mem::swap(&mut year, &mut day);
        } else {
            (year, month, day) = (day, year, month);
        }
    }
    let year = i32::try_from(year).expect("bounded year or swapped day");
    let parts = ClockDateParts {
        // Whitespace-separated numbers lose the lexical century marker in
        // dateutil, unlike punctuation-separated four-digit year tokens.
        year: if spaced && year < 100 {
            expand_short_year(year, context)
        } else {
            year
        },
        month,
        day,
        hour: 0,
        minute: 0,
        second: 0,
    };
    let result = if let Some(clock) = captures.name("clock") {
        finish_explicit_date_clock(text, clock.as_str(), parts, true)
    } else {
        Some(finish_clock_date_parts(text, &parts).map(TimestampTextValue::Timestamp))
    }?;
    Some(restore_year_overflow(text, result, overflow))
}

fn parse_year_month(
    text: &str,
    context: TimestampTextContext,
) -> Option<Result<TimestampTextValue, TimestampTextError>> {
    let captures = YEAR_MONTH_PATTERN.captures(text)?;
    let digits = &captures["year"];
    if matches!(digits.len(), 6 | 8 | 12 | 14) && &captures["sep"] != "." {
        return Some(finish_compact_pair(text, &captures, context));
    }
    let parsed = digits.parse::<i32>();
    let overflow = parsed.is_err();
    let first = u32::try_from(parsed.unwrap_or(i32::MAX)).expect("nonnegative decimal year");
    let second = numeric_field(&captures, "month");
    let clock = captures.name("clock").map(|value| value.as_str());
    if digits.len() == 4 && clock.is_none() && !text.ends_with(|c: char| c.is_ascii_whitespace()) {
        let iso = format!("{}-{:02}-01", &captures["year"], second);
        if let Some(value) = parse_iso_timestamp_with_fields(&iso) {
            return Some(value.map_err(TimestampTextError::from));
        }
    }
    if &captures["sep"] == "." && clock.is_none() && !text.starts_with('0') && first < 1000 {
        return Some(Err(TimestampTextError::NotDateLike {
            message: format!("Given date string \"{text}\" not likely a datetime"),
        }));
    }
    if digits.len() == 6 {
        return Some(finish_compact_pair(text, &captures, context));
    }
    if &captures["sep"] == "\\" {
        return Some(Err(TimestampTextError::DateParse {
            message: format!("Unknown datetime string format, unable to parse: {text}"),
        }));
    }
    let mut parts = decimal_date_parts(first, second, None, context);
    match &captures["sep"] {
        "-" | "/" => {
            if first > 31 {
                parts.year = i32::try_from(first).expect("bounded year");
            } else if second > 31 {
                parts.year = i32::try_from(second).expect("two digits");
            }
        }
        "." => {
            parts = ClockDateParts {
                year: 1,
                month: 1,
                day: 1,
                hour: 0,
                minute: 0,
                second: 0,
            };
            if matches!(digits.len() + 1 + captures["month"].len(), 6 | 8 | 12 | 14) {
                return Some(Err(TimestampTextError::DateParse {
                    message: format!("Unknown datetime string format, unable to parse: {text}"),
                }));
            } else if first > 31 {
                let year = i32::try_from(first).expect("bounded year");
                parts.year = if year < 100 {
                    expand_short_year(year, context)
                } else {
                    year
                };
            } else {
                parts.day = first;
            }
        }
        _ => {}
    }
    let result = if let Some(clock) = clock {
        finish_explicit_date_clock(text, clock, parts, false).unwrap_or_else(|| {
            Err(TimestampTextError::DateParse {
                message: format!("Unknown datetime string format, unable to parse: {text}"),
            })
        })
    } else {
        finish_clock_date_parts(text, &parts).map(TimestampTextValue::Timestamp)
    };
    Some(restore_year_overflow(text, result, overflow))
}

fn preserves_clock_fraction(clock: Option<&str>) -> bool {
    clock.is_none_or(|clock| {
        let zone = general_zone_captures(clock);
        let raw = zone.as_ref().map_or(clock, |zone| &zone["clock"]);
        HOUR_PERIOD_PATTERN.is_match(raw)
            || CLOCK_PATTERN
                .captures(raw)
                .is_some_and(|clock| clock.name("second").is_none())
    })
}

fn finish_compact_pair(
    text: &str,
    captures: &regex::Captures<'_>,
    context: TimestampTextContext,
) -> Result<TimestampTextValue, TimestampTextError> {
    let token = &captures["year"];
    let separator = &captures["sep"];
    let unknown = || TimestampTextError::DateParse {
        message: format!("Unknown datetime string format, unable to parse: {text}"),
    };
    if token.len() == 8 && separator == " " {
        let normalized = format!(
            "{}-{}-{}{}",
            &token[..4],
            &token[4..6],
            &token[6..],
            &text[captures.name("year").expect("required token").end()..]
        );
        if let Some(value) = parse_iso_timestamp_with_fields(&normalized) {
            return value.map_err(TimestampTextError::from);
        }
    }
    let mut parts = ClockDateParts {
        year: 1,
        month: 1,
        day: 1,
        hour: 0,
        minute: 0,
        second: 0,
    };
    let offset = if separator == "." {
        parts.hour = token[..2].parse().expect("two digits");
        parts.minute = token[2..4].parse().expect("two digits");
        parts.second = token[4..].parse().expect("two digits");
        None
    } else {
        if separator == "\\" || (token.len() > 8 && separator != "-") {
            return Err(unknown());
        }
        apply_compact_date(token, context, &mut parts);
        let following = i32::try_from(numeric_field(captures, "month")).expect("two digits");
        if token.len() <= 8 {
            if captures["month"].len() != 2 {
                return Err(unknown());
            }
            parts.hour = following;
            None
        } else {
            Some(-following * 60)
        }
    };
    let mut value = if separator == "." {
        finish_decimal_pair_clock(text, captures, parts, None, None, context)?
    } else {
        finish_compact_clock(text, captures, parts, offset)?
    };
    if separator == "."
        && preserves_clock_fraction(captures.name("clock").map(|clock| clock.as_str()))
    {
        value.local_datetime = value
            .local_datetime
            .with_nanosecond(decimal_nanos(&captures["month"]))
            .expect("one or two fractional digits");
    }
    Ok(TimestampTextValue::Timestamp(value))
}

fn finish_decimal_pair_clock(
    text: &str,
    captures: &regex::Captures<'_>,
    mut parts: ClockDateParts,
    date_token: Option<u32>,
    offset: Option<i32>,
    context: TimestampTextContext,
) -> Result<ParsedTimestampText, TimestampTextError> {
    if let Some(clock) = captures.name("clock") {
        let zone = general_zone_captures(clock.as_str());
        let raw = zone.as_ref().map_or(clock.as_str(), |zone| &zone["clock"]);
        if let Some(period) = HOUR_PERIOD_PATTERN.captures(raw) {
            let day = numeric_field(&period, "hour");
            // An attached number too large to be an hour may append a date
            // field; AM/PM then adjusts the already parsed clock.
            if period["gap"].is_empty() && (24..=31).contains(&day) && parts.hour <= 12 {
                if let Some(first) = date_token {
                    let date = decimal_date_parts(first, day, None, context);
                    parts.year = date.year;
                    parts.month = date.month;
                    parts.day = date.day;
                } else {
                    parts.day = day;
                }
                parts.hour = adjust_period(parts.hour, &period["period"]);
                let offset = zone
                    .as_ref()
                    .map(|zone| general_offset_minutes(text, &zone["zone"]))
                    .transpose()?
                    .or(offset);
                let normalized =
                    format!("{:02}:{:02}:{:02}", parts.hour, parts.minute, parts.second);
                return finish_date_clock(text, &normalized, parts, offset)
                    .expect("generated clock has valid lexical shape");
            }
        }
    }
    finish_compact_clock(text, captures, parts, offset)
}

fn restore_year_overflow(
    text: &str,
    result: Result<TimestampTextValue, TimestampTextError>,
    overflow: bool,
) -> Result<TimestampTextValue, TimestampTextError> {
    if overflow
        && !matches!(&result, Err(TimestampTextError::DateParse { message })
            if message.starts_with("Unknown datetime string format"))
    {
        return Err(TimestampTextBoundsError {
            message: format!("Parsing \"{text}\" to datetime overflows"),
        }
        .into());
    }
    result
}

fn finish_compact_following(
    text: &str,
    captures: &regex::Captures<'_>,
    context: TimestampTextContext,
) -> Result<TimestampTextValue, TimestampTextError> {
    let unknown = || TimestampTextError::DateParse {
        message: format!("Unknown datetime string format, unable to parse: {text}"),
    };
    let token = &captures["year"];
    let separator = &captures["ym_sep"];
    if token.len() == 8 && separator == " " {
        let normalized = format!(
            "{}-{}-{}{}",
            &token[..4],
            &token[4..6],
            &token[6..],
            &text[captures.name("year").expect("required token").end()..]
        );
        if let Some(result) = parse_iso_timestamp_with_fields(&normalized) {
            return result.map_err(TimestampTextError::from);
        }
    }
    if token.len() == 6 && separator == "." {
        return finish_decimal_compact_clock(text, captures, context);
    }
    if separator == "." && &captures["md_sep"] == " " {
        // dateutil dispatches the full decimal token by lexical length first;
        // twelve digits plus a dot and one digit form a malformed compact14.
        if token.len() + 1 + captures["month"].len() == 14 {
            return Err(unknown());
        }
        let parsed = token.parse::<i32>();
        let overflow = parsed.is_err();
        let first = u32::try_from(parsed.unwrap_or(i32::MAX)).expect("nonnegative decimal");
        let parts = decimal_date_parts(first, numeric_field(captures, "day"), None, context);
        let result = if let Some(clock) = captures.name("clock") {
            finish_explicit_date_clock(text, clock.as_str(), parts, false)
                .unwrap_or_else(|| Err(unknown()))
        } else {
            finish_clock_date_parts(text, &parts).map(TimestampTextValue::Timestamp)
        };
        return restore_year_overflow(text, result, overflow);
    }
    if &captures["md_sep"] != "-"
        || !(separator == "-" || (token.len() <= 8 && matches!(separator, "/" | " ")))
    {
        return Err(unknown());
    }
    let mut parts = ClockDateParts {
        year: 1,
        month: 1,
        day: 1,
        hour: 0,
        minute: 0,
        second: 0,
    };
    apply_compact_date(token, context, &mut parts);
    if token.len() <= 8 {
        if captures["month"].len() != 2 {
            return Err(unknown());
        }
        parts.hour = i32::try_from(numeric_field(captures, "month")).expect("two digits");
    }
    let offset = -i32::try_from(numeric_field(captures, "day")).expect("two digits") * 60;
    finish_compact_clock(text, captures, parts, Some(offset)).map(TimestampTextValue::Timestamp)
}

fn finish_decimal_compact_clock(
    text: &str,
    captures: &regex::Captures<'_>,
    context: TimestampTextContext,
) -> Result<TimestampTextValue, TimestampTextError> {
    let token = &captures["year"];
    let mut parts = ClockDateParts {
        year: 1,
        month: 1,
        day: 1,
        hour: token[..2].parse().expect("two digits"),
        minute: token[2..4].parse().expect("two digits"),
        second: token[4..].parse().expect("two digits"),
    };
    let last = numeric_field(captures, "day");
    let offset = match &captures["md_sep"] {
        "-" => Some(-i32::try_from(last).expect("two digits") * 60),
        "/" | " " => {
            if last > 31 {
                parts.year = expand_short_year(i32::try_from(last).expect("two digits"), context);
            } else {
                parts.day = last;
            }
            None
        }
        _ => {
            return Err(TimestampTextError::DateParse {
                message: format!("Unknown datetime string format, unable to parse: {text}"),
            });
        }
    };
    let keep_fraction =
        preserves_clock_fraction(captures.name("clock").map(|clock| clock.as_str()));
    let date_token = matches!(&captures["md_sep"], "/" | " ").then_some(last);
    let mut result = finish_decimal_pair_clock(text, captures, parts, date_token, offset, context)?;
    if keep_fraction {
        result.local_datetime = result
            .local_datetime
            .with_nanosecond(decimal_nanos(&captures["month"]))
            .expect("one or two fractional digits");
    }
    Ok(TimestampTextValue::Timestamp(result))
}

fn finish_compact_clock(
    text: &str,
    captures: &regex::Captures<'_>,
    parts: ClockDateParts,
    offset: Option<i32>,
) -> Result<ParsedTimestampText, TimestampTextError> {
    let unknown = || TimestampTextError::DateParse {
        message: format!("Unknown datetime string format, unable to parse: {text}"),
    };
    let generated;
    let clock = if let Some(clock) = captures.name("clock") {
        clock.as_str()
    } else {
        generated = format!("{:02}:{:02}:{:02}", parts.hour, parts.minute, parts.second);
        &generated
    };
    if general_zone_captures(clock).is_some() {
        finish_zoned_date_clock(text, clock, parts).unwrap_or_else(|| Err(unknown()))
    } else {
        finish_date_clock(text, clock, parts, offset).unwrap_or_else(|| Err(unknown()))
    }
}

fn finish_explicit_date_clock(
    text: &str,
    clock: &str,
    parts: ClockDateParts,
    complete_date: bool,
) -> Option<Result<TimestampTextValue, TimestampTextError>> {
    // With three date fields dateutil consumes a two-digit hour before it
    // encounters AM/PM; a partial date consumes the hour and period together.
    let zone = general_zone_captures(clock);
    let raw = zone.as_ref().map_or(clock, |zone| &zone["clock"]);
    if complete_date
        && HOUR_PERIOD_PATTERN
            .captures(raw)
            .is_some_and(|captures| numeric_field(&captures, "hour") > 12)
    {
        return Some(Err(TimestampTextError::DateParse {
            message: format!("Unknown datetime string format, unable to parse: {text}"),
        }));
    }
    if let Some(compact) = EXPLICIT_COMPACT_CLOCK_PATTERN.captures(raw) {
        if complete_date || compact["digits"].len() == 6 {
            return Some(
                finish_explicit_compact_clock(text, &compact, parts, zone.as_ref())
                    .map(TimestampTextValue::Timestamp),
            );
        }
    }
    let result = if CLOCK_PATTERN.is_match(clock) || HOUR_PERIOD_PATTERN.is_match(clock) {
        // An extra comma date token after a complete date is a lexical failure.
        Some(
            finish_date_clock(text, clock, parts, None).unwrap_or_else(|| {
                Err(TimestampTextError::DateParse {
                    message: format!("Unknown datetime string format, unable to parse: {text}"),
                })
            }),
        )
    } else {
        finish_zoned_date_clock(text, clock, parts)
    };
    result.map(|value| value.map(TimestampTextValue::Timestamp))
}

fn finish_explicit_compact_clock(
    text: &str,
    compact: &regex::Captures<'_>,
    parts: ClockDateParts,
    zone: Option<&regex::Captures<'_>>,
) -> Result<ParsedTimestampText, TimestampTextError> {
    let unknown = || TimestampTextError::DateParse {
        message: format!("Unknown datetime string format, unable to parse: {text}"),
    };
    let digits = &compact["digits"];
    let fraction = compact.name("fraction");
    if digits.len() != 6 && fraction.is_some() {
        return Err(unknown());
    }
    let mut normalized = match digits.len() {
        2 => format!("{digits}:00"),
        4 => format!("{}:{}", &digits[..2], &digits[2..]),
        _ => format!("{}:{}:{}", &digits[..2], &digits[2..4], &digits[4..]),
    };
    if let Some(fraction) = fraction {
        // Compact fractions have no colon-based precision recovery in Pandas.
        // The existing two-digit-second comma path has the same storage/local
        // microsecond distinction, so reuse it rather than recompute timestamps.
        normalized.push(',');
        normalized.push_str(fraction.as_str());
    }
    if let Some(period) = compact.name("period") {
        normalized.push_str(period.as_str());
    }
    let offset = zone
        .map(|zone| general_offset_minutes(text, &zone["zone"]))
        .transpose()?;
    finish_date_clock(text, &normalized, parts, offset)
        .expect("normalized compact clock has two-digit fields and no date tokens")
}

fn parse_zoned_clock(
    text: &str,
    context: TimestampTextContext,
) -> Option<Result<TimestampTextValue, TimestampTextError>> {
    let zone = general_zone_captures(text)?;
    if HOUR_PERIOD_PATTERN.is_match(&zone["clock"]) {
        return finish_zoned_date_clock(
            text,
            text,
            ClockDateParts {
                year: 1,
                month: 1,
                day: 1,
                hour: 0,
                minute: 0,
                second: 0,
            },
        )
        .map(|value| value.map(TimestampTextValue::Timestamp));
    }
    let clock = CLOCK_PATTERN.captures(&zone["clock"])?;
    let date = clock_reference_date(&clock, context.reference_date)?;
    if clock
        .name("separator")
        .is_some_and(|value| value.as_str() == ",")
        && clock["second"].len() == 1
        && !clock["fraction"].is_empty()
    {
        return Some(finish_zoned_date_token(text, &zone, &clock, context));
    }
    finish_zoned_date_clock(
        text,
        text,
        ClockDateParts {
            year: date.year(),
            month: date.month(),
            day: date.day(),
            hour: 0,
            minute: 0,
            second: 0,
        },
    )
    .map(|value| value.map(TimestampTextValue::Timestamp))
}

fn finish_zoned_date_clock(
    text: &str,
    clock: &str,
    parts: ClockDateParts,
) -> Option<Result<ParsedTimestampText, TimestampTextError>> {
    let captures = general_zone_captures(clock)?;
    let offset = match general_offset_minutes(text, &captures["zone"]) {
        Ok(offset) => offset,
        Err(error) => return Some(Err(error)),
    };
    let raw_clock = captures["clock"].trim_end_matches(|c: char| c.is_ascii_whitespace());
    let normalized;
    let clock = if raw_clock.bytes().all(|c| c.is_ascii_digit()) {
        // Fresh complete dates consume 2/4-digit clocks in the explicit-date
        // finisher. Here an existing compact clock may only be replaced by a
        // six-digit clock; a short numeric token is a date token, not an hour.
        if raw_clock.len() != 6 {
            return None;
        }
        normalized = format!(
            "{}:{}:{}",
            &raw_clock[..2],
            &raw_clock[2..4],
            &raw_clock[4..]
        );
        normalized.as_str()
    } else {
        raw_clock
    };
    finish_date_clock(text, clock, parts, Some(offset))
}

fn general_zone_captures(text: &str) -> Option<regex::Captures<'_>> {
    let captures = GENERAL_ZONE_PATTERN.captures(text)?;
    let zone = captures.name("zone").expect("required zone");
    if zone.as_str().as_bytes()[0].is_ascii_alphabetic()
        && text.as_bytes()[..zone.start()]
            .last()
            .is_some_and(u8::is_ascii_alphabetic)
    {
        return None; // PMUTC/XYZ are whole name tokens, not a suffix UTC/Z alias.
    }
    Some(captures)
}

fn general_offset_minutes(text: &str, zone: &str) -> Result<i32, TimestampTextError> {
    if let Some(rest) = ["UTC", "GMT", "Z", "z"]
        .iter()
        .find_map(|name| zone.strip_prefix(name))
    {
        let rest = rest.trim_start_matches(|c: char| c.is_ascii_whitespace());
        return if rest.is_empty() {
            Ok(0)
        } else {
            // dateutil interprets UTC+N/GMT+N as zones N hours behind UTC.
            general_offset_minutes(text, rest).map(|offset| -offset)
        };
    }
    let digits = &zone[1..];
    if digits.len() == 3 && !digits.contains(':') {
        return Err(TimestampTextError::DateParse {
            message: format!("Unknown datetime string format, unable to parse: {text}"),
        });
    }
    let (hours, minutes) = digits
        .split_once(':')
        .unwrap_or_else(|| digits.split_at(digits.len().min(2)));
    let minutes = if minutes.is_empty() {
        0
    } else {
        minutes.parse::<i32>().expect("bounded digits")
    };
    Ok(
        (hours.parse::<i32>().expect("bounded digits") * 60 + minutes)
            * if zone.starts_with('-') { -1 } else { 1 },
    )
}

fn finish_zoned_date_token(
    text: &str,
    zone: &regex::Captures<'_>,
    clock: &regex::Captures<'_>,
    context: TimestampTextContext,
) -> Result<TimestampTextValue, TimestampTextError> {
    let token = clock.name("fraction").expect("nonempty date token");
    let attached = zone.name("zone").expect("required zone").start() == token.end();
    let compact = matches!(token.as_str().len(), 6 | 8 | 12 | 14);
    if attached && !compact && zone["zone"].starts_with('-') {
        return finish_clock_hyphen_date(text, clock, &zone["zone"][1..], context);
    }
    let offset = general_offset_minutes(text, &zone["zone"])?;
    if attached
        && !compact
        && token
            .as_str()
            .parse::<u32>()
            .map_or(true, |number| !(1..=31).contains(&number))
    {
        return Err(TimestampTextError::DateParse {
            message: format!("Unknown datetime string format, unable to parse: {text}"),
        });
    }
    let value = interpret_clock_date_token(text, clock, context)?;
    validate_general_offset(text, offset)?;
    // Comma date tokens have integral-second precision; retain their local
    // calendar while reusing the shared checked UTC conversion.
    apply_general_offset(
        ParsedClock {
            ticks: value.local_datetime.and_utc().timestamp(),
            unit: TimeUnit::Second,
            local_datetime: value.local_datetime,
        },
        offset,
    )
    .map(TimestampTextValue::Timestamp)
}

fn finish_clock_hyphen_date(
    text: &str,
    clock: &regex::Captures<'_>,
    second: &str,
    context: TimestampTextContext,
) -> Result<TimestampTextValue, TimestampTextError> {
    let first = &clock["fraction"];
    if second.contains(':') || (first.len() > 2 && second.len() > 2) {
        return Err(TimestampTextError::DateParse {
            message: format!("Unknown datetime string format, unable to parse: {text}"),
        });
    }
    let first_value = first.parse::<i32>().map_err(|_| TimestampTextBoundsError {
        message: format!("Parsing \"{text}\" to datetime overflows"),
    })?;
    let second_value = second.parse::<u32>().expect("four digits");
    let default = clock_reference_date(clock, context.reference_date).expect("validated reference");
    let (mut year, month, mut day, explicit_year) = if first_value > 31 {
        (first_value, second_value, default.day(), true)
    } else if second_value > 31 {
        (
            i32::try_from(second_value).expect("four digits"),
            u32::try_from(first_value).expect("nonnegative digits"),
            default.day(),
            true,
        )
    } else {
        (
            default.year(),
            u32::try_from(first_value).expect("nonnegative digits"),
            second_value,
            false,
        )
    };
    if explicit_year {
        // An explicitly selected year from two at-most-two-digit fields is
        // necessarily below 100; longer spellings preserve their century.
        if first.len() <= 2 && second.len() <= 2 {
            year = expand_short_year(year, context);
        }
        if let Some(last) = NaiveDate::from_ymd_opt(year, month, 1)
            .and_then(|date| date.checked_add_months(Months::new(1)))
            .and_then(|date| date.pred_opt())
        {
            day = day.min(last.day());
        }
    }
    finish_clock_date_parts(
        text,
        &ClockDateParts {
            year,
            month,
            day,
            hour: i32::try_from(numeric_field(clock, "hour")).expect("two digits"),
            minute: numeric_field(clock, "minute"),
            second: numeric_field(clock, "second"),
        },
    )
    .map(TimestampTextValue::Timestamp)
}

fn validate_general_offset(text: &str, offset: i32) -> Result<(), TimestampTextError> {
    if offset.abs() >= 24 * 60 {
        Err(TimestampTextError::InvalidOffset {
            message: format!(
                "Parsed string \"{text}\" gives an invalid tzoffset, which must be between -timedelta(hours=24) and timedelta(hours=24)"
            ),
        })
    } else {
        Ok(())
    }
}

fn apply_general_offset(
    mut value: ParsedClock,
    offset: i32,
) -> Result<ParsedTimestampText, TimestampTextError> {
    let scale = match value.unit {
        TimeUnit::Second => 1,
        TimeUnit::Millisecond => 1_000,
        TimeUnit::Microsecond => 1_000_000,
        TimeUnit::Nanosecond => 1_000_000_000,
    };
    let utc_ticks = i128::from(value.ticks) - i128::from(offset) * 60 * scale;
    value.ticks = i64::try_from(utc_ticks).map_err(|_| TimestampTextError::OffsetOverflow)?;
    // Whole-minute offset adjustment cannot reach the NaT sentinel here:
    // general-parser nanoseconds require microseconds divisible by 1000,
    // whereas i64::MIN has a different submillisecond remainder. Coarser
    // resolutions with Python's bounded calendar are far inside i64 bounds.
    let timezone = if offset == 0 {
        "tzutc()".to_owned()
    } else {
        format!("tzoffset(None, {})", offset * 60)
    };
    Ok(value.into_timestamp(Some(timezone)))
}

fn parse_clock_failure(
    text: &str,
    reference_date: NaiveDate,
) -> Option<Result<TimestampTextValue, TimestampTextError>> {
    // Successful clocks and numeric date tokens have already been dispatched.
    // Do not turn an unsupported caller reference into a year-one timestamp.
    if !(1..=9999).contains(&reference_date.year()) {
        return None;
    }
    // Reuse the general clock validator, including AM/PM before numeric clock
    // errors. The valid default calendar cannot mask a clock error. Comma date
    // tokens must retain precedence (they can replace an invalid hour).
    finish_date_clock(
        text,
        text,
        ClockDateParts {
            year: 1,
            month: 1,
            day: 1,
            hour: 0,
            minute: 0,
            second: 0,
        },
        None,
    )
    .map(|value| value.map(TimestampTextValue::Timestamp))
}

fn parse_date_clock(
    text: &str,
    context: TimestampTextContext,
) -> Option<Result<TimestampTextValue, TimestampTextError>> {
    let date = DATE_CLOCK_PATTERN.captures(text)?;
    let mut month = numeric_field(&date, "month");
    let mut day = numeric_field(&date, "day");
    let mut year = numeric_field(&date, "year");
    let sep1 = &date["sep1"];
    let sep2 = &date["sep2"];
    let mut expand = (sep2 == " " || (sep1 != " " && sep1 != sep2)) && year <= 100;
    if (sep1 == ".") != (sep2 == ".") {
        if sep1 != " " && sep2 != " " {
            return Some(Err(TimestampTextError::DateParse {
                message: format!("Unknown datetime string format, unable to parse: {text}"),
            }));
        }
        // A lone dot belongs to a decimal token, whose fractional digits are
        // discarded when resolving the two remaining numeric date components.
        let other = if sep1 == "." { year } else { day };
        if let Some(clock) = date.name("clock") {
            if let Some(captures) = CLOCK_PATTERN.captures(clock.as_str()) {
                if captures
                    .name("separator")
                    .is_some_and(|value| value.as_str() == ",")
                    && captures
                        .name("second")
                        .is_some_and(|value| value.as_str().len() == 1)
                    && captures
                        .name("fraction")
                        .is_some_and(|value| !value.as_str().is_empty())
                {
                    return Some(finish_decimal_date_token(
                        text, &captures, month, other, context,
                    ));
                }
            }
        }
        if month > 31 {
            (year, month, day) = (month, other, 1);
            expand = other <= 100;
        } else if other > 31 {
            (year, day) = (other, 1);
            expand = other <= 100;
        } else {
            (year, day) = (1, other);
            expand = false;
        }
    } else if month > 31 {
        (year, month, day) = (month, day, year);
    } else if month > 12 {
        std::mem::swap(&mut month, &mut day);
    }
    let parts = ClockDateParts {
        year: if expand && year < 100 {
            expand_short_year(i32::try_from(year).expect("two digits"), context)
        } else {
            i32::try_from(year).expect("at most four digits")
        },
        month,
        day,
        hour: 0,
        minute: 0,
        second: 0,
    };
    if let Some(clock) = date.name("clock") {
        finish_explicit_date_clock(text, clock.as_str(), parts, (sep1 == ".") == (sep2 == "."))
    } else {
        Some(finish_clock_date_parts(text, &parts).map(TimestampTextValue::Timestamp))
    }
}

fn finish_decimal_date_token(
    text: &str,
    captures: &regex::Captures<'_>,
    first: u32,
    second: u32,
    context: TimestampTextContext,
) -> Result<TimestampTextValue, TimestampTextError> {
    let token = captures.name("fraction").expect("nonempty date token");
    let period = captures.name("period");
    let attached = period.is_some_and(|value| value.start() == token.end());
    let unknown = || TimestampTextError::DateParse {
        message: format!("Unknown datetime string format, unable to parse: {text}"),
    };
    if matches!(token.as_str().len(), 8 | 12 | 14) {
        return Err(unknown());
    }
    if token.as_str().len() != 6
        && period.is_none()
        && second > 100
        && token
            .as_str()
            .parse::<u32>()
            .map_or(true, |number| number > 100)
    {
        return Err(unknown());
    }
    let number = match token.as_str().parse::<i32>() {
        Ok(number) => number,
        Err(_) if attached => return Err(unknown()),
        Err(_) => {
            return Err(TimestampTextBoundsError {
                message: format!("Parsing \"{text}\" to datetime overflows"),
            }
            .into());
        }
    };
    let mut parts;
    if token.as_str().len() == 6 {
        parts = decimal_date_parts(first, second, None, context);
        parts.hour = number / 10000;
        parts.minute = u32::try_from(number / 100 % 100).expect("two digits");
        parts.second = u32::try_from(number % 100).expect("two digits");
    } else {
        let replacement = period.filter(|_| !attached || number < 24);
        if replacement.is_none() && attached && number > 31 {
            return Err(unknown());
        }
        parts = decimal_date_parts(
            first,
            second,
            replacement
                .is_none()
                .then_some(u32::try_from(number).expect("nonnegative digits")),
            context,
        );
        parts.hour = i32::try_from(numeric_field(captures, "hour")).expect("two digits");
        parts.minute = numeric_field(captures, "minute");
        parts.second = numeric_field(captures, "second");
        if let Some(period) = replacement {
            parts.hour = adjust_period(number, period.as_str());
            return finish_clock_date_parts(text, &parts).map(TimestampTextValue::Timestamp);
        }
    }
    if let Some(period) = period {
        if parts.hour > 12 {
            return Err(unknown());
        }
        parts.hour = adjust_period(parts.hour, period.as_str());
    }
    finish_clock_date_parts(text, &parts).map(TimestampTextValue::Timestamp)
}

fn decimal_date_parts(
    first: u32,
    second: u32,
    third: Option<u32>,
    context: TimestampTextContext,
) -> ClockDateParts {
    let (year, month, day, explicit_year) = if let Some(third) = third {
        if first > 31 {
            (first, second, third, true)
        } else if first > 12 {
            (third, second, first, true)
        } else {
            (third, first, second, true)
        }
    } else if first > 31 {
        (first, second, 1, true)
    } else if second > 31 {
        (second, first, 1, true)
    } else {
        (1, first, second, false)
    };
    let year = i32::try_from(year).expect("bounded date token");
    let year = if explicit_year && second <= 100 && third.unwrap_or(0) <= 100 && year < 100 {
        expand_short_year(year, context)
    } else {
        year
    };
    ClockDateParts {
        year,
        month,
        day,
        hour: 0,
        minute: 0,
        second: 0,
    }
}

fn named_month(name: &str) -> Option<u32> {
    let name = if name.eq_ignore_ascii_case("sept") {
        "Sep"
    } else {
        name
    };
    NaiveDate::parse_from_str(&format!("2000 {name} 01"), "%Y %B %d")
        .ok()
        .map(|date| date.month())
}

fn parse_partial_month_date(
    text: &str,
    context: TimestampTextContext,
) -> Option<Result<TimestampTextValue, TimestampTextError>> {
    let captures = PARTIAL_FIRST_MONTH_PATTERN
        .captures(text)
        .or_else(|| PARTIAL_LAST_MONTH_PATTERN.captures(text))?;
    let month = named_month(&captures["month"])?;
    let mut parts = ClockDateParts {
        year: 1,
        month,
        day: 1,
        hour: 0,
        minute: 0,
        second: 0,
    };
    if let Some(number) = captures.name("number") {
        let value = numeric_field(&captures, "number");
        let separator = &captures["sep"];
        let explicit_year = separator.eq_ignore_ascii_case(" of ");
        // Only numeric-first dots preserve lexical century markers; month
        // tokens consume slash/dash, while commas simply separate tokens.
        let numeric_first = number.start() < captures.name("month").expect("month capture").start();
        let lexical = (!numeric_first && matches!(separator, "-" | "/"))
            || (numeric_first
                && captures.name("number_suffix").is_none()
                && matches!(separator, "-" | "/" | "."));
        let century = lexical && number.as_str().len() > 2;
        if explicit_year || century || value > 31 {
            let year = i32::try_from(value).expect("at most four digits");
            parts.year = if year < 100 && !century {
                expand_short_year(year, context)
            } else {
                year
            };
        } else {
            parts.day = value;
        }
    }
    if let Some(clock) = captures.name("clock") {
        finish_explicit_date_clock(text, clock.as_str(), parts, false)
    } else {
        Some(finish_clock_date_parts(text, &parts).map(TimestampTextValue::Timestamp))
    }
}

enum MonthPosition {
    First,
    Middle,
    Last,
}

fn parse_named_month_composition(
    text: &str,
    context: TimestampTextContext,
) -> Option<Result<TimestampTextValue, TimestampTextError>> {
    let (captures, position) = [
        (&*MIDDLE_MONTH_PATTERN, MonthPosition::Middle),
        (&*FIRST_MONTH_PATTERN, MonthPosition::First),
        (&*LAST_MONTH_PATTERN, MonthPosition::Last),
    ]
    .into_iter()
    .find_map(|(pattern, position)| pattern.captures(text).map(|captures| (captures, position)))?;
    let month = named_month(&captures["month"])?;
    let first = numeric_field(&captures, "first");
    let last = numeric_field(&captures, "last");
    let explicit_year = captures["sep1"].eq_ignore_ascii_case(" of ");
    let first_suffix = captures.name("first_suffix").is_some();
    let last_suffix = captures.name("last_suffix").is_some();
    // A dot adjacent to the month splits the entire dotted lexical token.
    // Otherwise adjacent digits form one decimal (comma requires two digits).
    let decimal_separators = match position {
        MonthPosition::First => Some((&captures["sep2"], &captures["sep1"])),
        MonthPosition::Last => Some((
            &captures["sep1"],
            if last_suffix { " " } else { &captures["sep2"] },
        )),
        MonthPosition::Middle => None,
    };
    if let Some((numeric_separator, month_separator)) = decimal_separators {
        let decimal = !first_suffix
            && month_separator != "."
            && (numeric_separator == "."
                || (numeric_separator == "," && captures["first"].len() >= 2));
        if decimal {
            return finish_decimal_month(text, &captures, month, month_separator, context);
        }
    }
    // Punctuation passes lexical numbers to dateutil's Y/M/D accumulator;
    // whitespace passes numeric values and loses leading-zero century markers.
    let first_separator = matches!(&captures["sep1"], "-" | "/");
    let last_separator = matches!(&captures["sep2"], "-" | "/" | ".");
    let (first_lexical, last_lexical, default_first_year) = match position {
        MonthPosition::First => (
            first_separator || (!first_suffix && last_separator),
            !explicit_year
                && !first_suffix
                && last_separator
                && (!first_separator || captures["sep1"] == captures["sep2"]),
            first > 31,
        ),
        MonthPosition::Middle => {
            // Numeric tokens consume slash/dash/dot dates; a month token only
            // consumes slash/dash suffixes. Commas are skipped, not separators
            // that preserve leading-zero century markers.
            let numeric_separator = !first_suffix && matches!(&captures["sep1"], "-" | "/" | ".");
            let last_lexical = if numeric_separator {
                captures["sep1"] == captures["sep2"]
            } else {
                matches!(&captures["sep2"], "-" | "/")
            };
            (numeric_separator, last_lexical, first > 31)
        }
        MonthPosition::Last => (
            !first_suffix && matches!(&captures["sep1"], "-" | "/" | "."),
            (!first_suffix && matches!(&captures["sep1"], "-" | "/" | "."))
                || (!last_suffix && last_separator),
            last <= 31,
        ),
    };
    let first_year = explicit_year
        || if first_lexical {
            captures["first"].len() > 2
        } else {
            first > 100
        };
    let last_year = if last_lexical {
        captures["last"].len() > 2
    } else {
        last > 100
    };
    if first_year && last_year {
        return Some(Err(TimestampTextError::DateParse {
            message: format!("Unknown datetime string format, unable to parse: {text}"),
        }));
    }
    let (year, day) = if first_year || (!last_year && default_first_year) {
        (first, last)
    } else {
        (last, first)
    };
    let year = i32::try_from(year).expect("at most four digits");
    let parts = ClockDateParts {
        year: if year < 100 && (explicit_year || (!first_year && !last_year)) {
            expand_short_year(year, context)
        } else {
            year
        },
        month,
        day,
        hour: 0,
        minute: 0,
        second: 0,
    };
    if let Some(clock) = captures.name("clock") {
        finish_explicit_date_clock(text, clock.as_str(), parts, true)
    } else {
        Some(finish_clock_date_parts(text, &parts).map(TimestampTextValue::Timestamp))
    }
}

fn finish_decimal_month(
    text: &str,
    captures: &regex::Captures<'_>,
    month: u32,
    month_separator: &str,
    context: TimestampTextContext,
) -> Option<Result<TimestampTextValue, TimestampTextError>> {
    let first = numeric_field(captures, "first");
    let explicit_year = captures["sep1"].eq_ignore_ascii_case(" of ");
    let width = captures["first"].len() + 1 + captures["last"].len();
    if matches!(month_separator, "-" | "/") || (!explicit_year && matches!(width, 6 | 8)) {
        return Some(Err(TimestampTextError::DateParse {
            message: format!("Unknown datetime string format, unable to parse: {text}"),
        }));
    }
    if let Some(following) = captures
        .name("clock")
        .and_then(|clock| FOLLOWING_MONTH_NUMBER_PATTERN.captures(clock.as_str()))
    {
        return finish_decimal_month_number(text, captures, month, &following, context);
    }
    let parts = ClockDateParts {
        // The month `of` rule consumes and discards non-digit tokens.
        year: if !explicit_year && first > 31 {
            let year = i32::try_from(first).expect("at most four digits");
            if year < 100 {
                expand_short_year(year, context)
            } else {
                year
            }
        } else {
            1
        },
        month,
        day: if explicit_year || first > 31 {
            1
        } else {
            first
        },
        hour: 0,
        minute: 0,
        second: 0,
    };
    if let Some(clock) = captures.name("clock") {
        finish_explicit_date_clock(text, clock.as_str(), parts, false)
    } else {
        Some(finish_clock_date_parts(text, &parts).map(TimestampTextValue::Timestamp))
    }
}

fn finish_decimal_month_number(
    text: &str,
    captures: &regex::Captures<'_>,
    month: u32,
    following: &regex::Captures<'_>,
    context: TimestampTextContext,
) -> Option<Result<TimestampTextValue, TimestampTextError>> {
    let first = numeric_field(captures, "first");
    let discarded = captures["sep1"].eq_ignore_ascii_case(" of ");
    // The accumulator labels numeric values greater than 100 as years before
    // truncating decimals; 100.1 therefore differs from 100.0.
    let first_year =
        first > 100 || (first == 100 && captures["last"].bytes().any(|digit| digit != b'0'));
    // With `of` the decimal was discarded, so two following numeric fields
    // still fill the date before a subsequent clock can be consumed.
    let after_discard = if discarded {
        following
            .name("clock")
            .and_then(|clock| FOLLOWING_MONTH_NUMBER_PATTERN.captures(clock.as_str()))
    } else {
        None
    };
    let (first, first_year, discarded, following) = if let Some(extra) = &after_discard {
        let first = numeric_field(following, "number");
        (first, first > 100, false, extra)
    } else {
        (first, first_year, discarded, following)
    };
    let second = numeric_field(following, "number");
    let second_year = second > 100;
    if !discarded && first_year && second_year {
        return Some(Err(TimestampTextError::DateParse {
            message: format!("Unknown datetime string format, unable to parse: {text}"),
        }));
    }
    let (year, day, has_year) = if discarded {
        if second > 31 {
            (second, 1, true)
        } else {
            (1, second, false)
        }
    } else if first_year || (!second_year && first > 31) {
        (first, second, true)
    } else {
        (second, first, true)
    };
    let year = i32::try_from(year).expect("at most four digits");
    let century = (!discarded && first_year) || second_year;
    let parts = ClockDateParts {
        year: if has_year && !century && year < 100 {
            expand_short_year(year, context)
        } else {
            year
        },
        month,
        day,
        hour: 0,
        minute: 0,
        second: 0,
    };
    if let Some(clock) = following.name("clock") {
        finish_explicit_date_clock(text, clock.as_str(), parts, !discarded)
    } else {
        Some(finish_clock_date_parts(text, &parts).map(TimestampTextValue::Timestamp))
    }
}

fn parse_month_date(
    text: &str,
    context: TimestampTextContext,
) -> Option<Result<TimestampTextValue, TimestampTextError>> {
    let captures = MONTH_DATE_PATTERN.captures(text)?;
    let month = named_month(&captures["month"])?;
    let mut day = numeric_field(&captures, "day");
    let mut year = numeric_field(&captures, "year");
    if year <= 100 && day > 31 {
        std::mem::swap(&mut day, &mut year);
    }
    let year = i32::try_from(year).expect("at most four digits");
    let parts = ClockDateParts {
        year: if year < 100 {
            expand_short_year(year, context)
        } else {
            year
        },
        month,
        day,
        hour: 0,
        minute: 0,
        second: 0,
    };
    if let Some(clock) = captures.name("clock") {
        finish_explicit_date_clock(text, clock.as_str(), parts, true)
    } else {
        Some(finish_clock_date_parts(text, &parts).map(TimestampTextValue::Timestamp))
    }
}

fn numeric_field(captures: &regex::Captures<'_>, name: &str) -> u32 {
    captures.name(name).map_or(0, |value| {
        value.as_str().parse::<u32>().expect("bounded digits")
    })
}

fn finish_date_clock(
    text: &str,
    clock: &str,
    mut parts: ClockDateParts,
    offset: Option<i32>,
) -> Option<Result<ParsedTimestampText, TimestampTextError>> {
    let normalized;
    let clock = if let Some(captures) = HOUR_PERIOD_PATTERN.captures(clock) {
        let hour = i32::try_from(numeric_field(&captures, "hour")).expect("two digits");
        if captures["gap"].is_empty() && hour >= 24 {
            return Some(Err(TimestampTextError::DateParse {
                message: format!("Unknown datetime string format, unable to parse: {text}"),
            }));
        }
        normalized = format!(
            "{:02}:{:02}:{:02}",
            adjust_period(hour, &captures["period"]),
            parts.minute,
            parts.second
        );
        normalized.as_str()
    } else {
        clock
    };
    let clock = CLOCK_PATTERN.captures(clock)?;
    if clock
        .name("separator")
        .is_some_and(|value| value.as_str() == ",")
        && clock["second"].len() == 1
        && !clock["fraction"].is_empty()
    {
        return None; // Additional date tokens need the complete token parser.
    }
    let mut hour = i32::try_from(numeric_field(&clock, "hour")).expect("two digits");
    if let Some(period) = clock.name("period") {
        if hour > 12 {
            return Some(Err(TimestampTextError::DateParse {
                message: format!("Unknown datetime string format, unable to parse: {text}"),
            }));
        }
        hour = adjust_period(hour, period.as_str());
    }
    parts.hour = hour;
    parts.minute = numeric_field(&clock, "minute");
    parts.second = numeric_field(&clock, "second");
    Some(finish_clock_date_parts(text, &parts).and_then(|value| {
        if let Some(offset) = offset {
            validate_general_offset(text, offset)?;
        }
        let value = finish_general_clock(
            &clock,
            value.local_datetime.date(),
            u32::try_from(hour).expect("validated hour"),
        )
        .expect("validated calendar and clock without additional date tokens")
        .map_err(TimestampTextError::from)?;
        match offset {
            Some(offset) => apply_general_offset(value, offset),
            None => Ok(value.into_timestamp(None)),
        }
    }))
}

fn parse_delimited_date(text: &str) -> Option<Result<TimestampTextValue, TimestampTextError>> {
    let captures = DELIMITED_DATE_PATTERN.captures(text)?;
    let field = |primary, alternate| {
        captures
            .name(primary)
            .or_else(|| captures.name(alternate))
            .expect("one alternative supplies every required field")
            .as_str()
            .parse::<u32>()
            .expect("at most four digits")
    };
    let year = field("year", "year_only");
    if year < 1000 {
        return None;
    }
    let mut month = field("month", "month_only");
    let mut day = captures.name("day").map_or(1, |value| {
        value.as_str().parse::<u32>().expect("at most two digits")
    });
    if !(1..=31).contains(&month) || !(1..=31).contains(&day) || (month > 12 && day > 12) {
        return Some(Err(TimestampTextError::DateParse {
            message: format!("Invalid date specified ({month}/{day})"),
        }));
    }
    if month > 12 && captures.name("day").is_some() {
        std::mem::swap(&mut month, &mut day);
    }
    let parts = ClockDateParts {
        year: i32::try_from(year).expect("four digits"),
        month,
        day,
        hour: 0,
        minute: 0,
        second: 0,
    };
    Some(
        finish_clock_date_parts(text, &parts)
            .map(TimestampTextValue::Timestamp)
            .map_err(|error| TimestampTextError::InvalidCalendar {
                message: error
                    .to_string()
                    .strip_suffix(&format!(": {text}"))
                    .expect("calendar validation appends the original text")
                    .to_owned(),
            }),
    )
}

fn parse_numeric_date(
    text: &str,
    context: TimestampTextContext,
) -> Option<Result<TimestampTextValue, TimestampTextError>> {
    let trimmed = text.trim_start_matches(|c: char| c.is_ascii_whitespace());
    if trimmed.len() == 4 && trimmed.bytes().all(|c| c.is_ascii_digit()) {
        return parse_iso_timestamp_with_fields(&format!("{trimmed}-01-01"))
            .map(|result| result.map_err(TimestampTextError::from));
    }
    let captures = NUMERIC_DATE_PATTERN.captures(text)?;
    let digits = &captures["digits"];
    let period = captures.name("period");
    if digits.len() == 4 && period.is_some() {
        return None;
    }
    if digits.len() == 8 && trimmed.len() == 8 {
        let iso = format!("{}-{}-{}", &digits[..4], &digits[4..6], &digits[6..]);
        if let Some(result) = parse_iso_timestamp_with_fields(&iso) {
            return Some(result.map_err(TimestampTextError::from));
        }
    }
    if !text.starts_with('0')
        && period.is_none()
        && digits.parse::<u64>().expect("at most fourteen digits") < 1000
    {
        return Some(Err(TimestampTextError::NotDateLike {
            message: format!("Given date string \"{text}\" not likely a datetime"),
        }));
    }
    let mut parts = ClockDateParts {
        year: 1,
        month: 1,
        day: 1,
        hour: 0,
        minute: 0,
        second: 0,
    };
    if digits.len() == 4 {
        let number = digits.parse::<i32>().expect("four digits");
        if number <= 31 {
            parts.day = u32::try_from(number).expect("nonnegative digits");
        } else {
            parts.year = if number < 100 {
                expand_short_year(number, context)
            } else {
                number
            };
        }
    } else {
        apply_compact_date(digits, context, &mut parts);
    }
    if let Some(period) = period {
        // A date alone supplies no hour to which AM/PM can apply.
        if digits.len() <= 8 || parts.hour > 12 {
            return Some(Err(TimestampTextError::DateParse {
                message: format!("Unknown datetime string format, unable to parse: {text}"),
            }));
        }
        parts.hour = adjust_period(parts.hour, period.as_str());
    }
    Some(finish_clock_date_parts(text, &parts).map(TimestampTextValue::Timestamp))
}

fn parse_clock_date_token(
    text: &str,
    context: TimestampTextContext,
) -> Option<Result<TimestampTextValue, TimestampTextError>> {
    let captures = CLOCK_PATTERN.captures(text)?;
    let token = captures.name("fraction")?;
    if captures
        .name("separator")
        .expect("fraction requires a separator")
        .as_str()
        != ","
        || captures
            .name("second")
            .expect("fraction requires seconds")
            .as_str()
            .len()
            != 1
        || token.as_str().is_empty()
        || !(1..=9999).contains(&context.reference_date.year())
    {
        return None;
    }
    Some(interpret_clock_date_token(text, &captures, context).map(TimestampTextValue::Timestamp))
}

struct ClockDateParts {
    year: i32,
    month: u32,
    day: u32,
    hour: i32,
    minute: u32,
    second: u32,
}

fn interpret_clock_date_token(
    text: &str,
    captures: &regex::Captures<'_>,
    context: TimestampTextContext,
) -> Result<ParsedTimestampText, TimestampTextError> {
    let hour = captures["hour"].parse::<i32>().expect("two ASCII digits");
    let minute = captures["minute"].parse::<u32>().expect("two ASCII digits");
    let current = captures.name("hour").expect("required hour").start() == 0
        && captures["minute"].len() == 2
        && hour < 24
        && minute < 60;
    let date = if current {
        context.reference_date
    } else {
        NaiveDate::from_ymd_opt(1, 1, 1).expect("valid default date")
    };
    let mut parts = ClockDateParts {
        year: date.year(),
        month: date.month(),
        day: date.day(),
        hour,
        minute,
        second: captures["second"].parse().expect("one ASCII digit"),
    };
    let token = captures
        .name("fraction")
        .expect("required numeric date token");
    let period = captures.name("period");
    if matches!(token.as_str().len(), 6 | 8 | 12 | 14) {
        apply_compact_date(token.as_str(), context, &mut parts);
        if let Some(period) = period {
            if parts.hour > 12 {
                return Err(TimestampTextError::DateParse {
                    message: format!("Unknown datetime string format, unable to parse: {text}"),
                });
            }
            parts.hour = adjust_period(parts.hour, period.as_str());
        }
        return finish_clock_date_parts(text, &parts);
    }
    let attached_period = period.is_some_and(|value| value.start() == token.end());
    let unknown = || TimestampTextError::DateParse {
        message: format!("Unknown datetime string format, unable to parse: {text}"),
    };
    let number = match token.as_str().parse::<i32>() {
        Ok(number) => number,
        Err(_) if attached_period => return Err(unknown()),
        Err(_) => {
            return Err(TimestampTextBoundsError {
                message: format!("Parsing \"{text}\" to datetime overflows"),
            }
            .into());
        }
    };
    if let Some(period) = period.filter(|_| !attached_period || number < 24) {
        // The trailing numeric token can replace an already parsed hour.
        parts.hour = adjust_period(number, period.as_str());
    } else {
        if attached_period && number > 31 {
            return Err(unknown());
        }
        if number <= 31 {
            parts.day = u32::try_from(number).expect("nonnegative numeric token");
        } else {
            parts.year = if number < 100 {
                expand_short_year(number, context)
            } else {
                number
            };
        }
        if let Some(period) = period {
            if parts.hour > 12 {
                return Err(unknown());
            }
            parts.hour = adjust_period(parts.hour, period.as_str());
        }
    }
    finish_clock_date_parts(text, &parts)
}

fn expand_short_year(number: i32, context: TimestampTextContext) -> i32 {
    let parser_year = context.parser_initialized_date.year();
    let year = parser_year.div_euclid(100) * 100 + number;
    if year >= parser_year + 50 {
        year - 100
    } else if year < parser_year - 50 {
        year + 100
    } else {
        year
    }
}

fn apply_compact_date(token: &str, context: TimestampTextContext, parts: &mut ClockDateParts) {
    let digits = |start, end| {
        token[start..end]
            .parse::<u32>()
            .expect("bounded ASCII digits")
    };
    if token.len() == 6 {
        let (first, middle, last) = (digits(0, 2), digits(2, 4), digits(4, 6));
        let (year, month, day) = if first > 31 {
            (first, middle, last)
        } else if first > 12 {
            (last, middle, first)
        } else {
            (last, first, middle)
        };
        parts.year = expand_short_year(i32::try_from(year).expect("two digits"), context);
        parts.month = month;
        parts.day = day;
    } else {
        parts.year = i32::try_from(digits(0, 4)).expect("four digits");
        parts.month = digits(4, 6);
        parts.day = digits(6, 8);
        if token.len() > 8 {
            parts.hour = i32::try_from(digits(8, 10)).expect("two digits");
            parts.minute = digits(10, 12);
        }
        if token.len() == 14 {
            parts.second = digits(12, 14);
        }
    }
}

fn adjust_period(hour: i32, period: &str) -> i32 {
    if period.eq_ignore_ascii_case("pm") && hour < 12 {
        hour + 12
    } else if period.eq_ignore_ascii_case("am") && hour == 12 {
        0
    } else {
        hour
    }
}

fn finish_clock_date_parts(
    text: &str,
    parts: &ClockDateParts,
) -> Result<ParsedTimestampText, TimestampTextError> {
    let invalid = |message: String| TimestampTextError::DateParse {
        message: format!("{message}: {text}"),
    };
    if !(1..=9999).contains(&parts.year) {
        return Err(invalid(format!(
            "year must be in 1..9999, not {}",
            parts.year
        )));
    }
    if !(1..=12).contains(&parts.month) {
        return Err(invalid(format!(
            "month must be in 1..12, not {}",
            parts.month
        )));
    }
    let first =
        NaiveDate::from_ymd_opt(parts.year, parts.month, 1).expect("validated year and month");
    let date = first.with_day(parts.day).ok_or_else(|| {
        let last = first
            .checked_add_months(Months::new(1))
            .expect("next month fits Chrono")
            .pred_opt()
            .expect("month starts after year zero")
            .day();
        invalid(format!(
            "day {} must be in range 1..{last} for month {} in year {}",
            parts.day, parts.month, parts.year
        ))
    })?;
    if !(0..24).contains(&parts.hour) {
        return Err(invalid(format!(
            "hour must be in 0..23, not {}",
            parts.hour
        )));
    }
    if parts.minute >= 60 {
        return Err(invalid(format!(
            "minute must be in 0..59, not {}",
            parts.minute
        )));
    }
    if parts.second >= 60 {
        return Err(invalid(format!(
            "second must be in 0..59, not {}",
            parts.second
        )));
    }
    let local = date
        .and_hms_opt(
            u32::try_from(parts.hour).expect("validated hour"),
            parts.minute,
            parts.second,
        )
        .expect("validated clock");
    Ok(ParsedTimestampText {
        ticks: local.and_utc().timestamp(),
        unit: TimeUnit::Second,
        timezone: None,
        local_datetime: local,
    })
}

/// Recognize the implemented ISO-style date/time grammar and its source precision.
///
/// Handles four-digit year dates with consistent separators, optional colon or
/// compact hour/minute/second fields, up to eighteen
/// fractional digits, and numeric minute offsets. Missing markers, informal dates,
/// malformed components and other ISO spellings return `None` for later parsing.
/// No fallback, current-clock access, warnings or Python execution occurs here.
///
/// # Errors
///
/// A recognized valid calendar value returns `Some(Err(...))` if local precision
/// conversion or subsequent timezone adjustment exceeds the source bounds.
///
/// # Panics
///
/// Only if the static regex violates its bounded ASCII-digit capture invariants.
#[must_use]
pub fn parse_iso_timestamp_compatible(
    text: &str,
) -> Option<Result<TemporalFrameValue, TimestampTextBoundsError>> {
    parse_iso_timestamp_with_fields(text)
        .map(|result| result.map(TimestampTextValue::into_temporal))
}

fn parse_iso_timestamp_with_fields(
    text: &str,
) -> Option<Result<TimestampTextValue, TimestampTextBoundsError>> {
    let captures = ISO_PATTERN.captures(text.trim_matches(|c: char| c.is_ascii_whitespace()))?;
    if captures["ym_sep"] != captures["md_sep"] {
        return None;
    }
    if let Some(hour) = captures.name("hour") {
        if captures.name("minute").is_none() && hour.as_str().len() != 2 {
            return None;
        }
        if let Some(separator) = captures.name("hm_sep") {
            if captures
                .name("ms_sep")
                .is_some_and(|value| value.as_str() != separator.as_str())
            {
                return None;
            }
            if separator.as_str().is_empty()
                && ["hour", "minute", "second"].iter().any(|name| {
                    captures
                        .name(name)
                        .is_some_and(|value| value.as_str().len() != 2)
                })
            {
                return None;
            }
        }
    } else if text.ends_with(|c: char| c.is_ascii_whitespace()) {
        return None; // Date-only trailing whitespace enters dateutil, not ISO.
    }
    let date = NaiveDate::from_ymd_opt(
        i32::try_from(numeric_field(&captures, "year")).expect("four digits"),
        numeric_field(&captures, "month"),
        numeric_field(&captures, "day"),
    )?;
    let field = |name| {
        captures.name(name).map_or(0, |value| {
            value
                .as_str()
                .parse::<u32>()
                .expect("two ASCII digits fit u32")
        })
    };
    let fraction = captures.name("fraction").map(|value| value.as_str());
    let (unit, scale) = match fraction.map(str::len) {
        None => (TimeUnit::Second, 1_i128),
        Some(0..=3) => (TimeUnit::Millisecond, 1_000),
        Some(4..=6) => (TimeUnit::Microsecond, 1_000_000),
        Some(_) => (TimeUnit::Nanosecond, 1_000_000_000),
    };
    // Source digits beyond nanoseconds are discarded, not rounded.
    let nanos = fraction.map_or(0, decimal_nanos);
    let local = date.and_hms_nano_opt(field("hour"), field("minute"), field("second"), nanos)?;
    let zone = captures.name("zone").map(|value| value.as_str());
    // After an hours-only numeric offset, remaining whitespace makes the
    // source attempt (and fail) to parse offset minutes. Do not strip it into
    // a successful ISO result: dateutil's timezone type and precision differ.
    if zone.is_some_and(|value| value != "Z" && value.len() <= 3)
        && text.ends_with(|c: char| c.is_ascii_whitespace())
    {
        return None;
    }
    let offset_minutes = match zone {
        None | Some("Z") => 0,
        Some(zone) => iso_offset_minutes(zone)?,
    };
    let local_ticks = i128::from(local.and_utc().timestamp()) * scale
        + i128::from(nanos) / (1_000_000_000 / scale);
    if local_ticks <= i128::from(i64::MIN) || local_ticks > i128::from(i64::MAX) {
        return Some(Err(TimestampTextBoundsError {
            message: format!(
                "Out of bounds nanosecond timestamp: {}-{}",
                local.year(),
                local.format("%m-%d %H:%M:%S")
            ),
        }));
    }
    let utc_ticks = local_ticks - i128::from(offset_minutes) * 60 * scale;
    Some(finish_timestamp(
        local,
        utc_ticks,
        unit,
        zone,
        offset_minutes,
    ))
}

fn iso_offset_minutes(zone: &str) -> Option<i32> {
    let digits = &zone[1..];
    // Without a colon the source greedily takes two hour digits, then up to
    // two minute digits. Thus +083 is +08:03, not +00:83 or +08:30.
    let (hours, minutes) = digits
        .split_once(':')
        .unwrap_or_else(|| digits.split_at(digits.len().min(2)));
    let hours = hours.parse::<i32>().expect("one or two ASCII digits");
    let minutes = if minutes.is_empty() {
        0
    } else {
        minutes.parse::<i32>().expect("one or two ASCII digits")
    };
    if hours >= 24 || minutes >= 60 {
        return None;
    }
    Some((hours * 60 + minutes) * if zone.starts_with('-') { -1 } else { 1 })
}

/// Recognize a naive clock string using Pandas' general-parser precision rules.
///
/// `reference_date` is the local current date read by the caller. A valid H:MM or
/// HH:MM prefix selects it; other recognized clocks use 0001-01-01, including a
/// leading space or a one-digit minute. AM/PM and fractional seconds are retained.
/// Dot/comma fractions retain their distinct source resolution rules.
/// Timezone suffixes and other general-parser spellings still
/// return `None`, which must not be interpreted as a source parse failure.
/// The reference date must be representable by Python's date (years 1..=9999).
///
/// # Errors
///
/// Returns a source bounds error when inferred precision cannot represent the date.
///
/// # Panics
///
/// Only if the static regex violates its bounded ASCII-digit capture invariants.
#[must_use]
pub fn parse_clock_timestamp_compatible(
    text: &str,
    reference_date: NaiveDate,
) -> Option<Result<ParsedTimestampText, TimestampTextBoundsError>> {
    let captures = CLOCK_PATTERN.captures(text)?;
    let field = |name| {
        captures.name(name).map_or(0, |value| {
            value
                .as_str()
                .parse::<u32>()
                .expect("two ASCII digits fit u32")
        })
    };
    let mut hour = field("hour");
    if let Some(period) = captures.name("period") {
        if hour > 12 {
            return None;
        }
        hour = hour % 12 + u32::from(period.as_str().eq_ignore_ascii_case("pm")) * 12;
    }
    let date = clock_reference_date(&captures, reference_date)?;
    finish_general_clock(&captures, date, hour)
        .map(|result| result.map(|value| value.into_timestamp(None)))
}

fn clock_reference_date(
    captures: &regex::Captures<'_>,
    reference_date: NaiveDate,
) -> Option<NaiveDate> {
    let use_current_date = captures
        .name("hour")
        .expect("required hour capture")
        .start()
        == 0
        && captures["minute"].len() == 2
        && numeric_field(captures, "hour") < 24
        && numeric_field(captures, "minute") < 60;
    if use_current_date {
        if !(1..=9999).contains(&reference_date.year()) {
            return None;
        }
        Some(reference_date)
    } else {
        Some(NaiveDate::from_ymd_opt(1, 1, 1).expect("Python's default date is valid"))
    }
}

fn finish_general_clock(
    captures: &regex::Captures<'_>,
    date: NaiveDate,
    hour: u32,
) -> Option<Result<ParsedClock, TimestampTextBoundsError>> {
    let minute = captures["minute"].parse::<u32>().expect("two digits");
    let second = captures.name("second").map_or(0, |value| {
        value.as_str().parse::<u32>().expect("two digits")
    });
    let fraction = captures.name("fraction").map_or("", |value| value.as_str());
    // A comma after a one-digit second starts a date token in dateutil, not a
    // fraction (e.g. 9:30:0,2 selects day 2). Leave that general-date grammar to
    // the fallback rather than incorrectly recognizing a clock-only value.
    if captures
        .name("separator")
        .is_some_and(|value| value.as_str() == ",")
        && captures
            .name("second")
            .expect("separator requires seconds")
            .as_str()
            .len()
            == 1
        && !fraction.is_empty()
    {
        return None;
    }
    let nanos = decimal_nanos(fraction);
    let micros = nanos / 1_000;
    // Dateutil first materializes microseconds. Pandas recovers finer precision
    // only when that microsecond field is an exact multiple of a millisecond.
    let (unit, scale) = if micros % 1_000 != 0 {
        (TimeUnit::Microsecond, 1_000_000_i128)
    } else if captures
        .name("separator")
        .is_some_and(|value| value.as_str() == ",")
        || captures["minute"].len() != 2
        || captures
            .name("second")
            .is_none_or(|value| value.as_str().len() != 2)
    {
        (TimeUnit::Second, 1)
    } else {
        match fraction.len() {
            0 => (TimeUnit::Second, 1),
            1..=3 => (TimeUnit::Millisecond, 1_000),
            4..=6 => (TimeUnit::Microsecond, 1_000_000),
            _ => (TimeUnit::Nanosecond, 1_000_000_000),
        }
    };
    let field_nanos = if unit == TimeUnit::Nanosecond {
        nanos
    } else {
        micros * 1_000
    };
    let local = date.and_hms_nano_opt(hour, minute, second, field_nanos)?;
    let ticks = i128::from(local.and_utc().timestamp()) * scale
        + i128::from(nanos) / (1_000_000_000 / scale);
    if ticks <= i128::from(i64::MIN) || ticks > i128::from(i64::MAX) {
        return Some(Err(TimestampTextBoundsError {
            message: "Out of bounds nanosecond timestamp".to_owned(),
        }));
    }
    Some(Ok(ParsedClock {
        ticks: i64::try_from(ticks).expect("validated timestamp bounds"),
        unit,
        local_datetime: local,
    }))
}

fn decimal_nanos(digits: &str) -> u32 {
    let mut positions = digits.bytes().chain(std::iter::repeat(b'0'));
    (0..9).fold(0_u32, |value, _| {
        value * 10 + u32::from(positions.next().expect("infinite digit iterator") - b'0')
    })
}

fn finish_timestamp(
    local: NaiveDateTime,
    ticks: i128,
    unit: TimeUnit,
    zone: Option<&str>,
    offset_minutes: i32,
) -> Result<TimestampTextValue, TimestampTextBoundsError> {
    let ticks = i64::try_from(ticks).map_err(|_| {
        let (direction, limit) = if ticks < 0 {
            ("underflows", "1677-09-21 00:12:43.145224193")
        } else {
            ("overflows", "2262-04-11 23:47:16.854775807")
        };
        TimestampTextBoundsError {
            message: format!(
                "Converting {} {direction} past {limit}",
                local.format("%Y-%m-%d %H:%M:%S")
            ),
        }
    })?;
    if ticks == i64::MIN {
        return Ok(TimestampTextValue::NotATime);
    }
    let timezone = zone.map(|_| {
        if offset_minutes == 0 {
            "UTC".to_owned()
        } else {
            let sign = if offset_minutes < 0 { '-' } else { '+' };
            let minutes = offset_minutes.abs();
            format!("UTC{sign}{:02}:{:02}", minutes / 60, minutes % 60)
        }
    });
    Ok(TimestampTextValue::Timestamp(ParsedTimestampText {
        ticks,
        unit,
        timezone,
        local_datetime: local,
    }))
}
