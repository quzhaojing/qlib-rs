//! Typed trade-range rules compatible with `qlib.backtest.decision`.

use chrono::{DateTime, NaiveDate, NaiveDateTime, NaiveTime};
use thiserror::Error;

/// Failures emitted by a replaceable trade-calendar range provider.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TradeCalendarRangeError {
    /// The calendar could not resolve the requested closed timestamp range.
    #[error("trade calendar range provider error: {message}")]
    Provider {
        /// Provider diagnostic retained for adapters and logs.
        message: String,
    },
}

/// Narrow object-safe calendar boundary required by [`TradeRangeByTime`].
pub trait TradeCalendarRange: Send + Sync {
    /// Requested start timestamp whose date anchors the intraday rule, not the current bar.
    ///
    /// # Errors
    /// Returns missing initialization or provider access failures.
    fn start_time(&self) -> Result<NaiveDateTime, TradeCalendarRangeError>;

    /// Resolve a closed timestamp range to closed integer step indices.
    ///
    /// # Errors
    ///
    /// Returns a provider failure without changing its classification.
    fn get_range_idx(
        &self,
        start_time: NaiveDateTime,
        end_time: NaiveDateTime,
    ) -> Result<(i64, i64), TradeCalendarRangeError>;
}

/// Failures from constructing or evaluating a trade range.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TradeRangeError {
    /// A Python-facing time spelling cannot be represented by the typed core.
    #[error("invalid trade-range time: {input}")]
    InvalidTime {
        /// Original rejected spelling.
        input: String,
    },
    /// Qlib requires a strictly increasing intraday rule.
    #[error("trade-range start {start} must be earlier than end {end}")]
    InvalidBounds {
        /// Parsed start clock time.
        start: NaiveTime,
        /// Parsed end clock time.
        end: NaiveTime,
    },
    /// `TradeRangeByTime.__call__` cannot resolve indices without a calendar.
    #[error("trade calendar is necessary for TradeRangeByTime")]
    MissingCalendar,
    /// `IdxTradeRange` deliberately has no timestamp clipping implementation.
    #[error("IdxTradeRange does not implement timestamp clipping")]
    IndexTimeClippingUnsupported,
    /// A replaceable calendar implementation failed.
    #[error(transparent)]
    Calendar(#[from] TradeCalendarRangeError),
}

/// Object-safe decision range, expressed either in steps or closed timestamps.
pub trait TradeRange: Send + Sync {
    /// Return closed intraday clock bounds when this range is time based.
    ///
    /// Index-only and custom ranges default to no time bounds. Implementations corresponding to
    /// Python `TradeRangeByTime` override this method.
    fn time_bounds(&self) -> Option<(NaiveTime, NaiveTime)> {
        None
    }

    /// Resolve the closed tradable step range.
    ///
    /// `None` is accepted so the typed API preserves Python's explicit missing-calendar failure;
    /// index ranges ignore it, while time ranges reject it.
    ///
    /// # Errors
    ///
    /// Returns a missing-calendar or provider failure where applicable.
    fn range_indices(
        &self,
        calendar: Option<&dyn TradeCalendarRange>,
    ) -> Result<(i64, i64), TradeRangeError>;

    /// Intersect a closed timestamp interval with this rule.
    ///
    /// # Errors
    ///
    /// Index-only rules return [`TradeRangeError::IndexTimeClippingUnsupported`].
    fn clip_time_range(
        &self,
        start_time: NaiveDateTime,
        end_time: NaiveDateTime,
    ) -> Result<(NaiveDateTime, NaiveDateTime), TradeRangeError>;
}

/// Closed integer step range. Qlib accepts negative and reversed indices unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdxTradeRange {
    start_idx: i64,
    end_idx: i64,
}

impl IdxTradeRange {
    /// Construct an unvalidated index range, matching Python.
    #[must_use]
    pub const fn new(start_idx: i64, end_idx: i64) -> Self {
        Self { start_idx, end_idx }
    }

    /// Closed start index.
    #[must_use]
    pub const fn start_idx(self) -> i64 {
        self.start_idx
    }

    /// Closed end index.
    #[must_use]
    pub const fn end_idx(self) -> i64 {
        self.end_idx
    }
}

impl TradeRange for IdxTradeRange {
    fn range_indices(
        &self,
        _calendar: Option<&dyn TradeCalendarRange>,
    ) -> Result<(i64, i64), TradeRangeError> {
        Ok((self.start_idx, self.end_idx))
    }

    fn clip_time_range(
        &self,
        _start_time: NaiveDateTime,
        _end_time: NaiveDateTime,
    ) -> Result<(NaiveDateTime, NaiveDateTime), TradeRangeError> {
        Err(TradeRangeError::IndexTimeClippingUnsupported)
    }
}

/// Strictly increasing intraday clock range with both endpoints closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TradeRangeByTime {
    start_time: NaiveTime,
    end_time: NaiveTime,
}

impl TradeRangeByTime {
    /// Construct a time rule from already parsed clock values.
    ///
    /// # Errors
    ///
    /// Returns [`TradeRangeError::InvalidBounds`] unless `start_time < end_time`.
    pub fn new(start_time: NaiveTime, end_time: NaiveTime) -> Result<Self, TradeRangeError> {
        if start_time >= end_time {
            return Err(TradeRangeError::InvalidBounds {
                start: start_time,
                end: end_time,
            });
        }
        Ok(Self {
            start_time,
            end_time,
        })
    }

    /// Parse the documented time strings plus common Pandas timestamp spellings.
    ///
    /// # Errors
    ///
    /// Returns an invalid-time or invalid-bounds failure.
    pub fn parse(start_time: &str, end_time: &str) -> Result<Self, TradeRangeError> {
        Self::new(parse_clock(start_time)?, parse_clock(end_time)?)
    }

    /// Closed start clock time.
    #[must_use]
    pub const fn start_time(self) -> NaiveTime {
        self.start_time
    }

    /// Closed end clock time.
    #[must_use]
    pub const fn end_time(self) -> NaiveTime {
        self.end_time
    }
}

impl TradeRange for TradeRangeByTime {
    fn time_bounds(&self) -> Option<(NaiveTime, NaiveTime)> {
        Some((self.start_time, self.end_time))
    }

    fn range_indices(
        &self,
        calendar: Option<&dyn TradeCalendarRange>,
    ) -> Result<(i64, i64), TradeRangeError> {
        let calendar = calendar.ok_or(TradeRangeError::MissingCalendar)?;
        let date = calendar.start_time()?.date();
        calendar
            .get_range_idx(date.and_time(self.start_time), date.and_time(self.end_time))
            .map_err(Into::into)
    }

    fn clip_time_range(
        &self,
        start_time: NaiveDateTime,
        end_time: NaiveDateTime,
    ) -> Result<(NaiveDateTime, NaiveDateTime), TradeRangeError> {
        let date = start_time.date();
        let rule_start = date.and_time(self.start_time);
        let rule_end = date.and_time(self.end_time);
        Ok((rule_start.max(start_time), rule_end.min(end_time)))
    }
}

fn parse_clock(input: &str) -> Result<NaiveTime, TradeRangeError> {
    let input = input.trim();
    if let Ok(value) = DateTime::parse_from_rfc3339(input) {
        return Ok(value.time());
    }
    let normalized = input.replace('T', " ");
    if let Ok(value) = NaiveDateTime::parse_from_str(&normalized, "%Y-%m-%d %H:%M:%S%.f") {
        return Ok(value.time());
    }
    if let Ok(value) = NaiveDateTime::parse_from_str(&normalized, "%Y-%m-%d %H:%M") {
        return Ok(value.time());
    }
    if let Ok(value) = NaiveTime::parse_from_str(input, "%H:%M:%S%.f") {
        return Ok(value);
    }
    if let Ok(value) = NaiveTime::parse_from_str(input, "%H:%M") {
        return Ok(value);
    }
    if NaiveDate::parse_from_str(input, "%Y-%m-%d").is_ok() {
        return Ok(NaiveTime::MIN);
    }
    Err(TradeRangeError::InvalidTime {
        input: input.to_owned(),
    })
}
