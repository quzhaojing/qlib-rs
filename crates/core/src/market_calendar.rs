//! Intraday trading sessions and minute calendars compatible with `qlib.utils.time`.

use std::{
    num::NonZeroUsize,
    sync::{Arc, LazyLock, Mutex},
};

use chrono::{NaiveTime, TimeDelta};
use lru::LruCache;
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use thiserror::Error;

use crate::Region;

/// A half-open intraday trading session: `start <= time < end`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TradingSession {
    /// Inclusive market-open time.
    pub start: NaiveTime,
    /// Exclusive market-close time.
    pub end: NaiveTime,
}

/// Mainland China morning and afternoon sessions.
pub const CN_SESSIONS: [TradingSession; 2] = [
    TradingSession {
        start: NaiveTime::from_hms_opt(9, 30, 0).expect("valid CN market time"),
        end: NaiveTime::from_hms_opt(11, 30, 0).expect("valid CN market time"),
    },
    TradingSession {
        start: NaiveTime::from_hms_opt(13, 0, 0).expect("valid CN market time"),
        end: NaiveTime::from_hms_opt(15, 0, 0).expect("valid CN market time"),
    },
];

/// United States continuous regular session.
pub const US_SESSIONS: [TradingSession; 1] = [TradingSession {
    start: NaiveTime::from_hms_opt(9, 30, 0).expect("valid US market time"),
    end: NaiveTime::from_hms_opt(16, 0, 0).expect("valid US market time"),
}];

/// Taiwan continuous regular session.
pub const TW_SESSIONS: [TradingSession; 1] = [TradingSession {
    start: NaiveTime::from_hms_opt(9, 0, 0).expect("valid TW market time"),
    end: NaiveTime::from_hms_opt(13, 30, 0).expect("valid TW market time"),
}];

/// Maximum number of minute-calendar variants retained by Qlib.
pub const MINUTE_CALENDAR_CACHE_CAPACITY: usize = 240;

/// Failures while constructing an intraday minute calendar.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum MarketCalendarError {
    /// Pandas cannot represent the requested minute shift as a nanosecond duration.
    #[error("minute shift is outside the pandas Timedelta range: {shift}")]
    ShiftOutOfRange {
        /// Rejected signed minute count.
        shift: BigInt,
    },
}

type CalendarKey = (BigInt, Region);
type SharedCalendar = Arc<[NaiveTime]>;
const NANOS_PER_MINUTE: i64 = 60_000_000_000;
const CN_US_MAX_POSITIVE_SHIFT: i64 = 116_906_957;
const TW_MAX_POSITIVE_SHIFT: i64 = 116_906_927;

static MINUTE_CALENDARS: LazyLock<Mutex<LruCache<CalendarKey, SharedCalendar>>> =
    LazyLock::new(|| {
        Mutex::new(LruCache::new(
            NonZeroUsize::new(MINUTE_CALENDAR_CACHE_CAPACITY)
                .expect("Qlib cache capacity is non-zero"),
        ))
    });

impl Region {
    /// Return the regular intraday sessions used by Qlib's minute calendar.
    #[must_use]
    pub const fn trading_sessions(self) -> &'static [TradingSession] {
        match self {
            Self::Cn => &CN_SESSIONS,
            Self::Us => &US_SESSIONS,
            Self::Tw => &TW_SESSIONS,
        }
    }
}

/// Return the cached, end-exclusive minute calendar for a region.
///
/// `shift` follows Pandas `Series.shift` direction: a positive value moves every
/// clock time backward. Valid values are reduced modulo one day only while
/// constructing the calendar; the original integer remains part of the LRU key.
/// # Errors
///
/// Returns [`MarketCalendarError::ShiftOutOfRange`] when Pandas Timedelta would
/// reject the shift before applying it to the source calendar.
///
/// # Panics
///
/// Panics if another thread previously poisoned the process-global cache lock.
pub fn minute_calendar(
    shift: &BigInt,
    region: Region,
) -> Result<SharedCalendar, MarketCalendarError> {
    let shift_minutes = shift
        .to_i64()
        .filter(|value| value.checked_mul(NANOS_PER_MINUTE).is_some());
    let maximum_positive_shift = match region {
        Region::Cn | Region::Us => CN_US_MAX_POSITIVE_SHIFT,
        Region::Tw => TW_MAX_POSITIVE_SHIFT,
    };
    let Some(shift_minutes) = shift_minutes.filter(|value| *value <= maximum_positive_shift) else {
        return Err(MarketCalendarError::ShiftOutOfRange {
            shift: shift.clone(),
        });
    };
    Ok(cached_minute_calendar(
        (shift.clone(), region),
        shift_minutes,
    ))
}

/// Return Qlib's unshifted regular minute calendar for a region.
///
/// # Panics
///
/// Panics if another thread previously poisoned the process-global cache lock.
#[must_use]
pub fn regular_minute_calendar(region: Region) -> SharedCalendar {
    cached_minute_calendar((BigInt::from(0), region), 0)
}

fn cached_minute_calendar(key: CalendarKey, shift_minutes: i64) -> SharedCalendar {
    let region = key.1;
    let mut calendars = MINUTE_CALENDARS
        .lock()
        .expect("minute calendar generation cannot poison its cache lock");
    Arc::clone(calendars.get_or_insert(key, || {
        build_minute_calendar(shift_minutes, region.trading_sessions())
    }))
}

fn build_minute_calendar(shift: i64, sessions: &[TradingSession]) -> SharedCalendar {
    let wrapped_shift = shift % (24 * 60);
    let mut calendar = Vec::new();
    for session in sessions {
        let minute_count = session
            .end
            .signed_duration_since(session.start)
            .num_minutes();
        for offset in 0..minute_count {
            calendar.push(session.start + TimeDelta::minutes(offset - wrapped_shift));
        }
    }
    calendar.into()
}
