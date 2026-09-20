//! Source-observable mutable cache semantics for `qlib.utils.time.get_min_cal`.
//!
//! The ordinary [`crate::minute_calendar`] API deliberately returns immutable calendars keyed by
//! normalized Rust arguments. This compatibility facade is additive: it retains the raw Python
//! call shape as the cache key and returns a shared mutable list for consumers that need the
//! observable `functools.lru_cache` contract.

use std::{
    num::NonZeroUsize,
    sync::{Arc, Mutex, MutexGuard},
};

use chrono::NaiveTime;
use lru::LruCache;
use num_bigint::BigInt;
use thiserror::Error;

use crate::{MarketCalendarError, Region, minute_calendar};

/// The fixed `maxsize` declared by the Python `functools.lru_cache` decorator.
pub const TIME_CALENDAR_CACHE_MAXSIZE: usize = 240;

/// A dynamically typed value supported by the narrow `get_min_cal` call binder.
///
/// Python accepts additional runtime objects and lets Pandas or string formatting decide their
/// behavior. The native facade intentionally supports only the source contract's annotated
/// integer shift and string region boundary.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TimeCalendarValue {
    /// An arbitrary-size Python-compatible integer shift.
    Integer(BigInt),
    /// A Python-compatible region string.
    Text(String),
}

impl From<BigInt> for TimeCalendarValue {
    fn from(value: BigInt) -> Self {
        Self::Integer(value)
    }
}

impl From<i64> for TimeCalendarValue {
    fn from(value: i64) -> Self {
        Self::Integer(BigInt::from(value))
    }
}

impl From<&str> for TimeCalendarValue {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

impl From<String> for TimeCalendarValue {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

/// One keyword entry in insertion order.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TimeCalendarKeyword {
    /// Raw keyword name.
    pub name: String,
    /// Raw keyword value.
    pub value: TimeCalendarValue,
}

impl TimeCalendarKeyword {
    /// Construct an ordered keyword entry.
    #[must_use]
    pub fn new(name: impl Into<String>, value: impl Into<TimeCalendarValue>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
        }
    }
}

/// Raw positional and ordered-keyword call shape used as the cache key.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct TimeCalendarCall {
    /// Positional arguments in call order.
    pub positional: Vec<TimeCalendarValue>,
    /// Keyword arguments in source insertion order.
    pub keywords: Vec<TimeCalendarKeyword>,
}

impl TimeCalendarCall {
    /// Construct an arbitrary supported raw call shape.
    #[must_use]
    pub fn new(positional: Vec<TimeCalendarValue>, keywords: Vec<TimeCalendarKeyword>) -> Self {
        Self {
            positional,
            keywords,
        }
    }
}

/// Shared mutable list identity returned by a successful cached call.
pub type SharedMutableMinuteCalendar = Arc<Mutex<Vec<NaiveTime>>>;

/// `functools.lru_cache.cache_info()` fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeCalendarCacheInfo {
    /// Calls served from an existing raw-shape entry.
    pub hits: u64,
    /// Calls that reached argument binding and calendar construction.
    pub misses: u64,
    /// Configured maximum number of retained entries.
    pub maxsize: usize,
    /// Current retained entry count.
    pub currsize: usize,
}

/// Failures from binding or executing a raw `get_min_cal` call.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TimeCalendarCacheError {
    /// The raw call cannot bind to `get_min_cal(shift=0, region="cn")`.
    #[error("{message}")]
    Binding {
        /// Python-compatible binding diagnostic.
        message: String,
    },
    /// A supported parameter received a value outside the narrow annotated native types.
    #[error("get_min_cal {parameter} requires {expected}")]
    UnsupportedValueType {
        /// Parameter being bound.
        parameter: &'static str,
        /// Supported native kind.
        expected: &'static str,
    },
    /// The region string is not one of Qlib's three constants.
    #[error("{region} is not supported")]
    UnsupportedRegion {
        /// Rejected source string.
        region: String,
    },
    /// The shift cannot be represented by Pandas' minute Timedelta boundary.
    #[error(transparent)]
    Calendar(#[from] MarketCalendarError),
}

struct TimeCalendarCacheState {
    entries: LruCache<TimeCalendarCall, SharedMutableMinuteCalendar>,
    hits: u64,
    misses: u64,
}

/// Independent source-compatible cache facade for `get_min_cal`.
pub struct TimeCalendarCache {
    state: Mutex<TimeCalendarCacheState>,
}

impl Default for TimeCalendarCache {
    fn default() -> Self {
        Self::new()
    }
}

impl TimeCalendarCache {
    /// Construct an empty cache with the source `maxsize=240` policy.
    ///
    /// # Panics
    ///
    /// Panics only if the compile-time source cache size is changed to zero.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Mutex::new(TimeCalendarCacheState {
                entries: LruCache::new(
                    NonZeroUsize::new(TIME_CALENDAR_CACHE_MAXSIZE)
                        .expect("the source cache size is non-zero"),
                ),
                hits: 0,
                misses: 0,
            }),
        }
    }

    /// Execute one raw call, preserving raw-shape keys and shared mutable list identity.
    ///
    /// Failed calls increment `misses` but are never inserted and never evict a successful entry.
    ///
    /// # Errors
    ///
    /// Returns a binding or value-kind error for unsupported raw forms, an unsupported-region
    /// error for unknown strings, or the existing calendar range error for invalid shifts.
    ///
    /// # Panics
    ///
    /// Panics if an earlier in-process panic poisoned this facade's internal state lock.
    pub fn get(
        &self,
        call: &TimeCalendarCall,
    ) -> Result<SharedMutableMinuteCalendar, TimeCalendarCacheError> {
        {
            let mut state = self.lock_state();
            if let Some(calendar) = state.entries.get(call) {
                let calendar = Arc::clone(calendar);
                state.hits += 1;
                return Ok(calendar);
            }
            state.misses += 1;
        }

        let (shift, region) = bind_call(call)?;
        let calendar = minute_calendar(&shift, region)?;
        let mutable = Arc::new(Mutex::new(calendar.to_vec()));

        let mut state = self.lock_state();
        if state.entries.contains(call) {
            // `functools.lru_cache` lets concurrent misses finish independently. The first
            // completed value remains cached, while a later concurrent caller still receives
            // the distinct value it computed.
            return Ok(mutable);
        }
        state.entries.put(call.clone(), Arc::clone(&mutable));
        Ok(mutable)
    }

    /// Return current hit, miss, size-limit, and retained-entry counters.
    ///
    /// # Panics
    ///
    /// Panics if an earlier in-process panic poisoned this facade's internal state lock.
    #[must_use]
    pub fn cache_info(&self) -> TimeCalendarCacheInfo {
        let state = self.lock_state();
        TimeCalendarCacheInfo {
            hits: state.hits,
            misses: state.misses,
            maxsize: TIME_CALENDAR_CACHE_MAXSIZE,
            currsize: state.entries.len(),
        }
    }

    /// Clear retained entries and reset hit/miss counters.
    ///
    /// # Panics
    ///
    /// Panics if an earlier in-process panic poisoned this facade's internal state lock.
    pub fn cache_clear(&self) {
        let mut state = self.lock_state();
        state.entries.clear();
        state.hits = 0;
        state.misses = 0;
    }

    fn lock_state(&self) -> MutexGuard<'_, TimeCalendarCacheState> {
        self.state
            .lock()
            .expect("time calendar cache operations cannot poison the state lock")
    }
}

fn bind_call(call: &TimeCalendarCall) -> Result<(BigInt, Region), TimeCalendarCacheError> {
    if call.positional.len() > 2 {
        return Err(binding_error(format!(
            "get_min_cal() takes from 0 to 2 positional arguments but {} were given",
            call.positional.len()
        )));
    }

    let mut shift = call.positional.first();
    let mut region = call.positional.get(1);
    for keyword in &call.keywords {
        match keyword.name.as_str() {
            "shift" => {
                if shift.is_some() {
                    return Err(binding_error(
                        "get_min_cal() got multiple values for argument 'shift'",
                    ));
                }
                shift = Some(&keyword.value);
            }
            "region" => {
                if region.is_some() {
                    return Err(binding_error(
                        "get_min_cal() got multiple values for argument 'region'",
                    ));
                }
                region = Some(&keyword.value);
            }
            name => {
                return Err(binding_error(format!(
                    "get_min_cal() got an unexpected keyword argument '{name}'"
                )));
            }
        }
    }

    let shift = match shift {
        None => BigInt::from(0),
        Some(TimeCalendarValue::Integer(value)) => value.clone(),
        Some(TimeCalendarValue::Text(_)) => {
            return Err(TimeCalendarCacheError::UnsupportedValueType {
                parameter: "shift",
                expected: "an integer",
            });
        }
    };
    let region = match region {
        None => Region::Cn,
        Some(TimeCalendarValue::Text(value)) => match value.as_str() {
            "cn" => Region::Cn,
            "us" => Region::Us,
            "tw" => Region::Tw,
            _ => {
                return Err(TimeCalendarCacheError::UnsupportedRegion {
                    region: value.clone(),
                });
            }
        },
        Some(TimeCalendarValue::Integer(_)) => {
            return Err(TimeCalendarCacheError::UnsupportedValueType {
                parameter: "region",
                expected: "a string",
            });
        }
    };
    Ok((shift, region))
}

fn binding_error(message: impl Into<String>) -> TimeCalendarCacheError {
    TimeCalendarCacheError::Binding {
        message: message.into(),
    }
}
