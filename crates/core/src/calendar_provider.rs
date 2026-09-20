//! Cached typed calendar loading, slicing and index lookup from `CalendarProvider`.

use std::{
    collections::HashMap,
    ffi::{OsStr, OsString},
    num::NonZeroUsize,
    ops::Range,
    sync::{Arc, Mutex},
};

use chrono::NaiveDateTime;
use lru::LruCache;

use crate::{CalendarLoadError, ExecutionCalendarError, ExecutionCalendarProvider};

/// Native calendar source. Timestamp parsing and local/remote storage belong here,
/// outside the shared cache. Returned arrays retain source order and duplicates.
pub trait CalendarLoader: Send + Sync {
    /// # Errors
    /// Returns loading failures; failures are not inserted into the cache.
    fn load_calendar(
        &self,
        frequency: &str,
        future: bool,
    ) -> Result<Arc<[NaiveDateTime]>, ExecutionCalendarError>;
}

/// Explicitly supplied, already decoded calendars, usable with Arrow/file data ingestion.
/// Frequency spellings are deliberately not normalized.
#[derive(Default)]
pub struct CalendarCatalog {
    calendars: HashMap<(String, bool), Arc<[NaiveDateTime]>>,
}

impl CalendarCatalog {
    /// Add or replace a source calendar. A provider's existing cache is unaffected.
    pub fn insert(&mut self, frequency: String, future: bool, timestamps: Arc<[NaiveDateTime]>) {
        self.calendars.insert((frequency, future), timestamps);
    }
}

impl CalendarLoader for CalendarCatalog {
    fn load_calendar(
        &self,
        frequency: &str,
        future: bool,
    ) -> Result<Arc<[NaiveDateTime]>, ExecutionCalendarError> {
        self.calendars
            .get(&(frequency.to_owned(), future))
            .cloned()
            .ok_or_else(|| {
                ExecutionCalendarError::Provider(format!(
                    "calendar unavailable: {frequency}, future={future}"
                ))
            })
    }
}

struct CachedCalendar {
    timestamps: Arc<[NaiveDateTime]>,
    // Python's dictionary comprehension retains the LAST index of a repeated timestamp.
    indices: HashMap<NaiveDateTime, usize>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum CalendarCacheKey {
    Parsed(String, bool),
    Raw(OsString),
}

struct CalendarCacheEntries {
    recency: LruCache<CalendarCacheKey, ()>,
    parsed: HashMap<(String, bool), Arc<CachedCalendar>>,
    raw: HashMap<OsString, Arc<[String]>>,
}

impl CalendarCacheEntries {
    fn insert_key(&mut self, key: CalendarCacheKey) {
        if let Some((removed, ())) = self.recency.push(key, ()) {
            match removed {
                CalendarCacheKey::Parsed(frequency, future) => {
                    self.parsed.remove(&(frequency, future));
                }
                CalendarCacheKey::Raw(uri) => {
                    self.raw.remove(&uri);
                }
            }
        }
    }

    fn parsed(&mut self, frequency: &str, future: bool) -> Option<Arc<CachedCalendar>> {
        self.recency
            .get(&CalendarCacheKey::Parsed(frequency.to_owned(), future));
        self.parsed.get(&(frequency.to_owned(), future)).cloned()
    }

    fn raw(&mut self, uri: &OsStr) -> Option<Arc<[String]>> {
        self.recency.get(&CalendarCacheKey::Raw(uri.to_owned()));
        self.raw.get(uri).cloned()
    }
}

/// Share this object to reproduce the upstream calendar cache shared across providers.
/// A length bound uses LRU eviction; `None` means unbounded. This is not Python's
/// `sys.getsizeof` cache mode. Raw text and parsed calendars share one eviction budget.
/// Typed keys separate file URIs from frequency/future pairs.
pub struct CalendarCache {
    parsed_loading: Mutex<()>,
    raw_loading: Mutex<()>,
    entries: Mutex<CalendarCacheEntries>,
}

impl CalendarCache {
    #[must_use]
    pub fn new(capacity: Option<NonZeroUsize>) -> Self {
        Self {
            parsed_loading: Mutex::new(()),
            raw_loading: Mutex::new(()),
            entries: Mutex::new(CalendarCacheEntries {
                recency: capacity.map_or_else(LruCache::unbounded, LruCache::new),
                parsed: HashMap::new(),
                raw: HashMap::new(),
            }),
        }
    }

    /// # Errors
    /// Returns poisoned-cache failures without recovering partial state.
    pub fn clear(&self) -> Result<(), ExecutionCalendarError> {
        let _parsed = self.parsed_loading.lock().map_err(|_| cache_poisoned())?;
        let _raw = self.raw_loading.lock().map_err(|_| cache_poisoned())?;
        let mut entries = self.entries.lock().map_err(|_| cache_poisoned())?;
        entries.recency.clear();
        entries.parsed.clear();
        entries.raw.clear();
        Ok(())
    }

    /// Cache raw lines using the exact, unnormalized storage URI. The caller must
    /// check file existence before calling, even on hits, and resample after return.
    /// A parsed-calendar loader may call this method. Raw readers must not reenter
    /// either loader or clear the cache; lock order is parsed -> raw -> entries.
    /// Neither loader runs while the entries mutex is held. Failures are not cached.
    ///
    /// # Errors
    /// Preserves the reader's Value/Other error class so the local loader can apply
    /// its future-calendar fallback. Cache poisoning is always an Other failure.
    pub fn get_or_load_raw(
        &self,
        uri: &str,
        read: &mut dyn FnMut() -> Result<Arc<[String]>, CalendarLoadError>,
    ) -> Result<Arc<[String]>, CalendarLoadError> {
        self.get_or_load_raw_native(OsStr::new(uri), read)
    }

    /// Native-path counterpart of `get_or_load_raw`, preserving every OS string
    /// unit in cache identity. Text and native callers share the same cache and
    /// eviction budget. The same callback and lock-order restrictions apply.
    ///
    /// # Errors
    /// Preserves reader failures; poisoned cache locks return an Other failure.
    pub fn get_or_load_raw_native(
        &self,
        uri: &OsStr,
        read: &mut dyn FnMut() -> Result<Arc<[String]>, CalendarLoadError>,
    ) -> Result<Arc<[String]>, CalendarLoadError> {
        let _loading = self.raw_loading.lock().map_err(|_| raw_cache_poisoned())?;
        if let Some(lines) = self
            .entries
            .lock()
            .map_err(|_| raw_cache_poisoned())?
            .raw(uri)
        {
            return Ok(lines);
        }
        let lines = read()?;
        let mut entries = self.entries.lock().map_err(|_| raw_cache_poisoned())?;
        entries.insert_key(CalendarCacheKey::Raw(uri.to_owned()));
        entries.raw.insert(uri.to_owned(), Arc::clone(&lines));
        Ok(lines)
    }
}

/// Immutable view retaining the original cached buffer rather than copying a slice.
#[derive(Clone)]
pub struct CalendarSlice {
    timestamps: Arc<[NaiveDateTime]>,
    range: Range<usize>,
}

impl CalendarSlice {
    #[must_use]
    pub fn timestamps(&self) -> &[NaiveDateTime] {
        &self.timestamps[self.range.clone()]
    }

    /// Return the shared full buffer when possible; materialize only a proper subrange
    /// for interfaces requiring an owned, offset-free `Arc` slice.
    #[must_use]
    pub fn into_timestamps(self) -> Arc<[NaiveDateTime]> {
        if self.range == (0..self.timestamps.len()) {
            self.timestamps
        } else {
            Arc::from(self.timestamps())
        }
    }
}

/// Production implementation of calendar query/locate behavior over a native loader.
/// Source calendars are ordered, as required by upstream bisect. Cache callbacks must
/// not reenter parsed loading or clear the same cache. Raw loading is supported:
/// separate loading locks serialize publication without holding the entries mutex.
pub struct CachedCalendarProvider {
    loader: Arc<dyn CalendarLoader>,
    cache: Arc<CalendarCache>,
}

impl CachedCalendarProvider {
    #[must_use]
    pub fn new(loader: Arc<dyn CalendarLoader>, cache: Arc<CalendarCache>) -> Self {
        Self { loader, cache }
    }

    fn get_calendar(
        &self,
        frequency: &str,
        future: bool,
    ) -> Result<Arc<CachedCalendar>, ExecutionCalendarError> {
        let _loading = self
            .cache
            .parsed_loading
            .lock()
            .map_err(|_| cache_poisoned())?;
        let key = (frequency.to_owned(), future);
        if let Some(calendar) = self
            .cache
            .entries
            .lock()
            .map_err(|_| cache_poisoned())?
            .parsed(frequency, future)
        {
            return Ok(calendar);
        }
        let timestamps = self.loader.load_calendar(frequency, future)?;
        let indices = timestamps
            .iter()
            .copied()
            .enumerate()
            .map(|(index, time)| (time, index))
            .collect();
        let calendar = Arc::new(CachedCalendar {
            timestamps,
            indices,
        });
        let mut entries = self.cache.entries.lock().map_err(|_| cache_poisoned())?;
        entries.insert_key(CalendarCacheKey::Parsed(frequency.to_owned(), future));
        entries.parsed.insert(key, Arc::clone(&calendar));
        Ok(calendar)
    }

    /// Query a closed window, preserving duplicate and reversed-bound slice behavior.
    /// `None` bounds mean the first/last calendar timestamp here, whereas direct
    /// [`ExecutionCalendarProvider::locate_index`] treats them as upstream `NaT`.
    /// # Errors
    /// Returns loading, empty-calendar or index lookup failures.
    pub fn calendar_range(
        &self,
        start: Option<NaiveDateTime>,
        end: Option<NaiveDateTime>,
        frequency: &str,
        future: bool,
    ) -> Result<CalendarSlice, ExecutionCalendarError> {
        self.query(start, end, frequency, future, true)
            .map(|mut view| {
                view.range.end = view.range.end.max(view.range.start);
                view
            })
    }

    // Resolve one operation against one immutable cache snapshot. This avoids
    // mixing indices from a replacement entry with a previously returned buffer.
    fn query(
        &self,
        mut start: Option<NaiveDateTime>,
        mut end: Option<NaiveDateTime>,
        frequency: &str,
        future: bool,
        slicing: bool,
    ) -> Result<CalendarSlice, ExecutionCalendarError> {
        let calendar = self.get_calendar(frequency, future)?;
        if slicing {
            let (&first, &last) = calendar
                .timestamps
                .first()
                .zip(calendar.timestamps.last())
                .ok_or(ExecutionCalendarError::Index(0))?;
            let actual_start = start.unwrap_or(first);
            let actual_end = end.unwrap_or(last);
            if actual_start > last || actual_end < first {
                return Ok(CalendarSlice {
                    timestamps: Arc::clone(&calendar.timestamps),
                    range: 0..0,
                });
            }
            start = Some(actual_start);
            end = Some(actual_end);
        }
        let start_time = if let Some((time, _)) =
            start.and_then(|time| calendar.indices.get_key_value(&time))
        {
            *time
        } else {
            let index = start.map_or(0, |start| {
                calendar.timestamps.partition_point(|time| *time < start)
            });
            *calendar.timestamps.get(index).ok_or_else(|| ExecutionCalendarError::Provider(
                    "`start_time` uses a future date, if you want to get future trading days, you can use: `future=True`".to_owned()))?
        };
        let end_time =
            if let Some((time, _)) = end.and_then(|time| calendar.indices.get_key_value(&time)) {
                *time
            } else {
                let insertion = end.map_or(calendar.timestamps.len(), |end| {
                    calendar.timestamps.partition_point(|time| *time <= end)
                });
                let index = insertion
                    .checked_sub(1)
                    .unwrap_or(calendar.timestamps.len() - 1);
                calendar.timestamps[index]
            };
        // Both selected values come from the same loaded array used to construct this map.
        Ok(CalendarSlice {
            timestamps: Arc::clone(&calendar.timestamps),
            range: calendar.indices[&start_time]..calendar.indices[&end_time] + 1,
        })
    }
}

impl ExecutionCalendarProvider for CachedCalendarProvider {
    fn calendar(
        &self,
        frequency: &str,
        future: bool,
    ) -> Result<Arc<[NaiveDateTime]>, ExecutionCalendarError> {
        self.calendar_range(None, None, frequency, future)
            .map(CalendarSlice::into_timestamps)
    }

    fn locate_index(
        &self,
        start: Option<NaiveDateTime>,
        end: Option<NaiveDateTime>,
        frequency: &str,
        future: bool,
    ) -> Result<(i64, i64), ExecutionCalendarError> {
        self.query(start, end, frequency, future, false)
            .map(|view| {
                // A Rust slice's allocation cannot contain more than isize::MAX timestamp values.
                (
                    i64::try_from(view.range.start).expect("allocated calendar index fits i64"),
                    i64::try_from(view.range.end - 1).expect("allocated calendar index fits i64"),
                )
            })
    }
}

fn cache_poisoned() -> ExecutionCalendarError {
    ExecutionCalendarError::Provider("calendar cache lock poisoned".to_owned())
}

fn raw_cache_poisoned() -> CalendarLoadError {
    CalendarLoadError::Other("calendar cache lock poisoned".to_owned())
}

#[cfg(test)]
#[path = "../tests/support/calendar_cache_failures.rs"]
mod cache_failure_tests;
