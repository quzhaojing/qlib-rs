use chrono::{NaiveDateTime, TimeDelta};
use domain_core::{
    CachedCalendarProvider, CalendarCache, CalendarCatalog, CalendarLoadError, CalendarLoader,
    ExecutionCalendar, ExecutionCalendarError as Error, ExecutionCalendarProvider,
};
use serde_json::{Value, json};
use std::{
    num::NonZeroUsize,
    process::Command,
    sync::{Arc, Mutex},
};

fn time(seconds: i64) -> NaiveDateTime {
    NaiveDateTime::parse_from_str("2024-01-02 09:30:00", "%Y-%m-%d %H:%M:%S").unwrap()
        + TimeDelta::seconds(seconds)
}

struct Loader {
    values: Arc<[NaiveDateTime]>,
    reads: Arc<Mutex<Vec<(String, bool)>>>,
    fail: Mutex<bool>,
}
impl CalendarLoader for Loader {
    fn load_calendar(&self, frequency: &str, future: bool) -> Result<Arc<[NaiveDateTime]>, Error> {
        self.reads
            .lock()
            .unwrap()
            .push((frequency.to_owned(), future));
        if std::mem::take(&mut *self.fail.lock().unwrap()) {
            return Err(Error::Provider("load".to_owned()));
        }
        Ok(Arc::clone(&self.values))
    }
}

fn loader(values: &[i64]) -> Arc<Loader> {
    Arc::new(Loader {
        values: values.iter().copied().map(time).collect::<Vec<_>>().into(),
        reads: Arc::new(Mutex::new(Vec::new())),
        fail: Mutex::new(false),
    })
}

fn oracle() -> Value {
    let output = Command::new("python")
        .args([
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/calendar_provider_contract.py"
            ),
            r"D:\code\github\qlib\qlib",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn calendar_queries_and_direct_locate_match_upstream_including_duplicates_and_none() {
    let mut rows = Vec::new();
    for values in [vec![], vec![0, 60, 120], vec![0, 0, 60, 120, 120]] {
        let source = loader(&values);
        let provider =
            CachedCalendarProvider::new(source.clone(), Arc::new(CalendarCache::new(None)));
        for (start, end) in [
            (None, None),
            (Some(-60), Some(-30)),
            (Some(0), Some(0)),
            (Some(30), Some(90)),
            (Some(120), Some(60)),
            (Some(180), Some(240)),
            (None, Some(0)),
            (Some(0), None),
        ] {
            let calendar =
                match provider.calendar_range(start.map(time), end.map(time), "day", true) {
                    Ok(view) => json!(
                        view.timestamps()
                            .iter()
                            .map(|value| (*value - time(0)).num_seconds())
                            .collect::<Vec<_>>()
                    ),
                    Err(Error::Index(_)) => json!("IndexError"),
                    Err(error) => panic!("unexpected calendar error: {error}"),
                };
            let locate = match provider.locate_index(start.map(time), end.map(time), "day", true) {
                Ok(indices) => json!(indices),
                Err(Error::Provider(message)) if message.contains("future date") => {
                    json!("IndexError")
                }
                Err(error) => panic!("unexpected locate error: {error}"),
            };
            rows.push(json!({"values": values, "start": start, "end": end, "calendar": calendar, "locate_index": locate}));
        }
        assert_eq!(source.reads.lock().unwrap().len(), 1);
    }
    assert_eq!(json!(rows), oracle()["rows"]);
}

#[test]
fn shared_length_lru_preserves_raw_frequency_keys_and_retries_failed_loads() {
    let source = loader(&[0, 60, 120]);
    let cache = Arc::new(CalendarCache::new(NonZeroUsize::new(2)));
    let provider = CachedCalendarProvider::new(source.clone(), Arc::clone(&cache));
    for (frequency, future) in [
        ("day", false),
        ("1day", false),
        ("day", false),
        ("day", true),
        ("1day", false),
    ] {
        provider.calendar(frequency, future).unwrap();
    }
    let oracle = oracle();
    assert_eq!(json!(*source.reads.lock().unwrap()), oracle["lru_reads"]);
    cache.clear().unwrap();
    source.reads.lock().unwrap().clear();
    *source.fail.lock().unwrap() = true;
    assert_eq!(
        provider.calendar("day", false).unwrap_err(),
        Error::Provider("load".to_owned())
    );
    let first = provider.calendar("day", false).unwrap();
    let second = provider.calendar("day", false).unwrap();
    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(json!(*source.reads.lock().unwrap()), oracle["retry_reads"]);
    let other = loader(&[600, 660]);
    let second_provider = CachedCalendarProvider::new(other.clone(), Arc::clone(&cache));
    assert!(Arc::ptr_eq(
        &first,
        &second_provider.calendar("day", false).unwrap()
    ));
    assert!(other.reads.lock().unwrap().is_empty());
    cache.clear().unwrap();
    assert_eq!(
        &*second_provider.calendar("day", false).unwrap(),
        &[time(600), time(660)]
    );
    assert_eq!(&*first, &[time(0), time(60), time(120)]);
}

#[test]
fn catalog_and_cached_provider_drive_real_execution_windows_without_copying_full_buffers() {
    let timestamps: Arc<[NaiveDateTime]> = vec![time(0), time(60), time(120)].into();
    let mut catalog = CalendarCatalog::default();
    assert!(
        catalog
            .load_calendar("1min", true)
            .unwrap_err()
            .to_string()
            .contains("unavailable")
    );
    catalog.insert("1min".to_owned(), true, Arc::clone(&timestamps));
    catalog.insert("1min".to_owned(), true, Arc::clone(&timestamps));
    let provider = Arc::new(CachedCalendarProvider::new(
        Arc::new(catalog),
        Arc::new(CalendarCache::new(None)),
    ));
    assert!(Arc::ptr_eq(
        &provider.calendar("1min", true).unwrap(),
        &timestamps
    ));
    let slice = provider
        .calendar_range(Some(time(60)), Some(time(120)), "1min", true)
        .unwrap();
    assert_eq!(slice.clone().timestamps(), &[time(60), time(120)]);
    assert_eq!(&*slice.into_timestamps(), &[time(60), time(120)]);
    let mut cursor =
        ExecutionCalendar::new(provider, "1min".to_owned(), Some(time(0)), Some(time(60))).unwrap();
    assert_eq!(cursor.step_time(None, 0).unwrap(), (time(0), time(59)));
    cursor.step().unwrap();
    assert_eq!(cursor.step_time(None, 0).unwrap(), (time(60), time(119)));
    cursor.step().unwrap();
    assert!(cursor.finished());
}

struct PanickingLoader;
#[test]
fn concurrent_cache_misses_publish_one_complete_shared_calendar() {
    let source = loader(&[0, 60, 120]);
    let provider = Arc::new(CachedCalendarProvider::new(
        source.clone(),
        Arc::new(CalendarCache::new(None)),
    ));
    let barrier = Arc::new(std::sync::Barrier::new(8));
    let threads: Vec<_> = (0..8)
        .map(|_| {
            let provider = Arc::clone(&provider);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                provider.calendar("day", false).unwrap()
            })
        })
        .collect();
    for thread in threads {
        assert!(Arc::ptr_eq(&thread.join().unwrap(), &source.values));
    }
    assert_eq!(*source.reads.lock().unwrap(), [("day".to_owned(), false)]);
}

impl CalendarLoader for PanickingLoader {
    fn load_calendar(&self, _: &str, _: bool) -> Result<Arc<[NaiveDateTime]>, Error> {
        panic!("injected load panic")
    }
}

#[test]
fn poisoned_cache_is_never_silently_recovered() {
    let cache = Arc::new(CalendarCache::new(None));
    let provider = Arc::new(CachedCalendarProvider::new(
        Arc::new(PanickingLoader),
        Arc::clone(&cache),
    ));
    let thread_provider = Arc::clone(&provider);
    assert!(
        std::thread::spawn(move || thread_provider.calendar("day", false))
            .join()
            .is_err()
    );
    let error = Error::Provider("calendar cache lock poisoned".to_owned());
    assert_eq!(provider.calendar("day", false).unwrap_err(), error);
    assert_eq!(
        provider.locate_index(None, None, "day", false).unwrap_err(),
        error
    );
    assert_eq!(cache.clear().unwrap_err(), error);
}

struct RawBackedLoader {
    cache: Arc<CalendarCache>,
    events: Mutex<Vec<Value>>,
}

impl RawBackedLoader {
    fn raw(&self, uri: &str) -> Result<Arc<[String]>, CalendarLoadError> {
        self.events.lock().unwrap().push(json!(["check", uri]));
        self.cache.get_or_load_raw(uri, &mut || {
            self.events.lock().unwrap().push(json!(["read", uri]));
            Ok(vec![
                "2024-01-02 09:30:00".to_owned(),
                "2024-01-02 09:31:00".to_owned(),
            ]
            .into())
        })
    }
}

impl CalendarLoader for RawBackedLoader {
    fn load_calendar(&self, frequency: &str, future: bool) -> Result<Arc<[NaiveDateTime]>, Error> {
        self.events
            .lock()
            .unwrap()
            .push(json!(["parse", frequency, future]));
        self.raw(frequency)
            .map_err(|error| Error::Provider(error.to_string()))?
            .iter()
            .map(|line| {
                NaiveDateTime::parse_from_str(line, "%Y-%m-%d %H:%M:%S")
                    .map_err(|error| Error::Provider(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Arc::from)
    }
}

#[test]
fn raw_and_parsed_entries_share_upstream_eviction_order_without_recursive_locking() {
    let mut results = Vec::new();
    for capacity in [1, 2, 3] {
        let cache = Arc::new(CalendarCache::new(NonZeroUsize::new(capacity)));
        let source = Arc::new(RawBackedLoader {
            cache: Arc::clone(&cache),
            events: Mutex::new(Vec::new()),
        });
        let provider = CachedCalendarProvider::new(source.clone(), Arc::clone(&cache));
        for (operation, key) in [
            ("parsed", "day"),
            ("raw", "day"),
            ("parsed", "day"),
            ("raw", "other"),
            ("parsed", "day"),
            ("parsed", "1day"),
            ("raw", "day"),
            ("parsed", "day"),
        ] {
            if operation == "parsed" {
                assert_eq!(
                    &*provider.calendar(key, false).unwrap(),
                    &[time(0), time(60)]
                );
            } else {
                assert_eq!(source.raw(key).unwrap().len(), 2);
            }
        }
        results.push(source.events.lock().unwrap().clone());
        let retained = source.raw("day").unwrap();
        cache.clear().unwrap();
        let replaced = source.raw("day").unwrap();
        assert_eq!(retained, replaced);
        assert!(!Arc::ptr_eq(&retained, &replaced));
    }
    assert_eq!(json!(results), oracle()["mixed"]);
}

#[test]
fn raw_read_failures_retry_and_panics_poison_without_publishing_partial_data() {
    let cache = Arc::new(CalendarCache::new(None));
    let error = CalendarLoadError::Other("read".to_owned());
    assert_eq!(
        cache
            .get_or_load_raw("day", &mut || Err(error.clone()))
            .unwrap_err(),
        error
    );
    let lines: Arc<[String]> = vec!["line".to_owned()].into();
    let loaded = cache
        .get_or_load_raw("day", &mut || Ok(Arc::clone(&lines)))
        .unwrap();
    let cached = cache
        .get_or_load_raw("day", &mut || panic!("hit must not read"))
        .unwrap();
    assert!(Arc::ptr_eq(&loaded, &cached));
    let other = Arc::clone(&cache);
    assert!(
        std::thread::spawn(move || other.get_or_load_raw("panic", &mut || panic!("read panic")))
            .join()
            .is_err()
    );
    let poisoned = CalendarLoadError::Other("calendar cache lock poisoned".to_owned());
    assert_eq!(
        cache
            .get_or_load_raw("day", &mut || panic!("poison must not read"))
            .unwrap_err(),
        poisoned
    );
    assert_eq!(
        cache.clear().unwrap_err(),
        Error::Provider(poisoned.to_string())
    );
    assert_eq!(&*loaded, &["line"]);
}

#[test]
fn native_and_text_raw_keys_share_identity_and_eviction_budget() {
    use std::ffi::OsStr;

    let cache = CalendarCache::new(NonZeroUsize::new(1));
    let lines: Arc<[String]> = vec!["original".into()].into();
    let first = cache
        .get_or_load_raw("orig_fileC:\\data", &mut || Ok(Arc::clone(&lines)))
        .unwrap();
    let native = cache
        .get_or_load_raw_native(OsStr::new("orig_fileC:\\data"), &mut || {
            panic!("native caller must reuse the text entry")
        })
        .unwrap();
    assert!(Arc::ptr_eq(&first, &native));
    let replacement: Arc<[String]> = vec!["replacement".into()].into();
    cache
        .get_or_load_raw_native(OsStr::new("other"), &mut || Ok(Arc::clone(&replacement)))
        .unwrap();
    let other = cache
        .get_or_load_raw("other", &mut || {
            panic!("text caller must reuse native entry")
        })
        .unwrap();
    assert!(Arc::ptr_eq(&other, &replacement));
    let reloaded = cache
        .get_or_load_raw("orig_fileC:\\data", &mut || Ok(Arc::clone(&replacement)))
        .unwrap();
    assert!(Arc::ptr_eq(&reloaded, &replacement));
    assert_eq!(&*first, &["original"]);
}

#[test]
fn concurrent_raw_misses_publish_one_retained_buffer() {
    let cache = Arc::new(CalendarCache::new(None));
    let reads = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let barrier = Arc::new(std::sync::Barrier::new(8));
    let lines: Arc<[String]> = vec!["line".to_owned()].into();
    let threads: Vec<_> = (0..8)
        .map(|_| {
            let cache = Arc::clone(&cache);
            let reads = Arc::clone(&reads);
            let barrier = Arc::clone(&barrier);
            let lines = Arc::clone(&lines);
            std::thread::spawn(move || {
                barrier.wait();
                cache
                    .get_or_load_raw("day", &mut || {
                        reads.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        Ok(Arc::clone(&lines))
                    })
                    .unwrap()
            })
        })
        .collect();
    for thread in threads {
        assert!(Arc::ptr_eq(&thread.join().unwrap(), &lines));
    }
    assert_eq!(reads.load(std::sync::atomic::Ordering::SeqCst), 1);
}
