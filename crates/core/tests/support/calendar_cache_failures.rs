use super::*;

fn poison_entries(cache: &CalendarCache) {
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _entries = cache.entries.lock().unwrap();
            panic!("injected cache mutation failure");
        }))
        .is_err()
    );
}

struct PoisonDuringLoad(Arc<CalendarCache>);

impl CalendarLoader for PoisonDuringLoad {
    fn load_calendar(
        &self,
        _: &str,
        _: bool,
    ) -> Result<Arc<[NaiveDateTime]>, ExecutionCalendarError> {
        poison_entries(&self.0);
        Ok(Arc::from([]))
    }
}

#[test]
fn entry_poison_before_lookup_and_between_load_and_publication_is_not_recovered() {
    let cache = Arc::new(CalendarCache::new(None));
    let provider =
        CachedCalendarProvider::new(Arc::new(CalendarCatalog::default()), Arc::clone(&cache));
    poison_entries(&cache);
    assert_eq!(cache.clear().unwrap_err(), cache_poisoned());
    assert_eq!(
        provider.calendar("day", false).unwrap_err(),
        cache_poisoned()
    );
    assert_eq!(
        cache
            .get_or_load_raw("day", &mut || panic!("must not read"))
            .unwrap_err(),
        raw_cache_poisoned()
    );

    let cache = Arc::new(CalendarCache::new(None));
    assert_eq!(
        cache
            .get_or_load_raw("day", &mut || {
                poison_entries(&cache);
                Ok(Arc::from([]))
            })
            .unwrap_err(),
        raw_cache_poisoned()
    );
    let cache = Arc::new(CalendarCache::new(None));
    let provider =
        CachedCalendarProvider::new(Arc::new(PoisonDuringLoad(Arc::clone(&cache))), cache);
    assert_eq!(
        provider.calendar("day", false).unwrap_err(),
        cache_poisoned()
    );
}

fn poison_loading(lock: &Mutex<()>) {
    assert!(
        std::panic::catch_unwind(|| {
            let _loading = lock.lock().unwrap();
            panic!("injected loader panic");
        })
        .is_err()
    );
}

#[test]
fn complete_cache_lifecycle_preserves_values_retries_identity_and_loading_failures() {
    let cache = Arc::new(CalendarCache::new(None));
    let lines: Arc<[String]> = vec!["2024-01-02".to_owned()].into();
    let value_error = CalendarLoadError::Value("decode".to_owned());
    assert_eq!(
        cache
            .get_or_load_raw("file", &mut || Err(value_error.clone()))
            .unwrap_err(),
        value_error
    );
    let first = cache
        .get_or_load_raw("file", &mut || Ok(Arc::clone(&lines)))
        .unwrap();
    let second = cache
        .get_or_load_raw("file", &mut || panic!("cache hit must not read"))
        .unwrap();
    assert!(Arc::ptr_eq(&first, &second));

    let timestamps: Arc<[NaiveDateTime]> = vec![
        chrono::NaiveDate::from_ymd_opt(2024, 1, 2)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap(),
    ]
    .into();
    let mut catalog = CalendarCatalog::default();
    catalog.insert("day".to_owned(), false, Arc::clone(&timestamps));
    let provider = CachedCalendarProvider::new(Arc::new(catalog), Arc::clone(&cache));
    assert!(
        provider
            .get_calendar("missing", false)
            .err()
            .unwrap()
            .to_string()
            .contains("unavailable")
    );
    let first_calendar = provider.get_calendar("day", false).unwrap();
    let second_calendar = provider.get_calendar("day", false).unwrap();
    assert!(Arc::ptr_eq(&first_calendar, &second_calendar));
    assert_eq!(first_calendar.timestamps, timestamps);
    assert_eq!(
        first_calendar.indices.values().copied().collect::<Vec<_>>(),
        [0]
    );
    cache.clear().unwrap();
    assert_eq!(first.as_ref(), ["2024-01-02"]);
    assert!(!Arc::ptr_eq(
        &first_calendar,
        &provider.get_calendar("day", false).unwrap()
    ));

    poison_loading(&cache.raw_loading);
    assert_eq!(
        cache
            .get_or_load_raw("file", &mut || panic!("poison must not read"))
            .unwrap_err(),
        raw_cache_poisoned()
    );
    assert_eq!(cache.clear().unwrap_err(), cache_poisoned());
    poison_loading(&cache.parsed_loading);
    assert_eq!(
        provider.get_calendar("day", false).err().unwrap(),
        cache_poisoned()
    );
    assert_eq!(cache.clear().unwrap_err(), cache_poisoned());
}
