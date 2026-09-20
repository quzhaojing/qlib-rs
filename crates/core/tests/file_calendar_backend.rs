use domain_core::{
    CalendarBackendSource, CalendarCache, CalendarLoadError, CalendarTextEncoding, CalendarValue,
    DEFAULT_DATA_FREQUENCY, DataPathManager, FileCalendarBackend, IsoCalendarTimestampDecoder,
    LocalCalendarLoader, Region, TracingCalendarWarnings,
};
use indexmap::IndexMap;
use serde_json::{Value, json};
use std::{path::Path, process::Command, sync::Arc};

fn backend(root: &Path) -> FileCalendarBackend {
    FileCalendarBackend {
        paths: DataPathManager::new(
            IndexMap::from([(DEFAULT_DATA_FREQUENCY.into(), root.to_str().unwrap().into())]),
            IndexMap::new(),
        ),
        system: "Windows".into(),
        region: Region::Cn,
        minute_shift: 0.into(),
        text_decoder: Arc::new(CalendarTextEncoding::Utf8),
        timestamp_decoder: Arc::new(IsoCalendarTimestampDecoder),
        cache: Arc::new(CalendarCache::new(None)),
        enable_read_cache: true,
    }
}

#[cfg(windows)]
#[test]
fn native_roots_with_identical_lossy_display_have_independent_cache_entries() {
    use domain_core::path_initialization::{ConfigPathValue, PathConfiguration, PathSetting};
    use std::{ffi::OsString, os::windows::ffi::OsStringExt};

    let directory = tempfile::tempdir().unwrap();
    let cache = Arc::new(CalendarCache::new(std::num::NonZeroUsize::new(2)));
    let mut backends = Vec::new();
    for (unit, text) in [(0xd800, "first"), (0xd801, "second")] {
        let root = directory.path().join(OsString::from_wide(&[unit]));
        std::fs::create_dir_all(root.join("calendars")).unwrap();
        std::fs::write(root.join("calendars/day.txt"), format!("{text}\n")).unwrap();
        let mut config = PathConfiguration {
            provider_uri: PathSetting::Scalar(ConfigPathValue::Text(root.into_os_string())),
            mount_path: PathSetting::Scalar(ConfigPathValue::Null),
        };
        config.resolve_paths_native().unwrap();
        let PathSetting::Mapping(providers) = config.provider_uri else {
            panic!("successful initialization must normalize provider maps");
        };
        let providers = providers
            .into_iter()
            .map(|(key, value)| {
                let native = match value {
                    ConfigPathValue::Text(text) => text,
                    ConfigPathValue::Path(path) => path.into_os_string(),
                    other => panic!("unexpected configured provider: {other:?}"),
                };
                (key, native)
            })
            .collect();
        let mut source = backend(directory.path());
        source.paths = DataPathManager::from_native(providers, IndexMap::new());
        source.cache = Arc::clone(&cache);
        backends.push(source);
    }
    let first = backends[0].resolve_file("day", false).unwrap().1;
    let second = backends[1].resolve_file("day", false).unwrap().1;
    assert_ne!(first.as_os_str(), second.as_os_str());
    assert_eq!(first.to_string_lossy(), second.to_string_lossy());
    for _ in 0..2 {
        assert_eq!(
            values(&backends[0], "day", false).unwrap(),
            vec![CalendarValue::Text("first".into())]
        );
        assert_eq!(
            values(&backends[1], "day", false).unwrap(),
            vec![CalendarValue::Text("second".into())]
        );
    }
    std::fs::write(&first, "changed\n").unwrap();
    assert_eq!(
        values(&backends[0], "day", false).unwrap(),
        vec![CalendarValue::Text("first".into())]
    );
    cache.clear().unwrap();
    assert_eq!(
        values(&backends[0], "day", false).unwrap(),
        vec![CalendarValue::Text("changed".into())]
    );
    assert_eq!(
        values(&backends[1], "day", false).unwrap(),
        vec![CalendarValue::Text("second".into())]
    );
}

fn values(
    backend: &FileCalendarBackend,
    frequency: &str,
    future: bool,
) -> Result<Vec<CalendarValue>, CalendarLoadError> {
    backend.data(frequency, future)?.collect()
}

#[test]
fn native_calendar_writes_preserve_the_shared_read_cache_until_explicit_clear() {
    use domain_core::{CalendarTextArray, CalendarWriteMode, write_calendar_file};

    let directory = tempfile::tempdir().unwrap();
    let calendars = directory.path().join("calendars");
    std::fs::create_dir(&calendars).unwrap();
    let path = calendars.join("day.txt");
    let original = CalendarTextArray {
        shape: vec![2],
        values: vec!["2024-01-02".into(), "2024-01-03".into()],
    };
    write_calendar_file(&path, &original, CalendarWriteMode::Overwrite).unwrap();
    let backend = backend(directory.path());
    let cached = values(&backend, "day", false).unwrap();
    assert_eq!(
        cached,
        vec![
            CalendarValue::Text("2024-01-02".into()),
            CalendarValue::Text("2024-01-03".into())
        ]
    );
    let additional = CalendarTextArray {
        shape: vec![1],
        values: vec!["2024-01-04".into()],
    };
    write_calendar_file(&path, &additional, CalendarWriteMode::Append).unwrap();
    assert_eq!(
        std::fs::read(&path).unwrap(),
        b"2024-01-02\n2024-01-03\n2024-01-04\n"
    );
    assert_eq!(values(&backend, "day", false).unwrap(), cached);
    let empty = CalendarTextArray {
        shape: vec![0],
        values: vec![],
    };
    write_calendar_file(&path, &empty, CalendarWriteMode::Overwrite).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), b"");
    assert_eq!(values(&backend, "day", false).unwrap(), cached);
    backend.cache.clear().unwrap();
    assert_eq!(values(&backend, "day", false).unwrap(), vec![]);
}

fn snapshot(backend: &FileCalendarBackend, frequency: &str, future: bool) -> Value {
    match values(backend, frequency, future) {
        Ok(values) => {
            let (selected, path) = backend.resolve_file(frequency, future).unwrap();
            let values: Vec<_> = values
                .into_iter()
                .map(|value| match value {
                    CalendarValue::Text(text) => json!(["text", text]),
                    CalendarValue::Timestamp(time) => {
                        json!(["timestamp", time.format("%Y-%m-%dT%H:%M:%S").to_string()])
                    }
                })
                .collect();
            json!({"values": values, "selected": selected.to_string(), "file": path.file_name().unwrap().to_str().unwrap()})
        }
        Err(CalendarLoadError::Value(_)) => json!({"error": "Value"}),
        Err(CalendarLoadError::Other(_)) => json!({"error": "Other"}),
    }
}

#[test]
fn configured_file_backend_matches_source_frequency_and_resampling() {
    let output = Command::new("python")
        .env("PYTHONUTF8", "1")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/file_calendar_backend_contract.py"
        ))
        .arg("D:/code/github/qlib/qlib")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let expected: Value = serde_json::from_slice(&output.stdout).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let calendars = directory.path().join("calendars");
    std::fs::create_dir(&calendars).unwrap();
    std::fs::write(
        calendars.join("1min.txt"),
        "2024-01-02 09:31:00\n2024-01-02 09:30:00\n2024-01-02 09:31:00\n2024-01-03 09:30:00\n",
    )
    .unwrap();
    std::fs::write(calendars.join("1min_future.txt"), "bad-date\n").unwrap();
    std::fs::write(calendars.join("ignore.bin"), "").unwrap();
    let backend = backend(directory.path());
    for case in expected.as_array().unwrap() {
        assert_eq!(
            snapshot(
                &backend,
                case[0].as_str().unwrap(),
                case[1].as_bool().unwrap()
            ),
            case[2]
        );
    }
    let loader = LocalCalendarLoader::new(
        Arc::new(backend),
        Arc::new(IsoCalendarTimestampDecoder),
        Arc::new(TracingCalendarWarnings),
    );
    assert!(matches!(
        loader.load("1min", true),
        Err(CalendarLoadError::Value(_))
    ));
    assert_eq!(loader.load("5min", true).unwrap().len(), 2);
}

#[test]
fn existence_is_checked_before_cached_reads_and_cache_can_be_disabled() {
    let directory = tempfile::tempdir().unwrap();
    let mut backend = backend(directory.path());
    assert!(backend.supported_frequencies().unwrap().is_empty());
    assert!(matches!(
        values(&backend, "day", false),
        Err(CalendarLoadError::Value(_))
    ));
    std::fs::create_dir(directory.path().join("calendars")).unwrap();
    let path = directory.path().join("calendars/day.txt");
    std::fs::write(&path, "old\n").unwrap();
    assert_eq!(
        values(&backend, "day", false).unwrap(),
        [CalendarValue::Text("old".into())]
    );
    std::fs::write(&path, "new\n").unwrap();
    assert_eq!(
        values(&backend, "day", false).unwrap(),
        [CalendarValue::Text("old".into())]
    );
    backend.enable_read_cache = false;
    assert_eq!(
        values(&backend, "day", false).unwrap(),
        [CalendarValue::Text("new".into())]
    );
    backend.enable_read_cache = true;
    // Map-based discovery keeps the frequency available after file deletion.
    backend.paths.provider_uri =
        IndexMap::from([("day".into(), directory.path().to_str().unwrap().into())]);
    std::fs::remove_file(&path).unwrap();
    assert!(matches!(
        values(&backend, "day", false),
        Err(CalendarLoadError::Value(_))
    ));
    assert!(!path.exists());
    std::fs::write(&path, "restored\n").unwrap();
    assert_eq!(
        values(&backend, "day", false).unwrap(),
        [CalendarValue::Text("old".into())]
    );
    backend.cache.clear().unwrap();
    assert_eq!(
        values(&backend, "day", false).unwrap(),
        [CalendarValue::Text("restored".into())]
    );
}

#[test]
fn configuration_read_decode_and_resample_failures_keep_their_class() {
    let directory = tempfile::tempdir().unwrap();
    let mut backend = backend(directory.path());
    let calendars = directory.path().join("calendars");
    std::fs::create_dir(&calendars).unwrap();
    std::fs::write(calendars.join("nonsense.txt"), "").unwrap();
    assert!(matches!(
        backend.supported_frequencies(),
        Err(CalendarLoadError::Value(_))
    ));
    std::fs::remove_file(calendars.join("nonsense.txt")).unwrap();
    std::fs::write(calendars.join("0day.txt"), "2024-01-02").unwrap();
    assert!(matches!(
        values(&backend, "0min", false),
        Err(CalendarLoadError::Value(_))
    ));
    backend.paths.provider_uri =
        IndexMap::from([("day".into(), directory.path().to_str().unwrap().into())]);
    std::fs::create_dir(calendars.join("day.txt")).unwrap();
    assert!(matches!(
        values(&backend, "day", false),
        Err(CalendarLoadError::Other(_))
    ));
    std::fs::remove_dir(calendars.join("day.txt")).unwrap();
    std::fs::write(calendars.join("day.txt"), b"\xff").unwrap();
    assert!(matches!(
        values(&backend, "day", false),
        Err(CalendarLoadError::Value(_))
    ));
    backend.enable_read_cache = false;
    assert!(matches!(
        values(&backend, "day", false),
        Err(CalendarLoadError::Value(_))
    ));
    backend.paths.provider_uri =
        IndexMap::from([("1day".into(), directory.path().to_str().unwrap().into())]);
    assert!(matches!(
        values(&backend, "day", false),
        Err(CalendarLoadError::Other(_))
    ));
    backend.paths.provider_uri =
        IndexMap::from([(DEFAULT_DATA_FREQUENCY.into(), "host:/missing".into())]);
    assert!(matches!(
        backend.supported_frequencies(),
        Err(CalendarLoadError::Other(_))
    ));
    backend
        .paths
        .provider_uri
        .insert("invalid".into(), "root".into());
    assert!(matches!(
        backend.supported_frequencies(),
        Err(CalendarLoadError::Value(_))
    ));
    assert!(matches!(
        backend.resolve_file("day", false),
        Err(CalendarLoadError::Value(_))
    ));
    assert!(matches!(
        values(&backend, "day", false),
        Err(CalendarLoadError::Value(_))
    ));
}

#[cfg(windows)]
#[test]
fn native_glob_filters_suffix_before_rejecting_non_unicode_frequency() {
    use std::{ffi::OsString, os::windows::ffi::OsStringExt};
    let directory = tempfile::tempdir().unwrap();
    let calendars = directory.path().join("calendars");
    std::fs::create_dir(&calendars).unwrap();
    let name = |suffix: &str| {
        OsString::from_wide(
            &std::iter::once(0xd800)
                .chain(suffix.encode_utf16())
                .collect::<Vec<_>>(),
        )
    };
    std::fs::write(calendars.join(name(".bin")), "").unwrap();
    std::fs::write(calendars.join(name("_future.txt")), "").unwrap();
    std::fs::write(calendars.join("DAY.TXT"), "").unwrap();
    let backend = backend(directory.path());
    assert_eq!(
        backend.supported_frequencies().unwrap(),
        ["day".parse().unwrap()]
    );
    std::fs::write(calendars.join(name(".txt")), "").unwrap();
    assert!(matches!(
        backend.supported_frequencies(),
        Err(CalendarLoadError::Value(_))
    ));
}
