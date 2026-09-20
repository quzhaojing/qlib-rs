use std::{
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
};

use domain_core::{
    CalendarCache, CalendarLoadError, CalendarTextArray, CalendarTextEncoding, CalendarValue,
    CalendarWriteMode, DEFAULT_DATA_FREQUENCY, DataPathManager, FileCalendarBackend,
    FileCalendarStorage, IsoCalendarTimestampDecoder, Region,
};
use indexmap::IndexMap;
use serde_json::{Value, json};

fn storage(root: &Path, frequency: &str, named: bool) -> FileCalendarStorage {
    let key = if named {
        frequency
    } else {
        DEFAULT_DATA_FREQUENCY
    };
    FileCalendarStorage::new(
        FileCalendarBackend {
            paths: DataPathManager::from_native(
                IndexMap::from([(key.into(), root.as_os_str().to_owned())]),
                IndexMap::new(),
            ),
            system: "Windows".into(),
            region: Region::Cn,
            minute_shift: 0.into(),
            text_decoder: Arc::new(CalendarTextEncoding::Utf8),
            timestamp_decoder: Arc::new(IsoCalendarTimestampDecoder),
            cache: Arc::new(CalendarCache::new(None)),
            enable_read_cache: true,
        },
        frequency.into(),
        false,
    )
}

fn sequence_outcome(result: Result<Value, CalendarLoadError>) -> Value {
    match result {
        Ok(value) => json!({"value": value}),
        Err(CalendarLoadError::Value(_)) => json!({"error": "Value"}),
        Err(CalendarLoadError::Other(_)) => json!({"error": "Other"}),
    }
}

#[test]
#[cfg(windows)]
fn storage_frequency_enumeration_matches_native_source_names_and_refresh() {
    use std::{
        ffi::OsString,
        os::windows::ffi::{OsStrExt, OsStringExt},
    };
    let output = Command::new("python")
        .env("PYTHONUTF8", "1")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/calendar_storage_frequencies_contract.py"
        ))
        .arg("D:/code/github/qlib/qlib")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(cases.len(), 20);
    for case in cases {
        let directory = tempfile::tempdir().unwrap();
        let calendar = directory.path().join("calendars");
        let mode = case["mode"].as_str().unwrap();
        match mode {
            "missing" => (),
            "not_directory" => std::fs::write(&calendar, b"not a directory").unwrap(),
            _ => std::fs::create_dir(&calendar).unwrap(),
        }
        if matches!(mode, "files" | "directories") {
            for name in case["names"].as_array().unwrap() {
                let units: Vec<u16> = serde_json::from_value(name.clone()).unwrap();
                let path = calendar.join(OsString::from_wide(&units));
                if mode == "files" {
                    std::fs::write(path, b"unchanged").unwrap();
                } else {
                    std::fs::create_dir(path).unwrap();
                }
            }
        }
        let mut store = storage(
            directory.path(),
            case["frequency"].as_str().unwrap(),
            case["named"].as_bool().unwrap(),
        );
        for attempt in 0..2 {
            if attempt == 1 && calendar.is_dir() {
                std::fs::write(calendar.join("added_future.txt"), b"new").unwrap();
            }
            let result = sequence_outcome(store.storage_frequencies().map(|names| {
                json!(
                    names
                        .into_iter()
                        .map(|name| name.encode_wide().collect::<Vec<_>>())
                        .collect::<Vec<_>>()
                )
            }));
            assert_eq!(
                result, case["results"][attempt],
                "{case}, attempt {attempt}"
            );
            assert!(
                !calendar.join("day.txt").exists(),
                "enumeration must not create the selected calendar"
            );
        }
    }
    let directory = tempfile::tempdir().unwrap();
    let mut store = storage(directory.path(), "day", true);
    store
        .backend
        .paths
        .provider_uri
        .insert("day".into(), "host:/missing".into());
    assert!(matches!(
        store.storage_frequencies(),
        Err(CalendarLoadError::Other(_))
    ));
}

#[test]
fn insertion_matches_source_numeric_conversion_width_bounds_and_cache() {
    let output = Command::new("python")
        .env("PYTHONUTF8", "1")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/calendar_insert_contract.py"
        ))
        .arg("D:/code/github/qlib/qlib")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(cases.len(), 2484);
    let directory = tempfile::tempdir().unwrap();
    std::fs::create_dir(directory.path().join("calendars")).unwrap();
    let path = directory.path().join("calendars/day.txt");
    for (case_number, case) in cases.iter().enumerate() {
        if path.exists() {
            std::fs::remove_file(&path).unwrap();
        }
        if !case["initial"].is_null() {
            let rows: Vec<String> = serde_json::from_value(case["initial"].clone()).unwrap();
            let text = if rows.is_empty() {
                String::new()
            } else {
                rows.join("\n") + "\n"
            };
            std::fs::write(&path, text).unwrap();
        }
        let mut store = storage(directory.path(), "day", true);
        if !case["initial"].is_null() {
            store.data().unwrap();
        }
        let result = store.insert(
            &case["index"].as_str().unwrap().parse().unwrap(),
            case["value"].as_str().unwrap(),
        );
        let mut result = sequence_outcome(result.map(|()| Value::Null));
        result["bytes"] = if path.exists() {
            json!(std::fs::read(&path).unwrap())
        } else {
            Value::Null
        };
        result["cached"] = sequence_outcome(store.data().map(|rows| {
            json!(
                rows.into_iter()
                    .map(|row| match row {
                        CalendarValue::Text(text) => text,
                        CalendarValue::Timestamp(_) =>
                            panic!("same-frequency data must remain text"),
                    })
                    .collect::<Vec<_>>()
            )
        }));
        assert_eq!(result, case["result"], "case {case_number}: {case}");
    }
}

struct SequenceReadScript(
    std::sync::Mutex<std::collections::VecDeque<Result<String, CalendarLoadError>>>,
);

impl domain_core::CalendarTextDecoder for SequenceReadScript {
    fn decode_text(&self, bytes: &[u8]) -> Result<String, CalendarLoadError> {
        assert_eq!(bytes, b"a\nb\n");
        self.0
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected extra read")
    }
}

#[test]
fn sequence_read_failures_and_changed_second_read_preserve_file_and_error_order() {
    use domain_core::{CalendarAssignment, CalendarSelection};
    let directory = tempfile::tempdir().unwrap();
    std::fs::create_dir(directory.path().join("calendars")).unwrap();
    let path = directory.path().join("calendars/day.txt");
    std::fs::write(&path, b"a\nb\n").unwrap();
    let index = CalendarSelection::Index(0.into());
    for error in [
        CalendarLoadError::Value("decoder-value".into()),
        CalendarLoadError::Other("decoder-other".into()),
    ] {
        for op in 0..5 {
            let script = Arc::new(SequenceReadScript(std::sync::Mutex::new(
                [Err(error.clone())].into(),
            )));
            let mut store = storage(directory.path(), "day", true);
            store.backend.text_decoder = script.clone();
            let actual = match op {
                0 => store.index("a").map(|_| ()),
                1 => store.get_item(&index).map(|_| ()),
                2 => store.delete_item(&index),
                3 => store.set_item(&index, &CalendarAssignment::Text("new".into())),
                4 => store.insert(&0.into(), "new"),
                _ => unreachable!(),
            };
            assert_eq!(actual, Err(error.clone()));
            assert!(script.0.lock().unwrap().is_empty());
            assert_eq!(std::fs::read(&path).unwrap(), b"a\nb\n");
        }
        let script = Arc::new(SequenceReadScript(std::sync::Mutex::new(
            [Ok("a\nb\n".into()), Err(error.clone())].into(),
        )));
        let mut store = storage(directory.path(), "day", true);
        store.backend.text_decoder = script.clone();
        assert_eq!(store.remove("b"), Err(error));
        assert!(script.0.lock().unwrap().is_empty());
        assert_eq!(std::fs::read(&path).unwrap(), b"a\nb\n");
    }
    let script = Arc::new(SequenceReadScript(std::sync::Mutex::new(
        [Ok("a\nb\n".into()), Ok(String::new())].into(),
    )));
    let mut store = storage(directory.path(), "day", true);
    store.backend.text_decoder = script.clone();
    assert_eq!(
        store.remove("b"),
        Err(CalendarLoadError::Other("list index out of range".into()))
    );
    assert!(script.0.lock().unwrap().is_empty());
    assert_eq!(std::fs::read(&path).unwrap(), b"a\nb\n");
    // A fresh request must fail selection before opening or decoding anything.
    let mut invalid = storage(directory.path(), "invalid", true);
    invalid.backend.text_decoder = script;
    assert!(matches!(invalid.check(), Err(CalendarLoadError::Value(_))));
    assert_eq!(std::fs::read(&path).unwrap(), b"a\nb\n");
}

#[test]
#[cfg(windows)]
fn insert_conversion_precedes_write_open_and_write_failure_keeps_empty_file() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::create_dir(directory.path().join("calendars")).unwrap();
    let path = directory.path().join("calendars/day.txt");
    std::fs::write(&path, b"").unwrap();
    let original = std::fs::metadata(&path).unwrap().permissions();
    let mut read_only = original.clone();
    read_only.set_readonly(true);
    std::fs::set_permissions(&path, read_only).unwrap();
    let mut store = storage(directory.path(), "day", true);
    let invalid = store.insert(&0.into(), "not numeric");
    let valid = store.insert(&0.into(), "1");
    std::fs::set_permissions(&path, original).unwrap();
    assert!(matches!(invalid, Err(CalendarLoadError::Value(_))));
    assert!(matches!(valid, Err(CalendarLoadError::Other(_))));
    assert!(std::fs::read(&path).unwrap().is_empty());
}

#[test]
fn sequence_operations_match_source_values_errors_bytes_and_stale_cache() {
    use domain_core::{
        CalendarAssignment, CalendarSelection, CalendarSelectionValue, CalendarSliceSpec,
    };
    let output = Command::new("python")
        .env("PYTHONUTF8", "1")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/calendar_sequence_contract.py"
        ))
        .arg("D:/code/github/qlib/qlib")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(cases.len(), 11940);
    let directory = tempfile::tempdir().unwrap();
    std::fs::create_dir(directory.path().join("calendars")).unwrap();
    let path = directory.path().join("calendars/day.txt");
    for (case_number, case) in cases.iter().enumerate() {
        if path.exists() {
            std::fs::remove_file(&path).unwrap();
        }
        if !case["initial"].is_null() {
            let rows: Vec<String> = serde_json::from_value(case["initial"].clone()).unwrap();
            let text = if rows.is_empty() {
                String::new()
            } else {
                rows.join("\n") + "\n"
            };
            std::fs::write(&path, text).unwrap();
        }
        let mut store = storage(directory.path(), "day", true);
        if !case["initial"].is_null() {
            store.data().unwrap();
        }
        let op = case["op"].as_str().unwrap();
        let result = if op == "index" {
            store
                .index(case["selection"].as_str().unwrap())
                .map(|i| json!(i))
        } else if op == "remove" {
            store
                .remove(case["selection"].as_str().unwrap())
                .map(|()| Value::Null)
        } else {
            let selection = if let Some(index) = case["selection"].as_str() {
                CalendarSelection::Index(index.parse().unwrap())
            } else {
                let fields = case["selection"].as_array().unwrap();
                CalendarSelection::Slice(CalendarSliceSpec {
                    start: fields[0].as_str().map(|s| s.parse().unwrap()),
                    stop: fields[1].as_str().map(|s| s.parse().unwrap()),
                    step: fields[2].as_str().map(|s| s.parse().unwrap()),
                })
            };
            match op {
                "get" => store.get_item(&selection).map(|value| match value {
                    CalendarSelectionValue::One(value) => json!(value),
                    CalendarSelectionValue::Many(value) => json!(value),
                }),
                "delete" => store.delete_item(&selection).map(|()| Value::Null),
                "set" => {
                    let assignment = if let Some(text) = case["assignment"].as_str() {
                        CalendarAssignment::Text(text.into())
                    } else {
                        CalendarAssignment::Sequence(
                            serde_json::from_value(case["assignment"].clone()).unwrap(),
                        )
                    };
                    store
                        .set_item(&selection, &assignment)
                        .map(|()| Value::Null)
                }
                _ => panic!("unknown operation"),
            }
        };
        let mut result = sequence_outcome(result);
        result["bytes"] = if path.exists() {
            json!(std::fs::read(&path).unwrap())
        } else {
            Value::Null
        };
        result["cached"] = sequence_outcome(store.data().map(|rows| {
            json!(
                rows.into_iter()
                    .map(|row| match row {
                        CalendarValue::Text(text) => text,
                        CalendarValue::Timestamp(_) =>
                            panic!("same-frequency data must remain text"),
                    })
                    .collect::<Vec<_>>()
            )
        }));
        assert_eq!(result, case["result"], "case {case_number}: {case}");
    }
}

fn action(
    storage: &mut FileCalendarStorage,
    roots: &[PathBuf],
    action: &Value,
) -> Result<Value, CalendarLoadError> {
    let name = action[0].as_str().unwrap();
    match name {
        "support" => {
            return Ok(json!(
                storage
                    .supported_frequencies()?
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
            ));
        }
        "file_frequency" => return Ok(json!(storage.file_frequency()?.to_string())),
        "uri" => {
            let path = storage.uri()?;
            let (index, relative) = roots
                .iter()
                .enumerate()
                .find_map(|(i, root)| path.strip_prefix(root).ok().map(|p| (i, p)))
                .unwrap();
            return Ok(json!([
                index,
                relative.to_str().unwrap().replace('\\', "/")
            ]));
        }
        "data" => {
            return Ok(json!(
                storage
                    .data()?
                    .into_iter()
                    .map(|value| match value {
                        CalendarValue::Text(value) => json!(["text", value]),
                        CalendarValue::Timestamp(value) =>
                            json!(["timestamp", value.format("%Y-%m-%dT%H:%M:%S").to_string()]),
                    })
                    .collect::<Vec<_>>()
            ));
        }
        "read" => return Ok(json!(storage.read_calendar()?)),
        "length" => return Ok(json!(storage.len()?)),
        "empty" => return Ok(json!(storage.is_empty()?)),
        "clear" => storage.clear()?,
        "extend" | "overwrite" => {
            let values: Vec<String> = serde_json::from_value(action[1].clone()).unwrap();
            let input = CalendarTextArray {
                shape: vec![values.len()],
                values,
            };
            if name == "extend" {
                storage.extend(&input)?;
            } else {
                storage.write_calendar(&input, CalendarWriteMode::Overwrite)?;
            }
        }
        "overwrite_scalar" => {
            let values = CalendarTextArray {
                shape: vec![],
                values: vec![action[1].as_str().unwrap().into()],
            };
            storage.write_calendar(&values, CalendarWriteMode::Overwrite)?;
        }
        "cache_clear" => storage.backend.cache.clear().unwrap(),
        "set_cache" => storage.backend.enable_read_cache = action[1].as_bool().unwrap(),
        "set_frequency" => storage.frequency = action[1].as_str().unwrap().into(),
        "set_future" => storage.future = action[1].as_bool().unwrap(),
        "set_root" => {
            let index: usize = serde_json::from_value(action[1].clone()).unwrap();
            if storage.backend.paths.provider_uri.is_empty() {
                storage.backend.paths.provider_uri.insert(
                    DEFAULT_DATA_FREQUENCY.into(),
                    roots[index].as_os_str().to_owned(),
                );
            }
            for value in storage.backend.paths.provider_uri.values_mut() {
                roots[index].as_os_str().clone_into(value);
            }
        }
        "set_provider" => {
            for value in storage.backend.paths.provider_uri.values_mut() {
                *value = action[1].as_str().unwrap().into();
            }
            storage.backend.paths.mount_path.clear();
        }
        "clear_providers" => storage.backend.paths.provider_uri.clear(),
        "put" | "put_bytes" | "remove_file" => {
            let index: usize = serde_json::from_value(action[1].clone()).unwrap();
            let path = roots[index]
                .join("calendars")
                .join(action[2].as_str().unwrap());
            if name == "put" {
                std::fs::write(path, action[3].as_str().unwrap()).unwrap();
            } else if name == "put_bytes" {
                let bytes: Vec<u8> = serde_json::from_value(action[3].clone()).unwrap();
                std::fs::write(path, bytes).unwrap();
            } else {
                std::fs::remove_file(path).unwrap();
            }
        }
        _ => panic!("unknown action: {name}"),
    }
    Ok(Value::Null)
}

#[test]
fn persistent_storage_matches_real_source_instances_and_file_effects() {
    let output = Command::new("python")
        .env("PYTHONUTF8", "1")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/file_calendar_storage_contract.py"
        ))
        .arg("D:/code/github/qlib/qlib")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(cases.as_array().unwrap().len(), 12);
    for case in cases.as_array().unwrap() {
        let directory = tempfile::tempdir().unwrap();
        let roots: Vec<_> = (0..2)
            .map(|i| directory.path().join(i.to_string()))
            .collect();
        for root in &roots {
            std::fs::create_dir_all(root.join("calendars")).unwrap();
        }
        for (name, text) in case["initial"].as_object().unwrap() {
            std::fs::write(
                roots[0].join("calendars").join(name),
                text.as_str().unwrap(),
            )
            .unwrap();
        }
        let mut storage = storage(
            &roots[0],
            case["frequency"].as_str().unwrap(),
            case["named"] == true,
        );
        for (operation, expected) in case["actions"]
            .as_array()
            .unwrap()
            .iter()
            .zip(case["results"].as_array().unwrap())
        {
            let mut result = match action(&mut storage, &roots, operation) {
                Ok(value) => json!({"value": value}),
                Err(CalendarLoadError::Value(_)) => json!({"error": "Value"}),
                Err(CalendarLoadError::Other(_)) => json!({"error": "Other"}),
            };
            result["files"] = json!(
                roots
                    .iter()
                    .map(|root| {
                        std::fs::read_dir(root.join("calendars"))
                            .unwrap()
                            .map(|entry| {
                                let entry = entry.unwrap();
                                (
                                    entry.file_name().into_string().unwrap(),
                                    json!(std::fs::read(entry.path()).unwrap()),
                                )
                            })
                            .collect::<serde_json::Map<_, _>>()
                    })
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                &result, expected,
                "case={} action={operation}",
                case["name"]
            );
        }
    }
}
