use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    process::Command,
    sync::{Arc, RwLock},
};

use domain_core::{
    CalendarCache, CalendarLoadError, CalendarPathProvider, CalendarTextArray,
    CalendarTextEncoding, CalendarValue, DataPathManager, FileCalendarBackend, FileCalendarStorage,
    IsoCalendarTimestampDecoder, LiveCalendarPaths, Region,
};
use indexmap::IndexMap;
use serde_json::{Value, json};

fn providers(root: &Path) -> IndexMap<String, OsString> {
    IndexMap::from([("day".into(), root.as_os_str().to_owned())])
}

fn store(paths: LiveCalendarPaths) -> FileCalendarStorage<LiveCalendarPaths> {
    FileCalendarStorage::new(
        FileCalendarBackend {
            paths,
            system: "Windows".into(),
            region: Region::Cn,
            minute_shift: 0.into(),
            text_decoder: Arc::new(CalendarTextEncoding::Utf8),
            timestamp_decoder: Arc::new(IsoCalendarTimestampDecoder),
            cache: Arc::new(CalendarCache::new(None)),
            enable_read_cache: true,
        },
        "day".into(),
        false,
    )
}

fn apply_action(
    storage: &mut FileCalendarStorage<LiveCalendarPaths>,
    roots: &[PathBuf],
    action: &Value,
) -> Result<Value, CalendarLoadError> {
    let name = action[0].as_str().unwrap();
    match name {
        "data" => {
            return storage.data().map(|rows| {
                json!(
                    rows.into_iter()
                        .map(|row| match row {
                            CalendarValue::Text(value) => value,
                            CalendarValue::Timestamp(_) => panic!("same-frequency result is text"),
                        })
                        .collect::<Vec<_>>()
                )
            });
        }
        "uri" => {
            let path = storage.uri()?;
            return Ok(json!(
                roots
                    .iter()
                    .position(|root| root.join("calendars/day.txt") == path)
                    .unwrap()
            ));
        }
        "support" => {
            return storage
                .supported_frequencies()
                .map(|freqs| json!(freqs.iter().map(ToString::to_string).collect::<Vec<_>>()));
        }
        "clear" => storage.clear()?,
        "read" => return storage.read_calendar().map(|rows| json!(rows)),
        "extend" => storage.extend(&CalendarTextArray {
            shape: vec![1],
            values: vec!["new".into()],
        })?,
        "cache_clear" => storage.backend.cache.clear().unwrap(),
        _ => apply_configuration(&mut storage.backend.paths, roots, action),
    }
    Ok(Value::Null)
}

fn root_index(action: &Value) -> usize {
    serde_json::from_value(action[1].clone()).unwrap()
}

fn apply_configuration(paths: &mut LiveCalendarPaths, roots: &[PathBuf], action: &Value) {
    match action[0].as_str().unwrap() {
        "provider" => {
            paths.configuration.write().unwrap().provider_uri =
                providers(&roots[root_index(action)]);
        }
        "provider_empty" => paths.configuration.write().unwrap().provider_uri.clear(),
        "mount" => {
            paths.configuration.write().unwrap().mount_path.insert(
                "day".into(),
                Some(roots[root_index(action)].as_os_str().to_owned()),
            );
        }
        "override" => {
            paths.provider_override =
                Some(Arc::new(RwLock::new(providers(&roots[root_index(action)]))));
        }
        "override_root" => {
            paths
                .provider_override
                .as_ref()
                .unwrap()
                .write()
                .unwrap()
                .insert(
                    "day".into(),
                    roots[root_index(action)].as_os_str().to_owned(),
                );
        }
        "override_nfs" => {
            paths
                .provider_override
                .as_ref()
                .unwrap()
                .write()
                .unwrap()
                .insert("day".into(), "host:/data".into());
        }
        "override_empty" => paths
            .provider_override
            .as_ref()
            .unwrap()
            .write()
            .unwrap()
            .clear(),
        "drop_override" => paths.provider_override = None,
        _ => panic!("unknown configuration action"),
    }
}

#[test]
fn live_global_and_override_mutations_match_source_paths_data_cache_and_writes() {
    let output = Command::new("python")
        .env("PYTHONUTF8", "1")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/live_calendar_paths_contract.py"
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
    assert_eq!(cases.len(), 3);
    for case in cases {
        assert_eq!(case["actions"].as_array().unwrap().len(), 36);
        let directory = tempfile::tempdir().unwrap();
        let roots: Vec<_> = (0..3)
            .map(|i| directory.path().join(i.to_string()))
            .collect();
        for (i, root) in roots.iter().enumerate() {
            std::fs::create_dir_all(root.join("calendars")).unwrap();
            std::fs::write(root.join("calendars/day.txt"), format!("row{i}\n")).unwrap();
        }
        let configuration = Arc::new(RwLock::new(DataPathManager::from_native(
            providers(&roots[0]),
            IndexMap::from([("day".into(), Some(roots[1].as_os_str().to_owned()))]),
        )));
        let provider_override = match case["mode"].as_str().unwrap() {
            "global" => None,
            "local_override" => Some(Arc::new(RwLock::new(providers(&roots[2])))),
            "nfs_override" => Some(Arc::new(RwLock::new(IndexMap::from([(
                "day".into(),
                "host:/data".into(),
            )])))),
            _ => unreachable!(),
        };
        let mut storage = store(LiveCalendarPaths {
            configuration,
            provider_override,
        });
        for (i, action) in case["actions"].as_array().unwrap().iter().enumerate() {
            let mut result = match apply_action(&mut storage, &roots, action) {
                Ok(value) => json!({"value": value}),
                Err(CalendarLoadError::Value(_)) => json!({"error": "Value"}),
                Err(CalendarLoadError::Other(_)) => json!({"error": "Other"}),
            };
            result["files"] = json!(
                roots
                    .iter()
                    .map(|root| std::fs::read(root.join("calendars/day.txt")).unwrap())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                result, case["results"][i],
                "mode {}, action {i}: {action}",
                case["mode"]
            );
        }
    }
}

fn poison<T: Send + Sync + 'static>(lock: &Arc<RwLock<T>>) {
    let lock = Arc::clone(lock);
    assert!(
        std::thread::spawn(move || {
            let _guard = lock.write().unwrap();
            panic!("poison configuration");
        })
        .join()
        .is_err()
    );
}

#[test]
fn poisoned_configuration_is_reported_only_when_that_source_is_accessed() {
    let directory = tempfile::tempdir().unwrap();
    let configuration = Arc::new(RwLock::new(DataPathManager::from_native(
        providers(directory.path()),
        IndexMap::new(),
    )));
    let override_map = Arc::new(RwLock::new(providers(directory.path())));
    let mut storage = store(LiveCalendarPaths {
        configuration: configuration.clone(),
        provider_override: None,
    });
    poison(&configuration);
    let expected = CalendarLoadError::Other("calendar path configuration lock poisoned".into());
    assert_eq!(storage.backend.paths.provider_keys(), Err(expected.clone()));
    assert_eq!(
        storage.backend.paths.data_uri("day", "Windows"),
        Err(expected.clone())
    );
    assert_eq!(storage.supported_frequencies(), Err(expected.clone()));
    storage.backend.paths.provider_override = Some(override_map.clone());
    assert_eq!(storage.backend.paths.provider_keys().unwrap(), vec!["day"]);
    assert_eq!(
        storage.backend.paths.data_uri("day", "Windows"),
        Err(expected.clone())
    );
    let configuration = Arc::new(RwLock::new(DataPathManager::from_native(
        providers(directory.path()),
        IndexMap::new(),
    )));
    let paths = LiveCalendarPaths {
        configuration,
        provider_override: Some(override_map.clone()),
    };
    poison(&override_map);
    assert_eq!(paths.provider_keys(), Err(expected.clone()));
    assert_eq!(paths.data_uri("day", "Windows"), Err(expected));
}

#[test]
fn external_shared_maps_route_all_storage_methods_and_fresh_backend_requests() {
    use domain_core::{
        CalendarAssignment, CalendarBackendSource, CalendarSelection, CalendarSelectionValue,
        CalendarSliceSpec,
    };
    let directory = tempfile::tempdir().unwrap();
    let roots: Vec<_> = (0..3)
        .map(|i| directory.path().join(i.to_string()))
        .collect();
    for (i, root) in roots.iter().enumerate() {
        std::fs::create_dir_all(root.join("calendars")).unwrap();
        std::fs::write(root.join("calendars/day.txt"), format!("{i}\n")).unwrap();
    }
    let configuration = Arc::new(RwLock::new(DataPathManager::from_native(
        providers(&roots[0]),
        IndexMap::new(),
    )));
    let override_map = Arc::new(RwLock::new(providers(&roots[1])));
    let mut global = store(LiveCalendarPaths {
        configuration: configuration.clone(),
        provider_override: None,
    });
    let mut overridden = store(LiveCalendarPaths {
        configuration: configuration.clone(),
        provider_override: Some(override_map.clone()),
    });
    assert_eq!(
        global.data().unwrap(),
        vec![CalendarValue::Text("0".into())]
    );
    assert_eq!(
        overridden.data().unwrap(),
        vec![CalendarValue::Text("1".into())]
    );
    configuration.write().unwrap().provider_uri = providers(&roots[2]);
    *override_map.write().unwrap() = providers(&roots[0]);
    let index = CalendarSelection::Index(0.into());
    assert_eq!(
        global.get_item(&index).unwrap(),
        CalendarSelectionValue::One("2".into())
    );
    assert_eq!(
        overridden.get_item(&index).unwrap(),
        CalendarSelectionValue::One("0".into())
    );
    overridden
        .set_item(&index, &CalendarAssignment::Text("xy".into()))
        .unwrap();
    overridden.insert(&1.into(), "long").unwrap();
    assert_eq!(
        overridden
            .get_item(&CalendarSelection::Slice(CalendarSliceSpec::default()))
            .unwrap(),
        CalendarSelectionValue::Many(vec!["xy".into(), "lo".into()])
    );
    assert_eq!(overridden.index("lo").unwrap(), 1);
    overridden.delete_item(&index).unwrap();
    overridden.remove("lo").unwrap();
    assert!(overridden.is_empty().unwrap());
    assert_eq!(overridden.len().unwrap(), 0);
    #[cfg(windows)]
    assert_eq!(
        overridden.storage_frequencies().unwrap(),
        vec![OsString::from("day")]
    );
    let rows = global
        .backend
        .data("day", false)
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(rows, vec![CalendarValue::Text("2".into())]);
    configuration.write().unwrap().provider_uri =
        IndexMap::from([("bad".into(), roots[2].as_os_str().to_owned())]);
    assert_eq!(
        global
            .supported_frequencies()
            .unwrap()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
        vec!["day"]
    );
    assert!(matches!(
        global.backend.resolve_file("day", false),
        Err(CalendarLoadError::Value(_))
    ));
    assert!(
        std::fs::read(roots[0].join("calendars/day.txt"))
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        std::fs::read(roots[1].join("calendars/day.txt")).unwrap(),
        b"1\n"
    );
    assert_eq!(
        std::fs::read(roots[2].join("calendars/day.txt")).unwrap(),
        b"2\n"
    );
}
