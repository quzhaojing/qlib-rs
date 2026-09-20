#![cfg(windows)]

use domain_core::{
    CalendarCache, CalendarConstructionError, CalendarLoadError, CalendarPathProvider,
    CalendarProviderInput, CalendarProviderMap, CalendarRuntimeConfiguration,
    CalendarStorageFactory, CalendarTextEncoding, CalendarValue, DataPathManager,
    IsoCalendarTimestampDecoder,
    path_initialization::{ConfigPathOperations, ConfigPathValue, PathInitializationError},
    windows_config_paths::WindowsConfigPathOperations,
};
use indexmap::IndexMap;
use num_bigint::BigInt;
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    process::Command,
    sync::{Arc, Mutex, RwLock},
};

struct Runtime {
    region: RwLock<Option<String>>,
    events: Mutex<Vec<String>>,
}

impl CalendarRuntimeConfiguration for Runtime {
    fn region(&self) -> Result<Option<String>, CalendarLoadError> {
        self.events.lock().unwrap().push("region".into());
        let region = self.region.read().unwrap().clone();
        if region.as_deref() == Some("missing") {
            Err(CalendarLoadError::Other("region".into()))
        } else {
            Ok(region)
        }
    }
    fn minute_shift(&self) -> Result<BigInt, CalendarLoadError> {
        Ok(0.into())
    }
}

fn factory(root: &Path, region: Option<String>) -> (CalendarStorageFactory, Arc<Runtime>) {
    let runtime = Arc::new(Runtime {
        region: RwLock::new(region),
        events: Mutex::new(vec![]),
    });
    (
        CalendarStorageFactory {
            global_paths: Arc::new(RwLock::new(DataPathManager::from_native(
                IndexMap::from([("day".into(), root.as_os_str().to_owned())]),
                IndexMap::new(),
            ))),
            runtime: runtime.clone(),
            operations: Arc::new(WindowsConfigPathOperations),
            system: "Windows".into(),
            text_decoder: Arc::new(CalendarTextEncoding::Utf8),
            timestamp_decoder: Arc::new(IsoCalendarTimestampDecoder),
            cache: Arc::new(CalendarCache::new(None)),
        },
        runtime,
    )
}

fn mapping(entries: &[(&str, ConfigPathValue)]) -> CalendarProviderMap {
    Arc::new(RwLock::new(
        entries
            .iter()
            .map(|(k, v)| ((*k).into(), v.clone()))
            .collect(),
    ))
}

fn text(value: &str) -> ConfigPathValue {
    ConfigPathValue::Text(value.into())
}

fn input(mode: &str) -> (CalendarProviderInput, Option<CalendarProviderMap>) {
    let map = match mode {
        "global" => return (CalendarProviderInput::Scalar(ConfigPathValue::Null), None),
        "scalar" => return (CalendarProviderInput::Scalar(text(".")), None),
        "invalid" => {
            return (
                CalendarProviderInput::Scalar(ConfigPathValue::Unsupported("int".into())),
                None,
            );
        }
        "mapping" => mapping(&[("day", text("."))]),
        "nfs" => mapping(&[("day", text("host:/data"))]),
        "empty" => mapping(&[]),
        "partial" => mapping(&[
            ("day", text(".")),
            ("bad", ConfigPathValue::Null),
            ("week", text("untouched")),
        ]),
        _ => panic!("unknown mode"),
    };
    (CalendarProviderInput::Mapping(map.clone()), Some(map))
}

fn error_class(error: &CalendarConstructionError) -> &str {
    match error {
        CalendarConstructionError::Path(PathInitializationError::UnsupportedProvider(_)) => {
            "TypeError"
        }
        CalendarConstructionError::Path(PathInitializationError::InvalidProviderEntry(_)) => {
            "AttributeError"
        }
        CalendarConstructionError::Configuration(_) => "KeyError",
        other @ CalendarConstructionError::Path(_) => panic!("unexpected {other}"),
    }
}

#[test]
fn construction_matches_source_normalization_failure_order_aliasing_and_capture() {
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/calendar_constructor_contract.py"
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
    assert_eq!(cases.len(), 56);
    for case in cases {
        let region: Option<String> = serde_json::from_value(case["region"].clone()).unwrap();
        let (factory, runtime) = factory(Path::new("."), region.clone());
        let (provider, alias) = input(case["mode"].as_str().unwrap());
        let kwargs = Arc::new(json!({"enable_read_cache":false, "region":"us", "custom":[1,2]}));
        let result = factory.create(
            case["frequency"].as_str().unwrap().into(),
            true,
            provider,
            kwargs.clone(),
        );
        assert_eq!(
            json!(result.as_ref().err().map(error_class)),
            case["error"],
            "{case}"
        );
        let read_region = case["events"]
            .as_array()
            .unwrap()
            .iter()
            .any(|e| e == "get:region");
        assert_eq!(
            runtime.events.lock().unwrap().len(),
            usize::from(read_region),
            "{case}"
        );
        if let Ok(created) = result {
            assert!(Arc::ptr_eq(&created.kwargs, &kwargs));
            assert!(created.storage.future && created.storage.backend.enable_read_cache);
            assert_eq!(
                created.storage.frequency,
                case["frequency"].as_str().unwrap()
            );
            assert_eq!(created.storage.backend.region.captured_region, region);
            *runtime.region.write().unwrap() = Some("tw".into());
            assert_eq!(created.storage.backend.region.captured_region, region);
            if let Some(alias) = &alias {
                assert!(Arc::ptr_eq(
                    alias,
                    created
                        .storage
                        .backend
                        .paths
                        .provider_override
                        .as_ref()
                        .unwrap()
                ));
                alias
                    .write()
                    .unwrap()
                    .insert("external".into(), text("host:/changed"));
                assert!(
                    created
                        .storage
                        .backend
                        .paths
                        .provider_keys()
                        .unwrap()
                        .contains(&"external".into())
                );
            } else if case["mode"] == "global" {
                assert!(created.storage.backend.paths.provider_override.is_none());
            } else {
                let map = created
                    .storage
                    .backend
                    .paths
                    .provider_override
                    .as_ref()
                    .unwrap()
                    .read()
                    .unwrap();
                assert_eq!(
                    map["__DEFAULT_FREQ"],
                    text(std::env::current_dir().unwrap().to_str().unwrap())
                );
            }
        }
        if let Some(alias) = alias {
            assert_provider_state(&alias, &case);
        }
    }
}

fn assert_provider_state(alias: &CalendarProviderMap, case: &Value) {
    // Mirror the source probe's external action even after region lookup failed.
    if case["provider"].get("external").is_some() {
        alias
            .write()
            .unwrap()
            .insert("external".into(), text("host:/changed"));
    }
    let actual: serde_json::Map<String, Value> = alias
        .read()
        .unwrap()
        .iter()
        .map(|(key, value)| {
            let value = match value {
                ConfigPathValue::Text(v) => json!(v.to_str().unwrap()),
                ConfigPathValue::Null => Value::Null,
                other => panic!("unexpected {other:?}"),
            };
            (key.clone(), value)
        })
        .collect();
    assert_eq!(json!(actual), case["provider"], "{case}");
}

#[test]
fn constructed_storage_retains_live_override_mounts_and_global_paths() {
    let temp = tempfile::tempdir().unwrap();
    let roots = [temp.path().join("a"), temp.path().join("b")];
    for (i, root) in roots.iter().enumerate() {
        std::fs::create_dir_all(root.join("calendars")).unwrap();
        std::fs::write(root.join("calendars/day.txt"), format!("row{i}\n")).unwrap();
    }
    let (factory, _) = factory(&roots[0], Some("cn".into()));
    let mut inherited = factory
        .create(
            "day".into(),
            false,
            CalendarProviderInput::Scalar(ConfigPathValue::Null),
            (),
        )
        .unwrap();
    assert_eq!(
        inherited.storage.data().unwrap(),
        vec![CalendarValue::Text("row0".into())]
    );
    factory
        .global_paths
        .write()
        .unwrap()
        .provider_uri
        .insert("day".into(), roots[1].clone().into_os_string());
    assert_eq!(
        inherited.storage.data().unwrap(),
        vec![CalendarValue::Text("row1".into())]
    );
    let alias = mapping(&[("day", ConfigPathValue::Path(roots[0].clone()))]);
    let mut owned = factory
        .create(
            "day".into(),
            false,
            CalendarProviderInput::Mapping(alias.clone()),
            (),
        )
        .unwrap();
    assert_eq!(
        owned.storage.data().unwrap(),
        vec![CalendarValue::Text("row0".into())]
    );
    alias
        .write()
        .unwrap()
        .insert("day".into(), text("host:/calendar"));
    assert!(owned.storage.data().is_err());
    factory
        .global_paths
        .write()
        .unwrap()
        .mount_path
        .insert("day".into(), Some(roots[1].clone().into_os_string()));
    assert_eq!(
        owned.storage.data().unwrap(),
        vec![CalendarValue::Text("row1".into())]
    );
    alias
        .write()
        .unwrap()
        .insert("day".into(), ConfigPathValue::Path(roots[0].clone()));
    alias
        .write()
        .unwrap()
        .insert("unrelated".into(), ConfigPathValue::Null);
    assert_eq!(
        owned.storage.data().unwrap(),
        vec![CalendarValue::Text("row0".into())]
    );
    assert!(
        owned
            .storage
            .backend
            .paths
            .data_uri("missing", "Windows")
            .is_err()
    );
    alias.write().unwrap().insert(
        "__DEFAULT_FREQ".into(),
        ConfigPathValue::Path(roots[1].clone()),
    );
    assert_eq!(
        owned
            .storage
            .backend
            .paths
            .data_uri("missing", "Windows")
            .unwrap(),
        roots[1]
    );
    for invalid in [
        ConfigPathValue::Null,
        ConfigPathValue::Unsupported("int".into()),
    ] {
        alias.write().unwrap().insert("day".into(), invalid);
        assert!(owned.storage.data().is_err());
    }
    assert_scalar_path(&factory, &roots[0]);
}

fn assert_scalar_path(factory: &CalendarStorageFactory, root: &Path) {
    let mut scalar = factory
        .create(
            "day".into(),
            false,
            CalendarProviderInput::Scalar(ConfigPathValue::Path(root.to_owned())),
            (),
        )
        .unwrap();
    assert_eq!(
        scalar.storage.data().unwrap(),
        vec![CalendarValue::Text("row0".into())]
    );
}

struct FailPaths {
    resolve: bool,
}
impl ConfigPathOperations for FailPaths {
    fn expand_home(&self, path: &Path) -> Result<PathBuf, PathInitializationError> {
        if self.resolve {
            Ok(path.to_owned())
        } else {
            Err(PathInitializationError::Operation("expand".into()))
        }
    }
    fn resolve(&self, _path: &Path) -> Result<PathBuf, PathInitializationError> {
        Err(PathInitializationError::Operation("resolve".into()))
    }
}

#[test]
fn native_path_plugins_preserve_construction_and_later_lookup_failures() {
    for resolve in [false, true] {
        let (mut factory, runtime) = factory(Path::new("."), Some("cn".into()));
        factory.operations = Arc::new(FailPaths { resolve });
        let failed = factory.create(
            "day".into(),
            false,
            CalendarProviderInput::Scalar(text(".")),
            (),
        );
        assert!(matches!(
            failed,
            Err(CalendarConstructionError::Path(
                PathInitializationError::Operation(_)
            ))
        ));
        assert!(runtime.events.lock().unwrap().is_empty());
        let alias = mapping(&[("day", text("host:/data"))]);
        let created = factory
            .create(
                "day".into(),
                false,
                CalendarProviderInput::Mapping(alias.clone()),
                (),
            )
            .unwrap();
        alias
            .write()
            .unwrap()
            .insert("day".into(), ConfigPathValue::Path(".".into()));
        let error = created
            .storage
            .backend
            .paths
            .data_uri("day", "Windows")
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            if resolve { "resolve" } else { "expand" }
        );
    }
}

fn poison<T: Send + Sync + 'static>(value: &Arc<RwLock<T>>) {
    let value = value.clone();
    assert!(
        std::thread::spawn(move || {
            let _guard = value.write().unwrap();
            panic!("intentional poison");
        })
        .join()
        .is_err()
    );
}

#[test]
fn poisoned_maps_fail_only_when_accessed_and_are_never_recovered() {
    let (factory, _) = factory(Path::new("."), Some("cn".into()));
    let inherited = factory
        .create(
            "day".into(),
            false,
            CalendarProviderInput::Scalar(ConfigPathValue::Null),
            (),
        )
        .unwrap();
    let map = mapping(&[("day", text("host:/data"))]);
    let owned = factory
        .create(
            "day".into(),
            false,
            CalendarProviderInput::Mapping(map.clone()),
            (),
        )
        .unwrap();
    poison(&factory.global_paths);
    assert!(inherited.storage.backend.paths.provider_keys().is_err());
    assert!(
        inherited
            .storage
            .backend
            .paths
            .data_uri("day", "Windows")
            .is_err()
    );
    assert_eq!(
        owned.storage.backend.paths.provider_keys().unwrap(),
        ["day"]
    );
    assert!(
        owned
            .storage
            .backend
            .paths
            .data_uri("day", "Windows")
            .is_err()
    );
    poison(&map);
    assert!(owned.storage.backend.paths.provider_keys().is_err());
    assert!(
        owned
            .storage
            .backend
            .paths
            .data_uri("day", "Windows")
            .is_err()
    );
    let error = factory
        .create("day".into(), false, CalendarProviderInput::Mapping(map), ())
        .err()
        .unwrap();
    assert!(error.to_string().contains("lock poisoned"));
}

#[test]
fn construction_captures_region_but_null_defers_to_live_configuration() {
    let temp = tempfile::tempdir().unwrap();
    std::fs::create_dir(temp.path().join("calendars")).unwrap();
    std::fs::write(
        temp.path().join("calendars/1min.txt"),
        "2024-01-02T14:59:00\n",
    )
    .unwrap();
    let (factory, runtime) = factory(temp.path(), Some("cn".into()));
    let provider = || CalendarProviderInput::Scalar(ConfigPathValue::Path(temp.path().to_owned()));
    let mut captured = factory
        .create("60min".into(), false, provider(), ())
        .unwrap();
    *runtime.region.write().unwrap() = None;
    let mut deferred = factory
        .create("60min".into(), false, provider(), ())
        .unwrap();
    *runtime.region.write().unwrap() = Some("us".into());
    let expected = |text| {
        vec![CalendarValue::Timestamp(
            chrono::NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%M:%S").unwrap(),
        )]
    };
    assert_eq!(
        captured.storage.data().unwrap(),
        expected("2024-01-02T14:00:00")
    );
    assert_eq!(
        deferred.storage.data().unwrap(),
        expected("2024-01-02T14:30:00")
    );
    assert_eq!(runtime.events.lock().unwrap().len(), 3);
    *runtime.region.write().unwrap() = Some("unknown".into());
    let mut fresh = factory
        .create("60min".into(), false, provider(), ())
        .unwrap();
    assert!(matches!(
        fresh.storage.data(),
        Err(CalendarLoadError::Value(_))
    ));
    assert_eq!(
        captured.storage.data().unwrap(),
        expected("2024-01-02T14:00:00")
    );
}

#[test]
fn native_runtime_values_compose_with_factory_and_preserve_missing_null_and_poison() {
    use domain_core::CalendarRuntimeValues;
    let values = Arc::new(RwLock::new(CalendarRuntimeValues {
        region: None,
        minute_shift: None,
    }));
    let (mut factory, _) = factory(Path::new("."), Some("unused".into()));
    factory.runtime = values.clone();
    let input = || CalendarProviderInput::Scalar(ConfigPathValue::Null);
    assert!(factory.create("day".into(), false, input(), ()).is_err());
    assert!(
        values
            .minute_shift()
            .unwrap_err()
            .to_string()
            .contains("min_data_shift")
    );
    values.write().unwrap().region = Some(None);
    let created = factory.create("day".into(), false, input(), ()).unwrap();
    assert_eq!(created.storage.backend.region.captured_region, None);
    values.write().unwrap().region = Some(Some("tw".into()));
    values.write().unwrap().minute_shift = Some(7.into());
    assert_eq!(
        created
            .storage
            .backend
            .region
            .configuration
            .region()
            .unwrap(),
        Some("tw".into())
    );
    assert_eq!(
        created
            .storage
            .backend
            .region
            .configuration
            .minute_shift()
            .unwrap(),
        BigInt::from(7)
    );
    poison(&values);
    assert!(
        values
            .region()
            .unwrap_err()
            .to_string()
            .contains("lock poisoned")
    );
    assert!(
        values
            .minute_shift()
            .unwrap_err()
            .to_string()
            .contains("lock poisoned")
    );
    assert!(factory.create("day".into(), false, input(), ()).is_err());
}
