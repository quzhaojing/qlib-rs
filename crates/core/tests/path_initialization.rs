use domain_core::path_initialization::{
    ConfigPathOperations, ConfigPathValue, PathConfiguration, PathInitializationError, PathSetting,
    normalize_provider_map,
};
use indexmap::IndexMap;
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    process::Command,
    sync::Mutex,
};

struct TracePaths {
    events: Mutex<Vec<Value>>,
    fail_at: usize,
}

impl TracePaths {
    fn operation(&self, name: &str, path: &Path) -> Result<PathBuf, PathInitializationError> {
        let mut events = self.events.lock().unwrap();
        events.push(json!([name, path.to_str().unwrap()]));
        if events.len() == self.fail_at {
            return Err(PathInitializationError::Operation("injected".into()));
        }
        let prefix = if name == "expand" { "E" } else { "R" };
        Ok(format!("{prefix}[{}]", path.display()).into())
    }
}

impl ConfigPathOperations for TracePaths {
    fn expand_home(&self, path: &Path) -> Result<PathBuf, PathInitializationError> {
        self.operation("expand", path)
    }
    fn resolve(&self, path: &Path) -> Result<PathBuf, PathInitializationError> {
        self.operation("resolve", path)
    }
}

fn decode_value(value: &Value) -> ConfigPathValue {
    match value[0].as_str().unwrap() {
        "text" => ConfigPathValue::Text(value[1].as_str().unwrap().into()),
        "path" => ConfigPathValue::Path(value[1].as_str().unwrap().into()),
        "null" => ConfigPathValue::Null,
        "bad" => ConfigPathValue::Unsupported(value[1].as_str().unwrap().into()),
        _ => panic!("unknown fixture value"),
    }
}

fn decode(value: &Value) -> PathSetting {
    if let Some(scalar) = value.get("scalar") {
        PathSetting::Scalar(decode_value(scalar))
    } else {
        PathSetting::Mapping(
            value["map"]
                .as_array()
                .unwrap()
                .iter()
                .map(|pair| (pair[0].as_str().unwrap().into(), decode_value(&pair[1])))
                .collect(),
        )
    }
}

fn encode_value(value: &ConfigPathValue) -> Value {
    match value {
        ConfigPathValue::Text(text) => json!(["text", text.to_str().unwrap()]),
        ConfigPathValue::Path(path) => json!(["path", path.to_str().unwrap()]),
        ConfigPathValue::Null => json!(["null", null]),
        ConfigPathValue::Unsupported(kind) => json!(["bad", kind]),
    }
}

fn encode(value: &PathSetting) -> Value {
    match value {
        PathSetting::Scalar(value) => json!({"scalar": encode_value(value)}),
        PathSetting::Mapping(values) => {
            json!({"map": values.iter().map(|(key, value)| json!([key, encode_value(value)])).collect::<Vec<_>>()})
        }
    }
}

fn error_class(error: &PathInitializationError) -> &'static str {
    match error {
        PathInitializationError::NoneProvider => "ValueError",
        PathInitializationError::UnsupportedProvider(_)
        | PathInitializationError::InvalidMountEntry(_) => "TypeError",
        PathInitializationError::InvalidProviderEntry(_) => "AttributeError",
        PathInitializationError::MissingMount(_) => "AssertionError",
        PathInitializationError::Operation(_) => "RuntimeError",
        #[cfg(windows)]
        PathInitializationError::NativePath(_) => {
            panic!("trace-only operations cannot return native errors")
        }
    }
}

#[test]
fn ordered_initialization_and_partial_mutations_match_unchanged_source() {
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/path_initialization_contract.py"
        ))
        .arg("D:/code/github/qlib/qlib/config.py")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(cases.as_array().unwrap().len(), 700);
    for case in cases.as_array().unwrap() {
        let mut configuration = PathConfiguration {
            provider_uri: decode(&case["provider"]),
            mount_path: decode(&case["mount"]),
        };
        let paths = TracePaths {
            events: Mutex::new(Vec::new()),
            fail_at: serde_json::from_value(case["fail_at"].clone()).unwrap(),
        };
        let result = configuration.resolve_paths(&paths);
        assert_eq!(
            json!(result.as_ref().err().map(error_class)),
            case["error"],
            "{case}"
        );
        assert_eq!(
            json!(*paths.events.lock().unwrap()),
            case["events"],
            "{case}"
        );
        assert_eq!(
            encode(&configuration.provider_uri),
            case["after_provider"],
            "{case}"
        );
        assert_eq!(
            encode(&configuration.mount_path),
            case["after_mount"],
            "{case}"
        );
    }
}

#[test]
fn direct_map_normalization_preserves_nfs_values_and_prior_writes() {
    let paths = TracePaths {
        events: Mutex::new(Vec::new()),
        fail_at: 3,
    };
    let mut values = IndexMap::from([
        ("day".into(), ConfigPathValue::Text("first".into())),
        ("week".into(), ConfigPathValue::Text("host:/remote".into())),
        ("1min".into(), ConfigPathValue::Text("last".into())),
    ]);
    let error = normalize_provider_map(&mut values, &paths).unwrap_err();
    assert_eq!(error, PathInitializationError::Operation("injected".into()));
    assert_eq!(error.to_string(), "injected");
    assert_eq!(values["day"], ConfigPathValue::Text("R[E[first]]".into()));
    assert_eq!(values["week"], ConfigPathValue::Text("host:/remote".into()));
    assert_eq!(values["1min"], ConfigPathValue::Text("last".into()));
}

#[test]
fn missing_mount_reports_exact_keys_before_expanding_any_mount() {
    let paths = TracePaths {
        events: Mutex::new(Vec::new()),
        fail_at: 0,
    };
    let mut configuration = PathConfiguration {
        provider_uri: PathSetting::Mapping(IndexMap::from([
            ("day".into(), ConfigPathValue::Text("first".into())),
            ("week".into(), ConfigPathValue::Text("host:/remote".into())),
        ])),
        mount_path: PathSetting::Mapping(IndexMap::new()),
    };
    assert_eq!(
        configuration.resolve_paths(&paths),
        Err(PathInitializationError::MissingMount(vec![
            "day".into(),
            "week".into()
        ]))
    );
    assert_eq!(
        *paths.events.lock().unwrap(),
        [json!(["expand", "first"]), json!(["resolve", "E[first]"])]
    );
    assert_eq!(
        encode(&configuration.provider_uri),
        json!({"map": [["day", ["text", "R[E[first]]"]], ["week", ["text", "host:/remote"]]]})
    );
}

#[test]
fn native_error_diagnostics_are_explicit_and_stable() {
    for (error, expected) in [
        (
            PathInitializationError::NoneProvider,
            "provider_uri cannot be None",
        ),
        (
            PathInitializationError::UnsupportedProvider("int".into()),
            "provider_uri does not support int",
        ),
        (
            PathInitializationError::InvalidProviderEntry("NoneType".into()),
            "provider URI entry has no expanduser method: NoneType",
        ),
        (
            PathInitializationError::InvalidMountEntry("int".into()),
            "mount path is not path-like: int",
        ),
        (
            PathInitializationError::MissingMount(vec!["day".into()]),
            "mount_path is missing freq: [\"day\"]",
        ),
        (
            PathInitializationError::Operation("filesystem failure".into()),
            "filesystem failure",
        ),
    ] {
        assert_eq!(error.to_string(), expected);
    }
}

#[cfg(windows)]
#[test]
fn real_home_expansion_composes_without_losing_utf16_or_committing_failed_scalars() {
    use domain_core::windows_home::WindowsHomeEnvironment;
    use std::{ffi::OsString, os::windows::ffi::OsStringExt};
    struct HomePaths(WindowsHomeEnvironment);
    impl ConfigPathOperations for HomePaths {
        fn expand_home(&self, path: &Path) -> Result<PathBuf, PathInitializationError> {
            self.0
                .expand(path)
                .map_err(|error| PathInitializationError::Operation(error.to_string()))
        }
        fn resolve(&self, _: &Path) -> Result<PathBuf, PathInitializationError> {
            panic!("NFS roots and mount-only expansion must not resolve filesystem paths")
        }
    }
    let mut configuration = PathConfiguration {
        provider_uri: PathSetting::Scalar(ConfigPathValue::Text("host:/remote".into())),
        mount_path: PathSetting::Scalar(ConfigPathValue::Path("~/data".into())),
    };
    let original = configuration.clone();
    assert_eq!(
        configuration.resolve_paths(&HomePaths(WindowsHomeEnvironment::default())),
        Err(PathInitializationError::Operation(
            "Could not determine home directory.".into()
        ))
    );
    assert_eq!(configuration, original);
    let profile = OsString::from_wide(&[67, 58, 92, 0xd800]);
    configuration
        .resolve_paths(&HomePaths(WindowsHomeEnvironment {
            user_profile: Some(profile.clone()),
            ..WindowsHomeEnvironment::default()
        }))
        .unwrap();
    let PathSetting::Mapping(mounts) = configuration.mount_path else {
        panic!("normalized map required")
    };
    assert_eq!(
        mounts["__DEFAULT_FREQ"],
        ConfigPathValue::Text(PathBuf::from(profile).join("data").into_os_string())
    );
    assert_eq!(
        configuration.provider_uri,
        PathSetting::Mapping(IndexMap::from([(
            "__DEFAULT_FREQ".into(),
            ConfigPathValue::Text("host:/remote".into())
        )]))
    );
}
