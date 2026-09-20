#![cfg(windows)]
use domain_core::path_initialization::{
    ConfigPathValue, PathConfiguration, PathInitializationError, PathSetting,
};
use serde_json::{Value, json};
use std::{
    ffi::OsString,
    os::windows::{
        ffi::{OsStrExt, OsStringExt},
        fs::OpenOptionsExt,
    },
    process::Command,
};

fn decode_value(value: &Value) -> ConfigPathValue {
    match value[0].as_str().unwrap() {
        "null" => ConfigPathValue::Null,
        "bad" => ConfigPathValue::Unsupported("int".into()),
        kind => {
            let units: Vec<u16> = serde_json::from_value(value[1].clone()).unwrap();
            let text = OsString::from_wide(&units);
            if kind == "path" {
                ConfigPathValue::Path(text.into())
            } else {
                assert_eq!(kind, "text");
                ConfigPathValue::Text(text)
            }
        }
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
        ConfigPathValue::Text(text) => json!(["text", text.encode_wide().collect::<Vec<_>>()]),
        ConfigPathValue::Path(path) => {
            json!(["path", path.as_os_str().encode_wide().collect::<Vec<_>>()])
        }
        ConfigPathValue::Null => json!(["null", null]),
        ConfigPathValue::Unsupported(kind) => json!(["bad", kind]),
    }
}
fn encode(value: &PathSetting) -> Value {
    match value {
        PathSetting::Scalar(value) => json!({"scalar":encode_value(value)}),
        PathSetting::Mapping(values) => {
            json!({"map":values.iter().map(|(key,value)| json!([key,encode_value(value)])).collect::<Vec<_>>()})
        }
    }
}
fn error_value(error: &PathInitializationError) -> Value {
    let class = match error {
        PathInitializationError::NoneProvider => "ValueError",
        PathInitializationError::UnsupportedProvider(_)
        | PathInitializationError::InvalidMountEntry(_) => "TypeError",
        PathInitializationError::InvalidProviderEntry(_) => "AttributeError",
        PathInitializationError::MissingMount(_) => "AssertionError",
        PathInitializationError::Operation(_) => "RuntimeError",
        PathInitializationError::NativePath(native) => match native {
            path::PathQueryError::Windows { code, .. } => {
                return json!({"class":"OSError","code":code});
            }
            path::PathQueryError::EmbeddedNul | path::PathQueryError::NotSymbolicLink => {
                "ValueError"
            }
            path::PathQueryError::Allocation => "MemoryError",
            path::PathQueryError::InputTooLong => "OverflowError",
        },
    };
    json!({"class":class,"code":null})
}

#[test]
fn real_configuration_results_and_partial_updates_match_original_qlib() {
    let directory = tempfile::tempdir().unwrap();
    let _lock = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .share_mode(0)
        .open(directory.path().join("locked.txt"))
        .unwrap();
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/windows_config_paths.py"
        ))
        .arg("D:/code/github/qlib/qlib/config.py")
        .arg(directory.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(cases.len(), 364);
    for case in cases {
        let mut config = PathConfiguration {
            provider_uri: decode(&case["provider"]),
            mount_path: decode(&case["mount"]),
        };
        let error = config
            .resolve_paths_native()
            .err()
            .as_ref()
            .map(error_value);
        assert_eq!(json!(error), case["error"], "{case}");
        assert_eq!(
            encode(&config.provider_uri),
            case["after_provider"],
            "{case}"
        );
        assert_eq!(encode(&config.mount_path), case["after_mount"], "{case}");
    }
}
