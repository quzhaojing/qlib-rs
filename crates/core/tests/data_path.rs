use domain_core::{DataPathError, DataPathManager, ProviderUriKind, provider_uri_kind};
use indexmap::IndexMap;
use serde_json::{Value, json};
use std::{path::PathBuf, process::Command};

#[test]
fn data_root_lookup_matches_actual_source_paths_platforms_defaults_and_failures() {
    let output = Command::new("python")
        .args([
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/data_path_contract.py"
            ),
            r"D:\code\github\qlib\qlib\config.py",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let expected: Value = serde_json::from_slice(&output.stdout).unwrap();
    for row in expected["kinds"].as_array().unwrap() {
        let kind = match provider_uri_kind(row[0].as_str().unwrap()) {
            ProviderUriKind::Local => "local",
            ProviderUriKind::Nfs => "nfs",
        };
        assert_eq!(kind, row[1].as_str().unwrap(), "{row}");
    }
    for row in expected["paths"].as_array().unwrap() {
        let manager = DataPathManager::new(
            serde_json::from_value(row["provider"].clone()).unwrap(),
            serde_json::from_value(row["mount"].clone()).unwrap(),
        );
        let (result, error) =
            match manager.get_data_uri(row["freq"].as_str(), row["system"].as_str().unwrap()) {
                Ok(path) => (json!(path.to_str().unwrap()), Value::Null),
                Err(DataPathError::MissingProvider(_) | DataPathError::MissingMount(_)) => {
                    (Value::Null, json!("KeyError"))
                }
                Err(DataPathError::NullMount(_)) => (Value::Null, json!("TypeError")),
            };
        assert_eq!(result, row["result"], "{row}");
        assert_eq!(error, row["error"], "{row}");
    }
}

#[test]
fn typed_errors_and_map_changes_are_live_without_touching_the_filesystem() {
    let mut manager = DataPathManager::new(IndexMap::new(), IndexMap::new());
    let error = manager.get_data_uri(Some("day"), "Windows").unwrap_err();
    assert_eq!(
        error,
        DataPathError::MissingProvider("__DEFAULT_FREQ".to_owned())
    );
    assert_eq!(
        error.to_string(),
        "missing provider URI key: __DEFAULT_FREQ"
    );
    manager
        .provider_uri
        .insert("day".to_owned(), "host:/data".into());
    let error = manager.get_data_uri(Some("day"), "Windows").unwrap_err();
    assert_eq!(error, DataPathError::MissingMount("day".to_owned()));
    assert_eq!(error.to_string(), "missing mount path key: day");
    manager.mount_path.insert("day".to_owned(), None);
    let error = manager.get_data_uri(Some("day"), "Linux").unwrap_err();
    assert_eq!(error, DataPathError::NullMount("day".to_owned()));
    assert_eq!(error.to_string(), "mount path must be text, not None: day");
    manager
        .provider_uri
        .insert("day".to_owned(), "local-data".into());
    assert_eq!(
        manager.get_data_uri(Some("day"), "Windows").unwrap(),
        PathBuf::from("local-data")
    );
    assert_eq!(manager.mount_path["day"], None);
}
