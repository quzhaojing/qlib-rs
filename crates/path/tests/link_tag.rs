#![cfg(windows)]
use path::{PathQueryError, symbolic_link_by_open};
use serde_json::{Value, json};
use std::{
    ffi::OsString,
    os::windows::{ffi::OsStringExt, fs::OpenOptionsExt},
    path::Path,
    process::Command,
};

#[test]
fn native_tag_stage_matches_windows_queries_and_python_successes() {
    let directory = tempfile::tempdir().unwrap();
    let _locked = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .share_mode(0)
        .open(directory.path().join("locked.txt"))
        .unwrap();
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/link_tag.py"
        ))
        .arg(directory.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Vec<(Vec<u16>, Value)> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(cases.len(), 18);
    for (units, expected) in cases {
        let input = OsString::from_wide(&units);
        let actual = match symbolic_link_by_open(Path::new(&input)) {
            Ok(value) => json!({"value":value}),
            Err(PathQueryError::Windows { operation, code }) => {
                json!({"error":code,"operation":operation})
            }
            Err(PathQueryError::EmbeddedNul) => json!({"error":"nul"}),
            Err(error) => panic!("unexpected {error}"),
        };
        assert_eq!(actual, expected, "{input:?}");
    }
    assert_eq!(
        symbolic_link_by_open(&directory.path().join("junction")),
        Ok(false)
    );
    assert_eq!(
        symbolic_link_by_open(&directory.path().join("missing")),
        Ok(true)
    );
}
