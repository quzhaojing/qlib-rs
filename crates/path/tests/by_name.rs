#![cfg(windows)]
use path::{PathQueryError, symbolic_link_by_name};
use serde_json::{Value, json};
use std::{
    ffi::OsString,
    os::windows::{ffi::OsStringExt, fs::OpenOptionsExt},
    path::Path,
    process::Command,
};

#[test]
fn native_by_name_matches_query_errors_and_source_classification() {
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
            "/tests/fixtures/by_name.py"
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
        let actual = match symbolic_link_by_name(Path::new(&input)) {
            Ok(value) => json!({"value": value}),
            Err(PathQueryError::Windows { operation, code }) => {
                assert_eq!(operation, "GetFileInformationByName");
                json!({"error":code})
            }
            Err(PathQueryError::EmbeddedNul) => json!({"error":"nul"}),
            Err(error) => panic!("unexpected {error}"),
        };
        assert_eq!(actual, expected, "{input:?}");
    }
    assert_eq!(
        symbolic_link_by_name(&directory.path().join("junction")),
        Ok(false)
    );
    assert_eq!(
        symbolic_link_by_name(&directory.path().join("missing")),
        Ok(true)
    );
    // The by-name stage requires no open handle, even for this exclusive lock.
    assert_eq!(
        symbolic_link_by_name(&directory.path().join("locked.txt")),
        Ok(false)
    );
}
