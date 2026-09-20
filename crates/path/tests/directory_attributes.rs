#![cfg(windows)]
use path::{PathQueryError, directory_attributes};
use serde_json::{Value, json};
use std::{
    ffi::OsString,
    os::windows::{ffi::OsStringExt, fs::OpenOptionsExt},
    path::Path,
    process::Command,
};

#[test]
fn directory_search_matches_native_attributes_without_opening_locked_file() {
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
            "/tests/fixtures/directory_attributes.py"
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
    assert_eq!(cases.len(), 34);
    for (units, expected) in cases {
        let input = OsString::from_wide(&units);
        let actual = match directory_attributes(Path::new(&input)) {
            Ok(Some(value)) => json!({"attributes":value.attributes,"tag":value.reparse_tag}),
            Ok(None) => json!({"skip":true}),
            Err(PathQueryError::Windows { operation, code }) => {
                assert_eq!(operation, "FindFirstFileW");
                json!({"error":code})
            }
            Err(PathQueryError::EmbeddedNul) => json!({"error":"nul"}),
            Err(error) => panic!("unexpected {error}"),
        };
        assert_eq!(actual, expected, "{input:?}");
    }
    let locked = directory_attributes(&directory.path().join("locked.txt"))
        .unwrap()
        .unwrap();
    assert_eq!(locked.reparse_tag, 0);
    assert_eq!(
        std::fs::File::open(directory.path().join("locked.txt"))
            .unwrap_err()
            .raw_os_error(),
        Some(32)
    );
    assert_eq!(
        directory_attributes(&directory.path().join("junction"))
            .unwrap()
            .unwrap()
            .reparse_tag,
        0xa000_0003
    );
    assert_eq!(
        directory_attributes(&directory.path().join("relative"))
            .unwrap()
            .unwrap()
            .reparse_tag,
        0xa000_000c
    );
}
