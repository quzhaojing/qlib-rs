#![cfg(windows)]
use path::{PathQueryError, is_symbolic_link, lstat_attributes};
use serde_json::{Value, json};
use std::{
    ffi::OsString,
    os::windows::{ffi::OsStringExt, fs::OpenOptionsExt},
    path::Path,
    process::Command,
};

#[test]
fn native_stat_and_classification_match_actual_python() {
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
            "/tests/fixtures/stat_classification.py"
        ))
        .arg(directory.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Vec<(Vec<u16>, Value, bool)> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(cases.len(), 29);
    for (units, expected, islink) in cases {
        let input = OsString::from_wide(&units);
        let actual = match lstat_attributes(Path::new(&input)) {
            Ok(info) => json!({"attributes":info.attributes,"tag":info.reparse_tag}),
            Err(PathQueryError::Windows { code, .. }) => json!({"error":code}),
            Err(PathQueryError::EmbeddedNul) => json!({"error":"nul"}),
            Err(error) => panic!("unexpected {error}"),
        };
        assert_eq!(actual, expected, "{input:?}");
        assert_eq!(is_symbolic_link(Path::new(&input)), islink, "{input:?}");
    }
}
