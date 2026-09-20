#![cfg(windows)]
use std::{
    ffi::OsString,
    os::windows::ffi::{OsStrExt, OsStringExt},
    path::Path,
    process::Command,
};

#[test]
fn path_join_matches_python_drive_and_root_policy() {
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/join.py"
        ))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Vec<(Vec<Vec<u16>>, Vec<u16>)> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(cases.len(), 44_135);
    for (parts, expected) in cases {
        let native: Vec<_> = parts.iter().map(|part| OsString::from_wide(part)).collect();
        let paths: Vec<_> = native[1..].iter().map(Path::new).collect();
        let actual = path::join(Path::new(&native[0]), &paths);
        assert_eq!(
            actual.as_os_str().encode_wide().collect::<Vec<_>>(),
            expected,
            "{native:?}"
        );
    }
}
