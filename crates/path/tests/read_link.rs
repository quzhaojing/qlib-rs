#![cfg(windows)]
use path::{PathQueryError, read_link};
use serde_json::{Value, json};
use std::{
    ffi::OsString,
    os::windows::{
        ffi::{OsStrExt, OsStringExt},
        fs::OpenOptionsExt,
    },
    path::Path,
    process::Command,
};

#[test]
fn native_links_and_junctions_match_python() {
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
            "/tests/fixtures/read_link.py"
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
    for (input, expected) in cases {
        let input = OsString::from_wide(&input);
        let actual = match read_link(Path::new(&input)) {
            Ok(value) => json!({"value": value.as_os_str().encode_wide().collect::<Vec<_>>()}),
            Err(PathQueryError::EmbeddedNul | PathQueryError::NotSymbolicLink) => {
                json!({"error": "ValueError"})
            }
            Err(PathQueryError::Windows { code, .. }) => json!({"error": code}),
            Err(error) => panic!("unexpected {error}"),
        };
        assert_eq!(actual, expected, "{input:?}");
    }
    let destination = read_link(&directory.path().join("absolute")).unwrap();
    assert!(
        destination
            .as_os_str()
            .to_string_lossy()
            .starts_with(r"\\?\")
    );
    assert_eq!(
        read_link(&directory.path().join("relative"))
            .unwrap()
            .as_os_str(),
        "file.txt"
    );
    assert_eq!(
        std::fs::File::open(directory.path().join("locked.txt"))
            .unwrap_err()
            .raw_os_error(),
        Some(32)
    );
}
