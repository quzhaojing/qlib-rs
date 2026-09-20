#![cfg(windows)]
use std::{
    ffi::OsString,
    os::windows::ffi::{OsStrExt, OsStringExt},
    path::Path,
    process::Command,
};

#[test]
fn lexical_normalization_matches_actual_python() {
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/lexical.py"
        ))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Vec<(Vec<u16>, Vec<u16>)> = serde_json::from_slice(&output.stdout).unwrap();
    assert!(cases.len() > 10_000);
    for (input, expected) in cases {
        let native = OsString::from_wide(&input);
        let actual = path::normalize(Path::new(&native));
        assert_eq!(
            actual.as_os_str().encode_wide().collect::<Vec<_>>(),
            expected,
            "input={input:?}"
        );
        assert_eq!(path::normalize(&actual).as_os_str(), actual.as_os_str());
    }
}
