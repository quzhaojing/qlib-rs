#![cfg(windows)]
use std::{
    ffi::OsString,
    os::windows::ffi::{OsStrExt, OsStringExt},
    path::Path,
    process::Command,
};

#[test]
fn native_case_mapping_matches_python_across_utf16_and_unicode_planes() {
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/normalize_case.py"
        ))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Vec<(Vec<u16>, Vec<u16>)> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(cases.len(), 66_570);
    for (input, expected) in cases {
        let native = OsString::from_wide(&input);
        let actual = path::normalize_case(Path::new(&native)).unwrap();
        assert_eq!(
            actual.encode_wide().collect::<Vec<_>>(),
            expected,
            "input={input:?}"
        );
        assert_eq!(path::normalize_case(Path::new(&actual)).unwrap(), actual);
    }
    assert_eq!(
        path::normalize_case(Path::new("C:/A/../B")).unwrap(),
        "c:\\a\\..\\b"
    );
}
