use domain_core::{
    CalendarLoadError, CalendarTextEncoding, calendar_text_records, read_calendar_file,
};
use serde_json::{Value, json};
use std::process::Command;

#[test]
fn real_file_records_match_unchanged_source() {
    let output = Command::new("python")
        .env("PYTHONUTF8", "1")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/calendar_file_contract.py"
        ))
        .arg("D:/code/github/qlib/qlib/data/storage/file_storage.py")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let expected: Value = serde_json::from_slice(&output.stdout).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("calendar.txt");
    for case in expected["cases"].as_array().unwrap() {
        let text = case[0].as_str().unwrap();
        std::fs::write(&path, text).unwrap();
        assert_eq!(
            json!(read_calendar_file(&path, &CalendarTextEncoding::Utf8).unwrap()),
            case[1]
        );
        assert_eq!(json!(calendar_text_records(text)), case[1]);
    }
    // Check every native Unicode scalar against the host Python whitespace set.
    let whitespace = expected["whitespace"].as_str().unwrap();
    for character in (0..=0x0010_ffff).filter_map(char::from_u32) {
        let text = format!("{character}x{character}");
        let expected = if whitespace.contains(character) {
            "x"
        } else {
            &text
        };
        assert_eq!(calendar_text_records(&text), [expected], "{character:?}");
    }
}

#[test]
fn real_file_creation_io_failures_and_strict_decoding() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("calendar.txt");
    assert!(!path.exists());
    assert!(
        read_calendar_file(&path, &CalendarTextEncoding::Utf8)
            .unwrap()
            .is_empty()
    );
    assert_eq!(std::fs::read(&path).unwrap(), b"");
    let missing_parent = directory.path().join("missing/child.txt");
    assert!(matches!(
        read_calendar_file(&missing_parent, &CalendarTextEncoding::Utf8),
        Err(CalendarLoadError::Other(_))
    ));
    assert!(!missing_parent.parent().unwrap().exists());
    assert!(matches!(
        read_calendar_file(directory.path(), &CalendarTextEncoding::Utf8),
        Err(CalendarLoadError::Other(_))
    ));
    std::fs::write(&path, b"2024-01-02\n\xff").unwrap();
    assert!(matches!(
        read_calendar_file(&path, &CalendarTextEncoding::Utf8),
        Err(CalendarLoadError::Value(_))
    ));
    assert_eq!(std::fs::read(&path).unwrap(), b"2024-01-02\n\xff");
    std::fs::write(&path, b" \xd6\xd0\xce\xc4\r\n").unwrap();
    assert_eq!(
        read_calendar_file(&path, &CalendarTextEncoding::PythonGbk).unwrap(),
        ["中文"]
    );
}
