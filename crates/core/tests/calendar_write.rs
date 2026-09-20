use std::{cell::Cell, io::Write, path::PathBuf, process::Command};

use domain_core::{
    CalendarFileValues, CalendarLoadError, CalendarTextArray, CalendarWriteMode,
    write_calendar_file,
};
use serde_json::Value;

#[test]
fn binary_writes_match_source_shapes_bytes_errors_and_open_order() {
    let output = Command::new("python")
        .env("PYTHONUTF8", "1")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/calendar_write_contract.py"
        ))
        .arg("D:/code/github/qlib/qlib/data/storage/file_storage.py")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(cases.as_array().unwrap().len(), 96);
    for case in cases.as_array().unwrap() {
        let directory = tempfile::tempdir().unwrap();
        let path: PathBuf = match case["state"].as_str().unwrap() {
            "existing" => {
                let path = directory.path().join("file.txt");
                std::fs::write(&path, b"original\n").unwrap();
                path
            }
            "missing" => directory.path().join("file.txt"),
            "missing_parent" => directory.path().join("absent/file.txt"),
            "directory" => directory.path().to_owned(),
            state => panic!("unknown state: {state}"),
        };
        let array = CalendarTextArray {
            shape: serde_json::from_value(case["shape"].clone()).unwrap(),
            values: serde_json::from_value(case["values"].clone()).unwrap(),
        };
        let mode = if case["mode"] == "ab" {
            CalendarWriteMode::Append
        } else {
            CalendarWriteMode::Overwrite
        };
        let result = write_calendar_file(&path, &array, mode);
        let error = result.err().map(|error| match error {
            CalendarLoadError::Value(_) => "Value",
            CalendarLoadError::Other(_) => "Other",
        });
        assert_eq!(
            serde_json::to_value(error).unwrap(),
            case["error"],
            "{case}"
        );
        let data = path.is_file().then(|| std::fs::read(&path).unwrap());
        assert_eq!(serde_json::to_value(data).unwrap(), case["data"], "{case}");
        assert!(!directory.path().join("absent").exists());
    }
}

struct FailedValues {
    calls: Cell<usize>,
}

impl CalendarFileValues for FailedValues {
    fn write_values(&self, output: &mut dyn Write) -> Result<(), CalendarLoadError> {
        self.calls.set(self.calls.get() + 1);
        output.write_all(b"partial\n").unwrap();
        Err(CalendarLoadError::Value("conversion".into()))
    }
}

#[test]
fn plugin_failure_retains_partial_output_and_open_failure_skips_conversion() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("file.txt");
    let values = FailedValues {
        calls: Cell::new(0),
    };
    for mode in [CalendarWriteMode::Overwrite, CalendarWriteMode::Append] {
        std::fs::write(&path, b"old\n").unwrap();
        assert_eq!(
            write_calendar_file(&path, &values, mode),
            Err(CalendarLoadError::Value("conversion".into()))
        );
        let expected: &[u8] = if mode == CalendarWriteMode::Append {
            b"old\npartial\n"
        } else {
            b"partial\n"
        };
        assert_eq!(std::fs::read(&path).unwrap(), expected);
    }
    assert_eq!(values.calls.get(), 2);
    assert!(matches!(
        write_calendar_file(directory.path(), &values, CalendarWriteMode::Overwrite),
        Err(CalendarLoadError::Other(_))
    ));
    assert_eq!(values.calls.get(), 2);
    let malformed = CalendarTextArray {
        shape: vec![2],
        values: vec!["one".into()],
    };
    assert!(matches!(
        write_calendar_file(&path, &malformed, CalendarWriteMode::Overwrite),
        Err(CalendarLoadError::Value(_))
    ));
    assert_eq!(std::fs::read(&path).unwrap(), b"");
}
