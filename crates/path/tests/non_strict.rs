#![cfg(windows)]
use path::{FinalPathOperations, NativeFinalPathOperations, PathQueryError, final_path_non_strict};
use serde_json::{Value, json};
use std::os::windows::{
    ffi::{OsStrExt, OsStringExt},
    fs::OpenOptionsExt,
};
use std::{
    cell::RefCell,
    ffi::OsString,
    path::{Path, PathBuf},
    process::Command,
};

struct Queries {
    case: Value,
    trace: RefCell<Vec<Value>>,
}
impl Queries {
    fn query(&self, stage: &str, path: &Path) -> Result<PathBuf, PathQueryError> {
        let value = path.to_str().unwrap();
        self.trace.borrow_mut().push(json!([stage, value]));
        let step = &self.case[stage][value];
        if let Some(value) = step["value"].as_str() {
            return Ok(value.into());
        }
        if step.is_null() && stage == "deep" {
            return Ok(path.to_path_buf());
        }
        Err(match step["error"].as_str() {
            Some("nul") => PathQueryError::EmbeddedNul,
            Some("notlink") => PathQueryError::NotSymbolicLink,
            Some("allocation") => PathQueryError::Allocation,
            Some("length") => PathQueryError::InputTooLong,
            _ => PathQueryError::Windows {
                operation: "query",
                code: u32::try_from(step["error"].as_u64().unwrap_or(2)).unwrap(),
            },
        })
    }
}
impl FinalPathOperations for Queries {
    fn final_path(&self, path: &Path) -> Result<PathBuf, PathQueryError> {
        self.query("final", path)
    }
    fn read_link_deep(&self, path: &Path) -> Result<PathBuf, PathQueryError> {
        self.query("deep", path)
    }
    fn find_name(&self, path: &Path) -> Result<PathBuf, PathQueryError> {
        self.query("find", path)
    }
}
fn error_value(error: &PathQueryError) -> Value {
    match error {
        PathQueryError::Windows { code, .. } => json!({"error":code}),
        PathQueryError::EmbeddedNul => json!({"error":"nul"}),
        PathQueryError::NotSymbolicLink => json!({"error":"notlink"}),
        PathQueryError::Allocation => json!({"error":"allocation"}),
        PathQueryError::InputTooLong => json!({"error":"length"}),
    }
}
#[test]
fn complete_fallback_traces_match_actual_source() {
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/non_strict.py"
        ))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(cases.len(), 151);
    for expected in cases {
        let queries = Queries {
            case: expected["case"].clone(),
            trace: RefCell::default(),
        };
        let result = match final_path_non_strict(
            Path::new(queries.case["input"].as_str().unwrap()),
            &queries,
        ) {
            Ok(value) => json!({"value":value.to_str().unwrap()}),
            Err(error) => error_value(&error),
        };
        assert_eq!(result, expected["result"], "{}", queries.case);
        assert_eq!(
            *queries.trace.borrow(),
            *expected["trace"].as_array().unwrap(),
            "{}",
            queries.case
        );
    }
}

#[test]
fn native_fallback_matches_python_on_physical_links_and_missing_suffixes() {
    let directory = tempfile::tempdir().unwrap();
    let _lock = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .share_mode(0)
        .open(directory.path().join("locked.txt"))
        .unwrap();
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/non_strict_physical.py"
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
    assert_eq!(cases.len(), 27);
    for (input, expected) in cases {
        let input = OsString::from_wide(&input);
        let result = match final_path_non_strict(Path::new(&input), &NativeFinalPathOperations) {
            Ok(value) => json!({"value":value.as_os_str().encode_wide().collect::<Vec<_>>()}),
            Err(error) => error_value(&error),
        };
        assert_eq!(result, expected, "{input:?}");
    }
}
