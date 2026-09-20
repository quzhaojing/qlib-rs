#![cfg(windows)]
use path::{PathQueryError, RealPathOperations, real_path, real_path_with};
use serde_json::{Value, json};
use std::os::windows::{
    ffi::{OsStrExt, OsStringExt},
    fs::OpenOptionsExt,
};
use std::{
    cell::RefCell,
    collections::HashMap,
    ffi::OsString,
    path::{Path, PathBuf},
    process::Command,
};

struct Queries {
    case: Value,
    counts: RefCell<HashMap<String, usize>>,
    trace: RefCell<Vec<Value>>,
}
impl Queries {
    fn query(&self, stage: &str, path: Option<&Path>) -> Result<OsString, PathQueryError> {
        self.trace
            .borrow_mut()
            .push(path.map_or_else(|| json!([stage]), |p| json!([stage, p.to_str().unwrap()])));
        let mut counts = self.counts.borrow_mut();
        let count = counts.entry(stage.to_owned()).or_default();
        let step = &self.case[stage][*count];
        *count += 1;
        if step.is_null() {
            return match stage {
                "cwd" => Ok(OsString::from("C:/cwd")),
                "key" => path::normalize_case(path.unwrap()),
                _ => Ok(path.unwrap().as_os_str().to_owned()),
            };
        }
        if let Some(value) = step["value"].as_str() {
            return Ok(value.into());
        }
        Err(match step["error"].as_str() {
            Some("nul") => PathQueryError::EmbeddedNul,
            Some("notlink") => PathQueryError::NotSymbolicLink,
            Some("allocation") => PathQueryError::Allocation,
            Some("length") => PathQueryError::InputTooLong,
            _ => PathQueryError::Windows {
                operation: "query",
                code: u32::try_from(step["error"].as_u64().unwrap()).unwrap(),
            },
        })
    }
}
impl RealPathOperations for Queries {
    fn current_directory(&self) -> Result<PathBuf, PathQueryError> {
        self.query("cwd", None).map(Into::into)
    }
    fn case_key(&self, path: &Path) -> Result<OsString, PathQueryError> {
        self.query("key", Some(path))
    }
    fn final_path(&self, path: &Path) -> Result<PathBuf, PathQueryError> {
        self.query("final", Some(path)).map(Into::into)
    }
    fn non_strict(&self, path: &Path) -> Result<PathBuf, PathQueryError> {
        self.query("fallback", Some(path)).map(Into::into)
    }
}
fn failure(error: &PathQueryError) -> Value {
    match error {
        PathQueryError::Windows { code, .. } => json!({"error":code}),
        PathQueryError::EmbeddedNul => json!({"error":"nul"}),
        PathQueryError::NotSymbolicLink => json!({"error":"notlink"}),
        PathQueryError::Allocation => json!({"error":"allocation"}),
        PathQueryError::InputTooLong => json!({"error":"length"}),
    }
}

#[test]
fn outer_policy_matches_actual_source_results_and_query_order() {
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/real_path.py"
        ))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(cases.len(), 369);
    for expected in cases {
        let queries = Queries {
            case: expected["case"].clone(),
            counts: RefCell::default(),
            trace: RefCell::default(),
        };
        let result =
            match real_path_with(Path::new(queries.case["input"].as_str().unwrap()), &queries) {
                Ok(value) => json!({"value":value.to_str().unwrap()}),
                Err(error) => failure(&error),
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
fn real_native_resolution_matches_python() {
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
            "/tests/fixtures/real_path_physical.py"
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
    assert_eq!(cases.len(), 37);
    for (input, expected) in cases {
        let input = OsString::from_wide(&input);
        let result = match real_path(Path::new(&input)) {
            Ok(value) => json!({"value":value.as_os_str().encode_wide().collect::<Vec<_>>()}),
            Err(error) => failure(&error),
        };
        assert_eq!(result, expected, "{input:?}");
    }
}
