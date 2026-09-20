#![cfg(windows)]
use path::{LinkOperations, NativeLinkOperations, PathQueryError, read_link_deep};
use serde_json::{Value, json};
use std::{
    cell::{Cell, RefCell},
    ffi::OsString,
    path::{Path, PathBuf},
    process::Command,
};

use std::os::windows::ffi::{OsStrExt, OsStringExt};

#[test]
fn physical_chain_queries_match_python_with_native_classification() {
    let directory = tempfile::tempdir().unwrap();
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/deep_links.py"
        ))
        .arg(directory.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let queries = NativeLinkOperations;
    assert!(!queries.is_symbolic_link(&directory.path().join("junction")));
    let cases: Vec<(Vec<u16>, Value)> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(cases.len(), 18);
    for (input, expected) in cases {
        let input = OsString::from_wide(&input);
        let actual = match read_link_deep(Path::new(&input), &queries) {
            Ok(value) => json!({"value":value.as_os_str().encode_wide().collect::<Vec<_>>()}),
            Err(PathQueryError::Windows { code, .. }) => json!({"error":code}),
            Err(error) => panic!("unexpected {error}"),
        };
        assert_eq!(actual, expected, "{input:?}");
    }
}

struct Queries {
    case: Value,
    trace: RefCell<Vec<Value>>,
    key_calls: Cell<u64>,
}

impl LinkOperations for Queries {
    fn case_key(&self, path: &Path) -> Result<OsString, PathQueryError> {
        self.trace
            .borrow_mut()
            .push(json!(["key", path.to_str().unwrap()]));
        self.key_calls.set(self.key_calls.get() + 1);
        if self.case["key_fail"].as_u64() == Some(self.key_calls.get()) {
            return Err(PathQueryError::Allocation);
        }
        path::normalize_case(path)
    }
    fn read_link(&self, path: &Path) -> Result<PathBuf, PathQueryError> {
        let text = path.to_str().unwrap();
        self.trace.borrow_mut().push(json!(["read", text]));
        let step = &self.case["steps"][text];
        if let Some(value) = step["value"].as_str() {
            return Ok(PathBuf::from(value));
        }
        Err(match step["error"].as_str() {
            Some("nul") => PathQueryError::EmbeddedNul,
            Some("notlink") => PathQueryError::NotSymbolicLink,
            Some("allocation") => PathQueryError::Allocation,
            Some("length") => PathQueryError::InputTooLong,
            _ => PathQueryError::Windows {
                operation: "read",
                code: u32::try_from(step["error"].as_u64().unwrap_or(2)).unwrap(),
            },
        })
    }
    fn is_symbolic_link(&self, path: &Path) -> bool {
        let text = path.to_str().unwrap();
        self.trace.borrow_mut().push(json!(["islink", text]));
        self.case["steps"][text]["symbolic"]
            .as_bool()
            .unwrap_or(false)
    }
}

#[test]
fn full_query_traces_and_results_match_source_link_traversal() {
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/link_chain.py"
        ))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(cases.len(), 81);
    for expected in cases {
        let queries = Queries {
            case: expected["case"].clone(),
            trace: RefCell::default(),
            key_calls: Cell::new(0),
        };
        let result =
            match read_link_deep(Path::new(queries.case["input"].as_str().unwrap()), &queries) {
                Ok(value) => json!({"value":value.to_str().unwrap()}),
                Err(PathQueryError::Allocation) => json!({"error":"allocation"}),
                Err(PathQueryError::InputTooLong) => json!({"error":"length"}),
                Err(PathQueryError::Windows { code, .. }) => json!({"error":code}),
                Err(error) => panic!("unexpected {error}"),
            };
        assert_eq!(result, expected["result"], "case={}", queries.case);
        assert_eq!(
            *queries.trace.borrow(),
            expected["trace"].as_array().unwrap().clone(),
            "case={}",
            queries.case
        );
    }
}
