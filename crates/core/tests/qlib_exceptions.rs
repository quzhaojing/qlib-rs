use std::{error::Error, path::PathBuf, process::Command};

use domain_core::{
    ExpAlreadyExistError, LoadObjectError, QlibError, QlibException, RecorderInitializationError,
};
use serde::Deserialize;

#[derive(Debug, Deserialize, PartialEq)]
struct PythonExceptionSnapshot {
    source_sha256: String,
    public_names: Vec<String>,
    classes: Vec<PythonClassSnapshot>,
}

#[derive(Debug, Deserialize, PartialEq)]
struct PythonClassSnapshot {
    name: String,
    base: String,
    doc: Option<String>,
    public_members: Vec<String>,
    empty_args: Vec<String>,
    empty_text: String,
    message_args: Vec<String>,
    message_text: String,
    is_exception: bool,
    is_qlib_exception: bool,
}

fn assert_standard_error<T: Error>() {}
fn assert_qlib_error<T: QlibError>() {}

fn expected_class(
    name: &str,
    base: &str,
    doc: Option<&str>,
    is_qlib_exception: bool,
) -> PythonClassSnapshot {
    PythonClassSnapshot {
        name: name.to_owned(),
        base: base.to_owned(),
        doc: doc.map(str::to_owned),
        public_members: vec![],
        empty_args: vec![],
        empty_text: String::new(),
        message_args: vec!["message".to_owned()],
        message_text: "message".to_owned(),
        is_exception: true,
        is_qlib_exception,
    }
}

#[test]
fn native_error_types_preserve_messages_and_qlib_family_membership() {
    assert_standard_error::<QlibException>();
    assert_standard_error::<RecorderInitializationError>();
    assert_standard_error::<LoadObjectError>();
    assert_standard_error::<ExpAlreadyExistError>();
    assert_qlib_error::<QlibException>();
    assert_qlib_error::<RecorderInitializationError>();
    assert_qlib_error::<LoadObjectError>();

    let base = QlibException::default();
    assert_eq!(base.message(), None);
    assert_eq!(base.to_string(), "");
    assert_eq!(QlibError::message(&base), None);

    let recorder = RecorderInitializationError::new("already active");
    assert_eq!(recorder.message(), Some("already active"));
    assert_eq!(recorder.to_string(), "already active");
    assert_eq!(QlibError::message(&recorder), Some("already active"));

    let load = LoadObjectError::new("missing artifact");
    assert_eq!(load.message(), Some("missing artifact"));
    assert_eq!(load.to_string(), "missing artifact");
    assert_eq!(QlibError::message(&load), Some("missing artifact"));

    let exists = ExpAlreadyExistError::default();
    assert_eq!(exists.message(), None);
    assert_eq!(exists.to_string(), "");
    let exists = ExpAlreadyExistError::new("duplicate");
    assert_eq!(exists.message(), Some("duplicate"));
    assert_eq!(exists.to_string(), "duplicate");

    let base = QlibException::new("base failure");
    assert_eq!(base.message(), Some("base failure"));
    assert_eq!(base.to_string(), "base failure");
    let recorder = RecorderInitializationError::default();
    assert_eq!(recorder.message(), None);
    assert_eq!(recorder.to_string(), "");
    let load = LoadObjectError::default();
    assert_eq!(load.message(), None);
    assert_eq!(load.to_string(), "");
}

#[test]
fn live_python_module_freezes_class_order_hierarchy_docs_and_messages() {
    let source = std::env::var_os("QLIB_PYTHON_EXCEPTIONS").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/utils/exceptions.py"),
        PathBuf::from,
    );
    assert!(
        source.is_file(),
        "Python source not found: {}",
        source.display()
    );

    let script = r#"
import hashlib
import importlib.util
import json
import sys

path = sys.argv[1]
raw = open(path, "rb").read()
spec = importlib.util.spec_from_file_location("qlib_exceptions_contract", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
names = ["QlibException", "RecorderInitializationError", "LoadObjectError", "ExpAlreadyExistError"]
classes = []
for name in names:
    cls = getattr(module, name)
    empty = cls()
    message = cls("message")
    classes.append({
        "name": name,
        "base": cls.__base__.__name__,
        "doc": cls.__doc__,
        "public_members": [key for key in cls.__dict__ if not key.startswith("_")],
        "empty_args": list(empty.args),
        "empty_text": str(empty),
        "message_args": list(message.args),
        "message_text": str(message),
        "is_exception": isinstance(message, Exception),
        "is_qlib_exception": isinstance(message, module.QlibException),
    })
print(json.dumps({
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "public_names": [name for name in vars(module) if not name.startswith("_")],
    "classes": classes,
}, ensure_ascii=False))
"#;
    let python = std::env::var_os("PYTHON").unwrap_or_else(|| "python".into());
    let output = Command::new(python)
        .arg("-c")
        .arg(script)
        .arg(&source)
        .output()
        .expect("Python interpreter starts");
    assert!(
        output.status.success(),
        "exception snapshot failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let actual: PythonExceptionSnapshot =
        serde_json::from_slice(&output.stdout).expect("Python returns a JSON snapshot");
    let expected = PythonExceptionSnapshot {
        source_sha256: "654f03c6e15a1b13e6585963437717df13e1da2b11918f682a81692e51d59067"
            .to_owned(),
        public_names: [
            "QlibException",
            "RecorderInitializationError",
            "LoadObjectError",
            "ExpAlreadyExistError",
        ]
        .map(str::to_owned)
        .to_vec(),
        classes: vec![
            expected_class("QlibException", "Exception", None, true),
            expected_class(
                "RecorderInitializationError",
                "QlibException",
                Some("Error type for re-initialization when starting an experiment"),
                true,
            ),
            expected_class(
                "LoadObjectError",
                "QlibException",
                Some("Error type for Recorder when can not load object"),
                true,
            ),
            expected_class(
                "ExpAlreadyExistError",
                "Exception",
                Some("Experiment already exists"),
                false,
            ),
        ],
    };
    assert_eq!(actual, expected);
}
