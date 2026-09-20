use std::{path::PathBuf, process::Command};

use domain_core::InitialStateType;
use serde::Deserialize;

#[derive(Debug, Deserialize, PartialEq)]
struct PythonSeedSnapshot {
    source_sha256: String,
    module_doc: String,
    public_names: Vec<String>,
    body_kinds: Vec<String>,
    typevar_name: String,
    typevar_type: String,
    constraints: Vec<String>,
    bound: Option<String>,
    covariant: bool,
    contravariant: bool,
}

struct NonCloneState {
    value: i32,
}

fn identity<T>(value: InitialStateType<T>) -> T {
    value
}

#[test]
fn native_initial_state_alias_accepts_every_concrete_type_without_wrapping() {
    let integer: InitialStateType<i64> = 7;
    assert_eq!(identity(integer), 7);

    let state: InitialStateType<NonCloneState> = NonCloneState { value: 11 };
    assert_eq!(identity(state).value, 11);

    let mut text = String::from("seed");
    let reference: InitialStateType<&mut String> = &mut text;
    reference.push_str("-state");
    assert_eq!(text, "seed-state");
}

#[test]
fn live_python_module_freezes_unconstrained_typevar_and_public_surface() {
    let source = std::env::var_os("QLIB_PYTHON_RL_SEED").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/rl/seed.py"),
        PathBuf::from,
    );
    assert!(
        source.is_file(),
        "Python source not found: {}",
        source.display()
    );

    let script = r#"
import ast
import hashlib
import importlib.util
import json
import sys

path = sys.argv[1]
raw = open(path, "rb").read()
tree = ast.parse(raw, filename=path)
spec = importlib.util.spec_from_file_location("qlib_rl_seed_contract", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
typevar = module.InitialStateType
print(json.dumps({
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "module_doc": module.__doc__,
    "public_names": [name for name in vars(module) if not name.startswith("_")],
    "body_kinds": [type(node).__name__ for node in tree.body],
    "typevar_name": typevar.__name__,
    "typevar_type": f"{type(typevar).__module__}.{type(typevar).__qualname__}",
    "constraints": [f"{item.__module__}.{item.__qualname__}" for item in typevar.__constraints__],
    "bound": None if typevar.__bound__ is None else repr(typevar.__bound__),
    "covariant": typevar.__covariant__,
    "contravariant": typevar.__contravariant__,
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
        "RL seed snapshot failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let actual: PythonSeedSnapshot =
        serde_json::from_slice(&output.stdout).expect("Python returns a JSON snapshot");
    let expected = PythonSeedSnapshot {
        source_sha256: "bc86d479595fdb28cc1f546a87e97c45c060a7846c066e58834b1138afaaffb9"
            .to_owned(),
        module_doc: "Defines a set of initial state definitions and state-set definitions.\n\nWith single-asset order execution only, the only seed is order.\n"
            .to_owned(),
        public_names: ["TypeVar", "InitialStateType"]
            .map(str::to_owned)
            .to_vec(),
        body_kinds: ["Expr", "ImportFrom", "Assign", "Expr"]
            .map(str::to_owned)
            .to_vec(),
        typevar_name: "InitialStateType".to_owned(),
        typevar_type: "typing.TypeVar".to_owned(),
        constraints: vec![],
        bound: None,
        covariant: false,
        contravariant: false,
    };
    assert_eq!(actual, expected);
}
