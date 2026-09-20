use std::{path::PathBuf, process::Command};

use domain_core::BaseOptimizer;
use serde::Deserialize;

#[derive(Debug, Deserialize, PartialEq)]
struct PythonOptimizerBaseSnapshot {
    source_sha256: String,
    module_doc: Option<String>,
    public_names: Vec<String>,
    body_kinds: Vec<String>,
    bases: Vec<String>,
    class_doc: String,
    call_doc: String,
    signature: String,
    abstract_methods: Vec<String>,
    class_public_names: Vec<String>,
    verified_facts: Vec<String>,
    forwarded_args: Vec<i64>,
    forwarded_kwargs: Vec<(String, i64)>,
    concrete_result: Vec<i64>,
}

fn expected_python_snapshot() -> PythonOptimizerBaseSnapshot {
    PythonOptimizerBaseSnapshot {
        source_sha256: "e361589cb48220a6ddffef283b38ee9bbd7e442f34f65ef1d16f8db161d03d62"
            .to_owned(),
        module_doc: None,
        public_names: ["abc", "BaseOptimizer"].map(str::to_owned).to_vec(),
        body_kinds: ["Import", "ClassDef"].map(str::to_owned).to_vec(),
        bases: vec!["abc.ABC".to_owned()],
        class_doc: "Construct portfolio with a optimization related method".to_owned(),
        call_doc: "Generate a optimized portfolio allocation".to_owned(),
        signature: "(self, *args, **kwargs) -> object".to_owned(),
        abstract_methods: vec!["__call__".to_owned()],
        class_public_names: vec![],
        verified_facts: [
            "base-is-abstract",
            "call-is-abstract",
            "base-instantiation-fails",
            "incomplete-subclass-instantiation-fails",
            "direct-base-call-is-none",
        ]
        .map(str::to_owned)
        .to_vec(),
        forwarded_args: vec![3, 4],
        forwarded_kwargs: [("right".to_owned(), 5), ("ignored".to_owned(), 6)].to_vec(),
        concrete_result: vec![3, 5],
    }
}

struct PairOptimizer;

impl BaseOptimizer<(i64, i64)> for PairOptimizer {
    type Output = [i64; 2];

    fn call(&self, (left, right): (i64, i64)) -> Self::Output {
        [left, right]
    }
}

struct BorrowingOptimizer;

impl<'a> BaseOptimizer<&'a mut String> for BorrowingOptimizer {
    type Output = usize;

    fn call(&self, text: &'a mut String) -> Self::Output {
        text.push('!');
        text.len()
    }
}

#[test]
fn native_optimizer_trait_preserves_implementer_selected_inputs_and_outputs() {
    let optimizer: &dyn BaseOptimizer<(i64, i64), Output = [i64; 2]> = &PairOptimizer;
    assert_eq!(optimizer.call((3, 5)), [3, 5]);

    let mut text = String::from("weights");
    assert_eq!(BorrowingOptimizer.call(&mut text), 8);
    assert_eq!(text, "weights!");
}

#[test]
fn live_python_module_freezes_abstract_optimizer_contract() {
    let source = std::env::var_os("QLIB_PYTHON_OPTIMIZER_BASE").map_or_else(
        || {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../../qlib/qlib/contrib/strategy/optimizer/base.py")
        },
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
import inspect
import json
import sys

path = sys.argv[1]
raw = open(path, "rb").read()
tree = ast.parse(raw, filename=path)
spec = importlib.util.spec_from_file_location("qlib_optimizer_base_contract", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
base = module.BaseOptimizer

class Incomplete(base):
    pass

observed = {}
class Concrete(base):
    def __call__(self, *args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return [args[0], kwargs["right"]]

def instantiation_fails(cls):
    try:
        cls()
    except TypeError:
        return True
    return False

result = Concrete()(3, 4, right=5, ignored=6)
class_node = tree.body[1]
facts = []
if inspect.isabstract(base):
    facts.append("base-is-abstract")
if base.__call__.__isabstractmethod__:
    facts.append("call-is-abstract")
if instantiation_fails(base):
    facts.append("base-instantiation-fails")
if instantiation_fails(Incomplete):
    facts.append("incomplete-subclass-instantiation-fails")
if base.__call__(object(), 1, value=2) is None:
    facts.append("direct-base-call-is-none")
print(json.dumps({
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "module_doc": module.__doc__,
    "public_names": [name for name in vars(module) if not name.startswith("_")],
    "body_kinds": [type(node).__name__ for node in tree.body],
    "bases": [ast.unparse(item) for item in class_node.bases],
    "class_doc": base.__doc__,
    "call_doc": base.__call__.__doc__,
    "signature": str(inspect.signature(base.__call__)),
    "abstract_methods": sorted(base.__abstractmethods__),
    "class_public_names": [name for name in vars(base) if not name.startswith("_")],
    "verified_facts": facts,
    "forwarded_args": list(observed["args"]),
    "forwarded_kwargs": list(observed["kwargs"].items()),
    "concrete_result": result,
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
        "optimizer-base snapshot failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let actual: PythonOptimizerBaseSnapshot =
        serde_json::from_slice(&output.stdout).expect("Python returns a JSON snapshot");
    assert_eq!(actual, expected_python_snapshot());
}
