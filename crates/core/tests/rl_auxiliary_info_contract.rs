use std::{convert::Infallible, path::PathBuf, process::Command};

use domain_core::AuxiliaryInfoCollector;
use serde::Deserialize;

struct BorrowingCollector;

impl AuxiliaryInfoCollector<String> for BorrowingCollector {
    type AuxiliaryInfo = (usize, String);
    type Error = Infallible;

    fn collect(&self, state: &String) -> Result<Self::AuxiliaryInfo, Self::Error> {
        Ok((state.as_ptr() as usize, state.to_uppercase()))
    }
}

#[test]
fn generic_collector_call_delegates_with_borrowed_state_and_arbitrary_output() {
    let collector: &dyn AuxiliaryInfoCollector<String, AuxiliaryInfo = (usize, String), Error = Infallible> =
        &BorrowingCollector;
    let state = "mixed Case".to_owned();
    assert_eq!(
        collector.call(&state),
        Ok((state.as_ptr() as usize, "MIXED CASE".to_owned()))
    );
    assert_eq!(
        collector.collect(&state),
        Ok((state.as_ptr() as usize, "MIXED CASE".to_owned()))
    );
}

#[derive(Debug, Deserialize, PartialEq)]
struct PythonSnapshot {
    source_sha256: String,
    module_doc: Option<String>,
    public_names: Vec<String>,
    body_kinds: Vec<String>,
    exported_names: Vec<String>,
    typevar_name: String,
    typevar_type: String,
    typevar_constraints: Vec<String>,
    typevar_bound: Option<String>,
    typevar_variance: String,
    bases: Vec<String>,
    class_doc: String,
    class_public: Vec<String>,
    call_signature: String,
    collect_signature: String,
    collect_doc: String,
    facts: Vec<String>,
    forwarded_state: Vec<i64>,
    forwarded_output: Vec<String>,
}

#[test]
fn live_python_module_freezes_auxiliary_collector_surface_and_delegation() {
    assert_eq!(live_python_snapshot(), expected_python_snapshot());
}

fn live_python_snapshot() -> PythonSnapshot {
    let source = std::env::var_os("QLIB_PYTHON_RL_AUX_INFO").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/rl/aux_info.py"),
        PathBuf::from,
    );
    assert!(
        source.is_file(),
        "Python source not found: {}",
        source.display()
    );
    let output = Command::new(std::env::var_os("PYTHON").unwrap_or_else(|| "python".into()))
        .arg("-c")
        .arg(PYTHON_SNAPSHOT)
        .arg(&source)
        .output()
        .expect("Python interpreter starts");
    assert!(
        output.status.success(),
        "RL auxiliary-info snapshot failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("Python returns a JSON snapshot")
}

fn expected_python_snapshot() -> PythonSnapshot {
    PythonSnapshot {
        source_sha256: "ef100eaf90408f832bd5e2784cfbd7621234863952238832d30f2c870b2cae64"
            .to_owned(),
        module_doc: None,
        public_names: [
            "annotations",
            "TYPE_CHECKING",
            "Generic",
            "Optional",
            "TypeVar",
            "final",
            "StateType",
            "AuxInfoType",
            "AuxiliaryInfoCollector",
        ]
        .map(str::to_owned)
        .to_vec(),
        body_kinds: [
            "ImportFrom",
            "ImportFrom",
            "ImportFrom",
            "ImportFrom",
            "If",
            "Assign",
            "Assign",
            "ClassDef",
        ]
        .map(str::to_owned)
        .to_vec(),
        exported_names: vec!["AuxiliaryInfoCollector".to_owned()],
        typevar_name: "AuxInfoType".to_owned(),
        typevar_type: "typing.TypeVar".to_owned(),
        typevar_constraints: vec![],
        typevar_bound: None,
        typevar_variance: "invariant".to_owned(),
        bases: vec!["Generic[StateType, AuxInfoType]".to_owned()],
        class_doc: "Override this class to collect customized auxiliary information from environment."
            .to_owned(),
        class_public: ["env", "collect"].map(str::to_owned).to_vec(),
        call_signature: "(self, simulator_state: 'StateType') -> 'AuxInfoType'".to_owned(),
        collect_signature: "(self, simulator_state: 'StateType') -> 'AuxInfoType'".to_owned(),
        collect_doc: "Override this for customized auxiliary info.\nUsually useful in Multi-agent RL.\n\nParameters\n----------\nsimulator_state\n    Retrieved with ``simulator.get_state()``.\n\nReturns\n-------\nAuxiliary information.\n".to_owned(),
        facts: [
            "type-checking-import-is-passive",
            "default-env-is-none",
            "call-is-final",
            "default-collect-raises-exact-error",
            "call-is-inherited",
            "call-forwards-state-identity",
            "environment-can-be-injected",
            "final-marker-is-runtime-advisory",
        ]
        .map(str::to_owned)
        .to_vec(),
        forwarded_state: vec![2, 4, 6],
        forwarded_output: ["same-state", "collected"].map(str::to_owned).to_vec(),
    }
}

const PYTHON_SNAPSHOT: &str = r#"
import ast
import hashlib
import inspect
import json
import sys
import types
import typing

path = sys.argv[1]
raw = open(path, "rb").read()
tree = ast.parse(raw, filename=path)
qlib = types.ModuleType("qlib")
qlib.__path__ = []
rl = types.ModuleType("qlib.rl")
rl.__path__ = []
typehint = types.ModuleType("qlib.typehint")
typehint.final = typing.final
simulator = types.ModuleType("qlib.rl.simulator")
simulator.StateType = typing.TypeVar("StateType")
sys.modules.update({
    "qlib": qlib,
    "qlib.rl": rl,
    "qlib.typehint": typehint,
    "qlib.rl.simulator": simulator,
})
module = types.ModuleType("qlib.rl.aux_info")
module.__file__ = path
module.__package__ = "qlib.rl"
exec(compile(raw, path, "exec"), module.__dict__)

collector = module.AuxiliaryInfoCollector
typevar = module.AuxInfoType
facts = []
if "qlib.rl.utils.env_wrapper" not in sys.modules:
    facts.append("type-checking-import-is-passive")
if collector.env is None:
    facts.append("default-env-is-none")
if collector.__call__.__final__:
    facts.append("call-is-final")
try:
    collector().collect(object())
except NotImplementedError as error:
    if str(error) == "collect is not implemented!":
        facts.append("default-collect-raises-exact-error")

observed = {}
class Concrete(collector):
    def collect(self, state):
        observed["state"] = state
        return ["same-state", "collected"]

state = [2, 4, 6]
instance = Concrete()
output = instance(state)
if instance.__call__.__func__ is collector.__call__:
    facts.append("call-is-inherited")
if observed["state"] is state:
    facts.append("call-forwards-state-identity")
environment = object()
instance.env = environment
if instance.env is environment:
    facts.append("environment-can-be-injected")

class Override(collector):
    def __call__(self, state):
        return "override"
if Override()(state) == "override":
    facts.append("final-marker-is-runtime-advisory")

node = next(item for item in tree.body if isinstance(item, ast.ClassDef))
print(json.dumps({
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "module_doc": module.__doc__,
    "public_names": [name for name in vars(module) if not name.startswith("_")],
    "body_kinds": [type(item).__name__ for item in tree.body],
    "exported_names": module.__all__,
    "typevar_name": typevar.__name__,
    "typevar_type": f"{type(typevar).__module__}.{type(typevar).__qualname__}",
    "typevar_constraints": [repr(item) for item in typevar.__constraints__],
    "typevar_bound": None if typevar.__bound__ is None else repr(typevar.__bound__),
    "typevar_variance": "invariant" if not typevar.__covariant__ and not typevar.__contravariant__ else "variant",
    "bases": [ast.unparse(item) for item in node.bases],
    "class_doc": collector.__doc__,
    "class_public": [name for name in vars(collector) if not name.startswith("_")],
    "call_signature": str(inspect.signature(collector.__call__)),
    "collect_signature": str(inspect.signature(collector.collect)),
    "collect_doc": collector.collect.__doc__,
    "facts": facts,
    "forwarded_state": observed["state"],
    "forwarded_output": output,
}, ensure_ascii=False))
"#;
