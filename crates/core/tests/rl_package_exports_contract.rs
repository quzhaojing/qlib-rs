use std::{convert::Infallible, mem::size_of, path::PathBuf, process::Command};

use domain_core::{
    ActionInterpreter, Interpreter, Reward, RewardCombination, SampleSpace, Simulator,
    StateInterpreter,
};
use serde::Deserialize;

struct ExportedSpace;

impl SampleSpace<i32> for ExportedSpace {
    type Error = Infallible;

    fn validate(&self, _sample: &i32) -> Result<(), Self::Error> {
        Ok(())
    }
}

type ExportedStateInterpreter = dyn StateInterpreter<i32, Observation = i32, ObservationSpace = ExportedSpace, Error = Infallible>;
type ExportedActionInterpreter =
    dyn ActionInterpreter<i32, i32, Action = i32, ActionSpace = ExportedSpace, Error = Infallible>;

#[test]
fn crate_root_exposes_all_six_rl_package_names() {
    assert_ne!(size_of::<Option<&dyn Interpreter>>(), 0);
    assert_ne!(size_of::<Option<&ExportedStateInterpreter>>(), 0);
    assert_ne!(size_of::<Option<&ExportedActionInterpreter>>(), 0);
    assert_ne!(size_of::<Option<&dyn Reward<i32, Error = Infallible>>>(), 0);
    assert_ne!(size_of::<RewardCombination>(), 0);
    assert_ne!(
        size_of::<Option<&dyn Simulator<i32, i32, i32, Error = Infallible>>>(),
        0
    );
}

#[derive(Debug, Deserialize, PartialEq)]
struct PythonSnapshot {
    source_sha256: String,
    module_doc: Option<String>,
    public_names: Vec<String>,
    body_kinds: Vec<String>,
    imports: Vec<Vec<String>>,
    exported_names: Vec<String>,
    identity_names: Vec<String>,
    identity_modules: Vec<String>,
    facts: Vec<String>,
}

#[test]
fn live_python_package_initializer_freezes_reexport_identity_and_order() {
    assert_eq!(live_python_snapshot(), expected_python_snapshot());
}

fn live_python_snapshot() -> PythonSnapshot {
    let source = std::env::var_os("QLIB_PYTHON_RL_INIT").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/rl/__init__.py"),
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
        "RL package snapshot failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("Python returns a JSON snapshot")
}

fn expected_python_snapshot() -> PythonSnapshot {
    PythonSnapshot {
        source_sha256: "204139e0f97748a1bd02a54a7a244043c2a80ecaf2948ac74f865a00dbe06092"
            .to_owned(),
        module_doc: None,
        public_names: [
            "Interpreter",
            "StateInterpreter",
            "ActionInterpreter",
            "Reward",
            "RewardCombination",
            "Simulator",
        ]
        .map(str::to_owned)
        .to_vec(),
        body_kinds: ["ImportFrom", "ImportFrom", "ImportFrom", "Assign"]
            .map(str::to_owned)
            .to_vec(),
        imports: vec![
            ["Interpreter", "StateInterpreter", "ActionInterpreter"]
                .map(str::to_owned)
                .to_vec(),
            ["Reward", "RewardCombination"].map(str::to_owned).to_vec(),
            vec!["Simulator".to_owned()],
        ],
        exported_names: [
            "Interpreter",
            "StateInterpreter",
            "ActionInterpreter",
            "Reward",
            "RewardCombination",
            "Simulator",
        ]
        .map(str::to_owned)
        .to_vec(),
        identity_names: [
            "Interpreter",
            "StateInterpreter",
            "ActionInterpreter",
            "Reward",
            "RewardCombination",
            "Simulator",
        ]
        .map(str::to_owned)
        .to_vec(),
        identity_modules: [
            "qlib.rl.interpreter",
            "qlib.rl.interpreter",
            "qlib.rl.interpreter",
            "qlib.rl.reward",
            "qlib.rl.reward",
            "qlib.rl.simulator",
        ]
        .map(str::to_owned)
        .to_vec(),
        facts: [
            "all-is-independent-list",
            "all-matches-public-order",
            "imports-preserve-object-identity",
            "no-auxiliary-info-export",
        ]
        .map(str::to_owned)
        .to_vec(),
    }
}

const PYTHON_SNAPSHOT: &str = r#"
import ast
import hashlib
import json
import sys
import types

path = sys.argv[1]
raw = open(path, "rb").read()
tree = ast.parse(raw, filename=path)
package = types.ModuleType("qlib.rl")
package.__file__ = path
package.__package__ = "qlib.rl"
package.__path__ = []

sources = {}
for module_name, names in [
    ("qlib.rl.interpreter", ["Interpreter", "StateInterpreter", "ActionInterpreter"]),
    ("qlib.rl.reward", ["Reward", "RewardCombination"]),
    ("qlib.rl.simulator", ["Simulator"]),
]:
    module = types.ModuleType(module_name)
    for name in names:
        value = type(name, (), {"__module__": module_name})
        setattr(module, name, value)
        sources[name] = value
    sys.modules[module_name] = module
sys.modules["qlib.rl"] = package
exec(compile(raw, path, "exec"), package.__dict__)

public_names = [name for name in vars(package) if not name.startswith("_")]
facts = []
if package.__all__ is not public_names:
    facts.append("all-is-independent-list")
if package.__all__ == public_names:
    facts.append("all-matches-public-order")
if all(getattr(package, name) is sources[name] for name in package.__all__):
    facts.append("imports-preserve-object-identity")
if "AuxiliaryInfoCollector" not in vars(package) and "AuxiliaryInfoCollector" not in package.__all__:
    facts.append("no-auxiliary-info-export")

imports = [
    [alias.name for alias in node.names]
    for node in tree.body
    if isinstance(node, ast.ImportFrom)
]
print(json.dumps({
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "module_doc": package.__doc__,
    "public_names": public_names,
    "body_kinds": [type(node).__name__ for node in tree.body],
    "imports": imports,
    "exported_names": package.__all__,
    "identity_names": [getattr(package, name).__name__ for name in package.__all__],
    "identity_modules": [getattr(package, name).__module__ for name in package.__all__],
    "facts": facts,
}, ensure_ascii=False))
"#;
