use std::{convert::Infallible, path::PathBuf, process::Command};

use domain_core::Reward;
use serde::Deserialize;

#[derive(Debug, Deserialize, PartialEq)]
struct PythonRewardSnapshot {
    source_sha256: String,
    module_doc: Option<String>,
    public_names: Vec<String>,
    body_kinds: Vec<String>,
    typevar_name: String,
    typevar_type: String,
    typevar_constraints: Vec<String>,
    typevar_bound: Option<String>,
    reward_bases: Vec<String>,
    reward_doc: String,
    reward_class_public: Vec<String>,
    reward_call_signature: String,
    reward_method_signature: String,
    reward_method_doc: String,
    log_signature: String,
    combination_bases: Vec<String>,
    combination_doc: String,
    combination_public: Vec<String>,
    verified_facts: Vec<String>,
    forwarded_state: Vec<i64>,
    forwarded_value: f64,
    logged_name: String,
    logged_value: Vec<i64>,
}

struct DifferenceReward;

impl Reward<(i32, i32)> for DifferenceReward {
    type Error = Infallible;

    fn reward(&self, &(left, right): &(i32, i32)) -> Result<f64, Self::Error> {
        Ok(f64::from(left - right))
    }
}

#[test]
fn generic_reward_call_delegates_without_constraining_state_or_error() {
    let reward: &dyn Reward<(i32, i32), Error = Infallible> = &DifferenceReward;
    assert_eq!(reward.call(&(8, 3)), Ok(5.0));
    assert_eq!(reward.reward(&(2, 7)), Ok(-5.0));
}

#[test]
fn live_python_module_freezes_generic_reward_and_combination_surface() {
    assert_eq!(live_python_snapshot(), expected_python_snapshot());
}

fn live_python_snapshot() -> PythonRewardSnapshot {
    let source = std::env::var_os("QLIB_PYTHON_RL_REWARD").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/rl/reward.py"),
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
        "RL reward snapshot failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("Python returns a JSON snapshot")
}

fn expected_python_snapshot() -> PythonRewardSnapshot {
    PythonRewardSnapshot {
        source_sha256: "eeab0f252c8a1c02d19f97da09ba352db6fde5cfc02ff66fb0b11787d8a0566d"
            .to_owned(),
        module_doc: None,
        public_names: [
            "annotations",
            "TYPE_CHECKING",
            "Any",
            "Dict",
            "Generic",
            "Optional",
            "Tuple",
            "TypeVar",
            "final",
            "SimulatorState",
            "Reward",
            "RewardCombination",
        ]
        .map(str::to_owned)
        .to_vec(),
        body_kinds: [
            "ImportFrom",
            "ImportFrom",
            "ImportFrom",
            "If",
            "Assign",
            "ClassDef",
            "ClassDef",
        ]
        .map(str::to_owned)
        .to_vec(),
        typevar_name: "SimulatorState".to_owned(),
        typevar_type: "typing.TypeVar".to_owned(),
        typevar_constraints: vec![],
        typevar_bound: None,
        reward_bases: vec!["Generic[SimulatorState]".to_owned()],
        reward_doc: "\nReward calculation component that takes a single argument: state of simulator. Returns a real number: reward.\n\nSubclass should implement ``reward(simulator_state)`` to implement their own reward calculation recipe.\n"
            .to_owned(),
        reward_class_public: ["env", "reward", "log"].map(str::to_owned).to_vec(),
        reward_call_signature: "(self, simulator_state: 'SimulatorState') -> 'float'".to_owned(),
        reward_method_signature: "(self, simulator_state: 'SimulatorState') -> 'float'".to_owned(),
        reward_method_doc: "Implement this method for your own reward.".to_owned(),
        log_signature: "(self, name: 'str', value: 'Any') -> 'None'".to_owned(),
        combination_bases: vec!["Reward".to_owned()],
        combination_doc: "Combination of multiple reward.".to_owned(),
        combination_public: vec!["reward".to_owned()],
        verified_facts: [
            "default-env-is-none",
            "typevar-is-invariant",
            "reward-call-is-final",
            "default-reward-raises-exact-error",
            "missing-env-log-asserts",
            "call-delegates-to-override",
            "log-forwards-value-identity",
            "combination-call-is-inherited",
            "combination-preserves-mapping-identity",
        ]
        .map(str::to_owned)
        .to_vec(),
        forwarded_state: vec![1, 2, 3],
        forwarded_value: 6.0,
        logged_name: "vector".to_owned(),
        logged_value: vec![4, 5],
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
typehint = types.ModuleType("qlib.typehint")
typehint.final = typing.final
sys.modules["qlib"] = qlib
sys.modules["qlib.typehint"] = typehint
module = types.ModuleType("qlib.rl.reward")
module.__file__ = path
module.__package__ = "qlib.rl"
exec(compile(raw, path, "exec"), module.__dict__)
reward = module.Reward
combination = module.RewardCombination
typevar = module.SimulatorState

facts = []
if reward.env is None:
    facts.append("default-env-is-none")
if not typevar.__covariant__ and not typevar.__contravariant__:
    facts.append("typevar-is-invariant")
if reward.__call__.__final__:
    facts.append("reward-call-is-final")
try:
    reward()(object())
except NotImplementedError as error:
    if str(error) == "Implement reward calculation recipe in `reward()`." :
        facts.append("default-reward-raises-exact-error")
try:
    reward().log("missing", 1)
except AssertionError:
    facts.append("missing-env-log-asserts")

observed = {}
class Concrete(reward):
    def reward(self, state):
        observed["state"] = state
        return float(sum(state))

state = [1, 2, 3]
value = Concrete()(state)
if observed["state"] is state:
    facts.append("call-delegates-to-override")

class Logger:
    def add_scalar(self, name, value):
        observed["log"] = (name, value)

logged_value = [4, 5]
logging_reward = Concrete()
logging_reward.env = types.SimpleNamespace(logger=Logger())
logging_reward.log("vector", logged_value)
if observed["log"][1] is logged_value:
    facts.append("log-forwards-value-identity")

mapping = {"only": (Concrete(), 2.0)}
combined = combination(mapping)
if combined.__call__.__func__ is reward.__call__:
    facts.append("combination-call-is-inherited")
if combined.rewards is mapping:
    facts.append("combination-preserves-mapping-identity")

reward_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Reward")
combination_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RewardCombination")
print(json.dumps({
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "module_doc": module.__doc__,
    "public_names": [name for name in vars(module) if not name.startswith("_")],
    "body_kinds": [type(node).__name__ for node in tree.body],
    "typevar_name": typevar.__name__,
    "typevar_type": f"{type(typevar).__module__}.{type(typevar).__qualname__}",
    "typevar_constraints": [repr(item) for item in typevar.__constraints__],
    "typevar_bound": None if typevar.__bound__ is None else repr(typevar.__bound__),
    "reward_bases": [ast.unparse(item) for item in reward_node.bases],
    "reward_doc": reward.__doc__,
    "reward_class_public": [name for name in vars(reward) if not name.startswith("_")],
    "reward_call_signature": str(inspect.signature(reward.__call__)),
    "reward_method_signature": str(inspect.signature(reward.reward)),
    "reward_method_doc": reward.reward.__doc__,
    "log_signature": str(inspect.signature(reward.log)),
    "combination_bases": [ast.unparse(item) for item in combination_node.bases],
    "combination_doc": combination.__doc__,
    "combination_public": [name for name in vars(combination) if not name.startswith("_")],
    "verified_facts": facts,
    "forwarded_state": observed["state"],
    "forwarded_value": value,
    "logged_name": observed["log"][0],
    "logged_value": observed["log"][1],
}, ensure_ascii=False))
"#;
