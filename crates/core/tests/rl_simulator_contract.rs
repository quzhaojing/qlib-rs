use std::{convert::Infallible, path::PathBuf, process::Command};

use domain_core::{ActType, Simulator, StateType};
use serde::Deserialize;

#[derive(Debug, Deserialize, PartialEq)]
struct PythonSimulatorSnapshot {
    source_sha256: String,
    module_doc: Option<String>,
    public_names: Vec<String>,
    body_kinds: Vec<String>,
    state_type_doc: String,
    action_type_doc: String,
    typevars: Vec<TypeVarSnapshot>,
    bases: Vec<String>,
    class_doc: String,
    class_public_names: Vec<String>,
    init_signature: String,
    step_signature: String,
    step_doc: String,
    state_signature: String,
    state_doc: Option<String>,
    done_signature: String,
    done_doc: String,
    verified_facts: Vec<String>,
    forwarded_action: Vec<i64>,
    returned_state: Vec<i64>,
}

#[derive(Debug, Deserialize, PartialEq)]
struct TypeVarSnapshot {
    name: String,
    runtime_type: String,
    constraints: Vec<String>,
    bound: Option<String>,
}

struct CounterSimulator {
    value: i64,
}

impl Simulator<String, StateType<i64>, ActType<i64>> for CounterSimulator {
    type Error = Infallible;

    fn step(&mut self, action: i64) -> Result<(), Self::Error> {
        self.value += action;
        Ok(())
    }

    fn get_state(&self) -> Result<i64, Self::Error> {
        Ok(self.value)
    }

    fn done(&self) -> Result<bool, Self::Error> {
        Ok(self.value >= 5)
    }
}

#[test]
fn native_simulator_contract_keeps_initial_state_state_and_action_independent() {
    let simulator: &mut dyn Simulator<String, i64, i64, Error = Infallible> =
        &mut CounterSimulator { value: 1 };
    assert_eq!(simulator.get_state(), Ok(1));
    assert_eq!(simulator.done(), Ok(false));
    assert_eq!(simulator.step(4), Ok(()));
    assert_eq!(simulator.get_state(), Ok(5));
    assert_eq!(simulator.done(), Ok(true));
}

#[test]
fn live_python_module_freezes_generic_simulator_contract() {
    assert_eq!(live_python_snapshot(), expected_python_snapshot());
}

fn live_python_snapshot() -> PythonSimulatorSnapshot {
    let source = std::env::var_os("QLIB_PYTHON_RL_SIMULATOR").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/rl/simulator.py"),
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
        "RL simulator snapshot failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("Python returns a JSON snapshot")
}

fn expected_python_snapshot() -> PythonSimulatorSnapshot {
    PythonSimulatorSnapshot {
        source_sha256: "264d575976116ebab9c9a844f951fb19d6b6e0f2c5cfb20fc77a6a05bda7a4c2"
            .to_owned(),
        module_doc: None,
        public_names: [
            "annotations",
            "TYPE_CHECKING",
            "Any",
            "Generic",
            "Optional",
            "TypeVar",
            "InitialStateType",
            "StateType",
            "ActType",
            "Simulator",
        ]
        .map(str::to_owned)
        .to_vec(),
        body_kinds: [
            "ImportFrom",
            "ImportFrom",
            "ImportFrom",
            "If",
            "Assign",
            "Expr",
            "Assign",
            "Expr",
            "ClassDef",
        ]
        .map(str::to_owned)
        .to_vec(),
        state_type_doc: "StateType stores all the useful data in the simulation process\n(as well as utilities to generate/retrieve data when needed)."
            .to_owned(),
        action_type_doc: "This ActType is the type of action at the simulator end.".to_owned(),
        typevars: ["InitialStateType", "StateType", "ActType"]
            .into_iter()
            .map(|name| TypeVarSnapshot {
                name: name.to_owned(),
                runtime_type: "typing.TypeVar".to_owned(),
                constraints: vec![],
                bound: None,
            })
            .collect(),
        bases: vec!["Generic[InitialStateType, StateType, ActType]".to_owned()],
        class_doc: "\nSimulator that resets with ``__init__``, and transits with ``step(action)``.\n\nTo make the data-flow clear, we make the following restrictions to Simulator:\n\n1. The only way to modify the inner status of a simulator is by using ``step(action)``.\n2. External modules can *read* the status of a simulator by using ``simulator.get_state()``,\n   and check whether the simulator is in the ending state by calling ``simulator.done()``.\n\nA simulator is defined to be bounded with three types:\n\n- *InitialStateType* that is the type of the data used to create the simulator.\n- *StateType* that is the type of the **status** (state) of the simulator.\n- *ActType* that is the type of the **action**, which is the input received in each step.\n\nDifferent simulators might share the same StateType. For example, when they are dealing with the same task,\nbut with different simulation implementation. With the same type, they can safely share other components in the MDP.\n\nSimulators are ephemeral. The lifecycle of a simulator starts with an initial state, and ends with the trajectory.\nIn another word, when the trajectory ends, simulator is recycled.\nIf simulators want to share context between (e.g., for speed-up purposes),\nthis could be done by accessing the weak reference of environment wrapper.\n\nAttributes\n----------\nenv\n    A reference of env-wrapper, which could be useful in some corner cases.\n    Simulators are discouraged to use this, because it's prone to induce errors.\n"
            .to_owned(),
        class_public_names: ["env", "step", "get_state", "done"]
            .map(str::to_owned)
            .to_vec(),
        init_signature: "(self, initial: 'InitialStateType', **kwargs: 'Any') -> 'None'".to_owned(),
        step_signature: "(self, action: 'ActType') -> 'None'".to_owned(),
        step_doc: "Receives an action of ActType.\n\nSimulator should update its internal state, and return None.\nThe updated state can be retrieved with ``simulator.get_state()``.\n"
            .to_owned(),
        state_signature: "(self) -> 'StateType'".to_owned(),
        state_doc: None,
        done_signature: "(self) -> 'bool'".to_owned(),
        done_doc: "Check whether the simulator is in a \"done\" state.\nWhen simulator is in a \"done\" state,\nit should no longer receives any ``step`` request.\nAs simulators are ephemeral, to reset the simulator,\nthe old one should be destroyed and a new simulator can be created.\n"
            .to_owned(),
        verified_facts: [
            "typevars-are-invariant",
            "default-env-is-none",
            "base-init-ignores-input-and-keywords",
            "base-methods-raise-empty-not-implemented",
            "subclass-inherits-base-init",
            "subclass-step-receives-action-identity",
            "subclass-state-preserves-identity",
            "subclass-done-result-is-forwarded",
        ]
        .map(str::to_owned)
        .to_vec(),
        forwarded_action: vec![3, 4],
        returned_state: vec![7, 8],
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
seed = types.ModuleType("qlib.rl.seed")
seed.InitialStateType = typing.TypeVar("InitialStateType")
sys.modules["qlib"] = qlib
sys.modules["qlib.rl"] = rl
sys.modules["qlib.rl.seed"] = seed
module = types.ModuleType("qlib.rl.simulator")
module.__file__ = path
module.__package__ = "qlib.rl"
exec(compile(raw, path, "exec"), module.__dict__)
simulator = module.Simulator
typevars = [module.InitialStateType, module.StateType, module.ActType]

facts = []
if all(not item.__covariant__ and not item.__contravariant__ for item in typevars):
    facts.append("typevars-are-invariant")
if simulator.env is None:
    facts.append("default-env-is-none")
base = simulator(object(), ignored=1)
if vars(base) == {}:
    facts.append("base-init-ignores-input-and-keywords")
errors = []
for call in (lambda: base.step(object()), base.get_state, base.done):
    try:
        call()
    except NotImplementedError as error:
        errors.append(str(error))
if errors == ["", "", ""]:
    facts.append("base-methods-raise-empty-not-implemented")

observed = {}
state = [7, 8]
class Concrete(simulator):
    def step(self, action):
        observed["action"] = action
    def get_state(self):
        return state
    def done(self):
        return True

concrete = Concrete("initial", option=True)
if vars(concrete) == {}:
    facts.append("subclass-inherits-base-init")
action = [3, 4]
concrete.step(action)
if observed["action"] is action:
    facts.append("subclass-step-receives-action-identity")
returned_state = concrete.get_state()
if returned_state is state:
    facts.append("subclass-state-preserves-identity")
if concrete.done() is True:
    facts.append("subclass-done-result-is-forwarded")

class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef))
print(json.dumps({
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "module_doc": module.__doc__,
    "public_names": [name for name in vars(module) if not name.startswith("_")],
    "body_kinds": [type(node).__name__ for node in tree.body],
    "state_type_doc": ast.get_docstring(ast.Module(body=[tree.body[5]], type_ignores=[]), clean=False),
    "action_type_doc": ast.get_docstring(ast.Module(body=[tree.body[7]], type_ignores=[]), clean=False),
    "typevars": [{
        "name": item.__name__,
        "runtime_type": f"{type(item).__module__}.{type(item).__qualname__}",
        "constraints": [repr(value) for value in item.__constraints__],
        "bound": None if item.__bound__ is None else repr(item.__bound__),
    } for item in typevars],
    "bases": [ast.unparse(item) for item in class_node.bases],
    "class_doc": simulator.__doc__,
    "class_public_names": [name for name in vars(simulator) if not name.startswith("_")],
    "init_signature": str(inspect.signature(simulator.__init__)),
    "step_signature": str(inspect.signature(simulator.step)),
    "step_doc": simulator.step.__doc__,
    "state_signature": str(inspect.signature(simulator.get_state)),
    "state_doc": simulator.get_state.__doc__,
    "done_signature": str(inspect.signature(simulator.done)),
    "done_doc": simulator.done.__doc__,
    "verified_facts": facts,
    "forwarded_action": observed["action"],
    "returned_state": returned_state,
}, ensure_ascii=False))
"#;
