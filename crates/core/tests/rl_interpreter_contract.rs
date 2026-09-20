use std::{
    cell::RefCell, error::Error, fmt, fmt::Write as _, path::PathBuf, process::Command, rc::Rc,
};

use domain_core::{
    ActionInterpreter, GymSample, GymSpace, Interpreter, LeafSpace, SampleSpace, StateInterpreter,
    gym_space_contains,
};
use indexmap::IndexMap;
use serde::Deserialize;

#[derive(Clone, Debug, PartialEq, Eq)]
struct Bounded(&'static str, i32, i32);

impl LeafSpace<i32> for Bounded {
    fn contains(&self, value: &i32) -> bool {
        (self.1..=self.2).contains(value)
    }
}

impl fmt::Display for Bounded {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

fn leaf(name: &'static str, start: i32, end: i32) -> GymSpace<Bounded> {
    GymSpace::Leaf(Bounded(name, start, end))
}

fn value(value: i32) -> GymSample<i32> {
    GymSample::Leaf(value)
}

#[test]
fn recursive_gym_validation_preserves_order_normalization_and_error_context() {
    let tuple = GymSpace::Tuple(vec![leaf("Small", 0, 2), leaf("Large", 10, 20)]);
    let space = GymSpace::Dict(IndexMap::from([
        ("pair".to_owned(), tuple.clone()),
        ("single".to_owned(), leaf("Exact", 7, 7)),
    ]));
    let valid_pair = vec![value(2), value(12)];
    for pair in [
        GymSample::Tuple(valid_pair.clone()),
        GymSample::List(valid_pair.clone()),
        GymSample::Array(valid_pair),
    ] {
        let sample = GymSample::Dict(IndexMap::from([
            ("pair".to_owned(), pair),
            ("single".to_owned(), value(7)),
        ]));
        assert_eq!(gym_space_contains(&space, &sample), Ok(()));
        assert_eq!(SampleSpace::validate(&space, &sample), Ok(()));
    }

    let wrong_dict_kind = gym_space_contains(&space, &value(1)).unwrap_err();
    assert_eq!(
        wrong_dict_kind.message,
        "Sample must be a dict with same length as space."
    );
    let short_dict = GymSample::Dict(IndexMap::from([("pair".to_owned(), value(1))]));
    assert_eq!(
        gym_space_contains(&space, &short_dict).unwrap_err().message,
        "Sample must be a dict with same length as space."
    );
    let wrong_key = GymSample::Dict(IndexMap::from([
        ("other".to_owned(), value(1)),
        ("single".to_owned(), value(7)),
    ]));
    assert_eq!(
        gym_space_contains(&space, &wrong_key).unwrap_err().message,
        "Key pair not found in sample."
    );

    let nested = GymSample::Dict(IndexMap::from([
        (
            "pair".to_owned(),
            GymSample::List(vec![value(2), value(99)]),
        ),
        ("single".to_owned(), value(7)),
    ]));
    let error = gym_space_contains(&space, &nested).unwrap_err();
    assert_eq!(error.message, "Subspace of key pair validation error.");
    assert_eq!(*error.space, space);
    assert_eq!(*error.sample, nested);
    let tuple_error = error.cause().unwrap();
    assert_eq!(tuple_error.message, "Subspace of index 1 validation error.");
    assert!(matches!(*tuple_error.sample, GymSample::Tuple(_)));
    let leaf_error = tuple_error.cause().unwrap();
    assert_eq!(leaf_error.message, "Validation error reported by gym.");
    assert!(leaf_error.cause().is_none());
    assert_eq!(
        Error::source(&error).unwrap().to_string(),
        tuple_error.to_string()
    );
    assert_eq!(
        leaf_error.to_string(),
        "Validation error reported by gym.\n  Space: Large\n  Sample: 99"
    );
    assert_eq!(
        space.to_string(),
        "Dict({pair: Tuple((Small, Large)), single: Exact})"
    );
    assert_eq!(nested.to_string(), "{pair: (2, 99), single: 7}");
    exercise_format_failures(&space);
    exercise_format_failures(&nested);
}

struct LimitedWriter {
    remaining: usize,
}

impl fmt::Write for LimitedWriter {
    fn write_str(&mut self, value: &str) -> fmt::Result {
        if value.len() > self.remaining {
            Err(fmt::Error)
        } else {
            self.remaining -= value.len();
            Ok(())
        }
    }
}

fn exercise_format_failures(value: &impl fmt::Display) {
    let length = value.to_string().len();
    for remaining in 0..length {
        let mut writer = LimitedWriter { remaining };
        let _ = write!(&mut writer, "{value}");
    }
    let mut writer = LimitedWriter { remaining: length };
    assert!(write!(&mut writer, "{value}").is_ok());
}

#[test]
fn recursive_gym_validation_covers_tuple_and_leaf_boundary_failures() {
    let tuple = GymSpace::Tuple(vec![leaf("Only", 1, 1)]);
    assert_eq!(
        gym_space_contains(&tuple, &value(1)).unwrap_err().message,
        "Sample must be a tuple with same length as space."
    );
    let too_long = GymSample::Array(vec![value(1), value(1)]);
    let length_error = gym_space_contains(&tuple, &too_long).unwrap_err();
    assert_eq!(
        length_error.message,
        "Sample must be a tuple with same length as space."
    );
    assert_eq!(
        *length_error.sample,
        GymSample::Tuple(vec![value(1), value(1)])
    );
    assert_eq!(
        gym_space_contains(&leaf("Only", 1, 1), &GymSample::Tuple(vec![]))
            .unwrap_err()
            .message,
        "Validation error reported by gym."
    );
    assert_eq!(
        GymSpace::<Bounded>::Dict(IndexMap::new()).to_string(),
        "Dict({})"
    );
    assert_eq!(GymSpace::<Bounded>::Tuple(vec![]).to_string(), "Tuple(())");
    assert_eq!(GymSample::<i32>::List(vec![]).to_string(), "()");
    assert_eq!(GymSample::Array(vec![value(1)]).to_string(), "(1)");
}

#[derive(Clone)]
struct RecordingSpace {
    events: Rc<RefCell<Vec<&'static str>>>,
    maximum: i32,
}

impl SampleSpace<i32> for RecordingSpace {
    type Error = String;

    fn validate(&self, sample: &i32) -> Result<(), Self::Error> {
        self.events.borrow_mut().push("validate");
        if *sample <= self.maximum {
            Ok(())
        } else {
            Err("outside space".to_owned())
        }
    }
}

struct StateSide {
    space: RecordingSpace,
}

impl Interpreter for StateSide {}

impl StateInterpreter<String> for StateSide {
    type Observation = i32;
    type ObservationSpace = RecordingSpace;
    type Error = String;

    fn observation_space(&self) -> &Self::ObservationSpace {
        &self.space
    }

    fn interpret(&self, state: &String) -> Result<Self::Observation, Self::Error> {
        self.space.events.borrow_mut().push("interpret");
        state.parse().map_err(|_| "interpret failed".to_owned())
    }
}

struct ActionSide {
    space: RecordingSpace,
}

impl Interpreter for ActionSide {}

impl ActionInterpreter<String, i32> for ActionSide {
    type Action = String;
    type ActionSpace = RecordingSpace;
    type Error = String;

    fn action_space(&self) -> &Self::ActionSpace {
        &self.space
    }

    fn interpret(&self, state: &String, action: &i32) -> Result<Self::Action, Self::Error> {
        self.space.events.borrow_mut().push("interpret");
        if state == "fail" {
            Err("interpret failed".to_owned())
        } else {
            Ok(format!("{state}:{action}"))
        }
    }
}

#[test]
fn generic_interpreters_enforce_the_two_distinct_call_orders() {
    let state_events = Rc::new(RefCell::new(vec![]));
    let state = StateSide {
        space: RecordingSpace {
            events: Rc::clone(&state_events),
            maximum: 5,
        },
    };
    let state_object: &dyn StateInterpreter<
        String,
        Observation = i32,
        ObservationSpace = RecordingSpace,
        Error = String,
    > = &state;
    assert_eq!(state_object.call(&"4".to_owned()), Ok(4));
    assert_eq!(&*state_events.borrow(), &["interpret", "validate"]);
    state_events.borrow_mut().clear();
    assert_eq!(
        state_object.call(&"8".to_owned()),
        Err("outside space".to_owned())
    );
    assert_eq!(&*state_events.borrow(), &["interpret", "validate"]);
    state_events.borrow_mut().clear();
    assert_eq!(
        state_object.call(&"bad".to_owned()),
        Err("interpret failed".to_owned())
    );
    assert_eq!(&*state_events.borrow(), &["interpret"]);

    let action_events = Rc::new(RefCell::new(vec![]));
    let action = ActionSide {
        space: RecordingSpace {
            events: Rc::clone(&action_events),
            maximum: 5,
        },
    };
    let action_object: &dyn ActionInterpreter<
        String,
        i32,
        Action = String,
        ActionSpace = RecordingSpace,
        Error = String,
    > = &action;
    assert_eq!(
        action_object.call(&"state".to_owned(), &3),
        Ok("state:3".to_owned())
    );
    assert_eq!(&*action_events.borrow(), &["validate", "interpret"]);
    action_events.borrow_mut().clear();
    assert_eq!(
        action_object.call(&"state".to_owned(), &8),
        Err("outside space".to_owned())
    );
    assert_eq!(&*action_events.borrow(), &["validate"]);
    action_events.borrow_mut().clear();
    assert_eq!(
        action_object.call(&"fail".to_owned(), &3),
        Err("interpret failed".to_owned())
    );
    assert_eq!(&*action_events.borrow(), &["validate", "interpret"]);
}

#[derive(Debug, Deserialize, PartialEq)]
struct PythonSnapshot {
    source_sha256: String,
    module_doc: Option<String>,
    public_names: Vec<String>,
    body_kinds: Vec<String>,
    typevars: Vec<Vec<String>>,
    bases: Vec<Vec<String>>,
    signatures: Vec<String>,
    docs: Vec<Option<String>>,
    facts: Vec<String>,
    errors: Vec<Vec<String>>,
}

#[test]
fn live_python_module_freezes_interpreter_and_recursive_validation_contract() {
    assert_eq!(live_python_snapshot(), expected_python_snapshot());
}

fn live_python_snapshot() -> PythonSnapshot {
    let source = std::env::var_os("QLIB_PYTHON_RL_INTERPRETER").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/rl/interpreter.py"),
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
        "RL interpreter snapshot failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("Python returns a JSON snapshot")
}

fn expected_python_snapshot() -> PythonSnapshot {
    PythonSnapshot {
        source_sha256: "cffde7f3f65fdd3a2a7f5d7e4b3919d95c7951f8cec2fe7805bd2c338918b9e8"
            .to_owned(),
        module_doc: None,
        public_names: [
            "annotations",
            "Any",
            "Generic",
            "TypeVar",
            "gym",
            "np",
            "spaces",
            "final",
            "ActType",
            "StateType",
            "ObsType",
            "PolicyActType",
            "Interpreter",
            "StateInterpreter",
            "ActionInterpreter",
            "GymSpaceValidationError",
        ]
        .map(str::to_owned)
        .to_vec(),
        body_kinds: [
            "ImportFrom",
            "ImportFrom",
            "Import",
            "Import",
            "ImportFrom",
            "ImportFrom",
            "ImportFrom",
            "Assign",
            "Assign",
            "ClassDef",
            "ClassDef",
            "ClassDef",
            "FunctionDef",
            "ClassDef",
        ]
        .map(str::to_owned)
        .to_vec(),
        typevars: vec![
            vec!["ObsType".to_owned(), "invariant".to_owned()],
            vec!["PolicyActType".to_owned(), "invariant".to_owned()],
        ],
        bases: vec![
            vec![],
            vec![
                "Generic[StateType, ObsType]".to_owned(),
                "Interpreter".to_owned(),
            ],
            vec![
                "Generic[StateType, PolicyActType, ActType]".to_owned(),
                "Interpreter".to_owned(),
            ],
            vec!["Exception".to_owned()],
        ],
        signatures: [
            "(self) -> 'gym.Space'",
            "(self, simulator_state: 'StateType') -> 'ObsType'",
            "(self, obs: 'ObsType') -> 'None'",
            "(self, simulator_state: 'StateType') -> 'ObsType'",
            "(self) -> 'gym.Space'",
            "(self, simulator_state: 'StateType', action: 'PolicyActType') -> 'ActType'",
            "(self, action: 'PolicyActType') -> 'None'",
            "(self, simulator_state: 'StateType', action: 'PolicyActType') -> 'ActType'",
            "(space: 'gym.Space', x: 'Any') -> 'None'",
            "(self, message: 'str', space: 'gym.Space', x: 'Any') -> 'None'",
            "(self) -> 'str'",
        ]
        .map(str::to_owned)
        .to_vec(),
        docs: expected_docs(),
        facts: expected_facts(),
        errors: expected_errors(),
    }
}

fn expected_facts() -> Vec<String> {
    [
        "base-is-instantiable",
        "state-call-is-final",
        "action-call-is-final",
        "state-space-empty-not-implemented",
        "state-interpret-exact-not-implemented",
        "action-space-empty-not-implemented",
        "action-interpret-exact-not-implemented",
        "state-interpret-before-validation",
        "action-validation-before-interpret",
        "tuple-promotes-list",
        "tuple-promotes-ndarray",
        "dict-success",
        "dict-kind-error",
        "dict-length-error",
        "dict-key-error",
        "tuple-kind-error",
        "tuple-length-error-normalizes",
        "leaf-error",
        "error-retains-inputs",
        "nested-cause-preserved",
        "exact-error-string",
    ]
    .map(str::to_owned)
    .to_vec()
}

fn expected_docs() -> Vec<Option<String>> {
    vec![
        Some("Interpreter is a media between states produced by simulators and states needed by RL policies.\nInterpreters are two-way:\n\n1. From simulator state to policy state (aka observation), see :class:`StateInterpreter`.\n2. From policy action to action accepted by simulator, see :class:`ActionInterpreter`.\n\nInherit one of the two sub-classes to define your own interpreter.\nThis super-class is only used for isinstance check.\n\nInterpreters are recommended to be stateless, meaning that storing temporary information with ``self.xxx``\nin interpreter is anti-pattern. In future, we might support register some interpreter-related\nstates by calling ``self.env.register_state()``, but it's not planned for first iteration.\n".to_owned()),
        Some("State Interpreter that interpret execution result of qlib executor into rl env state".to_owned()),
        Some("Action Interpreter that interpret rl agent action into qlib orders".to_owned()),
        None,
        Some("Validate whether an observation belongs to the pre-defined observation space.".to_owned()),
        Some("Interpret the state of simulator.\n\nParameters\n----------\nsimulator_state\n    Retrieved with ``simulator.get_state()``.\n\nReturns\n-------\nState needed by policy. Should conform with the state space defined in ``observation_space``.\n".to_owned()),
        Some("Validate whether an action belongs to the pre-defined action space.".to_owned()),
        Some("Convert the policy action to simulator action.\n\nParameters\n----------\nsimulator_state\n    Retrieved with ``simulator.get_state()``.\naction\n    Raw action given by policy.\n\nReturns\n-------\nThe action needed by simulator,\n".to_owned()),
        Some("Strengthened version of gym.Space.contains.\nGiving more diagnostic information on why validation fails.\n\nThrow exception rather than returning true or false.\n".to_owned()),
    ]
}

fn expected_errors() -> Vec<Vec<String>> {
    vec![
        [
            "Subspace of key pair validation error.",
            "DictSpace",
            "{'pair': (1, 9)}",
        ]
        .map(str::to_owned)
        .to_vec(),
        [
            "Subspace of index 1 validation error.",
            "TupleSpace",
            "(1, 9)",
        ]
        .map(str::to_owned)
        .to_vec(),
        ["Validation error reported by gym.", "Two", "9"]
            .map(str::to_owned)
            .to_vec(),
    ]
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

class Space:
    def contains(self, value):
        return False
class Dict(Space):
    def __init__(self, spaces): self.spaces = spaces
    def __len__(self): return len(self.spaces)
    def __str__(self): return "DictSpace"
class Tuple(Space):
    def __init__(self, spaces): self.spaces = spaces
    def __len__(self): return len(self.spaces)
    def __iter__(self): return iter(self.spaces)
    def __str__(self): return "TupleSpace"
class Leaf(Space):
    def __init__(self, accepted, name): self.accepted, self.name = accepted, name
    def contains(self, value): return value in self.accepted
    def __str__(self): return self.name
class NdArray(list): pass

gym = types.ModuleType("gym")
gym.Space = Space
gym.spaces = types.SimpleNamespace(Dict=Dict, Tuple=Tuple)
np = types.ModuleType("numpy")
np.ndarray = NdArray
qlib = types.ModuleType("qlib")
qlib.__path__ = []
rl = types.ModuleType("qlib.rl")
rl.__path__ = []
typehint = types.ModuleType("qlib.typehint")
typehint.final = typing.final
simulator = types.ModuleType("qlib.rl.simulator")
simulator.ActType = typing.TypeVar("ActType")
simulator.StateType = typing.TypeVar("StateType")
sys.modules.update({
    "gym": gym, "numpy": np, "qlib": qlib, "qlib.rl": rl,
    "qlib.typehint": typehint, "qlib.rl.simulator": simulator,
})
module = types.ModuleType("qlib.rl.interpreter")
module.__file__ = path
module.__package__ = "qlib.rl"
exec(compile(raw, path, "exec"), module.__dict__)

facts = []
errors = []
if isinstance(module.Interpreter(), module.Interpreter): facts.append("base-is-instantiable")
if module.StateInterpreter.__call__.__final__: facts.append("state-call-is-final")
if module.ActionInterpreter.__call__.__final__: facts.append("action-call-is-final")
try: module.StateInterpreter().observation_space
except NotImplementedError as error:
    if str(error) == "": facts.append("state-space-empty-not-implemented")
try: module.StateInterpreter().interpret(object())
except NotImplementedError as error:
    if str(error) == "interpret is not implemented!": facts.append("state-interpret-exact-not-implemented")
try: module.ActionInterpreter().action_space
except NotImplementedError as error:
    if str(error) == "": facts.append("action-space-empty-not-implemented")
try: module.ActionInterpreter().interpret(object(), object())
except NotImplementedError as error:
    if str(error) == "interpret is not implemented!": facts.append("action-interpret-exact-not-implemented")

events = []
class StateSide(module.StateInterpreter):
    observation_space = Leaf({4}, "Obs")
    def interpret(self, state): events.append("interpret-state"); return state + 1
    def validate(self, obs): events.append("validate-state"); return super().validate(obs)
if StateSide()(3) == 4 and events == ["interpret-state", "validate-state"]:
    facts.append("state-interpret-before-validation")
events.clear()
class ActionSide(module.ActionInterpreter):
    action_space = Leaf({2}, "Act")
    def validate(self, action): events.append("validate-action"); return super().validate(action)
    def interpret(self, state, action): events.append("interpret-action"); return state + action
if ActionSide()(5, 2) == 7 and events == ["validate-action", "interpret-action"]:
    facts.append("action-validation-before-interpret")

tuple_space = Tuple([Leaf({1}, "One"), Leaf({2}, "Two")])
module._gym_space_contains(tuple_space, [1, 2]); facts.append("tuple-promotes-list")
module._gym_space_contains(tuple_space, NdArray([1, 2])); facts.append("tuple-promotes-ndarray")
dict_space = Dict({"pair": tuple_space})
module._gym_space_contains(dict_space, {"pair": (1, 2)}); facts.append("dict-success")
try: module._gym_space_contains(dict_space, (1, 2))
except module.GymSpaceValidationError as error:
    if error.message == "Sample must be a dict with same length as space.": facts.append("dict-kind-error")
try: module._gym_space_contains(dict_space, {})
except module.GymSpaceValidationError as error:
    if error.message == "Sample must be a dict with same length as space.": facts.append("dict-length-error")
try: module._gym_space_contains(dict_space, {"other": (1, 2)})
except module.GymSpaceValidationError as error:
    if error.message == "Key pair not found in sample.": facts.append("dict-key-error")
try: module._gym_space_contains(tuple_space, 1)
except module.GymSpaceValidationError as error:
    if error.message == "Sample must be a tuple with same length as space.": facts.append("tuple-kind-error")
too_long = [1, 2, 3]
try: module._gym_space_contains(tuple_space, too_long)
except module.GymSpaceValidationError as error:
    if error.message == "Sample must be a tuple with same length as space." and error.x == tuple(too_long):
        facts.append("tuple-length-error-normalizes")
try: module._gym_space_contains(Leaf({1}, "One"), 9)
except module.GymSpaceValidationError as error:
    if error.message == "Validation error reported by gym.": facts.append("leaf-error")
bad = {"pair": (1, 9)}
try:
    module._gym_space_contains(dict_space, bad)
except module.GymSpaceValidationError as error:
    if error.space is dict_space and error.x is bad: facts.append("error-retains-inputs")
    if isinstance(error.__cause__, module.GymSpaceValidationError) and isinstance(error.__cause__.__cause__, module.GymSpaceValidationError):
        facts.append("nested-cause-preserved")
    leaf_error = error.__cause__.__cause__
    if str(leaf_error) == "Validation error reported by gym.\n  Space: Two\n  Sample: 9":
        facts.append("exact-error-string")
    errors = [
        [error.message, str(error.space), str(error.x)],
        [error.__cause__.message, str(error.__cause__.space), str(error.__cause__.x)],
        [leaf_error.message, str(leaf_error.space), str(leaf_error.x)],
    ]

classes = [module.Interpreter, module.StateInterpreter, module.ActionInterpreter, module.GymSpaceValidationError]
nodes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
print(json.dumps({
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "module_doc": module.__doc__,
    "public_names": [name for name in vars(module) if not name.startswith("_")],
    "body_kinds": [type(node).__name__ for node in tree.body],
    "typevars": [[item.__name__, "invariant" if not item.__covariant__ and not item.__contravariant__ else "variant"] for item in [module.ObsType, module.PolicyActType]],
    "bases": [[ast.unparse(base) for base in nodes[item.__name__].bases] for item in classes],
    "signatures": [str(inspect.signature(item)) for item in [
        module.StateInterpreter.observation_space.fget, module.StateInterpreter.__call__,
        module.StateInterpreter.validate, module.StateInterpreter.interpret,
        module.ActionInterpreter.action_space.fget, module.ActionInterpreter.__call__,
        module.ActionInterpreter.validate, module.ActionInterpreter.interpret,
        module._gym_space_contains, module.GymSpaceValidationError.__init__,
        module.GymSpaceValidationError.__str__,
    ]],
    "docs": [item.__doc__ for item in classes] + [
        module.StateInterpreter.validate.__doc__, module.StateInterpreter.interpret.__doc__,
        module.ActionInterpreter.validate.__doc__, module.ActionInterpreter.interpret.__doc__,
        module._gym_space_contains.__doc__,
    ],
    "facts": facts,
    "errors": errors,
}, ensure_ascii=False))
"#;
