#![allow(clippy::float_cmp)]

use super::*;
use serde_json::{Value, json};
use std::{cell::RefCell, path::PathBuf, process::Command, rc::Rc, sync::Arc};

type Document = RlTrainerCheckpoint<i64, i64, i64>;
type Runtime = RlTrainerRuntime<f64>;

#[path = "rl_trainer_checkpoint_integration.rs"]
mod integration;

fn fixture() -> Value {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .args([
            root.join("tests/fixtures/rl_trainer_checkpoint_contract.py"),
            root.join("../../../qlib/qlib/rl/trainer/trainer.py"),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

fn runtime() -> Runtime {
    let runtime = Runtime::new(None);
    runtime
        .update(|state| {
            state.initialize();
            state.current_iter = Some(1.into());
            state.current_episode = Some(2.into());
            state.metrics = Some(IndexMap::from([("old".into(), 9.0)]));
        })
        .unwrap();
    runtime
}

fn snapshot(runtime: &Runtime) -> Value {
    runtime.read(|s|json!({"should_stop":s.should_stop,"current_iter":s.current_iter.as_ref().map(|v|v.to_string().parse::<i64>().unwrap()),
        "current_episode":s.current_episode.as_ref().map(|v|v.to_string().parse::<i64>().unwrap()),"current_stage":s.current_stage,"metrics":s.metrics})).unwrap()
}

fn document() -> Document {
    use RlCheckpointField::Present as P;
    Document {
        vessel: P(10),
        callbacks: P(IndexMap::from([
            ("foo".into(), 20),
            ("foo1".into(), 22),
            ("foo2".into(), 23),
            ("unused".into(), 29),
        ])),
        loggers: P(IndexMap::from([
            ("log".into(), 30),
            ("log1".into(), 31),
            ("unused".into(), 39),
        ])),
        should_stop: P(true),
        current_iter: P(12.into()),
        current_episode: P(34.into()),
        current_stage: P("val".into()),
        metrics: P(IndexMap::from([("score".into(), 5.0)])),
    }
}

fn omit(doc: &mut Document, path: &str) {
    use RlCheckpointField::Missing as X;
    match path {
        "vessel" => doc.vessel = X,
        "callbacks" => doc.callbacks = X,
        "loggers" => doc.loggers = X,
        "should_stop" => doc.should_stop = X,
        "current_iter" => doc.current_iter = X,
        "current_episode" => doc.current_episode = X,
        "current_stage" => doc.current_stage = X,
        "metrics" => doc.metrics = X,
        other => {
            let (group, name) = other.split_once('.').unwrap();
            let RlCheckpointField::Present(values) = (if group == "callbacks" {
                &mut doc.callbacks
            } else {
                &mut doc.loggers
            }) else {
                panic!("expected present")
            };
            values.shift_remove(name).unwrap();
        }
    }
}

struct Component {
    identity: String,
    name: String,
    value: i64,
    events: Rc<RefCell<Vec<String>>>,
    fail: Option<String>,
}
impl Component {
    fn event(&mut self, operation: &str) -> Result<(), String> {
        let event = format!("{operation}:{}", self.identity);
        self.events.borrow_mut().push(event.clone());
        if self.fail.as_ref() == Some(&event) {
            Err(event)
        } else {
            Ok(())
        }
    }
}
impl RlCheckpointState<i64> for Component {
    fn save_checkpoint(&mut self) -> Result<i64, String> {
        self.value += 1;
        self.event("save")?;
        Ok(self.value)
    }
    fn load_checkpoint(&mut self, value: &i64) -> Result<(), String> {
        self.value = *value;
        self.event("load")
    }
}
fn named(components: &mut [Component]) -> Vec<RlNamedCheckpointComponent<'_, i64>> {
    components
        .iter_mut()
        .map(|c| RlNamedCheckpointComponent {
            type_name: c.name.clone(),
            state: c,
        })
        .collect()
}
fn category(error: &RlTrainerCheckpointError) -> String {
    match error {
        RlTrainerCheckpointError::Component { message, .. } => message.clone(),
        RlTrainerCheckpointError::MissingField(path) => {
            format!("missing:{}", path.rsplit('.').next().unwrap())
        }
        RlTrainerCheckpointError::Uninitialized(field) => format!("uninitialized:{field}"),
        RlTrainerCheckpointError::Runtime(_) => panic!("unexpected poison"),
    }
}

#[test]
fn naming_collisions_unicode_and_graph_partial_failures_match_python() {
    let f = fixture();
    for case in f["names"].as_array().unwrap() {
        let names = case["input"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap());
        let output = named_rl_checkpoint_indices(names)
            .into_iter()
            .map(|(key, index)| json!([key, index]))
            .collect::<Vec<_>>();
        assert_eq!(json!(output), case["output"]);
    }
    assert_eq!(f["cases"].as_array().unwrap().len(), 33);
    for case in f["cases"].as_array().unwrap() {
        compare_case(case);
    }
}

fn compare_case(case: &Value) {
    let spec = &case["spec"];
    let runtime = runtime();
    if let Some(field) = spec["absent"].as_str() {
        runtime
            .update(|state| match field {
                "should_stop" => state.should_stop = None,
                "current_iter" => state.current_iter = None,
                "current_episode" => state.current_episode = None,
                "metrics" => state.metrics = None,
                _ => panic!("unknown"),
            })
            .unwrap();
    }
    let events = Rc::new(RefCell::new(vec![]));
    let mut components = vec![
        ("Vessel", "vessel"),
        ("Foo", "c0"),
        ("Foo1", "c1"),
        ("Foo", "c2"),
        ("Foo", "c3"),
        ("Log", "l0"),
        ("Log", "l1"),
    ]
    .into_iter()
    .map(|(name, id)| Component {
        name: name.into(),
        identity: id.into(),
        value: 0,
        events: events.clone(),
        fail: spec["fail"].as_str().map(str::to_owned),
    })
    .collect::<Vec<_>>();
    if spec["empty"] == true {
        components.truncate(1);
    }
    let mut doc = document();
    if let Some(path) = spec["missing"].as_str() {
        omit(&mut doc, path);
    }
    if spec["empty"] == true {
        omit(&mut doc, "callbacks");
        omit(&mut doc, "loggers");
    }
    let result = {
        let (vessel, rest) = components.split_first_mut().unwrap();
        let (callbacks, loggers) = rest.split_at_mut(if spec["empty"] == true { 0 } else { 4 });
        let mut callbacks = named(callbacks);
        let mut loggers = named(loggers);
        if spec["load"] == true {
            load_rl_trainer_checkpoint(&runtime, vessel, &mut callbacks, &mut loggers, &doc)
                .map(|()| None)
        } else {
            save_rl_trainer_checkpoint(&runtime, vessel, &mut callbacks, &mut loggers).map(Some)
        }
    };
    let (output, error) = match result {
        Ok(Some(doc)) => (round_trip_document(&doc), Value::Null),
        Ok(None) => (Value::Null, Value::Null),
        Err(error) => {
            assert_eq!(error, error.clone());
            assert!(!error.to_string().is_empty());
            (Value::Null, json!(category(&error)))
        }
    };
    assert_eq!(json!(*events.borrow()), case["events"], "{spec}");
    assert_eq!(
        json!(
            components
                .iter()
                .map(|c| json!([c.identity, c.value]))
                .collect::<Vec<_>>()
        ),
        case["components"],
        "{spec}"
    );
    assert_eq!(output, case["output"], "{spec}");
    assert_eq!(error, case["error"], "{spec}");
    assert_eq!(snapshot(&runtime), case["state"], "{spec}");
}

fn round_trip_document(doc: &Document) -> Value {
    let mut value = serde_json::to_value(doc).unwrap();
    assert_eq!(
        &serde_json::from_value::<Document>(value.clone()).unwrap(),
        doc
    );
    assert_eq!(
        &bincode::deserialize::<Document>(&bincode::serialize(doc).unwrap()).unwrap(),
        doc
    );
    assert_eq!(doc, &doc.clone());
    assert!(format!("{doc:?}").contains("vessel"));
    value["current_iter"] = json!(
        doc.current_iter
            .require("current_iter")
            .unwrap()
            .to_string()
            .parse::<i64>()
            .unwrap()
    );
    value["current_episode"] = json!(
        doc.current_episode
            .require("current_episode")
            .unwrap()
            .to_string()
            .parse::<i64>()
            .unwrap()
    );
    value
}

#[test]
fn missing_and_null_fields_are_distinct_and_serialization_is_explicit() {
    type Nullable = RlTrainerCheckpoint<Option<i64>, Option<i64>, Option<i64>>;
    let doc: Nullable =
        serde_json::from_value(json!({"vessel":null,"callbacks":{"none":null},"extra":123}))
            .unwrap();
    assert_eq!(doc.vessel, RlCheckpointField::Present(None));
    assert_eq!(doc.callbacks.require("callbacks").unwrap()["none"], None);
    assert!(doc.metrics.is_missing());
    assert_eq!(
        serde_json::to_value(&doc).unwrap(),
        json!({"vessel":null,"callbacks":{"none":null}})
    );
    assert_eq!(
        serde_json::to_value(Nullable::default()).unwrap(),
        json!({})
    );
    let missing = RlCheckpointField::<i64>::default();
    assert!(missing.is_missing());
    assert_eq!(
        serde_json::to_string(&RlCheckpointField::Present(7_i64)).unwrap(),
        "7"
    );
    assert!(
        serde_json::to_string(&missing)
            .unwrap_err()
            .to_string()
            .contains("no standalone value")
    );
    assert!(serde_json::from_value::<Nullable>(json!({"should_stop":"not a boolean"})).is_err());
}

struct Opaque(Arc<Vec<i64>>);
struct OpaqueState(Arc<Vec<i64>>);
impl RlCheckpointState<Opaque> for OpaqueState {
    fn save_checkpoint(&mut self) -> Result<Opaque, String> {
        Ok(Opaque(self.0.clone()))
    }
    fn load_checkpoint(&mut self, state: &Opaque) -> Result<(), String> {
        assert!(Arc::ptr_eq(&self.0, &state.0));
        Ok(())
    }
}

#[test]
fn opaque_payloads_do_not_require_clone_default_or_serialization() {
    let payload = Arc::new(vec![1, 2, 3]);
    let runtime = RlTrainerRuntime::<Arc<Vec<i64>>>::new(None);
    runtime
        .update(|s| {
            s.initialize();
            s.metrics = Some(IndexMap::from([("tensor".into(), payload.clone())]));
        })
        .unwrap();
    let mut vessel = OpaqueState(payload.clone());
    let mut callback = OpaqueState(payload.clone());
    let mut logger = OpaqueState(payload.clone());
    let mut callbacks = [RlNamedCheckpointComponent {
        type_name: "Callback".into(),
        state: &mut callback as &mut dyn RlCheckpointState<Opaque>,
    }];
    let mut loggers = [RlNamedCheckpointComponent {
        type_name: "Log".into(),
        state: &mut logger as &mut dyn RlCheckpointState<Opaque>,
    }];
    let doc =
        save_rl_trainer_checkpoint(&runtime, &mut vessel, &mut callbacks, &mut loggers).unwrap();
    assert!(Arc::ptr_eq(
        &doc.vessel.require("vessel").unwrap().0,
        &payload
    ));
    assert!(Arc::ptr_eq(
        &doc.metrics.require("metrics").unwrap()["tensor"],
        &payload
    ));
    load_rl_trainer_checkpoint(&runtime, &mut vessel, &mut callbacks, &mut loggers, &doc).unwrap();
    let missing = RlTrainerCheckpoint::<Opaque, Opaque, Opaque, Opaque>::default();
    assert!(missing.vessel.is_missing());
}

#[test]
fn poisoned_runtime_is_reported_only_after_component_side_effects() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let runtime = runtime();
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _ = runtime.update::<()>(|_| panic!("poison"));
        }))
        .is_err()
    );
    let events = Rc::new(RefCell::new(vec![]));
    let mut vessel = Component {
        identity: "vessel".into(),
        name: "Vessel".into(),
        value: 0,
        events: events.clone(),
        fail: None,
    };
    let mut callbacks: Vec<RlNamedCheckpointComponent<'_, i64>> = vec![];
    let mut loggers: Vec<RlNamedCheckpointComponent<'_, i64>> = vec![];
    let error = RlTrainerCheckpointError::Runtime(RlTrainerStateError::Poisoned);
    assert_eq!(
        save_rl_trainer_checkpoint(&runtime, &mut vessel, &mut callbacks, &mut loggers),
        Err(error.clone())
    );
    assert_eq!(vessel.value, 1);
    assert_eq!(
        load_rl_trainer_checkpoint(
            &runtime,
            &mut vessel,
            &mut callbacks,
            &mut loggers,
            &document()
        ),
        Err(error.clone())
    );
    assert_eq!(vessel.value, 10);
    assert_eq!(*events.borrow(), vec!["save:vessel", "load:vessel"]);
    assert!(!error.to_string().is_empty());
}
