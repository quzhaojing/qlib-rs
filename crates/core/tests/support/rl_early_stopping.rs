#![allow(clippy::float_cmp)]

use super::*;
use RlCheckpointField::{Missing, Present};

#[path = "rl_early_stopping_integration.rs"]
mod integration;
use indexmap::IndexMap;
use serde_json::{Value, json};
use std::{cell::RefCell, path::PathBuf, process::Command, rc::Rc, sync::Arc};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Weights {
    weights: Vec<i64>,
}

type EventAction = Rc<dyn Fn(&str, usize)>;

#[derive(Clone, Default)]
struct Events {
    entries: Rc<RefCell<Vec<Value>>>,
    fail: Option<String>,
    action: Option<EventAction>,
    null_snapshot: bool,
}
impl Events {
    fn push(&self, kind: &str, value: Value) -> Result<(), String> {
        let mut events = self.entries.borrow_mut();
        events.push(Value::Array(vec![json!(kind), value]));
        let count = events.iter().filter(|e| e[0] == kind).count();
        if let Some(action) = &self.action {
            action(kind, count);
        }
        if self.fail.as_deref() == Some(&format!("{kind}{count}"))
            || self.fail.as_deref() == Some(kind)
        {
            return Err(self.fail.clone().unwrap());
        }
        Ok(())
    }
}
impl RlEarlyStoppingLogger for Events {
    fn log(&mut self, level: RlEarlyStoppingLogLevel, message: &str) -> Result<(), String> {
        self.push(
            match level {
                RlEarlyStoppingLogLevel::Info => "info",
                RlEarlyStoppingLogLevel::Warning => "warning",
            },
            json!(message),
        )
    }
}
impl RlCheckpointSnapshot<Weights> for Events {
    fn snapshot(&mut self, state: &Weights) -> Result<Option<Weights>, String> {
        self.push("copy", json!(state.weights))?;
        if self.null_snapshot {
            return Ok(None);
        }
        BincodeRlCheckpointSnapshot.snapshot(state)
    }
}
struct Vessel {
    weights: Weights,
    events: Events,
}
impl RlCheckpointState<Weights> for Vessel {
    fn save_checkpoint(&mut self) -> Result<Weights, String> {
        self.events.push("save", json!(self.weights.weights))?;
        Ok(self.weights.clone())
    }
    fn load_checkpoint(&mut self, state: &Weights) -> Result<(), String> {
        self.weights.clone_from(state);
        self.events.push("load", json!(self.weights.weights))
    }
}
type Callback = RlEarlyStopping<Weights, Events, Events>;

fn number(value: &Value) -> f64 {
    match value.as_str() {
        Some("nan") => f64::NAN,
        Some("inf") => f64::INFINITY,
        Some("-inf") => f64::NEG_INFINITY,
        _ => value.as_f64().unwrap(),
    }
}
fn clean(value: f64) -> Value {
    if value.is_nan() {
        json!("nan")
    } else if value.is_infinite() {
        json!(if value.is_sign_positive() {
            "inf"
        } else {
            "-inf"
        })
    } else {
        json!(value)
    }
}
fn field<T>(value: &RlCheckpointField<T>, present: impl FnOnce(&T) -> Value) -> Value {
    match value {
        Missing => json!("MISSING"),
        Present(value) => present(value),
    }
}
fn state(value: &RlEarlyStoppingState<Weights>) -> Value {
    json!({"wait":field(&value.wait, |v|json!(v.to_string().parse::<i64>().unwrap())),
        "best":field(&value.best, |v|clean(*v)), "best_weights":field(&value.best_weights, |v|json!(v)),
        "best_iter":field(&value.best_iter, |v|json!(v.to_string().parse::<i64>().unwrap()))})
}
fn error_text(error: RlEarlyStoppingError) -> String {
    match error {
        RlEarlyStoppingError::Checkpoint(RlTrainerCheckpointError::MissingField(name)) => {
            format!("missing:{name}")
        }
        RlEarlyStoppingError::MissingTrainerField(name) => format!("missing:{name}"),
        RlEarlyStoppingError::Plugin { message, .. } => message,
        other => other.to_string(),
    }
}
fn omit(state: &mut RlEarlyStoppingState<Weights>, name: &str) {
    match name {
        "wait" => state.wait = Missing,
        "best" => state.best = Missing,
        "best_weights" => state.best_weights = Missing,
        "best_iter" => state.best_iter = Missing,
        _ => panic!("unexpected field"),
    }
}
fn document() -> RlEarlyStoppingState<Weights> {
    RlEarlyStoppingState {
        wait: Present(8.into()),
        best: Present(9.0),
        best_weights: Present(Some(Weights { weights: vec![10] })),
        best_iter: Present(11.into()),
    }
}
fn config(spec: &Value) -> RlEarlyStoppingConfig {
    RlEarlyStoppingConfig {
        monitor: spec["monitor"].as_str().unwrap_or("reward").into(),
        mode: spec["mode"].as_str().unwrap_or("max").into(),
        min_delta: spec.get("min_delta").map_or(0.0, number),
        patience: spec.get("patience").map_or_else(BigInt::default, |v| {
            v.as_str()
                .map_or_else(|| v.to_string(), str::to_owned)
                .parse()
                .unwrap()
        }),
        baseline: spec.get("baseline").filter(|v| !v.is_null()).map(number),
        restore_best_weights: spec["restore_best_weights"].as_bool().unwrap_or(false),
    }
}
fn exercise(
    spec: &Value,
    cb: &mut Callback,
    runtime: &RlTrainerRuntime<Option<f64>>,
    vessel: &mut Vessel,
    snapshots: &mut Vec<Value>,
) -> Result<(), RlEarlyStoppingError> {
    if !spec["fresh"].as_bool().unwrap_or(false) {
        cb.on_fit_start();
    }
    if let Some(name) = spec["absent"].as_str() {
        omit(&mut cb.state, name);
    }
    if let Some(name) = spec["load_missing"].as_str() {
        let mut doc = document();
        omit(&mut doc, name);
        cb.load_state_dict(&doc)?;
    }
    for (i, value) in spec["values"].as_array().into_iter().flatten().enumerate() {
        let i = i64::try_from(i).unwrap();
        runtime
            .update(|s| {
                s.current_iter = if spec["missing_iter"] == true {
                    None
                } else {
                    Some((spec["start"].as_i64().unwrap_or(0) + i).into())
                };
                s.metrics = if spec["missing_metrics"] == true {
                    None
                } else if value.is_null() {
                    Some(IndexMap::from([
                        ("other".into(), Some(5.0)),
                        ("reward".into(), None),
                    ]))
                } else {
                    Some(IndexMap::from([("reward".into(), Some(number(value)))]))
                };
            })
            .unwrap();
        vessel.weights.weights = vec![100 + i];
        cb.on_validate_end(runtime, vessel)?;
        snapshots.push(json!({"state":state(&cb.state_dict()?), "stopped":runtime.read(|s|s.should_stop).unwrap(), "weights":vessel.weights.weights}));
    }
    if spec["reuse"] == true {
        cb.on_fit_start();
    }
    Ok(())
}

#[test]
fn live_source_contract_covers_thresholds_snapshots_failures_and_partial_state() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .args([
            root.join("tests/fixtures/rl_early_stopping_contract.py"),
            root.join("../../../qlib/qlib/rl/trainer/callbacks.py"),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(cases.len(), 74);
    for expected in cases {
        let spec = &expected["spec"];
        let events = Events {
            fail: spec["fail"].as_str().map(str::to_owned),
            null_snapshot: spec["null_snapshot"] == true,
            ..Events::default()
        };
        let mut vessel = Vessel {
            weights: Weights { weights: vec![1] },
            events: events.clone(),
        };
        let runtime = RlTrainerRuntime::new(None);
        runtime.update(crate::RlTrainerState::initialize).unwrap();
        let mut snapshots = vec![];
        let (actual_state, error) =
            match Callback::new(config(spec), events.clone(), events.clone()) {
                Ok(mut cb) => {
                    let result = exercise(spec, &mut cb, &runtime, &mut vessel, &mut snapshots);
                    (state(&cb.state), result.err().map(error_text))
                }
                Err(error) => (Value::Null, Some(error_text(error))),
            };
        assert_eq!(
            json!({"spec":spec, "events":*events.entries.borrow(), "state":actual_state, "snapshots":snapshots,
            "stopped":runtime.read(|s|s.should_stop).unwrap(), "weights":vessel.weights.weights, "error":error}),
            expected,
            "spec: {spec}"
        );
    }
}

#[test]
fn callback_documents_roundtrip_and_missing_fields_remain_ordered() {
    let mut cb = Callback::new(
        RlEarlyStoppingConfig::default(),
        Events::default(),
        Events::default(),
    )
    .unwrap();
    assert!(cb.save_checkpoint().is_err());
    cb.on_fit_start();
    let initial = cb.save_checkpoint().unwrap();
    let bytes = bincode::serialize(&initial).unwrap();
    assert_eq!(
        bincode::deserialize::<RlEarlyStoppingState<Weights>>(&bytes).unwrap(),
        initial
    );
    cb.load_checkpoint(&document()).unwrap();
    assert_eq!(cb.state_dict().unwrap(), document());
    let json = serde_json::to_value(cb.state_dict().unwrap()).unwrap();
    assert_eq!(
        serde_json::from_value::<RlEarlyStoppingState<Weights>>(json).unwrap(),
        document()
    );
    assert!(format!("{:?}", cb.state).contains("best_weights"));
    for name in ["wait", "best", "best_weights", "best_iter"] {
        cb.load_state_dict(&document()).unwrap();
        omit(&mut cb.state, name);
        assert!(cb.state_dict().unwrap_err().to_string().contains(name));
        let mut missing = document();
        omit(&mut missing, name);
        assert!(cb.load_checkpoint(&missing).unwrap_err().contains(name));
    }
    let empty: RlEarlyStoppingState<Weights> = serde_json::from_str("{}").unwrap();
    assert_eq!(serde_json::to_string(&empty).unwrap(), "{}");
    let null: RlEarlyStoppingState<Weights> =
        serde_json::from_str(r#"{"best_weights":null}"#).unwrap();
    assert_eq!(null.best_weights, Present(None));
    assert!(serde_json::from_str::<RlEarlyStoppingState<Weights>>(r#"{"best":"bad"}"#).is_err());
}

fn poison(runtime: &RlTrainerRuntime<Option<f64>>) {
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = runtime.update::<()>(|_| panic!("injected poison"));
        }))
        .is_err()
    );
}

#[test]
fn runtime_failures_preserve_order_and_do_not_hold_locks_across_plugins() {
    for (hook, count, poisoned) in [
        ("initial", 0, true),
        ("copy", 1, true),
        ("copy", 2, true),
        ("copy", 2, false),
        ("info", 1, true),
        ("info", 1, false),
    ] {
        let runtime = Arc::new(RlTrainerRuntime::new(None));
        runtime
            .update(|s| {
                s.initialize();
                s.current_iter = Some(1.into());
                s.metrics = Some(IndexMap::from([("reward".into(), Some(1.0))]));
            })
            .unwrap();
        let shared = runtime.clone();
        let events = Events {
            action: Some(Rc::new(move |kind, n| {
                if kind == hook && n == count {
                    if poisoned {
                        poison(&shared);
                    } else {
                        shared.update(|s| s.current_iter = None).unwrap();
                    }
                }
            })),
            ..Events::default()
        };
        let mut cb = Callback::new(
            RlEarlyStoppingConfig {
                restore_best_weights: true,
                ..RlEarlyStoppingConfig::default()
            },
            events.clone(),
            events.clone(),
        )
        .unwrap();
        let mut vessel = Vessel {
            weights: Weights { weights: vec![3] },
            events: events.clone(),
        };
        cb.on_fit_start();
        if hook == "initial" {
            poison(&runtime);
        }
        let result = cb.on_validate_end(&runtime, &mut vessel).unwrap_err();
        assert_eq!(
            result,
            if poisoned {
                RlEarlyStoppingError::Runtime(RlTrainerStateError::Poisoned)
            } else {
                RlEarlyStoppingError::MissingTrainerField("current_iter")
            }
        );
        assert!(!format!("{result:?}: {result}").is_empty());
        assert!(events.entries.borrow().iter().all(|e| e[0] != "load"));
        assert_eq!(vessel.weights.weights, vec![3]);
        if hook != "initial" {
            assert_eq!(cb.state.best, Present(1.0));
        }
    }
}

struct BadCodec(bool);
impl Serialize for BadCodec {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if self.0 {
            Err(serde::ser::Error::custom("encode failure"))
        } else {
            7_u8.serialize(serializer)
        }
    }
}
impl<'de> Deserialize<'de> for BadCodec {
    fn deserialize<D: serde::Deserializer<'de>>(_: D) -> Result<Self, D::Error> {
        Err(serde::de::Error::custom("decode failure"))
    }
}
#[test]
fn bundled_snapshot_reports_both_codec_stages() {
    for (value, expected) in [(true, "encode failure"), (false, "decode failure")] {
        assert!(
            BincodeRlCheckpointSnapshot
                .snapshot(&BadCodec(value))
                .err()
                .unwrap()
                .contains(expected)
        );
    }
}

// Intentionally neither Clone nor Serialize: the snapshot adapter must own the model's
// deep-copy semantics instead of imposing a JSON/codec requirement on every model state.
struct Opaque(Rc<RefCell<Vec<i64>>>);
struct OpaqueVessel(Rc<RefCell<Vec<i64>>>);
struct DeepCopy;
impl RlCheckpointState<Opaque> for OpaqueVessel {
    fn save_checkpoint(&mut self) -> Result<Opaque, String> {
        Ok(Opaque(self.0.clone()))
    }
    fn load_checkpoint(&mut self, state: &Opaque) -> Result<(), String> {
        self.0.borrow_mut().clone_from(&state.0.borrow());
        Ok(())
    }
}
impl RlCheckpointSnapshot<Opaque> for DeepCopy {
    fn snapshot(&mut self, state: &Opaque) -> Result<Option<Opaque>, String> {
        Ok(Some(Opaque(Rc::new(RefCell::new(
            state.0.borrow().clone(),
        )))))
    }
}
#[test]
fn opaque_model_snapshots_detach_mutable_handles_without_clone_or_serde_bounds() {
    let weights = Rc::new(RefCell::new(vec![3]));
    let mut vessel = OpaqueVessel(weights.clone());
    let runtime = RlTrainerRuntime::<f64>::new(None);
    runtime
        .update(|s| {
            s.initialize();
            s.metrics = Some(IndexMap::from([("reward".into(), 1.0)]));
        })
        .unwrap();
    let mut cb = RlEarlyStopping::new(
        RlEarlyStoppingConfig {
            restore_best_weights: true,
            ..RlEarlyStoppingConfig::default()
        },
        DeepCopy,
        Events::default(),
    )
    .unwrap();
    cb.on_fit_start();
    cb.on_validate_end(&runtime, &mut vessel).unwrap();
    let best = &cb
        .state
        .best_weights
        .require("best_weights")
        .unwrap()
        .as_ref()
        .unwrap()
        .0;
    assert!(!Rc::ptr_eq(best, &weights));
    weights.borrow_mut()[0] = 99;
    assert_eq!(*best.borrow(), vec![3]);
    runtime.update(|s| s.current_iter = Some(1.into())).unwrap();
    cb.on_validate_end(&runtime, &mut vessel).unwrap();
    assert_eq!(*weights.borrow(), vec![3]);
    assert_eq!(runtime.read(|s| s.should_stop).unwrap(), Some(true));
}
