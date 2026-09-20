use std::{path::PathBuf, process::Command, sync::Mutex};

use serde_json::{Value, json};

use super::*;

struct TrainerData {
    iteration: BigInt,
    fast_dev: Option<i64>,
    fail: Option<TrainingTrainerField>,
}

struct Trainer(Mutex<TrainerData>);

impl Trainer {
    fn new() -> Self {
        Self(Mutex::new(TrainerData {
            iteration: (-1).into(),
            fast_dev: None,
            fail: None,
        }))
    }
}

impl TrainingTrainerView for Trainer {
    fn current_iteration(&self) -> Result<BigInt, String> {
        let state = self.0.lock().unwrap();
        if state.fail == Some(TrainingTrainerField::CurrentIteration) {
            Err("iteration access".into())
        } else {
            Ok(state.iteration.clone())
        }
    }

    fn fast_dev_run(&self) -> Result<Option<i64>, String> {
        let state = self.0.lock().unwrap();
        if state.fail == Some(TrainingTrainerField::FastDevRun) {
            Err("development access".into())
        } else {
            Ok(state.fast_dev)
        }
    }
}

struct PolicyData {
    payload: Arc<Value>,
    loaded: Option<Arc<Value>>,
    events: Vec<Value>,
    fail: Option<&'static str>,
}

struct Policy(Arc<Mutex<PolicyData>>);

impl TrainingPolicyState<Arc<Value>> for Policy {
    fn state_dict(&mut self) -> Result<Arc<Value>, String> {
        let mut state = self.0.lock().unwrap();
        state.events.push(json!("save"));
        if state.fail == Some("save") {
            Err("save failed".into())
        } else {
            Ok(Arc::clone(&state.payload))
        }
    }

    fn load_state_dict(&mut self, payload: &Arc<Value>) -> Result<(), String> {
        let mut state = self.0.lock().unwrap();
        state.events.push(json!(["load", payload.as_ref()]));
        if state.fail == Some("load") {
            Err("load failed".into())
        } else {
            state.loaded = Some(Arc::clone(payload));
            Ok(())
        }
    }
}

fn state_adapter() -> (TrainingVesselState<Arc<Value>>, Arc<Mutex<PolicyData>>) {
    let state = Arc::new(Mutex::new(PolicyData {
        payload: Arc::new(json!({"weights": [1, 2], "optimizer": null})),
        loaded: None,
        events: vec![],
        fail: None,
    }));
    (
        TrainingVesselState::new(Box::new(Policy(Arc::clone(&state)))),
        state,
    )
}

#[test]
fn attachment_reads_live_values_and_never_owns_the_trainer() {
    let mut binding = TrainingVesselBinding::default();
    assert!(matches!(
        binding.trainer(),
        Err(TrainingVesselBindingError::Unassigned)
    ));
    assert_eq!(
        binding.current_iteration(),
        Err(TrainingVesselBindingError::Unassigned)
    );
    assert_eq!(
        binding.fast_dev_run(),
        Err(TrainingVesselBindingError::Unassigned)
    );
    let concrete = Arc::new(Trainer::new());
    let view: Arc<dyn TrainingTrainerView> = concrete.clone();
    binding.assign_trainer(&view);
    assert_eq!(Arc::strong_count(&view), 2);
    assert_eq!(binding.current_iteration().unwrap(), BigInt::from(-1));
    assert_eq!(binding.fast_dev_run().unwrap(), None);
    {
        let mut state = concrete.0.lock().unwrap();
        state.iteration = BigInt::from(10).pow(30);
        state.fast_dev = Some(-2);
    }
    assert_eq!(
        binding.current_iteration().unwrap().to_string(),
        "1000000000000000000000000000000"
    );
    assert_eq!(binding.fast_dev_run().unwrap(), Some(-2));
    let pinned = binding.trainer().unwrap();
    drop(concrete);
    drop(view);
    assert_eq!(binding.fast_dev_run().unwrap(), Some(-2));
    drop(pinned);
    assert_eq!(
        binding.current_iteration(),
        Err(TrainingVesselBindingError::Expired)
    );
    assert_eq!(
        binding.fast_dev_run(),
        Err(TrainingVesselBindingError::Expired)
    );
    let replacement: Arc<dyn TrainingTrainerView> = Arc::new(Trainer::new());
    binding.assign_trainer(&replacement);
    assert_eq!(binding.current_iteration().unwrap(), BigInt::from(-1));
    let second: Arc<dyn TrainingTrainerView> = Arc::new(Trainer::new());
    binding.assign_trainer(&second);
    assert!(Arc::ptr_eq(&binding.trainer().unwrap(), &second));
    assert_eq!(Arc::weak_count(&replacement), 0);
}

#[test]
fn trainer_access_failures_are_typed_without_expiring_the_binding() {
    let trainer = Arc::new(Trainer::new());
    let view: Arc<dyn TrainingTrainerView> = trainer.clone();
    let mut binding = TrainingVesselBinding::default();
    binding.assign_trainer(&view);
    for (field, message, name) in [
        (
            TrainingTrainerField::CurrentIteration,
            "iteration access",
            "current_iteration",
        ),
        (
            TrainingTrainerField::FastDevRun,
            "development access",
            "fast_dev_run",
        ),
    ] {
        trainer.0.lock().unwrap().fail = Some(field);
        let expected = TrainingVesselBindingError::Access {
            field,
            message: message.into(),
        };
        if field == TrainingTrainerField::CurrentIteration {
            assert_eq!(binding.current_iteration(), Err(expected.clone()));
        } else {
            assert_eq!(binding.fast_dev_run(), Err(expected.clone()));
        }
        assert_eq!(field.to_string(), name);
        assert!(expected.to_string().contains(message));
        assert!(binding.trainer().is_ok());
    }
    trainer.0.lock().unwrap().fail = None;
    assert_eq!(binding.fast_dev_run().unwrap(), None);
    assert!(
        TrainingVesselBindingError::Expired
            .to_string()
            .contains("dropped")
    );
    assert!(
        TrainingVesselBindingError::Unassigned
            .to_string()
            .contains("no assigned")
    );
}

#[test]
fn checkpoints_preserve_payload_identity_order_and_policy_failure_state() {
    let (mut adapter, state) = state_adapter();
    let checkpoint = adapter.state_dict().unwrap();
    assert!(Arc::ptr_eq(
        &checkpoint.policy,
        &state.lock().unwrap().payload
    ));
    adapter.load_state_dict(&checkpoint).unwrap();
    assert!(Arc::ptr_eq(
        &checkpoint.policy,
        state.lock().unwrap().loaded.as_ref().unwrap()
    ));
    state.lock().unwrap().fail = Some("save");
    assert_eq!(
        adapter.state_dict(),
        Err(TrainingVesselStateError::Save("save failed".into()))
    );
    state.lock().unwrap().fail = Some("load");
    assert_eq!(
        adapter.load_state_dict(&checkpoint),
        Err(TrainingVesselStateError::Load("load failed".into()))
    );
    assert_eq!(state.lock().unwrap().events.len(), 4);
    assert!(Arc::ptr_eq(
        &checkpoint.policy,
        state.lock().unwrap().loaded.as_ref().unwrap()
    ));
    assert!(
        TrainingVesselStateError::Save("save failed".into())
            .to_string()
            .contains("save failed")
    );
    assert!(
        TrainingVesselStateError::Load("load failed".into())
            .to_string()
            .contains("load failed")
    );
    state.lock().unwrap().fail = None;
    adapter.load_state_dict(&checkpoint).unwrap();
    assert_eq!(state.lock().unwrap().events.len(), 5);
}

#[test]
fn checkpoint_envelope_requires_policy_even_for_optional_payloads() {
    for payload in [
        Value::Null,
        json!(4),
        json!("state"),
        json!([1, 2]),
        json!({"layer": [1,2]}),
    ] {
        let checkpoint = TrainingVesselCheckpoint {
            policy: payload.clone(),
        };
        assert_eq!(
            serde_json::to_value(&checkpoint).unwrap(),
            json!({"policy": payload})
        );
        let decoded: TrainingVesselCheckpoint<Value> =
            serde_json::from_value(json!({"policy": payload, "extra": 3})).unwrap();
        assert_eq!(checkpoint, decoded);
        let bytes = bincode::serialize(&TrainingVesselCheckpoint {
            policy: vec![1_i64, 2],
        })
        .unwrap();
        let binary: TrainingVesselCheckpoint<Vec<i64>> = bincode::deserialize(&bytes).unwrap();
        assert_eq!(binary.policy, vec![1, 2]);
    }
    assert!(serde_json::from_value::<TrainingVesselCheckpoint<Value>>(json!({})).is_err());
    assert!(serde_json::from_value::<TrainingVesselCheckpoint<Option<i64>>>(json!({})).is_err());
    let explicit_null: TrainingVesselCheckpoint<Option<i64>> =
        serde_json::from_value(json!({"policy": null})).unwrap();
    assert_eq!(explicit_null.policy, None);
}

#[test]
fn state_delegation_and_weak_attachment_match_live_python_source() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .args([
            root.join("tests/fixtures/training_vessel_state_contract.py"),
            root.join("../../../qlib/qlib/rl/trainer/vessel.py"),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let python: Value = serde_json::from_slice(&output.stdout).unwrap();
    let (mut adapter, state) = state_adapter();
    let checkpoint = adapter.state_dict().unwrap();
    adapter.load_state_dict(&checkpoint).unwrap();
    state.lock().unwrap().fail = Some("save");
    assert!(adapter.state_dict().is_err());
    state.lock().unwrap().fail = Some("load");
    assert!(adapter.load_state_dict(&checkpoint).is_err());
    assert_eq!(json!(state.lock().unwrap().events), python["events"]);
    assert_eq!(
        python["checkpoint"],
        json!({"keys":["policy"],"same_payload":true,"load_returns_none":true,"same_loaded_payload":true})
    );
    assert_eq!(
        python["errors"],
        json!([
            ["missing", "KeyError", 0],
            ["save", "RuntimeError", 1],
            ["load", "RuntimeError", 1]
        ])
    );
    let mut binding = TrainingVesselBinding::default();
    assert_eq!(python["binding"]["unassigned"], binding.trainer().is_err());
    let trainer = Arc::new(Trainer::new());
    let view: Arc<dyn TrainingTrainerView> = trainer.clone();
    binding.assign_trainer(&view);
    assert_eq!(
        python["binding"]["initial_iter"]
            .as_i64()
            .unwrap()
            .to_string(),
        binding.current_iteration().unwrap().to_string()
    );
    {
        let mut state = trainer.0.lock().unwrap();
        state.iteration = BigInt::from(10).pow(30);
        state.fast_dev = Some(-2);
    }
    assert_eq!(
        python["binding"]["updated"],
        json!([
            binding.current_iteration().unwrap().to_string(),
            binding.fast_dev_run().unwrap()
        ])
    );
    let weak = Arc::downgrade(&view);
    drop(trainer);
    drop(view);
    assert_eq!(python["binding"]["not_owned"], weak.upgrade().is_none());
    assert_eq!(
        python["binding"]["expired"],
        matches!(binding.trainer(), Err(TrainingVesselBindingError::Expired))
    );
    let replacement: Arc<dyn TrainingTrainerView> = Arc::new(Trainer::new());
    binding.assign_trainer(&replacement);
    assert_eq!(
        python["binding"]["replacement_iter"]
            .as_i64()
            .unwrap()
            .to_string(),
        binding.current_iteration().unwrap().to_string()
    );
}
