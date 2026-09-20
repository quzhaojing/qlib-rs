use std::{
    path::PathBuf,
    process::Command,
    sync::{Arc, Mutex},
};

use domain_core::{
    BoxedFiniteEnvironment, EnvironmentPluginError, FiniteBackendStep, FiniteDummyBackend,
    FiniteDummyBuildError, FiniteDummyError, FiniteEnvironment, FiniteObservationPredicate,
    FiniteVectorBackend, FiniteVectorEnv,
};
use serde_json::{Value, json};

type Events = Arc<Mutex<Vec<Value>>>;
type WorkerBox = BoxedFiniteEnvironment<i64, i32, f64, Value>;

struct Predicate;

impl FiniteObservationPredicate<i64> for Predicate {
    fn is_invalid(&mut self, observation: &i64) -> Result<bool, EnvironmentPluginError> {
        Ok(*observation < 0)
    }
}

fn plugin(message: &str) -> EnvironmentPluginError {
    EnvironmentPluginError::new(message)
}

struct Worker {
    id: usize,
    resets: usize,
    steps: usize,
    fail_reset: bool,
    fail_step: bool,
    events: Events,
}

impl FiniteEnvironment<i64, i32, f64, Value> for Worker {
    fn reset(&mut self) -> Result<i64, EnvironmentPluginError> {
        self.resets += 1;
        self.events
            .lock()
            .unwrap()
            .push(json!(["reset", self.id, self.resets]));
        if self.fail_reset {
            Err(plugin("reset"))
        } else {
            Ok(i64::try_from(self.id * 100 + self.resets).unwrap())
        }
    }

    fn step(
        &mut self,
        action: &i32,
    ) -> Result<FiniteBackendStep<i64, f64, Value>, EnvironmentPluginError> {
        self.steps += 1;
        self.events
            .lock()
            .unwrap()
            .push(json!(["step", self.id, self.steps, action]));
        if self.fail_step {
            Err(plugin("step"))
        } else {
            Ok(FiniteBackendStep {
                observation: Some(i64::try_from(self.id * 100 + self.steps).unwrap()),
                reward: Some(f64::from(*action)),
                done: self.steps == 2,
                info: Some(json!({"action": action, "worker": self.id})),
            })
        }
    }
}

fn worker(id: usize, events: &Events) -> WorkerBox {
    Box::new(Worker {
        id,
        resets: 0,
        steps: 0,
        fail_reset: false,
        fail_step: false,
        events: Arc::clone(events),
    })
}

fn failing_worker(id: usize, fail_reset: bool, fail_step: bool, events: &Events) -> WorkerBox {
    Box::new(Worker {
        id,
        resets: 0,
        steps: 0,
        fail_reset,
        fail_step,
        events: Arc::clone(events),
    })
}

fn python_contract() -> Value {
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/finite_dummy_contract.py");
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/rl/utils/finite_env.py");
    let output = Command::new("python")
        .arg(fixture)
        .arg(source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn repeated_factory_and_dummy_identity_match_live_qlib_source() {
    let python = python_contract();
    assert_eq!(
        python["dummy_bases"],
        json!(["FiniteVectorEnv", "DummyVectorEnv"])
    );
    assert_eq!(python["dummy_body"], json!(["Pass"]));
    assert_eq!(python["selected"], json!(["dummy", "subproc", "shmem"]));
    assert_eq!(python["factory_calls"], json!([3, 3, 3]));
    assert_eq!(python["same_factory_reference"], true);
    assert_eq!(python["same_logger_reference"], true);
    assert_eq!(python["invalid_key"], "invalid");

    let events = Arc::new(Mutex::new(Vec::new()));
    let calls = Arc::new(Mutex::new(0_usize));
    let factory_calls = Arc::clone(&calls);
    let factory_events = Arc::clone(&events);
    let mut factory = move || {
        let mut call_count = factory_calls.lock().unwrap();
        let id = *call_count;
        *call_count += 1;
        Ok(worker(id, &factory_events))
    };
    let backend = FiniteDummyBackend::from_factory(3, &mut factory).unwrap();
    assert_eq!(backend.environment_count(), 3);
    assert_eq!(*calls.lock().unwrap(), 3);
}

#[test]
fn synchronous_send_then_receive_preserves_duplicates_and_order() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut backend =
        FiniteDummyBackend::new(vec![worker(0, &events), worker(1, &events)]).unwrap();

    assert_eq!(
        backend.reset_environments(&[1, 0, 1]).unwrap(),
        vec![Some(102), Some(1), Some(102)]
    );
    assert_eq!(
        backend
            .step_environments(&[10, 20, 30], &[1, 0, 1])
            .unwrap(),
        vec![
            FiniteBackendStep {
                observation: Some(102),
                reward: Some(30.0),
                done: true,
                info: Some(json!({"action": 30, "worker": 1})),
            },
            FiniteBackendStep {
                observation: Some(1),
                reward: Some(20.0),
                done: false,
                info: Some(json!({"action": 20, "worker": 0})),
            },
            FiniteBackendStep {
                observation: Some(102),
                reward: Some(30.0),
                done: true,
                info: Some(json!({"action": 30, "worker": 1})),
            },
        ]
    );
    assert_eq!(
        *events.lock().unwrap(),
        [
            json!(["reset", 1, 1]),
            json!(["reset", 0, 1]),
            json!(["reset", 1, 2]),
            json!(["step", 1, 1, 10]),
            json!(["step", 0, 1, 20]),
            json!(["step", 1, 2, 30]),
        ]
    );
    assert!(backend.reset_environments(&[]).unwrap().is_empty());
    assert!(backend.step_environments(&[], &[]).unwrap().is_empty());
}

#[test]
fn construction_selection_and_worker_failures_retain_exact_boundaries() {
    assert!(matches!(
        FiniteDummyBackend::<i64, i32, f64, Value>::new(Vec::new()),
        Err(FiniteDummyBuildError::Empty)
    ));
    let mut unused_factory = || -> Result<WorkerBox, EnvironmentPluginError> {
        panic!("zero concurrency must not call the factory")
    };
    assert!(matches!(
        FiniteDummyBackend::from_factory(0, &mut unused_factory),
        Err(FiniteDummyBuildError::Empty)
    ));

    let calls = Arc::new(Mutex::new(0_usize));
    let factory_calls = Arc::clone(&calls);
    let mut failing_factory = move || {
        let mut call_count = factory_calls.lock().unwrap();
        let id = *call_count;
        *call_count += 1;
        if id == 1 {
            Err(plugin("factory"))
        } else {
            Ok(Box::new(Worker {
                id,
                resets: 0,
                steps: 0,
                fail_reset: false,
                fail_step: false,
                events: Arc::new(Mutex::new(Vec::new())),
            }) as WorkerBox)
        }
    };
    assert!(matches!(
        FiniteDummyBackend::from_factory(3, &mut failing_factory),
        Err(FiniteDummyBuildError::Environment {
            environment_id: 1,
            ..
        })
    ));
    assert_eq!(*calls.lock().unwrap(), 2);

    let events = Arc::new(Mutex::new(Vec::new()));
    let mut invalid = FiniteDummyBackend::new(vec![worker(0, &events)]).unwrap();
    assert_eq!(
        invalid.reset_environments(&[0, 1]),
        Err(FiniteDummyError::InvalidEnvironmentId {
            id: 1,
            environment_count: 1,
        })
    );
    assert_eq!(*events.lock().unwrap(), [json!(["reset", 0, 1])]);
    events.lock().unwrap().clear();
    assert_eq!(
        invalid.step_environments(&[1], &[0, 0]),
        Err(FiniteDummyError::ActionCount {
            required: 2,
            actual: 1,
        })
    );
    assert!(events.lock().unwrap().is_empty());
    assert_eq!(
        invalid.step_environments(&[1, 2], &[0, 1]),
        Err(FiniteDummyError::InvalidEnvironmentId {
            id: 1,
            environment_count: 1,
        })
    );
    assert_eq!(*events.lock().unwrap(), [json!(["step", 0, 1, 1])]);

    let reset_events = Arc::new(Mutex::new(Vec::new()));
    let mut reset_failure = FiniteDummyBackend::new(vec![
        worker(0, &reset_events),
        failing_worker(1, true, false, &reset_events),
    ])
    .unwrap();
    assert!(matches!(
        reset_failure.reset_environments(&[0, 1]),
        Err(FiniteDummyError::Reset {
            environment_id: 1,
            ..
        })
    ));
    assert_eq!(reset_events.lock().unwrap().len(), 2);

    let step_events = Arc::new(Mutex::new(Vec::new()));
    let mut step_failure = FiniteDummyBackend::new(vec![
        worker(0, &step_events),
        failing_worker(1, false, true, &step_events),
    ])
    .unwrap();
    assert!(matches!(
        step_failure.step_environments(&[4, 5], &[0, 1]),
        Err(FiniteDummyError::Step {
            environment_id: 1,
            ..
        })
    ));
    assert_eq!(step_events.lock().unwrap().len(), 2);
}

#[test]
fn backend_trait_composes_with_finite_vector_coordinator() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let backend = FiniteDummyBackend::new(vec![worker(0, &events), worker(1, &events)]).unwrap();
    let mut vector = FiniteVectorEnv::new(Box::new(backend), Box::new(Predicate), Vec::new());
    assert_eq!(
        vector.reset(None).unwrap().observations,
        vec![Some(1), Some(101)]
    );
    let stepped = vector.step(&[7, 8], None).unwrap();
    assert_eq!(stepped.transitions.len(), 2);
    assert_eq!(stepped.transitions[0].reward, Some(7.0));
    assert_eq!(stepped.transitions[1].reward, Some(8.0));
    assert_eq!(
        vector
            .alive_environment_ids()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![0, 1]
    );

    let mut erased: Box<dyn FiniteVectorBackend<i64, i32, f64, Value>> = Box::new(
        FiniteDummyBackend::new(vec![worker(0, &Arc::new(Mutex::new(Vec::new())))]).unwrap(),
    );
    assert_eq!(erased.environment_count(), 1);
    assert!(erased.reset(&[1]).unwrap_err().message.contains("outside"));
    assert!(
        erased
            .step(&[], &[0])
            .unwrap_err()
            .message
            .contains("exactly")
    );
}
