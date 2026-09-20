#![allow(clippy::float_cmp)]

use std::{
    collections::VecDeque,
    path::PathBuf,
    process::Command,
    sync::{Arc, Mutex},
};

use domain_core::{
    EnvironmentPluginError, FiniteBackendStep, FiniteObservation, FiniteObservationPredicate,
    FiniteVectorBackend, FiniteVectorEnv, FiniteVectorError, FiniteVectorLogger, FiniteVectorStage,
    RecursiveFiniteObservationPredicate,
};
use ndarray::array;
use serde_json::{Value, json};

type BackendResult<T> = Result<T, EnvironmentPluginError>;
type ResetPlan = BackendResult<Vec<Option<i64>>>;
type StepPlan = BackendResult<Vec<FiniteBackendStep<i64, f64, Value>>>;
type Environment = FiniteVectorEnv<i64, i64, f64, Value>;
type Events = Arc<Mutex<Vec<Value>>>;

fn plugin(message: &str) -> EnvironmentPluginError {
    EnvironmentPluginError::new(message)
}

struct Backend {
    count: usize,
    resets: VecDeque<ResetPlan>,
    steps: VecDeque<StepPlan>,
    events: Events,
}

impl FiniteVectorBackend<i64, i64, f64, Value> for Backend {
    fn environment_count(&self) -> usize {
        self.count
    }

    fn reset(&mut self, environment_ids: &[usize]) -> ResetPlan {
        self.events
            .lock()
            .unwrap()
            .push(json!(["reset", environment_ids]));
        self.resets.pop_front().unwrap_or_else(|| Ok(Vec::new()))
    }

    fn step(&mut self, actions: &[i64], environment_ids: &[usize]) -> StepPlan {
        self.events
            .lock()
            .unwrap()
            .push(json!(["step", environment_ids, actions]));
        self.steps.pop_front().unwrap_or_else(|| Ok(Vec::new()))
    }
}

struct Predicate;

impl FiniteObservationPredicate<i64> for Predicate {
    fn is_invalid(&mut self, observation: &i64) -> Result<bool, EnvironmentPluginError> {
        if *observation == -888 {
            Err(plugin("predicate"))
        } else {
            Ok(*observation == -999)
        }
    }
}

struct Logger {
    events: Events,
    fail: Option<&'static str>,
}

impl Logger {
    fn record(&self, name: &'static str, values: Value) -> Result<(), EnvironmentPluginError> {
        self.events
            .lock()
            .unwrap()
            .push(Value::Array(vec![Value::String(name.to_owned()), values]));
        if self.fail == Some(name) {
            Err(plugin(name))
        } else {
            Ok(())
        }
    }
}

impl FiniteVectorLogger<i64, f64, Value> for Logger {
    fn on_all_ready(&mut self) -> Result<(), EnvironmentPluginError> {
        self.record("ready", Value::Null)
    }

    fn on_all_done(&mut self) -> Result<(), EnvironmentPluginError> {
        self.record("done", Value::Null)
    }

    fn on_reset(
        &mut self,
        environment_id: usize,
        all_observations: &[Option<i64>],
    ) -> Result<(), EnvironmentPluginError> {
        self.record("reset", json!([environment_id, all_observations]))
    }

    fn on_step(
        &mut self,
        environment_id: usize,
        step: &FiniteBackendStep<i64, f64, Value>,
    ) -> Result<(), EnvironmentPluginError> {
        self.record(
            "step",
            json!([
                environment_id,
                step.observation,
                step.reward,
                step.done,
                step.info
            ]),
        )
    }
}

struct DefaultLogger;

impl FiniteVectorLogger<i64, f64, Value> for DefaultLogger {}

fn environment(
    count: usize,
    resets: Vec<ResetPlan>,
    steps: Vec<StepPlan>,
    backend_events: Events,
    loggers: Vec<Box<dyn FiniteVectorLogger<i64, f64, Value>>>,
) -> Environment {
    FiniteVectorEnv::new(
        Box::new(Backend {
            count,
            resets: resets.into(),
            steps: steps.into(),
            events: backend_events,
        }),
        Box::new(Predicate),
        loggers,
    )
}

fn guarded_operation(environment: &mut Environment) -> Result<usize, FiniteVectorError> {
    match environment.environment_count() {
        0 => Err(FiniteVectorError::Exhausted),
        2 => Err(FiniteVectorError::EmptySelection),
        _ => environment
            .reset(None)
            .map(|reset| reset.observations.len()),
    }
}

fn transition(
    observation: Option<i64>,
    reward: Option<f64>,
    done: bool,
    info: Option<Value>,
) -> FiniteBackendStep<i64, f64, Value> {
    FiniteBackendStep {
        observation,
        reward,
        done,
        info,
    }
}

fn python_contract() -> Value {
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/finite_vector_contract.py");
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
fn reset_step_retirement_defaults_and_zombie_match_live_python() {
    let python = python_contract();
    let backend_events = Arc::new(Mutex::new(Vec::new()));
    let logger_events = Arc::new(Mutex::new(Vec::new()));
    let mut environment = environment(
        2,
        vec![Ok(vec![Some(10), Some(-999)]), Ok(vec![Some(-999)])],
        vec![Ok(vec![transition(
            Some(20),
            Some(1.0),
            true,
            Some(json!({"source": 0})),
        )])],
        Arc::clone(&backend_events),
        vec![Box::new(Logger {
            events: Arc::clone(&logger_events),
            fail: None,
        })],
    );
    assert_eq!(environment.environment_count(), 2);
    assert!(!environment.is_zombie());
    assert!(!environment.is_collector_guarded());
    assert_eq!(
        environment
            .alive_environment_ids()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [0, 1]
    );

    let reset = environment.reset(None).unwrap();
    assert_eq!(reset.observations, [Some(10), Some(10)]);
    assert_eq!(json!(reset.observations), python["first_reset"]);
    assert_eq!(environment.unguarded_reset_warnings(), 1);
    assert_eq!(python["warning_count"], 1);
    assert_eq!(
        environment
            .alive_environment_ids()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [0]
    );

    let step = environment.step(&[100, 200], Some(&[1, 0])).unwrap();
    assert_eq!(step.transitions[0].observation, Some(10));
    assert_eq!(step.transitions[1].observation, Some(20));
    assert_eq!(step.transitions[0].reward, Some(1.0));
    assert_eq!(step.transitions[1].reward, Some(1.0));
    assert!(!step.transitions[0].done);
    assert!(step.transitions[1].done);
    assert_eq!(step.transitions[0].info, Some(json!({"source": 0})));
    assert_eq!(json!([10, 20]), python["step"][0]);
    assert_eq!(json!([1.0, 1.0]), python["step"][1]);
    assert_eq!(json!([false, true]), python["step"][2]);
    assert_eq!(
        environment.reset(Some(&[1, 0])),
        Err(FiniteVectorError::Exhausted)
    );
    assert_eq!(
        json!(&*backend_events.lock().unwrap()),
        python["backend_events"]
    );
    assert!(environment.is_zombie());
    assert_eq!(python["exhausted"], true);
    assert_eq!(python["zombie"], true);
    assert_eq!(environment.reset(None), Err(FiniteVectorError::Zombie));
    assert_eq!(environment.step(&[], None), Err(FiniteVectorError::Zombie));
    assert_eq!(
        *logger_events.lock().unwrap(),
        [
            json!(["reset", [0, [10, null]]]),
            json!(["step", [0, 20, 1.0, true, {"source": 0}]])
        ]
    );
}

#[test]
fn duplicates_zip_truncation_and_first_defaults_preserve_python_order() {
    let backend_events = Arc::new(Mutex::new(Vec::new()));
    let logger_events = Arc::new(Mutex::new(Vec::new()));
    let mut environment = environment(
        2,
        vec![Ok(vec![Some(5)])],
        vec![
            Ok(vec![transition(Some(7), Some(2.0), true, Some(json!(2)))]),
            Ok(vec![transition(Some(8), Some(9.0), false, Some(json!(9)))]),
        ],
        Arc::clone(&backend_events),
        vec![Box::new(Logger {
            events: Arc::clone(&logger_events),
            fail: None,
        })],
    );
    let reset = environment.reset(Some(&[0, 0])).unwrap();
    assert_eq!(reset.observations, [Some(5), Some(5)]);
    assert_eq!(
        environment
            .alive_environment_ids()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [1]
    );
    assert!(logger_events.lock().unwrap().is_empty());

    let first = environment.step(&[10, 20, 30], Some(&[0, 1, 1])).unwrap();
    assert_eq!(
        first.transitions,
        [
            transition(Some(5), Some(2.0), false, Some(json!(2))),
            transition(Some(5), Some(2.0), false, Some(json!(2))),
            transition(Some(7), Some(2.0), true, Some(json!(2))),
        ]
    );
    let second = environment.step(&[40, 50], Some(&[0, 1])).unwrap();
    assert_eq!(second.transitions[0].reward, Some(2.0));
    assert_eq!(second.transitions[0].info, Some(json!(2)));
    assert_eq!(second.transitions[1].reward, Some(9.0));
    assert_eq!(
        *backend_events.lock().unwrap(),
        [
            json!(["reset", [0, 0]]),
            json!(["step", [1, 1], [30, 30]]),
            json!(["step", [1], [50]])
        ]
    );
    assert_eq!(logger_events.lock().unwrap().len(), 3);
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one matrix verifies every coordinator plugin failure stage"
)]
fn selection_backend_predicate_and_logger_failures_are_typed() {
    let make = |resets: Vec<ResetPlan>, steps: Vec<StepPlan>, fail| {
        environment(
            2,
            resets,
            steps,
            Arc::new(Mutex::new(Vec::new())),
            vec![Box::new(Logger {
                events: Arc::new(Mutex::new(Vec::new())),
                fail,
            })],
        )
    };

    let mut invalid = make(Vec::new(), Vec::new(), None);
    assert_eq!(
        invalid.reset(Some(&[2])),
        Err(FiniteVectorError::InvalidEnvironmentId {
            id: 2,
            environment_count: 2
        })
    );
    assert_eq!(
        invalid.step(&[1], Some(&[3])),
        Err(FiniteVectorError::InvalidEnvironmentId {
            id: 3,
            environment_count: 2
        })
    );
    assert_eq!(
        invalid.step(&[1], None),
        Err(FiniteVectorError::ActionBatchTooShort {
            required: 2,
            actual: 1
        })
    );
    assert_eq!(
        invalid.reset(Some(&[])),
        Err(FiniteVectorError::EmptySelection)
    );
    assert_eq!(
        invalid.step(&[], Some(&[])),
        Err(FiniteVectorError::EmptySelection)
    );

    let mut backend_reset = make(vec![Err(plugin("reset"))], Vec::new(), None);
    assert!(matches!(
        backend_reset.reset(None),
        Err(FiniteVectorError::Component {
            stage: FiniteVectorStage::BackendReset,
            ..
        })
    ));
    let mut backend_step = make(
        vec![Ok(vec![Some(1), Some(2)])],
        vec![Err(plugin("step"))],
        None,
    );
    backend_step.reset(None).unwrap();
    assert!(matches!(
        backend_step.step(&[1, 2], None),
        Err(FiniteVectorError::Component {
            stage: FiniteVectorStage::BackendStep,
            ..
        })
    ));

    let mut reset_predicate = make(vec![Ok(vec![Some(-888), Some(2)])], Vec::new(), None);
    assert!(matches!(
        reset_predicate.reset(None),
        Err(FiniteVectorError::Component {
            stage: FiniteVectorStage::ResetObservation,
            environment_id: Some(0),
            ..
        })
    ));
    let mut step_predicate = make(
        vec![Ok(vec![Some(1), Some(2)])],
        vec![Ok(vec![transition(Some(-888), None, false, None)])],
        None,
    );
    step_predicate.reset(None).unwrap();
    assert!(matches!(
        step_predicate.step(&[1], Some(&[0])),
        Err(FiniteVectorError::Component {
            stage: FiniteVectorStage::StepObservation,
            environment_id: Some(0),
            ..
        })
    ));

    let mut reset_logger = make(vec![Ok(vec![Some(1), Some(2)])], Vec::new(), Some("reset"));
    assert!(matches!(
        reset_logger.reset(None),
        Err(FiniteVectorError::Component {
            stage: FiniteVectorStage::LoggerReset,
            environment_id: Some(0),
            ..
        })
    ));
    let mut step_logger = make(
        vec![Ok(vec![Some(1), Some(2)])],
        vec![Ok(vec![transition(
            Some(3),
            Some(1.0),
            false,
            Some(json!(1)),
        )])],
        Some("step"),
    );
    step_logger.reset(None).unwrap();
    assert!(matches!(
        step_logger.step(&[1], Some(&[0])),
        Err(FiniteVectorError::Component {
            stage: FiniteVectorStage::LoggerStep,
            environment_id: Some(0),
            ..
        })
    ));
}

#[test]
fn collector_guard_matches_exhaustion_error_and_logger_boundaries() {
    let python = python_contract();
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut success = environment(
        1,
        vec![Ok(vec![Some(7)])],
        Vec::new(),
        Arc::new(Mutex::new(Vec::new())),
        vec![Box::new(Logger {
            events: Arc::clone(&events),
            fail: None,
        })],
    );
    assert_eq!(success.collect_guarded(guarded_operation), Ok(Some(1)));
    assert!(!success.is_collector_guarded());
    assert_eq!(
        *events.lock().unwrap(),
        [
            json!(["ready", null]),
            json!(["reset", [0, [7]]]),
            json!(["done", null])
        ]
    );

    events.lock().unwrap().clear();
    let mut exhausted = environment(
        0,
        Vec::new(),
        Vec::new(),
        Arc::new(Mutex::new(Vec::new())),
        vec![Box::new(Logger {
            events: Arc::clone(&events),
            fail: None,
        })],
    );
    assert_eq!(
        exhausted.collect_guarded(guarded_operation),
        Ok(None::<usize>)
    );
    assert_eq!(
        *events.lock().unwrap(),
        [json!(["ready", null]), json!(["done", null])]
    );
    assert_eq!(python["guard_events"], json!([["ready"], ["done"]]));

    events.lock().unwrap().clear();
    let mut ordinary_error = environment(
        2,
        Vec::new(),
        Vec::new(),
        Arc::new(Mutex::new(Vec::new())),
        vec![Box::new(Logger {
            events: Arc::clone(&events),
            fail: None,
        })],
    );
    assert_eq!(
        ordinary_error.collect_guarded(guarded_operation),
        Err(FiniteVectorError::EmptySelection)
    );
    assert_eq!(*events.lock().unwrap(), [json!(["ready", null])]);
    assert!(!ordinary_error.is_collector_guarded());
    assert_eq!(python["error_events"], json!([["ready"]]));
    assert_eq!(python["error_guarded_flag"], false);

    let mut ready_failure = environment(
        1,
        Vec::new(),
        Vec::new(),
        Arc::new(Mutex::new(Vec::new())),
        vec![Box::new(Logger {
            events: Arc::new(Mutex::new(Vec::new())),
            fail: Some("ready"),
        })],
    );
    assert!(matches!(
        ready_failure.collect_guarded(guarded_operation),
        Err(FiniteVectorError::Component {
            stage: FiniteVectorStage::LoggerAllReady,
            ..
        })
    ));
    assert!(ready_failure.is_collector_guarded());

    let mut done_failure = environment(
        1,
        vec![Ok(vec![Some(7)])],
        Vec::new(),
        Arc::new(Mutex::new(Vec::new())),
        vec![Box::new(Logger {
            events: Arc::new(Mutex::new(Vec::new())),
            fail: Some("done"),
        })],
    );
    assert!(matches!(
        done_failure.collect_guarded(guarded_operation),
        Err(FiniteVectorError::Component {
            stage: FiniteVectorStage::LoggerAllDone,
            ..
        })
    ));
    assert!(!done_failure.is_collector_guarded());
}

#[test]
fn default_hooks_none_observations_zero_workers_and_recursive_predicate_are_explicit() {
    let mut default_logger = DefaultLogger;
    assert_eq!(default_logger.on_all_ready(), Ok(()));
    assert_eq!(default_logger.on_all_done(), Ok(()));

    let mut healthy_defaults = environment(
        1,
        vec![Ok(vec![Some(7)])],
        vec![Ok(vec![transition(
            Some(8),
            Some(1.0),
            false,
            Some(json!({"source": "default-logger"})),
        )])],
        Arc::new(Mutex::new(Vec::new())),
        vec![Box::new(DefaultLogger)],
    );
    assert_eq!(
        healthy_defaults.reset(None).unwrap().observations,
        vec![Some(7)]
    );
    assert_eq!(
        healthy_defaults.step(&[3], None).unwrap().transitions,
        vec![transition(
            Some(8),
            Some(1.0),
            false,
            Some(json!({"source": "default-logger"})),
        )]
    );

    let mut defaults = environment(
        1,
        vec![Ok(vec![None])],
        Vec::new(),
        Arc::new(Mutex::new(Vec::new())),
        Vec::new(),
    );
    assert_eq!(defaults.reset(None), Err(FiniteVectorError::Exhausted));
    assert!(defaults.is_zombie());

    let mut zero = environment(
        0,
        Vec::new(),
        Vec::new(),
        Arc::new(Mutex::new(Vec::new())),
        Vec::new(),
    );
    assert_eq!(zero.reset(None), Err(FiniteVectorError::Exhausted));

    let mut predicate = RecursiveFiniteObservationPredicate;
    assert!(
        !predicate
            .is_invalid(&FiniteObservation::Float64(array![1.0].into_dyn()))
            .unwrap()
    );
    assert!(
        predicate
            .is_invalid(&FiniteObservation::Float64(array![f64::NAN].into_dyn()))
            .unwrap()
    );
    assert!(
        predicate
            .is_invalid(&FiniteObservation::Bool(array![true].into_dyn()))
            .is_err()
    );
}
