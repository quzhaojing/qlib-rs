use std::{
    path::PathBuf,
    process::Command,
    sync::{Arc, Mutex},
};

use serde_json::json;

use super::*;
use crate::{
    ArrowTrainingMetricReducer, EnvironmentPluginError, FiniteBackendStep, FiniteDummyBackend,
    FiniteEnvironment, FiniteObservationPredicate, FiniteVectorBackend, FiniteVectorLogger,
    TrainingMetricScalar, TrainingMetricSink, TrainingTrainerView,
};

type Environment = FiniteVectorEnv<i64, i64, f64, Value>;
type Runner = TrainingVesselRunner<i64, i64, f64, Value, Vec<i64>>;

#[derive(Clone)]
struct Trace {
    events: Arc<Mutex<Vec<Value>>>,
    buffer_address: Arc<Mutex<Option<usize>>>,
    input: Value,
}
impl Trace {
    fn new(input: Value) -> Self {
        Self {
            events: Arc::default(),
            buffer_address: Arc::default(),
            input,
        }
    }
    fn record(&self, event: Value) -> Result<(), TrainingVesselRunError> {
        let stage = event[0].as_str().unwrap().to_owned();
        self.events.lock().unwrap().push(event);
        if self.input["fail"] == stage {
            Err(TrainingVesselRunError::Plugin {
                stage,
                message: "injected".into(),
            })
        } else if self.input["fail"] == format!("stop:{stage}") {
            Err(FiniteVectorError::Exhausted.into())
        } else {
            Ok(())
        }
    }
    fn snapshot(&self) -> Vec<Value> {
        self.events.lock().unwrap().clone()
    }
}

struct Trainer {
    trace: Trace,
    reads: Mutex<usize>,
}
impl TrainingTrainerView for Trainer {
    fn current_iteration(&self) -> Result<BigInt, String> {
        Ok(0.into())
    }
    fn fast_dev_run(&self) -> Result<Option<i64>, String> {
        let mut reads = self.reads.lock().unwrap();
        *reads += 1;
        let value = if *reads == 1 || self.trace.input["second"] == "unchanged" {
            &self.trace.input["fast"]
        } else {
            &self.trace.input["second"]
        };
        self.trace
            .record(json!(["fast_dev", value]))
            .map_err(|e| e.to_string())?;
        if self.trace.input["fail"] == "fast_dev_second" && *reads == 2 {
            return Err("second trainer read".into());
        }
        Ok(value.as_i64())
    }
}
struct Backend(Trace);
impl FiniteVectorBackend<i64, i64, f64, Value> for Backend {
    fn environment_count(&self) -> usize {
        let count = self.0.input["count"].as_u64().unwrap_or(2);
        self.0.record(json!(["len", count])).unwrap();
        usize::try_from(count).unwrap()
    }
    fn reset(&mut self, _: &[usize]) -> Result<Vec<Option<i64>>, EnvironmentPluginError> {
        panic!("contract collector does not reset")
    }
    fn step(
        &mut self,
        _: &[i64],
        _: &[usize],
    ) -> Result<Vec<FiniteBackendStep<i64, f64, Value>>, EnvironmentPluginError> {
        panic!("contract collector does not step")
    }
}
struct Predicate;
impl FiniteObservationPredicate<i64> for Predicate {
    fn is_invalid(&mut self, value: &i64) -> Result<bool, EnvironmentPluginError> {
        Ok(*value < 0)
    }
}
impl FiniteVectorLogger<i64, f64, Value> for Trace {
    fn on_all_ready(&mut self) -> Result<(), EnvironmentPluginError> {
        self.record(json!(["ready"]))
            .map_err(|e| EnvironmentPluginError::new(e.to_string()))
    }
    fn on_all_done(&mut self) -> Result<(), EnvironmentPluginError> {
        self.record(json!(["done"]))
            .map_err(|e| EnvironmentPluginError::new(e.to_string()))
    }
}
impl TrainingMetricSink for Trace {
    fn info(&mut self, message: &str) -> Result<(), String> {
        self.record(json!(["log", message]))
            .map_err(|e| e.to_string())
    }
}
fn metrics(entries: &[(&str, i64)]) -> TrainingVesselMetrics {
    entries
        .iter()
        .map(|(key, value)| {
            (
                (*key).into(),
                TrainingMetricValue::Scalar(TrainingMetricScalar::Integer((*value).into())),
            )
        })
        .collect()
}
impl TrainingRunPolicy<Vec<i64>> for Trace {
    fn set_mode(&mut self, mode: TrainingPolicyMode) -> Result<(), TrainingVesselRunError> {
        self.record(json!([
            "mode",
            match mode {
                TrainingPolicyMode::Train => "train",
                TrainingPolicyMode::Evaluation => "evaluation",
            }
        ]))
    }
    fn update(
        &mut self,
        size: u64,
        buffer: Option<&mut Vec<i64>>,
        options: &TrainingUpdateOptions,
    ) -> Result<TrainingVesselMetrics, TrainingVesselRunError> {
        if self.input["no_buffer"] == true {
            assert!(buffer.is_none());
        } else {
            let buffer = buffer.unwrap();
            assert_eq!(
                Some(buffer.as_ptr() as usize),
                *self.buffer_address.lock().unwrap()
            );
            assert_eq!(buffer, &[42]);
            buffer.push(99);
        }
        let options: serde_json::Map<String, Value> = options
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        self.record(json!(["update", size, options]))?;
        Ok(metrics(&[("shared", 20), ("loss", 3)]))
    }
}
struct Collector {
    trace: Trace,
    buffer: Option<Vec<i64>>,
}
impl TrainingRunCollector<i64, i64, f64, Value, Vec<i64>> for Collector {
    fn collect(
        &mut self,
        _: &mut (dyn TrainingRunPolicy<Vec<i64>> + 'static),
        env: &mut Environment,
        limit: TrainingCollectLimit,
    ) -> Result<TrainingVesselMetrics, TrainingVesselRunError> {
        assert!(env.is_collector_guarded());
        let options = match limit {
            TrainingCollectLimit::Episodes(count) => json!({"n_episode":count}),
            TrainingCollectLimit::Steps(count) => json!({"n_step":count.to_string()}),
        };
        self.trace.record(json!(["collect", options]))?;
        if self.trace.input["real"] == true {
            assert_eq!(env.reset(None)?.observations, vec![Some(7)]);
            let step = env.step(&[3], None)?;
            assert_eq!(step.transitions[0].observation, Some(10));
            assert!(step.transitions[0].done);
            // Exhaust the actual finite dataset via the next reset.
            env.reset(None)?;
            panic!("second reset must exhaust the finite worker");
        }
        Ok(metrics(&[("reward", 1), ("shared", 2)]))
    }
    fn buffer(&mut self) -> Result<Option<&mut Vec<i64>>, TrainingVesselRunError> {
        self.trace.record(json!(["buffer_access"]))?;
        Ok(self.buffer.as_mut())
    }
}
impl TrainingCollectorFactory<i64, i64, f64, Value, Vec<i64>> for Trace {
    fn create_buffer(
        &mut self,
        capacity: i64,
        count: usize,
    ) -> Result<Vec<i64>, TrainingVesselRunError> {
        self.record(json!(["buffer", capacity, count]))?;
        let buffer = vec![42];
        *self.buffer_address.lock().unwrap() = Some(buffer.as_ptr() as usize);
        Ok(buffer)
    }
    fn create_collector(
        &mut self,
        _: &mut (dyn TrainingRunPolicy<Vec<i64>> + 'static),
        env: &mut Environment,
        buffer: Option<Vec<i64>>,
        noise: bool,
    ) -> Result<BoxTrainingRunCollector<i64, i64, f64, Value, Vec<i64>>, TrainingVesselRunError>
    {
        assert!(env.is_collector_guarded());
        self.record(json!(["collector", buffer.is_some(), noise]))?;
        let buffer = if self.input["no_buffer"] == true {
            None
        } else {
            buffer
        };
        Ok(Box::new(Collector {
            trace: self.clone(),
            buffer,
        }))
    }
}
fn setup(input: Value) -> (Runner, Environment, Trace, Arc<dyn TrainingTrainerView>) {
    let trace = Trace::new(input);
    let trainer: Arc<dyn TrainingTrainerView> = Arc::new(Trainer {
        trace: trace.clone(),
        reads: Mutex::new(0),
    });
    let mut binding = TrainingVesselBinding::default();
    binding.assign_trainer(&trainer);
    let config = TrainingVesselRunConfig {
        update_kwargs: trace.input["kwargs"]
            .as_object()
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default(),
        ..TrainingVesselRunConfig::default()
    };
    let runner = Runner::new(
        Box::new(trace.clone()),
        Box::new(trace.clone()),
        binding,
        config,
        TrainingVesselLog::with_plugins(
            Box::new(ArrowTrainingMetricReducer),
            Box::new(trace.clone()),
        ),
    );
    let env = Environment::new(
        Box::new(Backend(trace.clone())),
        Box::new(Predicate),
        vec![Box::new(trace.clone())],
    );
    trace.events.lock().unwrap().clear();
    (runner, env, trace, trainer)
}
fn encoded_result(result: Option<TrainingVesselMetrics>) -> Value {
    result.map_or(Value::Null, |m| {
        Value::Array(
            m.into_iter()
                .map(|(key, value)| {
                    let TrainingMetricValue::Scalar(TrainingMetricScalar::Integer(value)) = value
                    else {
                        panic!("integer metric")
                    };
                    json!([key, value.to_string().parse::<i64>().unwrap()])
                })
                .collect(),
        )
    })
}

#[test]
fn live_python_collection_update_and_guard_contract() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = root.join("../../../qlib/qlib/rl");
    let output = Command::new("python")
        .args([
            root.join("tests/fixtures/training_vessel_runner_contract.py"),
            source.join("trainer/vessel.py"),
            source.join("utils/finite_env.py"),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(cases.len(), 42);
    for case in cases {
        let (mut runner, mut env, trace, _trainer) = setup(case["input"].clone());
        let result = match case["input"]["phase"].as_str().unwrap() {
            "train" => runner.train(&mut env),
            "validate" => runner.validate(&mut env),
            "test" => runner.test(&mut env),
            _ => unreachable!(),
        };
        assert_eq!(
            trace.snapshot(),
            case["events"].as_array().unwrap().clone(),
            "{}",
            case["input"]
        );
        assert_eq!(
            env.is_collector_guarded(),
            case["guarded"].as_bool().unwrap()
        );
        match result {
            Ok(metrics) => {
                assert!(case["error"].is_null(), "{case}");
                assert_eq!(encoded_result(metrics), case["result"]);
            }
            Err(error) => {
                assert!(!case["error"].is_null(), "{error}: {case}");
                assert!(!error.to_string().is_empty());
                assert_eq!(error, error.clone());
                match case["error"].as_str().unwrap() {
                    "TypeError" => {
                        assert!(matches!(error, TrainingVesselRunError::DuplicateKeyword(_)));
                    }
                    "StopIteration" => assert_eq!(
                        error,
                        TrainingVesselRunError::Environment(FiniteVectorError::Exhausted)
                    ),
                    "RuntimeError" => assert!(!matches!(
                        error,
                        TrainingVesselRunError::DuplicateKeyword(_)
                            | TrainingVesselRunError::Environment(FiniteVectorError::Exhausted)
                    )),
                    _ => panic!("unexpected Python error"),
                }
            }
        }
    }
}

#[test]
fn configuration_live_binding_and_collector_buffer_edges() {
    let (mut runner, mut env, trace, trainer) =
        setup(json!({"second":"unchanged","no_buffer":true}));
    runner.config.buffer_size = -1;
    runner.config.episode_per_iter = 0;
    assert_eq!(runner.config, runner.config.clone());
    assert!(format!("{:?}", runner.config).contains("-1"));
    assert!(runner.train(&mut env).unwrap().is_some());
    assert!(trace.snapshot().contains(&json!(["buffer", -1, 2])));
    assert!(
        trace
            .snapshot()
            .contains(&json!(["collect",{"n_episode":0}]))
    );
    // Duplicate keyword detection preserves input order, after evaluating collector.buffer.
    runner.config.update_kwargs = IndexMap::from([
        ("buffer".into(), json!(1)),
        ("sample_size".into(), json!(2)),
    ]);
    assert_eq!(
        runner.train(&mut env).err(),
        Some(TrainingVesselRunError::DuplicateKeyword("buffer".into()))
    );
    runner.config.update_kwargs.clear();
    drop(trainer);
    assert_eq!(
        runner.train(&mut env).err(),
        Some(TrainingVesselBindingError::Expired.into())
    );
    runner.binding = TrainingVesselBinding::default();
    assert_eq!(
        runner.train(&mut env).err(),
        Some(TrainingVesselBindingError::Unassigned.into())
    );
    assert_eq!(
        runner.test(&mut env).err(),
        Some(TrainingVesselRunError::Log(
            TrainingVesselLogError::Binding(TrainingVesselBindingError::Unassigned)
        ))
    );
    assert!(!env.is_collector_guarded());
    assert!(
        format!(
            "{:?}",
            TrainingCollectLimit::Steps(BigInt::from(INF) * usize::MAX)
        )
        .contains("Steps")
    );
    assert_eq!(
        TrainingCollectLimit::Episodes(None),
        TrainingCollectLimit::Episodes(None).clone()
    );
}

struct Worker {
    resets: usize,
}

// Deliberately neither Clone nor Serialize: a policy can receive owned tensor/runtime handles.
struct OpaqueOption(Arc<Mutex<i64>>);
struct OpaquePolicy(Arc<Mutex<i64>>);
impl TrainingRunPolicy<Vec<i64>, OpaqueOption> for OpaquePolicy {
    fn set_mode(&mut self, _: TrainingPolicyMode) -> Result<(), TrainingVesselRunError> {
        Ok(())
    }
    fn update(
        &mut self,
        sample_size: u64,
        buffer: Option<&mut Vec<i64>>,
        options: &TrainingUpdateOptions<OpaqueOption>,
    ) -> Result<TrainingVesselMetrics, TrainingVesselRunError> {
        assert_eq!(sample_size, 0);
        assert_eq!(buffer.unwrap(), &[10]);
        assert!(Arc::ptr_eq(&self.0, &options["tensor"].0));
        *options["tensor"].0.lock().unwrap() += 1;
        Ok(IndexMap::new())
    }
}
struct OpaqueCollector(Option<Vec<i64>>);
impl TrainingRunCollector<i64, i64, f64, Value, Vec<i64>, OpaqueOption> for OpaqueCollector {
    fn collect(
        &mut self,
        _: &mut (dyn TrainingRunPolicy<Vec<i64>, OpaqueOption> + 'static),
        _: &mut Environment,
        _: TrainingCollectLimit,
    ) -> Result<TrainingVesselMetrics, TrainingVesselRunError> {
        Ok(IndexMap::new())
    }
    fn buffer(&mut self) -> Result<Option<&mut Vec<i64>>, TrainingVesselRunError> {
        Ok(self.0.as_mut())
    }
}
struct OpaqueFactory;
impl TrainingCollectorFactory<i64, i64, f64, Value, Vec<i64>, OpaqueOption> for OpaqueFactory {
    fn create_buffer(&mut self, _: i64, _: usize) -> Result<Vec<i64>, TrainingVesselRunError> {
        Ok(vec![10])
    }
    fn create_collector(
        &mut self,
        _: &mut (dyn TrainingRunPolicy<Vec<i64>, OpaqueOption> + 'static),
        _: &mut Environment,
        buffer: Option<Vec<i64>>,
        _: bool,
    ) -> Result<
        BoxTrainingRunCollector<i64, i64, f64, Value, Vec<i64>, OpaqueOption>,
        TrainingVesselRunError,
    > {
        Ok(Box::new(OpaqueCollector(buffer)))
    }
}

#[test]
fn opaque_update_values_are_forwarded_without_serialization_or_cloning() {
    let (original, mut env, _, _trainer) = setup(json!({"second":"unchanged"}));
    let shared = Arc::new(Mutex::new(5));
    let mut runner = TrainingVesselRunner::new(
        Box::new(OpaquePolicy(shared.clone())),
        Box::new(OpaqueFactory),
        original.binding,
        TrainingVesselRunConfig {
            update_kwargs: IndexMap::from([("tensor".into(), OpaqueOption(shared.clone()))]),
            ..TrainingVesselRunConfig::default()
        },
        TrainingVesselLog::default(),
    );
    assert!(runner.train(&mut env).unwrap().unwrap().is_empty());
    assert_eq!(*shared.lock().unwrap(), 6);
    runner.binding = TrainingVesselBinding::default();
    assert!(runner.validate(&mut env).unwrap().unwrap().is_empty());
    assert!(runner.test(&mut env).unwrap().unwrap().is_empty());
    assert_eq!(*shared.lock().unwrap(), 6);
}
impl FiniteEnvironment<i64, i64, f64, Value> for Worker {
    fn reset(&mut self) -> Result<i64, EnvironmentPluginError> {
        self.resets += 1;
        Ok(if self.resets == 1 { 7 } else { -1 })
    }
    fn step(
        &mut self,
        action: &i64,
    ) -> Result<FiniteBackendStep<i64, f64, Value>, EnvironmentPluginError> {
        Ok(FiniteBackendStep {
            observation: Some(7 + action),
            reward: Some(1.0),
            done: true,
            info: Some(json!({})),
        })
    }
}

#[test]
fn actual_finite_dummy_exhaustion_skips_update_and_finishes_logging() {
    for phase in ["train", "validate", "test"] {
        let (mut runner, _, trace, _trainer) = setup(json!({"real":true,"second":"unchanged"}));
        let backend = FiniteDummyBackend::new(vec![Box::new(Worker { resets: 0 })]).unwrap();
        let mut env = Environment::new(
            Box::new(backend),
            Box::new(Predicate),
            vec![Box::new(trace.clone())],
        );
        let result = match phase {
            "train" => runner.train(&mut env),
            "validate" => runner.validate(&mut env),
            _ => runner.test(&mut env),
        };
        assert!(result.unwrap().is_none());
        assert_eq!(env.unguarded_reset_warnings(), 0);
        assert!(!env.is_collector_guarded());
        assert!(env.is_zombie());
        assert_eq!(trace.snapshot().last().unwrap(), &json!(["done"]));
        assert!(
            !trace
                .snapshot()
                .iter()
                .any(|e| e[0] == "update" || e[0] == "log")
        );
        // Exhaustion is suppressed, but reusing the resulting zombie is an ordinary failure.
        let error = runner.train(&mut env).err().unwrap();
        assert_eq!(
            error,
            TrainingVesselRunError::Environment(FiniteVectorError::Zombie)
        );
        assert_eq!(trace.snapshot().last().unwrap()[0], "collect");
        assert!(!env.is_collector_guarded());
    }
}
