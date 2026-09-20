#![allow(clippy::float_cmp)]

use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    path::PathBuf,
    process::Command,
};

use super::*;
use crate::{
    ArrowTrainingMetricReducer, RlLogBuffer, RlLogEntry, RlLogValue, TrainingMetricScalar,
    TrainingMetricSink, TrainingMetricValue, TrainingVesselBinding, TrainingVesselBindingError,
    TrainingVesselLog,
};
use serde_json::{Value, json};

type State = RlTrainerState<f64>;
type Runtime = RlTrainerRuntime<f64>;

fn fixture() -> Value {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .args([
            root.join("tests/fixtures/rl_trainer_state_contract.py"),
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
fn snapshot(state: &State) -> Value {
    json!({"should_stop":state.should_stop,"current_iter":state.current_iter.as_ref().map(ToString::to_string),
        "current_episode":state.current_episode.as_ref().map(ToString::to_string),"current_stage":state.current_stage,
        "metrics":state.metrics.as_ref().map(|m|m.iter().map(|(k,v)|json!([k,v])).collect::<Vec<_>>())})
}
struct Source {
    input: Value,
    events: Vec<String>,
}
impl Source {
    fn record(&mut self, field: &str) -> Result<(), RlTrainerStateError> {
        self.events.push(field.into());
        if self.input["fail"] == field {
            Err(RlTrainerStateError::Provider {
                field: field.into(),
                message: "injected".into(),
            })
        } else {
            Ok(())
        }
    }
    fn values(&self, scale: f64) -> RlTrainerMetrics {
        if self.input["empty"] == true {
            IndexMap::new()
        } else {
            IndexMap::from([
                ("reward".into(), 2.5 * scale),
                ("val/nested".into(), 4.0 * scale),
            ])
        }
    }
}
impl RlTrainerMetricSource<f64> for Source {
    fn global_episode(&mut self) -> Result<BigInt, RlTrainerStateError> {
        self.record("global_episode")?;
        Ok(77.into())
    }
    fn episode_metrics(&mut self) -> Result<RlTrainerMetrics, RlTrainerStateError> {
        self.record("episode_metrics")?;
        Ok(self.values(1.0))
    }
    fn collect_metrics(&mut self) -> Result<RlTrainerMetrics, RlTrainerStateError> {
        self.record("collect_metrics")?;
        Ok(self.values(2.0))
    }
}

#[test]
fn initialization_live_levels_and_metric_updates_match_python() {
    let f = fixture();
    let mut state = State::default();
    assert_eq!(snapshot(&state), f["initial"]);
    state.current_iter = Some(12.into());
    state.current_episode = Some(8.into());
    state.current_stage = "val".into();
    state.should_stop = Some(true);
    state.metrics = Some(IndexMap::from([("old".into(), 99.0)]));
    state.initialize();
    assert_eq!(snapshot(&state), f["initialized"]);
    state.current_stage = "test".into();
    state.current_iter = Some(7.into());
    state.should_stop = Some(true);
    state.initialize_iter();
    assert_eq!(snapshot(&state), f["iteration"]);
    assert_eq!(f["cases"].as_array().unwrap().len(), 27);
    for case in f["cases"].as_array().unwrap() {
        let input = &case["input"];
        let mut state = State {
            current_stage: input["stage"].as_str().unwrap().into(),
            ..State::default()
        };
        if input["have_metrics"] == true {
            state.metrics = Some(IndexMap::from([
                ("old".into(), 99.0),
                ("reward".into(), -1.0),
                ("val/reward".into(), -2.0),
            ]));
        }
        let mut source = Source {
            input: input.clone(),
            events: vec![],
        };
        let result = state.metrics_callback(
            input["episode"].as_bool().unwrap(),
            input["collect"].as_bool().unwrap(),
            &mut source,
        );
        let category = result.err().map(|error| {
            assert!(!error.to_string().is_empty());
            assert_eq!(error, error.clone());
            match error {
                RlTrainerStateError::Uninitialized(_) => "AttributeError",
                RlTrainerStateError::NoMetricEvent => "UnboundLocalError",
                RlTrainerStateError::Provider { .. } => "RuntimeError",
                _ => panic!("unexpected source error"),
            }
        });
        assert_eq!(json!(category), case["error"], "{input}");
        assert_eq!(json!(source.events), case["events"]);
        assert_eq!(snapshot(&state), case["state"]);
        let restored: State =
            serde_json::from_value(serde_json::to_value(&state).unwrap()).unwrap();
        assert_eq!(state, restored);
        assert_eq!(state, state.clone());
        assert!(format!("{state:?}").contains("current_stage"));
        assert_eq!(
            bincode::deserialize::<State>(&bincode::serialize(&state).unwrap()).unwrap(),
            state
        );
    }
    for case in f["levels"].as_array().unwrap() {
        let levels = case[0]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap());
        assert_eq!(minimum_rl_log_level(levels), case[1].as_i64().unwrap());
    }
    let mut levels = vec![30, 40];
    assert_eq!(minimum_rl_log_level(levels.clone()), 30);
    levels[1] = -1;
    assert_eq!(minimum_rl_log_level(levels), -1);
}

struct Sink(Arc<Mutex<Vec<String>>>);
impl TrainingMetricSink for Sink {
    fn info(&mut self, message: &str) -> Result<(), String> {
        self.0.lock().unwrap().push(message.into());
        Ok(())
    }
}

#[test]
fn real_log_buffer_updates_live_trainer_and_vessel_binding_across_stages() {
    let runtime = Arc::new(Runtime::new(Some(2)));
    assert!(
        runtime
            .current_iteration()
            .unwrap_err()
            .contains("not been initialized")
    );
    let view: Arc<dyn TrainingTrainerView> = runtime.clone();
    let mut binding = TrainingVesselBinding::default();
    binding.assign_trainer(&view);
    drop(view);
    let mut buffer = RlLogBuffer::<String, _>::new_buffer(20, runtime.buffer_callback());
    assert_eq!(Arc::strong_count(&runtime), 2);
    runtime
        .update(|state| {
            state.initialize();
            state.initialize_iter();
        })
        .unwrap();
    for (reward, score) in [(2.0, Some(4.0)), (6.0, None)] {
        let mut logs = IndexMap::from([(
            "reward".into(),
            RlLogEntry {
                level: 20,
                value: RlLogValue::Float(reward),
            },
        )]);
        if let Some(score) = score {
            logs.insert(
                "score".into(),
                RlLogEntry {
                    level: 20,
                    value: RlLogValue::Float(score),
                },
            );
        }
        buffer.on_env_reset(0);
        buffer.on_env_step(0, 999.0, true, Some(&logs)).unwrap();
    }
    assert_eq!(
        runtime
            .read(|s| s.metrics.as_ref().unwrap()["score"])
            .unwrap(),
        4.0
    );
    buffer.on_env_all_done().unwrap();
    runtime
        .read(|s| {
            assert_eq!(s.current_episode, Some(2.into()));
            assert_eq!(s.metrics.as_ref().unwrap()["reward"], 4.0);
            assert_eq!(s.metrics.as_ref().unwrap()["score"], 2.0);
        })
        .unwrap();
    runtime
        .update(|s| {
            s.current_stage = "val".into();
            s.current_iter = Some(1.into());
        })
        .unwrap();
    buffer.clear();
    buffer.on_env_reset(0);
    let logs = IndexMap::from([
        (
            "reward".into(),
            RlLogEntry {
                level: 20,
                value: RlLogValue::Float(8.0),
            },
        ),
        (
            "val/already".into(),
            RlLogEntry {
                level: 20,
                value: RlLogValue::Float(3.0),
            },
        ),
    ]);
    buffer.on_env_step(0, 999.0, true, Some(&logs)).unwrap();
    buffer.on_env_all_done().unwrap();
    runtime
        .read(|s| {
            assert_eq!(s.current_episode, Some(3.into()));
            assert_eq!(s.metrics.as_ref().unwrap()["reward"], 4.0);
            assert_eq!(s.metrics.as_ref().unwrap()["val/reward"], 8.0);
            assert_eq!(s.metrics.as_ref().unwrap()["val/val/already"], 3.0);
        })
        .unwrap();
    assert_live_vessel_view(&runtime, &binding);
    let iteration = binding.current_iteration().unwrap();
    drop(runtime);
    assert_eq!(binding.current_iteration().unwrap(), iteration);
    buffer.on_env_all_done().unwrap();
    drop(buffer);
    assert_eq!(
        binding.current_iteration(),
        Err(TrainingVesselBindingError::Expired)
    );
}

#[test]
fn callback_keeps_runtime_alive_like_the_python_bound_method_without_a_cycle() {
    let f = fixture();
    let runtime = Arc::new(Runtime::new(None));
    runtime
        .update(|state| {
            state.initialize();
            state.initialize_iter();
        })
        .unwrap();
    let observer = Arc::downgrade(&runtime);
    let mut callback = runtime.buffer_callback::<String>();
    drop(runtime);
    assert_eq!(
        json!(observer.upgrade().is_some()),
        f["lifetime"]["retained"]
    );
    let writer = RlLogWriterState {
        global_episode: 1.into(),
        ..RlLogWriterState::default()
    };
    let buffer = RlLogBufferState {
        latest_metrics: Some(IndexMap::from([("reward".into(), 2.0)])),
        ..RlLogBufferState::default()
    };
    callback(RlLogBufferEvent::Episode, &writer, &buffer).unwrap();
    assert_eq!(
        observer.upgrade().unwrap().read(snapshot).unwrap(),
        f["lifetime"]["after_callback"]
    );
    drop(callback);
    assert_eq!(
        json!(observer.upgrade().is_none()),
        f["lifetime"]["released"]
    );
}

fn assert_live_vessel_view(runtime: &Runtime, binding: &TrainingVesselBinding) {
    let messages = Arc::new(Mutex::new(vec![]));
    let mut logger = TrainingVesselLog::with_plugins(
        Box::new(ArrowTrainingMetricReducer),
        Box::new(Sink(messages.clone())),
    );
    logger
        .log(
            binding,
            "val/reward",
            &TrainingMetricValue::Scalar(TrainingMetricScalar::Float(8.0)),
        )
        .unwrap();
    assert_eq!(*messages.lock().unwrap(), vec!["[Iter 2] val/reward = 8.0"]);
    assert_eq!(binding.fast_dev_run().unwrap(), Some(2));
    for value in [None, Some(0), Some(-3)] {
        runtime.set_fast_dev_run(value).unwrap();
        assert_eq!(binding.fast_dev_run().unwrap(), value);
    }
    let huge = BigInt::from(10).pow(30);
    runtime
        .update(|s| s.current_iter = Some(huge.clone()))
        .unwrap();
    assert_eq!(binding.current_iteration().unwrap(), huge);
}

#[test]
fn buffer_read_failures_preserve_counter_updates_and_poison_is_not_hidden() {
    let runtime = Arc::new(Runtime::new(None));
    let writer = RlLogWriterState::<String> {
        global_episode: 99.into(),
        episode_count: BigInt::from(10).pow(1000),
        ..RlLogWriterState::default()
    };
    let mut buffer = RlLogBufferState::default();
    let mut callback = runtime.buffer_callback();
    assert!(
        callback(RlLogBufferEvent::Episode, &writer, &buffer)
            .unwrap_err()
            .contains("no episode metrics")
    );
    assert_eq!(
        runtime.read(|s| s.current_episode.clone()).unwrap(),
        Some(99.into())
    );
    buffer.aggregated_metrics.insert("reward".into(), 1.0);
    assert!(
        callback(RlLogBufferEvent::Collect, &writer, &buffer)
            .unwrap_err()
            .contains("cannot be represented")
    );
    assert_eq!(
        runtime.read(|s| s.current_episode.clone()).unwrap(),
        Some(99.into())
    );
    buffer.latest_metrics = Some(IndexMap::new());
    assert!(
        callback(RlLogBufferEvent::Episode, &writer, &buffer)
            .unwrap_err()
            .contains("Metrics")
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _ = runtime.update::<()>(|s| {
                s.current_stage = "partial".into();
                panic!("injected update panic");
            });
        }))
        .is_err()
    );
    assert_eq!(runtime.read(|_| ()), Err(RlTrainerStateError::Poisoned));
    assert_eq!(runtime.update(|_| ()), Err(RlTrainerStateError::Poisoned));
    assert_eq!(
        runtime.set_fast_dev_run(None),
        Err(RlTrainerStateError::Poisoned)
    );
    assert!(
        runtime
            .current_iteration()
            .unwrap_err()
            .contains("poisoned")
    );
    assert!(runtime.fast_dev_run().unwrap_err().contains("poisoned"));
    assert!(
        callback(RlLogBufferEvent::Collect, &writer, &buffer)
            .unwrap_err()
            .contains("poisoned")
    );
    let error = RlTrainerStateError::Buffer(RlLogError::NoEpisodeMetrics);
    assert_eq!(error, error.clone());
    assert!(!error.to_string().is_empty());
}

struct Opaque(Arc<Vec<i64>>);
struct OpaqueSource(Option<Opaque>);
impl RlTrainerMetricSource<Opaque> for OpaqueSource {
    fn global_episode(&mut self) -> Result<BigInt, RlTrainerStateError> {
        Ok(1.into())
    }
    fn episode_metrics(&mut self) -> Result<RlTrainerMetrics<Opaque>, RlTrainerStateError> {
        Ok(IndexMap::from([("payload".into(), self.0.take().unwrap())]))
    }
    fn collect_metrics(&mut self) -> Result<RlTrainerMetrics<Opaque>, RlTrainerStateError> {
        Ok(IndexMap::new())
    }
}

#[test]
fn opaque_metrics_move_without_cloning_and_nonfinite_numbers_survive_updates() {
    let payload = Arc::new(vec![1, 2, 3]);
    let mut source = OpaqueSource(Some(Opaque(payload.clone())));
    let mut state = RlTrainerState::<Opaque>::default();
    state.initialize_iter();
    state.current_stage = "val".into();
    state.metrics_callback(true, false, &mut source).unwrap();
    state.metrics_callback(false, true, &mut source).unwrap();
    assert!(Arc::ptr_eq(
        &state.metrics.as_ref().unwrap()["val/payload"].0,
        &payload
    ));
    assert!(source.0.is_none());
    let writer = RlLogWriterState::<String>::default();
    let buffer = RlLogBufferState {
        latest_metrics: Some(IndexMap::from([
            ("nan".into(), f64::NAN),
            ("inf".into(), f64::INFINITY),
        ])),
        aggregated_metrics: IndexMap::new(),
    };
    let mut source = RlLogBufferMetricSource {
        writer: &writer,
        buffer: &buffer,
    };
    let mut state = State::default();
    state.initialize_iter();
    state.metrics_callback(true, false, &mut source).unwrap();
    assert!(state.metrics.as_ref().unwrap()["nan"].is_nan());
    assert!(state.metrics.as_ref().unwrap()["inf"].is_infinite());
}
