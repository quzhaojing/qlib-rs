//! Actual GRU/PPO learning inside the Trainer/callback/native-file lifecycle.
use super::*;
use crate::rl_candle_checkpoint::CandlePolicySnapshot;
use crate::*;
use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
};

type Snapshot = TrainingVesselCheckpoint<CandlePolicySnapshot<()>>;
type Logger = RlLogWriter<String>;
type LoggerState = RlLogWriterState<String>;
type Graph = RlTrainerGraph<Snapshot, Option<u64>, LoggerState>;
type Document = RlTrainerCheckpoint<Snapshot, Option<u64>, LoggerState>;

#[derive(Clone, Debug, PartialEq)]
struct TensorBits {
    dtype: safetensors::Dtype,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}
#[derive(Clone, Debug, PartialEq)]
struct SnapshotValue {
    metadata: Option<HashMap<String, String>>,
    tensors: BTreeMap<String, TensorBits>,
}
fn snapshot_value(snapshot: &Snapshot) -> SnapshotValue {
    let bytes = &snapshot.policy.tensors;
    let (_, metadata) = safetensors::SafeTensors::read_metadata(bytes).unwrap();
    let tensors = safetensors::SafeTensors::deserialize(bytes).unwrap();
    SnapshotValue {
        metadata: metadata.metadata().clone(),
        tensors: tensors
            .iter()
            .map(|(name, tensor)| {
                (
                    name.to_owned(),
                    TensorBits {
                        dtype: tensor.dtype(),
                        shape: tensor.shape().to_vec(),
                        bytes: tensor.data().to_vec(),
                    },
                )
            })
            .collect(),
    }
}

fn header_end(snapshot: &Snapshot) -> usize {
    8 + usize::try_from(u64::from_le_bytes(
        snapshot.policy.tensors[..8].try_into().unwrap(),
    ))
    .unwrap()
}

pub(super) fn assert_same_snapshot(actual: &Snapshot, expected: &Snapshot) {
    assert_eq!(snapshot_value(actual), snapshot_value(expected));
}
fn replace_header(snapshot: &Snapshot, header: &impl serde::Serialize) -> Snapshot {
    let mut encoded = serde_json::to_vec(header).unwrap();
    encoded.resize(encoded.len() + (8 - encoded.len() % 8) % 8, b' ');
    let mut bytes = u64::try_from(encoded.len()).unwrap().to_le_bytes().to_vec();
    bytes.extend(encoded);
    bytes.extend_from_slice(&snapshot.policy.tensors[header_end(snapshot)..]);
    Snapshot {
        policy: CandlePolicySnapshot {
            tensors: bytes,
            metadata: (),
        },
    }
}

#[test]
fn snapshot_comparison_ignores_only_header_order_not_values_or_layout() {
    let variable = Var::new(&[0_f32, -0_f32], &Device::Cpu).unwrap();
    let state = crate::rl_candle_checkpoint::CandlePolicyState::new(
        IndexMap::from([
            ("weight".into(), variable.clone()),
            ("alias".into(), variable),
        ]),
        (),
    );
    let snapshot = Snapshot {
        policy: state.snapshot().unwrap(),
    };
    let original = snapshot_value(&snapshot);
    let header: Value =
        serde_json::from_slice(&snapshot.policy.tensors[8..header_end(&snapshot)]).unwrap();
    let reversed: IndexMap<_, _> = header.as_object().unwrap().iter().rev().collect();
    let reordered = replace_header(&snapshot, &reversed);
    assert_ne!(snapshot.policy.tensors, reordered.policy.tensors);
    assert_eq!(original, snapshot_value(&reordered));

    let mut different_bits = snapshot.clone();
    *different_bits.policy.tensors.last_mut().unwrap() ^= 0x80;
    assert_ne!(
        original,
        snapshot_value(&different_bits),
        "signed zero bits matter"
    );
    let mut different_shape = header.clone();
    different_shape["pweight"]["shape"] = json!([1, 2]);
    assert_ne!(
        original,
        snapshot_value(&replace_header(&snapshot, &different_shape))
    );
    let mut different_dtype = header.clone();
    different_dtype["pweight"]["dtype"] = json!("I32");
    assert_ne!(
        original,
        snapshot_value(&replace_header(&snapshot, &different_dtype))
    );
    let mut different_layout = header.clone();
    let layout = &mut different_layout["__metadata__"]["core.candle_policy.layout"];
    let mut decoded: Value = serde_json::from_str(layout.as_str().unwrap()).unwrap();
    decoded["entries"][1][1] = json!(1);
    *layout = json!(decoded.to_string());
    assert_ne!(
        original,
        snapshot_value(&replace_header(&snapshot, &different_layout))
    );
    let mut renamed = header;
    let tensor = renamed.as_object_mut().unwrap().remove("pweight").unwrap();
    renamed["prenamed"] = tensor;
    assert_ne!(
        original,
        snapshot_value(&replace_header(&snapshot, &renamed))
    );
}

type Runner = TrainingVesselRunner<
    RecurrentObservation,
    i64,
    f64,
    Value,
    CandleVectorReplayBuffer,
    Value,
    CandlePpoVessel<StdRng>,
>;

struct LiveLogger(Arc<Mutex<Logger>>);
impl RlCheckpointState<LoggerState> for LiveLogger {
    fn save_checkpoint(&mut self) -> Result<LoggerState, String> {
        Ok(self.0.lock().unwrap().state_dict().clone())
    }
    fn load_checkpoint(&mut self, state: &LoggerState) -> Result<(), String> {
        self.0.lock().unwrap().load_state_dict(state.clone());
        Ok(())
    }
}
// This fixture's environment has no application log entries. Forward its real
// reset/step/reward/done events, rather than synthesizing episodes in a callback.
impl FiniteVectorLogger<RecurrentObservation, f64, Value> for LiveLogger {
    fn on_all_ready(&mut self) -> Result<(), EnvironmentPluginError> {
        self.0.lock().unwrap().clear();
        Ok(())
    }
    fn on_all_done(&mut self) -> Result<(), EnvironmentPluginError> {
        self.0.lock().unwrap().on_env_all_done().unwrap();
        Ok(())
    }
    fn on_reset(
        &mut self,
        id: usize,
        _: &[Option<RecurrentObservation>],
    ) -> Result<(), EnvironmentPluginError> {
        self.0.lock().unwrap().on_env_reset(id);
        Ok(())
    }
    fn on_step(
        &mut self,
        id: usize,
        step: &FiniteBackendStep<RecurrentObservation, f64, Value>,
    ) -> Result<(), EnvironmentPluginError> {
        self.0
            .lock()
            .unwrap()
            .on_env_step(id, step.reward.unwrap(), step.done, Some(&IndexMap::new()))
            .unwrap();
        Ok(())
    }
}

struct Vessel {
    runner: Runner,
    logger: Arc<Mutex<Logger>>,
    watched: Tensor,
    attachments: usize,
    updates: usize,
}
impl RlCheckpointState<Snapshot> for Vessel {
    fn save_checkpoint(&mut self) -> Result<Snapshot, String> {
        self.runner.save_checkpoint()
    }
    fn load_checkpoint(&mut self, state: &Snapshot) -> Result<(), String> {
        assert!(self.attachments > 0, "attach before restoring policy");
        self.runner.load_checkpoint(state)
    }
}
impl RlTrainerVessel for Vessel {
    type Seed = DataQueue<usize>;
    type Environment = CandleCollectorEnvironment<Value>;
    fn assign_trainer(
        &mut self,
        runtime: &Arc<RlTrainerRuntime>,
    ) -> Result<(), RlTrainerDriverError> {
        let trainer: Arc<dyn TrainingTrainerView> = runtime.clone();
        self.runner.assign_trainer(&trainer);
        self.attachments += 1;
        Ok(())
    }
    fn seeds(&mut self, phase: RlTrainerPhase) -> Result<Self::Seed, RlTrainerDriverError> {
        assert_eq!(phase, RlTrainerPhase::Train);
        // A small native seed fixture selects this environment's worker count.
        // The production queue, not a no-op test context, owns entry and cleanup.
        Ok(DataQueue::new(
            Arc::new(vec![2_usize]),
            DataQueueConfig {
                repeat: 1,
                shuffle: false,
                ..DataQueueConfig::default()
            },
        ))
    }
    fn environment(
        &mut self,
        seed: &mut Self::Seed,
        _: &mut RlTrainerControl,
    ) -> Result<Self::Environment, RlTrainerDriverError> {
        assert!(seed.is_activated());
        let workers = seed.get().unwrap();
        assert_eq!(workers, 2);
        let trace = Arc::new(Mutex::new(Trace {
            ticks: vec![0; workers],
            resets: vec![0; workers],
            ..Trace::default()
        }));
        Ok(FiniteVectorEnv::new(
            Box::new(Backend(trace)),
            Box::new(Predicate),
            vec![Box::new(LiveLogger(self.logger.clone()))],
        ))
    }
    fn run(
        &mut self,
        phase: RlTrainerPhase,
        environment: &mut Self::Environment,
        _: &mut RlTrainerControl,
    ) -> Result<(), RlTrainerDriverError> {
        assert_eq!(phase, RlTrainerPhase::Train);
        let before = self.watched.to_vec2::<f32>().unwrap();
        let metrics = self.runner.train(environment).unwrap().unwrap();
        assert_eq!(metric_json(&metrics["n/ep"]), json!(3));
        assert_eq!(metric_json(&metrics["n/st"]), json!(5));
        let losses = metric_json(&metrics["loss"]);
        assert_eq!(losses.as_array().unwrap().len(), 2);
        assert!(
            losses
                .as_array()
                .unwrap()
                .iter()
                .all(|v| v.as_f64().unwrap().is_finite())
        );
        assert_ne!(self.watched.to_vec2::<f32>().unwrap(), before);
        self.updates += 1;
        Ok(())
    }
}

struct Counter(Arc<Mutex<u64>>);
impl RlCheckpointState<Option<u64>> for Counter {
    fn save_checkpoint(&mut self) -> Result<Option<u64>, String> {
        Ok(Some(*self.0.lock().unwrap()))
    }
    fn load_checkpoint(&mut self, state: &Option<u64>) -> Result<(), String> {
        *self.0.lock().unwrap() = state.unwrap();
        Ok(())
    }
}
struct UnitState;
impl RlCheckpointState<Option<u64>> for UnitState {
    fn save_checkpoint(&mut self) -> Result<Option<u64>, String> {
        Ok(None)
    }
    fn load_checkpoint(&mut self, state: &Option<u64>) -> Result<(), String> {
        assert_eq!(*state, None);
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq)]
struct Start {
    snapshot: SnapshotValue,
    counter: u64,
    logger: LoggerState,
    iteration: BigInt,
}
struct Observer {
    counter: Arc<Mutex<u64>>,
    starts: Arc<Mutex<Vec<Start>>>,
}
impl RlTrainerCallback<Vessel> for Observer {
    fn call(
        &mut self,
        hook: RlTrainerHook,
        control: &mut RlTrainerControl,
        vessel: &mut Vessel,
    ) -> Result<(), RlTrainerDriverError> {
        if hook == RlTrainerHook::FitStart {
            self.starts.lock().unwrap().push(Start {
                snapshot: snapshot_value(&vessel.save_checkpoint().unwrap()),
                counter: *self.counter.lock().unwrap(),
                logger: vessel.logger.lock().unwrap().state_dict().clone(),
                iteration: control.runtime.current_iteration().unwrap(),
            });
        }
        if hook == RlTrainerHook::IterEnd {
            *self.counter.lock().unwrap() += 1;
        }
        Ok(())
    }
}
fn graph(counter: &Arc<Mutex<u64>>, logger: &Arc<Mutex<Logger>>) -> Graph {
    Graph::new(
        vec![
            RlOwnedCheckpointComponent {
                type_name: "Observer".into(),
                state: Box::new(Counter(counter.clone())),
            },
            RlOwnedCheckpointComponent {
                type_name: "Checkpoint".into(),
                state: Box::new(UnitState),
            },
        ],
        vec![RlOwnedCheckpointComponent {
            type_name: "LogWriter".into(),
            state: Box::new(LiveLogger(logger.clone())),
        }],
    )
}
fn driver(
    path: &Path,
    counter: &Arc<Mutex<u64>>,
    logger: &Arc<Mutex<Logger>>,
    starts: &Arc<Mutex<Vec<Start>>>,
) -> RlTrainerDriver<Vessel> {
    let policy = real_policy();
    let watched = policy.policy.actor.parameters()["layer_out.0.weight"].clone();
    let runner = TrainingVesselRunner::with_policy(
        Box::new(policy),
        Box::new(CandleCollectorFactory),
        TrainingVesselBinding::default(),
        TrainingVesselRunConfig {
            buffer_size: 7,
            episode_per_iter: 3,
            update_kwargs: IndexMap::from([
                // Keep both reward trajectories in each normalized-advantage
                // minibatch: identical two-item advantages have zero source std.
                ("batch_size".into(), json!(5)),
                ("repeat".into(), json!(2)),
            ]),
        },
        TrainingVesselLog::default(),
    );
    let mut driver = RlTrainerDriver::new(
        Vessel {
            runner,
            logger: logger.clone(),
            watched,
            attachments: 0,
            updates: 0,
        },
        Arc::new(RlTrainerRuntime::new(None)),
        RlTrainerConfig {
            max_iters: Some(2.into()),
            val_every_n_iters: None,
        },
    );
    driver.callbacks.push(Box::new(Observer {
        counter: counter.clone(),
        starts: starts.clone(),
    }));
    let mut config = RlCheckpointConfig::new(path);
    config.every_n_iters = Some(1.into());
    driver.callbacks.push(Box::new(RlCheckpointCallback::new(
        config,
        SystemRlCheckpointClock,
        PythonRlCheckpointName,
        graph(counter, logger),
        FileRlCheckpointStorage::new(BincodeRlCheckpointCodec),
    )));
    driver
}

#[test]
fn actual_gru_learning_restores_native_trainer_graph_before_fit_start() {
    let directory = tempfile::tempdir().unwrap();
    let counter = Arc::new(Mutex::new(0));
    let logger = Arc::new(Mutex::new(Logger::new(20, NoopRlLogWriterHooks)));
    let starts = Arc::new(Mutex::new(Vec::new()));
    let mut driver = driver(directory.path(), &counter, &logger, &starts);
    let watched_id = driver.vessel.watched.id();
    driver.fit(None).unwrap();
    let mut storage = FileRlCheckpointStorage::new(BincodeRlCheckpointCodec);
    let first_path = directory.path().join("001.pth");
    let original_bytes = std::fs::read(&first_path).unwrap();
    let first: Document = storage.load(&first_path).unwrap();
    let second: Document = storage.load(&directory.path().join("002.pth")).unwrap();
    assert_eq!(
        first.current_iter.require("iter").unwrap(),
        &BigInt::from(1)
    );
    assert_eq!(
        second.current_iter.require("iter").unwrap(),
        &BigInt::from(2)
    );
    assert_ne!(
        snapshot_value(first.vessel.require("vessel").unwrap()),
        snapshot_value(second.vessel.require("vessel").unwrap())
    );
    assert_eq!(
        first.callbacks.require("callbacks").unwrap()["observer"],
        Some(1)
    );
    let saved_log = &first.loggers.require("loggers").unwrap()["logwriter"];
    assert_eq!(saved_log.global_episode, 3.into());
    assert_eq!(saved_log.global_step, 5.into());
    let latest: Document = storage.load(&directory.path().join("latest.pth")).unwrap();
    assert_eq!(latest, second);
    assert_eq!(driver.vessel.updates, 2);
    let expected = Start {
        snapshot: snapshot_value(first.vessel.require("vessel").unwrap()),
        counter: 1,
        logger: saved_log.clone(),
        iteration: 1.into(),
    };
    *counter.lock().unwrap() = 999;
    logger
        .lock()
        .unwrap()
        .load_state_dict(LoggerState::default());
    let tensor = &driver.vessel.watched;
    Var::from_tensor(tensor)
        .unwrap()
        .set(&tensor.zeros_like().unwrap())
        .unwrap();
    let mut graph = graph(&counter, &logger);
    let mut restore = RlCheckpointFileRestore {
        path: first_path.clone(),
        storage: &mut storage,
        graph: &mut graph,
    };
    driver.fit(Some(&mut restore)).unwrap();
    assert_eq!(starts.lock().unwrap().len(), 2);
    assert_eq!(starts.lock().unwrap()[1], expected);
    assert_eq!(driver.vessel.attachments, 2);
    assert_eq!(driver.vessel.updates, 3);
    assert_eq!(driver.vessel.watched.id(), watched_id);
    assert_eq!(*counter.lock().unwrap(), 2);
    assert_eq!(logger.lock().unwrap().state_dict().global_episode, 6.into());
    assert_eq!(logger.lock().unwrap().state_dict().global_step, 10.into());
    assert_eq!(std::fs::read(first_path).unwrap(), original_bytes);
    assert_ne!(
        snapshot_value(&driver.vessel.save_checkpoint().unwrap()),
        expected.snapshot
    );
    // Restoring source policy state deliberately retains Adam/RNG runtime state;
    // resumed training need not be bit-identical to an uninterrupted fresh run.
}
