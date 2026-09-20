use super::*;
use crate::rl_candle_checkpoint::{CandlePolicySnapshot, CandlePolicyState};
use crate::rl_policy_checkpoint::{
    PolicyCheckpointFile, PolicyCheckpointReader, TrainerPolicyCheckpoint,
};
use crate::rl_policy_weight::set_policy_weights;
use crate::{rl_checkpoint_callback::*, *};
use candle_core::{Device, Tensor, Var};
use indexmap::IndexMap;
use num_bigint::BigInt;
use serde_json::{Value, json};
use std::{
    cell::RefCell,
    rc::Rc,
    sync::{Arc, Mutex},
};

type Logger = RlLogWriter<String>;
type LoggerState = RlLogWriterState<String>;
type PolicySnapshot = CandlePolicySnapshot<u64>;
type VesselSnapshot = TrainingVesselCheckpoint<PolicySnapshot>;
type Graph = RlTrainerGraph<VesselSnapshot, Option<i64>, LoggerState>;
type Document = RlTrainerCheckpoint<VesselSnapshot, Option<i64>, LoggerState>;
type Saver = RlCheckpointCallback<
    SystemRlCheckpointClock,
    PythonRlCheckpointName,
    Graph,
    FileRlCheckpointStorage<BincodeRlCheckpointCodec>,
>;

struct CounterState(Arc<Mutex<i64>>);
impl RlCheckpointState<Option<i64>> for CounterState {
    fn save_checkpoint(&mut self) -> Result<Option<i64>, String> {
        Ok(Some(*self.0.lock().unwrap()))
    }
    fn load_checkpoint(&mut self, value: &Option<i64>) -> Result<(), String> {
        *self.0.lock().unwrap() = value.ok_or("counter state absent")?;
        Ok(())
    }
}
// Checkpoint inherits no-op state methods, so representing its state never locks the
// executing callback recursively. This must not be used for stateful callbacks.
struct UnitState;
impl RlCheckpointState<Option<i64>> for UnitState {
    fn save_checkpoint(&mut self) -> Result<Option<i64>, String> {
        Ok(None)
    }
    fn load_checkpoint(&mut self, _value: &Option<i64>) -> Result<(), String> {
        Ok(())
    }
}
struct LoggerHandle(Arc<Mutex<Logger>>);
impl RlCheckpointState<LoggerState> for LoggerHandle {
    fn save_checkpoint(&mut self) -> Result<LoggerState, String> {
        self.0.lock().unwrap().save_checkpoint()
    }
    fn load_checkpoint(&mut self, value: &LoggerState) -> Result<(), String> {
        self.0.lock().unwrap().load_checkpoint(value)
    }
}
struct Seed;
impl RlTrainerSeedContext for Seed {}
struct Vessel {
    weight: Var,
    shared_weight: Var,
    state: TrainingVesselState<PolicySnapshot>,
    attachments: usize,
    fail_save: bool,
}
impl Vessel {
    fn new() -> Self {
        let weight = Var::new(&[10_f32], &Device::Cpu).unwrap();
        let policy = CandlePolicyState::new(
            IndexMap::from([
                ("__metadata__".into(), weight.clone()),
                ("shared.weight".into(), weight.clone()),
            ]),
            7_u64,
        );
        Self {
            shared_weight: weight.clone(),
            weight,
            state: TrainingVesselState::new(Box::new(policy)),
            attachments: 0,
            fail_save: false,
        }
    }
    fn weights(&self) -> Vec<f32> {
        self.weight.to_vec1().unwrap()
    }
    fn set_weight(&self, value: f32) {
        self.weight
            .set(&Tensor::new(&[value], &Device::Cpu).unwrap())
            .unwrap();
    }
    fn forward(&self) -> f32 {
        self.weight
            .mul(&self.shared_weight)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap()
    }
}
impl RlCheckpointState<VesselSnapshot> for Vessel {
    fn save_checkpoint(&mut self) -> Result<VesselSnapshot, String> {
        if self.fail_save {
            Err("vessel save failed".into())
        } else {
            self.state.state_dict().map_err(|error| error.to_string())
        }
    }
    fn load_checkpoint(&mut self, value: &VesselSnapshot) -> Result<(), String> {
        self.state
            .load_state_dict(value)
            .map_err(|error| error.to_string())
    }
}
impl RlTrainerVessel for Vessel {
    type Seed = Seed;
    type Environment = ();
    fn assign_trainer(
        &mut self,
        _runtime: &Arc<RlTrainerRuntime>,
    ) -> Result<(), RlTrainerDriverError> {
        self.attachments += 1;
        Ok(())
    }
    fn seeds(&mut self, _phase: RlTrainerPhase) -> Result<Seed, RlTrainerDriverError> {
        Ok(Seed)
    }
    fn environment(
        &mut self,
        _seed: &mut Seed,
        _control: &mut RlTrainerControl,
    ) -> Result<(), RlTrainerDriverError> {
        Ok(())
    }
    fn run(
        &mut self,
        _phase: RlTrainerPhase,
        _environment: &mut (),
        control: &mut RlTrainerControl,
    ) -> Result<(), RlTrainerDriverError> {
        // A deterministic native model update, not a claim of migrated PPO training.
        self.weight
            .set(&self.weight.affine(1., 1.).unwrap())
            .unwrap();
        control
            .runtime
            .update(|s| *s.current_episode.as_mut().unwrap() += 100)?;
        Ok(())
    }
}
struct Observer {
    counter: Arc<Mutex<i64>>,
    logger: Arc<Mutex<Logger>>,
    starts: Arc<Mutex<Vec<Value>>>,
}
impl RlTrainerCallback<Vessel> for Observer {
    fn call(
        &mut self,
        hook: RlTrainerHook,
        control: &mut RlTrainerControl,
        vessel: &mut Vessel,
    ) -> Result<(), RlTrainerDriverError> {
        if hook == RlTrainerHook::FitStart {
            self.starts.lock().unwrap().push(json!({"counter":*self.counter.lock().unwrap(), "weights":vessel.weights(), "forward":vessel.forward(),
                "iteration":control.runtime.read(|s|s.current_iter.as_ref().unwrap().to_string())?,
                "logged_episodes":self.logger.lock().unwrap().state_dict().global_episode.to_string()}));
        }
        if hook == RlTrainerHook::IterEnd {
            *self.counter.lock().unwrap() += 1;
            let mut logger = self.logger.lock().unwrap();
            logger.on_env_reset(0);
            logger
                .on_env_step(0, 1.0, true, Some(&IndexMap::new()))
                .unwrap();
        }
        Ok(())
    }
}
struct SharedSaver(Rc<RefCell<Saver>>);
impl RlTrainerCallback<Vessel> for SharedSaver {
    fn call(
        &mut self,
        hook: RlTrainerHook,
        control: &mut RlTrainerControl,
        vessel: &mut Vessel,
    ) -> Result<(), RlTrainerDriverError> {
        self.0.borrow_mut().call(hook, control, vessel)
    }
}

struct Fixture {
    directory: tempfile::TempDir,
    counter: Arc<Mutex<i64>>,
    logger: Arc<Mutex<Logger>>,
    starts: Arc<Mutex<Vec<Value>>>,
    saver: Rc<RefCell<Saver>>,
    driver: RlTrainerDriver<Vessel>,
}
fn graph(counter: &Arc<Mutex<i64>>, logger: &Arc<Mutex<Logger>>) -> Graph {
    Graph::new(
        vec![
            RlOwnedCheckpointComponent {
                type_name: "Observer".into(),
                state: Box::new(CounterState(counter.clone())),
            },
            RlOwnedCheckpointComponent {
                type_name: "Checkpoint".into(),
                state: Box::new(UnitState),
            },
        ],
        vec![RlOwnedCheckpointComponent {
            type_name: "LogWriter".into(),
            state: Box::new(LoggerHandle(logger.clone())),
        }],
    )
}
impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let counter = Arc::new(Mutex::new(0));
        let logger = Arc::new(Mutex::new(Logger::new(20, NoopRlLogWriterHooks)));
        let starts = Arc::new(Mutex::new(vec![]));
        let mut config = RlCheckpointConfig::new(directory.path().join("checkpoints"));
        config.every_n_iters = Some(1.into());
        let saver = Rc::new(RefCell::new(RlCheckpointCallback::new(
            config,
            SystemRlCheckpointClock,
            PythonRlCheckpointName,
            graph(&counter, &logger),
            storage(),
        )));
        let mut driver = RlTrainerDriver::new(
            Vessel::new(),
            Arc::new(RlTrainerRuntime::new(None)),
            RlTrainerConfig {
                max_iters: Some(2.into()),
                val_every_n_iters: None,
            },
        );
        driver.callbacks.push(Box::new(Observer {
            counter: counter.clone(),
            logger: logger.clone(),
            starts: starts.clone(),
        }));
        driver.callbacks.push(Box::new(SharedSaver(saver.clone())));
        Self {
            directory,
            counter,
            logger,
            starts,
            saver,
            driver,
        }
    }
    fn path(&self, name: &str) -> PathBuf {
        self.directory.path().join("checkpoints").join(name)
    }
    fn corrupt_live_state(&mut self) {
        self.driver.vessel.set_weight(999.);
        *self.counter.lock().unwrap() = 999;
        self.logger
            .lock()
            .unwrap()
            .load_state_dict(LoggerState::default());
    }
}

#[test]
fn real_driver_files_restore_all_live_groups_before_fit_start() {
    let mut f = Fixture::new();
    assert!(!f.path("001.pth").exists());
    f.driver.fit(None).unwrap();
    let first_file_bytes = fs::read(f.path("001.pth")).unwrap();
    let first: Document = storage().load(&f.path("001.pth")).unwrap();
    let second: Document = storage().load(&f.path("002.pth")).unwrap();
    assert_eq!(
        first.current_iter.require("iter").unwrap(),
        &BigInt::from(1)
    );
    assert_eq!(
        first.current_episode.require("episode").unwrap(),
        &BigInt::from(100)
    );
    let first_policy = &first.vessel.require("vessel").unwrap().policy;
    assert_eq!(first_policy.metadata, 7);
    let mut policy_reader = PolicyCheckpointFile::<_, _, Document>::new(
        BincodeRlCheckpointCodec,
        TrainerPolicyCheckpoint,
    );
    let extracted = policy_reader.read_policy(&f.path("001.pth")).unwrap();
    assert_eq!(&extracted, first_policy);
    let mut weights = extracted.clone().into_policy_weights(None).unwrap();
    assert!(Arc::ptr_eq(
        &weights.weights["__metadata__"],
        &weights.weights["shared.weight"]
    ));
    assert_eq!(weights.metadata, 7);
    let materialized = Var::new(&[0_f32], &Device::Cpu).unwrap();
    let mut policy = CandlePolicyState::new(
        IndexMap::from([
            ("__metadata__".into(), materialized.clone()),
            ("shared.weight".into(), materialized.clone()),
        ]),
        99_u64,
    );
    set_policy_weights(&mut policy, &mut weights).unwrap();
    assert_eq!(materialized.to_vec1::<f32>().unwrap(), [11.]);
    assert_eq!(weights.metadata, 7);
    assert_eq!(
        policy_reader.read_policy(&f.path("latest.pth")).unwrap(),
        second.vessel.require("vessel").unwrap().policy
    );
    let mut independent = Vessel::new();
    independent
        .load_checkpoint(&TrainingVesselCheckpoint { policy: extracted })
        .unwrap();
    assert_eq!(independent.weights(), vec![11.]);
    assert_eq!(independent.forward().to_bits(), 121_f32.to_bits());
    assert_eq!(independent.weight.id(), independent.shared_weight.id());
    assert_eq!(
        first.callbacks.require("callbacks").unwrap(),
        &IndexMap::from([("observer".into(), Some(1)), ("checkpoint".into(), None)])
    );
    assert_eq!(
        first.loggers.require("loggers").unwrap()["logwriter"].global_episode,
        1.into()
    );
    assert!(!first.should_stop.require("stop").unwrap());
    assert!(*second.should_stop.require("stop").unwrap());
    assert_eq!(
        fs::read_link(f.path("latest.pth")).unwrap(),
        f.path("002.pth")
    );
    f.corrupt_live_state();
    let mut store = storage();
    let mut restore_graph = graph(&f.counter, &f.logger);
    let mut restore = RlCheckpointFileRestore {
        path: f.path("001.pth"),
        storage: &mut store,
        graph: &mut restore_graph,
    };
    f.driver.fit(Some(&mut restore)).unwrap();
    assert_eq!(
        f.starts.lock().unwrap()[1],
        json!({"counter":1,"weights":[11.0],"forward":121.0,"iteration":"1","logged_episodes":"1"})
    );
    assert_eq!(f.driver.vessel.weights(), vec![12.]);
    assert_eq!(f.driver.vessel.forward().to_bits(), 144_f32.to_bits());
    assert_eq!(
        f.driver.vessel.weight.id(),
        f.driver.vessel.shared_weight.id()
    );
    assert_eq!(fs::read(f.path("001.pth")).unwrap(), first_file_bytes);
    assert_eq!(*f.counter.lock().unwrap(), 2);
    assert_eq!(f.driver.vessel.attachments, 2);
    assert_eq!(f.saver.borrow().state.last_iter, Some(2.into()));
    let before = fs::read(f.path("002.pth")).unwrap();
    f.driver.test().unwrap();
    assert_eq!(fs::read(f.path("002.pth")).unwrap(), before);
}

#[test]
fn graph_errors_precede_file_open_and_decode_errors_precede_restore() {
    let mut f = Fixture::new();
    f.driver.fit(None).unwrap();
    let before = fs::read(f.path("002.pth")).unwrap();
    f.driver.vessel.fail_save = true;
    let error = f
        .saver
        .borrow_mut()
        .save(&mut f.driver.control, &mut f.driver.vessel)
        .unwrap_err();
    assert!(matches!(
        error,
        RlCheckpointCallbackError::Plugin { stage: "graph", .. }
    ));
    assert_eq!(fs::read(f.path("002.pth")).unwrap(), before);
    assert_eq!(f.saver.borrow().state.last_iter, Some(2.into()));
    f.driver.vessel.fail_save = false;
    f.corrupt_live_state();
    let mut store = storage();
    let mut restore_graph = graph(&f.counter, &f.logger);
    let mut restore = RlCheckpointFileRestore {
        path: f.path("absent"),
        storage: &mut store,
        graph: &mut restore_graph,
    };
    assert!(f.driver.fit(Some(&mut restore)).is_err());
    assert_eq!(f.driver.vessel.weights(), vec![999.]);
    assert_eq!(*f.counter.lock().unwrap(), 999);
    assert_eq!(f.starts.lock().unwrap().len(), 1);
    fs::write(&restore.path, [0_u8]).unwrap();
    assert!(f.driver.fit(Some(&mut restore)).is_err());
    assert_eq!(f.driver.vessel.weights(), vec![999.]);
    let mut doc: Document = storage().load(&f.path("001.pth")).unwrap();
    doc.callbacks = RlCheckpointField::Present(IndexMap::new());
    assert!(
        restore_graph
            .restore(&mut f.driver.control, &mut f.driver.vessel, &doc)
            .unwrap_err()
            .contains("callbacks.observer")
    );
    assert_eq!(f.driver.vessel.weights(), vec![11.]);
    assert_eq!(*f.counter.lock().unwrap(), 999);
}
