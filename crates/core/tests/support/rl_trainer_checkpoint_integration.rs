use super::*;
use crate::*;
use std::sync::Mutex;

type Buffer = RlLogBuffer<String, RlTrainerBufferCallback<String>>;
type BufferState = RlLogBufferCheckpoint<String>;
type VesselState = TrainingVesselCheckpoint<Vec<i64>>;

struct Policy {
    weights: Arc<Mutex<Vec<i64>>>,
    fail: bool,
}
impl TrainingPolicyState<Vec<i64>> for Policy {
    fn state_dict(&mut self) -> Result<Vec<i64>, String> {
        if self.fail {
            Err("save injected".into())
        } else {
            Ok(self.weights.lock().unwrap().clone())
        }
    }
    fn load_state_dict(&mut self, value: &Vec<i64>) -> Result<(), String> {
        if self.fail {
            Err("load injected".into())
        } else {
            self.weights.lock().unwrap().clone_from(value);
            Ok(())
        }
    }
}
struct Counter(Arc<Mutex<i64>>);
impl RlCheckpointState<i64> for Counter {
    fn save_checkpoint(&mut self) -> Result<i64, String> {
        Ok(*self.0.lock().unwrap())
    }
    fn load_checkpoint(&mut self, value: &i64) -> Result<(), String> {
        *self.0.lock().unwrap() = *value;
        Ok(())
    }
}
struct Logger(Arc<Mutex<Buffer>>);
impl RlCheckpointState<BufferState> for Logger {
    fn save_checkpoint(&mut self) -> Result<BufferState, String> {
        self.0.lock().unwrap().save_checkpoint()
    }
    fn load_checkpoint(&mut self, value: &BufferState) -> Result<(), String> {
        self.0.lock().unwrap().load_checkpoint(value)
    }
}
struct Seed;
impl RlTrainerSeedContext for Seed {}
struct Vessel {
    state: TrainingVesselState<Vec<i64>>,
    weights: Arc<Mutex<Vec<i64>>>,
    attached: usize,
}
impl RlCheckpointState<VesselState> for Vessel {
    fn save_checkpoint(&mut self) -> Result<VesselState, String> {
        self.state.save_checkpoint()
    }
    fn load_checkpoint(&mut self, value: &VesselState) -> Result<(), String> {
        self.state.load_checkpoint(value)
    }
}
impl RlTrainerVessel for Vessel {
    type Seed = Seed;
    type Environment = ();
    fn assign_trainer(&mut self, _runtime: &Arc<Runtime>) -> Result<(), RlTrainerDriverError> {
        self.attached += 1;
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
        _env: &mut (),
        _control: &mut RlTrainerControl,
    ) -> Result<(), RlTrainerDriverError> {
        self.weights.lock().unwrap()[0] += 1;
        Ok(())
    }
}
struct Callback {
    counter: Arc<Mutex<i64>>,
    logger: Arc<Mutex<Buffer>>,
    seen: Arc<Mutex<Vec<Value>>>,
}
impl RlTrainerCallback<Vessel> for Callback {
    fn call(
        &mut self,
        hook: RlTrainerHook,
        control: &mut RlTrainerControl,
        vessel: &mut Vessel,
    ) -> Result<(), RlTrainerDriverError> {
        if hook == RlTrainerHook::FitStart {
            self.seen.lock().unwrap().push(json!({"counter":*self.counter.lock().unwrap(),"weights":*vessel.weights.lock().unwrap(),
                "iteration":control.runtime.read(|s|s.current_iter.as_ref().unwrap().to_string()).unwrap(),
                "global_episode":self.logger.lock().unwrap().state_dict().global_episode.to_string()}));
        }
        if hook == RlTrainerHook::IterEnd {
            *self.counter.lock().unwrap() += 1;
        }
        Ok(())
    }
}

#[test]
fn graph_restoration_reaches_live_driver_callback_vessel_and_log_buffer_before_fit_start() {
    let runtime = Arc::new(runtime());
    runtime
        .update(|s| s.current_iter = Some(12.into()))
        .unwrap();
    let weights = Arc::new(Mutex::new(vec![42]));
    let vessel = Vessel {
        state: TrainingVesselState::new(Box::new(Policy {
            weights: weights.clone(),
            fail: false,
        })),
        weights: weights.clone(),
        attached: 0,
    };
    let buffer = Arc::new(Mutex::new(Buffer::new_buffer(
        20,
        runtime.buffer_callback(),
    )));
    let logs = IndexMap::from([(
        "reward".into(),
        RlLogEntry {
            level: 20,
            value: RlLogValue::Float(2.0),
        },
    )]);
    buffer.lock().unwrap().on_env_reset(0);
    buffer
        .lock()
        .unwrap()
        .on_env_step(0, 2.0, true, Some(&logs))
        .unwrap();
    let counter = Arc::new(Mutex::new(7));
    let seen = Arc::new(Mutex::new(vec![]));
    let mut driver = RlTrainerDriver::new(
        vessel,
        runtime.clone(),
        RlTrainerConfig {
            max_iters: Some(13.into()),
            val_every_n_iters: None,
        },
    );
    driver.callbacks.push(Box::new(Callback {
        counter: counter.clone(),
        logger: buffer.clone(),
        seen: seen.clone(),
    }));
    let mut callback = Counter(counter.clone());
    let mut logger = Logger(buffer.clone());
    let mut callbacks = [RlNamedCheckpointComponent {
        type_name: "Callback".into(),
        state: &mut callback as &mut dyn RlCheckpointState<i64>,
    }];
    let mut loggers = [RlNamedCheckpointComponent {
        type_name: "LogBuffer".into(),
        state: &mut logger as &mut dyn RlCheckpointState<BufferState>,
    }];
    let mut doc =
        save_rl_trainer_checkpoint(&runtime, &mut driver.vessel, &mut callbacks, &mut loggers)
            .unwrap();
    let saved_buffer = doc.loggers.require("loggers").unwrap()["logbuffer"].clone();
    assert_buffer_round_trip(&saved_buffer);
    runtime.update(crate::RlTrainerState::initialize).unwrap();
    *weights.lock().unwrap() = vec![99];
    *counter.lock().unwrap() = 100;
    buffer
        .lock()
        .unwrap()
        .load_buffer_state(RlLogBufferState::default(), RlLogWriterState::default());
    let mut restore = RlTrainerCheckpointRestore {
        checkpoint: &doc,
        callbacks: &mut callbacks,
        loggers: &mut loggers,
    };
    driver.fit(Some(&mut restore)).unwrap();
    assert_eq!(
        *seen.lock().unwrap(),
        vec![json!({"counter":7,"weights":[42],"iteration":"12","global_episode":"1"})]
    );
    assert_eq!(*weights.lock().unwrap(), vec![43]);
    assert_eq!(*counter.lock().unwrap(), 8);
    assert_eq!(driver.vessel.attached, 1);
    assert_eq!(
        runtime.read(|s| s.current_iter.clone()).unwrap(),
        Some(13.into())
    );
    let restored: BufferState = buffer.lock().unwrap().save_checkpoint().unwrap();
    assert_eq!(restored, saved_buffer);
    doc.vessel = RlCheckpointField::Missing;
    let mut restore = RlTrainerCheckpointRestore {
        checkpoint: &doc,
        callbacks: &mut callbacks,
        loggers: &mut loggers,
    };
    let error = driver.fit(Some(&mut restore)).unwrap_err();
    assert!(
        matches!(error,RlTrainerDriverError::Plugin {ref stage,..} if stage=="checkpoint_restore")
    );
    assert!(error.to_string().contains("vessel"));
    assert_eq!(driver.vessel.attached, 2);
    assert_eq!(seen.lock().unwrap().len(), 1);
}

fn assert_buffer_round_trip(state: &BufferState) {
    assert_eq!(state, &state.clone());
    assert!(format!("{state:?}").contains("writer"));
    assert_eq!(
        &bincode::deserialize::<BufferState>(&bincode::serialize(state).unwrap()).unwrap(),
        state
    );
    assert_eq!(
        &serde_json::from_value::<BufferState>(serde_json::to_value(state).unwrap()).unwrap(),
        state
    );
}

#[test]
fn concrete_writer_and_vessel_adapters_preserve_payloads_and_report_errors() {
    let weights = Arc::new(Mutex::new(vec![3]));
    let mut broken = TrainingVesselState::new(Box::new(Policy {
        weights,
        fail: true,
    }));
    assert!(
        broken
            .save_checkpoint()
            .unwrap_err()
            .contains("save injected")
    );
    assert!(
        broken
            .load_checkpoint(&VesselState { policy: vec![] })
            .unwrap_err()
            .contains("load injected")
    );
    let mut writer = RlLogWriter::<String>::new(20, NoopRlLogWriterHooks);
    writer.on_env_reset(1);
    writer
        .on_env_step(1, 2.0, true, Some(&IndexMap::new()))
        .unwrap();
    let checkpoint: RlLogWriterState<String> = writer.save_checkpoint().unwrap();
    writer.load_state_dict(RlLogWriterState::default());
    writer.load_checkpoint(&checkpoint).unwrap();
    assert_eq!(writer.state_dict(), &checkpoint);
}
