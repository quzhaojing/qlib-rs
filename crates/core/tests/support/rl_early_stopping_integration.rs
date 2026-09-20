use super::*;
use crate::{
    RlTrainerConfig, RlTrainerDriver, RlTrainerPhase, RlTrainerSeedContext, RlTrainerVessel,
    TrainingPolicyState, TrainingVesselCheckpoint, TrainingVesselState,
};
use std::sync::Mutex;

type PolicyState = TrainingVesselCheckpoint<Vec<f64>>;
type Early = RlEarlyStopping<PolicyState, BincodeRlCheckpointSnapshot, Events>;
struct Policy(Arc<Mutex<Vec<f64>>>);
impl TrainingPolicyState<Vec<f64>> for Policy {
    fn state_dict(&mut self) -> Result<Vec<f64>, String> {
        Ok(self.0.lock().unwrap().clone())
    }
    fn load_state_dict(&mut self, state: &Vec<f64>) -> Result<(), String> {
        self.0.lock().unwrap().clone_from(state);
        Ok(())
    }
}
struct Seeds;
impl RlTrainerSeedContext for Seeds {}
struct Training {
    state: TrainingVesselState<Vec<f64>>,
    weights: Arc<Mutex<Vec<f64>>>,
    runs: usize,
}
impl RlCheckpointState<PolicyState> for Training {
    fn save_checkpoint(&mut self) -> Result<PolicyState, String> {
        self.state.save_checkpoint()
    }
    fn load_checkpoint(&mut self, state: &PolicyState) -> Result<(), String> {
        self.state.load_checkpoint(state)
    }
}
impl RlTrainerVessel for Training {
    type Seed = Seeds;
    type Environment = ();
    fn assign_trainer(&mut self, _: &Arc<RlTrainerRuntime>) -> Result<(), RlTrainerDriverError> {
        Ok(())
    }
    fn seeds(&mut self, _: RlTrainerPhase) -> Result<Seeds, RlTrainerDriverError> {
        Ok(Seeds)
    }
    fn environment(
        &mut self,
        _: &mut Seeds,
        _: &mut RlTrainerControl,
    ) -> Result<(), RlTrainerDriverError> {
        Ok(())
    }
    fn run(
        &mut self,
        phase: RlTrainerPhase,
        (): &mut (),
        control: &mut RlTrainerControl,
    ) -> Result<(), RlTrainerDriverError> {
        if phase == RlTrainerPhase::Train {
            self.runs += 1;
            self.weights.lock().unwrap()[0] += 1.0;
        } else {
            control
                .runtime
                .update(|s| s.metrics = Some(IndexMap::from([("val/reward".into(), 1.0)])))?;
        }
        Ok(())
    }
}
struct SharedEarly(Rc<RefCell<Early>>);
impl RlTrainerCallback<Training> for SharedEarly {
    fn call(
        &mut self,
        hook: RlTrainerHook,
        control: &mut RlTrainerControl,
        vessel: &mut Training,
    ) -> Result<(), RlTrainerDriverError> {
        self.0.borrow_mut().call(hook, control, vessel)
    }
}

struct Restore<'a>(
    &'a mut crate::RlTrainerCheckpoint<PolicyState, RlEarlyStoppingState<PolicyState>, ()>,
    Rc<RefCell<Early>>,
);
impl crate::RlTrainerRestore<Training> for Restore<'_> {
    fn restore(
        &mut self,
        control: &mut RlTrainerControl,
        vessel: &mut Training,
    ) -> Result<(), RlTrainerDriverError> {
        let mut early = self.1.borrow_mut();
        crate::load_rl_trainer_checkpoint(
            &control.runtime,
            vessel,
            &mut [crate::RlNamedCheckpointComponent {
                type_name: "EarlyStopping".into(),
                state: &mut *early,
            }],
            &mut [],
            self.0,
        )
        .map_err(|e| RlTrainerDriverError::Plugin {
            stage: "restore".into(),
            message: e.to_string(),
        })
    }
}

#[test]
fn real_driver_stops_restores_best_policy_and_resets_callback_after_graph_resume() {
    let weights = Arc::new(Mutex::new(vec![0.0]));
    let training = Training {
        state: TrainingVesselState::new(Box::new(Policy(weights.clone()))),
        weights: weights.clone(),
        runs: 0,
    };
    let runtime = Arc::new(RlTrainerRuntime::new(None));
    let mut driver = RlTrainerDriver::new(
        training,
        runtime.clone(),
        RlTrainerConfig {
            max_iters: Some(10.into()),
            val_every_n_iters: Some(1.into()),
        },
    );
    let callback = Rc::new(RefCell::new(
        Early::new(
            RlEarlyStoppingConfig {
                monitor: "val/reward".into(),
                restore_best_weights: true,
                ..RlEarlyStoppingConfig::default()
            },
            BincodeRlCheckpointSnapshot,
            Events::default(),
        )
        .unwrap(),
    ));
    driver
        .callbacks
        .push(Box::new(SharedEarly(callback.clone())));
    driver.fit(None).unwrap();
    assert_eq!(driver.vessel.runs, 2);
    assert_eq!(
        runtime.read(|s| s.current_iter.clone()).unwrap(),
        Some(2.into())
    );
    assert_eq!(*weights.lock().unwrap(), vec![1.0]);
    assert_eq!(callback.borrow().state.best_iter, Present(0.into()));
    assert_eq!(callback.borrow().state.wait, Present(1.into()));
    let mut graph = {
        let mut callback = callback.borrow_mut();
        crate::save_rl_trainer_checkpoint::<_, _, (), _>(
            &runtime,
            &mut driver.vessel,
            &mut [crate::RlNamedCheckpointComponent {
                type_name: "EarlyStopping".into(),
                state: &mut *callback,
            }],
            &mut [],
        )
        .unwrap()
    };
    let decoded: crate::RlTrainerCheckpoint<PolicyState, RlEarlyStoppingState<PolicyState>, ()> =
        bincode::deserialize(&bincode::serialize(&graph).unwrap()).unwrap();
    assert_eq!(graph, decoded);
    // A changed live model must not change the callback's detached best snapshot.
    weights.lock().unwrap()[0] = 99.0;
    assert_eq!(
        callback
            .borrow()
            .state
            .best_weights
            .require("best_weights")
            .unwrap()
            .as_ref()
            .unwrap()
            .policy,
        vec![1.0]
    );
    // Resume an already stopped graph: fit-start still resets EarlyStopping even though
    // the driver then skips the training loop. This is the source's non-resuming wait rule.
    callback.borrow_mut().state.wait = Present(999.into());
    driver
        .fit(Some(&mut Restore(&mut graph, callback.clone())))
        .unwrap();
    assert_eq!(driver.vessel.runs, 2);
    assert_eq!(*weights.lock().unwrap(), vec![1.0]);
    assert_eq!(callback.borrow().state.wait, Present(0.into()));
    assert_eq!(callback.borrow().state.best_weights, Present(None));
    assert_eq!(callback.borrow().state.best, Present(f64::NEG_INFINITY));
    driver.test().unwrap();
    assert_eq!(callback.borrow().state.wait, Present(0.into()));
    // Dispatch propagates callback errors with stage information.
    runtime.update(|s| s.metrics = None).unwrap();
    assert!(
        matches!(callback.borrow_mut().call(RlTrainerHook::ValidateEnd, &mut driver.control, &mut driver.vessel), Err(RlTrainerDriverError::Plugin { stage, .. }) if stage == "early_stopping")
    );
}

struct Capture(Arc<Mutex<Vec<(String, String)>>>);
impl tracing::Subscriber for Capture {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        event.record(
            &mut |_: &tracing::field::Field, value: &dyn std::fmt::Debug| {
                self.0
                    .lock()
                    .unwrap()
                    .push((event.metadata().level().to_string(), format!("{value:?}")));
            },
        );
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}
#[test]
fn default_logger_delivers_info_and_warning_and_allows_disabled_logging() {
    let values = Arc::new(Mutex::new(vec![]));
    {
        let _guard = tracing::subscriber::set_default(Capture(values.clone()));
        TracingRlEarlyStoppingLogger
            .log(RlEarlyStoppingLogLevel::Info, "progress")
            .unwrap();
        TracingRlEarlyStoppingLogger
            .log(RlEarlyStoppingLogLevel::Warning, "missing")
            .unwrap();
    }
    assert_eq!(
        *values.lock().unwrap(),
        vec![
            ("INFO".into(), "progress".into()),
            ("WARN".into(), "missing".into())
        ]
    );
    let _guard = tracing::subscriber::set_default(tracing::subscriber::NoSubscriber::default());
    TracingRlEarlyStoppingLogger
        .log(RlEarlyStoppingLogLevel::Info, "disabled")
        .unwrap();
    TracingRlEarlyStoppingLogger
        .log(RlEarlyStoppingLogLevel::Warning, "disabled")
        .unwrap();
    assert_eq!(values.lock().unwrap().len(), 2);
}
