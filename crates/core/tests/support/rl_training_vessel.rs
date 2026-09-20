//! Exercise the production default vessel through real queue/runner/finite lifecycles.
use super::*;

type DefaultVessel = RlTrainingVessel<i64, i64, i64, f64, Info, Vec<i64>, Value, StoredPolicy>;
type DefaultDriver = RlTrainerDriver<DefaultVessel>;

#[derive(Default)]
struct Audit {
    updates: Arc<AtomicUsize>,
    policy_failure: Mutex<Option<&'static str>>,
    factory_failure: Mutex<bool>,
    queues: Mutex<Vec<Queue>>,
    live: Arc<AtomicUsize>,
    modes: Mutex<Vec<TrainingPolicyMode>>,
}
struct StoredPolicy(Arc<Audit>);
impl TrainingRunPolicy<Vec<i64>> for StoredPolicy {
    fn set_mode(&mut self, mode: TrainingPolicyMode) -> Result<(), TrainingVesselRunError> {
        self.0.modes.lock().unwrap().push(mode);
        if *self.0.policy_failure.lock().unwrap() == Some("mode") {
            Err(TrainingVesselRunError::Plugin {
                stage: "mode".into(),
                message: "mode failed".into(),
            })
        } else {
            Ok(())
        }
    }
    fn update(
        &mut self,
        size: u64,
        buffer: Option<&mut Vec<i64>>,
        options: &TrainingUpdateOptions,
    ) -> Result<TrainingVesselMetrics, TrainingVesselRunError> {
        Policy(self.0.updates.clone()).update(size, buffer, options)
    }
}
impl TrainingPolicyState<usize> for StoredPolicy {
    fn state_dict(&mut self) -> Result<usize, String> {
        if *self.0.policy_failure.lock().unwrap() == Some("save") {
            Err("save".into())
        } else {
            Ok(self.0.updates.load(Ordering::SeqCst))
        }
    }
    fn load_state_dict(&mut self, value: &usize) -> Result<(), String> {
        if *self.0.policy_failure.lock().unwrap() == Some("load") {
            return Err("load".into());
        }
        self.0.updates.store(*value, Ordering::SeqCst);
        Ok(())
    }
}

fn build() -> (DefaultDriver, Arc<Audit>) {
    let runtime = Arc::new(Runtime::new(Some(2)));
    let audit = Arc::new(Audit::default());
    let buffer = Arc::new(Mutex::new(LogBuffer::new_buffer(
        20,
        runtime.buffer_callback(),
    )));
    let runner = TrainingVesselRunner::with_policy(
        Box::new(StoredPolicy(audit.clone())),
        Box::new(Factory),
        TrainingVesselBinding::default(),
        TrainingVesselRunConfig::default(),
        TrainingVesselLog::default(),
    );
    let factory_audit = audit.clone();
    let factory = move |queue: &Queue, _: &mut RlTrainerControl| {
        assert!(queue.lock().unwrap().is_activated());
        factory_audit.queues.lock().unwrap().push(queue.clone());
        if *factory_audit.factory_failure.lock().unwrap() {
            return Err(RlTrainerDriverError::Plugin {
                stage: "environment_factory".into(),
                message: "construction failed".into(),
            });
        }
        factory_audit.live.fetch_add(1, Ordering::SeqCst);
        let backend = FiniteDummyBackend::new(vec![Box::new(Worker {
            queue: queue.clone(),
            live: factory_audit.live.clone(),
        })])
        .map_err(plugin)?;
        Ok(Environment::new(
            Box::new(backend),
            Box::new(Predicate),
            vec![Box::new(BufferLogger(buffer.clone()))],
        ))
    };
    let vessel = RlTrainingVessel::new(
        runner,
        TrainingVesselSeeds::new(
            Some(Arc::new(vec![7, 9])),
            Some(Arc::new(vec![11, 13])),
            Some(Arc::new(vec![17, 19])),
            None,
        ),
        Box::new(factory),
    );
    (
        RlTrainerDriver::new(
            vessel,
            runtime,
            RlTrainerConfig {
                max_iters: Some(2.into()),
                val_every_n_iters: Some(1.into()),
            },
        ),
        audit,
    )
}

struct ChangeSettings;
impl RlTrainerCallback<DefaultVessel> for ChangeSettings {
    fn call(
        &mut self,
        hook: RlTrainerHook,
        control: &mut RlTrainerControl,
        _: &mut DefaultVessel,
    ) -> Result<(), RlTrainerDriverError> {
        match hook {
            RlTrainerHook::TrainStart => control.runtime.set_fast_dev_run(Some(2))?,
            RlTrainerHook::ValidateStart | RlTrainerHook::TestStart => {
                control.runtime.set_fast_dev_run(Some(1))?;
            }
            _ => {}
        }
        Ok(())
    }
}

fn assert_cleaned(audit: &Audit, count: usize) {
    assert_eq!(audit.live.load(Ordering::SeqCst), 0);
    let queues = audit.queues.lock().unwrap();
    assert_eq!(queues.len(), count);
    for queue in &*queues {
        assert_eq!(queue.lock().unwrap().get(), Err(DataQueueError::Exhausted));
    }
}

#[test]
fn default_vessel_runs_all_phases_and_restores_the_owned_policy() {
    let (mut driver, audit) = build();
    driver.callbacks.push(Box::new(ChangeSettings));
    driver.fit(None).unwrap();
    assert_eq!(audit.updates.load(Ordering::SeqCst), 2);
    driver
        .control
        .runtime
        .read(|state| {
            assert_eq!(state.current_iter, Some(2.into()));
            assert_eq!(state.current_episode, Some(6.into()));
            assert_eq!(
                state.metrics.as_ref().unwrap(),
                &IndexMap::from([("reward".into(), 3.0), ("val/reward".into(), 3.0)])
            );
        })
        .unwrap();
    driver.test().unwrap();
    driver
        .control
        .runtime
        .read(|state| {
            assert_eq!(state.current_episode, Some(7.into()));
            assert_eq!(state.current_stage, "test");
            assert_eq!(
                state.metrics.as_ref().unwrap(),
                &IndexMap::from([("reward".into(), 3.0)])
            );
        })
        .unwrap();
    assert_eq!(
        *audit.modes.lock().unwrap(),
        [
            TrainingPolicyMode::Train,
            TrainingPolicyMode::Evaluation,
            TrainingPolicyMode::Train,
            TrainingPolicyMode::Evaluation,
            TrainingPolicyMode::Evaluation
        ]
    );
    assert_cleaned(&audit, 5);
    let state = driver.vessel.save_checkpoint().unwrap();
    assert_eq!(state, TrainingVesselCheckpoint { policy: 2 });
    audit.updates.store(99, Ordering::SeqCst);
    driver.vessel.load_checkpoint(&state).unwrap();
    assert_eq!(audit.updates.load(Ordering::SeqCst), 2);
    driver.fit(None).unwrap();
    assert_eq!(audit.updates.load(Ordering::SeqCst), 4);
    assert_cleaned(&audit, 9);
}

#[test]
fn default_vessel_does_not_wrap_factory_errors_or_skip_context_cleanup() {
    let (mut driver, audit) = build();
    *audit.factory_failure.lock().unwrap() = true;
    assert_eq!(
        driver.fit(None).unwrap_err(),
        RlTrainerDriverError::Plugin {
            stage: "environment_factory".into(),
            message: "construction failed".into(),
        }
    );
    assert_cleaned(&audit, 1);
    assert_eq!(audit.updates.load(Ordering::SeqCst), 0);
    assert!(audit.modes.lock().unwrap().is_empty());
}

#[test]
fn default_vessel_seed_failures_precede_factory_and_assignment_adds_no_strong_reference() {
    let (mut driver, audit) = build();
    let error = driver.vessel.seeds(RlTrainerPhase::Train).err().unwrap();
    assert_eq!(
        error,
        RlTrainerDriverError::Plugin {
            stage: "seed_factory".into(),
            message: TrainingVesselSeedError::Trainer {
                phase: TrainingSeedPhase::Train,
                source: TrainingVesselBindingError::Unassigned
            }
            .to_string()
        }
    );
    let references = Arc::strong_count(&driver.control.runtime);
    driver
        .vessel
        .assign_trainer(&driver.control.runtime)
        .unwrap();
    assert_eq!(Arc::strong_count(&driver.control.runtime), references);
    driver.vessel.seeds = TrainingVesselSeeds::new(None, None, None, None);
    for (phase, seed_phase) in [
        (RlTrainerPhase::Train, TrainingSeedPhase::Train),
        (RlTrainerPhase::Validation, TrainingSeedPhase::Val),
        (RlTrainerPhase::Test, TrainingSeedPhase::Test),
    ] {
        assert_eq!(
            driver.vessel.seeds(phase).err().unwrap(),
            RlTrainerDriverError::Plugin {
                stage: "seed_factory".into(),
                message: TrainingVesselSeedError::SeedIteratorNotAvailable { phase: seed_phase }
                    .to_string(),
            }
        );
    }
    assert_cleaned(&audit, 0);
}

#[test]
fn default_vessel_retains_run_stage_and_policy_checkpoint_failures() {
    let (mut driver, audit) = build();
    driver
        .vessel
        .assign_trainer(&driver.control.runtime)
        .unwrap();
    driver
        .control
        .runtime
        .update(RlTrainerState::initialize)
        .unwrap();
    *audit.policy_failure.lock().unwrap() = Some("mode");
    for (phase, stage) in [
        (RlTrainerPhase::Train, "train"),
        (RlTrainerPhase::Validation, "validate"),
        (RlTrainerPhase::Test, "test"),
    ] {
        let mut queue = driver.vessel.seeds(phase).unwrap();
        queue.enter().unwrap();
        let mut env = driver
            .vessel
            .environment(&mut queue, &mut driver.control)
            .unwrap();
        let error = driver
            .vessel
            .run(phase, &mut env, &mut driver.control)
            .unwrap_err();
        assert_eq!(
            error,
            RlTrainerDriverError::Plugin {
                stage: stage.into(),
                message: TrainingVesselRunError::Plugin {
                    stage: "mode".into(),
                    message: "mode failed".into()
                }
                .to_string()
            }
        );
        drop(env);
        assert!(!queue.exit(Some(&error)).unwrap());
    }
    assert_cleaned(&audit, 3);
    *audit.policy_failure.lock().unwrap() = Some("save");
    assert_eq!(
        driver.vessel.save_checkpoint().unwrap_err(),
        TrainingVesselStateError::Save("save".into()).to_string()
    );
    *audit.policy_failure.lock().unwrap() = Some("load");
    assert_eq!(
        driver
            .vessel
            .load_checkpoint(&TrainingVesselCheckpoint { policy: 88 })
            .unwrap_err(),
        TrainingVesselStateError::Load("load".into()).to_string()
    );
    assert_eq!(audit.updates.load(Ordering::SeqCst), 0);
}
