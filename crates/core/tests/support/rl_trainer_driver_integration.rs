use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;
use crate::*;

#[path = "rl_training_vessel.rs"]
mod default_vessel;

type Info = RlLogInfo<String>;
type Environment = FiniteVectorEnv<i64, i64, f64, Info>;
type Runner = TrainingVesselRunner<i64, i64, f64, Info, Vec<i64>>;
type LogBuffer = RlLogBuffer<String, RlTrainerBufferCallback<String>>;
type Queue = Arc<Mutex<DataQueue<i64>>>;

fn plugin(error: impl std::fmt::Display) -> RlTrainerDriverError {
    RlTrainerDriverError::Plugin {
        stage: "integration".into(),
        message: error.to_string(),
    }
}

struct QueueSeed {
    queue: Queue,
    live: Arc<AtomicUsize>,
}
impl RlTrainerSeedContext for QueueSeed {
    fn enter(&mut self) -> Result<(), RlTrainerDriverError> {
        RlTrainerSeedContext::enter(&mut self.queue)
    }
    fn exit(&mut self, error: Option<&RlTrainerDriverError>) -> Result<bool, RlTrainerDriverError> {
        assert!(error.is_none());
        assert_eq!(
            self.live.load(Ordering::SeqCst),
            0,
            "environment released before queue cleanup"
        );
        RlTrainerSeedContext::exit(&mut self.queue, error)
    }
}
struct Worker {
    queue: Queue,
    live: Arc<AtomicUsize>,
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.live.fetch_sub(1, Ordering::SeqCst);
    }
}
impl FiniteEnvironment<i64, i64, f64, Info> for Worker {
    fn reset(&mut self) -> Result<i64, EnvironmentPluginError> {
        match self.queue.lock().unwrap().get() {
            Ok(value) => Ok(value),
            Err(DataQueueError::Exhausted) => Ok(-1),
            Err(error) => Err(EnvironmentPluginError::new(error.to_string())),
        }
    }
    fn step(
        &mut self,
        _action: &i64,
    ) -> Result<FiniteBackendStep<i64, f64, Info>, EnvironmentPluginError> {
        Ok(FiniteBackendStep {
            observation: Some(0),
            reward: Some(3.0),
            done: true,
            info: Some(Info {
                log: Some(IndexMap::from([(
                    "reward".into(),
                    RlLogEntry {
                        level: 20,
                        value: RlLogValue::Float(3.0),
                    },
                )])),
            }),
        })
    }
}
struct Predicate;
impl FiniteObservationPredicate<i64> for Predicate {
    fn is_invalid(&mut self, value: &i64) -> Result<bool, EnvironmentPluginError> {
        Ok(*value < 0)
    }
}
struct BufferLogger(Arc<Mutex<LogBuffer>>);
impl FiniteVectorLogger<i64, f64, Info> for BufferLogger {
    fn on_all_ready(&mut self) -> Result<(), EnvironmentPluginError> {
        <LogBuffer as FiniteVectorLogger<i64, f64, Info>>::on_all_ready(&mut self.0.lock().unwrap())
    }
    fn on_all_done(&mut self) -> Result<(), EnvironmentPluginError> {
        <LogBuffer as FiniteVectorLogger<i64, f64, Info>>::on_all_done(&mut self.0.lock().unwrap())
    }
    fn on_reset(&mut self, id: usize, obs: &[Option<i64>]) -> Result<(), EnvironmentPluginError> {
        self.0.lock().unwrap().on_reset(id, obs)
    }
    fn on_step(
        &mut self,
        id: usize,
        step: &FiniteBackendStep<i64, f64, Info>,
    ) -> Result<(), EnvironmentPluginError> {
        self.0.lock().unwrap().on_step(id, step)
    }
}
struct Policy(Arc<AtomicUsize>);
impl TrainingRunPolicy<Vec<i64>> for Policy {
    fn set_mode(&mut self, _mode: TrainingPolicyMode) -> Result<(), TrainingVesselRunError> {
        Ok(())
    }
    fn update(
        &mut self,
        size: u64,
        buffer: Option<&mut Vec<i64>>,
        _options: &TrainingUpdateOptions,
    ) -> Result<TrainingVesselMetrics, TrainingVesselRunError> {
        assert_eq!(size, 0);
        assert_eq!(buffer.unwrap().len(), 2);
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(IndexMap::new())
    }
}
struct Collector(Option<Vec<i64>>);
impl<P: TrainingRunPolicy<Vec<i64>> + ?Sized + 'static>
    TrainingRunCollector<i64, i64, f64, Info, Vec<i64>, Value, P> for Collector
{
    fn collect(
        &mut self,
        _policy: &mut P,
        env: &mut Environment,
        limit: TrainingCollectLimit,
    ) -> Result<TrainingVesselMetrics, TrainingVesselRunError> {
        let episodes = match limit {
            TrainingCollectLimit::Episodes(Some(n)) => usize::try_from(n).unwrap(),
            TrainingCollectLimit::Steps(_) => usize::MAX,
            other @ TrainingCollectLimit::Episodes(_) => panic!("unexpected {other:?}"),
        };
        for _ in 0..episodes {
            let reset = env.reset(None)?;
            let observation = reset.observations[0].unwrap();
            let step = env.step(&[0], None)?;
            assert!(step.transitions[0].done);
            if let Some(buffer) = &mut self.0 {
                buffer.push(observation);
            }
        }
        Ok(IndexMap::new())
    }
    fn buffer(&mut self) -> Result<Option<&mut Vec<i64>>, TrainingVesselRunError> {
        Ok(self.0.as_mut())
    }
}
struct Factory;
impl<P: TrainingRunPolicy<Vec<i64>> + ?Sized + 'static>
    TrainingCollectorFactory<i64, i64, f64, Info, Vec<i64>, Value, P> for Factory
{
    fn create_buffer(
        &mut self,
        _capacity: i64,
        environments: usize,
    ) -> Result<Vec<i64>, TrainingVesselRunError> {
        assert_eq!(environments, 1);
        Ok(vec![])
    }
    fn create_collector(
        &mut self,
        _policy: &mut P,
        _env: &mut Environment,
        buffer: Option<Vec<i64>>,
        _noise: bool,
    ) -> Result<
        BoxTrainingRunCollector<i64, i64, f64, Info, Vec<i64>, Value, P>,
        TrainingVesselRunError,
    > {
        Ok(Box::new(Collector(buffer)))
    }
}
struct LiveVessel {
    seeds: TrainingVesselSeeds<i64>,
    binding: TrainingVesselBinding,
    runner: Runner,
    buffer: Arc<Mutex<LogBuffer>>,
    queues: Vec<Queue>,
    live: Arc<AtomicUsize>,
}
impl RlTrainerVessel for LiveVessel {
    type Seed = QueueSeed;
    type Environment = Environment;
    fn assign_trainer(&mut self, runtime: &Arc<Runtime>) -> Result<(), RlTrainerDriverError> {
        let view: Arc<dyn TrainingTrainerView> = runtime.clone();
        self.runner.assign_trainer(&view);
        self.binding.assign_trainer(&view);
        Ok(())
    }
    fn seeds(&mut self, phase: RlTrainerPhase) -> Result<QueueSeed, RlTrainerDriverError> {
        let phase = match phase {
            RlTrainerPhase::Train => TrainingSeedPhase::Train,
            RlTrainerPhase::Validation => TrainingSeedPhase::Val,
            RlTrainerPhase::Test => TrainingSeedPhase::Test,
        };
        let queue = self
            .seeds
            .seed_queue_with_trainer(phase, &self.binding)
            .map_err(plugin)?;
        let queue = Arc::new(Mutex::new(queue));
        self.queues.push(queue.clone());
        Ok(QueueSeed {
            queue,
            live: self.live.clone(),
        })
    }
    fn environment(
        &mut self,
        seed: &mut QueueSeed,
        _control: &mut RlTrainerControl,
    ) -> Result<Environment, RlTrainerDriverError> {
        assert!(seed.queue.lock().unwrap().is_activated());
        self.live.fetch_add(1, Ordering::SeqCst);
        let backend = FiniteDummyBackend::new(vec![Box::new(Worker {
            queue: seed.queue.clone(),
            live: self.live.clone(),
        })])
        .map_err(plugin)?;
        Ok(Environment::new(
            Box::new(backend),
            Box::new(Predicate),
            vec![Box::new(BufferLogger(self.buffer.clone()))],
        ))
    }
    fn run(
        &mut self,
        phase: RlTrainerPhase,
        env: &mut Environment,
        _control: &mut RlTrainerControl,
    ) -> Result<(), RlTrainerDriverError> {
        let result = match phase {
            RlTrainerPhase::Train => self.runner.train(env),
            RlTrainerPhase::Validation => self.runner.validate(env),
            RlTrainerPhase::Test => self.runner.test(env),
        }
        .map_err(plugin)?;
        assert_eq!(result.is_some(), phase == RlTrainerPhase::Train);
        assert_eq!(env.unguarded_reset_warnings(), 0);
        Ok(())
    }
}

struct ChangeSeedConfig;
impl RlTrainerCallback<LiveVessel> for ChangeSeedConfig {
    fn call(
        &mut self,
        hook: RlTrainerHook,
        control: &mut RlTrainerControl,
        _: &mut LiveVessel,
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

fn run_live_driver(changing: bool) {
    let runtime = Arc::new(Runtime::new(Some(2)));
    let updates = Arc::new(AtomicUsize::new(0));
    let buffer = Arc::new(Mutex::new(LogBuffer::new_buffer(
        20,
        runtime.buffer_callback(),
    )));
    let vessel = LiveVessel {
        binding: TrainingVesselBinding::default(),
        seeds: TrainingVesselSeeds::new(
            Some(Arc::new(vec![7, 9])),
            Some(Arc::new(vec![11, 13])),
            Some(Arc::new(vec![17, 19])),
            None,
        ),
        runner: Runner::new(
            Box::new(Policy(updates.clone())),
            Box::new(Factory),
            TrainingVesselBinding::default(),
            TrainingVesselRunConfig::default(),
            TrainingVesselLog::default(),
        ),
        buffer,
        queues: vec![],
        live: Arc::new(AtomicUsize::new(0)),
    };
    let mut driver = RlTrainerDriver::new(
        vessel,
        runtime.clone(),
        RlTrainerConfig {
            max_iters: Some(2.into()),
            val_every_n_iters: Some(1.into()),
        },
    );
    if changing {
        driver.callbacks.push(Box::new(ChangeSeedConfig));
    }
    driver.fit(None).unwrap();
    assert_eq!(updates.load(Ordering::SeqCst), 2);
    runtime
        .read(|state| {
            assert_eq!(state.current_iter, Some(2.into()));
            assert_eq!(
                state.current_episode,
                Some(if changing { 6.into() } else { 8.into() })
            );
            assert_eq!(state.current_stage, "val");
            assert_eq!(state.should_stop, Some(true));
            assert_eq!(
                state.metrics.as_ref().unwrap(),
                &IndexMap::from([("reward".into(), 3.0), ("val/reward".into(), 3.0)])
            );
        })
        .unwrap();
    driver.test().unwrap();
    assert_eq!(updates.load(Ordering::SeqCst), 2);
    runtime
        .read(|state| {
            assert_eq!(state.current_iter, Some(2.into()));
            assert_eq!(
                state.current_episode,
                Some(if changing { 7.into() } else { 10.into() })
            );
            assert_eq!(state.current_stage, "test");
            assert_eq!(state.should_stop, Some(true));
            assert_eq!(
                state.metrics.as_ref().unwrap(),
                &IndexMap::from([("reward".into(), 3.0)])
            );
        })
        .unwrap();
    assert_eq!(driver.vessel.queues.len(), 5);
    for queue in &driver.vessel.queues {
        assert_eq!(queue.lock().unwrap().get(), Err(DataQueueError::Exhausted));
    }
    assert_eq!(driver.vessel.live.load(Ordering::SeqCst), 0);
}

#[test]
fn driver_composes_real_seed_queues_vessel_runner_finite_env_and_metric_callback() {
    run_live_driver(false);
}

#[test]
fn phase_callbacks_change_live_seed_subsets_after_trainer_attachment() {
    run_live_driver(true);
}
