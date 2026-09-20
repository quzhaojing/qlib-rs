#![allow(clippy::float_cmp)]

use std::{
    collections::VecDeque,
    path::PathBuf,
    process::Command,
    sync::{Arc, Mutex},
};

use arrow_array::RecordBatch;
use arrow_schema::Schema;
use chrono::NaiveDateTime;
use domain_core::{
    BoxedFiniteEnvironment, EnvironmentActionInterpreter, EnvironmentAuxiliaryInfo,
    EnvironmentLogCollector, EnvironmentLogLevel, EnvironmentObservationSpace,
    EnvironmentPluginError, EnvironmentResetError, EnvironmentResetRunner, EnvironmentResetStage,
    EnvironmentReward, EnvironmentRewardError, EnvironmentSeedSource, EnvironmentSimulator,
    EnvironmentSimulatorFactory, EnvironmentStateInterpreter, EnvironmentStatus,
    EnvironmentStepError, EnvironmentStepInfo, EnvironmentStepRunner, FiniteDummyBackend,
    FiniteEnvironment, FiniteObservationPredicate, FiniteVectorEnv, FiniteVectorError, Order,
    OrderDir, SaoeBacktestData, SaoeState, SaoeStateParts,
};
use ndarray::Array1;
use serde_json::Value;

type Runner = EnvironmentStepRunner<i64, i64, i64, i64, ()>;
type ResetRunner = EnvironmentResetRunner<i64, i64, i64, i64, ()>;
type FactoryCalls = Arc<Mutex<Vec<Option<i64>>>>;
type InvalidCalls = Arc<Mutex<usize>>;
type RunnerInfo = EnvironmentStepInfo<i64, i64, ()>;
type FiniteRunner = dyn FiniteEnvironment<i64, i64, f64, RunnerInfo>;

struct InvalidObservationPredicate;

impl FiniteObservationPredicate<i64> for InvalidObservationPredicate {
    fn is_invalid(&mut self, observation: &i64) -> Result<bool, EnvironmentPluginError> {
        Ok(*observation == -999)
    }
}

fn state(value: i64) -> SaoeState {
    let empty = RecordBatch::new_empty(Arc::new(Schema::empty()));
    SaoeState::new(SaoeStateParts {
        order: Order::new("A", 1.0, OrderDir::Buy, None, None),
        cur_time: NaiveDateTime::default(),
        cur_step: value,
        position: 1.0,
        history_exec: empty.clone(),
        history_steps: empty.clone(),
        metrics: None,
        backtest_data: SaoeBacktestData {
            ticks_index: Vec::new(),
            ticks_for_order: Vec::new(),
            deal_prices: Array1::zeros(0),
            market_volumes: Array1::zeros(0),
            features: empty,
        },
        ticks_per_step: 1,
        ticks_index: Vec::new(),
        ticks_for_order: Vec::new(),
    })
}

#[derive(Clone, Copy)]
enum Failure {
    None,
    Stop,
    Error,
}

fn failure(value: Failure) -> Result<(), EnvironmentPluginError> {
    match value {
        Failure::None => Ok(()),
        Failure::Stop => Err(EnvironmentPluginError::stop_iteration()),
        Failure::Error => Err(EnvironmentPluginError::new("failure")),
    }
}

struct Simulator {
    value: i64,
    initial_state_failure: Failure,
}

impl EnvironmentSimulator<i64, i64, i64, i64> for Simulator {
    fn state(
        &self,
        status: &EnvironmentStatus<i64, i64, i64>,
    ) -> Result<SaoeState, EnvironmentPluginError> {
        if status.observation_history.is_empty() {
            failure(self.initial_state_failure)?;
        }
        Ok(state(self.value))
    }

    fn step(
        &mut self,
        action: i64,
        _status: &EnvironmentStatus<i64, i64, i64>,
    ) -> Result<(), EnvironmentPluginError> {
        self.value += action;
        Ok(())
    }

    fn done(
        &self,
        _status: &EnvironmentStatus<i64, i64, i64>,
    ) -> Result<bool, EnvironmentPluginError> {
        Ok(false)
    }
}

struct StateInterpreter {
    failures: Arc<Mutex<VecDeque<Failure>>>,
}

impl EnvironmentStateInterpreter<i64, i64, i64> for StateInterpreter {
    fn interpret(
        &mut self,
        state: &SaoeState,
        status: &EnvironmentStatus<i64, i64, i64>,
    ) -> Result<i64, EnvironmentPluginError> {
        if status.observation_history.is_empty() {
            failure(
                self.failures
                    .lock()
                    .unwrap()
                    .pop_front()
                    .unwrap_or(Failure::None),
            )?;
        }
        Ok(state.parts().cur_step * 10)
    }
}

struct ActionInterpreter;

impl EnvironmentActionInterpreter<i64, i64, i64, i64> for ActionInterpreter {
    fn interpret(
        &mut self,
        _state: &SaoeState,
        action: &i64,
        _status: &EnvironmentStatus<i64, i64, i64>,
    ) -> Result<i64, EnvironmentPluginError> {
        Ok(*action)
    }
}

impl EnvironmentReward<i64, i64, i64> for ActionInterpreter {
    fn reward(
        &mut self,
        _state: &SaoeState,
        _status: &EnvironmentStatus<i64, i64, i64>,
    ) -> Result<f64, EnvironmentRewardError> {
        Ok(0.0)
    }
}

impl EnvironmentAuxiliaryInfo<i64, i64, i64, ()> for ActionInterpreter {
    fn collect(
        &mut self,
        _state: &SaoeState,
        _status: &EnvironmentStatus<i64, i64, i64>,
    ) -> Result<(), EnvironmentPluginError> {
        Ok(())
    }
}

struct Factory {
    failure: Failure,
    simulator_failure: Failure,
    calls: Arc<Mutex<Vec<Option<i64>>>>,
}

impl EnvironmentSimulatorFactory<i64, i64, i64, i64> for Factory {
    fn create(
        &mut self,
        initial_state: Option<&i64>,
    ) -> Result<Box<dyn EnvironmentSimulator<i64, i64, i64, i64>>, EnvironmentPluginError> {
        self.calls.lock().unwrap().push(initial_state.copied());
        failure(self.failure)?;
        Ok(Box::new(Simulator {
            value: initial_state.copied().unwrap_or(7),
            initial_state_failure: self.simulator_failure,
        }))
    }
}

struct ObservationSpace {
    result: Result<i64, EnvironmentPluginError>,
    calls: Arc<Mutex<usize>>,
}

impl EnvironmentObservationSpace<i64> for ObservationSpace {
    fn invalid_observation(&mut self) -> Result<i64, EnvironmentPluginError> {
        *self.calls.lock().unwrap() += 1;
        self.result.clone()
    }
}

fn uninitialized(interpreter_failures: &[Failure]) -> Runner {
    EnvironmentStepRunner::uninitialized(
        Box::new(StateInterpreter {
            failures: Arc::new(Mutex::new(interpreter_failures.iter().copied().collect())),
        }),
        Box::new(ActionInterpreter),
        None,
        None,
        Arc::new(EnvironmentLogCollector::new(EnvironmentLogLevel::Debug)),
    )
}

fn reset_runner(
    source: EnvironmentSeedSource<i64>,
    factory_failure: Failure,
    simulator_failure: Failure,
    interpreter_failures: &[Failure],
    invalid: Result<i64, EnvironmentPluginError>,
) -> (ResetRunner, FactoryCalls, InvalidCalls) {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let invalid_calls = Arc::new(Mutex::new(0));
    (
        EnvironmentResetRunner::new(
            source,
            Box::new(Factory {
                failure: factory_failure,
                simulator_failure,
                calls: Arc::clone(&calls),
            }),
            Box::new(ObservationSpace {
                result: invalid,
                calls: Arc::clone(&invalid_calls),
            }),
            uninitialized(interpreter_failures),
        ),
        calls,
        invalid_calls,
    )
}

fn python_contract() -> Value {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/environment_reset_contract.py");
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../qlib/qlib/rl/utils/env_wrapper.py");
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
fn fallible_seed_errors_preserve_source_retry_status_and_exhaustion_order() {
    let python = python_contract();
    assert_eq!(
        python["iterator_error_then_retry"]["outputs"],
        serde_json::json!([
            ["ok", 20],
            ["error", "ValueError"],
            ["ok", 30],
            ["ok", "invalid"],
            ["error", "RuntimeError"]
        ])
    );
    assert_eq!(python["iterator_error_state"]["status"]["initial_state"], 2);
    assert_eq!(python["iterator_error_state"]["dead"], false);
    let reads = Arc::new(Mutex::new(0));
    let counted = reads.clone();
    let values = [
        Ok(2),
        Err(EnvironmentPluginError::new("seed")),
        Ok(3),
        Err(EnvironmentPluginError::stop_iteration()),
        Ok(4),
    ];
    let iterator = values
        .into_iter()
        .inspect(move |_| *counted.lock().unwrap() += 1);
    let (mut environment, calls, invalid_calls) = reset_runner(
        EnvironmentSeedSource::try_seeded(iterator),
        Failure::None,
        Failure::None,
        &[],
        Ok(-999),
    );
    assert_eq!(environment.reset().unwrap(), 20);
    let before = environment.status().unwrap().clone();
    environment
        .step_runner()
        .logger()
        .add_scalar("old", 1., EnvironmentLogLevel::Debug)
        .unwrap();
    assert_eq!(
        environment.reset().unwrap_err(),
        EnvironmentResetError::Component {
            stage: EnvironmentResetStage::SeedIterator,
            source: EnvironmentPluginError::new("seed"),
        }
    );
    assert_eq!(environment.status().unwrap(), &before);
    assert!(!environment.is_exhausted());
    assert!(environment.step_runner().logger().snapshot().is_empty());
    assert_eq!(*calls.lock().unwrap(), [Some(2)]);
    assert_eq!(*invalid_calls.lock().unwrap(), 0);
    assert_eq!(environment.reset().unwrap(), 30);
    assert_eq!(environment.reset().unwrap(), -999);
    assert_eq!(environment.reset(), Err(EnvironmentResetError::Dead));
    assert_eq!(*reads.lock().unwrap(), 4, "never read after StopIteration");
    assert_eq!(*calls.lock().unwrap(), [Some(2), Some(3)]);
    assert_eq!(*invalid_calls.lock().unwrap(), 1);
    assert_eq!(environment.status().unwrap().initial_state, Some(3));
    assert!(matches!(
        environment.step(1),
        Err(EnvironmentStepError::Exhausted)
    ));
}

#[test]
fn fallible_seed_end_and_invalid_observation_failure_remain_terminal() {
    let python = python_contract();
    assert_eq!(
        python["iterator_end"]["outputs"],
        serde_json::json!([["ok", 20], ["ok", "invalid"], ["error", "RuntimeError"]])
    );
    for invalid in [Ok(-999), Err(EnvironmentPluginError::new("invalid"))] {
        let (mut environment, calls, invalid_calls) = reset_runner(
            EnvironmentSeedSource::try_seeded([Ok(2)].into_iter()),
            Failure::None,
            Failure::None,
            &[],
            invalid.clone(),
        );
        assert_eq!(environment.reset().unwrap(), 20);
        let expected = invalid.map_err(|source| EnvironmentResetError::Component {
            stage: EnvironmentResetStage::InvalidObservation,
            source,
        });
        assert_eq!(environment.reset(), expected);
        assert!(environment.is_exhausted());
        assert_eq!(environment.reset(), Err(EnvironmentResetError::Dead));
        assert_eq!(*calls.lock().unwrap(), [Some(2)]);
        assert_eq!(*invalid_calls.lock().unwrap(), 1);
    }
}

#[test]
fn two_native_environments_consume_one_context_owned_queue_without_copying_seeds() {
    use domain_core::{DataQueue, DataQueueConfig, DataQueueError, RlTrainerSeedContext};
    let mut queue = Arc::new(Mutex::new(DataQueue::new(
        Arc::new(vec![2_i64, 3]),
        DataQueueConfig {
            repeat: 1,
            shuffle: false,
            queue_maxsize: 1,
            ..DataQueueConfig::default()
        },
    )));
    RlTrainerSeedContext::enter(&mut queue).unwrap();
    let mut environments = Vec::new();
    for _ in 0..2 {
        let handle = queue.clone();
        let iterator = std::iter::from_fn(move || match handle.lock().unwrap().get() {
            Ok(value) => Some(Ok(value)),
            Err(DataQueueError::Exhausted) => None,
            Err(error) => Some(Err(EnvironmentPluginError::new(error.to_string()))),
        });
        environments.push(reset_runner(
            EnvironmentSeedSource::try_seeded(iterator),
            Failure::None,
            Failure::None,
            &[],
            Ok(-999),
        ));
    }
    assert_eq!(environments[0].0.reset().unwrap(), 20);
    assert_eq!(environments[1].0.reset().unwrap(), 30);
    for (environment, calls, invalid_calls) in &mut environments {
        assert_eq!(environment.reset().unwrap(), -999);
        assert_eq!(calls.lock().unwrap().len(), 1);
        assert_eq!(*invalid_calls.lock().unwrap(), 1);
    }
    assert!(!RlTrainerSeedContext::exit(&mut queue, None).unwrap());
    assert!(queue.lock().unwrap().done());
    assert!(matches!(
        queue.lock().unwrap().get(),
        Err(DataQueueError::Exhausted)
    ));
}

#[test]
fn seeded_and_unseeded_reset_lifecycles_match_live_python() {
    let python = python_contract();
    assert_eq!(
        python["seeded"]["outputs"],
        serde_json::json!([["ok", 20], ["ok", "invalid"], ["error", "RuntimeError"]])
    );
    assert_eq!(
        python["unseeded"]["outputs"],
        serde_json::json!([["ok", 70], ["ok", 70]])
    );

    let (mut seeded, calls, invalid_calls) = reset_runner(
        EnvironmentSeedSource::seeded([2].into_iter()),
        Failure::None,
        Failure::None,
        &[],
        Ok(-999),
    );
    assert!(seeded.status().is_none());
    assert_eq!(seeded.reset().unwrap(), 20);
    assert_eq!(seeded.status().unwrap().initial_state, Some(2));
    assert_eq!(seeded.status().unwrap().observation_history, [20]);
    assert_eq!(seeded.reset().unwrap(), -999);
    assert!(seeded.is_exhausted());
    assert!(seeded.step_runner().try_status().is_some());
    assert!(matches!(
        seeded.step(1),
        Err(EnvironmentStepError::Exhausted)
    ));
    assert_eq!(seeded.reset(), Err(EnvironmentResetError::Dead));
    assert_eq!(*calls.lock().unwrap(), [Some(2)]);
    assert_eq!(*invalid_calls.lock().unwrap(), 1);

    let (mut unseeded, calls, _) = reset_runner(
        EnvironmentSeedSource::unseeded(),
        Failure::None,
        Failure::None,
        &[],
        Ok(-999),
    );
    assert_eq!(unseeded.reset().unwrap(), 70);
    assert_eq!(unseeded.reset().unwrap(), 70);
    assert!(!unseeded.is_exhausted());
    assert_eq!(*calls.lock().unwrap(), [None, None]);
}

#[test]
fn stop_iteration_from_every_reset_plugin_permanently_exhausts() {
    let python = python_contract();
    for name in ["factory_stop", "state_stop", "interpreter_stop"] {
        assert_eq!(
            python[name]["outputs"],
            serde_json::json!([["ok", "invalid"]])
        );
        assert_eq!(python[name]["dead"], true);
    }

    for (factory, simulator, interpreter) in [
        (Failure::Stop, Failure::None, Failure::None),
        (Failure::None, Failure::Stop, Failure::None),
        (Failure::None, Failure::None, Failure::Stop),
    ] {
        let (mut environment, _, invalid_calls) = reset_runner(
            EnvironmentSeedSource::seeded([1].into_iter()),
            factory,
            simulator,
            &[interpreter],
            Ok(-999),
        );
        assert_eq!(environment.reset().unwrap(), -999);
        assert!(environment.is_exhausted());
        assert_eq!(*invalid_calls.lock().unwrap(), 1);
    }
}

#[test]
fn ordinary_failures_retain_python_partial_state_and_do_not_mark_dead() {
    let python = python_contract();
    assert_eq!(python["factory_error"]["dead"], false);
    assert!(python["factory_error"]["status"].is_null());
    assert_eq!(python["state_error"]["dead"], false);
    assert_eq!(python["state_error"]["status"]["initial_state"], 1);
    assert_eq!(
        python["state_error"]["status"]["obs_history"],
        serde_json::json!([])
    );

    let (mut factory_error, _, _) = reset_runner(
        EnvironmentSeedSource::seeded([1].into_iter()),
        Failure::Error,
        Failure::None,
        &[],
        Ok(-999),
    );
    assert!(matches!(
        factory_error.reset(),
        Err(EnvironmentResetError::Component {
            stage: EnvironmentResetStage::SimulatorFactory,
            ..
        })
    ));
    assert!(!factory_error.is_exhausted());
    assert!(factory_error.status().is_none());

    let (mut state_error, _, _) = reset_runner(
        EnvironmentSeedSource::seeded([1].into_iter()),
        Failure::None,
        Failure::Error,
        &[],
        Ok(-999),
    );
    assert!(matches!(
        state_error.reset(),
        Err(EnvironmentResetError::Component {
            stage: EnvironmentResetStage::InitialObservation,
            ..
        })
    ));
    assert!(!state_error.is_exhausted());
    assert_eq!(state_error.status().unwrap().initial_state, Some(1));
    assert!(state_error.status().unwrap().observation_history.is_empty());
}

#[test]
fn invalid_generation_errors_and_step_before_reset_are_typed() {
    let normal = EnvironmentPluginError::new("normal");
    assert!(!normal.is_stop_iteration());
    assert!(EnvironmentPluginError::stop_iteration().is_stop_iteration());

    let mut runner = uninitialized(&[]);
    assert!(runner.try_status().is_none());
    assert!(matches!(
        runner.step(1),
        Err(EnvironmentStepError::NotReset)
    ));

    let mut optional_plugins: Runner = EnvironmentStepRunner::uninitialized(
        Box::new(StateInterpreter {
            failures: Arc::new(Mutex::new(VecDeque::new())),
        }),
        Box::new(ActionInterpreter),
        Some(Box::new(ActionInterpreter)),
        Some(Box::new(ActionInterpreter)),
        Arc::new(EnvironmentLogCollector::new(EnvironmentLogLevel::Debug)),
    );
    assert!(matches!(
        optional_plugins.step(1),
        Err(EnvironmentStepError::NotReset)
    ));

    let (mut environment, _, invalid_calls) = reset_runner(
        EnvironmentSeedSource::seeded(std::iter::empty()),
        Failure::None,
        Failure::None,
        &[],
        Err(EnvironmentPluginError::new("invalid")),
    );
    assert!(matches!(
        environment.reset(),
        Err(EnvironmentResetError::Component {
            stage: EnvironmentResetStage::InvalidObservation,
            ..
        })
    ));
    assert!(environment.is_exhausted());
    assert_eq!(*invalid_calls.lock().unwrap(), 1);
}

#[test]
fn reset_environment_delegates_subsequent_steps() {
    let (mut environment, _, _) = reset_runner(
        EnvironmentSeedSource::unseeded(),
        Failure::None,
        Failure::None,
        &[],
        Ok(-999),
    );
    assert_eq!(environment.reset().unwrap(), 70);
    let output = environment.step(2).unwrap();
    assert_eq!(output.observation, 90);
    assert_eq!(output.reward, 0.0);
    assert!(!output.done);
    assert_eq!(environment.status().unwrap().action_history, [2]);
}

#[test]
fn reset_runner_adapter_maps_outputs_errors_and_runs_two_dummy_workers() {
    let (mut healthy, _, _) = reset_runner(
        EnvironmentSeedSource::unseeded(),
        Failure::None,
        Failure::None,
        &[],
        Ok(-999),
    );
    assert_eq!(FiniteRunner::reset(&mut healthy).unwrap(), 70);
    let transition = FiniteRunner::step(&mut healthy, &2).unwrap();
    assert_eq!(transition.observation, Some(90));
    assert_eq!(transition.reward, Some(0.0));
    assert!(!transition.done);
    assert_eq!(transition.info.unwrap().auxiliary_info, ());

    let (mut factory_error, _, _) = reset_runner(
        EnvironmentSeedSource::unseeded(),
        Failure::Error,
        Failure::None,
        &[],
        Ok(-999),
    );
    assert!(
        FiniteRunner::reset(&mut factory_error)
            .unwrap_err()
            .message
            .contains("SimulatorFactory")
    );

    let (mut dead, _, _) = reset_runner(
        EnvironmentSeedSource::seeded(std::iter::empty()),
        Failure::None,
        Failure::None,
        &[],
        Ok(-999),
    );
    assert_eq!(FiniteRunner::reset(&mut dead).unwrap(), -999);
    assert!(
        FiniteRunner::reset(&mut dead)
            .unwrap_err()
            .message
            .contains("dead environment")
    );
    assert!(
        FiniteRunner::step(&mut dead, &1)
            .unwrap_err()
            .message
            .contains("seed iterator is exhausted")
    );

    let (mut not_reset, _, _) = reset_runner(
        EnvironmentSeedSource::unseeded(),
        Failure::None,
        Failure::None,
        &[],
        Ok(-999),
    );
    assert!(
        FiniteRunner::step(&mut not_reset, &1)
            .unwrap_err()
            .message
            .contains("has not been reset")
    );

    let factory_calls = Arc::new(Mutex::new(0_i64));
    let factory_counter = Arc::clone(&factory_calls);
    let mut factory = move || {
        let mut count = factory_counter.lock().unwrap();
        *count += 1;
        let initial_state = *count;
        let (runner, _, _) = reset_runner(
            EnvironmentSeedSource::seeded([initial_state].into_iter()),
            Failure::None,
            Failure::None,
            &[],
            Ok(-999),
        );
        Ok(Box::new(runner) as BoxedFiniteEnvironment<i64, i64, f64, RunnerInfo>)
    };
    let backend = FiniteDummyBackend::from_factory(2, &mut factory).unwrap();
    assert_eq!(*factory_calls.lock().unwrap(), 2);
    let mut vector = FiniteVectorEnv::new(
        Box::new(backend),
        Box::new(InvalidObservationPredicate),
        Vec::new(),
    );
    assert_eq!(
        vector.reset(None).unwrap().observations,
        vec![Some(10), Some(20)]
    );
    let stepped = vector.step(&[3, 4], None).unwrap();
    assert_eq!(stepped.transitions[0].observation, Some(40));
    assert_eq!(stepped.transitions[1].observation, Some(60));
    assert_eq!(stepped.transitions[0].reward, Some(0.0));
    assert_eq!(stepped.transitions[1].reward, Some(0.0));
    assert_eq!(vector.reset(None), Err(FiniteVectorError::Exhausted));
    assert!(vector.is_zombie());
}
