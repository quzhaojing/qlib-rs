#![allow(clippy::float_cmp)]

use std::{
    path::PathBuf,
    process::Command,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use arrow_array::RecordBatch;
use arrow_schema::Schema;
use chrono::NaiveDateTime;
use domain_core::{
    EnvironmentActionInterpreter, EnvironmentAuxInfo, EnvironmentAuxiliaryInfo,
    EnvironmentLogCollector, EnvironmentLogError, EnvironmentLogLevel, EnvironmentLogValue,
    EnvironmentPluginError, EnvironmentReward, EnvironmentRewardError, EnvironmentSimulator,
    EnvironmentStateInterpreter, EnvironmentStatus, EnvironmentStepError, EnvironmentStepRunner,
    EnvironmentStepStage, Order, OrderDir, PpoReward, SaoeBacktestData, SaoeEnvironmentReward,
    SaoeReward, SaoeRewardError, SaoeRewardLogSink, SaoeState, SaoeStateParts,
};
use indexmap::IndexMap;
use ndarray::Array1;
use num_bigint::BigInt;
use serde_json::{Value, json};

type Events = Arc<Mutex<Vec<String>>>;

fn plugin(message: &str) -> EnvironmentPluginError {
    EnvironmentPluginError::new(message)
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

struct Simulator {
    value: i64,
    done: bool,
    failure: Option<&'static str>,
    conflict: Option<&'static str>,
    state_calls: AtomicUsize,
    events: Events,
    logger: Option<Arc<dyn SaoeRewardLogSink>>,
    attachments: Arc<AtomicUsize>,
}

impl EnvironmentSimulator<i64, i64, i64, i64> for Simulator {
    fn set_logger(&mut self, logger: Option<Arc<dyn SaoeRewardLogSink>>) {
        self.logger = logger;
        self.attachments.fetch_add(1, Ordering::SeqCst);
    }

    fn state(
        &self,
        status: &EnvironmentStatus<i64, i64, i64>,
    ) -> Result<SaoeState, EnvironmentPluginError> {
        let call = self.state_calls.fetch_add(1, Ordering::SeqCst);
        let stage = if call == 0 { "pre_state" } else { "post_state" };
        if call == 0 {
            assert_eq!(status.cur_step, BigInt::from(0_u8));
            assert_eq!(status.action_history, [3]);
        } else {
            assert_eq!(status.cur_step, BigInt::from(1_u8));
            assert_eq!(status.done, self.done);
        }
        self.events.lock().unwrap().push(stage.to_owned());
        if self.failure == Some(stage) {
            Err(plugin(stage))
        } else {
            Ok(state(self.value))
        }
    }

    fn step(
        &mut self,
        action: i64,
        status: &EnvironmentStatus<i64, i64, i64>,
    ) -> Result<(), EnvironmentPluginError> {
        assert_eq!(status.cur_step, BigInt::from(1_u8));
        assert!(!status.done);
        self.events.lock().unwrap().push("sim_step".to_owned());
        if self.failure == Some("sim_step") {
            return Err(plugin("sim_step"));
        }
        self.value += action;
        self.logger
            .as_ref()
            .unwrap()
            .log_scalar(self.conflict.unwrap_or("sim_metric"), 101.0)
            .map_err(|error| plugin(&error.message))
    }

    fn done(
        &self,
        status: &EnvironmentStatus<i64, i64, i64>,
    ) -> Result<bool, EnvironmentPluginError> {
        assert_eq!(status.cur_step, BigInt::from(1_u8));
        assert!(!status.done);
        self.events.lock().unwrap().push("done".to_owned());
        if self.failure == Some("done") {
            Err(plugin("done"))
        } else {
            Ok(self.done)
        }
    }
}

struct ActionInterpreter {
    failure: Option<&'static str>,
    events: Events,
    attachments: Arc<AtomicUsize>,
}

impl EnvironmentActionInterpreter<i64, i64, i64, i64> for ActionInterpreter {
    fn set_logger(&mut self, logger: Option<Arc<dyn SaoeRewardLogSink>>) {
        assert!(logger.is_some());
        self.attachments.fetch_add(1, Ordering::SeqCst);
    }

    fn interpret(
        &mut self,
        _state: &SaoeState,
        action: &i64,
        status: &EnvironmentStatus<i64, i64, i64>,
    ) -> Result<i64, EnvironmentPluginError> {
        assert_eq!(status.cur_step, BigInt::from(0_u8));
        assert_eq!(status.action_history, [3]);
        self.events.lock().unwrap().push("action".to_owned());
        if self.failure == Some("action") {
            Err(plugin("action"))
        } else {
            Ok(*action - 1)
        }
    }
}

struct StateInterpreter {
    failure: Option<&'static str>,
    events: Events,
    attachments: Arc<AtomicUsize>,
}

impl EnvironmentStateInterpreter<i64, i64, i64> for StateInterpreter {
    fn set_logger(&mut self, logger: Option<Arc<dyn SaoeRewardLogSink>>) {
        assert!(logger.is_some());
        self.attachments.fetch_add(1, Ordering::SeqCst);
    }

    fn interpret(
        &mut self,
        state: &SaoeState,
        status: &EnvironmentStatus<i64, i64, i64>,
    ) -> Result<i64, EnvironmentPluginError> {
        assert_eq!(status.cur_step, BigInt::from(1_u8));
        assert_eq!(status.observation_history, [10]);
        self.events.lock().unwrap().push("state_interp".to_owned());
        if self.failure == Some("state_interp") {
            Err(plugin("state_interp"))
        } else {
            Ok(state.parts().cur_step)
        }
    }
}

struct Reward {
    failure: Option<&'static str>,
    events: Events,
    logger: Option<Arc<dyn SaoeRewardLogSink>>,
    attachments: Arc<AtomicUsize>,
}

impl EnvironmentReward<i64, i64, i64> for Reward {
    fn set_logger(&mut self, logger: Option<Arc<dyn SaoeRewardLogSink>>) {
        self.logger = logger;
        self.attachments.fetch_add(1, Ordering::SeqCst);
    }

    fn reward(
        &mut self,
        _state: &SaoeState,
        status: &EnvironmentStatus<i64, i64, i64>,
    ) -> Result<f64, EnvironmentRewardError> {
        self.events
            .lock()
            .unwrap()
            .push(format!("reward_done:{}", status.done));
        if self.failure == Some("reward") {
            return Err(plugin("reward").into());
        }
        self.logger
            .as_ref()
            .unwrap()
            .log_scalar("reward_metric", 7.0)
            .map_err(|error| plugin(&error.message))?;
        Ok(7.0)
    }
}

struct Auxiliary {
    failure: Option<&'static str>,
    events: Events,
    attachments: Arc<AtomicUsize>,
}

impl EnvironmentAuxiliaryInfo<i64, i64, i64, EnvironmentAuxInfo> for Auxiliary {
    fn set_logger(&mut self, logger: Option<Arc<dyn SaoeRewardLogSink>>) {
        assert!(logger.is_some());
        self.attachments.fetch_add(1, Ordering::SeqCst);
    }

    fn collect(
        &mut self,
        _state: &SaoeState,
        status: &EnvironmentStatus<i64, i64, i64>,
    ) -> Result<EnvironmentAuxInfo, EnvironmentPluginError> {
        self.events
            .lock()
            .unwrap()
            .push(format!("aux_rewards:{}", status.reward_history.len()));
        if self.failure == Some("aux") {
            return Err(plugin("aux"));
        }
        Ok(IndexMap::from([(
            "seen_step".to_owned(),
            json!(status.cur_step.to_string().parse::<u64>().unwrap()),
        )]))
    }
}

type Runner = EnvironmentStepRunner<i64, i64, i64, i64, EnvironmentAuxInfo>;

fn runner(
    done: bool,
    with_reward: bool,
    with_auxiliary: bool,
    minimum: EnvironmentLogLevel,
    failure: Option<&'static str>,
    conflict: Option<&'static str>,
) -> (Runner, Events, Arc<AtomicUsize>) {
    let events = Arc::new(Mutex::new(Vec::new()));
    let attachments = Arc::new(AtomicUsize::new(0));
    let simulator = Simulator {
        value: 10,
        done,
        failure,
        conflict,
        state_calls: AtomicUsize::new(0),
        events: Arc::clone(&events),
        logger: None,
        attachments: Arc::clone(&attachments),
    };
    let state_interpreter = StateInterpreter {
        failure,
        events: Arc::clone(&events),
        attachments: Arc::clone(&attachments),
    };
    let action_interpreter = ActionInterpreter {
        failure,
        events: Arc::clone(&events),
        attachments: Arc::clone(&attachments),
    };
    let reward = with_reward.then(|| {
        Box::new(Reward {
            failure,
            events: Arc::clone(&events),
            logger: None,
            attachments: Arc::clone(&attachments),
        }) as Box<dyn EnvironmentReward<i64, i64, i64>>
    });
    let auxiliary = with_auxiliary.then(|| {
        Box::new(Auxiliary {
            failure,
            events: Arc::clone(&events),
            attachments: Arc::clone(&attachments),
        }) as Box<dyn EnvironmentAuxiliaryInfo<i64, i64, i64, EnvironmentAuxInfo>>
    });
    let logger = Arc::new(EnvironmentLogCollector::new(minimum));
    (
        EnvironmentStepRunner::new(
            Box::new(simulator),
            Box::new(state_interpreter),
            Box::new(action_interpreter),
            reward,
            auxiliary,
            logger,
            Some(5),
            10,
        ),
        events,
        attachments,
    )
}

fn status_json(status: &EnvironmentStatus<i64, i64, i64>) -> Value {
    json!({
        "cur_step": status.cur_step.to_string().parse::<u64>().unwrap(),
        "done": status.done,
        "initial_state": status.initial_state,
        "obs_history": status.observation_history,
        "action_history": status.action_history,
        "reward_history": status.reward_history,
    })
}

fn logs_json(logs: &domain_core::EnvironmentLogs<i64, i64>) -> Value {
    let values = logs
        .iter()
        .map(|(name, entry)| {
            let value = match &entry.value {
                EnvironmentLogValue::Scalar(value) => json!(value),
                EnvironmentLogValue::Observation(value)
                | EnvironmentLogValue::PolicyAction(value) => json!(value),
            };
            (name.clone(), json!([entry.level as i32, value]))
        })
        .collect::<serde_json::Map<_, _>>();
    Value::Object(values)
}

fn python_contract() -> Value {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/environment_step_contract.py");
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

fn assert_typed_environment_logs(
    logs: &domain_core::RlLogContents<EnvironmentLogValue<i64, i64>>,
    original: &domain_core::EnvironmentLogs<i64, i64>,
    level: i64,
) {
    use domain_core::RlLogValue;
    let expected_names = original
        .iter()
        .filter(|(_, entry)| i64::from(entry.level as i32) >= level)
        .map(|(name, _)| name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        logs.keys().map(String::as_str).collect::<Vec<_>>(),
        expected_names
    );
    for (name, value) in logs {
        let expected = match &original[name].value {
            EnvironmentLogValue::Scalar(value) => RlLogValue::Float(*value),
            value => RlLogValue::Other(value.clone()),
        };
        assert_eq!(*value, expected);
    }
}

#[test]
fn real_environment_info_drives_collector_termination_and_typed_log_buffer() {
    use domain_core::rl_candle_collector::CandleTerminationInfo;
    use domain_core::{
        EnvironmentStepInfo, FiniteBackendStep, FiniteVectorLogger, RlLogBuffer, RlLogBufferEvent,
        RlLogBufferState, RlLogError, RlLogWriterState,
    };
    type Payload = EnvironmentLogValue<i64, i64>;
    type Info = EnvironmentStepInfo<i64, i64, EnvironmentAuxInfo>;
    type Logger = dyn FiniteVectorLogger<i64, f64, Info>;
    let source = python_contract();
    for level in [10, 20] {
        let (mut environment, _, _) =
            runner(true, true, true, EnvironmentLogLevel::Debug, None, None);
        let output = environment.step(3).unwrap();
        assert_eq!(
            logs_json(&output.info.logs),
            source["success"]["output"][3]["log"]
        );
        assert!(!output.info.time_limit_truncated().unwrap());
        let mut step = FiniteBackendStep {
            observation: Some(output.observation),
            reward: Some(output.reward),
            done: output.done,
            info: Some(output.info),
        };
        step.info
            .as_mut()
            .unwrap()
            .auxiliary_info
            .insert("TimeLimit.truncated".into(), json!(true));
        let original = step.clone();
        assert!(!step.info.as_ref().unwrap().time_limit_truncated().unwrap());
        let failures = Arc::new(Mutex::new(false));
        let switch = failures.clone();
        let mut buffer = RlLogBuffer::new_buffer(
            level,
            move |event: RlLogBufferEvent, _: &RlLogWriterState<Payload>, _: &RlLogBufferState| {
                if *switch.lock().unwrap() {
                    Err(format!("{event:?} failed"))
                } else {
                    Ok(())
                }
            },
        );
        let logger: &mut Logger = &mut buffer;
        logger.on_all_ready().unwrap();
        logger.on_reset(0, &[Some(10)]).unwrap();
        logger.on_step(0, &step).unwrap();
        logger.on_all_done().unwrap();
        assert_eq!(step, original);
        let state = buffer.state_dict();
        assert_eq!(state.global_step, BigInt::from(1));
        assert_eq!(state.global_episode, BigInt::from(1));
        assert_eq!(state.episode_rewards[&0], [7.0]);
        let logs = &state.episode_logs[&0][0];
        assert_typed_environment_logs(logs, &step.info.as_ref().unwrap().logs, level);
        let metrics = buffer.buffer_state().episode_metrics().unwrap();
        assert_eq!(metrics["reward"], 7.0);
        assert_eq!(metrics["steps_per_episode"], 1.0);
        assert!(!metrics.contains_key("obs"));
        assert!(!metrics.contains_key("policy_act"));
        let logger: &mut Logger = &mut buffer;
        let mut missing = FiniteBackendStep::default();
        assert_eq!(
            logger.on_step(0, &missing).unwrap_err().message,
            RlLogError::MissingReward.to_string()
        );
        missing.reward = Some(1.0);
        assert_eq!(
            logger.on_step(0, &missing).unwrap_err().message,
            RlLogError::MissingLog.to_string()
        );
        *failures.lock().unwrap() = true;
        assert_eq!(
            logger.on_step(0, &step).unwrap_err().message,
            RlLogError::Callback {
                event: RlLogBufferEvent::Episode,
                message: "Episode failed".into()
            }
            .to_string()
        );
        assert_eq!(
            logger.on_all_done().unwrap_err().message,
            RlLogError::Callback {
                event: RlLogBufferEvent::Collect,
                message: "Collect failed".into()
            }
            .to_string()
        );
        assert_eq!(buffer.state_dict().global_step, BigInt::from(3));
    }
}

#[test]
fn complete_and_optional_step_paths_match_live_python_source() {
    let python = python_contract();
    let (mut environment, events, attachments) =
        runner(true, true, true, EnvironmentLogLevel::Debug, None, None);
    assert_eq!(attachments.load(Ordering::SeqCst), 5);
    let output = environment.step(3).unwrap();
    assert_eq!(output.observation, 12);
    assert_eq!(output.reward, 7.0);
    assert!(output.done);
    assert_eq!(
        status_json(environment.status()),
        python["success"]["status"]
    );
    assert_eq!(
        logs_json(&output.info.logs),
        python["success"]["output"][3]["log"]
    );
    assert_eq!(
        Value::Object(
            output
                .info
                .auxiliary_info
                .iter()
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect(),
        ),
        python["success"]["output"][3]["aux_info"]
    );
    assert_eq!(
        *events.lock().unwrap(),
        [
            "pre_state",
            "action",
            "sim_step",
            "done",
            "post_state",
            "state_interp",
            "reward_done:true",
            "aux_rewards:1",
        ]
    );
    assert_eq!(
        output
            .info
            .logs
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        [
            "sim_metric",
            "reward_metric",
            "steps_per_episode",
            "reward",
            "obs",
            "policy_act",
        ]
    );
    assert_eq!(
        logs_json(&environment.logger().snapshot()),
        logs_json(&output.info.logs)
    );

    let (mut fallback, _, attachments) = runner(
        false,
        false,
        false,
        EnvironmentLogLevel::Periodic,
        None,
        None,
    );
    assert_eq!(attachments.load(Ordering::SeqCst), 3);
    let output = fallback.step(3).unwrap();
    assert_eq!(status_json(fallback.status()), python["fallback"]["status"]);
    assert_eq!(
        logs_json(&output.info.logs),
        python["fallback"]["output"][3]["log"]
    );
    assert!(output.info.auxiliary_info.is_empty());
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one table covers every Python partial-failure boundary"
)]
fn every_component_and_final_log_failure_retains_the_python_partial_state() {
    let python = python_contract();
    let component_cases = [
        ("pre_state", EnvironmentStepStage::PreActionState),
        ("action", EnvironmentStepStage::ActionInterpreter),
        ("sim_step", EnvironmentStepStage::SimulatorStep),
        ("done", EnvironmentStepStage::SimulatorDone),
        ("post_state", EnvironmentStepStage::PostStepState),
        ("state_interp", EnvironmentStepStage::StateInterpreter),
        ("aux", EnvironmentStepStage::AuxiliaryInfo),
    ];
    for (name, expected_stage) in component_cases {
        let (mut runner, _, _) = runner(
            true,
            true,
            true,
            EnvironmentLogLevel::Debug,
            Some(name),
            None,
        );
        assert!(matches!(
            runner.step(3),
            Err(EnvironmentStepError::Component { stage, .. }) if stage == expected_stage
        ));
        assert_eq!(
            status_json(runner.status()),
            python["failures"][name]["status"]
        );
        assert_eq!(
            logs_json(&runner.logger().snapshot()),
            python["failures"][name]["retained_logs"]
        );
    }

    let (mut reward_failure, _, _) = runner(
        true,
        true,
        true,
        EnvironmentLogLevel::Debug,
        Some("reward"),
        None,
    );
    assert!(matches!(
        reward_failure.step(3),
        Err(EnvironmentStepError::Reward(
            EnvironmentRewardError::Plugin(_)
        ))
    ));
    assert_eq!(
        status_json(reward_failure.status()),
        python["failures"]["reward"]["status"]
    );

    let log_cases = [
        (
            "log_steps",
            "steps_per_episode",
            EnvironmentStepStage::StepsPerEpisodeLog,
        ),
        ("log_reward", "reward", EnvironmentStepStage::RewardLog),
        ("log_obs", "obs", EnvironmentStepStage::ObservationLog),
        (
            "log_action",
            "policy_act",
            EnvironmentStepStage::PolicyActionLog,
        ),
    ];
    for (name, conflict, expected_stage) in log_cases {
        let (mut runner, _, _) = runner(
            true,
            true,
            true,
            EnvironmentLogLevel::Debug,
            None,
            Some(conflict),
        );
        assert!(matches!(
            runner.step(3),
            Err(EnvironmentStepError::Log {
                stage,
                source: EnvironmentLogError::DuplicateMetric(_),
            }) if stage == expected_stage
        ));
        assert_eq!(
            status_json(runner.status()),
            python["failures"][name]["status"]
        );
        assert_eq!(
            logs_json(&runner.logger().snapshot()),
            python["failures"][name]["retained_logs"]
        );
    }

    let (mut exhausted, events, _) =
        runner(true, true, true, EnvironmentLogLevel::Debug, None, None);
    exhausted.mark_exhausted();
    assert!(matches!(
        exhausted.step(3),
        Err(EnvironmentStepError::Exhausted)
    ));
    assert!(events.lock().unwrap().is_empty());
    assert_eq!(
        status_json(exhausted.status()),
        python["failures"]["dead"]["status"]
    );
}

struct BoundReward {
    logger: Option<Arc<dyn SaoeRewardLogSink>>,
}

struct DefaultBindings;

impl EnvironmentSimulator<i64, i64, i64, i64> for DefaultBindings {
    fn state(
        &self,
        _status: &EnvironmentStatus<i64, i64, i64>,
    ) -> Result<SaoeState, EnvironmentPluginError> {
        Ok(state(0))
    }

    fn step(
        &mut self,
        _action: i64,
        _status: &EnvironmentStatus<i64, i64, i64>,
    ) -> Result<(), EnvironmentPluginError> {
        Ok(())
    }

    fn done(
        &self,
        _status: &EnvironmentStatus<i64, i64, i64>,
    ) -> Result<bool, EnvironmentPluginError> {
        Ok(false)
    }
}

impl EnvironmentActionInterpreter<i64, i64, i64, i64> for DefaultBindings {
    fn interpret(
        &mut self,
        _state: &SaoeState,
        action: &i64,
        _status: &EnvironmentStatus<i64, i64, i64>,
    ) -> Result<i64, EnvironmentPluginError> {
        Ok(*action)
    }
}

impl EnvironmentStateInterpreter<i64, i64, i64> for DefaultBindings {
    fn interpret(
        &mut self,
        state: &SaoeState,
        _status: &EnvironmentStatus<i64, i64, i64>,
    ) -> Result<i64, EnvironmentPluginError> {
        Ok(state.parts().cur_step)
    }
}

impl EnvironmentReward<i64, i64, i64> for DefaultBindings {
    fn reward(
        &mut self,
        _state: &SaoeState,
        _status: &EnvironmentStatus<i64, i64, i64>,
    ) -> Result<f64, EnvironmentRewardError> {
        Ok(0.0)
    }
}

impl EnvironmentAuxiliaryInfo<i64, i64, i64, EnvironmentAuxInfo> for DefaultBindings {
    fn collect(
        &mut self,
        _state: &SaoeState,
        _status: &EnvironmentStatus<i64, i64, i64>,
    ) -> Result<EnvironmentAuxInfo, EnvironmentPluginError> {
        Ok(IndexMap::new())
    }
}

impl SaoeReward for BoundReward {
    fn set_logger(&mut self, logger: Option<Arc<dyn SaoeRewardLogSink>>) {
        self.logger = logger;
    }

    fn reward(&self, _state: &SaoeState) -> Result<f64, SaoeRewardError> {
        self.logger
            .as_ref()
            .ok_or(SaoeRewardError::MissingLogger)?
            .log_scalar("bound_reward", 2.0)?;
        Ok(2.0)
    }
}

#[test]
fn collector_bound_saoe_reward_unbounded_steps_and_status_wire_are_explicit() {
    let mut uninitialized: Runner = EnvironmentStepRunner::uninitialized(
        Box::new(DefaultBindings),
        Box::new(DefaultBindings),
        None,
        None,
        Arc::new(EnvironmentLogCollector::new(EnvironmentLogLevel::Debug)),
    );
    assert!(matches!(
        uninitialized.step(1),
        Err(EnvironmentStepError::NotReset)
    ));

    let mut defaults = DefaultBindings;
    EnvironmentSimulator::<i64, i64, i64, i64>::set_logger(&mut defaults, None);
    EnvironmentActionInterpreter::<i64, i64, i64, i64>::set_logger(&mut defaults, None);
    EnvironmentStateInterpreter::<i64, i64, i64>::set_logger(&mut defaults, None);
    EnvironmentReward::<i64, i64, i64>::set_logger(&mut defaults, None);
    EnvironmentAuxiliaryInfo::<i64, i64, i64, EnvironmentAuxInfo>::set_logger(&mut defaults, None);
    let mut ppo = PpoReward::new(4, 0, 239);
    SaoeReward::set_logger(&mut ppo, None);

    let filtered = EnvironmentLogCollector::<i64, i64>::new(EnvironmentLogLevel::Info);
    filtered
        .add_scalar("debug", 1.0, EnvironmentLogLevel::Debug)
        .unwrap();
    filtered
        .add_step_count(
            "periodic",
            &BigInt::from(3_u8),
            EnvironmentLogLevel::Periodic,
        )
        .unwrap();
    filtered
        .add_observation("debug_obs", &1, EnvironmentLogLevel::Debug)
        .unwrap();
    filtered
        .add_policy_action("debug_act", &1, EnvironmentLogLevel::Debug)
        .unwrap();
    filtered
        .add_scalar("info", 2.0, EnvironmentLogLevel::Info)
        .unwrap();
    filtered
        .add_scalar("critical", 3.0, EnvironmentLogLevel::Critical)
        .unwrap();
    assert_eq!(filtered.snapshot().len(), 2);
    filtered.reset();
    assert!(filtered.snapshot().is_empty());

    let duplicate_sink = EnvironmentLogCollector::<i64, i64>::new(EnvironmentLogLevel::Debug);
    SaoeRewardLogSink::log_scalar(&duplicate_sink, "duplicate", 1.0).unwrap();
    assert!(SaoeRewardLogSink::log_scalar(&duplicate_sink, "duplicate", 2.0).is_err());

    let logger = EnvironmentLogCollector::<i64, i64>::new(EnvironmentLogLevel::Debug);
    let huge = BigInt::from(10_u8).pow(10_000_u32);
    assert!(matches!(
        logger.add_step_count("huge", &huge, EnvironmentLogLevel::Periodic),
        Err(EnvironmentLogError::StepCountNotRepresentable(value)) if value == huge
    ));

    let events = Arc::new(Mutex::new(Vec::new()));
    let attachments = Arc::new(AtomicUsize::new(0));
    let logger = Arc::new(EnvironmentLogCollector::new(EnvironmentLogLevel::Debug));
    let mut runner: Runner = EnvironmentStepRunner::new(
        Box::new(Simulator {
            value: 10,
            done: false,
            failure: None,
            conflict: None,
            state_calls: AtomicUsize::new(0),
            events: Arc::clone(&events),
            logger: None,
            attachments: Arc::clone(&attachments),
        }),
        Box::new(StateInterpreter {
            failure: None,
            events: Arc::clone(&events),
            attachments: Arc::clone(&attachments),
        }),
        Box::new(ActionInterpreter {
            failure: None,
            events,
            attachments,
        }),
        Some(Box::new(SaoeEnvironmentReward::new(Box::new(
            BoundReward { logger: None },
        )))),
        None,
        logger,
        Some(5_i64),
        10_i64,
    );
    let output = runner.step(3_i64).unwrap();
    assert_eq!(output.reward, 2.0);
    assert_eq!(
        output
            .info
            .logs
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["sim_metric", "bound_reward", "reward", "obs", "policy_act"]
    );

    let encoded = serde_json::to_string(runner.status()).unwrap();
    let decoded: EnvironmentStatus<i64, i64, i64> = serde_json::from_str(&encoded).unwrap();
    assert_eq!(&decoded, runner.status());
}
