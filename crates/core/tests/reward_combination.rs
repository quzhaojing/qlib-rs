#![allow(clippy::float_cmp)]

use std::{
    path::PathBuf,
    process::Command,
    sync::{Arc, Mutex},
};

use arrow_array::RecordBatch;
use arrow_schema::Schema;
use chrono::NaiveDateTime;
use domain_core::{
    Order, OrderDir, RewardCombination, SaoeBacktestData, SaoeReward, SaoeRewardError,
    SaoeRewardLogError, SaoeRewardLogSink, SaoeState, SaoeStateParts, WeightedSaoeReward,
};
use indexmap::IndexMap;
use ndarray::Array1;
use serde_json::Value;

type Events = Arc<Mutex<Vec<String>>>;
type Logs = Arc<Mutex<Vec<(String, f64)>>>;

struct FixedReward {
    name: &'static str,
    value: f64,
    events: Events,
    fail: bool,
}

impl SaoeReward for FixedReward {
    fn reward(&self, _state: &SaoeState) -> Result<f64, SaoeRewardError> {
        self.events.lock().unwrap().push(self.name.to_owned());
        if self.fail {
            Err(SaoeRewardError::InvalidOrderAmount(self.value))
        } else {
            Ok(self.value)
        }
    }
}

struct Logger {
    logs: Logs,
    fail_name: Option<&'static str>,
}

impl SaoeRewardLogSink for Logger {
    fn log_scalar(&self, name: &str, value: f64) -> Result<(), SaoeRewardLogError> {
        self.logs.lock().unwrap().push((name.to_owned(), value));
        if self.fail_name == Some(name) {
            Err(SaoeRewardLogError {
                message: name.to_owned(),
            })
        } else {
            Ok(())
        }
    }
}

fn state() -> SaoeState {
    let empty = RecordBatch::new_empty(Arc::new(Schema::empty()));
    SaoeState::new(SaoeStateParts {
        order: Order::new("A", 1.0, OrderDir::Buy, None, None),
        cur_time: NaiveDateTime::default(),
        cur_step: 0,
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

fn fixed(name: &'static str, value: f64, events: &Events, fail: bool) -> Arc<dyn SaoeReward> {
    Arc::new(FixedReward {
        name,
        value,
        events: Arc::clone(events),
        fail,
    })
}

fn sink(fail_name: Option<&'static str>) -> (Arc<dyn SaoeRewardLogSink>, Logs) {
    let logs = Arc::new(Mutex::new(Vec::new()));
    (
        Arc::new(Logger {
            logs: Arc::clone(&logs),
            fail_name,
        }),
        logs,
    )
}

fn map(entries: Vec<(&str, Arc<dyn SaoeReward>, f64)>) -> IndexMap<String, WeightedSaoeReward> {
    entries
        .into_iter()
        .map(|(name, reward, weight)| (name.to_owned(), (reward, weight)))
        .collect()
}

#[test]
fn ordered_weighting_and_logging_match_live_python_source() {
    let detached_events = Arc::new(Mutex::new(Vec::new()));
    let mut detached = FixedReward {
        name: "detached",
        value: 2.5,
        events: Arc::clone(&detached_events),
        fail: false,
    };
    SaoeReward::set_logger(&mut detached, None);
    assert_eq!(detached.reward(&state()).unwrap(), 2.5);
    assert_eq!(*detached_events.lock().unwrap(), ["detached"]);

    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/reward_combination_contract.py");
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/rl/reward.py");
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
    let expected: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        expected["child_failure"],
        serde_json::json!([true, ["first", "broken"], [["first", 6.0]]])
    );
    assert_eq!(
        expected["log_failure"],
        serde_json::json!([true, ["first"], [["first", 6.0]]])
    );
    assert_eq!(expected["missing"], serde_json::json!([true, ["first"]]));
    assert_eq!(
        expected["self_log"],
        serde_json::json!([true, ["inner"], []])
    );
    assert_eq!(expected["ieee"], serde_json::json!([true, true, true]));

    let events = Arc::new(Mutex::new(Vec::new()));
    let rewards = map(vec![
        ("alpha", fixed("alpha", 1.5, &events, false), 2.0),
        ("beta", fixed("beta", 4.0, &events, false), -0.5),
        ("gamma", fixed("gamma", -7.0, &events, false), 0.0),
    ]);
    let (logger, logs) = sink(None);
    let actual = RewardCombination::new(rewards)
        .with_logger(logger)
        .reward(&state())
        .unwrap();

    assert_eq!(actual, expected["success"].as_f64().unwrap());
    assert_eq!(
        *events.lock().unwrap(),
        expected["calls"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_owned())
            .collect::<Vec<_>>()
    );
    let expected_logs = expected["logs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| {
            (
                entry[0].as_str().unwrap().to_owned(),
                entry[1].as_f64().unwrap(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(*logs.lock().unwrap(), expected_logs);

    assert_eq!(
        RewardCombination::new(IndexMap::new())
            .reward(&state())
            .unwrap(),
        0.0
    );
    assert_eq!(expected["empty"].as_f64().unwrap(), 0.0);
}

#[test]
fn component_logger_and_attachment_failures_stop_at_the_python_stage() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let rewards = map(vec![
        ("first", fixed("first", 2.0, &events, false), 3.0),
        ("broken", fixed("broken", 4.0, &events, true), 1.0),
        ("never", fixed("never", 8.0, &events, false), 1.0),
    ]);
    let (logger, logs) = sink(None);
    assert!(matches!(
        RewardCombination::new(rewards)
            .with_logger(logger)
            .reward(&state()),
        Err(SaoeRewardError::InvalidOrderAmount(4.0))
    ));
    assert_eq!(*events.lock().unwrap(), ["first", "broken"]);
    assert_eq!(*logs.lock().unwrap(), [("first".to_owned(), 6.0)]);

    let events = Arc::new(Mutex::new(Vec::new()));
    let rewards = map(vec![
        ("first", fixed("first", 2.0, &events, false), 3.0),
        ("never", fixed("never", 8.0, &events, false), 1.0),
    ]);
    let (logger, logs) = sink(Some("first"));
    assert!(matches!(
        RewardCombination::new(rewards)
            .with_logger(logger)
            .reward(&state()),
        Err(SaoeRewardError::Log(_))
    ));
    assert_eq!(*events.lock().unwrap(), ["first"]);
    assert_eq!(*logs.lock().unwrap(), [("first".to_owned(), 6.0)]);

    let events = Arc::new(Mutex::new(Vec::new()));
    let rewards = map(vec![("first", fixed("first", 2.0, &events, false), 3.0)]);
    let mut combination = RewardCombination::new(rewards);
    assert!(matches!(
        combination.reward(&state()),
        Err(SaoeRewardError::MissingCombinationLogger)
    ));
    assert_eq!(*events.lock().unwrap(), ["first"]);

    let (logger, logs) = sink(None);
    combination.set_logger(Some(logger));
    assert_eq!(combination.reward(&state()).unwrap(), 6.0);
    combination.set_logger(None);
    assert!(matches!(
        combination.reward(&state()),
        Err(SaoeRewardError::MissingCombinationLogger)
    ));
    assert_eq!(*logs.lock().unwrap(), [("first".to_owned(), 6.0)]);

    let child_events = Arc::new(Mutex::new(Vec::new()));
    let child = fixed("inner", -1.0, &child_events, true);
    let (logger, outer_logs) = sink(None);
    assert!(matches!(
        RewardCombination::new(map(vec![("outer", child, 3.0)]))
            .with_logger(logger)
            .reward(&state()),
        Err(SaoeRewardError::InvalidOrderAmount(-1.0))
    ));
    assert!(outer_logs.lock().unwrap().is_empty());
}

#[test]
fn duplicate_keys_and_ieee_accumulation_follow_python_dict_and_float_rules() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut rewards = IndexMap::new();
    rewards.insert(
        "same".to_owned(),
        (fixed("replaced", 99.0, &events, false), 1.0),
    );
    rewards.insert(
        "second".to_owned(),
        (fixed("second", -1.0, &events, false), f64::INFINITY),
    );
    rewards.insert(
        "same".to_owned(),
        (fixed("same", 1.0, &events, false), f64::INFINITY),
    );
    let (logger, logs) = sink(None);
    let result = RewardCombination::new(rewards)
        .with_logger(logger)
        .reward(&state())
        .unwrap();
    assert!(result.is_nan());
    assert_eq!(*events.lock().unwrap(), ["same", "second"]);
    let logs = logs.lock().unwrap();
    assert_eq!(logs[0].0, "same");
    assert!(logs[0].1.is_infinite() && logs[0].1.is_sign_positive());
    assert_eq!(logs[1].0, "second");
    assert!(logs[1].1.is_infinite() && logs[1].1.is_sign_negative());
}
