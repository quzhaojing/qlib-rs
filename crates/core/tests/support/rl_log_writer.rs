#![allow(clippy::float_cmp)]

use super::*;
use crate::{FiniteDummyBackend, FiniteEnvironment, FiniteObservationPredicate, FiniteVectorEnv};
use serde_json::{Value, json};
use std::{
    path::PathBuf,
    process::Command,
    sync::{Arc, Mutex},
};

type Events = Arc<Mutex<Vec<Value>>>;
type Logger = dyn FiniteVectorLogger<i64, f64, RlLogInfo<Value>>;
fn fixture() -> Value {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .args([
            root.join("tests/fixtures/rl_log_writer_contract.py"),
            root.join("../../../qlib/qlib/rl/utils/log.py"),
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
fn float(value: f64) -> Value {
    if value.is_nan() {
        json!("nan")
    } else if value == f64::INFINITY {
        json!("inf")
    } else if value == f64::NEG_INFINITY {
        json!("-inf")
    } else {
        json!(value)
    }
}
fn value(value: &RlLogValue<Value>) -> Value {
    match value {
        RlLogValue::Float(v) => float(*v),
        RlLogValue::Other(v) => v.clone(),
    }
}
fn decode(value: &Value) -> RlLogValue<Value> {
    if value.is_f64() {
        RlLogValue::Float(value.as_f64().unwrap())
    } else {
        RlLogValue::Other(value.clone())
    }
}
fn contents(values: &RlLogContents<Value>) -> Value {
    Value::Object(values.iter().map(|(k, v)| (k.clone(), value(v))).collect())
}
fn state(state: &RlLogWriterState<Value>) -> Value {
    let mut active = state.active_env_ids.iter().copied().collect::<Vec<_>>();
    active.sort_unstable();
    json!({
        "episode_count":state.episode_count.to_i64().unwrap(),"step_count":state.step_count.to_i64().unwrap(),
        "global_step":state.global_step.to_i64().unwrap(),"global_episode":state.global_episode.to_i64().unwrap(),
        "active_env_ids":active,
        "episode_lengths":state.episode_lengths.iter().map(|(k,v)|(k.to_string(),json!(v.to_i64().unwrap()))).collect::<serde_json::Map<_,_>>(),
        "episode_rewards":state.episode_rewards.iter().map(|(k,v)|(k.to_string(),Value::Array(v.iter().map(|v|float(*v)).collect()))).collect::<serde_json::Map<_,_>>(),
        "episode_logs":state.episode_logs.iter().map(|(k,v)|(k.to_string(),Value::Array(v.iter().map(contents).collect()))).collect::<serde_json::Map<_,_>>()
    })
}
fn buffered(writer: &RlLogWriterState<Value>, buffer: &RlLogBufferState) -> Value {
    let mut result = state(writer);
    result["latest_metrics"] = serde_json::to_value(&buffer.latest_metrics).unwrap();
    result["aggregated_metrics"] = serde_json::to_value(&buffer.aggregated_metrics).unwrap();
    result
}
fn steps(fixture: &Value) -> Vec<(usize, f64, bool, RlLeveledLogs<Value>)> {
    fixture["steps"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| {
            (
                s[0].as_u64().unwrap().try_into().unwrap(),
                s[1].as_f64().unwrap(),
                s[2].as_bool().unwrap(),
                s[3].as_array()
                    .unwrap()
                    .iter()
                    .map(|e| {
                        (
                            e[0].as_str().unwrap().to_owned(),
                            RlLogEntry {
                                level: e[1][0].as_i64().unwrap(),
                                value: decode(&e[1][1]),
                            },
                        )
                    })
                    .collect(),
            )
        })
        .collect()
}
struct Hooks {
    events: Events,
    fail: Value,
}
impl RlLogWriterHooks<Value> for Hooks {
    fn log_step(
        &mut self,
        reward: f64,
        logs: &RlLogContents<Value>,
        s: &RlLogWriterState<Value>,
    ) -> Result<(), RlLogError> {
        self.events.lock().unwrap().push(json!([
            "step",
            reward,
            contents(logs),
            s.step_count.to_i64(),
            s.episode_count.to_i64()
        ]));
        if self.fail == "step" {
            Err(RlLogError::Hook("step".into()))
        } else {
            Ok(())
        }
    }
    fn log_episode(
        &mut self,
        length: &BigInt,
        rewards: &[f64],
        logs: &[RlLogContents<Value>],
        s: &RlLogWriterState<Value>,
    ) -> Result<(), RlLogError> {
        self.events.lock().unwrap().push(json!([
            "episode",
            length.to_i64(),
            rewards,
            logs.iter().map(contents).collect::<Vec<_>>(),
            s.step_count.to_i64(),
            s.episode_count.to_i64()
        ]));
        if self.fail == "episode" {
            Err(RlLogError::Hook("episode".into()))
        } else {
            Ok(())
        }
    }
}

#[test]
fn writer_events_counters_partial_failures_and_clear_match_live_source() {
    let f = fixture();
    for case in f["writer"].as_array().unwrap() {
        let events = Events::default();
        let mut writer = RlLogWriter::new(
            20,
            Hooks {
                events: events.clone(),
                fail: case["fail"].clone(),
            },
        );
        if case["missing"] != "unreset" {
            writer.on_env_reset(0);
            writer.on_env_reset(1);
        }
        if case["missing"] == "episode_rewards" {
            writer.state.episode_rewards.shift_remove(&0);
        }
        if case["missing"] == "episode_logs" {
            writer.state.episode_logs.shift_remove(&0);
        }
        let mut states = Vec::new();
        let mut error = None;
        for (id, reward, done, logs) in steps(&f) {
            let result = writer.on_env_step(
                id,
                reward,
                done,
                if case["missing"] == "log" {
                    None
                } else {
                    Some(&logs)
                },
            );
            if let Err(e) = result {
                assert!(!e.to_string().is_empty());
                assert_eq!(e, e.clone());
                error = Some(e);
                break;
            }
            states.push(state(writer.state_dict()));
        }
        assert_eq!(error.is_some(), !case["error"].is_null());
        assert_eq!(states, case["states"].as_array().unwrap().clone());
        assert_eq!(
            *events.lock().unwrap(),
            case["events"].as_array().unwrap().clone()
        );
        assert_eq!(state(writer.state_dict()), case["before_clear"]);
        writer.on_env_all_done().unwrap();
        writer.clear();
        assert_eq!(state(writer.state_dict()), case["after_clear"]);
        let checkpoint = writer.state_dict().clone();
        let restored: RlLogWriterState<Value> =
            serde_json::from_value(serde_json::to_value(&checkpoint).unwrap()).unwrap();
        assert_eq!(restored, checkpoint);
        writer.load_state_dict(restored);
        assert_eq!(state(writer.state_dict()), case["after_clear"]);
        assert!(format!("{:?}", writer.state_dict()).contains("episode_count"));
    }
}

#[test]
fn buffer_metrics_sparse_denominators_callbacks_and_restore_match_source() {
    let f = fixture();
    for case in f["buffer"].as_array().unwrap() {
        let events = Events::default();
        let captured = events.clone();
        let fail = case["fail"].clone();
        let mut writer = RlLogBuffer::new_buffer(
            20,
            move |event, w: &RlLogWriterState<Value>, b: &RlLogBufferState| {
                captured.lock().unwrap().push(json!([
                    event == RlLogBufferEvent::Episode,
                    event == RlLogBufferEvent::Collect,
                    buffered(w, b),
                    b.collect_metrics(&w.episode_count).unwrap()
                ]));
                if fail
                    == match event {
                        RlLogBufferEvent::Episode => "episode",
                        RlLogBufferEvent::Collect => "collect",
                    }
                {
                    Err("callback".into())
                } else {
                    Ok(())
                }
            },
        );
        assert_eq!(
            buffered(writer.state_dict(), writer.buffer_state()),
            case["initial"]
        );
        assert_eq!(
            writer.buffer_state().episode_metrics(),
            Err(RlLogError::NoEpisodeMetrics)
        );
        writer.on_env_reset(0);
        writer.on_env_reset(1);
        let mut error = None;
        for (id, reward, done, logs) in steps(&f) {
            if let Err(e) = writer.on_env_step(id, reward, done, Some(&logs)) {
                error = Some(e);
                break;
            }
        }
        if error.is_none() {
            error = writer.on_env_all_done().err();
        }
        assert_eq!(error.is_some(), !case["error"].is_null());
        if let Some(e) = error {
            assert!(!e.to_string().is_empty());
            assert_eq!(e, e.clone());
        }
        assert_eq!(
            *events.lock().unwrap(),
            case["events"].as_array().unwrap().clone()
        );
        assert_eq!(
            buffered(writer.state_dict(), writer.buffer_state()),
            case["before_clear"]
        );
        assert_eq!(
            serde_json::to_value(writer.buffer_state().episode_metrics().unwrap()).unwrap(),
            case["episode"]
        );
        assert_eq!(
            serde_json::to_value(
                writer
                    .buffer_state()
                    .collect_metrics(&writer.state_dict().episode_count)
                    .unwrap()
            )
            .unwrap(),
            case["collect"]
        );
        assert_eq!(
            json!(
                writer
                    .buffer_state()
                    .episode_metrics()
                    .unwrap()
                    .keys()
                    .collect::<Vec<_>>()
            ),
            case["latest_order"]
        );
        assert_eq!(
            json!(
                writer
                    .buffer_state()
                    .aggregated_metrics
                    .keys()
                    .collect::<Vec<_>>()
            ),
            case["aggregated_order"]
        );
        assert_buffer_restore(&mut writer, case);
    }
}

fn assert_buffer_restore<C>(writer: &mut RlLogBuffer<Value, C>, case: &Value)
where
    C: FnMut(RlLogBufferEvent, &RlLogWriterState<Value>, &RlLogBufferState) -> Result<(), String>,
{
    let restored_state = writer.state_dict().clone();
    let restored_buffer = writer.buffer_state().clone();
    writer.clear();
    assert_eq!(
        buffered(writer.state_dict(), writer.buffer_state()),
        case["after_clear"]
    );
    writer.load_buffer_state(restored_buffer, restored_state);
    assert_eq!(
        buffered(writer.state_dict(), writer.buffer_state()),
        case["restored"]
    );
    let saved = serde_json::to_value(writer.buffer_state()).unwrap();
    assert_eq!(
        &serde_json::from_value::<RlLogBufferState>(saved).unwrap(),
        writer.buffer_state()
    );
}

#[test]
fn aggregation_types_special_values_and_checkpoint_edges_are_explicit() {
    let f = fixture();
    for case in f["aggregation"].as_array().unwrap() {
        let values: Vec<_> = case["values"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(index, v)| {
                if case["float_mask"][index] == false {
                    RlLogValue::Other(v.clone())
                } else {
                    match v.as_str() {
                        Some("nan") => RlLogValue::Float(f64::NAN),
                        Some("inf") => RlLogValue::Float(f64::INFINITY),
                        Some("-inf") => RlLogValue::Float(f64::NEG_INFINITY),
                        _ => decode(v),
                    }
                }
            })
            .collect();
        let result = aggregate_rl_logs(&values, case["name"].as_str());
        if case["error"].is_null() {
            assert_eq!(value(&result.unwrap()), case["result"]);
        } else {
            assert_eq!(result, Err(RlLogError::EmptyAggregation));
        }
    }
    let opaque = Arc::new(vec![1, 2, 3]);
    let values = [RlLogValue::Other(opaque.clone()), RlLogValue::Float(1.0)];
    let RlLogValue::Other(output) = aggregate_rl_logs(&values, None).unwrap() else {
        panic!("identity");
    };
    assert!(Arc::ptr_eq(&output, &opaque));
    let mut buffer = RlLogBufferState::default();
    let huge = BigInt::from(10).pow(1000);
    assert!(buffer.collect_metrics(&huge).unwrap().is_empty());
    buffer.aggregated_metrics.insert("value".into(), 1.0);
    assert_eq!(
        buffer.collect_metrics(&huge),
        Err(RlLogError::EpisodeCountNotRepresentable(huge))
    );
    assert_eq!(
        buffer.collect_metrics(&BigInt::from(0)).unwrap()["value"],
        f64::INFINITY
    );
    assert_eq!(
        buffer.collect_metrics(&BigInt::from(-2)).unwrap()["value"],
        -0.5
    );
    assert!(serde_json::from_value::<RlLogBufferState>(json!({"aggregated_metrics":{}})).is_err());
    let empty = RlLogBufferState::default();
    assert_eq!(
        bincode::deserialize::<RlLogBufferState>(&bincode::serialize(&empty).unwrap()).unwrap(),
        empty
    );
    let mut writer = RlLogWriter::<Value>::new(i64::MIN, NoopRlLogWriterHooks);
    writer.on_env_reset(0);
    writer.state.global_step = BigInt::from(10).pow(30);
    writer
        .on_env_step(0, 0.0, true, Some(&IndexMap::new()))
        .unwrap();
    assert_eq!(writer.state.global_step, BigInt::from(10).pow(30) + 1);
    writer.on_env_all_done().unwrap();
    let encoded = bincode::serialize(writer.state_dict()).unwrap();
    assert_eq!(
        &bincode::deserialize::<RlLogWriterState<Value>>(&encoded).unwrap(),
        writer.state_dict()
    );
    writer.on_env_reset(0);
    assert!(writer.state.episode_logs[&0].is_empty());
    assert_eq!(writer.state.episode_count, BigInt::from(1));
}

struct Worker {
    count: usize,
}

#[test]
fn restored_counts_and_empty_collect_match_python_numeric_semantics() {
    let f = fixture();
    let buffer = RlLogBufferState {
        latest_metrics: None,
        aggregated_metrics: IndexMap::from([("value".into(), 1.0)]),
    };
    for case in f["count_edges"].as_array().unwrap() {
        let count: BigInt = case["count"].as_str().unwrap().parse().unwrap();
        let result = buffer.collect_metrics(&count);
        if case["error"].is_null() {
            assert_eq!(float(result.unwrap()["value"]), case["result"]["value"]);
        } else {
            assert_eq!(result, Err(RlLogError::EpisodeCountNotRepresentable(count)));
            assert_eq!(case["error"], "OverflowError");
        }
    }
    let mut calls = 0;
    let mut buffer = RlLogBuffer::new_buffer(
        20,
        |event, w: &RlLogWriterState<String>, b: &RlLogBufferState| {
            assert_eq!(event, RlLogBufferEvent::Collect);
            assert_eq!(w.episode_count, BigInt::default());
            assert!(b.latest_metrics.is_none());
            assert!(b.collect_metrics(&w.episode_count).unwrap().is_empty());
            calls += 1;
            Ok(())
        },
    );
    buffer.on_env_all_done().unwrap();
    drop(buffer);
    assert_eq!(calls, 1);
    let mut writer = RlLogWriter::new(20, NoopRlLogWriterHooks);
    writer.on_env_reset(3);
    let logs = IndexMap::from([(
        "text".into(),
        RlLogEntry {
            level: 20,
            value: RlLogValue::Other("opaque".to_owned()),
        },
    )]);
    writer.on_env_step(3, 2.0, true, Some(&logs)).unwrap();
    assert_eq!(
        &bincode::deserialize::<RlLogWriterState<String>>(
            &bincode::serialize(writer.state_dict()).unwrap()
        )
        .unwrap(),
        writer.state_dict()
    );
}
impl FiniteEnvironment<i64, i64, f64, RlLogInfo<Value>> for Worker {
    fn reset(&mut self) -> Result<i64, EnvironmentPluginError> {
        self.count += 1;
        Ok(if self.count <= 2 { 0 } else { -1 })
    }
    fn step(
        &mut self,
        _: &i64,
    ) -> Result<FiniteBackendStep<i64, f64, RlLogInfo<Value>>, EnvironmentPluginError> {
        Ok(FiniteBackendStep {
            observation: Some(1),
            reward: Some(10.0),
            done: true,
            info: Some(RlLogInfo {
                log: Some(IndexMap::from([(
                    "reward".into(),
                    RlLogEntry {
                        level: 20,
                        value: RlLogValue::Float(2.0),
                    },
                )])),
            }),
        })
    }
}
struct Predicate;
impl FiniteObservationPredicate<i64> for Predicate {
    fn is_invalid(&mut self, v: &i64) -> Result<bool, EnvironmentPluginError> {
        Ok(*v < 0)
    }
}

#[test]
fn finite_logger_adapter_drives_real_episodes_and_reports_failures() {
    let events = Events::default();
    let captured = events.clone();
    let buffer = RlLogBuffer::new_buffer(
        20,
        move |event, w: &RlLogWriterState<Value>, b: &RlLogBufferState| {
            captured.lock().unwrap().push(json!([
                format!("{event:?}"),
                w.global_episode.to_i64(),
                b.collect_metrics(&w.episode_count).unwrap()
            ]));
            Ok(())
        },
    );
    let backend = FiniteDummyBackend::new(vec![Box::new(Worker { count: 0 })]).unwrap();
    let mut env = FiniteVectorEnv::new(
        Box::new(backend),
        Box::new(Predicate),
        vec![Box::new(buffer)],
    );
    let result = env.collect_guarded(|env| {
        for _ in 0..2 {
            env.reset(None)?;
            env.step(&[0], None)?;
        }
        env.reset(None)
    });
    assert!(result.unwrap().is_none());
    assert_eq!(
        *events.lock().unwrap(),
        vec![
            json!(["Episode",1,{"reward":2.0}]),
            json!(["Episode",2,{"reward":2.0}]),
            json!(["Collect",2,{"reward":2.0}])
        ]
    );
    assert_eq!(env.unguarded_reset_warnings(), 0);
    let mut buffer = RlLogBuffer::new_buffer(
        20,
        |_: RlLogBufferEvent, _: &RlLogWriterState<Value>, _: &RlLogBufferState| {
            Err("failure".into())
        },
    );
    let logger: &mut Logger = &mut buffer;
    logger.on_all_ready().unwrap();
    logger.on_reset(0, &[Some(1)]).unwrap();
    let mut step = FiniteBackendStep::<i64, f64, RlLogInfo<Value>>::default();
    assert!(
        logger
            .on_step(0, &step)
            .unwrap_err()
            .to_string()
            .contains("no reward")
    );
    step.reward = Some(1.0);
    assert!(
        logger
            .on_step(0, &step)
            .unwrap_err()
            .to_string()
            .contains("no log")
    );
    step.info = Some(RlLogInfo { log: None });
    assert!(logger.on_step(0, &step).is_err());
    step.info = Some(RlLogInfo {
        log: Some(IndexMap::new()),
    });
    step.done = true;
    assert!(
        logger
            .on_step(0, &step)
            .unwrap_err()
            .to_string()
            .contains("Episode")
    );
    assert!(
        logger
            .on_all_done()
            .unwrap_err()
            .to_string()
            .contains("Collect")
    );
    assert!(buffer.buffer_state().episode_metrics().unwrap().is_empty());
    assert_eq!(buffer.state_dict().global_step, BigInt::from(3));
}
