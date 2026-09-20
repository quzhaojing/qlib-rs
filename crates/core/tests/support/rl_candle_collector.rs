use super::*;
use crate::rl_candle_network::{RecurrentConfig, RecurrentFeatures, RecurrentKind};
use crate::rl_candle_policy::{CandlePpo, CandlePpoConfig};
use crate::{
    EnvironmentPluginError, FiniteBackendStep, FiniteObservationPredicate, FiniteVectorBackend,
    TrainingPolicyMode, TrainingTrainerView, TrainingUpdateOptions, TrainingVesselBinding,
    TrainingVesselLog, TrainingVesselRunConfig, TrainingVesselRunner,
};
use candle_core::{DType, Device, Var};
use candle_nn::{VarBuilder, VarMap};
use rand::{SeedableRng, rngs::StdRng};
use serde_json::json;
use std::sync::Mutex;

#[path = "rl_candle_trainer_checkpoint.rs"]
mod trainer_checkpoint;

#[test]
fn production_default_vessel_trains_real_gru_and_restores_its_original_parameters() {
    use crate::rl_candle_checkpoint::CandlePolicySnapshot;
    use crate::{
        RlCheckpointState, RlTrainerConfig, RlTrainerControl, RlTrainerDriver, RlTrainerRuntime,
        RlTrainingSeed, RlTrainingVessel, TrainingVesselCheckpoint, TrainingVesselSeeds,
    };
    let policy = real_policy();
    let watched = policy.policy.actor.parameters()["layer_out.0.weight"].clone();
    let before = watched.to_vec2::<f32>().unwrap();
    let runner = TrainingVesselRunner::with_policy(
        Box::new(policy),
        Box::new(CandleCollectorFactory),
        TrainingVesselBinding::default(),
        TrainingVesselRunConfig {
            buffer_size: 7,
            episode_per_iter: 3,
            update_kwargs: IndexMap::from([
                ("batch_size".into(), json!(5)),
                ("repeat".into(), json!(2)),
            ]),
        },
        TrainingVesselLog::default(),
    );
    let queues = Arc::new(Mutex::new(Vec::new()));
    let captured = queues.clone();
    let factory = move |queue: &RlTrainingSeed<usize>, _: &mut RlTrainerControl| {
        assert!(queue.lock().unwrap().is_activated());
        let workers = queue.lock().unwrap().get().unwrap();
        assert_eq!(workers, 2);
        captured.lock().unwrap().push(queue.clone());
        Ok(FiniteVectorEnv::new(
            Box::new(Backend(Arc::new(Mutex::new(Trace {
                ticks: vec![0; workers],
                resets: vec![0; workers],
                ..Trace::default()
            })))),
            Box::new(Predicate),
            vec![],
        ))
    };
    let vessel = RlTrainingVessel::new(
        runner,
        TrainingVesselSeeds::new(Some(Arc::new(vec![2_usize])), None, None, None),
        Box::new(factory),
    );
    let mut driver = RlTrainerDriver::new(
        vessel,
        Arc::new(RlTrainerRuntime::new(None)),
        RlTrainerConfig {
            max_iters: Some(1.into()),
            val_every_n_iters: None,
        },
    );
    driver.fit(None).unwrap();
    let trained = watched.to_vec2::<f32>().unwrap();
    assert_ne!(trained, before);
    assert!(trained.iter().flatten().all(|value| value.is_finite()));
    let saved: TrainingVesselCheckpoint<CandlePolicySnapshot<()>> =
        driver.vessel.save_checkpoint().unwrap();
    Var::from_tensor(&watched)
        .unwrap()
        .set(&watched.zeros_like().unwrap())
        .unwrap();
    assert_ne!(watched.to_vec2::<f32>().unwrap(), trained);
    driver.vessel.load_checkpoint(&saved).unwrap();
    assert_eq!(watched.to_vec2::<f32>().unwrap(), trained);
    trainer_checkpoint::assert_same_snapshot(&driver.vessel.save_checkpoint().unwrap(), &saved);
    driver.fit(None).unwrap();
    assert_ne!(watched.to_vec2::<f32>().unwrap(), trained);
    let queues = queues.lock().unwrap();
    assert_eq!(queues.len(), 2);
    for queue in &*queues {
        assert_eq!(
            queue.lock().unwrap().get(),
            Err(crate::DataQueueError::Exhausted)
        );
    }
}

#[derive(Default)]
struct Trace {
    events: Vec<Value>,
    ticks: Vec<usize>,
    resets: Vec<usize>,
    fail: Option<String>,
}
type Shared = Arc<Mutex<Trace>>;
struct Backend(Shared);
struct Predicate;
impl FiniteObservationPredicate<RecurrentObservation> for Predicate {
    fn is_invalid(&mut self, _: &RecurrentObservation) -> Result<bool, EnvironmentPluginError> {
        Ok(false)
    }
}
fn observation(id: usize, tick: usize) -> RecurrentObservation {
    let device = &Device::Cpu;
    RecurrentObservation {
        data_processed: Tensor::new(
            &[[[id.to_f32().unwrap(), tick.to_f32().unwrap()], [1., 2.]]],
            device,
        )
        .unwrap(),
        cur_tick: Tensor::new(&[i64::try_from(tick % 3).unwrap()], device).unwrap(),
        cur_step: Tensor::new(&[i64::try_from(tick % 2).unwrap()], device).unwrap(),
        position_history: Tensor::ones((1, 2), DType::F32, device).unwrap(),
        target: Tensor::ones(1, DType::F32, device).unwrap(),
        num_step: Tensor::new(&[2_i64], device).unwrap(),
        acquiring: Tensor::new(&[i64::try_from(id % 2).unwrap()], device).unwrap(),
    }
}
fn pairs(obs: &RecurrentObservation) -> Vec<Vec<f32>> {
    obs.data_processed
        .to_vec3::<f32>()
        .unwrap()
        .into_iter()
        .map(|row| row[0].clone())
        .collect()
}
impl FiniteVectorBackend<RecurrentObservation, i64, f64, Value> for Backend {
    fn environment_count(&self) -> usize {
        self.0.lock().unwrap().ticks.len()
    }
    fn reset(
        &mut self,
        ids: &[usize],
    ) -> Result<Vec<Option<RecurrentObservation>>, EnvironmentPluginError> {
        let mut trace = self.0.lock().unwrap();
        let mut result = Vec::new();
        for &id in ids {
            trace.resets[id] += 1;
            trace.events.push(json!(["reset", id]));
            if trace.fail.as_deref() == Some("always_reset")
                || (id == 0
                    && ((trace.fail.as_deref() == Some("reset_finished") && trace.resets[id] == 2)
                        || (trace.fail.as_deref() == Some("reset_final") && trace.resets[id] == 3)))
            {
                return Err(EnvironmentPluginError::new("reset"));
            }
            trace.ticks[id] = 0;
            let exhausted = (trace.fail.as_deref() == Some("exhaust_finished")
                && trace.resets[id] >= 2)
                || (trace.fail.as_deref() == Some("exhaust_final") && trace.resets[id] >= 3);
            result.push((!exhausted).then(|| observation(id, 0)));
        }
        Ok(result)
    }
    fn step(
        &mut self,
        actions: &[i64],
        ids: &[usize],
    ) -> Result<Vec<FiniteBackendStep<RecurrentObservation, f64, Value>>, EnvironmentPluginError>
    {
        let mut trace = self.0.lock().unwrap();
        let mut result = Vec::new();
        for (&id, &action) in ids.iter().zip(actions) {
            trace.events.push(json!(["step", id, action]));
            if trace.fail.as_deref() == Some("step") {
                return Err(EnvironmentPluginError::new("step"));
            }
            trace.ticks[id] += 1;
            let done = trace.fail.as_deref() != Some("long_episode")
                && trace.ticks[id] == if id == 0 { 1 } else { 3 };
            let truncated = if trace.fail.as_deref() == Some("truncation") {
                json!("invalid")
            } else {
                json!(id == 1 && done)
            };
            let mut next = observation(id, trace.ticks[id]);
            if trace.fail.as_deref() == Some("bad_next") && id == 1 {
                next.target = next.target.to_dtype(DType::F64).unwrap();
            }
            result.push(FiniteBackendStep {
                observation: Some(next),
                reward: if trace.fail.as_deref() == Some("missing_reward") {
                    None
                } else {
                    Some((id + 1).to_f64().unwrap())
                },
                done,
                info: Some(json!({"TimeLimit.truncated": truncated})),
            });
        }
        Ok(result)
    }
}
fn environment(count: usize, fail: Option<&str>) -> (CandleCollectorEnvironment<Value>, Shared) {
    let trace = Arc::new(Mutex::new(Trace {
        ticks: vec![0; count],
        resets: vec![0; count],
        fail: fail.map(str::to_owned),
        ..Trace::default()
    }));
    (
        FiniteVectorEnv::new(
            Box::new(Backend(Arc::clone(&trace))),
            Box::new(Predicate),
            vec![],
        ),
        trace,
    )
}
struct Policy(Shared);
impl TrainingRunPolicy<CandleVectorReplayBuffer> for Policy {
    fn set_mode(&mut self, _: TrainingPolicyMode) -> Result<(), TrainingVesselRunError> {
        Ok(())
    }
    fn update(
        &mut self,
        _: u64,
        _: Option<&mut CandleVectorReplayBuffer>,
        _: &TrainingUpdateOptions,
    ) -> Result<TrainingVesselMetrics, TrainingVesselRunError> {
        panic!("collector must not learn")
    }
}
impl CandleCollectionPolicy for Policy {
    fn actions(&mut self, obs: &RecurrentObservation) -> Result<Tensor, String> {
        let mut trace = self.0.lock().unwrap();
        trace.events.push(json!(["act", pairs(obs)]));
        if trace.fail.as_deref() == Some("forward") {
            return Err("forward".into());
        }
        Ok(Tensor::zeros(obs.cur_tick.dims(), DType::I64, &Device::Cpu).unwrap())
    }
    fn exploration_noise(
        &mut self,
        actions: Tensor,
        _: &RecurrentObservation,
    ) -> Result<Tensor, String> {
        let mut trace = self.0.lock().unwrap();
        trace.events.push(json!(["noise"]));
        if trace.fail.as_deref() == Some("noise") {
            return Err("noise".into());
        }
        Ok(Tensor::new(
            actions
                .to_vec1::<i64>()
                .unwrap()
                .iter()
                .map(|value| value + 1)
                .collect::<Vec<_>>(),
            &Device::Cpu,
        )
        .unwrap())
    }
    fn map_action(&mut self, actions: &Tensor) -> Result<Vec<i64>, String> {
        let mut trace = self.0.lock().unwrap();
        let actions = actions.to_vec1::<i64>().unwrap();
        trace.events.push(json!(["map", actions]));
        if trace.fail.as_deref() == Some("map") {
            return Err("map".into());
        }
        Ok(actions.into_iter().map(|value| value + 10).collect())
    }
}
struct ConstantClock;
impl CandleCollectorClock for ConstantClock {
    fn seconds(&mut self) -> f64 {
        100.
    }
}

fn metric_json(value: &TrainingMetricValue) -> Value {
    match value {
        TrainingMetricValue::Scalar(TrainingMetricScalar::Integer(value)) => {
            json!(value.to_i64().unwrap())
        }
        TrainingMetricValue::Scalar(TrainingMetricScalar::Float(value)) => json!(value),
        TrainingMetricValue::Numeric(array)
            if array.data_type() == &arrow_schema::DataType::Int64 =>
        {
            json!(
                array
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            )
        }
        TrainingMetricValue::Numeric(array) => json!(
            array
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .values()
                .to_vec()
        ),
        TrainingMetricValue::Scalar(_) => panic!("unexpected metric type"),
    }
}
fn check_metrics(actual: &TrainingVesselMetrics, expected: &Value) {
    assert_eq!(
        actual.keys().map(String::as_str).collect::<Vec<_>>(),
        [
            "n/ep", "n/st", "rews", "lens", "idxs", "rew", "len", "rew_std", "len_std"
        ]
    );
    for (name, value) in actual {
        let actual = metric_json(value);
        if let (Some(left), Some(right)) = (actual.as_f64(), expected[name].as_f64()) {
            assert!((left - right).abs() < 1e-12, "{name}: {left} != {right}");
        } else {
            assert_eq!(actual, expected[name]);
        }
    }
}
fn check_state(collector: &CandleCollector, expected: &Value, trace: &Shared) {
    assert_eq!(json!(collector.statistics().steps), expected["steps"]);
    assert_eq!(json!(collector.statistics().episodes), expected["episodes"]);
    assert_eq!(
        collector.statistics().seconds.to_bits(),
        expected["seconds"].as_f64().unwrap().to_bits()
    );
    let children = collector.replay().children();
    assert_eq!(
        json!(
            children
                .iter()
                .map(crate::rl_replay_index::ReplayIndex::len)
                .collect::<Vec<_>>()
        ),
        expected["lengths"]
    );
    assert_eq!(
        json!(
            children
                .iter()
                .enumerate()
                .map(|(id, child)| id * child.capacity() + child.last_index())
                .collect::<Vec<_>>()
        ),
        expected["last"]
    );
    assert_eq!(
        json!(
            children
                .iter()
                .map(crate::rl_replay_index::ReplayIndex::next_write_index)
                .collect::<Vec<_>>()
        ),
        expected["next_write"]
    );
    assert_eq!(json!(trace.lock().unwrap().events), expected["events"]);
}
fn check_replay(collector: &CandleCollector, expected: &Value) {
    let batch = collector
        .replay()
        .sample_batch(0, &mut StdRng::seed_from_u64(1))
        .unwrap();
    assert_eq!(
        json!({"indices": batch.indices, "actions": batch.actions.to_vec1::<i64>().unwrap(),
        "rewards": batch.rewards.to_vec(), "terminated": batch.terminated.to_vec(),
        "truncated": batch.truncated.to_vec(), "obs": pairs(&batch.observations),
        "following": pairs(&batch.next_observations), "unfinished": batch.unfinished_indices}),
        *expected
    );
}

#[test]
fn collection_events_metrics_replay_and_failures_match_actual_tianshou() {
    let fixture: Value =
        serde_json::from_str(include_str!("../fixtures/rl_candle_collector.json")).unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 18);
    for case in cases {
        let (mut env, trace) = environment(2, case["fail"].as_str());
        let replay = case["capacity"].as_u64().map(|capacity| {
            CandleVectorReplayBuffer::new(usize::try_from(capacity).unwrap(), 2).unwrap()
        });
        let mut collector =
            CandleCollector::new(&mut env, replay, case["noise"].as_bool().unwrap()).unwrap();
        collector.set_clock(Box::new(ConstantClock));
        let mut policy = Policy(Arc::clone(&trace));
        for (limit, expected) in case["limits"]
            .as_array()
            .unwrap()
            .iter()
            .zip(case["output"].as_array().unwrap())
        {
            let count = limit[1].as_i64().unwrap();
            let limit = if limit[0] == "episodes" {
                TrainingCollectLimit::Episodes(Some(count))
            } else {
                TrainingCollectLimit::Steps(count.into())
            };
            let result = collector.collect(&mut policy, &mut env, limit);
            assert_eq!(result.is_err(), !expected["error"].is_null(), "{case}");
            if let Ok(metrics) = result {
                check_metrics(&metrics, &expected["metrics"]);
                check_replay(&collector, &expected["stored"]);
            }
            check_state(&collector, expected, &trace);
        }
    }
}

struct Trainer;
impl TrainingTrainerView for Trainer {
    fn current_iteration(&self) -> Result<BigInt, String> {
        Ok(0.into())
    }
    fn fast_dev_run(&self) -> Result<Option<i64>, String> {
        Ok(None)
    }
}
fn real_policy() -> CandlePpoVessel<StdRng> {
    let variables = VarMap::new();
    let builder = VarBuilder::from_varmap(&variables, DType::F32, &Device::Cpu);
    let extractor = Arc::new(
        RecurrentFeatures::new(
            RecurrentConfig {
                data_dim: 2,
                hidden_dim: 2,
                output_dim: 3,
                kind: RecurrentKind::Gru,
                layers: 1,
            },
            builder.pp("extractor"),
        )
        .unwrap(),
    );
    let policy = CandlePpo::new(extractor, 3, &builder, CandlePpoConfig::new(0.003)).unwrap();
    for tensor in policy
        .actor
        .parameters()
        .values()
        .chain(policy.critic.parameters().values())
    {
        Var::from_tensor(tensor)
            .unwrap()
            .set(&tensor.ones_like().unwrap().affine(0.1, 0.).unwrap())
            .unwrap();
    }
    CandlePpoVessel::new(policy, StdRng::seed_from_u64(47))
}

#[test]
fn typed_vessel_runs_actual_gru_collection_training_and_metric_logging() {
    let policy = real_policy();
    let watched = policy.policy.actor.parameters()["layer_out.0.weight"].clone();
    let before = watched.to_vec2::<f32>().unwrap();
    let trainer: Arc<dyn TrainingTrainerView> = Arc::new(Trainer);
    let (mut environment, trace) = environment(2, None);
    let mut runner = TrainingVesselRunner::with_policy(
        Box::new(policy),
        Box::new(CandleCollectorFactory),
        TrainingVesselBinding::default(),
        TrainingVesselRunConfig {
            buffer_size: 7,
            episode_per_iter: 3,
            update_kwargs: IndexMap::from([
                ("batch_size".into(), json!(2)),
                ("repeat".into(), json!(2)),
            ]),
        },
        TrainingVesselLog::default(),
    );
    runner.assign_trainer(&trainer);
    let snapshot = persist_native_policy_envelope(
        &crate::RlCheckpointState::save_checkpoint(&mut runner).unwrap(),
    );
    let metrics = runner.train(&mut environment).unwrap().unwrap();
    assert_eq!(metric_json(&metrics["n/ep"]), json!(3));
    assert_eq!(metric_json(&metrics["n/st"]), json!(5));
    for name in ["loss", "loss/clip", "loss/vf", "loss/ent"] {
        let values = metric_json(&metrics[name]);
        assert_eq!(values.as_array().unwrap().len(), 4);
        assert!(
            values
                .as_array()
                .unwrap()
                .iter()
                .all(|value| value.as_f64().unwrap().is_finite())
        );
    }
    assert_ne!(watched.to_vec2::<f32>().unwrap(), before);
    let mut corrupt = snapshot.clone();
    corrupt.policy.tensors.clear();
    let after = watched.to_vec2::<f32>().unwrap();
    assert!(matches!(
        runner.load_state_dict(&corrupt),
        Err(crate::TrainingVesselStateError::Load(_))
    ));
    assert_eq!(watched.to_vec2::<f32>().unwrap(), after);
    assert!(crate::RlCheckpointState::load_checkpoint(&mut runner, &corrupt).is_err());
    crate::RlCheckpointState::load_checkpoint(&mut runner, &snapshot).unwrap();
    assert_eq!(watched.to_vec2::<f32>().unwrap(), before);
    let actions: Vec<_> = trace
        .lock()
        .unwrap()
        .events
        .iter()
        .filter(|event| event[0] == "step")
        .map(|event| event[2].as_i64().unwrap())
        .collect();
    assert_eq!(actions.len(), 5);
    assert!(actions.iter().all(|action| (0..3).contains(action)));
    assert!(!environment.is_collector_guarded());
}

#[test]
fn dqn_vessel_collects_with_independent_gru_target_and_scalar_loss() {
    use crate::rl_candle_dqn_policy::{CandleDqn, CandleDqnConfig};
    use crate::rl_candle_dqn_vessel::CandleDqnVessel;
    let parameters = VarMap::new();
    let builder = VarBuilder::from_varmap(&parameters, DType::F32, &Device::Cpu);
    let features = Arc::new(
        RecurrentFeatures::new(
            RecurrentConfig {
                data_dim: 2,
                hidden_dim: 2,
                output_dim: 3,
                kind: RecurrentKind::Gru,
                layers: 1,
            },
            builder.pp("extractor"),
        )
        .unwrap(),
    );
    let mut policy = CandleDqn::new(
        features,
        3,
        &builder,
        CandleDqnConfig {
            target_update_frequency: 2,
            steps: 3,
            ..CandleDqnConfig::new(0.003)
        },
    )
    .unwrap();
    for variable in policy.optimizer.parameters() {
        variable
            .set(&variable.ones_like().unwrap().affine(0.1, 0.).unwrap())
            .unwrap();
    }
    policy.sync_weight().unwrap();
    policy.set_epsilon(0.5);
    let old = policy.target_model().unwrap().parameters()["layer_out.0.weight"].clone();
    let watched = policy.model.parameters()["layer_out.0.weight"].clone();
    assert_ne!(old.id(), watched.id());
    let before = watched.to_vec2::<f32>().unwrap();
    let old_before = old.to_vec2::<f32>().unwrap();
    let runtime = CandleDqnVessel::new(policy, StdRng::seed_from_u64(47));
    let trainer: Arc<dyn TrainingTrainerView> = Arc::new(Trainer);
    let (mut env, trace) = environment(2, None);
    let mut runner = TrainingVesselRunner::with_policy(
        Box::new(runtime),
        Box::new(CandleCollectorFactory),
        TrainingVesselBinding::default(),
        TrainingVesselRunConfig {
            buffer_size: 7,
            episode_per_iter: 3,
            update_kwargs: IndexMap::new(),
        },
        TrainingVesselLog::default(),
    );
    runner.assign_trainer(&trainer);
    let snapshot = persist_native_policy_envelope(&runner.state_dict().unwrap());
    let metrics = runner.train(&mut env).unwrap().unwrap();
    assert_eq!(metric_json(&metrics["n/st"]), json!(5));
    assert_eq!(metric_json(&metrics["n/ep"]), json!(3));
    assert!(
        matches!(&metrics["loss"], TrainingMetricValue::Scalar(TrainingMetricScalar::Float(loss)) if loss.is_finite())
    );
    assert_ne!(watched.to_vec2::<f32>().unwrap(), before);
    assert_eq!(old.to_vec2::<f32>().unwrap(), old_before);
    let mut corrupt = snapshot.clone();
    corrupt.policy.tensors.clear();
    let after = watched.to_vec2::<f32>().unwrap();
    assert!(matches!(
        runner.load_state_dict(&corrupt),
        Err(crate::TrainingVesselStateError::Load(_))
    ));
    assert_eq!(watched.to_vec2::<f32>().unwrap(), after);
    runner.load_state_dict(&snapshot).unwrap();
    assert_eq!(watched.to_vec2::<f32>().unwrap(), before);
    assert_eq!(old.to_vec2::<f32>().unwrap(), old_before);
    assert!(!env.is_collector_guarded());
    assert_eq!(
        trace
            .lock()
            .unwrap()
            .events
            .iter()
            .filter(|v| v[0] == "step")
            .count(),
        5
    );
}

fn persist_native_policy_envelope(
    state: &crate::TrainingVesselCheckpoint<crate::rl_candle_checkpoint::CandlePolicySnapshot<()>>,
) -> crate::TrainingVesselCheckpoint<crate::rl_candle_checkpoint::CandlePolicySnapshot<()>> {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("native-policy.bin");
    std::fs::write(&path, bincode::serialize(state).unwrap()).unwrap();
    let restored = bincode::deserialize(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(*state, restored);
    restored
}

#[test]
fn reset_limits_empty_batches_and_metadata_validation_keep_explicit_boundaries() {
    assert!(!().time_limit_truncated().unwrap());
    assert!(!json!({}).time_limit_truncated().unwrap());
    assert!(
        json!({"TimeLimit.truncated": true})
            .time_limit_truncated()
            .unwrap()
    );
    assert!(
        json!({"TimeLimit.truncated": 1})
            .time_limit_truncated()
            .is_err()
    );
    assert!(stack_observations(&[]).is_err());
    assert!(stack_observations(&[None]).is_err());
    let (mut env, trace) = environment(2, None);
    let mut policy = Policy(Arc::clone(&trace));
    let mut collector = CandleCollector::new(&mut env, None, false).unwrap();
    assert!(
        collector
            .collect(&mut policy, &mut env, TrainingCollectLimit::Episodes(None))
            .is_err()
    );
    collector
        .collect(&mut policy, &mut env, TrainingCollectLimit::Steps(2.into()))
        .unwrap();
    assert_eq!(collector.statistics().steps, 2);
    assert!(collector.statistics().seconds > 0.);
    let saved = collector.statistics().clone();
    trace.lock().unwrap().fail = Some("always_reset".into());
    assert!(collector.reset(&mut env, true).is_err());
    assert_eq!(collector.statistics(), &saved);
    assert_eq!(collector.replay().len(), 2);
    trace.lock().unwrap().fail = None;
    collector.reset(&mut env, false).unwrap();
    assert_eq!(
        collector.statistics(),
        &CandleCollectionStatistics::default()
    );
    assert_eq!(collector.replay().len(), 2);
    collector.reset_buffer(true);
    assert!(collector.replay().is_empty());
    collector.reset(&mut env, true).unwrap();
    let (mut other, _) = environment(1, None);
    assert!(
        collector
            .collect(
                &mut policy,
                &mut other,
                TrainingCollectLimit::Steps(1.into())
            )
            .is_err()
    );
    let small = CandleVectorReplayBuffer::new(1, 1).unwrap();
    assert!(CandleCollector::new(&mut env, Some(small), false).is_err());
    let (mut empty, _) = environment(0, None);
    assert!(CandleCollector::new(&mut empty, None, false).is_err());
}

#[test]
fn exhausted_collection_retains_replay_and_only_reached_statistics() {
    for (failure, recorded_steps) in [("exhaust_finished", 0), ("exhaust_final", 1)] {
        let (mut env, trace) = environment(1, Some(failure));
        let mut collector = CandleCollector::new(&mut env, None, false).unwrap();
        collector.set_clock(Box::new(ConstantClock));
        let mut policy = Policy(Arc::clone(&trace));
        let result = env.collect_guarded_with(
            |env| collector.collect(&mut policy, env, TrainingCollectLimit::Episodes(Some(1))),
            |error| {
                matches!(
                    error,
                    TrainingVesselRunError::Environment(crate::FiniteVectorError::Exhausted)
                )
            },
        );
        assert!(result.unwrap().is_none());
        assert!(env.is_zombie());
        assert!(!env.is_collector_guarded());
        assert_eq!(collector.statistics().steps, recorded_steps);
        assert_eq!(collector.statistics().episodes, recorded_steps);
        assert_eq!(collector.replay().len(), 1);
        let stored = collector.replay().get(&[0]).unwrap();
        assert_eq!(stored.rewards.to_vec(), [1.]);
        assert_eq!(stored.terminated.to_vec(), [true]);
        assert_eq!(stored.actions.to_vec1::<i64>().unwrap(), [0]);
        assert_eq!(trace.lock().unwrap().resets, [recorded_steps + 2]);
    }
}

#[test]
fn real_vessel_exhaustion_skips_learning_in_train_validate_and_test() {
    for phase in ["train", "validate", "test"] {
        let policy = real_policy();
        let watched = policy.policy.actor.parameters()["layer_out.0.weight"].clone();
        let before = watched.to_vec2::<f32>().unwrap();
        let trainer: Arc<dyn TrainingTrainerView> = Arc::new(Trainer);
        let (mut env, trace) = environment(1, Some("exhaust_finished"));
        let mut runner = TrainingVesselRunner::with_policy(
            Box::new(policy),
            Box::new(CandleCollectorFactory),
            TrainingVesselBinding::default(),
            // Missing learning keywords would fail if collection incorrectly reached update.
            TrainingVesselRunConfig {
                episode_per_iter: 1,
                ..Default::default()
            },
            TrainingVesselLog::default(),
        );
        runner.assign_trainer(&trainer);
        let result = match phase {
            "train" => runner.train(&mut env),
            "validate" => runner.validate(&mut env),
            _ => runner.test(&mut env),
        };
        assert!(result.unwrap().is_none());
        assert_eq!(watched.to_vec2::<f32>().unwrap(), before);
        assert!(env.is_zombie());
        assert!(!env.is_collector_guarded());
        assert_eq!(trace.lock().unwrap().ticks, [0]);
        assert_eq!(trace.lock().unwrap().resets, [2]);
        assert_eq!(trace.lock().unwrap().events.len(), 3);
    }
}

#[test]
fn collection_without_completed_episodes_and_failed_rewards_preserves_state() {
    let (mut env, trace) = environment(2, Some("long_episode"));
    let mut collector = CandleCollector::new(&mut env, None, false).unwrap();
    let mut policy = Policy(Arc::clone(&trace));
    let metrics = collector
        .collect(&mut policy, &mut env, TrainingCollectLimit::Steps(2.into()))
        .unwrap();
    check_metrics(
        &metrics,
        &json!({"n/ep": 0, "n/st": 2, "rews": [], "lens": [],
        "idxs": [], "rew": 0., "len": 0., "rew_std": 0., "len_std": 0.}),
    );
    assert_eq!(collector.replay().unfinished_indices().unwrap(), [0, 1]);
    let retained = collector.replay;
    let mut restored = CandleCollector::new(&mut env, Some(retained), false).unwrap();
    assert_eq!(restored.replay().len(), 2);
    assert_eq!(
        restored.statistics(),
        &CandleCollectionStatistics::default()
    );
    let mismatched = observation(0, 0);
    let mut bad = observation(1, 0);
    bad.target = bad.target.to_dtype(DType::F64).unwrap();
    assert!(stack_observations(&[Some(mismatched), Some(bad)]).is_err());
    restored.observations.clear();
    let result = restored.collect(&mut policy, &mut env, TrainingCollectLimit::Steps(2.into()));
    assert!(
        matches!(result, Err(TrainingVesselRunError::Plugin { stage, .. }) if stage == "collector state")
    );

    let (mut env, trace) = environment(1, Some("missing_reward"));
    let mut collector = CandleCollector::new(&mut env, None, false).unwrap();
    let result = collector.collect(
        &mut Policy(Arc::clone(&trace)),
        &mut env,
        TrainingCollectLimit::Steps(1.into()),
    );
    assert!(
        matches!(result, Err(TrainingVesselRunError::Plugin { stage, .. }) if stage == "collector reward")
    );
    assert!(collector.replay().is_empty());
    assert_eq!(
        collector.statistics(),
        &CandleCollectionStatistics::default()
    );
    assert_eq!(trace.lock().unwrap().ticks, [1]);
    assert_eq!(trace.lock().unwrap().resets, [1]);
}

#[test]
fn replay_write_failure_keeps_environment_step_but_does_not_reset_or_count_it() {
    let (mut env, trace) = environment(1, None);
    let empty = CandleVectorReplayBuffer::new(0, 1).unwrap();
    let mut collector = CandleCollector::new(&mut env, Some(empty), false).unwrap();
    let result = collector.collect(
        &mut Policy(Arc::clone(&trace)),
        &mut env,
        TrainingCollectLimit::Steps(1.into()),
    );
    assert!(
        matches!(result, Err(TrainingVesselRunError::Plugin { stage, .. }) if stage == "collector replay")
    );
    assert_eq!(trace.lock().unwrap().ticks, [1]);
    assert_eq!(trace.lock().unwrap().resets, [1]);
    assert_eq!(
        collector.statistics(),
        &CandleCollectionStatistics::default()
    );
    assert!(collector.replay().is_empty());
    assert_eq!(
        pairs(collector.observations[0].as_ref().unwrap()),
        [vec![0., 0.]]
    );
}

struct SequenceClock(std::collections::VecDeque<f64>);
impl CandleCollectorClock for SequenceClock {
    fn seconds(&mut self) -> f64 {
        self.0.pop_front().unwrap()
    }
}

#[test]
fn elapsed_clock_and_checked_statistics_keep_source_mutation_order() {
    for elapsed in [-1., 2., f64::NAN] {
        let (mut env, trace) = environment(1, None);
        let mut collector = CandleCollector::new(&mut env, None, false).unwrap();
        collector.set_clock(Box::new(SequenceClock([0., elapsed].into())));
        collector
            .collect(
                &mut Policy(trace),
                &mut env,
                TrainingCollectLimit::Steps(1.into()),
            )
            .unwrap();
        if elapsed.is_nan() {
            assert!(collector.statistics().seconds.is_nan());
        } else {
            assert_eq!(
                collector.statistics().seconds.to_bits(),
                elapsed.max(1e-9).to_bits()
            );
        }
    }
    for step_overflow in [false, true] {
        let (mut env, trace) = environment(1, None);
        let mut collector = CandleCollector::new(&mut env, None, false).unwrap();
        if step_overflow {
            collector.statistics.steps = usize::MAX;
        } else {
            collector.statistics.episodes = usize::MAX;
        }
        let error = collector
            .collect(
                &mut Policy(trace),
                &mut env,
                TrainingCollectLimit::Steps(1.into()),
            )
            .err()
            .expect("overflow must fail collection");
        assert!(
            matches!(error, TrainingVesselRunError::Plugin { stage, .. } if stage == "collector statistics")
        );
        assert_eq!(
            collector.statistics.steps,
            if step_overflow { usize::MAX } else { 1 }
        );
        assert_eq!(
            collector.statistics.episodes,
            if step_overflow { 0 } else { usize::MAX }
        );
        assert_eq!(collector.statistics.seconds.to_bits(), 0_f64.to_bits());
        assert_eq!(collector.replay().len(), 1);
    }
    let mut progress = Progress {
        steps: usize::MAX,
        ..Default::default()
    };
    assert!(progress.record(&[false], &[]).is_err());
    assert_eq!(progress.steps, usize::MAX);
    // These guards are reachable on 64-bit platforms; narrower usize cannot overflow i64.
    if let Ok(too_large) = usize::try_from(u64::try_from(i64::MAX).unwrap() + 1) {
        let mut progress = Progress::default();
        let episode = ReplayEpisode {
            index: 0,
            reward: 3.,
            length: too_large,
            start: 0,
        };
        assert!(progress.record(&[true], &[episode]).is_err());
        assert_eq!(progress.rewards, [3.]);
        assert!(progress.lengths.is_empty());
        let episode = ReplayEpisode {
            index: 0,
            reward: 5.,
            length: 1,
            start: too_large,
        };
        assert!(progress.record(&[true], &[episode]).is_err());
        assert_eq!(progress.rewards, [3., 5.]);
        assert_eq!(progress.lengths, [1]);
        assert!(progress.starts.is_empty());
    }
}

#[test]
fn native_policy_defaults_and_factory_errors_are_typed() {
    assert!(
        !crate::RlLogInfo::<Value> { log: None }
            .time_limit_truncated()
            .unwrap()
    );
    let mut policy = real_policy();
    let actions = Tensor::new(&[2_i64, 1], &Device::Cpu).unwrap();
    assert_eq!(policy.map_action(&actions).unwrap(), [2, 1]);
    let noise = policy
        .exploration_noise(actions.clone(), &observation(0, 0))
        .unwrap();
    assert_eq!(noise.id(), actions.id());
    assert!(
        policy
            .map_action(&actions.to_dtype(DType::F32).unwrap())
            .is_err()
    );
    let mut invalid = observation(0, 0);
    invalid.data_processed = Tensor::zeros((1, 2, 7), DType::F32, &Device::Cpu).unwrap();
    assert!(policy.actions(&invalid).is_err());
    let factory: &mut dyn TrainingCollectorFactory<
        RecurrentObservation,
        i64,
        f64,
        Value,
        CandleVectorReplayBuffer,
        Value,
        CandlePpoVessel<StdRng>,
    > = &mut CandleCollectorFactory;
    assert!(factory.create_buffer(-1, 1).is_err());
    assert!(factory.create_buffer(1, 0).is_err());
    let (mut env, _) = environment(1, Some("always_reset"));
    assert!(
        factory
            .create_collector(&mut policy, &mut env, None, false)
            .is_err()
    );
}

#[test]
fn malformed_current_and_next_observation_batches_stop_at_the_correct_stage() {
    for bad_current in [false, true] {
        let (mut env, trace) = environment(2, Some("bad_next"));
        let mut collector = CandleCollector::new(&mut env, None, false).unwrap();
        if bad_current {
            let row = collector.observations[1].as_mut().unwrap();
            row.target = row.target.to_dtype(DType::F64).unwrap();
        }
        let result = collector.collect(
            &mut Policy(Arc::clone(&trace)),
            &mut env,
            TrainingCollectLimit::Steps(2.into()),
        );
        assert!(
            matches!(result, Err(TrainingVesselRunError::Plugin { stage, .. }) if stage == "collector observation")
        );
        assert!(collector.replay().is_empty());
        assert_eq!(
            collector.statistics(),
            &CandleCollectionStatistics::default()
        );
        assert_eq!(
            trace.lock().unwrap().ticks,
            vec![usize::from(!bad_current); 2]
        );
        assert_eq!(trace.lock().unwrap().resets, [1, 1]);
        let count = trace
            .lock()
            .unwrap()
            .events
            .iter()
            .filter(|event| event[0] == "act")
            .count();
        assert_eq!(count, usize::from(!bad_current));
    }
}
