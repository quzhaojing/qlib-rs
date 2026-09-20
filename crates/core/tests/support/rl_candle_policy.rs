use super::*;
use crate::rl_candle_network_fixture::{Record, builder};
use candle_core::Var;
use ndarray::array;
use rand::{RngCore, SeedableRng, rngs::StdRng};
use serde::Deserialize;
use std::sync::Mutex;

#[path = "rl_candle_policy_weights.rs"]
mod weights;

struct Features {
    parameters: IndexMap<String, Tensor>,
    calls: Mutex<Vec<usize>>,
}
impl CandleFeatureExtractor for Features {
    fn output_dim(&self) -> usize {
        2
    }
    fn parameters(&self) -> &IndexMap<String, Tensor> {
        &self.parameters
    }
    fn forward(&self, obs: &RecurrentObservation) -> candle_core::Result<Tensor> {
        self.calls.lock().unwrap().push(obs.data_processed.dim(0)?);
        obs.data_processed.broadcast_mul(&self.parameters["scale"])
    }
}

#[derive(Deserialize)]
struct Statistics {
    mean: f64,
    variance: f64,
    count: usize,
}
impl Statistics {
    fn compare(&self, actual: &ReturnStatistics, dtype: &str) {
        let tolerance = if dtype == "float64" { 1e-11 } else { 1e-7 };
        assert!((self.mean - actual.mean).abs() < tolerance);
        assert!((self.variance - actual.variance).abs() < tolerance);
        assert_eq!(self.count, actual.count);
    }
}

#[derive(Deserialize)]
struct Case {
    dtype: String,
    normalize: bool,
    recompute: bool,
    initial: IndexMap<String, Record>,
    observations: Record,
    next_observations: Record,
    indices: Vec<usize>,
    unfinished_indices: Vec<usize>,
    prepared: IndexMap<String, Record>,
    stats_before: Statistics,
    metrics: IndexMap<String, Vec<f64>>,
    final_prepared: IndexMap<String, Record>,
    stats_after: Statistics,
    final_weights: IndexMap<String, Record>,
    initialized: usize,
}
fn cases() -> Vec<Case> {
    #[derive(Deserialize)]
    struct Fixture {
        cases: Vec<Case>,
    }
    let fixture: Fixture =
        serde_json::from_str(include_str!("../fixtures/rl_candle_policy.json")).unwrap();
    assert_eq!(fixture.cases.len(), 8);
    fixture.cases
}

fn observation(data: Tensor) -> RecurrentObservation {
    let length = data.dim(0).unwrap();
    let floats = Tensor::ones(length, data.dtype(), data.device()).unwrap();
    let indices = Tensor::zeros(length, DType::I64, data.device()).unwrap();
    RecurrentObservation {
        data_processed: data,
        cur_tick: indices.clone(),
        cur_step: indices.clone(),
        position_history: floats.unsqueeze(1).unwrap(),
        target: floats,
        num_step: indices.clone(),
        acquiring: indices,
    }
}

fn rollout(case: &Case) -> CandlePpoRollout {
    CandlePpoRollout {
        observations: observation(case.observations.tensor()),
        next_observations: observation(case.next_observations.tensor()),
        actions: Tensor::new(&[0_i64, 1, 2, 0, 1], &Device::Cpu).unwrap(),
        rewards: array![0.2, -0.4, 0.8, 0.1, -0.3],
        terminated: array![false, true, false, false, false],
        truncated: array![false, false, false, true, false],
        bootstrap_valid: array![true, false, true, true, true],
        indices: case.indices.clone(),
        unfinished_indices: case.unfinished_indices.clone(),
        replay_weights: None,
    }
}

fn setup(case: &Case) -> (Arc<Features>, IndexMap<String, Tensor>, CandlePpoConfig) {
    let weights: IndexMap<_, _> = case
        .initial
        .iter()
        .map(|(name, record)| {
            (
                name.clone(),
                Var::from_tensor(&record.tensor())
                    .unwrap()
                    .as_tensor()
                    .clone(),
            )
        })
        .collect();
    let extractor = Arc::new(Features {
        parameters: IndexMap::from([("scale".into(), weights["actor.extractor.scale"].clone())]),
        calls: Mutex::default(),
    });
    let mut config = CandlePpoConfig::new(0.003);
    config.weight_decay = 0.1;
    config.gamma = 0.9;
    config.gae_lambda = 0.95;
    config.max_batch_size = 2;
    config.reward_normalization = case.normalize;
    config.loss.value_clip = case.normalize;
    config.max_grad_norm = Some(if case.normalize { 0.2 } else { 0. });
    config.recompute_advantage = case.recompute;
    (extractor, weights, config)
}

fn compare_prepared(expected: &IndexMap<String, Record>, actual: &PreparedPpoRollout<'_>) {
    for (name, tensor) in [
        ("old_values", &actual.targets.old_values),
        ("returns", &actual.targets.returns),
        ("advantages", &actual.targets.advantages),
        ("actions", &actual.actions),
        ("old_log_prob", &actual.old_log_prob),
    ] {
        expected[name].compare(tensor, name);
    }
}

#[test]
fn checkpoint_restores_parameters_without_rewinding_ppo_runtime() {
    use crate::rl_candle_vessel::CandlePpoVessel;
    let case = cases().remove(0);
    let (features, weights, config) = setup(&case);
    let policy = CandlePpo::new(features, 3, &builder(&weights), config).unwrap();
    let mut runtime = CandlePpoVessel::new(policy, StdRng::seed_from_u64(73));
    let snapshot = runtime.state_dict().unwrap();
    let rollout = rollout(&case);
    let mut batch = runtime.policy.process(&rollout, &mut runtime.rng).unwrap();
    runtime
        .policy
        .learn(&mut batch, 5, 1, &mut runtime.rng)
        .unwrap();
    runtime.policy.return_statistics = ReturnStatistics {
        mean: 4.,
        variance: 9.,
        count: 23,
    };
    runtime.policy.updating = true;
    let mut expected_rng = runtime.rng.clone();
    assert_eq!(runtime.policy.optimizer.initialized_parameter_count(), 5);
    runtime.load_state_dict(&snapshot).unwrap();
    for (name, expected) in &case.initial {
        expected.compare(
            &runtime.policy.policy_state(()).unwrap().variables()[name],
            name,
        );
    }
    assert_eq!(runtime.policy.optimizer.initialized_parameter_count(), 5);
    assert!(runtime.policy.is_updating());
    assert_eq!(
        runtime.policy.return_statistics,
        ReturnStatistics {
            mean: 4.,
            variance: 9.,
            count: 23
        }
    );
    assert_eq!(runtime.rng.next_u64(), expected_rng.next_u64());
}

#[test]
fn real_qlib_constructor_process_and_three_adam_repeats_match() {
    for case in cases() {
        let (features, weights, config) = setup(&case);
        let mut policy = CandlePpo::new(features.clone(), 3, &builder(&weights), config).unwrap();
        assert_eq!(policy.optimizer.parameters().len(), 5);
        let rollout = rollout(&case);
        let mut rng = StdRng::seed_from_u64(73);
        let mut batch = policy.process(&rollout, &mut rng).unwrap();
        compare_prepared(&case.prepared, &batch);
        case.stats_before
            .compare(&policy.return_statistics, &case.dtype);
        // Critic current/next are interleaved; actor capture runs afterwards.
        assert_eq!(*features.calls.lock().unwrap(), [2, 2, 3, 3, 2, 3]);
        assert!(!batch.actions.is_variable());
        let metrics = policy.learn(&mut batch, 5, 3, &mut rng).unwrap();
        assert_eq!(
            metrics.keys().collect::<Vec<_>>(),
            case.metrics.keys().collect::<Vec<_>>()
        );
        for (name, expected) in &case.metrics {
            let record = Record {
                shape: vec![3],
                dtype: "float64".into(),
                values: expected.clone(),
            };
            if case.dtype == "float64" {
                record.compare(
                    &Tensor::new(metrics[name].as_slice(), &Device::Cpu).unwrap(),
                    name,
                );
            } else {
                for (actual, expected) in metrics[name].iter().zip(expected) {
                    assert!(
                        (actual - expected).abs() < 1e-7 + 2e-5 * expected.abs(),
                        "{name}: {actual} != {expected}"
                    );
                }
            }
        }
        compare_prepared(&case.final_prepared, &batch);
        case.stats_after
            .compare(&policy.return_statistics, &case.dtype);
        for (name, expected) in &case.final_weights {
            expected.compare(&weights[name], name);
        }
        assert_eq!(
            policy.optimizer.initialized_parameter_count(),
            case.initialized
        );
    }
}

#[test]
fn zero_repeats_modes_minibatches_and_failures_preserve_state_boundaries() {
    let case = cases().remove(0);
    let (features, weights, mut config) = setup(&case);
    config.max_grad_norm = None;
    let mut policy = CandlePpo::new(features.clone(), 3, &builder(&weights), config).unwrap();
    let mut rollout = rollout(&case);
    let mut rng = StdRng::seed_from_u64(41);
    policy.set_mode(TrainingPolicyMode::Evaluation);
    let mut expected_rng = rng.clone();
    let state = Arc::new(7);
    let output = policy
        .forward(&rollout.observations, state.clone(), &mut rng)
        .unwrap();
    assert!(Arc::ptr_eq(&output.state, &state));
    assert_eq!(rng.next_u64(), expected_rng.next_u64());
    let mut batch = policy.process(&rollout, &mut rng).unwrap();
    expected_rng = rng.clone();
    let metrics = policy.learn(&mut batch, 0, 0, &mut rng).unwrap();
    assert_eq!(metrics.len(), 4);
    assert!(metrics.values().all(Vec::is_empty));
    assert_eq!(policy.optimizer.initialized_parameter_count(), 0);
    assert_eq!(rng.next_u64(), expected_rng.next_u64());
    assert!(matches!(
        policy.learn(&mut batch, 0, 1, &mut rng),
        Err(CandlePpoError::Batch(_))
    ));
    policy.set_mode(TrainingPolicyMode::Train);
    features.calls.lock().unwrap().clear();
    let metrics = policy.learn(&mut batch, 2, 2, &mut rng).unwrap();
    assert!(
        metrics
            .values()
            .all(|values| values.len() == 4 && values.iter().all(|v| v.is_finite()))
    );
    assert_eq!(*features.calls.lock().unwrap(), [2, 2, 3, 3, 2, 2, 3, 3]);
    assert_eq!(policy.optimizer.initialized_parameter_count(), 5);
    rollout.actions = Tensor::new(&[0.5_f32; 5], &Device::Cpu).unwrap();
    assert!(policy.process(&rollout, &mut rng).is_err());
    rollout.rewards = array![];
    assert!(policy.process(&rollout, &mut rng).is_err());
    assert!(matches!(
        positions_tensor(&[usize::MAX], &Device::Cpu),
        Err(CandlePpoError::IndexRange)
    ));
}

#[test]
fn invalid_configuration_and_missing_head_parameters_are_errors() {
    let case = cases().remove(0);
    let (features, weights, config) = setup(&case);
    for bad in [-1., 1.1, f64::NAN] {
        for changed in [
            CandlePpoConfig {
                gamma: bad,
                ..config
            },
            CandlePpoConfig {
                gae_lambda: bad,
                ..config
            },
        ] {
            assert!(matches!(
                CandlePpo::new(features.clone(), 3, &builder(&weights), changed),
                Err(CandlePpoError::Configuration(_))
            ));
        }
    }
    for dual in [0., 1., f64::NAN] {
        let changed = CandlePpoConfig {
            loss: PpoLossConfig {
                dual_clip: Some(dual),
                ..config.loss
            },
            ..config
        };
        assert!(CandlePpo::new(features.clone(), 3, &builder(&weights), changed).is_err());
    }
    let changed = CandlePpoConfig {
        loss: PpoLossConfig {
            value_clip: true,
            ..config.loss
        },
        ..config
    };
    assert!(CandlePpo::new(features.clone(), 3, &builder(&weights), changed).is_err());
    for name in ["actor.layer_out.0.weight", "critic.value_out.weight"] {
        let mut missing = weights.clone();
        missing.shift_remove(name);
        assert!(CandlePpo::new(features.clone(), 3, &builder(&missing), config).is_err());
    }
    let changed = CandlePpoConfig {
        learning_rate: -0.1,
        ..config
    };
    assert!(CandlePpo::new(features, 3, &builder(&weights), changed).is_err());
}

struct Replay {
    rollout: Option<CandlePpoRollout>,
    fail: Option<String>,
    events: Arc<Mutex<Vec<String>>>,
    weight_id: candle_core::TensorId,
}
impl CandlePpoReplay for Replay {
    fn sample(&mut self, size: u64, _rng: &mut dyn RngCore) -> Result<CandlePpoRollout, String> {
        assert_eq!(size, 0);
        self.events.lock().unwrap().push("sample".into());
        if self.fail.as_deref() == Some("sample") {
            return Err("sample".into());
        }
        Ok(self.rollout.take().unwrap())
    }
    fn update_weights(&mut self, indices: &[usize], weights: &Tensor) -> Result<(), String> {
        assert_eq!(indices, [0, 1, 2, 3, 4]);
        assert_eq!(weights.id(), self.weight_id);
        self.events.lock().unwrap().push("post-process".into());
        if self.fail.as_deref() == Some("post-process") {
            return Err("post-process".into());
        }
        Ok(())
    }
}
struct Scheduler {
    fail: bool,
    events: Arc<Mutex<Vec<String>>>,
}
impl CandlePpoScheduler for Scheduler {
    fn step(&mut self, optimizer: &mut CandleAdam) -> Result<(), String> {
        self.events.lock().unwrap().push("scheduler".into());
        assert_eq!(optimizer.initialized_parameter_count(), 5);
        optimizer.set_learning_rate(0.01).unwrap();
        if self.fail {
            Err("scheduler".into())
        } else {
            Ok(())
        }
    }
}

#[derive(Deserialize)]
struct UpdateCase {
    failure: Option<String>,
    initial: bool,
    events: Vec<(String, bool)>,
    updating: bool,
    error: Option<String>,
}

fn check_update_case(expected: &UpdateCase, case: &Case) {
    let (features, weights, config) = setup(case);
    let mut policy = CandlePpo::new(features, 3, &builder(&weights), config).unwrap();
    policy.updating = expected.initial;
    let mut rollout = rollout(case);
    if expected.failure.as_deref() == Some("process") {
        rollout.actions = Tensor::new(&[0.5_f32; 5], &Device::Cpu).unwrap();
    }
    let priorities = Tensor::new(&[0.1_f32; 5], &Device::Cpu).unwrap();
    let weight_id = priorities.id();
    rollout.replay_weights = Some(priorities);
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut buffer = Replay {
        rollout: Some(rollout),
        fail: expected.failure.clone(),
        events: events.clone(),
        weight_id,
    };
    let mut scheduler = Scheduler {
        fail: expected.failure.as_deref() == Some("scheduler"),
        events: events.clone(),
    };
    let buffer: Option<&mut dyn CandlePpoReplay> =
        if expected.failure.as_deref() == Some("no-buffer") {
            None
        } else {
            Some(&mut buffer)
        };
    let result = policy.update(
        buffer,
        Some(&mut scheduler),
        CandlePpoUpdate {
            sample_size: 0,
            batch_size: if expected.failure.as_deref() == Some("learn") {
                0
            } else {
                5
            },
            repeat: 1,
        },
        &mut StdRng::seed_from_u64(91),
    );
    assert_eq!(result.is_err(), expected.error.is_some());
    assert_eq!(policy.is_updating(), expected.updating);
    let expected_events: Vec<_> = expected
        .events
        .iter()
        .filter(|(stage, _)| !matches!(stage.as_str(), "process" | "learn"))
        .map(|(stage, _)| stage.clone())
        .collect();
    assert_eq!(*events.lock().unwrap(), expected_events);
    let learned = matches!(
        expected.failure.as_deref(),
        None | Some("post-process" | "scheduler")
    );
    assert_eq!(
        policy.optimizer.initialized_parameter_count(),
        if learned { 5 } else { 0 }
    );
    let processed = !matches!(expected.failure.as_deref(), Some("sample" | "no-buffer"));
    assert_eq!(
        policy.return_statistics.count,
        if processed { 5 } else { 0 }
    );
    if expected.failure.as_deref() == Some("no-buffer") {
        assert!(result.unwrap().is_empty());
    }
    if expected_events
        .last()
        .is_some_and(|event| event == "scheduler")
    {
        assert_eq!(
            policy.optimizer.learning_rate().to_bits(),
            0.01_f64.to_bits()
        );
    }
}

#[test]
fn real_base_update_failure_and_success_states_match_fourteen_source_cases() {
    #[derive(Deserialize)]
    struct Fixture {
        cases: Vec<UpdateCase>,
    }
    let fixture: Fixture =
        serde_json::from_str(include_str!("../fixtures/rl_policy_update.json")).unwrap();
    assert_eq!(fixture.cases.len(), 14);
    let case = cases()
        .into_iter()
        .find(|case| case.normalize && !case.recompute)
        .unwrap();
    for expected in fixture.cases {
        check_update_case(&expected, &case);
    }
}

struct OrdinaryReplay(Option<CandlePpoRollout>);
impl CandlePpoReplay for OrdinaryReplay {
    fn sample(&mut self, size: u64, _rng: &mut dyn RngCore) -> Result<CandlePpoRollout, String> {
        assert_eq!(size, 17);
        Ok(self.0.take().unwrap())
    }
}

#[test]
fn update_without_scheduler_and_without_priority_support_is_valid() {
    let case = cases().remove(0);
    for has_weights in [false, true] {
        let (features, weights, config) = setup(&case);
        let mut policy = CandlePpo::new(features, 3, &builder(&weights), config).unwrap();
        let mut rollout = rollout(&case);
        rollout.replay_weights =
            has_weights.then(|| Tensor::ones(5, DType::F32, &Device::Cpu).unwrap());
        let mut buffer = OrdinaryReplay(Some(rollout));
        let result = policy
            .update(
                Some(&mut buffer),
                None,
                CandlePpoUpdate {
                    sample_size: 17,
                    batch_size: 2,
                    repeat: 0,
                },
                &mut StdRng::seed_from_u64(19),
            )
            .unwrap();
        assert_eq!(result.len(), 4);
        assert!(result.values().all(Vec::is_empty));
        assert!(!policy.is_updating());
        assert_eq!(policy.optimizer.initialized_parameter_count(), 0);
    }
}

#[test]
fn preprocessing_and_learning_errors_leave_only_completed_side_effects() {
    let case = cases().into_iter().find(|case| case.normalize).unwrap();
    let (features, weights, config) = setup(&case);
    let mut policy = CandlePpo::new(features.clone(), 3, &builder(&weights), config).unwrap();
    let mut input = rollout(&case);
    let mut rng = StdRng::seed_from_u64(4);
    input.terminated = array![false];
    assert!(matches!(
        policy.process(&input, &mut rng),
        Err(CandlePpoError::Returns(_))
    ));
    assert_eq!(policy.return_statistics.count, 0);
    input = rollout(&case);
    input.next_observations.cur_tick = Tensor::new(0_i64, &Device::Cpu).unwrap();
    assert!(matches!(
        policy.process(&input, &mut rng),
        Err(CandlePpoError::Tensor(_))
    ));
    assert_eq!(policy.return_statistics.count, 0);
    input = rollout(&case);
    input.actions = Tensor::new(&[0_i64], &Device::Cpu).unwrap();
    assert!(matches!(
        policy.process(&input, &mut rng),
        Err(CandlePpoError::Tensor(_))
    ));
    assert_eq!(policy.return_statistics.count, 5);
    input = rollout(&case);
    let mut prepared = policy.process(&input, &mut rng).unwrap();
    prepared.old_log_prob = prepared.old_log_prob.to_dtype(DType::F64).unwrap();
    assert!(matches!(
        policy.learn(&mut prepared, 5, 1, &mut rng),
        Err(CandlePpoError::Tensor(_))
    ));
    assert_eq!(policy.optimizer.initialized_parameter_count(), 0);
    let config = CandlePpoConfig {
        max_batch_size: 0,
        ..config
    };
    let mut policy = CandlePpo::new(features, 3, &builder(&weights), config).unwrap();
    assert!(matches!(
        policy.process(&input, &mut rng),
        Err(CandlePpoError::Batch(_))
    ));
    assert_eq!(policy.return_statistics.count, 0);
}
