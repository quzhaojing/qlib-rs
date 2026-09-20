use super::*;
use crate::rl_candle_network_fixture::Record;
use crate::rl_candle_replay::CandleReplayTransition;
use candle_core::{DType, Device};
use ndarray::{Array1, array};
use rand::{SeedableRng, rngs::StdRng};
use serde::Deserialize;
use std::sync::{
    Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

#[derive(Default)]
struct Hooks {
    calls: AtomicUsize,
    events: Mutex<Vec<String>>,
    modes: Mutex<Vec<(candle_core::TensorId, bool)>>,
    fail: Option<String>,
}
impl Hooks {
    fn enter(&self, name: &str) -> Result<(), String> {
        self.events.lock().unwrap().push(name.into());
        if self.fail.as_deref() == Some(name) {
            Err(name.into())
        } else {
            Ok(())
        }
    }
}
struct Features {
    parameters: IndexMap<String, Tensor>,
    mode: AtomicBool,
    hooks: Arc<Hooks>,
    wrong_copy: bool,
}
impl CandleFeatureExtractor for Features {
    fn output_dim(&self) -> usize {
        2
    }
    fn parameters(&self) -> &IndexMap<String, Tensor> {
        &self.parameters
    }
    fn forward(&self, observation: &RecurrentObservation) -> candle_core::Result<Tensor> {
        self.hooks.modes.lock().unwrap().push((
            self.parameters["scale"].id(),
            self.mode.load(Ordering::Relaxed),
        ));
        if self.hooks.calls.fetch_add(1, Ordering::Relaxed) == 2 {
            self.hooks.enter("learn").map_err(candle_core::Error::Msg)?;
        }
        observation
            .data_processed
            .broadcast_mul(&self.parameters["scale"])
    }
    fn rebuild(
        &self,
        parameters: IndexMap<String, Tensor>,
    ) -> candle_core::Result<Arc<dyn CandleFeatureExtractor>> {
        Ok(Arc::new(Self {
            parameters: if self.wrong_copy {
                self.parameters.clone()
            } else {
                parameters
            },
            mode: AtomicBool::new(self.mode.load(Ordering::Relaxed)),
            hooks: self.hooks.clone(),
            wrong_copy: self.wrong_copy,
        }))
    }
    fn set_mode(&self, mode: TrainingPolicyMode) {
        self.mode
            .store(mode == TrainingPolicyMode::Train, Ordering::Relaxed);
    }
}

#[derive(Deserialize)]
struct Step {
    returns: Record,
    loss: f64,
    td: Record,
    weights: IndexMap<String, Record>,
    iteration: u64,
    target_grad_absent: bool,
}
#[derive(Deserialize)]
struct Case {
    frequency: i64,
    double: bool,
    huber: bool,
    independent: bool,
    initial: IndexMap<String, Record>,
    observations: Record,
    next_observations: Record,
    steps: Vec<Step>,
    mode: Vec<bool>,
    max_actions: usize,
}
#[derive(Deserialize)]
struct Lifecycle {
    failure: Option<String>,
    events: Vec<String>,
    error: Option<String>,
    updating: bool,
    iteration: u64,
}
#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
    lifecycle: Vec<Lifecycle>,
    weight_cases: Vec<WeightCase>,
}
#[derive(Deserialize)]
struct WeightCase {
    frequency: i64,
    initial: IndexMap<String, Record>,
    error: Option<String>,
    input_keys: Vec<String>,
    #[serde(rename = "final")]
    final_weights: IndexMap<String, Record>,
}
fn fixture() -> Fixture {
    serde_json::from_str(include_str!("../fixtures/rl_candle_dqn_policy.json")).unwrap()
}
fn config(case: &Case) -> CandleDqnConfig {
    CandleDqnConfig {
        weight_decay: 0.1,
        gamma: 0.9,
        steps: 3,
        target_update_frequency: case.frequency,
        is_double: case.double,
        huber: case.huber,
        ..CandleDqnConfig::new(0.003)
    }
}
fn weights(case: &Case) -> PolicyWeights<Tensor> {
    PolicyWeights {
        weights: case
            .initial
            .iter()
            .map(|(name, value)| (name.clone(), Arc::new(value.tensor())))
            .collect(),
        metadata: (),
    }
}
fn setup(case: &Case, hooks: Arc<Hooks>) -> (Arc<Features>, IndexMap<String, Tensor>) {
    let parameters: IndexMap<_, _> = case
        .initial
        .iter()
        .filter(|(name, _)| name.starts_with("model."))
        .map(|(name, record)| {
            (
                name.clone(),
                Var::from_tensor(&record.tensor()).unwrap().into_inner(),
            )
        })
        .collect();
    let features = Arc::new(Features {
        parameters: IndexMap::from([("scale".into(), parameters["model.extractor.scale"].clone())]),
        mode: AtomicBool::new(true),
        hooks,
        wrong_copy: false,
    });
    (features, parameters)
}
fn builder(parameters: &IndexMap<String, Tensor>) -> VarBuilder<'static> {
    VarBuilder::from_tensors(
        parameters.clone().into_iter().collect(),
        parameters["model.extractor.scale"].dtype(),
        &Device::Cpu,
    )
}
fn policy(case: &Case, hooks: Arc<Hooks>) -> CandleDqn {
    let (features, parameters) = setup(case, hooks);
    CandleDqn::new_with_weights(
        features,
        3,
        &builder(&parameters),
        config(case),
        &mut weights(case),
    )
    .unwrap()
}
fn observation(data: Tensor) -> RecurrentObservation {
    let size = data.dim(0).unwrap();
    let f = Tensor::zeros(size, data.dtype(), data.device()).unwrap();
    let i = Tensor::zeros(size, DType::I64, data.device()).unwrap();
    RecurrentObservation {
        data_processed: data,
        cur_tick: i.clone(),
        cur_step: i.clone(),
        position_history: f.unsqueeze(1).unwrap(),
        target: f,
        num_step: i.clone(),
        acquiring: i,
    }
}
fn replay(case: &Case) -> CandleReplayBuffer {
    let mut buffer = CandleReplayBuffer::new(8);
    let obs = observation(case.observations.tensor());
    let next = observation(case.next_observations.tensor());
    for (index, reward) in [0.2, -0.4, 0.8, 0.1, -0.3].into_iter().enumerate() {
        let id = Tensor::new(&[i64::try_from(index).unwrap()], &Device::Cpu).unwrap();
        buffer
            .add(&CandleReplayTransition {
                observation: obs.select_batch(&id).unwrap(),
                next_observation: next.select_batch(&id).unwrap(),
                action: Tensor::new(&[i64::try_from(index % 3).unwrap()], &Device::Cpu).unwrap(),
                reward,
                terminated: index == 1,
                truncated: index == 3,
            })
            .unwrap();
    }
    buffer
}
fn compare_weights(policy: &CandleDqn, expected: &IndexMap<String, Record>) {
    let state = policy.policy_state(()).unwrap();
    assert_eq!(
        state.variables().keys().collect::<Vec<_>>(),
        expected.keys().collect::<Vec<_>>()
    );
    for (name, value) in state.variables() {
        expected[name].compare(value, name);
    }
}

#[test]
fn checkpoint_restores_both_models_without_rewinding_dqn_runtime() {
    use crate::rl_candle_dqn_vessel::CandleDqnVessel;
    let case = fixture()
        .cases
        .into_iter()
        .find(|case| case.frequency == 2)
        .unwrap();
    let mut runtime = CandleDqnVessel::new(
        policy(&case, Arc::new(Hooks::default())),
        StdRng::seed_from_u64(19),
    );
    let snapshot = runtime.state_dict().unwrap();
    let buffer = replay(&case);
    let mut batch: CandleDqnRollout = buffer.get(&[0, 1, 2, 3, 4]).unwrap().into();
    runtime.policy.process(&mut batch, &buffer).unwrap();
    runtime.policy.learn(&mut batch).unwrap();
    runtime.policy.iteration = 17;
    runtime.policy.set_epsilon(0.7);
    runtime.policy.updating = true;
    let mut expected_rng = runtime.rng.clone();
    runtime.load_state_dict(&snapshot).unwrap();
    compare_weights(&runtime.policy, &case.initial);
    assert_eq!(runtime.policy.iteration(), 17);
    assert_eq!(runtime.policy.epsilon.to_bits(), 0.7_f64.to_bits());
    assert_eq!(runtime.policy.optimizer.initialized_parameter_count(), 3);
    assert_eq!(runtime.policy.action_count(), Some(3));
    assert!(runtime.policy.is_updating());
    assert_eq!(runtime.rng.next_u64(), expected_rng.next_u64());
    let (features, _) = setup(&case, Arc::new(Hooks::default()));
    runtime.policy.model =
        DqnModel::new(features, 3, VarBuilder::zeros(DType::F32, &Device::Cpu)).unwrap();
    assert!(runtime.state_dict().is_err());
    assert!(runtime.load_state_dict(&snapshot).is_err());
    let mut runner = crate::TrainingVesselRunner::<_, _, _, (), _, _, _>::with_policy(
        Box::new(runtime),
        Box::new(crate::rl_candle_collector::CandleCollectorFactory),
        crate::TrainingVesselBinding::default(),
        crate::TrainingVesselRunConfig::default(),
        crate::TrainingVesselLog::default(),
    );
    assert!(matches!(
        runner.state_dict(),
        Err(crate::TrainingVesselStateError::Save(_))
    ));
    assert!(crate::RlCheckpointState::save_checkpoint(&mut runner).is_err());
    assert!(matches!(
        runner.load_state_dict(&crate::TrainingVesselCheckpoint { policy: snapshot }),
        Err(crate::TrainingVesselStateError::Load(_))
    ));
}

#[test]
fn actual_qlib_target_cadence_nstep_learning_and_all_state_values_match() {
    let cases = fixture().cases;
    assert_eq!(cases.len(), 16);
    for case in cases {
        let hooks = Arc::new(Hooks::default());
        let mut policy = policy(&case, hooks.clone());
        assert!(case.independent);
        assert_eq!(policy.action_count(), None);
        assert_eq!(policy.optimizer.initialized_parameter_count(), 0);
        if let Some(old) = policy.target_model() {
            for (name, tensor) in old.parameters() {
                assert_ne!(tensor.id(), policy.model.parameters()[name].id());
            }
        }
        compare_weights(&policy, &case.initial);
        let buffer = replay(&case);
        for step in &case.steps {
            let mut batch: CandleDqnRollout = buffer.get(&[4, 2, 0, 1, 3]).unwrap().into();
            batch.targets.weight =
                Some(Tensor::new(&[0.2_f64, 0.5, 1., 1.5, 2.], &Device::Cpu).unwrap());
            policy.process(&mut batch, &buffer).unwrap();
            step.returns
                .compare(batch.targets.returns.as_ref().unwrap(), "processed returns");
            let loss = policy.learn(&mut batch).unwrap();
            assert!((loss - step.loss).abs() < 1e-7 + step.loss.abs() * 2e-6);
            step.td
                .compare(batch.targets.weight.as_ref().unwrap(), "priority");
            assert_eq!(policy.iteration(), step.iteration);
            compare_weights(&policy, &step.weights);
            if let Some(old) = policy.target_model() {
                let gradients = batch
                    .targets
                    .weight
                    .as_ref()
                    .unwrap()
                    .sum_all()
                    .unwrap()
                    .backward()
                    .unwrap();
                assert!(step.target_grad_absent);
                for tensor in old.parameters().values() {
                    assert!(gradients.get(tensor).is_none());
                }
            }
            assert!(!policy.is_updating());
        }
        policy.set_mode(TrainingPolicyMode::Evaluation);
        assert_eq!(policy.mode(), TrainingPolicyMode::Evaluation);
        policy.set_mode(TrainingPolicyMode::Train);
        assert_eq!(case.mode, [true, true]);
        policy
            .target_q(&CandleDqnObservation {
                observation: observation(case.observations.tensor()),
                mask: None,
            })
            .unwrap();
        let modes = hooks.modes.lock().unwrap();
        if let Some(old) = policy.target_model() {
            assert_eq!(
                modes.last().unwrap(),
                &(old.parameters()["extractor.scale"].id(), false)
            );
        } else {
            assert!(modes.last().unwrap().1);
        }
        assert_eq!(policy.action_count(), Some(case.max_actions));
    }
}

struct ReplayHook {
    buffer: CandleReplayBuffer,
    hooks: Arc<Hooks>,
}
impl CandleNStepReplay for ReplayHook {
    fn rewards(&self) -> Result<Array1<f64>, String> {
        self.hooks.enter("process")?;
        self.buffer.rewards()
    }
    fn next_indices(&self, indices: &[usize]) -> Result<Vec<usize>, String> {
        self.buffer.next_indices(indices)
    }
    fn bootstrap_mask(&self, indices: &[usize]) -> Result<Array1<bool>, String> {
        self.buffer.bootstrap_mask(indices)
    }
    fn end_flags(&self) -> Result<Array1<bool>, String> {
        self.buffer.end_flags()
    }
    fn unfinished(&self) -> Result<Vec<usize>, String> {
        self.buffer.unfinished()
    }
}
impl CandleDqnReplay for ReplayHook {
    fn sample(&mut self, size: u64, rng: &mut dyn RngCore) -> Result<CandleDqnRollout, String> {
        self.hooks.enter("sample")?;
        CandleDqnReplay::sample(&mut self.buffer, size, rng)
    }
    fn next_observations(&self, indices: &[usize]) -> Result<CandleDqnObservation, String> {
        CandleDqnReplay::next_observations(&self.buffer, indices)
    }
    fn update_weights(&mut self, indices: &[usize], weights: &Tensor) -> Result<(), String> {
        assert_eq!(indices.len(), 5);
        assert_eq!(weights.dims(), [5]);
        self.hooks.enter("priority")
    }
}
struct Scheduler(Arc<Hooks>);
impl CandleDqnScheduler for Scheduler {
    fn step(&mut self, _: &mut CandleAdam) -> Result<(), String> {
        self.0.enter("scheduler")
    }
}
#[test]
fn source_update_failure_order_updating_flag_and_committed_iteration_match() {
    let fixture = fixture();
    let case = fixture
        .cases
        .iter()
        .find(|c| c.frequency == 2 && !c.huber && c.double)
        .unwrap();
    assert_eq!(fixture.lifecycle.len(), 6);
    for expected in fixture.lifecycle {
        let hooks = Arc::new(Hooks {
            fail: expected.failure.clone(),
            ..Default::default()
        });
        let mut policy = policy(case, hooks.clone());
        let mut buffer = ReplayHook {
            buffer: replay(case),
            hooks: hooks.clone(),
        };
        let mut scheduler = Scheduler(hooks.clone());
        let mut rng = StdRng::seed_from_u64(3);
        let result = policy.update(Some(&mut buffer), Some(&mut scheduler), 0, &mut rng);
        assert_eq!(result.is_err(), expected.error.is_some());
        if let Some(error) = expected.error {
            assert!(result.unwrap_err().to_string().contains(&error));
        }
        assert_eq!(*hooks.events.lock().unwrap(), expected.events);
        assert_eq!(policy.is_updating(), expected.updating);
        assert_eq!(policy.iteration(), expected.iteration);
        assert!(
            policy
                .update(None, Some(&mut scheduler), 0, &mut rng)
                .unwrap()
                .is_empty()
        );
        assert_eq!(policy.is_updating(), expected.updating);
    }
}

#[test]
fn configuration_copy_and_exploration_failures_do_not_silently_share_state() {
    let case = fixture().cases.remove(0);
    let (features, parameters) = setup(&case, Arc::new(Hooks::default()));
    for config in [
        CandleDqnConfig {
            gamma: f64::NAN,
            ..config(&case)
        },
        CandleDqnConfig {
            steps: 0,
            ..config(&case)
        },
    ] {
        assert!(matches!(
            CandleDqn::new(features.clone(), 3, &builder(&parameters), config),
            Err(CandleDqnError::Configuration(_))
        ));
    }
    let bad = Arc::new(Features {
        parameters: features.parameters.clone(),
        mode: AtomicBool::new(true),
        hooks: features.hooks.clone(),
        wrong_copy: true,
    });
    assert!(
        CandleDqn::new(
            bad,
            3,
            &builder(&parameters),
            CandleDqnConfig {
                target_update_frequency: 1,
                ..config(&case)
            }
        )
        .is_err()
    );
    let mut policy = policy(&case, Arc::new(Hooks::default()));
    assert!(matches!(
        policy.sync_weight(),
        Err(CandleDqnError::MissingTarget)
    ));
    let mut actions = [0, 1];
    let mut rng = StdRng::seed_from_u64(9);
    let mut expected = rng.clone();
    policy.exploration(&mut actions, None, &mut rng).unwrap();
    assert_eq!(rng.next_u64(), expected.next_u64());
    policy.set_epsilon(0.5);
    for _ in 0..2 {
        let _ = expected.random::<f64>();
    }
    assert!(matches!(
        policy.exploration(&mut actions, None, &mut rng),
        Err(CandleDqnError::MissingActionCount)
    ));
    assert_eq!(rng.next_u64(), expected.next_u64());
    assert_eq!(actions, [0, 1]);
    let input = CandleDqnObservation {
        observation: observation(case.observations.tensor()),
        mask: None,
    };
    assert_eq!(
        policy.forward(&input, "retained").unwrap().state,
        "retained"
    );
    policy.set_epsilon(1.);
    let mask = array![[0., 1., 0.]];
    policy
        .exploration(&mut actions, Some(&mask.view()), &mut rng)
        .unwrap();
    assert_eq!(actions, [1, 1]);
}

#[test]
fn full_and_failed_target_weight_loads_match_source_retry_mutations() {
    let fixture = fixture();
    assert_eq!(fixture.weight_cases.len(), 6);
    for expected in fixture.weight_cases {
        let case = fixture
            .cases
            .iter()
            .find(|c| c.frequency == expected.frequency)
            .unwrap();
        let mut policy = policy(case, Arc::new(Hooks::default()));
        let mut loaded = PolicyWeights {
            weights: expected
                .initial
                .iter()
                .map(|(name, record)| (name.clone(), Arc::new(record.tensor())))
                .collect(),
            metadata: String::from("retained"),
        };
        let result = set_policy_weights(&mut policy, &mut loaded);
        assert_eq!(result.is_err(), expected.error.is_some());
        assert_eq!(
            loaded.weights.keys().collect::<Vec<_>>(),
            expected.input_keys.iter().collect::<Vec<_>>()
        );
        assert_eq!(loaded.metadata, "retained");
        compare_weights(&policy, &expected.final_weights);
        assert_eq!(policy.iteration(), 0);
        assert_eq!(policy.optimizer.initialized_parameter_count(), 0);
        let snapshot = policy.policy_state(()).unwrap().snapshot().unwrap();
        let materialized = snapshot.into_policy_weights(None).unwrap();
        assert_eq!(materialized.weights.len(), expected.final_weights.len());
        for (name, value) in &materialized.weights {
            expected.final_weights[name].compare(value, name);
        }
    }
}

struct WithoutRebuild(Arc<Features>);
impl CandleFeatureExtractor for WithoutRebuild {
    fn output_dim(&self) -> usize {
        2
    }
    fn parameters(&self) -> &IndexMap<String, Tensor> {
        &self.0.parameters
    }
    fn forward(&self, input: &RecurrentObservation) -> candle_core::Result<Tensor> {
        self.0.forward(input)
    }
}
#[test]
fn explicit_rebuild_capability_and_alias_preserving_copies_are_required() {
    let variable = Var::new(&[1_f32, 2.], &Device::Cpu).unwrap();
    let constant = Tensor::new(&[3_f32, 4.], &Device::Cpu).unwrap();
    let original = IndexMap::from([
        ("first".into(), variable.as_tensor().clone()),
        ("alias".into(), variable.as_tensor().clone()),
        ("constant".into(), constant.clone()),
    ]);
    let copied = crate::rl_candle_network::independent_parameters(&original).unwrap();
    assert_eq!(copied["first"].id(), copied["alias"].id());
    assert_ne!(copied["first"].id(), variable.id());
    assert_ne!(copied["constant"].id(), constant.id());
    assert!(!copied["constant"].is_variable());
    Var::from_tensor(&copied["first"])
        .unwrap()
        .set(&Tensor::new(&[9_f32, 8.], &Device::Cpu).unwrap())
        .unwrap();
    assert_eq!(variable.to_vec1::<f32>().unwrap(), [1., 2.]);
    assert_eq!(copied["alias"].to_vec1::<f32>().unwrap(), [9., 8.]);
    let case = fixture().cases.remove(0);
    let (features, parameters) = setup(&case, Arc::new(Hooks::default()));
    let unsupported = Arc::new(WithoutRebuild(features));
    let enabled = CandleDqnConfig {
        target_update_frequency: 1,
        ..config(&case)
    };
    let Err(error) = CandleDqn::new(unsupported.clone(), 3, &builder(&parameters), enabled) else {
        panic!("must reject unsupported target reconstruction")
    };
    assert!(error.to_string().contains("target reconstruction"));
    let mut disabled = CandleDqn::new(
        unsupported,
        3,
        &builder(&parameters),
        CandleDqnConfig {
            target_update_frequency: -2,
            ..enabled
        },
    )
    .unwrap();
    assert!(disabled.target_model().is_none());
    disabled.set_mode(TrainingPolicyMode::Evaluation);
    assert_eq!(disabled.mode(), TrainingPolicyMode::Evaluation);
}

#[test]
fn late_counter_errors_and_early_preprocessing_errors_retain_reached_state() {
    let case = fixture().cases.remove(0);
    let mut policy = policy(&case, Arc::new(Hooks::default()));
    let buffer = replay(&case);
    let mut batch: CandleDqnRollout = buffer.get(&[0, 1, 2, 3, 4]).unwrap().into();
    policy.config.reward_normalization = true;
    assert!(matches!(
        policy.process(&mut batch, &buffer),
        Err(CandleDqnError::Returns(
            CandleNStepError::RewardNormalization
        ))
    ));
    assert_eq!(policy.action_count(), None);
    assert!(batch.targets.returns.is_none());
    policy.config.reward_normalization = false;
    policy.process(&mut batch, &buffer).unwrap();
    policy.iteration = u64::MAX;
    let before = policy.model.parameters()["layer_out.0.weight"]
        .to_vec2::<f32>()
        .unwrap();
    assert!(matches!(
        policy.learn(&mut batch),
        Err(CandleDqnError::Iteration)
    ));
    assert_eq!(policy.iteration(), u64::MAX);
    assert_eq!(batch.targets.weight.unwrap().dims(), [5]);
    assert_ne!(
        policy.model.parameters()["layer_out.0.weight"]
            .to_vec2::<f32>()
            .unwrap(),
        before
    );
    assert_eq!(policy.optimizer.initialized_parameter_count(), 3);
    let (features, _) = setup(&case, Arc::new(Hooks::default()));
    policy.model = DqnModel::new(features, 3, VarBuilder::zeros(DType::F32, &Device::Cpu)).unwrap();
    assert!(policy.policy_state(()).is_err());
    let mut loaded = weights(&case);
    let input_ids: Vec<_> = loaded.weights.values().map(|tensor| tensor.id()).collect();
    assert!(matches!(
        policy.load_weights(&mut loaded),
        Err(PolicyWeightLoadError::Runtime(_))
    ));
    assert_eq!(
        loaded
            .weights
            .values()
            .map(|tensor| tensor.id())
            .collect::<Vec<_>>(),
        input_ids
    );
}

#[test]
fn failed_target_inference_preserves_batch_and_optimizer_state() {
    let case = fixture().cases.remove(0);
    let hooks = Arc::new(Hooks {
        calls: AtomicUsize::new(2),
        fail: Some("learn".into()),
        ..Hooks::default()
    });
    let mut policy = policy(&case, hooks.clone());
    let buffer = replay(&case);
    let mut batch: CandleDqnRollout = buffer.get(&[0, 1]).unwrap().into();
    let returns = Tensor::new(&[7_f32, 8.], &Device::Cpu).unwrap();
    let priority = Tensor::new(&[0.2_f64, 0.8], &Device::Cpu).unwrap();
    batch.targets.returns = Some(returns.clone());
    batch.targets.weight = Some(priority.clone());
    let error = policy.process(&mut batch, &buffer).unwrap_err();
    assert!(matches!(error, CandleDqnError::Returns(_)));
    assert!(error.to_string().contains("learn"));
    assert_eq!(batch.targets.returns.as_ref().unwrap().id(), returns.id());
    assert_eq!(batch.targets.weight.as_ref().unwrap().id(), priority.id());
    assert_eq!(policy.action_count(), None);
    assert_eq!(policy.iteration(), 0);
    assert_eq!(policy.optimizer.initialized_parameter_count(), 0);
    assert_eq!(*hooks.events.lock().unwrap(), ["learn"]);
    compare_weights(&policy, &case.initial);
}

#[test]
fn vessel_inference_and_uncached_exploration_errors_preserve_actions_and_draw_order() {
    use crate::rl_candle_collector::CandleCollectionPolicy;
    use crate::rl_candle_dqn_vessel::CandleDqnVessel;
    let case = fixture().cases.remove(0);
    let mut runtime = CandleDqnVessel::new(
        policy(&case, Arc::new(Hooks::default())),
        StdRng::seed_from_u64(71),
    );
    let malformed = observation(Tensor::zeros((2, 3), DType::F32, &Device::Cpu).unwrap());
    assert!(runtime.actions(&malformed).is_err());
    assert_eq!(runtime.policy.action_count(), None);
    runtime.policy.set_epsilon(1.);
    let actions = Tensor::new(&[1_i64, 2], &Device::Cpu).unwrap();
    let mut expected_rng = runtime.rng.clone();
    for _ in 0..2 {
        let _ = expected_rng.random::<f64>();
    }
    let error = runtime
        .exploration_noise(actions.clone(), &malformed)
        .unwrap_err();
    assert_eq!(error, CandleDqnError::MissingActionCount.to_string());
    assert_eq!(actions.to_vec1::<i64>().unwrap(), [1, 2]);
    assert_eq!(runtime.rng.next_u64(), expected_rng.next_u64());
    assert_eq!(runtime.policy.iteration(), 0);
    assert_eq!(runtime.policy.optimizer.initialized_parameter_count(), 0);
    compare_weights(&runtime.policy, &case.initial);
}

#[test]
fn independent_actor_copy_preserves_cross_extractor_head_parameter_alias() {
    let case = fixture().cases.remove(0);
    let (features, _) = setup(&case, Arc::new(Hooks::default()));
    let shared = features.parameters["scale"].clone();
    let head = Var::new(&[[0.1_f32, 0.2], [0.3, 0.4]], &Device::Cpu).unwrap();
    let parameters = std::collections::HashMap::from([
        ("layer_out.0.weight".into(), head.into_inner()),
        ("layer_out.0.bias".into(), shared.clone()),
    ]);
    let actor = DqnModel::new(
        features,
        2,
        VarBuilder::from_tensors(parameters, DType::F32, &Device::Cpu),
    )
    .unwrap();
    let copied = actor.independent_copy().unwrap();
    let copied_scale = &copied.parameters()["extractor.scale"];
    assert_eq!(
        copied_scale.id(),
        copied.parameters()["layer_out.0.bias"].id()
    );
    assert_ne!(copied_scale.id(), shared.id());
    let before = shared.to_vec1::<f32>().unwrap();
    Var::from_tensor(copied_scale)
        .unwrap()
        .set(&Tensor::new(&[9_f32, 8.], &Device::Cpu).unwrap())
        .unwrap();
    assert_eq!(shared.to_vec1::<f32>().unwrap(), before);
    assert_eq!(
        copied.parameters()["layer_out.0.bias"]
            .to_vec1::<f32>()
            .unwrap(),
        [9., 8.]
    );
}

#[test]
fn native_vessel_options_scheduler_and_replay_errors_remain_ordered() {
    use crate::rl_candle_collector::CandleCollectionPolicy;
    use crate::rl_candle_dqn_vessel::CandleDqnVessel;
    use crate::{TrainingRunPolicy, TrainingUpdateOptions, TrainingVesselRunError};
    let case = fixture()
        .cases
        .into_iter()
        .find(|c| c.frequency == 2)
        .unwrap();
    let mut runtime = CandleDqnVessel::new(
        policy(&case, Arc::new(Hooks::default())),
        StdRng::seed_from_u64(2),
    );
    let mut buffer = replay(&case);
    for key in ["sample_size", "buffer"] {
        let result = TrainingRunPolicy::<CandleReplayBuffer>::update(
            &mut runtime,
            0,
            None,
            &IndexMap::from([(key.into(), serde_json::json!(null))]),
        );
        assert!(matches!(
            result,
            Err(TrainingVesselRunError::DuplicateKeyword(_))
        ));
    }
    let options = IndexMap::from([("repeat".into(), serde_json::json!("ignored"))]);
    runtime.scheduler = Some(Box::new(Scheduler(Arc::new(Hooks::default()))));
    let metrics = TrainingRunPolicy::update(&mut runtime, 0, Some(&mut buffer), &options).unwrap();
    assert!(
        matches!(&metrics["loss"], crate::TrainingMetricValue::Scalar(crate::TrainingMetricScalar::Float(value)) if value.is_finite())
    );
    let mut empty = CandleReplayBuffer::new(2);
    assert!(
        TrainingRunPolicy::update(
            &mut runtime,
            0,
            Some(&mut empty),
            &TrainingUpdateOptions::new()
        )
        .is_err()
    );
    assert!(!runtime.policy.is_updating());
    assert!(
        runtime
            .exploration_noise(
                Tensor::new(&[0_f32], &Device::Cpu).unwrap(),
                &observation(case.observations.tensor())
            )
            .is_err()
    );
    let mut vector = CandleVectorReplayBuffer::new(4, 2).unwrap();
    assert!(CandleDqnReplay::sample(&mut vector, 0, &mut runtime.rng).is_err());
    assert!(CandleDqnReplay::next_observations(&vector, &[0]).is_err());
    assert!(CandleDqnReplay::next_observations(&empty, &[0]).is_err());
    assert!(
        runtime
            .policy
            .update(None, None, 0, &mut runtime.rng)
            .unwrap()
            .is_empty()
    );
}
