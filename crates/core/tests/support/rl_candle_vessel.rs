use super::*;
use crate::rl_candle_network::CandleFeatureExtractor;
use crate::rl_candle_policy::{CandlePpoConfig, CandlePpoRollout};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use indexmap::IndexMap;
use ndarray::array;
use rand::{SeedableRng, rngs::StdRng};
use serde::Deserialize;
use serde_json::json;

struct Features(IndexMap<String, Tensor>);
impl CandleFeatureExtractor for Features {
    fn output_dim(&self) -> usize {
        2
    }
    fn parameters(&self) -> &IndexMap<String, Tensor> {
        &self.0
    }
    fn forward(&self, obs: &RecurrentObservation) -> candle_core::Result<Tensor> {
        Ok(obs.data_processed.clone())
    }
}

fn runtime() -> CandlePpoVessel<StdRng> {
    let variables = VarMap::new();
    let builder = VarBuilder::from_varmap(&variables, DType::F32, &Device::Cpu);
    let mut config = CandlePpoConfig::new(0.003);
    config.loss.normalize_advantage = false;
    let policy = CandlePpo::new(Arc::new(Features(IndexMap::new())), 3, &builder, config).unwrap();
    for parameter in policy.optimizer.parameters() {
        parameter
            .set(&parameter.ones_like().unwrap().affine(0.1, 0.).unwrap())
            .unwrap();
    }
    CandlePpoVessel::new(policy, StdRng::seed_from_u64(81))
}

fn rollout() -> CandlePpoRollout {
    let floats = Tensor::ones(5, DType::F32, &Device::Cpu).unwrap();
    let integers = Tensor::zeros(5, DType::I64, &Device::Cpu).unwrap();
    let observations = RecurrentObservation {
        data_processed: Tensor::new(
            &[[0_f32, 0.2], [0.1, 0.2], [0.2, 0.2], [0.3, 0.2], [0.4, 0.2]],
            &Device::Cpu,
        )
        .unwrap(),
        cur_tick: integers.clone(),
        cur_step: integers.clone(),
        num_step: integers.clone(),
        acquiring: integers,
        position_history: floats.unsqueeze(1).unwrap(),
        target: floats,
    };
    CandlePpoRollout {
        next_observations: observations.detached(),
        observations,
        actions: Tensor::new(&[0_i64, 1, 2, 0, 1], &Device::Cpu).unwrap(),
        rewards: array![0., 0.1, 0.2, 0.3, 0.4],
        terminated: array![false, true, false, false, true],
        truncated: array![false, false, false, false, false],
        bootstrap_valid: array![true, false, true, true, false],
        indices: vec![0, 1, 2, 3, 4],
        unfinished_indices: vec![],
        replay_weights: None,
    }
}

struct Buffer {
    fail: bool,
    calls: usize,
}
impl CandlePpoReplay for Buffer {
    fn sample(&mut self, size: u64, _rng: &mut dyn RngCore) -> Result<CandlePpoRollout, String> {
        assert_eq!(size, 0);
        self.calls += 1;
        if self.fail {
            Err("sample sentinel".into())
        } else {
            Ok(rollout())
        }
    }
}

#[derive(Deserialize)]
struct Case {
    options: TrainingUpdateOptions,
    missing_buffer: bool,
    error: Option<String>,
    lengths: Option<IndexMap<String, usize>>,
    state: ExpectedState,
    permutation_consumed: bool,
}

#[derive(Deserialize)]
struct ExpectedState {
    updating: bool,
    count: usize,
    learned: bool,
}

fn check_case(case: &Case) {
    let mut runtime = runtime();
    let mut buffer = Buffer {
        fail: false,
        calls: 0,
    };
    let mut reference_rng = runtime.rng.clone();
    let duplicate = case.options.contains_key("sample_size") || case.options.contains_key("buffer");
    if !case.missing_buffer && !duplicate {
        // Separate real preprocessing establishes the baseline RNG consumption
        // before resolving dynamic learn options (sampling also occurs there).
        let mut reference_policy = super::tests::runtime().policy;
        reference_policy
            .process(&rollout(), &mut reference_rng)
            .unwrap();
    }
    if case.error.is_some() && case.permutation_consumed {
        minibatch_indices(5, 1, true, true, &mut reference_rng).unwrap();
    }
    let result = runtime.update(
        0,
        if case.missing_buffer {
            None
        } else {
            Some(&mut buffer)
        },
        &case.options,
    );
    assert_eq!(result.is_err(), case.error.is_some(), "{:?}", case.options);
    assert_eq!(runtime.policy.is_updating(), case.state.updating);
    assert_eq!(runtime.policy.return_statistics.count, case.state.count);
    assert_eq!(
        runtime.policy.optimizer.initialized_parameter_count() > 0,
        case.state.learned
    );
    assert_eq!(
        buffer.calls,
        usize::from(!case.missing_buffer && !duplicate)
    );
    if duplicate {
        assert!(matches!(
            result,
            Err(TrainingVesselRunError::DuplicateKeyword(_))
        ));
    }
    if case.error.is_some() || case.missing_buffer {
        assert_eq!(
            runtime.rng.next_u64(),
            reference_rng.next_u64(),
            "{:?}",
            case.options
        );
    }
    if let Some(lengths) = &case.lengths {
        let result = result.unwrap();
        assert_eq!(
            result.keys().collect::<Vec<_>>(),
            lengths.keys().collect::<Vec<_>>()
        );
        for (name, expected_len) in lengths {
            let TrainingMetricValue::Numeric(values) = &result[name] else {
                panic!("expected Arrow metric");
            };
            assert_eq!(values.data_type(), &arrow_schema::DataType::Float64);
            assert_eq!(values.len(), *expected_len);
            assert_eq!(values.null_count(), 0);
        }
    }
}

#[test]
fn default_vessel_keywords_follow_fifty_six_real_source_calls() {
    #[derive(Deserialize)]
    struct Fixture {
        cases: Vec<Case>,
    }
    let fixture: Fixture =
        serde_json::from_str(include_str!("../fixtures/rl_candle_vessel.json")).unwrap();
    assert_eq!(fixture.cases.len(), 56);
    for case in fixture.cases {
        check_case(&case);
    }
}

struct Scheduler(bool);
impl CandlePpoScheduler for Scheduler {
    fn step(
        &mut self,
        optimizer: &mut crate::rl_candle_optimizer::CandleAdam,
    ) -> Result<(), String> {
        optimizer.set_learning_rate(0.01).unwrap();
        if self.0 {
            Err("schedule sentinel".into())
        } else {
            Ok(())
        }
    }
}

#[test]
fn runtime_retains_mode_state_scheduler_and_failure_mutations() {
    for fail in [false, true] {
        let mut runtime = runtime();
        runtime.scheduler = Some(Box::new(Scheduler(fail)));
        let mut expected_rng = runtime.rng.clone();
        <CandlePpoVessel<_> as TrainingRunPolicy<Buffer>>::set_mode(
            &mut runtime,
            TrainingPolicyMode::Evaluation,
        )
        .unwrap();
        let state = Arc::new(7);
        let prediction = runtime
            .forward(&rollout().observations, state.clone())
            .unwrap();
        assert!(Arc::ptr_eq(&state, &prediction.state));
        assert_eq!(runtime.rng.next_u64(), expected_rng.next_u64());
        <CandlePpoVessel<_> as TrainingRunPolicy<Buffer>>::set_mode(
            &mut runtime,
            TrainingPolicyMode::Train,
        )
        .unwrap();
        let options =
            IndexMap::from([("batch_size".into(), json!(5)), ("repeat".into(), json!(1))]);
        let mut buffer = Buffer {
            fail: false,
            calls: 0,
        };
        let result = runtime.update(0, Some(&mut buffer), &options);
        assert_eq!(result.is_err(), fail);
        assert_eq!(runtime.policy.is_updating(), fail);
        assert_eq!(
            runtime.policy.optimizer.learning_rate().to_bits(),
            0.01_f64.to_bits()
        );
        assert!(
            runtime
                .update(0, None::<&mut Buffer>, &IndexMap::new())
                .is_ok()
        );
        assert_eq!(runtime.policy.is_updating(), fail);
    }
}

#[test]
fn sampling_failure_precedes_options_and_large_native_integers_do_not_narrow() {
    let mut runtime = runtime();
    let mut buffer = Buffer {
        fail: true,
        calls: 0,
    };
    let error = runtime
        .update(0, Some(&mut buffer), &IndexMap::new())
        .err()
        .unwrap();
    assert!(error.to_string().contains("sample sentinel"));
    assert!(!runtime.policy.is_updating());
    assert_eq!(runtime.policy.return_statistics.count, 0);
    let options = IndexMap::from([
        ("batch_size".into(), json!(u64::MAX)),
        ("repeat".into(), json!(1)),
    ]);
    let mut rng = StdRng::seed_from_u64(1);
    assert_eq!(
        learn_options(&options, 5, &mut rng).unwrap(),
        (usize::MAX, 1)
    );
}
