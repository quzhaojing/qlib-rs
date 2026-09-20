use super::*;
use candle_core::{Device, Var};
use rand::{SeedableRng, rngs::StdRng};
use serde::Deserialize;
use std::sync::Arc;

#[derive(Deserialize)]
struct Record {
    shape: Vec<usize>,
    dtype: String,
    values: Vec<String>,
}
impl Record {
    fn tensor(&self) -> Tensor {
        let dtype = match self.dtype.as_str() {
            "torch.float16" => DType::F16,
            "torch.bfloat16" => DType::BF16,
            "torch.float32" => DType::F32,
            "torch.float64" => DType::F64,
            "torch.int64" => DType::I64,
            _ => panic!("unexpected fixture dtype"),
        };
        Tensor::from_vec(
            self.values
                .iter()
                .map(|v| v.parse::<f64>().unwrap())
                .collect::<Vec<_>>(),
            self.shape.as_slice(),
            &Device::Cpu,
        )
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
    }
    fn compare(&self, tensor: &Tensor, context: &str) {
        assert_eq!(tensor.shape().dims(), self.shape, "{context}");
        assert_eq!(tensor.dtype(), self.tensor().dtype(), "{context}");
        let actual = tensor
            .to_dtype(DType::F64)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f64>()
            .unwrap();
        for (&actual, expected) in actual.iter().zip(&self.values) {
            let expected = expected.parse::<f64>().unwrap();
            if expected.is_nan() {
                assert!(actual.is_nan(), "{context}");
            } else if matches!(tensor.dtype(), DType::F16 | DType::BF16 | DType::I64)
                || expected.is_infinite()
            {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "{context}: {actual} != {expected}"
                );
            } else {
                let (abs, rel) = if tensor.dtype() == DType::F64 {
                    (1e-12, 1e-11)
                } else {
                    (1e-7, 2e-5)
                };
                assert!(
                    (actual - expected).abs() <= abs + rel * expected.abs(),
                    "{context}: {actual} != {expected}"
                );
            }
        }
    }
}
#[derive(Deserialize)]
struct Case {
    name: String,
    raw: Record,
    probabilities: Record,
    logits: Record,
    actions: Record,
    deterministic_actions: Record,
    log_prob: Record,
    entropy: Record,
    gradients: Record,
}
#[derive(Deserialize)]
struct Failure {
    name: String,
    raw: Record,
}
#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
    errors: Vec<Failure>,
}
fn fixture() -> Fixture {
    serde_json::from_str(include_str!("../fixtures/rl_candle_categorical.json")).unwrap()
}

#[test]
fn forward_distributions_and_gradients_match_real_tianshou_and_torch() {
    let cases = fixture().cases;
    assert_eq!(cases.len(), 20);
    for case in cases {
        let raw = Var::from_tensor(&case.raw.tensor()).unwrap();
        let state = Arc::new(());
        let mut rng = StdRng::seed_from_u64(41);
        let mut unused_rng = rng.clone();
        let forward = categorical_forward(
            raw.as_tensor().clone(),
            state.clone(),
            TrainingPolicyMode::Evaluation,
            true,
            &mut rng,
        )
        .unwrap();
        assert_eq!(rng.random::<u64>(), unused_rng.random::<u64>());
        assert!(Arc::ptr_eq(&state, &forward.state));
        assert_eq!(forward.logits.id(), raw.id());
        case.raw.compare(&forward.logits, &case.name);
        case.deterministic_actions
            .compare(&forward.actions, &case.name);
        case.probabilities.compare(
            forward.distribution.probabilities(),
            &format!("{} probabilities", case.name),
        );
        case.logits.compare(
            forward.distribution.logits(),
            &format!("{} logits", case.name),
        );
        let log_prob = forward
            .distribution
            .log_prob(&case.actions.tensor())
            .unwrap();
        let entropy = forward.distribution.entropy().unwrap();
        case.log_prob
            .compare(&log_prob, &format!("{} log_prob", case.name));
        case.entropy
            .compare(&entropy, &format!("{} entropy", case.name));
        let loss = (log_prob.sum_all().unwrap() + entropy.sum_all().unwrap()).unwrap();
        let gradients = loss.backward().unwrap();
        case.gradients.compare(
            gradients.get(&raw).unwrap(),
            &format!("{} gradients", case.name),
        );
    }
}

#[test]
fn sampled_actions_are_reproducible_respect_zero_weights_and_follow_probabilities() {
    let raw = Tensor::new(&[0.1_f64, 0.3, 0.6], &Device::Cpu)
        .unwrap()
        .broadcast_as((30_000, 3))
        .unwrap();
    let distribution = CandleCategorical::from_probabilities(&raw).unwrap();
    let sample = distribution.sample(&mut StdRng::seed_from_u64(41)).unwrap();
    assert_eq!(sample.dtype(), DType::I64);
    assert_eq!(sample.dims(), &[30_000]);
    let values = sample.to_vec1::<i64>().unwrap();
    assert_eq!(
        values,
        distribution
            .sample(&mut StdRng::seed_from_u64(41))
            .unwrap()
            .to_vec1::<i64>()
            .unwrap()
    );
    let mut counts = [0_u32; 3];
    for value in values {
        counts[usize::try_from(value).unwrap()] += 1;
    }
    for (&observed, probability) in counts.iter().zip([0.1_f64, 0.3, 0.6]) {
        let expected = 30_000. * probability;
        let sigma = (expected * (1. - probability)).sqrt();
        assert!((f64::from(observed) - expected).abs() < 6. * sigma);
    }
    for (mode, deterministic) in [
        (TrainingPolicyMode::Train, true),
        (TrainingPolicyMode::Train, false),
        (TrainingPolicyMode::Evaluation, false),
    ] {
        let mut rng = StdRng::seed_from_u64(93);
        let mut unused = rng.clone();
        let raw = Tensor::new(&[[0_f32, 1., 0.], [1., 0., 0.]], &Device::Cpu).unwrap();
        let result = categorical_forward(raw, (), mode, deterministic, &mut rng).unwrap();
        assert_eq!(result.actions.to_vec1::<i64>().unwrap(), [1, 0]);
        assert_ne!(rng.random::<u64>(), unused.random::<u64>());
    }
}

#[test]
fn malformed_probabilities_actions_and_empty_shapes_follow_the_boundary() {
    assert_eq!(SourceNormalize.name(), "source-categorical-normalization");
    let errors = fixture().errors;
    assert_eq!(errors.len(), 6);
    for case in errors {
        assert!(
            CandleCategorical::from_probabilities(&case.raw.tensor()).is_err(),
            "{}",
            case.name
        );
    }
    let raw = Tensor::new(&[[1_f64, 2., 3.]; 2], &Device::Cpu).unwrap();
    let distribution = CandleCategorical::from_probabilities(&raw).unwrap();
    for value in [f64::NAN, f64::INFINITY, -1., 0.5, 3.] {
        assert!(
            distribution
                .log_prob(&Tensor::new(value, &Device::Cpu).unwrap())
                .is_err()
        );
    }
    assert!(
        distribution
            .log_prob(&Tensor::zeros(4, DType::I64, &Device::Cpu).unwrap())
            .is_err()
    );
    assert!(
        CandleCategorical::from_probabilities(
            &Tensor::ones((2, 3), DType::U8, &Device::Cpu).unwrap()
        )
        .is_err()
    );
    assert!(
        CandleCategorical::from_probabilities(
            &Tensor::full(f32::MAX, (2, 3), &Device::Cpu).unwrap()
        )
        .is_err()
    );
    let empty = CandleCategorical::from_probabilities(
        &Tensor::zeros((0, 3), DType::F32, &Device::Cpu).unwrap(),
    )
    .unwrap();
    assert_eq!(
        empty.sample(&mut StdRng::seed_from_u64(1)).unwrap().dims(),
        &[0]
    );
    assert_eq!(empty.entropy().unwrap().dims(), &[0]);
    assert_eq!(
        empty
            .log_prob(&Tensor::zeros(0, DType::I64, &Device::Cpu).unwrap())
            .unwrap()
            .dims(),
        &[0]
    );
    let empty = CandleCategorical::from_probabilities(
        &Tensor::zeros((0, 0), DType::F32, &Device::Cpu).unwrap(),
    )
    .unwrap();
    assert!(empty.sample(&mut StdRng::seed_from_u64(1)).is_err());
}
