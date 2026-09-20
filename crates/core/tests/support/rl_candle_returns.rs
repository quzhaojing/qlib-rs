use super::*;
use candle_core::{Device, Var};
use ndarray::array;
use serde::Deserialize;

#[derive(Deserialize)]
struct Statistics {
    mean: String,
    variance: String,
    count: usize,
}
#[derive(Deserialize)]
struct Step {
    rewards: Vec<String>,
    values: Vec<String>,
    next_values: Vec<String>,
    terminated: Vec<bool>,
    truncated: Vec<bool>,
    bootstrap_valid: Vec<bool>,
    indices: Vec<usize>,
    unfinished_indices: Vec<usize>,
    before: Statistics,
    after: Statistics,
    old_values: Vec<String>,
    returns: Vec<String>,
    advantages: Vec<String>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    dtype: String,
    normalize: bool,
    gamma: f64,
    gae_lambda: f64,
    steps: Vec<Step>,
}
#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

fn numbers(values: &[String]) -> Array1<f64> {
    values.iter().map(|value| value.parse().unwrap()).collect()
}

fn close(actual: f64, expected: f64, exact: bool, context: &str) {
    if expected.is_nan() {
        assert!(actual.is_nan(), "{context}: {actual}");
    } else if exact || expected.is_infinite() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{context}: {actual} != {expected}"
        );
    } else {
        // F64 batch reductions may sum in a different order than NumPy. This
        // bound is tighter than the existing full PPO loss/gradient tolerance.
        assert!(
            (actual - expected).abs() <= 1e-14 + 1e-13 * expected.abs(),
            "{context}: {actual} != {expected}"
        );
    }
}

fn check_statistics(actual: &ReturnStatistics, expected: &Statistics, context: &str) {
    assert_eq!(actual.count, expected.count, "{context}");
    close(actual.mean, expected.mean.parse().unwrap(), false, context);
    close(
        actual.variance,
        expected.variance.parse().unwrap(),
        false,
        context,
    );
}

#[test]
fn targets_and_running_statistics_match_real_policy_preprocessing() {
    let fixture: Fixture =
        serde_json::from_str(include_str!("../fixtures/rl_candle_returns.json")).unwrap();
    assert_eq!(fixture.cases.len(), 22);
    for case in fixture.cases {
        let dtype = if case.dtype == "torch.float32" {
            DType::F32
        } else {
            DType::F64
        };
        let initial = &case.steps[0].before;
        let mut stats = ReturnStatistics {
            mean: initial.mean.parse().unwrap(),
            variance: initial.variance.parse().unwrap(),
            count: initial.count,
        };
        assert_eq!(case.steps.len(), 3);
        for step in case.steps {
            let values = Var::from_tensor(
                &Tensor::new(numbers(&step.values).to_vec(), &Device::Cpu)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap(),
            )
            .unwrap();
            let next_values = Tensor::new(numbers(&step.next_values).to_vec(), &Device::Cpu)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let rewards = numbers(&step.rewards);
            let terminated = Array1::from(step.terminated);
            let truncated = Array1::from(step.truncated);
            let bootstrap_valid = Array1::from(step.bootstrap_valid);
            check_statistics(&stats, &step.before, &case.name);
            let prepared = prepare_returns(
                &CandleReturnInput {
                    rewards: rewards.view(),
                    terminated: terminated.view(),
                    truncated: truncated.view(),
                    bootstrap_valid: bootstrap_valid.view(),
                    indices: &step.indices,
                    unfinished_indices: &step.unfinished_indices,
                    values: &values,
                    next_values: &next_values,
                },
                case.gamma,
                case.gae_lambda,
                case.normalize.then_some(&mut stats),
            )
            .unwrap();
            check_statistics(&stats, &step.after, &case.name);
            for (actual, expected) in [
                (&prepared.old_values, &step.old_values),
                (&prepared.returns, &step.returns),
                (&prepared.advantages, &step.advantages),
            ] {
                assert_eq!(actual.dtype(), dtype);
                assert_eq!(actual.dims(), &[rewards.len()]);
                assert!(
                    actual
                        .sum_all()
                        .unwrap()
                        .backward()
                        .unwrap()
                        .get(&values)
                        .is_none()
                );
                let actual = actual
                    .to_dtype(DType::F64)
                    .unwrap()
                    .to_vec1::<f64>()
                    .unwrap();
                for (&actual, &expected) in actual.iter().zip(&numbers(expected)) {
                    close(actual, expected, dtype == DType::F32, &case.name);
                }
            }
        }
    }
}

#[test]
fn malformed_batches_fail_before_statistics_change_and_views_flatten() {
    let rewards = array![0.1, 0.2];
    let flags = array![false, true];
    let values = Tensor::new(&[[0.2_f64, 0.3]], &Device::Cpu)
        .unwrap()
        .t()
        .unwrap();
    let next = Tensor::new(&[0.3_f64, 0.4], &Device::Cpu).unwrap();
    let mut input = CandleReturnInput {
        rewards: rewards.view(),
        terminated: flags.view(),
        truncated: flags.view(),
        bootstrap_valid: flags.view(),
        indices: &[0, 1],
        unfinished_indices: &[],
        values: &values,
        next_values: &next,
    };
    let initial = ReturnStatistics::default();
    let mut stats = initial.clone();
    let empty = array![];
    input.rewards = empty.view();
    let error = prepare_returns(&input, 1., 1., Some(&mut stats)).unwrap_err();
    assert!(matches!(error, CandleReturnError::Empty));
    assert!(error.to_string().contains("nonempty"));
    assert_eq!(stats, initial);
    input.rewards = rewards.view();
    let integers = Tensor::new(&[1_i64, 2], &Device::Cpu).unwrap();
    input.values = &integers;
    assert!(matches!(
        prepare_returns(&input, 1., 1., Some(&mut stats)),
        Err(CandleReturnError::Dtype)
    ));
    input.values = &values;
    input.next_values = &integers;
    let error = prepare_returns(&input, 1., 1., Some(&mut stats)).unwrap_err();
    assert!(matches!(error, CandleReturnError::Dtype));
    assert!(error.to_string().contains("dtype"));
    input.next_values = &next;
    input.indices = &[0];
    let error = prepare_returns(&input, 1., 1., Some(&mut stats)).unwrap_err();
    assert!(matches!(
        error,
        CandleReturnError::Episodic(EpisodicReturnError::Length {
            field: "indices",
            ..
        })
    ));
    assert!(error.to_string().contains("indices"));
    assert_eq!(stats, initial);
    input.indices = &[0, 1];
    let output = prepare_returns(&input, 1., 1., Some(&mut stats)).unwrap();
    assert_eq!(output.old_values.to_vec1::<f64>().unwrap(), vec![0.2, 0.3]);
    assert_eq!(stats.count, 2);
}

#[test]
fn scalar_statistics_keep_source_empty_and_nonfinite_merge_behavior() {
    let empty = array![];
    let mut stats = ReturnStatistics::default();
    stats.update(&empty.view());
    assert_eq!(stats.count, 0);
    assert!(stats.mean.is_nan() && stats.variance.is_nan());
    stats.update(&array![1., 2.].view());
    assert_eq!(stats.count, 2);
    assert!(stats.mean.is_nan() && stats.variance.is_nan());
    let mut stats = ReturnStatistics::default();
    stats.update(&array![1., 3.].view());
    assert_eq!(
        stats,
        ReturnStatistics {
            mean: 2.,
            variance: 1.,
            count: 2
        }
    );
    stats.update(&empty.view());
    assert_eq!(stats.count, 2);
    assert!(stats.mean.is_nan() && stats.variance.is_nan());
    let mut full = ReturnStatistics {
        count: usize::MAX,
        ..ReturnStatistics::default()
    };
    let before = full.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        full.update(&array![1.].view());
    }));
    assert!(result.is_err());
    assert_eq!(full, before);
}
