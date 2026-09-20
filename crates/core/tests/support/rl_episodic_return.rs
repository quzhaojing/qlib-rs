use super::*;
use ndarray::array;
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    name: String,
    gamma: f64,
    gae_lambda: f64,
    rewards: Vec<String>,
    terminated: Vec<bool>,
    truncated: Vec<bool>,
    bootstrap_valid: Vec<bool>,
    indices: Vec<usize>,
    unfinished_indices: Vec<usize>,
    next_values: Option<Vec<String>>,
    values: Option<Vec<String>>,
    returns: Vec<String>,
    advantages: Vec<String>,
}
#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}
fn numbers(values: &[String]) -> Array1<f64> {
    values.iter().map(|value| value.parse().unwrap()).collect()
}

#[test]
fn returns_match_real_replay_buffers_and_numba_gae_including_nonfinite_values() {
    let fixture: Fixture =
        serde_json::from_str(include_str!("../fixtures/rl_episodic_return.json")).unwrap();
    assert_eq!(fixture.cases.len(), 48);
    for case in fixture.cases {
        let rewards = numbers(&case.rewards);
        let terminated = Array1::from(case.terminated);
        let truncated = Array1::from(case.truncated);
        let bootstrap_valid = Array1::from(case.bootstrap_valid);
        let next_values = case.next_values.as_deref().map(numbers);
        let values = case.values.as_deref().map(numbers);
        let input = EpisodicReturnInput {
            rewards: rewards.view(),
            terminated: terminated.view(),
            truncated: truncated.view(),
            bootstrap_valid: bootstrap_valid.view(),
            indices: &case.indices,
            unfinished_indices: &case.unfinished_indices,
            next_values: next_values.as_ref().map(|values| values.view()),
            values: values.as_ref().map(|values| values.view()),
        };
        let output = episodic_returns(&input, case.gamma, case.gae_lambda).unwrap();
        for (actual, expected) in [
            (&output.returns, numbers(&case.returns)),
            (&output.advantages, numbers(&case.advantages)),
        ] {
            assert_eq!(actual.len(), expected.len());
            for (&actual, &expected) in actual.iter().zip(&expected) {
                if expected.is_nan() {
                    assert!(actual.is_nan(), "{}", case.name);
                } else {
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "{}: {actual} != {expected}",
                        case.name
                    );
                }
            }
        }
    }
}

#[test]
fn malformed_lengths_are_typed_and_unused_bootstrap_masks_are_not_inspected() {
    let values = array![0.2, 0.3];
    let flags = array![false, true];
    let short_values = array![1.];
    let short_flags = array![true];
    let base = EpisodicReturnInput {
        rewards: values.view(),
        terminated: flags.view(),
        truncated: flags.view(),
        bootstrap_valid: flags.view(),
        indices: &[0, 1],
        unfinished_indices: &[],
        next_values: Some(values.view()),
        values: Some(values.view()),
    };
    for field in [
        "terminated",
        "truncated",
        "indices",
        "next_values",
        "bootstrap_valid",
        "values",
    ] {
        let mut input = base.clone();
        match field {
            "terminated" => input.terminated = short_flags.view(),
            "truncated" => input.truncated = short_flags.view(),
            "indices" => input.indices = &[0],
            "next_values" => input.next_values = Some(short_values.view()),
            "bootstrap_valid" => input.bootstrap_valid = short_flags.view(),
            "values" => input.values = Some(short_values.view()),
            _ => unreachable!(),
        }
        let error = episodic_returns(&input, 1., 1.).unwrap_err();
        assert_eq!(
            error,
            EpisodicReturnError::Length {
                field,
                actual: 1,
                expected: 2
            }
        );
        assert!(error.to_string().contains(field));
    }
    let mut missing = base.clone();
    missing.next_values = None;
    missing.bootstrap_valid = short_flags.view();
    for lambda in [0.95, f64::NAN, f64::INFINITY] {
        let error = episodic_returns(&missing, 1., lambda).unwrap_err();
        assert_eq!(error, EpisodicReturnError::MissingBootstrap);
        assert!(error.to_string().contains("lambda"));
    }
    let actual = episodic_returns(&missing, 1., 1.).unwrap();
    assert_eq!(actual.returns.to_vec(), vec![0.2, 0.3]);
    assert_eq!(actual.advantages.to_vec(), vec![0., 0.]);
    missing.values = None;
    let actual = episodic_returns(&missing, 1., 1.).unwrap();
    assert_eq!(actual.returns.to_vec(), vec![0.5, 0.3]);
}
