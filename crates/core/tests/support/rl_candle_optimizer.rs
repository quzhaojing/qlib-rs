use super::*;
use crate::rl_candle_network_fixture::Record;
use candle_core::{DType, Device};
use serde::Deserialize;

#[derive(Deserialize)]
struct Step {
    learning_rate: f64,
    gradients: Vec<Option<Record>>,
    parameters: Vec<Record>,
    initialized: usize,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    decay: f64,
    initial: Vec<Record>,
    steps: Vec<Step>,
}
#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[test]
fn adam_updates_match_torch_with_missing_gradients_decay_and_rate_changes() {
    let fixture: Fixture =
        serde_json::from_str(include_str!("../fixtures/rl_candle_adam.json")).unwrap();
    assert_eq!(fixture.cases.len(), 12);
    for case in fixture.cases {
        let variables: Vec<_> = case
            .initial
            .iter()
            .map(|record| Var::from_tensor(&record.tensor()).unwrap())
            .collect();
        // Actor/critic can register the same extractor; it must be updated once.
        let mut parameters: Vec<_> = variables
            .iter()
            .map(|var| var.as_tensor().clone())
            .collect();
        parameters.push(variables[0].as_tensor().clone());
        let mut optimizer = CandleAdam::new(parameters, 0.003, case.decay).unwrap();
        assert_eq!(optimizer.parameters().len(), 2);
        assert_eq!(optimizer.initialized_parameter_count(), 0);
        for (index, step) in case.steps.iter().enumerate() {
            optimizer.set_learning_rate(step.learning_rate).unwrap();
            assert_eq!(
                optimizer.learning_rate().to_bits(),
                step.learning_rate.to_bits()
            );
            let mut gradients = Tensor::new(0_f32, &Device::Cpu)
                .unwrap()
                .backward()
                .unwrap();
            for (parameter, gradient) in variables.iter().zip(&step.gradients) {
                if let Some(gradient) = gradient {
                    gradients.insert(parameter, gradient.tensor());
                }
            }
            optimizer
                .step(&gradients)
                .unwrap_or_else(|error| panic!("{} step {index}: {error}", case.name));
            assert_eq!(optimizer.initialized_parameter_count(), step.initialized);
            for (parameter, expected) in variables.iter().zip(&step.parameters) {
                expected.compare(parameter, &format!("{} step {index}", case.name));
            }
        }
    }
}

#[test]
fn invalid_parameters_and_gradients_do_not_mutate_live_state() {
    let device = &Device::Cpu;
    let a = Var::new(&[1_f32, 2.], device).unwrap();
    let b = Var::new(&[3_f32, 4.], device).unwrap();
    for value in [-1., f64::NAN, f64::NEG_INFINITY] {
        assert!(CandleAdam::new(vec![a.as_tensor().clone()], value, 0.).is_err());
        assert!(CandleAdam::new(vec![a.as_tensor().clone()], 0.01, value).is_err());
    }
    assert!(CandleAdam::new(Vec::new(), 0.01, 0.).is_err());
    assert!(CandleAdam::new(vec![Tensor::new(1_f32, device).unwrap()], 0.01, 0.).is_err());
    assert!(
        CandleAdam::new(
            vec![Var::new(&[1_i64], device).unwrap().as_tensor().clone()],
            0.01,
            0.
        )
        .is_err()
    );
    let mut optimizer =
        CandleAdam::new(vec![a.as_tensor().clone(), b.as_tensor().clone()], 0.01, 0.).unwrap();
    assert!(optimizer.set_learning_rate(-1.).is_err());
    assert_eq!(optimizer.learning_rate().to_bits(), 0.01_f64.to_bits());
    for invalid in [
        Tensor::ones(3, DType::F32, device).unwrap(),
        Tensor::ones(2, DType::F64, device).unwrap(),
    ] {
        let mut gradients = Tensor::new(0_f32, device).unwrap().backward().unwrap();
        gradients.insert(&a, Tensor::ones(2, DType::F32, device).unwrap());
        gradients.insert(&b, invalid);
        assert!(optimizer.clip_grad_norm(&mut gradients, 1.).is_err());
        assert_eq!(
            gradients.get(&a).unwrap().to_vec1::<f32>().unwrap(),
            [1., 1.]
        );
        assert!(optimizer.step(&gradients).is_err());
        assert_eq!(a.to_vec1::<f32>().unwrap(), [1., 2.]);
        assert_eq!(b.to_vec1::<f32>().unwrap(), [3., 4.]);
        assert_eq!(optimizer.initialized_parameter_count(), 0);
    }
}

#[derive(Deserialize)]
struct ClipCase {
    name: String,
    limit: f64,
    parameters: Vec<Record>,
    before: Vec<Option<Record>>,
    norm: Record,
    after: Vec<Option<Record>>,
}
#[derive(Deserialize)]
struct ClipSpecial {
    value: String,
    limit: String,
    norm: String,
    after: Vec<String>,
}
#[derive(Deserialize)]
struct ClipFixture {
    cases: Vec<ClipCase>,
    special: Vec<ClipSpecial>,
}

#[test]
fn gradient_clipping_matches_torch_including_mixed_and_reduced_precision() {
    let fixture: ClipFixture =
        serde_json::from_str(include_str!("../fixtures/rl_candle_gradient_clip.json")).unwrap();
    assert_eq!(fixture.cases.len(), 37);
    for case in fixture.cases {
        let variables: Vec<_> = case
            .parameters
            .iter()
            .map(|record| Var::from_tensor(&record.tensor()).unwrap())
            .collect();
        let mut parameters: Vec<_> = variables.iter().map(|v| v.as_tensor().clone()).collect();
        parameters.push(variables[0].as_tensor().clone());
        let optimizer = CandleAdam::new(parameters, 0.01, 0.).unwrap();
        let mut gradients = Tensor::new(0_f32, &Device::Cpu)
            .unwrap()
            .backward()
            .unwrap();
        let unrelated = Var::new(9_f32, &Device::Cpu).unwrap();
        gradients.insert(&unrelated, unrelated.as_tensor().clone());
        for (variable, record) in variables.iter().zip(&case.before) {
            if let Some(record) = record {
                gradients.insert(variable, record.tensor());
            }
        }
        let norm = optimizer
            .clip_grad_norm(&mut gradients, case.limit)
            .unwrap();
        case.norm.compare(&norm, &format!("{} norm", case.name));
        assert_eq!(optimizer.initialized_parameter_count(), 0);
        for ((variable, expected), original) in
            variables.iter().zip(&case.after).zip(&case.parameters)
        {
            original.compare(variable, "clipping must not update parameter values");
            match expected {
                Some(record) => record.compare(gradients.get(variable).unwrap(), &case.name),
                None => assert!(gradients.get(variable).is_none()),
            }
        }
        assert_eq!(
            gradients
                .get(&unrelated)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
                .to_bits(),
            9_f32.to_bits()
        );
    }
}

#[test]
fn gradient_clipping_preserves_torch_default_nonfinite_behavior() {
    let fixture: ClipFixture =
        serde_json::from_str(include_str!("../fixtures/rl_candle_gradient_clip.json")).unwrap();
    assert_eq!(fixture.special.len(), 6);
    for case in fixture.special {
        let variable = Var::zeros(2, DType::F32, &Device::Cpu).unwrap();
        let optimizer = CandleAdam::new(vec![variable.as_tensor().clone()], 0.01, 0.).unwrap();
        let mut gradients = Tensor::new(0_f32, &Device::Cpu)
            .unwrap()
            .backward()
            .unwrap();
        gradients.insert(
            &variable,
            Tensor::new(&[case.value.parse::<f32>().unwrap(), 4.], &Device::Cpu).unwrap(),
        );
        let norm = optimizer
            .clip_grad_norm(&mut gradients, case.limit.parse().unwrap())
            .unwrap();
        let actual = std::iter::once(norm.to_scalar::<f32>().unwrap())
            .chain(gradients.get(&variable).unwrap().to_vec1::<f32>().unwrap());
        let expected = std::iter::once(case.norm).chain(case.after);
        for (actual, expected) in actual.zip(expected) {
            let expected = expected.parse::<f32>().unwrap();
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
        assert_eq!(optimizer.initialized_parameter_count(), 0);
    }
}
