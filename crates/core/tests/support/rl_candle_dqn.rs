use super::*;
use crate::rl_candle_network_fixture::Record;
use candle_core::{Device, Var};
use num_traits::ToPrimitive;
use rand::{RngCore, SeedableRng, rngs::StdRng};
use serde::Deserialize;
use std::cell::Cell;

#[path = "rl_candle_dqn_masks.rs"]
mod masks;

#[derive(Deserialize)]
struct Inference {
    logits: Record,
    mask: Option<Vec<Vec<f64>>>,
    double: bool,
    old: Option<Record>,
    q: Record,
    actions: Vec<i64>,
    state: String,
    target: Record,
}
#[derive(Deserialize)]
struct Step {
    returns: Record,
    weight: Option<Record>,
    loss: f64,
    td: Record,
    gradient: Record,
    r#final: Record,
}
#[derive(Deserialize)]
struct Learning {
    initial: Record,
    huber: bool,
    steps: Vec<Step>,
}
#[derive(Deserialize)]
struct Exploration {
    eps: f64,
    mask: Option<Vec<Vec<f64>>>,
    draws: Vec<f64>,
    actions: Vec<i64>,
}
#[derive(Deserialize)]
struct Fixture {
    inference: Vec<Inference>,
    learning: Vec<Learning>,
    exploration: Vec<Exploration>,
}
fn fixture() -> Fixture {
    serde_json::from_str(include_str!("../fixtures/rl_candle_dqn.json")).unwrap()
}
fn array(values: &[Vec<f64>]) -> Array2<f64> {
    Array2::from_shape_vec((values.len(), values[0].len()), values.concat()).unwrap()
}
fn mask_tensor(values: &[Vec<f64>]) -> Tensor {
    Tensor::from_vec(
        values.concat(),
        (values.len(), values[0].len()),
        &Device::Cpu,
    )
    .unwrap()
}

#[test]
fn source_inference_masks_raw_logits_and_double_targets_match() {
    let cases = fixture().inference;
    assert_eq!(cases.len(), 24);
    for case in cases {
        let logits = case.logits.tensor();
        let mask = case.mask.as_deref().map(mask_tensor);
        case.q
            .compare(&dqn_q_values(&logits, mask.as_ref()).unwrap(), "masked q");
        let result = dqn_forward(logits.clone(), "kept", mask.as_ref()).unwrap();
        assert_eq!(result.logits.id(), logits.id());
        assert_eq!(result.actions.to_vec1::<i64>().unwrap(), case.actions);
        assert_eq!(result.state, case.state);
        let old = case.old.as_ref().map(Record::tensor);
        case.target.compare(
            &dqn_target_values(&result, old.as_ref(), case.double).unwrap(),
            "target",
        );
    }
}

#[test]
fn source_three_adam_steps_match_loss_td_gradients_weights_and_parameters() {
    let cases = fixture().learning;
    assert_eq!(cases.len(), 28);
    let actions = Tensor::new(&[0_i64, -1, 1], &Device::Cpu).unwrap();
    for case in cases {
        let variable = Var::from_tensor(&case.initial.tensor()).unwrap();
        let mut optimizer =
            CandleAdam::new(vec![variable.as_tensor().clone()], 0.003, 0.1).unwrap();
        assert_eq!(case.steps.len(), 3);
        for step in case.steps {
            let returns = step.returns.tensor();
            let weight = step.weight.as_ref().map(Record::tensor);
            let prepared =
                dqn_loss(&variable, &actions, &returns, weight.as_ref(), case.huber).unwrap();
            let gradients = prepared.loss.backward().unwrap();
            step.gradient
                .compare(gradients.get(&variable).unwrap(), "gradient");
            step.td.compare(&prepared.td_error, "td");
            let mut batch = CandleNStepBatch {
                returns: Some(returns.clone()),
                weight,
            };
            let actual = learn_dqn_batch(
                &mut batch,
                &actions,
                || Ok(variable.as_tensor().clone()),
                &mut optimizer,
                case.huber,
            )
            .unwrap();
            assert!(
                (actual - step.loss).abs() < 1e-7 + step.loss.abs() * 2e-6,
                "{actual} != {}",
                step.loss
            );
            step.td.compare(
                batch.weight.as_ref().unwrap(),
                "published signed priorities",
            );
            assert_eq!(batch.returns.unwrap().id(), returns.id());
            step.r#final.compare(&variable, "Adam parameter");
            assert_eq!(optimizer.initialized_parameter_count(), 1);
        }
    }
}

struct Draws {
    values: std::vec::IntoIter<f64>,
}
impl RngCore for Draws {
    fn next_u64(&mut self) -> u64 {
        (self.values.next().expect("unexpected extra draw") * 9_007_199_254_740_992.)
            .to_u64()
            .unwrap()
            << 11
    }
    fn next_u32(&mut self) -> u32 {
        panic!("f64 uses u64 draws")
    }
    fn fill_bytes(&mut self, _: &mut [u8]) {
        panic!("f64 does not fill bytes")
    }
}
#[test]
fn source_epsilon_threshold_draw_order_and_masked_exploration_match() {
    let cases = fixture().exploration;
    assert_eq!(cases.len(), 14);
    for case in cases {
        let mut rng = Draws {
            values: case.draws.into_iter(),
        };
        let mut actions = [2, 0, 1];
        let mask = case.mask.as_deref().map(array);
        dqn_exploration(
            &mut actions,
            case.eps,
            3,
            mask.as_ref().map(|m| m.view()).as_ref(),
            &mut rng,
        )
        .unwrap();
        assert_eq!(actions.as_slice(), case.actions);
        assert_eq!(rng.values.len(), 0);
    }
}

#[test]
fn invalid_batches_preserve_source_priority_pop_and_error_order() {
    let variable = Var::new(&[[1_f32, 2.]], &Device::Cpu).unwrap();
    let mut optimizer = CandleAdam::new(vec![variable.as_tensor().clone()], 0.003, 0.).unwrap();
    let valid = Tensor::new(&[0_i64], &Device::Cpu).unwrap();
    let returns = Tensor::new(&[3_f32], &Device::Cpu).unwrap();
    for actions in [
        Tensor::new(&[0_f32], &Device::Cpu).unwrap(),
        Tensor::new(&[0_i64, 1], &Device::Cpu).unwrap(),
        Tensor::new(&[2_i64], &Device::Cpu).unwrap(),
        Tensor::new(&[-3_i64], &Device::Cpu).unwrap(),
        Tensor::new(&[i64::MIN], &Device::Cpu).unwrap(),
        Tensor::new(&[i64::MAX], &Device::Cpu).unwrap(),
    ] {
        let mut batch = CandleNStepBatch {
            returns: Some(returns.clone()),
            weight: Some(returns.clone()),
        };
        assert!(
            learn_dqn_batch(
                &mut batch,
                &actions,
                || Ok(variable.as_tensor().clone()),
                &mut optimizer,
                false
            )
            .is_err()
        );
        assert!(batch.weight.is_none());
        assert_eq!(optimizer.initialized_parameter_count(), 0);
        assert_eq!(variable.to_vec2::<f32>().unwrap(), [[1., 2.]]);
    }
    let mut batch = CandleNStepBatch {
        returns: None,
        weight: Some(returns.clone()),
    };
    let error = learn_dqn_batch(
        &mut batch,
        &valid,
        || Err(Error::Msg("model failed".into())),
        &mut optimizer,
        false,
    )
    .unwrap_err();
    assert!(error.to_string().contains("model failed"));
    assert!(batch.weight.is_none());
    let invalid_actions = Tensor::new(&[0_f32], &Device::Cpu).unwrap();
    let error = learn_dqn_batch(
        &mut batch,
        &invalid_actions,
        || Ok(variable.as_tensor().clone()),
        &mut optimizer,
        false,
    )
    .unwrap_err();
    assert!(error.to_string().contains("I64 batch vector"));
    let entered = Cell::new(false);
    let error = learn_dqn_batch(
        &mut batch,
        &valid,
        || {
            entered.set(true);
            Ok(variable.as_tensor().clone())
        },
        &mut optimizer,
        false,
    )
    .unwrap_err();
    assert!(entered.get());
    assert!(error.to_string().contains("lacks returns"));
    assert!(dqn_forward(returns.clone(), (), None).is_err());
    assert!(
        dqn_q_values(
            &variable,
            Some(&Tensor::zeros((2, 3), DType::F32, &Device::Cpu).unwrap())
        )
        .is_err()
    );
    let malformed = Tensor::zeros((2, 2), DType::F32, &Device::Cpu).unwrap();
    // Huber does not inspect weights even when their shapes would break MSE.
    assert!(dqn_loss(&variable, &valid, &returns, Some(&malformed), true).is_ok());
    assert!(
        dqn_loss(
            &variable,
            &valid,
            &Tensor::zeros(3, DType::F32, &Device::Cpu).unwrap(),
            Some(&malformed),
            false
        )
        .is_err()
    );
}

#[test]
fn nonfinite_and_empty_inference_and_exploration_are_not_sanitized() {
    let logits = Tensor::new(
        &[[1_f64, f64::NAN, 9.], [f64::INFINITY, 0., f64::INFINITY]],
        &Device::Cpu,
    )
    .unwrap();
    let result = dqn_forward(logits.clone(), (), None).unwrap();
    assert_eq!(result.actions.to_vec1::<i64>().unwrap(), [1, 0]);
    assert!(
        dqn_target_values(&result, None, false)
            .unwrap()
            .to_vec1::<f64>()
            .unwrap()[0]
            .is_nan()
    );
    let masked = dqn_q_values(&logits, Some(&logits.ones_like().unwrap()))
        .unwrap()
        .to_vec2::<f64>()
        .unwrap();
    assert!(masked.iter().flatten().all(|v| v.is_nan()));
    let empty = Tensor::zeros((0, 3), DType::F32, &Device::Cpu).unwrap();
    assert_eq!(
        dqn_forward(empty.clone(), (), None).unwrap().actions.dims(),
        [0]
    );
    assert!(dqn_q_values(&empty, Some(&empty)).is_err());
    assert!(
        dqn_forward(
            Tensor::zeros((2, 0), DType::F32, &Device::Cpu).unwrap(),
            (),
            None
        )
        .is_err()
    );
    let mut rng = StdRng::seed_from_u64(3);
    let mut actions = [1_i64];
    let wrong = ndarray::array![[1., 1.], [0., 0.]];
    assert!(dqn_exploration(&mut actions, 1., 3, Some(&wrong.view()), &mut rng).is_err());
    assert_eq!(actions, [1]);
    assert!(dqn_exploration(&mut actions, 1., 0, None, &mut rng).is_err());
    dqn_exploration(&mut actions, 0., 0, Some(&wrong.view()), &mut rng).unwrap();
    dqn_exploration(&mut actions, f64::NAN, 3, None, &mut rng).unwrap();
    assert_eq!(actions, [1]);
    let mask = ndarray::array![[f64::NAN, 3., f64::NAN]];
    dqn_exploration(&mut actions, 1., 3, Some(&mask.view()), &mut rng).unwrap();
    assert_eq!(actions, [0]);
    let mask = ndarray::array![[0., f64::NAN, f64::NAN]];
    dqn_exploration(&mut actions, 1., 3, Some(&mask.view()), &mut rng).unwrap();
    assert_eq!(actions, [1]);
    dqn_exploration(&mut [], 1., 3, None, &mut rng).unwrap();
}

#[test]
fn huber_nonfinite_slopes_match_torch_without_inactive_square_overflow() {
    let logits = Var::new(&[[0_f32], [0.], [0.]], &Device::Cpu).unwrap();
    let actions = Tensor::new(&[0_i64, 0, 0], &Device::Cpu).unwrap();
    let returns = Tensor::new(&[f32::INFINITY, f32::NEG_INFINITY, f32::NAN], &Device::Cpu).unwrap();
    let result = dqn_loss(&logits, &actions, &returns, None, true).unwrap();
    assert!(result.loss.to_scalar::<f32>().unwrap().is_nan());
    let gradient = result
        .loss
        .backward()
        .unwrap()
        .get(&logits)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    assert_eq!(gradient[0][0].to_bits(), (-1_f32 / 3.).to_bits());
    assert_eq!(gradient[1][0].to_bits(), (1_f32 / 3.).to_bits());
    assert!(gradient[2][0].is_nan());
}
