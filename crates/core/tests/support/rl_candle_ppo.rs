use super::*;
use crate::rl_candle_categorical::epsilon;
use crate::rl_candle_network_fixture::Record;
use candle_core::{Device, Var};
use indexmap::IndexMap;

#[derive(Deserialize)]
struct Case {
    name: String,
    config: PpoLossConfig,
    inputs: IndexMap<String, Record>,
    metrics: IndexMap<String, f64>,
    gradients: IndexMap<String, Record>,
}
#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

fn input(tensors: &IndexMap<String, Tensor>) -> PpoLossInput<'_> {
    PpoLossInput {
        probabilities: &tensors["probabilities"],
        values: &tensors["values"],
        actions: &tensors["actions"],
        old_log_prob: &tensors["old_log_prob"],
        advantages: &tensors["advantages"],
        returns: &tensors["returns"],
        old_values: tensors.get("old_values"),
    }
}
fn cases() -> Vec<Case> {
    let fixture: Fixture =
        serde_json::from_str(include_str!("../fixtures/rl_candle_ppo_loss.json")).unwrap();
    assert_eq!(fixture.cases.len(), 24);
    fixture.cases
}
fn tensors(case: &Case) -> IndexMap<String, Tensor> {
    case.inputs
        .iter()
        .map(|(name, record)| {
            let tensor = record.tensor();
            let tensor = if name == "probabilities" || name == "values" {
                Var::from_tensor(&tensor).unwrap().as_tensor().clone()
            } else {
                tensor
            };
            (name.clone(), tensor)
        })
        .collect()
}

#[test]
fn losses_and_actor_critic_gradients_match_actual_tianshou_learn() {
    for case in cases() {
        let tensors = tensors(&case);
        let output = case.config.loss(&input(&tensors)).unwrap();
        for (name, tensor) in [
            ("loss", &output.total),
            ("loss/clip", &output.policy),
            ("loss/vf", &output.value),
            ("loss/ent", &output.entropy),
        ] {
            let dtype = if tensor.dtype() == DType::F64 {
                "float64"
            } else {
                "float32"
            };
            Record {
                shape: vec![],
                dtype: dtype.into(),
                values: vec![case.metrics[name]],
            }
            .compare(tensor, &format!("{} {name}", case.name));
        }
        let gradients = output.total.backward().unwrap();
        for (name, expected) in &case.gradients {
            expected.compare(
                gradients
                    .get(&tensors[name])
                    .unwrap_or_else(|| panic!("{} missing {name} gradient", case.name)),
                &format!("{} {name}", case.name),
            );
        }
    }
}

#[test]
fn scalar_clamp_keeps_boundary_gradients_and_minmax_keep_nan_gradients() {
    let tensor = Var::new(&[-0.25_f32, 0., 0.25, f32::NAN], &Device::Cpu).unwrap();
    let result = clamp(&tensor, -0.25, 0.25).unwrap();
    assert!(result.to_vec1::<f32>().unwrap()[3].is_nan());
    assert_eq!(
        result
            .sum_all()
            .unwrap()
            .backward()
            .unwrap()
            .get(&tensor)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        [1., 1., 1., 0.]
    );
    for (min, max) in [(f64::NAN, 1.), (0., f64::NAN)] {
        let result = clamp(&tensor, min, max).unwrap();
        assert!(result.to_vec1::<f32>().unwrap().iter().all(|v| v.is_nan()));
        assert_eq!(
            result
                .sum_all()
                .unwrap()
                .backward()
                .unwrap()
                .get(&tensor)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            [0.; 4]
        );
    }
    for maximum in [false, true] {
        let left = Var::new(&[1_f32, f32::NAN, 2.], &Device::Cpu).unwrap();
        let right = Var::new(&[1_f32, 3., f32::NAN], &Device::Cpu).unwrap();
        let output = extreme(&left, &right, maximum).unwrap();
        let values = output.to_vec1::<f32>().unwrap();
        assert_eq!(values[0].to_bits(), 1_f32.to_bits());
        assert!(values[1..].iter().all(|v| v.is_nan()));
        let gradients = output.sum_all().unwrap().backward().unwrap();
        for var in [&left, &right] {
            assert_eq!(
                gradients.get(var).unwrap().to_vec1::<f32>().unwrap(),
                [0.5, 1., 1.]
            );
        }
    }
}

#[test]
fn loss_rejects_malformed_inputs_and_preserves_default_configuration() {
    let defaults = PpoLossConfig::default();
    assert_eq!(defaults.eps_clip.to_bits(), 0.3_f64.to_bits());
    assert!(defaults.normalize_advantage && defaults.value_clip);
    assert_eq!(defaults.value_weight.to_bits(), 1_f64.to_bits());
    assert_eq!(defaults.entropy_weight.to_bits(), 0.01_f64.to_bits());
    assert!(defaults.dual_clip.is_none());
    let case = cases().remove(0);
    let valid = tensors(&case);
    for dual in [1., 0., f64::NAN] {
        let config = PpoLossConfig {
            dual_clip: Some(dual),
            ..defaults
        };
        assert!(config.loss(&input(&valid)).is_err());
    }
    for (name, tensor) in [
        (
            "probabilities",
            Tensor::zeros(4, DType::F32, &Device::Cpu).unwrap(),
        ),
        (
            "probabilities",
            Tensor::zeros((0, 3), DType::F32, &Device::Cpu).unwrap(),
        ),
        (
            "probabilities",
            Tensor::zeros((4, 0), DType::F32, &Device::Cpu).unwrap(),
        ),
        (
            "probabilities",
            Tensor::zeros((4, 3), DType::U8, &Device::Cpu).unwrap(),
        ),
        (
            "values",
            Tensor::zeros(3, DType::F32, &Device::Cpu).unwrap(),
        ),
        (
            "returns",
            Tensor::zeros(4, DType::F64, &Device::Cpu).unwrap(),
        ),
        (
            "actions",
            Tensor::zeros(3, DType::I64, &Device::Cpu).unwrap(),
        ),
        (
            "actions",
            Tensor::new(&[0.5_f32, 0., 0., 0.], &Device::Cpu).unwrap(),
        ),
        (
            "actions",
            Tensor::new(&[-1_i64, 0, 0, 0], &Device::Cpu).unwrap(),
        ),
        (
            "actions",
            Tensor::new(&[0_i64, 0, 0, 3], &Device::Cpu).unwrap(),
        ),
        (
            "probabilities",
            Tensor::zeros((4, 3), DType::F32, &Device::Cpu).unwrap(),
        ),
        (
            "probabilities",
            Tensor::new(&[[-1_f32, 1., 1.]; 4], &Device::Cpu).unwrap(),
        ),
    ] {
        let mut invalid = valid.clone();
        invalid.insert(name.into(), tensor);
        assert!(defaults.loss(&input(&invalid)).is_err(), "{name}");
    }
    for (dtype, expected) in [
        (DType::F16, 0.000_976_562_5_f64),
        (DType::BF16, 0.007_812_5),
    ] {
        assert_eq!(epsilon(dtype).unwrap().to_bits(), expected.to_bits());
    }
}

#[test]
fn old_values_are_optional_and_only_validated_when_clipping_reads_them() {
    let case = cases().remove(0);
    let tensors = tensors(&case);
    let unclipped = PpoLossConfig {
        value_clip: false,
        normalize_advantage: false,
        ..PpoLossConfig::default()
    };
    let baseline = unclipped.loss(&input(&tensors)).unwrap();
    let baseline_grad = baseline.total.backward().unwrap();
    let malformed_shape = Tensor::zeros(1, DType::F32, &Device::Cpu).unwrap();
    let malformed_dtype = Tensor::zeros(4, DType::I64, &Device::Cpu).unwrap();
    for old_values in [None, Some(&malformed_shape), Some(&malformed_dtype)] {
        let mut batch = input(&tensors);
        batch.old_values = old_values;
        let output = unclipped.loss(&batch).unwrap();
        assert_eq!(
            output.total.to_scalar::<f32>().unwrap().to_bits(),
            baseline.total.to_scalar::<f32>().unwrap().to_bits()
        );
        let grad = output.total.backward().unwrap();
        for name in ["probabilities", "values"] {
            assert_eq!(
                grad.get(&tensors[name])
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
                baseline_grad
                    .get(&tensors[name])
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            );
        }
        assert!(PpoLossConfig::default().loss(&batch).is_err());
    }
    let mut overflow = tensors.clone();
    overflow.insert(
        "probabilities".into(),
        Tensor::full(f32::MAX, (4, 3), &Device::Cpu).unwrap(),
    );
    assert!(unclipped.loss(&input(&overflow)).is_err());
}
