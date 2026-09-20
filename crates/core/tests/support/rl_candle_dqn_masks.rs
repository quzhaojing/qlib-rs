use crate::rl_candle_dqn::{dqn_forward, dqn_q_values};
use candle_core::{DType, Device, Tensor, WithDType};
use serde::Deserialize;
use std::{fmt::Debug, str::FromStr};

#[derive(Deserialize)]
struct Record {
    shape: Vec<usize>,
    dtype: String,
    values: Vec<String>,
}
impl Record {
    fn typed<T: WithDType + FromStr>(&self) -> Tensor
    where
        T::Err: Debug,
    {
        Tensor::from_vec(
            self.values
                .iter()
                .map(|value| value.parse::<T>().unwrap())
                .collect(),
            self.shape.as_slice(),
            &Device::Cpu,
        )
        .unwrap()
    }
    fn tensor(&self) -> Tensor {
        match self.dtype.as_str() {
            "uint8" => self.typed::<u8>(),
            "uint32" => self.typed::<u32>(),
            "int64" => self.typed::<i64>(),
            name => {
                let dtype = match name {
                    "float16" => DType::F16,
                    "bfloat16" => DType::BF16,
                    "float32" => DType::F32,
                    "float64" => DType::F64,
                    _ => panic!("unexpected dtype {name}"),
                };
                self.typed::<f64>().to_dtype(dtype).unwrap()
            }
        }
    }
    fn compare(&self, actual: &Tensor, context: &str) {
        assert_eq!(actual.dims(), self.shape);
        assert_eq!(actual.dtype(), self.tensor().dtype());
        let values = actual
            .to_dtype(DType::F64)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f64>()
            .unwrap();
        for (actual, expected) in values.iter().zip(&self.values) {
            let expected: f64 = expected.parse().unwrap();
            if expected.is_nan() {
                assert!(actual.is_nan(), "{context}: expected NaN, got {actual}");
            } else {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "{context}: {actual} != {expected}"
                );
            }
        }
    }
}
#[derive(Deserialize)]
struct Case {
    layout: String,
    mask: Record,
    logits: Record,
    q: Record,
    actions: Vec<i64>,
}
#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[test]
fn source_mask_dtype_wrap_rounding_broadcast_and_strides_match_exactly() {
    let fixture: Fixture =
        serde_json::from_str(include_str!("../fixtures/rl_candle_dqn_masks.json")).unwrap();
    assert_eq!(fixture.cases.len(), 88);
    for case in fixture.cases {
        let mask = case.mask.tensor();
        let logits = case.logits.tensor();
        let axis = mask.rank();
        let strided = Tensor::stack(&[&mask, &mask], axis)
            .unwrap()
            .narrow(axis, 0, 1)
            .unwrap()
            .squeeze(axis)
            .unwrap();
        if mask.elem_count() > 1 {
            assert!(!strided.is_contiguous());
        }
        for input in [&mask, &strided] {
            let context = format!(
                "{} / {} / {}",
                case.mask.dtype, case.logits.dtype, case.layout
            );
            case.q
                .compare(&dqn_q_values(&logits, Some(input)).unwrap(), &context);
            let output = dqn_forward(logits.clone(), "unchanged", Some(input)).unwrap();
            assert_eq!(
                output.actions.to_vec1::<i64>().unwrap(),
                case.actions,
                "{context}"
            );
            assert_eq!(output.state, "unchanged");
            assert_eq!(output.logits.id(), logits.id());
        }
    }
}
