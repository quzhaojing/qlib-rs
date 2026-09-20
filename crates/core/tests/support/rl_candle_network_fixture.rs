use candle_core::{DType, Device, Tensor, Var, backprop::GradStore};
use candle_nn::VarBuilder;
use indexmap::IndexMap;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize)]
pub struct Record {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub values: Vec<f64>,
}
impl Record {
    pub fn tensor(&self) -> Tensor {
        let dtype = match self.dtype.as_str() {
            "float32" => DType::F32,
            "float64" => DType::F64,
            "float16" => DType::F16,
            "bfloat16" => DType::BF16,
            "int64" => DType::I64,
            other => panic!("unexpected oracle dtype {other}"),
        };
        Tensor::from_vec(self.values.clone(), self.shape.as_slice(), &Device::Cpu)
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
    }
    pub fn compare(&self, actual: &Tensor, context: &str) {
        assert_eq!(actual.dims(), self.shape, "{context} shape");
        assert_eq!(actual.dtype(), self.tensor().dtype(), "{context} dtype");
        let values = actual
            .to_dtype(DType::F64)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f64>()
            .unwrap();
        if matches!(self.dtype.as_str(), "float16" | "bfloat16") {
            let expected = self
                .tensor()
                .to_dtype(DType::F64)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f64>()
                .unwrap();
            assert_eq!(values, expected, "{context}: reduced-precision values");
            return;
        }
        // F32 recurrence accumulates rounding across matmuls; double precision
        // retains a tighter bound. The reduced-precision fixture is exact above.
        let (absolute, relative) = match self.dtype.as_str() {
            "float64" => (1e-12, 1e-11),
            _ => (1e-7, 2e-5),
        };
        for (index, (actual, expected)) in values.iter().zip(&self.values).enumerate() {
            assert!(
                (actual - expected).abs() <= absolute + relative * expected.abs(),
                "{context}[{index}]: actual={actual}, expected={expected}"
            );
        }
    }
}

#[derive(Deserialize)]
pub struct Case {
    pub name: String,
    pub kind: String,
    pub config: serde_json::Value,
    pub weights: IndexMap<String, Record>,
    pub aliases: Vec<Vec<String>>,
    pub inputs: IndexMap<String, Record>,
    pub outputs: IndexMap<String, Record>,
    pub gradients: IndexMap<String, Option<Record>>,
    pub input_gradients: IndexMap<String, Option<Record>>,
    #[serde(default)]
    pub intermediate_gradients: IndexMap<String, Record>,
}
#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}
pub fn cases(kind: &str) -> Vec<Case> {
    let fixture: Fixture =
        serde_json::from_str(include_str!("../fixtures/rl_candle_network.json")).unwrap();
    assert_eq!(fixture.cases.len(), 17);
    fixture
        .cases
        .into_iter()
        .filter(|case| case.kind == kind)
        .collect()
}
impl Case {
    pub fn weights(&self) -> IndexMap<String, Tensor> {
        let mut weights: IndexMap<_, _> = self
            .weights
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
        for group in &self.aliases {
            let first = weights[&group[0]].clone();
            for name in &group[1..] {
                weights.insert(name.clone(), first.clone());
            }
        }
        weights
    }
    pub fn inputs(&self) -> IndexMap<String, Tensor> {
        self.inputs
            .iter()
            .map(|(name, record)| {
                let tensor = record.tensor();
                let tensor = if tensor.dtype().is_float() {
                    Var::from_tensor(&tensor).unwrap().as_tensor().clone()
                } else {
                    tensor
                };
                (name.clone(), tensor)
            })
            .collect()
    }
    pub fn check_gradients(
        &self,
        store: &GradStore,
        weights: &IndexMap<String, Tensor>,
        inputs: &IndexMap<String, Tensor>,
    ) {
        for (expected, tensors) in [(&self.gradients, weights), (&self.input_gradients, inputs)] {
            for (name, record) in expected {
                let gradient = store.get(&tensors[name]);
                let context = format!("{} gradient {name}", self.name);
                match record {
                    Some(record) => record.compare(
                        gradient.unwrap_or_else(|| panic!("missing {context}")),
                        &context,
                    ),
                    None => assert!(gradient.is_none(), "unexpected {context}"),
                }
            }
        }
    }
}
pub fn builder(weights: &IndexMap<String, Tensor>) -> VarBuilder<'static> {
    VarBuilder::from_tensors(
        weights
            .iter()
            .map(|(n, t)| (n.clone(), t.clone()))
            .collect::<HashMap<_, _>>(),
        weights.first().unwrap().1.dtype(),
        &Device::Cpu,
    )
}
