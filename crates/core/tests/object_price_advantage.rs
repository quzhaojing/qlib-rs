use std::{path::PathBuf, process::Command, rc::Rc};

use domain_core::object_price_advantage::{
    FloatObjectArithmetic, ObjectPriceAdvantage, ObjectPriceAdvantageError, ObjectPriceArithmetic,
    object_price_advantage,
};
use ndarray::{ArrayD, Axis, IxDyn, ShapeBuilder, Slice};
use serde::Deserialize;
use serde_json::{Value, json};
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
#[error("{message}")]
struct ProbeError {
    kind: &'static str,
    message: &'static str,
}

struct Arithmetic {
    failure: Option<(String, usize)>,
    events: Vec<Value>,
    last: Option<Rc<usize>>,
}

impl Arithmetic {
    fn run(&mut self, stage: &str, name: usize, operand: &str) -> Result<usize, ProbeError> {
        self.events.push(json!([stage, name, operand]));
        if self
            .failure
            .as_ref()
            .is_some_and(|(fail_stage, fail_name)| fail_stage == stage && *fail_name == name)
        {
            return Err(ProbeError {
                kind: "RuntimeError",
                message: "sentinel",
            });
        }
        Ok(name)
    }
}

impl ObjectPriceArithmetic for Arithmetic {
    type Input = usize;
    type Quotient = usize;
    type Difference = usize;
    type Value = Rc<usize>;
    type Error = ProbeError;

    fn divide(&mut self, value: &usize, baseline: f64) -> Result<usize, ProbeError> {
        self.run("divide", *value, &format!("{baseline:.1}"))
    }
    fn subtract(&mut self, value: &usize, buy: bool) -> Result<usize, ProbeError> {
        self.run(if buy { "one_minus" } else { "minus_one" }, *value, "1")
    }
    fn multiply(&mut self, value: &usize) -> Result<Rc<usize>, ProbeError> {
        let result = Rc::new(self.run("multiply", *value, "10000")?);
        self.last = Some(result.clone());
        Ok(result)
    }
    fn finish_scalar(&mut self, _value: Rc<usize>) -> Result<Rc<usize>, ProbeError> {
        Err(ProbeError {
            kind: "AttributeError",
            message: "'Item' object has no attribute 'size'",
        })
    }
}

#[derive(Deserialize)]
struct Oracle {
    objects: Vec<Value>,
    floats: Vec<FloatCase>,
}

#[derive(Debug, Deserialize)]
struct FloatCase {
    shape: Vec<usize>,
    value: String,
    baseline: String,
    direction: i64,
    output: Option<Vec<String>>,
    scalar: Option<bool>,
    error: Option<Vec<String>>,
}

fn object_case(expected: &Value) {
    let base = ArrayD::from_shape_vec(IxDyn(&[2, 3]), (0..6).collect()).unwrap();
    let scalar = ArrayD::from_elem(IxDyn(&[]), 0);
    let fortran = ArrayD::from_shape_vec(IxDyn(&[2, 3]).f(), vec![0, 3, 1, 4, 2, 5]).unwrap();
    let broadcast = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![0, 1, 2]).unwrap();
    let three = ArrayD::from_shape_vec(IxDyn(&[1, 2, 3]), (0..6).collect()).unwrap();
    let mut view = base.view();
    match expected["view"].as_str().unwrap() {
        "c" => {}
        "fortran" => view = fortran.view(),
        "broadcast" => view = broadcast.broadcast(IxDyn(&[2, 3])).unwrap(),
        "three" => view = three.view().permuted_axes(vec![2, 0, 1]),
        "transpose_reverse" => {
            view.invert_axis(Axis(1));
            view = view.reversed_axes();
        }
        "transpose" => view = view.reversed_axes(),
        "reverse" => view.invert_axis(Axis(1)),
        "both_reverse" => {
            view.invert_axis(Axis(0));
            view.invert_axis(Axis(1));
        }
        "slice" => view.slice_axis_inplace(Axis(1), Slice::new(0, None, 2)),
        "empty" => view.slice_axis_inplace(Axis(0), Slice::new(0, Some(0), 1)),
        "singleton" => {
            view.slice_axis_inplace(Axis(0), Slice::new(0, Some(1), 1));
            view.slice_axis_inplace(Axis(1), Slice::new(0, Some(1), 1));
        }
        "zero_dim" => view = scalar.view(),
        other => panic!("unknown view {other}"),
    }
    let mut arithmetic = Arithmetic {
        failure: serde_json::from_value(expected["failure"].clone()).unwrap(),
        events: vec![],
        last: None,
    };
    let result = object_price_advantage(
        &view,
        expected["baseline"].as_f64().unwrap(),
        expected["direction"].as_i64().unwrap(),
        &mut arithmetic,
    );
    assert_eq!(json!(arithmetic.events), expected["events"], "{expected}");
    match result {
        Ok(result) => {
            let (values, is_scalar) = match result {
                ObjectPriceAdvantage::ZeroBaseline(values) => (
                    values
                        .iter()
                        .map(|&v| usize::try_from(v).unwrap())
                        .collect::<Vec<_>>(),
                    false,
                ),
                ObjectPriceAdvantage::Scalar(value) => {
                    assert!(Rc::ptr_eq(&value, arithmetic.last.as_ref().unwrap()));
                    (vec![*value], true)
                }
                ObjectPriceAdvantage::Array(values) => {
                    assert_eq!(json!(values.shape()), expected["shape"]);
                    (values.iter().map(|v| **v).collect(), false)
                }
            };
            assert_eq!(json!(values), expected["output"], "{expected}");
            assert_eq!(json!(is_scalar), expected["scalar"]);
        }
        Err(error) => {
            assert!(
                format!("{error:?}").contains("Direction")
                    || format!("{error:?}").contains("Operation")
            );
            let kind = match &error {
                ObjectPriceAdvantageError::Direction(_) => "ValueError",
                ObjectPriceAdvantageError::Operation(value) => value.kind,
            };
            assert_eq!(
                json!([kind, &error.to_string()]),
                expected["error"],
                "{expected}"
            );
        }
    }
}

fn float_case(case: &FloatCase) {
    let prices = ArrayD::from_elem(IxDyn(&case.shape), case.value.parse::<f64>().unwrap());
    let result = object_price_advantage(
        &prices.view(),
        case.baseline.parse().unwrap(),
        case.direction,
        &mut FloatObjectArithmetic,
    );
    match result {
        Ok(result) => {
            let (values, scalar) = match result {
                ObjectPriceAdvantage::ZeroBaseline(values) => (vec![0.0; values.len()], false),
                ObjectPriceAdvantage::Scalar(value) => (vec![value], true),
                ObjectPriceAdvantage::Array(values) => (values.iter().copied().collect(), false),
            };
            let bits: Vec<_> = values
                .iter()
                .map(|v| {
                    if v.is_nan() {
                        "nan".to_owned()
                    } else {
                        format!("{:016x}", v.to_bits())
                    }
                })
                .collect();
            assert_eq!(Some(bits), case.output, "{case:?}");
            assert_eq!(Some(scalar), case.scalar, "{case:?}");
        }
        Err(error) => assert_eq!(
            Some(vec!["ValueError".to_owned(), error.to_string()]),
            case.error,
            "{case:?}"
        ),
    }
}

#[test]
fn object_stages_identity_failures_and_numeric_scalars_match_real_numpy() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = std::env::var_os("QLIB_PYTHON_RL_EXECUTION_UTILS").map_or_else(
        || root.join("../../../qlib/qlib/rl/order_execution/utils.py"),
        PathBuf::from,
    );
    let output = Command::new(std::env::var_os("PYTHON").unwrap_or_else(|| "python".into()))
        .arg(root.join("tests/fixtures/object_price_advantage.py"))
        .arg(source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let oracle: Oracle = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(oracle.objects.len(), 240);
    assert_eq!(oracle.floats.len(), 768);
    oracle.objects.iter().for_each(object_case);
    oracle.floats.iter().for_each(float_case);
}
