use std::{path::PathBuf, process::Command};

use domain_core::{
    builtin_object_price::{
        BuiltinPriceArithmetic, BuiltinPriceArithmeticError, BuiltinPriceNumber, BuiltinPriceType,
        BuiltinPriceValue,
    },
    object_price_advantage::{
        ObjectPriceAdvantage, ObjectPriceAdvantageError, object_price_advantage,
    },
};
use ndarray::{ArrayD, IxDyn};
use num_complex::Complex;
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Debug, Deserialize)]
struct Case {
    input: Value,
    shape: Vec<usize>,
    baseline: String,
    direction: i64,
    output: Option<Vec<Value>>,
    scalar: Option<bool>,
    error: Option<Vec<String>>,
}

fn input(value: &Value) -> BuiltinPriceValue {
    let kind = value["kind"].as_str().unwrap();
    match kind {
        "bool" => BuiltinPriceValue::Bool(value["value"].as_str().unwrap().parse().unwrap()),
        "integer" => BuiltinPriceValue::Integer(value["value"].as_str().unwrap().parse().unwrap()),
        "float" => BuiltinPriceValue::Float(value["value"].as_str().unwrap().parse().unwrap()),
        "complex" => BuiltinPriceValue::Complex(Complex::new(
            value["value"].as_str().unwrap().parse().unwrap(),
            value["imag"].as_str().unwrap().parse().unwrap(),
        )),
        other => BuiltinPriceValue::NonNumeric(match other {
            "NoneType" => BuiltinPriceType::None,
            "str" => BuiltinPriceType::String,
            "bytes" => BuiltinPriceType::Bytes,
            "list" => BuiltinPriceType::List,
            "tuple" => BuiltinPriceType::Tuple,
            "dict" => BuiltinPriceType::Dict,
            "set" => BuiltinPriceType::Set,
            "frozenset" => BuiltinPriceType::FrozenSet,
            "range" => BuiltinPriceType::Range,
            "ellipsis" => BuiltinPriceType::Ellipsis,
            "NotImplementedType" => BuiltinPriceType::NotImplemented,
            other => panic!("unexpected type {other}"),
        }),
    }
}

fn bits(value: f64) -> String {
    if value.is_nan() {
        "nan".to_owned()
    } else {
        format!("{:016x}", value.to_bits())
    }
}

fn snapshot(value: &BuiltinPriceNumber) -> Value {
    match value {
        BuiltinPriceNumber::Float(value) => json!({"kind":"float", "real": bits(*value)}),
        BuiltinPriceNumber::Complex(value) => {
            json!({"kind":"complex", "real": bits(value.re), "imag": bits(value.im)})
        }
    }
}

fn compare(case: &Case) {
    let prices = if let Some(values) = case.input.as_array() {
        ArrayD::from_shape_vec(IxDyn(&case.shape), values.iter().map(input).collect()).unwrap()
    } else {
        ArrayD::from_elem(IxDyn(&case.shape), input(&case.input))
    };
    let result = object_price_advantage(
        &prices.view(),
        case.baseline.parse().unwrap(),
        case.direction,
        &mut BuiltinPriceArithmetic,
    );
    match result {
        Ok(result) => {
            let (output, scalar) = match result {
                ObjectPriceAdvantage::ZeroBaseline(values) => (
                    values
                        .iter()
                        .map(|v| json!({"kind":"integer", "value":v.to_string()}))
                        .collect(),
                    false,
                ),
                ObjectPriceAdvantage::Scalar(value) => (vec![snapshot(&value)], true),
                ObjectPriceAdvantage::Array(values) => {
                    assert_eq!(values.shape(), case.shape);
                    (values.iter().map(snapshot).collect(), false)
                }
            };
            assert_eq!(Some(output), case.output, "{case:?}");
            assert_eq!(Some(scalar), case.scalar, "{case:?}");
        }
        Err(error) => {
            let kind = match &error {
                ObjectPriceAdvantageError::Direction(_) => "ValueError",
                ObjectPriceAdvantageError::Operation(
                    BuiltinPriceArithmeticError::IntegerOverflow,
                ) => "OverflowError",
                ObjectPriceAdvantageError::Operation(
                    BuiltinPriceArithmeticError::UnsupportedDivision(_),
                ) => "TypeError",
            };
            assert_eq!(
                Some(vec![kind.to_owned(), error.to_string()]),
                case.error,
                "{case:?}"
            );
            assert!(!format!("{error:?}").is_empty());
        }
    }
}

#[test]
fn builtin_object_payloads_match_live_python_types_values_and_errors() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = std::env::var_os("QLIB_PYTHON_RL_EXECUTION_UTILS").map_or_else(
        || root.join("../../../qlib/qlib/rl/order_execution/utils.py"),
        PathBuf::from,
    );
    let result = Command::new(std::env::var_os("PYTHON").unwrap_or_else(|| "python".into()))
        .arg(root.join("tests/fixtures/builtin_object_price.py"))
        .arg(source)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let cases: Vec<Case> = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(cases.len(), 4_608);
    cases.iter().for_each(compare);
}
