use std::{fmt::Debug, path::PathBuf, process::Command, str::FromStr};

use domain_core::price_advantage::{
    ArrayPriceAdvantage, PriceAdvantageError, PriceElement, PriceFloat, price_advantage,
    price_advantage_array,
};
use half::f16;
use ndarray::{ArrayD, Axis, IxDyn, array};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Debug, Deserialize)]
struct Case {
    dtype: String,
    shape: Vec<usize>,
    values: Vec<String>,
    baseline: String,
    direction: i64,
    output: Option<Value>,
    error: Option<String>,
}

fn snapshot(dtype: &str, shape: &[usize], values: impl Iterator<Item = f64>) -> Value {
    json!({"dtype": dtype, "shape": shape, "bits": values.map(|v| format!("{:016x}", v.to_bits())).collect::<Vec<_>>()})
}

fn array_case<T>(case: &Case, output_dtype: &str) -> Result<Value, PriceAdvantageError>
where
    T: PriceElement + FromStr,
    T::Err: Debug,
{
    let prices = ArrayD::from_shape_vec(
        IxDyn(&case.shape),
        case.values
            .iter()
            .map(|v| v.parse::<T>().unwrap())
            .collect(),
    )
    .unwrap();
    match price_advantage_array(
        &prices.view(),
        case.baseline.parse().unwrap(),
        case.direction,
    )? {
        ArrayPriceAdvantage::ZeroBaseline(values) => Ok(snapshot(
            &case.dtype,
            values.shape(),
            values.iter().map(|v| v.promoted().as_f64()),
        )),
        ArrayPriceAdvantage::Scalar(value) => Ok(snapshot("scalar", &[], std::iter::once(value))),
        ArrayPriceAdvantage::Array(values) => Ok(snapshot(
            output_dtype,
            values.shape(),
            values.iter().map(|v| v.as_f64()),
        )),
    }
}

#[test]
fn real_numpy_differential_covers_real_dtypes_shapes_directions_and_ieee_values() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = std::env::var_os("QLIB_PYTHON_RL_EXECUTION_UTILS").map_or_else(
        || root.join("../../../qlib/qlib/rl/order_execution/utils.py"),
        PathBuf::from,
    );
    let result = Command::new(std::env::var_os("PYTHON").unwrap_or_else(|| "python".into()))
        .arg(root.join("tests/fixtures/price_advantage.py"))
        .arg(source)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let cases: Vec<Case> = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(cases.len(), 3528);
    for case in &cases {
        let actual = match case.dtype.as_str() {
            "scalar" => price_advantage(
                case.values[0].parse().unwrap(),
                case.baseline.parse().unwrap(),
                case.direction,
            )
            .map(|v| snapshot("scalar", &[], std::iter::once(v))),
            "float16" => array_case::<f16>(case, "float16"),
            "float32" => array_case::<f32>(case, "float32"),
            "float64" => array_case::<f64>(case, "float64"),
            "bool" => array_case::<bool>(case, "float64"),
            "int8" => array_case::<i8>(case, "float64"),
            "uint8" => array_case::<u8>(case, "float64"),
            "int16" => array_case::<i16>(case, "float64"),
            "uint16" => array_case::<u16>(case, "float64"),
            "int32" => array_case::<i32>(case, "float64"),
            "uint32" => array_case::<u32>(case, "float64"),
            "int64" => array_case::<i64>(case, "float64"),
            "uint64" => array_case::<u64>(case, "float64"),
            other => panic!("unknown dtype {other}"),
        };
        match actual {
            Ok(value) => assert_eq!(Some(value), case.output, "{case:?}"),
            Err(error) => {
                assert_eq!(Some(error.to_string()), case.error, "{case:?}");
                assert_eq!(error, PriceAdvantageError(case.direction));
                assert!(format!("{error:?}").contains("PriceAdvantageError"));
            }
        }
    }
}

#[test]
fn reversed_transposed_views_preserve_logical_values_and_leave_input_unchanged() {
    let prices = array![[1.0, 2.0], [3.0, 4.0]];
    let mut reversed = prices.t();
    reversed.invert_axis(Axis(0));
    let ArrayPriceAdvantage::Array(actual) = price_advantage_array(&reversed, 2.0, 0).unwrap()
    else {
        panic!("matrix expected")
    };
    assert_eq!(actual, array![[0.0, 10_000.0], [-5_000.0, 5_000.0]]);
    assert_eq!(prices, array![[1.0, 2.0], [3.0, 4.0]]);
}
