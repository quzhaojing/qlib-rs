use std::{fmt::Debug, path::PathBuf, process::Command, str::FromStr};

use domain_core::{
    complex_price_advantage::{
        ComplexPriceAdvantage, ComplexPriceComponent, complex_price_advantage,
    },
    price_advantage::PriceAdvantageError,
};
use ndarray::{ArrayD, Axis, IxDyn, array};
use num_complex::Complex;
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Debug, Deserialize)]
struct Case {
    dtype: String,
    shape: Vec<usize>,
    values: Vec<[String; 2]>,
    baseline: String,
    direction: i64,
    output: Option<Value>,
    error: Option<String>,
}

fn snapshot(dtype: &str, shape: &[usize], values: impl Iterator<Item = Complex<f64>>) -> Value {
    json!({"dtype": dtype, "shape": shape, "bits": values.map(|v| [format!("{:016x}", v.re.to_bits()), format!("{:016x}", v.im.to_bits())]).collect::<Vec<_>>()})
}

fn run_case<T>(case: &Case) -> Result<Value, PriceAdvantageError>
where
    T: ComplexPriceComponent + FromStr,
    T::Err: Debug,
{
    let prices = ArrayD::from_shape_vec(
        IxDyn(&case.shape),
        case.values
            .iter()
            .map(|v| Complex::new(v[0].parse::<T>().unwrap(), v[1].parse::<T>().unwrap()))
            .collect(),
    )
    .unwrap();
    match complex_price_advantage(
        &prices.view(),
        case.baseline.parse().unwrap(),
        case.direction,
    )? {
        ComplexPriceAdvantage::Scalar(value) => Ok(snapshot("scalar", &[], std::iter::once(value))),
        ComplexPriceAdvantage::Array(values) => Ok(snapshot(
            &case.dtype,
            values.shape(),
            values.iter().map(|v| Complex::new(v.re.as_(), v.im.as_())),
        )),
    }
}

#[test]
fn complex_dtypes_match_numpy_bitwise_for_each_component_and_shape() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = std::env::var_os("QLIB_PYTHON_RL_EXECUTION_UTILS").map_or_else(
        || root.join("../../../qlib/qlib/rl/order_execution/utils.py"),
        PathBuf::from,
    );
    let result = Command::new(std::env::var_os("PYTHON").unwrap_or_else(|| "python".into()))
        .arg(root.join("tests/fixtures/complex_price_advantage.py"))
        .arg(source)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let cases: Vec<Case> = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(cases.len(), 50_688);
    for case in &cases {
        let actual = match case.dtype.as_str() {
            "complex64" => run_case::<f32>(case),
            "complex128" => run_case::<f64>(case),
            other => panic!("unknown dtype {other}"),
        };
        match actual {
            Ok(value) => assert_eq!(Some(value), case.output, "{case:?}"),
            Err(error) => assert_eq!(Some(error.to_string()), case.error, "{case:?}"),
        }
    }
}

#[test]
fn complex_views_preserve_component_order_and_original_storage() {
    let prices = array![[Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]];
    let original = prices.clone();
    let mut view = prices.t();
    view.invert_axis(Axis(0));
    let ComplexPriceAdvantage::Array(actual) = complex_price_advantage(&view, 2.0, 0).unwrap()
    else {
        panic!("array expected")
    };
    assert_eq!(
        actual,
        array![
            [Complex::new(5_000.0, 20_000.0)],
            [Complex::new(-5_000.0, 10_000.0)]
        ]
    );
    assert_eq!(prices, original);
}
