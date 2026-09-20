use std::{path::PathBuf, process::Command};

use chrono::TimeDelta;
use domain_core::{
    EPS, EPS_T, FloatOrNdarray, INF, ONE_DAY, ONE_MIN, REG_CN, REG_TW, REG_US, Region,
};
use ndarray::{arr1, arr2};
use serde::Deserialize;
use strum::VariantArray;

#[derive(Debug, Deserialize, PartialEq)]
struct PythonConstantSnapshot {
    eps: f64,
    eps_t_ns: i64,
    inf: i64,
    one_day_ns: i64,
    one_min_ns: i64,
    reg_cn: String,
    reg_tw: String,
    reg_us: String,
    typevar_constraints: Vec<String>,
    typevar_name: String,
    public_names: Vec<String>,
    runtime_types: Vec<String>,
    source_sha256: String,
}

fn duration_ns(value: TimeDelta) -> i64 {
    value
        .num_nanoseconds()
        .expect("Qlib duration constants fit in i64 nanoseconds")
}

fn accepts_float_or_ndarray<T: FloatOrNdarray>(_value: &T) {}

#[test]
fn runtime_constants_match_the_python_contract() {
    assert_eq!(REG_CN, "cn");
    assert_eq!(REG_US, "us");
    assert_eq!(REG_TW, "tw");
    assert_eq!(EPS.to_bits(), 1e-12_f64.to_bits());
    assert_eq!(INF, 1_000_000_000_000_000_000);
    assert_eq!(duration_ns(ONE_DAY), 86_400_000_000_000);
    assert_eq!(duration_ns(ONE_MIN), 60_000_000_000);
    assert_eq!(duration_ns(EPS_T), 1_000_000_000);
    accepts_float_or_ndarray(&1.0_f64);
    accepts_float_or_ndarray(&arr1(&[1_i64, 2]));
    accepts_float_or_ndarray(&arr2(&[[1.0_f32, 2.0], [3.0, 4.0]]));
}

#[test]
fn regions_support_stable_strings_parsing_and_json() {
    let expected = [
        (Region::Cn, REG_CN),
        (Region::Us, REG_US),
        (Region::Tw, REG_TW),
    ];

    assert_eq!(Region::VARIANTS.len(), expected.len());
    for (region, text) in expected {
        assert_eq!(region.code(), text);
        assert_eq!(region.as_ref(), text);
        assert_eq!(region.to_string(), text);
        assert_eq!(text.parse::<Region>(), Ok(region));

        let json = serde_json::to_string(&region).expect("region serialization succeeds");
        assert_eq!(json, format!("\"{text}\""));
        assert_eq!(
            serde_json::from_str::<Region>(&json).expect("region deserialization succeeds"),
            region
        );
    }

    assert!("CN".parse::<Region>().is_err());
    assert!(serde_json::from_str::<Region>("\"unknown\"").is_err());
}

#[test]
fn rust_values_match_a_live_python_snapshot() {
    let source = std::env::var_os("QLIB_PYTHON_CONSTANT").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/constant.py"),
        PathBuf::from,
    );
    assert!(
        source.is_file(),
        "Python source not found: {}",
        source.display()
    );

    let script = r#"
import importlib.util
import hashlib
import json
import sys

raw = open(sys.argv[1], "rb").read()
spec = importlib.util.spec_from_file_location("qlib_constant_snapshot", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(json.dumps({
    "eps": module.EPS,
    "eps_t_ns": module.EPS_T.value,
    "inf": module.INF,
    "one_day_ns": module.ONE_DAY.value,
    "one_min_ns": module.ONE_MIN.value,
    "reg_cn": module.REG_CN,
    "reg_tw": module.REG_TW,
    "reg_us": module.REG_US,
    "typevar_constraints": [f"{item.__module__}.{item.__qualname__}" for item in module.float_or_ndarray.__constraints__],
    "typevar_name": module.float_or_ndarray.__name__,
    "public_names": [name for name in vars(module) if not name.startswith("_")],
    "runtime_types": [
        f"{type(value).__module__}.{type(value).__qualname__}"
        for value in [module.REG_CN, module.EPS, module.INF, module.ONE_DAY, module.float_or_ndarray]
    ],
    "source_sha256": hashlib.sha256(raw).hexdigest(),
}, sort_keys=True))
"#;
    let python = std::env::var_os("PYTHON").unwrap_or_else(|| "python".into());
    let output = Command::new(python)
        .arg("-c")
        .arg(script)
        .arg(&source)
        .output()
        .expect("Python interpreter starts");
    assert!(
        output.status.success(),
        "Python snapshot failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let actual: PythonConstantSnapshot =
        serde_json::from_slice(&output.stdout).expect("Python returns a JSON snapshot");
    let expected = PythonConstantSnapshot {
        eps: EPS,
        eps_t_ns: duration_ns(EPS_T),
        inf: INF,
        one_day_ns: duration_ns(ONE_DAY),
        one_min_ns: duration_ns(ONE_MIN),
        reg_cn: REG_CN.to_owned(),
        reg_tw: REG_TW.to_owned(),
        reg_us: REG_US.to_owned(),
        typevar_constraints: vec!["builtins.float".to_owned(), "numpy.ndarray".to_owned()],
        typevar_name: "float_or_ndarray".to_owned(),
        public_names: [
            "TypeVar",
            "np",
            "pd",
            "REG_CN",
            "REG_US",
            "REG_TW",
            "EPS",
            "INF",
            "ONE_DAY",
            "ONE_MIN",
            "EPS_T",
            "float_or_ndarray",
        ]
        .map(str::to_owned)
        .to_vec(),
        runtime_types: [
            "builtins.str",
            "builtins.float",
            "builtins.int",
            "pandas._libs.tslibs.timedeltas.Timedelta",
            "typing.TypeVar",
        ]
        .map(str::to_owned)
        .to_vec(),
        source_sha256: "52dbdc78a580b75c7bda9b6532f65d791d1e882d91181a1e6b8ca107e6c39da8"
            .to_owned(),
    };
    assert_eq!(actual, expected);
}
