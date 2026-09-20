use std::{collections::BTreeMap, path::PathBuf, process::Command};

use domain_core::{Region, RegionConfig};
use serde::Deserialize;
use strum::VariantArray;

#[derive(Debug, Deserialize)]
struct PythonRegionConfig {
    deal_price: String,
    limit_threshold: Option<f64>,
    trade_unit: u32,
}

fn assert_threshold_eq(actual: Option<f64>, expected: Option<f64>) {
    assert_eq!(
        actual.map(f64::to_bits),
        expected.map(f64::to_bits),
        "limit-threshold bits differ"
    );
}

fn assert_config_eq(actual: &RegionConfig, expected: &PythonRegionConfig) {
    assert_eq!(actual.trade_unit, expected.trade_unit);
    assert_threshold_eq(actual.limit_threshold, expected.limit_threshold);
    assert_eq!(actual.deal_price, expected.deal_price);
}

#[test]
fn every_region_has_the_expected_owned_defaults() {
    let cases = [
        (Region::Cn, 100, Some(0.095)),
        (Region::Us, 1, None),
        (Region::Tw, 1_000, Some(0.1)),
    ];

    assert_eq!(cases.len(), Region::VARIANTS.len());
    for (region, trade_unit, limit_threshold) in cases {
        let actual = region.defaults();
        assert_eq!(actual.trade_unit, trade_unit);
        assert_threshold_eq(actual.limit_threshold, limit_threshold);
        assert_eq!(actual.deal_price, "close");
    }

    let mut first = Region::Cn.defaults();
    first.deal_price.push_str("_mutated");
    assert_eq!(Region::Cn.defaults().deal_price, "close");
}

#[test]
fn region_config_round_trips_through_json() {
    for region in Region::VARIANTS {
        let config = region.defaults();
        let json = serde_json::to_string(&config).expect("region config serializes");
        let decoded: RegionConfig =
            serde_json::from_str(&json).expect("region config deserializes");
        assert_eq!(decoded.trade_unit, config.trade_unit);
        assert_threshold_eq(decoded.limit_threshold, config.limit_threshold);
        assert_eq!(decoded.deal_price, config.deal_price);
    }
}

#[test]
fn rust_defaults_match_the_python_source_dictionary() {
    let source = std::env::var_os("QLIB_PYTHON_CONFIG").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/config.py"),
        PathBuf::from,
    );
    assert!(
        source.is_file(),
        "Python source not found: {}",
        source.display()
    );

    let script = r#"
import ast
import json
import sys

tree = ast.parse(open(sys.argv[1], encoding="utf-8").read(), filename=sys.argv[1])
for statement in tree.body:
    if isinstance(statement, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "_default_region_config" for target in statement.targets):
        expression = ast.Expression(statement.value)
        config = eval(compile(expression, sys.argv[1], "eval"), {"REG_CN": "cn", "REG_US": "us", "REG_TW": "tw"})
        print(json.dumps(config, sort_keys=True))
        break
else:
    raise RuntimeError("_default_region_config was not found")
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

    let snapshot: BTreeMap<String, PythonRegionConfig> =
        serde_json::from_slice(&output.stdout).expect("Python returns a JSON object");
    assert_eq!(snapshot.len(), Region::VARIANTS.len());
    for region in Region::VARIANTS {
        let expected = snapshot
            .get(region.code())
            .expect("Python contains every Rust region");
        assert_config_eq(&region.defaults(), expected);
    }
}
