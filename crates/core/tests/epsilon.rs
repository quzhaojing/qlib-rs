use std::{path::PathBuf, process::Command, str::FromStr};

use chrono::NaiveDateTime;
use domain_core::{
    EpsilonDirection, EpsilonError, epsilon_change, epsilon_change_backward, epsilon_change_str,
};
use serde_json::{Value, json};

fn timestamp(text: &str) -> NaiveDateTime {
    NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S%.f").expect("fixture timestamp is valid")
}

#[test]
fn directions_shift_exactly_one_second_and_have_stable_strings() {
    let value = timestamp("2021-01-01 00:00:00.123456789");
    assert_eq!(
        epsilon_change_backward(value),
        Ok(timestamp("2020-12-31 23:59:59.123456789"))
    );
    assert_eq!(
        epsilon_change(value, EpsilonDirection::Forward),
        Ok(timestamp("2021-01-01 00:00:01.123456789"))
    );
    assert_eq!(
        epsilon_change_str(value, "backward"),
        Ok(timestamp("2020-12-31 23:59:59.123456789"))
    );
    assert_eq!(
        epsilon_change_str(value, "forward"),
        Ok(timestamp("2021-01-01 00:00:01.123456789"))
    );

    for (direction, text) in [
        (EpsilonDirection::Backward, "backward"),
        (EpsilonDirection::Forward, "forward"),
    ] {
        assert_eq!(direction.to_string(), text);
        assert_eq!(EpsilonDirection::from_str(text), Ok(direction));
        let encoded = serde_json::to_string(&direction).expect("direction serializes");
        assert_eq!(encoded, format!("\"{text}\""));
        assert_eq!(
            serde_json::from_str::<EpsilonDirection>(&encoded).expect("direction deserializes"),
            direction
        );
    }
    assert!(EpsilonDirection::from_str("Backward").is_err());
    assert!(serde_json::from_str::<EpsilonDirection>("\"sideways\"").is_err());
}

#[test]
fn invalid_directions_and_every_pandas_range_boundary_are_typed_errors() {
    let value = timestamp("2021-01-01 00:00:00");
    for direction in ["", "Backward", "sideways"] {
        assert_eq!(
            epsilon_change_str(value, direction),
            Err(EpsilonError::InvalidDirection {
                direction: direction.to_owned()
            })
        );
    }

    let minimum = timestamp("1677-09-21 00:12:43.145224193");
    let maximum = timestamp("2262-04-11 23:47:16.854775807");
    assert_eq!(
        epsilon_change(minimum, EpsilonDirection::Forward),
        Ok(timestamp("1677-09-21 00:12:44.145224193"))
    );
    assert_eq!(
        epsilon_change(maximum, EpsilonDirection::Backward),
        Ok(timestamp("2262-04-11 23:47:15.854775807"))
    );
    assert_eq!(
        epsilon_change(minimum, EpsilonDirection::Backward),
        Err(EpsilonError::ResultOutOfRange {
            timestamp: minimum,
            direction: EpsilonDirection::Backward
        })
    );
    assert_eq!(
        epsilon_change(maximum, EpsilonDirection::Forward),
        Err(EpsilonError::ResultOutOfRange {
            timestamp: maximum,
            direction: EpsilonDirection::Forward
        })
    );

    for outside in [
        timestamp("1677-09-21 00:12:43.145224192"),
        timestamp("1600-01-01 00:00:00"),
        timestamp("9999-12-31 23:59:58"),
    ] {
        assert_eq!(
            epsilon_change(outside, EpsilonDirection::Forward),
            Err(EpsilonError::InputOutOfRange { timestamp: outside })
        );
    }
}

#[test]
fn rust_contract_matches_a_live_python_epsilon_snapshot() {
    let source = std::env::var_os("QLIB_PYTHON_TIME").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/utils/time.py"),
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
import pandas as pd

tree = ast.parse(open(sys.argv[1], encoding="utf-8").read(), filename=sys.argv[1])
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "epsilon_change")
module = ast.Module(body=[function], type_ignores=[])
ast.fix_missing_locations(module)
namespace = {"pd": pd}
exec(compile(module, sys.argv[1], "exec"), namespace)
shift = namespace["epsilon_change"]

result = {}
for name, value, direction, use_default in [
    ("default", pd.Timestamp("2021-01-01 00:00:00.123456789"), "backward", True),
    ("forward", pd.Timestamp("2021-01-01 00:00:00.123456789"), "forward", False),
    ("timezone", pd.Timestamp("2021-03-14 01:59:59-05:00"), "forward", False),
    ("minimum_forward", pd.Timestamp.min, "forward", False),
    ("maximum_backward", pd.Timestamp.max, "backward", False),
    ("nat_backward", pd.NaT, "backward", False),
    ("nat_forward", pd.NaT, "forward", False),
]:
    shifted = shift(value) if use_default else shift(value, direction)
    result[name] = {"value": str(shifted), "is_nat": bool(pd.isna(shifted)), "tz": str(getattr(shifted, "tz", None))}

for name, value, direction in [
    ("minimum_backward", pd.Timestamp.min, "backward"),
    ("maximum_forward", pd.Timestamp.max, "forward"),
    ("far", pd.Timestamp("9999-12-31 23:59:58"), "forward"),
    ("uppercase", pd.Timestamp("2021-01-01"), "Backward"),
    ("empty", pd.Timestamp("2021-01-01"), ""),
]:
    try:
        shift(value, direction)
    except Exception as error:
        result[name] = type(error).__name__
print(json.dumps(result, sort_keys=True))
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
    let actual: Value =
        serde_json::from_slice(&output.stdout).expect("Python returns a JSON snapshot");
    let expected = json!({
        "default": {"value": "2020-12-31 23:59:59.123456789", "is_nat": false, "tz": "None"},
        "forward": {"value": "2021-01-01 00:00:01.123456789", "is_nat": false, "tz": "None"},
        "timezone": {"value": "2021-03-14 02:00:00-05:00", "is_nat": false, "tz": "UTC-05:00"},
        "minimum_forward": {"value": "1677-09-21 00:12:44.145224193", "is_nat": false, "tz": "None"},
        "maximum_backward": {"value": "2262-04-11 23:47:15.854775807", "is_nat": false, "tz": "None"},
        "nat_backward": {"value": "NaT", "is_nat": true, "tz": "None"},
        "nat_forward": {"value": "NaT", "is_nat": true, "tz": "None"},
        "minimum_backward": "OutOfBoundsDatetime",
        "maximum_forward": "OutOfBoundsDatetime",
        "far": "OutOfBoundsDatetime",
        "uppercase": "ValueError",
        "empty": "ValueError",
    });
    assert_eq!(actual, expected);
}
