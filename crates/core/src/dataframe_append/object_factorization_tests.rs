use super::*;
use crate::dataframe_append::tuple_objects::tests::{cell, compare};
use serde_json::{Value, json};
use std::{path::PathBuf, process::Command};

fn check(values: &[V], expected: &Value) {
    let before = format!("{values:?}");
    let unique = object_index_is_unique(values).unwrap();
    assert_eq!(json!(unique), expected["unique"], "{expected}");
    assert_eq!(expected["multi"].get("error").is_none(), unique);
    for sentinel in [false, true] {
        let result = factorize_index_objects(values, sentinel).unwrap();
        let expected = &expected[if sentinel { "True" } else { "False" }];
        let codes = result
            .codes
            .iter()
            .map(|code| code.map_or(-1, |code| i64::try_from(code).unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(json!(codes), expected["codes"], "{expected}, {values:?}");
        compare(&result.uniques, expected["uniques"].as_array().unwrap());
        // Repeated factoring of the representatives must be idempotent, except
        // the optional top-level NaN that is intentionally represented by -1.
        let repeated = factorize_index_objects(&result.uniques, sentinel).unwrap();
        compare(&repeated.uniques, expected["uniques"].as_array().unwrap());
    }
    assert_eq!(format!("{values:?}"), before);
}

#[test]
fn object_level_equivalence_and_stable_factorization_match_actual_source() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/dataframe_object_factorization.py"))
        .arg(root.join("../../../qlib/qlib/rl/order_execution/utils.py"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let contract: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(contract["pandas"], "2.3.3");
    assert_eq!(contract["numpy"], "2.4.0");
    assert_eq!(
        contract["digest"],
        "2e5bc16d4cc253e6a9ecdb220557dc97333c9ec855d12ae271d85f06e56aae54"
    );
    let values = contract["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(cell)
        .collect::<Vec<_>>();
    assert_eq!(values.len(), 93);
    let pairs = contract["pairs"].as_array().unwrap();
    assert_eq!(pairs.len(), 8649);
    assert_eq!(pairs.iter().filter(|p| p["unique"] == false).count(), 209);
    for case in pairs {
        let left = usize::try_from(case["left"].as_u64().unwrap()).unwrap();
        let right = usize::try_from(case["right"].as_u64().unwrap()).unwrap();
        check(&[values[left].clone(), values[right].clone()], case);
    }
    let histories = contract["histories"].as_array().unwrap();
    assert_eq!(histories.len(), 6);
    for case in histories {
        check(
            &case["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(cell)
                .collect::<Vec<_>>(),
            case,
        );
    }
}

#[test]
fn invalid_temporal_keys_fail_even_after_a_duplicate_or_missing_value() {
    let valid = V::Scalar(T::Builtin(B::None));
    for invalid in [
        V::Scalar(T::Timestamp {
            ticks: i64::MIN,
            unit: TimeUnit::Second,
            timezone: None,
        }),
        V::Scalar(T::Duration {
            ticks: i64::MIN,
            unit: TimeUnit::Nanosecond,
        }),
    ] {
        for value in [invalid.clone(), V::Tuple(vec![invalid])] {
            let values = [valid.clone(), valid.clone(), value];
            let before = format!("{values:?}");
            assert_eq!(
                object_index_is_unique(&values).unwrap_err().to_string(),
                "Invalid argument error: temporal object uses reserved NaT ticks"
            );
            for sentinel in [false, true] {
                assert_eq!(
                    factorize_index_objects(&values, sentinel)
                        .unwrap_err()
                        .to_string(),
                    "Invalid argument error: temporal object uses reserved NaT ticks"
                );
            }
            assert_eq!(format!("{values:?}"), before);
        }
    }
}
