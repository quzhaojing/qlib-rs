use super::*;
use crate::dataframe_append::tuple_objects::tests::{cell, compare as compare_cells};
use serde_json::{Value, json};
use std::{path::PathBuf, process::Command};

fn check(values: &[TupleFrameValue], expected: &Value) {
    let before = format!("{values:?}");
    let result = factorize_level_objects(values).unwrap();
    // This primitive implements factorize(sort=True), with sort=False fallback.
    // Categorical's subsequent construction is a separate boundary (and can
    // refactor representatives again); do not conflate those two stages.
    let expected = if expected["sorted"].get("error").is_none() {
        &expected["sorted"]
    } else {
        expected.get("fallback").unwrap_or(&expected["multi"])
    };
    let codes = result
        .codes
        .iter()
        .map(|c| c.map_or(-1, |v| i64::try_from(v).unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(json!(codes), expected["codes"], "{values:?}: {expected}");
    compare_cells(&result.uniques, expected["uniques"].as_array().unwrap());
    assert_eq!(before, format!("{values:?}"));
}

#[test]
fn mixed_sort_preserves_missing_tail_and_rejects_incompatible_values() {
    let text = Key::Text(crate::RlCheckpointText::from_utf8("a"));
    let keys = vec![
        text,
        Key::None,
        Key::Integer(2.into()),
        Key::Na,
        Key::Nat,
        Key::Nan,
    ];
    assert_eq!(mixed(&keys), Some(vec![2, 0, 1, 3, 4, 5]));
    assert_eq!(mixed(&[Key::Timestamp(0, false), Key::Duration(0)]), None);
}

#[test]
fn sorted_object_long_sequences_match_source() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/dataframe_sorted_sequences.py"))
        .arg(root.join("../../../qlib/qlib/rl/order_execution/utils.py"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let contract: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        contract["digest"],
        "8b97ccc36953849ad4272cd8cb5d915ea5ffa47f0778b36085f1db929bdf57a7"
    );
    let cases = contract["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 432);
    for case in cases {
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
fn sorted_object_invalid_keys_do_not_publish_codes() {
    let invalid = TupleFrameValue::Scalar(super::super::TemporalFrameValue::Duration {
        ticks: i64::MIN,
        unit: arrow_schema::TimeUnit::Second,
    });
    let values = [TupleFrameValue::Tuple(vec![]), invalid];
    assert_eq!(
        factorize_level_objects(&values).unwrap_err().to_string(),
        "Invalid argument error: temporal object uses reserved NaT ticks"
    );
}

#[test]
fn sorted_object_levels_and_categorical_fallback_match_source() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/dataframe_sorted_factorization.py"))
        .arg(root.join("../../../qlib/qlib/rl/order_execution/utils.py"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let contract: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        contract["digest"],
        "500dbb6034120b2c0e5cefb04f4a95cf819edc8b4e199258470bf82bd2ac6dba"
    );
    let values = contract["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(cell)
        .collect::<Vec<_>>();
    let pairs = contract["pairs"].as_array().unwrap();
    assert_eq!(pairs.len(), 8649);
    for case in pairs {
        let left = usize::try_from(case["left"].as_u64().unwrap()).unwrap();
        let right = usize::try_from(case["right"].as_u64().unwrap()).unwrap();
        check(&[values[left].clone(), values[right].clone()], case);
    }
    for case in contract["histories"].as_array().unwrap() {
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
