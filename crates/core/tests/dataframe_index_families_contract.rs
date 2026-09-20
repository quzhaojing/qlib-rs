//! Source characterization for index families not yet represented natively.
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::PathBuf, process::Command};

#[test]
fn source_index_families_preserve_metadata_and_directional_coercion() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/dataframe_index_families.py"))
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
        "8b371ec25bd85b0ad03573f1e346679c903617527c7163cec32c871f485d36e4"
    );
    assert_eq!(contract["inputs"].as_object().unwrap().len(), 44);
    assert_eq!(contract["chains"].as_array().unwrap().len(), 220);
    let cases = contract["pairs"].as_array().unwrap();
    assert_eq!(cases.len(), 7744);
    let mut families = BTreeMap::new();
    for case in cases {
        assert!(case.get("error").is_none(), "{case}");
        *families
            .entry(case["output"]["index"]["kind"].as_str().unwrap())
            .or_insert(0_usize) += 1;
    }
    assert_eq!(
        families,
        BTreeMap::from([
            ("Index", 5350),
            ("RangeIndex", 168),
            ("CategoricalIndex", 610),
            ("PeriodIndex", 386),
            ("IntervalIndex", 386),
            ("DatetimeIndex", 382),
            ("TimedeltaIndex", 236),
            ("MultiIndex", 226),
        ])
    );
    let find = |left: &str, right: &str, right_columns: bool| {
        &cases
            .iter()
            .find(|c| {
                c["left"] == left
                    && c["right"] == right
                    && c["left_columns"] == true
                    && c["right_columns"] == right_columns
            })
            .unwrap()["output"]["index"]
    };
    let range = find("range_start", "range_empty", false);
    assert_eq!(range["kind"], "RangeIndex");
    assert_eq!(range["range"], json!([0, 2, 1]));
    assert_eq!(find("range_start", "range_next", true)["kind"], "Index");
    let multi = find("multi_values", "tuple_values", true);
    assert_eq!(multi["kind"], "MultiIndex");
    assert_eq!(multi["names"], json!([["none"], ["none"]]));
    assert_eq!(multi["codes"], json!([[0, 1, 0, 1], [0, 1, 0, 1]]));
    assert_eq!(find("tuple_values", "multi_values", true)["kind"], "Index");
    // Actual source truncates longer right tuples to the existing MultiIndex depth.
    assert_eq!(
        find("multi_values", "multi_three", true)["values"],
        multi["values"]
    );
    let category = find("category", "category_reordered", true);
    assert_eq!(category["kind"], "CategoricalIndex");
    assert_eq!(category["ordered"], false);
    assert_eq!(category["codes"], json!([0, 1, 1, 0]));
    assert_eq!(
        category["categories"],
        contract["inputs"]["category"]["categories"]
    );
    assert_eq!(find("period_month", "period_day", true)["dtype"], "object");
    assert_eq!(
        find("interval_right", "interval_left", true)["dtype"],
        "object"
    );
    let nullable = find("nullable_Int64_missing", "int_values", true);
    assert_eq!(nullable["dtype"], "Int64");
    assert_eq!(nullable["values"][1], json!(["pd.NA"]));
    assert_eq!(
        find("datetime_regular", "datetime_next", true)["freq"],
        Value::Null
    );
}
