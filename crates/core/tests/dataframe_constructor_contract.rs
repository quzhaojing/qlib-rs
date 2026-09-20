//! Actual-source constructor requirements, not native constructor parity.
use serde_json::{Value, json};
use std::{path::PathBuf, process::Command};

#[test]
fn dataframe_construction_preserves_inference_alignment_and_error_order() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/dataframe_constructor_contract.py"))
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
        "f98dd8ffe714ec2686ab5521551561e485e372f1bd6dfc4cd4173796178eb904"
    );
    let cases = contract["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 1158);
    assert_eq!(
        cases.iter().filter(|c| c.get("output").is_some()).count(),
        1012
    );
    assert_eq!(
        cases.iter().filter(|c| c["error"] == "ValueError").count(),
        123
    );
    assert_eq!(
        cases.iter().filter(|c| c["error"] == "KeyError").count(),
        21
    );
    assert_eq!(
        cases
            .iter()
            .filter(|c| c["error"] == "InvalidIndexError")
            .count(),
        2
    );
    assert!(
        cases
            .iter()
            .all(|c| c["construction_warnings"] == json!([]))
    );
    assert_scalars_and_records(cases);
    assert_alignment_and_errors(cases);
}

fn case<'a>(cases: &'a [Value], name: &str, left: &str) -> &'a Value {
    cases
        .iter()
        .find(|c| c["name"] == name && c["left"] == left)
        .unwrap()
}

fn assert_scalars_and_records(cases: &[Value]) {
    let scalar = case(cases, "scalar/integer/3", "empty");
    assert_eq!(
        scalar["construction"]["dtypes"],
        json!(["datetime64[ns]", "int64", "object"])
    );
    assert_eq!(
        scalar["construction"]["values"][1],
        json!([["int", "-3"], ["int", "-3"], ["int", "-3"]])
    );
    assert_eq!(scalar["output"]["dtypes"], json!(["int64", "object"]));
    assert_eq!(scalar["warnings"], json!([]));
    let empty = case(cases, "scalar/integer/0", "empty");
    assert_eq!(
        empty["construction"]["dtypes"],
        scalar["construction"]["dtypes"]
    );
    assert_eq!(empty["construction"]["values"], json!([[], [], []]));
    for (name, dtype, value) in [
        ("unsigned", "uint64", "18446744073709551615"),
        ("bigint", "object", "1208925819614629174706176"),
    ] {
        let c = case(cases, &format!("scalar/{name}/3"), "empty");
        assert_eq!(c["construction"]["dtypes"][1], dtype);
        assert_eq!(c["construction"]["values"][1][0], json!(["int", value]));
    }
    let missing = case(cases, "records/integer/none/False", "empty");
    assert_eq!(
        missing["construction"]["dtypes"],
        json!(["datetime64[ns]", "float64", "object"])
    );
    assert_eq!(
        missing["construction"]["values"][1],
        json!([["float", "-0x1.8000000000000p+1"], ["float", "nan"]])
    );
    let boolean = case(cases, "records/false/none/False", "empty");
    assert_eq!(boolean["construction"]["dtypes"][1], "object");
    assert_eq!(
        boolean["construction"]["values"][1],
        json!([["bool", false], ["none"]])
    );
    let omitted = case(cases, "records/false/none/True", "empty");
    assert_eq!(
        omitted["construction"]["values"][1],
        json!([["bool", false], ["float", "nan"]])
    );
    let initialized = case(cases, "scalar/integer/3", "initialized");
    assert_eq!(initialized["output"]["dtypes"], json!(["object", "object"]));
    assert_eq!(initialized["warnings"].as_array().unwrap().len(), 1);
    assert_eq!(initialized["warnings"][0][0], "FutureWarning");
    assert_eq!(initialized["output"]["index_dtype"], "datetime64[ns]");
}

fn assert_alignment_and_errors(cases: &[Value]) {
    let series = case(cases, "series/[2, 3]", "empty");
    assert_eq!(
        series["construction"]["index"],
        json!([["int", "0"], ["int", "1"], ["int", "2"], ["int", "3"]])
    );
    assert_eq!(
        series["construction"]["values"][1],
        json!([
            ["float", "nan"],
            ["float", "nan"],
            ["float", "0x1.0000000000000p+0"],
            ["float", "0x1.0000000000000p+1"]
        ])
    );
    assert_eq!(series["output"]["index"][2], json!(["NaT"]));
    let reordered = case(cases, "special/record_order", "empty");
    assert_eq!(
        reordered["construction"]["columns"][0],
        json!(["str", [120]])
    );
    assert_eq!(
        reordered["construction"]["blocks"],
        json!([["float64", [0, 2]], ["int64", [1]]])
    );
    assert_eq!(reordered["output"]["blocks"], json!([["float64", [0, 1]]]));
    for (name, error, message) in [
        (
            "special/scalar_only",
            "ValueError",
            "If using all scalar values, you must pass an index",
        ),
        (
            "special/unequal",
            "ValueError",
            "All arrays must be of the same length",
        ),
        (
            "special/rank3",
            "ValueError",
            "Must pass 2-d input. shape=(1, 1, 1)",
        ),
        (
            "series/[0, 0]",
            "ValueError",
            "cannot reindex on an axis with duplicate labels",
        ),
    ] {
        let c = case(cases, name, "empty");
        assert_eq!(c["error"], error);
        assert_eq!(c["message"], message);
        assert_eq!(c["construction"]["message"], message);
    }
    for (name, error, message) in [
        (
            "special/missing_datetime",
            "KeyError",
            "\"None of ['datetime'] are in the columns\"",
        ),
        (
            "special/duplicate_datetime",
            "ValueError",
            "Index data must be 1-dimensional",
        ),
    ] {
        let c = case(cases, name, "empty");
        assert!(c["construction"].get("error").is_none());
        assert_eq!(c["error"], error);
        assert_eq!(c["message"], message);
    }
    assert!(
        case(cases, "special/duplicate_data", "empty")
            .get("output")
            .is_some()
    );
    assert_eq!(
        case(cases, "special/duplicate_data", "populated")["error"],
        "InvalidIndexError"
    );
}
