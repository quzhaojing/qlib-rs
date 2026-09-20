//! Actual-source missing-object characterization; not a native parity claim.
use serde_json::{Value, json};
use std::{path::PathBuf, process::Command};

#[test]
fn missing_object_identity_depends_on_dtype_and_block() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/dataframe_missing_contract.py"))
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
        "df1b31bdb290fe75ff04105e699af7ba59f3fd3b4d8024030de5d6184488c192"
    );
    let pairs = contract["pairs"].as_array().unwrap();
    assert_eq!(pairs.len(), 841);
    assert_eq!(contract["chains"].as_array().unwrap().len(), 1682);
    let pair = |left: &str, right: &str| {
        pairs
            .iter()
            .find(|c| c["left"] == left && c["right"] == right)
            .unwrap()
    };
    for (left, right, prefix) in [
        (
            "object_none_nan",
            "object_integer",
            json!([["none"], ["none"]]),
        ),
        (
            "object_nan_none",
            "object_integer",
            json!([["float", "nan"], ["float", "nan"]]),
        ),
        (
            "object_none_nan",
            "int64_finite",
            json!([["none"], ["float", "nan"]]),
        ),
        ("object_nat", "float32_finite", json!([["NaT"], ["NaT"]])),
    ] {
        let case = pair(left, right);
        assert_eq!(case["output"]["dtypes"], json!(["object"]));
        assert_eq!(
            json!(&case["output"]["values"][0].as_array().unwrap()[..2]),
            prefix
        );
        assert_eq!(case["warnings"], json!([]));
    }
    let na = pair("object_na", "float32_finite");
    assert_eq!(na["output"]["dtypes"], json!(["float32"]));
    assert_eq!(na["warnings"].as_array().unwrap().len(), 1);
    assert_eq!(na["warnings"][0][0], "FutureWarning");
    let text = pair("object_text", "object_empty");
    assert_eq!(
        text["output"]["values"][0],
        json!([["str", []], ["str", [20013, 25991, 55296]]])
    );
    let blocks = contract["blocks"].as_array().unwrap();
    assert_eq!(blocks.len(), 384);
    assert_all_na_order(blocks);
    for fragmented in [false, true] {
        let case = blocks
            .iter()
            .find(|c| {
                c["state"] == "object_none_nan"
                    && c["sibling"] == "object_integer"
                    && c["partner"] == "float32_finite"
                    && c["swapped"] == false
                    && c["left_fragmented"] == fragmented
                    && c["right_fragmented"] == false
                    && c["reverse"] == false
            })
            .unwrap();
        assert_eq!(
            case["output"]["dtypes"],
            if fragmented {
                json!(["float32", "object"])
            } else {
                json!(["object", "object"])
            }
        );
        assert_eq!(
            case["warnings"].as_array().unwrap().len(),
            usize::from(fragmented)
        );
    }
}

fn assert_all_na_order(blocks: &[Value]) {
    for reverse in [false, true] {
        let case = blocks
            .iter()
            .find(|c| {
                c["state"] == "object_none_nan"
                    && c["sibling"] == "object_nan_none"
                    && c["partner"] == "object_finite_float"
                    && c["swapped"] == false
                    && c["left_fragmented"] == false
                    && c["right_fragmented"] == false
                    && c["reverse"] == reverse
            })
            .unwrap();
        let missing = if reverse {
            json!(["float", "nan"])
        } else {
            json!(["none"])
        };
        for column in case["output"]["values"].as_array().unwrap() {
            assert_eq!(column[0], missing);
            assert_eq!(column[1], missing);
        }
        assert_eq!(case["output"]["dtypes"], json!(["object", "object"]));
        assert_eq!(case["warnings"], json!([]));
    }
}
