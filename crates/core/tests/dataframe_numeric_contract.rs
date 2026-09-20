//! Source characterization: no claim that native numeric promotion is complete.

use std::{path::PathBuf, process::Command};

use serde_json::{Value, json};

#[test]
fn source_numeric_promotion_requires_block_metadata() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/dataframe_numeric_contract.py"))
        .arg(root.join("../../../qlib/qlib/rl/order_execution/utils.py"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let contract: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_object_contract(&contract);
    assert_eq!(contract["pandas"], "2.3.3");
    assert_eq!(contract["numpy"], "2.4.0");
    assert_eq!(
        contract["digest"],
        "fb3fda0b9bd6f71c99f6c9d57347bae8d8ade5c61ac6a8fbc012d8977bb4ef71"
    );
    let numeric = contract["numeric"].as_array().unwrap();
    let blocks = contract["blocks"].as_array().unwrap();
    assert_eq!(numeric.len(), 1521);
    assert_eq!(blocks.len(), 16);
    let finite = |left: &str, right: &str| {
        numeric
            .iter()
            .find(|case| {
                case["left"] == json!([left, "finite"]) && case["right"] == json!([right, "finite"])
            })
            .unwrap()
    };
    assert_eq!(
        finite("bool", "float32")["output"]["dtypes"],
        json!(["object"])
    );
    assert_eq!(
        finite("float32", "bool")["output"]["dtypes"],
        json!(["float32"])
    );
    assert_eq!(
        finite("int64", "uint64")["output"]["dtypes"],
        json!(["float64"])
    );
    assert_eq!(
        finite("int8", "float16")["output"]["dtypes"],
        json!(["float16"])
    );
    assert_eq!(
        finite("int16", "float16")["output"]["dtypes"],
        json!(["float32"])
    );
    let block = |fragmented: bool| {
        blocks
            .iter()
            .find(|case| {
                case["left_fragmented"] == fragmented
                    && case["right_fragmented"] == false
                    && case["reverse"] == false
                    && case["all_na"] == false
            })
            .unwrap()
    };
    assert_eq!(
        block(false)["left_input"]["values"],
        block(true)["left_input"]["values"]
    );
    assert_eq!(
        block(false)["left_input"]["dtypes"],
        block(true)["left_input"]["dtypes"]
    );
    assert_eq!(
        block(false)["output"]["dtypes"],
        json!(["float64", "float64"])
    );
    assert_eq!(
        block(true)["output"]["dtypes"],
        json!(["float32", "float64"])
    );
    assert_ne!(
        block(false)["left_input"]["blocks"],
        block(true)["left_input"]["blocks"]
    );
    assert!(!block(true)["warnings"].as_array().unwrap().is_empty());
}

fn assert_object_contract(contract: &Value) {
    assert_eq!(
        contract["object_digest"],
        "1332cf72009d83f3ed1bdef4b586be24cacfba1a4a2a4ba44e509ea4f15d73fd"
    );
    let chains = contract["object_contract"]["chains"].as_array().unwrap();
    let blocks = contract["object_contract"]["blocks"].as_array().unwrap();
    assert_eq!(chains.len(), 2808);
    assert_eq!(blocks.len(), 96);
    for (dtype, expected) in [
        ("bool", json!([["bool", true], ["bool", false]])),
        (
            "int64",
            json!([
                ["int", "-9223372036854775808"],
                ["int", "9223372036854775807"]
            ]),
        ),
        (
            "uint64",
            json!([["int", "0"], ["int", "18446744073709551615"]]),
        ),
    ] {
        let case = chains
            .iter()
            .find(|c| {
                c["left"] == json!(["bool", "finite"])
                    && c["right"] == json!(["float32", "finite"])
                    && c["third"] == json!([dtype, "extreme"])
            })
            .unwrap();
        assert_eq!(case["intermediate"]["dtypes"], json!(["object"]));
        assert_eq!(case["output"]["dtypes"], json!(["object"]));
        let values = case["output"]["values"][0].as_array().unwrap();
        assert_eq!(json!(&values[4..]), expected);
        assert_eq!(values[0], json!(["float", "0x0.0p+0"]));
        assert_eq!(case["warnings"], json!([]));
    }
    for fragmented in [false, true] {
        let case = blocks
            .iter()
            .find(|c| {
                c["dtype"] == "float32"
                    && c["bool_first"] == false
                    && c["left_fragmented"] == fragmented
                    && c["right_fragmented"] == false
                    && c["reverse"] == false
                    && c["all_na"] == false
            })
            .unwrap();
        assert_eq!(
            case["output"]["dtypes"],
            if fragmented {
                json!(["object", "float32"])
            } else {
                json!(["float32", "float32"])
            }
        );
    }
    let reverts = chains
        .iter()
        .filter(|c| {
            c["intermediate"]["dtypes"] == json!(["object"])
                && c["output"]["dtypes"] != json!(["object"])
        })
        .collect::<Vec<_>>();
    assert_eq!(reverts.len(), 72);
    for case in reverts {
        assert_eq!(case["warnings"].as_array().unwrap().len(), 1);
        assert_eq!(case["warnings"][0][0], "FutureWarning");
    }
}
