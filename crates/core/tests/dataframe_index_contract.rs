//! Actual-source characterization; general index migration is not yet accepted.
use serde_json::{Value, json};
use std::{path::PathBuf, process::Command};

#[test]
fn initialized_and_chained_index_contract_is_frozen_from_actual_source() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/dataframe_index_contract.py"))
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
        "5c6261d2f19cf03db471565079a6e85ba7d84ea77eca7b757f775d128ea79a26"
    );
    let inputs = contract["inputs"].as_object().unwrap();
    assert_eq!(inputs.len(), 83);
    for name in ["float16_empty", "float16_finite", "float16_na"] {
        assert_eq!(
            inputs[name],
            json!({"error":"NotImplementedError", "message":"float16 indexes are not supported"})
        );
    }
    assert_eq!(
        inputs.values().filter(|v| v.get("error").is_some()).count(),
        3
    );
    let pairs = contract["pairs"].as_array().unwrap();
    assert_eq!(pairs.len(), 19_920);
    assert_eq!(
        pairs.iter().filter(|c| c.get("error").is_some()).count(),
        720
    );
    assert_eq!(
        pairs
            .iter()
            .filter(|c| !c["warnings"].as_array().unwrap().is_empty())
            .count(),
        5_802
    );
    for case in pairs {
        if case.get("error").is_some() {
            assert!(case["right"].as_str().unwrap().starts_with("float16_"));
            assert_eq!(case["error"], "NotImplementedError");
            assert_eq!(case["message"], "float16 indexes are not supported");
        } else {
            let name = if case["name"] == "datetime" {
                json!(["str", "datetime".chars().map(u32::from).collect::<Vec<_>>()])
            } else {
                json!(["none"])
            };
            assert_eq!(case["output"]["name"], name);
        }
    }
    let chains = contract["chains"].as_array().unwrap();
    assert_eq!(chains.len(), 581);
    let chain = chains
        .iter()
        .find(|c| {
            c["first"] == "timestamp_ns_UTC_finite" && c["second"] == "timestamp_s_None_finite"
        })
        .unwrap();
    assert_eq!(chain["initial"]["dtype"], "object");
    assert_eq!(chain["initial"]["values"], json!([]));
    let first = &chain["first_output"];
    assert_eq!(first["output"]["kind"], "DatetimeIndex");
    assert_eq!(first["output"]["dtype"], "datetime64[ns, UTC]");
    assert_eq!(first["warnings"].as_array().unwrap().len(), 1);
    assert_eq!(first["warnings"][0][0], "FutureWarning");
    let second = &chain["second_output"];
    assert_eq!(second["output"]["kind"], "Index");
    assert_eq!(second["output"]["dtype"], "object");
    assert_eq!(second["warnings"], json!([]));
    assert_eq!(
        second["output"]["values"],
        json!([
            ["Timestamp", "datetime64[ns]", "1704153600000000000", "UTC"],
            ["NaT"],
            ["Timestamp", "datetime64[s]", "1704153600", "None"],
            ["NaT"]
        ])
    );
}
