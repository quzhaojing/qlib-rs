//! Source characterization for categorical concatenation; not native parity.
use serde_json::Value;
use std::{path::PathBuf, process::Command};

#[test]
fn actual_categorical_append_contract_is_frozen() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/dataframe_categorical_append.py"))
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
        "3664562ff16b25ab4991959ee4058f9c21c74ea721984fb35ec3e6f7286bf72d"
    );
    assert_eq!(contract["inputs"].as_object().unwrap().len(), 74);
    let pairs = contract["pairs"].as_array().unwrap();
    assert_eq!(pairs.len(), 21904);
    let mut counts = [0; 5];
    for case in pairs {
        let position = match case["output"]["index"]["kind"].as_str() {
            Some("CategoricalIndex") => 0,
            Some("Index") => 1,
            Some("DatetimeIndex") => 2,
            Some("TimedeltaIndex") => 3,
            None => {
                assert_eq!(case["error"], "TypeError");
                assert_eq!(case["message"], "dtype of categories must be the same");
                4
            }
            other => panic!("unexpected index family: {other:?}"),
        };
        counts[position] += 1;
    }
    assert_eq!(counts, [8032, 12960, 584, 292, 36]);
}
