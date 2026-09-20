//! Source characterization only; native typed-record inference is still pending.
use serde_json::{Value, json};
use std::{path::PathBuf, process::Command};

#[test]
fn records_reinfer_typed_scalars_and_preserve_discovery_order() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/dataframe_records_contract.py"))
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
    assert_eq!(contract["samples"], 44);
    assert_eq!(
        contract["digest"],
        "ea9d83e7e17d1d3ed5400f48ed58a1ca353475ce01474a34b7e6721c49204ae1"
    );
    let cases = contract["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 1584);
    assert!(cases.iter().all(|c| c.get("output").is_some()));
    assert!(
        cases
            .iter()
            .all(|c| c["construction_warnings"] == json!([]))
    );
    let find = |name: &str, missing: &str, order: &str| {
        &cases
            .iter()
            .find(|c| {
                c["name"] == name
                    && c["missing"] == missing
                    && c["order"] == order
                    && c["left"] == "empty"
            })
            .unwrap()["construction"]
    };
    for (name, dtype) in [("numpy_int8", "int8"), ("numpy_float32", "float32")] {
        assert_eq!(find(name, "single", "forward")["dtypes"][0], dtype);
        for missing in ["omitted", "none", "nan"] {
            assert_eq!(find(name, missing, "forward")["dtypes"][0], "float64");
        }
        for missing in ["nat", "na"] {
            assert_eq!(find(name, missing, "forward")["dtypes"][0], "object");
        }
    }
    for (name, dtype) in [
        ("timestamp_s_None", "datetime64[ns]"),
        ("timestamp_s_UTC", "datetime64[ns, UTC]"),
        ("timestamp_s_Asia/Shanghai", "datetime64[ns, Asia/Shanghai]"),
        ("duration_s", "timedelta64[ns]"),
    ] {
        for missing in ["single", "omitted", "none", "nan", "nat"] {
            assert_eq!(find(name, missing, "forward")["dtypes"][0], dtype);
        }
        assert_eq!(find(name, "na", "forward")["dtypes"][0], "object");
    }
    for name in ["timestamp_outside_ns", "duration_outside_ns"] {
        assert_eq!(find(name, "single", "forward")["dtypes"][0], "object");
        assert_eq!(
            find(name, "single", "forward")["values"][0][0][1],
            if name == "timestamp_outside_ns" {
                "datetime64[s]"
            } else {
                "timedelta64[s]"
            }
        );
    }
    let forward = find("bool", "omitted", "forward");
    let reverse = find("bool", "omitted", "reverse");
    let text = |s: &str| json!(["str", s.chars().map(u32::from).collect::<Vec<_>>()]);
    assert_eq!(
        forward["columns"],
        json!([text("x"), text("datetime"), text("stock_id")])
    );
    assert_eq!(
        reverse["columns"],
        json!([text("stock_id"), text("datetime"), text("x")])
    );
    assert_eq!(
        forward["values"][0],
        json!([["bool", true], ["float", "nan"]])
    );
    assert_eq!(
        reverse["values"][2],
        json!([["float", "nan"], ["bool", true]])
    );
    assert_eq!(
        find("bool", "none", "forward")["values"][0],
        json!([["bool", true], ["none"]])
    );
}
