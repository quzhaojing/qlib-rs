//! Source behavior needed by the tuple/MultiIndex native representation.
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::PathBuf, process::Command};

fn verify_index(index: &Value) {
    let rows = index["values"].as_array().unwrap().len();
    let depth = usize::try_from(index["nlevels"].as_u64().unwrap()).unwrap();
    assert_eq!(index["names"].as_array().unwrap().len(), depth);
    if index["kind"] != "MultiIndex" {
        assert_eq!(depth, 1);
        return;
    }
    let levels = index["levels"].as_array().unwrap();
    let codes = index["codes"].as_array().unwrap();
    assert_eq!(levels.len(), depth);
    assert_eq!(codes.len(), depth);
    for (level, codes) in levels.iter().zip(codes) {
        verify_index(level);
        let level_len = i64::try_from(level["values"].as_array().unwrap().len()).unwrap();
        let codes = codes.as_array().unwrap();
        assert_eq!(codes.len(), rows);
        for code in codes {
            assert!((-1..level_len).contains(&code.as_i64().unwrap()));
        }
    }
    for value in index["values"].as_array().unwrap() {
        assert_eq!(value[0], "tuple");
        assert_eq!(value[1].as_array().unwrap().len(), depth);
    }
}

#[test]
fn source_multi_index_factoring_fallback_and_metadata_validation() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/dataframe_multi_index_contract.py"))
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
        "6cc7f3f1e47d0f6e40f1ecbc2a7b6ba52d5fa7ede33bb23e665e440ff9ec7ecd"
    );
    let inputs = contract["inputs"].as_object().unwrap();
    assert_eq!(inputs.len(), 28);
    for index in inputs.values() {
        verify_index(index);
    }
    let pairs = contract["pairs"].as_array().unwrap();
    assert_eq!(pairs.len(), 3136);
    let mut families = BTreeMap::new();
    for case in pairs {
        assert!(case.get("error").is_none(), "{case}");
        let index = &case["output"]["index"];
        verify_index(index);
        assert_eq!(index["values"], case["output"]["frame"]["index"]);
        let row_count = |side: &str| {
            inputs[case[side].as_str().unwrap()]["values"]
                .as_array()
                .unwrap()
                .len()
        };
        assert_eq!(
            index["values"].as_array().unwrap().len(),
            row_count("left") + row_count("right")
        );
        *families
            .entry(index["kind"].as_str().unwrap())
            .or_insert(0_usize) += 1;
    }
    assert_eq!(
        families,
        BTreeMap::from([("Index", 2008), ("MultiIndex", 1128)])
    );
    verify_directional_rules(&contract);
    verify_constructors_and_chains(&contract);
}

fn verify_directional_rules(contract: &Value) {
    let inputs = contract["inputs"].as_object().unwrap();
    let pairs = contract["pairs"].as_array().unwrap();
    let find = |left: &str, right: &str, right_columns: bool| {
        &pairs
            .iter()
            .find(|c| {
                c["left"] == left
                    && c["right"] == right
                    && c["left_columns"] == true
                    && c["right_columns"] == right_columns
            })
            .unwrap()["output"]["index"]
    };
    // Longer tuples truncate; shorter/ragged tuples cause lossless object fallback.
    let truncated = find("multi_two", "tuple_three", true);
    assert_eq!(truncated["kind"], "MultiIndex");
    assert_eq!(truncated["codes"], json!([[1, 0, 1, 0], [1, 0, 1, 0]]));
    assert_eq!(truncated["names"], json!([["none"], ["none"]]));
    for (left, right) in [
        ("multi_three", "tuple_one"),
        ("multi_two", "tuple_ragged"),
        ("multi_two", "scalar_text"),
        ("tuple_two", "multi_two"),
    ] {
        let result = find(left, right, true);
        assert_eq!(result["kind"], "Index");
        let mut expected = inputs[left]["values"].as_array().unwrap().clone();
        expected.extend(inputs[right]["values"].as_array().unwrap().iter().cloned());
        assert_eq!(result["values"], json!(expected));
    }
    let missing = find("multi_missing", "tuple_two", true);
    assert_eq!(
        missing["codes"],
        json!([[1, -1, 0, 1, 0], [-1, 0, 1, 1, 0]])
    );
    assert_eq!(missing["levels"][1]["dtype"], "float64");
    assert_eq!(missing["levels"][0]["values"].as_array().unwrap().len(), 2);
    assert_eq!(
        find("multi_category", "tuple_two", true)["levels"][0]["kind"],
        "Index"
    );
    for name in [
        "multi_tuple_names",
        "multi_category",
        "multi_nullable",
        "multi_empty",
    ] {
        assert_eq!(find(name, "object_empty", false), &inputs[name]);
    }
    // Even an ignored 0x0 manager causes MultiIndex.append([]) to rebuild the
    // descriptor and discard sortorder. RangeIndex's identity rule is not generic.
    let mut unsorted = inputs["multi_sorted"].clone();
    unsorted["sortorder"] = Value::Null;
    assert_eq!(find("multi_sorted", "object_empty", false), &unsorted);
    assert_eq!(
        find("multi_sorted", "object_empty", true)["sortorder"],
        Value::Null
    );
    assert_eq!(find("multi_empty", "object_empty", true)["kind"], "Index");
}

fn verify_constructors_and_chains(contract: &Value) {
    let constructors = contract["constructors"].as_array().unwrap();
    assert_eq!(constructors.len(), 14);
    assert_eq!(
        constructors
            .iter()
            .filter(|c| c.get("error").is_some())
            .count(),
        10
    );
    for case in constructors {
        assert_eq!(case["warnings"], json!([]));
        if case.get("output").is_some() {
            verify_index(&case["output"]);
        } else {
            assert!(case["message"].as_str().is_some_and(|m| !m.is_empty()));
        }
    }
    let constructor = |name: &str| constructors.iter().find(|c| c["name"] == name).unwrap();
    assert_eq!(constructor("negative_sortorder")["output"]["sortorder"], -1);
    assert_eq!(
        constructor("float_codes")["output"]["codes"],
        json!([[0, 1]])
    );
    assert_eq!(
        constructor("missing_in_levels")["output"]["codes"],
        json!([[-1, 1]])
    );
    assert_eq!(constructor("unhashable_name")["error"], "TypeError");
    let chains = contract["chains"].as_array().unwrap();
    assert_eq!(chains.len(), 112);
    for case in chains {
        verify_index(&case["initial"]);
        for step in ["first_output", "second_output"] {
            assert!(case[step].get("error").is_none(), "{case}");
            verify_index(&case[step]["output"]["index"]);
        }
    }
}
