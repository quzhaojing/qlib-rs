use super::*;
use crate::dataframe_append::{
    self as append, IndexedFrame,
    inference::tests::{atom, check},
};
use serde_json::{Value, json};
use std::{path::PathBuf, process::Command};

fn check_batch(batch: &RecordBatch, snapshot: &Value) {
    assert_eq!(
        batch.num_rows(),
        snapshot["index"].as_array().unwrap().len()
    );
    assert_eq!(
        batch.num_columns(),
        snapshot["columns"].as_array().unwrap().len()
    );
    for (i, array) in batch.columns().iter().enumerate() {
        check(
            array,
            &json!({"dtype": snapshot["dtypes"][i], "values": snapshot["values"][i]}),
        );
        assert_eq!(
            json!([
                "str",
                batch
                    .schema_ref()
                    .field(i)
                    .name()
                    .chars()
                    .map(u32::from)
                    .collect::<Vec<_>>()
            ]),
            snapshot["columns"][i]
        );
    }
}

#[test]
fn record_assembly_matches_actual_source_including_missing_fields() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/dataframe_builtin_records.py"))
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
        "93132e49fa28866acf5de042e6640a9807b7f1193591b0df0bf35b4e3179ce74"
    );
    let cases = contract["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 1106);
    let empty = IndexedFrame::new(
        Arc::new(Int64Array::from(Vec::<i64>::new())),
        None,
        frame_from_columns(&IndexMap::new()).unwrap(),
    )
    .unwrap();
    for case in cases {
        let records = case["records"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|pair| (pair[0].as_str().unwrap().to_owned(), atom(&pair[1])))
                    .collect::<IndexMap<_, _>>()
            })
            .collect::<Vec<_>>();
        let before = format!("{records:?}");
        let built = frame_from_builtin_records(&records).unwrap();
        check_batch(&built, &case["construction"]);
        let mut warnings = vec![];
        let result = append::dataframe_append_with_warnings(&empty, &built, &mut |message| {
            warnings.push(json!(["FutureWarning", message]));
        });
        if let Some(expected) = case.get("output") {
            let result = result.unwrap();
            check_batch(result.data(), expected);
            check(
                result.index(),
                &json!({"dtype": expected["index_dtype"], "values": expected["index"]}),
            );
            assert_eq!(result.index_name(), Some("datetime"));
            let groups = result
                .blocks()
                .iter()
                .map(|group| json!([expected["dtypes"][group[0]], group]))
                .collect::<Vec<_>>();
            assert_eq!(json!(groups), expected["blocks"]);
        } else {
            assert_eq!(case["error"], "KeyError");
            assert!(matches!(
                result,
                Err(append::DataframeAppendError::MissingDatetime)
            ));
        }
        assert_eq!(json!(warnings), case["warnings"]);
        assert_eq!(format!("{records:?}"), before);
    }
}
