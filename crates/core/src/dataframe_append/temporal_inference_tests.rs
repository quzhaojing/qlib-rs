use super::*;
use crate::dataframe_append::{
    self as append, FrameColumnInput, IndexedFrame,
    temporal_objects::tests::{compare_cells, decode},
};
use arrow_array::{Array, Int64Array, RecordBatch};
use arrow_schema::DataType;
use indexmap::IndexMap;
use serde_json::{Value, json};
use std::{path::PathBuf, process::Command};

#[test]
fn non_utc_python_range_overflow_retains_the_original_object() {
    let output = Command::new("python").args(["-c", "import pandas as p,numpy as n,json; x=p.Timestamp('3000-01-01',tz='Etc/GMT-8'); a=p.Index(n.array([x],dtype=object)); print(json.dumps([str(a.dtype),str(a[0].asm8.dtype),str(a[0].asm8.view('i8')),str(a[0].tz)]))"]).output().unwrap();
    assert!(output.status.success());
    let source: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        source,
        json!(["object", "datetime64[s]", "32503651200", "Etc/GMT-8"])
    );
    let values = vec![V::Timestamp {
        ticks: 32_503_651_200,
        unit: TimeUnit::Second,
        timezone: Some("Etc/GMT-8".into()),
    }];
    let array = infer_temporal_frame_values(&values).unwrap();
    assert_eq!(array.data_type(), &append::temporal_frame_dtype());
    assert_eq!(append::temporal_frame_values(&array).unwrap(), values);
}

fn check(array: &ArrayRef, dtype: &str, values: &Value) {
    if dtype == "object" && array.data_type() == &append::temporal_frame_dtype() {
        compare_cells(
            &append::temporal_frame_values(array).unwrap(),
            values.as_array().unwrap(),
        );
    } else if dtype.starts_with("datetime64[") || dtype.starts_with("timedelta64[") {
        let expected = if dtype.starts_with("datetime64[") {
            DataType::Timestamp(
                TimeUnit::Nanosecond,
                dtype
                    .strip_prefix("datetime64[ns, ")
                    .map(|s| s.strip_suffix(']').unwrap().into()),
            )
        } else {
            DataType::Duration(TimeUnit::Nanosecond)
        };
        assert_eq!(array.data_type(), &expected);
        let ints = arrow_cast::cast(array, &DataType::Int64).unwrap();
        let ints = ints.as_any().downcast_ref::<Int64Array>().unwrap();
        for (i, value) in values.as_array().unwrap().iter().enumerate() {
            if value[0] == "NaT" {
                assert!(ints.is_null(i));
            } else {
                assert!(!ints.is_null(i));
                assert_eq!(
                    ints.value(i),
                    value[2].as_str().unwrap().parse::<i64>().unwrap()
                );
            }
        }
    } else {
        append::inference::tests::check(array, &json!({"dtype":dtype,"values":values}));
    }
}

fn check_batch(batch: &RecordBatch, expected: &Value) {
    assert_eq!(
        batch.num_rows(),
        expected["index"].as_array().unwrap().len()
    );
    assert_eq!(
        batch.num_columns(),
        expected["columns"].as_array().unwrap().len()
    );
    for (i, array) in batch.columns().iter().enumerate() {
        check(
            array,
            expected["dtypes"][i].as_str().unwrap(),
            &expected["values"][i],
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
            expected["columns"][i]
        );
    }
}

fn check_append(empty: &IndexedFrame, batch: &RecordBatch, case: &Value) {
    let mut warnings = vec![];
    let result = append::dataframe_append_with_warnings(empty, batch, &mut |message| {
        warnings.push(json!(["FutureWarning", message]));
    })
    .unwrap();
    let expected = &case["output"];
    check_batch(result.data(), expected);
    check(
        result.index(),
        expected["index_dtype"].as_str().unwrap(),
        &expected["index"],
    );
    assert_eq!(result.index_name(), Some("datetime"));
    let blocks = result
        .blocks()
        .iter()
        .map(|g| json!([expected["dtypes"][g[0]], g]))
        .collect::<Vec<_>>();
    assert_eq!(json!(blocks), expected["blocks"]);
    assert_eq!(json!(warnings), case["warnings"]);
}

#[test]
fn temporal_list_and_record_inference_match_actual_source() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/dataframe_temporal_inference.py"))
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
        "236252f667fdb5e190921c65e56373085bbaf3b2e10fa3ff41ada258ab0c55f9"
    );
    let atoms = contract["atoms"]
        .as_array()
        .unwrap()
        .iter()
        .map(decode)
        .collect::<Vec<_>>();
    let cases = contract["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 1999);
    let empty = IndexedFrame::new(
        Arc::new(Int64Array::from(Vec::<i64>::new())),
        None,
        append::frame_from_columns(&IndexMap::new()).unwrap(),
    )
    .unwrap();
    for case in cases {
        let values = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| atoms[usize::try_from(id.as_u64().unwrap()).unwrap()].clone())
            .collect::<Vec<_>>();
        let before = format!("{values:?}");
        let inferred = infer_temporal_frame_values(&values).unwrap();
        let c = &case["construction"];
        check(&inferred, c["dtypes"][1].as_str().unwrap(), &c["values"][1]);
        let dates = (0..values.len())
            .map(|i| B::Int(i64::try_from(i).unwrap()))
            .collect::<Vec<_>>();
        let columns = IndexMap::from([
            ("datetime".into(), FrameColumnInput::Untyped(dates.clone())),
            ("x".into(), FrameColumnInput::Temporal(values.clone())),
        ]);
        let batch = append::frame_from_columns(&columns).unwrap();
        check_batch(&batch, c);
        check_append(&empty, &batch, case);
        let records = dates
            .into_iter()
            .zip(values.iter())
            .map(|(date, value)| {
                IndexMap::from([
                    ("datetime".into(), V::Builtin(date)),
                    ("x".into(), value.clone()),
                ])
            })
            .collect::<Vec<_>>();
        let batch = append::frame_from_temporal_records(&records).unwrap();
        check_batch(&batch, &case["records"]);
        if !records.is_empty() {
            check_append(&empty, &batch, case);
        }
        assert_eq!(format!("{values:?}"), before);
    }
    let omitted = contract["omitted"].as_array().unwrap();
    assert_eq!(
        contract["omitted_digest"],
        "8caf83d0b2d1b804dc283a0232c6c38c49e08f164d4c8775ce3848666fb455d8"
    );
    assert_eq!(omitted.len(), 96);
    for case in omitted {
        let records = case["records"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|pair| (pair[0].as_str().unwrap().to_owned(), decode(&pair[1])))
                    .collect::<IndexMap<_, _>>()
            })
            .collect::<Vec<_>>();
        let batch = append::frame_from_temporal_records(&records).unwrap();
        check_batch(&batch, &case["construction"]);
        check_append(&empty, &batch, case);
    }
}

#[test]
fn reserved_temporal_ticks_are_errors_even_on_object_fallback() {
    let bad = V::Timestamp {
        ticks: i64::MIN,
        unit: TimeUnit::Nanosecond,
        timezone: None,
    };
    for values in [vec![bad.clone()], vec![V::Builtin(B::Bool(false)), bad]] {
        assert_eq!(
            infer_temporal_frame_values(&values)
                .unwrap_err()
                .to_string(),
            "Invalid argument error: temporal object uses reserved NaT ticks"
        );
    }
}
