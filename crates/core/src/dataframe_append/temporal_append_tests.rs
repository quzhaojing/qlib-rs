use super::*;
use arrow_array::{Array, Int64Array};
use arrow_schema::TimeUnit;
use indexmap::IndexMap;
use serde_json::{Value, json};
use std::{collections::HashMap, path::PathBuf, process::Command};

pub(super) fn column(dtype: &str, values: &Value, mode: usize) -> ArrayRef {
    let items = values.as_array().unwrap();
    if dtype == "object" {
        if mode == 1
            || items
                .iter()
                .any(|v| matches!(v[0].as_str().unwrap(), "Timestamp" | "Timedelta"))
        {
            return temporal_frame_array(
                &items
                    .iter()
                    .map(temporal_objects::tests::decode)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        }
        return missing_append_tests::source_column(dtype, values, mode == 2);
    }
    if dtype.starts_with("datetime64[") || dtype.starts_with("timedelta64[") {
        let inside = dtype.split('[').nth(1).unwrap().strip_suffix(']').unwrap();
        let mut parts = inside.split(", ");
        let unit = match parts.next().unwrap() {
            "s" => TimeUnit::Second,
            "ms" => TimeUnit::Millisecond,
            "us" => TimeUnit::Microsecond,
            "ns" => TimeUnit::Nanosecond,
            _ => panic!("unit"),
        };
        let kind = if dtype.starts_with("datetime") {
            DataType::Timestamp(unit, parts.next().map(Into::into))
        } else {
            DataType::Duration(unit)
        };
        let ticks = Arc::new(Int64Array::from(
            items
                .iter()
                .map(|v| {
                    if v[0] == "NaT" {
                        None
                    } else {
                        Some(v[2].as_str().unwrap().parse::<i64>().unwrap())
                    }
                })
                .collect::<Vec<_>>(),
        )) as ArrayRef;
        return arrow_cast::cast(&ticks, &kind).unwrap();
    }
    missing_append_tests::source_column(dtype, values, false)
}

fn text(value: &Value) -> String {
    value[1]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| char::from_u32(u32::try_from(v.as_u64().unwrap()).unwrap()).unwrap())
        .collect()
}

fn batch(snapshot: &Value, mode: usize) -> RecordBatch {
    let columns = snapshot["columns"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, name)| {
            (
                text(name),
                FrameColumnInput::Array(column(
                    snapshot["dtypes"][i].as_str().unwrap(),
                    &snapshot["values"][i],
                    mode,
                )),
            )
        })
        .collect::<IndexMap<_, _>>();
    frame_from_columns(&columns).unwrap()
}

fn index(rows: usize) -> ArrayRef {
    Arc::new(Int64Array::from_iter_values(
        (0..rows).map(|i| i64::try_from(i).unwrap()),
    ))
}

fn left(snapshot: &Value, mode: usize) -> IndexedFrame {
    let data = batch(snapshot, mode);
    let groups = snapshot["blocks"]
        .as_array()
        .unwrap()
        .iter()
        .map(|g| {
            g[1].as_array()
                .unwrap()
                .iter()
                .map(|i| usize::try_from(i.as_u64().unwrap()).unwrap())
                .collect()
        })
        .collect();
    IndexedFrame::new(index(data.num_rows()), None, data)
        .unwrap()
        .with_blocks(groups)
        .unwrap()
}

fn check(frame: &IndexedFrame, warnings: &[Value], case: &Value) {
    let expected = &case["output"];
    assert_eq!(
        frame.data().num_rows(),
        expected["index"].as_array().unwrap().len()
    );
    assert_eq!(
        frame.data().num_columns(),
        expected["columns"].as_array().unwrap().len()
    );
    for (i, array) in frame.data().columns().iter().enumerate() {
        assert_eq!(
            frame.data().schema_ref().field(i).name(),
            &text(&expected["columns"][i])
        );
        let dtype = expected["dtypes"][i].as_str().unwrap();
        let values = &expected["values"][i];
        if dtype == "object" && array.data_type() == &temporal_frame_dtype() {
            temporal_objects::tests::compare_cells(
                &temporal_frame_values(array).unwrap(),
                values.as_array().unwrap(),
            );
        } else if dtype.starts_with("datetime") || dtype.starts_with("timedelta") {
            let wanted = column(dtype, values, 0);
            assert_eq!(array.data_type(), wanted.data_type(), "{case}");
            assert_eq!(array.to_data(), wanted.to_data());
        } else {
            if dtype == "object" {
                assert!(blocks::logical_object(array.data_type()), "{case}");
            } else {
                assert_eq!(
                    array.data_type(),
                    column(dtype, values, 0).data_type(),
                    "{case}"
                );
            }
            block_tests::compare_values(array, values);
        }
    }
    block_tests::compare_values(frame.index(), &expected["index"]);
    let name = if expected["index_name"][0] == "none" {
        None
    } else {
        Some(text(&expected["index_name"]))
    };
    assert_eq!(frame.index_name(), name.as_deref());
    let blocks = frame
        .blocks()
        .iter()
        .map(|g| json!([expected["dtypes"][g[0]], g]))
        .collect::<Vec<_>>();
    assert_eq!(json!(blocks), expected["blocks"]);
    assert_eq!(json!(warnings), case["warnings"]);
}

#[test]
fn temporal_objects_native_arrays_and_alignment_match_actual_source() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/dataframe_temporal_append.py"))
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
        "d3d4102c727cb5583ba58779f52aded8f0a376a2ef0bbd6bb4626aafc7727930"
    );
    assert_eq!(contract["inputs"].as_object().unwrap().len(), 83);
    for mode in 0..3 {
        let inputs = contract["inputs"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(name, snapshot)| {
                (
                    name.clone(),
                    column(
                        snapshot["dtypes"][0].as_str().unwrap(),
                        &snapshot["values"][0],
                        mode,
                    ),
                )
            })
            .collect::<HashMap<_, _>>();
        for kind in ["pairs", "alignment"] {
            let cases = contract[kind].as_array().unwrap();
            assert_eq!(cases.len(), 6889);
            for case in cases {
                let a = &inputs[case["left"].as_str().unwrap()];
                let b = &inputs[case["right"].as_str().unwrap()];
                let frame = IndexedFrame::new(
                    index(a.len()),
                    None,
                    block_tests::batch(vec![("x", a.clone())], a.len()),
                )
                .unwrap();
                let right = block_tests::batch(
                    vec![
                        ("datetime", index(b.len())),
                        (if kind == "pairs" { "x" } else { "y" }, b.clone()),
                    ],
                    b.len(),
                );
                let mut warnings = vec![];
                let result = dataframe_append_with_warnings(&frame, &right, &mut |message| {
                    warnings.push(json!(["FutureWarning", message]));
                });
                assert!(
                    result.is_ok(),
                    "mode={mode} {kind} {} / {}: {result:?}",
                    case["left"],
                    case["right"]
                );
                check(&result.unwrap(), &warnings, case);
            }
        }
        let cases = contract["blocks"].as_array().unwrap();
        assert_eq!(cases.len(), 72);
        for case in cases {
            let frame = left(&case["left_input"], mode);
            let right = batch(&case["right_input"], mode);
            let mut warnings = vec![];
            let result = dataframe_append_with_warnings(&frame, &right, &mut |message| {
                warnings.push(json!(["FutureWarning", message]));
            })
            .unwrap();
            check(&result, &warnings, case);
        }
    }
}
