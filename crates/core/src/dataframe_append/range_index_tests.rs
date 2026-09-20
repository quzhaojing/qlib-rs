use super::*;
use arrow_array::{Float64Array, Int64Array};
use num_bigint::BigInt;
use serde_json::{Value, json};
use std::{collections::HashMap, path::PathBuf, process::Command};

fn contract(file: &str) -> Value {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures").join(file))
        .arg(root.join("../../../qlib/qlib/rl/order_execution/utils.py"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

fn metadata(bounds: &Value) -> IndexMetadata {
    let integer = |i| -> BigInt {
        let v = &bounds[i];
        v.as_str()
            .map_or_else(|| v.to_string(), str::to_owned)
            .parse()
            .unwrap()
    };
    IndexMetadata::Range {
        start: integer(0),
        stop: integer(1),
        step: integer(2),
    }
}

fn name(value: &Value) -> Option<String> {
    if value[0] == "none" {
        None
    } else {
        Some(
            value[1]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| char::from_u32(u32::try_from(c.as_u64().unwrap()).unwrap()).unwrap())
                .collect(),
        )
    }
}

fn ordinal(rows: usize) -> ArrayRef {
    if rows == 0 {
        Arc::new(Float64Array::from(Vec::<f64>::new()))
    } else {
        Arc::new(Int64Array::from_iter_values(
            (0..rows).map(|i| i64::try_from(i).unwrap()),
        ))
    }
}

fn frame(
    index: ArrayRef,
    name: Option<String>,
    columns: bool,
    meta: IndexMetadata,
) -> IndexedFrame {
    let data = block_tests::batch(
        if columns {
            vec![("x", ordinal(index.len()))]
        } else {
            vec![]
        },
        index.len(),
    );
    IndexedFrame::new(index, name, data)
        .unwrap()
        .with_index_metadata(meta)
        .unwrap()
}

fn check(left: &IndexedFrame, right: &ArrayRef, case: &Value) {
    let mut columns = vec![("datetime", right.clone())];
    if case["right_columns"] == true {
        columns.push(("x", ordinal(right.len())));
    }
    let right = block_tests::batch(columns, right.len());
    let before = (left.index().to_data(), left.index_metadata().clone());
    let mut warnings = Vec::new();
    let result = dataframe_append_with_warnings(left, &right, &mut |m| {
        warnings.push(json!(["FutureWarning", m]));
    })
    .unwrap();
    let expected = &case["output"];
    let index = &expected["index"];
    let meta = if index["kind"] == "RangeIndex" {
        metadata(&index["range"])
    } else {
        IndexMetadata::Array
    };
    assert_eq!(result.index_metadata(), &meta, "{case}");
    assert_eq!(
        result.index_name(),
        name(&index["names"][0]).as_deref(),
        "{case}"
    );
    let dtype = index["dtype"].as_str().unwrap();
    if dtype == "object" {
        assert!(blocks::logical_object(result.index().data_type()));
        temporal_objects::tests::compare_cells(
            &temporal_cast::values(result.index()).unwrap(),
            index["values"].as_array().unwrap(),
        );
    } else {
        let wanted = temporal_append_tests::column(dtype, &index["values"], 0);
        assert_eq!(result.index().data_type(), wanted.data_type(), "{case}");
        if temporal_cast::is_temporal(wanted.data_type()) {
            assert_eq!(result.index().to_data(), wanted.to_data(), "{case}");
        } else {
            block_tests::compare_values(result.index(), &index["values"]);
        }
    }
    let data = &expected["frame"];
    assert_eq!(
        result.data().num_columns(),
        data["columns"].as_array().unwrap().len()
    );
    assert_eq!(
        result.data().num_rows(),
        index["values"].as_array().unwrap().len()
    );
    for (i, array) in result.data().columns().iter().enumerate() {
        assert_eq!(result.data().schema().field(i).name(), "x");
        let wanted = temporal_append_tests::column(
            data["dtypes"][i].as_str().unwrap(),
            &data["values"][i],
            0,
        );
        assert_eq!(array.data_type(), wanted.data_type(), "{case}");
        block_tests::compare_values(array, &data["values"][i]);
    }
    assert_eq!(json!(warnings), case["warnings"], "{case}");
    assert_eq!(
        before,
        (left.index().to_data(), left.index_metadata().clone())
    );
    if result.data().num_rows() > 0 || result.data().num_columns() > 0 {
        let empty = Arc::new(Int64Array::from(Vec::<i64>::new())) as ArrayRef;
        let ignored = block_tests::batch(vec![("datetime", empty)], 0);
        let repeated = dataframe_append(&result, &ignored).unwrap();
        assert_eq!(repeated.index_metadata(), result.index_metadata());
        assert_eq!(repeated.index().to_data(), result.index().to_data());
        assert_eq!(repeated.index_name(), result.index_name());
    }
}

#[test]
fn range_identity_and_empty_manager_transitions_match_source() {
    let c = contract("dataframe_index_families.py");
    assert_eq!(
        c["digest"],
        "8b371ec25bd85b0ad03573f1e346679c903617527c7163cec32c871f485d36e4"
    );
    let inputs = c["inputs"]
        .as_object()
        .unwrap()
        .iter()
        .filter(|(n, _)| {
            n.starts_with("range_")
                || n.starts_with("int_")
                || *n == "object_empty"
                || n.starts_with("datetime_")
                || n.starts_with("timedelta_")
        })
        .map(|(n, s)| {
            (
                n.clone(),
                temporal_append_tests::column(s["dtype"].as_str().unwrap(), &s["values"], 0),
            )
        })
        .collect::<HashMap<_, _>>();
    assert_eq!(inputs.len(), 16);
    let mut checked = 0;
    for case in c["pairs"].as_array().unwrap() {
        let ln = case["left"].as_str().unwrap();
        let rn = case["right"].as_str().unwrap();
        if let (Some(left), Some(right)) = (inputs.get(ln), inputs.get(rn)) {
            let input = &c["inputs"][ln];
            let meta = if input["kind"] == "RangeIndex" {
                metadata(&input["range"])
            } else {
                IndexMetadata::Array
            };
            let frame = frame(
                left.clone(),
                name(&input["names"][0]),
                case["left_columns"] == true,
                meta,
            );
            check(&frame, right, case);
            checked += 1;
        }
    }
    assert_eq!(checked, 1024);
}

#[test]
fn wide_range_bounds_and_invalid_metadata_are_explicit() {
    let c = contract("dataframe_range_boundaries.py");
    assert_eq!(
        c["digest"],
        "1f3172f530b7e365953d0bc15937395fac8488b6b8980348e3522516416dae97"
    );
    assert_eq!(c["cases"].as_array().unwrap().len(), 96);
    for case in c["cases"].as_array().unwrap() {
        let input = c["constructors"]
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["bounds"] == case["bounds"])
            .unwrap();
        let values = Arc::new(Int64Array::from_iter_values(
            input["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().parse::<i64>().unwrap()),
        )) as ArrayRef;
        let left = frame(
            values,
            Some("history".into()),
            case["left_columns"] == true,
            metadata(&case["bounds"]),
        );
        let right = if case["right"] == "int_empty" {
            Arc::new(Int64Array::from(Vec::<i64>::new())) as ArrayRef
        } else {
            ordinal(2)
        };
        check(&left, &right, case);
    }
    let base = frame(ordinal(2), None, true, IndexMetadata::default());
    for (bounds, error) in [
        (json!([0, 2, 0]), "range step must not be zero"),
        (json!([0, 3, 1]), "range bounds do not match index length"),
        (json!([1, 3, 1]), "range values do not match bounds"),
        (
            json!(["0", "1208925819614629174706176", "1"]),
            "range bounds do not match index length",
        ),
    ] {
        assert_eq!(
            base.clone()
                .with_index_metadata(metadata(&bounds))
                .unwrap_err()
                .to_string(),
            format!("invalid index metadata: {error}")
        );
    }
    for (index, error) in [
        (
            Arc::new(Float64Array::from(vec![0., 1.])) as ArrayRef,
            "range requires Int64 storage",
        ),
        (
            Arc::new(Int64Array::from(vec![Some(0), None])) as ArrayRef,
            "range values do not match bounds",
        ),
    ] {
        let frame =
            IndexedFrame::new(index, None, block_tests::batch(vec![("x", ordinal(2))], 2)).unwrap();
        assert!(
            frame
                .with_index_metadata(metadata(&json!([0, 2, 1])))
                .unwrap_err()
                .to_string()
                .ends_with(error)
        );
    }
    assert_eq!(base.index_metadata(), &IndexMetadata::Array);
}
