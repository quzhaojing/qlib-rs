use super::*;
use arrow_array::{Float64Array, Int64Array, new_null_array};
use serde_json::{Value, json};
use std::{collections::HashMap, path::PathBuf, process::Command};
use tuple_objects::tests::{cell, compare};

pub(super) fn array(index: &Value, legacy_scalars: bool) -> ArrayRef {
    if index["dtype"] != "object" {
        return temporal_append_tests::column(
            index["dtype"].as_str().unwrap(),
            &index["values"],
            0,
        );
    }
    let values = index["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(cell)
        .collect::<Vec<_>>();
    if legacy_scalars
        && values
            .iter()
            .all(|v| matches!(v, TupleFrameValue::Scalar(_)))
    {
        return temporal_frame_array(
            &index["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(temporal_objects::tests::decode)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    }
    tuple_frame_array(&values).unwrap()
}

pub(super) fn ordinal(rows: usize) -> ArrayRef {
    if rows == 0 {
        Arc::new(Float64Array::from(Vec::<f64>::new()))
    } else {
        Arc::new(Int64Array::from_iter_values(
            (0..rows).map(|i| i64::try_from(i).unwrap()),
        ))
    }
}

pub(super) fn append(
    left: &IndexedFrame,
    right: &ArrayRef,
    right_columns: bool,
    case: &Value,
) -> IndexedFrame {
    let mut columns = vec![("datetime", right.clone())];
    if right_columns {
        columns.push(("x", ordinal(right.len())));
    }
    let other = block_tests::batch(columns, right.len());
    append_batch(left, &other, case)
}

pub(super) fn append_batch(left: &IndexedFrame, other: &RecordBatch, case: &Value) -> IndexedFrame {
    let before = (left.index().to_data(), other.column(0).to_data());
    let mut warnings = vec![];
    let result = dataframe_append_with_warnings(left, other, &mut |m| {
        warnings.push(json!(["FutureWarning", m]));
    })
    .unwrap_or_else(|e| panic!("{case}: {e}"));
    assert_eq!(json!(warnings), case["warnings"], "{case}");
    let expected = &case["output"]["index"];
    if expected["kind"] == "CategoricalIndex" {
        assert!(
            matches!(result.index_metadata(), IndexMetadata::Categorical(_)),
            "{case}"
        );
        categorical_append_tests::check(&result, &case["output"]);
        assert_eq!(left.index().to_data(), before.0);
        assert_eq!(other.column(0).to_data(), before.1);
        return result;
    }
    if expected.get("string").is_some() {
        super::string_append::tests::compare(&result, expected);
    } else if expected.get("masked").is_some() {
        super::nullable_append::tests::compare(&result, expected);
    } else if expected["dtype"] == "object" {
        assert!(
            blocks::logical_object(result.index().data_type())
                || result.index().data_type() == &tuple_frame_dtype(),
            "{case}"
        );
        compare(
            &tuple_index::values(result.index()).unwrap(),
            expected["values"].as_array().unwrap(),
        );
    } else {
        let wanted = array(expected, false);
        assert_eq!(result.index().data_type(), wanted.data_type(), "{case}");
        if temporal_cast::is_temporal(wanted.data_type()) {
            assert_eq!(result.index().to_data(), wanted.to_data(), "{case}");
        } else {
            block_tests::compare_values(result.index(), &expected["values"]);
        }
    }
    let name = if expected["kind"] == "MultiIndex" || expected["names"][0][0] == "none" {
        None
    } else {
        Some("datetime")
    };
    assert_eq!(result.index_name(), name, "{case}");
    if expected["kind"] == "MultiIndex" {
        multi_append_tests::compare_metadata(result.index_metadata(), expected);
    } else if expected.get("masked").is_none() && expected.get("string").is_none() {
        assert_eq!(result.index_metadata(), &IndexMetadata::Array);
    }
    let data = &case["output"]["frame"];
    assert_eq!(
        result.data().num_rows(),
        expected["values"].as_array().unwrap().len()
    );
    assert_eq!(
        result.data().num_columns(),
        data["columns"].as_array().unwrap().len()
    );
    for (i, column) in result.data().columns().iter().enumerate() {
        assert_eq!(result.data().schema().field(i).name(), "x");
        let wanted = temporal_append_tests::column(
            data["dtypes"][i].as_str().unwrap(),
            &data["values"][i],
            0,
        );
        assert!(
            blocks::logical_type_eq(column.data_type(), wanted.data_type()),
            "{case}"
        );
        if blocks::logical_object(column.data_type()) {
            temporal_objects::tests::compare_cells(
                &temporal_cast::values(column).unwrap(),
                data["values"][i].as_array().unwrap(),
            );
        } else {
            block_tests::compare_values(column, &data["values"][i]);
        }
    }
    assert_eq!(left.index().to_data(), before.0);
    assert_eq!(other.column(0).to_data(), before.1);
    result
}

#[test]
fn ordinary_tuple_indexes_match_source_pairs_and_continuous_append() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/dataframe_tuple_index.py"))
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
        "f7668ab2dbfc93826a2cd9d937cee2e3186b748c39b270025786033c14f378aa"
    );
    assert_eq!(contract["inputs"].as_object().unwrap().len(), 27);
    assert_eq!(contract["pairs"].as_array().unwrap().len(), 2916);
    assert_eq!(contract["chains"].as_array().unwrap().len(), 108);
    for legacy in [false, true] {
        let arrays: HashMap<_, _> = contract["inputs"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(name, value)| (name.as_str(), array(value, legacy)))
            .collect();
        for case in contract["pairs"].as_array().unwrap() {
            let index = arrays[case["left"].as_str().unwrap()].clone();
            let data = block_tests::batch(
                if case["left_columns"] == true {
                    vec![("x", ordinal(index.len()))]
                } else {
                    vec![]
                },
                index.len(),
            );
            let left = IndexedFrame::new(index, Some("datetime".into()), data).unwrap();
            append(
                &left,
                &arrays[case["right"].as_str().unwrap()],
                case["right_columns"] == true,
                case,
            );
        }
        for case in contract["chains"].as_array().unwrap() {
            let index = array(&case["initial"], legacy);
            let data = block_tests::batch(vec![("x", builtin_frame_array(&[]).unwrap())], 0);
            let left = IndexedFrame::new(index, Some("datetime".into()), data).unwrap();
            let first = append(
                &left,
                &arrays[case["first"].as_str().unwrap()],
                true,
                &case["first_output"],
            );
            append(
                &first,
                &arrays[case["second"].as_str().unwrap()],
                true,
                &case["second_output"],
            );
        }
    }
}

#[test]
fn tuple_index_invalid_inputs_are_not_hidden_by_tuple_inference() {
    let valid = tuple_frame_array(&[TupleFrameValue::Tuple(vec![])]).unwrap();
    let invalid = new_null_array(&tuple_frame_dtype(), 1);
    let joined = concat(&[valid.as_ref(), invalid.as_ref()]).unwrap();
    let before = joined.to_data();
    assert!(tuple_index::infer(&joined, &mut ArrowOperations).is_err());
    assert_eq!(joined.to_data(), before);
    let foreign: ArrayRef = Arc::new(arrow_array::BinaryArray::from(vec![b"x".as_slice()]));
    assert!(tuple_index::values(&foreign).is_err());
    for (left, right) in [
        (&invalid, &valid),
        (&valid, &invalid),
        (&foreign, &valid),
        (&valid, &foreign),
    ] {
        assert!(
            tuple_index::join(left, right, &tuple_frame_dtype(), &mut ArrowOperations).is_err()
        );
    }
}

pub(super) struct Failure {
    pub(super) at: usize,
    pub(super) calls: usize,
}

impl Failure {
    fn check(&mut self) -> Result<(), ArrowError> {
        self.calls += 1;
        if self.calls == self.at {
            Err(ArrowError::ComputeError(
                "tuple index backend failed".into(),
            ))
        } else {
            Ok(())
        }
    }
}

impl FrameOperations for Failure {
    fn batch(
        &mut self,
        _fields: Vec<Field>,
        _columns: Vec<ArrayRef>,
        _rows: usize,
    ) -> Result<RecordBatch, ArrowError> {
        panic!("failed index operations must not publish a data batch")
    }
    fn tuple_array(&mut self, values: &[TupleFrameValue]) -> Result<ArrayRef, ArrowError> {
        self.check()?;
        tuple_frame_array(values)
    }
    fn infer_index(&mut self, values: &[TemporalFrameValue]) -> Result<ArrayRef, ArrowError> {
        self.check()?;
        infer_temporal_frame_values(values)
    }
    fn cast(&mut self, array: &ArrayRef, dtype: &DataType) -> Result<ArrayRef, ArrowError> {
        ArrowOperations.cast(array, dtype)
    }
    fn concat(&mut self, left: &ArrayRef, right: &ArrayRef) -> Result<ArrayRef, ArrowError> {
        ArrowOperations.concat(left, right)
    }
}

#[test]
fn tuple_index_boxing_and_ignored_manager_inference_failures_are_atomic() {
    let object = tuple_frame_array(&[TupleFrameValue::Scalar(TemporalFrameValue::Builtin(
        BuiltinFrameValue::Int(1),
    ))])
    .unwrap();
    let before = object.to_data();
    for at in [1, 2] {
        let mut backend = Failure { at, calls: 0 };
        let error =
            tuple_index::join(&object, &object, &tuple_frame_dtype(), &mut backend).unwrap_err();
        assert_eq!(
            error.to_string(),
            "Compute error: tuple index backend failed"
        );
        assert_eq!(backend.calls, at);
        assert_eq!(object.to_data(), before);
    }
    for left_empty in [false, true] {
        let left_index = if left_empty {
            object.slice(0, 0)
        } else {
            object.clone()
        };
        let left = IndexedFrame::new(
            left_index.clone(),
            None,
            block_tests::batch(vec![], left_index.len()),
        )
        .unwrap();
        let right_index = if left_empty {
            object.clone()
        } else {
            object.slice(0, 0)
        };
        let right = block_tests::batch(vec![("datetime", right_index.clone())], right_index.len());
        let mut backend = Failure {
            at: if left_empty { 2 } else { 1 },
            calls: 0,
        };
        let mut warnings = vec![];
        let error = append_with_operations(&left, &right, &mut backend, &mut |m| {
            warnings.push(m.to_owned());
        })
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "Compute error: tuple index backend failed"
        );
        assert_eq!(backend.calls, backend.at);
        assert!(warnings.is_empty());
        assert_eq!(left.index().to_data(), left_index.to_data());
        assert_eq!(right.column(0).to_data(), right_index.to_data());
    }
}
