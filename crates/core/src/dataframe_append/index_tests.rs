use super::*;
use arrow_array::{Float16Array, Int64Array};
use serde_json::{Value, json};
use std::{collections::HashMap, path::PathBuf, process::Command};

fn ordinal(rows: usize) -> ArrayRef {
    Arc::new(Int64Array::from_iter_values(
        (0..rows).map(|i| i64::try_from(i).unwrap()),
    ))
}

fn check(
    result: Result<IndexedFrame, DataframeAppendError>,
    warnings: &[Value],
    case: &Value,
) -> Option<IndexedFrame> {
    if case.get("error").is_some() {
        assert_eq!(
            result.unwrap_err().to_string(),
            case["message"].as_str().unwrap(),
            "{case}"
        );
        assert_eq!(json!(warnings), case["warnings"], "{case}");
        return None;
    }
    let frame = result.unwrap_or_else(|e| panic!("{case}: {e}"));
    let expected = &case["output"];
    let dtype = expected["dtype"].as_str().unwrap();
    let array = frame.index();
    if dtype == "object" {
        assert!(
            blocks::logical_object(array.data_type()),
            "{case}: {:?}",
            array.data_type()
        );
        temporal_objects::tests::compare_cells(
            &temporal_cast::values(array).unwrap(),
            expected["values"].as_array().unwrap(),
        );
    } else {
        let wanted = temporal_append_tests::column(dtype, &expected["values"], 0);
        assert_eq!(array.data_type(), wanted.data_type(), "{case}");
        if temporal_cast::is_temporal(array.data_type()) {
            assert_eq!(array.to_data(), wanted.to_data(), "{case}");
        } else {
            block_tests::compare_values(array, &expected["values"]);
        }
    }
    let name = if expected["name"][0] == "none" {
        None
    } else {
        Some("datetime")
    };
    assert_eq!(frame.index_name(), name, "{case}");
    assert_eq!(json!(warnings), case["warnings"], "{case}");
    Some(frame)
}

fn append(frame: &IndexedFrame, values: &ArrayRef, case: &Value) -> Option<IndexedFrame> {
    let right = block_tests::batch(
        vec![("datetime", values.clone()), ("x", ordinal(values.len()))],
        values.len(),
    );
    let before = (frame.index().to_data(), values.to_data());
    let mut warnings = Vec::new();
    let result = dataframe_append_with_warnings(frame, &right, &mut |m| {
        warnings.push(json!(["FutureWarning", m]));
    });
    assert_eq!(before, (frame.index().to_data(), values.to_data()));
    check(result, &warnings, case)
}

#[test]
fn native_index_pairs_and_initialized_chains_match_source() {
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
    assert_eq!(
        contract["digest"],
        "5c6261d2f19cf03db471565079a6e85ba7d84ea77eca7b757f775d128ea79a26"
    );
    for mode in 0..3 {
        let inputs = contract["inputs"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(name, s)| {
                let array = if name.starts_with("float16_") {
                    let v = match name.as_str() {
                        "float16_empty" => vec![],
                        "float16_finite" => vec![half::f16::ZERO, half::f16::ONE],
                        _ => vec![half::f16::NAN; 2],
                    };
                    Arc::new(Float16Array::from(v)) as ArrayRef
                } else {
                    temporal_append_tests::column(s["dtype"].as_str().unwrap(), &s["values"], mode)
                };
                (name.clone(), array)
            })
            .collect::<HashMap<_, _>>();
        let cases = contract["pairs"].as_array().unwrap();
        assert_eq!(cases.len(), 19_920);
        for case in cases {
            let index = &inputs[case["left"].as_str().unwrap()];
            let frame = IndexedFrame::new(
                index.clone(),
                case["name"].as_str().map(str::to_owned),
                block_tests::batch(vec![("x", ordinal(index.len()))], index.len()),
            )
            .unwrap();
            append(&frame, &inputs[case["right"].as_str().unwrap()], case);
        }
        for case in contract["chains"].as_array().unwrap() {
            let empty = temporal_append_tests::column("object", &json!([]), mode);
            let frame = IndexedFrame::new(
                empty.clone(),
                Some("datetime".into()),
                block_tests::batch(vec![("x", empty)], 0),
            )
            .unwrap();
            if let Some(frame) = append(
                &frame,
                &inputs[case["first"].as_str().unwrap()],
                &case["first_output"],
            ) {
                append(
                    &frame,
                    &inputs[case["second"].as_str().unwrap()],
                    &case["second_output"],
                );
            }
        }
    }
}

#[test]
fn invalid_index_payloads_fail_without_warnings_or_input_mutation() {
    let bad = arrow_array::new_null_array(&temporal_frame_dtype(), 1);
    let valid = ordinal(1);
    let mut warnings = Vec::new();
    let mut sink = |message: &str| {
        warnings.push(message.to_owned());
    };
    assert!(
        index::from_column(&bad, &mut ArrowOperations, &mut sink)
            .unwrap_err()
            .to_string()
            .contains("typed null")
    );
    let original = bad.to_data();
    let frame = IndexedFrame::new(
        bad.clone(),
        None,
        block_tests::batch(vec![("x", valid.clone())], 1),
    )
    .unwrap();
    let right = block_tests::batch(vec![("datetime", valid.clone()), ("x", valid)], 1);
    assert!(
        dataframe_append_with_warnings(&frame, &right, &mut sink)
            .unwrap_err()
            .to_string()
            .contains("typed null")
    );
    assert!(warnings.is_empty());
    assert_eq!(bad.to_data(), original);
    let half = Arc::new(Float16Array::from(vec![half::f16::ONE])) as ArrayRef;
    assert!(matches!(
        IndexedFrame::new(half, None, block_tests::batch(vec![("x", ordinal(1))], 1)),
        Err(DataframeAppendError::Float16Index)
    ));
}

struct BrokenIndexOperations {
    infer_calls: usize,
}

impl FrameOperations for BrokenIndexOperations {
    fn infer_index(&mut self, values: &[TemporalFrameValue]) -> Result<ArrayRef, ArrowError> {
        self.infer_calls += 1;
        assert_eq!(values.len(), 1);
        Err(ArrowError::ComputeError("index inference failed".into()))
    }
    fn cast(&mut self, array: &ArrayRef, dtype: &DataType) -> Result<ArrayRef, ArrowError> {
        ArrowOperations.cast(array, dtype)
    }
    fn concat(&mut self, left: &ArrayRef, right: &ArrayRef) -> Result<ArrayRef, ArrowError> {
        Ok(Arc::new(arrow_array::StringArray::from(vec![
            "invalid backend payload";
            left.len()
                + right
                    .len()
        ])))
    }
    fn batch(
        &mut self,
        fields: Vec<Field>,
        columns: Vec<ArrayRef>,
        rows: usize,
    ) -> Result<RecordBatch, ArrowError> {
        ArrowOperations.batch(fields, columns, rows)
    }
}

#[test]
fn index_backend_errors_and_unsupported_array_adapters_are_not_suppressed() {
    let object =
        temporal_frame_array(&[TemporalFrameValue::Builtin(BuiltinFrameValue::Int(1))]).unwrap();
    let mut operations = BrokenIndexOperations { infer_calls: 0 };
    let mut warnings = Vec::new();
    let error = index::from_column(&object, &mut operations, &mut |m| {
        warnings.push(m.to_owned());
    })
    .unwrap_err();
    assert_eq!(error.to_string(), "Compute error: index inference failed");
    assert_eq!(operations.infer_calls, 1);
    assert!(warnings.is_empty());
    let boolean = Arc::new(arrow_array::BooleanArray::from(vec![true])) as ArrayRef;
    let numbers = ordinal(1);
    let error = index::join(&boolean, &numbers, &mut operations, &mut ignore_warning).unwrap_err();
    assert!(error.to_string().contains("does not support Utf8"));
    assert_eq!(operations.infer_calls, 1);
    let strings = Arc::new(arrow_array::StringArray::from(vec!["x"])) as ArrayRef;
    for (left, right) in [(&boolean, &strings), (&strings, &boolean)] {
        assert!(matches!(
            index::join(left, right, &mut ArrowOperations, &mut ignore_warning),
            Err(DataframeAppendError::DtypeAdapter(_, _))
        ));
    }
    // A backend implementing only index inference still retains normal batch publication.
    let batch = operations
        .batch(
            vec![Field::new("x", DataType::Int64, false)],
            vec![numbers.clone()],
            1,
        )
        .unwrap();
    assert_eq!(batch.column(0).to_data(), numbers.to_data());
    let tuple_object = tuple_frame_array(&[TupleFrameValue::Scalar(TemporalFrameValue::Builtin(
        BuiltinFrameValue::Int(1),
    ))])
    .unwrap();
    let before = tuple_object.to_data();
    let error = tuple_index::infer(&tuple_object, &mut operations).unwrap_err();
    assert_eq!(error.to_string(), "Compute error: index inference failed");
    assert_eq!(operations.infer_calls, 2);
    assert_eq!(tuple_object.to_data(), before);
}
