use super::super::*;
use serde_json::{Value, json};
use std::{path::PathBuf, process::Command};

pub(in crate::dataframe_append) fn descriptor(value: &Value) -> StringIndexDescriptor {
    let values = value["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| {
            if value[0] == "str" {
                let TemporalFrameValue::Builtin(text) = temporal_objects::tests::decode(value)
                else {
                    panic!("expected builtin text")
                };
                text
            } else {
                BuiltinFrameValue::None
            }
        })
        .collect::<Vec<_>>();
    StringIndexDescriptor::new(
        builtin_frame_array(&values).unwrap(),
        if value["string"]["storage"] == "python" {
            StringStorage::Python
        } else {
            StringStorage::PyArrow
        },
        if value["string"]["missing"] == "NA" {
            StringMissing::PandasNa
        } else {
            StringMissing::NaN
        },
    )
    .unwrap()
}

pub(in crate::dataframe_append) fn compare(result: &IndexedFrame, expected: &Value) {
    let IndexMetadata::String(actual) = result.index_metadata() else {
        panic!("lost string identity: {expected}")
    };
    let wanted = descriptor(expected);
    assert_eq!(actual.as_ref(), &wanted);
    assert_eq!(actual.dtype_name(), expected["dtype"]);
    assert_eq!(result.index().to_data(), actual.storage().to_data());
    builtin_object_tests::compare(actual.values(), &expected["values"]);
}

fn frame(value: &Value, columns: bool) -> IndexedFrame {
    if value.get("string").is_none() {
        return nullable_append::tests::frame(value, columns);
    }
    let descriptor = descriptor(value);
    let rows = descriptor.storage().len();
    let data = block_tests::batch(
        if columns {
            vec![("x", tuple_index_tests::ordinal(rows))]
        } else {
            vec![]
        },
        rows,
    );
    IndexedFrame::from_string_index(descriptor, Some("datetime".into()), data).unwrap()
}

fn other(value: &Value, columns: bool) -> RecordBatch {
    if value.get("string").is_none() {
        return nullable_append::tests::other(value, columns);
    }
    let descriptor = descriptor(value);
    let rows = descriptor.storage().len();
    let mut fields = vec![descriptor.field("datetime")];
    let mut arrays = vec![descriptor.storage()];
    if columns {
        let array = tuple_index_tests::ordinal(rows);
        fields.push(Field::new("x", array.data_type().clone(), true));
        arrays.push(array);
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap()
}

fn append_case(left: &IndexedFrame, right: &RecordBatch, case: &Value) -> Option<IndexedFrame> {
    if case.get("error").is_some() {
        let mut warnings = vec![];
        let result = dataframe_append_with_warnings(left, right, &mut |m| {
            warnings.push(json!(["FutureWarning", m]));
        });
        let Err(error) = result else {
            panic!("expected source error for {case}");
        };
        assert_eq!(error.to_string(), case["message"].as_str().unwrap());
        match case["error"].as_str().unwrap() {
            "UnicodeEncodeError" => assert!(matches!(
                error,
                DataframeAppendError::StringIndex(StringIndexError::Unicode { .. })
            )),
            "TypeError" => assert!(matches!(error, DataframeAppendError::CategoricalIndex(_))),
            unexpected => panic!("unexpected source error: {unexpected}: {case}"),
        }
        assert_eq!(json!(warnings), case["warnings"]);
        None
    } else {
        Some(tuple_index_tests::append_batch(left, right, case))
    }
}

#[test]
fn string_continuous_append_matches_actual_source_histories() {
    compare_histories(
        "dataframe_string_chains.py",
        "326acfd66f70cf67bd5cb1a09a430ec0cfc757a4701caf3222168e5d1c876692",
        2640,
        (5232, 56),
    );
}

#[test]
fn string_tuple_category_histories_match_actual_source() {
    compare_histories(
        "dataframe_string_tuple_category_chains.py",
        "67ea217fbc518489ed0c2f97fc7a0783ef8e1b79ec7e8c38d7a6539e770fe22d",
        2304,
        (3852, 784),
    );
}

#[test]
fn string_mixed_category_histories_match_actual_source() {
    compare_histories(
        "dataframe_string_category_chains.py",
        "626373f48a5eb6b9885acf36dca11d2574bf167ccdcc7b8c24b55b77c6bac129",
        4704,
        (9408, 0),
    );
}

#[test]
fn string_temporal_category_histories_match_actual_source() {
    compare_histories(
        "dataframe_string_temporal_category_chains.py",
        "c5d784328485c426a12f59e9ff0d1ff2dee8d3d542ada5aaf140d839c87d3a2f",
        4608,
        (8880, 336),
    );
}

fn compare_histories(fixture: &str, digest: &str, count: usize, totals: (usize, usize)) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures").join(fixture))
        .arg(root.join("../../../qlib/qlib/rl/order_execution/utils.py"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let contract: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(contract["digest"], digest);
    let cases = contract["cases"].as_array().unwrap();
    assert_eq!(cases.len(), count);
    let mut calls = 0;
    let mut errors = 0;
    for case in cases {
        let columns = case["columns"] == true;
        let left = frame(&contract["inputs"][case["left"].as_str().unwrap()], columns);
        let right = other(
            &contract["inputs"][case["first_name"].as_str().unwrap()],
            columns,
        );
        let before = (
            left.index().to_data(),
            left.index_metadata().clone(),
            left.data().clone(),
            right.clone(),
        );
        calls += 1;
        if let Some(first) = append_case(&left, &right, &case["first"]) {
            let next = other(
                &contract["inputs"][case["second_name"].as_str().unwrap()],
                true,
            );
            let first_before = (
                first.index().to_data(),
                first.index_metadata().clone(),
                first.data().clone(),
                next.clone(),
            );
            calls += 1;
            errors += usize::from(append_case(&first, &next, &case["second"]).is_none());
            assert_eq!(
                (
                    first.index().to_data(),
                    first.index_metadata().clone(),
                    first.data().clone(),
                    next
                ),
                first_before
            );
        } else {
            errors += 1;
            assert!(case["second"].is_null());
        }
        assert_eq!(
            (
                left.index().to_data(),
                left.index_metadata().clone(),
                left.data().clone(),
                right
            ),
            before
        );
    }
    assert_eq!((calls, errors), totals);
}

fn native_string() -> StringIndexDescriptor {
    StringIndexDescriptor::new(
        builtin_frame_array(&[
            BuiltinFrameValue::Text(crate::RlCheckpointText::from_utf8("a")),
            BuiltinFrameValue::None,
        ])
        .unwrap(),
        StringStorage::Python,
        StringMissing::PandasNa,
    )
    .unwrap()
}

#[test]
fn string_attachment_and_column_import_reject_invalid_identity() {
    let string = native_string();
    let array = string.storage();
    assert!(
        IndexedFrame::from_string_index(string.clone(), None, block_tests::batch(vec![], 0))
            .is_err()
    );
    let frame = IndexedFrame::new(array.clone(), None, block_tests::batch(vec![], 2)).unwrap();
    let metadata = IndexMetadata::String(Arc::new(string.clone()));
    assert_eq!(
        frame
            .clone()
            .with_index_metadata(metadata.clone())
            .unwrap()
            .index_metadata(),
        &metadata
    );
    assert!(
        IndexedFrame::new(array.slice(0, 1), None, block_tests::batch(vec![], 1))
            .unwrap()
            .with_index_metadata(metadata)
            .is_err()
    );
    let bad_field = string.field("datetime").with_metadata(
        [(
            super::super::string_index::DTYPE_KEY.into(),
            "unknown".into(),
        )]
        .into(),
    );
    let right =
        RecordBatch::try_new(Arc::new(Schema::new(vec![bad_field])), vec![array.clone()]).unwrap();
    let before = (frame.index().to_data(), right.clone());
    let error = dataframe_append_with_warnings(&frame, &right, &mut |_| {
        panic!("import failure precedes warnings")
    })
    .unwrap_err();
    assert_eq!(
        error.to_string(),
        "invalid string index storage: unknown string dtype metadata"
    );
    assert_eq!((frame.index().to_data(), right), before);
}

#[test]
fn string_append_backend_failures_preserve_inputs() {
    let string = native_string();
    let string_frame = IndexedFrame::from_string_index(
        string.clone(),
        Some("datetime".into()),
        block_tests::batch(vec![], 2),
    )
    .unwrap();
    let object = IndexedFrame::new(
        string.storage(),
        Some("datetime".into()),
        block_tests::batch(vec![], 2),
    )
    .unwrap();
    for left in [&string_frame, &object] {
        for typed in [false, true] {
            let field = if typed {
                string.field("datetime")
            } else {
                Field::new("datetime", string.storage().data_type().clone(), true)
            };
            let right =
                RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![string.storage()])
                    .unwrap();
            let before = (
                left.index().to_data(),
                left.index_metadata().clone(),
                left.data().clone(),
                right.clone(),
            );
            let mut baseline = categorical_append_tests::Failure {
                at: usize::MAX,
                calls: 0,
            };
            let result = append_with_operations(left, &right, &mut baseline, &mut |_| {
                panic!("unexpected warning")
            })
            .unwrap();
            assert_eq!(result.index().len(), 4);
            assert!(baseline.calls > 0);
            for at in 1..=baseline.calls {
                let mut backend = categorical_append_tests::Failure { at, calls: 0 };
                let error = append_with_operations(left, &right, &mut backend, &mut |_| {
                    panic!("unexpected warning")
                })
                .unwrap_err();
                assert_eq!(
                    error.to_string(),
                    "Compute error: categorical backend failed"
                );
                assert_eq!(backend.calls, at);
                assert_eq!(
                    (
                        left.index().to_data(),
                        left.index_metadata().clone(),
                        left.data().clone(),
                        right.clone()
                    ),
                    before
                );
            }
        }
    }
}

struct InvalidConcat(ArrayRef);
impl FrameOperations for InvalidConcat {
    fn cast(&mut self, array: &ArrayRef, dtype: &DataType) -> Result<ArrayRef, ArrowError> {
        ArrowOperations.cast(array, dtype)
    }
    fn concat(&mut self, _: &ArrayRef, _: &ArrayRef) -> Result<ArrayRef, ArrowError> {
        Ok(self.0.clone())
    }
    fn batch(
        &mut self,
        _: Vec<Field>,
        _: Vec<ArrayRef>,
        _: usize,
    ) -> Result<RecordBatch, ArrowError> {
        panic!("invalid index cannot publish a batch")
    }
}

#[test]
fn string_subclass_boxing_and_empty_legacy_inference_failures_are_atomic() {
    let string = StringIndexDescriptor::new(
        native_string().storage(),
        StringStorage::PyArrow,
        StringMissing::NaN,
    )
    .unwrap();
    let empty = StringIndexDescriptor::new(
        string.storage().slice(0, 0),
        StringStorage::Python,
        StringMissing::PandasNa,
    )
    .unwrap();
    let category = CategoricalIndexDescriptor::new(
        Arc::new(arrow_array::Int64Array::from(vec![1])),
        vec![0],
        false,
    )
    .unwrap();
    let multi = MultiIndexDescriptor::new(
        vec![Arc::new(arrow_array::Int64Array::from(vec![1]))],
        vec![vec![0]],
        vec![TupleFrameValue::Scalar(TemporalFrameValue::Builtin(
            BuiltinFrameValue::None,
        ))],
        None,
    )
    .unwrap();
    let cases = [
        (
            category.storage(),
            IndexMetadata::Categorical(Arc::new(category)),
            string.storage(),
            IndexMetadata::String(Arc::new(string.clone())),
        ),
        (
            tuple_frame_array(&multi.values()).unwrap(),
            IndexMetadata::Multi(Arc::new(multi)),
            string.storage(),
            IndexMetadata::String(Arc::new(string.clone())),
        ),
        (
            empty.storage(),
            IndexMetadata::String(Arc::new(empty)),
            string.storage(),
            IndexMetadata::Array,
        ),
    ];
    for (left, lm, right, rm) in cases {
        let before = (left.to_data(), lm.clone(), right.to_data(), rm.clone());
        let mut baseline = categorical_append_tests::Failure {
            at: usize::MAX,
            calls: 0,
        };
        let (output, metadata) = super::join(&left, &lm, &right, &rm, &mut baseline, &mut |_| {
            panic!("unexpected warning")
        })
        .unwrap();
        assert_eq!(output.len(), left.len() + right.len());
        assert_eq!(metadata, IndexMetadata::Array);
        assert!(baseline.calls > 0);
        for at in 1..=baseline.calls {
            let mut backend = categorical_append_tests::Failure { at, calls: 0 };
            let error = super::join(&left, &lm, &right, &rm, &mut backend, &mut |_| {
                panic!("unexpected warning")
            })
            .unwrap_err();
            assert_eq!(
                error.to_string(),
                "Compute error: categorical backend failed"
            );
            assert_eq!(backend.calls, at);
            assert_eq!(
                (left.to_data(), lm.clone(), right.to_data(), rm.clone()),
                before
            );
        }
    }
}

#[test]
fn string_concat_rejects_changed_backend_shape_or_type() {
    let string = native_string();
    let metadata = IndexMetadata::String(Arc::new(string.clone()));
    let before = string.storage().to_data();
    for (array, expected) in [
        (
            string.storage(),
            "invalid index metadata: string concatenation changed row count",
        ),
        (
            builtin_frame_array(&vec![BuiltinFrameValue::Int(1); 4]).unwrap(),
            "invalid string index storage: expected text or None cell",
        ),
    ] {
        let error = super::join(
            &string.storage(),
            &metadata,
            &string.storage(),
            &metadata,
            &mut InvalidConcat(array),
            &mut |_| panic!("unexpected warning"),
        )
        .unwrap_err();
        assert_eq!(error.to_string(), expected);
        assert_eq!(string.storage().to_data(), before);
    }
}

#[test]
fn string_category_missing_comparison_retains_legacy_text_storage() {
    let text = BuiltinFrameValue::Text(crate::RlCheckpointText::from_utf8("z"));
    let categories = builtin_frame_array(&[BuiltinFrameValue::Text(
        crate::RlCheckpointText::from_utf8("a"),
    )])
    .unwrap();
    let category = CategoricalIndexDescriptor::new(categories.clone(), vec![0], false).unwrap();
    for storage in [StringStorage::Python, StringStorage::PyArrow] {
        let string = StringIndexDescriptor::new(
            builtin_frame_array(&[text.clone(), BuiltinFrameValue::None]).unwrap(),
            storage,
            StringMissing::PandasNa,
        )
        .unwrap();
        assert_eq!(
            super::category_codes(&category, &string),
            Some(vec![-1, -1])
        );
        let (array, metadata) = super::join(
            &category.storage(),
            &IndexMetadata::Categorical(Arc::new(category.clone())),
            &string.storage(),
            &IndexMetadata::String(Arc::new(string)),
            &mut ArrowOperations,
            &mut |_| panic!("unexpected warning"),
        )
        .unwrap();
        let IndexMetadata::Categorical(result) = metadata else {
            panic!("lost category")
        };
        assert_eq!(result.codes(), vec![0, -1, -1]);
        assert_eq!(result.categories().to_data(), categories.to_data());
        assert_eq!(array.to_data(), result.storage().to_data());
        assert_eq!(category.codes(), vec![0]);
    }
}

#[test]
fn string_directional_append_matches_actual_source_matrix() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/dataframe_string_append.py"))
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
        "097eeb0eeb2d2527ee0b00e98bbb38e8c57b9a0cbd741a19fe790bef583ea7be"
    );
    let cases = contract["pairs"].as_array().unwrap();
    assert_eq!(cases.len(), 16720);
    assert_eq!(
        cases.iter().filter(|c| c.get("error").is_some()).count(),
        128
    );
    for case in cases {
        let left = frame(
            &contract["inputs"][case["left"].as_str().unwrap()],
            case["left_columns"] == true,
        );
        let right = other(
            &contract["inputs"][case["right"].as_str().unwrap()],
            case["right_columns"] == true,
        );
        let before = (
            left.index().to_data(),
            left.index_metadata().clone(),
            left.data().clone(),
            right.clone(),
        );
        if case.get("error").is_some() {
            let mut warnings = vec![];
            let error = dataframe_append_with_warnings(&left, &right, &mut |m| {
                warnings.push(json!(["FutureWarning", m]));
            })
            .unwrap_err();
            assert_eq!(case["error"], "UnicodeEncodeError");
            assert_eq!(error.to_string(), case["message"].as_str().unwrap());
            assert!(
                matches!(
                    error,
                    DataframeAppendError::StringIndex(StringIndexError::Unicode { .. })
                ),
                "{case}: {error}"
            );
            assert_eq!(json!(warnings), case["warnings"]);
        } else {
            tuple_index_tests::append_batch(&left, &right, case);
        }
        assert_eq!(
            (
                left.index().to_data(),
                left.index_metadata().clone(),
                left.data().clone(),
                right
            ),
            before
        );
    }
}

#[test]
fn string_mixed_category_append_matches_actual_source_matrix() {
    compare_category_pairs(
        "dataframe_string_categories.py",
        "aeaeb095806961429fe0ddd7bda83c775d006b386044aceb0b4d98e6708e3539",
        34496,
        0,
    );
}

#[test]
fn string_temporal_category_append_matches_actual_source_matrix() {
    compare_category_pairs(
        "dataframe_string_temporal_categories.py",
        "ba845d199cca2cdfe98f4d20d2e2152e8bc478b9552cc82a060c815ac81d5e3d",
        33792,
        2272,
    );
}

#[test]
fn string_tuple_category_append_matches_actual_source_matrix() {
    compare_category_pairs(
        "dataframe_string_tuple_categories.py",
        "d3582c1cdc617ec5892c725ef27cf77167db6268d65abe7fc555d21a33dc4abf",
        16896,
        4800,
    );
}

#[test]
fn string_nested_category_append_matches_actual_source_matrix() {
    compare_category_pairs(
        "dataframe_string_nested_categories.py",
        "5d5209ce59802481021a4e5892dd62ad8c74ee1ba805a2103c1ba766ca72897c",
        16896,
        2784,
    );
}

#[test]
fn string_category_hash_order_matches_actual_source_matrix() {
    compare_category_pairs(
        "dataframe_string_hash_order.py",
        "63465a09810bb0ddeb60673749a4354297f4a0fe4e514b3e7a329c8685bd5e13",
        8448,
        6384,
    );
}

fn compare_category_pairs(fixture: &str, digest: &str, count: usize, errors: usize) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures").join(fixture))
        .arg(root.join("../../../qlib/qlib/rl/order_execution/utils.py"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let contract: Value = serde_json::from_slice(&output.stdout).unwrap();
    let cases = contract["pairs"].as_array().unwrap();
    assert_eq!(contract["digest"], digest);
    assert_eq!(cases.len(), count);
    assert_eq!(
        cases.iter().filter(|c| c.get("error").is_some()).count(),
        errors
    );
    for case in cases {
        let left = frame(
            &contract["inputs"][case["left"].as_str().unwrap()],
            case["left_columns"] == true,
        );
        let right = other(
            &contract["inputs"][case["right"].as_str().unwrap()],
            case["right_columns"] == true,
        );
        let before = (
            left.index().to_data(),
            left.index_metadata().clone(),
            left.data().clone(),
            right.clone(),
        );
        assert_eq!(
            append_case(&left, &right, case).is_none(),
            case.get("error").is_some(),
            "{case}"
        );
        assert_eq!(
            (
                left.index().to_data(),
                left.index_metadata().clone(),
                left.data().clone(),
                right
            ),
            before
        );
    }
}
