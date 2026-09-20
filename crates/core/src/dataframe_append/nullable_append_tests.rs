use super::super::*;
use arrow_array::Array;
use serde_json::Value;
use std::{path::PathBuf, process::Command};

fn descriptor(value: &Value) -> NullableIndexDescriptor {
    NullableIndexDescriptor::new(nullable_index::tests::array(&value["masked"])).unwrap()
}

pub(in crate::dataframe_append) fn compare(result: &IndexedFrame, expected: &Value) {
    let IndexMetadata::Nullable(actual) = result.index_metadata() else {
        panic!("lost nullable metadata: {expected}")
    };
    let wanted = descriptor(expected);
    assert_eq!(actual.dtype_name(), wanted.dtype_name());
    assert_eq!(result.index().data_type(), wanted.storage().data_type());
    temporal_objects::tests::compare_cells(actual.values(), expected["values"].as_array().unwrap());
    for i in 0..result.index().len() {
        assert_eq!(result.index().is_null(i), wanted.storage().is_null(i));
    }
}

pub(in crate::dataframe_append) fn frame(value: &Value, columns: bool) -> IndexedFrame {
    if value["kind"] == "MultiIndex" {
        return multi_append_tests::frame(value, columns, false);
    }
    let rows = value["values"].as_array().unwrap().len();
    let data = block_tests::batch(
        if columns {
            vec![("x", tuple_index_tests::ordinal(rows))]
        } else {
            vec![]
        },
        rows,
    );
    if value["kind"] == "CategoricalIndex" {
        IndexedFrame::from_categorical_index(
            categorical_append_tests::descriptor(value),
            Some("datetime".into()),
            data,
        )
        .unwrap()
    } else if value.get("masked").is_some() {
        IndexedFrame::from_nullable_index(descriptor(value), Some("datetime".into()), data).unwrap()
    } else {
        IndexedFrame::new(
            tuple_index_tests::array(value, false),
            Some("datetime".into()),
            data,
        )
        .unwrap()
    }
}

pub(in crate::dataframe_append) fn other(value: &Value, columns: bool) -> RecordBatch {
    if value.get("masked").is_none() {
        return categorical_append_tests::other(value, columns);
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

#[test]
fn nullable_numeric_and_object_append_matches_source_matrix() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/dataframe_nullable_append.py"))
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
        "15576dced26955d7b3e9ebc463c86f3f6b378bd5de32afb2a447ce44c692458c"
    );
    let cases = contract["pairs"].as_array().unwrap();
    assert_eq!(cases.len(), 18768);
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
            left.index_metadata().clone(),
            left.data().clone(),
            right.clone(),
        );
        tuple_index_tests::append_batch(&left, &right, case);
        assert_eq!(
            (left.index_metadata().clone(), left.data().clone(), right),
            before
        );
    }
}

#[test]
fn nullable_categorical_directional_append_matches_source_matrix() {
    source_matrix(
        "dataframe_nullable_categorical.py",
        "a5dcdc7eaec988371cea2c3f2736218591473e626176d566ddadf7764dc4c6a0",
        39008,
        320,
    );
}

#[test]
fn nullable_numeric_boundaries_match_source_matrix() {
    source_matrix(
        "dataframe_nullable_boundaries.py",
        "e1994fdc15526d395cc3951b5be148b58f232b6f8b49426adf349e9e0276f7b1",
        18876,
        284,
    );
}

#[test]
fn nullable_numeric_boundary_histories_match_source() {
    source_histories(
        "dataframe_nullable_boundary_chains.py",
        "da6d89412dde999b2ba4ab3f2994688904e2fa1e8e5a95448b4bcb0866f9d8a2",
        880,
        (1710, 58),
    );
}

#[test]
fn masked_category_numeric_boundary_histories_match_source() {
    source_histories(
        "dataframe_masked_category_boundary_chains.py",
        "645a513f1ba4c0ebb3fe4b176a4ff3dfb05c02c02c7e9d8f0333ec3e376a152a",
        990,
        (1980, 0),
    );
}

fn source_histories(fixture: &str, digest: &str, chain_count: usize, counts: (usize, usize)) {
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
    let chains = contract["chains"].as_array().unwrap();
    assert_eq!(chains.len(), chain_count);
    let mut calls = 0;
    let mut errors = 0;
    for chain in chains {
        let mut current = frame(
            &contract["inputs"][chain["left"].as_str().unwrap()],
            chain["left_columns"] == true,
        );
        for step in chain["steps"].as_array().unwrap() {
            calls += 1;
            let right = other(
                &contract["inputs"][step["right"].as_str().unwrap()],
                step["right_columns"] == true,
            );
            let before = (
                current.index().to_data(),
                current.index_metadata().clone(),
                current.data().clone(),
                right.clone(),
            );
            if step.get("error").is_some() {
                errors += 1;
                let mut warnings = vec![];
                let result = dataframe_append_with_warnings(&current, &right, &mut |m| {
                    warnings.push(serde_json::json!(["FutureWarning", m]));
                });
                assert_eq!(serde_json::json!(warnings), step["warnings"], "{chain}");
                assert_eq!(
                    result.unwrap_err().to_string(),
                    step["message"].as_str().unwrap(),
                    "{chain}"
                );
                assert_eq!(
                    (
                        current.index().to_data(),
                        current.index_metadata().clone(),
                        current.data().clone(),
                        right
                    ),
                    before
                );
            } else {
                let next = tuple_index_tests::append_batch(&current, &right, step);
                assert_eq!(
                    (
                        current.index().to_data(),
                        current.index_metadata().clone(),
                        current.data().clone(),
                        right
                    ),
                    before
                );
                current = next;
            }
        }
    }
    assert_eq!((calls, errors), counts);
}

#[test]
fn masked_category_numeric_boundaries_match_source_matrix() {
    source_matrix(
        "dataframe_masked_category_boundaries.py",
        "4be3589f46e2dcb62b72dfc5b13908fc658b832a7793c564897492ce9f76e08e",
        42276,
        224,
    );
}

#[test]
fn masked_category_nonempty_append_matches_source_matrix() {
    source_matrix(
        "dataframe_categorical_nullable_append.py",
        "b402eb81eb624b3a7f75ad9c0692ded13f7baefa7e6e41dae1b9a633a9257770",
        23660,
        440,
    );
}

fn source_matrix(fixture: &str, digest: &str, pair_count: usize, error_count: usize) {
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
    let cases = contract["pairs"].as_array().unwrap();
    assert_eq!(cases.len(), pair_count);
    assert_eq!(
        cases.iter().filter(|c| c.get("error").is_some()).count(),
        error_count
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
            let result = dataframe_append_with_warnings(&left, &right, &mut |m| {
                warnings.push(serde_json::json!(["FutureWarning", m]));
            });
            assert_eq!(serde_json::json!(warnings), case["warnings"], "{case}");
            assert_eq!(
                result.unwrap_err().to_string(),
                case["message"].as_str().unwrap(),
                "{case}"
            );
        } else {
            let comparison = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                tuple_index_tests::append_batch(&left, &right, case)
            }));
            assert!(comparison.is_ok(), "{case}");
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
fn nullable_attachment_and_field_errors_are_not_silently_demoted() {
    let array: ArrayRef = Arc::new(arrow_array::Int64Array::from(vec![Some(1), None]));
    let descriptor = NullableIndexDescriptor::new(array.clone()).unwrap();
    assert!(matches!(
        IndexedFrame::from_nullable_index(descriptor.clone(), None, block_tests::batch(vec![], 0)),
        Err(DataframeAppendError::IndexLength)
    ));
    let plain = IndexedFrame::new(array.clone(), None, block_tests::batch(vec![], 2)).unwrap();
    assert_eq!(plain.index_metadata(), &IndexMetadata::Array);
    let attached = plain
        .clone()
        .with_index_metadata(IndexMetadata::Nullable(Arc::new(descriptor.clone())))
        .unwrap();
    assert!(matches!(
        attached.index_metadata(),
        IndexMetadata::Nullable(_)
    ));
    let different =
        NullableIndexDescriptor::new(Arc::new(arrow_array::Int64Array::from(vec![1, 2]))).unwrap();
    assert_eq!(
        plain
            .with_index_metadata(IndexMetadata::Nullable(Arc::new(different)))
            .unwrap_err()
            .to_string(),
        "invalid index metadata: NullableIndex descriptor does not match storage"
    );
    let bad_field = descriptor
        .field("datetime")
        .with_metadata([(nullable_index::DTYPE_KEY.into(), "Float64".into())].into());
    let right = RecordBatch::try_new(Arc::new(Schema::new(vec![bad_field])), vec![array]).unwrap();
    assert_eq!(
        dataframe_append_with_warnings(&attached, &right, &mut |_| panic!(
            "invalid field must not warn"
        ))
        .unwrap_err()
        .to_string(),
        "invalid masked index storage: masked dtype metadata does not match array"
    );
}

#[test]
fn nullable_append_backend_failures_preserve_inputs() {
    for array in [
        Arc::new(arrow_array::Int64Array::from(vec![Some(1), None])) as ArrayRef,
        Arc::new(arrow_array::BooleanArray::from(vec![Some(true), None])),
    ] {
        for empty in [false, true] {
            let array = if empty {
                array.slice(0, 0)
            } else {
                array.clone()
            };
            let left = IndexedFrame::from_nullable_index(
                NullableIndexDescriptor::new(array.clone()).unwrap(),
                Some("datetime".into()),
                block_tests::batch(
                    vec![("x", tuple_index_tests::ordinal(array.len()))],
                    array.len(),
                ),
            )
            .unwrap();
            for values in [
                Arc::new(arrow_array::Float64Array::from(vec![f64::NAN, 1.5])) as ArrayRef,
                tuple_frame_array(&[TupleFrameValue::Scalar(TemporalFrameValue::Builtin(
                    BuiltinFrameValue::Int(3),
                ))])
                .unwrap(),
            ] {
                let right = block_tests::batch(
                    vec![
                        ("datetime", values.clone()),
                        ("x", tuple_index_tests::ordinal(values.len())),
                    ],
                    values.len(),
                );
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
                let result =
                    append_with_operations(&left, &right, &mut baseline, &mut ignore_warning)
                        .unwrap();
                assert_eq!(result.data().num_rows(), array.len() + values.len());
                for at in 1..=baseline.calls {
                    let mut backend = categorical_append_tests::Failure { at, calls: 0 };
                    assert_eq!(
                        append_with_operations(&left, &right, &mut backend, &mut ignore_warning)
                            .unwrap_err()
                            .to_string(),
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
}

struct InvalidBackend {
    oversized_cast: bool,
}

#[test]
fn nullable_categorical_failures_and_negative_membership_preserve_inputs() {
    let category = CategoricalIndexDescriptor::new(
        Arc::new(arrow_array::UInt8Array::from(vec![0, 1])),
        vec![0, -1],
        false,
    )
    .unwrap();
    let nullable = NullableIndexDescriptor::new(Arc::new(arrow_array::Int64Array::from(vec![
        Some(-1),
        None,
    ])))
    .unwrap();
    assert_eq!(
        super::category_codes(&category, &nullable, &mut ArrowOperations).unwrap(),
        Some(vec![-1, -1])
    );
    let categories = [
        Arc::new(arrow_array::Int64Array::from(vec![0, 1])) as ArrayRef,
        Arc::new(arrow_array::BooleanArray::from(vec![false, true])),
        tuple_frame_array(&[TupleFrameValue::Scalar(TemporalFrameValue::Builtin(
            BuiltinFrameValue::Text(crate::RlCheckpointText::from_utf8("a")),
        ))])
        .unwrap(),
    ];
    for values in categories {
        let category = CategoricalIndexDescriptor::new(values, vec![0, -1], false).unwrap();
        let masked =
            NullableIndexDescriptor::new(Arc::new(arrow_array::Int64Array::from(vec![5, 6])))
                .unwrap();
        for reverse in [false, true] {
            let (left, field, right) = if reverse {
                (
                    IndexedFrame::from_categorical_index(
                        category.clone(),
                        Some("datetime".into()),
                        block_tests::batch(vec![], 2),
                    )
                    .unwrap(),
                    masked.field("datetime"),
                    masked.storage(),
                )
            } else {
                (
                    IndexedFrame::from_nullable_index(
                        masked.clone(),
                        Some("datetime".into()),
                        block_tests::batch(vec![], 2),
                    )
                    .unwrap(),
                    category.field("datetime"),
                    category.storage(),
                )
            };
            let right =
                RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![right]).unwrap();
            let before = (
                left.index().to_data(),
                left.index_metadata().clone(),
                right.clone(),
            );
            let mut baseline = categorical_append_tests::Failure {
                at: usize::MAX,
                calls: 0,
            };
            assert_eq!(
                append_with_operations(&left, &right, &mut baseline, &mut ignore_warning)
                    .unwrap()
                    .data()
                    .num_rows(),
                4
            );
            for at in 1..=baseline.calls {
                let mut backend = categorical_append_tests::Failure { at, calls: 0 };
                assert_eq!(
                    append_with_operations(&left, &right, &mut backend, &mut ignore_warning)
                        .unwrap_err()
                        .to_string(),
                    "Compute error: categorical backend failed"
                );
                assert_eq!(backend.calls, at);
                assert_eq!(
                    (
                        left.index().to_data(),
                        left.index_metadata().clone(),
                        right.clone()
                    ),
                    before
                );
            }
        }
    }
}

struct TruncatedCategories(usize);

#[test]
fn masked_category_unsigned_lookup_narrowing_matches_source_and_is_atomic() {
    for (dtype, categories, values, expected) in [
        (DataType::UInt8, vec![0, 1], vec![Some(257), None], None),
        (
            DataType::UInt8,
            vec![0, 1],
            vec![Some(i64::MAX), None],
            Some(vec![-1, -1]),
        ),
        (
            DataType::UInt8,
            vec![0, 1],
            vec![Some(1), None],
            Some(vec![1, -1]),
        ),
        (
            DataType::UInt8,
            vec![0, 1],
            vec![Some(-1), None],
            Some(vec![-1, -1]),
        ),
        (DataType::UInt8, vec![0, 1], vec![None, None], None),
        (
            DataType::UInt16,
            vec![0, 1],
            vec![Some(257), None],
            Some(vec![-1, -1]),
        ),
        (
            DataType::UInt64,
            vec![1_u64 << 63],
            vec![Some(i64::MAX), None],
            Some(vec![-1, -1]),
        ),
    ] {
        let categories =
            arrow_cast::cast(&arrow_array::UInt64Array::from(categories), &dtype).unwrap();
        let category = CategoricalIndexDescriptor::from_nullable_categories(
            NullableIndexDescriptor::new(categories).unwrap(),
            vec![0, -1],
            false,
        )
        .unwrap();
        let nullable =
            NullableIndexDescriptor::new(Arc::new(arrow_array::Int64Array::from(values))).unwrap();
        let before = (category.storage().to_data(), nullable.storage().to_data());
        let mut baseline = categorical_append_tests::Failure {
            at: usize::MAX,
            calls: 0,
        };
        assert_eq!(
            super::category_codes(&category, &nullable, &mut baseline).unwrap(),
            expected
        );
        for at in 1..=baseline.calls {
            let mut backend = categorical_append_tests::Failure { at, calls: 0 };
            assert_eq!(
                super::category_codes(&category, &nullable, &mut backend)
                    .unwrap_err()
                    .to_string(),
                "Compute error: categorical backend failed"
            );
            assert_eq!(backend.calls, at);
        }
        assert_eq!(
            (category.storage().to_data(), nullable.storage().to_data()),
            before
        );
    }
}

struct UnsignedLookupBackend(ArrayRef);

impl FrameOperations for UnsignedLookupBackend {
    fn cast(&mut self, _: &ArrayRef, dtype: &DataType) -> Result<ArrayRef, ArrowError> {
        assert_eq!(dtype, &DataType::UInt64);
        Ok(self.0.clone())
    }
    fn concat(&mut self, _: &ArrayRef, _: &ArrayRef) -> Result<ArrayRef, ArrowError> {
        panic!("invalid lookup cannot concatenate");
    }
    fn batch(
        &mut self,
        _: Vec<Field>,
        _: Vec<ArrayRef>,
        _: usize,
    ) -> Result<RecordBatch, ArrowError> {
        panic!("invalid lookup cannot publish a batch");
    }
}

#[test]
fn unsigned_category_lookup_rejects_changed_backend_dtype_and_length() {
    let input: ArrayRef = Arc::new(arrow_array::Int64Array::from(vec![Some(1), None]));
    let before = input.to_data();
    for (output, message) in [
        (
            arrow_array::new_null_array(&DataType::Int64, 2),
            "unsigned lookup cast changed dtype",
        ),
        (
            arrow_array::new_null_array(&DataType::UInt64, 0),
            "unsigned lookup cast changed array length",
        ),
    ] {
        assert_eq!(
            super::unsigned_lookup_keys(&input, 255, &mut UnsignedLookupBackend(output))
                .unwrap_err()
                .to_string(),
            message
        );
        assert_eq!(input.to_data(), before);
    }
}

#[test]
fn masked_category_precision_recovery_is_conditional_and_atomic() {
    for (values, expected) in [
        (
            [9_007_199_254_740_993, 9_007_199_254_740_995],
            [9_007_199_254_740_993, 9_007_199_254_740_995],
        ),
        (
            [9_007_199_254_740_993, 9_007_199_254_740_996],
            [9_007_199_254_740_992, 9_007_199_254_740_996],
        ),
        (
            [9_007_199_254_740_992, 9_007_199_254_740_993],
            [9_007_199_254_740_992, 9_007_199_254_740_992],
        ),
    ] {
        let category = CategoricalIndexDescriptor::from_nullable_categories(
            NullableIndexDescriptor::new(Arc::new(arrow_array::Int64Array::from(values.to_vec())))
                .unwrap(),
            vec![0, 1, -1],
            false,
        )
        .unwrap();
        let storage = category.storage();
        let before = storage.to_data();
        let metadata = IndexMetadata::Categorical(Arc::new(category));
        let mut baseline = categorical_append_tests::Failure {
            at: usize::MAX,
            calls: 0,
        };
        let result = super::cast(&storage, &metadata, &DataType::Int64, &mut baseline).unwrap();
        assert_eq!(
            result.to_data(),
            arrow_array::Int64Array::from(vec![Some(expected[0]), Some(expected[1]), None])
                .to_data()
        );
        for at in 1..=baseline.calls {
            let mut backend = categorical_append_tests::Failure { at, calls: 0 };
            assert_eq!(
                super::cast(&storage, &metadata, &DataType::Int64, &mut backend)
                    .unwrap_err()
                    .to_string(),
                "Compute error: categorical backend failed"
            );
            assert_eq!(backend.calls, at);
            assert_eq!(storage.to_data(), before);
        }
    }
}

#[test]
fn masked_category_precision_rejects_invalid_rounding_and_retains_all_missing() {
    let category = CategoricalIndexDescriptor::from_nullable_categories(
        NullableIndexDescriptor::new(Arc::new(arrow_array::UInt64Array::from(vec![u64::MAX])))
            .unwrap(),
        vec![-1],
        false,
    )
    .unwrap();
    let storage = category.storage();
    let before = storage.to_data();
    let rounded = arrow_array::new_null_array(&DataType::Float64, 1);
    assert!(!super::recover_category_precision(&category, &rounded).unwrap());
    let invalid: ArrayRef = Arc::new(arrow_array::Int64Array::from(vec![1]));
    assert_eq!(
        super::recover_category_precision(&category, &invalid)
            .unwrap_err()
            .to_string(),
        "categorical numeric materialization changed dtype"
    );
    let invalid = arrow_array::new_null_array(&tuple_frame_dtype(), 1);
    assert_eq!(
        super::recover_category_precision(&category, &invalid)
            .unwrap_err()
            .to_string(),
        "Invalid argument error: tuple object has null row"
    );
    let result = super::cast(
        &storage,
        &IndexMetadata::Categorical(Arc::new(category)),
        &DataType::UInt64,
        &mut ArrowOperations,
    )
    .unwrap();
    assert_eq!(
        result.to_data(),
        arrow_array::new_null_array(&DataType::UInt64, 1).to_data()
    );
    assert_eq!(storage.to_data(), before);
}

#[test]
fn categorical_missing_integer_materialization_preserves_source_rounding() {
    let original = [9_007_199_254_740_993_i64, 9_007_199_254_740_995];
    for missing in [false, true] {
        let codes = if missing { vec![0, 1, -1] } else { vec![0, 1] };
        let category = CategoricalIndexDescriptor::new(
            Arc::new(arrow_array::Int64Array::from(original.to_vec())),
            codes,
            false,
        )
        .unwrap();
        let storage = category.storage();
        let before = storage.to_data();
        let output = super::cast(
            &storage,
            &IndexMetadata::Categorical(Arc::new(category)),
            &DataType::Int64,
            &mut ArrowOperations,
        )
        .unwrap();
        let wanted = if missing {
            vec![
                Some(9_007_199_254_740_992),
                Some(9_007_199_254_740_996),
                None,
            ]
        } else {
            original.into_iter().map(Some).collect()
        };
        assert_eq!(
            output.to_data(),
            arrow_array::Int64Array::from(wanted).to_data()
        );
        assert_eq!(storage.to_data(), before);
    }
}

#[test]
fn categorical_floating_lookup_cast_failures_preserve_inputs() {
    let category = CategoricalIndexDescriptor::new(
        Arc::new(arrow_array::Int64Array::from(vec![i64::MAX])),
        vec![0],
        false,
    )
    .unwrap();
    let nullable = NullableIndexDescriptor::new(Arc::new(arrow_array::Float64Array::from(vec![
        9_223_372_036_854_775_808.0,
    ])))
    .unwrap();
    let before = (category.storage().to_data(), nullable.storage().to_data());
    let mut baseline = categorical_append_tests::Failure {
        at: usize::MAX,
        calls: 0,
    };
    assert_eq!(
        super::category_codes(&category, &nullable, &mut baseline).unwrap(),
        Some(vec![0])
    );
    for at in 1..=baseline.calls {
        let mut backend = categorical_append_tests::Failure { at, calls: 0 };
        assert_eq!(
            super::category_codes(&category, &nullable, &mut backend)
                .unwrap_err()
                .to_string(),
            "Compute error: categorical backend failed"
        );
        assert_eq!(backend.calls, at);
        assert_eq!(
            (category.storage().to_data(), nullable.storage().to_data()),
            before
        );
    }
}

struct InvalidLookup;

impl FrameOperations for InvalidLookup {
    fn cast(&mut self, _: &ArrayRef, _: &DataType) -> Result<ArrayRef, ArrowError> {
        Ok(arrow_array::new_null_array(&tuple_frame_dtype(), 1))
    }
    fn concat(&mut self, _: &ArrayRef, _: &ArrayRef) -> Result<ArrayRef, ArrowError> {
        panic!("invalid lookup must precede concatenation")
    }
    fn batch(
        &mut self,
        _: Vec<Field>,
        _: Vec<ArrayRef>,
        _: usize,
    ) -> Result<RecordBatch, ArrowError> {
        panic!("invalid lookup must precede publication")
    }
}

#[test]
fn categorical_lookup_rejects_invalid_backend_objects() {
    let array: ArrayRef = Arc::new(arrow_array::Int64Array::from(vec![1]));
    let before = array.to_data();
    assert_eq!(
        super::cast_keys(&array, &DataType::Float64, &mut InvalidLookup)
            .unwrap_err()
            .to_string(),
        "Invalid argument error: tuple object has null row"
    );
    assert_eq!(array.to_data(), before);
    assert_eq!(
        super::cast_keys_with(&array, &DataType::Float64, &mut ArrowOperations, |_| Ok(
            vec![TupleFrameValue::Scalar(TemporalFrameValue::Timestamp {
                ticks: i64::MIN,
                unit: arrow_schema::TimeUnit::Second,
                timezone: None,
            })]
        ))
        .unwrap_err()
        .to_string(),
        "Invalid argument error: temporal object uses reserved NaT ticks"
    );
    let half = arrow_cast::cast(array.as_ref(), &DataType::Float16).unwrap();
    assert!(CategoricalIndexDescriptor::new(half, vec![0], false).is_err());
    assert_eq!(array.to_data(), before);
}

impl FrameOperations for TruncatedCategories {
    fn cast(&mut self, array: &ArrayRef, dtype: &DataType) -> Result<ArrayRef, ArrowError> {
        if self.0 > 0 {
            self.0 -= 1;
            return ArrowOperations.cast(array, dtype);
        }
        Ok(arrow_array::new_null_array(dtype, 0))
    }
    fn concat(&mut self, _: &ArrayRef, _: &ArrayRef) -> Result<ArrayRef, ArrowError> {
        panic!("invalid take must fail before concatenation")
    }
    fn batch(
        &mut self,
        _: Vec<Field>,
        _: Vec<ArrayRef>,
        _: usize,
    ) -> Result<RecordBatch, ArrowError> {
        panic!("invalid take must fail before publication")
    }
}

#[test]
fn categorical_to_masked_take_checks_backend_bounds() {
    let category = CategoricalIndexDescriptor::new(
        Arc::new(arrow_array::Int64Array::from(vec![1])),
        vec![0],
        false,
    )
    .unwrap();
    let array = category.storage();
    let before = array.to_data();
    let error = super::cast(
        &array,
        &IndexMetadata::Categorical(Arc::new(category)),
        &DataType::Float64,
        &mut TruncatedCategories(0),
    )
    .unwrap_err();
    assert!(error.to_string().contains("out of bounds"));
    assert_eq!(array.to_data(), before);
}

#[test]
fn masked_category_precision_restoration_checks_backend_bounds() {
    let category = CategoricalIndexDescriptor::from_nullable_categories(
        NullableIndexDescriptor::new(Arc::new(arrow_array::Int64Array::from(vec![
            9_007_199_254_740_993,
        ])))
        .unwrap(),
        vec![0, -1],
        false,
    )
    .unwrap();
    let array = category.storage();
    let before = array.to_data();
    let error = super::cast(
        &array,
        &IndexMetadata::Categorical(Arc::new(category)),
        &DataType::Int64,
        &mut TruncatedCategories(1),
    )
    .unwrap_err();
    assert!(error.to_string().contains("out of bounds"));
    assert_eq!(array.to_data(), before);
}

impl FrameOperations for InvalidBackend {
    fn cast(&mut self, array: &ArrayRef, dtype: &DataType) -> Result<ArrayRef, ArrowError> {
        if self.oversized_cast {
            Ok(arrow_array::new_null_array(dtype, 65))
        } else {
            ArrowOperations.cast(array, dtype)
        }
    }
    fn concat(&mut self, _: &ArrayRef, _: &ArrayRef) -> Result<ArrayRef, ArrowError> {
        Ok(Arc::new(arrow_array::StringArray::from(vec!["invalid"])))
    }
    fn batch(
        &mut self,
        _: Vec<Field>,
        _: Vec<ArrayRef>,
        _: usize,
    ) -> Result<RecordBatch, ArrowError> {
        panic!("invalid index must not be published")
    }
}

#[test]
fn nullable_rejects_unsupported_and_inconsistent_backend_values() {
    let array: ArrayRef = Arc::new(arrow_array::Float64Array::from(vec![Some(1.), None]));
    let metadata = IndexMetadata::Nullable(Arc::new(
        NullableIndexDescriptor::new(array.clone()).unwrap(),
    ));
    let before = array.to_data();
    let half = arrow_array::new_null_array(&DataType::Float16, 1);
    assert!(matches!(
        super::join(
            &array,
            &metadata,
            &half,
            &IndexMetadata::Array,
            &mut ArrowOperations,
            &mut ignore_warning
        ),
        Err(DataframeAppendError::DtypeAdapter { .. })
    ));
    assert_eq!(
        super::cast_with(
            &half,
            &IndexMetadata::Array,
            &DataType::Float64,
            &mut ArrowOperations,
            |_| Err(ArrowError::ComputeError("masked decoder failed".into())),
        )
        .unwrap_err()
        .to_string(),
        "Compute error: masked decoder failed"
    );
    let malformed = arrow_array::new_null_array(&tuple_frame_dtype(), 1);
    assert!(
        super::join(
            &array,
            &metadata,
            &malformed,
            &IndexMetadata::Array,
            &mut ArrowOperations,
            &mut ignore_warning
        )
        .is_err()
    );
    let native: ArrayRef = Arc::new(arrow_array::Float64Array::from(vec![Some(1.), None]));
    let cast = super::cast(
        &native,
        &IndexMetadata::Array,
        &DataType::Float64,
        &mut ArrowOperations,
    )
    .unwrap();
    assert_eq!(cast.to_data(), native.to_data());
    assert!(
        super::cast(
            &native,
            &IndexMetadata::Array,
            &DataType::Float64,
            &mut InvalidBackend {
                oversized_cast: true
            }
        )
        .is_err()
    );
    let error = super::join(
        &array,
        &metadata,
        &array,
        &metadata,
        &mut InvalidBackend {
            oversized_cast: false,
        },
        &mut ignore_warning,
    )
    .unwrap_err();
    assert!(matches!(error, DataframeAppendError::NullableIndex(_)));
    assert!(error.to_string().contains("Utf8"));
    assert_eq!(array.to_data(), before);
}

#[test]
fn nullable_empty_append_accepts_builtin_object_encoding_without_warning() {
    let empty: ArrayRef = Arc::new(arrow_array::BooleanArray::from(Vec::<bool>::new()));
    let metadata = IndexMetadata::Nullable(Arc::new(
        NullableIndexDescriptor::new(empty.clone()).unwrap(),
    ));
    let values =
        builtin_frame_array(&[BuiltinFrameValue::Bool(true), BuiltinFrameValue::None]).unwrap();
    let (output, actual) = super::join(
        &empty,
        &metadata,
        &values,
        &IndexMetadata::Array,
        &mut ArrowOperations,
        &mut |_| panic!("object survivor must not warn"),
    )
    .unwrap();
    assert_eq!(actual, IndexMetadata::Array);
    assert_eq!(
        temporal_cast::values(&output).unwrap(),
        vec![
            TemporalFrameValue::Builtin(BuiltinFrameValue::Bool(true)),
            TemporalFrameValue::Builtin(BuiltinFrameValue::None),
        ]
    );
    assert_eq!(values.len(), 2);
}

#[test]
fn multi_index_nullable_input_failures_preserve_inputs() {
    let descriptor = MultiIndexDescriptor::new(
        vec![Arc::new(arrow_array::Int64Array::from(vec![1]))],
        vec![vec![0]],
        vec![TupleFrameValue::Scalar(TemporalFrameValue::Builtin(
            BuiltinFrameValue::None,
        ))],
        None,
    )
    .unwrap();
    let left = IndexedFrame::from_multi_index(descriptor, block_tests::batch(vec![], 1)).unwrap();
    for array in [
        Arc::new(arrow_array::Int64Array::from(vec![Some(1), None])) as ArrayRef,
        Arc::new(arrow_array::BooleanArray::from(vec![Some(true), None])),
    ] {
        let descriptor = NullableIndexDescriptor::new(array).unwrap();
        let right = RecordBatch::try_new(
            Arc::new(Schema::new(vec![descriptor.field("datetime")])),
            vec![descriptor.storage()],
        )
        .unwrap();
        let before = (
            left.index().to_data(),
            left.index_metadata().clone(),
            right.clone(),
        );
        let mut baseline = categorical_append_tests::Failure {
            at: usize::MAX,
            calls: 0,
        };
        assert_eq!(
            append_with_operations(&left, &right, &mut baseline, &mut ignore_warning)
                .unwrap()
                .data()
                .num_rows(),
            3
        );
        for at in 1..=baseline.calls {
            let mut backend = categorical_append_tests::Failure { at, calls: 0 };
            assert_eq!(
                append_with_operations(&left, &right, &mut backend, &mut ignore_warning)
                    .unwrap_err()
                    .to_string(),
                "Compute error: categorical backend failed"
            );
            assert_eq!(backend.calls, at);
            assert_eq!(
                (
                    left.index().to_data(),
                    left.index_metadata().clone(),
                    right.clone()
                ),
                before
            );
        }
    }
}
