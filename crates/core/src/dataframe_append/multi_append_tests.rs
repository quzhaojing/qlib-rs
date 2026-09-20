use super::*;
use serde_json::{Value, json};
use std::{path::PathBuf, process::Command};
use tuple_index_tests::{append, array, ordinal};
use tuple_objects::tests::{cell, compare};

#[test]
fn multi_index_reconstruction_failures_preserve_inputs() {
    let tuple = |v| {
        TupleFrameValue::Tuple(vec![TupleFrameValue::Scalar(TemporalFrameValue::Builtin(
            BuiltinFrameValue::Int(v),
        ))])
    };
    let populated = tuple_frame_array(&[tuple(1), tuple(2)]).unwrap();
    let empty = tuple_frame_array(&[]).unwrap();
    let scalar = temporal_frame_array(&[TemporalFrameValue::Timestamp {
        ticks: 0,
        unit: arrow_schema::TimeUnit::Second,
        timezone: None,
    }])
    .unwrap();
    let empty_tuples = tuple_frame_array(&[
        TupleFrameValue::Tuple(vec![]),
        TupleFrameValue::Tuple(vec![]),
    ])
    .unwrap();
    for (left, right) in [
        (&populated, &populated),
        (&populated, &empty_tuples),
        (&empty, &empty_tuples),
        (&empty, &scalar),
    ] {
        let before = (left.to_data(), right.to_data());
        let mut baseline = tuple_index_tests::Failure {
            at: usize::MAX,
            calls: 0,
        };
        assert!(multi_append::join(left, right, &mut baseline).is_ok());
        for at in 1..=baseline.calls {
            let mut backend = tuple_index_tests::Failure { at, calls: 0 };
            let error = multi_append::join(left, right, &mut backend).unwrap_err();
            assert_eq!(
                error.to_string(),
                "Compute error: tuple index backend failed"
            );
            assert_eq!(backend.calls, at);
            assert_eq!((left.to_data(), right.to_data()), before);
        }
    }
    let malformed = arrow_array::new_null_array(&tuple_frame_dtype(), 1);
    assert!(multi_append::join(&malformed, &populated, &mut ArrowOperations).is_err());
    assert!(multi_append::join(&populated, &malformed, &mut ArrowOperations).is_err());
    let descriptor = MultiIndexDescriptor::new(
        vec![Arc::new(arrow_array::Int64Array::from(vec![1]))],
        vec![vec![0]],
        vec![TupleFrameValue::Scalar(TemporalFrameValue::Builtin(
            BuiltinFrameValue::None,
        ))],
        None,
    )
    .unwrap();
    let frame = IndexedFrame::from_multi_index(descriptor, block_tests::batch(vec![], 1)).unwrap();
    let before = frame.index().to_data();
    let other = block_tests::batch(vec![("datetime", populated.clone())], 2);
    let error = append_with_operations(
        &frame,
        &other,
        &mut tuple_index_tests::Failure { at: 1, calls: 0 },
        &mut ignore_warning,
    )
    .unwrap_err();
    assert_eq!(
        error.to_string(),
        "Compute error: tuple index backend failed"
    );
    assert_eq!(frame.index().to_data(), before);
    for dtype in [
        DataType::Timestamp(arrow_schema::TimeUnit::Second, None),
        DataType::Duration(arrow_schema::TimeUnit::Millisecond),
    ] {
        let coarse = arrow_array::new_null_array(&dtype, 1);
        assert!(multi_append::join(&populated, &coarse, &mut ArrowOperations).is_ok());
        assert!(multi_append::join(&populated, &coarse.slice(0, 0), &mut ArrowOperations).is_ok());
    }
}

struct WrongBackend {
    empty_storage: bool,
}

impl FrameOperations for WrongBackend {
    fn tuple_array(&mut self, values: &[TupleFrameValue]) -> Result<ArrayRef, ArrowError> {
        tuple_frame_array(if self.empty_storage { &[] } else { values })
    }
    fn infer_index(&mut self, _values: &[TemporalFrameValue]) -> Result<ArrayRef, ArrowError> {
        Ok(Arc::new(arrow_array::Int64Array::from(Vec::<i64>::new())))
    }
    fn cast(&mut self, array: &ArrayRef, dtype: &DataType) -> Result<ArrayRef, ArrowError> {
        ArrowOperations.cast(array, dtype)
    }
    fn concat(&mut self, left: &ArrayRef, right: &ArrayRef) -> Result<ArrayRef, ArrowError> {
        ArrowOperations.concat(left, right)
    }
    fn batch(
        &mut self,
        _fields: Vec<Field>,
        _columns: Vec<ArrayRef>,
        _rows: usize,
    ) -> Result<RecordBatch, ArrowError> {
        panic!("index reconstruction must fail before batch publication")
    }
}

#[test]
fn multi_index_inconsistent_backend_outputs_are_not_published() {
    let tuple = tuple_frame_array(&[TupleFrameValue::Tuple(vec![TupleFrameValue::Scalar(
        TemporalFrameValue::Builtin(BuiltinFrameValue::Int(1)),
    )])])
    .unwrap();
    let error = multi_append::join(
        &tuple,
        &tuple,
        &mut WrongBackend {
            empty_storage: false,
        },
    )
    .unwrap_err();
    assert!(matches!(
        error,
        DataframeAppendError::MultiIndexReconstruction(_)
    ));
    assert!(
        error
            .to_string()
            .contains("code max (0) >= length of level (0)")
    );
    let empty = tuple_frame_array(&[]).unwrap();
    let empty_tuple = tuple_frame_array(&[TupleFrameValue::Tuple(vec![])]).unwrap();
    let error = multi_append::join(
        &empty,
        &empty_tuple,
        &mut WrongBackend {
            empty_storage: true,
        },
    )
    .unwrap_err();
    assert_eq!(
        error.to_string(),
        "invalid index metadata: empty-tuple backend changed level values"
    );
}

pub(super) fn compare_metadata(metadata: &IndexMetadata, expected: &Value) {
    let IndexMetadata::Multi(descriptor) = metadata else {
        panic!("expected MultiIndex: {expected}")
    };
    assert_eq!(json!(descriptor.sortorder()), expected["sortorder"]);
    let codes = descriptor
        .codes()
        .iter()
        .map(|c| {
            c.iter()
                .map(|c| c.map_or(-1, |v| i64::try_from(v).unwrap()))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(json!(codes), expected["codes"], "{expected}");
    compare(descriptor.names(), expected["names"].as_array().unwrap());
    let levels = expected["levels"].as_array().unwrap();
    assert_eq!(descriptor.levels().len(), levels.len());
    for (((actual, category), nullable), expected) in descriptor
        .levels()
        .iter()
        .zip(descriptor.categorical_levels())
        .zip(descriptor.nullable_levels())
        .zip(levels)
    {
        if expected["kind"] == "CategoricalIndex" {
            assert!(nullable.is_none());
            let wanted = categorical_append_tests::descriptor(expected);
            assert_eq!(category.as_ref().unwrap(), &wanted);
            assert_eq!(actual.to_data(), wanted.storage().to_data());
            continue;
        }
        assert!(category.is_none());
        if let Some(masked) = expected.get("masked") {
            let wanted =
                NullableIndexDescriptor::new(nullable_index::tests::array(masked)).unwrap();
            assert_eq!(nullable.as_ref().unwrap(), &wanted);
            assert_eq!(actual.to_data(), wanted.storage().to_data());
            temporal_objects::tests::compare_cells(
                wanted.values(),
                expected["values"].as_array().unwrap(),
            );
            continue;
        }
        assert!(nullable.is_none());
        if expected["dtype"] == "object" {
            assert!(
                blocks::logical_object(actual.data_type())
                    || actual.data_type() == &tuple_frame_dtype()
            );
        } else {
            assert_eq!(
                actual.data_type(),
                array(expected, false).data_type(),
                "{expected}"
            );
        }
        compare(
            &tuple_index::values(actual).unwrap(),
            expected["values"].as_array().unwrap(),
        );
    }
}

pub(super) fn frame(snapshot: &Value, columns: bool, legacy: bool) -> IndexedFrame {
    let index = array(snapshot, legacy);
    let data = block_tests::batch(
        if columns {
            vec![("x", ordinal(index.len()))]
        } else {
            vec![]
        },
        index.len(),
    );
    frame_with_data(snapshot, data, legacy)
}

fn frame_with_data(snapshot: &Value, data: RecordBatch, legacy: bool) -> IndexedFrame {
    if snapshot["kind"] == "MultiIndex" {
        let levels = snapshot["levels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                if let Some(masked) = v.get("masked") {
                    MultiIndexLevel::Nullable(
                        NullableIndexDescriptor::new(nullable_index::tests::array(masked)).unwrap(),
                    )
                } else if v["kind"] == "CategoricalIndex" {
                    MultiIndexLevel::Categorical(categorical_append_tests::descriptor(v))
                } else {
                    MultiIndexLevel::Array(array(v, legacy))
                }
            })
            .collect();
        let codes = snapshot["codes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                c.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_i64().unwrap())
                    .collect()
            })
            .collect();
        let names = snapshot["names"]
            .as_array()
            .unwrap()
            .iter()
            .map(cell)
            .collect();
        let descriptor =
            MultiIndexDescriptor::from_levels(levels, codes, names, snapshot["sortorder"].as_i64())
                .unwrap();
        IndexedFrame::from_multi_index(descriptor, data).unwrap()
    } else {
        let name = (snapshot["names"][0][0] != "none").then(|| "datetime".into());
        IndexedFrame::new(array(snapshot, legacy), name, data).unwrap()
    }
}

#[test]
fn multi_index_directional_append_and_continuous_identity_match_source() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/dataframe_multi_native_append.py"))
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
        "e672b1c494a4bb20b210cadfbc82687c0db612f6e3f712ace43ffcd206803e64"
    );
    assert_eq!(contract["inputs"].as_object().unwrap().len(), 32);
    assert_eq!(contract["pairs"].as_array().unwrap().len(), 4096);
    assert_eq!(contract["chains"].as_array().unwrap().len(), 128);
    for legacy in [false, true] {
        for case in contract["pairs"].as_array().unwrap() {
            let left = frame(
                &contract["inputs"][case["left"].as_str().unwrap()],
                case["left_columns"] == true,
                legacy,
            );
            let before = left.index_metadata().clone();
            let right = array(&contract["inputs"][case["right"].as_str().unwrap()], legacy);
            append(&left, &right, case["right_columns"] == true, case);
            assert_eq!(left.index_metadata(), &before);
        }
        for case in contract["chains"].as_array().unwrap() {
            let initial = frame_with_data(
                &case["initial"],
                block_tests::batch(vec![("x", builtin_frame_array(&[]).unwrap())], 0),
                legacy,
            );
            let first = array(&contract["inputs"][case["first"].as_str().unwrap()], legacy);
            let current = append(&initial, &first, true, &case["first_output"]);
            let second = array(
                &contract["inputs"][case["second"].as_str().unwrap()],
                legacy,
            );
            append(&current, &second, true, &case["second_output"]);
        }
    }
}

#[test]
fn categorical_multi_levels_actual_append_and_continuous_reconstruction_match_source() {
    multi_level_chains(
        "dataframe_multi_category_chains.py",
        "7969b04d2de7b85d3fdf4e4fa9e931c3e768a8adda42c3a642195c11b0c4e69e",
        3648,
    );
}

#[test]
fn categorical_multi_level_boundaries_continuous_append_match_source() {
    multi_level_chains(
        "dataframe_multi_category_boundary_chains.py",
        "a64396b1eac3ffe3c080bd21c14bbb130c18f7e967de0b56d828a5bd427487ff",
        5760,
    );
}

#[test]
fn nullable_multi_levels_actual_append_and_continuous_reconstruction_match_source() {
    multi_level_chains(
        "dataframe_multi_nullable_chains.py",
        "82219f24f6778d7d62a86220971ce0a2ec64b2ea8f7908a58e45a525cbec1cb7",
        1680,
    );
}

fn multi_level_chains(fixture: &str, digest: &str, count: usize) {
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
    for case in cases {
        let initial = frame(
            &case["left"]["index"],
            !case["left"]["frame"]["columns"]
                .as_array()
                .unwrap()
                .is_empty(),
            false,
        );
        let before = (initial.index_metadata().clone(), initial.data().clone());
        let right = categorical_append_tests::other(
            &contract["inputs"][case["right"].as_str().unwrap()],
            case["right_columns"] == true,
        );
        let first = tuple_index_tests::append_batch(&initial, &right, &case["first"]);
        let next = categorical_append_tests::other(&contract["inputs"]["tuple_two"], true);
        tuple_index_tests::append_batch(&first, &next, &case["second"]);
        assert_eq!(
            (initial.index_metadata().clone(), initial.data().clone()),
            before
        );
    }
}

#[test]
fn multi_index_categorical_failures_preserve_inputs() {
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
    let tuple = tuple_frame_array(&[TupleFrameValue::Tuple(vec![TupleFrameValue::Scalar(
        TemporalFrameValue::Builtin(BuiltinFrameValue::Int(2)),
    )])])
    .unwrap();
    for categories in [ordinal(1), tuple] {
        let descriptor = CategoricalIndexDescriptor::new(categories, vec![0, -1], true).unwrap();
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
        let result =
            append_with_operations(&left, &right, &mut baseline, &mut ignore_warning).unwrap();
        assert_eq!(result.data().num_rows(), 3);
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

#[test]
fn multi_index_categorical_numpy_boxing_matches_source() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/dataframe_multi_categorical.py"))
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
        "111de6d2660ff0d2705b19ff964d389696c99982ad9342e1c9838f64948212f6"
    );
    let cases = contract["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 888);
    for case in cases {
        let left = frame(
            &case["left"]["index"],
            !case["left"]["frame"]["columns"]
                .as_array()
                .unwrap()
                .is_empty(),
            false,
        );
        let right = categorical_append_tests::other(&case["input"], case["right_columns"] == true);
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
fn multi_index_numpy_temporal_boxing_matches_source() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/dataframe_multi_numpy_temporal.py"))
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
        "1102853476e50153c206a75574160b93220bc42521b7b62d339fa3cdb51469cd"
    );
    let cases = contract["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 576);
    for case in cases {
        let left = frame(
            &case["left"]["index"],
            !case["left"]["frame"]["columns"]
                .as_array()
                .unwrap()
                .is_empty(),
            false,
        );
        let right = array(&case["input"], false);
        append(&left, &right, case["right_columns"] == true, case);
    }
}

#[test]
fn utc_object_surrogate_year_and_inference_boundaries_match_source() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/dataframe_utc_object_inference.py"))
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
        "5ad619dec565e95586c6d144e97cc4bc5dab3fcf4c9dc5fbeb3a0c1ed15c9648"
    );
    let cases = contract["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 2640);
    for case in cases {
        let left = frame(&case["left"]["index"], true, false);
        append(&left, &array(&case["input"], false), true, case);
    }
}
