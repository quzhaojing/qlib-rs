use super::*;
use serde_json::{Value, json};
use std::{path::PathBuf, process::Command};
use tuple_index_tests::{array, ordinal};
use tuple_objects::tests::compare;

#[test]
fn categorical_recode_preserves_left_object_storage_across_supported_encodings() {
    let cells = [1, 2, 99].map(|value| TemporalFrameValue::Builtin(BuiltinFrameValue::Int(value)));
    let scalar = temporal_frame_array(&cells).unwrap();
    let tuple = tuple_frame_array(
        &cells
            .iter()
            .cloned()
            .map(TupleFrameValue::Scalar)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    for (left, right) in [(&scalar, &tuple), (&tuple, &scalar)] {
        let left_descriptor =
            CategoricalIndexDescriptor::new(left.clone(), vec![0, -1], true).unwrap();
        let right_descriptor =
            CategoricalIndexDescriptor::new(right.clone(), vec![1], true).unwrap();
        let frame = IndexedFrame::from_categorical_index(
            left_descriptor,
            Some("datetime".into()),
            block_tests::batch(vec![], 2),
        )
        .unwrap();
        let other = RecordBatch::try_new(
            Arc::new(Schema::new(vec![right_descriptor.field("datetime")])),
            vec![right_descriptor.storage()],
        )
        .unwrap();
        let result = dataframe_append_with_warnings(&frame, &other, &mut |_| {
            panic!("equivalent object categories must not warn")
        })
        .unwrap();
        let IndexMetadata::Categorical(descriptor) = result.index_metadata() else {
            panic!("lost categorical identity")
        };
        assert_eq!(descriptor.categories().to_data(), left.to_data());
        assert_eq!(descriptor.codes(), vec![0, -1, 1]);
        assert!(descriptor.ordered());
        assert_eq!(result.data().num_rows(), 3);
    }
}

pub(super) struct Failure {
    pub(super) at: usize,
    pub(super) calls: usize,
}

impl Failure {
    fn step(&mut self) -> Result<(), ArrowError> {
        self.calls += 1;
        if self.calls == self.at {
            Err(ArrowError::ComputeError(
                "categorical backend failed".into(),
            ))
        } else {
            Ok(())
        }
    }
}

impl FrameOperations for Failure {
    fn cast(&mut self, array: &ArrayRef, dtype: &DataType) -> Result<ArrayRef, ArrowError> {
        self.step()?;
        ArrowOperations.cast(array, dtype)
    }
    fn concat(&mut self, left: &ArrayRef, right: &ArrayRef) -> Result<ArrayRef, ArrowError> {
        self.step()?;
        ArrowOperations.concat(left, right)
    }
    fn tuple_array(&mut self, values: &[TupleFrameValue]) -> Result<ArrayRef, ArrowError> {
        self.step()?;
        tuple_frame_array(values)
    }
    fn infer_index(&mut self, values: &[TemporalFrameValue]) -> Result<ArrayRef, ArrowError> {
        self.step()?;
        infer_temporal_frame_values(values)
    }
    fn batch(
        &mut self,
        fields: Vec<Field>,
        columns: Vec<ArrayRef>,
        rows: usize,
    ) -> Result<RecordBatch, ArrowError> {
        self.step()?;
        ArrowOperations.batch(fields, columns, rows)
    }
}

#[test]
fn categorical_append_backend_failures_preserve_inputs() {
    let integers: ArrayRef = Arc::new(arrow_array::Int64Array::from(vec![1, 2]));
    let text = tuple_frame_array(&[TupleFrameValue::Scalar(TemporalFrameValue::Builtin(
        BuiltinFrameValue::Text(crate::RlCheckpointText::from_utf8("a")),
    ))])
    .unwrap();
    let numeric_object =
        temporal_frame_array(&[TemporalFrameValue::Builtin(BuiltinFrameValue::Int(9))]).unwrap();
    for (categories, codes) in [
        (integers.clone(), vec![0, -1]),
        (text.clone(), vec![0, -1]),
        (integers.clone(), vec![]),
        (text, vec![]),
    ] {
        let rows = codes.len();
        let descriptor = CategoricalIndexDescriptor::new(categories, codes, false).unwrap();
        let left = IndexedFrame::from_categorical_index(
            descriptor,
            Some("datetime".into()),
            block_tests::batch(vec![("x", ordinal(rows))], rows),
        )
        .unwrap();
        for right in [
            integers.clone(),
            Arc::new(arrow_array::Float64Array::from(vec![3.5])) as ArrayRef,
            numeric_object.clone(),
        ] {
            let other = block_tests::batch(
                vec![("datetime", right.clone()), ("x", ordinal(right.len()))],
                right.len(),
            );
            let before = (
                left.index().to_data(),
                left.index_metadata().clone(),
                other.clone(),
            );
            let mut baseline = Failure {
                at: usize::MAX,
                calls: 0,
            };
            let expected =
                append_with_operations(&left, &other, &mut baseline, &mut ignore_warning).unwrap();
            assert_eq!(expected.data().num_rows(), rows + right.len());
            for at in 1..=baseline.calls {
                let mut backend = Failure { at, calls: 0 };
                assert_eq!(
                    append_with_operations(&left, &other, &mut backend, &mut ignore_warning)
                        .unwrap_err()
                        .to_string(),
                    "Compute error: categorical backend failed"
                );
                assert_eq!(backend.calls, at);
                assert_eq!(
                    (
                        left.index().to_data(),
                        left.index_metadata().clone(),
                        other.clone()
                    ),
                    before
                );
            }
        }
    }
}

pub(super) fn descriptor(value: &Value) -> CategoricalIndexDescriptor {
    if let Some(masked) = value["categories"].get("masked") {
        return CategoricalIndexDescriptor::from_nullable_categories(
            NullableIndexDescriptor::new(nullable_index::tests::array(masked)).unwrap(),
            value["codes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect(),
            value["ordered"] == true,
        )
        .unwrap();
    }
    CategoricalIndexDescriptor::new(
        array(&value["categories"], false),
        value["codes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect(),
        value["ordered"] == true,
    )
    .unwrap()
}

fn frame(value: &Value, columns: bool) -> IndexedFrame {
    let rows = value["values"].as_array().unwrap().len();
    let data = block_tests::batch(
        if columns {
            vec![("x", ordinal(rows))]
        } else {
            vec![]
        },
        rows,
    );
    if value["kind"] == "CategoricalIndex" {
        IndexedFrame::from_categorical_index(descriptor(value), Some("datetime".into()), data)
            .unwrap()
    } else {
        IndexedFrame::new(array(value, false), Some("datetime".into()), data).unwrap()
    }
}

pub(super) fn other(value: &Value, columns: bool) -> RecordBatch {
    let (field, values) = if value["kind"] == "CategoricalIndex" {
        let descriptor = descriptor(value);
        (descriptor.field("datetime"), descriptor.storage())
    } else {
        let values = array(value, false);
        (
            Field::new("datetime", values.data_type().clone(), true),
            values,
        )
    };
    let rows = values.len();
    let mut fields = vec![field];
    let mut arrays = vec![values];
    if columns {
        let values = ordinal(rows);
        fields.push(Field::new("x", values.data_type().clone(), true));
        arrays.push(values);
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap()
}

pub(super) fn check(result: &IndexedFrame, expected: &Value) {
    let index = &expected["index"];
    if index["kind"] == "CategoricalIndex" {
        let IndexMetadata::Categorical(actual) = result.index_metadata() else {
            panic!("categorical metadata lost")
        };
        let wanted = descriptor(index);
        assert_eq!(actual.codes(), wanted.codes());
        assert_eq!(actual.ordered(), wanted.ordered());
        assert_eq!(actual.categories().to_data(), wanted.categories().to_data());
        compare(&actual.values(), index["values"].as_array().unwrap());
    } else {
        assert_eq!(result.index_metadata(), &IndexMetadata::Array);
        let wanted = array(index, false);
        assert!(
            result.index().data_type() == wanted.data_type()
                || (index["dtype"] == "object"
                    && (blocks::logical_object(result.index().data_type())
                        || result.index().data_type() == &tuple_frame_dtype())),
            "actual {:?}, wanted {:?}",
            result.index().data_type(),
            wanted.data_type()
        );
        compare(
            &tuple_index::values(result.index()).unwrap(),
            index["values"].as_array().unwrap(),
        );
    }
    assert_eq!(result.index_name(), Some("datetime"));
    let data = &expected["frame"];
    assert_eq!(
        result.data().num_rows(),
        index["values"].as_array().unwrap().len()
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
        assert!(blocks::logical_type_eq(
            column.data_type(),
            wanted.data_type()
        ));
        if blocks::logical_object(column.data_type()) {
            temporal_objects::tests::compare_cells(
                &temporal_cast::values(column).unwrap(),
                data["values"][i].as_array().unwrap(),
            );
        } else {
            block_tests::compare_values(column, &data["values"][i]);
        }
    }
}

#[test]
fn categorical_continuous_history_matches_actual_source() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/dataframe_categorical_chains.py"))
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
        "4c68c0eb03b0f6f4e86b1f9b10d4e317091c2ad2fc917dc7694884648a004e9a"
    );
    let chains = contract["chains"].as_array().unwrap();
    assert_eq!(chains.len(), 370);
    for chain in chains {
        let mut current = IndexedFrame::new(
            tuple_frame_array(&[]).unwrap(),
            Some("datetime".into()),
            block_tests::batch(vec![("x", builtin_frame_array(&[]).unwrap())], 0),
        )
        .unwrap();
        for stage in ["first", "second"] {
            let expected = &chain[format!("{stage}_output")];
            let right = other(&contract["inputs"][chain[stage].as_str().unwrap()], true);
            let before = (
                current.index().to_data(),
                current.index_metadata().clone(),
                current.data().clone(),
                right.clone(),
            );
            let mut warnings = vec![];
            let result = dataframe_append_with_warnings(&current, &right, &mut |message| {
                warnings.push(json!(["FutureWarning", message]));
            });
            assert_eq!(json!(warnings), expected["warnings"], "{chain}");
            assert_eq!(
                (
                    current.index().to_data(),
                    current.index_metadata().clone(),
                    current.data().clone(),
                    right
                ),
                before
            );
            if let Some(message) = expected.get("message") {
                assert_eq!(result.unwrap_err().to_string(), message.as_str().unwrap());
                break;
            }
            current = result.unwrap_or_else(|error| panic!("{chain}: {error}"));
            check(&current, &expected["output"]);
        }
    }
}

#[test]
fn categorical_directional_append_matches_complete_source_matrix() {
    run_contract(
        "dataframe_categorical_append.py",
        "3664562ff16b25ab4991959ee4058f9c21c74ea721984fb35ec3e6f7286bf72d",
        21904,
    );
}

#[test]
fn categorical_dtype_membership_and_missing_markers_match_source() {
    run_contract(
        "dataframe_categorical_membership.py",
        "02f8eab72c7836667b82ab6e8badf0fa356bf41b95324a51b5ad6eaa02f85ab6",
        1120,
    );
}

#[test]
fn categorical_empty_category_sets_match_source() {
    run_contract(
        "dataframe_categorical_empty.py",
        "fe91b72e304d8c409016dd7f709a767a9c337eb063afcf7eb4f50f332d0c0a7c",
        512,
    );
}

#[test]
fn categorical_temporal_units_and_object_categories_match_source() {
    run_contract(
        "dataframe_categorical_temporal.py",
        "ffbd153010ee99989d8cc82e7020e196fd488860e0804e5013982c9b40131465",
        3600,
    );
}

fn run_contract(fixture: &str, digest: &str, count: usize) {
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
    assert_eq!(cases.len(), count);
    for case in cases {
        let left = frame(
            &contract["inputs"][case["left"].as_str().unwrap()],
            case["left_columns"] == true,
        );
        let right = other(
            &contract["inputs"][case["right"].as_str().unwrap()],
            case["right_columns"] == true,
        );
        let before = (left.index().to_data(), right.clone());
        let mut warnings = vec![];
        let result = dataframe_append_with_warnings(&left, &right, &mut |message| {
            warnings.push(json!(["FutureWarning", message]));
        });
        assert_eq!(json!(warnings), case["warnings"], "{case}");
        if case.get("error").is_some() {
            assert_eq!(
                result.unwrap_err().to_string(),
                case["message"].as_str().unwrap(),
                "{case}"
            );
        } else {
            let result = result.unwrap_or_else(|error| panic!("{case}: {error}"));
            let comparison = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                check(&result, &case["output"]);
            }));
            assert!(comparison.is_ok(), "{case}");
        }
        assert_eq!((left.index().to_data(), right), before);
    }
}
