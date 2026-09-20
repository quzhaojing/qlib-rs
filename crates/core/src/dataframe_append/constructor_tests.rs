use super::*;
use crate::dataframe_append::{self as append, IndexedFrame};
use arrow_array::{Array, Float32Array};
use serde_json::{Value, json};
use std::{path::PathBuf, process::Command};

fn text(value: &str) -> V {
    V::Text(crate::RlCheckpointText::from(value))
}

fn input(name: &str) -> FrameColumnInput {
    FrameColumnInput::Scalar(match name {
        "none" => V::None,
        "na" => V::PandasNa,
        "nat" => V::NotATime,
        "false" => V::Bool(false),
        "integer" => V::Int(-3),
        "unsigned" => V::UInt(u64::MAX),
        "float" => V::Float(1.25),
        "negative_zero" => V::Float(-0.),
        "nan" => V::Float(f64::NAN),
        "text" => text("中文"),
        "surrogate" => {
            V::Text(crate::RlCheckpointText::try_from_code_points([120, 0xd800]).unwrap())
        }
        _ => panic!("unexpected scalar"),
    })
}

fn expected_array(dtype: &str, values: &Value) -> ArrayRef {
    let values = values.as_array().unwrap();
    match dtype {
        "bool" => Arc::new(BooleanArray::from(
            values
                .iter()
                .map(|v| v[1].as_bool().unwrap())
                .collect::<Vec<_>>(),
        )),
        "int64" => Arc::new(Int64Array::from(
            values
                .iter()
                .map(|v| v[1].as_str().unwrap().parse::<i64>().unwrap())
                .collect::<Vec<_>>(),
        )),
        "uint64" => Arc::new(UInt64Array::from(
            values
                .iter()
                .map(|v| v[1].as_str().unwrap().parse::<u64>().unwrap())
                .collect::<Vec<_>>(),
        )),
        "float64" => Arc::new(Float64Array::from(
            values
                .iter()
                .map(|v| append::block_tests::expected_float(v[1].as_str().unwrap()))
                .collect::<Vec<_>>(),
        )),
        "object" => {
            let values = values
                .iter()
                .map(|v| match v[0].as_str().unwrap() {
                    "none" => V::None,
                    "pd.NA" => V::PandasNa,
                    "str" => V::Text(
                        crate::RlCheckpointText::try_from_code_points(
                            v[1].as_array()
                                .unwrap()
                                .iter()
                                .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap()),
                        )
                        .unwrap(),
                    ),
                    _ => panic!("unexpected object"),
                })
                .collect::<Vec<_>>();
            builtin_frame_array(&values).unwrap()
        }
        _ => {
            let kind = if dtype.starts_with("timedelta") {
                DataType::Duration(TimeUnit::Nanosecond)
            } else {
                DataType::Timestamp(
                    if dtype.contains("[ns") {
                        TimeUnit::Nanosecond
                    } else {
                        TimeUnit::Second
                    },
                    dtype.contains("UTC").then(|| "UTC".into()),
                )
            };
            let array = Arc::new(Int64Array::from(
                values
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
            arrow_cast::cast(&array, &kind).unwrap()
        }
    }
}

fn check_batch(batch: &RecordBatch, snapshot: &Value) {
    assert_eq!(
        batch.num_columns(),
        snapshot["dtypes"].as_array().unwrap().len()
    );
    for (i, array) in batch.columns().iter().enumerate() {
        let expected = expected_array(
            snapshot["dtypes"][i].as_str().unwrap(),
            &snapshot["values"][i],
        );
        assert_eq!(array.to_data(), expected.to_data(), "{snapshot}");
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
fn scalar_column_maps_match_actual_construction_and_empty_frame_append() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/dataframe_constructor_contract.py"))
        .arg(root.join("../../../qlib/qlib/rl/order_execution/utils.py"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let contract: Value = serde_json::from_slice(&output.stdout).unwrap();
    let cases = contract["cases"].as_array().unwrap();
    let empty = IndexedFrame::new(
        Arc::new(Int64Array::from(Vec::<i64>::new())),
        None,
        frame_from_columns(&IndexMap::new()).unwrap(),
    )
    .unwrap();
    for name in [
        "none",
        "na",
        "nat",
        "false",
        "integer",
        "unsigned",
        "float",
        "negative_zero",
        "nan",
        "text",
        "surrogate",
        "timestamp",
        "utc",
        "timedelta",
    ] {
        for rows in [0, 1, 3] {
            let key = format!("scalar/{name}/{rows}");
            let case = cases
                .iter()
                .find(|c| c["name"] == key && c["left"] == "empty")
                .unwrap();
            let c = &case["construction"];
            let dates = Arc::new(arrow_array::TimestampNanosecondArray::from(
                (0..rows)
                    .map(|i| 1_704_153_600_000_000_000_i64 + i * 86_400_000_000_000)
                    .collect::<Vec<_>>(),
            )) as ArrayRef;
            let scalar = match name {
                "timestamp" => FrameColumnInput::TypedScalar(Arc::new(
                    arrow_array::TimestampSecondArray::from(vec![1_704_153_600]),
                )),
                "utc" => FrameColumnInput::TypedScalar(Arc::new(
                    arrow_array::TimestampSecondArray::from(vec![1_704_153_600])
                        .with_timezone("UTC"),
                )),
                "timedelta" => FrameColumnInput::TypedScalar(Arc::new(
                    arrow_array::DurationNanosecondArray::from(vec![3_600_000_000_000]),
                )),
                _ => input(name),
            };
            let columns = IndexMap::from([
                ("datetime".into(), FrameColumnInput::Array(dates.clone())),
                ("x".into(), scalar),
                (
                    "stock_id".into(),
                    FrameColumnInput::Scalar(text("SH600000")),
                ),
            ]);
            let built = frame_from_columns(&columns).unwrap();
            assert!(Arc::ptr_eq(built.column(0), &dates));
            check_batch(&built, c);
            let mut warnings = vec![];
            let result = append::dataframe_append_with_warnings(&empty, &built, &mut |message| {
                warnings.push(json!(["FutureWarning", message]));
            })
            .unwrap();
            check_batch(result.data(), &case["output"]);
            let groups = result
                .blocks()
                .iter()
                .map(|group| json!([case["output"]["dtypes"][group[0]], group]))
                .collect::<Vec<_>>();
            assert_eq!(json!(groups), case["output"]["blocks"]);
            assert_eq!(result.index().to_data(), dates.to_data());
            assert_eq!(result.index_name(), Some("datetime"));
            assert_eq!(json!(warnings), case["warnings"]);
        }
    }
}

#[test]
fn map_shape_validation_and_integer_inference_preserve_inputs() {
    let vector = Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef;
    let mut columns = IndexMap::from([("a".into(), FrameColumnInput::Scalar(V::UInt(3)))]);
    assert_eq!(
        frame_from_columns(&columns).unwrap_err().to_string(),
        "If using all scalar values, you must pass an index"
    );
    columns.insert("b".into(), FrameColumnInput::Array(vector.clone()));
    let built = frame_from_columns(&columns).unwrap();
    assert_eq!(built.column(0).data_type(), &DataType::Int64);
    assert!(Arc::ptr_eq(built.column(1), &vector));
    columns.insert(
        "c".into(),
        FrameColumnInput::Array(Arc::new(Int64Array::from(vec![1]))),
    );
    assert_eq!(
        frame_from_columns(&columns).unwrap_err().to_string(),
        "All arrays must be of the same length"
    );
    columns.shift_remove("c");
    for size in [0, 2] {
        columns.insert(
            "c".into(),
            FrameColumnInput::TypedScalar(Arc::new(Float32Array::from(vec![1.; size]))),
        );
        let error = frame_from_columns(&columns).unwrap_err();
        assert_eq!(
            error.to_string(),
            "typed scalar must contain exactly one element"
        );
        assert!(format!("{error:?}").contains("ScalarLength"));
    }
    columns.insert(
        "c".into(),
        FrameColumnInput::TypedScalar(Arc::new(Float32Array::from(vec![1.]))),
    );
    let built = frame_from_columns(&columns).unwrap();
    assert_eq!(built.column(2).data_type(), &DataType::Float32);
    assert_eq!(built.num_rows(), 2);
}

struct FailAt {
    fail: usize,
    calls: usize,
}
impl FailAt {
    fn record(&mut self) -> Result<(), ArrowError> {
        let call = self.calls;
        self.calls += 1;
        if call == self.fail {
            Err(ArrowError::ComputeError(
                "constructor operation failed".into(),
            ))
        } else {
            Ok(())
        }
    }
}
impl ConstructorOperations for FailAt {
    fn infer_temporal(&mut self, values: &[T]) -> Result<ArrayRef, ArrowError> {
        self.record()?;
        ArrowConstructor.infer_temporal(values)
    }
    fn infer(&mut self, values: &[V]) -> Result<ArrayRef, ArrowError> {
        self.record()?;
        ArrowConstructor.infer(values)
    }
    fn scalar(&mut self, value: &V) -> Result<ArrayRef, ArrowError> {
        self.record()?;
        ArrowConstructor.scalar(value)
    }
    fn repeat(&mut self, value: &ArrayRef, rows: usize) -> Result<ArrayRef, ArrowError> {
        self.record()?;
        ArrowConstructor.repeat(value, rows)
    }
    fn batch(
        &mut self,
        fields: Vec<Field>,
        values: Vec<ArrayRef>,
        rows: usize,
    ) -> Result<RecordBatch, ArrowError> {
        self.record()?;
        ArrowConstructor.batch(fields, values, rows)
    }
}

#[test]
fn constructor_errors_never_publish_partial_batches() {
    let vector = Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef;
    let columns = IndexMap::from([
        ("a".into(), FrameColumnInput::Array(vector.clone())),
        ("b".into(), FrameColumnInput::Scalar(V::None)),
        (
            "c".into(),
            FrameColumnInput::TypedScalar(vector.slice(0, 1)),
        ),
    ]);
    let mut columns = columns;
    columns.insert(
        "d".into(),
        FrameColumnInput::Untyped(vec![V::None, V::None]),
    );
    for fail in 0..5 {
        let mut operations = FailAt { fail, calls: 0 };
        let error = construct(&columns, &mut operations).unwrap_err();
        assert!(matches!(error, FrameConstructionError::Arrow(_)));
        assert_eq!(
            error.to_string(),
            "Compute error: constructor operation failed"
        );
        assert_eq!(operations.calls, fail + 1);
        assert_eq!(
            vector
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .as_ref(),
            &[1, 2]
        );
    }
}

#[test]
fn record_inference_and_batch_errors_preserve_the_input() {
    let records = vec![IndexMap::from([
        ("a".into(), V::Int(7)),
        ("b".into(), V::None),
    ])];
    let before = format!("{records:?}");
    for fail in 0..3 {
        let mut operations = FailAt { fail, calls: 0 };
        let error = construct_records(
            &records,
            V::Float(f64::NAN),
            |ops, values| ops.infer(values),
            &mut operations,
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "Compute error: constructor operation failed"
        );
        assert_eq!(operations.calls, fail + 1);
        assert_eq!(format!("{records:?}"), before);
    }
    for size in 0..3 {
        let empty_records = vec![IndexMap::new(); size];
        let batch = frame_from_builtin_records(&empty_records).unwrap();
        assert_eq!(batch.num_columns(), 0);
        assert_eq!(batch.num_rows(), size);
    }
}

#[test]
fn temporal_construction_errors_and_empty_record_rows_are_preserved() {
    let value = T::Timestamp {
        ticks: 1,
        unit: TimeUnit::Second,
        timezone: None,
    };
    let records = vec![IndexMap::from([
        ("datetime".into(), value.clone()),
        ("x".into(), T::Builtin(V::Int(7))),
    ])];
    let before = format!("{records:?}");
    for fail in 0..3 {
        let mut ops = FailAt { fail, calls: 0 };
        let error = construct_records(
            &records,
            T::Builtin(V::Float(f64::NAN)),
            |ops, values| ops.infer_temporal(values),
            &mut ops,
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "Compute error: constructor operation failed"
        );
        assert_eq!(ops.calls, fail + 1);
        assert_eq!(format!("{records:?}"), before);
    }
    let mut columns = IndexMap::from([("x".into(), FrameColumnInput::Temporal(vec![value]))]);
    let error = construct(&columns, &mut FailAt { fail: 0, calls: 0 }).unwrap_err();
    assert_eq!(
        error.to_string(),
        "Compute error: constructor operation failed"
    );
    columns.insert("y".into(), FrameColumnInput::Untyped(vec![]));
    assert_eq!(
        frame_from_columns(&columns).unwrap_err().to_string(),
        "All arrays must be of the same length"
    );
    for size in 0..3 {
        let batch = frame_from_temporal_records(&vec![IndexMap::new(); size]).unwrap();
        assert_eq!(batch.num_rows(), size);
        assert_eq!(batch.num_columns(), 0);
    }
}
