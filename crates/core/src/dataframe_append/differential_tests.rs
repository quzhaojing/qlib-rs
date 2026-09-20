use std::{path::PathBuf, process::Command, sync::Arc};

use super::{DataframeAppendError, EmptyColumnAxis, IndexedFrame, dataframe_append};
use arrow_array::{
    Array, ArrayRef, Float64Array, Int64Array, RecordBatch, RecordBatchOptions,
    TimestampNanosecondArray,
};
use arrow_schema::{DataType, Field, Schema};
use serde::Deserialize;
use serde_json::{Value, json};

fn batch(columns: Vec<(&str, ArrayRef)>, rows: usize) -> RecordBatch {
    let fields: Vec<_> = columns
        .iter()
        .map(|(name, a)| Field::new(*name, a.data_type().clone(), true))
        .collect();
    RecordBatch::try_new_with_options(
        Arc::new(Schema::new(fields)),
        columns.into_iter().map(|(_, a)| a).collect(),
        &RecordBatchOptions::new().with_row_count(Some(rows)),
    )
    .unwrap()
}

fn integers(values: Vec<i64>) -> ArrayRef {
    Arc::new(Int64Array::from(values))
}

fn index(values: Vec<i64>, kind: &str) -> ArrayRef {
    match kind {
        "int64" => integers(values),
        "datetime64[ns]" => Arc::new(TimestampNanosecondArray::from(values)),
        "datetime64[ns, UTC]" => {
            Arc::new(TimestampNanosecondArray::from(values).with_timezone("UTC"))
        }
        _ => panic!("unexpected index {kind}"),
    }
}

fn data(columns: &[(String, String)], rows: usize) -> Vec<(&str, ArrayRef)> {
    columns
        .iter()
        .enumerate()
        .map(|(position, (name, dtype))| {
            let ints: Vec<_> = (1..=i64::try_from(rows).unwrap())
                .map(|i| i + 10 * i64::try_from(position).unwrap())
                .collect();
            let array = if dtype == "int64" {
                integers(ints)
            } else {
                Arc::new(Float64Array::from(
                    ints.into_iter()
                        .map(|i| i32::try_from(i).map(f64::from).unwrap())
                        .collect::<Vec<_>>(),
                )) as ArrayRef
            };
            (name.as_str(), array)
        })
        .collect()
}

fn values(array: &ArrayRef) -> Vec<Value> {
    (0..array.len())
        .map(|i| {
            if array.is_null(i) {
                return Value::Null;
            }
            match array.data_type() {
                DataType::Timestamp(_, _) => json!(
                    array
                        .as_any()
                        .downcast_ref::<TimestampNanosecondArray>()
                        .unwrap()
                        .value(i)
                ),
                DataType::Int64 => json!(
                    array
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap()
                        .value(i)
                ),
                DataType::Float64 => json!(
                    array
                        .as_any()
                        .downcast_ref::<Float64Array>()
                        .unwrap()
                        .value(i)
                ),
                other => panic!("unexpected dtype {other}"),
            }
        })
        .collect()
}

fn snapshot(frame: &IndexedFrame) -> Value {
    let columns: Vec<_> = frame
        .data()
        .schema_ref()
        .fields()
        .iter()
        .zip(frame.data().columns())
        .map(|(f, a)| {
            let dtype = match a.data_type() {
                DataType::Int64 => "int64",
                DataType::Float64 => "float64",
                other => panic!("unexpected {other}"),
            };
            json!([f.name(), dtype, values(a)])
        })
        .collect();
    let index_kind = match frame.index().data_type() {
        DataType::Int64 => "int64",
        DataType::Timestamp(_, None) => "datetime64[ns]",
        DataType::Timestamp(_, Some(_)) => "datetime64[ns, UTC]",
        other => panic!("unexpected index dtype {other}"),
    };
    json!({"index": values(frame.index()), "index_kind":index_kind, "name":frame.index_name(), "columns":columns,"column_axis":format!("{:?}",frame.column_axis())})
}

#[derive(Debug, Deserialize)]
struct Case {
    left: Vec<(String, String)>,
    right: Vec<(String, String)>,
    n: usize,
    m: usize,
    name: Option<String>,
    index_kind: String,
    output: Option<Value>,
    error: Option<String>,
    explicit_empty_columns: bool,
}

#[test]
fn actual_pandas_alignment() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/dataframe_append.py"))
        .arg(root.join("../../../qlib/qlib/rl/order_execution/utils.py"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Vec<Case> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(cases.len(), 9720);
    for case in cases {
        let frame = IndexedFrame::new(
            index(
                (0..case.n).map(|i| 3 - i64::try_from(i).unwrap()).collect(),
                &case.index_kind,
            ),
            case.name.clone(),
            batch(data(&case.left, case.n), case.n),
        )
        .unwrap()
        .with_empty_column_axis(if case.explicit_empty_columns {
            EmptyColumnAxis::ObjectIndex
        } else {
            EmptyColumnAxis::RangeIndex
        });
        let before = snapshot(&frame);
        let mut columns = data(&case.right, case.m);
        columns.insert(
            case.m % (case.right.len() + 1),
            ("datetime", index(vec![3; case.m], &case.index_kind)),
        );
        let other = batch(columns, case.m);
        let actual = dataframe_append(&frame, &other);
        if let Some(error) = &case.error {
            let actual = actual.unwrap_err();
            assert!(matches!(actual, DataframeAppendError::DuplicateColumns));
            assert_eq!(actual.to_string(), *error, "{case:?}");
        } else {
            assert_eq!(
                Some(snapshot(
                    &actual.unwrap_or_else(|error| panic!("{case:?}: {error}"))
                )),
                case.output,
                "{case:?}"
            );
        }
        assert_eq!(snapshot(&frame), before);
    }
}

#[test]
fn native_validation_boundaries() {
    assert!(matches!(
        IndexedFrame::new(integers(vec![1]), None, batch(vec![], 0)),
        Err(DataframeAppendError::IndexLength)
    ));
    let frame = IndexedFrame::new(integers(vec![]), None, batch(vec![], 0)).unwrap();
    assert!(matches!(
        dataframe_append(&frame, &batch(vec![], 0)),
        Err(DataframeAppendError::MissingDatetime)
    ));
    assert!(matches!(
        dataframe_append(
            &frame,
            &batch(
                vec![
                    ("datetime", integers(vec![])),
                    ("datetime", integers(vec![]))
                ],
                0
            )
        ),
        Err(DataframeAppendError::DuplicateDatetime)
    ));
    let duplicate = batch(vec![("a", integers(vec![])), ("a", integers(vec![]))], 0);
    let duplicate_frame = IndexedFrame::new(integers(vec![]), None, duplicate).unwrap();
    let other = batch(
        vec![("datetime", integers(vec![])), ("a", integers(vec![]))],
        0,
    );
    assert!(matches!(
        dataframe_append(&duplicate_frame, &other),
        Err(DataframeAppendError::DuplicateColumns)
    ));
    let duplicate_other = batch(
        vec![
            ("datetime", integers(vec![])),
            ("a", integers(vec![])),
            ("a", integers(vec![])),
        ],
        0,
    );
    let unique_frame = IndexedFrame::new(
        integers(vec![]),
        None,
        batch(vec![("a", integers(vec![]))], 0),
    )
    .unwrap();
    assert!(matches!(
        dataframe_append(&unique_frame, &duplicate_other),
        Err(DataframeAppendError::DuplicateColumns)
    ));
}

#[test]
fn native_arrow_dtype_and_index_boundaries() {
    use arrow_array::{NullArray, StringArray};
    let make = |array: ArrayRef| {
        IndexedFrame::new(
            integers(vec![1]),
            Some("datetime".into()),
            batch(vec![("a", array)], 1),
        )
        .unwrap()
    };
    let append = |array: ArrayRef| batch(vec![("datetime", integers(vec![1])), ("a", array)], 1);
    let nulls = Arc::new(NullArray::new(1)) as ArrayRef;
    for (left, right) in [
        (nulls.clone(), integers(vec![2])),
        (integers(vec![2]), nulls.clone()),
        (nulls.clone(), nulls),
    ] {
        let result = dataframe_append(&make(left.clone()), &append(right.clone())).unwrap();
        let array = result.data().column(0);
        assert_eq!(array.len(), 2);
        let missing = |a: &ArrayRef, i| a.data_type() == &DataType::Null || a.is_null(i);
        assert_eq!(missing(array, 0), missing(&left, 0));
        assert_eq!(missing(array, 1), missing(&right, 0));
    }
    let strings = Arc::new(StringArray::from(vec!["x"])) as ArrayRef;
    let error = dataframe_append(&make(strings.clone()), &append(integers(vec![2]))).unwrap_err();
    assert!(matches!(
        error,
        DataframeAppendError::DtypeAdapter(DataType::Utf8, DataType::Int64)
    ));
    assert!(error.to_string().contains("dtype adapter"));
    let frame = IndexedFrame::new(strings, None, batch(vec![("a", integers(vec![1]))], 1)).unwrap();
    assert!(matches!(
        dataframe_append(&frame, &append(integers(vec![2]))),
        Err(DataframeAppendError::DtypeAdapter(_, _))
    ));
    let error: DataframeAppendError =
        arrow_schema::ArrowError::ComputeError("concat failure".into()).into();
    assert!(error.to_string().contains("concat failure"));
    assert!(!format!("{frame:?}").is_empty());
}

#[test]
fn chained_append_retains_empty_axis_semantics() {
    let frame = IndexedFrame::new(integers(vec![1]), None, batch(vec![], 1)).unwrap();
    let repeated = batch(
        vec![
            ("datetime", integers(vec![2])),
            ("a", integers(vec![3])),
            ("a", integers(vec![4])),
        ],
        1,
    );
    let unchanged =
        dataframe_append(&frame, &batch(vec![("datetime", integers(vec![]))], 0)).unwrap();
    assert_eq!(unchanged.column_axis(), EmptyColumnAxis::RangeIndex);
    let result = dataframe_append(&unchanged, &repeated).unwrap();
    assert_eq!(
        values(result.data().column(0)),
        vec![Value::Null, json!(3.0)]
    );
    assert_eq!(
        values(result.data().column(1)),
        vec![Value::Null, json!(4.0)]
    );
    let changed =
        dataframe_append(&frame, &batch(vec![("datetime", integers(vec![5]))], 1)).unwrap();
    assert_eq!(changed.column_axis(), EmptyColumnAxis::ObjectIndex);
    assert!(matches!(
        dataframe_append(&changed, &repeated),
        Err(DataframeAppendError::DuplicateColumns)
    ));
}
