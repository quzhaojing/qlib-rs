use std::{path::PathBuf, process::Command, sync::Arc};

use arrow_array::{
    Array, ArrayRef, BooleanArray, Float16Array, Float32Array, Float64Array, Int64Array,
    RecordBatch, RecordBatchOptions, StringArray, TimestampNanosecondArray,
};
use arrow_schema::{DataType, Field, Schema};
use domain_core::{
    ValidEdge, ValidValueError, first_valid_value, first_valid_values, last_valid_value,
    last_valid_values, valid_value, valid_values,
};
use half::f16;
use serde_json::{Value, json};

fn float64_value(array: &ArrayRef) -> Option<f64> {
    let array = array
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("fixture is Float64");
    (!array.is_null(0)).then(|| array.value(0))
}

#[test]
fn arrays_select_each_edge_and_preserve_all_missing_values() {
    let floats = Float64Array::from(vec![
        None,
        Some(f64::NAN),
        Some(1.5),
        None,
        Some(2.5),
        Some(f64::NAN),
    ]);
    assert_eq!(
        float64_value(&first_valid_value(&floats).unwrap()),
        Some(1.5)
    );
    assert_eq!(
        float64_value(&last_valid_value(&floats).unwrap()),
        Some(2.5)
    );

    let all_null = Int64Array::from(vec![None, None]);
    let first = valid_value(&all_null, ValidEdge::First).unwrap();
    let last = valid_value(&all_null, ValidEdge::Last).unwrap();
    assert!(first.is_null(0));
    assert!(last.is_null(0));
    assert_eq!(first.data_type(), &DataType::Int64);
    assert_eq!(last.data_type(), &DataType::Int64);

    let all_nan = Float64Array::from(vec![f64::NAN, f64::NAN]);
    assert!(
        float64_value(&first_valid_value(&all_nan).unwrap())
            .unwrap()
            .is_nan()
    );
    assert!(
        float64_value(&last_valid_value(&all_nan).unwrap())
            .unwrap()
            .is_nan()
    );

    let empty = StringArray::from(Vec::<Option<&str>>::new());
    assert_eq!(first_valid_value(&empty), Err(ValidValueError::EmptyArray));
    assert_eq!(last_valid_value(&empty), Err(ValidValueError::EmptyArray));
}

#[test]
fn every_arrow_float_width_treats_nan_as_missing() {
    let f16s = Float16Array::from(vec![
        Some(f16::NAN),
        Some(f16::from_f32(1.25)),
        Some(f16::NAN),
    ]);
    let selected = first_valid_value(&f16s).unwrap();
    assert_eq!(
        selected
            .as_any()
            .downcast_ref::<Float16Array>()
            .unwrap()
            .value(0),
        f16::from_f32(1.25)
    );

    let f32s = Float32Array::from(vec![Some(f32::NAN), Some(1.25), Some(f32::NAN)]);
    let selected = last_valid_value(&f32s).unwrap();
    assert_eq!(
        selected
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .value(0)
            .to_bits(),
        1.25_f32.to_bits()
    );
}

fn mixed_batch() -> RecordBatch {
    RecordBatch::try_from_iter([
        (
            "float",
            Arc::new(Float64Array::from(vec![
                Some(f64::NAN),
                Some(1.0),
                None,
                Some(2.0),
                Some(f64::NAN),
            ])) as ArrayRef,
        ),
        (
            "integer",
            Arc::new(Int64Array::from(vec![None, Some(1), None, Some(2), None])) as ArrayRef,
        ),
        (
            "boolean",
            Arc::new(BooleanArray::from(vec![
                None,
                Some(false),
                None,
                Some(true),
                None,
            ])) as ArrayRef,
        ),
        (
            "string",
            Arc::new(StringArray::from(vec![
                None,
                Some("a"),
                None,
                Some("b"),
                None,
            ])) as ArrayRef,
        ),
        (
            "timestamp",
            Arc::new(TimestampNanosecondArray::from(vec![
                None,
                Some(10),
                None,
                Some(20),
                None,
            ])) as ArrayRef,
        ),
    ])
    .expect("fixture RecordBatch is valid")
}

#[test]
fn record_batches_select_columns_independently_and_preserve_schema() {
    let input = mixed_batch();
    for (
        output,
        expected_float,
        expected_integer,
        expected_boolean,
        expected_string,
        expected_time,
    ) in [
        (first_valid_values(&input), 1.0, 1, false, "a", 10),
        (last_valid_values(&input), 2.0, 2, true, "b", 20),
    ] {
        assert_eq!(output.num_rows(), 1);
        assert_eq!(output.schema(), input.schema());
        assert_eq!(float64_value(output.column(0)), Some(expected_float));
        assert_eq!(
            output
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0),
            expected_integer
        );
        assert_eq!(
            output
                .column(2)
                .as_any()
                .downcast_ref::<BooleanArray>()
                .unwrap()
                .value(0),
            expected_boolean
        );
        assert_eq!(
            output
                .column(3)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value(0),
            expected_string
        );
        assert_eq!(
            output
                .column(4)
                .as_any()
                .downcast_ref::<TimestampNanosecondArray>()
                .unwrap()
                .value(0),
            expected_time
        );
    }

    assert_eq!(
        valid_values(&input, ValidEdge::First),
        first_valid_values(&input)
    );
    assert_eq!(
        valid_values(&input, ValidEdge::Last),
        last_valid_values(&input)
    );
}

#[test]
fn empty_record_batch_shapes_are_returned_unchanged() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Float64,
        true,
    )]));
    let zero_rows = RecordBatch::new_empty(schema);
    assert_eq!(first_valid_values(&zero_rows), zero_rows);

    let options = RecordBatchOptions::new().with_row_count(Some(2));
    let zero_columns =
        RecordBatch::try_new_with_options(Arc::new(Schema::empty()), Vec::new(), &options)
            .expect("zero-column fixture is valid");
    assert_eq!(last_valid_values(&zero_columns), zero_columns);
}

#[test]
fn valid_value_contract_matches_live_python_source() {
    let source = std::env::var_os("QLIB_PYTHON_RESAM").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/utils/resam.py"),
        PathBuf::from,
    );
    assert!(
        source.is_file(),
        "Python source not found: {}",
        source.display()
    );
    let script = r#"
import ast, json, sys
import numpy as np
import pandas as pd
tree = ast.parse(open(sys.argv[1], encoding="utf-8").read(), filename=sys.argv[1])
body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"get_valid_value", "_ts_data_valid"}]
module = ast.Module(body=body, type_ignores=[])
ast.fix_missing_locations(module)
ns = {"pd": pd}
exec(compile(module, sys.argv[1], "exec"), ns)

series = {
    "float64": pd.Series([np.nan, 1.5, np.nan, 2.5, np.nan], dtype="float64"),
    "float32": pd.Series([np.nan, 1.25, np.nan], dtype="float32"),
    "integer": pd.Series([pd.NA, 1, pd.NA, 2, pd.NA], dtype="Int64"),
    "boolean": pd.Series([pd.NA, False, pd.NA, True, pd.NA], dtype="boolean"),
    "string": pd.Series([None, "a", None, "b", None], dtype="object"),
    "timestamp": pd.Series([pd.NaT, pd.Timestamp("1970-01-01 00:00:00.000000010"), pd.NaT,
                            pd.Timestamp("1970-01-01 00:00:00.000000020"), pd.NaT]),
}
result = {}
for name, values in series.items():
    result[name] = {
        "dtype": str(values.dtype),
        "first": str(ns["get_valid_value"](values, last=False)),
        "last": str(ns["get_valid_value"](values, last=True)),
    }
for name, values in {
    "all_nan": pd.Series([np.nan, np.nan], dtype="float64"),
    "all_null": pd.Series([pd.NA, pd.NA], dtype="Int64"),
}.items():
    result[name] = {
        "first": str(ns["get_valid_value"](values, last=False)),
        "last": str(ns["get_valid_value"](values, last=True)),
    }
try:
    ns["get_valid_value"](pd.Series([], dtype="float64"))
except Exception as error:
    result["empty_error"] = type(error).__name__
print(json.dumps(result, sort_keys=True))
"#;
    let output = Command::new(std::env::var_os("PYTHON").unwrap_or_else(|| "python".into()))
        .arg("-c")
        .arg(script)
        .arg(&source)
        .output()
        .expect("Python interpreter starts");
    assert!(
        output.status.success(),
        "Python snapshot failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let actual: Value = serde_json::from_slice(&output.stdout).expect("valid Python JSON");
    let expected = json!({
        "float64": {"dtype": "float64", "first": "1.5", "last": "2.5"},
        "float32": {"dtype": "float32", "first": "1.25", "last": "1.25"},
        "integer": {"dtype": "Int64", "first": "1", "last": "2"},
        "boolean": {"dtype": "boolean", "first": "False", "last": "True"},
        "string": {"dtype": "object", "first": "a", "last": "b"},
        "timestamp": {
            "dtype": "datetime64[ns]",
            "first": "1970-01-01 00:00:00.000000010",
            "last": "1970-01-01 00:00:00.000000020"
        },
        "all_nan": {"first": "nan", "last": "nan"},
        "all_null": {"first": "<NA>", "last": "<NA>"},
        "empty_error": "IndexError"
    });
    assert_eq!(actual, expected);
}
