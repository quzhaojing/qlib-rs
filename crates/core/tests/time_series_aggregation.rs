use std::{collections::BTreeMap, path::PathBuf, process::Command, sync::Arc};

use arrow_array::{
    Array, ArrayRef, BooleanArray, Float16Array, Float32Array, Float64Array, Int8Array, Int16Array,
    Int32Array, Int64Array, RecordBatch, StringArray, TimestampNanosecondArray, UInt8Array,
    UInt16Array, UInt32Array, UInt64Array,
};
use arrow_schema::DataType;
use chrono::NaiveDateTime;
use domain_core::{
    BuiltInAggregation, TimeRange, TimeSeriesAggregationError, TimeSeriesIndex,
    TimeSeriesIndexOrder, TimeSeriesSelectionError, aggregate_time_series,
    aggregate_time_series_str, aggregate_time_series_with_arguments,
};
use half::f16;
use serde_json::{Value, json};

fn timestamp(day: u32) -> i64 {
    NaiveDateTime::parse_from_str(&format!("2024-01-{day:02} 00:00:00"), "%Y-%m-%d %H:%M:%S")
        .unwrap()
        .and_utc()
        .timestamp_nanos_opt()
        .unwrap()
}

fn batch(columns: Vec<(&str, ArrayRef)>) -> RecordBatch {
    RecordBatch::try_from_iter(columns).unwrap()
}

fn columns(input: &RecordBatch, indices: &[usize]) -> RecordBatch {
    let schema = input.schema_ref();
    batch(
        indices
            .iter()
            .map(|index| {
                (
                    schema.field(*index).name().as_str(),
                    input.column(*index).clone(),
                )
            })
            .collect(),
    )
}

fn single_index() -> TimeSeriesIndex {
    TimeSeriesIndex::Datetime {
        datetime: "datetime".to_owned(),
    }
}

fn multi_index(order: TimeSeriesIndexOrder) -> TimeSeriesIndex {
    TimeSeriesIndex::InstrumentDatetime {
        instrument: "instrument".to_owned(),
        datetime: "datetime".to_owned(),
        order,
    }
}

fn aggregate(
    input: &RecordBatch,
    index: &TimeSeriesIndex,
    method: BuiltInAggregation,
) -> RecordBatch {
    aggregate_time_series(input, index, TimeRange::default(), method)
        .unwrap()
        .unwrap()
}

fn value<'a, T: Array + 'static>(batch: &'a RecordBatch, name: &str) -> &'a T {
    batch
        .column_by_name(name)
        .unwrap()
        .as_any()
        .downcast_ref()
        .unwrap()
}

#[allow(
    clippy::needless_pass_by_value,
    reason = "owned inline maps keep method-kwargs test cases compact"
)]
fn aggregate_with(
    input: &RecordBatch,
    index: &TimeSeriesIndex,
    method: BuiltInAggregation,
    arguments: BTreeMap<String, Value>,
) -> Result<Option<RecordBatch>, TimeSeriesAggregationError> {
    aggregate_time_series_with_arguments(input, index, TimeRange::default(), method, &arguments)
}

#[test]
#[allow(
    clippy::float_cmp,
    reason = "fixtures use exactly representable integer-valued IEEE results"
)]
fn single_index_default_reductions_match_pandas_missing_and_identity_rules() {
    let input = batch(vec![
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(vec![
                timestamp(3),
                timestamp(1),
                timestamp(2),
            ])),
        ),
        (
            "float",
            Arc::new(Float64Array::from(vec![Some(0.0), None, Some(2.0)])),
        ),
        (
            "bool",
            Arc::new(BooleanArray::from(vec![Some(true), None, Some(true)])),
        ),
    ]);

    let sum = aggregate(&input, &single_index(), BuiltInAggregation::Sum);
    assert_eq!(value::<Float64Array>(&sum, "float").value(0), 2.0);
    assert_eq!(value::<Int64Array>(&sum, "bool").value(0), 2);
    let mean = aggregate(&input, &single_index(), BuiltInAggregation::Mean);
    assert_eq!(value::<Float64Array>(&mean, "float").value(0), 1.0);
    assert_eq!(value::<Float64Array>(&mean, "bool").value(0), 1.0);
    let product = aggregate(&input, &single_index(), BuiltInAggregation::Product);
    assert_eq!(value::<Float64Array>(&product, "float").value(0), 0.0);
    assert_eq!(value::<Int64Array>(&product, "bool").value(0), 1);
    let all = aggregate(&input, &single_index(), BuiltInAggregation::All);
    assert!(!value::<BooleanArray>(&all, "float").value(0));
    assert!(value::<BooleanArray>(&all, "bool").value(0));

    let missing = batch(vec![
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(vec![timestamp(1)])),
        ),
        ("value", Arc::new(Float64Array::from(vec![None]))),
        ("value16", Arc::new(Float16Array::from(vec![None]))),
        ("value32", Arc::new(Float32Array::from(vec![None]))),
    ]);
    assert_eq!(
        value::<Float64Array>(
            &aggregate(&missing, &single_index(), BuiltInAggregation::Sum),
            "value"
        )
        .value(0),
        0.0
    );
    assert_eq!(
        value::<Float64Array>(
            &aggregate(&missing, &single_index(), BuiltInAggregation::Product),
            "value"
        )
        .value(0),
        1.0
    );
    assert_eq!(
        value::<Float16Array>(
            &aggregate(&missing, &single_index(), BuiltInAggregation::Product),
            "value16"
        )
        .value(0),
        f16::from_f32(1.0)
    );
    assert!(
        value::<Float64Array>(
            &aggregate(&missing, &single_index(), BuiltInAggregation::Mean),
            "value"
        )
        .is_null(0)
    );
    let missing_mean = aggregate(&missing, &single_index(), BuiltInAggregation::Mean);
    assert!(value::<Float16Array>(&missing_mean, "value16").is_null(0));
    assert!(value::<Float32Array>(&missing_mean, "value32").is_null(0));
    assert!(
        value::<BooleanArray>(
            &aggregate(&missing, &single_index(), BuiltInAggregation::All),
            "value"
        )
        .value(0)
    );
}

#[test]
fn multi_index_groups_sort_drop_null_keys_and_apply_every_method() {
    let input = batch(vec![
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(vec![
                timestamp(2),
                timestamp(2),
                timestamp(1),
                timestamp(1),
                timestamp(1),
            ])),
        ),
        (
            "instrument",
            Arc::new(StringArray::from(vec![
                Some("B"),
                Some("A"),
                Some("A"),
                Some("B"),
                None,
            ])),
        ),
        (
            "a",
            Arc::new(Float64Array::from(vec![
                Some(2.0),
                Some(2.0),
                Some(1.0),
                Some(4.0),
                Some(99.0),
            ])),
        ),
        (
            "b",
            Arc::new(BooleanArray::from(vec![
                Some(true),
                Some(false),
                Some(true),
                Some(true),
                Some(false),
            ])),
        ),
    ]);
    let index = multi_index(TimeSeriesIndexOrder::DatetimeInstrument);

    let sum = aggregate(&input, &index, BuiltInAggregation::Sum);
    assert_eq!(
        value::<StringArray>(&sum, "instrument")
            .iter()
            .collect::<Vec<_>>(),
        [Some("A"), Some("B")]
    );
    assert_eq!(value::<Float64Array>(&sum, "a").values(), &[3.0, 6.0]);
    assert_eq!(value::<Int64Array>(&sum, "b").values(), &[1, 2]);
    let mean = aggregate(&input, &index, BuiltInAggregation::Mean);
    assert_eq!(value::<Float64Array>(&mean, "a").values(), &[1.5, 3.0]);
    let product = aggregate(&input, &index, BuiltInAggregation::Product);
    assert_eq!(value::<Float64Array>(&product, "a").values(), &[2.0, 8.0]);
    let all = aggregate(&input, &index, BuiltInAggregation::All);
    assert_eq!(
        value::<BooleanArray>(&all, "b").iter().collect::<Vec<_>>(),
        [Some(false), Some(true)]
    );
    let first = aggregate(&input, &index, BuiltInAggregation::First);
    assert_eq!(value::<Float64Array>(&first, "a").values(), &[1.0, 4.0]);
    let last = aggregate(&input, &index, BuiltInAggregation::Last);
    assert_eq!(value::<Float64Array>(&last, "a").values(), &[2.0, 2.0]);
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "all Arrow numeric widths share one contract matrix"
)]
fn every_numeric_width_preserves_pandas_scalar_result_types() {
    let input = batch(vec![
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(vec![
                timestamp(1),
                timestamp(2),
            ])),
        ),
        ("i8", Arc::new(Int8Array::from(vec![1, 2]))),
        ("i16", Arc::new(Int16Array::from(vec![1, 2]))),
        ("i32", Arc::new(Int32Array::from(vec![1, 2]))),
        ("i64", Arc::new(Int64Array::from(vec![1, 2]))),
        ("u8", Arc::new(UInt8Array::from(vec![1, 2]))),
        ("u16", Arc::new(UInt16Array::from(vec![1, 2]))),
        ("u32", Arc::new(UInt32Array::from(vec![1, 2]))),
        ("u64", Arc::new(UInt64Array::from(vec![1, 2]))),
        (
            "f16",
            Arc::new(Float16Array::from(vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
            ])),
        ),
        ("f32", Arc::new(Float32Array::from(vec![1.0, 2.0]))),
        ("f64", Arc::new(Float64Array::from(vec![1.0, 2.0]))),
        ("bool", Arc::new(BooleanArray::from(vec![true, true]))),
    ]);
    for method in [BuiltInAggregation::Sum, BuiltInAggregation::Product] {
        let result = aggregate(&input, &single_index(), method);
        for name in ["i8", "i16", "i32", "i64", "bool"] {
            assert_eq!(
                result.column_by_name(name).unwrap().data_type(),
                &DataType::Int64
            );
        }
        for name in ["u8", "u16", "u32", "u64"] {
            assert_eq!(
                result.column_by_name(name).unwrap().data_type(),
                &DataType::UInt64
            );
        }
        assert_eq!(
            result.column_by_name("f16").unwrap().data_type(),
            &DataType::Float16
        );
        assert_eq!(
            result.column_by_name("f32").unwrap().data_type(),
            &DataType::Float32
        );
        assert_eq!(
            result.column_by_name("f64").unwrap().data_type(),
            &DataType::Float64
        );
    }
    let mean = aggregate(&input, &single_index(), BuiltInAggregation::Mean);
    for name in [
        "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "bool", "f64",
    ] {
        assert_eq!(
            mean.column_by_name(name).unwrap().data_type(),
            &DataType::Float64
        );
    }
    assert_eq!(
        mean.column_by_name("f16").unwrap().data_type(),
        &DataType::Float16
    );
    assert_eq!(
        mean.column_by_name("f32").unwrap().data_type(),
        &DataType::Float32
    );
    let all = aggregate(&input, &single_index(), BuiltInAggregation::All);
    assert!(
        all.columns()
            .iter()
            .all(|column| column.data_type() == &DataType::Boolean)
    );
}

#[test]
fn grouped_integer_results_narrow_when_safe_and_promote_on_overflow() {
    let input = batch(vec![
        (
            "instrument",
            Arc::new(StringArray::from(vec!["A", "A", "B", "B"])),
        ),
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(vec![
                timestamp(1),
                timestamp(2),
                timestamp(1),
                timestamp(2),
            ])),
        ),
        (
            "i8_overflow",
            Arc::new(Int8Array::from(vec![100, 100, 100, 100])),
        ),
        ("i16", Arc::new(Int16Array::from(vec![2, 3, 2, 3]))),
        ("i32", Arc::new(Int32Array::from(vec![2, 3, 2, 3]))),
        ("u8", Arc::new(UInt8Array::from(vec![2, 3, 2, 3]))),
        ("u16", Arc::new(UInt16Array::from(vec![2, 3, 2, 3]))),
        ("u32", Arc::new(UInt32Array::from(vec![2, 3, 2, 3]))),
    ]);
    for method in [BuiltInAggregation::Sum, BuiltInAggregation::Product] {
        let result = aggregate(
            &input,
            &multi_index(TimeSeriesIndexOrder::InstrumentDatetime),
            method,
        );
        assert_eq!(
            result.column_by_name("i8_overflow").unwrap().data_type(),
            &DataType::Int64
        );
        assert_eq!(
            result.column_by_name("i16").unwrap().data_type(),
            &DataType::Int16
        );
        assert_eq!(
            result.column_by_name("i32").unwrap().data_type(),
            &DataType::Int32
        );
        assert_eq!(
            result.column_by_name("u8").unwrap().data_type(),
            &DataType::UInt8
        );
        assert_eq!(
            result.column_by_name("u16").unwrap().data_type(),
            &DataType::UInt16
        );
        assert_eq!(
            result.column_by_name("u32").unwrap().data_type(),
            &DataType::UInt32
        );
    }
}

#[test]
fn dispatch_empty_and_every_error_remain_explicit() {
    for (text, method) in [
        ("all", BuiltInAggregation::All),
        ("sum", BuiltInAggregation::Sum),
        ("mean", BuiltInAggregation::Mean),
        ("prod", BuiltInAggregation::Product),
        ("first", BuiltInAggregation::First),
        ("last", BuiltInAggregation::Last),
    ] {
        assert_eq!(text.parse::<BuiltInAggregation>().unwrap(), method);
        assert_eq!(method.to_string(), text);
        assert_eq!(
            serde_json::from_str::<BuiltInAggregation>(&format!("\"{text}\"")).unwrap(),
            method
        );
    }
    let empty = batch(vec![
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(Vec::<i64>::new())),
        ),
        ("value", Arc::new(Float64Array::from(Vec::<f64>::new()))),
    ]);
    assert!(
        aggregate_time_series(
            &empty,
            &single_index(),
            TimeRange::default(),
            BuiltInAggregation::Sum
        )
        .unwrap()
        .is_none()
    );
    assert!(
        aggregate_time_series_str(&empty, &single_index(), TimeRange::default(), "sum")
            .unwrap()
            .is_none()
    );

    let input = batch(vec![
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(vec![timestamp(1)])),
        ),
        ("text", Arc::new(StringArray::from(vec!["x"]))),
    ]);
    assert!(matches!(
        aggregate_time_series_str(&input, &single_index(), TimeRange::default(), "median"),
        Err(TimeSeriesAggregationError::UnsupportedMethod { method }) if method == "median"
    ));
    for method in [BuiltInAggregation::First, BuiltInAggregation::Last] {
        assert!(matches!(
            aggregate_time_series(&input, &single_index(), TimeRange::default(), method),
            Err(TimeSeriesAggregationError::RequiresInstrumentGrouping { method: actual }) if actual == method
        ));
    }
    for method in [
        BuiltInAggregation::All,
        BuiltInAggregation::Sum,
        BuiltInAggregation::Mean,
        BuiltInAggregation::Product,
    ] {
        assert!(matches!(
            aggregate_time_series(&input, &single_index(), TimeRange::default(), method),
            Err(TimeSeriesAggregationError::UnsupportedDataType { method: actual, data_type })
                if actual == method && data_type == DataType::Utf8
        ));
    }
    let wrong_index = TimeSeriesIndex::Datetime {
        datetime: "missing".to_owned(),
    };
    assert!(matches!(
        aggregate_time_series(&input, &wrong_index, TimeRange::default(), BuiltInAggregation::Sum),
        Err(TimeSeriesAggregationError::Selection(TimeSeriesSelectionError::MissingIndexColumn { column }))
            if column == "missing"
    ));

    let null_groups = batch(vec![
        (
            "instrument",
            Arc::new(StringArray::from(vec![None::<&str>])),
        ),
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(vec![timestamp(1)])),
        ),
        ("value", Arc::new(Float64Array::from(vec![1.0]))),
    ]);
    assert!(
        aggregate_time_series(
            &null_groups,
            &multi_index(TimeSeriesIndexOrder::InstrumentDatetime),
            TimeRange::default(),
            BuiltInAggregation::Sum
        )
        .unwrap()
        .is_none()
    );
}

#[test]
#[allow(
    clippy::float_cmp,
    clippy::too_many_lines,
    reason = "one matrix verifies interacting single-index kwargs and Arrow shapes"
)]
fn single_index_method_kwargs_control_missing_counts_and_column_selection() {
    let input = batch(vec![
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(vec![
                timestamp(1),
                timestamp(2),
                timestamp(3),
            ])),
        ),
        (
            "number",
            Arc::new(Float64Array::from(vec![Some(1.0), None, Some(3.0)])),
        ),
        (
            "zero",
            Arc::new(Float64Array::from(vec![Some(1.0), None, Some(0.0)])),
        ),
        (
            "flag",
            Arc::new(BooleanArray::from(vec![Some(true), None, Some(true)])),
        ),
        ("text", Arc::new(StringArray::from(vec!["x", "y", "z"]))),
    ]);

    let numeric_only = BTreeMap::from([("numeric_only".to_owned(), json!(true))]);
    let sum = aggregate_with(
        &input,
        &single_index(),
        BuiltInAggregation::Sum,
        numeric_only,
    )
    .unwrap()
    .unwrap();
    assert_eq!(sum.num_columns(), 3);
    assert_eq!(value::<Float64Array>(&sum, "number").value(0), 4.0);
    assert_eq!(value::<Int64Array>(&sum, "flag").value(0), 2);
    assert!(sum.column_by_name("text").is_none());

    let bool_only = BTreeMap::from([("bool_only".to_owned(), json!(true))]);
    let all = aggregate_with(&input, &single_index(), BuiltInAggregation::All, bool_only)
        .unwrap()
        .unwrap();
    assert_eq!(all.num_columns(), 1);
    assert!(value::<BooleanArray>(&all, "flag").value(0));

    for method in [
        BuiltInAggregation::Sum,
        BuiltInAggregation::Mean,
        BuiltInAggregation::Product,
    ] {
        let output = aggregate_with(
            &columns(&input, &[0, 1]),
            &single_index(),
            method,
            BTreeMap::from([("skipna".to_owned(), json!(false))]),
        )
        .unwrap()
        .unwrap();
        assert!(output.column(0).is_null(0));
    }

    for method in [BuiltInAggregation::Sum, BuiltInAggregation::Product] {
        let output = aggregate_with(
            &columns(&input, &[0, 1]),
            &single_index(),
            method,
            BTreeMap::from([("min_count".to_owned(), json!(3))]),
        )
        .unwrap()
        .unwrap();
        assert!(output.column(0).is_null(0));
    }

    let all = aggregate_with(
        &input,
        &single_index(),
        BuiltInAggregation::All,
        BTreeMap::from([("skipna".to_owned(), json!(false))]),
    )
    .unwrap_err();
    assert!(matches!(
        all,
        TimeSeriesAggregationError::UnsupportedDataType {
            method: BuiltInAggregation::All,
            data_type: DataType::Utf8
        }
    ));
    let numeric = columns(&input, &[0, 1, 2, 3]);
    let all = aggregate_with(
        &numeric,
        &single_index(),
        BuiltInAggregation::All,
        BTreeMap::from([("skipna".to_owned(), json!(false))]),
    )
    .unwrap()
    .unwrap();
    assert!(value::<BooleanArray>(&all, "number").is_null(0));
    assert!(!value::<BooleanArray>(&all, "zero").value(0));
    assert!(value::<BooleanArray>(&all, "flag").is_null(0));

    let nan = batch(vec![
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(vec![timestamp(1)])),
        ),
        ("value", Arc::new(Float64Array::from(vec![f64::NAN]))),
    ]);
    let all_nan = aggregate_with(
        &nan,
        &single_index(),
        BuiltInAggregation::All,
        BTreeMap::from([("skipna".to_owned(), json!(false))]),
    )
    .unwrap()
    .unwrap();
    assert!(value::<BooleanArray>(&all_nan, "value").value(0));
}

#[test]
#[allow(
    clippy::float_cmp,
    reason = "fixtures use exactly representable integer-valued IEEE results"
)]
fn grouped_method_kwargs_preserve_layout_specific_pandas_rules() {
    let input = batch(vec![
        (
            "instrument",
            Arc::new(StringArray::from(vec!["A", "A", "B"])),
        ),
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(vec![
                timestamp(1),
                timestamp(2),
                timestamp(1),
            ])),
        ),
        (
            "value",
            Arc::new(Float64Array::from(vec![Some(1.0), None, Some(4.0)])),
        ),
        ("text", Arc::new(StringArray::from(vec!["x", "y", "z"]))),
    ]);
    let index = multi_index(TimeSeriesIndexOrder::InstrumentDatetime);

    let numeric = aggregate_with(
        &input,
        &index,
        BuiltInAggregation::Sum,
        BTreeMap::from([("numeric_only".to_owned(), json!(true))]),
    )
    .unwrap()
    .unwrap();
    assert_eq!(numeric.num_columns(), 2);
    assert!(numeric.column_by_name("text").is_none());

    for method in [BuiltInAggregation::Sum, BuiltInAggregation::Product] {
        let output = aggregate_with(
            &columns(&input, &[0, 1, 2]),
            &index,
            method,
            BTreeMap::from([("min_count".to_owned(), json!(2))]),
        )
        .unwrap()
        .unwrap();
        assert!(value::<Float64Array>(&output, "value").is_null(0));
        assert!(value::<Float64Array>(&output, "value").is_null(1));
    }

    let first = aggregate_with(
        &input,
        &index,
        BuiltInAggregation::First,
        BTreeMap::from([("min_count".to_owned(), json!(2))]),
    )
    .unwrap()
    .unwrap();
    assert!(value::<Float64Array>(&first, "value").is_null(0));
    assert!(value::<StringArray>(&first, "text").is_null(1));

    let last = aggregate_with(
        &input,
        &index,
        BuiltInAggregation::Last,
        BTreeMap::from([("skipna".to_owned(), json!(false))]),
    )
    .unwrap()
    .unwrap();
    assert!(value::<Float64Array>(&last, "value").is_null(0));
    assert_eq!(value::<Float64Array>(&last, "value").value(1), 4.0);

    let first_positional = aggregate_with(
        &input,
        &index,
        BuiltInAggregation::First,
        BTreeMap::from([("skipna".to_owned(), json!(false))]),
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        value::<Float64Array>(&first_positional, "value").value(0),
        1.0
    );

    let negative = aggregate_with(
        &input,
        &index,
        BuiltInAggregation::First,
        BTreeMap::from([("min_count".to_owned(), json!(-5))]),
    )
    .unwrap()
    .unwrap();
    assert_eq!(value::<Float64Array>(&negative, "value").value(0), 1.0);
}

#[test]
fn every_method_kwargs_validation_failure_is_typed() {
    let single = batch(vec![
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(vec![timestamp(1)])),
        ),
        ("value", Arc::new(Int64Array::from(vec![1]))),
    ]);
    let grouped = batch(vec![
        ("instrument", Arc::new(StringArray::from(vec!["A"]))),
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(vec![timestamp(1)])),
        ),
        ("value", Arc::new(Int64Array::from(vec![1]))),
    ]);
    let multi = multi_index(TimeSeriesIndexOrder::InstrumentDatetime);

    for axis in [Value::Null, json!(0), json!("index")] {
        aggregate_with(
            &single,
            &single_index(),
            BuiltInAggregation::Sum,
            BTreeMap::from([("axis".to_owned(), axis)]),
        )
        .unwrap();
    }
    assert!(matches!(
        aggregate_with(
            &single,
            &single_index(),
            BuiltInAggregation::Sum,
            BTreeMap::from([("axis".to_owned(), json!("columns"))])
        ),
        Err(TimeSeriesAggregationError::InvalidArgument { argument, .. }) if argument == "axis"
    ));

    for (method, argument) in [
        (BuiltInAggregation::All, "skipna"),
        (BuiltInAggregation::Sum, "numeric_only"),
        (BuiltInAggregation::All, "bool_only"),
    ] {
        assert!(matches!(
            aggregate_with(
                &single,
                &single_index(),
                method,
                BTreeMap::from([(argument.to_owned(), json!("invalid"))])
            ),
            Err(TimeSeriesAggregationError::InvalidArgument { argument: rejected, .. })
                if rejected == argument
        ));
    }
    for invalid in [json!(1.5), json!(u64::MAX)] {
        assert!(matches!(
            aggregate_with(
                &single,
                &single_index(),
                BuiltInAggregation::Sum,
                BTreeMap::from([("min_count".to_owned(), invalid)])
            ),
            Err(TimeSeriesAggregationError::InvalidArgument { argument, .. })
                if argument == "min_count"
        ));
    }

    for (method, argument) in [
        (BuiltInAggregation::Sum, "skipna"),
        (BuiltInAggregation::Mean, "skipna"),
        (BuiltInAggregation::All, "bool_only"),
        (BuiltInAggregation::Mean, "min_count"),
        (BuiltInAggregation::Product, "engine"),
    ] {
        assert!(matches!(
            aggregate_with(
                &grouped,
                &multi,
                method,
                BTreeMap::from([(argument.to_owned(), Value::Null)])
            ),
            Err(TimeSeriesAggregationError::UnsupportedArgument { argument: rejected, .. })
                if rejected == argument
        ));
    }
}

#[test]
fn built_in_aggregations_match_live_python_resam_ts_data() {
    let utils =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/utils/__init__.py");
    let dataset =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/data/dataset/utils.py");
    let resam = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/utils/resam.py");
    let script = r#"
import ast, json, sys
from typing import Callable, Union
import numpy as np
import pandas as pd
def fn(path, name):
    tree=ast.parse(open(path, encoding="utf-8").read(), filename=path)
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
nodes=[fn(sys.argv[1], "lazy_sort_index"), fn(sys.argv[2], "get_level_index"), fn(sys.argv[3], "resam_ts_data")]
nodes[2].body=[n for n in nodes[2].body if not isinstance(n, ast.ImportFrom)]
module=ast.Module(body=nodes, type_ignores=[]); ast.fix_missing_locations(module)
ns={"pd":pd,"np":np,"Callable":Callable,"Union":Union,"is_deprecated_lexsorted_pandas":True}
exec(compile(module, sys.argv[3], "exec"), ns)
idx=pd.DatetimeIndex(["2024-01-03","2024-01-01","2024-01-02"],name="datetime")
single=pd.DataFrame({"a":[0.0,np.nan,2.0],"b":[True,True,True]},index=idx)
mi=pd.MultiIndex.from_tuples([("B","2024-01-02"),("A","2024-01-02"),("A","2024-01-01"),("B","2024-01-01")],names=["instrument","datetime"])
multi=pd.DataFrame({"a":[2.0,2.0,1.0,4.0],"b":[True,False,True,True]},index=mi)
def clean(v):
    if isinstance(v,pd.Series): return [None if pd.isna(x) else x for x in v.tolist()]
    return [[None if pd.isna(x) else x for x in row] for row in v.to_numpy().tolist()]
out={}
for method in ["all","sum","mean","prod"]: out["single_"+method]=clean(ns["resam_ts_data"](single,method=method))
for method in ["all","sum","mean","prod","first","last"]: out["multi_"+method]=clean(ns["resam_ts_data"](multi,method=method))
print(json.dumps(out,sort_keys=True))
"#;
    let output = Command::new(std::env::var_os("PYTHON").unwrap_or_else(|| "python".into()))
        .arg("-c")
        .arg(script)
        .arg(utils)
        .arg(dataset)
        .arg(resam)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let actual: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        actual,
        json!({
            "single_all": [false, true], "single_sum": [2.0, 3.0],
            "single_mean": [1.0, 1.0], "single_prod": [0.0, 1.0],
        "multi_all": [[true, false], [true, true]],
        "multi_sum": [[3.0, 1.0], [6.0, 2.0]],
            "multi_mean": [[1.5, 0.5], [3.0, 1.0]],
        "multi_prod": [[2.0, 0.0], [8.0, 1.0]],
            "multi_first": [[1.0, true], [4.0, true]],
            "multi_last": [[2.0, false], [2.0, true]]
        })
    );
}
