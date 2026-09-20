use std::{any::Any, path::PathBuf, process::Command, sync::Arc};

use arrow_array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray,
    TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
    TimestampSecondArray, UInt64Array,
};
use arrow_buffer::NullBuffer;
use arrow_data::ArrayData;
use arrow_schema::{DataType, TimeUnit};
use domain_core::{AlphaLongShortReturn, AlphaReturnError, AlphaSeries, calc_long_short_return};
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug)]
struct DelegatingArray(ArrayRef);

unsafe impl Array for DelegatingArray {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn to_data(&self) -> ArrayData {
        self.0.to_data()
    }

    fn into_data(self) -> ArrayData {
        self.0.to_data()
    }

    fn data_type(&self) -> &DataType {
        self.0.data_type()
    }

    fn slice(&self, offset: usize, length: usize) -> ArrayRef {
        self.0.slice(offset, length)
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn offset(&self) -> usize {
        self.0.offset()
    }

    fn nulls(&self) -> Option<&NullBuffer> {
        self.0.nulls()
    }

    fn get_buffer_memory_size(&self) -> usize {
        self.0.get_buffer_memory_size()
    }

    fn get_array_memory_size(&self) -> usize {
        self.0.get_array_memory_size()
    }
}

#[derive(Debug, Deserialize)]
struct Fixture {
    ast: String,
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    name: String,
    pred: Series,
    label: Series,
    date_col: String,
    quantile: String,
    dropna: bool,
    result: ResultSnapshot,
}

#[derive(Debug, Deserialize)]
struct Series {
    level_names: [String; 2],
    level_types: [String; 2],
    levels: [Vec<Value>; 2],
    values: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ResultSnapshot {
    long_short_kind: Option<String>,
    long_short_columns: Option<[String; 2]>,
    long_short_column_types: Option<[String; 2]>,
    long_short_index_class: Option<String>,
    long_short_index_names: Option<Vec<String>>,
    dates: Option<Vec<Option<String>>>,
    date_type: Option<String>,
    index_name: Option<String>,
    long_short_name: Option<String>,
    average_name: Option<String>,
    long_short_bits: Option<Vec<String>>,
    average_bits: Option<Vec<String>>,
    error_type: Option<String>,
    error: Option<String>,
}

fn float(value: &str) -> f64 {
    match value {
        "nan" => f64::NAN,
        "inf" => f64::INFINITY,
        "-inf" => f64::NEG_INFINITY,
        value => value.parse().unwrap(),
    }
}

fn native(series: &Series) -> AlphaSeries {
    let levels: [ArrayRef; 2] = std::array::from_fn(|index| {
        let values = &series.levels[index];
        match series.level_types[index].as_str() {
            "utf8" => Arc::new(StringArray::from(
                values
                    .iter()
                    .map(|value| value.as_str().map(str::to_owned))
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            "int64" => Arc::new(Int64Array::from(
                values.iter().map(Value::as_i64).collect::<Vec<_>>(),
            )) as ArrayRef,
            "uint64" => Arc::new(UInt64Array::from(
                values.iter().map(Value::as_u64).collect::<Vec<_>>(),
            )) as ArrayRef,
            "timestamp_ns" => Arc::new(TimestampNanosecondArray::from(
                values.iter().map(Value::as_i64).collect::<Vec<_>>(),
            )) as ArrayRef,
            "timestamp_ns_utc" => Arc::new(
                TimestampNanosecondArray::from(
                    values.iter().map(Value::as_i64).collect::<Vec<_>>(),
                )
                .with_timezone("UTC"),
            ) as ArrayRef,
            other => panic!("unknown fixture index type {other}"),
        }
    });
    AlphaSeries::try_new(
        series.level_names.clone(),
        levels,
        Arc::new(Float64Array::from(
            series
                .values
                .iter()
                .map(|value| float(value))
                .collect::<Vec<_>>(),
        )),
    )
    .unwrap()
}

fn bits(values: &Float64Array) -> Vec<String> {
    values
        .iter()
        .map(|value| format!("{:016x}", value.unwrap_or(f64::NAN).to_bits()))
        .collect()
}

fn index_snapshot(values: &ArrayRef) -> (String, Vec<Option<String>>) {
    match values.data_type() {
        DataType::Utf8 => (
            "utf8".to_owned(),
            values
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .iter()
                .map(|value| value.map(str::to_owned))
                .collect(),
        ),
        DataType::Int64 => (
            "int64".to_owned(),
            values
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .iter()
                .map(|value| value.map(|value| value.to_string()))
                .collect(),
        ),
        DataType::Timestamp(TimeUnit::Nanosecond, timezone) => (
            if timezone.is_some() {
                "timestamp_ns_utc".to_owned()
            } else {
                "timestamp_ns".to_owned()
            },
            values
                .as_any()
                .downcast_ref::<TimestampNanosecondArray>()
                .unwrap()
                .iter()
                .map(|value| value.map(|value| value.to_string()))
                .collect(),
        ),
        other => panic!("unexpected result index type {other}"),
    }
}

fn live_fixture() -> Fixture {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = std::env::var_os("QLIB_PYTHON_ALPHA").map_or_else(
        || root.join("../../../qlib/qlib/contrib/eva/alpha.py"),
        PathBuf::from,
    );
    let output = Command::new(std::env::var_os("PYTHON").unwrap_or_else(|| "python".into()))
        .arg(root.join("tests/fixtures/alpha_returns_contract.py"))
        .arg(source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn outputs_and_failures_match_live_source_ast() {
    let fixture = live_fixture();
    assert!(
        fixture
            .ast
            .starts_with("FunctionDef(name='calc_long_short_return'")
    );
    assert!(fixture.ast.contains("attr='nlargest'"));
    assert!(fixture.ast.contains("attr='nsmallest'"));
    assert_eq!(fixture.cases.len(), 18);

    for case in fixture.cases {
        let pred = native(&case.pred);
        let label = native(&case.label);
        assert_eq!(pred.level_names(), &case.pred.level_names);
        let result = calc_long_short_return(
            &pred,
            &label,
            &case.date_col,
            float(&case.quantile),
            case.dropna,
        );
        if case.result.long_short_kind.as_deref() == Some("frame") {
            let result = result.unwrap_or_else(|error| panic!("{}: {error}", case.name));
            assert_eq!(
                result.long_short.empty_frame_columns(),
                case.result.long_short_columns.as_ref()
            );
            assert_eq!(
                result.long_short.empty_frame_column_types(),
                Some(&[DataType::Float64, DataType::Float64])
            );
            assert_eq!(
                case.result.long_short_column_types.unwrap(),
                ["float64", "float64"]
            );
            assert_eq!(case.result.long_short_index_class.as_deref(), Some("Index"));
            assert_eq!(
                case.result.long_short_index_names.as_deref(),
                Some([case.date_col.clone()].as_slice())
            );
            let (index_name, index) = result.long_short.empty_frame_index().unwrap();
            assert_eq!(index_name, case.date_col);
            assert_eq!(index_snapshot(index).1, Vec::<Option<String>>::new());
            assert_eq!(result.average.values().len(), 0, "{}", case.name);
            assert_eq!(result.average.name(), case.result.average_name.as_deref());
            assert_eq!(
                result.average.index_name(),
                case.result.index_name.as_deref().unwrap()
            );
            assert_eq!(
                index_snapshot(result.average.dates()).0,
                case.result.date_type.unwrap()
            );
            assert_eq!(case.result.dates.unwrap(), Vec::<Option<String>>::new());
            assert_eq!(case.result.average_bits.unwrap(), Vec::<String>::new());
        } else if let Some(expected) = case.result.long_short_bits {
            let result = result.unwrap_or_else(|error| panic!("{}: {error}", case.name));
            let long_short = result.long_short.as_series().unwrap();
            let (date_type, dates) = index_snapshot(long_short.dates());
            assert_eq!(dates, case.result.dates.unwrap(), "{}", case.name);
            assert_eq!(date_type, case.result.date_type.unwrap(), "{}", case.name);
            assert_eq!(long_short.index_name(), case.result.index_name.unwrap());
            assert_eq!(long_short.name(), case.result.long_short_name.as_deref());
            assert_eq!(result.average.name(), case.result.average_name.as_deref());
            assert_eq!(bits(long_short.values()), expected, "{}", case.name);
            assert_eq!(
                bits(result.average.values()),
                case.result.average_bits.unwrap(),
                "{}",
                case.name
            );
        } else {
            let error = result.unwrap_err();
            let expected = match case.name.as_str() {
                "nan_quantile_error" => AlphaReturnError::InvalidQuantile,
                "missing_date_level_error" => {
                    AlphaReturnError::DateLevelNotFound("missing".to_owned())
                }
                "incompatible_duplicate_alignment_error" => AlphaReturnError::NonUniqueAlignment,
                other => panic!("unexpected error case {other}"),
            };
            assert_eq!(error, expected, "{}", case.name);
            assert!(case.result.error_type.is_some());
            assert!(case.result.error.is_some());
        }
    }
}

fn sample(names: [&str; 2], dates: &[Option<&str>], instruments: &[Option<&str>]) -> AlphaSeries {
    let levels: [ArrayRef; 2] = [
        Arc::new(StringArray::from(dates.to_vec())),
        Arc::new(StringArray::from(instruments.to_vec())),
    ];
    AlphaSeries::try_new(
        names.map(str::to_owned),
        levels,
        Arc::new(Float64Array::from(vec![1.0; dates.len()])),
    )
    .unwrap()
}

#[test]
fn typed_boundary_validates_shapes_names_alignment_and_quantiles() {
    assert_eq!(
        AlphaSeries::try_new(
            ["date".to_owned(), "id".to_owned()],
            [
                Arc::new(StringArray::from(vec!["d"])) as ArrayRef,
                Arc::new(StringArray::from(Vec::<&str>::new())) as ArrayRef,
            ],
            Arc::new(Float64Array::from(vec![1.0])),
        )
        .unwrap_err(),
        AlphaReturnError::LengthMismatch
    );
    assert_eq!(
        AlphaSeries::try_new(
            ["date".to_owned(), "date".to_owned()],
            [
                Arc::new(StringArray::from(vec!["d"])) as ArrayRef,
                Arc::new(StringArray::from(vec!["a"])) as ArrayRef,
            ],
            Arc::new(Float64Array::from(vec![1.0])),
        )
        .unwrap_err(),
        AlphaReturnError::DuplicateLevelName("date".to_owned())
    );

    let pred = sample(["date", "id"], &[Some("d")], &[Some("a")]);
    let label = sample(["when", "id"], &[Some("d")], &[Some("a")]);
    assert_eq!(
        calc_long_short_return(&pred, &label, "date", 0.2, false).unwrap_err(),
        AlphaReturnError::IndexNamesDiffer
    );
    let label = sample(["date", "id"], &[Some("d")], &[Some("a")]);
    assert_eq!(
        calc_long_short_return(&pred, &label, "date", f64::INFINITY, false).unwrap_err(),
        AlphaReturnError::InvalidQuantile
    );
}

#[test]
fn reversed_date_level_null_values_and_inputs_are_preserved() {
    let index: [ArrayRef; 2] = [
        Arc::new(StringArray::from(vec![Some("b"), Some("a")])),
        Arc::new(StringArray::from(vec![Some("d"), Some("d")])),
    ];
    let pred_values = Arc::new(Float64Array::from(vec![Some(2.0), None]));
    let label_values = Arc::new(Float64Array::from(vec![20.0, 10.0]));
    let pred = AlphaSeries::try_new(
        ["instrument".to_owned(), "datetime".to_owned()],
        index.clone(),
        Arc::clone(&pred_values),
    )
    .unwrap();
    let label = AlphaSeries::try_new(
        ["instrument".to_owned(), "datetime".to_owned()],
        index,
        Arc::clone(&label_values),
    )
    .unwrap();
    let output = calc_long_short_return(&pred, &label, "datetime", 1.0, false).unwrap();
    assert_eq!(
        index_snapshot(output.average.dates()).1,
        [Some("d".to_owned())]
    );
    assert_eq!(
        output.average.values().value(0).to_bits(),
        15.0_f64.to_bits()
    );
    assert!(Arc::ptr_eq(pred.values(), &pred_values));
    assert!(Arc::ptr_eq(label.values(), &label_values));
    assert!(pred.values().is_null(1));
}

fn typed_series(levels: [ArrayRef; 2], values: Vec<f64>) -> AlphaSeries {
    AlphaSeries::try_new(
        ["datetime".to_owned(), "instrument".to_owned()],
        levels,
        Arc::new(Float64Array::from(values)),
    )
    .unwrap()
}

#[test]
fn every_supported_date_dtype_is_retained_and_unsupported_types_are_atomic() {
    let timestamps: Vec<(ArrayRef, DataType)> = vec![
        (
            Arc::new(TimestampSecondArray::from(vec![2, 1])),
            DataType::Timestamp(TimeUnit::Second, None),
        ),
        (
            Arc::new(TimestampMillisecondArray::from(vec![2, 1])),
            DataType::Timestamp(TimeUnit::Millisecond, None),
        ),
        (
            Arc::new(TimestampMicrosecondArray::from(vec![2, 1])),
            DataType::Timestamp(TimeUnit::Microsecond, None),
        ),
        (
            Arc::new(TimestampNanosecondArray::from(vec![2, 1])),
            DataType::Timestamp(TimeUnit::Nanosecond, None),
        ),
    ];
    for (dates, expected) in timestamps {
        let levels = [
            dates,
            Arc::new(StringArray::from(vec!["b", "a"])) as ArrayRef,
        ];
        let pred = typed_series(levels.clone(), vec![2.0, 1.0]);
        let label = typed_series(levels, vec![20.0, 10.0]);
        let output = calc_long_short_return(&pred, &label, "datetime", 1.0, false).unwrap();
        assert_eq!(
            output.long_short.as_series().unwrap().dates().data_type(),
            &expected
        );
    }

    let unsigned: [ArrayRef; 2] = [
        Arc::new(UInt64Array::from(vec![2, 1])),
        Arc::new(StringArray::from(vec!["b", "a"])),
    ];
    let pred = typed_series(unsigned.clone(), vec![2.0, 1.0]);
    let label = typed_series(unsigned, vec![20.0, 10.0]);
    let output = calc_long_short_return(&pred, &label, "datetime", 1.0, false).unwrap();
    assert_eq!(output.average.dates().data_type(), &DataType::UInt64);

    assert_eq!(
        AlphaSeries::try_new(
            ["datetime".to_owned(), "instrument".to_owned()],
            [
                Arc::new(BooleanArray::from(vec![true])) as ArrayRef,
                Arc::new(StringArray::from(vec!["a"])) as ArrayRef,
            ],
            Arc::new(Float64Array::from(vec![1.0])),
        )
        .unwrap_err(),
        AlphaReturnError::UnsupportedIndexType(DataType::Boolean)
    );
    assert_eq!(
        AlphaSeries::try_new(
            ["datetime".to_owned(), "instrument".to_owned()],
            [
                Arc::new(StringArray::from(vec!["d"])) as ArrayRef,
                Arc::new(BooleanArray::from(vec![true])) as ArrayRef,
            ],
            Arc::new(Float64Array::from(vec![1.0])),
        )
        .unwrap_err(),
        AlphaReturnError::UnsupportedIndexType(DataType::Boolean)
    );
}

#[test]
fn supported_dtype_with_noncanonical_array_returns_typed_error_atomically() {
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(vec!["d"])),
        Arc::new(Int64Array::from(vec![1])),
        Arc::new(UInt64Array::from(vec![1])),
        Arc::new(TimestampSecondArray::from(vec![1])),
        Arc::new(TimestampMillisecondArray::from(vec![1])),
        Arc::new(TimestampMicrosecondArray::from(vec![1])),
        Arc::new(TimestampNanosecondArray::from(vec![1])),
    ];

    for inner in arrays {
        let expected_type = inner.data_type().clone();
        let expected_data = inner.to_data();
        let values = Arc::new(Float64Array::from(vec![1.0]));
        let values_before = Arc::clone(&values);
        let wrapped = Arc::new(DelegatingArray(Arc::clone(&inner))) as ArrayRef;
        assert_eq!(wrapped.data_type(), &expected_type);
        assert_eq!(wrapped.to_data(), expected_data);

        assert_eq!(
            AlphaSeries::try_new(
                ["datetime".to_owned(), "instrument".to_owned()],
                [wrapped, Arc::new(StringArray::from(vec!["a"]))],
                Arc::clone(&values),
            )
            .unwrap_err(),
            AlphaReturnError::UnsupportedIndexType(expected_type)
        );
        assert_eq!(inner.to_data(), expected_data);
        assert!(Arc::ptr_eq(&values, &values_before));
        assert_eq!(values.value(0).to_bits(), 1.0_f64.to_bits());
    }
}

#[test]
fn remaining_alignment_and_error_boundaries_are_explicit() {
    let pred = typed_series(
        [
            Arc::new(Int64Array::from(vec![1])) as ArrayRef,
            Arc::new(StringArray::from(vec!["a"])) as ArrayRef,
        ],
        vec![1.0],
    );
    let different_type = typed_series(
        [
            Arc::new(UInt64Array::from(vec![1])) as ArrayRef,
            Arc::new(StringArray::from(vec!["a"])) as ArrayRef,
        ],
        vec![1.0],
    );
    assert_eq!(
        calc_long_short_return(&pred, &different_type, "datetime", 1.0, false).unwrap_err(),
        AlphaReturnError::IndexTypesDiffer
    );

    let duplicate_label = typed_series(
        [
            Arc::new(Int64Array::from(vec![1, 1])) as ArrayRef,
            Arc::new(StringArray::from(vec!["a", "a"])) as ArrayRef,
        ],
        vec![4.0, 4.0],
    );
    let output = calc_long_short_return(&pred, &duplicate_label, "datetime", 0.5, false).unwrap();
    assert_eq!(bits(output.average.values()), ["4010000000000000"]);

    let extra_unique = typed_series(
        [
            Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef,
            Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef,
        ],
        vec![1.0, 2.0],
    );
    assert_eq!(
        calc_long_short_return(&duplicate_label, &extra_unique, "datetime", 0.5, false)
            .unwrap_err(),
        AlphaReturnError::NonUniqueAlignment
    );
    assert_eq!(
        calc_long_short_return(&pred, &pred, "datetime", 1.0e30, false).unwrap_err(),
        AlphaReturnError::InvalidQuantile
    );

    let two_rows = typed_series(
        [
            Arc::new(Int64Array::from(vec![1, 1])) as ArrayRef,
            Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef,
        ],
        vec![1.0, 2.0],
    );
    assert_eq!(
        calc_long_short_return(&two_rows, &two_rows, "datetime", f64::MAX * 0.75, false,)
            .unwrap_err(),
        AlphaReturnError::InvalidQuantile
    );

    let duplicate_pred = typed_series(
        [
            Arc::new(Int64Array::from(vec![1, 1])) as ArrayRef,
            Arc::new(StringArray::from(vec!["a", "a"])) as ArrayRef,
        ],
        vec![1.0, 2.0],
    );
    let duplicate_other = typed_series(
        [
            Arc::new(Int64Array::from(vec![1, 1])) as ArrayRef,
            Arc::new(StringArray::from(vec!["b", "b"])) as ArrayRef,
        ],
        vec![3.0, 4.0],
    );
    assert_eq!(
        calc_long_short_return(&duplicate_pred, &duplicate_other, "datetime", 0.5, false)
            .unwrap_err(),
        AlphaReturnError::NonUniqueAlignment
    );
}

#[test]
fn missing_key_order_all_nan_ranking_and_empty_group_output_are_defined() {
    for (first, second) in [(None, Some("a")), (Some("a"), None)] {
        let pred = typed_series(
            [
                Arc::new(StringArray::from(vec![Some("d")])) as ArrayRef,
                Arc::new(StringArray::from(vec![first])) as ArrayRef,
            ],
            vec![f64::NAN],
        );
        let label = typed_series(
            [
                Arc::new(StringArray::from(vec![Some("d")])) as ArrayRef,
                Arc::new(StringArray::from(vec![second])) as ArrayRef,
            ],
            vec![f64::NAN],
        );
        let output = calc_long_short_return(&pred, &label, "datetime", 2.0, false).unwrap();
        assert!(
            output
                .long_short
                .as_series()
                .unwrap()
                .values()
                .value(0)
                .is_nan()
        );
    }

    let pred = typed_series(
        [
            Arc::new(StringArray::from(vec![None::<&str>])) as ArrayRef,
            Arc::new(StringArray::from(vec!["a"])) as ArrayRef,
        ],
        vec![1.0],
    );
    let label = typed_series(
        [
            Arc::new(StringArray::from(vec![None::<&str>])) as ArrayRef,
            Arc::new(StringArray::from(vec!["b"])) as ArrayRef,
        ],
        vec![2.0],
    );
    let output = calc_long_short_return(&pred, &label, "datetime", 1.0, false).unwrap();
    assert_eq!(output.average.values().len(), 0);

    let missing_dates = typed_series(
        [
            Arc::new(StringArray::from(vec![None, None::<&str>])) as ArrayRef,
            Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef,
        ],
        vec![f64::NAN, f64::NAN],
    );
    let output =
        calc_long_short_return(&missing_dates, &missing_dates, "datetime", 1.0, false).unwrap();
    assert_eq!(
        output.long_short.empty_frame_columns(),
        Some(&["pred".to_owned(), "label".to_owned()])
    );
    assert!(matches!(
        output.long_short,
        AlphaLongShortReturn::EmptyFrame { .. }
    ));
}
