use std::{path::PathBuf, process::Command, sync::Arc};

use arrow_array::{
    Array, ArrayRef, Int64Array, RecordBatch, StringArray, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use chrono::{NaiveDate, NaiveDateTime};
use domain_core::{
    TimeRange, TimeSeriesIndex, TimeSeriesIndexOrder, TimeSeriesSelectionError, select_time_series,
};
use serde_json::{Value, json};

fn timestamp(text: &str) -> NaiveDateTime {
    NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S%.f").expect("fixture timestamp is valid")
}

fn nanos(text: &str) -> i64 {
    timestamp(text)
        .and_utc()
        .timestamp_nanos_opt()
        .expect("fixture is in Pandas range")
}

fn datetime_index() -> TimeSeriesIndex {
    TimeSeriesIndex::Datetime {
        datetime: "datetime".to_owned(),
    }
}

fn batch(columns: Vec<(&str, ArrayRef)>) -> RecordBatch {
    RecordBatch::try_from_iter(columns).expect("fixture batch is valid")
}

fn int_values(batch: &RecordBatch, name: &str) -> Vec<i64> {
    batch
        .column_by_name(name)
        .expect("fixture column exists")
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("fixture column is Int64")
        .values()
        .to_vec()
}

fn time_values(batch: &RecordBatch) -> Vec<Option<i64>> {
    batch
        .column_by_name("datetime")
        .expect("datetime exists")
        .as_any()
        .downcast_ref::<TimestampNanosecondArray>()
        .expect("fixture datetime is nanoseconds")
        .iter()
        .collect()
}

#[test]
fn single_index_selection_is_stable_sorted_inclusive_and_optional() {
    let input = batch(vec![
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(vec![
                nanos("2024-01-03 00:00:00"),
                nanos("2024-01-01 00:00:00"),
                nanos("2024-01-02 00:00:00"),
                nanos("2024-01-02 00:00:00"),
            ])),
        ),
        ("value", Arc::new(Int64Array::from(vec![30, 10, 20, 21]))),
    ]);
    let selected = select_time_series(
        &input,
        &datetime_index(),
        TimeRange {
            start: Some(timestamp("2024-01-02 00:00:00")),
            end: Some(timestamp("2024-01-03 00:00:00")),
        },
    )
    .unwrap()
    .unwrap();
    assert_eq!(int_values(&selected, "value"), [20, 21, 30]);
    assert_eq!(
        time_values(&selected),
        [
            Some(nanos("2024-01-02 00:00:00")),
            Some(nanos("2024-01-02 00:00:00")),
            Some(nanos("2024-01-03 00:00:00")),
        ]
    );

    assert!(
        select_time_series(
            &input,
            &datetime_index(),
            TimeRange {
                start: Some(timestamp("2024-01-03 00:00:00")),
                end: Some(timestamp("2024-01-02 00:00:00")),
            },
        )
        .unwrap()
        .is_none()
    );

    let end_only = select_time_series(
        &input,
        &datetime_index(),
        TimeRange {
            start: None,
            end: Some(timestamp("2024-01-02 00:00:00")),
        },
    )
    .unwrap()
    .unwrap();
    assert_eq!(int_values(&end_only, "value"), [10, 20, 21]);

    let sorted = batch(vec![
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(vec![
                nanos("2024-01-01 00:00:00"),
                nanos("2024-01-02 00:00:00"),
            ])),
        ),
        ("value", Arc::new(Int64Array::from(vec![1, 2]))),
    ]);
    let unchanged = select_time_series(&sorted, &datetime_index(), TimeRange::default())
        .unwrap()
        .unwrap();
    assert!(Arc::ptr_eq(sorted.column(0), unchanged.column(0)));
    assert!(Arc::ptr_eq(sorted.column(1), unchanged.column(1)));

    let sorted_duplicates = batch(vec![
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(vec![
                nanos("2024-01-01 00:00:00"),
                nanos("2024-01-01 00:00:00"),
                nanos("2024-01-02 00:00:00"),
            ])),
        ),
        ("value", Arc::new(Int64Array::from(vec![1, 2, 3]))),
    ]);
    let duplicates =
        select_time_series(&sorted_duplicates, &datetime_index(), TimeRange::default())
            .unwrap()
            .unwrap();
    assert_eq!(int_values(&duplicates, "value"), [1, 2, 3]);
}

#[test]
fn both_multi_index_orders_sort_by_declared_level_order() {
    let datetimes = Arc::new(TimestampNanosecondArray::from(vec![
        nanos("2024-01-03 00:00:00"),
        nanos("2024-01-02 00:00:00"),
        nanos("2024-01-01 00:00:00"),
        nanos("2024-01-01 00:00:00"),
        nanos("2024-01-02 00:00:00"),
    ])) as ArrayRef;
    let instruments = Arc::new(StringArray::from(vec!["B", "A", "B", "A", "A"])) as ArrayRef;
    let values = Arc::new(Int64Array::from(vec![5, 3, 4, 1, 2])) as ArrayRef;
    let input = batch(vec![
        ("instrument", instruments.clone()),
        ("datetime", datetimes.clone()),
        ("value", values.clone()),
    ]);
    let range = TimeRange {
        start: Some(timestamp("2024-01-02 00:00:00")),
        end: Some(timestamp("2024-01-03 00:00:00")),
    };
    let instrument_first = select_time_series(
        &input,
        &TimeSeriesIndex::InstrumentDatetime {
            instrument: "instrument".to_owned(),
            datetime: "datetime".to_owned(),
            order: TimeSeriesIndexOrder::InstrumentDatetime,
        },
        range,
    )
    .unwrap()
    .unwrap();
    assert_eq!(int_values(&instrument_first, "value"), [3, 2, 5]);

    let datetime_first_input = batch(vec![
        ("datetime", datetimes),
        ("instrument", instruments),
        ("value", values),
    ]);
    let datetime_first = select_time_series(
        &datetime_first_input,
        &TimeSeriesIndex::InstrumentDatetime {
            instrument: "instrument".to_owned(),
            datetime: "datetime".to_owned(),
            order: TimeSeriesIndexOrder::DatetimeInstrument,
        },
        range,
    )
    .unwrap()
    .unwrap();
    assert_eq!(int_values(&datetime_first, "value"), [3, 2, 5]);
}

#[test]
fn nat_empty_rows_and_index_only_batches_match_pandas_empty_behavior() {
    let input = batch(vec![
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(vec![
                None,
                Some(nanos("2024-01-02 00:00:00")),
                Some(nanos("2024-01-01 00:00:00")),
            ])),
        ),
        ("value", Arc::new(Int64Array::from(vec![9, 2, 1]))),
    ]);
    let unbounded = select_time_series(&input, &datetime_index(), TimeRange::default())
        .unwrap()
        .unwrap();
    assert_eq!(int_values(&unbounded, "value"), [1, 2, 9]);
    assert_eq!(
        time_values(&unbounded),
        [
            Some(nanos("2024-01-01 00:00:00")),
            Some(nanos("2024-01-02 00:00:00")),
            None,
        ]
    );
    let bounded = select_time_series(
        &input,
        &datetime_index(),
        TimeRange {
            start: Some(timestamp("2024-01-01 00:00:00")),
            end: None,
        },
    )
    .unwrap()
    .unwrap();
    assert_eq!(int_values(&bounded, "value"), [1, 2]);

    let empty = batch(vec![
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(Vec::<i64>::new())),
        ),
        ("value", Arc::new(Int64Array::from(Vec::<i64>::new()))),
    ]);
    assert!(
        select_time_series(&empty, &datetime_index(), TimeRange::default())
            .unwrap()
            .is_none()
    );
    let index_only = batch(vec![(
        "datetime",
        Arc::new(TimestampNanosecondArray::from(vec![nanos(
            "2024-01-01 00:00:00",
        )])),
    )]);
    assert!(
        select_time_series(&index_only, &datetime_index(), TimeRange::default())
            .unwrap()
            .is_none()
    );
}

#[test]
fn all_timestamp_units_and_timezone_metadata_use_exact_inclusive_instants() {
    let start = timestamp("1970-01-01 00:00:00");
    let range = TimeRange {
        start: Some(start),
        end: Some(start),
    };
    let cases: Vec<ArrayRef> = vec![
        Arc::new(TimestampSecondArray::from(vec![
            None,
            Some(-1),
            Some(0),
            Some(1),
        ])),
        Arc::new(TimestampMillisecondArray::from(vec![
            None,
            Some(-1),
            Some(0),
            Some(1),
        ])),
        Arc::new(TimestampMicrosecondArray::from(vec![
            None,
            Some(-1),
            Some(0),
            Some(1),
        ])),
        Arc::new(
            TimestampNanosecondArray::from(vec![None, Some(-1), Some(0), Some(1)])
                .with_timezone("+08:00"),
        ),
    ];
    for datetime in cases {
        let input = batch(vec![
            ("datetime", datetime),
            ("value", Arc::new(Int64Array::from(vec![99, -1, 0, 1]))),
        ]);
        let selected = select_time_series(&input, &datetime_index(), range)
            .unwrap()
            .unwrap();
        assert_eq!(int_values(&selected, "value"), [0]);
    }
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "the typed error matrix shares one minimal valid Arrow fixture"
)]
fn invalid_layout_types_and_bounds_are_typed() {
    let valid_time = Arc::new(TimestampNanosecondArray::from(vec![0])) as ArrayRef;
    let value = Arc::new(Int64Array::from(vec![1])) as ArrayRef;
    let input = batch(vec![
        ("datetime", valid_time.clone()),
        ("value", value.clone()),
    ]);

    assert!(matches!(
        select_time_series(
            &input,
            &TimeSeriesIndex::Datetime {
                datetime: "missing".to_owned()
            },
            TimeRange::default()
        ),
        Err(TimeSeriesSelectionError::MissingIndexColumn { column }) if column == "missing"
    ));

    let duplicate_schema = Arc::new(Schema::new(vec![
        Field::new(
            "datetime",
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            false,
        ),
        Field::new(
            "datetime",
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            false,
        ),
        Field::new("value", DataType::Int64, false),
    ]));
    let duplicate = RecordBatch::try_new(
        duplicate_schema,
        vec![valid_time.clone(), valid_time.clone(), value.clone()],
    )
    .unwrap();
    assert!(matches!(
        select_time_series(&duplicate, &datetime_index(), TimeRange::default()),
        Err(TimeSeriesSelectionError::AmbiguousIndexColumn { column }) if column == "datetime"
    ));

    assert!(matches!(
        select_time_series(
            &input,
            &TimeSeriesIndex::InstrumentDatetime {
                instrument: "datetime".to_owned(),
                datetime: "datetime".to_owned(),
                order: TimeSeriesIndexOrder::InstrumentDatetime,
            },
            TimeRange::default()
        ),
        Err(TimeSeriesSelectionError::DuplicateIndexColumns { column }) if column == "datetime"
    ));

    let multi_index = |instrument: &str, datetime: &str| TimeSeriesIndex::InstrumentDatetime {
        instrument: instrument.to_owned(),
        datetime: datetime.to_owned(),
        order: TimeSeriesIndexOrder::InstrumentDatetime,
    };
    assert!(matches!(
        select_time_series(
            &input,
            &multi_index("missing_instrument", "datetime"),
            TimeRange::default()
        ),
        Err(TimeSeriesSelectionError::MissingIndexColumn { column })
            if column == "missing_instrument"
    ));
    assert!(matches!(
        select_time_series(
            &input,
            &multi_index("value", "missing_datetime"),
            TimeRange::default()
        ),
        Err(TimeSeriesSelectionError::MissingIndexColumn { column })
            if column == "missing_datetime"
    ));

    let wrong_type = batch(vec![
        ("datetime", Arc::new(Int64Array::from(vec![0]))),
        ("value", value.clone()),
    ]);
    assert!(matches!(
        select_time_series(&wrong_type, &datetime_index(), TimeRange::default()),
        Err(TimeSeriesSelectionError::InvalidDatetimeType { column, data_type })
            if column == "datetime" && data_type == DataType::Int64
    ));

    let outside = NaiveDate::from_ymd_opt(1600, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    assert!(matches!(
        select_time_series(
            &input,
            &datetime_index(),
            TimeRange { start: Some(outside), end: None }
        ),
        Err(TimeSeriesSelectionError::TimeBoundOutOfRange { timestamp }) if timestamp == outside
    ));
    assert!(matches!(
        select_time_series(
            &input,
            &datetime_index(),
            TimeRange { start: None, end: Some(outside) }
        ),
        Err(TimeSeriesSelectionError::TimeBoundOutOfRange { timestamp }) if timestamp == outside
    ));
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "live Python differential fixture stays with its assertions"
)]
fn selection_matches_live_python_resam_ts_data_with_no_aggregation() {
    let utils_source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/utils/__init__.py");
    let dataset_source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/data/dataset/utils.py");
    let resam_source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/utils/resam.py");
    assert!(utils_source.is_file() && dataset_source.is_file() && resam_source.is_file());
    let script = r#"
import ast, json, sys
from typing import Callable, Union
import pandas as pd

def function(path, name):
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
lazy = function(sys.argv[1], "lazy_sort_index")
get_level = function(sys.argv[2], "get_level_index")
resample = function(sys.argv[3], "resam_ts_data")
resample.body = [node for node in resample.body if not isinstance(node, ast.ImportFrom)]
module = ast.Module(body=[lazy, get_level, resample], type_ignores=[])
ast.fix_missing_locations(module)
namespace = {"pd": pd, "Callable": Callable, "Union": Union, "is_deprecated_lexsorted_pandas": True}
exec(compile(module, sys.argv[3], "exec"), namespace)

def iso(value): return value.isoformat()
single_index = pd.DatetimeIndex(["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-02"], name="datetime")
single = pd.Series([30, 10, 20, 21], index=single_index)
single_out = namespace["resam_ts_data"](single, "2024-01-02", "2024-01-03", method=None)

multi_index = pd.MultiIndex.from_tuples([
    ("B", pd.Timestamp("2024-01-03")), ("A", pd.Timestamp("2024-01-02")),
    ("B", pd.Timestamp("2024-01-01")), ("A", pd.Timestamp("2024-01-01")),
    ("A", pd.Timestamp("2024-01-02"))], names=["instrument", "datetime"])
multi = pd.DataFrame({"value": [5, 3, 4, 1, 2]}, index=multi_index)
multi_out = namespace["resam_ts_data"](multi, "2024-01-02", "2024-01-03", method=None)

result = {
    "single": [[iso(index), int(value)] for index, value in single_out.items()],
    "multi": [[instrument, iso(datetime), int(row["value"])]
              for (instrument, datetime), row in multi_out.iterrows()],
    "reversed_is_none": namespace["resam_ts_data"](
        single, "2024-01-03", "2024-01-02", method=None) is None,
}
print(json.dumps(result, sort_keys=True))
"#;
    let output = Command::new(std::env::var_os("PYTHON").unwrap_or_else(|| "python".into()))
        .arg("-c")
        .arg(script)
        .arg(&utils_source)
        .arg(&dataset_source)
        .arg(&resam_source)
        .output()
        .expect("Python interpreter starts");
    assert!(
        output.status.success(),
        "Python snapshot failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let actual: Value = serde_json::from_slice(&output.stdout).expect("valid Python JSON");
    let expected = json!({
        "single": [
            ["2024-01-02T00:00:00", 20],
            ["2024-01-02T00:00:00", 21],
            ["2024-01-03T00:00:00", 30]
        ],
        "multi": [
            ["A", "2024-01-02T00:00:00", 3],
            ["A", "2024-01-02T00:00:00", 2],
            ["B", "2024-01-03T00:00:00", 5]
        ],
        "reversed_is_none": true
    });
    assert_eq!(actual, expected);
}
