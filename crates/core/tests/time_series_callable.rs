use std::{
    borrow::Cow,
    collections::BTreeMap,
    io,
    path::PathBuf,
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use arrow_array::{
    Array, ArrayRef, BooleanArray, Float16Array, Float32Array, Float64Array, Int8Array, Int16Array,
    Int32Array, Int64Array, RecordBatch, StringArray, TimestampNanosecondArray, UInt8Array,
    UInt16Array, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use chrono::NaiveDateTime;
use domain_core::{
    AggregationArguments, AggregatorError, AggregatorPhase, CompoundedReturnAggregator,
    LastValidAggregator, TimeRange, TimeSeriesAggregator, TimeSeriesCallableError, TimeSeriesIndex,
    TimeSeriesIndexOrder, TimeSeriesSelectionError, aggregate_time_series_with,
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

fn single_index() -> TimeSeriesIndex {
    TimeSeriesIndex::Datetime {
        datetime: "datetime".to_owned(),
    }
}

fn multi_index() -> TimeSeriesIndex {
    TimeSeriesIndex::InstrumentDatetime {
        instrument: "instrument".to_owned(),
        datetime: "datetime".to_owned(),
        order: TimeSeriesIndexOrder::DatetimeInstrument,
    }
}

fn run(
    input: &RecordBatch,
    index: &TimeSeriesIndex,
    aggregator: &dyn TimeSeriesAggregator,
    arguments: &AggregationArguments,
) -> RecordBatch {
    aggregate_time_series_with(input, index, TimeRange::default(), aggregator, arguments)
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

#[test]
#[allow(
    clippy::float_cmp,
    reason = "fixtures use exactly representable integer-valued IEEE results"
)]
fn last_valid_adapter_matches_single_and_grouped_ts_data_valid() {
    let single = batch(vec![
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(vec![
                timestamp(3),
                timestamp(1),
                timestamp(2),
            ])),
        ),
        (
            "a",
            Arc::new(Float64Array::from(vec![Some(3.0), None, Some(2.0)])),
        ),
        (
            "text",
            Arc::new(StringArray::from(vec![Some("three"), None, Some("two")])),
        ),
    ]);
    let last = run(
        &single,
        &single_index(),
        &LastValidAggregator,
        &BTreeMap::new(),
    );
    assert_eq!(value::<Float64Array>(&last, "a").value(0), 3.0);
    assert_eq!(value::<StringArray>(&last, "text").value(0), "three");

    let first_arguments = BTreeMap::from([("last".to_owned(), Value::Bool(false))]);
    let first = run(
        &single,
        &single_index(),
        &LastValidAggregator,
        &first_arguments,
    );
    assert_eq!(value::<Float64Array>(&first, "a").value(0), 2.0);
    assert_eq!(value::<StringArray>(&first, "text").value(0), "two");

    let grouped = batch(vec![
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(vec![
                timestamp(2),
                timestamp(2),
                timestamp(1),
                timestamp(1),
            ])),
        ),
        (
            "instrument",
            Arc::new(StringArray::from(vec!["B", "A", "A", "B"])),
        ),
        (
            "value",
            Arc::new(Float64Array::from(vec![
                Some(2.0),
                None,
                Some(1.0),
                Some(4.0),
            ])),
        ),
    ]);
    let output = run(
        &grouped,
        &multi_index(),
        &LastValidAggregator,
        &BTreeMap::new(),
    );
    assert_eq!(
        value::<StringArray>(&output, "instrument")
            .iter()
            .collect::<Vec<_>>(),
        [Some("A"), Some("B")]
    );
    assert_eq!(
        value::<Float64Array>(&output, "value").values(),
        &[1.0, 2.0]
    );
}

#[test]
#[allow(
    clippy::float_cmp,
    clippy::too_many_lines,
    reason = "all numeric Arrow widths share the compounded-return contract matrix"
)]
fn compounded_return_uses_arrow_arithmetic_for_every_numeric_width() {
    let input = batch(vec![
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(vec![
                timestamp(1),
                timestamp(2),
            ])),
        ),
        ("bool", Arc::new(BooleanArray::from(vec![true, false]))),
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
        (
            "f64",
            Arc::new(Float64Array::from(vec![Some(f64::NAN), None])),
        ),
    ]);
    let output = run(
        &input,
        &single_index(),
        &CompoundedReturnAggregator,
        &BTreeMap::new(),
    );
    assert_eq!(value::<Int64Array>(&output, "bool").value(0), 2);
    for name in ["i8", "i16", "i32", "i64"] {
        assert_eq!(value::<Int64Array>(&output, name).value(0), 6);
    }
    for name in ["u8", "u16", "u32", "u64"] {
        assert_eq!(value::<UInt64Array>(&output, name).value(0), 6);
    }
    assert_eq!(
        value::<Float16Array>(&output, "f16").value(0),
        f16::from_f32(6.0)
    );
    assert_eq!(value::<Float32Array>(&output, "f32").value(0), 6.0);
    assert_eq!(value::<Float64Array>(&output, "f64").value(0), 1.0);

    let grouped = batch(vec![
        (
            "instrument",
            Arc::new(StringArray::from(vec!["B", "A", "A", "B"])),
        ),
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(vec![
                timestamp(2),
                timestamp(2),
                timestamp(1),
                timestamp(1),
            ])),
        ),
        (
            "value",
            Arc::new(Float64Array::from(vec![
                Some(2.0),
                None,
                Some(1.0),
                Some(4.0),
            ])),
        ),
    ]);
    let grouped_output = run(
        &grouped,
        &multi_index(),
        &CompoundedReturnAggregator,
        &BTreeMap::new(),
    );
    assert_eq!(
        value::<Float64Array>(&grouped_output, "value").values(),
        &[2.0, 15.0]
    );
}

#[derive(Debug)]
struct ProjectionAggregator {
    calls: AtomicUsize,
}

impl ProjectionAggregator {
    fn new() -> Self {
        Self {
            calls: AtomicUsize::new(0),
        }
    }
}

impl TimeSeriesAggregator for ProjectionAggregator {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed("projection")
    }

    fn output_schema(
        &self,
        input: &RecordBatch,
        _index: &TimeSeriesIndex,
        arguments: &AggregationArguments,
    ) -> Result<SchemaRef, AggregatorError> {
        if !matches!(
            arguments.get("mode").and_then(Value::as_str),
            Some("rows" | "empty")
        ) {
            return Err(Box::new(io::Error::other("mode must be rows")));
        }
        Ok(Arc::new(Schema::new(vec![
            input
                .schema_ref()
                .field_with_name("datetime")
                .unwrap()
                .clone(),
            input.schema_ref().field_with_name("value").unwrap().clone(),
        ])))
    }

    fn aggregate_group(
        &self,
        group: &RecordBatch,
        _index: &TimeSeriesIndex,
        arguments: &AggregationArguments,
    ) -> Result<RecordBatch, AggregatorError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if arguments.get("mode") == Some(&Value::String("empty".to_owned())) {
            return Ok(RecordBatch::new_empty(Arc::new(Schema::new(vec![
                group
                    .schema_ref()
                    .field_with_name("datetime")
                    .unwrap()
                    .clone(),
                group.schema_ref().field_with_name("value").unwrap().clone(),
            ]))));
        }
        Ok(batch(vec![
            (
                "datetime",
                group.column_by_name("datetime").unwrap().clone(),
            ),
            ("value", group.column_by_name("value").unwrap().clone()),
        ]))
    }
}

#[test]
fn plugin_contract_supports_kwargs_multi_row_single_and_empty_outputs() {
    let grouped = batch(vec![
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(vec![
                timestamp(2),
                timestamp(2),
                timestamp(1),
                timestamp(1),
            ])),
        ),
        (
            "instrument",
            Arc::new(StringArray::from(vec!["B", "A", "A", "B"])),
        ),
        ("value", Arc::new(Int64Array::from(vec![2, 2, 1, 4]))),
    ]);
    let arguments = BTreeMap::from([("mode".to_owned(), Value::String("rows".to_owned()))]);
    let plugin = ProjectionAggregator::new();
    let output = run(&grouped, &multi_index(), &plugin, &arguments);
    assert_eq!(plugin.calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        value::<StringArray>(&output, "instrument")
            .iter()
            .collect::<Vec<_>>(),
        [Some("A"), Some("A"), Some("B"), Some("B")]
    );
    assert_eq!(
        value::<Int64Array>(&output, "value").values(),
        &[1, 2, 4, 2]
    );

    let single_plugin = ProjectionAggregator::new();
    let single = batch(vec![
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(vec![
                timestamp(2),
                timestamp(1),
            ])),
        ),
        ("value", Arc::new(Int64Array::from(vec![2, 1]))),
    ]);
    let single_output = run(&single, &single_index(), &single_plugin, &arguments);
    assert_eq!(single_plugin.calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        value::<Int64Array>(&single_output, "value").values(),
        &[1, 2]
    );

    let null_groups = batch(vec![
        (
            "instrument",
            Arc::new(StringArray::from(vec![None::<&str>])),
        ),
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(vec![timestamp(1)])),
        ),
        ("value", Arc::new(Int64Array::from(vec![1]))),
    ]);
    let empty_plugin = ProjectionAggregator::new();
    let empty = run(&null_groups, &multi_index(), &empty_plugin, &arguments);
    assert_eq!(empty.num_rows(), 0);
    assert_eq!(empty_plugin.calls.load(Ordering::SeqCst), 0);
    assert_eq!(empty.schema_ref().field(0).name(), "instrument");

    let empty_arguments = BTreeMap::from([("mode".to_owned(), Value::String("empty".to_owned()))]);
    let empty_plugin = ProjectionAggregator::new();
    let empty = run(&grouped, &multi_index(), &empty_plugin, &empty_arguments);
    assert_eq!(empty.num_rows(), 0);
    assert_eq!(empty_plugin.calls.load(Ordering::SeqCst), 2);
    assert_eq!(empty.schema_ref().field(0).name(), "instrument");
}

enum FailureMode {
    Schema,
    Group,
    Reserved,
    Mismatch,
}

struct FailureAggregator(FailureMode);

impl TimeSeriesAggregator for FailureAggregator {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed("failure")
    }

    fn output_schema(
        &self,
        _input: &RecordBatch,
        _index: &TimeSeriesIndex,
        _arguments: &AggregationArguments,
    ) -> Result<SchemaRef, AggregatorError> {
        match self.0 {
            FailureMode::Schema => Err(Box::new(io::Error::other("schema failed"))),
            FailureMode::Reserved => Ok(Arc::new(Schema::new(vec![Field::new(
                "instrument",
                DataType::Utf8,
                false,
            )]))),
            FailureMode::Group | FailureMode::Mismatch => {
                Ok(Arc::new(Schema::new(vec![Field::new(
                    "result",
                    DataType::Int64,
                    false,
                )])))
            }
        }
    }

    fn aggregate_group(
        &self,
        _group: &RecordBatch,
        _index: &TimeSeriesIndex,
        _arguments: &AggregationArguments,
    ) -> Result<RecordBatch, AggregatorError> {
        match self.0 {
            FailureMode::Group => Err(Box::new(io::Error::other("group failed"))),
            FailureMode::Mismatch => Ok(batch(vec![(
                "result",
                Arc::new(Float64Array::from(vec![1.0])),
            )])),
            FailureMode::Schema | FailureMode::Reserved => unreachable!(),
        }
    }
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "all public callable lifecycle errors share one fixture"
)]
fn selection_and_every_plugin_lifecycle_failure_are_typed() {
    let input = batch(vec![
        ("instrument", Arc::new(StringArray::from(vec!["A"]))),
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(vec![timestamp(1)])),
        ),
        ("value", Arc::new(Int64Array::from(vec![1]))),
    ]);
    assert_eq!(AggregatorPhase::Schema.to_string(), "schema");
    assert_eq!(AggregatorPhase::Group.to_string(), "group");
    assert_eq!(
        serde_json::from_str::<AggregatorPhase>("\"group\"").unwrap(),
        AggregatorPhase::Group
    );

    let wrong_index = TimeSeriesIndex::Datetime {
        datetime: "missing".to_owned(),
    };
    assert!(matches!(
        aggregate_time_series_with(
            &input,
            &wrong_index,
            TimeRange::default(),
            &LastValidAggregator,
            &BTreeMap::new()
        ),
        Err(TimeSeriesCallableError::Selection(
            TimeSeriesSelectionError::MissingIndexColumn { column }
        )) if column == "missing"
    ));

    assert!(matches!(
        aggregate_time_series_with(
            &input,
            &multi_index(),
            TimeRange::default(),
            &FailureAggregator(FailureMode::Schema),
            &BTreeMap::new()
        ),
        Err(TimeSeriesCallableError::Aggregator { aggregator, phase: AggregatorPhase::Schema, source })
            if aggregator == "failure" && source.to_string() == "schema failed"
    ));
    assert!(matches!(
        aggregate_time_series_with(
            &input,
            &multi_index(),
            TimeRange::default(),
            &FailureAggregator(FailureMode::Group),
            &BTreeMap::new()
        ),
        Err(TimeSeriesCallableError::Aggregator { aggregator, phase: AggregatorPhase::Group, source })
            if aggregator == "failure" && source.to_string() == "group failed"
    ));
    assert!(matches!(
        aggregate_time_series_with(
            &input,
            &multi_index(),
            TimeRange::default(),
            &FailureAggregator(FailureMode::Reserved),
            &BTreeMap::new()
        ),
        Err(TimeSeriesCallableError::ReservedInstrumentColumn { aggregator, column })
            if aggregator == "failure" && column == "instrument"
    ));
    assert!(matches!(
        aggregate_time_series_with(
            &input,
            &multi_index(),
            TimeRange::default(),
            &FailureAggregator(FailureMode::Mismatch),
            &BTreeMap::new()
        ),
        Err(TimeSeriesCallableError::OutputSchemaMismatch { aggregator, group: 0, expected, actual })
            if aggregator == "failure"
                && expected.field(0).data_type() == &DataType::Int64
                && actual.field(0).data_type() == &DataType::Float64
    ));

    for mode in [FailureMode::Group, FailureMode::Mismatch] {
        assert!(
            aggregate_time_series_with(
                &input,
                &single_index(),
                TimeRange::default(),
                &FailureAggregator(mode),
                &BTreeMap::new()
            )
            .is_err()
        );
    }

    let empty = batch(vec![
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(Vec::<i64>::new())),
        ),
        ("value", Arc::new(Int64Array::from(Vec::<i64>::new()))),
    ]);
    assert!(
        aggregate_time_series_with(
            &empty,
            &single_index(),
            TimeRange::default(),
            &FailureAggregator(FailureMode::Schema),
            &BTreeMap::new()
        )
        .unwrap()
        .is_none()
    );
}

#[test]
fn bundled_adapters_validate_kwargs_types_and_value_schemas() {
    let input = batch(vec![
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(vec![timestamp(1)])),
        ),
        ("value", Arc::new(StringArray::from(vec!["x"]))),
    ]);
    let unknown = BTreeMap::from([("unknown".to_owned(), Value::Bool(true))]);
    assert!(matches!(
        aggregate_time_series_with(
            &input,
            &single_index(),
            TimeRange::default(),
            &LastValidAggregator,
            &unknown
        ),
        Err(TimeSeriesCallableError::Aggregator {
            phase: AggregatorPhase::Schema,
            ..
        })
    ));
    let invalid = BTreeMap::from([("last".to_owned(), Value::String("false".to_owned()))]);
    assert!(
        aggregate_time_series_with(
            &input,
            &single_index(),
            TimeRange::default(),
            &LastValidAggregator,
            &invalid
        )
        .is_err()
    );
    assert!(
        aggregate_time_series_with(
            &input,
            &single_index(),
            TimeRange::default(),
            &CompoundedReturnAggregator,
            &unknown
        )
        .is_err()
    );
    assert!(matches!(
        aggregate_time_series_with(
            &input,
            &single_index(),
            TimeRange::default(),
            &CompoundedReturnAggregator,
            &BTreeMap::new()
        ),
        Err(TimeSeriesCallableError::Aggregator { phase: AggregatorPhase::Schema, source, .. })
            if source.to_string().contains("Utf8")
    ));

    assert!(
        LastValidAggregator
            .aggregate_group(&input, &single_index(), &unknown)
            .is_err()
    );
    assert!(
        CompoundedReturnAggregator
            .aggregate_group(&input, &single_index(), &unknown)
            .is_err()
    );
    assert!(
        CompoundedReturnAggregator
            .aggregate_group(&input, &single_index(), &BTreeMap::new())
            .is_err()
    );
}

#[test]
fn bundled_callable_adapters_match_live_python_source() {
    let utils =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/utils/__init__.py");
    let dataset =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/data/dataset/utils.py");
    let resam = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/utils/resam.py");
    let script = r#"
import ast,json,sys
from functools import partial
from typing import Callable,Union
import numpy as np,pandas as pd
def fn(path,name):
 t=ast.parse(open(path,encoding="utf-8").read(),filename=path)
 return next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)
nodes=[fn(sys.argv[1],"lazy_sort_index"),fn(sys.argv[2],"get_level_index"),fn(sys.argv[3],"resam_ts_data"),fn(sys.argv[3],"get_valid_value"),fn(sys.argv[3],"_ts_data_valid")]
nodes[2].body=[n for n in nodes[2].body if not isinstance(n,ast.ImportFrom)]
m=ast.Module(body=nodes,type_ignores=[]);ast.fix_missing_locations(m)
ns={"pd":pd,"np":np,"Callable":Callable,"Union":Union,"is_deprecated_lexsorted_pandas":True};exec(compile(m,sys.argv[3],"exec"),ns)
last=partial(ns["_ts_data_valid"],last=True)
idx=pd.DatetimeIndex(["2024-01-02","2024-01-01"],name="datetime")
s=pd.Series([2.0,np.nan],index=idx,name="value")
mi=pd.MultiIndex.from_tuples([("B","2024-01-02"),("A","2024-01-02"),("A","2024-01-01"),("B","2024-01-01")],names=["instrument","datetime"])
ms=pd.Series([2.0,np.nan,1.0,4.0],index=mi,name="value")
def vals(v): return [None if pd.isna(x) else x for x in v.tolist()] if isinstance(v,pd.Series) else v
out={
 "single_last":ns["resam_ts_data"](s,method=last),
 "single_first":ns["resam_ts_data"](s,method=last,method_kwargs={"last":False}),
 "single_comp":ns["resam_ts_data"](s,method=lambda x:(x+1).prod()),
 "multi_last":vals(ns["resam_ts_data"](ms,method=last)),
 "multi_comp":vals(ns["resam_ts_data"](ms,method=lambda x:(x+1).prod())),
}
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
            "single_last": 2.0,
            "single_first": 2.0,
            "single_comp": 3.0,
            "multi_last": [1.0, 2.0],
            "multi_comp": [2.0, 15.0]
        })
    );
}
