use std::{collections::BTreeMap, path::PathBuf, process::Command, sync::Arc};

use arrow_array::{
    Array, ArrayRef, Int64Array, RecordBatch, StringArray, TimestampNanosecondArray,
};
use arrow_schema::DataType;
use chrono::NaiveDateTime;
use domain_core::{
    AggregatorPhase, BuiltInAggregation, LastValidAggregator, TimeRange,
    TimeSeriesAggregationError, TimeSeriesCallableError, TimeSeriesIndex, TimeSeriesIndexOrder,
    TimeSeriesMethod, TimeSeriesResampleError, TimeSeriesSelectionError, resample_time_series,
};
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
        order: TimeSeriesIndexOrder::InstrumentDatetime,
    }
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
fn unified_dispatch_selects_once_and_routes_every_method_variant() {
    let input = batch(vec![
        (
            "instrument",
            Arc::new(StringArray::from(vec!["B", "A", "A"])),
        ),
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(vec![
                timestamp(2),
                timestamp(2),
                timestamp(1),
            ])),
        ),
        ("value", Arc::new(Int64Array::from(vec![4, 2, 1]))),
    ]);
    let index = multi_index();
    let ignored = BTreeMap::from([("not_used".to_owned(), json!(true))]);
    let selected = resample_time_series(
        &input,
        &index,
        TimeRange::default(),
        TimeSeriesMethod::Selection,
        &ignored,
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        value::<StringArray>(&selected, "instrument")
            .iter()
            .collect::<Vec<_>>(),
        [Some("A"), Some("A"), Some("B")]
    );
    assert_eq!(value::<Int64Array>(&selected, "value").values(), &[1, 2, 4]);

    let sum = resample_time_series(
        &input,
        &index,
        TimeRange::default(),
        TimeSeriesMethod::BuiltIn(BuiltInAggregation::Sum),
        &BTreeMap::new(),
    )
    .unwrap()
    .unwrap();
    assert_eq!(value::<Int64Array>(&sum, "value").values(), &[3, 4]);

    let first = resample_time_series(
        &input,
        &index,
        TimeRange::default(),
        TimeSeriesMethod::Callable(&LastValidAggregator),
        &BTreeMap::from([("last".to_owned(), json!(false))]),
    )
    .unwrap()
    .unwrap();
    assert_eq!(value::<Int64Array>(&first, "value").values(), &[1, 4]);

    let default = resample_time_series(
        &input,
        &index,
        TimeRange::default(),
        TimeSeriesMethod::default(),
        &BTreeMap::new(),
    )
    .unwrap()
    .unwrap();
    assert_eq!(value::<Int64Array>(&default, "value").values(), &[2, 4]);

    assert_eq!(format!("{:?}", TimeSeriesMethod::Selection), "Selection");
    assert_eq!(
        format!("{:?}", TimeSeriesMethod::BuiltIn(BuiltInAggregation::Mean)),
        "BuiltIn(Mean)"
    );
    assert_eq!(
        format!("{:?}", TimeSeriesMethod::Callable(&LastValidAggregator)),
        "Callable(\"ts_data_last\")"
    );
    assert!(matches!(
        TimeSeriesMethod::built_in("sum").unwrap(),
        TimeSeriesMethod::BuiltIn(BuiltInAggregation::Sum)
    ));
    assert!(matches!(
        TimeSeriesMethod::built_in("median"),
        Err(TimeSeriesAggregationError::UnsupportedMethod { method }) if method == "median"
    ));
}

#[test]
fn unified_dispatch_preserves_empty_short_circuit_and_typed_error_domains() {
    let input = batch(vec![
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(vec![timestamp(1)])),
        ),
        ("value", Arc::new(StringArray::from(vec!["x"]))),
    ]);
    let missing_index = TimeSeriesIndex::Datetime {
        datetime: "missing".to_owned(),
    };
    assert!(matches!(
        resample_time_series(
            &input,
            &missing_index,
            TimeRange::default(),
            TimeSeriesMethod::Selection,
            &BTreeMap::new()
        ),
        Err(TimeSeriesResampleError::Selection(
            TimeSeriesSelectionError::MissingIndexColumn { column }
        )) if column == "missing"
    ));
    assert!(matches!(
        resample_time_series(
            &input,
            &single_index(),
            TimeRange::default(),
            TimeSeriesMethod::BuiltIn(BuiltInAggregation::Mean),
            &BTreeMap::new()
        ),
        Err(TimeSeriesResampleError::BuiltIn(
            TimeSeriesAggregationError::UnsupportedDataType {
                data_type: DataType::Utf8,
                ..
            }
        ))
    ));
    assert!(matches!(
        resample_time_series(
            &input,
            &single_index(),
            TimeRange::default(),
            TimeSeriesMethod::Callable(&LastValidAggregator),
            &BTreeMap::from([("unknown".to_owned(), Value::Null)])
        ),
        Err(TimeSeriesResampleError::Callable(
            TimeSeriesCallableError::Aggregator {
                phase: AggregatorPhase::Schema,
                ..
            }
        ))
    ));

    let empty = batch(vec![
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(Vec::<i64>::new())),
        ),
        ("value", Arc::new(Int64Array::from(Vec::<i64>::new()))),
    ]);
    for method in [
        TimeSeriesMethod::Selection,
        TimeSeriesMethod::BuiltIn(BuiltInAggregation::Sum),
        TimeSeriesMethod::Callable(&LastValidAggregator),
    ] {
        assert!(
            resample_time_series(
                &empty,
                &single_index(),
                TimeRange::default(),
                method,
                &BTreeMap::from([("invalid".to_owned(), json!(true))])
            )
            .unwrap()
            .is_none()
        );
    }
}

#[test]
fn unified_method_kwargs_snapshot_matches_live_python_source() {
    let utils =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/utils/__init__.py");
    let dataset =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/data/dataset/utils.py");
    let resam = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/utils/resam.py");
    let script = r#"
import ast,json,sys
from typing import Callable,Union
import numpy as np,pandas as pd
def fn(path,name):
 t=ast.parse(open(path,encoding="utf-8").read(),filename=path)
 return next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)
nodes=[fn(sys.argv[1],"lazy_sort_index"),fn(sys.argv[2],"get_level_index"),fn(sys.argv[3],"resam_ts_data")]
nodes[2].body=[n for n in nodes[2].body if not isinstance(n,ast.ImportFrom)]
m=ast.Module(body=nodes,type_ignores=[]);ast.fix_missing_locations(m)
ns={"pd":pd,"np":np,"Callable":Callable,"Union":Union,"is_deprecated_lexsorted_pandas":True};exec(compile(m,sys.argv[3],"exec"),ns)
idx=pd.DatetimeIndex(["2024-01-02","2024-01-01"],name="datetime")
s=pd.DataFrame({"value":[np.nan,1.0],"flag":[True,False],"text":["x","y"]},index=idx)
mi=pd.MultiIndex.from_tuples([("B","2024-01-01"),("A","2024-01-02"),("A","2024-01-01")],names=["instrument","datetime"])
g=pd.DataFrame({"value":[4.0,np.nan,1.0],"text":["z","y","x"]},index=mi)
def vals(v): return [None if pd.isna(x) else x for x in v.tolist()]
out={
 "selection":vals(ns["resam_ts_data"](s[["value"]],method=None,method_kwargs={"ignored":1})["value"]),
 "numeric_sum":vals(ns["resam_ts_data"](s,method="sum",method_kwargs={"numeric_only":True})),
 "bool_only":vals(ns["resam_ts_data"](s,method="all",method_kwargs={"bool_only":True})),
 "group_sum":vals(ns["resam_ts_data"](g,method="sum",method_kwargs={"numeric_only":True})["value"]),
 "group_first_min2":vals(ns["resam_ts_data"](g,method="first",method_kwargs={"min_count":2})["value"]),
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
            "bool_only": [false],
            "group_first_min2": [null, null],
            "group_sum": [1.0, 4.0],
            "numeric_sum": [1.0, 1.0],
            "selection": [1.0, null]
        })
    );
}
