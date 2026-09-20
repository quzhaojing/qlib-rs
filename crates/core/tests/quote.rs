use std::{path::PathBuf, process::Command, sync::Arc};

use arrow_array::{
    Array, ArrayRef, BooleanArray, Float16Array, Float32Array, Float64Array, Int8Array, Int16Array,
    Int32Array, Int64Array, RecordBatch, StringArray, TimestampNanosecondArray, UInt8Array,
    UInt16Array, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use chrono::NaiveDateTime;
use domain_core::{
    ArrowQuote, BuiltInAggregation, Quote, QuoteData, QuoteError, QuoteMethod, TimeRange,
    TimeSeriesAggregationError, TimeSeriesResampleError, TimeSeriesSelectionError,
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

fn quote_batch() -> RecordBatch {
    batch(vec![
        (
            "instrument",
            Arc::new(StringArray::from(vec![
                Some("B"),
                Some("A"),
                Some("A"),
                None,
            ])),
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
        ("price", Arc::new(Int32Array::from(vec![4, 2, 1, 9]))),
        (
            "flag",
            Arc::new(BooleanArray::from(vec![true, false, true, true])),
        ),
        (
            "valid",
            Arc::new(Float64Array::from(vec![
                Some(4.0),
                Some(3.0),
                None,
                Some(9.0),
            ])),
        ),
        (
            "text",
            Arc::new(StringArray::from(vec!["b", "y", "x", "n"])),
        ),
    ])
}

fn scalar<T: Array + 'static>(data: QuoteData) -> ArrayRef {
    let QuoteData::Scalar(value) = data else {
        panic!("expected scalar quote data");
    };
    assert!(value.as_any().is::<T>());
    value
}

#[test]
#[allow(
    clippy::float_cmp,
    clippy::too_many_lines,
    reason = "integer-valued Float64 SingleData fixtures are exactly representable"
)]
fn arrow_quote_partitions_stocks_and_returns_scalar_or_single_data_shapes() {
    let quote = ArrowQuote::try_new(&quote_batch(), "instrument", "datetime").unwrap();
    let provider: &dyn Quote = &quote;
    assert_eq!(provider.get_all_stock(), ["A", "B"]);

    let raw = provider
        .get_data("A", TimeRange::default(), "price", QuoteMethod::Selection)
        .unwrap()
        .unwrap();
    let QuoteData::Series(raw) = raw else {
        panic!("expected a series");
    };
    assert_eq!(raw.num_columns(), 2);
    assert_eq!(raw.column(1).data_type(), &DataType::Float64);
    assert_eq!(
        raw.column(0)
            .as_any()
            .downcast_ref::<TimestampNanosecondArray>()
            .unwrap()
            .values(),
        &[timestamp(1), timestamp(2)]
    );
    assert_eq!(
        raw.column(1)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .values(),
        &[1.0, 2.0]
    );

    let bool_series = provider
        .get_data("B", TimeRange::default(), "flag", QuoteMethod::Selection)
        .unwrap()
        .unwrap();
    let QuoteData::Series(bool_series) = bool_series else {
        panic!("expected a series");
    };
    assert_eq!(
        bool_series
            .column(1)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(0),
        1.0
    );

    let sum = provider
        .get_data(
            "A",
            TimeRange::default(),
            "price",
            QuoteMethod::BuiltIn(BuiltInAggregation::Sum),
        )
        .unwrap()
        .unwrap();
    let sum = scalar::<Int64Array>(sum);
    assert_eq!(
        sum.as_any().downcast_ref::<Int64Array>().unwrap().value(0),
        3
    );

    let all = provider
        .get_data(
            "A",
            TimeRange::default(),
            "flag",
            QuoteMethod::BuiltIn(BuiltInAggregation::All),
        )
        .unwrap()
        .unwrap();
    assert!(
        !scalar::<BooleanArray>(all)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap()
            .value(0)
    );

    let last = provider
        .get_data("A", TimeRange::default(), "valid", QuoteMethod::LastValid)
        .unwrap()
        .unwrap();
    assert_eq!(
        scalar::<Float64Array>(last)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(0),
        3.0
    );

    assert!(
        provider
            .get_data(
                "A",
                TimeRange {
                    start: Some(
                        NaiveDateTime::parse_from_str("2024-01-03 00:00:00", "%Y-%m-%d %H:%M:%S")
                            .unwrap(),
                    ),
                    end: None,
                },
                "price",
                QuoteMethod::Selection,
            )
            .unwrap()
            .is_none()
    );
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one matrix exercises every public quote error domain"
)]
fn quote_method_parsing_constructor_and_query_failures_are_typed() {
    assert_eq!(QuoteMethod::default(), QuoteMethod::Selection);
    assert_eq!(
        QuoteMethod::from_python_name(None).unwrap(),
        QuoteMethod::Selection
    );
    assert_eq!(
        QuoteMethod::from_python_name(Some("ts_data_last")).unwrap(),
        QuoteMethod::LastValid
    );
    assert_eq!(
        QuoteMethod::from_python_name(Some("mean")).unwrap(),
        QuoteMethod::BuiltIn(BuiltInAggregation::Mean)
    );
    for (method, name) in [
        (QuoteMethod::Selection, None),
        (QuoteMethod::BuiltIn(BuiltInAggregation::All), Some("all")),
        (QuoteMethod::BuiltIn(BuiltInAggregation::Sum), Some("sum")),
        (QuoteMethod::BuiltIn(BuiltInAggregation::Mean), Some("mean")),
        (
            QuoteMethod::BuiltIn(BuiltInAggregation::Product),
            Some("prod"),
        ),
        (
            QuoteMethod::BuiltIn(BuiltInAggregation::First),
            Some("first"),
        ),
        (QuoteMethod::BuiltIn(BuiltInAggregation::Last), Some("last")),
        (QuoteMethod::LastValid, Some("ts_data_last")),
    ] {
        assert_eq!(method.python_name(), name);
    }
    assert!(matches!(
        QuoteMethod::from_python_name(Some("median")),
        Err(QuoteError::UnsupportedMethod { method }) if method == "median"
    ));

    let quote = ArrowQuote::try_new(&quote_batch(), "instrument", "datetime").unwrap();
    assert!(matches!(
        quote.get_data("C", TimeRange::default(), "price", QuoteMethod::Selection),
        Err(QuoteError::MissingStock { stock }) if stock == "C"
    ));
    for field in ["missing", "datetime"] {
        assert!(matches!(
            quote.get_data("A", TimeRange::default(), field, QuoteMethod::Selection),
            Err(QuoteError::MissingField { field: rejected, .. }) if rejected == field
        ));
    }
    assert!(matches!(
        quote.get_data("A", TimeRange::default(), "text", QuoteMethod::Selection),
        Err(QuoteError::UnsupportedSeriesType {
            data_type: DataType::Utf8,
            ..
        })
    ));
    assert!(matches!(
        quote.get_data("A", TimeRange::default(), "text", QuoteMethod::LastValid),
        Err(QuoteError::UnsupportedScalarType {
            data_type: DataType::Utf8
        })
    ));
    assert!(matches!(
        quote.get_data(
            "A",
            TimeRange::default(),
            "price",
            QuoteMethod::BuiltIn(BuiltInAggregation::Last)
        ),
        Err(QuoteError::Resample(TimeSeriesResampleError::BuiltIn(
            TimeSeriesAggregationError::RequiresInstrumentGrouping {
                method: BuiltInAggregation::Last
            }
        )))
    ));

    let invalid_instrument = batch(vec![
        ("instrument", Arc::new(Int64Array::from(vec![1]))),
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(vec![timestamp(1)])),
        ),
        ("price", Arc::new(Int64Array::from(vec![1]))),
    ]);
    assert!(matches!(
        ArrowQuote::try_new(&invalid_instrument, "instrument", "datetime"),
        Err(QuoteError::InvalidInstrumentType {
            data_type: DataType::Int64,
            ..
        })
    ));
    assert!(matches!(
        ArrowQuote::try_new(&quote_batch(), "instrument", "missing"),
        Err(QuoteError::Selection(
            TimeSeriesSelectionError::MissingIndexColumn { column }
        )) if column == "missing"
    ));

    let duplicate_schema = Arc::new(Schema::new(vec![
        Field::new("instrument", DataType::Utf8, false),
        Field::new(
            "datetime",
            DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, None),
            false,
        ),
        Field::new("price", DataType::Int64, false),
        Field::new("price", DataType::Int64, false),
    ]));
    let duplicate = RecordBatch::try_new(
        duplicate_schema,
        vec![
            Arc::new(StringArray::from(vec!["A"])),
            Arc::new(TimestampNanosecondArray::from(vec![timestamp(1)])),
            Arc::new(Int64Array::from(vec![1])),
            Arc::new(Int64Array::from(vec![2])),
        ],
    )
    .unwrap();
    let duplicate = ArrowQuote::try_new(&duplicate, "instrument", "datetime").unwrap();
    assert!(matches!(
        duplicate.get_data("A", TimeRange::default(), "price", QuoteMethod::Selection),
        Err(QuoteError::AmbiguousField { field, .. }) if field == "price"
    ));

    let empty = batch(vec![
        (
            "instrument",
            Arc::new(StringArray::from(Vec::<&str>::new())),
        ),
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(Vec::<i64>::new())),
        ),
        ("price", Arc::new(Int64Array::from(Vec::<i64>::new()))),
    ]);
    let empty = ArrowQuote::try_new(&empty, "instrument", "datetime").unwrap();
    assert!(empty.get_all_stock().is_empty());
}

#[test]
fn every_arrow_numeric_scalar_type_crosses_the_quote_boundary() {
    let input = batch(vec![
        ("instrument", Arc::new(StringArray::from(vec!["A"]))),
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(vec![timestamp(1)])),
        ),
        ("bool", Arc::new(BooleanArray::from(vec![true]))),
        ("i8", Arc::new(Int8Array::from(vec![1]))),
        ("i16", Arc::new(Int16Array::from(vec![1]))),
        ("i32", Arc::new(Int32Array::from(vec![1]))),
        ("i64", Arc::new(Int64Array::from(vec![1]))),
        ("u8", Arc::new(UInt8Array::from(vec![1]))),
        ("u16", Arc::new(UInt16Array::from(vec![1]))),
        ("u32", Arc::new(UInt32Array::from(vec![1]))),
        ("u64", Arc::new(UInt64Array::from(vec![1]))),
        (
            "f16",
            Arc::new(Float16Array::from(vec![f16::from_f32(1.0)])),
        ),
        ("f32", Arc::new(Float32Array::from(vec![1.0]))),
        ("f64", Arc::new(Float64Array::from(vec![1.0]))),
    ]);
    let quote = ArrowQuote::try_new(&input, "instrument", "datetime").unwrap();
    for (field, expected) in [
        ("bool", DataType::Boolean),
        ("i8", DataType::Int8),
        ("i16", DataType::Int16),
        ("i32", DataType::Int32),
        ("i64", DataType::Int64),
        ("u8", DataType::UInt8),
        ("u16", DataType::UInt16),
        ("u32", DataType::UInt32),
        ("u64", DataType::UInt64),
        ("f16", DataType::Float16),
        ("f32", DataType::Float32),
        ("f64", DataType::Float64),
    ] {
        let value = quote
            .get_data("A", TimeRange::default(), field, QuoteMethod::LastValid)
            .unwrap()
            .unwrap();
        let QuoteData::Scalar(value) = value else {
            panic!("expected scalar");
        };
        assert_eq!(value.data_type(), &expected);
    }
}

#[test]
fn arrow_quote_matches_live_pandas_quote_orchestration() {
    let utils =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/utils/__init__.py");
    let dataset =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/data/dataset/utils.py");
    let resam = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/utils/resam.py");
    let quote = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../qlib/qlib/backtest/high_performance_ds.py");
    let script = r#"
import ast,json,logging,sys
from functools import partial
from types import SimpleNamespace
from typing import Callable,Iterable,Optional,Union
import numpy as np,pandas as pd
class IndexData: pass
class SingleData(IndexData):
 def __init__(self,series): self.index=series.index; self.data=np.array(series.values).astype(np.float64)
idd=SimpleNamespace(SingleData=SingleData)
def get_module_logger(*args,**kwargs): return logging.getLogger("probe")
def fn(path,name):
 t=ast.parse(open(path,encoding="utf-8").read(),filename=path)
 return next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)
def cls(path,name):
 t=ast.parse(open(path,encoding="utf-8").read(),filename=path)
 return next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name==name)
nodes=[fn(sys.argv[1],"lazy_sort_index"),fn(sys.argv[2],"get_level_index"),fn(sys.argv[3],"resam_ts_data"),fn(sys.argv[3],"get_valid_value"),fn(sys.argv[3],"_ts_data_valid"),cls(sys.argv[4],"BaseQuote"),cls(sys.argv[4],"PandasQuote")]
nodes[2].body=[n for n in nodes[2].body if not isinstance(n,ast.ImportFrom)]
m=ast.Module(body=nodes,type_ignores=[]);ast.fix_missing_locations(m)
ns=globals();ns["is_deprecated_lexsorted_pandas"]=True;exec(compile(m,sys.argv[4],"exec"),ns)
ns["ts_data_last"]=partial(ns["_ts_data_valid"],last=True)
mi=pd.MultiIndex.from_tuples([("B",pd.Timestamp("2024-01-02")),("A",pd.Timestamp("2024-01-02")),("A",pd.Timestamp("2024-01-01")),(None,pd.Timestamp("2024-01-01"))],names=["instrument","datetime"])
df=pd.DataFrame({"price":[4,2,1,9],"flag":[True,False,True,True],"valid":[4.0,3.0,np.nan,9.0],"text":["b","y","x","n"]},index=mi)
q=ns["PandasQuote"](df,"day")
def attempt(call):
 try: return ["ok",call()]
 except Exception as e: return [type(e).__name__,str(e)]
raw=q.get_data("A",None,None,"price",None)
out={
 "stocks":list(q.get_all_stock()),
 "raw_values":raw.data.tolist(),
 "raw_index":[str(x) for x in raw.index],
 "sum":int(q.get_data("A",None,None,"price","sum")),
 "all":bool(q.get_data("A",None,None,"flag","all")),
 "last_valid":float(q.get_data("A",None,None,"valid","ts_data_last")),
 "empty":q.get_data("A","2024-01-03",None,"price",None),
 "missing_stock":attempt(lambda:q.get_data("C",None,None,"price",None))[0],
 "missing_field":attempt(lambda:q.get_data("A",None,None,"missing",None))[0],
 "string_series":attempt(lambda:q.get_data("A",None,None,"text",None))[0],
 "string_scalar":attempt(lambda:q.get_data("A",None,None,"text","ts_data_last"))[0],
 "last":attempt(lambda:q.get_data("A",None,None,"price","last"))[0],
 "unknown":attempt(lambda:q.get_data("A",None,None,"price","not_a_method"))[0],
}
print(json.dumps(out,sort_keys=True))
"#;
    let output = Command::new(std::env::var_os("PYTHON").unwrap_or_else(|| "python".into()))
        .arg("-c")
        .arg(script)
        .arg(utils)
        .arg(dataset)
        .arg(resam)
        .arg(quote)
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
            "all": false,
            "empty": null,
            "last": "TypeError",
            "last_valid": 3.0,
            "missing_field": "KeyError",
            "missing_stock": "KeyError",
            "raw_index": ["2024-01-01 00:00:00", "2024-01-02 00:00:00"],
            "raw_values": [1.0, 2.0],
            "stocks": ["A", "B"],
            "string_scalar": "ValueError",
            "string_series": "ValueError",
            "sum": 3,
            "unknown": "AttributeError"
        })
    );
}
