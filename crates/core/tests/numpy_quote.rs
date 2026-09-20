use std::{path::PathBuf, process::Command, sync::Arc};

use arrow_array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int32Array, RecordBatch, StringArray,
    TimestampNanosecondArray,
};
use arrow_schema::DataType;
use chrono::NaiveDateTime;
use domain_core::{
    BuiltInAggregation, FrequencyError, FrequencyUnit, NUMPY_QUOTE_CACHE_CAPACITY, NumpyQuote,
    Quote, QuoteData, QuoteError, QuoteMethod, TimeRange,
};
use serde_json::{Value, json};

fn timestamp(value: &str) -> i64 {
    datetime(value).and_utc().timestamp_nanos_opt().unwrap()
}

fn datetime(value: &str) -> NaiveDateTime {
    NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S").unwrap()
}

fn range(start: &str, end: &str) -> TimeRange {
    TimeRange {
        start: Some(datetime(start)),
        end: Some(datetime(end)),
    }
}

fn batch(columns: Vec<(&str, ArrayRef)>) -> RecordBatch {
    RecordBatch::try_from_iter(columns).unwrap()
}

fn quote_batch() -> RecordBatch {
    batch(vec![
        (
            "instrument",
            Arc::new(StringArray::from(vec!["A", "A", "A", "B"])),
        ),
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(vec![
                timestamp("2024-01-02 09:30:00"),
                timestamp("2024-01-02 09:31:00"),
                timestamp("2024-01-02 11:29:00"),
                timestamp("2024-01-02 09:30:00"),
            ])),
        ),
        (
            "value",
            Arc::new(Float64Array::from(vec![
                Some(1.0),
                None,
                Some(7.0),
                Some(4.0),
            ])),
        ),
        (
            "missing",
            Arc::new(Float64Array::from(vec![None, None, None, None])),
        ),
        ("flag", Arc::new(Int32Array::from(vec![1, 0, 1, 1]))),
    ])
}

fn scalar(data: Option<QuoteData>) -> ArrayRef {
    let Some(QuoteData::Scalar(value)) = data else {
        panic!("expected scalar quote data");
    };
    value
}

fn f64_scalar(data: Option<QuoteData>) -> f64 {
    scalar(data)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .value(0)
}

#[test]
#[allow(clippy::float_cmp, reason = "fixture values are exactly representable")]
fn numpy_quote_reproduces_float64_selection_and_every_aggregation() {
    let quote =
        NumpyQuote::try_new(&quote_batch(), "instrument", "datetime", "5min", "cn").unwrap();
    assert_eq!(quote.get_all_stock(), ["A", "B"]);
    let slow = range("2024-01-02 09:30:00", "2024-01-02 09:31:00");

    let Some(QuoteData::Series(raw)) = quote.get_data_str("A", slow, "value", None).unwrap() else {
        panic!("expected series");
    };
    assert_eq!(raw.column(1).data_type(), &DataType::Float64);
    let raw = raw
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(raw.value(0), 1.0);
    assert!(raw.value(1).is_nan());
    assert_eq!(
        f64_scalar(quote.get_data_str("A", slow, "value", Some("sum")).unwrap()),
        1.0
    );
    assert_eq!(
        f64_scalar(
            quote
                .get_data_str("A", slow, "value", Some("mean"))
                .unwrap()
        ),
        1.0
    );
    assert!(
        f64_scalar(
            quote
                .get_data_str("A", slow, "value", Some("last"))
                .unwrap()
        )
        .is_nan()
    );
    assert_eq!(
        f64_scalar(
            quote
                .get_data_str("A", slow, "value", Some("ts_data_last"))
                .unwrap()
        ),
        1.0
    );
    assert!(
        !scalar(quote.get_data_str("A", slow, "flag", Some("all")).unwrap())
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap()
            .value(0)
    );
    assert!(
        f64_scalar(
            quote
                .get_data_str("A", slow, "missing", Some("mean"))
                .unwrap()
        )
        .is_nan()
    );
    assert!(
        quote
            .get_data_str("A", slow, "missing", Some("ts_data_last"))
            .unwrap()
            .is_none()
    );

    let empty = range("2024-01-03 09:30:00", "2024-01-03 09:31:00");
    assert!(
        quote
            .get_data_str("A", empty, "value", Some("last"))
            .unwrap()
            .is_none()
    );
    assert!(
        quote
            .get_data_str("A", empty, "value", Some("ts_data_last"))
            .unwrap()
            .is_none()
    );
}

#[test]
#[allow(clippy::float_cmp, reason = "fixture values are exactly representable")]
#[allow(
    clippy::too_many_lines,
    reason = "one behavior matrix keeps Python's evaluation order auditable"
)]
fn numpy_quote_preserves_fast_path_ordering_and_typed_failures() {
    let quote =
        NumpyQuote::try_new(&quote_batch(), "instrument", "datetime", "5min", "cn").unwrap();
    let fast = range("2024-01-02 09:30:00", "2024-01-02 09:30:30");
    assert_eq!(
        f64_scalar(
            quote
                .get_data_str("A", fast, "value", Some("bogus"))
                .unwrap()
        ),
        1.0
    );
    assert!(
        quote
            .get_data_str("A", fast, "absent", Some("bogus"))
            .unwrap()
            .is_none()
    );
    assert!(
        quote
            .get_data_str(
                "A",
                range("2024-01-02 09:32:00", "2024-01-02 09:32:30"),
                "value",
                None,
            )
            .unwrap()
            .is_none()
    );
    assert_eq!(
        f64_scalar(
            quote
                .get_data_str(
                    "A",
                    range("2024-01-02 11:29:00", "2024-01-02 15:00:00"),
                    "value",
                    Some("bogus"),
                )
                .unwrap()
        ),
        7.0
    );
    let maximum = chrono::NaiveDate::MAX.and_hms_opt(0, 0, 0).unwrap();
    assert!(
        quote
            .get_data_str(
                "A",
                TimeRange {
                    start: Some(maximum),
                    end: Some(maximum),
                },
                "value",
                None,
            )
            .is_err()
    );

    let duplicate_time = batch(vec![
        ("instrument", Arc::new(StringArray::from(vec!["A", "A"]))),
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(vec![
                timestamp("2024-01-02 09:30:00"),
                timestamp("2024-01-02 09:30:00"),
            ])),
        ),
        ("value", Arc::new(Int32Array::from(vec![1, 2]))),
    ]);
    let duplicate =
        NumpyQuote::try_new(&duplicate_time, "instrument", "datetime", "min", "cn").unwrap();
    assert!(matches!(
        duplicate
            .get_data_str("A", fast, "value", Some("bogus"))
            .unwrap(),
        Some(QuoteData::Series(series)) if series.num_rows() == 2
    ));

    let slow = range("2024-01-02 09:30:00", "2024-01-02 09:31:00");
    assert!(matches!(
        quote.get_data_str("A", slow, "value", Some("bogus")),
        Err(QuoteError::UnsupportedMethod { method }) if method == "bogus"
    ));
    assert!(matches!(
        quote.get_data_str("A", slow, "absent", None),
        Err(QuoteError::MissingField { field, .. }) if field == "absent"
    ));
    assert!(matches!(
        quote.get_data_str("A", slow, "absent", Some("last")),
        Err(QuoteError::MissingField { field, .. }) if field == "absent"
    ));
    assert!(matches!(
        quote.get_data_str("A", slow, "absent", Some("ts_data_last")),
        Err(QuoteError::MissingField { field, .. }) if field == "absent"
    ));
    assert!(
        quote
            .get_data_str("Z", TimeRange::default(), "value", Some("bogus"))
            .unwrap()
            .is_none()
    );
    assert!(matches!(
        quote.get_data_str(
            "A",
            TimeRange {
                start: None,
                end: slow.end
            },
            "value",
            None,
        ),
        Err(QuoteError::MissingRangeBound { bound: "start" })
    ));
    assert!(matches!(
        quote.get_data_str(
            "A",
            TimeRange {
                start: slow.start,
                end: None
            },
            "value",
            None,
        ),
        Err(QuoteError::MissingRangeBound { bound: "end" })
    ));

    let invalid_region =
        NumpyQuote::try_new(&quote_batch(), "instrument", "datetime", "day", "xx").unwrap();
    assert!(
        invalid_region
            .get_data_str("Z", TimeRange::default(), "value", None)
            .unwrap()
            .is_none()
    );
    assert!(matches!(
        invalid_region.get_data_str("A", slow, "value", None),
        Err(QuoteError::UnsupportedRegion { region }) if region == "xx"
    ));

    let provider: &dyn Quote = &quote;
    for method in [
        QuoteMethod::BuiltIn(BuiltInAggregation::Product),
        QuoteMethod::BuiltIn(BuiltInAggregation::First),
    ] {
        assert!(matches!(
            provider.get_data("A", slow, "value", method),
            Err(QuoteError::UnsupportedMethod { .. })
        ));
    }
    assert!(
        f64_scalar(
            provider
                .get_data(
                    "A",
                    slow,
                    "value",
                    QuoteMethod::BuiltIn(BuiltInAggregation::Last),
                )
                .unwrap(),
        )
        .is_nan()
    );
    assert_eq!(
        f64_scalar(
            provider
                .get_data("A", slow, "value", QuoteMethod::LastValid)
                .unwrap()
        ),
        1.0
    );
}

#[test]
fn numpy_quote_constructor_and_lru_cache_match_python_boundaries() {
    assert!(NumpyQuote::try_new(&quote_batch(), "instrument", "datetime", "2day", "cn").is_ok());
    assert!(matches!(
        NumpyQuote::try_new(&quote_batch(), "instrument", "datetime", "week", "cn"),
        Err(QuoteError::UnsupportedNumpyFrequency {
            unit: FrequencyUnit::Week,
            ..
        })
    ));
    assert!(matches!(
        NumpyQuote::try_new(&quote_batch(), "instrument", "datetime", "bad!", "cn"),
        Err(QuoteError::Frequency(
            FrequencyError::UnsupportedFormat { .. }
        ))
    ));
    let strings = batch(vec![
        ("instrument", Arc::new(StringArray::from(vec!["A"]))),
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(vec![timestamp(
                "2024-01-02 09:30:00",
            )])),
        ),
        ("bad", Arc::new(StringArray::from(vec!["not-a-number"]))),
    ]);
    assert!(matches!(
        NumpyQuote::try_new(&strings, "instrument", "datetime", "min", "cn"),
        Err(QuoteError::Float64Conversion {
            field,
            data_type: DataType::Utf8,
            ..
        }) if field == "bad"
    ));
    let invalid_instrument = batch(vec![
        ("instrument", Arc::new(Int32Array::from(vec![1]))),
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(vec![timestamp(
                "2024-01-02 09:30:00",
            )])),
        ),
        ("value", Arc::new(Int32Array::from(vec![1]))),
    ]);
    assert!(matches!(
        NumpyQuote::try_new(&invalid_instrument, "instrument", "datetime", "min", "cn"),
        Err(QuoteError::InvalidInstrumentType { .. })
    ));

    let quote = NumpyQuote::try_new(&quote_batch(), "instrument", "datetime", "min", "cn").unwrap();
    let fast = range("2024-01-02 09:30:00", "2024-01-02 09:30:30");
    let first = scalar(quote.get_data_str("A", fast, "value", None).unwrap());
    let second = scalar(quote.get_data_str("A", fast, "value", None).unwrap());
    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(quote.cache_len(), 1);

    assert!(
        quote
            .get_data_str("A", fast, "value", Some("ignored"))
            .is_ok()
    );
    assert_eq!(quote.cache_len(), 2);
    let slow = range("2024-01-02 09:30:00", "2024-01-02 09:31:00");
    assert!(
        quote
            .get_data_str("A", slow, "value", Some("error"))
            .is_err()
    );
    assert_eq!(quote.cache_len(), 2);

    for second in 0..=NUMPY_QUOTE_CACHE_CAPACITY {
        let start = datetime("2024-01-03 09:30:00")
            + chrono::TimeDelta::seconds(i64::try_from(second).unwrap());
        quote
            .get_data_str(
                "A",
                TimeRange {
                    start: Some(start),
                    end: Some(start + chrono::TimeDelta::milliseconds(500)),
                },
                "value",
                None,
            )
            .unwrap();
    }
    assert_eq!(quote.cache_len(), NUMPY_QUOTE_CACHE_CAPACITY);
}

#[test]
fn numpy_quote_contract_matches_live_python_source() {
    let time = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/utils/time.py");
    let quote = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../qlib/qlib/backtest/high_performance_ds.py");
    let script = r#"
import ast,json,logging,re,sys
from functools import lru_cache
from types import SimpleNamespace
from typing import Dict,List,Optional,Tuple,Union
import numpy as np,pandas as pd
REG_CN,REG_TW,REG_US="cn","tw","us"
def get_module_logger(*args,**kwargs): return logging.getLogger("probe")
def node(path,kind,name):
 tree=ast.parse(open(path,encoding="utf-8").read(),filename=path)
 return next(n for n in tree.body if isinstance(n,kind) and n.name==name)
nodes=[node(sys.argv[1],ast.FunctionDef,"is_single_value"),node(sys.argv[1],ast.ClassDef,"Freq"),node(sys.argv[2],ast.ClassDef,"BaseQuote"),node(sys.argv[2],ast.ClassDef,"NumpyQuote")]
module=ast.Module(body=nodes,type_ignores=[]);ast.fix_missing_locations(module)
class Locator:
 def __init__(self,owner,positional=False): self.owner,self.positional=owner,positional
 def __getitem__(self,key):
  out=(self.owner.obj.iloc if self.positional else self.owner.obj.loc)[key]
  return SingleData(out) if isinstance(out,pd.Series) else out
class SingleData:
 def __init__(self,obj): self.obj=obj;self.data=np.asarray(obj.values);self.index=obj.index
 @property
 def loc(self): return Locator(self)
 @property
 def iloc(self): return Locator(self,True)
 @property
 def empty(self): return self.obj.empty
 def __array__(self,dtype=None,copy=None): return np.asarray(self.data,dtype=dtype)
 def __getitem__(self,key): return self.obj.iloc[key] if isinstance(key,int) else self.obj[key]
 def __len__(self): return len(self.obj)
 def all(self): return self.obj.all()
 def isna(self): return SingleData(self.obj.isna())
class MultiData:
 def __init__(self,obj): self.obj=obj.astype(np.float64)
 @property
 def loc(self): return Locator(self)
 def sort_index(self): self.obj.sort_index(inplace=True)
idd=SimpleNamespace(MultiData=MultiData)
ns=globals();exec(compile(module,sys.argv[2],"exec"),ns)
mi=pd.MultiIndex.from_tuples([("A",pd.Timestamp("2024-01-02 09:30")),("A",pd.Timestamp("2024-01-02 09:31")),("A",pd.Timestamp("2024-01-02 11:29"))],names=["instrument","datetime"])
df=pd.DataFrame({"value":[1.0,np.nan,7.0],"missing":[np.nan,np.nan,np.nan]},index=mi)
q=ns["NumpyQuote"](df,"5min","cn")
bad=ns["NumpyQuote"](df,"day","xx")
def attempt(call):
 try: return ["ok",call()]
 except Exception as e: return [type(e).__name__,str(e)]
slow=(pd.Timestamp("2024-01-02 09:30"),pd.Timestamp("2024-01-02 09:31"))
out={
 "stocks":list(q.get_all_stock()),"stored_freq":str(q.freq),
 "fast_unknown":q.get_data("A",slow[0],slow[0]+pd.Timedelta(seconds=30),"value","bogus"),
 "cn_close":q.get_data("A",pd.Timestamp("2024-01-02 11:29"),pd.Timestamp("2024-01-02 15:00"),"value","bogus"),
 "sum":q.get_data("A",*slow,"value","sum"),"mean":q.get_data("A",*slow,"value","mean"),
 "last_nan":bool(np.isnan(q.get_data("A",*slow,"value","last"))),
 "last_valid":q.get_data("A",*slow,"value","ts_data_last"),
 "all_missing":q.get_data("A",*slow,"missing","ts_data_last"),
 "missing_stock":q.get_data("Z",None,None,"value","bogus"),
 "bad_region_missing":bad.get_data("Z",None,None,"value",None),
 "bad_region_existing":attempt(lambda:bad.get_data("A",*slow,"value",None))[0],
 "slow_unknown":attempt(lambda:q.get_data("A",*slow,"value","bogus"))[0],
 "week_ctor":attempt(lambda:ns["NumpyQuote"](df,"week","cn"))[0],
 "cache_max":q.get_data.cache_info().maxsize,
}
print(json.dumps(out,sort_keys=True))
"#;
    let output = Command::new(std::env::var_os("PYTHON").unwrap_or_else(|| "python".into()))
        .arg("-c")
        .arg(script)
        .arg(time)
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
            "all_missing": null,
            "bad_region_existing": "NotImplementedError",
            "bad_region_missing": null,
            "cache_max": 512,
            "cn_close": 7.0,
            "fast_unknown": 1.0,
            "last_nan": true,
            "last_valid": 1.0,
            "mean": 1.0,
            "missing_stock": null,
            "slow_unknown": "ValueError",
            "stocks": ["A"],
            "stored_freq": "0 days 00:01:00",
            "sum": 1.0,
            "week_ctor": "ValueError"
        })
    );
}
