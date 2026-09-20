#![allow(
    clippy::float_cmp,
    reason = "Exchange compatibility fixtures assert exact thresholds and Arrow casts"
)]

use std::{collections::VecDeque, path::PathBuf, process::Command, sync::Arc, sync::Mutex};

use arrow_array::{
    ArrayRef, BooleanArray, Float64Array, Int32Array, RecordBatch, StringArray,
    TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
    TimestampSecondArray,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use chrono::NaiveDateTime;
use domain_core::{
    ArrowQuote, BasePriceAggregation, BasePriceConfig, BasePriceDataProvider, BasePriceRequest,
    BasePriceSource, BuiltInAggregation, DealPriceFields, ExchangeQuoteError,
    ExchangeQuoteProvider, Indicator, MarketDataValue, NumpyQuote, OrderDir, Quote, QuoteData,
    QuoteError, QuoteMethod, TimeRange,
};
use serde_json::{Value, json};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Call {
    stock: String,
    field: String,
    method: QuoteMethod,
}

struct ScriptQuote {
    responses: Mutex<VecDeque<Result<Option<QuoteData>, QuoteError>>>,
    calls: Mutex<Vec<Call>>,
}

impl ScriptQuote {
    fn new(responses: Vec<Result<Option<QuoteData>, QuoteError>>) -> Arc<Self> {
        Arc::new(Self {
            responses: Mutex::new(responses.into()),
            calls: Mutex::new(Vec::new()),
        })
    }

    fn calls(&self) -> Vec<Call> {
        self.calls.lock().unwrap().clone()
    }
}

impl Quote for ScriptQuote {
    fn get_all_stock(&self) -> Vec<String> {
        vec!["A".to_owned()]
    }

    fn get_data(
        &self,
        stock: &str,
        _range: TimeRange,
        field: &str,
        method: QuoteMethod,
    ) -> Result<Option<QuoteData>, QuoteError> {
        self.calls.lock().unwrap().push(Call {
            stock: stock.to_owned(),
            field: field.to_owned(),
            method,
        });
        self.responses
            .lock()
            .unwrap()
            .pop_front()
            .expect("the test script supplies one response per expected query")
    }
}

fn exchange(
    responses: Vec<Result<Option<QuoteData>, QuoteError>>,
) -> (ExchangeQuoteProvider, Arc<ScriptQuote>) {
    let quote = ScriptQuote::new(responses);
    let provider =
        ExchangeQuoteProvider::new(quote.clone(), DealPriceFields::directional("$ask", "$bid"));
    (provider, quote)
}

fn scalar(value: f64) -> QuoteData {
    QuoteData::Scalar(Arc::new(Float64Array::from(vec![value])))
}

fn nullable_scalar(value: Option<f64>) -> QuoteData {
    QuoteData::Scalar(Arc::new(Float64Array::from(vec![value])))
}

fn datetime(text: &str) -> NaiveDateTime {
    NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S%.f").unwrap()
}

fn nanos(text: &str) -> i64 {
    datetime(text).and_utc().timestamp_nanos_opt().unwrap()
}

fn raw_series(timestamps: ArrayRef, values: ArrayRef) -> QuoteData {
    QuoteData::Series(
        RecordBatch::try_from_iter(vec![("datetime", timestamps), ("value", values)]).unwrap(),
    )
}

fn f64_value(value: Option<&MarketDataValue>) -> f64 {
    let Some(MarketDataValue::Scalar(value)) = value else {
        panic!("expected scalar market data")
    };
    *value
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one cohesive matrix covers Exchange field selection, fallback, and query failures"
)]
fn exchange_fields_fallback_and_volume_match_python_ordering() {
    let shared = DealPriceFields::shared("close").unwrap();
    assert_eq!(shared.buy(), "$close");
    assert_eq!(shared.sell(), "$close");
    assert_eq!(
        DealPriceFields::shared("$open").unwrap(),
        DealPriceFields::directional("$open", "$open")
    );
    assert!(matches!(
        DealPriceFields::shared(""),
        Err(ExchangeQuoteError::EmptySharedDealPrice)
    ));
    let directional = DealPriceFields::directional("ask", "");
    assert_eq!(directional.buy(), "ask");
    assert_eq!(directional.sell(), "");

    let (provider, quote) = exchange(vec![Ok(None)]);
    let cloned = provider.clone();
    assert_eq!(cloned.fields().buy(), "$ask");
    assert!(
        provider
            .get_deal_price(
                "A",
                TimeRange::default(),
                OrderDir::Sell,
                QuoteMethod::Selection
            )
            .unwrap()
            .is_none()
    );
    assert_eq!(quote.calls()[0].field, "$bid");

    for invalid in [
        None,
        Some(scalar(f64::NAN)),
        Some(scalar(-1.0)),
        Some(scalar(0.0)),
        Some(scalar(1.0e-8)),
        Some(nullable_scalar(None)),
        Some(QuoteData::Scalar(Arc::new(BooleanArray::from(vec![false])))),
    ] {
        let (provider, quote) = exchange(vec![Ok(invalid), Ok(Some(scalar(99.0)))]);
        let result = provider
            .get_deal_price(
                "A",
                TimeRange::default(),
                OrderDir::Buy,
                QuoteMethod::LastValid,
            )
            .unwrap()
            .unwrap();
        let QuoteData::Scalar(result) = result else {
            panic!("expected fallback scalar")
        };
        assert_eq!(
            result
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .value(0),
            99.0
        );
        assert_eq!(
            quote
                .calls()
                .iter()
                .map(|call| call.field.as_str())
                .collect::<Vec<_>>(),
            ["$ask", "$close"]
        );
    }

    for valid in [
        scalar(f64::from_bits(1.0e-8_f64.to_bits() + 1)),
        scalar(f64::INFINITY),
        QuoteData::Scalar(Arc::new(BooleanArray::from(vec![true]))),
    ] {
        let (provider, quote) = exchange(vec![Ok(Some(valid))]);
        assert!(
            provider
                .get_deal_price(
                    "A",
                    TimeRange::default(),
                    OrderDir::Buy,
                    QuoteMethod::BuiltIn(BuiltInAggregation::Last),
                )
                .unwrap()
                .is_some()
        );
        assert_eq!(quote.calls().len(), 1);
    }

    let series = raw_series(
        Arc::new(TimestampNanosecondArray::from(vec![nanos(
            "2024-01-02 09:30:00",
        )])),
        Arc::new(Float64Array::from(vec![1.0])),
    );
    let (provider, _) = exchange(vec![Ok(Some(series))]);
    assert!(matches!(
        provider.get_deal_price(
            "A",
            TimeRange::default(),
            OrderDir::Buy,
            QuoteMethod::LastValid
        ),
        Err(ExchangeQuoteError::AggregatedSeries)
    ));

    for value in [
        Arc::new(Float64Array::from(Vec::<f64>::new())) as ArrayRef,
        Arc::new(Float64Array::from(vec![1.0, 2.0])) as ArrayRef,
    ] {
        let length = value.len();
        let (provider, _) = exchange(vec![Ok(Some(QuoteData::Scalar(value)))]);
        assert!(matches!(
            provider.get_deal_price(
                "A",
                TimeRange::default(),
                OrderDir::Buy,
                QuoteMethod::LastValid
            ),
            Err(ExchangeQuoteError::ScalarLength { length: actual }) if actual == length
        ));
    }
    let (provider, _) = exchange(vec![Ok(Some(QuoteData::Scalar(Arc::new(
        StringArray::from(vec!["bad"]),
    ))))]);
    assert!(matches!(
        provider.get_deal_price(
            "A",
            TimeRange::default(),
            OrderDir::Buy,
            QuoteMethod::LastValid
        ),
        Err(ExchangeQuoteError::ScalarType { data_type }) if data_type == DataType::Utf8
    ));

    let (provider, quote) = exchange(vec![Ok(Some(scalar(7.0)))]);
    assert!(
        provider
            .get_volume(
                "A",
                TimeRange::default(),
                QuoteMethod::BuiltIn(BuiltInAggregation::Sum)
            )
            .unwrap()
            .is_some()
    );
    assert_eq!(quote.calls()[0].field, "$volume");
    let (provider, _) = exchange(vec![Err(QuoteError::MissingStock {
        stock: "A".to_owned(),
    })]);
    assert!(matches!(
        provider.get_volume("A", TimeRange::default(), QuoteMethod::Selection),
        Err(ExchangeQuoteError::Quote(QuoteError::MissingStock { stock })) if stock == "A"
    ));
}

fn quote_batch() -> RecordBatch {
    RecordBatch::try_from_iter(vec![
        (
            "instrument",
            Arc::new(StringArray::from(vec!["A", "A"])) as ArrayRef,
        ),
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(vec![
                nanos("2024-01-02 09:30:00"),
                nanos("2024-01-02 09:31:00"),
            ])) as ArrayRef,
        ),
        (
            "$ask",
            Arc::new(Float64Array::from(vec![Some(10.0), None])) as ArrayRef,
        ),
        (
            "$bid",
            Arc::new(Float64Array::from(vec![9.0, 11.0])) as ArrayRef,
        ),
        (
            "$close",
            Arc::new(Float64Array::from(vec![8.0, 12.0])) as ArrayRef,
        ),
        (
            "$volume",
            Arc::new(Float64Array::from(vec![100.0, 200.0])) as ArrayRef,
        ),
    ])
    .unwrap()
}

fn assert_real_quote_provider(quote: Arc<dyn Quote>) {
    let provider = ExchangeQuoteProvider::new(quote, DealPriceFields::directional("$ask", "$bid"));
    let range = TimeRange {
        start: Some(datetime("2024-01-02 09:30:00")),
        end: Some(datetime("2024-01-02 09:31:00")),
    };
    let buy = BasePriceDataProvider::deal_price(&provider, "A", range, OrderDir::Buy)
        .unwrap()
        .unwrap();
    let MarketDataValue::Series(buy) = buy else {
        panic!("expected raw price series")
    };
    assert_eq!(
        buy.timestamps(),
        [
            datetime("2024-01-02 09:30:00"),
            datetime("2024-01-02 09:31:00")
        ]
    );
    assert_eq!(buy.values()[0], 10.0);
    assert!(buy.values()[1].is_nan());
    let sell = BasePriceDataProvider::deal_price(&provider, "A", range, OrderDir::Sell)
        .unwrap()
        .unwrap();
    let MarketDataValue::Series(sell) = sell else {
        panic!("expected raw sell series")
    };
    assert_eq!(sell.values(), [9.0, 11.0]);
    let volume = BasePriceDataProvider::volume(&provider, "A", range)
        .unwrap()
        .unwrap();
    let MarketDataValue::Series(volume) = volume else {
        panic!("expected raw volume series")
    };
    assert_eq!(volume.values(), [100.0, 200.0]);

    let twap = Indicator::new()
        .get_base_volume_price(
            BasePriceRequest {
                stock: "A",
                start_time: range.start.unwrap(),
                end_time: range.end.unwrap(),
                direction: OrderDir::Buy,
                trade_range: None,
                config: BasePriceConfig::default(),
            },
            &provider,
        )
        .unwrap()
        .unwrap();
    assert_eq!(twap.base_price, 10.0);
    assert_eq!(twap.base_volume, 1.0);
    let vwap = Indicator::new()
        .get_base_volume_price(
            BasePriceRequest {
                stock: "A",
                start_time: range.start.unwrap(),
                end_time: range.end.unwrap(),
                direction: OrderDir::Sell,
                trade_range: None,
                config: BasePriceConfig {
                    aggregation: BasePriceAggregation::Vwap,
                    source: BasePriceSource::DealPrice,
                },
            },
            &provider,
        )
        .unwrap()
        .unwrap();
    assert_eq!(vwap.base_price, 3100.0 / 300.0);
    assert_eq!(vwap.base_volume, 300.0);
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "conversion matrix covers every timestamp unit and malformed plugin shape"
)]
fn provider_converts_real_quotes_scalars_series_and_every_failure() {
    let batch = quote_batch();
    assert_real_quote_provider(Arc::new(
        ArrowQuote::try_new(&batch, "instrument", "datetime").unwrap(),
    ));
    assert_real_quote_provider(Arc::new(
        NumpyQuote::try_new(&batch, "instrument", "datetime", "1min", "cn").unwrap(),
    ));

    for (data, expected) in [
        (QuoteData::Scalar(Arc::new(Int32Array::from(vec![7]))), 7.0),
        (
            QuoteData::Scalar(Arc::new(BooleanArray::from(vec![true]))),
            1.0,
        ),
    ] {
        let (provider, _) = exchange(vec![Ok(Some(data))]);
        assert_eq!(
            f64_value(
                BasePriceDataProvider::deal_price(
                    &provider,
                    "A",
                    TimeRange::default(),
                    OrderDir::Buy
                )
                .unwrap()
                .as_ref()
            ),
            expected
        );
    }
    let (provider, _) = exchange(vec![Ok(Some(nullable_scalar(None)))]);
    assert!(
        f64_value(
            BasePriceDataProvider::deal_price(&provider, "A", TimeRange::default(), OrderDir::Buy)
                .unwrap()
                .as_ref()
        )
        .is_nan()
    );
    let (provider, _) = exchange(vec![Ok(None)]);
    assert!(
        BasePriceDataProvider::volume(&provider, "A", TimeRange::default())
            .unwrap()
            .is_none()
    );

    let instant = datetime("2024-01-02 09:30:00");
    let timestamp_arrays: Vec<ArrayRef> = vec![
        Arc::new(TimestampSecondArray::from(vec![
            instant.and_utc().timestamp(),
        ])),
        Arc::new(TimestampMillisecondArray::from(vec![
            instant.and_utc().timestamp_millis(),
        ])),
        Arc::new(TimestampMicrosecondArray::from(vec![
            instant.and_utc().timestamp_micros(),
        ])),
        Arc::new(TimestampNanosecondArray::from(vec![
            instant.and_utc().timestamp_nanos_opt().unwrap(),
        ])),
    ];
    for timestamps in timestamp_arrays {
        let series = raw_series(
            timestamps,
            Arc::new(Float64Array::from(vec![Option::<f64>::None])),
        );
        let (provider, _) = exchange(vec![Ok(Some(series))]);
        let value = BasePriceDataProvider::volume(&provider, "A", TimeRange::default())
            .unwrap()
            .unwrap();
        let MarketDataValue::Series(value) = value else {
            panic!("expected series")
        };
        assert_eq!(value.timestamps(), [instant]);
        assert!(value.values()[0].is_nan());
    }

    let bad_shape = QuoteData::Series(
        RecordBatch::try_from_iter(vec![(
            "only",
            Arc::new(Float64Array::from(vec![1.0])) as ArrayRef,
        )])
        .unwrap(),
    );
    let bad_timestamp = raw_series(
        Arc::new(Int32Array::from(vec![1])),
        Arc::new(Float64Array::from(vec![1.0])),
    );
    let bad_value = raw_series(
        Arc::new(TimestampNanosecondArray::from(vec![nanos(
            "2024-01-02 09:30:00",
        )])),
        Arc::new(Int32Array::from(vec![1])),
    );
    let null_timestamp = raw_series(
        Arc::new(TimestampNanosecondArray::from(vec![Option::<i64>::None])),
        Arc::new(Float64Array::from(vec![1.0])),
    );
    let out_of_range = raw_series(
        Arc::new(TimestampSecondArray::from(vec![i64::MAX])),
        Arc::new(Float64Array::from(vec![1.0])),
    );
    for (data, message) in [
        (bad_shape, "exactly two columns"),
        (bad_timestamp, "timestamp column has unsupported"),
        (bad_value, "value column must be Float64"),
        (null_timestamp, "null timestamp"),
        (out_of_range, "outside Chrono's range"),
    ] {
        let (provider, _) = exchange(vec![Ok(Some(data))]);
        let error =
            BasePriceDataProvider::volume(&provider, "A", TimeRange::default()).unwrap_err();
        assert!(error.to_string().contains(message), "{error}");
    }

    let (provider, _) = exchange(vec![Ok(Some(QuoteData::Scalar(Arc::new(
        StringArray::from(vec!["bad"]),
    ))))]);
    assert!(
        BasePriceDataProvider::volume(&provider, "A", TimeRange::default())
            .unwrap_err()
            .to_string()
            .contains("unsupported Arrow type Utf8")
    );
    let (provider, _) = exchange(vec![Err(QuoteError::MissingField {
        stock: "A".to_owned(),
        field: "$volume".to_owned(),
    })]);
    assert!(
        BasePriceDataProvider::volume(&provider, "A", TimeRange::default())
            .unwrap_err()
            .to_string()
            .contains("field $volume is not present")
    );
    let (provider, _) = exchange(vec![Err(QuoteError::MissingField {
        stock: "A".to_owned(),
        field: "$ask".to_owned(),
    })]);
    assert!(
        BasePriceDataProvider::deal_price(&provider, "A", TimeRange::default(), OrderDir::Buy)
            .unwrap_err()
            .to_string()
            .contains("field $ask is not present")
    );

    let (provider, _) = exchange(vec![
        Ok(Some(scalar(0.0))),
        Err(QuoteError::MissingField {
            stock: "A".to_owned(),
            field: "$close".to_owned(),
        }),
    ]);
    assert!(matches!(
        provider.get_deal_price(
            "A",
            TimeRange::default(),
            OrderDir::Buy,
            QuoteMethod::LastValid
        ),
        Err(ExchangeQuoteError::Quote(QuoteError::MissingField { field, .. }))
            if field == "$close"
    ));

    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "datetime",
            DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
            false,
        ),
        Field::new("value", DataType::Float64, false),
    ]));
    let timezone_series = QuoteData::Series(
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(
                    TimestampNanosecondArray::from(vec![nanos("2024-01-02 09:30:00")])
                        .with_timezone("UTC"),
                ),
                Arc::new(Float64Array::from(vec![1.0])),
            ],
        )
        .unwrap(),
    );
    let (provider, _) = exchange(vec![Ok(Some(timezone_series))]);
    assert!(matches!(
        BasePriceDataProvider::volume(&provider, "A", TimeRange::default()).unwrap(),
        Some(MarketDataValue::Series(_))
    ));
}

#[test]
fn exchange_quote_contract_matches_live_python_source() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib");
    let script = r"
import ast,json,sys
from enum import IntEnum
from typing import *
import numpy as np
def cls(path,name):
 t=ast.parse(open(path,encoding='utf-8').read(),filename=path);return next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name==name)
exec(compile(ast.Module(body=[cls(sys.argv[2],'OrderDir')],type_ignores=[]),'live','exec'),globals());exchange=cls(sys.argv[1],'Exchange');methods=[next(n for n in exchange.body if isinstance(n,ast.FunctionDef) and n.name==x) for x in ['get_close','get_volume','get_deal_price']];exec(compile(ast.Module(body=methods,type_ignores=[]),'live','exec'),globals());init=next(n for n in exchange.body if isinstance(n,ast.FunctionDef) and n.name=='__init__');body=[n for n in init.body if 157<=n.lineno<=165];args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='self'),ast.arg(arg='deal_price')],vararg=None,kwonlyargs=[],kw_defaults=[],kwarg=None,defaults=[]);f=ast.FunctionDef(name='configure',args=args,body=body+[ast.Return(value=ast.Tuple(elts=[ast.Attribute(ast.Name('self',ast.Load()),'buy_price',ast.Load()),ast.Attribute(ast.Name('self',ast.Load()),'sell_price',ast.Load())],ctx=ast.Load()))],decorator_list=[]);ast.fix_missing_locations(f);exec(compile(ast.Module(body=[f],type_ignores=[]),'live','exec'),globals())
class L:
 def __init__(self):self.n=0
 def warning(self,_):self.n+=1
class Q:
 def __init__(self,values):self.values=values;self.calls=[]
 def get_data(self,stock,start,end,field,method):self.calls.append([field,method]);return self.values.pop(0)
class X:
 get_close=get_close;get_volume=get_volume;get_deal_price=get_deal_price
 def __init__(self,values):self.buy_price='$ask';self.sell_price='$bid';self.quote=Q(values);self.logger=L()
def config(v):
 x=object.__new__(X)
 try:return ['ok',*configure(x,v)]
 except Exception as e:return [type(e).__name__]
def run(direction,method,values):
 x=X(values)
 try:r=x.get_deal_price('A','s','e',direction,method);result='nan' if isinstance(r,float) and np.isnan(r) else r
 except Exception as e:result=type(e).__name__
 return [result,x.quote.calls,x.logger.n]
out={'config':[config('close'),config('$open'),config(('ask','bid')),config(''),config(3)],'selection':[run(OrderDir.SELL,None,['series']),run(OrderDir.BUY,None,[None])],'fallback':[run(OrderDir.BUY,'ts_data_last',[None,99]),run(OrderDir.BUY,'ts_data_last',[np.nan,99]),run(OrderDir.BUY,'ts_data_last',[1e-8,99]),run(OrderDir.BUY,'ts_data_last',[np.nextafter(1e-8,np.inf)])]};x=X(['volume',7]);out['volume']=[x.get_volume('A','s','e',None),x.get_volume('A','s','e','sum'),x.quote.calls];print(json.dumps(out,separators=(',',':')))
";
    let output = Command::new(std::env::var_os("PYTHON").unwrap_or_else(|| "python".into()))
        .arg("-c")
        .arg(script)
        .arg(root.join("qlib/backtest/exchange.py"))
        .arg(root.join("qlib/backtest/decision.py"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let actual: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        actual["config"],
        json!([
            ["ok", "$close", "$close"],
            ["ok", "$open", "$open"],
            ["ok", "ask", "bid"],
            ["IndexError"],
            ["NotImplementedError"]
        ])
    );
    assert_eq!(
        actual["selection"],
        json!([["series", [["$bid", null]], 0], [null, [["$ask", null]], 0]])
    );
    assert_eq!(actual["fallback"][0][0], json!(99));
    assert_eq!(actual["fallback"][1][0], json!(99));
    assert_eq!(actual["fallback"][2][0], json!(99));
    assert_eq!(actual["fallback"][0][2], json!(2));
    assert_eq!(actual["fallback"][3][1].as_array().unwrap().len(), 1);
    assert_eq!(
        actual["volume"],
        json!(["volume", 7, [["$volume", null], ["$volume", "sum"]]])
    );
}
