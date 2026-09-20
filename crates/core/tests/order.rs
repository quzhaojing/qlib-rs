use std::{path::PathBuf, process::Command, str::FromStr};

use chrono::{NaiveDate, NaiveDateTime};
use domain_core::{
    DenseIndicatorTransform, DenseIndicatorValue, DenseMetric, NumpyOrderIndicator, Order,
    OrderDir, OrderError, ParseOrderDirectionTransform, SingleData, transfer_dense,
};
use serde_json::{Value, json};
use strum::VariantArray;

fn timestamp(
    year: i32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
    nanos: u32,
) -> NaiveDateTime {
    NaiveDate::from_ymd_opt(year, month, day)
        .unwrap()
        .and_hms_nano_opt(hour, minute, second, nanos)
        .unwrap()
}

#[test]
fn direction_values_parsers_and_dense_array_rules_match_python() {
    assert_eq!(OrderDir::VARIANTS, &[OrderDir::Sell, OrderDir::Buy]);
    assert_eq!(OrderDir::Sell.value(), 0);
    assert_eq!(OrderDir::Buy.value(), 1);
    assert_eq!(OrderDir::Sell.sign(), -1);
    assert_eq!(OrderDir::Buy.sign(), 1);
    assert_eq!(OrderDir::Sell.to_string(), "sell");
    assert_eq!(OrderDir::Buy.as_ref(), "buy");
    assert_eq!(OrderDir::from_str("BUY").unwrap(), OrderDir::Buy);
    assert_eq!(OrderDir::parse_text(" sell ").unwrap(), OrderDir::Sell);
    assert!(matches!(
        OrderDir::parse_text("hold"),
        Err(OrderError::UnsupportedDirection(value)) if value == "hold"
    ));
    assert_eq!(OrderDir::try_from(0).unwrap(), OrderDir::Sell);
    assert_eq!(OrderDir::try_from(1).unwrap(), OrderDir::Buy);
    assert!(matches!(
        OrderDir::try_from(2),
        Err(OrderError::InvalidDirectionCode(2))
    ));

    for value in [1.0, f64::INFINITY] {
        assert_eq!(OrderDir::parse_number(value), OrderDir::Buy);
    }
    for value in [0.0, -0.0, -1.0, f64::NEG_INFINITY, f64::NAN] {
        assert_eq!(OrderDir::parse_number(value), OrderDir::Sell);
    }
    let parsed = OrderDir::parse_values(&[2.0, 0.0, -1.0, f64::NAN]);
    assert_eq!(&parsed[..3], &[1.0, 0.0, 0.0]);
    assert!(parsed[3].is_nan());
}

#[test]
#[allow(clippy::float_cmp)]
fn order_state_deltas_keys_and_resets_are_typed() {
    let start = timestamp(2024, 1, 2, 12, 34, 56, 123_456_789);
    let end = timestamp(2024, 1, 2, 15, 0, 0, 0);
    let mut buy = Order::new("A", 10.0, Order::BUY, Some(start), Some(end));
    assert_eq!(Order::SELL, OrderDir::Sell);
    assert_eq!(buy.stock_id(), "A");
    assert_eq!(buy.amount(), 10.0);
    assert_eq!(buy.direction(), OrderDir::Buy);
    assert_eq!(buy.start_time(), Some(start));
    assert_eq!(buy.end_time(), Some(end));
    assert_eq!(buy.deal_amount(), 0.0);
    assert_eq!(buy.factor(), None);
    assert_eq!(buy.sign(), 1);
    assert_eq!(buy.amount_delta(), 10.0);
    assert_eq!(buy.deal_amount_delta(), 0.0);

    buy.set_deal_amount(4.0);
    buy.set_factor(Some(2.0));
    assert_eq!(buy.deal_amount(), 4.0);
    assert_eq!(buy.factor(), Some(2.0));
    assert_eq!(buy.deal_amount_delta(), 4.0);
    assert_eq!(buy.key(), ("A", Some(start), Some(end), OrderDir::Buy));
    let day = timestamp(2024, 1, 2, 0, 0, 0, 123_456_789);
    assert_eq!(buy.day_timestamp().unwrap(), day);
    assert_eq!(buy.key_by_day().unwrap(), ("A", day, OrderDir::Buy));

    buy.reset_results();
    assert_eq!(buy.deal_amount(), 0.0);
    assert_eq!(buy.factor(), None);
    let sell = Order::try_new("B", -2.0, 0, Some(start), Some(end)).unwrap();
    assert_eq!(sell.sign(), -1);
    assert_eq!(sell.amount_delta(), 2.0);
    assert_eq!(sell.deal_amount_delta().to_bits(), (-0.0_f64).to_bits());
    assert!(matches!(
        Order::try_new("X", 1.0, 2, Some(start), Some(end)),
        Err(OrderError::InvalidDirectionCode(2))
    ));
    let missing = Order::new("X", 1.0, OrderDir::Sell, None, None);
    assert!(matches!(
        missing.day_timestamp(),
        Err(OrderError::MissingStartTime)
    ));
    assert!(matches!(
        missing.key_by_day(),
        Err(OrderError::MissingStartTime)
    ));

    assert_eq!(
        OrderError::UnsupportedDirection("hold".to_owned()).to_string(),
        "direction not supported: hold"
    );
    assert_eq!(
        OrderError::InvalidDirectionCode(3).to_string(),
        "direction code must be 0 (sell) or 1 (buy), got 3"
    );
    assert_eq!(
        OrderError::MissingStartTime.to_string(),
        "order start time is required for its day key"
    );
}

struct DuplicateMetric;

impl DenseMetric for DuplicateMetric {
    fn index(&self) -> &[String] {
        static INDEX: std::sync::LazyLock<Vec<String>> =
            std::sync::LazyLock::new(|| vec!["x".to_owned(), "x".to_owned()]);
        &INDEX
    }

    fn values(&self) -> &[f64] {
        &[1.0, -1.0]
    }
}

#[test]
fn parse_direction_transform_integrates_with_dense_indicators() {
    let transform = ParseOrderDirectionTransform::new("trade_dir");
    assert_eq!(transform.input_names(), &["trade_dir".to_owned()]);
    let source = SingleData::from_f64([
        ("a", Some(2.0)),
        ("b", Some(0.0)),
        ("c", Some(-1.0)),
        ("d", None),
    ])
    .unwrap();
    let result = transform.apply(&[&source]).unwrap();
    let DenseIndicatorValue::Metric(result) = result else {
        panic!("expected metric transform result")
    };
    assert_eq!(result.values()[..3], [1.0, 0.0, 0.0]);
    assert!(result.values()[3].is_nan());
    assert_eq!(
        transform.apply(&[]).unwrap_err(),
        "Order.parse_dir expects one dense metric, got 0"
    );
    assert_eq!(
        transform.apply(&[&source, &source]).unwrap_err(),
        "Order.parse_dir expects one dense metric, got 2"
    );
    assert!(
        transform
            .apply(&[&DuplicateMetric])
            .unwrap_err()
            .contains("duplicate")
    );

    let mut indicator = NumpyOrderIndicator::new();
    indicator.assign("trade_dir", source);
    assert!(
        transfer_dense(&mut indicator, &transform, Some("parsed"))
            .unwrap()
            .is_none()
    );
    assert_eq!(
        indicator.get_index_data("parsed").values()[..3],
        [1.0, 0.0, 0.0]
    );
}

#[test]
fn contract_matches_live_python_order_classes() {
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/backtest/decision.py");
    let script = r"
import ast,json,sys
from dataclasses import dataclass
from enum import IntEnum
from typing import *
import numpy as np,pandas as pd
t=ast.parse(open(sys.argv[1],encoding='utf-8').read(),filename=sys.argv[1])
nodes=[next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name==name) for name in ['OrderDir','Order']]
m=ast.Module(body=nodes,type_ignores=[]);ast.fix_missing_locations(m);exec(compile(m,sys.argv[1],'exec'),globals())
def iso(v):return None if v is None else v.isoformat()
def snap(o):return {'direction':int(o.direction),'direction_type':type(o.direction).__name__,'deal':o.deal_amount,'factor':o.factor,'sign':o.sign,'amount_delta':o.amount_delta,'deal_delta':o.deal_amount_delta,'date':iso(o.date),'key':[o.key[0],iso(o.key[1]),iso(o.key[2]),int(o.key[3])],'day_key':[o.key_by_day[0],iso(o.key_by_day[1]),int(o.key_by_day[2])]}
s=pd.Timestamp('2024-01-02 12:34:56.123456789');e=pd.Timestamp('2024-01-02 15:00:00')
buy=Order('A',10,OrderDir.BUY,s,e,7,3);sell=Order('B',-2,0,s,e,7,3);floating=Order('C',2,1.0,s,e)
buy.deal_amount=4;buy.factor=2
parsed=[]
for value in [OrderDir.SELL,' BUY ','sell',2,0,-2,2.5,0.0,-0.0,float('nan'),float('inf'),float('-inf'),True,False,np.int64(3),np.float32(-1)]:
 result=Order.parse_dir(value);parsed.append([str(value),int(result),type(result).__name__])
array=Order.parse_dir(np.array([[2.0,0.0],[-1.0,np.nan]]))
out={'enum':[int(OrderDir.SELL),int(OrderDir.BUY)],'buy':snap(buy),'sell':snap(sell),'floating_type':type(floating.direction).__name__,'parse':parsed,'array':[[None if np.isnan(v) else float(v) for v in row] for row in array.tolist()]}
for key,call in [('bad_ctor',lambda:Order('X',1,2,s,e)),('bad_text',lambda:Order.parse_dir('hold')),('bad_type',lambda:Order.parse_dir([])),('missing_date',lambda:Order('X',1,0,None,None).date)]:
 try:call()
 except Exception as error:out[key]=type(error).__name__
print(json.dumps(out,sort_keys=True))
";
    let output = Command::new(std::env::var_os("PYTHON").unwrap_or_else(|| "python".into()))
        .arg("-c")
        .arg(script)
        .arg(source)
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
            "array":[[1.0,0.0],[0.0,null]],"bad_ctor":"NotImplementedError",
            "bad_text":"NotImplementedError","bad_type":"NotImplementedError",
            "buy":{"amount_delta":10,"date":"2024-01-02T00:00:00.123456789","day_key":["A","2024-01-02T00:00:00.123456789",1],"deal":4,"deal_delta":4,"direction":1,"direction_type":"OrderDir","factor":2,"key":["A","2024-01-02T12:34:56.123456789","2024-01-02T15:00:00",1],"sign":1},
            "enum":[0,1],"floating_type":"float","missing_date":"AttributeError",
            "parse":[["0",0,"OrderDir"],[" BUY ",1,"OrderDir"],["sell",0,"OrderDir"],["2",1,"OrderDir"],["0",0,"OrderDir"],["-2",0,"OrderDir"],["2.5",1,"OrderDir"],["0.0",0,"OrderDir"],["-0.0",0,"OrderDir"],["nan",0,"OrderDir"],["inf",1,"OrderDir"],["-inf",0,"OrderDir"],["True",1,"OrderDir"],["False",0,"OrderDir"],["3",1,"OrderDir"],["-1.0",0,"OrderDir"]],
            "sell":{"amount_delta":2,"date":"2024-01-02T00:00:00.123456789","day_key":["B","2024-01-02T00:00:00.123456789",0],"deal":0.0,"deal_delta":-0.0,"direction":0,"direction_type":"int","factor":null,"key":["B","2024-01-02T12:34:56.123456789","2024-01-02T15:00:00",0],"sign":-1}
        })
    );
}
