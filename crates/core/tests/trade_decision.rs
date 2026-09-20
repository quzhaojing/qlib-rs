use std::{
    path::PathBuf,
    process::Command,
    sync::{Arc, Mutex},
};

use chrono::NaiveDateTime;
use domain_core::{
    EMPTY_ORDER_AMOUNT, EmptyTradeDecision, IdxTradeRange, Order, OrderDecision, OrderDir,
    OrderTradeDecision, RangeLimitDefault, SharedTradeRange, TradeCalendarRange,
    TradeCalendarRangeError, TradeDecision, TradeDecisionError, TradeRangeByTime, TradeRangeError,
};
use serde_json::{Value, json};

fn timestamp(text: &str) -> NaiveDateTime {
    NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S%.f").unwrap()
}

fn order(
    stock: &str,
    amount: f64,
    start: Option<NaiveDateTime>,
    end: Option<NaiveDateTime>,
) -> Order {
    Order::new(stock, amount, OrderDir::Buy, start, end)
}

#[test]
fn typed_decisions_fill_times_remain_mutable_and_expose_plugin_steps() {
    let start = timestamp("2024-01-02 09:30:00");
    let end = timestamp("2024-01-02 10:00:00");
    let supplied_start = timestamp("2024-01-02 09:35:00");
    let supplied_end = timestamp("2024-01-02 09:50:00");
    let range: SharedTradeRange = Arc::new(IdxTradeRange::new(2, 5));
    let mut decision = OrderTradeDecision::from_orders(
        vec![
            order("A", 2.0, None, Some(supplied_end)),
            order("B", 0.0, Some(supplied_start), None),
        ],
        start,
        end,
        Some(Arc::clone(&range)),
    );

    assert_eq!(decision.start_time(), start);
    assert_eq!(decision.end_time(), end);
    assert_eq!(decision.total_step(), None);
    assert_eq!(decision.orders()[0].start_time(), Some(start));
    assert_eq!(decision.orders()[0].end_time(), Some(supplied_end));
    assert_eq!(decision.orders()[1].start_time(), Some(supplied_start));
    assert_eq!(decision.orders()[1].end_time(), Some(end));
    assert!(!decision.is_empty());
    assert!(Arc::ptr_eq(decision.shared_trade_range().unwrap(), &range));
    assert_eq!(
        decision.trade_range().unwrap().range_indices(None),
        Ok((2, 5))
    );

    decision.orders_mut()[0].set_deal_amount(1.5);
    assert_eq!(
        decision.items()[0].deal_amount().to_bits(),
        1.5_f64.to_bits()
    );
    let old = decision.replace_orders(vec![order("C", 0.0, None, None)]);
    assert_eq!(old.len(), 2);
    assert_eq!(decision.orders()[0].start_time(), Some(start));
    assert_eq!(decision.orders()[0].end_time(), Some(end));
    assert!(decision.is_empty());

    let plugin: &dyn OrderDecision = &decision;
    assert_eq!(plugin.orders()[0].stock_id(), "C");
    assert!(plugin.is_empty());
    let plugin_step = plugin.base_price_step();
    assert_eq!((plugin_step.start_time, plugin_step.end_time), (start, end));
    assert_eq!(
        plugin_step.trade_range.unwrap().range_indices(None),
        Ok((2, 5))
    );
    let direct_step = decision.base_price_step();
    assert_eq!(direct_step.start_time, start);
    assert_eq!(direct_step.end_time, end);

    let mut generic = TradeDecision::from_items(vec!["opaque".to_owned()], start, end, None);
    generic.items_mut().push("second".to_owned());
    assert_eq!(
        generic.replace_items(vec!["replacement".to_owned()]),
        ["opaque", "second"]
    );
    assert_eq!(generic.items(), ["replacement"]);

    for (amount, expected_empty) in [
        (0.0, true),
        (EMPTY_ORDER_AMOUNT, true),
        (f64::from_bits(EMPTY_ORDER_AMOUNT.to_bits() + 1), false),
        (-2.0, true),
        (f64::NAN, true),
    ] {
        let candidate =
            OrderTradeDecision::from_orders(vec![order("X", amount, None, None)], start, end, None);
        assert_eq!(candidate.is_empty(), expected_empty);
        assert_eq!(OrderDecision::is_empty(&candidate), expected_empty);
    }
}

#[test]
fn empty_decisions_and_range_propagation_preserve_outer_metadata() {
    let start = timestamp("2024-01-02 09:30:00");
    let end = timestamp("2024-01-02 10:00:00");
    let outer_range: SharedTradeRange = Arc::new(IdxTradeRange::new(1, 3));
    let outer =
        OrderTradeDecision::from_orders(Vec::new(), start, end, Some(Arc::clone(&outer_range)));
    let mut inner_missing = OrderTradeDecision::from_orders(Vec::new(), start, end, None);
    let existing_range: SharedTradeRange = Arc::new(IdxTradeRange::new(8, 9));
    let mut inner_existing =
        OrderTradeDecision::from_orders(Vec::new(), start, end, Some(Arc::clone(&existing_range)));

    outer.propagate_trade_range_to(&mut inner_missing);
    outer.propagate_trade_range_to(&mut inner_existing);
    assert!(Arc::ptr_eq(
        inner_missing.shared_trade_range().unwrap(),
        &outer_range
    ));
    assert!(Arc::ptr_eq(
        inner_existing.shared_trade_range().unwrap(),
        &existing_range
    ));

    inner_existing.set_trade_range(None);
    assert!(inner_existing.trade_range().is_none());
    inner_existing.set_trade_range(Some(Arc::clone(&existing_range)));
    assert!(inner_existing.trade_range().is_some());
    inner_existing.set_total_step(10);
    assert_eq!(inner_existing.total_step(), Some(10));
    inner_existing.clear_total_step();
    assert_eq!(inner_existing.total_step(), None);

    let mut empty = EmptyTradeDecision::new(start, end, Some(Arc::new(IdxTradeRange::new(4, 6))));
    assert!(OrderDecision::orders_mut(&mut empty).is_empty());
    assert!(empty.items().is_empty());
    assert!(empty.is_empty());
    assert_eq!(empty.core().start_time(), start);
    empty.core_mut().set_total_step(7);
    assert_eq!(empty.core().total_step(), Some(7));
    let plugin: &dyn OrderDecision = &empty;
    assert!(plugin.orders().is_empty());
    assert!(plugin.is_empty());
    let step = plugin.base_price_step();
    assert_eq!((step.start_time, step.end_time), (start, end));
    assert_eq!(step.trade_range.unwrap().range_indices(None), Ok((4, 6)));
}

struct Calendar {
    start: NaiveDateTime,
    result: Result<(i64, i64), TradeCalendarRangeError>,
    calls: Mutex<usize>,
}

impl TradeCalendarRange for Calendar {
    fn start_time(&self) -> Result<NaiveDateTime, TradeCalendarRangeError> {
        Ok(self.start)
    }

    fn get_range_idx(
        &self,
        _start_time: NaiveDateTime,
        _end_time: NaiveDateTime,
    ) -> Result<(i64, i64), TradeCalendarRangeError> {
        *self.calls.lock().unwrap() += 1;
        self.result.clone()
    }
}

#[test]
fn range_fallback_clipping_and_provider_errors_match_python() {
    let start = timestamp("2024-01-02 09:30:00");
    let end = timestamp("2024-01-02 10:00:00");
    let mut missing = TradeDecision::<String>::from_items(Vec::new(), start, end, None);
    assert_eq!(
        missing.range_limit(None, RangeLimitDefault::Error),
        Err(TradeDecisionError::MissingRange)
    );
    assert_eq!(
        missing.range_limit(None, RangeLimitDefault::Value(Some((4, 7)))),
        Ok(Some((4, 7)))
    );
    assert_eq!(
        missing.range_limit(None, RangeLimitDefault::Value(None)),
        Ok(None)
    );

    let index_range: SharedTradeRange = Arc::new(IdxTradeRange::new(-2, 12));
    missing.set_trade_range(Some(index_range));
    assert_eq!(
        missing.range_limit(None, RangeLimitDefault::Error),
        Ok(Some((-2, 12)))
    );
    missing.set_total_step(5);
    assert_eq!(
        missing.range_limit(None, RangeLimitDefault::Error),
        Ok(Some((0, 4)))
    );

    for (range, total, expected) in [
        ((-2, 3), 5, (0, 3)),
        ((2, 12), 5, (2, 4)),
        ((9, -2), 5, (9, -2)),
        ((-1, 0), 0, (0, -1)),
    ] {
        let mut candidate = TradeDecision::<String>::from_items(
            Vec::new(),
            start,
            end,
            Some(Arc::new(IdxTradeRange::new(range.0, range.1))),
        );
        candidate.set_total_step(total);
        assert_eq!(
            candidate.range_limit(None, RangeLimitDefault::Error),
            Ok(Some(expected))
        );
    }

    let timed: SharedTradeRange = Arc::new(TradeRangeByTime::parse("09:40", "09:50").unwrap());
    let timed_decision = TradeDecision::<String>::from_items(Vec::new(), start, end, Some(timed));
    assert_eq!(
        timed_decision.range_limit(None, RangeLimitDefault::Error),
        Err(TradeDecisionError::MissingRange)
    );
    assert_eq!(
        timed_decision.range_limit(None, RangeLimitDefault::Value(Some((1, 2)))),
        Ok(Some((1, 2)))
    );

    let successful = Calendar {
        start,
        result: Ok((6, 7)),
        calls: Mutex::new(0),
    };
    assert_eq!(
        timed_decision.range_limit(Some(&successful), RangeLimitDefault::Error),
        Ok(Some((6, 7)))
    );
    assert_eq!(*successful.calls.lock().unwrap(), 1);

    let provider_error = TradeCalendarRangeError::Provider {
        message: "offline".to_owned(),
    };
    let failed = Calendar {
        start,
        result: Err(provider_error.clone()),
        calls: Mutex::new(0),
    };
    assert_eq!(
        timed_decision.range_limit(Some(&failed), RangeLimitDefault::Value(Some((1, 2)))),
        Err(TradeDecisionError::TradeRange(TradeRangeError::Calendar(
            provider_error
        )))
    );
}

#[test]
fn order_decision_contract_matches_live_python_source() {
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/backtest/decision.py");
    let script = r"
import ast,json,math,sys
from abc import abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import *
import numpy as np,pandas as pd
p=sys.argv[1];t=ast.parse(open(p,encoding='utf-8').read(),filename=p)
w={'OrderDir','Order','TradeRange','IdxTradeRange','TradeRangeByTime','BaseTradeDecision','EmptyTradeDecision','TradeDecisionWO'}
n=[t.body[0]]+[x for x in t.body if isinstance(x,ast.ClassDef) and x.name in w]
DecisionType=TypeVar('DecisionType');BaseStrategy=TradeCalendarManager=object
def concat_date_time(d,c):return pd.Timestamp.combine(d,c)
warnings=[]
class Logger:
 def warning(self,msg):warnings.append(msg)
def get_module_logger(name):return Logger()
exec(compile(ast.Module(body=n,type_ignores=[]),p,'exec'),globals())
class Cal:
 def __init__(self):self.calls=0
 def get_step_time(self):self.calls+=1;return pd.Timestamp('2024-01-02 09:30'),pd.Timestamp('2024-01-02 10:00')
class Strategy:
 def __init__(self):self.trade_calendar=Cal()
def o(code,amount,start=None,end=None):return Order(code,amount,OrderDir.BUY,start,end)
s=Strategy();original=[o('A',2,None,pd.Timestamp('2024-01-02 09:50')),o('B',0,pd.Timestamp('2024-01-02 09:35'),None)];d=TradeDecisionWO(original,s,(2,5))
normal={'calendar_calls':s.trade_calendar.calls,'same_list':d.get_decision() is original,'times':[[str(x.start_time),str(x.end_time)] for x in original],'range':d.trade_range(None),'empty':d.empty()}
amounts={k:TradeDecisionWO([o(k,v)],Strategy()).empty() for k,v in [('zero',0.0),('edge',1e-6),('above',math.nextafter(1e-6,math.inf)),('negative',-2.0),('nan',float('nan'))]}
class Mixed(BaseTradeDecision):
 def __init__(self,v):super().__init__(Strategy());self.v=v
 def get_decision(self):return self.v
mixed={'non_order_first':Mixed([object(),o('A',2)]).empty(),'positive_first':Mixed([o('A',2),object()]).empty(),'zero_then_non_order':Mixed([o('A',0),object()]).empty()}
outer=TradeDecisionWO([],Strategy(),(1,3));im=TradeDecisionWO([],Strategy());ie=TradeDecisionWO([],Strategy(),(8,9));outer.mod_inner_decision(im);outer.mod_inner_decision(ie)
prop={'same':im.trade_range is outer.trade_range,'missing':im.trade_range(None),'existing':ie.trade_range(None)}
def outcome(f):
 try:return ['ok',f()]
 except BaseException as e:return [type(e).__name__,str(e)]
r={};r['missing_error']=outcome(lambda:TradeDecisionWO([],Strategy()).get_range_limit());r['missing_value']=outcome(lambda:TradeDecisionWO([],Strategy()).get_range_limit(default_value=[4,7]));r['missing_none']=outcome(lambda:TradeDecisionWO([],Strategy()).get_range_limit(default_value=None))
for key,rv,total in [('unclipped',(-2,12),None),('both',(-2,12),5),('left',(-2,3),5),('right',(2,12),5),('condition_false',(9,-2),5),('zero',(-1,0),0)]:
 x=TradeDecisionWO([],Strategy(),rv);x.total_step=total;r[key]=outcome(lambda x=x:x.get_range_limit())
empty=EmptyTradeDecision(Strategy(),(4,6))
try:TradeDecisionWO([object()],Strategy());bad='ok'
except BaseException as e:bad=type(e).__name__
print(json.dumps({'normal':normal,'amounts':amounts,'mixed':mixed,'propagation':prop,'range':r,'empty':{'items':empty.get_decision(),'empty':empty.empty(),'range':empty.trade_range(None)},'bad_item':bad},allow_nan=True))
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
    let expected = json!({
        "normal": {
            "calendar_calls": 2,
            "same_list": true,
            "times": [["2024-01-02 09:30:00", "2024-01-02 09:50:00"], ["2024-01-02 09:35:00", "2024-01-02 10:00:00"]],
            "range": [2, 5],
            "empty": false
        },
        "amounts": {"zero": true, "edge": true, "above": false, "negative": true, "nan": true},
        "mixed": {"non_order_first": true, "positive_first": false, "zero_then_non_order": true},
        "propagation": {"same": true, "missing": [1, 3], "existing": [8, 9]},
        "range": {
            "missing_error": ["NotImplementedError", "The decision didn't provide an index range"],
            "missing_value": ["ok", [4, 7]],
            "missing_none": ["ok", null],
            "unclipped": ["ok", [-2, 12]],
            "both": ["ok", [0, 4]],
            "left": ["ok", [0, 3]],
            "right": ["ok", [2, 4]],
            "condition_false": ["ok", [9, -2]],
            "zero": ["ok", [0, -1]]
        },
        "empty": {"items": [], "empty": true, "range": [4, 6]},
        "bad_item": "AssertionError"
    });
    assert_eq!(actual, expected);
}
