use std::{path::PathBuf, process::Command};

use chrono::NaiveDateTime;
use domain_core::{
    EmptyTradeDecision, Order, OrderDecision, OrderDir, OrderTradeDecision, SimulatorExecutorError,
    SimulatorTradeType, TradeRange, retrieve_orders_from_decision, simulator_order_iterator,
};
use serde_json::{Value, json};
use strum::VariantArray;

fn timestamp(text: &str) -> NaiveDateTime {
    NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S").unwrap()
}

fn order(stock: &str, direction: OrderDir) -> Order {
    Order::new(
        stock,
        1.0,
        direction,
        Some(timestamp("2024-01-02 09:30:00")),
        Some(timestamp("2024-01-02 10:00:00")),
    )
}

fn stocks(orders: &[&mut Order]) -> Vec<String> {
    orders
        .iter()
        .map(|order| order.stock_id().to_owned())
        .collect()
}

fn decision() -> OrderTradeDecision {
    OrderTradeDecision::from_orders(
        vec![
            order("S1", OrderDir::Sell),
            order("B1", OrderDir::Buy),
            order("S2", OrderDir::Sell),
            order("B2", OrderDir::Buy),
            order("B3", OrderDir::Buy),
        ],
        timestamp("2024-01-02 09:30:00"),
        timestamp("2024-01-02 10:00:00"),
        None,
    )
}

#[test]
fn trade_type_parsing_is_exact_and_stably_rendered() {
    assert_eq!(
        SimulatorTradeType::VARIANTS,
        [SimulatorTradeType::Serial, SimulatorTradeType::Parallel]
    );
    assert_eq!(
        SimulatorTradeType::parse_text("serial"),
        Ok(SimulatorTradeType::Serial)
    );
    assert_eq!(
        SimulatorTradeType::parse_text("parallel"),
        Ok(SimulatorTradeType::Parallel)
    );
    assert_eq!(SimulatorTradeType::Serial.to_string(), "serial");
    assert_eq!(SimulatorTradeType::Parallel.as_ref(), "parallel");

    for invalid in ["", "Parallel", " parallel ", "sequential"] {
        assert_eq!(
            SimulatorTradeType::parse_text(invalid),
            Err(SimulatorExecutorError::UnsupportedTradeType {
                trade_type: invalid.to_owned()
            })
        );
    }
    assert_eq!(
        SimulatorTradeType::parse_text("bad")
            .unwrap_err()
            .to_string(),
        "unsupported simulator trade type: bad"
    );
}

#[test]
fn serial_and_parallel_batches_preserve_reference_and_stable_order_rules() {
    let mut serial_decision = decision();
    let mut serial = simulator_order_iterator(&mut serial_decision, "serial").unwrap();
    assert_eq!(stocks(&serial), ["S1", "B1", "S2", "B2", "B3"]);
    serial[0].set_deal_amount(7.0);
    drop(serial);
    assert_eq!(
        serial_decision
            .orders()
            .iter()
            .map(Order::stock_id)
            .collect::<Vec<_>>(),
        ["S1", "B1", "S2", "B2", "B3"]
    );
    assert_eq!(
        serial_decision.orders()[0].deal_amount().to_bits(),
        7.0_f64.to_bits()
    );

    let mut parallel_decision = decision();
    let mut parallel = simulator_order_iterator(&mut parallel_decision, "parallel").unwrap();
    assert_eq!(stocks(&parallel), ["B1", "B2", "B3", "S1", "S2"]);
    parallel[0].set_factor(Some(3.0));
    drop(parallel);
    assert_eq!(
        parallel_decision
            .orders()
            .iter()
            .map(Order::stock_id)
            .collect::<Vec<_>>(),
        ["S1", "B1", "S2", "B2", "B3"]
    );
    assert_eq!(parallel_decision.orders()[1].factor(), Some(3.0));

    let mut retrieved_decision = decision();
    let mut retrieved = retrieve_orders_from_decision(&mut retrieved_decision);
    assert_eq!(stocks(&retrieved), ["S1", "B1", "S2", "B2", "B3"]);
    retrieved.reverse();
    drop(retrieved);
    assert_eq!(retrieved_decision.orders()[0].stock_id(), "S1");
}

struct TrackingDecision {
    orders: Vec<Order>,
    orders_mut_calls: usize,
    start: NaiveDateTime,
    end: NaiveDateTime,
}

impl OrderDecision for TrackingDecision {
    fn orders(&self) -> &[Order] {
        &self.orders
    }

    fn orders_mut(&mut self) -> &mut [Order] {
        self.orders_mut_calls += 1;
        &mut self.orders
    }

    fn start_time(&self) -> NaiveDateTime {
        self.start
    }

    fn end_time(&self) -> NaiveDateTime {
        self.end
    }

    fn trade_range(&self) -> Option<&dyn TradeRange> {
        None
    }
}

#[test]
fn empty_and_invalid_modes_preserve_extraction_before_validation() {
    let start = timestamp("2024-01-02 09:30:00");
    let end = timestamp("2024-01-02 10:00:00");
    let mut empty = EmptyTradeDecision::new(start, end, None);
    assert!(
        simulator_order_iterator(&mut empty, "parallel")
            .unwrap()
            .is_empty()
    );
    assert!(retrieve_orders_from_decision(&mut empty).is_empty());

    for invalid in ["Parallel", " parallel ", "bad"] {
        let mut tracked = TrackingDecision {
            orders: vec![order("A", OrderDir::Buy)],
            orders_mut_calls: 0,
            start,
            end,
        };
        assert_eq!(
            simulator_order_iterator(&mut tracked, invalid),
            Err(SimulatorExecutorError::UnsupportedTradeType {
                trade_type: invalid.to_owned()
            })
        );
        assert_eq!(tracked.orders_mut_calls, 1);
        assert_eq!(tracked.orders[0].stock_id(), "A");
    }
}

#[test]
fn simulator_order_iteration_matches_live_python_source() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib");
    let decision_source = root.join("qlib/backtest/decision.py");
    let executor_source = root.join("qlib/backtest/executor.py");
    let script = r"
import ast,json,sys
from dataclasses import dataclass
from enum import IntEnum
from typing import *
import numpy as np,pandas as pd
dp,ep=sys.argv[1:3];dt=ast.parse(open(dp,encoding='utf-8').read(),filename=dp)
dn=[dt.body[0]]+[x for x in dt.body if isinstance(x,ast.ClassDef) and x.name in {'OrderDir','Order'}];exec(compile(ast.Module(body=dn,type_ignores=[]),dp,'exec'),globals())
et=ast.parse(open(ep,encoding='utf-8').read(),filename=ep);retrieve=next(x for x in et.body if isinstance(x,ast.FunctionDef) and x.name=='_retrieve_orders_from_decision');sim=next(x for x in et.body if isinstance(x,ast.ClassDef) and x.name=='SimulatorExecutor');iterator=next(x for x in sim.body if isinstance(x,ast.FunctionDef) and x.name=='_get_order_iterator');exec(compile(ast.Module(body=[retrieve],type_ignores=[]),ep,'exec'),globals());ns={};exec(compile(ast.Module(body=[iterator],type_ignores=[]),ep,'exec'),globals(),ns);get_iter=ns['_get_order_iterator']
def o(code,d):return Order(code,1.0,d,pd.Timestamp('2024-01-02 09:30'),pd.Timestamp('2024-01-02 10:00'))
class D:
 def __init__(self,v):self.v=v;self.calls=0
 def get_decision(self):self.calls+=1;return self.v
class E:
 TT_SERIAL='serial';TT_PARAL='parallel'
 def __init__(self,t):self.trade_type=t
orders=[o('S1',OrderDir.SELL),o('B1',OrderDir.BUY),o('S2',OrderDir.SELL),o('B2',OrderDir.BUY),o('B3',OrderDir.BUY)];r={}
for mode in ['serial','parallel']:
 d=D(orders);out=get_iter(E(mode),d);r[mode]={'codes':[x.stock_id for x in out],'new_list':out is not orders,'same_objects':set(map(id,out))==set(map(id,orders)),'original':[x.stock_id for x in orders],'calls':d.calls}
out=get_iter(E('serial'),D(orders));out[0].deal_amount=7;r['mutation']=orders[0].deal_amount;r['empty']=[x.stock_id for x in get_iter(E('parallel'),D([]))]
def result(vals,mode):
 d=D(vals)
 try:return ['ok',[x.stock_id for x in get_iter(E(mode),d)],d.calls]
 except BaseException as e:return [type(e).__name__,str(e),d.calls]
r['invalid']={str(v):result(orders,v) for v in ['Parallel',' parallel ',None,1]};r['non_order_serial']=result([object()],'serial');r['non_order_invalid']=result([object()],'bad')
class TD(D):
 def get_decision(self):self.calls+=1;return tuple(self.v)
t=TD(orders);r['tuple']=['ok',[x.stock_id for x in get_iter(E('serial'),t)],t.calls]
print(json.dumps(r))
";
    let output = Command::new(std::env::var_os("PYTHON").unwrap_or_else(|| "python".into()))
        .arg("-c")
        .arg(script)
        .arg(decision_source)
        .arg(executor_source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let actual: Value = serde_json::from_slice(&output.stdout).unwrap();
    let expected = json!({
        "serial": {"codes": ["S1", "B1", "S2", "B2", "B3"], "new_list": true, "same_objects": true, "original": ["S1", "B1", "S2", "B2", "B3"], "calls": 1},
        "parallel": {"codes": ["B1", "B2", "B3", "S1", "S2"], "new_list": true, "same_objects": true, "original": ["S1", "B1", "S2", "B2", "B3"], "calls": 1},
        "mutation": 7,
        "empty": [],
        "invalid": {
            "Parallel": ["NotImplementedError", "This type of input is not supported", 1],
            " parallel ": ["NotImplementedError", "This type of input is not supported", 1],
            "None": ["NotImplementedError", "This type of input is not supported", 1],
            "1": ["NotImplementedError", "This type of input is not supported", 1]
        },
        "non_order_serial": ["AssertionError", "", 1],
        "non_order_invalid": ["AssertionError", "", 1],
        "tuple": ["ok", ["S1", "B1", "S2", "B2", "B3"], 1]
    });
    assert_eq!(actual, expected);
}
