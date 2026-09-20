use std::{path::PathBuf, process::Command};

use domain_core::{
    CASH_SETTLEMENT, InfinitePositionCash, NO_SETTLEMENT, PositionCash, PositionCashError,
};
use serde_json::{Value, json};

fn number(value: f64) -> Value {
    if value.is_nan() {
        json!("NaN")
    } else if value == f64::INFINITY {
        json!("Infinity")
    } else if value == f64::NEG_INFINITY {
        json!("-Infinity")
    } else {
        json!(value)
    }
}

fn snapshot(cash: &PositionCash) -> Value {
    json!({
        "state": cash.settlement_type(),
        "cash": number(cash.available_cash()),
        "delay": cash.delayed_cash().map_or_else(|| json!("absent"), number),
        "available": number(cash.cash(false)),
        "total": number(cash.cash(true)),
    })
}

#[test]
fn construction_restore_and_disabled_settlement_preserve_residual_delay() {
    let initial = PositionCash::new(100.0);
    assert_eq!(initial.available_cash().to_bits(), 100.0_f64.to_bits());
    assert_eq!(initial.delayed_cash(), None);
    assert_eq!(initial.settlement_type(), NO_SETTLEMENT);
    assert_eq!(initial.cash(false).to_bits(), 100.0_f64.to_bits());
    assert_eq!(initial.cash(true).to_bits(), 100.0_f64.to_bits());

    let mut restored = PositionCash::restore(100.0, Some(9.0), NO_SETTLEMENT);
    assert_eq!(restored.cash(false).to_bits(), 100.0_f64.to_bits());
    assert_eq!(restored.cash(true).to_bits(), 109.0_f64.to_bits());
    restored.settle_commit().unwrap();
    restored.settle_start(NO_SETTLEMENT).unwrap();
    restored.settle_start(NO_SETTLEMENT).unwrap();
    assert_eq!(restored.delayed_cash(), Some(9.0));

    restored.settle_start(CASH_SETTLEMENT).unwrap();
    assert_eq!(
        restored.delayed_cash().unwrap().to_bits(),
        0.0_f64.to_bits()
    );
    assert_eq!(restored.cash(true).to_bits(), 100.0_f64.to_bits());
}

#[test]
fn sale_purchase_and_commit_follow_exact_cash_arithmetic() {
    let mut delayed = PositionCash::new(100.0);
    delayed.settle_start(CASH_SETTLEMENT).unwrap();
    delayed.record_sale_proceeds(30.0, 2.0).unwrap();
    assert_eq!(delayed.available_cash().to_bits(), 100.0_f64.to_bits());
    assert_eq!(delayed.delayed_cash(), Some(28.0));
    assert_eq!(delayed.cash(true).to_bits(), 128.0_f64.to_bits());
    delayed.settle_commit().unwrap();
    assert_eq!(delayed.available_cash().to_bits(), 128.0_f64.to_bits());
    assert_eq!(delayed.delayed_cash(), None);
    assert_eq!(delayed.settlement_type(), NO_SETTLEMENT);

    let mut immediate = PositionCash::new(100.0);
    immediate.record_sale_proceeds(30.0, 2.0).unwrap();
    assert_eq!(immediate.available_cash().to_bits(), 128.0_f64.to_bits());

    let mut purchase = PositionCash::new(100.0);
    purchase.settle_start(CASH_SETTLEMENT).unwrap();
    purchase.pay_for_purchase(30.0, 2.0);
    assert_eq!(purchase.available_cash().to_bits(), 68.0_f64.to_bits());
    assert_eq!(purchase.delayed_cash(), Some(0.0));
    purchase.settle_commit().unwrap();
    assert_eq!(purchase.available_cash().to_bits(), 68.0_f64.to_bits());

    for special in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut cash = PositionCash::new(100.0);
        cash.settle_start(CASH_SETTLEMENT).unwrap();
        cash.record_sale_proceeds(special, 0.0).unwrap();
        assert_eq!(cash.delayed_cash().unwrap().to_bits(), special.to_bits());
        cash.settle_commit().unwrap();
        assert_eq!(cash.available_cash().to_bits(), (100.0 + special).to_bits());
    }
}

#[test]
fn nested_unknown_and_missing_delay_errors_retain_reached_state() {
    let mut nested = PositionCash::new(100.0);
    nested.settle_start(CASH_SETTLEMENT).unwrap();
    assert_eq!(
        nested.settle_start("other"),
        Err(PositionCashError::NestedSettlement {
            active: CASH_SETTLEMENT.to_owned(),
        })
    );
    assert_eq!(nested.settlement_type(), CASH_SETTLEMENT);
    assert_eq!(nested.delayed_cash(), Some(0.0));

    let mut unknown = PositionCash::new(100.0);
    unknown.settle_start("weird").unwrap();
    assert_eq!(
        unknown.record_sale_proceeds(30.0, 2.0),
        Err(PositionCashError::UnsupportedSettlement {
            settlement_type: "weird".to_owned(),
        })
    );
    assert_eq!(unknown.available_cash().to_bits(), 100.0_f64.to_bits());
    assert_eq!(
        unknown.settle_commit(),
        Err(PositionCashError::UnsupportedSettlement {
            settlement_type: "weird".to_owned(),
        })
    );
    assert_eq!(unknown.settlement_type(), "weird");

    let mut missing = PositionCash::restore(100.0, None, CASH_SETTLEMENT);
    assert_eq!(
        missing.record_sale_proceeds(30.0, 2.0),
        Err(PositionCashError::MissingDelayedCash)
    );
    assert_eq!(
        missing.settle_commit(),
        Err(PositionCashError::MissingDelayedCash)
    );
    assert_eq!(missing.available_cash().to_bits(), 100.0_f64.to_bits());
    assert_eq!(missing.settlement_type(), CASH_SETTLEMENT);
}

#[test]
fn infinite_position_cash_ignores_every_operation() {
    let mut cash = InfinitePositionCash;
    assert!(cash.cash(false).is_infinite() && cash.cash(false).is_sign_positive());
    assert!(cash.cash(true).is_infinite() && cash.cash(true).is_sign_positive());
    cash.settle_start("weird");
    cash.settle_start(CASH_SETTLEMENT);
    cash.record_sale_proceeds(f64::NAN, f64::INFINITY);
    cash.pay_for_purchase(f64::INFINITY, f64::NAN);
    cash.settle_commit();
    assert_eq!(cash, InfinitePositionCash);
}

fn live_python_snapshot() -> Value {
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/backtest/position.py");
    let script = r"
import ast,json,math,numpy as np,sys
p=sys.argv[1];t=ast.parse(open(p,encoding='utf-8').read());c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='Position');names={'_del_stock','_sell_stock','_buy_stock','settle_start','settle_commit','get_cash'};body=[]
for n in c.body:
 if isinstance(n,ast.FunctionDef) and n.name in names:
  n.returns=None
  for a in n.args.args:a.annotation=None
  body.append(n)
C=ast.ClassDef(name='P',bases=[],keywords=[],decorator_list=[],body=[ast.Assign(targets=[ast.Name(id='ST_CASH',ctx=ast.Store())],value=ast.Constant('cash')),ast.Assign(targets=[ast.Name(id='ST_NO',ctx=ast.Store())],value=ast.Constant('None'))]+body);ns={'np':np};exec(compile(ast.fix_missing_locations(ast.Module(body=[C],type_ignores=[])),p,'exec'),ns);P=ns['P']
def pos(delay=None,state='None'):
 x=P();x._settle_type=state;x.position={'cash':100.,'A':{'amount':10.,'price':10.}}
 if delay is not None:x.position['cash_delay']=delay
 return x
def v(z):
 if isinstance(z,float) and math.isnan(z):return 'NaN'
 if z==float('inf'):return 'Infinity'
 if z==float('-inf'):return '-Infinity'
 return z
def s(x):return {'state':x._settle_type,'cash':v(x.position['cash']),'delay':v(x.position['cash_delay']) if 'cash_delay' in x.position else 'absent','available':v(x.get_cash()),'total':v(x.get_cash(True))}
r=[];x=pos(9.);x.settle_commit();x.settle_start('None');x.settle_start('None');r.append(s(x));x.settle_start('cash');r.append(s(x));x._sell_stock('A',30.,2.,10.);r.append(s(x));x.settle_commit();r.append(s(x));x=pos();x.settle_start('cash');x._buy_stock('A',30.,2.,10.);r.append(s(x));x.settle_commit();r.append(s(x));x=pos();x.settle_start('weird')
try:x._sell_stock('A',30.,2.,10.)
except Exception as e:r.append(type(e).__name__+':'+str(e))
r.append(s(x))
try:x.settle_commit()
except Exception as e:r.append(type(e).__name__+':'+str(e))
r.append(s(x));print(json.dumps(r,separators=(',',':')))
";
    let output = Command::new("python")
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
    serde_json::from_slice(&output.stdout).unwrap()
}

fn rust_snapshot() -> Value {
    let mut rows = Vec::new();
    let mut cash = PositionCash::restore(100.0, Some(9.0), NO_SETTLEMENT);
    cash.settle_commit().unwrap();
    cash.settle_start(NO_SETTLEMENT).unwrap();
    cash.settle_start(NO_SETTLEMENT).unwrap();
    rows.push(snapshot(&cash));
    cash.settle_start(CASH_SETTLEMENT).unwrap();
    rows.push(snapshot(&cash));
    cash.record_sale_proceeds(30.0, 2.0).unwrap();
    rows.push(snapshot(&cash));
    cash.settle_commit().unwrap();
    rows.push(snapshot(&cash));

    let mut cash = PositionCash::new(100.0);
    cash.settle_start(CASH_SETTLEMENT).unwrap();
    cash.pay_for_purchase(30.0, 2.0);
    rows.push(snapshot(&cash));
    cash.settle_commit().unwrap();
    rows.push(snapshot(&cash));

    let mut cash = PositionCash::new(100.0);
    cash.settle_start("weird").unwrap();
    let sale_error = cash.record_sale_proceeds(30.0, 2.0).unwrap_err();
    rows.push(json!(match sale_error {
        PositionCashError::UnsupportedSettlement { .. } =>
            "NotImplementedError:This type of input is not supported",
        _ => unreachable!(),
    }));
    rows.push(snapshot(&cash));
    let commit_error = cash.settle_commit().unwrap_err();
    rows.push(json!(match commit_error {
        PositionCashError::UnsupportedSettlement { .. } =>
            "NotImplementedError:This type of input is not supported",
        _ => unreachable!(),
    }));
    rows.push(snapshot(&cash));
    Value::Array(rows)
}

#[test]
fn cash_settlement_matches_live_python_source() {
    assert_eq!(rust_snapshot(), live_python_snapshot());
}
