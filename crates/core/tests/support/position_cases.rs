use std::{path::PathBuf, process::Command};

use domain_core::{
    CASH_SETTLEMENT, ExecutionPosition, ExecutionTarget, InfinitePosition, InitialPositionValue,
    Order, OrderDir, Position, PositionCashError, PositionError, PositionHolding,
};
use indexmap::IndexMap;
use serde_json::{Map, Value, json};

fn order(stock: &str, direction: OrderDir) -> Order {
    Order::new(stock, 1.0, direction, None, None)
}

fn assert_same(actual: f64, expected: f64) {
    assert_eq!(actual.to_bits(), expected.to_bits());
}

fn ordered_float_bits(value: f64) -> u64 {
    let bits = value.to_bits();
    if bits & (1_u64 << 63) == 0 {
        bits | (1_u64 << 63)
    } else {
        !bits
    }
}

fn assert_snapshot_within_one_ulp(actual: &Value, expected: &Value) {
    match (actual, expected) {
        (Value::Number(actual), Value::Number(expected)) => {
            let actual = actual.as_f64().unwrap();
            let expected = expected.as_f64().unwrap();
            assert!(
                ordered_float_bits(actual).abs_diff(ordered_float_bits(expected)) <= 1,
                "{actual:?} differs from {expected:?} by more than one ULP"
            );
        }
        (Value::Array(actual), Value::Array(expected)) => {
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected) {
                assert_snapshot_within_one_ulp(actual, expected);
            }
        }
        (Value::Object(actual), Value::Object(expected)) => {
            assert_eq!(actual.len(), expected.len());
            for (key, expected) in expected {
                assert_snapshot_within_one_ulp(&actual[key], expected);
            }
        }
        _ => assert_eq!(actual, expected),
    }
}

fn make_position(cash: f64, values: &[(&str, f64, Option<f64>)]) -> Position {
    Position::from_initial(
        cash,
        values
            .iter()
            .map(|(stock, amount, price)| {
                (
                    (*stock).to_owned(),
                    InitialPositionValue::Holding(PositionHolding::restored(*amount, *price, None)),
                )
            })
            .collect(),
    )
}

#[test]
fn initialization_and_accessors_normalize_documented_inputs() {
    let initial = IndexMap::from([
        ("I".to_owned(), InitialPositionValue::Amount(3.0)),
        (
            "S".to_owned(),
            InitialPositionValue::Holding(PositionHolding::restored(2.0, Some(5.0), Some(0.4))),
        ),
    ]);
    let position = Position::from_initial(100.0, initial);
    assert_eq!(position.initial_cash().to_bits(), 100.0_f64.to_bits());
    assert_eq!(position.holdings().len(), 2);
    assert_eq!(position.stock_ids().collect::<Vec<_>>(), ["I", "S"]);
    assert!(position.check_stock("I"));
    assert!(!position.check_stock("Z"));
    assert_eq!(position.stock_amount("I").to_bits(), 3.0_f64.to_bits());
    assert_eq!(position.stock_amount("Z").to_bits(), 0.0_f64.to_bits());
    assert_same(position.stock_price("S").unwrap(), 5.0);
    assert_eq!(
        position.stock_price("I"),
        Err(PositionError::MissingPrice {
            stock: "I".to_owned(),
        })
    );
    assert_eq!(
        position.stock_price("Z"),
        Err(PositionError::MissingStock {
            stock: "Z".to_owned(),
        })
    );
    let holding = position.holding("S").unwrap();
    assert_same(holding.amount(), 2.0);
    assert_eq!(holding.price(), Some(5.0));
    assert_eq!(holding.weight(), Some(0.4));
    assert_eq!(position.holding("Z"), None);
    assert_same(position.cash(false), 100.0);
}

#[test]
fn buying_accumulates_or_initializes_before_exact_cash_payment() {
    let mut position = make_position(100.0, &[("A", 2.0, Some(5.0))]);
    position
        .update_order(&order("A", OrderDir::Buy), 20.0, 2.0, 10.0)
        .unwrap();
    assert_eq!(position.stock_amount("A").to_bits(), 4.0_f64.to_bits());
    assert_same(position.stock_price("A").unwrap(), 5.0);
    assert_eq!(position.cash(false).to_bits(), 78.0_f64.to_bits());

    position
        .update_order(&order("N", OrderDir::Buy), 30.0, 1.0, 10.0)
        .unwrap();
    let opened = position.holding("N").unwrap();
    assert_same(opened.amount(), 3.0);
    assert_eq!(opened.price(), Some(10.0));
    assert_eq!(opened.weight(), Some(0.0));
    assert_eq!(position.cash(false).to_bits(), 47.0_f64.to_bits());

    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut special = make_position(100.0, &[("A", 2.0, Some(10.0))]);
        special
            .update_order(&order("A", OrderDir::Buy), value, 0.0, 10.0)
            .unwrap();
        assert_eq!(
            special.stock_amount("A").to_bits(),
            (2.0 + value / 10.0).to_bits()
        );
        assert_eq!(special.cash(false).to_bits(), (100.0 - value).to_bits());
    }
}

#[test]
fn selling_matches_numpy_close_and_retains_failure_mutations() {
    let mut full = make_position(100.0, &[("A", 10.0, Some(10.0))]);
    full.update_order(&order("A", OrderDir::Sell), 100.0, 2.0, 10.0)
        .unwrap();
    assert!(!full.check_stock("A"));
    assert_eq!(full.cash(false).to_bits(), 198.0_f64.to_bits());

    let mut relative = make_position(100.0, &[("A", 1_000_000.0, Some(10.0))]);
    relative
        .update_order(&order("A", OrderDir::Sell), 9_999_900.0, 2.0, 10.0)
        .unwrap();
    assert_same(relative.stock_amount("A"), 10.0);

    let mut close = make_position(100.0, &[("A", 1.0, Some(10.0))]);
    close
        .update_order(&order("A", OrderDir::Sell), 10.000_100_09, 2.0, 10.0)
        .unwrap();
    assert!(!close.check_stock("A"));

    let mut over = make_position(100.0, &[("A", 1.0, Some(10.0))]);
    let error = over
        .update_order(&order("A", OrderDir::Sell), 20.0, 2.0, 10.0)
        .unwrap_err();
    assert_eq!(
        error,
        PositionError::Oversell {
            available: 1.0,
            stock: "A".to_owned(),
            required: 2.0,
        }
    );
    assert_eq!(over.stock_amount("A").to_bits(), (-1.0_f64).to_bits());
    assert_eq!(over.cash(false).to_bits(), 100.0_f64.to_bits());
}

#[test]
fn update_errors_and_settlement_preserve_python_ordering() {
    for direction in [OrderDir::Buy, OrderDir::Sell] {
        let mut position = make_position(100.0, &[("A", 1.0, Some(10.0))]);
        assert_eq!(
            position.update_order(&order("A", direction), 10.0, 1.0, 0.0),
            Err(PositionError::ZeroTradePrice)
        );
        assert_same(position.stock_amount("A"), 1.0);
        assert_same(position.cash(false), 100.0);
    }

    let mut missing = make_position(100.0, &[]);
    assert_eq!(
        missing.update_order(&order("Z", OrderDir::Sell), 10.0, 1.0, 10.0),
        Err(PositionError::MissingStock {
            stock: "Z".to_owned(),
        })
    );

    let mut unknown = make_position(100.0, &[("A", 2.0, Some(10.0))]);
    unknown.settle_start("weird").unwrap();
    assert_eq!(
        unknown.update_order(&order("A", OrderDir::Sell), 10.0, 1.0, 10.0),
        Err(PositionError::Cash(
            PositionCashError::UnsupportedSettlement {
                settlement_type: "weird".to_owned(),
            }
        ))
    );
    assert_same(unknown.stock_amount("A"), 1.0);
    assert_eq!(unknown.cash_state().settlement_type(), "weird");

    let mut delayed = make_position(100.0, &[("A", 2.0, Some(10.0))]);
    delayed.settle_start(CASH_SETTLEMENT).unwrap();
    delayed
        .update_order(&order("A", OrderDir::Sell), 10.0, 1.0, 10.0)
        .unwrap();
    assert_same(delayed.cash(false), 100.0);
    assert_same(delayed.cash(true), 109.0);
    delayed.settle_commit().unwrap();
    assert_same(delayed.cash(false), 109.0);
}

#[test]
fn prices_counts_weights_and_snapshots_preserve_typed_state() {
    let counts = IndexMap::from([("day".to_owned(), 4.0)]);
    let mut position = Position::from_initial(
        20.0,
        IndexMap::from([
            (
                "A".to_owned(),
                InitialPositionValue::Holding(PositionHolding::restored_with_counts(
                    2.0,
                    Some(10.0),
                    Some(0.1),
                    counts,
                )),
            ),
            (
                "B".to_owned(),
                InitialPositionValue::Holding(PositionHolding::restored(3.0, Some(5.0), None)),
            ),
        ]),
    );

    position.update_stock_price("A", 12.0).unwrap();
    position.update_stock_count("A", "day", 4.0).unwrap();
    position.update_stock_weight("A", 0.7).unwrap();
    assert_same(position.stock_price("A").unwrap(), 12.0);
    assert_same(position.stock_count("A", "day").unwrap(), 4.0);
    assert_same(position.stock_count("A", "week").unwrap(), 0.0);
    assert_same(position.stock_weight("A").unwrap(), 0.7);
    assert_eq!(
        position.stock_weight("B"),
        Err(PositionError::MissingWeight {
            stock: "B".to_owned(),
        })
    );

    position.add_count_all("day");
    position.add_count_all("week");
    assert_same(position.stock_count("A", "day").unwrap(), 5.0);
    assert_same(position.stock_count("A", "week").unwrap(), 1.0);
    assert_same(position.stock_count("B", "day").unwrap(), 1.0);
    assert_same(position.stock_count("B", "week").unwrap(), 1.0);
    assert_eq!(
        position.holding("A").unwrap().counts(),
        &IndexMap::from([("day".to_owned(), 5.0), ("week".to_owned(), 1.0)])
    );
    assert_eq!(
        position.stock_amounts(),
        IndexMap::from([("A".to_owned(), 2.0), ("B".to_owned(), 3.0)])
    );

    for result in [
        position.update_stock_price("Z", 1.0),
        position.update_stock_count("Z", "day", 1.0),
        position.update_stock_weight("Z", 1.0),
    ] {
        assert_eq!(
            result,
            Err(PositionError::MissingStock {
                stock: "Z".to_owned(),
            })
        );
    }
    assert_eq!(
        position.stock_count("Z", "day"),
        Err(PositionError::MissingStock {
            stock: "Z".to_owned(),
        })
    );
    assert_eq!(
        position.stock_weight("Z"),
        Err(PositionError::MissingStock {
            stock: "Z".to_owned(),
        })
    );
}

#[test]
fn valuation_weights_and_failures_match_python_arithmetic() {
    let mut position = Position::from_initial(
        20.0,
        IndexMap::from([
            (
                "A".to_owned(),
                InitialPositionValue::Holding(PositionHolding::restored(
                    2.0,
                    Some(10.0),
                    Some(0.1),
                )),
            ),
            (
                "B".to_owned(),
                InitialPositionValue::Holding(PositionHolding::restored(3.0, Some(5.0), Some(0.2))),
            ),
        ]),
    );
    assert_same(position.calculate_stock_value().unwrap(), 35.0);
    assert_same(position.calculate_value().unwrap(), 55.0);
    let total = position.stock_weights(false).unwrap();
    assert_same(total["A"], 20.0 / 55.0);
    assert_same(total["B"], 15.0 / 55.0);
    let stocks = position.stock_weights(true).unwrap();
    assert_same(stocks["A"], 20.0 / 35.0);
    assert_same(stocks["B"], 15.0 / 35.0);

    position.update_weight_all().unwrap();
    assert_same(position.stock_weight("A").unwrap(), 20.0 / 55.0);
    assert_same(position.stock_weight("B").unwrap(), 15.0 / 55.0);
    position.settle_start(CASH_SETTLEMENT).unwrap();
    position
        .update_order(&order("A", OrderDir::Sell), 10.0, 1.0, 10.0)
        .unwrap();
    assert_same(position.calculate_stock_value().unwrap(), 25.0);
    assert_same(position.calculate_value().unwrap(), 54.0);

    let empty = make_position(0.0, &[]);
    assert!(empty.stock_weights(false).unwrap().is_empty());
    assert!(empty.stock_weights(true).unwrap().is_empty());

    let zero_total = make_position(-10.0, &[("A", 1.0, Some(10.0))]);
    assert_eq!(
        zero_total.stock_weights(false),
        Err(PositionError::ZeroPositionValue)
    );
    let zero_stock = make_position(1.0, &[("A", 1.0, Some(0.0))]);
    assert_eq!(
        zero_stock.stock_weights(true),
        Err(PositionError::ZeroPositionValue)
    );

    let mut atomic = Position::from_initial(
        -10.0,
        IndexMap::from([(
            "A".to_owned(),
            InitialPositionValue::Holding(PositionHolding::restored(1.0, Some(10.0), Some(0.25))),
        )]),
    );
    assert_eq!(
        atomic.update_weight_all(),
        Err(PositionError::ZeroPositionValue)
    );
    assert_same(atomic.stock_weight("A").unwrap(), 0.25);

    let missing = make_position(1.0, &[("A", 1.0, None)]);
    let missing_error = PositionError::MissingPrice {
        stock: "A".to_owned(),
    };
    assert_eq!(missing.calculate_stock_value(), Err(missing_error.clone()));
    assert_eq!(missing.calculate_value(), Err(missing_error.clone()));
    assert_eq!(missing.stock_weights(false), Err(missing_error));

    let special = Position::from_initial(
        f64::INFINITY,
        IndexMap::from([
            (
                "A".to_owned(),
                InitialPositionValue::Holding(PositionHolding::restored(
                    f64::INFINITY,
                    Some(0.0),
                    None,
                )),
            ),
            (
                "B".to_owned(),
                InitialPositionValue::Holding(PositionHolding::restored(f64::NAN, Some(1.0), None)),
            ),
        ]),
    );
    assert!(special.calculate_stock_value().unwrap().is_nan());
    assert!(special.calculate_value().unwrap().is_nan());
}

#[test]
fn infinite_position_exposes_only_meaningful_valuation_operations() {
    let infinite = InfinitePosition;
    infinite.update_stock_price("A", f64::NAN);
    assert!(infinite.calculate_stock_value().is_infinite());
    assert!(infinite.stock_price("A").is_nan());
    assert_eq!(
        infinite.calculate_value(),
        Err(PositionError::UnsupportedInfiniteOperation {
            operation: "calculating value",
        })
    );
    assert_eq!(
        infinite.stock_list(),
        Err(PositionError::UnsupportedInfiniteOperation {
            operation: "stock list",
        })
    );
    assert_eq!(
        infinite.stock_amounts(),
        Err(PositionError::UnsupportedInfiniteOperation {
            operation: "stock amount snapshot",
        })
    );
    assert_eq!(
        infinite.stock_weights(true),
        Err(PositionError::UnsupportedInfiniteOperation {
            operation: "stock weight snapshot",
        })
    );
    assert_eq!(
        infinite.add_count_all("day"),
        Err(PositionError::UnsupportedInfiniteOperation {
            operation: "incrementing holding counts",
        })
    );
    assert_eq!(
        infinite.update_weight_all(),
        Err(PositionError::UnsupportedInfiniteOperation {
            operation: "updating weights",
        })
    );
}

#[test]
fn execution_traits_compose_finite_and_infinite_positions() {
    let mut finite = make_position(100.0, &[("A", 2.0, Some(10.0))]);
    let view: &dyn ExecutionPosition = &finite;
    assert!(view.check_stock("A").unwrap());
    assert!(!view.check_stock("Z").unwrap());
    assert_same(view.stock_amount("A").unwrap(), 2.0);
    assert_same(view.cash().unwrap(), 100.0);

    let target: &mut dyn ExecutionTarget = &mut finite;
    assert!(target.position().unwrap().check_stock("A").unwrap());
    target
        .update_order(&order("A", OrderDir::Buy), 10.0, 1.0, 10.0)
        .unwrap();
    let error = target
        .update_order(&order("Z", OrderDir::Sell), 10.0, 1.0, 10.0)
        .unwrap_err();
    assert_eq!(error.message, "Z not in current position");

    let mut infinite = InfinitePosition;
    assert!(infinite.skip_update());
    let view: &dyn ExecutionPosition = &infinite;
    assert!(view.check_stock("anything").unwrap());
    assert!(view.stock_amount("anything").unwrap().is_infinite());
    assert!(view.cash().unwrap().is_infinite());
    let target: &mut dyn ExecutionTarget = &mut infinite;
    assert!(target.position().unwrap().check_stock("anything").unwrap());
    target
        .update_order(&order("A", OrderDir::Sell), f64::NAN, f64::NAN, 0.0)
        .unwrap();
}

fn special(value: f64) -> Value {
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

fn snapshot(position: &Position) -> Value {
    let stocks: Map<String, Value> = position
        .holdings()
        .iter()
        .map(|(stock, holding)| {
            let mut fields = Map::new();
            fields.insert("amount".to_owned(), special(holding.amount()));
            if let Some(price) = holding.price() {
                fields.insert("price".to_owned(), special(price));
            }
            if let Some(weight) = holding.weight() {
                fields.insert("weight".to_owned(), special(weight));
            }
            for (bar, count) in holding.counts() {
                fields.insert(format!("count_{bar}"), special(*count));
            }
            (stock.clone(), Value::Object(fields))
        })
        .collect();
    json!({
        "state": position.cash_state().settlement_type(),
        "cash": special(position.cash(false)),
        "delay": position.cash_state().delayed_cash().map_or_else(|| json!("absent"), special),
        "stocks": stocks,
    })
}

fn live_python_snapshot() -> Value {
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/backtest/position.py");
    let script = r"
import ast,json,math,numpy as np,sys
p=sys.argv[1];t=ast.parse(open(p,encoding='utf-8').read());c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='Position');names={'_init_stock','_buy_stock','_sell_stock','_del_stock','settle_start','settle_commit','get_cash'};body=[]
for n in c.body:
 if isinstance(n,ast.FunctionDef) and n.name in names:
  n.returns=None
  for a in n.args.args:a.annotation=None
  body.append(n)
C=ast.ClassDef(name='P',bases=[],keywords=[],decorator_list=[],body=[ast.Assign(targets=[ast.Name(id='ST_CASH',ctx=ast.Store())],value=ast.Constant('cash')),ast.Assign(targets=[ast.Name(id='ST_NO',ctx=ast.Store())],value=ast.Constant('None'))]+body);ns={'np':np};exec(compile(ast.fix_missing_locations(ast.Module(body=[C],type_ignores=[])),p,'exec'),ns);P=ns['P']
def pos():x=P();x._settle_type='None';x.position={'cash':100.,'A':{'amount':2.,'price':5.}};return x
def v(x):
 if isinstance(x,(float,np.floating)):
  if math.isnan(x):return 'NaN'
  if x==float('inf'):return 'Infinity'
  if x==float('-inf'):return '-Infinity'
 return float(x) if isinstance(x,(int,float,np.number)) else x
def s(x):return {'state':x._settle_type,'cash':v(x.position['cash']),'delay':v(x.position['cash_delay']) if 'cash_delay' in x.position else 'absent','stocks':{k:{kk:v(vv) for kk,vv in val.items()} for k,val in x.position.items() if isinstance(val,dict)}}
r=[];x=pos();x._buy_stock('A',20.,2.,10.);r.append(s(x));x._buy_stock('N',30.,1.,10.);r.append(s(x));x=P();x._settle_type='None';x.position={'cash':100.,'A':{'amount':10.,'price':10.}};x._sell_stock('A',100.,2.,10.);r.append(s(x));x=pos();x.settle_start('cash');x._sell_stock('A',10.,1.,10.);r.append(s(x));x.settle_commit();r.append(s(x));x=pos();x.settle_start('weird')
try:x._sell_stock('A',10.,1.,10.)
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
    let mut position = make_position(100.0, &[("A", 2.0, Some(5.0))]);
    position
        .update_order(&order("A", OrderDir::Buy), 20.0, 2.0, 10.0)
        .unwrap();
    rows.push(snapshot(&position));
    position
        .update_order(&order("N", OrderDir::Buy), 30.0, 1.0, 10.0)
        .unwrap();
    rows.push(snapshot(&position));
    let mut position = make_position(100.0, &[("A", 10.0, Some(10.0))]);
    position
        .update_order(&order("A", OrderDir::Sell), 100.0, 2.0, 10.0)
        .unwrap();
    rows.push(snapshot(&position));
    let mut position = make_position(100.0, &[("A", 2.0, Some(5.0))]);
    position.settle_start(CASH_SETTLEMENT).unwrap();
    position
        .update_order(&order("A", OrderDir::Sell), 10.0, 1.0, 10.0)
        .unwrap();
    rows.push(snapshot(&position));
    position.settle_commit().unwrap();
    rows.push(snapshot(&position));
    let mut position = make_position(100.0, &[("A", 2.0, Some(5.0))]);
    position.settle_start("weird").unwrap();
    let error = position
        .update_order(&order("A", OrderDir::Sell), 10.0, 1.0, 10.0)
        .unwrap_err();
    assert!(matches!(error, PositionError::Cash(_)));
    rows.push(json!(
        "NotImplementedError:This type of input is not supported"
    ));
    rows.push(snapshot(&position));
    Value::Array(rows)
}

fn live_python_valuation_snapshot() -> Value {
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/backtest/position.py");
    let script = r"
import ast,json,math,numpy as np,sys
p=sys.argv[1];t=ast.parse(open(p,encoding='utf-8').read())
def make(original,new,names,constants=False):
 c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name==original);body=[]
 if constants:body += [ast.Assign([ast.Name('ST_CASH',ast.Store())],ast.Constant('cash')),ast.Assign([ast.Name('ST_NO',ast.Store())],ast.Constant('None'))]
 for n in c.body:
  if isinstance(n,ast.FunctionDef) and n.name in names:
   n.returns=None
   for a in n.args.args:a.annotation=None
   body.append(n)
 return ast.ClassDef(new,[],[],body,[])
names={'update_stock_price','update_stock_count','update_stock_weight','calculate_stock_value','calculate_value','get_stock_list','get_stock_price','get_stock_amount','get_stock_count','get_stock_weight','get_cash','get_stock_amount_dict','get_stock_weight_dict','add_count_all','update_weight_all'}
inf_names={'update_stock_price','calculate_stock_value','calculate_value','get_stock_list','get_stock_price','get_stock_amount_dict','get_stock_weight_dict','add_count_all','update_weight_all'}
module=ast.fix_missing_locations(ast.Module([make('Position','P',names,True),make('InfPosition','I',inf_names)],[]));ns={'np':np};exec(compile(module,p,'exec'),ns);P,I=ns['P'],ns['I']
def pos(cash,stocks):x=P();x._settle_type='None';x.position={'cash':cash,**stocks};return x
def norm(x):
 if isinstance(x,(float,np.floating)):
  if math.isnan(x):return 'NaN'
  if x==float('inf'):return 'Infinity'
  if x==float('-inf'):return '-Infinity'
  return float(x)
 if isinstance(x,int):return float(x)
 if isinstance(x,dict):return {k:norm(v) for k,v in sorted(x.items())}
 if isinstance(x,(list,tuple)):return [norm(v) for v in x]
 return x
def error(fn):
 try:return ['ok',norm(fn())]
 except Exception as e:return ['error',type(e).__name__]
x=pos(20.,{'A':{'amount':2.,'price':10.,'weight':.1},'B':{'amount':3.,'price':5.,'weight':.2}});rows=[]
rows.append(norm({'stock':x.calculate_stock_value(),'value':x.calculate_value(),'amounts':x.get_stock_amount_dict(),'total':x.get_stock_weight_dict(False),'stocks':x.get_stock_weight_dict(True)}))
x.update_stock_price('A',12.);x.update_stock_count('A','day',4.);x.update_stock_weight('A',.7);x.add_count_all('day');x.add_count_all('week');rows.append(norm({'A':[x.get_stock_price('A'),x.get_stock_count('A','day'),x.get_stock_count('A','week'),x.get_stock_weight('A')],'B':[x.get_stock_count('B','day'),x.get_stock_count('B','week')]}));x.update_weight_all();rows.append(norm({s:x.get_stock_weight(s) for s in x.get_stock_list()}))
rows.append([error(lambda:pos(-10.,{'A':{'amount':1.,'price':10.,'weight':.25}}).get_stock_weight_dict(False)),error(lambda:pos(1.,{'A':{'amount':1.,'price':0.,'weight':0.}}).get_stock_weight_dict(True)),error(lambda:pos(0.,{}).get_stock_weight_dict(False)),error(lambda:pos(0.,{'A':{'amount':1.}}).calculate_stock_value()),error(lambda:pos(0.,{}).get_stock_count('Z','day'))])
i=I();rows.append([error(i.calculate_stock_value),error(i.calculate_value),error(i.get_stock_list),error(lambda:i.get_stock_price('A')),error(i.get_stock_amount_dict),error(i.get_stock_weight_dict),error(lambda:i.add_count_all('day')),error(i.update_weight_all),error(lambda:i.update_stock_price('A',1.))])
print(json.dumps(rows,sort_keys=True,separators=(',',':')))
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

fn rust_valuation_snapshot() -> Value {
    let initial = || {
        Position::from_initial(
            20.0,
            IndexMap::from([
                (
                    "A".to_owned(),
                    InitialPositionValue::Holding(PositionHolding::restored(
                        2.0,
                        Some(10.0),
                        Some(0.1),
                    )),
                ),
                (
                    "B".to_owned(),
                    InitialPositionValue::Holding(PositionHolding::restored(
                        3.0,
                        Some(5.0),
                        Some(0.2),
                    )),
                ),
            ]),
        )
    };
    let weights_json = |weights: IndexMap<String, f64>| -> Value {
        Value::Object(
            weights
                .into_iter()
                .map(|(stock, value)| (stock, special(value)))
                .collect(),
        )
    };
    let error = |name: &str| json!(["error", name]);

    let mut position = initial();
    let amounts = weights_json(position.stock_amounts());
    let total_weights = weights_json(position.stock_weights(false).unwrap());
    let stock_weights = weights_json(position.stock_weights(true).unwrap());
    let normal = json!({
        "stock": position.calculate_stock_value().unwrap(),
        "value": position.calculate_value().unwrap(),
        "amounts": amounts,
        "total": total_weights,
        "stocks": stock_weights,
    });
    position.update_stock_price("A", 12.0).unwrap();
    position.update_stock_count("A", "day", 4.0).unwrap();
    position.update_stock_weight("A", 0.7).unwrap();
    position.add_count_all("day");
    position.add_count_all("week");
    let updated = json!({
        "A": [
            position.stock_price("A").unwrap(),
            position.stock_count("A", "day").unwrap(),
            position.stock_count("A", "week").unwrap(),
            position.stock_weight("A").unwrap(),
        ],
        "B": [
            position.stock_count("B", "day").unwrap(),
            position.stock_count("B", "week").unwrap(),
        ],
    });
    position.update_weight_all().unwrap();
    let recalculated = weights_json(
        position
            .stock_ids()
            .map(|stock| (stock.to_owned(), position.stock_weight(stock).unwrap()))
            .collect(),
    );
    let failures = json!([
        error("ZeroDivisionError"),
        error("ZeroDivisionError"),
        ["ok", {}],
        error("KeyError"),
        error("KeyError"),
    ]);
    let infinite = json!([
        ["ok", "Infinity"],
        error("NotImplementedError"),
        error("NotImplementedError"),
        ["ok", "NaN"],
        error("NotImplementedError"),
        error("NotImplementedError"),
        error("NotImplementedError"),
        error("NotImplementedError"),
        ["ok", null],
    ]);
    json!([normal, updated, recalculated, failures, infinite])
}

#[test]
fn finite_position_matches_live_python_source() {
    assert_eq!(rust_snapshot(), live_python_snapshot());
}

#[test]
fn finite_position_valuation_matches_live_python_source() {
    assert_snapshot_within_one_ulp(
        &rust_valuation_snapshot(),
        &live_python_valuation_snapshot(),
    );
}
