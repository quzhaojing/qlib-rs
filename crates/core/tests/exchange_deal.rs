use std::{
    collections::{HashMap, VecDeque},
    path::PathBuf,
    process::Command,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use arrow_array::{ArrayRef, BooleanArray, Float64Array, RecordBatch, TimestampSecondArray};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use chrono::NaiveDateTime;
use domain_core::{
    BuiltInAggregation, DealPriceFields, ExchangeDealError, ExchangeDealExecutor,
    ExchangeQuoteProvider, ExchangeTradeCalculator, ExchangeTradeConfig, ExchangeTradeError,
    ExchangeVolumeLimiter, ExecutionMarketProvider, ExecutionMarketProviderError,
    ExecutionPosition, ExecutionPositionError, ExecutionTarget, ExecutionTargetError, Order,
    OrderDealResult, OrderDir, OrderTradabilityProvider, OrderTradabilityProviderError, Quote,
    QuoteData, QuoteError, QuoteMethod, TimeRange,
};
use serde_json::{Value, json};
use tracing::{
    Event, Metadata, Subscriber,
    span::{Attributes, Id, Record},
};

fn timestamp(text: &str) -> NaiveDateTime {
    NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S").unwrap()
}

fn order(direction: OrderDir, amount: f64) -> Order {
    let mut order = Order::new(
        "A",
        amount,
        direction,
        Some(timestamp("2024-01-02 09:30:00")),
        Some(timestamp("2024-01-02 10:00:00")),
    );
    order.set_deal_amount(7.0);
    order.set_factor(Some(8.0));
    order
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum QuoteCall {
    Stocks,
    Data(String, TimeRange, String, QuoteMethod),
}

enum QuoteResult {
    Missing,
    Float(f64),
    Boolean(bool),
    Series(Vec<Option<f64>>),
    MalformedSeries,
    Failure,
}

struct QueueQuote {
    stocks: Vec<String>,
    results: Mutex<VecDeque<QuoteResult>>,
    calls: Arc<Mutex<Vec<QuoteCall>>>,
}

impl Quote for QueueQuote {
    fn get_all_stock(&self) -> Vec<String> {
        self.calls.lock().unwrap().push(QuoteCall::Stocks);
        self.stocks.clone()
    }

    fn get_data(
        &self,
        stock: &str,
        range: TimeRange,
        field: &str,
        method: QuoteMethod,
    ) -> Result<Option<QuoteData>, QuoteError> {
        self.calls.lock().unwrap().push(QuoteCall::Data(
            stock.to_owned(),
            range,
            field.to_owned(),
            method,
        ));
        match self.results.lock().unwrap().pop_front().unwrap() {
            QuoteResult::Missing => Ok(None),
            QuoteResult::Float(value) => {
                Ok(Some(QuoteData::Scalar(Arc::new(Float64Array::from(vec![
                    value,
                ])))))
            }
            QuoteResult::Boolean(value) => {
                Ok(Some(QuoteData::Scalar(Arc::new(BooleanArray::from(vec![
                    value,
                ])))))
            }
            QuoteResult::Series(values) => Ok(Some(QuoteData::Series(series(values)))),
            QuoteResult::MalformedSeries => Ok(Some(QuoteData::Series(RecordBatch::new_empty(
                Arc::new(Schema::empty()),
            )))),
            QuoteResult::Failure => Err(QuoteError::MissingStock {
                stock: stock.to_owned(),
            }),
        }
    }
}

fn series(values: Vec<Option<f64>>) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "datetime",
            DataType::Timestamp(TimeUnit::Second, None),
            false,
        ),
        Field::new("value", DataType::Float64, true),
    ]));
    let timestamps: ArrayRef = Arc::new(TimestampSecondArray::from(
        (0..values.len())
            .map(|value| Some(i64::try_from(value).unwrap()))
            .collect::<Vec<_>>(),
    ));
    let values: ArrayRef = Arc::new(Float64Array::from(values));
    RecordBatch::try_new(schema, vec![timestamps, values]).unwrap()
}

fn quote_provider(
    stocks: &[&str],
    results: Vec<QuoteResult>,
) -> (ExchangeQuoteProvider, Arc<Mutex<Vec<QuoteCall>>>) {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let quote = QueueQuote {
        stocks: stocks.iter().map(ToString::to_string).collect(),
        results: Mutex::new(results.into()),
        calls: Arc::clone(&calls),
    };
    (
        ExchangeQuoteProvider::new(Arc::new(quote), DealPriceFields::directional("ask", "bid")),
        calls,
    )
}

#[test]
fn exchange_quote_tradability_preserves_suspension_short_circuit_and_limit_truthiness() {
    for (stocks, results, direction, expected, expected_queries) in [
        (vec![], vec![], OrderDir::Buy, false, 0),
        (
            vec!["A"],
            vec![QuoteResult::Missing],
            OrderDir::Buy,
            false,
            1,
        ),
        (
            vec!["A"],
            vec![QuoteResult::Float(f64::NAN)],
            OrderDir::Buy,
            false,
            1,
        ),
        (
            vec!["A"],
            vec![QuoteResult::Series(vec![])],
            OrderDir::Sell,
            false,
            1,
        ),
        (
            vec!["A"],
            vec![
                QuoteResult::Series(vec![None, Some(2.0)]),
                QuoteResult::Missing,
            ],
            OrderDir::Sell,
            true,
            2,
        ),
        (
            vec!["A"],
            vec![QuoteResult::Float(10.0), QuoteResult::Boolean(false)],
            OrderDir::Buy,
            true,
            2,
        ),
        (
            vec!["A"],
            vec![QuoteResult::Float(10.0), QuoteResult::Float(f64::NAN)],
            OrderDir::Buy,
            false,
            2,
        ),
        (
            vec!["A"],
            vec![QuoteResult::Float(10.0), QuoteResult::Float(-1.0)],
            OrderDir::Sell,
            false,
            2,
        ),
    ] {
        let (provider, calls) = quote_provider(&stocks, results);
        let candidate = order(direction, 1.0);
        assert_eq!(provider.is_tradable(&candidate), Ok(expected));
        let calls = calls.lock().unwrap();
        assert_eq!(
            calls
                .iter()
                .filter(|call| matches!(call, QuoteCall::Data(..)))
                .count(),
            expected_queries
        );
        if expected_queries == 2 {
            let field = if direction == OrderDir::Buy {
                "limit_buy"
            } else {
                "limit_sell"
            };
            assert!(matches!(
                calls.last(),
                Some(QuoteCall::Data(_, _, actual, QuoteMethod::BuiltIn(BuiltInAggregation::All)))
                    if actual == field
            ));
        }
    }
}

#[test]
fn exchange_quote_tradability_propagates_close_and_limit_shape_or_provider_failures() {
    for results in [
        vec![QuoteResult::Failure],
        vec![QuoteResult::MalformedSeries],
        vec![QuoteResult::Float(10.0), QuoteResult::Failure],
        vec![QuoteResult::Float(10.0), QuoteResult::MalformedSeries],
    ] {
        let (provider, _) = quote_provider(&["A"], results);
        assert!(provider.is_tradable(&order(OrderDir::Buy, 1.0)).is_err());
    }
}

#[derive(Clone)]
struct StaticMarket {
    price: Result<Option<f64>, ExecutionMarketProviderError>,
}

impl ExecutionMarketProvider for StaticMarket {
    fn deal_price(
        &self,
        _stock: &str,
        _range: TimeRange,
        _direction: OrderDir,
    ) -> Result<Option<f64>, ExecutionMarketProviderError> {
        self.price.clone()
    }

    fn market_volume(
        &self,
        _stock: &str,
        _range: TimeRange,
    ) -> Result<Option<f64>, ExecutionMarketProviderError> {
        Ok(Some(1000.0))
    }

    fn factor(
        &self,
        _stock: &str,
        _range: TimeRange,
    ) -> Result<Option<f64>, ExecutionMarketProviderError> {
        Ok(None)
    }
}

#[derive(Clone)]
struct StaticTradability {
    result: Result<bool, OrderTradabilityProviderError>,
    calls: Arc<AtomicUsize>,
}

impl OrderTradabilityProvider for StaticTradability {
    fn is_tradable(&self, _order: &Order) -> Result<bool, OrderTradabilityProviderError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.result.clone()
    }
}

#[derive(Default)]
struct PositionView;

impl ExecutionPosition for PositionView {
    fn check_stock(&self, _stock: &str) -> Result<bool, ExecutionPositionError> {
        Ok(true)
    }

    fn stock_amount(&self, _stock: &str) -> Result<f64, ExecutionPositionError> {
        Ok(100.0)
    }

    fn cash(&self) -> Result<f64, ExecutionPositionError> {
        Ok(f64::INFINITY)
    }
}

#[derive(Clone, Debug, PartialEq)]
struct UpdateCall {
    deal_amount: f64,
    trade_value: f64,
    trade_cost: f64,
    trade_price: f64,
}

struct TestTarget {
    view: PositionView,
    view_error: Option<ExecutionTargetError>,
    update_error: Option<ExecutionTargetError>,
    position_calls: Arc<AtomicUsize>,
    updates: Vec<UpdateCall>,
}

impl TestTarget {
    fn successful() -> Self {
        Self {
            view: PositionView,
            view_error: None,
            update_error: None,
            position_calls: Arc::new(AtomicUsize::new(0)),
            updates: Vec::new(),
        }
    }
}

impl ExecutionTarget for TestTarget {
    fn position(&self) -> Result<&dyn ExecutionPosition, ExecutionTargetError> {
        self.position_calls.fetch_add(1, Ordering::Relaxed);
        self.view_error
            .as_ref()
            .map_or(Ok(&self.view), |error| Err(error.clone()))
    }

    fn update_order(
        &mut self,
        order: &Order,
        trade_value: f64,
        trade_cost: f64,
        trade_price: f64,
    ) -> Result<(), ExecutionTargetError> {
        self.updates.push(UpdateCall {
            deal_amount: order.deal_amount(),
            trade_value,
            trade_cost,
            trade_price,
        });
        self.update_error.clone().map_or(Ok(()), Err)
    }
}

fn make_executor(
    tradability: Result<bool, OrderTradabilityProviderError>,
    price: Result<Option<f64>, ExecutionMarketProviderError>,
) -> (ExchangeDealExecutor, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let calculator = ExchangeTradeCalculator::new(
        ExchangeTradeConfig {
            open_cost: 0.0,
            close_cost: 0.0,
            min_cost: 0.0,
            impact_cost: 0.0,
            trade_with_adjusted_price: true,
            trade_unit: None,
        },
        Arc::new(StaticMarket { price }),
        ExchangeVolumeLimiter::new(None, None, None),
    );
    (
        ExchangeDealExecutor::new(
            Arc::new(StaticTradability {
                result: tradability,
                calls: Arc::clone(&calls),
            }),
            calculator,
        ),
        calls,
    )
}

struct EventCounter(Arc<AtomicUsize>);

impl Subscriber for EventCounter {
    fn enabled(&self, _metadata: &Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, _span: &Attributes<'_>) -> Id {
        Id::from_u64(1)
    }

    fn record(&self, _span: &Id, _values: &Record<'_>) {}

    fn record_follows_from(&self, _span: &Id, _follows: &Id) {}

    fn event(&self, _event: &Event<'_>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }

    fn enter(&self, _span: &Id) {}

    fn exit(&self, _span: &Id) {}
}

#[test]
fn blocked_orders_short_circuit_conflicts_calculation_and_update_after_mutation() {
    let (executor, calls) = make_executor(Ok(false), Ok(None));
    let mut account = TestTarget::successful();
    let mut position = TestTarget::successful();
    let mut candidate = order(OrderDir::Buy, 1.0);
    let events = Arc::new(AtomicUsize::new(0));
    let result = tracing::subscriber::with_default(EventCounter(Arc::clone(&events)), || {
        executor.deal_order(
            &mut candidate,
            Some(&mut account),
            Some(&mut position),
            &HashMap::new(),
        )
    })
    .unwrap();
    assert_eq!(result.trade_value.to_bits(), 0.0_f64.to_bits());
    assert_eq!(result.trade_cost.to_bits(), 0.0_f64.to_bits());
    assert!(result.trade_price.is_nan());
    assert_eq!(candidate.deal_amount().to_bits(), 0.0_f64.to_bits());
    assert_eq!(candidate.factor(), Some(8.0));
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(events.load(Ordering::Relaxed), 1);
    assert!(account.updates.is_empty() && position.updates.is_empty());
}

#[test]
fn tradable_orders_reject_conflicts_then_route_none_account_and_position_targets() {
    let (executor, _) = make_executor(Ok(true), Ok(Some(10.0)));
    let mut account = TestTarget::successful();
    let mut position = TestTarget::successful();
    let mut conflict = order(OrderDir::Buy, 1.0);
    assert_eq!(
        executor.deal_order(
            &mut conflict,
            Some(&mut account),
            Some(&mut position),
            &HashMap::new(),
        ),
        Err(ExchangeDealError::ConflictingTargets)
    );
    assert_eq!(conflict.deal_amount().to_bits(), 7.0_f64.to_bits());

    let mut no_target = order(OrderDir::Buy, 1.0);
    let result = executor
        .deal_order(&mut no_target, None, None, &HashMap::new())
        .unwrap();
    assert_eq!(
        result,
        OrderDealResult {
            trade_value: 10.0,
            trade_cost: 0.0,
            trade_price: 10.0,
        }
    );

    for account_selected in [true, false] {
        let mut target = TestTarget::successful();
        let mut candidate = order(OrderDir::Buy, 1.0);
        let result = if account_selected {
            executor.deal_order(
                &mut candidate,
                Some(&mut target),
                None,
                &HashMap::from([("A".to_owned(), 1.5)]),
            )
        } else {
            executor.deal_order(
                &mut candidate,
                None,
                Some(&mut target),
                &HashMap::from([("A".to_owned(), 1.5)]),
            )
        }
        .unwrap();
        assert_eq!(target.position_calls.load(Ordering::Relaxed), 1);
        assert_eq!(target.updates.len(), 1);
        assert_eq!(target.updates[0].deal_amount.to_bits(), 1.0_f64.to_bits());
        assert_eq!(
            target.updates[0].trade_value.to_bits(),
            result.trade_value.to_bits()
        );
    }
}

#[test]
fn update_threshold_and_nonfinite_values_match_python_comparison() {
    for (price, expected_updates) in [
        (1e-5, 0),
        (f64::from_bits(1e-5_f64.to_bits() + 1), 1),
        (f64::NAN, 0),
        (-1.0, 0),
        (f64::INFINITY, 1),
    ] {
        let (executor, _) = make_executor(Ok(true), Ok(Some(price)));
        let mut target = TestTarget::successful();
        let mut candidate = order(OrderDir::Buy, 1.0);
        let result = executor
            .deal_order(&mut candidate, None, Some(&mut target), &HashMap::new())
            .unwrap();
        assert_eq!(result.trade_value.to_bits(), price.to_bits());
        assert_eq!(target.updates.len(), expected_updates);
    }
}

#[test]
fn every_provider_calculation_target_access_and_update_failure_is_typed_and_ordered() {
    let tradability_failure = OrderTradabilityProviderError {
        message: "policy offline".to_owned(),
    };
    let (executor, _) = make_executor(Err(tradability_failure.clone()), Ok(Some(10.0)));
    let mut candidate = order(OrderDir::Buy, 1.0);
    assert_eq!(
        executor.deal_order(&mut candidate, None, None, &HashMap::new()),
        Err(ExchangeDealError::Tradability(tradability_failure))
    );
    assert_eq!(candidate.deal_amount().to_bits(), 7.0_f64.to_bits());

    let (executor, _) = make_executor(Ok(true), Ok(None));
    assert_eq!(
        executor.deal_order(&mut candidate, None, None, &HashMap::new()),
        Err(ExchangeDealError::Trade(
            ExchangeTradeError::MissingDealPrice
        ))
    );

    let mut target = TestTarget::successful();
    assert_eq!(
        executor.deal_order(&mut candidate, None, Some(&mut target), &HashMap::new(),),
        Err(ExchangeDealError::Trade(
            ExchangeTradeError::MissingDealPrice
        ))
    );
    assert_eq!(target.position_calls.load(Ordering::Relaxed), 1);
    assert!(target.updates.is_empty());

    let target_failure = ExecutionTargetError {
        message: "target offline".to_owned(),
    };
    let (executor, _) = make_executor(Ok(true), Ok(Some(10.0)));
    let mut target = TestTarget::successful();
    target.view_error = Some(target_failure.clone());
    let mut candidate = order(OrderDir::Buy, 1.0);
    assert_eq!(
        executor.deal_order(&mut candidate, None, Some(&mut target), &HashMap::new(),),
        Err(ExchangeDealError::Target(target_failure.clone()))
    );
    assert_eq!(candidate.deal_amount().to_bits(), 7.0_f64.to_bits());

    target.view_error = None;
    target.update_error = Some(target_failure.clone());
    assert_eq!(
        executor.deal_order(&mut candidate, None, Some(&mut target), &HashMap::new(),),
        Err(ExchangeDealError::Target(target_failure))
    );
    assert_eq!(candidate.deal_amount().to_bits(), 1.0_f64.to_bits());
    assert_eq!(target.updates.len(), 1);
}

fn float_snapshot(value: f64) -> Value {
    json!(value.to_bits())
}

#[test]
fn deal_order_matches_live_python_source_for_results_mutation_threshold_and_ordering() {
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/backtest/exchange.py");
    let script = r"
import ast,json,math,struct,sys,numpy as np
from collections import defaultdict
p=sys.argv[1];t=ast.parse(open(p,encoding='utf-8').read());c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='Exchange');nodes=[next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==x) for x in ['check_order','deal_order']]
class S(ast.NodeTransformer):
 def visit_FunctionDef(self,n):
  n.returns=None
  for a in [*n.args.posonlyargs,*n.args.args,*n.args.kwonlyargs]:a.annotation=None
  return self.generic_visit(n)
nodes=[S().visit(n) for n in nodes];ns={'np':np,'defaultdict':defaultdict};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),p,'exec'),ns)
class O:
 def __init__(self):self.stock_id='A';self.start_time='s';self.end_time='e';self.direction=1;self.amount=1.;self.deal_amount=7.;self.factor=8.
 def __repr__(self):return 'ORDER'
class L:
 def debug(self,x):pass
class P:
 def __init__(self):self.n=0
 def update_order(self,**kw):self.n+=1
class E:
 check_order=ns['check_order'];deal_order=ns['deal_order']
 def __init__(self,ok,p):self.ok=ok;self.p=p;self.logger=L()
 def is_stock_tradable(self,*a):return self.ok
 def _calc_trade_info_by_order(self,o,pos,dealt):
  o.deal_amount=1.;o.factor=None;c=-float('nan') if math.isinf(self.p) else self.p if math.isnan(self.p) else 0.;return self.p,self.p,c
def f(x):return struct.unpack('>Q',struct.pack('>d',x))[0]
out=[]
for ok,p,target in [(False,10.,False),(True,10.,False),(True,1e-5,True),(True,np.nextafter(1e-5,np.inf),True),(True,float('nan'),True),(True,-1.,True),(True,float('inf'),True)]:
 o=O();q=P();r=E(ok,p).deal_order(o,position=q if target else None,dealt_order_amount={'A':1.5});out.append([[f(x) for x in r],f(o.deal_amount),o.factor,q.n])
print(json.dumps(out,separators=(',',':')))
";
    let output = Command::new("python")
        .arg("-c")
        .arg(script)
        .arg(&source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let python: Value = serde_json::from_slice(&output.stdout).unwrap();

    let mut rust = Vec::new();
    for (tradable, price, selected) in [
        (false, 10.0, false),
        (true, 10.0, false),
        (true, 1e-5, true),
        (true, f64::from_bits(1e-5_f64.to_bits() + 1), true),
        (true, f64::NAN, true),
        (true, -1.0, true),
        (true, f64::INFINITY, true),
    ] {
        let (executor, _) = make_executor(Ok(tradable), Ok(Some(price)));
        let mut candidate = order(OrderDir::Buy, 1.0);
        let mut target = TestTarget::successful();
        let result = if selected {
            executor.deal_order(
                &mut candidate,
                None,
                Some(&mut target),
                &HashMap::from([("A".to_owned(), 1.5)]),
            )
        } else {
            executor.deal_order(
                &mut candidate,
                None,
                None,
                &HashMap::from([("A".to_owned(), 1.5)]),
            )
        }
        .unwrap();
        rust.push(json!([
            [
                float_snapshot(result.trade_value),
                float_snapshot(result.trade_cost),
                float_snapshot(result.trade_price)
            ],
            float_snapshot(candidate.deal_amount()),
            candidate.factor(),
            target.updates.len()
        ]));
    }
    assert_eq!(Value::Array(rust), python);
}
