use std::{
    collections::{HashMap, VecDeque},
    path::PathBuf,
    process::Command,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use arrow_array::{Float64Array, RecordBatch};
use arrow_schema::Schema;
use chrono::NaiveDateTime;
use domain_core::{
    DealPriceFields, ExchangeQuoteProvider, ExchangeVolumeError, ExchangeVolumeLimiter, Order,
    OrderDir, Quote, QuoteData, QuoteError, QuoteMethod, TimeRange, VolumeLimit, VolumeLimitKind,
    VolumeLimitProvider, VolumeLimitProviderError,
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
        100.0,
        direction,
        Some(timestamp("2024-01-02 09:30:00")),
        Some(timestamp("2024-01-02 10:00:00")),
    );
    order.set_deal_amount(amount);
    order
}

fn current(field: &str) -> VolumeLimit {
    VolumeLimit::new(VolumeLimitKind::Current, field)
}

fn cumulative(field: &str) -> VolumeLimit {
    VolumeLimit::new(VolumeLimitKind::Cumulative, field)
}

type ProviderCall = (String, TimeRange, VolumeLimit);
type ProviderResults = VecDeque<Result<Option<f64>, VolumeLimitProviderError>>;
type QuoteCall = (String, TimeRange, String, QuoteMethod);

#[derive(Clone)]
struct TestVolumeProvider {
    results: Arc<Mutex<ProviderResults>>,
    calls: Arc<Mutex<Vec<ProviderCall>>>,
}

impl TestVolumeProvider {
    fn new(results: Vec<Result<Option<f64>, VolumeLimitProviderError>>) -> Self {
        Self {
            results: Arc::new(Mutex::new(results.into())),
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl VolumeLimitProvider for TestVolumeProvider {
    fn volume_limit(
        &self,
        stock: &str,
        range: TimeRange,
        limit: &VolumeLimit,
    ) -> Result<Option<f64>, VolumeLimitProviderError> {
        self.calls
            .lock()
            .unwrap()
            .push((stock.to_owned(), range, limit.clone()));
        self.results.lock().unwrap().pop_front().unwrap()
    }
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
fn typed_configuration_and_directional_short_circuits_are_explicit() {
    assert_eq!(
        VolumeLimitKind::from_python_name("current"),
        Ok(VolumeLimitKind::Current)
    );
    assert_eq!(
        VolumeLimitKind::from_python_name("cum"),
        Ok(VolumeLimitKind::Cumulative)
    );
    assert_eq!(VolumeLimitKind::Current.python_name(), "current");
    assert_eq!(VolumeLimitKind::Cumulative.python_name(), "cum");
    assert_eq!(
        VolumeLimitKind::from_python_name("future"),
        Err(ExchangeVolumeError::UnsupportedLimitKind {
            kind: "future".to_owned()
        })
    );

    let limit = current("$askV1");
    assert_eq!(limit.kind(), VolumeLimitKind::Current);
    assert_eq!(limit.field(), "$askV1");

    let unrestricted = ExchangeVolumeLimiter::new(None, None, None);
    assert_eq!(unrestricted.buy_limits(), None);
    assert_eq!(unrestricted.sell_limits(), None);
    for direction in [OrderDir::Buy, OrderDir::Sell] {
        let mut candidate = order(direction, -0.0);
        assert_eq!(
            unrestricted.clip_amount_by_volume(&mut candidate, &HashMap::new()),
            Ok(Some(-0.0))
        );
        assert_eq!(candidate.deal_amount().to_bits(), (-0.0_f64).to_bits());
    }

    let configured = ExchangeVolumeLimiter::new(Some(vec![limit.clone()]), Some(vec![]), None);
    assert_eq!(configured.buy_limits(), Some([limit].as_slice()));
    assert_eq!(configured.sell_limits(), Some([].as_slice()));

    let mut buy = order(OrderDir::Buy, 10.0);
    assert_eq!(
        configured.clip_amount_by_volume(&mut buy, &HashMap::new()),
        Err(ExchangeVolumeError::MissingProvider)
    );
    assert_eq!(buy.deal_amount().to_bits(), 10.0_f64.to_bits());

    let mut sell = order(OrderDir::Sell, 10.0);
    assert_eq!(
        configured.clip_amount_by_volume(&mut sell, &HashMap::new()),
        Err(ExchangeVolumeError::EmptyLimits {
            direction: OrderDir::Sell
        })
    );
    assert_eq!(sell.deal_amount().to_bits(), 10.0_f64.to_bits());
}

fn clipped_bits(
    direction: OrderDir,
    limits: Vec<VolumeLimit>,
    values: Vec<f64>,
    dealt: Option<f64>,
    original: f64,
) -> u64 {
    let provider: Arc<dyn VolumeLimitProvider> = Arc::new(TestVolumeProvider::new(
        values.into_iter().map(|v| Ok(Some(v))).collect(),
    ));
    let (buy, sell) = match direction {
        OrderDir::Buy => (Some(limits), None),
        OrderDir::Sell => (None, Some(limits)),
    };
    let limiter = ExchangeVolumeLimiter::new(buy, sell, Some(provider));
    let mut order = order(direction, original);
    let dealt = dealt.map_or_else(HashMap::new, |value| {
        HashMap::from([("A".to_owned(), value)])
    });
    assert_eq!(limiter.clip_amount_by_volume(&mut order, &dealt), Ok(None));
    order.deal_amount().to_bits()
}

#[test]
fn clipping_preserves_query_order_cumulative_math_and_python_float_ordering() {
    let provider = TestVolumeProvider::new(vec![Ok(Some(9.0)), Ok(Some(20.0))]);
    let calls = Arc::clone(&provider.calls);
    let limiter = ExchangeVolumeLimiter::new(
        Some(vec![current("current"), cumulative("cum")]),
        None,
        Some(Arc::new(provider)),
    );
    let mut candidate = order(OrderDir::Buy, 10.0);
    let events = Arc::new(AtomicUsize::new(0));
    tracing::subscriber::with_default(EventCounter(Arc::clone(&events)), || {
        assert_eq!(
            limiter.clip_amount_by_volume(&mut candidate, &HashMap::from([("A".to_owned(), 15.0)])),
            Ok(None)
        );
    });
    assert_eq!(events.load(Ordering::Relaxed), 1);
    assert_eq!(candidate.deal_amount().to_bits(), 5.0_f64.to_bits());
    assert_eq!(
        calls.lock().unwrap().as_slice(),
        [
            (
                "A".to_owned(),
                TimeRange {
                    start: Some(timestamp("2024-01-02 09:30:00")),
                    end: Some(timestamp("2024-01-02 10:00:00"))
                },
                current("current")
            ),
            (
                "A".to_owned(),
                TimeRange {
                    start: Some(timestamp("2024-01-02 09:30:00")),
                    end: Some(timestamp("2024-01-02 10:00:00"))
                },
                cumulative("cum")
            )
        ]
    );

    assert_eq!(
        clipped_bits(OrderDir::Sell, vec![current("sell")], vec![3.0], None, 10.0),
        3.0_f64.to_bits()
    );
    assert_eq!(
        clipped_bits(
            OrderDir::Buy,
            vec![current("large")],
            vec![20.0],
            None,
            10.0
        ),
        10.0_f64.to_bits()
    );

    for (limits, values, dealt, original, expected) in [
        (
            vec![current("nan"), current("five")],
            vec![f64::NAN, 5.0],
            None,
            10.0,
            f64::NAN,
        ),
        (
            vec![current("five"), current("nan")],
            vec![5.0, f64::NAN],
            None,
            10.0,
            5.0,
        ),
        (vec![current("negative")], vec![-3.0], None, 10.0, 0.0),
        (
            vec![current("positive-infinity")],
            vec![f64::INFINITY],
            None,
            10.0,
            10.0,
        ),
        (
            vec![current("negative-infinity")],
            vec![f64::NEG_INFINITY],
            None,
            10.0,
            0.0,
        ),
        (vec![current("negative-zero")], vec![-0.0], None, 10.0, -0.0),
        (vec![current("finite")], vec![5.0], None, f64::NAN, 5.0),
        (
            vec![cumulative("cumulative")],
            vec![12.0],
            Some(f64::NAN),
            10.0,
            f64::NAN,
        ),
    ] {
        assert_eq!(
            clipped_bits(OrderDir::Buy, limits, values, dealt, original),
            expected.to_bits()
        );
    }
}

#[test]
fn provider_and_cumulative_failures_are_atomic_and_ordered() {
    let failure = VolumeLimitProviderError {
        message: "offline".to_owned(),
    };
    let provider = TestVolumeProvider::new(vec![Err(failure.clone())]);
    let calls = Arc::clone(&provider.calls);
    let limiter =
        ExchangeVolumeLimiter::new(Some(vec![current("first")]), None, Some(Arc::new(provider)));
    let mut candidate = order(OrderDir::Buy, 10.0);
    assert_eq!(
        limiter.clip_amount_by_volume(&mut candidate, &HashMap::new()),
        Err(ExchangeVolumeError::Provider(failure))
    );
    assert_eq!(calls.lock().unwrap().len(), 1);
    assert_eq!(candidate.deal_amount().to_bits(), 10.0_f64.to_bits());

    let provider = TestVolumeProvider::new(vec![Ok(None), Ok(Some(4.0))]);
    let calls = Arc::clone(&provider.calls);
    let limiter = ExchangeVolumeLimiter::new(
        Some(vec![current("missing"), current("second")]),
        None,
        Some(Arc::new(provider)),
    );
    let mut candidate = order(OrderDir::Buy, 10.0);
    assert_eq!(
        limiter.clip_amount_by_volume(&mut candidate, &HashMap::new()),
        Err(ExchangeVolumeError::MissingLimitValue {
            field: "missing".to_owned()
        })
    );
    assert_eq!(calls.lock().unwrap().len(), 2);
    assert_eq!(candidate.deal_amount().to_bits(), 10.0_f64.to_bits());

    let provider = TestVolumeProvider::new(vec![Ok(Some(4.0)), Ok(None)]);
    let calls = Arc::clone(&provider.calls);
    let limiter = ExchangeVolumeLimiter::new(
        Some(vec![current("first"), current("missing-second")]),
        None,
        Some(Arc::new(provider)),
    );
    let mut candidate = order(OrderDir::Buy, 10.0);
    assert_eq!(
        limiter.clip_amount_by_volume(&mut candidate, &HashMap::new()),
        Err(ExchangeVolumeError::MissingLimitValue {
            field: "missing-second".to_owned()
        })
    );
    assert_eq!(calls.lock().unwrap().len(), 2);
    assert_eq!(candidate.deal_amount().to_bits(), 10.0_f64.to_bits());

    let provider = TestVolumeProvider::new(vec![Ok(None)]);
    let limiter = ExchangeVolumeLimiter::new(
        Some(vec![cumulative("cum")]),
        None,
        Some(Arc::new(provider)),
    );
    let mut candidate = order(OrderDir::Buy, 10.0);
    assert_eq!(
        limiter.clip_amount_by_volume(&mut candidate, &HashMap::from([("A".to_owned(), 1.0)])),
        Err(ExchangeVolumeError::MissingLimitValue {
            field: "cum".to_owned()
        })
    );
    assert_eq!(candidate.deal_amount().to_bits(), 10.0_f64.to_bits());

    let provider = TestVolumeProvider::new(vec![Ok(Some(12.0))]);
    let calls = Arc::clone(&provider.calls);
    let limiter = ExchangeVolumeLimiter::new(
        Some(vec![cumulative("cum")]),
        None,
        Some(Arc::new(provider)),
    );
    let mut candidate = order(OrderDir::Buy, 10.0);
    assert_eq!(
        limiter.clip_amount_by_volume(&mut candidate, &HashMap::new()),
        Err(ExchangeVolumeError::MissingDealtAmount {
            stock: "A".to_owned()
        })
    );
    assert_eq!(calls.lock().unwrap().len(), 1);
    assert_eq!(candidate.deal_amount().to_bits(), 10.0_f64.to_bits());
}

#[derive(Clone)]
enum LimitQuoteResult {
    Scalar(f64),
    Missing,
    Series,
    Failure,
}

struct LimitQuote {
    result: LimitQuoteResult,
    calls: Arc<Mutex<Vec<QuoteCall>>>,
}

impl Quote for LimitQuote {
    fn get_all_stock(&self) -> Vec<String> {
        vec!["A".to_owned()]
    }

    fn get_data(
        &self,
        stock: &str,
        range: TimeRange,
        field: &str,
        method: QuoteMethod,
    ) -> Result<Option<QuoteData>, QuoteError> {
        self.calls
            .lock()
            .unwrap()
            .push((stock.to_owned(), range, field.to_owned(), method));
        match self.result {
            LimitQuoteResult::Scalar(value) => {
                Ok(Some(QuoteData::Scalar(Arc::new(Float64Array::from(vec![
                    value,
                ])))))
            }
            LimitQuoteResult::Missing => Ok(None),
            LimitQuoteResult::Series => Ok(Some(QuoteData::Series(RecordBatch::new_empty(
                Arc::new(Schema::empty()),
            )))),
            LimitQuoteResult::Failure => Err(QuoteError::MissingStock {
                stock: stock.to_owned(),
            }),
        }
    }
}

#[test]
fn exchange_quote_provider_supplies_typed_volume_limit_scalars() {
    for (kind, method) in [
        (
            VolumeLimitKind::Current,
            QuoteMethod::BuiltIn(domain_core::BuiltInAggregation::Sum),
        ),
        (VolumeLimitKind::Cumulative, QuoteMethod::LastValid),
    ] {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let exchange = ExchangeQuoteProvider::new(
            Arc::new(LimitQuote {
                result: LimitQuoteResult::Scalar(7.0),
                calls: Arc::clone(&calls),
            }),
            DealPriceFields::shared("close").unwrap(),
        );
        assert_eq!(
            VolumeLimitProvider::volume_limit(
                &exchange,
                "A",
                TimeRange::default(),
                &VolumeLimit::new(kind, "capacity")
            ),
            Ok(Some(7.0))
        );
        assert_eq!(
            calls.lock().unwrap().as_slice(),
            [(
                "A".to_owned(),
                TimeRange::default(),
                "capacity".to_owned(),
                method
            )]
        );
    }

    for (result, expected) in [
        (LimitQuoteResult::Missing, Ok(None)),
        (
            LimitQuoteResult::Series,
            Err(VolumeLimitProviderError {
                message: "aggregated quote result must be scalar, not a series".to_owned(),
            }),
        ),
        (
            LimitQuoteResult::Failure,
            Err(VolumeLimitProviderError {
                message: "stock A is not present in the quote".to_owned(),
            }),
        ),
    ] {
        let exchange = ExchangeQuoteProvider::new(
            Arc::new(LimitQuote {
                result,
                calls: Arc::new(Mutex::new(Vec::new())),
            }),
            DealPriceFields::shared("close").unwrap(),
        );
        assert_eq!(
            VolumeLimitProvider::volume_limit(
                &exchange,
                "A",
                TimeRange::default(),
                &current("capacity")
            ),
            expected
        );
    }
}

#[test]
fn volume_clipping_matches_live_python_exchange_method() {
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/backtest/exchange.py");
    let script = r"
import ast,json,math,struct,sys
from typing import List,Optional,cast
class Order:
 BUY=1;SELL=0
 def __init__(self,d=1,a=10.0):self.stock_id='A';self.direction=d;self.deal_amount=a;self.start_time='s';self.end_time='e'
 def __repr__(self):return 'ORDER'
p=sys.argv[1];t=ast.parse(open(p,encoding='utf-8').read(),filename=p);c=next(x for x in t.body if isinstance(x,ast.ClassDef) and x.name=='Exchange');m=next(x for x in c.body if isinstance(x,ast.FunctionDef) and x.name=='_clip_amount_by_volume');ns={};exec(compile(ast.Module(body=[m],type_ignores=[]),p,'exec'),globals(),ns)
class Q:
 def __init__(self,v):self.v=iter(v);self.calls=[]
 def get_data(self,*a,**k):self.calls.append([k['field'],k['method']]);x=next(self.v);(_ for _ in ()).throw(x) if isinstance(x,BaseException) else None;return x
class L:
 def debug(self,*a):pass
class X:_clip_amount_by_volume=ns['_clip_amount_by_volume']
def bits(x):return struct.unpack('>Q',struct.pack('>d',float(x)))[0]
def run(buy,sell,values,dealt,d=1,a=10.0):
 x=X();x.buy_vol_limit=buy;x.sell_vol_limit=sell;x.quote=Q(values);x.logger=L();o=Order(d,a)
 try:r=x._clip_amount_by_volume(o,dealt);out=['ok',None if r is None else bits(r)]
 except BaseException as e:out=[type(e).__name__,str(e)]
 return [out,bits(o.deal_amount),x.quote.calls]
r={'none':run(None,None,[],{},a=-0.0),'current':run([('current','c')],None,[4.0],{}),'large':run([('current','c')],None,[20.0],{}),'cum':run([('cum','u')],None,[12.0],{'A':5.0}),'mixed':run([('current','c'),('cum','u')],None,[9.0,20.0],{'A':15.0}),'sell':run(None,[('current','s')],[3.0],{},d=0),'nan_first':run([('current','n'),('current','f')],None,[float('nan'),5.0],{}),'nan_second':run([('current','f'),('current','n')],None,[5.0,float('nan')],{}),'negative_zero':run([('current','z')],None,[-0.0],{}),'empty':run([],None,[],{}),'missing_dealt':run([('cum','u')],None,[12.0],{}),'provider_error':run([('current','c')],None,[RuntimeError('offline')],{})};print(json.dumps(r,separators=(',',':')))
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
        "none": [["ok", (-0.0_f64).to_bits()], (-0.0_f64).to_bits(), []],
        "current": [["ok", null], 4.0_f64.to_bits(), [["c", "sum"]]],
        "large": [["ok", null], 10.0_f64.to_bits(), [["c", "sum"]]],
        "cum": [["ok", null], 7.0_f64.to_bits(), [["u", "ts_data_last"]]],
        "mixed": [["ok", null], 5.0_f64.to_bits(), [["c", "sum"], ["u", "ts_data_last"]]],
        "sell": [["ok", null], 3.0_f64.to_bits(), [["s", "sum"]]],
        "nan_first": [["ok", null], f64::NAN.to_bits(), [["n", "sum"], ["f", "sum"]]],
        "nan_second": [["ok", null], 5.0_f64.to_bits(), [["f", "sum"], ["n", "sum"]]],
        "negative_zero": [["ok", null], (-0.0_f64).to_bits(), [["z", "sum"]]],
        "empty": [["ValueError", "min() iterable argument is empty"], 10.0_f64.to_bits(), []],
        "missing_dealt": [["KeyError", "'A'"], 10.0_f64.to_bits(), [["u", "ts_data_last"]]],
        "provider_error": [["RuntimeError", "offline"], 10.0_f64.to_bits(), [["c", "sum"]]]
    });
    assert_eq!(actual, expected);
}
