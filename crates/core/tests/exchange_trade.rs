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
    BuiltInAggregation, DealPriceFields, ExchangeQuoteProvider, ExchangeSizingError,
    ExchangeTradeCalculator, ExchangeTradeConfig, ExchangeTradeError, ExchangeVolumeLimiter,
    ExecutionMarketProvider, ExecutionMarketProviderError, ExecutionPosition,
    ExecutionPositionError, Order, OrderDir, Quote, QuoteData, QuoteError, QuoteMethod, TimeRange,
    TradeInfo, VolumeLimit, VolumeLimitKind, VolumeLimitProvider, VolumeLimitProviderError,
};
use serde_json::{Value, json};
use tracing::{
    Event, Metadata, Subscriber,
    span::{Attributes, Id, Record},
};

fn timestamp(text: &str) -> NaiveDateTime {
    NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S").unwrap()
}

fn bounded_range() -> TimeRange {
    TimeRange {
        start: Some(timestamp("2024-01-02 09:30:00")),
        end: Some(timestamp("2024-01-02 10:00:00")),
    }
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

fn config() -> ExchangeTradeConfig {
    ExchangeTradeConfig {
        open_cost: 0.01,
        close_cost: 0.02,
        min_cost: 5.0,
        impact_cost: 0.1,
        trade_with_adjusted_price: false,
        trade_unit: Some(10.0),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum MarketCall {
    Price(String, TimeRange, OrderDir),
    Volume(String, TimeRange),
    Factor(String, TimeRange),
}

#[derive(Clone)]
struct TestMarket {
    price: Result<Option<f64>, ExecutionMarketProviderError>,
    volume: Result<Option<f64>, ExecutionMarketProviderError>,
    factor: Result<Option<f64>, ExecutionMarketProviderError>,
    calls: Arc<Mutex<Vec<MarketCall>>>,
}

impl TestMarket {
    fn values(price: Option<f64>, volume: Option<f64>, factor: Option<f64>) -> Self {
        Self {
            price: Ok(price),
            volume: Ok(volume),
            factor: Ok(factor),
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl ExecutionMarketProvider for TestMarket {
    fn deal_price(
        &self,
        stock: &str,
        range: TimeRange,
        direction: OrderDir,
    ) -> Result<Option<f64>, ExecutionMarketProviderError> {
        self.calls
            .lock()
            .unwrap()
            .push(MarketCall::Price(stock.to_owned(), range, direction));
        self.price.clone()
    }

    fn market_volume(
        &self,
        stock: &str,
        range: TimeRange,
    ) -> Result<Option<f64>, ExecutionMarketProviderError> {
        self.calls
            .lock()
            .unwrap()
            .push(MarketCall::Volume(stock.to_owned(), range));
        self.volume.clone()
    }

    fn factor(
        &self,
        stock: &str,
        range: TimeRange,
    ) -> Result<Option<f64>, ExecutionMarketProviderError> {
        self.calls
            .lock()
            .unwrap()
            .push(MarketCall::Factor(stock.to_owned(), range));
        self.factor.clone()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum PositionCall {
    Check(String),
    Amount(String),
    Cash,
}

struct TestPosition {
    check: Result<bool, ExecutionPositionError>,
    amount: Result<f64, ExecutionPositionError>,
    cash: Result<f64, ExecutionPositionError>,
    calls: Arc<Mutex<Vec<PositionCall>>>,
}

impl TestPosition {
    fn values(has_stock: bool, amount: f64, cash: f64) -> Self {
        Self {
            check: Ok(has_stock),
            amount: Ok(amount),
            cash: Ok(cash),
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl ExecutionPosition for TestPosition {
    fn check_stock(&self, stock: &str) -> Result<bool, ExecutionPositionError> {
        self.calls
            .lock()
            .unwrap()
            .push(PositionCall::Check(stock.to_owned()));
        self.check.clone()
    }

    fn stock_amount(&self, stock: &str) -> Result<f64, ExecutionPositionError> {
        self.calls
            .lock()
            .unwrap()
            .push(PositionCall::Amount(stock.to_owned()));
        self.amount.clone()
    }

    fn cash(&self) -> Result<f64, ExecutionPositionError> {
        self.calls.lock().unwrap().push(PositionCall::Cash);
        self.cash.clone()
    }
}

#[derive(Clone)]
struct FixedVolumeProvider {
    result: Result<Option<f64>, VolumeLimitProviderError>,
}

impl VolumeLimitProvider for FixedVolumeProvider {
    fn volume_limit(
        &self,
        _stock: &str,
        _range: TimeRange,
        _limit: &VolumeLimit,
    ) -> Result<Option<f64>, VolumeLimitProviderError> {
        self.result.clone()
    }
}

fn unrestricted() -> ExchangeVolumeLimiter {
    ExchangeVolumeLimiter::new(None, None, None)
}

fn calculator(
    config: ExchangeTradeConfig,
    market: TestMarket,
    limiter: ExchangeVolumeLimiter,
) -> ExchangeTradeCalculator {
    ExchangeTradeCalculator::new(config, Arc::new(market), limiter)
}

fn assert_info_bits(actual: TradeInfo, expected: (f64, f64, f64)) {
    assert_eq!(actual.trade_price.to_bits(), expected.0.to_bits());
    assert_eq!(actual.trade_value.to_bits(), expected.1.to_bits());
    assert_eq!(actual.trade_cost.to_bits(), expected.2.to_bits());
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
fn market_lookup_failures_preserve_python_call_and_mutation_boundaries() {
    let base = config();
    let market = TestMarket::values(Some(10.0), Some(1000.0), Some(1.0));
    let calls = Arc::clone(&market.calls);
    let calc = calculator(base, market, unrestricted());
    assert_eq!(calc.config(), base);
    let mut candidate = order(OrderDir::Buy, 23.0);
    let result = calc
        .calculate(&mut candidate, None, &HashMap::new())
        .unwrap();
    assert_info_bits(result, (10.0, 200.0, 5.0));
    assert_eq!(candidate.deal_amount().to_bits(), 20.0_f64.to_bits());
    assert_eq!(candidate.factor(), Some(1.0));
    assert_eq!(calls.lock().unwrap().len(), 3);

    for (price, volume, expected) in [
        (Some(10.0), None, ExchangeTradeError::MissingMarketVolume),
        (None, Some(1000.0), ExchangeTradeError::MissingDealPrice),
    ] {
        let market = TestMarket::values(price, volume, Some(1.0));
        let calls = Arc::clone(&market.calls);
        let calc = calculator(base, market, unrestricted());
        let mut candidate = order(OrderDir::Buy, 23.0);
        assert_eq!(
            calc.calculate(&mut candidate, None, &HashMap::new()),
            Err(expected)
        );
        assert_eq!(calls.lock().unwrap().len(), 2);
        assert_eq!(candidate.deal_amount().to_bits(), 7.0_f64.to_bits());
        assert_eq!(candidate.factor(), Some(8.0));
    }

    let provider_failure = ExecutionMarketProviderError {
        message: "offline".to_owned(),
    };
    for (stage, expected_calls) in [("price", 1), ("volume", 2), ("factor", 3)] {
        let mut market = TestMarket::values(Some(10.0), Some(1000.0), Some(1.0));
        match stage {
            "price" => market.price = Err(provider_failure.clone()),
            "volume" => market.volume = Err(provider_failure.clone()),
            "factor" => market.factor = Err(provider_failure.clone()),
            _ => unreachable!(),
        }
        let calls = Arc::clone(&market.calls);
        let calc = calculator(base, market, unrestricted());
        let mut candidate = order(OrderDir::Buy, 23.0);
        assert_eq!(
            calc.calculate(&mut candidate, None, &HashMap::new()),
            Err(ExchangeTradeError::MarketProvider(provider_failure.clone()))
        );
        assert_eq!(calls.lock().unwrap().len(), expected_calls);
        assert_eq!(candidate.deal_amount().to_bits(), 7.0_f64.to_bits());
        assert_eq!(candidate.factor(), Some(8.0));
    }

    let limiter = ExchangeVolumeLimiter::new(
        Some(vec![VolumeLimit::new(VolumeLimitKind::Current, "capacity")]),
        None,
        Some(Arc::new(FixedVolumeProvider {
            result: Err(VolumeLimitProviderError {
                message: "capacity offline".to_owned(),
            }),
        })),
    );
    let calc = calculator(
        base,
        TestMarket::values(Some(10.0), Some(1000.0), Some(1.0)),
        limiter,
    );
    let mut candidate = order(OrderDir::Buy, 23.0);
    assert!(matches!(
        calc.calculate(&mut candidate, None, &HashMap::new()),
        Err(ExchangeTradeError::Volume(_))
    ));
    assert_eq!(candidate.deal_amount().to_bits(), 23.0_f64.to_bits());
    assert_eq!(candidate.factor(), Some(1.0));

    let calc = calculator(
        base,
        TestMarket::values(Some(10.0), Some(1000.0), None),
        unrestricted(),
    );
    let mut buy = order(OrderDir::Buy, 23.0);
    assert_eq!(
        calc.calculate(&mut buy, None, &HashMap::new()),
        Err(ExchangeTradeError::Sizing(
            ExchangeSizingError::MissingFactorInput
        ))
    );
    assert_eq!(buy.deal_amount().to_bits(), 23.0_f64.to_bits());
    assert_eq!(buy.factor(), None);

    let mut sell = order(OrderDir::Sell, 23.0);
    let info = calc.calculate(&mut sell, None, &HashMap::new()).unwrap();
    assert_info_bits(info, (10.0, 230.0, 5.0));
    assert_eq!(sell.factor(), None);
}

#[test]
fn buy_cash_paths_round_in_python_order_and_emit_expected_events() {
    let events = Arc::new(AtomicUsize::new(0));
    tracing::subscriber::with_default(EventCounter(Arc::clone(&events)), || {
        for (cash, expected_amount, expected_value, expected_cost) in [
            (1000.0, 20.0, 200.0, 5.0),
            (1.0, 0.0, 0.0, 0.0),
            (150.0, 10.0, 100.0, 5.0),
            (5.0, 0.0, 0.0, 0.0),
            (235.0, 20.0, 200.0, 5.0),
        ]
            as [(f64, f64, f64, f64); 5]
        {
            let position = TestPosition::values(true, 0.0, cash);
            let calc = calculator(
                config(),
                TestMarket::values(Some(10.0), Some(1000.0), Some(1.0)),
                unrestricted(),
            );
            let mut candidate = order(OrderDir::Buy, 23.0);
            let info = calc
                .calculate(&mut candidate, Some(&position), &HashMap::new())
                .unwrap();
            assert_info_bits(info, (10.0, expected_value, expected_cost));
            assert_eq!(candidate.deal_amount().to_bits(), expected_amount.to_bits());
            assert_eq!(
                position.calls.lock().unwrap().as_slice(),
                [PositionCall::Cash]
            );
        }
    });
    assert_eq!(events.load(Ordering::Relaxed), 3);

    let no_unit = ExchangeTradeConfig {
        trade_unit: None,
        ..config()
    };
    let position = TestPosition::values(true, 0.0, 150.0);
    let calc = calculator(
        no_unit,
        TestMarket::values(Some(10.0), Some(1000.0), None),
        unrestricted(),
    );
    let mut candidate = order(OrderDir::Buy, 23.0);
    let info = calc
        .calculate(&mut candidate, Some(&position), &HashMap::new())
        .unwrap();
    assert_eq!(candidate.deal_amount().to_bits(), 14.5_f64.to_bits());
    assert_eq!(info.trade_value.to_bits(), 145.0_f64.to_bits());

    let failure = ExecutionPositionError {
        message: "cash offline".to_owned(),
    };
    let position = TestPosition {
        check: Ok(true),
        amount: Ok(0.0),
        cash: Err(failure.clone()),
        calls: Arc::new(Mutex::new(Vec::new())),
    };
    let calc = calculator(
        config(),
        TestMarket::values(Some(10.0), Some(1000.0), Some(1.0)),
        unrestricted(),
    );
    let mut candidate = order(OrderDir::Buy, 23.0);
    assert_eq!(
        calc.calculate(&mut candidate, Some(&position), &HashMap::new()),
        Err(ExchangeTradeError::Position(failure))
    );
    assert_eq!(candidate.deal_amount().to_bits(), 23.0_f64.to_bits());

    let zero_ratio = ExchangeTradeConfig {
        open_cost: 0.0,
        impact_cost: 0.0,
        ..config()
    };
    let position = TestPosition::values(true, 0.0, 150.0);
    let calc = calculator(
        zero_ratio,
        TestMarket::values(Some(10.0), Some(1000.0), Some(1.0)),
        unrestricted(),
    );
    let mut candidate = order(OrderDir::Buy, 23.0);
    assert_eq!(
        calc.calculate(&mut candidate, Some(&position), &HashMap::new()),
        Err(ExchangeTradeError::Sizing(
            ExchangeSizingError::ZeroCostRatio
        ))
    );
}

#[test]
fn buy_rounding_failures_propagate_from_full_and_cash_limited_paths() {
    for position in [
        None,
        Some(TestPosition::values(true, 0.0, 150.0)),
        Some(TestPosition::values(true, 0.0, 1000.0)),
    ] {
        let calc = calculator(
            config(),
            TestMarket::values(Some(10.0), Some(1000.0), Some(0.0)),
            unrestricted(),
        );
        let mut candidate = order(OrderDir::Buy, 23.0);
        assert_eq!(
            calc.calculate(
                &mut candidate,
                position.as_ref().map(|value| value as _),
                &HashMap::new()
            ),
            Err(ExchangeTradeError::Sizing(ExchangeSizingError::ZeroFactor))
        );
        assert_eq!(candidate.factor(), Some(0.0));
    }
}

#[test]
fn sell_position_paths_preserve_numpy_isclose_rounding_and_failure_order() {
    for (position, expected_amount, expected_calls) in [
        (
            TestPosition::values(true, 23.0, 0.0),
            23.0,
            vec![
                PositionCall::Check("A".to_owned()),
                PositionCall::Amount("A".to_owned()),
                PositionCall::Cash,
            ],
        ),
        (
            TestPosition::values(true, 23.0001, 0.0),
            23.0,
            vec![
                PositionCall::Check("A".to_owned()),
                PositionCall::Amount("A".to_owned()),
                PositionCall::Cash,
            ],
        ),
        (
            TestPosition::values(true, 18.0, 0.0),
            10.0,
            vec![
                PositionCall::Check("A".to_owned()),
                PositionCall::Amount("A".to_owned()),
                PositionCall::Cash,
            ],
        ),
        (
            TestPosition::values(false, 99.0, 0.0),
            0.0,
            vec![PositionCall::Check("A".to_owned()), PositionCall::Cash],
        ),
    ]
        as [(TestPosition, f64, Vec<PositionCall>); 4]
    {
        let calls = Arc::clone(&position.calls);
        let calc = calculator(
            config(),
            TestMarket::values(Some(10.0), Some(1000.0), Some(1.0)),
            unrestricted(),
        );
        let mut candidate = order(OrderDir::Sell, 23.0);
        let info = calc
            .calculate(&mut candidate, Some(&position), &HashMap::new())
            .unwrap();
        assert_eq!(candidate.deal_amount().to_bits(), expected_amount.to_bits());
        assert_eq!(calls.lock().unwrap().as_slice(), expected_calls);
        assert_eq!(
            info.trade_value.to_bits(),
            (expected_amount * 10.0).to_bits()
        );
    }

    let events = Arc::new(AtomicUsize::new(0));
    let position = TestPosition::values(true, 18.0, -200.0);
    let calc = calculator(
        config(),
        TestMarket::values(Some(10.0), Some(1000.0), Some(1.0)),
        unrestricted(),
    );
    let mut candidate = order(OrderDir::Sell, 23.0);
    tracing::subscriber::with_default(EventCounter(Arc::clone(&events)), || {
        let info = calc
            .calculate(&mut candidate, Some(&position), &HashMap::new())
            .unwrap();
        assert_info_bits(info, (10.0, 0.0, 0.0));
    });
    assert_eq!(events.load(Ordering::Relaxed), 1);
}

#[test]
fn sell_nonfinite_position_and_failure_paths_preserve_python_order() {
    for (amount, held, expected) in [
        (f64::INFINITY, f64::INFINITY, f64::INFINITY),
        (23.0, f64::INFINITY, 20.0),
        (f64::NAN, 23.0, 20.0),
        (23.0, f64::NAN, f64::NAN),
    ] {
        let position = TestPosition::values(true, held, f64::INFINITY);
        let calc = calculator(
            config(),
            TestMarket::values(Some(10.0), Some(1000.0), Some(1.0)),
            unrestricted(),
        );
        let mut candidate = order(OrderDir::Sell, amount);
        let _ = calc
            .calculate(&mut candidate, Some(&position), &HashMap::new())
            .unwrap();
        assert_eq!(candidate.deal_amount().to_bits(), expected.to_bits());
    }

    let failure = ExecutionPositionError {
        message: "position offline".to_owned(),
    };
    for stage in ["check", "amount", "cash"] {
        let position = TestPosition {
            check: if stage == "check" {
                Err(failure.clone())
            } else {
                Ok(true)
            },
            amount: if stage == "amount" {
                Err(failure.clone())
            } else {
                Ok(18.0)
            },
            cash: if stage == "cash" {
                Err(failure.clone())
            } else {
                Ok(0.0)
            },
            calls: Arc::new(Mutex::new(Vec::new())),
        };
        let calc = calculator(
            config(),
            TestMarket::values(Some(10.0), Some(1000.0), Some(1.0)),
            unrestricted(),
        );
        let mut candidate = order(OrderDir::Sell, 23.0);
        assert_eq!(
            calc.calculate(&mut candidate, Some(&position), &HashMap::new()),
            Err(ExchangeTradeError::Position(failure.clone()))
        );
        assert_eq!(candidate.factor(), Some(1.0));
    }

    let position = TestPosition::values(true, 18.0, 0.0);
    let calc = calculator(
        config(),
        TestMarket::values(Some(10.0), Some(1000.0), Some(0.0)),
        unrestricted(),
    );
    let mut candidate = order(OrderDir::Sell, 23.0);
    assert_eq!(
        calc.calculate(&mut candidate, Some(&position), &HashMap::new()),
        Err(ExchangeTradeError::Sizing(ExchangeSizingError::ZeroFactor))
    );

    let position = TestPosition::values(true, 23.0, 0.0);
    let calc = calculator(
        config(),
        TestMarket::values(Some(10.0), Some(1000.0), Some(0.0)),
        unrestricted(),
    );
    let mut candidate = order(OrderDir::Sell, 23.0);
    let info = calc
        .calculate(&mut candidate, Some(&position), &HashMap::new())
        .unwrap();
    assert_info_bits(info, (10.0, 230.0, 5.0));
}

#[test]
fn impact_cost_final_fee_and_trade_value_threshold_match_python_floats() {
    for (price, volume, expected_value, expected_cost) in [
        (10.0, 0.0, 200.0, 22.0),
        (10.0, f64::NAN, 200.0, 22.0),
        (10.0, f64::INFINITY, 200.0, 5.0),
        (0.0, 1000.0, 0.0, 0.0),
        (f64::NAN, 1000.0, f64::NAN, f64::NAN),
    ] {
        let calc = calculator(
            config(),
            TestMarket::values(Some(price), Some(volume), Some(1.0)),
            unrestricted(),
        );
        let mut candidate = order(OrderDir::Buy, 23.0);
        let info = calc
            .calculate(&mut candidate, None, &HashMap::new())
            .unwrap();
        assert_eq!(info.trade_value.to_bits(), expected_value.to_bits());
        assert_eq!(info.trade_cost.to_bits(), expected_cost.to_bits());
    }

    let threshold = 1e-5_f64;
    for (price, expected_cost) in [
        (threshold, 0.0),
        (f64::from_bits(threshold.to_bits() + 1), 5.0),
        (-1.0, 0.0),
    ] as [(f64, f64); 3]
    {
        let threshold_config = ExchangeTradeConfig {
            close_cost: 0.1,
            min_cost: 5.0,
            impact_cost: 0.0,
            trade_unit: None,
            ..config()
        };
        let calc = calculator(
            threshold_config,
            TestMarket::values(Some(price), Some(1.0), None),
            unrestricted(),
        );
        let mut candidate = order(OrderDir::Sell, 1.0);
        let info = calc
            .calculate(&mut candidate, None, &HashMap::new())
            .unwrap();
        assert_eq!(info.trade_cost.to_bits(), expected_cost.to_bits());
    }

    let nan_minimum = ExchangeTradeConfig {
        min_cost: f64::NAN,
        impact_cost: 0.0,
        trade_unit: None,
        ..config()
    };
    let calc = calculator(
        nan_minimum,
        TestMarket::values(Some(10.0), Some(1000.0), None),
        unrestricted(),
    );
    let mut candidate = order(OrderDir::Sell, 23.0);
    let info = calc
        .calculate(&mut candidate, None, &HashMap::new())
        .unwrap();
    assert_eq!(info.trade_cost.to_bits(), (230.0_f64 * 0.02).to_bits());
}

#[derive(Clone)]
enum QuoteResult {
    Scalar(f64),
    Missing,
    Series,
    Failure,
}

type QuoteCall = (String, TimeRange, String, QuoteMethod);

struct QueueQuote {
    results: Mutex<VecDeque<QuoteResult>>,
    calls: Arc<Mutex<Vec<QuoteCall>>>,
}

impl Quote for QueueQuote {
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
        match self.results.lock().unwrap().pop_front().unwrap() {
            QuoteResult::Scalar(value) => {
                Ok(Some(QuoteData::Scalar(Arc::new(Float64Array::from(vec![
                    value,
                ])))))
            }
            QuoteResult::Missing => Ok(None),
            QuoteResult::Series => Ok(Some(QuoteData::Series(RecordBatch::new_empty(Arc::new(
                Schema::empty(),
            ))))),
            QuoteResult::Failure => Err(QuoteError::MissingStock {
                stock: stock.to_owned(),
            }),
        }
    }
}

fn quote_provider(
    results: Vec<QuoteResult>,
) -> (ExchangeQuoteProvider, Arc<Mutex<Vec<QuoteCall>>>) {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let provider = ExchangeQuoteProvider::new(
        Arc::new(QueueQuote {
            results: Mutex::new(results.into()),
            calls: Arc::clone(&calls),
        }),
        DealPriceFields::directional("ask", "bid"),
    );
    (provider, calls)
}

#[test]
fn exchange_quote_adapter_supplies_all_execution_market_inputs_and_failures() {
    let (provider, calls) = quote_provider(vec![
        QuoteResult::Scalar(10.0),
        QuoteResult::Scalar(1000.0),
        QuoteResult::Scalar(2.0),
    ]);
    assert_eq!(
        ExecutionMarketProvider::deal_price(&provider, "A", TimeRange::default(), OrderDir::Buy),
        Ok(Some(10.0))
    );
    assert_eq!(
        ExecutionMarketProvider::market_volume(&provider, "A", TimeRange::default()),
        Ok(Some(1000.0))
    );
    assert_eq!(
        ExecutionMarketProvider::factor(&provider, "A", bounded_range()),
        Ok(Some(2.0))
    );
    assert_eq!(
        calls.lock().unwrap().as_slice(),
        [
            (
                "A".to_owned(),
                TimeRange::default(),
                "ask".to_owned(),
                QuoteMethod::LastValid
            ),
            (
                "A".to_owned(),
                TimeRange::default(),
                "$volume".to_owned(),
                QuoteMethod::BuiltIn(BuiltInAggregation::Sum)
            ),
            (
                "A".to_owned(),
                bounded_range(),
                "$factor".to_owned(),
                QuoteMethod::LastValid
            )
        ]
    );

    for (initial, fallback, expected) in [
        (
            QuoteResult::Missing,
            QuoteResult::Scalar(9.0),
            Ok(Some(9.0)),
        ),
        (
            QuoteResult::Scalar(f64::NAN),
            QuoteResult::Scalar(8.0),
            Ok(Some(8.0)),
        ),
        (
            QuoteResult::Scalar(1e-8),
            QuoteResult::Scalar(7.0),
            Ok(Some(7.0)),
        ),
        (
            QuoteResult::Scalar(f64::from_bits(1e-8_f64.to_bits() + 1)),
            QuoteResult::Scalar(99.0),
            Ok(Some(f64::from_bits(1e-8_f64.to_bits() + 1))),
        ),
    ] {
        let (provider, calls) = quote_provider(vec![initial, fallback]);
        assert_eq!(
            ExecutionMarketProvider::deal_price(
                &provider,
                "A",
                TimeRange::default(),
                OrderDir::Sell
            ),
            expected
        );
        let expected_calls = if expected == Ok(Some(f64::from_bits(1e-8_f64.to_bits() + 1))) {
            1
        } else {
            2
        };
        assert_eq!(calls.lock().unwrap().len(), expected_calls);
    }
}

#[test]
fn exchange_quote_adapter_propagates_shape_provider_and_factor_edge_cases() {
    for result in [QuoteResult::Series, QuoteResult::Failure] {
        let (provider, _) = quote_provider(vec![result]);
        assert!(
            ExecutionMarketProvider::deal_price(
                &provider,
                "A",
                TimeRange::default(),
                OrderDir::Buy
            )
            .is_err()
        );
    }
    let (provider, _) = quote_provider(vec![QuoteResult::Missing, QuoteResult::Failure]);
    assert!(
        ExecutionMarketProvider::deal_price(&provider, "A", TimeRange::default(), OrderDir::Buy)
            .is_err()
    );

    for method in ["volume", "factor"] {
        for result in [
            QuoteResult::Missing,
            QuoteResult::Series,
            QuoteResult::Failure,
        ] {
            let (provider, _) = quote_provider(vec![result]);
            let value = if method == "volume" {
                ExecutionMarketProvider::market_volume(&provider, "A", TimeRange::default())
            } else {
                ExecutionMarketProvider::factor(&provider, "A", bounded_range())
            };
            if matches!(value, Ok(None)) {
                assert_eq!(value, Ok(None));
            } else {
                assert!(value.is_err());
            }
        }
    }

    let (provider, calls) = quote_provider(vec![]);
    assert!(ExecutionMarketProvider::factor(&provider, "A", TimeRange::default()).is_err());
    assert!(
        ExecutionMarketProvider::factor(
            &provider,
            "A",
            TimeRange {
                start: Some(timestamp("2024-01-02 09:30:00")),
                end: None,
            },
        )
        .is_err()
    );
    assert_eq!(calls.lock().unwrap().len(), 0);
    assert_eq!(
        ExecutionMarketProvider::factor(&provider, "B", bounded_range()),
        Ok(None)
    );
    assert_eq!(calls.lock().unwrap().len(), 0);
}

#[test]
fn trade_info_calculation_matches_live_python_exchange_method() {
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/backtest/exchange.py");
    let script = r"
import ast,json,math,struct,sys
import numpy as np
from typing import Optional,Tuple,cast
class Order:
 SELL=0;BUY=1
 def __init__(self,d=1,a=23.0):self.stock_id='A';self.direction=d;self.amount=a;self.start_time='s';self.end_time='e';self.deal_amount=7.0;self.factor=8.0
 def __repr__(self):return 'ORDER'
p=sys.argv[1];t=ast.parse(open(p,encoding='utf-8').read(),filename=p);c=next(x for x in t.body if isinstance(x,ast.ClassDef) and x.name=='Exchange');names=['_get_factor_or_raise_error','round_amount_by_trade_unit','_get_buy_amount_by_cash_limit','_calc_trade_info_by_order'];ns={};exec(compile(ast.Module(body=[next(x for x in c.body if isinstance(x,ast.FunctionDef) and x.name==n) for n in names],type_ignores=[]),p,'exec'),globals(),ns)
class P:
 def __init__(self,has=True,stock=23.0,cash=1000.0):self.has=has;self.stock=stock;self.cashv=cash
 def check_stock(self,s):return self.has
 def get_stock_amount(self,s):return self.stock
 def get_cash(self):return self.cashv
class L:
 def debug(self,*a):pass
class X:
 _get_factor_or_raise_error=ns['_get_factor_or_raise_error'];round_amount_by_trade_unit=ns['round_amount_by_trade_unit'];_get_buy_amount_by_cash_limit=ns['_get_buy_amount_by_cash_limit'];_calc_trade_info_by_order=ns['_calc_trade_info_by_order']
 def __init__(self,price=10.,volume=1000.,factor=1.,clip=None):self.price=price;self.volume=volume;self.factorv=factor;self.clip=clip;self.trade_unit=10.;self.trade_w_adj_price=False;self.open_cost=.01;self.close_cost=.02;self.min_cost=5.;self.impact_cost=.1;self.logger=L()
 def get_deal_price(self,*a,**k):return self.price
 def get_volume(self,*a,**k):return self.volume
 def get_factor(self,*a,**k):return self.factorv
 def _clip_amount_by_volume(self,o,d):o.deal_amount=o.deal_amount if self.clip is None else min(self.clip,o.deal_amount)
def bits(x):return struct.unpack('>Q',struct.pack('>d',float(x)))[0]
def run(d=1,a=23.,pos=None,**kw):
 x=X(**kw);o=Order(d,a)
 try:r=['ok',[bits(v) for v in x._calc_trade_info_by_order(o,pos,{})]]
 except BaseException as e:r=[type(e).__name__,str(e)]
 return [r,bits(o.deal_amount),None if o.factor is None else bits(o.factor)]
r={'buy':run(),'buy_cost':run(pos=P(cash=1)),'buy_partial':run(pos=P(cash=150)),'sell':run(d=0),'sell_round':run(d=0,pos=P(stock=18,cash=0)),'sell_missing':run(d=0,pos=P(has=False,cash=0)),'clip':run(clip=12),'zero_volume':run(volume=0),'nan_volume':run(volume=float('nan')),'inf_volume':run(volume=float('inf')),'zero_price':run(price=0),'nan_price':run(price=float('nan')),'factor_none_buy':run(factor=None),'factor_none_sell':run(d=0,factor=None)};print(json.dumps(r,separators=(',',':')))
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
        "buy": [["ok", [10.0_f64.to_bits(), 200.0_f64.to_bits(), 5.0_f64.to_bits()]], 20.0_f64.to_bits(), 1.0_f64.to_bits()],
        "buy_cost": [["ok", [10.0_f64.to_bits(), 0.0_f64.to_bits(), 0.0_f64.to_bits()]], 0.0_f64.to_bits(), 1.0_f64.to_bits()],
        "buy_partial": [["ok", [10.0_f64.to_bits(), 100.0_f64.to_bits(), 5.0_f64.to_bits()]], 10.0_f64.to_bits(), 1.0_f64.to_bits()],
        "sell": [["ok", [10.0_f64.to_bits(), 230.0_f64.to_bits(), 5.0_f64.to_bits()]], 23.0_f64.to_bits(), 1.0_f64.to_bits()],
        "sell_round": [["ok", [10.0_f64.to_bits(), 100.0_f64.to_bits(), 5.0_f64.to_bits()]], 10.0_f64.to_bits(), 1.0_f64.to_bits()],
        "sell_missing": [["ok", [10.0_f64.to_bits(), 0.0_f64.to_bits(), 0.0_f64.to_bits()]], 0.0_f64.to_bits(), 1.0_f64.to_bits()],
        "clip": [["ok", [10.0_f64.to_bits(), 100.0_f64.to_bits(), 5.0_f64.to_bits()]], 10.0_f64.to_bits(), 1.0_f64.to_bits()],
        "zero_volume": [["ok", [10.0_f64.to_bits(), 200.0_f64.to_bits(), 22.0_f64.to_bits()]], 20.0_f64.to_bits(), 1.0_f64.to_bits()],
        "nan_volume": [["ok", [10.0_f64.to_bits(), 200.0_f64.to_bits(), 22.0_f64.to_bits()]], 20.0_f64.to_bits(), 1.0_f64.to_bits()],
        "inf_volume": [["ok", [10.0_f64.to_bits(), 200.0_f64.to_bits(), 5.0_f64.to_bits()]], 20.0_f64.to_bits(), 1.0_f64.to_bits()],
        "zero_price": [["ok", [0.0_f64.to_bits(), 0.0_f64.to_bits(), 0.0_f64.to_bits()]], 20.0_f64.to_bits(), 1.0_f64.to_bits()],
        "nan_price": [["ok", [f64::NAN.to_bits(), f64::NAN.to_bits(), f64::NAN.to_bits()]], 20.0_f64.to_bits(), 1.0_f64.to_bits()],
        "factor_none_buy": [["ValueError", "`factor` and (`stock_id`, `start_time`, `end_time`) can't both be None"], 23.0_f64.to_bits(), null],
        "factor_none_sell": [["ok", [10.0_f64.to_bits(), 230.0_f64.to_bits(), 5.0_f64.to_bits()]], 23.0_f64.to_bits(), null]
    });
    assert_eq!(actual, expected);
}
