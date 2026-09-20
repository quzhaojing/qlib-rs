use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    panic::{AssertUnwindSafe, catch_unwind},
    path::PathBuf,
    process::Command,
    sync::{Arc, Mutex, RwLock},
};

use chrono::{NaiveDate, NaiveDateTime};
use domain_core::{
    ExchangeDealError, ExchangeDealExecutor, ExchangeTradeCalculator, ExchangeTradeConfig,
    ExchangeVolumeLimiter, ExecutionMarketProvider, ExecutionMarketProviderError,
    ExecutionPosition, ExecutionPositionError, ExecutionTarget, ExecutionTargetError, Order,
    OrderDealResult, OrderDir, OrderExecution, OrderTradabilityProvider,
    OrderTradabilityProviderError, OrderTradeDecision, SimulatorCalendar, SimulatorCalendarError,
    SimulatorCollectionError, SimulatorCollector, SimulatorDealProvider, SimulatorExecutionLog,
    SimulatorExecutionReporter, SimulatorExecutorError, SimulatorReporterError, TimeRange,
    format_simulator_execution,
};
use serde_json::{Value, json};

use domain_core::decision_construction::{
    DecisionConstructionStrategy, DecisionOrderItem, SharedDecisionOrders,
    SharedOrderDecisionConstruction,
};
use domain_core::shared_simulator::{SharedSimulatorError, shared_simulator_order_iterator};

struct ConstructionClock;

impl DecisionConstructionStrategy for ConstructionClock {
    type Error = std::convert::Infallible;
    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), Self::Error> {
        let time = timestamp("2024-01-02 09:30:00");
        Ok((time, time))
    }
}

struct ClearingDealer {
    inner: TestDealer,
    original: SharedDecisionOrders,
}

impl SimulatorDealProvider for ClearingDealer {
    fn deal_order(
        &self,
        order: &mut Order,
        account: &mut dyn ExecutionTarget,
        fills: &HashMap<String, f64>,
    ) -> Result<OrderDealResult, ExchangeDealError> {
        self.original.write().unwrap().clear();
        self.inner.deal_order(order, account, fills)
    }
}

fn shared_collection_snapshot(mode: &str, fail: usize) -> Value {
    let buy = Arc::new(RwLock::new(Order::new("B", 1.0, OrderDir::Buy, None, None)));
    let sell = Arc::new(RwLock::new(Order::new(
        "S",
        1.0,
        OrderDir::Sell,
        None,
        None,
    )));
    let original = Arc::new(RwLock::new(vec![
        DecisionOrderItem::Order(Arc::clone(&sell)),
        DecisionOrderItem::Order(Arc::clone(&buy)),
        DecisionOrderItem::Order(Arc::clone(&buy)),
    ]));
    let mut decision = SharedOrderDecisionConstruction::new(ConstructionClock);
    decision.initialize(&original, None, ()).unwrap();
    let time = timestamp("2024-01-02 09:30:00");
    assert_eq!(buy.read().unwrap().start_time(), Some(time));
    let mut harness = make_harness(mode, vec![Ok(time); 4], None, false, None);
    let calendar = QueueCalendar {
        values: Mutex::new(vec![Ok((time, time)); 4].into()),
        events: Arc::clone(&harness.events),
    };
    let dealer = ClearingDealer {
        original: Arc::clone(&original),
        inner: TestDealer {
            call: Mutex::new(0),
            fail_at: fail.checked_sub(1),
            events: Arc::clone(&harness.events),
        },
    };
    harness.collector = SimulatorCollector::new(
        mode,
        Arc::new(calendar),
        Arc::new(dealer),
        Arc::new(TestReporter {
            error: None,
            logs: Arc::clone(&harness.logs),
            events: Arc::clone(&harness.events),
        }),
        false,
    );
    let result = harness.collector.collect_shared_data(
        decision.orders.as_ref().unwrap(),
        &mut harness.account,
        0,
    );
    let (rows, same_list, error) = match result {
        Ok(collection) => {
            let rows: Vec<Value> = collection
                .execution_result()
                .iter()
                .map(|execution| {
                    let order = execution.order.read().unwrap();
                    assert!(Arc::ptr_eq(
                        &execution.order,
                        if order.stock_id() == "B" { &buy } else { &sell }
                    ));
                    json!([
                        order.stock_id(),
                        order.deal_amount(),
                        execution.trade_value,
                        execution.trade_cost,
                        execution.trade_price
                    ])
                })
                .collect();
            (
                Some(rows),
                Some(collection.execution_result().as_ptr() == collection.trade_info().as_ptr()),
                None,
            )
        }
        Err(SharedSimulatorError::Collection(SimulatorCollectionError::Deal(
            ExchangeDealError::Tradability(error),
        ))) => (None, None, Some(error.message)),
        Err(error) => panic!("unexpected {error}"),
    };
    let events: Vec<Value> = harness
        .events
        .lock()
        .unwrap()
        .iter()
        .map(|event| match event {
            Event::Calendar => json!("calendar"),
            Event::Deal(stock, fills) => json!([
                stock,
                fills
                    .iter()
                    .map(|(stock, bits)| (stock, f64::from_bits(*bits)))
                    .collect::<BTreeMap<_, _>>()
            ]),
            _ => panic!("unexpected event"),
        })
        .collect();
    json!({"rows": rows, "same_list": same_list, "error": error,
        "buy_amount": buy.read().unwrap().deal_amount(), "sell_amount": sell.read().unwrap().deal_amount(),
        "original_length": original.read().unwrap().len(), "fills": harness.collector.dealt_order_amount(), "events": events})
}

#[test]
fn shared_constructor_to_collector_preserves_live_source_aliases_and_failed_deals() {
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/shared_simulator_contract.py"
        ))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let source: Value = serde_json::from_slice(&output.stdout).unwrap();
    for mode in ["serial", "parallel"] {
        for fail in [0, 2] {
            assert_eq!(
                shared_collection_snapshot(mode, fail),
                source[format!("{mode}-{fail}")]
            );
        }
    }
}

#[test]
fn shared_execution_tuples_survive_list_replacement_without_copying_orders() {
    let time = timestamp("2024-01-02 09:30:00");
    let order = Arc::new(RwLock::new(Order::new("B", 1.0, OrderDir::Buy, None, None)));
    let orders = Arc::new(RwLock::new(vec![DecisionOrderItem::Order(Arc::clone(
        &order,
    ))]));
    let mut harness = make_harness("serial", vec![Ok(time); 4], None, false, None);
    let mut first = harness
        .collector
        .collect_shared_data(&orders, &mut harness.account, 0)
        .unwrap();
    let tuple = Arc::clone(&first.execution_result()[0]);
    let numbers = (tuple.trade_value, tuple.trade_cost, tuple.trade_price);
    first.execution_result_mut().push(Arc::clone(&tuple));
    let mut flattened = first.execution_result().to_vec();
    first.execution_result_mut().clear();
    let second = harness
        .collector
        .collect_shared_data(&orders, &mut harness.account, 0)
        .unwrap();
    first
        .execution_result_mut()
        .push(Arc::clone(&second.execution_result()[0]));
    flattened.extend(second.execution_result().iter().cloned());
    order.write().unwrap().set_deal_amount(8.0);
    assert_eq!(first.execution_result().len(), 1);
    assert_eq!(flattened.len(), 3);
    assert!(Arc::ptr_eq(&flattened[0], &flattened[1]));
    assert!(!Arc::ptr_eq(&flattened[0], &flattened[2]));
    assert!(Arc::ptr_eq(&flattened[2], &first.execution_result()[0]));
    assert_eq!(
        (
            flattened[0].trade_value,
            flattened[0].trade_cost,
            flattened[0].trade_price
        ),
        numbers
    );
    for row in flattened {
        assert!(Arc::ptr_eq(&row.order, &order));
        assert_eq!(
            row.order.read().unwrap().deal_amount().to_bits(),
            8.0_f64.to_bits()
        );
    }
}

#[test]
fn shared_extraction_checks_items_before_mode_and_never_reorders_original_list() {
    let sell = Arc::new(RwLock::new(Order::new(
        "S",
        1.0,
        OrderDir::Sell,
        None,
        None,
    )));
    let buy = Arc::new(RwLock::new(Order::new("B", 1.0, OrderDir::Buy, None, None)));
    let orders = Arc::new(RwLock::new(vec![
        DecisionOrderItem::Order(Arc::clone(&sell)),
        DecisionOrderItem::Order(Arc::clone(&buy)),
    ]));
    let sorted = shared_simulator_order_iterator(&orders, "parallel").unwrap();
    assert!(Arc::ptr_eq(&sorted[0], &buy));
    assert!(Arc::ptr_eq(&sorted[1], &sell));
    let original = shared_simulator_order_iterator(&orders, "serial").unwrap();
    assert!(Arc::ptr_eq(&original[0], &sell));
    assert!(matches!(
        shared_simulator_order_iterator(&orders, "bad"),
        Err(SharedSimulatorError::Iterator(_))
    ));
    orders
        .write()
        .unwrap()
        .push(DecisionOrderItem::Other(Arc::new(())));
    assert!(matches!(
        shared_simulator_order_iterator(&orders, "bad"),
        Err(SharedSimulatorError::InvalidOrder(2))
    ));
    orders.write().unwrap().clear();
    assert!(
        shared_simulator_order_iterator(&orders, "parallel")
            .unwrap()
            .is_empty()
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _guard = sell.write().unwrap();
            panic!("poison");
        }))
        .is_err()
    );
    orders.write().unwrap().push(DecisionOrderItem::Order(sell));
    assert!(matches!(
        shared_simulator_order_iterator(&orders, "parallel"),
        Err(SharedSimulatorError::OrderPoisoned(0))
    ));
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _guard = orders.write().unwrap();
            panic!("poison");
        }))
        .is_err()
    );
    assert!(matches!(
        shared_simulator_order_iterator(&orders, "serial"),
        Err(SharedSimulatorError::ListPoisoned)
    ));
}

#[test]
fn shared_collection_calendar_and_lock_failures_keep_stage_order() {
    let time = timestamp("2024-01-02 09:30:00");
    let orders: SharedDecisionOrders = Arc::new(RwLock::new(vec![]));
    let mut empty = make_harness("serial", vec![Ok(time)], None, false, None);
    assert!(
        empty
            .collector
            .collect_shared_data(&orders, &mut empty.account, 0)
            .unwrap()
            .trade_info()
            .is_empty()
    );
    let failure = || {
        Err(SimulatorCalendarError {
            message: "calendar".to_owned(),
        })
    };
    let mut initial = make_harness("bad", vec![failure()], None, false, None);
    assert!(matches!(
        initial
            .collector
            .collect_shared_data(&orders, &mut initial.account, 0),
        Err(SharedSimulatorError::Collection(
            SimulatorCollectionError::Calendar(_)
        ))
    ));
    let mut invalid = make_harness("bad", vec![Ok(time)], None, false, None);
    assert!(matches!(
        invalid
            .collector
            .collect_shared_data(&orders, &mut invalid.account, 0),
        Err(SharedSimulatorError::Iterator(_))
    ));
    let order = Arc::new(RwLock::new(Order::new("A", 1.0, OrderDir::Buy, None, None)));
    orders
        .write()
        .unwrap()
        .push(DecisionOrderItem::Order(Arc::clone(&order)));
    let mut per_order = make_harness("serial", vec![Ok(time), failure()], None, false, None);
    assert!(matches!(
        per_order
            .collector
            .collect_shared_data(&orders, &mut per_order.account, 0),
        Err(SharedSimulatorError::Collection(
            SimulatorCollectionError::Calendar(_)
        ))
    ));
    assert!(per_order.collector.deal_day().is_none());
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _guard = order.write().unwrap();
            panic!("poison");
        }))
        .is_err()
    );
    let mut locked = make_harness("serial", vec![Ok(time), Ok(time)], None, false, None);
    assert!(matches!(
        locked
            .collector
            .collect_shared_data(&orders, &mut locked.account, 0),
        Err(SharedSimulatorError::OrderPoisoned(0))
    ));
    assert_eq!(locked.collector.deal_day(), Some(time.date()));
    assert_eq!(
        locked.events.lock().unwrap().as_slice(),
        [Event::Calendar, Event::Calendar]
    );
}

fn timestamp(text: &str) -> NaiveDateTime {
    NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S").unwrap()
}

fn day(text: &str) -> NaiveDate {
    NaiveDate::parse_from_str(text, "%Y-%m-%d").unwrap()
}

fn make_decision(spec: &[(&str, OrderDir)]) -> OrderTradeDecision {
    OrderTradeDecision::from_orders(
        spec.iter()
            .map(|(stock, direction)| {
                Order::new(
                    *stock,
                    1.0,
                    *direction,
                    Some(timestamp("2024-01-02 09:30:00")),
                    Some(timestamp("2024-01-02 10:00:00")),
                )
            })
            .collect(),
        timestamp("2024-01-02 09:30:00"),
        timestamp("2024-01-02 10:00:00"),
        None,
    )
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Event {
    Calendar,
    Deal(String, BTreeMap<String, u64>),
    Position,
    Cash,
    Report,
}

struct QueueCalendar {
    values: Mutex<VecDeque<Result<(NaiveDateTime, NaiveDateTime), SimulatorCalendarError>>>,
    events: Arc<Mutex<Vec<Event>>>,
}

impl SimulatorCalendar for QueueCalendar {
    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), SimulatorCalendarError> {
        self.events.lock().unwrap().push(Event::Calendar);
        self.values.lock().unwrap().pop_front().unwrap()
    }
}

struct TestDealer {
    call: Mutex<usize>,
    fail_at: Option<usize>,
    events: Arc<Mutex<Vec<Event>>>,
}

impl SimulatorDealProvider for TestDealer {
    fn deal_order(
        &self,
        order: &mut Order,
        _account: &mut dyn ExecutionTarget,
        dealt_order_amount: &HashMap<String, f64>,
    ) -> Result<OrderDealResult, ExchangeDealError> {
        let mut call = self.call.lock().unwrap();
        let snapshot = dealt_order_amount
            .iter()
            .map(|(stock, amount)| (stock.clone(), amount.to_bits()))
            .collect();
        self.events
            .lock()
            .unwrap()
            .push(Event::Deal(order.stock_id().to_owned(), snapshot));
        let deal_amount = f64::from(u32::try_from(*call + 1).unwrap());
        order.set_deal_amount(deal_amount);
        order.set_factor(Some(2.0));
        if self.fail_at == Some(*call) {
            return Err(ExchangeDealError::Tradability(
                OrderTradabilityProviderError {
                    message: "deal failed".to_owned(),
                },
            ));
        }
        *call += 1;
        Ok(OrderDealResult {
            trade_value: deal_amount * 10.0,
            trade_cost: 1.0,
            trade_price: 10.0,
        })
    }
}

struct AlwaysTradable;

impl OrderTradabilityProvider for AlwaysTradable {
    fn is_tradable(&self, _order: &Order) -> Result<bool, OrderTradabilityProviderError> {
        Ok(true)
    }
}

struct FixedMarket;

impl ExecutionMarketProvider for FixedMarket {
    fn deal_price(
        &self,
        _stock: &str,
        _range: TimeRange,
        _direction: OrderDir,
    ) -> Result<Option<f64>, ExecutionMarketProviderError> {
        Ok(Some(10.0))
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

struct TestAccount {
    cash: Result<f64, ExecutionPositionError>,
    position_error: Option<ExecutionTargetError>,
    events: Arc<Mutex<Vec<Event>>>,
}

impl ExecutionPosition for TestAccount {
    fn check_stock(&self, _stock: &str) -> Result<bool, ExecutionPositionError> {
        Ok(true)
    }

    fn stock_amount(&self, _stock: &str) -> Result<f64, ExecutionPositionError> {
        Ok(100.0)
    }

    fn cash(&self) -> Result<f64, ExecutionPositionError> {
        self.events.lock().unwrap().push(Event::Cash);
        self.cash.clone()
    }
}

impl ExecutionTarget for TestAccount {
    fn position(&self) -> Result<&dyn ExecutionPosition, ExecutionTargetError> {
        self.events.lock().unwrap().push(Event::Position);
        self.position_error
            .as_ref()
            .map_or(Ok(self as &dyn ExecutionPosition), |error| {
                Err(error.clone())
            })
    }

    fn update_order(
        &mut self,
        _order: &Order,
        _trade_value: f64,
        _trade_cost: f64,
        _trade_price: f64,
    ) -> Result<(), ExecutionTargetError> {
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
struct LogSnapshot {
    start: NaiveDateTime,
    stock: String,
    value: f64,
    cash: f64,
}

struct TestReporter {
    error: Option<SimulatorReporterError>,
    logs: Arc<Mutex<Vec<LogSnapshot>>>,
    events: Arc<Mutex<Vec<Event>>>,
}

impl SimulatorExecutionReporter for TestReporter {
    fn report(&self, log: SimulatorExecutionLog<'_>) -> Result<(), SimulatorReporterError> {
        self.events.lock().unwrap().push(Event::Report);
        self.logs.lock().unwrap().push(LogSnapshot {
            start: log.trade_start_time,
            stock: log.execution.order.stock_id().to_owned(),
            value: log.execution.trade_value,
            cash: log.cash,
        });
        self.error.clone().map_or(Ok(()), Err)
    }
}

struct Harness {
    collector: SimulatorCollector,
    account: TestAccount,
    events: Arc<Mutex<Vec<Event>>>,
    logs: Arc<Mutex<Vec<LogSnapshot>>>,
}

fn make_harness(
    trade_type: &str,
    times: Vec<Result<NaiveDateTime, SimulatorCalendarError>>,
    fail_deal_at: Option<usize>,
    verbose: bool,
    reporter_error: Option<SimulatorReporterError>,
) -> Harness {
    let events = Arc::new(Mutex::new(Vec::new()));
    let logs = Arc::new(Mutex::new(Vec::new()));
    let calendar = QueueCalendar {
        values: Mutex::new(
            times
                .into_iter()
                .map(|value| value.map(|time| (time, time)))
                .collect(),
        ),
        events: Arc::clone(&events),
    };
    let dealer = TestDealer {
        call: Mutex::new(0),
        fail_at: fail_deal_at,
        events: Arc::clone(&events),
    };
    let reporter = TestReporter {
        error: reporter_error,
        logs: Arc::clone(&logs),
        events: Arc::clone(&events),
    };
    Harness {
        collector: SimulatorCollector::new(
            trade_type,
            Arc::new(calendar),
            Arc::new(dealer),
            Arc::new(reporter),
            verbose,
        ),
        account: TestAccount {
            cash: Ok(99.0),
            position_error: None,
            events: Arc::clone(&events),
        },
        events,
        logs,
    }
}

#[test]
fn empty_batch_reads_initial_calendar_and_exposes_one_shared_result_view() {
    let mut harness = make_harness(
        "serial",
        vec![Ok(timestamp("2024-01-02 09:30:00"))],
        None,
        false,
        None,
    );
    let mut decision = make_decision(&[]);
    let collection = harness
        .collector
        .collect_data(&mut decision, &mut harness.account, 7)
        .unwrap();
    assert!(collection.execution_result().is_empty());
    assert_eq!(
        collection.execution_result().as_ptr(),
        collection.trade_info().as_ptr()
    );
    assert_eq!(harness.collector.deal_day(), None);
    assert!(harness.collector.dealt_order_amount().is_empty());
    assert_eq!(harness.events.lock().unwrap().as_slice(), [Event::Calendar]);
}

#[test]
fn exchange_deal_adapter_and_buy_formatter_compose_completed_slices() {
    let calculator = ExchangeTradeCalculator::new(
        ExchangeTradeConfig {
            open_cost: 0.0,
            close_cost: 0.0,
            min_cost: 0.0,
            impact_cost: 0.0,
            trade_with_adjusted_price: true,
            trade_unit: None,
        },
        Arc::new(FixedMarket),
        ExchangeVolumeLimiter::new(None, None, None),
    );
    let executor = ExchangeDealExecutor::new(Arc::new(AlwaysTradable), calculator);
    let dealer: &dyn SimulatorDealProvider = &executor;
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut account = TestAccount {
        cash: Ok(99.0),
        position_error: None,
        events,
    };
    let mut candidate = Order::new(
        "A",
        1.0,
        OrderDir::Buy,
        Some(timestamp("2024-01-02 09:30:00")),
        Some(timestamp("2024-01-02 10:00:00")),
    );
    let result = dealer
        .deal_order(&mut candidate, &mut account, &HashMap::new())
        .unwrap();
    let execution = OrderExecution {
        order: &candidate,
        trade_value: result.trade_value,
        trade_cost: result.trade_cost,
        trade_price: result.trade_price,
    };
    assert_eq!(
        format_simulator_execution(SimulatorExecutionLog {
            trade_start_time: timestamp("2024-01-02 09:30:00"),
            execution,
            cash: 99.0,
        }),
        "[I 2024-01-02 09:30:00]: buy A, price 10.00, amount 1.0, deal_amount 1.0, factor None, value 10.00, cash 99.00."
    );
}

#[test]
fn intraday_state_retains_same_or_older_days_and_replaces_on_later_days() {
    for (per_order_times, initial_day, initial, expected_day, expected) in [
        (
            vec!["2024-01-02 10:00:00", "2024-01-02 10:30:00"],
            Some(day("2024-01-02")),
            HashMap::from([("A".to_owned(), 5.0)]),
            day("2024-01-02"),
            HashMap::from([("A".to_owned(), 8.0)]),
        ),
        (
            vec!["2024-01-01 10:00:00", "2024-01-01 10:30:00"],
            Some(day("2024-01-02")),
            HashMap::from([("A".to_owned(), 5.0)]),
            day("2024-01-02"),
            HashMap::from([("A".to_owned(), 8.0)]),
        ),
        (
            vec!["2024-01-02 10:00:00", "2024-01-03 09:30:00"],
            Some(day("2024-01-02")),
            HashMap::from([("Z".to_owned(), 4.0)]),
            day("2024-01-03"),
            HashMap::from([("A".to_owned(), 2.0)]),
        ),
    ] {
        let mut times = vec![Ok(timestamp("2024-01-02 09:30:00"))];
        times.extend(
            per_order_times
                .into_iter()
                .map(|value| Ok(timestamp(value))),
        );
        let mut harness = make_harness("serial", times, None, false, None);
        harness
            .collector
            .restore_intraday_state(initial_day, initial);
        let mut decision = make_decision(&[("A", OrderDir::Buy), ("A", OrderDir::Sell)]);
        let collection = harness
            .collector
            .collect_data(&mut decision, &mut harness.account, 0)
            .unwrap();
        assert_eq!(collection.execution_result().len(), 2);
        assert_eq!(harness.collector.deal_day(), Some(expected_day));
        assert_eq!(harness.collector.dealt_order_amount(), &expected);
    }

    let mut harness = make_harness(
        "serial",
        vec![
            Ok(timestamp("2024-01-02 09:30:00")),
            Ok(timestamp("2024-01-02 10:00:00")),
        ],
        None,
        false,
        None,
    );
    harness
        .collector
        .restore_intraday_state(None, HashMap::from([("OLD".to_owned(), 9.0)]));
    let mut decision = make_decision(&[("A", OrderDir::Buy)]);
    harness
        .collector
        .collect_data(&mut decision, &mut harness.account, 0)
        .unwrap();
    assert_eq!(
        harness.collector.dealt_order_amount(),
        &HashMap::from([("A".to_owned(), 1.0)])
    );
}

#[test]
fn parallel_collection_preserves_buy_first_results_order_mutation_and_fill_snapshots() {
    let mut harness = make_harness(
        "parallel",
        vec![
            Ok(timestamp("2024-01-02 09:30:00")),
            Ok(timestamp("2024-01-02 09:30:00")),
            Ok(timestamp("2024-01-02 09:30:00")),
        ],
        None,
        false,
        None,
    );
    let mut decision = make_decision(&[("S", OrderDir::Sell), ("B", OrderDir::Buy)]);
    let collection = harness
        .collector
        .collect_data(&mut decision, &mut harness.account, 0)
        .unwrap();
    let executions = collection.execution_result();
    assert_eq!(executions[0].order.stock_id(), "B");
    assert_eq!(executions[1].order.stock_id(), "S");
    assert_eq!(
        executions[0].order.deal_amount().to_bits(),
        1.0_f64.to_bits()
    );
    assert_eq!(
        executions[1].order.deal_amount().to_bits(),
        2.0_f64.to_bits()
    );
    let events = harness.events.lock().unwrap();
    assert!(matches!(&events[2], Event::Deal(stock, fills) if stock == "B" && fills.is_empty()));
    assert!(
        matches!(&events[4], Event::Deal(stock, fills) if stock == "S" && fills.get("B") == Some(&1.0_f64.to_bits()))
    );
}

#[test]
fn calendar_iterator_and_second_deal_failures_retain_only_reached_state() {
    let calendar_failure = SimulatorCalendarError {
        message: "calendar failed".to_owned(),
    };
    let mut harness = make_harness(
        "serial",
        vec![Err(calendar_failure.clone())],
        None,
        false,
        None,
    );
    let mut decision = make_decision(&[("A", OrderDir::Buy)]);
    assert_eq!(
        harness
            .collector
            .collect_data(&mut decision, &mut harness.account, 0)
            .unwrap_err(),
        SimulatorCollectionError::Calendar(calendar_failure.clone())
    );

    let mut harness = make_harness(
        "bad",
        vec![Ok(timestamp("2024-01-02 09:30:00"))],
        None,
        false,
        None,
    );
    assert_eq!(
        harness
            .collector
            .collect_data(&mut decision, &mut harness.account, 0)
            .unwrap_err(),
        SimulatorCollectionError::Iterator(SimulatorExecutorError::UnsupportedTradeType {
            trade_type: "bad".to_owned(),
        })
    );

    let mut harness = make_harness(
        "serial",
        vec![
            Ok(timestamp("2024-01-02 09:30:00")),
            Ok(timestamp("2024-01-02 09:30:00")),
            Ok(timestamp("2024-01-02 09:30:00")),
        ],
        Some(1),
        false,
        None,
    );
    let mut decision = make_decision(&[("A", OrderDir::Buy), ("B", OrderDir::Buy)]);
    assert!(matches!(
        harness
            .collector
            .collect_data(&mut decision, &mut harness.account, 0),
        Err(SimulatorCollectionError::Deal(_))
    ));
    assert_eq!(
        harness.collector.dealt_order_amount(),
        &HashMap::from([("A".to_owned(), 1.0)])
    );
    assert_eq!(
        decision.orders()[1].deal_amount().to_bits(),
        2.0_f64.to_bits()
    );

    let mut harness = make_harness(
        "serial",
        vec![
            Ok(timestamp("2024-01-02 09:30:00")),
            Err(calendar_failure.clone()),
        ],
        None,
        false,
        None,
    );
    let mut decision = make_decision(&[("A", OrderDir::Buy)]);
    assert_eq!(
        harness
            .collector
            .collect_data(&mut decision, &mut harness.account, 0)
            .unwrap_err(),
        SimulatorCollectionError::Calendar(calendar_failure)
    );
    assert_eq!(
        decision.orders()[0].deal_amount().to_bits(),
        0.0_f64.to_bits()
    );
}

#[test]
fn verbose_reporting_runs_after_fill_accounting_and_propagates_each_late_failure() {
    let reporter_failure = SimulatorReporterError {
        message: "report failed".to_owned(),
    };
    let mut harness = make_harness(
        "serial",
        vec![
            Ok(timestamp("2024-01-02 09:30:00")),
            Ok(timestamp("2024-01-02 09:30:00")),
        ],
        None,
        true,
        None,
    );
    let mut decision = make_decision(&[("A", OrderDir::Sell)]);
    let collection = harness
        .collector
        .collect_data(&mut decision, &mut harness.account, 0)
        .unwrap();
    let execution = collection.execution_result()[0];
    assert_eq!(
        format_simulator_execution(SimulatorExecutionLog {
            trade_start_time: timestamp("2024-01-02 09:30:00"),
            execution,
            cash: 99.0,
        }),
        "[I 2024-01-02 09:30:00]: sell A, price 10.00, amount 1.0, deal_amount 1.0, factor 2.0, value 10.00, cash 99.00."
    );
    assert_eq!(harness.logs.lock().unwrap()[0].stock, "A");
    assert_eq!(
        harness.events.lock().unwrap().as_slice(),
        [
            Event::Calendar,
            Event::Calendar,
            Event::Deal("A".to_owned(), BTreeMap::new()),
            Event::Position,
            Event::Cash,
            Event::Report,
        ]
    );

    for failure in ["position", "cash", "report"] {
        let mut harness = make_harness(
            "serial",
            vec![
                Ok(timestamp("2024-01-02 09:30:00")),
                Ok(timestamp("2024-01-02 09:30:00")),
            ],
            None,
            true,
            (failure == "report").then_some(reporter_failure.clone()),
        );
        if failure == "position" {
            harness.account.position_error = Some(ExecutionTargetError {
                message: "position failed".to_owned(),
            });
        } else if failure == "cash" {
            harness.account.cash = Err(ExecutionPositionError {
                message: "cash failed".to_owned(),
            });
        }
        let mut decision = make_decision(&[("A", OrderDir::Buy)]);
        let error = harness
            .collector
            .collect_data(&mut decision, &mut harness.account, 0)
            .unwrap_err();
        assert!(matches!(
            (failure, error),
            ("position", SimulatorCollectionError::Target(_))
                | ("cash", SimulatorCollectionError::Position(_))
                | ("report", SimulatorCollectionError::Reporter(_))
        ));
        assert_eq!(harness.collector.dealt_order_amount().get("A"), Some(&1.0));
    }
}

fn live_rollover_snapshot() -> Value {
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/backtest/executor.py");
    let script = r"
import ast,json,sys
from collections import defaultdict
import pandas as pd
p=sys.argv[1];t=ast.parse(open(p,encoding='utf-8').read());c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='SimulatorExecutor');n=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='_collect_data');n.returns=None
for a in n.args.args:a.annotation=None
ns={'defaultdict':defaultdict};exec(compile(ast.fix_missing_locations(ast.Module(body=[n],type_ignores=[])),p,'exec'),ns)
class O:
 def __init__(self,s):self.stock_id=s;self.direction=1;self.amount=1.;self.deal_amount=0.;self.factor=None
class C:
 def __init__(self):self.v=iter([pd.Timestamp('2024-01-02 09:30'),pd.Timestamp('2024-01-02 10:00'),pd.Timestamp('2024-01-03 09:30')])
 def get_step_time(self):x=next(self.v);return x,x
class X:
 def __init__(self):self.n=0;self.seen=[]
 def deal_order(self,o,**kw):self.seen.append(dict(kw['dealt_order_amount']));self.n+=1;o.deal_amount=float(self.n);o.factor=2.;return 10.*self.n,1.,10.
class E:
 _collect_data=ns['_collect_data']
 def _get_order_iterator(self,d):return d
e=E();e.trade_calendar=C();e.trade_exchange=X();e.trade_account=object();e.verbose=False;e.deal_day=pd.Timestamp('2024-01-02');e.dealt_order_amount=defaultdict(float,{'Z':4.});o=[O('A'),O('B')];r=e._collect_data(o);print(json.dumps({'seen':e.trade_exchange.seen,'amounts':dict(e.dealt_order_amount),'day':str(e.deal_day),'result':[[x[0].stock_id,x[1],x[2],x[3]] for x in r[0]],'alias':r[0] is r[1]['trade_info'],'orders':[[x.stock_id,x.deal_amount,x.factor] for x in o]},separators=(',',':')))
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

#[test]
fn collection_rollover_matches_live_python_source() {
    let python = live_rollover_snapshot();
    let mut harness = make_harness(
        "serial",
        vec![
            Ok(timestamp("2024-01-02 09:30:00")),
            Ok(timestamp("2024-01-02 10:00:00")),
            Ok(timestamp("2024-01-03 09:30:00")),
        ],
        None,
        false,
        None,
    );
    harness.collector.restore_intraday_state(
        Some(day("2024-01-02")),
        HashMap::from([("Z".to_owned(), 4.0)]),
    );
    let mut decision = make_decision(&[("A", OrderDir::Buy), ("B", OrderDir::Buy)]);
    let collection = harness
        .collector
        .collect_data(&mut decision, &mut harness.account, 0)
        .unwrap();
    let seen: Vec<Value> = harness
        .events
        .lock()
        .unwrap()
        .iter()
        .filter_map(|event| match event {
            Event::Deal(_, values) => Some(json!(
                values
                    .iter()
                    .map(|(key, bits)| (key.clone(), f64::from_bits(*bits)))
                    .collect::<BTreeMap<_, _>>()
            )),
            _ => None,
        })
        .collect();
    let result: Vec<Value> = collection
        .execution_result()
        .iter()
        .map(|execution| {
            json!([
                execution.order.stock_id(),
                execution.trade_value,
                execution.trade_cost,
                execution.trade_price
            ])
        })
        .collect();
    let orders: Vec<Value> = collection
        .execution_result()
        .iter()
        .map(|execution| {
            json!([
                execution.order.stock_id(),
                execution.order.deal_amount(),
                execution.order.factor()
            ])
        })
        .collect();
    let rust = json!({
        "seen": seen,
        "amounts": harness.collector.dealt_order_amount(),
        "day": "2024-01-03 00:00:00",
        "result": result,
        "alias": collection.execution_result().as_ptr() == collection.trade_info().as_ptr(),
        "orders": orders,
    });
    assert_eq!(rust, python);
}
