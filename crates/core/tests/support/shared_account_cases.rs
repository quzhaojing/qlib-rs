use super::*;
use domain_core::decision_construction::{
    DecisionConstructionStrategy, DecisionOrderItem, SharedOrderDecisionConstruction,
};
use domain_core::shared_executor_lifecycle::{
    SharedAtomicBarEnd, SharedAtomicExecutorAccount, SharedAtomicExecutorLifecycle,
    SharedAtomicResult,
};
use domain_core::{Indicator, IndicatorError, IndicatorStore, PandasOrderIndicator};
use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::RwLock,
};

struct Clock;
impl DecisionConstructionStrategy for Clock {
    type Error = std::convert::Infallible;
    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), Self::Error> {
        let start = time("2024-01-02 09:30:00");
        Ok((start, start))
    }
}
type Input = SharedOrderDecisionConstruction<Clock, ()>;

impl domain_core::decision_update::SharedDecisionUpdateStrategy<()> for Clock {
    fn update_trade_decision(
        &self,
        _: &domain_core::decision_update::SharedLiveDecision<Self, ()>,
        _: &dyn domain_core::DecisionUpdateCalendar,
    ) -> Result<
        Option<domain_core::decision_update::SharedLiveDecision<Self, ()>>,
        domain_core::DecisionUpdateStrategyError,
    > {
        Ok(None)
    }
}

fn input(order: &Arc<RwLock<Order>>) -> Input {
    let mut input = Input::new(Clock);
    input
        .initialize(
            &Arc::new(RwLock::new(vec![DecisionOrderItem::Order(Arc::clone(
                order,
            ))])),
            None,
            (),
        )
        .unwrap();
    input
}

fn collector(events: &Arc<Mutex<Vec<String>>>) -> SimulatorCollector {
    SimulatorCollector::new(
        "serial",
        Arc::new(FixedSimulatorCalendar {
            events: Arc::clone(events),
            time: time("2024-01-02 09:30:00"),
        }),
        Arc::new(ApplyingDealer {
            events: Arc::clone(events),
        }),
        Arc::new(SilentReporter),
        false,
    )
}

fn result(order: &Arc<RwLock<Order>>) -> SharedAtomicResult {
    let input = input(order);
    let rows = collector(&Arc::default())
        .collect_shared_data(
            input.orders.as_ref().unwrap(),
            &mut Account::new(InfinitePosition, false),
            0,
        )
        .unwrap();
    rows.into_shared()
}

struct MutatingMarket {
    order: Arc<RwLock<Order>>,
    events: Arc<Mutex<Vec<String>>>,
}
impl AccountBarMarket for MutatingMarket {
    fn is_suspended(&self, _stock: &str, _range: TimeRange) -> Result<bool, AccountBarMarketError> {
        self.events.lock().unwrap().push("market".to_owned());
        self.order.write().unwrap().set_deal_amount(2.0);
        Ok(true)
    }
    fn close(&self, _stock: &str, _range: TimeRange) -> Result<f64, AccountBarMarketError> {
        panic!("suspended")
    }
}

struct ResetIndicator {
    native: Indicator<NumpyOrderIndicator>,
    order: Arc<RwLock<Order>>,
    events: Arc<Mutex<Vec<String>>>,
}
impl AccountIndicator for ResetIndicator {
    fn reset(&mut self) -> Result<(), AccountIndicatorError> {
        self.events.lock().unwrap().push("reset".to_owned());
        assert_eq!(
            self.order.read().unwrap().deal_amount().to_bits(),
            2.0_f64.to_bits()
        );
        self.order.write().unwrap().set_deal_amount(3.0);
        AccountIndicator::reset(&mut self.native)
    }
    fn update_shared_atomic(
        &mut self,
        value: &SharedAtomicResult,
    ) -> Result<(), AccountIndicatorError> {
        self.events.lock().unwrap().push("read".to_owned());
        AccountIndicator::update_shared_atomic(&mut self.native, value)
    }
    fn update_atomic(&mut self, value: &[OrderExecution<'_>]) -> Result<(), AccountIndicatorError> {
        AccountIndicator::update_atomic(&mut self.native, value)
    }
    fn update_nested(
        &mut self,
        value: NestedAccountIndicatorUpdate<'_>,
    ) -> Result<(), AccountIndicatorError> {
        AccountIndicator::update_nested(&mut self.native, value)
    }
    fn calculate(&mut self, value: IndicatorConfig) -> Result<(), AccountIndicatorError> {
        self.events.lock().unwrap().push("calculate".to_owned());
        AccountIndicator::calculate(&mut self.native, value)
    }
    fn record(&mut self, value: NaiveDateTime) -> Result<(), AccountIndicatorError> {
        self.events.lock().unwrap().push("record".to_owned());
        AccountIndicator::record(&mut self.native, value)
    }
    fn trade_indicator(&self) -> &domain_core::SharedTradeIndicator {
        AccountIndicator::trade_indicator(&self.native)
    }
    fn order_indicator_snapshot(&self) -> Result<NumpyOrderIndicator, AccountIndicatorError> {
        AccountIndicator::order_indicator_snapshot(&self.native)
    }
    fn order_indicator_handle(
        &self,
    ) -> Result<domain_core::SharedOrderIndicator<NumpyOrderIndicator>, AccountIndicatorError> {
        AccountIndicator::order_indicator_handle(&self.native)
    }
    fn recorded_trade_indicator(
        &self,
        value: NaiveDateTime,
    ) -> Option<&domain_core::SharedTradeIndicator> {
        AccountIndicator::recorded_trade_indicator(&self.native, value)
    }
    fn trade_indicator_report(
        &self,
    ) -> Result<domain_core::TradeIndicatorReport, AccountIndicatorError> {
        AccountIndicator::trade_indicator_report(&self.native)
    }
}

fn source_read_timing() -> serde_json::Value {
    let output = std::process::Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/shared_account_read_contract.py"
        ))
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
fn native_shared_account_reads_orders_after_market_and_indicator_reset() {
    verify_shared_account_read_timing(false);
    verify_shared_account_read_timing(true);
}

fn verify_shared_account_read_timing(live: bool) {
    let source = source_read_timing();
    let events = Arc::default();
    let order = Arc::new(RwLock::new(Order::new("B", 2.0, OrderDir::Buy, None, None)));
    let mut input = input(&order);
    let start = time("2024-01-02 09:30:00");
    let lifecycle = SharedAtomicExecutorLifecycle {
        calendar: Arc::new(FixedLifecycleCalendar {
            events: Arc::clone(&events),
            start,
            end: start,
        }),
        track_data: false,
        settle_type: "cash".to_owned(),
        indicator_config: IndicatorConfig::default(),
    };
    let market = MutatingMarket {
        order: Arc::clone(&order),
        events: Arc::clone(&events),
    };
    let mut account = Account::new(finite_position(100.0, None), false);
    account.replace_indicator(Box::new(ResetIndicator {
        native: Indicator::new(),
        order: Arc::clone(&order),
        events: Arc::clone(&events),
    }));
    let result = if live {
        let decision: domain_core::decision_update::LiveDecisionHandle =
            Arc::new(RwLock::new(SharedOrderDecisionConstruction {
                strategy: Arc::new(Clock),
                base: input.base,
                total_step: input.total_step,
                orders: input.orders,
                details: input.details,
            }));
        lifecycle
            .collect_live_data(
                &mut collector(&events),
                &decision,
                &mut AtomicAccountAdapter::new(&mut account, &market, false),
                None,
                None,
                0,
            )
            .unwrap()
    } else {
        lifecycle
            .collect_data(
                &mut collector(&events),
                &mut input,
                &mut AtomicAccountAdapter::new(&mut account, &market, false),
                None,
                None,
                0,
            )
            .unwrap()
    };
    let metrics = account.order_indicator_snapshot().unwrap();
    let stages: Vec<String> = events
        .lock()
        .unwrap()
        .iter()
        .filter(|event| {
            ["market", "reset", "read", "calculate", "record"].contains(&event.as_str())
        })
        .cloned()
        .collect();
    assert_eq!(
        serde_json::json!({"events": stages, "deal_amount": metrics.metric_snapshot("deal_amount").unwrap().values()[0], "amount": metrics.metric_snapshot("amount").unwrap().values()[0]}),
        source
    );
    assert_eq!(
        metrics.metric_snapshot("deal_amount").unwrap().values(),
        [3.0]
    );
    assert_eq!(metrics.metric_snapshot("ffr").unwrap().values(), [1.5]);
    assert_eq!(
        account
            .execution_position()
            .unwrap()
            .cash()
            .unwrap()
            .to_bits(),
        95.0_f64.to_bits()
    );
    assert!(Arc::ptr_eq(&result.lock().unwrap()[0].order, &order));
    assert_shared_account_events(&events);
    order.write().unwrap().set_deal_amount(4.0);
    assert_eq!(
        metrics.metric_snapshot("deal_amount").unwrap().values(),
        [3.0]
    );
    assert_eq!(
        result.lock().unwrap()[0]
            .order
            .read()
            .unwrap()
            .deal_amount()
            .to_bits(),
        4.0_f64.to_bits()
    );
}

fn assert_shared_account_events(events: &Arc<Mutex<Vec<String>>>) {
    assert_eq!(
        events.lock().unwrap().as_slice(),
        [
            "simulator-time",
            "simulator-time",
            "deal:B:95.0",
            "lifecycle-time",
            "market",
            "reset",
            "read",
            "calculate",
            "record",
            "lifecycle-step"
        ]
    );
}

fn bar(result: SharedAtomicResult) -> SharedAtomicBarEnd {
    let start = time("2024-01-02 09:30:00");
    SharedAtomicBarEnd {
        trade_start_time: start,
        trade_end_time: start,
        trade_info: result,
        indicator_config: IndicatorConfig::default(),
    }
}

#[test]
fn account_rejects_unsupported_backends_and_maps_live_input_failures_after_reset() {
    let order = Arc::new(RwLock::new(Order::new("B", 2.0, OrderDir::Buy, None, None)));
    let rows = result(&order);
    let mut input = input(&order);
    let market = SuspendedMarket {
        events: Arc::default(),
        failure: false,
    };
    let mut account = Account::new(InfinitePosition, false);
    account.replace_indicator(Box::<FailingSnapshotIndicator>::default());
    let failure = AtomicAccountAdapter::new(&mut account, &market, false)
        .update_shared_bar_end(bar(Arc::clone(&rows)), &mut input)
        .unwrap_err();
    assert!(failure.message.contains("does not support shared atomic"));
    account.replace_indicator(Box::new(Indicator::new()));
    AtomicAccountAdapter::new(&mut account, &market, false)
        .update_shared_bar_end(bar(Arc::clone(&rows)), &mut input)
        .unwrap();
    let old = account.order_indicator_handle().unwrap();
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _guard = order.write().unwrap();
            panic!("order poison");
        }))
        .is_err()
    );
    let error = AtomicAccountAdapter::new(&mut account, &market, false)
        .update_shared_bar_end(bar(Arc::clone(&rows)), &mut input)
        .unwrap_err();
    assert!(
        error
            .message
            .contains("shared execution order 0 lock poisoned")
    );
    assert!(
        account
            .order_indicator_snapshot()
            .unwrap()
            .metric_names()
            .next()
            .is_none()
    );
    assert!(old.read().unwrap().metric_snapshot("deal_amount").is_some());
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _guard = rows.lock().unwrap();
            panic!("list poison");
        }))
        .is_err()
    );
    let error = AtomicAccountAdapter::new(&mut account, &market, false)
        .update_shared_bar_end(bar(rows), &mut input)
        .unwrap_err();
    assert!(
        error
            .message
            .contains("shared execution list lock poisoned")
    );
}

fn shared_metric_failures<S: IndicatorStore + 'static>() {
    let order = Arc::new(RwLock::new(Order::new("B", 2.0, OrderDir::Buy, None, None)));
    let rows = result(&order);
    let mut native = Indicator::<S>::default();
    native.update_shared_order_indicators(&rows).unwrap();
    assert_eq!(native.order_snapshot().unwrap()["ffr"].values(), [0.5]);
    rows.lock().unwrap().clear();
    native.update_shared_order_indicators(&rows).unwrap();
    assert!(
        native.order_snapshot().unwrap()["amount"]
            .values()
            .is_empty()
    );
    let store = Arc::clone(native.order_indicator());
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _guard = store.write().unwrap();
            panic!("store poison");
        }))
        .is_err()
    );
    assert_eq!(
        native.update_shared_order_indicators(&rows),
        Err(IndicatorError::OrderStorePoisoned)
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _guard = rows.lock().unwrap();
            panic!("list poison");
        }))
        .is_err()
    );
    assert_eq!(
        native.update_shared_order_indicators(&rows),
        Err(IndicatorError::ExecutionListPoisoned)
    );
    let rows = result(&order);
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _guard = order.write().unwrap();
            panic!("order poison");
        }))
        .is_err()
    );
    assert_eq!(
        native.update_shared_order_indicators(&rows),
        Err(IndicatorError::ExecutionOrderPoisoned(0))
    );
}

#[test]
fn shared_metric_store_failures_and_empty_inputs_preserve_both_backends() {
    shared_metric_failures::<NumpyOrderIndicator>();
    shared_metric_failures::<PandasOrderIndicator>();
}
