use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use chrono::NaiveDateTime;
use domain_core::{
    Account, AccountBarMarket, AccountBarMarketError, AccountIndicator, AccountIndicatorError,
    AtomicAccountAdapter, AtomicBarEnd, AtomicDecisionSnapshot, AtomicExecutorAccount,
    AtomicExecutorLifecycle, ExecutionTarget, ExecutionTargetError, ExecutorLifecycleCalendar,
    ExecutorLifecycleCalendarError, IndicatorConfig, IndicatorStoreAccess, InfinitePosition,
    InitialPositionValue, MetricSnapshot, NestedAccountIndicatorUpdate, NumpyOrderIndicator, Order,
    OrderDealResult, OrderDir, OrderExecution, OrderTradeDecision, Position, PositionHolding,
    SimulatorCalendar, SimulatorCalendarError, SimulatorCollector, SimulatorDealProvider,
    SimulatorExecutionLog, SimulatorExecutionReporter, SimulatorReporterError, TimeRange,
};
use indexmap::IndexMap;

#[path = "support/shared_account_cases.rs"]
mod shared_account_cases;

fn time(text: &str) -> NaiveDateTime {
    NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S").unwrap()
}

struct FixedSimulatorCalendar {
    events: Arc<Mutex<Vec<String>>>,
    time: NaiveDateTime,
}

impl SimulatorCalendar for FixedSimulatorCalendar {
    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), SimulatorCalendarError> {
        self.events
            .lock()
            .unwrap()
            .push("simulator-time".to_owned());
        Ok((self.time, self.time))
    }
}

struct FixedLifecycleCalendar {
    events: Arc<Mutex<Vec<String>>>,
    start: NaiveDateTime,
    end: NaiveDateTime,
}

impl ExecutorLifecycleCalendar for FixedLifecycleCalendar {
    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), ExecutorLifecycleCalendarError> {
        self.events
            .lock()
            .unwrap()
            .push("lifecycle-time".to_owned());
        Ok((self.start, self.end))
    }

    fn step(&self) -> Result<(), ExecutorLifecycleCalendarError> {
        self.events
            .lock()
            .unwrap()
            .push("lifecycle-step".to_owned());
        Ok(())
    }
}

struct ApplyingDealer {
    events: Arc<Mutex<Vec<String>>>,
}

impl SimulatorDealProvider for ApplyingDealer {
    fn deal_order(
        &self,
        order: &mut Order,
        account: &mut dyn ExecutionTarget,
        _dealt_order_amount: &HashMap<String, f64>,
    ) -> Result<OrderDealResult, domain_core::ExchangeDealError> {
        let (trade_value, trade_cost, trade_price) = if order.stock_id() == "A" {
            (10.0, 1.0, 10.0)
        } else {
            (5.0, 0.0, 5.0)
        };
        order.set_deal_amount(1.0);
        account.update_order(order, trade_value, trade_cost, trade_price)?;
        let cash = account.position()?.cash().map_err(|error| {
            domain_core::ExchangeDealError::Target(ExecutionTargetError {
                message: error.to_string(),
            })
        })?;
        self.events
            .lock()
            .unwrap()
            .push(format!("deal:{}:{cash:?}", order.stock_id()));
        Ok(OrderDealResult {
            trade_value,
            trade_cost,
            trade_price,
        })
    }
}

struct SilentReporter;

impl SimulatorExecutionReporter for SilentReporter {
    fn report(&self, _log: SimulatorExecutionLog<'_>) -> Result<(), SimulatorReporterError> {
        Ok(())
    }
}

struct SuspendedMarket {
    events: Arc<Mutex<Vec<String>>>,
    failure: bool,
}

impl AccountBarMarket for SuspendedMarket {
    fn is_suspended(&self, stock: &str, _range: TimeRange) -> Result<bool, AccountBarMarketError> {
        self.events.lock().unwrap().push(format!("market:{stock}"));
        if self.failure {
            return Err(AccountBarMarketError {
                message: "market".to_owned(),
            });
        }
        Ok(true)
    }

    fn close(&self, _stock: &str, _range: TimeRange) -> Result<f64, AccountBarMarketError> {
        unreachable!()
    }
}

#[derive(Default)]
struct FailingSnapshotIndicator {
    trade: domain_core::SharedTradeIndicator,
    history: IndexMap<NaiveDateTime, domain_core::SharedTradeIndicator>,
}

impl AccountIndicator for FailingSnapshotIndicator {
    fn reset(&mut self) -> Result<(), AccountIndicatorError> {
        Ok(())
    }

    fn update_atomic(
        &mut self,
        _executions: &[OrderExecution<'_>],
    ) -> Result<(), AccountIndicatorError> {
        Ok(())
    }

    fn update_nested(
        &mut self,
        _update: NestedAccountIndicatorUpdate<'_>,
    ) -> Result<(), AccountIndicatorError> {
        Ok(())
    }

    fn calculate(&mut self, _config: IndicatorConfig) -> Result<(), AccountIndicatorError> {
        Ok(())
    }

    fn record(&mut self, _trade_start_time: NaiveDateTime) -> Result<(), AccountIndicatorError> {
        Ok(())
    }

    fn trade_indicator(&self) -> &domain_core::SharedTradeIndicator {
        &self.trade
    }

    fn order_indicator_snapshot(&self) -> Result<NumpyOrderIndicator, AccountIndicatorError> {
        Err(AccountIndicatorError {
            message: "snapshot failed".to_owned(),
        })
    }

    fn order_indicator_handle(
        &self,
    ) -> Result<domain_core::SharedOrderIndicator<NumpyOrderIndicator>, AccountIndicatorError> {
        Err(AccountIndicatorError {
            message: "handle failed".to_owned(),
        })
    }

    fn recorded_trade_indicator(
        &self,
        time: NaiveDateTime,
    ) -> Option<&domain_core::SharedTradeIndicator> {
        self.history.get(&time)
    }

    fn trade_indicator_report(
        &self,
    ) -> Result<domain_core::TradeIndicatorReport, AccountIndicatorError> {
        Ok(domain_core::TradeIndicatorReport::from_shared_history(&self.history).unwrap())
    }
}

fn finite_position(cash: f64, holding: Option<(&str, f64, f64)>) -> Position {
    let positions = holding.map_or_else(IndexMap::new, |(stock, amount, price)| {
        IndexMap::from([(
            stock.to_owned(),
            InitialPositionValue::Holding(PositionHolding::restored(
                amount,
                Some(price),
                Some(0.0),
            )),
        )])
    });
    Position::from_initial(cash, positions)
}

#[test]
fn concrete_account_adapter_runs_the_full_atomic_lifecycle_with_cash_settlement() {
    let start = time("2024-01-02 09:30:00");
    let end = time("2024-01-02 09:30:59");
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut collector = SimulatorCollector::new(
        "serial",
        Arc::new(FixedSimulatorCalendar {
            events: Arc::clone(&events),
            time: start,
        }),
        Arc::new(ApplyingDealer {
            events: Arc::clone(&events),
        }),
        Arc::new(SilentReporter),
        false,
    );
    let lifecycle = AtomicExecutorLifecycle::new(
        Arc::new(FixedLifecycleCalendar {
            events: Arc::clone(&events),
            start,
            end,
        }),
        false,
        "cash",
        IndicatorConfig::default(),
    );
    let market = SuspendedMarket {
        events: Arc::clone(&events),
        failure: false,
    };
    let mut account = Account::new(finite_position(0.0, Some(("A", 1.0, 10.0))), false);
    let mut decision = OrderTradeDecision::from_orders(
        vec![
            Order::new("A", 1.0, OrderDir::Sell, Some(start), Some(end)),
            Order::new("B", 1.0, OrderDir::Buy, Some(start), Some(end)),
        ],
        start,
        end,
        None,
    );
    {
        let mut adapter = AtomicAccountAdapter::new(&mut account, &market, false);
        let collection = lifecycle
            .collect_data(&mut collector, &mut decision, &mut adapter, None, None, 0)
            .unwrap();
        assert_eq!(collection.execution_result().len(), 2);
        assert_eq!(
            collection.execution_result().as_ptr(),
            collection.trade_info().as_ptr()
        );
    }
    assert_eq!(
        events.lock().unwrap().as_slice(),
        [
            "simulator-time",
            "simulator-time",
            "deal:A:0.0",
            "simulator-time",
            "deal:B:-5.0",
            "lifecycle-time",
            "market:B",
            "lifecycle-step",
        ]
    );
    let position = account.execution_position().unwrap();
    assert_eq!(position.cash().unwrap().to_bits(), 4.0_f64.to_bits());
    assert!(!position.check_stock("A").unwrap());
    assert_eq!(
        position.stock_amount("B").unwrap().to_bits(),
        1.0_f64.to_bits()
    );
    assert!(
        account
            .indicator()
            .read()
            .unwrap()
            .recorded_trade_indicator(start)
            .is_some()
    );
}

fn empty_bar(start: NaiveDateTime, atomic: bool) -> AtomicBarEnd<'static> {
    AtomicBarEnd {
        trade_start_time: start,
        trade_end_time: start,
        atomic,
        outer_decision: AtomicDecisionSnapshot {
            start_time: start,
            end_time: start,
            order_count: 0,
            has_trade_range: false,
        },
        trade_info: &[],
        indicator_config: IndicatorConfig::default(),
    }
}

#[test]
fn adapter_maps_atomic_validation_market_and_finite_settlement_failures() {
    let start = time("2024-01-02 09:30:00");
    let events = Arc::new(Mutex::new(Vec::new()));
    let market = SuspendedMarket {
        events: Arc::clone(&events),
        failure: true,
    };
    let mut account = Account::new(finite_position(10.0, Some(("A", 1.0, 10.0))), false);
    let mut adapter = AtomicAccountAdapter::new(&mut account, &market, false);
    assert_eq!(
        adapter
            .update_bar_end(empty_bar(start, false))
            .unwrap_err()
            .message,
        "atomic account adapter requires atomic bar-end input"
    );
    assert!(events.lock().unwrap().is_empty());
    assert!(
        adapter
            .update_bar_end(empty_bar(start, true))
            .unwrap_err()
            .message
            .contains("account bar market error: market")
    );
    assert_eq!(events.lock().unwrap().as_slice(), ["market:A"]);

    adapter.settle_start("cash").unwrap();
    assert_eq!(
        adapter.settle_start("cash").unwrap_err().message,
        "account position error: settlement cannot be nested while cash is active"
    );
    adapter.settle_commit().unwrap();

    let mut unsupported = Account::new(finite_position(10.0, None), false);
    let mut adapter = AtomicAccountAdapter::new(&mut unsupported, &market, false);
    adapter.settle_start("other").unwrap();
    assert_eq!(
        adapter.settle_commit().unwrap_err().message,
        "account position error: unsupported settlement type: other"
    );
}

#[test]
fn infinite_position_adapter_exposes_target_and_ignores_settlement() {
    let market = SuspendedMarket {
        events: Arc::new(Mutex::new(Vec::new())),
        failure: false,
    };
    let mut account = Account::new(InfinitePosition, false);
    let mut adapter = AtomicAccountAdapter::new(&mut account, &market, false);
    adapter.settle_start("anything").unwrap();
    adapter.settle_commit().unwrap();
    assert!(
        adapter
            .execution_target()
            .unwrap()
            .position()
            .unwrap()
            .cash()
            .unwrap()
            .is_infinite()
    );
    assert_eq!(
        IndicatorStoreAccess::metric_names(&adapter.order_indicator_snapshot().unwrap()).count(),
        0
    );

    let mut failing = Account::new(InfinitePosition, false);
    failing.replace_indicator(Box::new(FailingSnapshotIndicator::default()));
    let adapter = AtomicAccountAdapter::new(&mut failing, &market, false);
    assert_eq!(
        adapter.order_indicator_snapshot().unwrap_err().message,
        "account indicator plugin error: snapshot failed"
    );
    assert_eq!(
        adapter.order_indicator_handle().unwrap_err().message,
        "account indicator plugin error: handle failed"
    );
}

#[test]
fn raw_account_and_atomic_handles_survive_reset_replacement_and_account_drop() {
    let mut account = Account::new(InfinitePosition, false);
    let market = SuspendedMarket {
        events: Arc::default(),
        failure: true,
    };
    let original = account.order_indicator_handle().unwrap();
    original.write().unwrap().assign_snapshot(
        "trade_price",
        MetricSnapshot::try_new(vec!["A".into()], vec![3.0]).unwrap(),
    );
    let frozen = account.order_indicator_snapshot().unwrap();
    let retained = {
        let adapter = AtomicAccountAdapter::new(&mut account, &market, false);
        // Identity publication must not read/reacquire this independently held store lock.
        let _guard = original.write().unwrap();
        let first = adapter.order_indicator_handle().unwrap();
        let second = adapter.order_indicator_handle().unwrap();
        assert!(Arc::ptr_eq(&original, &first));
        assert!(Arc::ptr_eq(&first, &second));
        first
    };
    assert!(account.indicator().try_write().is_ok());
    assert!(market.events.lock().unwrap().is_empty());
    account.indicator().write().unwrap().reset().unwrap();
    let after_reset = account.order_indicator_handle().unwrap();
    assert!(!Arc::ptr_eq(&retained, &after_reset));
    retained.write().unwrap().assign_snapshot(
        "trade_price",
        MetricSnapshot::try_new(vec!["A".into()], vec![7.0]).unwrap(),
    );
    assert_eq!(
        frozen.metric_snapshot("trade_price").unwrap().values(),
        [3.0]
    );
    let old_engine = account.replace_indicator(Box::new(domain_core::Indicator::new()));
    let after_replace = account.order_indicator_handle().unwrap();
    assert!(!Arc::ptr_eq(&after_reset, &after_replace));
    assert!(Arc::ptr_eq(
        &after_reset,
        &old_engine.read().unwrap().order_indicator_handle().unwrap()
    ));
    drop(old_engine);
    drop(account);
    assert_eq!(
        retained
            .read()
            .unwrap()
            .metric_snapshot("trade_price")
            .unwrap()
            .values(),
        [7.0]
    );
    assert_eq!(
        IndicatorStoreAccess::metric_names(&*after_reset.read().unwrap()).count(),
        0
    );
    assert_eq!(
        IndicatorStoreAccess::metric_names(&*after_replace.read().unwrap()).count(),
        0
    );
}

#[test]
fn raw_handle_publication_distinguishes_store_poison_from_engine_poison() {
    let mut account = Account::new(InfinitePosition, false);
    let market = SuspendedMarket {
        events: Arc::default(),
        failure: true,
    };
    let raw = account.order_indicator_handle().unwrap();
    let poison = raw.clone();
    assert!(
        std::thread::spawn(move || {
            let _guard = poison.write().unwrap();
            panic!("poison order store only");
        })
        .join()
        .is_err()
    );
    assert!(Arc::ptr_eq(
        &raw,
        &account.order_indicator_handle().unwrap()
    ));
    {
        let adapter = AtomicAccountAdapter::new(&mut account, &market, false);
        assert!(Arc::ptr_eq(
            &raw,
            &adapter.order_indicator_handle().unwrap()
        ));
        assert_eq!(
            adapter.order_indicator_snapshot().unwrap_err().message,
            "account indicator plugin error: order indicator store lock poisoned"
        );
    }
    assert!(account.indicator().try_write().is_ok());
    account.indicator().write().unwrap().reset().unwrap();
    assert!(account.order_indicator_handle().unwrap().read().is_ok());
    assert!(raw.read().is_err());
    let engine = account.indicator().clone();
    assert!(
        std::thread::spawn(move || {
            let _guard = engine.write().unwrap();
            panic!("poison indicator engine");
        })
        .join()
        .is_err()
    );
    assert_eq!(
        account.order_indicator_handle().unwrap_err().to_string(),
        "account indicator plugin error: account indicator lock poisoned"
    );
    let adapter = AtomicAccountAdapter::new(&mut account, &market, false);
    assert_eq!(
        adapter.order_indicator_handle().unwrap_err().message,
        "account indicator plugin error: account indicator lock poisoned"
    );
    assert!(market.events.lock().unwrap().is_empty());
}
