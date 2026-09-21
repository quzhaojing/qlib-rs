use std::{
    collections::VecDeque,
    process::Command,
    sync::{Arc, Mutex, RwLock},
};

use arrow_array::{Float64Array, Int8Array, RecordBatch, StringArray};
use arrow_schema::Schema;
use chrono::{NaiveDateTime, TimeDelta};
use domain_core::{
    AllOnePolicy, ConcreteSaoeStateAdapter, DummyStateInterpreter, LiveSaoeState, Order, OrderDir,
    OwnedOrderExecution, SaoeActionInterpreter, SaoeActionSpace, SaoeAdapterConfig,
    SaoeAdapterContext, SaoeAdapterError, SaoeAdapterMarket, SaoeBacktestData,
    SaoeInterpreterError, SaoeMarketSlice, SaoeMetricRow, SaoeNumeric, SaoeObservation,
    SaoePluginError, SaoePolicy, SaoePolicyAction, SaoePolicyPipeline, SaoeState,
    SaoeStateInterpreter, SaoeStateProvider, SharedOrderExecution, SharedSaoeHistory,
    TwapRelativeActionInterpreter,
};
use ndarray::arr1;
use serde_json::Value;

#[path = "support/saoe_live_adapter_contract.rs"]
mod live_contract;

#[path = "support/saoe_live_market.rs"]
mod live_market;

fn time(minute: i64) -> NaiveDateTime {
    NaiveDateTime::parse_from_str("2024-01-02 09:30:00", "%Y-%m-%d %H:%M:%S").unwrap()
        + TimeDelta::minutes(minute)
}

fn backtest(ticks: Vec<NaiveDateTime>, order_ticks: Vec<NaiveDateTime>) -> SaoeBacktestData {
    SaoeBacktestData {
        ticks_index: ticks,
        ticks_for_order: order_ticks,
        deal_prices: arr1(&[]),
        market_volumes: arr1(&[]),
        features: RecordBatch::new_empty(Arc::new(Schema::empty())),
    }
}

fn config() -> SaoeAdapterConfig {
    SaoeAdapterConfig {
        backtest_data: backtest((0..6).map(time).collect(), (0..6).map(time).collect()),
        deal_prices: arr1(&[10.0, 11.0, 12.0, 13.0, 14.0, 15.0]),
        ticks_per_step: 2,
        data_granularity: 1,
        start_step: 0,
    }
}

#[derive(Default)]
struct ContextState {
    fail: Option<&'static str>,
    step: i64,
    pa: f64,
    warnings: usize,
}

struct Context(Mutex<ContextState>);

impl Context {
    fn new() -> Self {
        Self(Mutex::new(ContextState {
            step: 3,
            pa: 7.5,
            ..ContextState::default()
        }))
    }

    fn check(&self, stage: &'static str) -> Result<(), SaoePluginError> {
        if self.0.lock().unwrap().fail == Some(stage) {
            return Err(SaoePluginError {
                message: stage.to_owned(),
            });
        }
        Ok(())
    }
}

impl SaoeAdapterContext for Context {
    fn current_trade_step(&self) -> Result<i64, SaoePluginError> {
        self.check("step")?;
        Ok(self.0.lock().unwrap().step)
    }

    fn latest_price_advantage(&self) -> Result<f64, SaoePluginError> {
        self.check("pa")?;
        Ok(self.0.lock().unwrap().pa)
    }

    fn warn_overfill(&self, _execution_volume: f64, _position: f64) -> Result<(), SaoePluginError> {
        self.check("warning")?;
        self.0.lock().unwrap().warnings += 1;
        Ok(())
    }
}

struct Market {
    responses: Mutex<VecDeque<Result<SaoeMarketSlice, SaoePluginError>>>,
}

impl Market {
    fn new(responses: impl IntoIterator<Item = SaoeMarketSlice>) -> Self {
        Self {
            responses: Mutex::new(responses.into_iter().map(Ok).collect()),
        }
    }

    fn failing() -> Self {
        Self {
            responses: Mutex::new(VecDeque::from([Err(SaoePluginError {
                message: "market".to_owned(),
            })])),
        }
    }
}

impl SaoeAdapterMarket for Market {
    fn market_slice(
        &self,
        _stock_id: &str,
        _start: NaiveDateTime,
        _end: NaiveDateTime,
        _direction: OrderDir,
    ) -> Result<SaoeMarketSlice, SaoePluginError> {
        self.responses
            .lock()
            .unwrap()
            .pop_front()
            .expect("test supplies every market response")
    }
}

fn slice(volume: &[f64], price: &[f64]) -> SaoeMarketSlice {
    SaoeMarketSlice {
        volume: arr1(volume),
        price: arr1(price),
    }
}

fn order(direction: OrderDir) -> Order {
    Order::new("A", 10.0, direction, Some(time(0)), Some(time(5)))
}

fn execution(minute: i64, deal_amount: f64) -> SharedOrderExecution {
    let mut order = Order::new(
        "A",
        deal_amount,
        OrderDir::Buy,
        Some(time(minute)),
        Some(time(minute)),
    );
    order.set_deal_amount(deal_amount);
    OwnedOrderExecution {
        order,
        trade_value: 0.0,
        trade_cost: 0.0,
        trade_price: 0.0,
    }
    .into_shared()
}

fn populated_alias_adapter() -> (ConcreteSaoeStateAdapter, Arc<RwLock<Order>>) {
    let context = Arc::new(Context::new());
    let market = Arc::new(Market::new([
        slice(&[100.0, 200.0], &[10.0, 12.0]),
        slice(&[300.0, 400.0], &[14.0, 16.0]),
    ]));
    let order = Arc::new(RwLock::new(order(OrderDir::Buy)));
    let mut adapter = ConcreteSaoeStateAdapter::new(market, context, config()).unwrap();
    adapter.reset_shared_order(&order).unwrap();
    adapter
        .update_executions(&[execution(0, 2.0), execution(1, 3.0)], (0, 1))
        .unwrap();
    (adapter, order)
}

fn poison_lock<T: Send + Sync + 'static>(value: Arc<RwLock<T>>) {
    assert!(
        std::thread::spawn(move || {
            let _guard = value.write().unwrap();
            panic!("deliberate live-state alias poison");
        })
        .join()
        .is_err()
    );
}

fn assert_backtest_field_poison<T: Send + Sync + 'static>(
    state: &LiveSaoeState,
    field: Arc<RwLock<T>>,
    name: &'static str,
) {
    poison_lock(field);
    assert!(matches!(
        state.snapshot(),
        Err(SaoeAdapterError::BacktestDataFieldPoisoned(actual)) if actual == name
    ));
}

struct AliasStateInterpreter;

impl SaoeStateInterpreter for AliasStateInterpreter {
    fn interpret(&self, _: &SaoeState) -> Result<SaoeObservation, SaoeInterpreterError> {
        unreachable!("live pipeline must dispatch the alias-aware callback")
    }

    fn interpret_live(
        &self,
        state: &LiveSaoeState,
    ) -> Result<SaoeObservation, SaoeInterpreterError> {
        "pipeline".clone_into(&mut state.history_exec().write().unwrap()[0].stock_id);
        Ok(SaoeObservation::Dummy { dummy: 1 })
    }
}

struct AliasActionInterpreter;

impl SaoeActionInterpreter for AliasActionInterpreter {
    fn action_space(&self) -> SaoeActionSpace {
        SaoeActionSpace::NonNegativeContinuous
    }

    fn interpret(&self, _: &SaoeState, _: SaoePolicyAction) -> Result<f64, SaoeInterpreterError> {
        unreachable!("live pipeline must retain aliases through the action callback")
    }

    fn interpret_live(
        &self,
        state: &LiveSaoeState,
        _: SaoePolicyAction,
    ) -> Result<f64, SaoeInterpreterError> {
        assert_eq!(state.history_exec().read().unwrap()[0].stock_id, "pipeline");
        Ok(3.0)
    }
}

struct PoisonPolicy(SharedSaoeHistory);

impl SaoePolicy for PoisonPolicy {
    fn actions(
        &mut self,
        observations: &[SaoeObservation],
    ) -> Result<Vec<SaoePolicyAction>, SaoeInterpreterError> {
        poison_lock(Arc::clone(&self.0));
        Ok(vec![SaoePolicyAction::Continuous(1.0); observations.len()])
    }
}

struct FailingPolicy;

impl SaoePolicy for FailingPolicy {
    fn actions(
        &mut self,
        _: &[SaoeObservation],
    ) -> Result<Vec<SaoePolicyAction>, SaoeInterpreterError> {
        Err(SaoeInterpreterError::ProcessedDataPlugin(
            "live policy failed".to_owned(),
        ))
    }
}

fn python_contract() -> Value {
    let output = Command::new(std::env::var_os("PYTHON").unwrap_or_else(|| "python".into()))
        .args([
            "-W",
            "ignore",
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/saoe_state_adapter_contract.py"
            ),
            r"D:\code\github\qlib\qlib\rl\order_execution\strategy.py",
            r"D:\code\github\qlib\qlib\rl\order_execution\utils.py",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1.0e-9 * expected.abs().max(1.0),
        "{actual} != {expected}"
    );
}

#[test]
fn shared_calendar_mutation_and_poison_reach_execution_without_partial_commit() {
    use domain_core::{
        time_calendar_cache::{TimeCalendarCall, TimeCalendarKeyword, default_time_calendar_cache},
        time_compat::TimeCompatError,
    };
    // Isolate the process-wide cache from concurrently running adapter tests;
    // production code is identical in the child, including its shared cache.
    const CHILD: &str = "QLIB_SAOE_CALENDAR_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "shared_calendar_mutation_and_poison_reach_execution_without_partial_commit",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "child stdout={}\nstderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    let expected = python_contract();
    let cache = default_time_calendar_cache();
    for label in ["remove_open", "empty"] {
        cache.cache_clear();
        let calendar = cache
            .get(&TimeCalendarCall::new(
                Vec::new(),
                vec![TimeCalendarKeyword::new("region", "cn")],
            ))
            .unwrap();
        if label == "remove_open" {
            calendar.lock().unwrap().remove(0);
        } else {
            calendar.lock().unwrap().clear();
        }
        let market = Arc::new(Market::new([slice(&[100.0, 200.0], &[10.0, 12.0])]));
        let mut adapter =
            ConcreteSaoeStateAdapter::new(market, Arc::new(Context::new()), config()).unwrap();
        adapter.reset_order(&order(OrderDir::Buy)).unwrap();
        adapter
            .update_executions(&[execution(1, 2.0)], (0, 1))
            .unwrap();
        let actual: Vec<_> = adapter
            .history_exec()
            .unwrap()
            .iter()
            .map(|row| row.deal_amount)
            .collect();
        assert_eq!(
            serde_json::json!(actual),
            expected["calendar_mutation_exec"][label]
        );
    }
    cache.cache_clear();
    let calendar = cache
        .get(&TimeCalendarCall::new(
            Vec::new(),
            vec![TimeCalendarKeyword::new("region", "cn")],
        ))
        .unwrap();
    assert!(
        std::thread::spawn(move || {
            let _guard = calendar.lock().unwrap();
            panic!("test calendar value poison");
        })
        .join()
        .is_err()
    );
    let market = Arc::new(Market::new([slice(&[100.0, 200.0], &[10.0, 12.0])]));
    let context = Arc::new(Context::new());
    let mut adapter =
        ConcreteSaoeStateAdapter::new(market.clone(), context.clone(), config()).unwrap();
    adapter.reset_order(&order(OrderDir::Buy)).unwrap();
    let error = adapter
        .update_executions(&[execution(0, 2.0)], (0, 1))
        .unwrap_err();
    assert!(matches!(
        error,
        SaoeAdapterError::TimeCompatibility(TimeCompatError::CalendarLockPoisoned)
    ));
    assert_eq!(
        error.to_string(),
        "mutable minute calendar lock is poisoned"
    );
    assert!(adapter.history_exec().unwrap().is_empty());
    assert!(adapter.history_steps().unwrap().is_empty());
    close(adapter.position(), 10.0);
    assert_eq!(adapter.cur_time(), Some(time(0)));
    assert_eq!(market.responses.lock().unwrap().len(), 1);
    assert_eq!(context.0.lock().unwrap().warnings, 0);
    cache.cache_clear();
    adapter
        .update_executions(&[execution(0, 2.0)], (0, 1))
        .unwrap();
    close(adapter.position(), 8.0);
    assert_eq!(adapter.history_exec().unwrap().len(), 2);
    assert!(market.responses.lock().unwrap().is_empty());
}

#[test]
fn nanosecond_execution_uses_source_range_clock_precision() {
    let expected = python_contract();
    let market = Arc::new(Market::new([slice(&[100.0, 200.0], &[10.0, 12.0])]));
    let mut adapter =
        ConcreteSaoeStateAdapter::new(market, Arc::new(Context::new()), config()).unwrap();
    adapter.reset_order(&order(OrderDir::Buy)).unwrap();
    let mut fill = Order::new(
        "A",
        2.0,
        OrderDir::Buy,
        Some(time(0) + TimeDelta::nanoseconds(1)),
        Some(time(0)),
    );
    fill.set_deal_amount(2.0);
    let execution = OwnedOrderExecution {
        order: fill,
        trade_value: 0.0,
        trade_cost: 0.0,
        trade_price: 0.0,
    }
    .into_shared();
    adapter.update_executions(&[execution], (0, 1)).unwrap();
    let actual: Vec<_> = adapter
        .history_exec()
        .unwrap()
        .iter()
        .map(|row| row.deal_amount)
        .collect();
    assert_eq!(serde_json::json!(actual), expected["nanosecond_exec"]);
}

#[test]
fn live_state_preserves_current_aliases_and_replaces_histories_on_append() {
    let context = Arc::new(Context::new());
    let market = Arc::new(Market::new([
        slice(&[100.0, 200.0], &[10.0, 12.0]),
        slice(&[300.0, 400.0], &[14.0, 16.0]),
    ]));
    let shared_order = Arc::new(RwLock::new(order(OrderDir::Buy)));
    let mut adapter = ConcreteSaoeStateAdapter::new(market, context, config()).unwrap();
    adapter.reset_shared_order(&shared_order).unwrap();
    adapter
        .update_executions(&[execution(0, 2.0), execution(1, 3.0)], (0, 1))
        .unwrap();

    let old_state = adapter.live_state().unwrap();
    assert!(Arc::ptr_eq(old_state.order(), &shared_order));
    *shared_order.write().unwrap() =
        Order::new("Z", 10.0, OrderDir::Buy, Some(time(0)), Some(time(5)));
    "M".clone_into(&mut old_state.history_exec().write().unwrap()[0].stock_id);
    old_state.history_steps().write().unwrap()[0].amount = 42.0;

    adapter
        .update_executions(&[execution(2, 1.0), execution(3, 1.0)], (2, 3))
        .unwrap();
    let current = adapter.live_state().unwrap();
    assert!(Arc::ptr_eq(current.order(), old_state.order()));
    assert!(!Arc::ptr_eq(
        current.history_exec(),
        old_state.history_exec()
    ));
    assert!(!Arc::ptr_eq(
        current.history_steps(),
        old_state.history_steps()
    ));
    assert_eq!(old_state.history_exec().read().unwrap().len(), 2);
    assert_eq!(old_state.history_steps().read().unwrap().len(), 1);
    assert_eq!(current.history_exec().read().unwrap().len(), 4);
    assert_eq!(current.history_steps().read().unwrap().len(), 2);
    assert_eq!(current.history_exec().read().unwrap()[0].stock_id, "M");
    assert_eq!(
        current.history_steps().read().unwrap()[0].amount.to_bits(),
        42.0_f64.to_bits()
    );
    assert_eq!(current.history_exec().read().unwrap()[3].stock_id, "Z");
    assert_eq!(current.snapshot().unwrap().parts().order.stock_id(), "Z");

    let expected = &live_contract::source()["aliases"];
    assert_eq!(
        serde_json::json!({
            "order_alias": Arc::ptr_eq(current.order(), old_state.order()),
            "initial_history_alias": !Arc::ptr_eq(current.history_exec(), old_state.history_exec())
                && !Arc::ptr_eq(current.history_steps(), old_state.history_steps()),
            "old_lengths": [old_state.history_exec().read().unwrap().len(), old_state.history_steps().read().unwrap().len()],
            "new_lengths": [current.history_exec().read().unwrap().len(), current.history_steps().read().unwrap().len()],
            "mutated_exec_stock": current.history_exec().read().unwrap()[0].stock_id,
            "mutated_step_amount": current.history_steps().read().unwrap()[0].amount,
            "new_exec_stock": current.history_exec().read().unwrap()[3].stock_id,
            "state_stock": current.snapshot().unwrap().parts().order.stock_id(),
        }),
        *expected
    );
}

#[test]
fn live_final_metrics_are_shared_until_regeneration_replaces_the_dictionary() {
    let context = Arc::new(Context::new());
    let market = Arc::new(Market::new([slice(&[100.0, 200.0], &[10.0, 12.0])]));
    let shared_order = Arc::new(RwLock::new(order(OrderDir::Buy)));
    let mut adapter = ConcreteSaoeStateAdapter::new(market, context, config()).unwrap();
    adapter.reset_shared_order(&shared_order).unwrap();
    adapter
        .update_executions(&[execution(0, 2.0), execution(1, 3.0)], (0, 1))
        .unwrap();
    let before_finalize = adapter.live_state().unwrap();
    assert!(before_finalize.metrics().is_none());

    adapter.finalize_metrics().unwrap();
    let first = adapter.live_state().unwrap();
    let second = adapter.live_state().unwrap();
    let old_metrics = Arc::clone(first.metrics().unwrap());
    assert!(Arc::ptr_eq(
        first.metrics().unwrap(),
        second.metrics().unwrap()
    ));
    "M".clone_into(&mut old_metrics.write().unwrap().stock_id);
    assert_eq!(
        second
            .snapshot()
            .unwrap()
            .parts()
            .metrics
            .as_ref()
            .unwrap()
            .stock_id,
        "M"
    );

    adapter.finalize_metrics().unwrap();
    let current = adapter.live_state().unwrap();
    assert!(!Arc::ptr_eq(current.metrics().unwrap(), &old_metrics));
    assert_eq!(old_metrics.read().unwrap().stock_id, "M");
    assert_eq!(current.metrics().unwrap().read().unwrap().stock_id, "A");

    assert_eq!(
        serde_json::json!({
            "before_finalize_none": before_finalize.metrics().is_none(),
            "same_before_replacement": Arc::ptr_eq(first.metrics().unwrap(), second.metrics().unwrap()),
            "mutation_visible": second.snapshot().unwrap().parts().metrics.as_ref().unwrap().stock_id,
            "replaced_after_finalize": !Arc::ptr_eq(current.metrics().unwrap(), &old_metrics),
            "old_stock": old_metrics.read().unwrap().stock_id,
            "new_stock": current.metrics().unwrap().read().unwrap().stock_id,
        }),
        live_contract::source()["metric-aliases"]
    );

    poison_lock(Arc::clone(current.metrics().unwrap()));
    assert!(matches!(
        current.snapshot(),
        Err(SaoeAdapterError::MetricsPoisoned)
    ));
    assert!(matches!(
        domain_core::saoe_live_registry::LiveSaoeStateAdapter::state(&adapter),
        Err(error) if error.message.contains("metrics lock poisoned")
    ));
    adapter.finalize_metrics().unwrap();
    assert_eq!(
        adapter
            .live_state()
            .unwrap()
            .snapshot()
            .unwrap()
            .parts()
            .metrics
            .as_ref()
            .unwrap()
            .stock_id,
        "A"
    );
}

#[test]
fn live_backtest_object_shares_mutations_and_state_ticks_retain_rebound_children() {
    let context = Arc::new(Context::new());
    let market = Arc::new(Market::new([
        slice(&[100.0, 200.0], &[10.0, 12.0]),
        slice(&[300.0, 400.0], &[14.0, 16.0]),
    ]));
    let shared_order = Arc::new(RwLock::new(order(OrderDir::Buy)));
    let mut adapter = ConcreteSaoeStateAdapter::new(market, context, config()).unwrap();
    adapter.reset_shared_order(&shared_order).unwrap();
    adapter
        .update_executions(&[execution(0, 2.0), execution(1, 3.0)], (0, 1))
        .unwrap();

    let old_state = adapter.live_state().unwrap();
    let old_ticks = Arc::clone(old_state.ticks_index());
    let old_order_ticks = Arc::clone(old_state.ticks_for_order());
    let handles = old_state.backtest_data().read().unwrap().clone();
    *handles.deal_prices.write().unwrap() = arr1(&[99.0, 12.0]);
    let rebound_ticks = Arc::new(RwLock::new(
        (1..=5).map(time).collect::<Vec<NaiveDateTime>>(),
    ));
    let rebound_order_ticks = Arc::new(RwLock::new(vec![time(1), time(2)]));
    {
        let mut backtest_data = old_state.backtest_data().write().unwrap();
        backtest_data.ticks_index = Arc::clone(&rebound_ticks);
        backtest_data.ticks_for_order = Arc::clone(&rebound_order_ticks);
    }

    adapter
        .update_executions(&[execution(1, 1.0), execution(2, 1.0)], (1, 2))
        .unwrap();
    adapter.finalize_metrics().unwrap();
    let current = adapter.live_state().unwrap();
    let old_snapshot = old_state.snapshot().unwrap();
    let current_snapshot = current.snapshot().unwrap();
    let SaoeNumeric::Scalar(_) = current_snapshot
        .parts()
        .metrics
        .as_ref()
        .unwrap()
        .market_volume
    else {
        panic!("expected scalar final metric")
    };
    let domain_core::SaoeTime::Scalar(metric_time) =
        current_snapshot.parts().metrics.as_ref().unwrap().datetime
    else {
        panic!("expected scalar final timestamp")
    };

    assert_eq!(old_snapshot.parts().ticks_index[0], time(0));
    assert_eq!(old_snapshot.parts().ticks_for_order.len(), 6);
    assert_eq!(old_snapshot.parts().backtest_data.ticks_index[0], time(1));
    assert_eq!(old_snapshot.parts().backtest_data.ticks_for_order.len(), 2);
    assert_eq!(current_snapshot.parts().ticks_index[0], time(1));
    assert_eq!(current_snapshot.parts().ticks_for_order.len(), 2);
    assert_eq!(
        current_snapshot.parts().backtest_data.deal_prices[0].to_bits(),
        99.0_f64.to_bits()
    );

    assert_eq!(
        serde_json::json!({
            "same_backtest_object": Arc::ptr_eq(old_state.backtest_data(), current.backtest_data()),
            "old_ticks_retained": Arc::ptr_eq(old_state.ticks_index(), &old_ticks)
                && old_ticks.read().unwrap()[0] == time(0),
            "old_order_ticks_retained": Arc::ptr_eq(old_state.ticks_for_order(), &old_order_ticks)
                && old_order_ticks.read().unwrap().len() == 6,
            "current_ticks_rebound": Arc::ptr_eq(current.ticks_index(), &rebound_ticks)
                && current.ticks_index().read().unwrap()[0] == time(1),
            "current_order_ticks_rebound": Arc::ptr_eq(current.ticks_for_order(), &rebound_order_ticks)
                && current.ticks_for_order().read().unwrap().len() == 2,
            "deal_mutation_visible": current_snapshot.parts().backtest_data.deal_prices[0],
            "adapter_time": current_snapshot.parts().cur_time.to_string(),
            "metric_time": metric_time.to_string(),
        }),
        live_contract::source()["backtest-aliases"]
    );
}

#[test]
fn live_state_and_adapter_report_each_alias_poison_boundary() {
    let (mut adapter, reset_order) = populated_alias_adapter();
    let state = adapter.live_state().unwrap();
    poison_lock(Arc::clone(state.history_exec()));
    assert!(matches!(
        state.snapshot(),
        Err(SaoeAdapterError::HistoryExecPoisoned)
    ));
    assert!(matches!(
        adapter.history_exec(),
        Err(SaoeAdapterError::HistoryExecPoisoned)
    ));
    assert!(matches!(
        adapter.update_executions(&[execution(2, 1.0)], (2, 3)),
        Err(SaoeAdapterError::HistoryExecPoisoned)
    ));
    adapter.reset_shared_order(&reset_order).unwrap();
    assert!(adapter.history_exec().unwrap().is_empty());

    let (mut adapter, _) = populated_alias_adapter();
    let state = adapter.live_state().unwrap();
    poison_lock(Arc::clone(state.history_exec()));
    assert!(matches!(
        adapter.finalize_metrics(),
        Err(SaoeAdapterError::HistoryExecPoisoned)
    ));

    let (mut adapter, _) = populated_alias_adapter();
    let state = adapter.live_state().unwrap();
    poison_lock(Arc::clone(state.history_steps()));
    assert!(matches!(
        state.snapshot(),
        Err(SaoeAdapterError::HistoryStepsPoisoned)
    ));
    assert!(matches!(
        adapter.history_steps(),
        Err(SaoeAdapterError::HistoryStepsPoisoned)
    ));
    assert!(matches!(
        adapter.update_executions(&[execution(2, 1.0)], (2, 3)),
        Err(SaoeAdapterError::HistoryStepsPoisoned)
    ));
    assert!(matches!(
        adapter.finalize_metrics(),
        Err(SaoeAdapterError::HistoryStepsPoisoned)
    ));

    let (adapter, poisoned_order) = populated_alias_adapter();
    let state = adapter.live_state().unwrap();
    poison_lock(poisoned_order);
    assert!(matches!(
        state.snapshot(),
        Err(SaoeAdapterError::AdapterOrderPoisoned)
    ));
    assert!(adapter.live_state().is_ok());

    let context = Arc::new(Context::new());
    let market = Arc::new(Market::new([]));
    let order = Arc::new(RwLock::new(order(OrderDir::Buy)));
    let mut adapter = ConcreteSaoeStateAdapter::new(market, context.clone(), config()).unwrap();
    adapter.reset_shared_order(&order).unwrap();
    context.0.lock().unwrap().fail = Some("step");
    assert!(matches!(
        domain_core::saoe_live_registry::LiveSaoeStateAdapter::live_state(&adapter),
        Err(error) if error.message.contains("step")
    ));
}

#[test]
fn live_backtest_aliases_report_object_child_and_rebound_tick_failures() {
    let (mut adapter, _) = populated_alias_adapter();
    let state = adapter.live_state().unwrap();
    poison_lock(Arc::clone(state.backtest_data()));
    assert!(matches!(
        state.snapshot(),
        Err(SaoeAdapterError::BacktestDataPoisoned)
    ));
    assert!(matches!(
        adapter.live_state(),
        Err(SaoeAdapterError::BacktestDataPoisoned)
    ));
    assert!(matches!(
        adapter.update_executions(&[execution(2, 1.0)], (2, 3)),
        Err(SaoeAdapterError::BacktestDataPoisoned)
    ));

    let (mut adapter, reset_order) = populated_alias_adapter();
    let state = adapter.live_state().unwrap();
    poison_lock(Arc::clone(state.backtest_data()));
    assert!(matches!(
        adapter.reset_shared_order(&reset_order),
        Err(SaoeAdapterError::BacktestDataPoisoned)
    ));
    let (mut adapter, reset_order) = populated_alias_adapter();
    let state = adapter.live_state().unwrap();
    let handles = state.backtest_data().read().unwrap().clone();
    poison_lock(handles.ticks_for_order);
    assert!(matches!(
        adapter.reset_shared_order(&reset_order),
        Err(SaoeAdapterError::BacktestDataFieldPoisoned(
            "ticks_for_order"
        ))
    ));

    let (mut adapter, _) = populated_alias_adapter();
    let state = adapter.live_state().unwrap();
    let handles = state.backtest_data().read().unwrap().clone();
    poison_lock(handles.ticks_index);
    assert!(matches!(
        adapter.update_executions(&[execution(2, 1.0)], (2, 3)),
        Err(SaoeAdapterError::BacktestDataFieldPoisoned("ticks_index"))
    ));

    let (mut adapter, _) = populated_alias_adapter();
    let state = adapter.live_state().unwrap();
    poison_lock(Arc::clone(state.backtest_data()));
    assert!(matches!(
        adapter.finalize_metrics(),
        Err(SaoeAdapterError::BacktestDataPoisoned)
    ));
    let (mut adapter, _) = populated_alias_adapter();
    let state = adapter.live_state().unwrap();
    let handles = state.backtest_data().read().unwrap().clone();
    poison_lock(handles.ticks_index);
    assert!(matches!(
        adapter.finalize_metrics(),
        Err(SaoeAdapterError::BacktestDataFieldPoisoned("ticks_index"))
    ));
}

#[test]
fn live_backtest_snapshot_reports_child_rebind_and_empty_tick_failures() {
    let backtest_field = |name| {
        let (adapter, _) = populated_alias_adapter();
        let state = adapter.live_state().unwrap();
        let handles = state.backtest_data().read().unwrap().clone();
        (state, handles, name)
    };
    let (state, handles, name) = backtest_field("ticks_index");
    assert_backtest_field_poison(&state, handles.ticks_index, name);
    let (state, handles, name) = backtest_field("ticks_for_order");
    assert_backtest_field_poison(&state, handles.ticks_for_order, name);
    let (state, handles, name) = backtest_field("deal_prices");
    assert_backtest_field_poison(&state, handles.deal_prices, name);
    let (state, handles, name) = backtest_field("market_volumes");
    assert_backtest_field_poison(&state, handles.market_volumes, name);
    let (state, handles, name) = backtest_field("features");
    assert_backtest_field_poison(&state, handles.features, name);

    let (adapter, _) = populated_alias_adapter();
    let state = adapter.live_state().unwrap();
    state.backtest_data().write().unwrap().ticks_index = Arc::new(RwLock::new(vec![time(0)]));
    assert_backtest_field_poison(&state, Arc::clone(state.ticks_index()), "state.ticks_index");
    let (adapter, _) = populated_alias_adapter();
    let state = adapter.live_state().unwrap();
    state.backtest_data().write().unwrap().ticks_for_order = Arc::new(RwLock::new(vec![time(0)]));
    assert_backtest_field_poison(
        &state,
        Arc::clone(state.ticks_for_order()),
        "state.ticks_for_order",
    );

    let (mut adapter, reset_order) = populated_alias_adapter();
    let state = adapter.live_state().unwrap();
    let handles = state.backtest_data().read().unwrap().clone();
    handles.ticks_index.write().unwrap().clear();
    assert!(matches!(
        adapter.finalize_metrics(),
        Err(SaoeAdapterError::EmptyTicks)
    ));
    handles.ticks_for_order.write().unwrap().clear();
    assert!(matches!(
        adapter.reset_shared_order(&reset_order),
        Err(SaoeAdapterError::EmptyOrderTicks)
    ));
}

#[test]
fn live_pipeline_retains_aliases_across_state_policy_and_action_callbacks() {
    let context = Arc::new(Context::new());
    let market = Arc::new(Market::new([slice(&[100.0, 200.0], &[10.0, 12.0])]));
    let shared_order = Arc::new(RwLock::new(order(OrderDir::Buy)));
    let mut adapter = ConcreteSaoeStateAdapter::new(market, context, config()).unwrap();
    adapter.reset_shared_order(&shared_order).unwrap();
    adapter
        .update_executions(&[execution(0, 2.0), execution(1, 3.0)], (0, 1))
        .unwrap();
    let state = adapter.live_state().unwrap();
    let mut pipeline = SaoePolicyPipeline::new(
        Box::new(AliasStateInterpreter),
        Box::new(AllOnePolicy::default()),
        Box::new(AliasActionInterpreter),
    );
    let mut emitted = false;
    let decisions = pipeline
        .decisions_from_live(|| {
            if emitted {
                Ok::<_, SaoeInterpreterError>(None)
            } else {
                emitted = true;
                Ok(Some(state.clone()))
            }
        })
        .unwrap();
    assert_eq!(decisions.len(), 1);
    assert_eq!(decisions[0].execution_volume.to_bits(), 3.0_f64.to_bits());
    assert_eq!(adapter.history_exec().unwrap()[0].stock_id, "pipeline");
}

#[test]
fn live_pipeline_default_materialization_is_fresh_and_reports_alias_failures() {
    let context = Arc::new(Context::new());
    context.0.lock().unwrap().step = 1;
    let market = Arc::new(Market::new([slice(&[100.0, 200.0], &[10.0, 12.0])]));
    let shared_order = Arc::new(RwLock::new(order(OrderDir::Buy)));
    let mut adapter = ConcreteSaoeStateAdapter::new(market, context, config()).unwrap();
    adapter.reset_shared_order(&shared_order).unwrap();
    adapter
        .update_executions(&[execution(0, 2.0), execution(1, 3.0)], (0, 1))
        .unwrap();
    let state = adapter.live_state().unwrap();

    let mut default_pipeline = SaoePolicyPipeline::new(
        Box::new(DummyStateInterpreter),
        Box::new(AllOnePolicy::default()),
        Box::new(TwapRelativeActionInterpreter),
    );
    let mut emitted = false;
    let decisions = default_pipeline
        .decisions_from_live(|| {
            if std::mem::replace(&mut emitted, true) {
                Ok::<_, SaoeInterpreterError>(None)
            } else {
                Ok(Some(state.clone()))
            }
        })
        .unwrap();
    assert_eq!(decisions[0].execution_volume.to_bits(), 2.5_f64.to_bits());

    let mut policy_failure = SaoePolicyPipeline::new(
        Box::new(AliasStateInterpreter),
        Box::new(FailingPolicy),
        Box::new(AliasActionInterpreter),
    );
    let mut emitted = false;
    assert!(matches!(
        policy_failure.decisions_from_live(|| {
            if std::mem::replace(&mut emitted, true) {
                Ok::<_, SaoeInterpreterError>(None)
            } else {
                Ok(Some(state.clone()))
            }
        }),
        Err(SaoeInterpreterError::ProcessedDataPlugin(message)) if message == "live policy failed"
    ));

    let mut action_failure = SaoePolicyPipeline::new(
        Box::new(AliasStateInterpreter),
        Box::new(PoisonPolicy(Arc::clone(state.history_exec()))),
        Box::new(TwapRelativeActionInterpreter),
    );
    let mut emitted = false;
    assert!(matches!(
        action_failure.decisions_from_live(|| {
            if std::mem::replace(&mut emitted, true) {
                Ok::<_, SaoeInterpreterError>(None)
            } else {
                Ok(Some(state.clone()))
            }
        }),
        Err(SaoeInterpreterError::LiveState(_))
    ));

    let context = Arc::new(Context::new());
    let market = Arc::new(Market::new([]));
    let poisoned_order = Arc::new(RwLock::new(order(OrderDir::Buy)));
    let mut adapter = ConcreteSaoeStateAdapter::new(market, context, config()).unwrap();
    adapter.reset_shared_order(&poisoned_order).unwrap();
    let poisoned = adapter.live_state().unwrap();
    poison_lock(poisoned_order);
    let mut state_failure = SaoePolicyPipeline::new(
        Box::new(DummyStateInterpreter),
        Box::new(AllOnePolicy::default()),
        Box::new(TwapRelativeActionInterpreter),
    );
    assert!(matches!(
        state_failure.decisions_from_live(|| Ok::<_, SaoeInterpreterError>(Some(poisoned.clone()))),
        Err(SaoeInterpreterError::LiveState(_))
    ));
}

fn assert_row(row: &SaoeMetricRow, expected: &Value) {
    assert_eq!(row.stock_id, expected["stock_id"]);
    assert_eq!(
        row.direction as i8,
        i8::try_from(expected["direction"].as_i64().unwrap()).unwrap()
    );
    for (actual, key) in [
        (row.market_volume, "market_volume"),
        (row.market_price, "market_price"),
        (row.amount, "amount"),
        (row.inner_amount, "inner_amount"),
        (row.deal_amount, "deal_amount"),
        (row.trade_price, "trade_price"),
        (row.trade_value, "trade_value"),
        (row.position, "position"),
        (row.ffr, "ffr"),
        (row.pa, "pa"),
    ] {
        close(actual, expected[key].as_f64().unwrap());
    }
    assert_eq!(row.datetime.to_string(), expected["datetime"]);
}

fn scalar(value: &SaoeNumeric) -> f64 {
    let SaoeNumeric::Scalar(value) = value else {
        panic!("expected scalar metric")
    };
    *value
}

#[test]
fn execution_reads_live_order_and_poison_stops_before_market_or_state_mutation() {
    let market = Arc::new(Market::new([slice(&[100.0], &[10.0])]));
    let mut adapter =
        ConcreteSaoeStateAdapter::new(market.clone(), Arc::new(Context::new()), config()).unwrap();
    adapter.reset_order(&order(OrderDir::Buy)).unwrap();
    let row = execution(0, 1.0);
    let original = Arc::clone(&row.order);
    original.write().unwrap().set_deal_amount(3.0);
    adapter
        .update_executions(&[Arc::clone(&row)], (0, 0))
        .unwrap();
    close(adapter.state_snapshot().unwrap().parts().position, 7.0);
    let before = adapter.state_snapshot().unwrap().encode().unwrap();
    assert!(
        std::panic::catch_unwind(|| {
            let _guard = original.write().unwrap();
            panic!("poison original execution order");
        })
        .is_err()
    );
    assert!(matches!(
        adapter.update_executions(&[row], (0, 0)),
        Err(SaoeAdapterError::ExecutionOrderPoisoned),
    ));
    assert_eq!(adapter.state_snapshot().unwrap().encode().unwrap(), before);
    assert!(market.responses.lock().unwrap().is_empty());
}

#[test]
fn shared_provider_reads_live_updates_final_metrics_and_runtime_without_rewriting_snapshots() {
    let context = Arc::new(Context::new());
    let adapter = ConcreteSaoeStateAdapter::new(
        Arc::new(Market::new([slice(&[100.0, 120.0], &[10.0, 12.0])])),
        context.clone(),
        config(),
    )
    .unwrap();
    let shared: Arc<Mutex<dyn SaoeStateProvider>> = Arc::new(Mutex::new(adapter));
    let mut writer = Arc::clone(&shared);
    let reader = Arc::clone(&shared);
    let seed = order(OrderDir::Buy);
    writer.reset(&seed).unwrap();
    let initial = reader.state(&seed).unwrap();
    let initial_bytes = initial.encode().unwrap();
    assert!(initial.parts().metrics.is_none());
    close(initial.parts().position, 10.0);
    writer
        .update(&[execution(0, 2.0), execution(1, 3.0)], (0, 1))
        .unwrap();
    let updated = reader.state(&seed).unwrap();
    close(updated.parts().position, 5.0);
    assert_eq!(updated.parts().history_exec.num_rows(), 2);
    assert_eq!(updated.parts().history_steps.num_rows(), 1);
    assert!(updated.parts().metrics.is_none());
    writer.finalize().unwrap();
    context.0.lock().unwrap().step = 9;
    let finalized = reader.state(&seed).unwrap();
    assert_eq!(finalized.parts().cur_step, 9);
    assert!(finalized.parts().metrics.is_some());
    assert!(updated.parts().metrics.is_none());
    assert_eq!(initial.encode().unwrap(), initial_bytes);
    assert_eq!(
        finalized.encode().unwrap(),
        shared
            .lock()
            .unwrap()
            .state(&seed)
            .unwrap()
            .encode()
            .unwrap()
    );
    drop(writer);
    assert!(reader.state(&seed).unwrap().parts().metrics.is_some());
    drop(reader);
    assert_eq!(Arc::strong_count(&shared), 1);
    let weak = Arc::downgrade(&shared);
    drop(shared);
    assert!(weak.upgrade().is_none());
}

#[test]
fn shared_provider_preserves_native_failures_and_rejects_poison_for_every_operation() {
    let adapter = ConcreteSaoeStateAdapter::new(
        Arc::new(Market::failing()),
        Arc::new(Context::new()),
        config(),
    )
    .unwrap();
    let mut shared = Arc::new(Mutex::new(adapter));
    let seed = order(OrderDir::Buy);
    let expected = SaoeAdapterError::NotInitialized.to_string();
    assert_eq!(shared.state(&seed).unwrap_err().message, expected);
    assert_eq!(shared.update(&[], (0, 1)).unwrap_err().message, expected);
    assert_eq!(shared.finalize().unwrap_err().message, expected);
    let invalid = Order::new("A", 1.0, OrderDir::Buy, None, Some(time(1)));
    let expected_reset = shared.lock().unwrap().reset(&invalid).unwrap_err();
    assert_eq!(shared.reset(&invalid).unwrap_err(), expected_reset);
    shared.reset(&seed).unwrap();
    let before = shared.state(&seed).unwrap().encode().unwrap();
    let error = shared.update(&[execution(0, 1.0)], (0, 0)).unwrap_err();
    assert_eq!(
        error.message,
        SaoePluginError {
            message: "market".to_owned()
        }
        .to_string()
    );
    assert_eq!(shared.state(&seed).unwrap().encode().unwrap(), before);
    let poison = Arc::clone(&shared);
    assert!(
        std::thread::spawn(move || {
            let _guard = poison.lock().unwrap();
            panic!("poison shared provider for boundary test");
        })
        .join()
        .is_err()
    );
    let message = format!(
        "shared SAOE state provider lock poisoned: {}",
        shared.lock().err().unwrap()
    );
    assert_eq!(shared.reset(&seed).unwrap_err().message, message);
    assert_eq!(shared.state(&seed).unwrap_err().message, message);
    assert_eq!(shared.update(&[], (0, 0)).unwrap_err().message, message);
    assert_eq!(shared.finalize().unwrap_err().message, message);
    assert!(shared.is_poisoned());
}

#[test]
fn concrete_adapter_matches_live_python_updates_histories_and_final_metrics() {
    let market = Arc::new(Market::new([
        slice(&[100.0, f64::NAN], &[10.0, f64::NAN]),
        slice(&[120.0, 140.0], &[12.0, 14.0]),
        slice(&[160.0], &[16.0]),
    ]));
    let context = Arc::new(Context::new());
    let mut adapter = ConcreteSaoeStateAdapter::new(market, context.clone(), config()).unwrap();
    close(adapter.twap_price(), 12.5);
    adapter.reset_order(&order(OrderDir::Buy)).unwrap();
    adapter
        .update_executions(&[execution(0, 2.0), execution(1, 3.0)], (0, 1))
        .unwrap();
    close(adapter.position(), 5.0);
    assert_eq!(adapter.cur_time(), Some(time(2)));
    adapter
        .update_executions(&[execution(2, 4.0), execution(3, 4.0)], (2, 3))
        .unwrap();
    close(adapter.position(), 0.0);
    assert_eq!(context.0.lock().unwrap().warnings, 1);
    adapter.update_executions(&[], (4, 4)).unwrap();
    assert_eq!(adapter.cur_time(), Some(time(5)));
    adapter.finalize_metrics().unwrap();

    let python = python_contract();
    let expected = &python["snapshots"][3];
    for (row, expected) in adapter
        .history_exec()
        .unwrap()
        .iter()
        .zip(expected["history_exec"].as_array().unwrap())
    {
        assert_row(row, expected);
    }
    for (row, expected) in adapter
        .history_steps()
        .unwrap()
        .iter()
        .zip(expected["history_steps"].as_array().unwrap())
    {
        assert_row(row, expected);
    }
    assert_eq!(adapter.history_exec().unwrap().len(), 5);
    assert_eq!(adapter.history_steps().unwrap().len(), 3);

    let state = adapter.state_snapshot().unwrap();
    assert_eq!(state.parts().cur_step, 3);
    assert_eq!(state.parts().history_exec.num_rows(), 5);
    assert_eq!(state.parts().history_steps.num_rows(), 3);
    assert_eq!(
        state.parts().history_exec.schema().field(0).name(),
        "datetime"
    );
    assert_eq!(state.parts().history_exec.schema().field(12).name(), "pa");
    let stocks = state.parts().history_exec.column(1);
    assert_eq!(
        stocks
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0),
        "A"
    );
    let directions = state.parts().history_exec.column(2);
    assert_eq!(
        directions
            .as_any()
            .downcast_ref::<Int8Array>()
            .unwrap()
            .value(0),
        1
    );
    let amounts = state.parts().history_exec.column(5);
    close(
        amounts
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(3),
        2.5,
    );
    let metrics = state.parts().metrics.as_ref().unwrap();
    close(scalar(&metrics.market_volume), 620.0);
    close(scalar(&metrics.market_price), 12.4);
    close(scalar(&metrics.amount), 10.0);
    close(scalar(&metrics.trade_price), 11.5);
    close(scalar(&metrics.trade_value), 115.0);
    close(scalar(&metrics.position), -10.0);
    close(scalar(&metrics.ffr), 1.0);
    close(scalar(&metrics.pa), 800.0);
    let decoded = domain_core::SaoeState::decode(&state.encode().unwrap()).unwrap();
    assert_eq!(decoded.parts().history_exec, state.parts().history_exec);
}

#[test]
#[allow(clippy::too_many_lines)]
fn adapter_validates_configuration_order_ranges_shapes_and_times() {
    let market = Arc::new(Market::new([]));
    let context = Arc::new(Context::new());
    let mut empty_ticks = config();
    empty_ticks.backtest_data.ticks_index.clear();
    assert!(matches!(
        ConcreteSaoeStateAdapter::new(market.clone(), context.clone(), empty_ticks),
        Err(SaoeAdapterError::EmptyTicks)
    ));
    let mut empty_order_ticks = config();
    empty_order_ticks.backtest_data.ticks_for_order.clear();
    assert!(matches!(
        ConcreteSaoeStateAdapter::new(market.clone(), context.clone(), empty_order_ticks,),
        Err(SaoeAdapterError::EmptyOrderTicks)
    ));
    let mut zero = config();
    zero.data_granularity = 0;
    assert!(matches!(
        ConcreteSaoeStateAdapter::new(market.clone(), context.clone(), zero),
        Err(SaoeAdapterError::ZeroGranularity)
    ));
    let mut incompatible = config();
    incompatible.data_granularity = 3;
    assert!(matches!(
        ConcreteSaoeStateAdapter::new(market.clone(), context.clone(), incompatible,),
        Err(SaoeAdapterError::IncompatibleGranularity { .. })
    ));
    let mut huge = config();
    huge.ticks_per_step = usize::MAX;
    huge.data_granularity = usize::MAX;
    assert!(matches!(
        ConcreteSaoeStateAdapter::new(market.clone(), context.clone(), huge),
        Err(SaoeAdapterError::GranularityTooLarge)
    ));

    let market = Arc::new(Market::new([]));
    let mut adapter = ConcreteSaoeStateAdapter::new(market, context.clone(), config()).unwrap();
    assert_eq!(adapter.cur_time(), None);
    assert!(matches!(
        adapter.state_snapshot(),
        Err(SaoeAdapterError::NotInitialized)
    ));
    assert!(matches!(
        adapter.finalize_metrics(),
        Err(SaoeAdapterError::NotInitialized)
    ));
    assert!(matches!(
        adapter.update_executions(&[], (0, 1)),
        Err(SaoeAdapterError::NotInitialized)
    ));
    assert!(SaoeStateProvider::state(&adapter, &order(OrderDir::Buy)).is_err());
    assert!(SaoeStateProvider::update(&mut adapter, &[], (0, 1)).is_err());
    assert!(SaoeStateProvider::finalize(&mut adapter).is_err());
    assert!(matches!(
        adapter.reset_order(&Order::new("A", 1.0, OrderDir::Buy, None, Some(time(1)))),
        Err(SaoeAdapterError::Order(_))
    ));
    assert!(matches!(
        adapter.reset_order(&Order::new("A", 1.0, OrderDir::Buy, Some(time(0)), None)),
        Err(SaoeAdapterError::MissingEndTime)
    ));
    adapter.reset_order(&order(OrderDir::Buy)).unwrap();
    for range in [(-1, 0), (2, 1), (0, 6)] {
        assert!(matches!(
            adapter.update_executions(&[], range),
            Err(SaoeAdapterError::InvalidStepRange { .. })
        ));
    }

    let market = Arc::new(Market::new([
        slice(&[1.0, 2.0], &[1.0]),
        slice(&[1.0], &[1.0, 2.0]),
    ]));
    let mut adapter = ConcreteSaoeStateAdapter::new(market, context.clone(), config()).unwrap();
    adapter.reset_order(&order(OrderDir::Buy)).unwrap();
    assert!(matches!(
        adapter.update_executions(&[], (0, 1)),
        Err(SaoeAdapterError::VectorLength {
            name: "market price",
            ..
        })
    ));
    assert!(matches!(
        adapter.update_executions(&[], (0, 1)),
        Err(SaoeAdapterError::VectorLength {
            name: "market volume",
            ..
        })
    ));

    let gap_config = SaoeAdapterConfig {
        backtest_data: backtest(vec![time(0), time(2)], vec![time(0), time(2)]),
        deal_prices: arr1(&[1.0, 2.0]),
        ticks_per_step: 2,
        data_granularity: 1,
        start_step: 0,
    };
    let market = Arc::new(Market::new([slice(&[1.0, 2.0], &[1.0, 2.0])]));
    let mut adapter = ConcreteSaoeStateAdapter::new(market, context, gap_config).unwrap();
    adapter.reset_order(&order(OrderDir::Buy)).unwrap();
    assert!(matches!(
        adapter.update_executions(&[], (0, 1)),
        Err(SaoeAdapterError::VectorLength {
            name: "timestamp",
            ..
        })
    ));
}

#[test]
#[allow(clippy::too_many_lines)]
fn adapter_propagates_plugins_execution_indexing_and_partial_mutations() {
    let context = Arc::new(Context::new());
    let market = Arc::new(Market::failing());
    let mut adapter = ConcreteSaoeStateAdapter::new(market, context.clone(), config()).unwrap();
    adapter.reset_order(&order(OrderDir::Buy)).unwrap();
    assert!(matches!(
        adapter.update_executions(&[], (0, 1)),
        Err(SaoeAdapterError::Plugin(_))
    ));

    for stage in ["warning", "pa"] {
        let context = Arc::new(Context::new());
        context.0.lock().unwrap().fail = Some(stage);
        let market = Arc::new(Market::new([slice(&[1.0, 1.0], &[1.0, 1.0])]));
        let mut adapter = ConcreteSaoeStateAdapter::new(market, context, config()).unwrap();
        adapter.reset_order(&order(OrderDir::Buy)).unwrap();
        let executions = if stage == "warning" {
            vec![execution(0, 20.0)]
        } else {
            Vec::new()
        };
        assert!(matches!(
            adapter.update_executions(&executions, (0, 1)),
            Err(SaoeAdapterError::Plugin(_))
        ));
        assert!(adapter.history_exec().unwrap().is_empty());
    }

    let market = Arc::new(Market::new([slice(&[1.0, 1.0], &[1.0, 1.0])]));
    let mut adapter = ConcreteSaoeStateAdapter::new(market, context.clone(), config()).unwrap();
    adapter.reset_order(&order(OrderDir::Buy)).unwrap();
    assert!(matches!(
        adapter.update_executions(&[execution(2, 1.0)], (0, 1)),
        Err(SaoeAdapterError::ExecutionOutsideStep { .. })
    ));
    assert!(matches!(
        adapter.update_executions(&[execution(0, 1.0)], (1, 1)),
        Err(SaoeAdapterError::ExecutionOutsideStep { .. })
    ));
    let missing_start: SharedOrderExecution = OwnedOrderExecution {
        order: Order::new("A", 1.0, OrderDir::Buy, None, Some(time(0))),
        trade_value: 0.0,
        trade_cost: 0.0,
        trade_price: 0.0,
    }
    .into_shared();
    assert!(matches!(
        adapter.update_executions(&[missing_start], (0, 1)),
        Err(SaoeAdapterError::Order(_))
    ));
    let missing_end: SharedOrderExecution = OwnedOrderExecution {
        order: Order::new("A", 1.0, OrderDir::Buy, Some(time(0)), None),
        trade_value: 0.0,
        trade_cost: 0.0,
        trade_price: 0.0,
    }
    .into_shared();
    assert!(matches!(
        adapter.update_executions(&[missing_end], (0, 1)),
        Err(SaoeAdapterError::MissingEndTime)
    ));
    let outside: SharedOrderExecution = OwnedOrderExecution {
        order: Order::new("A", 1.0, OrderDir::Buy, Some(time(-90)), Some(time(-90))),
        trade_value: 0.0,
        trade_cost: 0.0,
        trade_price: 0.0,
    }
    .into_shared();
    adapter.update_executions(&[outside], (0, 1)).unwrap();
    assert_eq!(adapter.history_exec().unwrap().len(), 2);

    let context = Arc::new(Context::new());
    context.0.lock().unwrap().fail = Some("step");
    let market = Arc::new(Market::new([]));
    let mut adapter = ConcreteSaoeStateAdapter::new(market, context.clone(), config()).unwrap();
    adapter.reset_order(&order(OrderDir::Buy)).unwrap();
    assert!(matches!(
        adapter.state_snapshot(),
        Err(SaoeAdapterError::Plugin(_))
    ));

    let context = Arc::new(Context::new());
    let shifted = Order::new(
        "A",
        10.0,
        OrderDir::Buy,
        Some(time(0) + TimeDelta::seconds(30)),
        Some(time(5)),
    );
    let market = Arc::new(Market::new([slice(&[1.0, 1.0], &[1.0, 1.0])]));
    let mut adapter = ConcreteSaoeStateAdapter::new(market, context.clone(), config()).unwrap();
    adapter.reset_order(&shifted).unwrap();
    assert!(matches!(
        adapter.update_executions(&[], (0, 1)),
        Err(SaoeAdapterError::MissingCurrentTime(_))
    ));
    assert_eq!(adapter.history_exec().unwrap().len(), 2);
    assert_eq!(adapter.history_steps().unwrap().len(), 1);

    let duplicate_config = SaoeAdapterConfig {
        backtest_data: backtest(vec![time(0), time(0), time(1)], vec![time(0)]),
        deal_prices: arr1(&[1.0]),
        ticks_per_step: 1,
        data_granularity: 1,
        start_step: 0,
    };
    let market = Arc::new(Market::new([slice(&[1.0], &[1.0])]));
    let mut adapter =
        ConcreteSaoeStateAdapter::new(market, context.clone(), duplicate_config).unwrap();
    adapter.reset_order(&order(OrderDir::Buy)).unwrap();
    assert!(matches!(
        adapter.update_executions(&[], (0, 0)),
        Err(SaoeAdapterError::MissingCurrentTime(_))
    ));

    let context = Arc::new(Context::new());
    let market = Arc::new(Market::new([slice(&[1.0], &[1.0])]));
    let mut adapter = ConcreteSaoeStateAdapter::new(market, context.clone(), config()).unwrap();
    adapter.reset_order(&order(OrderDir::Buy)).unwrap();
    adapter
        .update_executions(&[execution(0, 10.5)], (0, 0))
        .unwrap();
    close(adapter.position(), 0.0);
    assert_eq!(context.0.lock().unwrap().warnings, 0);
}

#[test]
#[allow(clippy::too_many_lines)]
fn scalar_metric_edges_and_provider_trait_preserve_python_semantics() {
    let python = python_contract();
    let context = Arc::new(Context::new());
    let market = Arc::new(Market::new([slice(&[1.0], &[15.0])]));
    let mut sell_config = config();
    sell_config.deal_prices = arr1(&[12.5]);
    let mut adapter = ConcreteSaoeStateAdapter::new(market, context.clone(), sell_config).unwrap();
    let mut sell = Order::new("A", 1.0, OrderDir::Sell, Some(time(0)), Some(time(1)));
    SaoeStateProvider::reset(&mut adapter, &sell).unwrap();
    sell.set_deal_amount(1.0);
    let sell_execution: SharedOrderExecution = OwnedOrderExecution {
        order: sell,
        trade_value: 0.0,
        trade_cost: 0.0,
        trade_price: 0.0,
    }
    .into_shared();
    SaoeStateProvider::update(&mut adapter, &[sell_execution], (0, 0)).unwrap();
    close(
        adapter.history_steps().unwrap()[0].pa,
        python["edges"]["sell"].as_f64().unwrap(),
    );
    SaoeStateProvider::finalize(&mut adapter).unwrap();
    let state = SaoeStateProvider::state(&adapter, &order(OrderDir::Sell)).unwrap();
    assert!(state.parts().metrics.is_some());
    let missing_start = Order::new("A", 1.0, OrderDir::Buy, None, Some(time(1)));
    assert!(
        SaoeStateProvider::reset(&mut adapter, &missing_start)
            .unwrap_err()
            .message
            .contains("start time")
    );

    let market = Arc::new(Market::new([slice(&[1.0], &[15.0])]));
    let mut zero_config = config();
    zero_config.deal_prices = arr1(&[0.0]);
    let mut adapter = ConcreteSaoeStateAdapter::new(market, context.clone(), zero_config).unwrap();
    adapter.reset_order(&order(OrderDir::Buy)).unwrap();
    adapter
        .update_executions(&[execution(0, 1.0)], (0, 0))
        .unwrap();
    close(adapter.history_steps().unwrap()[0].pa, 0.0);

    let market = Arc::new(Market::new([slice(&[f64::NAN], &[f64::NAN])]));
    let mut nan_config = config();
    nan_config.deal_prices = arr1(&[f64::NAN]);
    let mut adapter = ConcreteSaoeStateAdapter::new(market, context.clone(), nan_config).unwrap();
    adapter.reset_order(&order(OrderDir::Buy)).unwrap();
    adapter.update_executions(&[], (0, 0)).unwrap();
    assert!(adapter.history_exec().unwrap()[0].market_price.is_nan());
    assert!(adapter.history_steps().unwrap()[0].market_price.is_nan());
    close(adapter.history_steps().unwrap()[0].pa, 0.0);

    let market = Arc::new(Market::new([]));
    let mut adapter = ConcreteSaoeStateAdapter::new(market, context.clone(), config()).unwrap();
    adapter.reset_order(&order(OrderDir::Buy)).unwrap();
    adapter.finalize_metrics().unwrap();
    let state = adapter.state_snapshot().unwrap();
    let metrics = state.parts().metrics.as_ref().unwrap();
    close(scalar(&metrics.market_volume), 0.0);
    assert!(scalar(&metrics.market_price).is_nan());
    close(scalar(&metrics.position), 10.0);
    close(scalar(&metrics.pa), 10_000.0);

    for (direction, expected) in [(OrderDir::Buy, f64::MIN), (OrderDir::Sell, f64::MAX)] {
        let market = Arc::new(Market::new([slice(&[1.0], &[f64::INFINITY])]));
        let mut edge_config = config();
        edge_config.deal_prices = arr1(&[1.0]);
        let mut adapter =
            ConcreteSaoeStateAdapter::new(market, context.clone(), edge_config).unwrap();
        let edge_order = Order::new("A", 1.0, direction, Some(time(0)), Some(time(1)));
        adapter.reset_order(&edge_order).unwrap();
        let mut fill = edge_order;
        fill.set_deal_amount(1.0);
        let fill: SharedOrderExecution = OwnedOrderExecution {
            order: fill,
            trade_value: 0.0,
            trade_cost: 0.0,
            trade_price: 0.0,
        }
        .into_shared();
        adapter.update_executions(&[fill], (0, 0)).unwrap();
        assert_eq!(
            adapter.history_steps().unwrap()[0].pa.to_bits(),
            expected.to_bits()
        );
    }

    let far = NaiveDateTime::parse_from_str("3000-01-02 09:30:00", "%Y-%m-%d %H:%M:%S").unwrap();
    let far_end = far + TimeDelta::minutes(1);
    let far_config = SaoeAdapterConfig {
        backtest_data: backtest(vec![far], vec![far]),
        deal_prices: arr1(&[1.0]),
        ticks_per_step: 1,
        data_granularity: 1,
        start_step: 0,
    };
    let market = Arc::new(Market::new([slice(&[1.0], &[1.0])]));
    let mut adapter = ConcreteSaoeStateAdapter::new(market, context.clone(), far_config).unwrap();
    adapter
        .reset_order(&Order::new(
            "A",
            1.0,
            OrderDir::Buy,
            Some(far),
            Some(far_end),
        ))
        .unwrap();
    adapter.update_executions(&[], (0, 0)).unwrap();
    assert!(matches!(
        adapter.state_snapshot(),
        Err(SaoeAdapterError::Arrow(_))
    ));

    let market = Arc::new(Market::new([slice(&[1.0], &[1.0])]));
    let mut adapter = ConcreteSaoeStateAdapter::new(market, context, config()).unwrap();
    adapter
        .reset_order(&Order::new(
            "A",
            1.0,
            OrderDir::Buy,
            Some(far),
            Some(far_end),
        ))
        .unwrap();
    assert!(matches!(
        adapter.update_executions(&[], (0, 0)),
        Err(SaoeAdapterError::MissingCurrentTime(_))
    ));
    assert!(matches!(
        adapter.state_snapshot(),
        Err(SaoeAdapterError::Arrow(_))
    ));
}
