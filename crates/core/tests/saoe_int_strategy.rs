use std::{
    process::Command,
    sync::{Arc, Mutex},
};

use arrow_array::RecordBatch;
use arrow_schema::Schema;
use chrono::{NaiveDateTime, TimeDelta};
use domain_core::{
    DummyStateInterpreter, LiveSaoeState, LiveSaoeStateParts, NestedCalendar, NestedDecisionUpdate,
    NestedOuterDecision, NestedOuterDecisionError, NestedStrategy, Order, OrderDecision, OrderDir,
    OwnedOrderExecution, SaoeActionInterpreter, SaoeActionSpace, SaoeAdapterRegistry,
    SaoeBacktestData, SaoeCalendar, SaoeDecisionCalendar, SaoeIntDecisionBuilder,
    SaoeIntStateProvider, SaoeIntStrategy, SaoeIntStrategyError, SaoeInterpreterError,
    SaoeOrderFactory, SaoePluginError, SaoePolicy, SaoePolicyAction, SaoePolicyPipeline, SaoeState,
    SaoeStateAdapterFactory, SaoeStateParts, SaoeStateProvider, SharedOrderExecution,
};
use ndarray::arr1;
use serde_json::{Value, json};

#[path = "support/saoe_live_generation.rs"]
mod live_generation;

#[path = "support/saoe_live_registry.rs"]
mod live_registry;

fn time(minute: i64) -> NaiveDateTime {
    NaiveDateTime::parse_from_str("2024-01-02 09:30:00", "%Y-%m-%d %H:%M:%S").unwrap()
        + TimeDelta::minutes(minute)
}

fn state_for(order: &Order) -> SaoeState {
    let empty = RecordBatch::new_empty(Arc::new(Schema::empty()));
    SaoeState::new(SaoeStateParts {
        order: order.clone(),
        cur_time: time(0),
        cur_step: 0,
        position: order.amount(),
        history_exec: empty.clone(),
        history_steps: empty.clone(),
        metrics: None,
        backtest_data: SaoeBacktestData {
            ticks_index: Vec::new(),
            ticks_for_order: Vec::new(),
            deal_prices: arr1(&[]),
            market_volumes: arr1(&[]),
            features: empty,
        },
        ticks_per_step: 1,
        ticks_index: Vec::new(),
        ticks_for_order: Vec::new(),
    })
}

fn plugin_error(message: &str) -> SaoePluginError {
    SaoePluginError {
        message: message.to_owned(),
    }
}

struct States {
    events: Arc<Mutex<Vec<String>>>,
    fail_on: Option<String>,
}

impl SaoeIntStateProvider for States {
    fn reset(
        &mut self,
        _outer: &dyn domain_core::NestedOuterDecision,
    ) -> Result<(), SaoePluginError> {
        Ok(())
    }

    fn state(&self, order: &Order) -> Result<SaoeState, SaoePluginError> {
        self.events
            .lock()
            .unwrap()
            .push(format!("state:{}", order.stock_id()));
        if self.fail_on.as_deref() == Some(order.stock_id()) {
            return Err(plugin_error("state"));
        }
        Ok(state_for(order))
    }

    fn update(
        &mut self,
        _executions: &[domain_core::SharedOrderExecution],
        _step_range: (i64, i64),
    ) -> Result<(), SaoePluginError> {
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), SaoePluginError> {
        Ok(())
    }
}

struct Policy {
    actions: Vec<SaoePolicyAction>,
    fail: bool,
}

impl SaoePolicy for Policy {
    fn actions(
        &mut self,
        _observations: &[domain_core::SaoeObservation],
    ) -> Result<Vec<SaoePolicyAction>, SaoeInterpreterError> {
        if self.fail {
            return Err(SaoeInterpreterError::ProcessedDataPlugin(
                "policy".to_owned(),
            ));
        }
        Ok(self.actions.clone())
    }
}

#[derive(Default)]
struct DirectAction;

impl SaoeActionInterpreter for DirectAction {
    fn action_space(&self) -> SaoeActionSpace {
        SaoeActionSpace::NonNegativeContinuous
    }

    fn interpret(
        &self,
        _state: &SaoeState,
        action: SaoePolicyAction,
    ) -> Result<f64, SaoeInterpreterError> {
        match action {
            SaoePolicyAction::Continuous(value) => Ok(value),
            SaoePolicyAction::Discrete(_) => Err(SaoeInterpreterError::ExpectedContinuous),
        }
    }
}

struct Orders {
    events: Arc<Mutex<Vec<String>>>,
    fail_on: Option<String>,
}

impl SaoeOrderFactory for Orders {
    fn create(
        &mut self,
        stock_id: &str,
        amount: Option<f64>,
        direction: OrderDir,
    ) -> Result<Order, SaoePluginError> {
        self.events
            .lock()
            .unwrap()
            .push(format!("order:{stock_id}:{}:{direction}", amount.unwrap()));
        if self.fail_on.as_deref() == Some(stock_id) {
            return Err(plugin_error("order"));
        }
        Ok(Order::new(stock_id, amount.unwrap(), direction, None, None))
    }
}

struct Calendar {
    events: Arc<Mutex<Vec<String>>>,
    time_calls: Mutex<usize>,
    frequency_calls: Mutex<usize>,
    fail_time_on: Option<usize>,
    fail_frequency_on: Option<usize>,
}

impl SaoeCalendar for Calendar {
    fn available_step_range(&self) -> Result<(i64, i64), SaoePluginError> {
        Ok((0, 1))
    }

    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), SaoePluginError> {
        self.events.lock().unwrap().push("time".to_owned());
        let mut calls = self.time_calls.lock().unwrap();
        *calls += 1;
        if self.fail_time_on == Some(*calls) {
            return Err(plugin_error("time"));
        }
        Ok((time(0), time(1)))
    }
}

impl SaoeDecisionCalendar for Calendar {
    fn frequency(&self) -> Result<String, SaoePluginError> {
        self.events.lock().unwrap().push("freq".to_owned());
        let mut calls = self.frequency_calls.lock().unwrap();
        *calls += 1;
        if self.fail_frequency_on == Some(*calls) {
            return Err(plugin_error("frequency"));
        }
        Ok("1min".to_owned())
    }
}

fn builder(
    actions: Vec<SaoePolicyAction>,
    events: Arc<Mutex<Vec<String>>>,
    state_failure: Option<&str>,
    policy_failure: bool,
    order_failure: Option<&str>,
    fail_time_on: Option<usize>,
    fail_frequency_on: Option<usize>,
) -> SaoeIntDecisionBuilder {
    SaoeIntDecisionBuilder::new(
        Box::new(States {
            events: Arc::clone(&events),
            fail_on: state_failure.map(str::to_owned),
        }),
        SaoePolicyPipeline::new(
            Box::new(DummyStateInterpreter),
            Box::new(Policy {
                actions,
                fail: policy_failure,
            }),
            Box::new(DirectAction),
        ),
        Box::new(Orders {
            events: Arc::clone(&events),
            fail_on: order_failure.map(str::to_owned),
        }),
        Arc::new(Calendar {
            events,
            time_calls: Mutex::new(0),
            frequency_calls: Mutex::new(0),
            fail_time_on,
            fail_frequency_on,
        }),
    )
}

fn outer_orders() -> Vec<Order> {
    vec![
        Order::new("A", 10.0, OrderDir::Buy, Some(time(0)), Some(time(9))),
        Order::new("B", 20.0, OrderDir::Sell, Some(time(0)), Some(time(9))),
        Order::new("C", 30.0, OrderDir::Buy, Some(time(0)), Some(time(9))),
    ]
}

struct Outer {
    decision: TestDecision,
}

impl Outer {
    fn new(orders: Vec<Order>) -> Self {
        Self {
            decision: TestDecision { orders },
        }
    }
}

struct TestDecision {
    orders: Vec<Order>,
}

impl OrderDecision for TestDecision {
    fn orders(&self) -> &[Order] {
        &self.orders
    }

    fn orders_mut(&mut self) -> &mut [Order] {
        &mut self.orders
    }

    fn start_time(&self) -> NaiveDateTime {
        time(0)
    }

    fn end_time(&self) -> NaiveDateTime {
        time(9)
    }

    fn trade_range(&self) -> Option<&dyn domain_core::TradeRange> {
        None
    }
}

impl NestedOuterDecision for Outer {
    fn order_decision(&self) -> &dyn OrderDecision {
        &self.decision
    }

    fn order_decision_mut(&mut self) -> &mut dyn OrderDecision {
        &mut self.decision
    }

    fn update(
        &mut self,
        _calendar: &dyn NestedCalendar,
    ) -> Result<NestedDecisionUpdate, NestedOuterDecisionError> {
        Ok(NestedDecisionUpdate::Unchanged)
    }

    fn is_empty(&self) -> Result<bool, NestedOuterDecisionError> {
        Ok(self.decision.is_empty())
    }

    fn range_limit(
        &self,
        _calendar: &dyn NestedCalendar,
    ) -> Result<Option<(i64, i64)>, NestedOuterDecisionError> {
        Ok(None)
    }

    fn modify_inner_decision(
        &self,
        _decision: &mut dyn OrderDecision,
    ) -> Result<(), NestedOuterDecisionError> {
        Ok(())
    }
}

#[derive(Clone, Default)]
struct AdapterFailures {
    factory: Option<String>,
    state: Option<String>,
    update: Option<String>,
    finalize: Option<String>,
}

struct AdapterFactory {
    events: Arc<Mutex<Vec<String>>>,
    count: usize,
    failures: AdapterFailures,
}

impl SaoeStateAdapterFactory for AdapterFactory {
    fn create(
        &mut self,
        order: &Order,
        outer: &dyn NestedOuterDecision,
    ) -> Result<Box<dyn SaoeStateProvider>, SaoePluginError> {
        self.count += 1;
        self.events.lock().unwrap().push(format!(
            "factory:{}:{}",
            order.stock_id(),
            outer.order_decision().orders().len()
        ));
        if self.failures.factory.as_deref() == Some(order.stock_id()) {
            return Err(plugin_error("factory"));
        }
        Ok(Box::new(Adapter {
            label: format!("{}#{}", order.stock_id(), self.count),
            events: Arc::clone(&self.events),
            failures: self.failures.clone(),
        }))
    }
}

struct Adapter {
    label: String,
    events: Arc<Mutex<Vec<String>>>,
    failures: AdapterFailures,
}

impl SaoeStateProvider for Adapter {
    fn reset(&mut self, order: &Order) -> Result<(), SaoePluginError> {
        self.events.lock().unwrap().push(format!(
            "adapter-reset:{}:{}",
            self.label,
            order.stock_id()
        ));
        Ok(())
    }

    fn state(&self, order: &Order) -> Result<SaoeState, SaoePluginError> {
        self.events
            .lock()
            .unwrap()
            .push(format!("adapter-state:{}", self.label));
        if self.failures.state.as_deref() == Some(order.stock_id()) {
            return Err(plugin_error("adapter-state"));
        }
        Ok(state_for(order))
    }

    fn update(
        &mut self,
        executions: &[SharedOrderExecution],
        step_range: (i64, i64),
    ) -> Result<(), SaoePluginError> {
        self.events.lock().unwrap().push(format!(
            "adapter-update:{}:{}:{}-{}",
            self.label,
            executions.len(),
            step_range.0,
            step_range.1
        ));
        if self.failures.update.as_deref() == Some(self.label.split('#').next().unwrap()) {
            return Err(plugin_error("adapter-update"));
        }
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), SaoePluginError> {
        self.events
            .lock()
            .unwrap()
            .push(format!("adapter-finalize:{}", self.label));
        if self.failures.finalize.as_deref() == Some(self.label.split('#').next().unwrap()) {
            return Err(plugin_error("adapter-finalize"));
        }
        Ok(())
    }
}

fn registry(events: Arc<Mutex<Vec<String>>>, failures: AdapterFailures) -> SaoeAdapterRegistry {
    SaoeAdapterRegistry::new(Box::new(AdapterFactory {
        events,
        count: 0,
        failures,
    }))
}

fn execution(order: Order) -> SharedOrderExecution {
    OwnedOrderExecution {
        order,
        trade_value: 0.0,
        trade_cost: 0.0,
        trade_price: 0.0,
    }
    .into_shared()
}

struct LifecycleCalendar {
    events: Arc<Mutex<Vec<String>>>,
    range: (i64, i64),
    fail_range: bool,
}

impl SaoeCalendar for LifecycleCalendar {
    fn available_step_range(&self) -> Result<(i64, i64), SaoePluginError> {
        self.events.lock().unwrap().push("range".to_owned());
        if self.fail_range {
            return Err(plugin_error("range"));
        }
        Ok(self.range)
    }

    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), SaoePluginError> {
        self.events.lock().unwrap().push("time".to_owned());
        Ok((time(0), time(1)))
    }
}

impl SaoeDecisionCalendar for LifecycleCalendar {
    fn frequency(&self) -> Result<String, SaoePluginError> {
        self.events.lock().unwrap().push("freq".to_owned());
        Ok("1min".to_owned())
    }
}

fn lifecycle_strategy(
    actions: Vec<SaoePolicyAction>,
    events: Arc<Mutex<Vec<String>>>,
    range: (i64, i64),
    fail_range: bool,
    failures: AdapterFailures,
) -> SaoeIntStrategy {
    SaoeIntStrategy::new(SaoeIntDecisionBuilder::new(
        Box::new(registry(Arc::clone(&events), failures)),
        SaoePolicyPipeline::new(
            Box::new(DummyStateInterpreter),
            Box::new(Policy {
                actions,
                fail: false,
            }),
            Box::new(DirectAction),
        ),
        Box::new(Orders {
            events: Arc::clone(&events),
            fail_on: None,
        }),
        Arc::new(LifecycleCalendar {
            events,
            range,
            fail_range,
        }),
    ))
}

fn python_contract() -> Value {
    let output = Command::new("python")
        .args([
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/saoe_int_strategy_contract.py"
            ),
            r"D:\code\github\qlib\qlib\rl\order_execution\strategy.py",
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

#[test]
#[allow(clippy::float_cmp)]
fn decision_builder_matches_python_order_filtering_details_and_call_order() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut builder = builder(
        vec![
            SaoePolicyAction::Continuous(0.0),
            SaoePolicyAction::Continuous(2.0),
            SaoePolicyAction::Continuous(f64::NAN),
        ],
        Arc::clone(&events),
        None,
        false,
        None,
        None,
        None,
    );
    let mut decision = builder.generate(&outer_orders()).unwrap();
    assert_eq!(decision.core().orders().len(), 2);
    assert_eq!(decision.core().orders()[0].stock_id(), "B");
    assert!(decision.core().orders()[1].amount().is_nan());
    assert_eq!(decision.details().len(), 3);
    assert_eq!(decision.details()[0].instrument, "A");
    assert_eq!(decision.details()[0].execution_volume, 0.0);
    assert_eq!(
        decision.details()[1].action,
        Some(SaoePolicyAction::Continuous(2.0))
    );
    assert!(decision.details()[2].execution_volume.is_nan());
    assert_eq!(decision.details()[2].frequency, "1min");
    assert_eq!(decision.details()[2].datetime, time(0));
    assert_eq!(OrderDecision::start_time(&decision), time(0));
    assert_eq!(OrderDecision::end_time(&decision), time(1));
    assert!(OrderDecision::trade_range(&decision).is_none());
    assert_eq!(OrderDecision::orders(&decision).len(), 2);
    OrderDecision::orders_mut(&mut decision)[0].set_deal_amount(1.0);
    assert_eq!(decision.core_mut().orders()[0].deal_amount(), 1.0);
    assert_eq!(
        events.lock().unwrap().as_slice(),
        [
            "state:A",
            "state:B",
            "state:C",
            "order:B:2:sell",
            "order:C:NaN:buy",
            "time",
            "freq",
            "time",
            "freq",
            "time",
            "freq",
            "time",
        ]
    );

    let python = python_contract();
    assert_eq!(python["orders"], json!([["B", 2.0, 0]]));
    assert_eq!(
        python["events"],
        json!([
            "order:B:2:0",
            "time",
            "freq",
            "time",
            "freq",
            "time",
            "update:A:1:1-3",
            "update:B:1:1-3",
            "finalize:A",
            "finalize:B"
        ])
    );
    assert_eq!(python["details"][0]["rl_exec_vol"], 0.0);
    assert_eq!(python["details"][1]["rl_action"], 2.0);
}

#[test]
fn empty_batch_still_captures_the_decision_interval() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut empty = builder(
        Vec::new(),
        Arc::clone(&events),
        None,
        false,
        None,
        None,
        None,
    );
    let decision = empty.generate(&[]).unwrap();
    assert!(decision.orders().is_empty());
    assert!(decision.details().is_empty());
    assert_eq!(events.lock().unwrap().as_slice(), ["time"]);
}

#[test]
fn every_plugin_failure_stops_at_the_owning_boundary() {
    let cases = [
        (
            builder(
                vec![SaoePolicyAction::Continuous(1.0); 3],
                Arc::new(Mutex::new(Vec::new())),
                Some("B"),
                false,
                None,
                None,
                None,
            ),
            SaoeIntStrategyError::State(plugin_error("state")),
        ),
        (
            builder(
                vec![],
                Arc::new(Mutex::new(Vec::new())),
                None,
                true,
                None,
                None,
                None,
            ),
            SaoeIntStrategyError::Pipeline(SaoeInterpreterError::ProcessedDataPlugin(
                "policy".to_owned(),
            )),
        ),
        (
            builder(
                vec![SaoePolicyAction::Discrete(1); 3],
                Arc::new(Mutex::new(Vec::new())),
                None,
                false,
                None,
                None,
                None,
            ),
            SaoeIntStrategyError::Pipeline(SaoeInterpreterError::ExpectedContinuous),
        ),
        (
            builder(
                vec![SaoePolicyAction::Continuous(1.0); 3],
                Arc::new(Mutex::new(Vec::new())),
                None,
                false,
                Some("B"),
                None,
                None,
            ),
            SaoeIntStrategyError::OrderFactory(plugin_error("order")),
        ),
        (
            builder(
                vec![SaoePolicyAction::Continuous(0.0); 3],
                Arc::new(Mutex::new(Vec::new())),
                None,
                false,
                None,
                Some(2),
                None,
            ),
            SaoeIntStrategyError::Calendar(plugin_error("time")),
        ),
        (
            builder(
                vec![SaoePolicyAction::Continuous(0.0); 3],
                Arc::new(Mutex::new(Vec::new())),
                None,
                false,
                None,
                None,
                Some(1),
            ),
            SaoeIntStrategyError::Calendar(plugin_error("frequency")),
        ),
        (
            builder(
                vec![SaoePolicyAction::Continuous(0.0); 3],
                Arc::new(Mutex::new(Vec::new())),
                None,
                false,
                None,
                Some(4),
                None,
            ),
            SaoeIntStrategyError::Calendar(plugin_error("time")),
        ),
    ];
    for (mut builder, expected) in cases {
        assert_eq!(builder.generate(&outer_orders()).err().unwrap(), expected);
    }
}

#[test]
fn poisoned_execution_order_stops_grouping_before_adapter_callbacks() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut registry = registry(Arc::clone(&events), AdapterFailures::default());
    let orders = outer_orders();
    registry.reset(&Outer::new(orders.clone())).unwrap();
    let row = execution(orders[0].clone());
    let before = events.lock().unwrap().clone();
    assert!(
        std::panic::catch_unwind(|| {
            let _guard = row.order.write().unwrap();
            panic!("poison execution order");
        })
        .is_err()
    );
    assert_eq!(
        registry.update(&[row], (1, 3)).unwrap_err().message,
        "SAOE execution order lock poisoned"
    );
    assert_eq!(*events.lock().unwrap(), before);
}

#[test]
fn ordered_adapter_registry_overwrites_duplicate_keys_groups_then_updates_and_finalizes() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut registry = registry(Arc::clone(&events), AdapterFailures::default());
    assert!(registry.is_empty());
    assert_eq!(registry.len(), 0);

    let orders = vec![
        Order::new("A", 10.0, OrderDir::Buy, Some(time(0)), Some(time(9))),
        Order::new("B", 20.0, OrderDir::Sell, Some(time(0)), Some(time(9))),
        Order::new("A", 30.0, OrderDir::Buy, Some(time(0)), Some(time(9))),
    ];
    let outer = Outer::new(orders.clone());
    registry.reset(&outer).unwrap();
    assert!(!registry.is_empty());
    assert_eq!(registry.len(), 2);
    assert_eq!(
        registry
            .state(&orders[0])
            .unwrap()
            .parts()
            .position
            .to_bits(),
        10.0_f64.to_bits()
    );

    registry
        .update(
            &[
                execution(orders[1].clone()),
                execution(Order::new(
                    "X",
                    1.0,
                    OrderDir::Buy,
                    Some(time(0)),
                    Some(time(1)),
                )),
                execution(orders[0].clone()),
            ],
            (1, 3),
        )
        .unwrap();
    registry.finalize().unwrap();
    assert_eq!(
        events.lock().unwrap().as_slice(),
        [
            "factory:A:3",
            "factory:B:3",
            "factory:A:3",
            "adapter-state:A#3",
            "adapter-update:A#3:1:1-3",
            "adapter-update:B#2:1:1-3",
            "adapter-finalize:A#3",
            "adapter-finalize:B#2",
        ]
    );

    let missing = Order::new("MISSING", 1.0, OrderDir::Buy, Some(time(0)), Some(time(1)));
    assert!(
        registry
            .state(&missing)
            .unwrap_err()
            .message
            .contains("missing")
    );
}

#[test]
fn adapter_registry_preserves_partial_state_at_every_failure_boundary() {
    let orders = outer_orders();
    let outer = Outer::new(orders.clone());

    let events = Arc::new(Mutex::new(Vec::new()));
    let mut factory_failure = registry(
        Arc::clone(&events),
        AdapterFailures {
            factory: Some("B".to_owned()),
            ..AdapterFailures::default()
        },
    );
    assert_eq!(factory_failure.reset(&outer), Err(plugin_error("factory")));
    assert_eq!(factory_failure.len(), 1);
    assert_eq!(
        events.lock().unwrap().as_slice(),
        ["factory:A:3", "factory:B:3"]
    );

    let events = Arc::new(Mutex::new(Vec::new()));
    let mut missing_key = registry(Arc::clone(&events), AdapterFailures::default());
    let bad_outer = Outer::new(vec![Order::new(
        "BAD",
        1.0,
        OrderDir::Buy,
        None,
        Some(time(1)),
    )]);
    assert!(
        missing_key
            .reset(&bad_outer)
            .unwrap_err()
            .message
            .contains("start time")
    );
    assert!(missing_key.is_empty());
    assert_eq!(events.lock().unwrap().as_slice(), ["factory:BAD:1"]);
}

#[test]
fn adapter_registry_maps_state_update_and_finalize_failures_in_order() {
    let orders = outer_orders();
    let outer = Outer::new(orders.clone());
    let mut state_failure = registry(
        Arc::new(Mutex::new(Vec::new())),
        AdapterFailures {
            state: Some("A".to_owned()),
            ..AdapterFailures::default()
        },
    );
    state_failure.reset(&outer).unwrap();
    assert_eq!(
        state_failure.state(&orders[0]).unwrap_err(),
        plugin_error("adapter-state")
    );
    assert!(
        state_failure
            .state(&Order::new("BAD", 1.0, OrderDir::Buy, None, None))
            .unwrap_err()
            .message
            .contains("start time")
    );

    let events = Arc::new(Mutex::new(Vec::new()));
    let mut update_key_failure = registry(Arc::clone(&events), AdapterFailures::default());
    update_key_failure.reset(&outer).unwrap();
    events.lock().unwrap().clear();
    assert!(
        update_key_failure
            .update(
                &[execution(
                    Order::new("BAD", 1.0, OrderDir::Buy, None, None,)
                )],
                (1, 2),
            )
            .unwrap_err()
            .message
            .contains("start time")
    );
    assert!(events.lock().unwrap().is_empty());

    let events = Arc::new(Mutex::new(Vec::new()));
    let mut update_failure = registry(
        Arc::clone(&events),
        AdapterFailures {
            update: Some("B".to_owned()),
            ..AdapterFailures::default()
        },
    );
    update_failure.reset(&outer).unwrap();
    events.lock().unwrap().clear();
    assert_eq!(
        update_failure.update(&[], (1, 2)),
        Err(plugin_error("adapter-update"))
    );
    assert_eq!(
        events.lock().unwrap().as_slice(),
        ["adapter-update:A#1:0:1-2", "adapter-update:B#2:0:1-2"]
    );

    let events = Arc::new(Mutex::new(Vec::new()));
    let mut finalize_failure = registry(
        Arc::clone(&events),
        AdapterFailures {
            finalize: Some("B".to_owned()),
            ..AdapterFailures::default()
        },
    );
    finalize_failure.reset(&outer).unwrap();
    events.lock().unwrap().clear();
    assert_eq!(
        finalize_failure.finalize(),
        Err(plugin_error("adapter-finalize"))
    );
    assert_eq!(
        events.lock().unwrap().as_slice(),
        ["adapter-finalize:A#1", "adapter-finalize:B#2"]
    );
}

#[test]
fn lifecycle_strategy_matches_python_range_grouping_and_finalization() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut strategy = lifecycle_strategy(
        vec![
            SaoePolicyAction::Continuous(0.0),
            SaoePolicyAction::Continuous(2.0),
        ],
        Arc::clone(&events),
        (1, 3),
        false,
        AdapterFailures::default(),
    );
    assert!(
        strategy
            .generate_trade_decision(None)
            .err()
            .unwrap()
            .message
            .contains("not been reset")
    );

    let mut outer = Outer::new(outer_orders()[..2].to_vec());
    strategy.reset(&outer).unwrap();
    assert_eq!(strategy.last_step_range(), (0, 0));
    strategy.alter_outer_decision(&mut outer).unwrap();
    let decision = strategy.generate_trade_decision(None).unwrap();
    assert_eq!(strategy.last_step_range(), (1, 3));
    assert_eq!(decision.orders().len(), 1);
    assert_eq!(decision.orders()[0].stock_id(), "B");
    strategy
        .post_execute(&[
            execution(outer_orders()[1].clone()),
            execution(Order::new(
                "X",
                1.0,
                OrderDir::Buy,
                Some(time(0)),
                Some(time(1)),
            )),
            execution(outer_orders()[0].clone()),
        ])
        .unwrap();
    strategy.post_upper_level().unwrap();
    assert!(
        events
            .lock()
            .unwrap()
            .contains(&"adapter-update:A#1:1:1-3".to_owned())
    );
    assert!(
        events
            .lock()
            .unwrap()
            .contains(&"adapter-update:B#2:1:1-3".to_owned())
    );

    let python = python_contract();
    assert_eq!(python["lifecycle_generated"], "inner-decision");
    assert_eq!(python["lifecycle_range"], json!([2, 4]));
}

#[test]
fn lifecycle_strategy_enforces_the_zero_range_execution_invariant() {
    let one_order = Outer::new(outer_orders()[..1].to_vec());
    let fill = vec![SaoePolicyAction::Continuous(0.0)];
    let mut zero = lifecycle_strategy(
        fill,
        Arc::new(Mutex::new(Vec::new())),
        (0, 0),
        false,
        AdapterFailures::default(),
    );
    zero.reset(&one_order).unwrap();
    zero.generate_trade_decision(None).unwrap();
    zero.post_execute(&[]).unwrap();
    assert!(
        zero.post_execute(&[execution(outer_orders()[0].clone())])
            .unwrap_err()
            .message
            .contains("zero-length")
    );
}

#[test]
fn lifecycle_strategy_maps_every_owned_plugin_failure() {
    let one_order = Outer::new(outer_orders()[..1].to_vec());
    let fill = vec![SaoePolicyAction::Continuous(0.0)];
    let mut reset_failure = lifecycle_strategy(
        fill.clone(),
        Arc::new(Mutex::new(Vec::new())),
        (1, 2),
        false,
        AdapterFailures {
            factory: Some("A".to_owned()),
            ..AdapterFailures::default()
        },
    );
    assert!(
        reset_failure
            .reset(&one_order)
            .unwrap_err()
            .message
            .contains("factory")
    );
    assert_eq!(reset_failure.last_step_range(), (0, 0));

    let mut range_failure = lifecycle_strategy(
        fill.clone(),
        Arc::new(Mutex::new(Vec::new())),
        (1, 2),
        true,
        AdapterFailures::default(),
    );
    range_failure.reset(&one_order).unwrap();
    assert!(
        range_failure
            .generate_trade_decision(None)
            .err()
            .unwrap()
            .message
            .contains("range")
    );
    assert_eq!(range_failure.last_step_range(), (0, 0));

    let mut state_failure = lifecycle_strategy(
        fill.clone(),
        Arc::new(Mutex::new(Vec::new())),
        (1, 2),
        false,
        AdapterFailures {
            state: Some("A".to_owned()),
            ..AdapterFailures::default()
        },
    );
    state_failure.reset(&one_order).unwrap();
    assert!(
        state_failure
            .generate_trade_decision(None)
            .err()
            .unwrap()
            .message
            .contains("adapter-state")
    );
    assert_eq!(state_failure.last_step_range(), (1, 2));

    let mut update_failure = lifecycle_strategy(
        fill.clone(),
        Arc::new(Mutex::new(Vec::new())),
        (1, 2),
        false,
        AdapterFailures {
            update: Some("A".to_owned()),
            ..AdapterFailures::default()
        },
    );
    update_failure.reset(&one_order).unwrap();
    update_failure.generate_trade_decision(None).unwrap();
    assert!(
        update_failure
            .post_execute(&[])
            .unwrap_err()
            .message
            .contains("adapter-update")
    );

    let mut finalize_failure = lifecycle_strategy(
        fill,
        Arc::new(Mutex::new(Vec::new())),
        (1, 2),
        false,
        AdapterFailures {
            finalize: Some("A".to_owned()),
            ..AdapterFailures::default()
        },
    );
    finalize_failure.reset(&one_order).unwrap();
    assert!(
        finalize_failure
            .post_upper_level()
            .unwrap_err()
            .message
            .contains("adapter-finalize")
    );
}
