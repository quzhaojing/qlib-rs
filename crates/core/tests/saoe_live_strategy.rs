use std::sync::{Arc, Mutex, RwLock};

use arrow_array::RecordBatch;
use arrow_schema::Schema;
use chrono::{NaiveDate, NaiveDateTime, NaiveTime, TimeDelta};
use domain_core::decision_construction::{
    ConstructedDecisionBase, DecisionOrderItem, DecisionTotalStep, SharedDecisionOrders,
    SharedOrderDecisionConstruction,
};
use domain_core::decision_update::{
    DecisionUpdateStrategyError, LiveDecisionHandle, SharedDecisionUpdateStrategy,
    SharedLiveDecision,
};
use domain_core::live_nested_executor::{LiveNestedStrategy, LiveNestedStrategyProgress};
use domain_core::nested_executor::{SharedNestedResult, SharedOrderExecution};
use domain_core::saoe_live_generation::LiveSaoeGeneration;
use domain_core::saoe_live_registry::{
    LiveSaoeAdapterFactory, LiveSaoeAdapterRegistry, LiveSaoeStateAdapter,
};
use domain_core::{
    AllOnePolicy, DummyStateInterpreter, LiveSaoeBacktestData, LiveSaoeIntStrategy, LiveSaoeState,
    LiveSaoeStateParts, Order, OrderDir, SaoeActionInterpreter, SaoeActionSpace, SaoeBacktestData,
    SaoeDecisionCalendar, SaoeInterpreterError, SaoeOrderFactory, SaoePluginError,
    SaoePolicyAction, SaoePolicyPipeline, SaoeState, SharedSaoeHistory, SharedTradeRange,
    TradeCalendarRange, TradeCalendarRangeError, TradeRange, TradeRangeByTime, TradeRangeError,
};
use ndarray::arr1;

fn time(minute: i64) -> NaiveDateTime {
    NaiveDate::from_ymd_opt(2024, 1, 2)
        .unwrap()
        .and_hms_opt(9, 30, 0)
        .unwrap()
        + TimeDelta::minutes(minute)
}

fn plugin_error(message: &str) -> SaoePluginError {
    SaoePluginError {
        message: message.to_owned(),
    }
}

struct Origin;

impl SharedDecisionUpdateStrategy<()> for Origin {
    fn update_trade_decision(
        &self,
        _decision: &SharedLiveDecision<Self, ()>,
        _calendar: &dyn domain_core::DecisionUpdateCalendar,
    ) -> Result<Option<SharedLiveDecision<Self, ()>>, DecisionUpdateStrategyError> {
        Ok(None)
    }
}

fn typed_outer(
    order: Arc<RwLock<Order>>,
    range: Option<SharedTradeRange>,
) -> SharedLiveDecision<Origin, ()> {
    let orders: SharedDecisionOrders = Arc::new(RwLock::new(vec![DecisionOrderItem::Order(order)]));
    Arc::new(RwLock::new(SharedOrderDecisionConstruction {
        strategy: Arc::new(Origin),
        base: Some(ConstructedDecisionBase {
            start_time: time(0),
            end_time: time(4),
            trade_range: range,
        }),
        total_step: DecisionTotalStep::Value(5),
        orders: Some(orders),
        details: Some(()),
    }))
}

fn outer(order: Arc<RwLock<Order>>, range: SharedTradeRange) -> LiveDecisionHandle {
    typed_outer(order, Some(range))
}

#[derive(Default)]
struct PoisonAfterRange {
    decision: Mutex<Option<SharedLiveDecision<Origin, ()>>>,
}

impl TradeRange for PoisonAfterRange {
    fn range_indices(
        &self,
        _calendar: Option<&dyn TradeCalendarRange>,
    ) -> Result<(i64, i64), TradeRangeError> {
        let decision = self.decision.lock().unwrap().as_ref().unwrap().clone();
        assert!(
            std::thread::spawn(move || {
                let _guard = decision.write().unwrap();
                panic!("poison decision after its base was read");
            })
            .join()
            .is_err()
        );
        Ok((0, 4))
    }

    fn clip_time_range(
        &self,
        start_time: NaiveDateTime,
        end_time: NaiveDateTime,
    ) -> Result<(NaiveDateTime, NaiveDateTime), TradeRangeError> {
        Ok((start_time, end_time))
    }
}

struct LengthCalendar;

impl domain_core::DecisionUpdateCalendar for LengthCalendar {
    fn trade_len(&self) -> Result<i64, domain_core::DecisionUpdateCalendarError> {
        Ok(3)
    }
}

struct Calendar {
    events: Arc<Mutex<Vec<String>>>,
}

impl domain_core::SaoeCalendar for Calendar {
    fn available_step_range(&self) -> Result<(i64, i64), SaoePluginError> {
        self.events.lock().unwrap().push("available".into());
        Ok((0, 4))
    }

    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), SaoePluginError> {
        self.events.lock().unwrap().push("time".into());
        Ok((time(1), time(2)))
    }
}

impl SaoeDecisionCalendar for Calendar {
    fn frequency(&self) -> Result<String, SaoePluginError> {
        self.events.lock().unwrap().push("frequency".into());
        Ok("1min".into())
    }
}

impl TradeCalendarRange for Calendar {
    fn start_time(&self) -> Result<NaiveDateTime, TradeCalendarRangeError> {
        Ok(time(0))
    }

    fn get_range_idx(
        &self,
        _start_time: NaiveDateTime,
        _end_time: NaiveDateTime,
    ) -> Result<(i64, i64), TradeCalendarRangeError> {
        self.events.lock().unwrap().push("range".into());
        Ok((1, 3))
    }
}

struct FailingCalendar {
    fail_available: bool,
    fail_range: bool,
    fail_step_on: Option<usize>,
    step_calls: Mutex<usize>,
}

impl domain_core::SaoeCalendar for FailingCalendar {
    fn available_step_range(&self) -> Result<(i64, i64), SaoePluginError> {
        if self.fail_available {
            return Err(plugin_error("available"));
        }
        Ok((0, 4))
    }

    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), SaoePluginError> {
        let mut calls = self.step_calls.lock().unwrap();
        *calls += 1;
        if self.fail_step_on == Some(*calls) {
            return Err(plugin_error("step"));
        }
        Ok((time(1), time(2)))
    }
}

impl SaoeDecisionCalendar for FailingCalendar {
    fn frequency(&self) -> Result<String, SaoePluginError> {
        Ok("1min".into())
    }
}

impl TradeCalendarRange for FailingCalendar {
    fn start_time(&self) -> Result<NaiveDateTime, TradeCalendarRangeError> {
        Ok(time(0))
    }

    fn get_range_idx(
        &self,
        _start_time: NaiveDateTime,
        _end_time: NaiveDateTime,
    ) -> Result<(i64, i64), TradeCalendarRangeError> {
        if self.fail_range {
            return Err(TradeCalendarRangeError::Provider {
                message: "range".into(),
            });
        }
        Ok((1, 3))
    }
}

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
        let SaoePolicyAction::Continuous(value) = action else {
            return Err(SaoeInterpreterError::ExpectedContinuous);
        };
        Ok(value)
    }
}

struct Orders {
    events: Arc<Mutex<Vec<String>>>,
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
            .push(format!("child:{stock_id}:{}:{direction}", amount.unwrap()));
        Ok(Order::new(stock_id, amount.unwrap(), direction, None, None))
    }
}

struct Adapter {
    state: LiveSaoeState,
    events: Arc<Mutex<Vec<String>>>,
}

fn live_state(order: &Arc<RwLock<Order>>) -> LiveSaoeState {
    let ticks = vec![time(0), time(1), time(2), time(3)];
    let empty = RecordBatch::new_empty(Arc::new(Schema::empty()));
    let backtest = LiveSaoeBacktestData::from_owned(SaoeBacktestData {
        ticks_index: ticks.clone(),
        ticks_for_order: ticks,
        deal_prices: arr1(&[10.0]),
        market_volumes: arr1(&[100.0]),
        features: empty,
    })
    .into_shared();
    let handles = backtest.read().unwrap().clone();
    let history: SharedSaoeHistory = Arc::new(RwLock::new(Vec::new()));
    LiveSaoeState::new(LiveSaoeStateParts {
        order: Arc::clone(order),
        cur_time: time(0),
        cur_step: 0,
        position: 10.0,
        history_exec: Arc::clone(&history),
        history_steps: history,
        metrics: None,
        backtest_data: backtest,
        ticks_index: handles.ticks_index,
        ticks_for_order: handles.ticks_for_order,
        ticks_per_step: 1,
    })
}

impl LiveSaoeStateAdapter for Adapter {
    fn live_state(&self) -> Result<LiveSaoeState, SaoePluginError> {
        self.events.lock().unwrap().push("state".into());
        Ok(self.state.clone())
    }

    fn state(&self) -> Result<SaoeState, SaoePluginError> {
        self.state
            .snapshot()
            .map_err(|error| plugin_error(&error.to_string()))
    }

    fn update(
        &mut self,
        executions: &[SharedOrderExecution],
        range: (i64, i64),
    ) -> Result<(), SaoePluginError> {
        self.events.lock().unwrap().push(format!(
            "update:{}:{}:{}",
            executions.len(),
            range.0,
            range.1
        ));
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), SaoePluginError> {
        self.events.lock().unwrap().push("finalize".into());
        Ok(())
    }
}

struct Adapters {
    events: Arc<Mutex<Vec<String>>>,
}

impl LiveSaoeAdapterFactory for Adapters {
    fn create(
        &mut self,
        order: &Arc<RwLock<Order>>,
        _outer: &LiveDecisionHandle,
        _range: &SharedTradeRange,
    ) -> Result<Box<dyn LiveSaoeStateAdapter>, SaoePluginError> {
        self.events.lock().unwrap().push("adapter".into());
        Ok(Box::new(Adapter {
            state: live_state(order),
            events: Arc::clone(&self.events),
        }))
    }
}

struct FailingAdapter {
    state: LiveSaoeState,
    mode: &'static str,
}

impl LiveSaoeStateAdapter for FailingAdapter {
    fn live_state(&self) -> Result<LiveSaoeState, SaoePluginError> {
        if self.mode == "state" {
            return Err(plugin_error("state"));
        }
        Ok(self.state.clone())
    }

    fn state(&self) -> Result<SaoeState, SaoePluginError> {
        self.state
            .snapshot()
            .map_err(|error| plugin_error(&error.to_string()))
    }

    fn update(
        &mut self,
        _executions: &[SharedOrderExecution],
        _range: (i64, i64),
    ) -> Result<(), SaoePluginError> {
        if self.mode == "update" {
            return Err(plugin_error("update"));
        }
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), SaoePluginError> {
        if self.mode == "finalize" {
            return Err(plugin_error("finalize"));
        }
        Ok(())
    }
}

struct FailingAdapters {
    mode: &'static str,
}

impl LiveSaoeAdapterFactory for FailingAdapters {
    fn create(
        &mut self,
        order: &Arc<RwLock<Order>>,
        _outer: &LiveDecisionHandle,
        _range: &SharedTradeRange,
    ) -> Result<Box<dyn LiveSaoeStateAdapter>, SaoePluginError> {
        assert_ne!(self.mode, "panic", "poison live strategy runtime");
        if self.mode == "create" {
            return Err(plugin_error("create"));
        }
        Ok(Box::new(FailingAdapter {
            state: live_state(order),
            mode: self.mode,
        }))
    }
}

fn failing_strategy(
    adapter_mode: &'static str,
    calendar: Arc<FailingCalendar>,
) -> Arc<LiveSaoeIntStrategy> {
    LiveSaoeIntStrategy::new(
        LiveSaoeAdapterRegistry::new(Box::new(FailingAdapters { mode: adapter_mode })),
        LiveSaoeGeneration::new(
            SaoePolicyPipeline::new(
                Box::new(DummyStateInterpreter),
                Box::new(AllOnePolicy::default()),
                Box::new(DirectAction),
            ),
            Box::new(Orders {
                events: Arc::new(Mutex::new(Vec::new())),
            }),
            Arc::clone(&calendar) as Arc<dyn SaoeDecisionCalendar>,
        ),
        Arc::clone(&calendar) as Arc<dyn SaoeDecisionCalendar>,
        calendar as Arc<dyn TradeCalendarRange>,
    )
}

fn good_failing_calendar() -> Arc<FailingCalendar> {
    Arc::new(FailingCalendar {
        fail_available: false,
        fail_range: false,
        fail_step_on: None,
        step_calls: Mutex::new(0),
    })
}

#[test]
fn shared_strategy_composes_reset_generation_update_finalization_and_origin_identity() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let calendar = Arc::new(Calendar {
        events: Arc::clone(&events),
    });
    let strategy = LiveSaoeIntStrategy::new(
        LiveSaoeAdapterRegistry::new(Box::new(Adapters {
            events: Arc::clone(&events),
        })),
        LiveSaoeGeneration::new(
            SaoePolicyPipeline::new(
                Box::new(DummyStateInterpreter),
                Box::new(AllOnePolicy::new(SaoePolicyAction::Continuous(2.0))),
                Box::new(DirectAction),
            ),
            Box::new(Orders {
                events: Arc::clone(&events),
            }),
            Arc::clone(&calendar) as Arc<dyn SaoeDecisionCalendar>,
        ),
        Arc::clone(&calendar) as Arc<dyn SaoeDecisionCalendar>,
        Arc::clone(&calendar) as Arc<dyn TradeCalendarRange>,
    );
    let order = Arc::new(RwLock::new(Order::new(
        "A",
        10.0,
        OrderDir::Buy,
        Some(time(0)),
        Some(time(4)),
    )));
    let range: SharedTradeRange = Arc::new(
        TradeRangeByTime::new(
            NaiveTime::from_hms_opt(9, 30, 0).unwrap(),
            NaiveTime::from_hms_opt(9, 34, 0).unwrap(),
        )
        .unwrap(),
    );
    let outer = outer(order, range);
    let mut lifecycle = Arc::clone(&strategy);
    lifecycle.reset(&outer).unwrap();
    let altered = lifecycle.alter_outer_decision(outer.clone()).unwrap();
    assert!(Arc::ptr_eq(&altered, &outer));
    assert_eq!(strategy.last_step_range().unwrap(), (0, 0));
    let before = Arc::strong_count(&strategy);
    let LiveNestedStrategyProgress::Ready(decision) = lifecycle.begin(None).unwrap() else {
        panic!("immediate strategy must not suspend");
    };
    assert_eq!(Arc::strong_count(&strategy), before + 1);
    assert_eq!(strategy.last_step_range().unwrap(), (1, 3));
    let children = decision.orders().unwrap();
    let children = children.read().unwrap();
    let DecisionOrderItem::Order(child) = &children[0] else {
        panic!("generated item must be an order");
    };
    let child = child.read().unwrap();
    assert_eq!(child.amount().to_bits(), 2.0_f64.to_bits());
    assert_eq!(child.start_time(), Some(time(1)));
    assert_eq!(child.end_time(), Some(time(2)));
    drop(child);
    drop(children);
    assert!(decision.clone().update(&LengthCalendar).unwrap().is_none());
    let executions: SharedNestedResult = Arc::new(Mutex::new(Vec::new()));
    lifecycle.post_execute(&executions).unwrap();
    lifecycle.post_upper_level().unwrap();
    assert_eq!(
        *events.lock().unwrap(),
        [
            "adapter",
            "available",
            "range",
            "state",
            "child:A:2:buy",
            "time",
            "frequency",
            "time",
            "time",
            "update:0:1:3",
            "finalize"
        ]
    );
}

fn test_range() -> SharedTradeRange {
    Arc::new(
        TradeRangeByTime::new(
            NaiveTime::from_hms_opt(9, 30, 0).unwrap(),
            NaiveTime::from_hms_opt(9, 34, 0).unwrap(),
        )
        .unwrap(),
    )
}

fn test_order() -> Arc<RwLock<Order>> {
    Arc::new(RwLock::new(Order::new(
        "A",
        10.0,
        OrderDir::Buy,
        Some(time(0)),
        Some(time(4)),
    )))
}

fn test_outer() -> LiveDecisionHandle {
    outer(test_order(), test_range())
}

#[test]
fn shared_strategy_requires_reset_and_keeps_default_non_suspending_hooks() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let calendar = Arc::new(Calendar { events });
    let strategy = LiveSaoeIntStrategy::new(
        LiveSaoeAdapterRegistry::new(Box::new(Adapters {
            events: Arc::new(Mutex::new(Vec::new())),
        })),
        LiveSaoeGeneration::new(
            SaoePolicyPipeline::new(
                Box::new(DummyStateInterpreter),
                Box::new(AllOnePolicy::default()),
                Box::new(DirectAction),
            ),
            Box::new(Orders {
                events: Arc::new(Mutex::new(Vec::new())),
            }),
            Arc::clone(&calendar) as Arc<dyn SaoeDecisionCalendar>,
        ),
        Arc::clone(&calendar) as Arc<dyn SaoeDecisionCalendar>,
        calendar as Arc<dyn TradeCalendarRange>,
    );
    let mut lifecycle = Arc::clone(&strategy);
    assert!(lifecycle.begin(None).is_err());
    assert!(lifecycle.resume(None).is_err());
    assert!(lifecycle.close().is_ok());
}

#[test]
fn shared_strategy_maps_reset_calendar_range_generation_construction_and_adapter_failures() {
    let create_failure = failing_strategy("create", good_failing_calendar());
    assert!(Arc::clone(&create_failure).reset(&test_outer()).is_err());

    let available_calendar = Arc::new(FailingCalendar {
        fail_available: true,
        fail_range: false,
        fail_step_on: None,
        step_calls: Mutex::new(0),
    });
    let available_failure = failing_strategy("none", available_calendar);
    let outer = test_outer();
    let mut lifecycle = Arc::clone(&available_failure);
    lifecycle.reset(&outer).unwrap();
    assert!(lifecycle.begin(None).is_err());

    let range_calendar = Arc::new(FailingCalendar {
        fail_available: false,
        fail_range: true,
        fail_step_on: None,
        step_calls: Mutex::new(0),
    });
    let range_failure = failing_strategy("none", range_calendar);
    let mut lifecycle = Arc::clone(&range_failure);
    lifecycle.reset(&outer).unwrap();
    assert!(lifecycle.begin(None).is_err());

    let state_failure = failing_strategy("state", good_failing_calendar());
    let mut lifecycle = Arc::clone(&state_failure);
    lifecycle.reset(&outer).unwrap();
    assert!(lifecycle.begin(None).is_err());

    for fail_step_on in [1, 2] {
        let calendar = Arc::new(FailingCalendar {
            fail_available: false,
            fail_range: false,
            fail_step_on: Some(fail_step_on),
            step_calls: Mutex::new(0),
        });
        let strategy = failing_strategy("none", calendar);
        let mut lifecycle = Arc::clone(&strategy);
        lifecycle.reset(&outer).unwrap();
        assert!(lifecycle.begin(None).is_err());
    }

    let update_failure = failing_strategy("update", good_failing_calendar());
    let mut lifecycle = Arc::clone(&update_failure);
    lifecycle.reset(&outer).unwrap();
    let _ = lifecycle.begin(None).unwrap();
    assert!(
        lifecycle
            .post_execute(&Arc::new(Mutex::new(Vec::new())))
            .is_err()
    );

    let finalize_failure = failing_strategy("finalize", good_failing_calendar());
    let mut lifecycle = Arc::clone(&finalize_failure);
    lifecycle.reset(&outer).unwrap();
    assert!(lifecycle.post_upper_level().is_err());
}

#[test]
fn shared_strategy_observes_live_outer_metadata_variants_and_lock_failure() {
    let typed = typed_outer(
        Arc::new(RwLock::new(Order::new(
            "A",
            10.0,
            OrderDir::Buy,
            Some(time(0)),
            Some(time(4)),
        ))),
        Some(test_range()),
    );
    let erased: LiveDecisionHandle = typed.clone();
    let missing_range = failing_strategy("none", good_failing_calendar());
    let mut lifecycle = Arc::clone(&missing_range);
    lifecycle.reset(&erased).unwrap();
    typed.write().unwrap().base.as_mut().unwrap().trade_range = None;
    assert!(lifecycle.begin(None).is_err());

    let typed = typed_outer(test_order(), Some(test_range()));
    let erased: LiveDecisionHandle = typed.clone();
    let missing_base = failing_strategy("none", good_failing_calendar());
    let mut lifecycle = Arc::clone(&missing_base);
    lifecycle.reset(&erased).unwrap();
    typed.write().unwrap().base = None;
    assert!(lifecycle.begin(None).is_err());

    let typed = typed_outer(test_order(), Some(test_range()));
    let erased: LiveDecisionHandle = typed.clone();
    let unset_total = failing_strategy("none", good_failing_calendar());
    let mut lifecycle = Arc::clone(&unset_total);
    lifecycle.reset(&erased).unwrap();
    typed.write().unwrap().total_step = DecisionTotalStep::Unset;
    assert!(matches!(
        lifecycle.begin(None).unwrap(),
        LiveNestedStrategyProgress::Ready(_)
    ));

    let poison_rule = Arc::new(PoisonAfterRange::default());
    let typed = typed_outer(
        test_order(),
        Some(Arc::clone(&poison_rule) as SharedTradeRange),
    );
    let erased: LiveDecisionHandle = typed.clone();
    *poison_rule.decision.lock().unwrap() = Some(typed);
    let poisoned_total = failing_strategy("none", good_failing_calendar());
    let mut lifecycle = Arc::clone(&poisoned_total);
    lifecycle.reset(&erased).unwrap();
    assert!(lifecycle.begin(None).is_err());
    poison_rule.decision.lock().unwrap().take();
}

#[test]
fn poisoned_shared_strategy_reports_every_runtime_entry_without_recovery() {
    let strategy = failing_strategy("panic", good_failing_calendar());
    let poison = Arc::clone(&strategy);
    let outer = test_outer();
    let poison_outer = outer.clone();
    assert!(
        std::thread::spawn(move || {
            let mut lifecycle = poison;
            lifecycle.reset(&poison_outer).unwrap();
        })
        .join()
        .is_err()
    );
    assert!(strategy.last_step_range().is_err());
    let mut lifecycle = Arc::clone(&strategy);
    assert!(lifecycle.reset(&outer).is_err());
    assert!(lifecycle.begin(None).is_err());
    assert!(
        lifecycle
            .post_execute(&Arc::new(Mutex::new(Vec::new())))
            .is_err()
    );
    assert!(lifecycle.post_upper_level().is_err());
}
