use super::*;
use domain_core::decision_construction::{
    DecisionConstructionStrategy, DecisionOrderItem, SharedOrderDecisionConstruction,
};
use domain_core::shared_executor_lifecycle::{
    SharedAtomicBarEnd, SharedAtomicExecutorAccount, SharedAtomicExecutorLifecycle,
    SharedAtomicLifecycleError, SharedAtomicResult, SharedExecutorDecisionTracker,
    SharedExecutorReturnSink,
};
use std::sync::RwLock;

struct Clock;
impl DecisionConstructionStrategy for Clock {
    type Error = std::convert::Infallible;
    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), Self::Error> {
        Ok((
            timestamp("2024-01-02 09:30:00"),
            timestamp("2024-01-02 10:00:00"),
        ))
    }
}
type Input = SharedOrderDecisionConstruction<Clock, ()>;

fn live_input() -> domain_core::decision_update::SharedLiveDecision<Clock, ()> {
    let old = shared_input();
    Arc::new(RwLock::new(SharedOrderDecisionConstruction {
        strategy: Arc::new(Clock),
        base: old.base,
        total_step: old.total_step,
        orders: old.orders,
        details: old.details,
    }))
}

#[test]
fn live_atomic_session_yields_original_then_observes_mutations_or_closes_without_work() {
    use domain_core::shared_executor_lifecycle::{
        LiveAtomicEvent, LiveAtomicSession, LiveAtomicSessionError,
    };
    for action in ["mutate", "range", "close", "drop"] {
        let mut h = harness(true, "None", None);
        let life = lifecycle(&h, true, "None", None);
        let typed = live_input();
        let decision: domain_core::decision_update::LiveDecisionHandle = typed.clone();
        let mut session = LiveAtomicSession::new(
            &life,
            &mut h.collector,
            &mut h.account,
            decision.clone(),
            None,
            0,
        );
        let LiveAtomicEvent::TrackedDecision(yielded) = session.resume().unwrap() else {
            panic!("tracking");
        };
        assert!(Arc::ptr_eq(&yielded, &decision));
        assert!(h.events.lock().unwrap().is_empty());
        assert!(typed.try_write().is_ok());
        if action == "drop" {
            drop(session);
            assert!(h.events.lock().unwrap().is_empty());
            continue;
        }
        if action == "close" {
            session.close();
            session.close();
            assert_eq!(session.resume().err(), Some(LiveAtomicSessionError::Closed));
            assert!(h.events.lock().unwrap().is_empty());
            continue;
        }
        if action == "range" {
            yielded
                .inherit_range(Some(Arc::new(IdxTradeRange::new(0, 0))))
                .unwrap();
            assert_eq!(
                session.resume().err(),
                Some(LiveAtomicSessionError::Execution(
                    SharedAtomicLifecycleError::UnsupportedRange(0, 0)
                ))
            );
            session.close();
            assert_eq!(session.resume().err(), Some(LiveAtomicSessionError::Failed));
            assert!(h.events.lock().unwrap().is_empty());
        } else {
            let orders = yielded.orders().unwrap();
            let items = orders.read().unwrap();
            let DecisionOrderItem::Order(order) = &items[0] else {
                panic!("order");
            };
            order.write().unwrap().set_factor(Some(9.0));
            drop(items);
            let LiveAtomicEvent::Complete(result) = session.resume().unwrap() else {
                panic!("completion");
            };
            assert_eq!(result.lock().unwrap().len(), 1);
            assert!(
                h.events
                    .lock()
                    .unwrap()
                    .contains(&Event::Deal(Some(9.0_f64.to_bits())))
            );
            session.close();
            assert_eq!(
                session.resume().err(),
                Some(LiveAtomicSessionError::Complete)
            );
        }
    }
}

#[test]
fn untracked_atomic_session_completes_directly_and_can_cancel_before_first_resume() {
    use domain_core::shared_executor_lifecycle::{
        LiveAtomicEvent, LiveAtomicSession, LiveAtomicSessionError,
    };
    for close in [false, true] {
        let mut h = harness(false, "None", None);
        let life = lifecycle(&h, false, "None", None);
        let decision: domain_core::decision_update::LiveDecisionHandle = live_input();
        let mut session =
            LiveAtomicSession::new(&life, &mut h.collector, &mut h.account, decision, None, 0);
        if close {
            session.close();
            assert_eq!(session.resume().err(), Some(LiveAtomicSessionError::Closed));
            assert!(h.events.lock().unwrap().is_empty());
        } else {
            assert!(matches!(
                session.resume().unwrap(),
                LiveAtomicEvent::Complete(_)
            ));
            assert_eq!(
                session.resume().err(),
                Some(LiveAtomicSessionError::Complete)
            );
        }
    }
}

#[test]
fn live_atomic_failures_stop_at_the_reached_stage_and_keep_mutations() {
    for (failure, last) in [
        (
            Failure::Account(Stage::Start),
            Event::SettleStart("cash".to_owned()),
        ),
        (Failure::Account(Stage::Target), Event::Target),
        (Failure::SimulatorTime, Event::SimulatorTime),
        (Failure::LifecycleTime, Event::LifecycleTime),
        (Failure::Account(Stage::Bar), Event::Bar),
        (Failure::LifecycleStep, Event::LifecycleStep),
        (Failure::Account(Stage::Commit), Event::SettleCommit),
    ] {
        let mut h = harness(false, "cash", Some(failure));
        let life = lifecycle(&h, false, "cash", Some(failure));
        let decision: domain_core::decision_update::LiveDecisionHandle = live_input();
        let result =
            life.collect_live_data(&mut h.collector, &decision, &mut h.account, None, None, 0);
        assert!(result.is_err());
        assert_eq!(h.events.lock().unwrap().last(), Some(&last));
        if !h.account.shared_bars.is_empty() {
            assert_eq!(h.account.shared_bars[0].lock().unwrap().len(), 1);
            assert!(decision.base().unwrap().trade_range.is_some());
        }
    }
}

#[test]
fn live_atomic_optional_hooks_and_sink_failure_preserve_source_order() {
    for (track, has_tracker, tracker_fails, sink_fails) in [
        (false, true, true, false),
        (true, false, false, false),
        (true, true, true, false),
        (true, true, false, true),
    ] {
        let mut h = harness(track, "None", None);
        let life = lifecycle(&h, track, "None", None);
        let decision: domain_core::decision_update::LiveDecisionHandle = live_input();
        let tracker = Tracker {
            events: Arc::clone(&h.events),
            fail: tracker_fails,
        };
        let tracker = has_tracker.then_some(
            &tracker as &dyn domain_core::shared_executor_lifecycle::LiveExecutorDecisionTracker,
        );
        let mut sink = SharedSink {
            events: Arc::clone(&h.events),
            retained: None,
            clear: true,
            fail: sink_fails,
        };
        let result = life.collect_live_data(
            &mut h.collector,
            &decision,
            &mut h.account,
            tracker,
            sink_fails.then_some(&mut sink as &mut dyn SharedExecutorReturnSink),
            0,
        );
        if track && has_tracker && tracker_fails {
            assert!(matches!(
                result,
                Err(SharedAtomicLifecycleError::Tracker(_))
            ));
            assert_eq!(*h.events.lock().unwrap(), [Event::Track]);
            let orders = decision.orders().unwrap();
            let items = orders.read().unwrap();
            let DecisionOrderItem::Order(order) = &items[0] else {
                panic!("order");
            };
            assert_eq!(
                order.read().unwrap().factor().map(f64::to_bits),
                Some(7.0_f64.to_bits())
            );
        } else if sink_fails {
            assert!(matches!(
                result,
                Err(SharedAtomicLifecycleError::ReturnSink(_))
            ));
            assert!(h.account.shared_bars[0].lock().unwrap().is_empty());
            assert!(Arc::ptr_eq(
                sink.retained.as_ref().unwrap(),
                &h.account.shared_bars[0]
            ));
            assert_eq!(h.events.lock().unwrap().last(), Some(&Event::Return(1)));
        } else {
            assert_eq!(result.unwrap().lock().unwrap().len(), 1);
            assert!(!h.events.lock().unwrap().contains(&Event::Track));
        }
    }
}

#[test]
fn live_atomic_validation_occurs_before_settlement_but_list_access_after_target() {
    use domain_core::decision_construction::{DecisionAccessError, DecisionTotalStep};
    use domain_core::decision_update::LiveDecisionAccessError;
    for action in ["base", "range", "provider", "orders"] {
        let mut h = harness(false, "cash", None);
        let life = lifecycle(&h, false, "cash", None);
        let typed = live_input();
        {
            let mut current = typed.write().unwrap();
            match action {
                "base" => current.base = None,
                "range" => {
                    current.base.as_mut().unwrap().trade_range =
                        Some(Arc::new(IdxTradeRange::new(-1, 9)));
                    current.total_step = DecisionTotalStep::Value(3);
                }
                "provider" => {
                    current.base.as_mut().unwrap().trade_range = Some(Arc::new(FailingRange));
                }
                "orders" => current.orders = None,
                _ => unreachable!(),
            }
        }
        let decision: domain_core::decision_update::LiveDecisionHandle = typed;
        let error = life
            .collect_live_data(&mut h.collector, &decision, &mut h.account, None, None, 0)
            .err()
            .unwrap();
        match action {
            "base" => assert_eq!(
                error,
                SharedAtomicLifecycleError::LiveDecision(LiveDecisionAccessError::MissingBase)
            ),
            "range" => assert_eq!(error, SharedAtomicLifecycleError::UnsupportedRange(0, 2)),
            "provider" => assert!(matches!(
                error,
                SharedAtomicLifecycleError::LiveDecision(LiveDecisionAccessError::Range(_))
            )),
            "orders" => assert_eq!(
                error,
                SharedAtomicLifecycleError::LiveDecision(LiveDecisionAccessError::Access(
                    DecisionAccessError::MissingOrders
                ))
            ),
            _ => unreachable!(),
        }
        if action == "orders" {
            assert_eq!(
                *h.events.lock().unwrap(),
                [Event::SettleStart("cash".into()), Event::Target]
            );
        } else {
            assert!(h.events.lock().unwrap().is_empty());
        }
    }
}

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

impl domain_core::shared_executor_lifecycle::LiveAtomicExecutorAccount for Account {
    fn update_live_bar_end(
        &mut self,
        bar: SharedAtomicBarEnd,
        decision: &domain_core::decision_update::LiveDecisionHandle,
    ) -> Result<(), AtomicExecutorAccountError> {
        self.events.lock().unwrap().push(Event::Bar);
        decision
            .inherit_range(Some(Arc::new(IdxTradeRange::new(2, 4))))
            .unwrap();
        let original = decision.orders().unwrap();
        let items = original.read().unwrap();
        let DecisionOrderItem::Order(order) = &items[0] else {
            panic!("order");
        };
        assert!(Arc::ptr_eq(order, &bar.trade_info.lock().unwrap()[0].order));
        self.shared_bars.push(bar.trade_info);
        fail_account(self.fail, Stage::Bar)
    }
}

impl domain_core::shared_executor_lifecycle::LiveExecutorDecisionTracker for Tracker {
    fn track_live(
        &self,
        decision: &domain_core::decision_update::LiveDecisionHandle,
    ) -> Result<(), ExecutorDecisionTrackerError> {
        self.events.lock().unwrap().push(Event::Track);
        let orders = decision.orders().unwrap();
        let items = orders.read().unwrap();
        let DecisionOrderItem::Order(order) = &items[0] else {
            panic!("order");
        };
        order.write().unwrap().set_factor(Some(7.0));
        if self.fail {
            return Err(ExecutorDecisionTrackerError {
                message: "track".into(),
            });
        }
        Ok(())
    }
}

#[test]
fn live_atomic_lifecycle_transports_decision_and_result_without_guards_or_copies() {
    for failure in [None, Some(Failure::Account(Stage::Bar))] {
        let mut h = harness(true, "cash", failure);
        let lifecycle = lifecycle(&h, true, "cash", failure);
        let old = shared_input();
        let typed = Arc::new(RwLock::new(SharedOrderDecisionConstruction {
            strategy: Arc::new(Clock),
            base: old.base,
            total_step: old.total_step,
            orders: old.orders,
            details: old.details,
        }));
        let decision: domain_core::decision_update::LiveDecisionHandle = typed.clone();
        let tracker = Tracker {
            events: Arc::clone(&h.events),
            fail: false,
        };
        let mut sink = SharedSink {
            events: Arc::clone(&h.events),
            retained: None,
            clear: false,
            fail: false,
        };
        let result = lifecycle.collect_live_data(
            &mut h.collector,
            &decision,
            &mut h.account,
            Some(&tracker),
            Some(&mut sink),
            3,
        );
        assert!(
            typed
                .try_write()
                .unwrap()
                .base
                .as_ref()
                .unwrap()
                .trade_range
                .is_some()
        );
        assert_eq!(h.account.shared_bars.len(), 1);
        if failure.is_some() {
            assert!(matches!(
                result,
                Err(SharedAtomicLifecycleError::Account(_))
            ));
            assert!(sink.retained.is_none());
            assert_eq!(h.events.lock().unwrap().last(), Some(&Event::Bar));
        } else {
            let result = result.unwrap();
            assert!(Arc::ptr_eq(&result, &h.account.shared_bars[0]));
            assert!(Arc::ptr_eq(&result, sink.retained.as_ref().unwrap()));
            assert_eq!(h.events.lock().unwrap().last(), Some(&Event::Return(1)));
        }
    }
}

fn shared_input() -> Input {
    let mut input = Input::new(Clock);
    let orders = Arc::new(RwLock::new(vec![DecisionOrderItem::Order(Arc::new(
        RwLock::new(Order::new("A", 2.0, OrderDir::Buy, None, None)),
    ))]));
    input.initialize(&orders, None, ()).unwrap();
    input
}

fn lifecycle(
    harness: &Harness,
    track: bool,
    settle: &str,
    failure: Option<Failure>,
) -> SharedAtomicExecutorLifecycle {
    SharedAtomicExecutorLifecycle {
        calendar: Arc::new(LifecycleCalendar {
            events: Arc::clone(&harness.events),
            fail_time: matches!(failure, Some(Failure::LifecycleTime)),
            fail_step: matches!(failure, Some(Failure::LifecycleStep)),
        }),
        track_data: track,
        settle_type: settle.to_owned(),
        indicator_config: IndicatorConfig::default(),
    }
}

impl SharedAtomicExecutorAccount<Clock, ()> for Account {
    fn update_shared_bar_end(
        &mut self,
        bar: SharedAtomicBarEnd,
        decision: &mut Input,
    ) -> Result<(), AtomicExecutorAccountError> {
        self.events.lock().unwrap().push(Event::Bar);
        assert_eq!(bar.trade_start_time, timestamp("2024-01-02 09:30:00"));
        assert_eq!(bar.trade_end_time, timestamp("2024-01-02 10:00:00"));
        assert_eq!(bar.indicator_config, IndicatorConfig::default());
        decision.total_step = domain_core::decision_construction::DecisionTotalStep::Value(9);
        let result = bar.trade_info.lock().unwrap();
        result[0].order.write().unwrap().set_factor(Some(11.0));
        drop(result);
        self.shared_bars.push(bar.trade_info);
        fail_account(self.fail, Stage::Bar)
    }
}

impl SharedExecutorDecisionTracker<Clock, ()> for Tracker {
    fn track(&self, decision: &mut Input) -> Result<(), ExecutorDecisionTrackerError> {
        self.events.lock().unwrap().push(Event::Track);
        let items = decision.orders.as_ref().unwrap().read().unwrap();
        let DecisionOrderItem::Order(order) = &items[0] else {
            panic!("order")
        };
        order.write().unwrap().set_factor(Some(7.0));
        if self.fail {
            return Err(ExecutorDecisionTrackerError {
                message: "track".to_owned(),
            });
        }
        Ok(())
    }
}

struct SharedSink {
    events: Arc<Mutex<Vec<Event>>>,
    retained: Option<SharedAtomicResult>,
    clear: bool,
    fail: bool,
}
impl SharedExecutorReturnSink for SharedSink {
    fn store(&mut self, result: SharedAtomicResult) -> Result<(), ExecutorReturnSinkError> {
        let mut rows = result.lock().unwrap();
        self.events.lock().unwrap().push(Event::Return(rows.len()));
        if self.clear {
            rows.clear();
        }
        drop(rows);
        self.retained = Some(result);
        if self.fail {
            return Err(ExecutorReturnSinkError {
                message: "sink".to_owned(),
            });
        }
        Ok(())
    }
}

#[test]
fn shared_lifecycle_matches_source_order_and_keeps_callback_identities() {
    let mut h = harness(true, "cash", None);
    let lifecycle = lifecycle(&h, true, "cash", None);
    let mut input = shared_input();
    let tracker = Tracker {
        events: Arc::clone(&h.events),
        fail: false,
    };
    let mut sink = SharedSink {
        events: Arc::clone(&h.events),
        retained: None,
        clear: false,
        fail: false,
    };
    let result = lifecycle
        .collect_data(
            &mut h.collector,
            &mut input,
            &mut h.account,
            Some(&tracker),
            Some(&mut sink),
            3,
        )
        .unwrap();
    assert!(Arc::ptr_eq(&result, &h.account.shared_bars[0]));
    assert!(Arc::ptr_eq(&result, sink.retained.as_ref().unwrap()));
    assert_eq!(
        input.total_step,
        domain_core::decision_construction::DecisionTotalStep::Value(9)
    );
    assert_eq!(
        result.lock().unwrap()[0]
            .order
            .read()
            .unwrap()
            .factor()
            .map(f64::to_bits),
        Some(11.0_f64.to_bits())
    );
    let events: Vec<String> = h
        .events
        .lock()
        .unwrap()
        .iter()
        .filter_map(|event| match event {
            Event::Track => Some("track".to_owned()),
            Event::SettleStart(value) => Some(format!("start:{value}")),
            Event::Deal(value) => {
                assert_eq!(*value, Some(7.0_f64.to_bits()));
                Some("collect:3".to_owned())
            }
            Event::LifecycleTime => Some("time".to_owned()),
            Event::Bar => Some("bar:True:1".to_owned()),
            Event::LifecycleStep => Some("step".to_owned()),
            Event::SettleCommit => Some("commit".to_owned()),
            Event::Return(count) => Some(format!("return:{count}")),
            Event::Target | Event::SimulatorTime => None,
        })
        .collect();
    assert_eq!(
        json!({"events": events, "ret": ["R"], "stored": ["R"]}),
        live_python_snapshot()
    );
}

#[test]
fn native_atomic_result_moves_to_nested_transport_without_rewrapping_or_copying() {
    let mut h = harness(false, "None", None);
    let lifecycle = lifecycle(&h, false, "None", None);
    let mut input = shared_input();
    let mut sink = SharedSink {
        events: Arc::clone(&h.events),
        retained: None,
        clear: false,
        fail: false,
    };
    let result = lifecycle
        .collect_data(
            &mut h.collector,
            &mut input,
            &mut h.account,
            None,
            Some(&mut sink),
            0,
        )
        .unwrap();
    let nested: domain_core::nested_executor::SharedNestedResult = result;
    assert!(Arc::ptr_eq(&nested, &h.account.shared_bars[0]));
    assert!(Arc::ptr_eq(&nested, sink.retained.as_ref().unwrap()));
    let row = Arc::clone(&nested.lock().unwrap()[0]);
    let original = {
        let items = input.orders.as_ref().unwrap().read().unwrap();
        let DecisionOrderItem::Order(order) = &items[0] else {
            panic!("order")
        };
        Arc::clone(order)
    };
    assert!(Arc::ptr_eq(&row.order, &original));
    original.write().unwrap().set_deal_amount(9.0);
    assert_eq!(
        row.order.read().unwrap().deal_amount().to_bits(),
        9.0_f64.to_bits()
    );
    nested
        .try_lock()
        .expect("no callback guard retained")
        .clear();
    assert!(h.account.shared_bars[0].lock().unwrap().is_empty());
    assert!(sink.retained.as_ref().unwrap().lock().unwrap().is_empty());
    drop(input);
    drop(h);
    drop(sink);
    nested.lock().unwrap().push(Arc::clone(&row));
    assert!(Arc::ptr_eq(&nested.lock().unwrap()[0], &row));
}

#[test]
fn result_list_mutations_survive_sink_success_and_failure_and_optional_hooks() {
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/shared_atomic_result_contract.py"
        ))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let source: Value = serde_json::from_slice(&output.stdout).unwrap();
    for fail in [false, true] {
        let mut h = harness(false, "None", None);
        let lifecycle = lifecycle(&h, false, "None", None);
        let mut input = shared_input();
        let tracker = Tracker {
            events: Arc::clone(&h.events),
            fail: true,
        };
        let mut sink = SharedSink {
            events: Arc::clone(&h.events),
            retained: None,
            clear: true,
            fail,
        };
        let result = lifecycle.collect_data(
            &mut h.collector,
            &mut input,
            &mut h.account,
            Some(&tracker),
            Some(&mut sink),
            0,
        );
        assert_eq!(result.is_err(), fail);
        assert_eq!(
            json!({
                "ok": result.is_ok(),
                "bar_len": h.account.shared_bars[0].lock().unwrap().len(),
                "sink_len": sink.retained.as_ref().unwrap().lock().unwrap().len(),
                "same_list": Arc::ptr_eq(&h.account.shared_bars[0], sink.retained.as_ref().unwrap()),
                "decision_marker": match input.total_step {
                    domain_core::decision_construction::DecisionTotalStep::Value(value) => Some(value),
                    _ => None,
                },
            }),
            source[if fail { "True" } else { "False" }]
        );
        assert!(h.account.shared_bars[0].lock().unwrap().is_empty());
        assert!(Arc::ptr_eq(
            &h.account.shared_bars[0],
            sink.retained.as_ref().unwrap()
        ));
        if let Ok(returned) = result {
            assert!(Arc::ptr_eq(&returned, sink.retained.as_ref().unwrap()));
        }
        assert!(!h.events.lock().unwrap().contains(&Event::SettleCommit));
    }
    let mut h = harness(true, "None", None);
    lifecycle(&h, true, "None", None)
        .collect_data(
            &mut h.collector,
            &mut shared_input(),
            &mut h.account,
            None,
            None,
            0,
        )
        .unwrap();
    assert_eq!(h.account.shared_bars.len(), 1);
}

#[test]
fn shared_lifecycle_validates_partial_initialization_and_range_before_settlement() {
    let mut h = harness(false, "None", None);
    let lifecycle = lifecycle(&h, false, "None", None);
    let mut input = Input::new(Clock);
    assert!(matches!(
        lifecycle.collect_data(&mut h.collector, &mut input, &mut h.account, None, None, 0),
        Err(SharedAtomicLifecycleError::MissingBase)
    ));
    input = shared_input();
    input.base.as_mut().unwrap().trade_range = Some(Arc::new(IdxTradeRange::new(-1, 9)));
    input.total_step = domain_core::decision_construction::DecisionTotalStep::Value(3);
    assert!(matches!(
        lifecycle.collect_data(&mut h.collector, &mut input, &mut h.account, None, None, 0),
        Err(SharedAtomicLifecycleError::UnsupportedRange(0, 2))
    ));
    input.base.as_mut().unwrap().trade_range = Some(Arc::new(FailingRange));
    assert!(matches!(
        lifecycle.collect_data(&mut h.collector, &mut input, &mut h.account, None, None, 0),
        Err(SharedAtomicLifecycleError::Range(_))
    ));
    assert!(h.events.lock().unwrap().is_empty());
    input.base.as_mut().unwrap().trade_range = None;
    input.orders = None;
    assert!(matches!(
        lifecycle.collect_data(&mut h.collector, &mut input, &mut h.account, None, None, 0),
        Err(SharedAtomicLifecycleError::MissingOrders)
    ));
    input = shared_input();
    input.base.as_mut().unwrap().trade_range =
        Some(Arc::new(TradeRangeByTime::parse("09:30", "10:00").unwrap()));
    lifecycle
        .collect_data(&mut h.collector, &mut input, &mut h.account, None, None, 0)
        .unwrap();
}

#[test]
fn each_shared_lifecycle_failure_stops_without_rollback() {
    for (failure, last) in [
        (
            Failure::Account(Stage::Start),
            Event::SettleStart("cash".to_owned()),
        ),
        (Failure::Account(Stage::Target), Event::Target),
        (Failure::SimulatorTime, Event::SimulatorTime),
        (Failure::LifecycleTime, Event::LifecycleTime),
        (Failure::Account(Stage::Bar), Event::Bar),
        (Failure::LifecycleStep, Event::LifecycleStep),
        (Failure::Account(Stage::Commit), Event::SettleCommit),
    ] {
        let mut h = harness(false, "cash", Some(failure));
        let lifecycle = lifecycle(&h, false, "cash", Some(failure));
        let mut input = shared_input();
        assert!(
            lifecycle
                .collect_data(&mut h.collector, &mut input, &mut h.account, None, None, 0)
                .is_err()
        );
        assert_eq!(h.events.lock().unwrap().last(), Some(&last));
        if !h.account.shared_bars.is_empty() {
            assert_eq!(
                input.total_step,
                domain_core::decision_construction::DecisionTotalStep::Value(9)
            );
            assert_eq!(h.account.shared_bars[0].lock().unwrap().len(), 1);
        }
    }
    let mut h = harness(true, "cash", None);
    let tracker = Tracker {
        events: Arc::clone(&h.events),
        fail: true,
    };
    assert!(matches!(
        lifecycle(&h, true, "cash", None).collect_data(
            &mut h.collector,
            &mut shared_input(),
            &mut h.account,
            Some(&tracker),
            None,
            0
        ),
        Err(SharedAtomicLifecycleError::Tracker(_))
    ));
    assert_eq!(h.events.lock().unwrap().as_slice(), [Event::Track]);
}
