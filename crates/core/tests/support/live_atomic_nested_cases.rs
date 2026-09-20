use super::*;
use domain_core::decision_construction::{
    ConstructedDecisionBase, DecisionOrderItem, SharedOrderDecisionConstruction,
};
use domain_core::decision_update::{
    LiveDecisionHandle, SharedDecisionUpdateStrategy, SharedLiveDecision,
};
use domain_core::nested_executor::{
    LiveNestedControlEvent, LiveNestedInnerProgress, NestedExecutorResume, NestedInnerControlMode,
};
use std::sync::RwLock;

#[test]
fn native_child_tracking_retains_identity_and_executes_mutations_only_on_resume() {
    let start = time("2024-01-02 09:30:00");
    let end = time("2024-01-02 09:30:59");
    let calendar = Arc::new(SharedInnerCalendar::new(start, end));
    let account = Arc::new(Mutex::new(Account::new(InfinitePosition, false)));
    let mut inner: Box<dyn NestedInnerExecutor> = Box::new(
        OwnedAtomicNestedInnerAdapter::new(
            calendar.clone(),
            collector(calendar.clone(), false),
            account.clone(),
            Arc::new(Services),
            "None",
            IndicatorConfig::default(),
            false,
        )
        .with_live_tracking(true),
    );
    assert_eq!(inner.live_control_mode(), NestedInnerControlMode::Delegated);
    assert_eq!(inner.control_mode(), NestedInnerControlMode::Framework);
    let (decision, order) = request(start, end);
    let LiveNestedInnerProgress::Suspended(LiveNestedControlEvent::TrackedDecision(tracked)) =
        inner.begin_live_collect_data(decision.clone(), 7).unwrap()
    else {
        panic!("tracking expected")
    };
    assert!(Arc::ptr_eq(&decision, &tracked));
    assert_eq!(inner.trade_step().unwrap(), 0);
    assert!(account.try_lock().is_ok());
    assert!(decision.orders().unwrap().try_write().is_ok());
    assert!(
        inner
            .begin_live_collect_data(decision.clone(), 8)
            .err()
            .unwrap()
            .message
            .contains("already suspended")
    );
    *order.write().unwrap() = Order::new("A", 3.0, OrderDir::Buy, Some(start), Some(end));
    // Python discards the sent value at the tracking yield, including an action.
    let LiveNestedInnerProgress::Complete(done) = inner
        .resume_live_collect_data(NestedExecutorResume::Action(Some(999.0)))
        .unwrap()
    else {
        panic!("completion expected")
    };
    assert!(Arc::ptr_eq(&decision, &done.decision));
    let rows = done.executions.lock().unwrap();
    assert!(Arc::ptr_eq(&rows[0].order, &order));
    assert_eq!(
        rows[0].order.read().unwrap().deal_amount().to_bits(),
        3.0_f64.to_bits()
    );
    assert_eq!(inner.trade_step().unwrap(), 1);
    assert!(account.try_lock().is_ok());
    assert!(
        inner
            .resume_live_collect_data(NestedExecutorResume::Continue)
            .is_err()
    );
    inner.close_live_collect_data().unwrap();
}

#[test]
fn native_child_close_drop_and_failed_resume_never_reexecute_pending_orders() {
    let start = time("2024-01-02 09:30:00");
    let end = time("2024-01-02 09:30:59");
    for action in ["close", "drop", "failure"] {
        let calendar = Arc::new(SharedInnerCalendar::new(start, end));
        let account = Arc::new(Mutex::new(Account::new(InfinitePosition, false)));
        let mut inner = OwnedAtomicNestedInnerAdapter::new(
            calendar.clone(),
            collector(calendar.clone(), true),
            account.clone(),
            Arc::new(Services),
            "None",
            IndicatorConfig::default(),
            false,
        )
        .with_live_tracking(true);
        let (decision, order) = request(start, end);
        let weak = Arc::downgrade(&decision);
        assert!(matches!(
            inner.begin_live_collect_data(decision, 0).unwrap(),
            LiveNestedInnerProgress::Suspended(_)
        ));
        assert!(weak.upgrade().is_some());
        if action == "close" {
            inner.close_live_collect_data().unwrap();
            inner.close_live_collect_data().unwrap();
            assert!(
                inner
                    .resume_live_collect_data(NestedExecutorResume::Continue)
                    .is_err()
            );
        } else if action == "failure" {
            assert!(
                inner
                    .resume_live_collect_data(NestedExecutorResume::Continue)
                    .err()
                    .unwrap()
                    .message
                    .contains("deal failed")
            );
            assert!(
                inner
                    .resume_live_collect_data(NestedExecutorResume::Continue)
                    .err()
                    .unwrap()
                    .message
                    .contains("not suspended")
            );
        }
        drop(inner);
        assert!(weak.upgrade().is_none());
        assert_eq!(NestedCalendar::trade_step(&*calendar).unwrap(), 0);
        assert_eq!(
            order.read().unwrap().deal_amount().to_bits(),
            0.0_f64.to_bits()
        );
        assert!(account.try_lock().is_ok());
    }
}

#[test]
fn native_child_without_tracking_completes_immediately() {
    let start = time("2024-01-02 09:30:00");
    let end = time("2024-01-02 09:30:59");
    let calendar = Arc::new(SharedInnerCalendar::new(start, end));
    let account = Arc::new(Mutex::new(Account::new(InfinitePosition, false)));
    let mut inner = OwnedAtomicNestedInnerAdapter::new(
        calendar.clone(),
        collector(calendar, false),
        account,
        Arc::new(Services),
        "None",
        IndicatorConfig::default(),
        false,
    );
    assert_eq!(inner.live_control_mode(), NestedInnerControlMode::Framework);
    let (decision, _) = request(start, end);
    let LiveNestedInnerProgress::Complete(done) =
        inner.begin_live_collect_data(decision.clone(), 0).unwrap()
    else {
        panic!("completion expected")
    };
    assert!(Arc::ptr_eq(&decision, &done.decision));
    assert_eq!(done.executions.lock().unwrap().len(), 1);
    assert!(inner.finished().unwrap());
}

struct Origin;
impl SharedDecisionUpdateStrategy<()> for Origin {
    fn update_trade_decision(
        &self,
        _: &SharedLiveDecision<Self, ()>,
        _: &dyn domain_core::DecisionUpdateCalendar,
    ) -> Result<Option<SharedLiveDecision<Self, ()>>, domain_core::DecisionUpdateStrategyError>
    {
        Ok(None)
    }
}

pub(super) fn request(
    start: NaiveDateTime,
    end: NaiveDateTime,
) -> (LiveDecisionHandle, Arc<RwLock<Order>>) {
    let order = Arc::new(RwLock::new(Order::new(
        "A",
        2.0,
        OrderDir::Buy,
        Some(start),
        Some(end),
    )));
    let mut decision = SharedOrderDecisionConstruction::new(Arc::new(Origin));
    decision.base = Some(ConstructedDecisionBase {
        start_time: start,
        end_time: end,
        trade_range: None,
    });
    decision.orders = Some(Arc::new(RwLock::new(vec![DecisionOrderItem::Order(
        order.clone(),
    )])));
    decision.details = Some(());
    (Arc::new(RwLock::new(decision)), order)
}

#[test]
fn native_owned_child_returns_original_decision_and_live_orders_across_thread_move() {
    let start = time("2024-01-02 09:30:00");
    let end = time("2024-01-02 09:30:59");
    let calendar = Arc::new(SharedInnerCalendar::new(start, end));
    let account = Arc::new(Mutex::new(Account::new(
        domain_core::Position::from_initial(100.0, indexmap::IndexMap::new()),
        false,
    )));
    let prior = Arc::new(Mutex::new(Vec::new()));
    let collector = SimulatorCollector::new(
        "serial",
        calendar.clone(),
        Arc::new(RememberingDealer(prior.clone())),
        Arc::new(SilentReporter),
        false,
    );
    let mut inner: Box<dyn NestedInnerExecutor> = Box::new(OwnedAtomicNestedInnerAdapter::new(
        calendar,
        collector,
        account.clone(),
        Arc::new(Services),
        "cash",
        IndicatorConfig::default(),
        false,
    ));
    let (decision, order) = request(start, end);
    let completion = inner.collect_live_decision(decision.clone(), 0).unwrap();
    assert!(Arc::ptr_eq(&completion.decision, &decision));
    assert!(Arc::ptr_eq(
        &completion.executions.lock().unwrap()[0].order,
        &order
    ));
    order.write().unwrap().set_deal_amount(0.5);
    assert_eq!(
        completion.executions.lock().unwrap()[0]
            .order
            .read()
            .unwrap()
            .deal_amount()
            .to_bits(),
        0.5_f64.to_bits()
    );
    assert_eq!(
        inner
            .order_indicator_snapshot()
            .unwrap()
            .metric("deal_amount")
            .unwrap()
            .values()[0]
            .to_bits(),
        2.0_f64.to_bits()
    );
    assert!(account.try_lock().is_ok());
    let result = std::thread::spawn(move || {
        inner.reset_window(start, end).unwrap();
        let (next, _) = request(start, end);
        inner.collect_live_decision(next, 1).unwrap()
    })
    .join()
    .unwrap();
    assert_eq!(result.executions.lock().unwrap().len(), 1);
    assert_eq!(*prior.lock().unwrap(), [None, Some(2.0)]);
    assert_eq!(
        account
            .try_lock()
            .unwrap()
            .execution_position()
            .unwrap()
            .cash()
            .unwrap()
            .to_bits(),
        60.0_f64.to_bits()
    );
    drop(decision);
    assert!(Arc::ptr_eq(
        &completion.executions.lock().unwrap()[0].order,
        &order
    ));
}

#[test]
fn native_owned_child_maps_execution_and_poison_failures_without_advancing() {
    let start = time("2024-01-02 09:30:00");
    let end = time("2024-01-02 09:30:59");
    let calendar = Arc::new(SharedInnerCalendar::new(start, end));
    let account = Arc::new(Mutex::new(Account::new(InfinitePosition, false)));
    let mut inner = OwnedAtomicNestedInnerAdapter::new(
        calendar.clone(),
        collector(calendar, true),
        account.clone(),
        Arc::new(Services),
        "None",
        IndicatorConfig::default(),
        false,
    );
    let (decision, _) = request(start, end);
    assert!(
        inner
            .collect_live_decision(decision.clone(), 0)
            .err()
            .unwrap()
            .message
            .contains("deal failed")
    );
    assert_eq!(inner.trade_step().unwrap(), 0);
    assert!(account.try_lock().is_ok());
    let poisoned = account.clone();
    assert!(
        std::thread::spawn(move || {
            let _guard = poisoned.lock().unwrap();
            panic!("poison");
        })
        .join()
        .is_err()
    );
    assert_eq!(
        inner.collect_live_data(&decision, 0).err().unwrap().message,
        "owned atomic account lock poisoned"
    );
    assert_eq!(inner.trade_step().unwrap(), 0);
}

#[test]
fn legacy_child_rejects_live_input_without_implicit_order_snapshot() {
    let start = time("2024-01-02 09:30:00");
    let end = time("2024-01-02 09:30:59");
    let calendar = Arc::new(SharedInnerCalendar::new(start, end));
    let mut collector = collector(calendar.clone(), false);
    let mut account = Account::new(InfinitePosition, false);
    let mut adapter = AtomicAccountAdapter::new(&mut account, &Services, false);
    let mut legacy = AtomicNestedInnerAdapter::new(
        calendar,
        &mut collector,
        &mut adapter,
        "None",
        IndicatorConfig::default(),
    );
    let (decision, order) = request(start, end);
    assert_eq!(
        legacy.live_control_mode(),
        NestedInnerControlMode::Framework
    );
    assert!(
        legacy
            .begin_live_collect_data(decision.clone(), 0)
            .err()
            .unwrap()
            .message
            .contains("does not support live")
    );
    assert!(
        legacy
            .resume_live_collect_data(NestedExecutorResume::Continue)
            .err()
            .unwrap()
            .message
            .contains("not suspended")
    );
    legacy.close_live_collect_data().unwrap();
    assert_eq!(
        legacy
            .collect_live_decision(decision, 0)
            .err()
            .unwrap()
            .message,
        "inner executor does not support live decisions"
    );
    assert_eq!(legacy.trade_step().unwrap(), 0);
    assert_eq!(
        order.read().unwrap().deal_amount().to_bits(),
        0.0_f64.to_bits()
    );
}
