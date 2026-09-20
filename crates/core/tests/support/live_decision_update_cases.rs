use super::*;
use domain_core::decision_construction::{DecisionTotalStep, SharedOrderDecisionConstruction};
use domain_core::decision_update::{
    SharedDecisionUpdateError, SharedDecisionUpdateStrategy, SharedLiveDecision,
    update_shared_trade_decision,
};
use std::sync::RwLock;

struct RejectRangeAssignment(domain_core::decision_update::LiveDecisionHandle);

impl domain_core::decision_update::LiveDecision for RejectRangeAssignment {
    fn base(
        &self,
    ) -> Result<
        domain_core::decision_construction::ConstructedDecisionBase,
        domain_core::decision_update::LiveDecisionAccessError,
    > {
        self.0.base()
    }
    fn total_step(
        &self,
    ) -> Result<DecisionTotalStep, domain_core::decision_update::LiveDecisionAccessError> {
        self.0.total_step()
    }
    fn inherit_range(
        &self,
        range: Option<domain_core::SharedTradeRange>,
    ) -> Result<(), domain_core::decision_update::LiveDecisionAccessError> {
        self.0.inherit_range(range)?;
        Err(domain_core::decision_update::LiveDecisionAccessError::DecisionPoisoned)
    }
    fn orders(
        &self,
    ) -> Result<
        domain_core::decision_construction::SharedDecisionOrders,
        domain_core::decision_update::LiveDecisionAccessError,
    > {
        self.0.orders()
    }
    fn is_empty(&self) -> Result<bool, domain_core::decision_update::LiveDecisionAccessError> {
        self.0.is_empty()
    }
    fn update(
        self: Arc<Self>,
        calendar: &dyn DecisionUpdateCalendar,
    ) -> Result<Option<domain_core::decision_update::LiveDecisionHandle>, SharedDecisionUpdateError>
    {
        self.0.clone().update(calendar)
    }
}

#[test]
fn propagation_preserves_plugin_assignment_failure_after_mutation() {
    use domain_core::decision_construction::ConstructedDecisionBase;
    use domain_core::decision_update::{LiveDecisionAccessError, LiveDecisionHandle};
    let outer = new_decision("none", &Arc::default());
    let inner = new_decision("none", &Arc::default());
    let rule: domain_core::SharedTradeRange = Arc::new(domain_core::IdxTradeRange::new(2, 4));
    for (decision, range) in [(&outer, Some(rule.clone())), (&inner, None)] {
        decision.write().unwrap().base = Some(ConstructedDecisionBase {
            start_time: at("2020-01-01 09:30:00"),
            end_time: at("2020-01-01 15:00:00"),
            trade_range: range,
        });
    }
    let outer: LiveDecisionHandle = outer;
    let original: LiveDecisionHandle = inner;
    let failing: LiveDecisionHandle = Arc::new(RejectRangeAssignment(original.clone()));
    assert_eq!(
        outer.modify_inner_decision(&failing),
        Err(LiveDecisionAccessError::DecisionPoisoned)
    );
    assert!(Arc::ptr_eq(
        original.base().unwrap().trade_range.as_ref().unwrap(),
        &rule
    ));
}

struct ReentrantRange {
    decision: std::sync::Weak<RwLock<SharedOrderDecisionConstruction<Arc<LiveStrategy>, String>>>,
    action: &'static str,
}

impl domain_core::TradeRange for ReentrantRange {
    fn range_indices(
        &self,
        _: Option<&dyn domain_core::TradeCalendarRange>,
    ) -> Result<(i64, i64), domain_core::TradeRangeError> {
        let decision = self.decision.upgrade().unwrap();
        let mut guard = decision
            .try_write()
            .expect("range callback can mutate original decision");
        guard.total_step = DecisionTotalStep::Value(3);
        match self.action {
            "missing" => Err(domain_core::TradeRangeError::MissingCalendar),
            "fail" => Err(domain_core::TradeRangeError::IndexTimeClippingUnsupported),
            "poison" => {
                drop(guard);
                assert!(
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let _guard = decision.write().unwrap();
                        panic!("poison during range evaluation");
                    }))
                    .is_err()
                );
                Ok((-2, 10))
            }
            _ => Ok((-2, 10)),
        }
    }
    fn clip_time_range(
        &self,
        _: NaiveDateTime,
        _: NaiveDateTime,
    ) -> Result<(NaiveDateTime, NaiveDateTime), domain_core::TradeRangeError> {
        Err(domain_core::TradeRangeError::IndexTimeClippingUnsupported)
    }
}

#[test]
fn shared_range_uses_post_callback_state_and_preserves_fallback_failure_order() {
    use domain_core::decision_construction::ConstructedDecisionBase;
    use domain_core::decision_update::{LiveDecisionAccessError, LiveDecisionHandle};
    use domain_core::{IdxTradeRange, RangeLimitDefault, TradeDecisionError, TradeRangeError};
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/live_decision_tracking_contract.py"
        ))
        .output()
        .unwrap();
    assert!(output.status.success());
    let expected: Value = serde_json::from_slice(&output.stdout).unwrap();
    for action in ["ok", "missing", "fail", "poison"] {
        let decision = new_decision("none", &Arc::default());
        let erased: LiveDecisionHandle = decision.clone();
        assert_eq!(
            erased.range_limit(None, RangeLimitDefault::Error),
            Err(LiveDecisionAccessError::MissingBase)
        );
        decision.write().unwrap().base = Some(ConstructedDecisionBase {
            start_time: at("2020-01-01 09:30:00"),
            end_time: at("2020-01-01 15:00:00"),
            trade_range: None,
        });
        assert_eq!(
            erased
                .range_limit(None, RangeLimitDefault::Value(Some((-8, 99))))
                .unwrap(),
            Some((-8, 99))
        );
        assert_eq!(
            erased.range_limit(None, RangeLimitDefault::Error),
            Err(LiveDecisionAccessError::Range(
                TradeDecisionError::MissingRange
            ))
        );
        decision.write().unwrap().base.as_mut().unwrap().trade_range =
            Some(Arc::new(IdxTradeRange::new(-2, 10)));
        for state in [
            DecisionTotalStep::Missing,
            DecisionTotalStep::Unset,
            DecisionTotalStep::Value(99),
            DecisionTotalStep::Value(0),
        ] {
            decision.write().unwrap().total_step = state;
            let result = erased.range_limit(None, RangeLimitDefault::Error).unwrap();
            let expected = match state {
                DecisionTotalStep::Value(0) => (0, -1),
                DecisionTotalStep::Value(_) => (0, 10),
                _ => (-2, 10),
            };
            assert_eq!(result, Some(expected));
        }
        decision.write().unwrap().total_step = DecisionTotalStep::Value(99);
        decision.write().unwrap().base.as_mut().unwrap().trade_range =
            Some(Arc::new(ReentrantRange {
                decision: Arc::downgrade(&decision),
                action,
            }));
        let result = erased.range_limit(None, RangeLimitDefault::Value(Some((-8, 99))));
        match action {
            "ok" => assert_eq!(json!(result.unwrap().unwrap()), expected["live_range"]),
            "missing" => {
                assert_eq!(result.unwrap(), Some((-8, 99)));
                assert_eq!(
                    erased.range_limit(None, RangeLimitDefault::Error),
                    Err(LiveDecisionAccessError::Range(
                        TradeDecisionError::MissingRange
                    ))
                );
            }
            "fail" => assert_eq!(
                result,
                Err(LiveDecisionAccessError::Range(
                    TradeDecisionError::TradeRange(TradeRangeError::IndexTimeClippingUnsupported)
                ))
            ),
            "poison" => assert_eq!(result, Err(LiveDecisionAccessError::DecisionPoisoned)),
            _ => unreachable!(),
        }
    }
}

#[test]
fn default_live_propagation_matches_source_lazy_access_and_self_aliasing() {
    use domain_core::decision_construction::ConstructedDecisionBase;
    use domain_core::decision_update::{LiveDecisionAccessError, LiveDecisionHandle};
    let source = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/live_decision_tracking_contract.py"
        ))
        .output()
        .unwrap();
    assert!(source.status.success());
    let expected: Value = serde_json::from_slice(&source.stdout).unwrap();
    let outer = new_decision("none", &Arc::default());
    let inner = new_decision("none", &Arc::default());
    let outer_handle: LiveDecisionHandle = outer.clone();
    let inner_handle: LiveDecisionHandle = inner.clone();
    let missing: LiveDecisionHandle = new_decision("none", &Arc::default());
    let rule: domain_core::SharedTradeRange = Arc::new(domain_core::IdxTradeRange::new(2, 4));
    let base = |range| ConstructedDecisionBase {
        start_time: at("2020-01-01 09:30:00"),
        end_time: at("2020-01-01 15:00:00"),
        trade_range: range,
    };
    outer.write().unwrap().base = Some(base(Some(rule.clone())));
    inner.write().unwrap().base = Some(base(None));
    outer_handle.modify_inner_decision(&inner_handle).unwrap();
    let retained = Arc::ptr_eq(
        inner_handle.base().unwrap().trade_range.as_ref().unwrap(),
        &rule,
    );
    missing.modify_inner_decision(&inner_handle).unwrap();
    let preserved = Arc::ptr_eq(
        inner_handle.base().unwrap().trade_range.as_ref().unwrap(),
        &rule,
    );
    inner.write().unwrap().base.as_mut().unwrap().trade_range = None;
    inner_handle.modify_inner_decision(&inner_handle).unwrap();
    let alias = inner_handle.base().unwrap().trade_range.is_none();
    let missing_inner =
        outer_handle.modify_inner_decision(&missing) == Err(LiveDecisionAccessError::MissingBase);
    let missing_outer =
        missing.modify_inner_decision(&inner_handle) == Err(LiveDecisionAccessError::MissingBase);
    assert_eq!(
        json!([retained, preserved, alias, missing_inner, missing_outer]),
        expected["propagation"]
    );
}

#[test]
fn erased_metadata_is_live_and_range_inheritance_preserves_rule_identity() {
    use domain_core::decision_construction::ConstructedDecisionBase;
    use domain_core::decision_update::{LiveDecisionAccessError, LiveDecisionHandle};
    let decision = new_decision("none", &Arc::default());
    let erased: LiveDecisionHandle = decision.clone();
    assert_eq!(
        erased.base().err(),
        Some(LiveDecisionAccessError::MissingBase)
    );
    assert_eq!(
        erased.inherit_range(None),
        Err(LiveDecisionAccessError::MissingBase)
    );
    let start = at("2020-01-01 09:30:00");
    let end = at("2020-01-01 15:00:00");
    decision.write().unwrap().base = Some(ConstructedDecisionBase {
        start_time: start,
        end_time: end,
        trade_range: None,
    });
    erased.inherit_range(None).unwrap();
    assert!(erased.base().unwrap().trade_range.is_none());
    let rule: domain_core::SharedTradeRange = Arc::new(domain_core::IdxTradeRange::new(2, 4));
    erased.inherit_range(Some(rule.clone())).unwrap();
    assert!(Arc::ptr_eq(
        erased.base().unwrap().trade_range.as_ref().unwrap(),
        &rule
    ));
    let other: domain_core::SharedTradeRange = Arc::new(domain_core::IdxTradeRange::new(8, 9));
    erased.inherit_range(Some(other)).unwrap();
    erased.inherit_range(None).unwrap();
    let view = erased.base().unwrap();
    assert_eq!((view.start_time, view.end_time), (start, end));
    assert!(Arc::ptr_eq(view.trade_range.as_ref().unwrap(), &rule));
    decision.write().unwrap().base.as_mut().unwrap().start_time = end;
    assert_eq!(erased.base().unwrap().start_time, end);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = decision.write().unwrap();
            panic!("poison metadata");
        }))
        .is_err()
    );
    assert_eq!(
        erased.base().err(),
        Some(LiveDecisionAccessError::DecisionPoisoned)
    );
    assert_eq!(
        erased.inherit_range(None),
        Err(LiveDecisionAccessError::DecisionPoisoned)
    );
}

#[test]
fn erased_transport_retains_allocation_and_live_membership() {
    use domain_core::decision_construction::{DecisionAccessError, DecisionOrderItem};
    use domain_core::decision_update::{LiveDecisionAccessError, LiveDecisionHandle};
    for action in ["none", "self", "replace", "fail"] {
        let events = Arc::default();
        let decision = new_decision(action, &events);
        let erased: LiveDecisionHandle = decision.clone();
        assert_eq!(
            Arc::as_ptr(&decision).cast::<()>(),
            Arc::as_ptr(&erased).cast::<()>()
        );
        let missing = LiveDecisionAccessError::Access(DecisionAccessError::MissingOrders);
        assert_eq!(erased.orders().err(), Some(missing));
        assert_eq!(
            erased.is_empty(),
            Err(LiveDecisionAccessError::Access(
                DecisionAccessError::MissingOrders
            ))
        );
        let orders = Arc::new(RwLock::new(Vec::new()));
        decision.write().unwrap().orders = Some(orders.clone());
        assert!(Arc::ptr_eq(&erased.orders().unwrap(), &orders));
        assert!(erased.is_empty().unwrap());
        orders
            .write()
            .unwrap()
            .push(DecisionOrderItem::Other(Arc::new(42)));
        assert_eq!(erased.orders().unwrap().read().unwrap().len(), 1);
        assert!(erased.is_empty().unwrap());
        let replacement_list = Arc::new(RwLock::new(Vec::new()));
        decision.write().unwrap().orders = Some(replacement_list.clone());
        assert!(Arc::ptr_eq(&erased.orders().unwrap(), &replacement_list));
        let calendar = LiveCalendar {
            decision: decision.clone(),
            events,
            fail: false,
        };
        let result = erased.clone().update(&calendar);
        match action {
            "none" => assert!(result.unwrap().is_none()),
            "self" => assert!(Arc::ptr_eq(&result.unwrap().unwrap(), &erased)),
            "replace" => assert!(!Arc::ptr_eq(&result.unwrap().unwrap(), &erased)),
            "fail" => assert!(matches!(
                result,
                Err(SharedDecisionUpdateError::Strategy(_))
            )),
            _ => unreachable!(),
        }
        assert_eq!(decision.read().unwrap().details.as_deref(), Some("changed"));
    }
}

#[test]
fn erased_transport_reports_poison_without_accessing_strategy() {
    use domain_core::decision_update::{LiveDecisionAccessError, LiveDecisionHandle};
    let events = Arc::default();
    let decision = new_decision("self", &events);
    let erased: LiveDecisionHandle = decision.clone();
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = decision.write().unwrap();
            panic!("poison");
        }))
        .is_err()
    );
    assert_eq!(
        erased.orders().err(),
        Some(LiveDecisionAccessError::DecisionPoisoned)
    );
    assert_eq!(
        erased.is_empty(),
        Err(LiveDecisionAccessError::DecisionPoisoned)
    );
    let calendar = Calendar {
        events: Arc::default(),
        result: Ok(7),
    };
    assert_eq!(
        erased.update(&calendar).err(),
        Some(SharedDecisionUpdateError::DecisionPoisoned)
    );
    assert!(events.lock().unwrap().is_empty());
}

struct LiveStrategy {
    action: &'static str,
    events: Arc<Mutex<Vec<Value>>>,
}

impl SharedDecisionUpdateStrategy<String> for LiveStrategy {
    fn update_trade_decision(
        &self,
        decision: &SharedLiveDecision<Self, String>,
        _calendar: &dyn DecisionUpdateCalendar,
    ) -> Result<Option<SharedLiveDecision<Self, String>>, DecisionUpdateStrategyError> {
        let mut current = decision
            .try_write()
            .expect("no framework guard across callback");
        assert!(std::ptr::eq(self, current.strategy.as_ref()));
        assert_eq!(current.total_step, DecisionTotalStep::Value(7));
        self.events.lock().unwrap().push(json!(["strategy", 7]));
        current.details = Some("changed".to_owned());
        drop(current);
        match self.action {
            "fail" => Err(DecisionUpdateStrategyError {
                message: "strategy".to_owned(),
            }),
            "self" => Ok(Some(Arc::clone(decision))),
            "replace" => Ok(Some(new_decision("none", &self.events))),
            "none" => Ok(None),
            _ => panic!("test action"),
        }
    }
}

fn new_decision(
    action: &'static str,
    events: &Arc<Mutex<Vec<Value>>>,
) -> SharedLiveDecision<LiveStrategy, String> {
    Arc::new(RwLock::new(SharedOrderDecisionConstruction::new(Arc::new(
        LiveStrategy {
            action,
            events: Arc::clone(events),
        },
    ))))
}

struct LiveCalendar {
    decision: SharedLiveDecision<LiveStrategy, String>,
    events: Arc<Mutex<Vec<Value>>>,
    fail: bool,
}

impl DecisionUpdateCalendar for LiveCalendar {
    fn trade_len(&self) -> Result<i64, DecisionUpdateCalendarError> {
        assert!(
            self.decision.try_write().is_ok(),
            "calendar may reenter decision"
        );
        self.events.lock().unwrap().push(json!("calendar"));
        if self.fail {
            Err(calendar_error("calendar"))
        } else {
            Ok(7)
        }
    }
}

#[test]
fn original_shared_strategy_update_matches_partial_source_and_retains_identity() {
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/live_decision_tracking_contract.py"
        ))
        .output()
        .unwrap();
    assert!(output.status.success());
    let source: Value = serde_json::from_slice(&output.stdout).unwrap();
    for (calendar_fail, action, source_index) in
        [(true, "none", 0), (false, "fail", 1), (false, "self", 2)]
    {
        let events = Arc::default();
        let decision = new_decision(action, &events);
        let calendar = LiveCalendar {
            decision: Arc::clone(&decision),
            events: Arc::clone(&events),
            fail: calendar_fail,
        };
        let result = update_shared_trade_decision(&decision, &calendar);
        if calendar_fail {
            assert_eq!(
                result.err(),
                Some(SharedDecisionUpdateError::Calendar(calendar_error(
                    "calendar"
                )))
            );
        } else if action == "fail" {
            assert_eq!(
                result.err(),
                Some(SharedDecisionUpdateError::Strategy(
                    DecisionUpdateStrategyError {
                        message: "strategy".into()
                    }
                ))
            );
        } else {
            assert!(Arc::ptr_eq(&result.unwrap().unwrap(), &decision));
        }
        let current = decision.read().unwrap();
        assert!(current.base.is_none());
        assert!(current.orders.is_none());
        let total = match current.total_step {
            DecisionTotalStep::Value(value) => Some(value),
            _ => None,
        };
        assert_eq!(
            json!({"events": *events.lock().unwrap(), "total_step": total, "marker": current.details}),
            source["update"][source_index]
        );
    }
}

#[test]
fn shared_update_distinguishes_none_replacement_and_poison_after_calendar() {
    for action in ["none", "replace"] {
        let events = Arc::default();
        let decision = new_decision(action, &events);
        let calendar = LiveCalendar {
            decision: Arc::clone(&decision),
            events,
            fail: false,
        };
        let returned = update_shared_trade_decision(&decision, &calendar).unwrap();
        if action == "none" {
            assert!(returned.is_none());
        } else {
            let replacement = returned.unwrap();
            assert!(!Arc::ptr_eq(&replacement, &decision));
            assert_eq!(
                replacement.read().unwrap().total_step,
                DecisionTotalStep::Missing
            );
        }
        assert_eq!(decision.read().unwrap().details.as_deref(), Some("changed"));
    }
    let events = Arc::default();
    let decision = new_decision("self", &events);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = decision.write().unwrap();
            panic!("poison decision");
        }))
        .is_err()
    );
    let calendar_events = Arc::default();
    let calendar = Calendar {
        events: Arc::clone(&calendar_events),
        result: Ok(7),
    };
    assert_eq!(
        update_shared_trade_decision(&decision, &calendar).err(),
        Some(SharedDecisionUpdateError::DecisionPoisoned)
    );
    assert_eq!(*calendar_events.lock().unwrap(), ["calendar"]);
    assert!(events.lock().unwrap().is_empty());
    assert_eq!(
        decision.read().err().unwrap().into_inner().total_step,
        DecisionTotalStep::Missing
    );
}
