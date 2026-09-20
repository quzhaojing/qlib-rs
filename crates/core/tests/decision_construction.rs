use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    process::Command,
    sync::{Arc, Mutex, RwLock},
};

use chrono::{NaiveDateTime, TimeDelta};
use domain_core::decision_construction::{
    DecisionAccessError, DecisionConstructionError, DecisionOrderItem, DecisionRangeInput,
    SaoeDecisionOrigin, SharedDecisionOrders, SharedOrderDecisionConstruction,
};
use domain_core::{
    IdxTradeRange, Order, OrderDir, SaoeCalendar, SaoePluginError, SharedTradeRange,
};
use serde_json::{Value, json};

fn time(day: i64) -> NaiveDateTime {
    NaiveDateTime::parse_from_str("2024-01-01 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        + TimeDelta::days(day)
}

struct UpdateCalendar(Result<i64, domain_core::DecisionUpdateCalendarError>);

impl domain_core::DecisionUpdateCalendar for UpdateCalendar {
    fn trade_len(&self) -> Result<i64, domain_core::DecisionUpdateCalendarError> {
        self.0.clone()
    }
}

#[test]
fn total_step_assignment_is_independent_and_reinitialization_keeps_source_failure_order() {
    use domain_core::decision_construction::DecisionTotalStep;
    assert_eq!(DecisionTotalStep::default(), DecisionTotalStep::Missing);
    let failure = domain_core::DecisionUpdateCalendarError {
        message: "calendar".to_owned(),
    };
    for initial in [
        DecisionTotalStep::Missing,
        DecisionTotalStep::Unset,
        DecisionTotalStep::Value(99),
    ] {
        let mut state = SharedOrderDecisionConstruction::<_, ()>::new(());
        state.total_step = initial;
        assert_eq!(
            state.refresh_total_step(&UpdateCalendar(Err(failure.clone()))),
            Err(failure.clone())
        );
        assert_eq!(state.total_step, initial);
        for value in [7, 0, -2] {
            state
                .refresh_total_step(&UpdateCalendar(Ok(value)))
                .unwrap();
            assert_eq!(state.total_step, DecisionTotalStep::Value(value));
            assert!(state.base.is_none());
            assert!(state.orders.is_none());
            assert!(state.details.is_none());
        }
    }
    for fail in [Some(1), Some(2), None] {
        let calendar = Calendar {
            fail,
            ..Calendar::default()
        };
        let mut state = SharedOrderDecisionConstruction::new(SaoeDecisionOrigin {
            strategy: &(),
            calendar: &calendar,
        });
        state.refresh_total_step(&UpdateCalendar(Ok(7))).unwrap();
        let orders = Arc::default();
        let result = state.initialize(&orders, None, ());
        assert_eq!(result.is_err(), fail.is_some());
        assert_eq!(
            state.total_step,
            if fail == Some(1) {
                DecisionTotalStep::Value(7)
            } else {
                DecisionTotalStep::Unset
            }
        );
        assert_eq!(state.base.is_some(), fail != Some(1));
    }
}

#[derive(Default)]
struct Calendar {
    calls: Mutex<i64>,
    fail: Option<i64>,
}

impl SaoeCalendar for Calendar {
    fn available_step_range(&self) -> Result<(i64, i64), SaoePluginError> {
        panic!("construction must not read the range")
    }

    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), SaoePluginError> {
        let mut calls = self.calls.lock().unwrap();
        *calls += 1;
        if self.fail == Some(*calls) {
            return Err(SaoePluginError {
                message: format!("calendar-{calls}"),
            });
        }
        Ok((time(*calls), time(*calls) + TimeDelta::hours(1)))
    }
}

fn order() -> Arc<RwLock<Order>> {
    Arc::new(RwLock::new(Order::new(
        "A",
        1.0,
        OrderDir::Buy,
        None,
        Some(time(20)),
    )))
}

#[test]
fn real_source_construction_and_partial_failures_match() {
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/decision_surface_audit.py"
        ))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let source: Value = serde_json::from_slice(&output.stdout).unwrap();
    for (name, fail, invalid) in [
        ("success", None, false),
        ("first_failure", Some(1), false),
        ("second_failure", Some(2), false),
        ("partial_failure", None, true),
    ] {
        let calendar = Calendar {
            fail,
            ..Calendar::default()
        };
        let strategy = Arc::new("source-strategy".to_owned());
        let origin = SaoeDecisionOrigin {
            strategy: &strategy,
            calendar: &calendar,
        };
        let first = order();
        let orders: SharedDecisionOrders = Arc::new(RwLock::new(vec![DecisionOrderItem::Order(
            Arc::clone(&first),
        )]));
        if invalid {
            orders
                .write()
                .unwrap()
                .push(DecisionOrderItem::Other(Arc::new("invalid")));
        }
        let details = Arc::new("details".to_owned());
        let mut state = SharedOrderDecisionConstruction::new(origin);
        let result = state.initialize(
            &orders,
            Some(DecisionRangeInput::Indices(2, 5)),
            Arc::clone(&details),
        );
        let error = match result {
            Ok(()) => None,
            Err(DecisionConstructionError::Calendar(error)) => {
                Some(format!("RuntimeError:{}", error.message))
            }
            Err(DecisionConstructionError::InvalidOrder(_)) => Some("AssertionError:".to_owned()),
            Err(error) => panic!("unexpected {error}"),
        };
        let first = first.read().unwrap();
        assert_eq!(
            json!({
                "calls": *calendar.calls.lock().unwrap(),
                "decision_start": state.base.as_ref().map(|base| {
                    assert_eq!(base.start_time, time(1));
                    assert_eq!(base.end_time, time(1) + TimeDelta::hours(1));
                    assert_eq!(state.total_step, domain_core::decision_construction::DecisionTotalStep::Unset);
                    assert_eq!(base.trade_range.as_ref().unwrap().range_indices(None).unwrap(), (2, 5));
                    "start-1"
                }),
                "order_start": first.start_time().map(|start| { assert_eq!(start, time(2)); "start-2" }),
                "order_end": ({ assert_eq!(first.end_time(), Some(time(20))); "explicit-end" }),
                "strategy_identity": Arc::ptr_eq(state.strategy.strategy, &strategy),
                "order_list_identity": state.orders.as_ref().is_some_and(|actual| Arc::ptr_eq(actual, &orders)),
                "has_total_step": state.total_step != domain_core::decision_construction::DecisionTotalStep::Missing,
                "has_details": state.details.is_some(),
                "error": error,
            }),
            source[name],
            "{name}"
        );
        if let Some(actual) = state.details.as_ref() {
            assert!(Arc::ptr_eq(actual, &details));
        }
    }
}

#[test]
fn shared_identity_empty_lists_defaults_and_reinitialization_are_observable() {
    let calendar = Calendar::default();
    let origin = SaoeDecisionOrigin {
        strategy: &(),
        calendar: &calendar,
    };
    let mut state = SharedOrderDecisionConstruction::new(origin);
    let first = order();
    let orders: SharedDecisionOrders = Arc::new(RwLock::new(vec![
        DecisionOrderItem::Order(Arc::clone(&first)),
        DecisionOrderItem::Order(Arc::clone(&first)),
    ]));
    let range: SharedTradeRange = Arc::new(IdxTradeRange::new(-2, 9));
    state
        .initialize(
            &orders,
            Some(DecisionRangeInput::Rule(Arc::clone(&range))),
            None::<String>,
        )
        .unwrap();
    assert!(Arc::ptr_eq(
        state.base.as_ref().unwrap().trade_range.as_ref().unwrap(),
        &range
    ));
    assert_eq!(state.details, Some(None));
    first.write().unwrap().set_deal_amount(3.0);
    let list = state.orders.as_ref().unwrap().read().unwrap();
    for item in list.iter() {
        let DecisionOrderItem::Order(handle) = item else {
            panic!("order expected")
        };
        assert!(Arc::ptr_eq(handle, &first));
        assert_eq!(
            handle.read().unwrap().deal_amount().to_bits(),
            3.0_f64.to_bits()
        );
    }
    drop(list);
    orders
        .write()
        .unwrap()
        .push(DecisionOrderItem::Other(Arc::new("late-invalid")));
    let result = state.initialize(&orders, None, Some("new".to_owned()));
    assert!(matches!(
        result,
        Err(DecisionConstructionError::InvalidOrder(2))
    ));
    assert_eq!(state.details, Some(None));
    assert!(state.base.as_ref().unwrap().trade_range.is_none());
    assert_eq!(first.read().unwrap().start_time(), Some(time(2)));
    let invalid = orders.read().unwrap();
    let DecisionOrderItem::Other(payload) = &invalid[2] else {
        panic!("original invalid object expected")
    };
    assert_eq!(payload.downcast_ref::<&str>(), Some(&"late-invalid"));
    drop(invalid);
    orders.write().unwrap().clear();
    state
        .initialize(&orders, None, Some("final".to_owned()))
        .unwrap();
    assert_eq!(*calendar.calls.lock().unwrap(), 6);
    assert_eq!(state.details, Some(Some("final".to_owned())));
    assert!(state.orders.as_ref().unwrap().read().unwrap().is_empty());
}

#[test]
fn poisoned_storage_reports_failure_after_prior_mutations() {
    let calendar = Calendar::default();
    let origin = SaoeDecisionOrigin {
        strategy: &(),
        calendar: &calendar,
    };
    let mut state = SharedOrderDecisionConstruction::new(origin);
    let first = order();
    let poisoned = order();
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _guard = poisoned.write().unwrap();
            panic!("poison order");
        }))
        .is_err()
    );
    let orders = Arc::new(RwLock::new(vec![
        DecisionOrderItem::Order(Arc::clone(&first)),
        DecisionOrderItem::Order(poisoned),
    ]));
    assert!(matches!(
        state.initialize(&orders, None, ()),
        Err(DecisionConstructionError::OrderPoisoned(1))
    ));
    assert_eq!(first.read().unwrap().start_time(), Some(time(2)));
    assert!(state.details.is_none());
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _guard = orders.write().unwrap();
            panic!("poison list");
        }))
        .is_err()
    );
    assert!(matches!(
        state.initialize(&orders, None, ()),
        Err(DecisionConstructionError::ListPoisoned)
    ));
    assert!(Arc::ptr_eq(state.orders.as_ref().unwrap(), &orders));
    assert_eq!(*calendar.calls.lock().unwrap(), 4);
}

// Exercise the same observable failure contract for opaque, optional and unit detail payloads.
// The payload type must not change constructor ordering or error-side-effect behavior.
fn payload_lifecycle<S: ?Sized, D: Clone>(strategy: &S, details: D) {
    for (fail, expected_calls) in [(Some(1), 1), (Some(2), 2), (None, 2)] {
        let calendar = Calendar {
            fail,
            ..Calendar::default()
        };
        let origin = SaoeDecisionOrigin {
            strategy,
            calendar: &calendar,
        };
        let mut state = SharedOrderDecisionConstruction::new(origin);
        let first = order();
        let orders = Arc::new(RwLock::new(vec![DecisionOrderItem::Order(Arc::clone(
            &first,
        ))]));
        let result = state.initialize(
            &orders,
            Some(DecisionRangeInput::Indices(0, 1)),
            details.clone(),
        );
        assert_eq!(*calendar.calls.lock().unwrap(), expected_calls);
        if fail.is_some() {
            assert!(matches!(
                result,
                Err(DecisionConstructionError::Calendar(_))
            ));
            assert!(first.read().unwrap().start_time().is_none());
            assert!(state.details.is_none());
        } else {
            result.unwrap_or_else(|_| panic!("successful calendar must initialize"));
            assert_eq!(first.read().unwrap().start_time(), Some(time(2)));
            assert!(state.details.is_some());
        }
    }
    let calendar = Calendar::default();
    let origin = SaoeDecisionOrigin {
        strategy,
        calendar: &calendar,
    };
    let mut state = SharedOrderDecisionConstruction::new(origin);
    let orders = Arc::new(RwLock::new(vec![]));
    let range: SharedTradeRange = Arc::new(IdxTradeRange::new(0, 3));
    state
        .initialize(
            &orders,
            Some(DecisionRangeInput::Rule(Arc::clone(&range))),
            details.clone(),
        )
        .unwrap_or_else(|_| panic!("empty list must initialize"));
    assert!(Arc::ptr_eq(
        state.base.as_ref().unwrap().trade_range.as_ref().unwrap(),
        &range
    ));
    orders
        .write()
        .unwrap()
        .push(DecisionOrderItem::Other(Arc::new("invalid")));
    assert!(matches!(
        state.initialize(&orders, None, details.clone()),
        Err(DecisionConstructionError::InvalidOrder(0))
    ));
    let poisoned = order();
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _guard = poisoned.write().unwrap();
            panic!("poison order");
        }))
        .is_err()
    );
    orders.write().unwrap()[0] = DecisionOrderItem::Order(poisoned);
    assert!(matches!(
        state.initialize(&orders, None, details.clone()),
        Err(DecisionConstructionError::OrderPoisoned(0))
    ));
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _guard = orders.write().unwrap();
            panic!("poison list");
        }))
        .is_err()
    );
    assert!(matches!(
        state.initialize(&orders, None, details),
        Err(DecisionConstructionError::ListPoisoned)
    ));
    assert!(state.details.is_some());
}

#[test]
fn payload_representation_does_not_change_initialization_or_failure_behavior() {
    payload_lifecycle(
        &Arc::new("strategy".to_owned()),
        Arc::new("details".to_owned()),
    );
    payload_lifecycle(&(), None::<String>);
    payload_lifecycle(&(), ());
}

#[test]
fn live_decision_access_matches_source_membership_thresholds_and_replacement() {
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/decision_surface_audit.py"
        ))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let source: Value = serde_json::from_slice(&output.stdout).unwrap();
    let mut state = SharedOrderDecisionConstruction::<_, ()>::new(());
    assert!(state.get_decision().is_none());
    assert_eq!(state.is_empty(), Err(DecisionAccessError::MissingOrders));
    let orders: SharedDecisionOrders = Arc::default();
    state.orders = Some(Arc::clone(&orders));
    let cases: &[&[Option<f64>]] = &[
        &[],
        &[Some(0.0)],
        &[Some(1e-6)],
        &[Some(1.000_001e-6)],
        &[Some(f64::NAN)],
        &[Some(f64::NEG_INFINITY)],
        &[Some(f64::INFINITY)],
        &[None, Some(1.0)],
        &[Some(1.0), None],
        &[Some(0.0), None, Some(1.0)],
    ];
    let mut actual = Vec::new();
    for amounts in cases {
        *orders.write().unwrap() = amounts
            .iter()
            .map(|amount| match amount {
                Some(amount) => DecisionOrderItem::Order(Arc::new(RwLock::new(Order::new(
                    "A",
                    *amount,
                    OrderDir::Buy,
                    None,
                    None,
                )))),
                None => DecisionOrderItem::Other(Arc::new(())),
            })
            .collect();
        assert!(Arc::ptr_eq(state.get_decision().unwrap(), &orders));
        actual.push(state.is_empty().unwrap());
    }
    assert_eq!(json!(actual), source["live_empty"]);
    let original = order();
    let replacement = Arc::new(RwLock::new(vec![DecisionOrderItem::Order(Arc::clone(
        &original,
    ))]));
    state.orders = Some(Arc::clone(&replacement));
    assert!(Arc::ptr_eq(state.get_decision().unwrap(), &replacement));
    assert!(!Arc::ptr_eq(state.get_decision().unwrap(), &orders));
    assert!(!state.is_empty().unwrap());
    *original.write().unwrap() = Order::new("A", 0.0, OrderDir::Buy, None, None);
    assert!(state.is_empty().unwrap());
}

#[test]
fn live_decision_access_only_locks_items_reached_before_short_circuit() {
    let poisoned = order();
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _guard = poisoned.write().unwrap();
            panic!("poison order");
        }))
        .is_err()
    );
    let orders = Arc::new(RwLock::new(vec![
        DecisionOrderItem::Other(Arc::new(())),
        DecisionOrderItem::Order(Arc::clone(&poisoned)),
    ]));
    let mut state = SharedOrderDecisionConstruction::<_, ()>::new(());
    state.orders = Some(Arc::clone(&orders));
    assert!(state.is_empty().unwrap());
    orders.write().unwrap()[0] = DecisionOrderItem::Order(order());
    assert!(!state.is_empty().unwrap());
    orders.write().unwrap().remove(0);
    assert_eq!(state.is_empty(), Err(DecisionAccessError::OrderPoisoned(0)));
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _guard = orders.write().unwrap();
            panic!("poison list");
        }))
        .is_err()
    );
    assert!(Arc::ptr_eq(state.get_decision().unwrap(), &orders));
    assert_eq!(state.is_empty(), Err(DecisionAccessError::ListPoisoned));
}
