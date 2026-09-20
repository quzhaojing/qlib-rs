use std::{
    process::Command,
    sync::{Arc, Mutex},
};

use chrono::NaiveDateTime;
use domain_core::{
    DecisionUpdate, DecisionUpdateCalendar, DecisionUpdateCalendarError, DecisionUpdateError,
    DecisionUpdateStrategy, DecisionUpdateStrategyError, NestedCalendar, NestedCalendarError,
    NestedDecisionCalendarAdapter, TradeDecision, update_trade_decision,
};
use serde_json::{Value, json};

#[path = "support/live_decision_update_cases.rs"]
mod live_decision_update_cases;

fn at(value: &str) -> NaiveDateTime {
    NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S").unwrap()
}

#[test]
fn source_live_decision_identity_contract_exposes_remaining_native_boundaries() {
    // Source characterization only: the current owned tracking DTO is not claimed equivalent.
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/live_decision_tracking_contract.py"
        ))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let source: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(source["tracking"]["close"], json!([]));
    assert_eq!(source["tracking"]["range"], json!(["range"]));
    assert_eq!(
        source["tracking"]["mutate"],
        json!(["range", ["collect", true, 9], ["account", true, 9], "step"])
    );
    assert_eq!(
        source["update"][0],
        json!({"events": ["calendar"], "total_step": null, "marker": null})
    );
    for index in [1, 2] {
        assert_eq!(
            source["update"][index],
            json!({
                "events": ["calendar", ["strategy", 7]], "total_step": 7, "marker": "changed"
            })
        );
    }
}

fn decision(item: &str) -> TradeDecision<String> {
    TradeDecision::from_items(
        vec![item.to_owned()],
        at("2024-01-02 09:30:00"),
        at("2024-01-02 10:00:00"),
        None,
    )
}

struct Calendar {
    result: Result<i64, DecisionUpdateCalendarError>,
    events: Arc<Mutex<Vec<String>>>,
}

impl DecisionUpdateCalendar for Calendar {
    fn trade_len(&self) -> Result<i64, DecisionUpdateCalendarError> {
        self.events.lock().unwrap().push("calendar".to_owned());
        self.result.clone()
    }
}

enum Action {
    Unchanged,
    Current,
    Replacement,
    Fail,
}

struct Strategy {
    action: Action,
    events: Arc<Mutex<Vec<String>>>,
}

impl DecisionUpdateStrategy<String> for Strategy {
    fn update_trade_decision(
        &mut self,
        current: &mut TradeDecision<String>,
        calendar: &dyn DecisionUpdateCalendar,
    ) -> Result<DecisionUpdate<String>, DecisionUpdateStrategyError> {
        self.events
            .lock()
            .unwrap()
            .push(format!("strategy:{:?}", current.total_step()));
        current.items_mut().push("mutated".to_owned());
        match self.action {
            Action::Unchanged => Ok(DecisionUpdate::Unchanged),
            Action::Current => Ok(DecisionUpdate::Current),
            Action::Replacement => {
                assert_eq!(calendar.trade_len().unwrap(), current.total_step().unwrap());
                Ok(DecisionUpdate::Replacement(decision("replacement")))
            }
            Action::Fail => Err(DecisionUpdateStrategyError {
                message: "strategy".to_owned(),
            }),
        }
    }
}

fn calendar_error(message: &str) -> DecisionUpdateCalendarError {
    DecisionUpdateCalendarError {
        message: message.to_owned(),
    }
}

#[test]
fn unchanged_current_and_replacement_preserve_order_and_identity_semantics() {
    for (length, action, expected_kind) in [
        (3, Action::Unchanged, "unchanged"),
        (0, Action::Current, "current"),
        (-2, Action::Replacement, "replacement"),
    ] {
        let events = Arc::new(Mutex::new(Vec::new()));
        let calendar = Calendar {
            result: Ok(length),
            events: Arc::clone(&events),
        };
        let mut strategy = Strategy {
            action,
            events: Arc::clone(&events),
        };
        let mut current = decision("original");
        let update = update_trade_decision(&mut current, &calendar, &mut strategy).unwrap();
        assert_eq!(current.total_step(), Some(length));
        assert_eq!(current.items(), ["original", "mutated"]);
        match (expected_kind, update) {
            ("unchanged", DecisionUpdate::Unchanged) | ("current", DecisionUpdate::Current) => {}
            ("replacement", DecisionUpdate::Replacement(replacement)) => {
                assert_eq!(replacement.items(), ["replacement"]);
                assert_eq!(replacement.total_step(), None);
            }
            _ => panic!("unexpected update action"),
        }
        let events = events.lock().unwrap();
        assert_eq!(events[0], "calendar");
        assert_eq!(events[1], format!("strategy:Some({length})"));
    }
}

#[test]
fn failures_stop_at_the_source_stage_and_retain_reached_mutations() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let calendar = Calendar {
        result: Err(calendar_error("offline")),
        events: Arc::clone(&events),
    };
    let mut strategy = Strategy {
        action: Action::Unchanged,
        events: Arc::clone(&events),
    };
    let mut current = decision("original");
    current.set_total_step(99);
    match update_trade_decision(&mut current, &calendar, &mut strategy) {
        Err(error) => assert_eq!(
            error,
            DecisionUpdateError::Calendar(calendar_error("offline"))
        ),
        Ok(_) => panic!("calendar failure must stop the update"),
    }
    assert_eq!(current.total_step(), Some(99));
    assert_eq!(current.items(), ["original"]);
    assert_eq!(*events.lock().unwrap(), ["calendar"]);

    let events = Arc::new(Mutex::new(Vec::new()));
    let calendar = Calendar {
        result: Ok(i64::MAX),
        events: Arc::clone(&events),
    };
    let mut strategy = Strategy {
        action: Action::Fail,
        events: Arc::clone(&events),
    };
    match update_trade_decision(&mut current, &calendar, &mut strategy) {
        Err(error) => assert_eq!(
            error,
            DecisionUpdateError::Strategy(DecisionUpdateStrategyError {
                message: "strategy".to_owned()
            })
        ),
        Ok(_) => panic!("strategy failure must be retained"),
    }
    assert_eq!(current.total_step(), Some(i64::MAX));
    assert_eq!(current.items(), ["original", "mutated"]);
    assert_eq!(
        *events.lock().unwrap(),
        ["calendar", "strategy:Some(9223372036854775807)"]
    );
}

struct ExistingCalendar {
    result: Result<i64, NestedCalendarError>,
}

impl NestedCalendar for ExistingCalendar {
    fn finished(&self) -> Result<bool, NestedCalendarError> {
        unreachable!()
    }

    fn trade_len(&self) -> Result<i64, NestedCalendarError> {
        self.result.clone()
    }

    fn trade_step(&self) -> Result<i64, NestedCalendarError> {
        unreachable!()
    }

    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), NestedCalendarError> {
        unreachable!()
    }

    fn step(&self) -> Result<(), NestedCalendarError> {
        unreachable!()
    }
}

#[test]
fn existing_nested_calendar_adapts_success_and_failure_without_extra_calls() {
    let nested = ExistingCalendar { result: Ok(12) };
    let adapter = NestedDecisionCalendarAdapter::new(&nested);
    assert_eq!(adapter.trade_len(), Ok(12));

    let nested = ExistingCalendar {
        result: Err(NestedCalendarError {
            message: "nested".to_owned(),
        }),
    };
    let adapter = NestedDecisionCalendarAdapter::new(&nested);
    assert_eq!(adapter.trade_len(), Err(calendar_error("nested")));
}

#[test]
fn unchanged_source_ast_freezes_results_failures_and_side_effect_order() {
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/decision_update_contract.py"
        ))
        .output()
        .expect("Python characterization fixture runs");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Value = serde_json::from_slice(&output.stdout).expect("fixture emits JSON");
    assert_eq!(
        cases[0],
        json!({
            "events": [["calendar"], ["strategy", 3, true]],
            "total_step": 3,
            "marker": "mutated",
            "returned": "none",
            "error": null
        })
    );
    assert_eq!(cases[1]["returned"], "self");
    assert_eq!(cases[1]["total_step"], 0);
    assert_eq!(cases[2]["returned"], "replacement");
    assert_eq!(cases[2]["total_step"], -2);
    assert_eq!(cases[3]["error"], "RuntimeError:strategy");
    assert_eq!(cases[3]["total_step"], 7);
    assert_eq!(cases[3]["marker"], "mutated");
    assert_eq!(cases[4]["error"], "RuntimeError:calendar");
    assert_eq!(cases[4]["total_step"], 99);
    assert_eq!(cases[4]["marker"], "original");
    assert_eq!(cases[4]["events"], json!([["calendar"]]));
}
