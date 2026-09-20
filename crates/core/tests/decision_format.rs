use std::{
    cell::{Cell, RefCell},
    process::Command,
    rc::Rc,
};

use domain_core::{DecisionFrequency, FormattedDecisions, format_decisions};
use serde_json::{Value, json};
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
#[error("frequency:{name}:{call}")]
struct FrequencyFailure {
    name: &'static str,
    call: usize,
}

#[derive(Debug)]
struct Decision {
    name: &'static str,
    frequency: &'static str,
    calls: Cell<usize>,
    fail_on: Option<usize>,
    events: Rc<RefCell<Vec<Value>>>,
}

impl DecisionFrequency for Decision {
    type Error = FrequencyFailure;

    fn frequency(&self) -> Result<String, Self::Error> {
        let call = self.calls.get() + 1;
        self.calls.set(call);
        self.events
            .borrow_mut()
            .push(json!(["frequency", self.name, call]));
        if self.fail_on == Some(call) {
            Err(FrequencyFailure {
                name: self.name,
                call,
            })
        } else {
            Ok(self.frequency.to_owned())
        }
    }
}

fn decisions(
    specifications: &[(&'static str, &'static str, Option<usize>)],
) -> (Vec<Decision>, Rc<RefCell<Vec<Value>>>) {
    let events = Rc::new(RefCell::new(Vec::new()));
    let decisions = specifications
        .iter()
        .map(|&(name, frequency, fail_on)| Decision {
            name,
            frequency,
            calls: Cell::new(0),
            fail_on,
            events: Rc::clone(&events),
        })
        .collect();
    (decisions, events)
}

fn serialize(tree: Option<&FormattedDecisions<'_, Decision>>) -> Value {
    match tree {
        None => Value::Null,
        Some(tree) => json!({
            "frequency": tree.frequency,
            "items": tree.items.iter().map(|item| json!([
                item.decision.name,
                serialize(item.nested.as_ref()),
            ])).collect::<Vec<_>>(),
        }),
    }
}

fn rust_case(name: &str, specifications: &[(&'static str, &'static str, Option<usize>)]) -> Value {
    let (decisions, events) = decisions(specifications);
    let result = format_decisions(&decisions);
    let (output, error) = match &result {
        Ok(tree) => (serialize(tree.as_ref()), Value::Null),
        Err(error) => (Value::Null, json!(error.to_string())),
    };
    json!({"name": name, "output": output, "events": events.borrow().clone(), "error": error})
}

#[test]
fn actual_source_contract_matches_tree_shapes_failures_and_call_counts() {
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/format_decisions_contract.py"
        ))
        .output()
        .expect("Python characterization fixture runs");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let actual: Value = serde_json::from_slice(&output.stdout).expect("fixture emits JSON");

    let expected = json!([
        rust_case("empty", &[]),
        rust_case("single", &[("d0", "day", None)]),
        rust_case(
            "flat",
            &[
                ("d0", "day", None),
                ("d1", "day", None),
                ("d2", "day", None)
            ]
        ),
        rust_case(
            "nested",
            &[
                ("d0", "day", None),
                ("m0", "1min", None),
                ("m1", "1min", None),
                ("d1", "day", None),
                ("m2", "1min", None),
                ("d2", "day", None),
            ]
        ),
        rust_case(
            "irregular",
            &[
                ("d0", "day", None),
                ("h0", "hour", None),
                ("m0", "1min", None),
                ("h1", "hour", None),
                ("d1", "day", None),
            ]
        ),
        rust_case(
            "root_failure",
            &[("d0", "day", Some(1)), ("tail", "tick", None)],
        ),
        rust_case(
            "scan_failure",
            &[
                ("d0", "day", None),
                ("m0", "1min", Some(1)),
                ("tail", "tick", None),
            ]
        ),
        rust_case(
            "nested_failure",
            &[
                ("d0", "day", None),
                ("m0", "1min", Some(2)),
                ("d1", "day", None),
                ("tail", "tick", None),
            ]
        ),
        rust_case(
            "final_nested_failure",
            &[("d0", "day", None), ("m0", "1min", Some(2))],
        ),
    ]);
    assert_eq!(actual, expected);
}

#[test]
fn formatting_borrows_the_original_decisions() {
    let (decisions, _) = decisions(&[("d0", "day", None), ("m0", "1min", None)]);
    let tree = format_decisions(&decisions).unwrap().unwrap();
    assert!(std::ptr::eq(
        tree.items[0].decision,
        std::ptr::from_ref(&decisions[0])
    ));
    let nested = tree.items[0].nested.as_ref().unwrap();
    assert!(std::ptr::eq(
        nested.items[0].decision,
        std::ptr::from_ref(&decisions[1])
    ));
}
