use std::process::Command;

use domain_core::{TradeDecisionReprContext, format_trade_decision_repr};
use serde_json::{Value, json};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Failure {
    Class,
    Strategy,
    Range,
    Orders,
}

struct Context {
    class_name: String,
    strategy: String,
    trade_range: String,
    order_count: usize,
    failure: Option<Failure>,
    events: Vec<&'static str>,
}

impl TradeDecisionReprContext for Context {
    type Error = Failure;

    fn class_name(&mut self) -> Result<String, Self::Error> {
        self.events.push("class");
        (self.failure != Some(Failure::Class))
            .then(|| self.class_name.clone())
            .ok_or(Failure::Class)
    }

    fn strategy_text(&mut self) -> Result<String, Self::Error> {
        self.events.push("strategy");
        (self.failure != Some(Failure::Strategy))
            .then(|| self.strategy.clone())
            .ok_or(Failure::Strategy)
    }

    fn trade_range_text(&mut self) -> Result<String, Self::Error> {
        self.events.push("range");
        (self.failure != Some(Failure::Range))
            .then(|| self.trade_range.clone())
            .ok_or(Failure::Range)
    }

    fn order_count(&mut self) -> Result<usize, Self::Error> {
        self.events.push("orders");
        (self.failure != Some(Failure::Orders))
            .then_some(self.order_count)
            .ok_or(Failure::Orders)
    }
}

fn context(failure: Option<Failure>) -> Context {
    Context {
        class_name: "TradeDecisionWithDetails".to_owned(),
        strategy: "策略\nalpha; beta".to_owned(),
        trade_range: "(2, 5)".to_owned(),
        order_count: usize::MAX,
        failure,
        events: Vec::new(),
    }
}

#[test]
fn exact_text_preserves_dynamic_values_unicode_and_native_count_width() {
    let mut source = context(None);
    assert_eq!(
        format_trade_decision_repr(&mut source),
        Ok(format!(
            "class: TradeDecisionWithDetails; strategy: 策略\nalpha; beta; trade_range: (2, 5); order_list[{}]",
            usize::MAX
        ))
    );
    assert_eq!(source.events, ["class", "strategy", "range", "orders"]);

    source.class_name = "TradeDecisionWO".to_owned();
    source.strategy = "None".to_owned();
    source.trade_range = "None".to_owned();
    source.order_count = 0;
    source.events.clear();
    assert_eq!(
        format_trade_decision_repr(&mut source),
        Ok("class: TradeDecisionWO; strategy: None; trade_range: None; order_list[0]".to_owned())
    );
}

#[test]
fn each_failure_stops_before_every_later_dynamic_value() {
    let cases = [
        (Failure::Class, vec!["class"]),
        (Failure::Strategy, vec!["class", "strategy"]),
        (Failure::Range, vec!["class", "strategy", "range"]),
        (
            Failure::Orders,
            vec!["class", "strategy", "range", "orders"],
        ),
    ];
    for (failure, expected) in cases {
        let mut source = context(Some(failure));
        assert_eq!(format_trade_decision_repr(&mut source), Err(failure));
        assert_eq!(source.events, expected);
    }
}

#[test]
fn unchanged_method_ast_matches_text_subclass_name_and_failure_cutoffs() {
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/trade_decision_repr_contract.py"
        ))
        .output()
        .expect("Python characterization fixture runs");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Value = serde_json::from_slice(&output.stdout).expect("fixture emits JSON");
    assert_eq!(cases.as_array().unwrap().len(), 5);
    assert_eq!(
        cases[0],
        json!({
            "value":"class: TradeDecisionWO; strategy: 策略\nalpha; trade_range: None; order_list[0]",
            "error":null,
            "events":["get_strategy","format_strategy:","get_range","get_orders","len_orders"]
        })
    );
    assert_eq!(
        cases[1]["value"],
        "class: TradeDecisionWithDetails; strategy: strategy; trade_range: (2, 5); order_list[3]"
    );
    assert_eq!(
        cases[1]["events"],
        json!([
            "get_strategy",
            "format_strategy:",
            "get_range",
            "format_range:",
            "get_orders",
            "len_orders"
        ])
    );
    assert_eq!(cases[2]["error"], "RuntimeError:strategy-failed");
    assert_eq!(
        cases[2]["events"],
        json!(["get_strategy", "format_strategy:"])
    );
    assert_eq!(cases[3]["error"], "RuntimeError:range-failed");
    assert_eq!(
        cases[3]["events"],
        json!([
            "get_strategy",
            "format_strategy:",
            "get_range",
            "format_range:"
        ])
    );
    assert_eq!(cases[4]["error"], "RuntimeError:orders-failed");
    assert_eq!(
        cases[4]["events"],
        json!([
            "get_strategy",
            "format_strategy:",
            "get_range",
            "format_range:",
            "get_orders",
            "len_orders"
        ])
    );
}
