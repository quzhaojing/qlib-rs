use std::{
    process::Command,
    sync::{Arc, Mutex},
};

use chrono::NaiveDateTime;
use domain_core::{
    IdxTradeRange, Order, OrderDecision, OrderDir, OrderTradeDecision, SharedTradeRange,
    TradeDecisionWithDetails,
};
use serde_json::{Value, json};

fn at(value: &str) -> NaiveDateTime {
    NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S").unwrap()
}

fn order(
    stock: &str,
    amount: f64,
    start: Option<NaiveDateTime>,
    end: Option<NaiveDateTime>,
) -> Order {
    Order::new(stock, amount, OrderDir::Buy, start, end)
}

#[test]
fn arbitrary_details_retain_identity_and_support_owned_replacement() {
    let start = at("2024-01-02 09:30:00");
    let end = at("2024-01-02 10:00:00");
    let first = Arc::new(Mutex::new(vec!["first"]));
    let second = Arc::new(Mutex::new(vec!["second"]));
    let core = OrderTradeDecision::from_orders(Vec::new(), start, end, None);
    let mut decision = TradeDecisionWithDetails::new(core, Some(Arc::clone(&first)));

    assert!(Arc::ptr_eq(decision.details().as_ref().unwrap(), &first));
    decision
        .details_mut()
        .as_ref()
        .unwrap()
        .lock()
        .unwrap()
        .push("mutated");
    assert_eq!(*first.lock().unwrap(), ["first", "mutated"]);

    let old = decision.replace_details(Some(Arc::clone(&second)));
    assert!(Arc::ptr_eq(old.as_ref().unwrap(), &first));
    assert!(Arc::ptr_eq(decision.details().as_ref().unwrap(), &second));
    let (core, details) = decision.into_parts();
    assert!(core.orders().is_empty());
    assert!(Arc::ptr_eq(details.as_ref().unwrap(), &second));

    let absent = TradeDecisionWithDetails::<Option<String>>::new(core, None);
    assert_eq!(absent.details(), &None);
}

#[test]
fn parent_normalization_precedes_wrapper_access_and_trait_delegation() {
    let start = at("2024-01-02 09:30:00");
    let end = at("2024-01-02 10:00:00");
    let supplied_start = at("2024-01-02 09:40:00");
    let supplied_end = at("2024-01-02 09:50:00");
    let range: SharedTradeRange = Arc::new(IdxTradeRange::new(2, 5));
    let mut decision = TradeDecisionWithDetails::from_orders(
        vec![
            order("A", 0.0, None, Some(supplied_end)),
            order("B", 2.0, Some(supplied_start), None),
        ],
        start,
        end,
        Some(Arc::clone(&range)),
        vec!["row-a", "row-b"],
    );

    assert_eq!(decision.core().orders()[0].start_time(), Some(start));
    assert_eq!(decision.core().orders()[0].end_time(), Some(supplied_end));
    assert_eq!(
        decision.core().orders()[1].start_time(),
        Some(supplied_start)
    );
    assert_eq!(decision.core().orders()[1].end_time(), Some(end));
    decision.core_mut().orders_mut()[0].set_deal_amount(0.5);
    assert_eq!(
        decision.core().orders()[0].deal_amount().to_bits(),
        0.5_f64.to_bits()
    );

    let plugin: &mut dyn OrderDecision = &mut decision;
    assert_eq!(plugin.orders().len(), 2);
    plugin.orders_mut()[1].set_deal_amount(1.5);
    assert_eq!(plugin.start_time(), start);
    assert_eq!(plugin.end_time(), end);
    assert_eq!(
        plugin.trade_range().unwrap().range_indices(None),
        Ok((2, 5))
    );
    assert!(!plugin.is_empty());
    let step = plugin.base_price_step();
    assert_eq!((step.start_time, step.end_time), (start, end));
    assert_eq!(step.trade_range.unwrap().range_indices(None), Ok((2, 5)));
    assert_eq!(
        decision.core().orders()[1].deal_amount().to_bits(),
        1.5_f64.to_bits()
    );
}

#[test]
fn unchanged_subclass_ast_freezes_parent_order_identity_and_failure_cutoff() {
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/trade_decision_with_details_contract.py"
        ))
        .output()
        .expect("Python characterization fixture runs");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Value = serde_json::from_slice(&output.stdout).expect("fixture emits JSON");
    assert_eq!(cases.as_array().unwrap().len(), 3);
    assert_eq!(cases[0]["events"], json!([["parent", 1, null]]));
    assert_eq!(cases[0]["orders"], json!(["original", "parent-mutated"]));
    assert_eq!(cases[0]["order_identity"], true);
    assert_eq!(cases[0]["strategy_identity"], true);
    assert_eq!(cases[0]["details_present"], true);
    assert_eq!(cases[0]["details_identity"], true);
    assert_eq!(cases[0]["details_is_none"], true);
    assert_eq!(cases[0]["error"], Value::Null);

    assert_eq!(cases[1]["events"], json!([["parent", 1, [2, 5]]]));
    assert_eq!(cases[1]["details_present"], true);
    assert_eq!(cases[1]["details_identity"], true);
    assert_eq!(cases[1]["details_is_none"], false);
    assert_eq!(cases[1]["error"], Value::Null);

    assert_eq!(cases[2]["events"], json!([["parent", 1, [8, 9]]]));
    assert_eq!(cases[2]["orders"], json!(["original", "parent-mutated"]));
    assert_eq!(cases[2]["details_present"], false);
    assert_eq!(cases[2]["details_identity"], false);
    assert_eq!(cases[2]["error"], "RuntimeError:parent-failed");
}
