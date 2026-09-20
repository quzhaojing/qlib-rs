//! Source behavior required before the owned numerical adapter can adopt live order transport.
use serde_json::{Value, json};
use std::process::Command;

pub(super) fn source() -> Value {
    let output = Command::new("python")
        .args([
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/saoe_live_adapter_contract.py"
            ),
            r"D:\code\github\qlib\qlib\rl\order_execution\strategy.py",
            r"D:\code\github\qlib\qlib\rl\order_execution\utils.py",
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
fn actual_adapter_source_reads_live_order_between_market_indicator_and_state_callbacks() {
    let cases = source();
    assert_eq!(cases.as_object().unwrap().len(), 13);
    for (mode, stock, amount, metric_stock, metric_amount, minute) in [
        ("normal", "A", 10.0, "A", 10.0, "32"),
        ("init", "I", 20.0, "I", 20.0, "31"),
        ("volume", "V", 20.0, "V", 20.0, "31"),
        ("price", "P", 40.0, "P", 40.0, "31"),
        ("indicator", "Q", 50.0, "Q", 50.0, "31"),
        ("state", "S", 80.0, "A", 10.0, "32"),
    ] {
        let volume_stock = if mode == "init" { "I" } else { "A" };
        let (price_stock, direction) = match mode {
            "init" => ("I", 0),
            "volume" => ("V", 0),
            _ => ("A", 1),
        };
        let initial_time = if mode == "init" { "31" } else { "30" };
        assert_eq!(
            cases[mode],
            json!({
                "events": ["start", "baseline", format!("volume:{volume_stock}"), format!("price:{price_stock}:{direction}"), "indicator", "step"],
                "initial": {"position": 10.0, "time": format!("2024-01-02 09:{initial_time}:00"), "amount": if mode == "init" {20.0} else {10.0}},
                "error": null, "position": 5.0, "time": format!("2024-01-02 09:{minute}:00"),
                "state_stock": stock, "state_amount": amount, "state_step": 2, "state_metrics_none": true,
                "exec_stock": [metric_stock, metric_stock], "exec_ffr": [2.0 / metric_amount, 3.0 / metric_amount],
                "step_count": 1, "metric_stock": stock, "metric_ffr": 5.0 / amount, "metric_position": 0.0,
            }),
            "{mode}"
        );
    }
}

#[test]
fn actual_adapter_final_metrics_alias_is_shared_then_replaced_on_regeneration() {
    assert_eq!(
        source()["metric-aliases"],
        json!({
            "before_finalize_none": true,
            "same_before_replacement": true,
            "mutation_visible": "M",
            "replaced_after_finalize": true,
            "old_stock": "M",
            "new_stock": "A",
        })
    );
}

#[test]
fn actual_adapter_backtest_object_and_tick_field_aliases_follow_rebinding() {
    assert_eq!(
        source()["backtest-aliases"],
        json!({
            "same_backtest_object": true,
            "old_ticks_retained": true,
            "old_order_ticks_retained": true,
            "current_ticks_rebound": true,
            "current_order_ticks_rebound": true,
            "deal_mutation_visible": 99.0,
            "adapter_time": "2024-01-02 09:33:00",
            "metric_time": "2024-01-02 09:31:00",
        })
    );
}

#[test]
fn actual_adapter_state_aliases_are_current_then_histories_are_replaced_on_append() {
    assert_eq!(
        source()["aliases"],
        json!({
            "order_alias": true,
            "initial_history_alias": true,
            "old_lengths": [2, 1],
            "new_lengths": [4, 2],
            "mutated_exec_stock": "M",
            "mutated_step_amount": 42.0,
            "new_exec_stock": "Z",
            "state_stock": "Z",
        })
    );
}

#[test]
fn actual_adapter_source_failure_order_preserves_committed_history_and_position() {
    let cases = source();
    for mode in [
        "volume-failure",
        "price-failure",
        "indicator-failure",
        "next-failure",
    ] {
        let mut events = vec!["start", "baseline", "volume:A"];
        if mode != "volume-failure" {
            events.push("price:A:1");
        }
        if !["volume-failure", "price-failure"].contains(&mode) {
            events.push("indicator");
        }
        events.push("step");
        let committed = mode == "next-failure";
        assert_eq!(
            cases[mode],
            json!({
                "events": events, "initial": {"position": 10.0, "time": "2024-01-02 09:30:00", "amount": 10.0},
                "error": if committed {"TypeError"} else {"RuntimeError"},
                "position": if committed {5.0} else {10.0}, "time": "2024-01-02 09:30:00",
                "state_stock": "A", "state_amount": 10.0, "state_step": 2, "state_metrics_none": true,
                "exec_stock": if committed {vec!["A", "A"]} else {vec![]},
                "exec_ffr": if committed {vec![0.2, 0.3]} else {vec![]},
                "step_count": i32::from(committed), "metric_stock": "A", "metric_ffr": if committed {0.5} else {0.0},
                "metric_position": if committed {0.0} else {10.0},
            }),
            "{mode}"
        );
    }
}
