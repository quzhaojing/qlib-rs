//! Source characterization for the live SAOE migration, not a native parity claim.
use std::process::Command;

use serde_json::{Value, json};

#[path = "support/saoe_lifecycle_contract.rs"]
mod lifecycle;

#[test]
fn actual_saoe_source_observes_mutations_between_callbacks_and_truncates_zip() {
    let output = Command::new("python")
        .args([
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/saoe_live_orders_contract.py"
            ),
            r"D:\code\github\qlib\qlib\rl\order_execution\strategy.py",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let actual: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(actual.as_object().unwrap().len(), 9);
    for (mode, states, children, details) in [
        ("normal", vec!["A", "B"], vec!["A", "B"], vec!["A", "B"]),
        (
            "append",
            vec!["A", "B", "C"],
            vec!["A", "B", "C"],
            vec!["A", "B", "C"],
        ),
        ("delete", vec!["A", "C"], vec!["A", "C"], vec!["A", "C"]),
        ("policy", vec!["A", "B"], vec!["X", "Y"], vec!["X", "Y"]),
        ("factory", vec!["A", "B"], vec!["A", "B"], vec!["D", "E"]),
        ("details", vec!["A", "B"], vec!["A", "B"], vec!["A", "E"]),
        ("short", vec!["A", "B"], vec!["A"], vec!["A"]),
        ("long", vec!["A", "B"], vec!["A", "B"], vec!["A", "B"]),
    ] {
        let mut events = vec!["read".to_owned()];
        for name in &states {
            events.push(format!("state:{name}"));
            events.push(format!("observe:{name}"));
        }
        events.push(format!("policy:{}", states.join(",")));
        for (index, name) in states.iter().take(children.len()).enumerate() {
            events.push(format!("action:{name}:{}", index + 1));
        }
        events.push("read".to_owned());
        let mut expected_children = Vec::new();
        for (index, name) in children.iter().enumerate() {
            events.push(format!("create:{name}"));
            expected_children.push(json!([name, index + 1, 1]));
        }
        for _ in &details {
            events.push("time".to_owned());
            events.push("freq".to_owned());
        }
        events.push("construct".to_owned());
        assert_eq!(
            actual[mode],
            json!({"events": events, "children": expected_children,
                "details": details, "error": null}),
            "source scenario {mode}"
        );
    }
    assert_eq!(
        actual["failure"],
        json!({"events": ["read", "state:A", "observe:A", "state:B", "observe:B"],
            "children": null, "details": null, "error": "observation failed"})
    );
}
