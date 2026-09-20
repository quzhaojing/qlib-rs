use std::process::Command;

use serde_json::{Value, json};

fn source_cases() -> Value {
    let output = Command::new("python")
        .args([
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/saoe_lifecycle_contract.py"
            ),
            r"D:\code\github\qlib\qlib\rl\order_execution\strategy.py",
            r"D:\code\github\qlib\qlib\strategy\base.py",
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
fn source_reset_retains_callback_order_live_membership_and_partial_state() {
    let cases = source_cases();
    let cases = &cases["reset"];
    assert_eq!(cases.as_object().unwrap().len(), 12);
    for (mode, creates, keys, entries, error) in [
        (
            "normal",
            vec!["A:0", "B:1"],
            vec!["A", "B"],
            json!([["A", "A:0"], ["B", "B:1"]]),
            None,
        ),
        (
            "late-key",
            vec!["A:0", "B:1"],
            vec!["X", "B"],
            json!([["X", "A:0"], ["B", "B:1"]]),
            None,
        ),
        (
            "append",
            vec!["A:0", "B:1", "C:2"],
            vec!["A", "B", "C"],
            json!([["A", "A:0"], ["B", "B:1"], ["C", "C:2"]]),
            None,
        ),
        (
            "delete",
            vec!["A:0"],
            vec!["A"],
            json!([["A", "A:0"]]),
            None,
        ),
        (
            "replace",
            vec!["A:0", "B:1"],
            vec!["A", "B"],
            json!([["A", "A:0"], ["B", "B:1"]]),
            None,
        ),
        (
            "duplicate",
            vec!["A:0", "B:1", "A:2"],
            vec!["A", "B", "A"],
            json!([["A", "A:2"], ["B", "B:1"]]),
            None,
        ),
        (
            "factory-failure",
            vec!["A:0", "B:1"],
            vec!["A"],
            json!([["A", "A:0"]]),
            Some("RuntimeError:factory failed"),
        ),
        (
            "key-failure",
            vec!["A:0", "B:1"],
            vec!["A", "B"],
            json!([["A", "A:0"]]),
            Some("RuntimeError:key failed"),
        ),
    ] {
        let mut events = vec![
            "level".to_owned(),
            "empty".to_owned(),
            "range".to_owned(),
            "orders".to_owned(),
        ];
        for (index, label) in creates.iter().enumerate() {
            events.push(format!("create:{label}"));
            if let Some(key) = keys.get(index) {
                events.push(format!("key:{key}"));
            }
        }
        assert_eq!(
            cases[mode],
            json!({"events": events, "entries": entries,
            "last": [0, 0], "error": error, "outer_retained": false,
            "registry_retained": false}),
            "{mode}"
        );
    }
}

#[test]
fn source_reset_none_empty_and_failure_have_distinct_retained_state() {
    let cases = source_cases();
    let cases = &cases["reset"];
    for (mode, events, error) in [
        (
            "no-range",
            vec!["level", "empty", "range"],
            Some("AssertionError:"),
        ),
        ("empty", vec!["level", "empty"], None),
        ("none", vec!["level"], None),
    ] {
        assert_eq!(
            cases[mode],
            json!({"events": events, "entries": [],
            "last": [0, 0], "error": error, "outer_retained": mode == "none",
            "registry_retained": false}),
            "{mode}"
        );
    }
    assert_eq!(
        cases["base-failure"],
        json!({"events": ["level"], "entries": [["old", "old"]],
        "last": [7, 9], "error": "RuntimeError:base failed", "outer_retained": true,
        "registry_retained": true})
    );
}

#[test]
fn source_post_groups_original_rows_before_updates_and_stops_at_first_failure() {
    let cases = source_cases();
    let cases = &cases["post"];
    assert_eq!(cases.as_object().unwrap().len(), 7);
    for mode in [
        "normal",
        "none",
        "zero-empty",
        "zero-rows",
        "mutate",
        "update-failure",
        "final-failure",
    ] {
        let mut events = Vec::new();
        if !["none", "zero-empty", "zero-rows"].contains(&mode) {
            events.extend([
                json!("key:B"),
                json!("key:A"),
                json!("key:B"),
                json!("key:X"),
            ]);
        }
        if !mode.starts_with("zero") {
            events.push(json!([
                "update",
                "A",
                if mode == "none" {
                    json!([])
                } else {
                    json!([["A", 12]])
                }
            ]));
            let name = if mode == "mutate" { "Z" } else { "B" };
            events.push(json!([
                "update",
                "B",
                if mode == "none" {
                    json!([])
                } else {
                    json!([[name, 11], ["B", 13]])
                }
            ]));
            if mode != "update-failure" {
                events.push(json!(["update", "C", []]));
            }
        }
        if !["update-failure", "zero-rows"].contains(&mode) {
            events.extend([json!("final:A"), json!("final:B")]);
            if mode != "final-failure" {
                events.push(json!("final:C"));
            }
        }
        let error = match mode {
            "zero-rows" => Some("AssertionError:"),
            "update-failure" => Some("RuntimeError:update failed"),
            "final-failure" => Some("RuntimeError:final failed"),
            _ => None,
        };
        assert_eq!(
            cases[mode],
            json!({"events": events, "error": error}),
            "{mode}"
        );
    }
}
