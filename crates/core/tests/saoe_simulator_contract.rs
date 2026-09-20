//! Source characterization for the not-yet-complete native simulator wrapper.
use std::process::Command;

use serde_json::{Value, json};

#[test]
fn source_simulator_freezes_initial_advance_live_final_state_and_failure_order() {
    let output = Command::new("python")
        .args([
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/saoe_simulator_contract.py"
            ),
            r"D:\code\github\qlib\qlib\rl\order_execution\simulator_qlib.py",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Value = serde_json::from_slice(&output.stdout).unwrap();
    let build = |cash: Value, position: &str| {
        json!([
            "build",
            cash,
            position,
            "2024-01-02 00:00:00",
            "2024-01-03 00:00:00"
        ])
    };
    for (name, cash, position, config) in [
        (
            "unlimited",
            json!(1_000_000_000_000_u64),
            "InfPosition",
            None,
        ),
        ("zero", json!(0), "Position", Some(json!({}))),
        (
            "positive",
            json!(25),
            "Position",
            Some(json!({"region":"cn"})),
        ),
    ] {
        let mut events = Vec::new();
        if let Some(config) = config {
            events.push(json!(["init", config]));
        }
        events.extend([
            build(cash, position),
            json!(["collect", "2024-01-02 00:00:00"]),
            json!(["action", 7.0]),
            json!(["forwarded", 7.0]),
        ]);
        assert_eq!(
            cases[name],
            json!({"events":events,"decisions":2,"final_position":0,"terminal_step_rejected":true})
        );
    }
    assert_eq!(
        cases["init_failure"],
        json!({"failure":"init failed","events":[["init",{}]]})
    );
    assert_eq!(
        cases["build_failure"],
        json!({"failure":"build failed","events":[build(json!(1_000_000_000_000_u64),"InfPosition")]})
    );
    assert_eq!(
        cases["advance_failure"],
        json!({"failure":"advance failed","events":[build(json!(1_000_000_000_000_u64),"InfPosition"),["collect","2024-01-02 00:00:00"],["action",7.0],["forwarded",7.0]],"decisions":2})
    );
}
