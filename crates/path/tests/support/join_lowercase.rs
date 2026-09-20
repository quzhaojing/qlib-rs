use super::*;
use std::process::Command;

#[test]
fn unicode16_lowercase_matches_actual_python() {
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/join_lowercase.py"
        ))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Vec<(Vec<u16>, Vec<u16>)> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(cases.len(), 66_687);
    for (input, expected) in cases {
        let input = OsString::from_wide(&input);
        assert_eq!(
            lowercase(&input).encode_wide().collect::<Vec<_>>(),
            expected,
            "{input:?}"
        );
    }
}
