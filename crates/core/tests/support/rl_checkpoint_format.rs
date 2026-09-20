use super::*;
use serde_json::Value;
use std::{path::PathBuf, process::Command, sync::Arc};

fn scalar(tag: &Value) -> TrainingMetricScalar {
    let text = tag[1].as_str().unwrap();
    match tag[0].as_str().unwrap() {
        "int" => TrainingMetricScalar::Integer(text.parse().unwrap()),
        "float" => TrainingMetricScalar::Float(text.parse().unwrap()),
        "text" => TrainingMetricScalar::Text(text.into()),
        "bool" => TrainingMetricScalar::Boolean(text == "true"),
        "null" => TrainingMetricScalar::Null,
        other => panic!("unknown tag {other}"),
    }
}

#[test]
fn composed_scalar_engines_match_all_live_python_probe_cases() {
    compare_oracle("--scalar-probes", 508);
}

#[test]
fn extended_scalar_formats_match_live_python() {
    compare_oracle("--extended-scalar-probes", 1350);
}

#[test]
fn large_precision_formats_match_live_python_across_types_and_layouts() {
    compare_oracle("--large-scalar-probes", 2675);
}

#[test]
fn fractional_grouping_matches_python_across_types_precision_and_padding() {
    compare_oracle("--fractional-grouping-probes", 18217);
}

#[test]
fn unicode_numeric_fields_preserve_flags_fill_and_type_contracts() {
    compare_oracle("--unicode-numeric-probes", 30872);
    assert_eq!(
        normalize_numeric_fields("９２２３３７２０３６８５４７７５８０７").unwrap(),
        "9223372036854775807"
    );
}

fn compare_oracle(mode: &str, count: usize) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/rl_checkpoint_filename_contract.py"))
        .arg(mode)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(cases.len(), count);
    let mut differences = Vec::new();
    for case in cases {
        let result =
            format_rl_checkpoint_scalar(&scalar(&case["value"]), case["format"].as_str().unwrap());
        if case["error"].is_null() {
            if result.as_deref() != Ok(case["output"].as_str().unwrap()) {
                differences.push(format!(
                    "value={} spec={} expected_length={} actual={:?}",
                    case["value"],
                    case["format"],
                    case["output"].as_str().unwrap().len(),
                    result
                        .as_ref()
                        .map(|text| (text.len(), text.chars().take(80).collect::<String>()))
                ));
            }
        } else if result.is_ok() {
            differences.push(format!("{case}: {result:?}"));
        }
    }
    assert!(differences.is_empty(), "{}", differences.join("\n"));
}

struct DisplayOnly;
impl crate::TrainingMetricDisplay for DisplayOnly {
    fn render(&self) -> Result<String, String> {
        panic!("scalar formatter must not invoke a display-only custom value")
    }
}

#[test]
fn custom_protocols_are_not_silently_coerced_and_format_input_is_not_a_template() {
    let custom = TrainingMetricScalar::Custom(Arc::new(DisplayOnly));
    assert!(
        format_rl_checkpoint_scalar(&custom, "")
            .unwrap_err()
            .contains("model-aware")
    );
    for value in [
        TrainingMetricScalar::Boolean(true),
        TrainingMetricScalar::Float(1.0),
        TrainingMetricScalar::Text("x".into()),
        TrainingMetricScalar::Integer(1.into()),
        TrainingMetricScalar::Null,
    ] {
        assert!(format_rl_checkpoint_scalar(&value, "}{0").is_err());
    }
}

#[test]
fn character_formats_preserve_unicode_scalars_and_reject_unrepresentable_surrogates() {
    for codepoint in [0_u32, 0xd7ff, 0xe000, 0x10_ffff] {
        let result =
            format_rl_checkpoint_scalar(&TrainingMetricScalar::Integer(codepoint.into()), "c")
                .unwrap();
        assert_eq!(result, char::from_u32(codepoint).unwrap().to_string());
    }
    // Python can construct lone surrogates, but Rust String cannot. This explicit error
    // is a tracked representation boundary, not claimed as identical Python behavior.
    for codepoint in [0xd800_u32, 0xdbff, 0xdc00, 0xdfff] {
        let result =
            format_rl_checkpoint_scalar(&TrainingMetricScalar::Integer(codepoint.into()), "c");
        assert!(result.unwrap_err().contains("codepoint"));
    }
}
