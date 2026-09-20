use super::*;
use serde_json::Value;
use std::{
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};

pub(crate) fn metadata(data: &Value) -> RlCheckpointNumericLocale {
    RlCheckpointNumericLocale::new(
        data["decimal"].as_str().unwrap().into(),
        data["separator"].as_str().unwrap().into(),
        &data["grouping"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| u8::try_from(v.as_u64().unwrap()).unwrap())
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

#[test]
fn locale_formats_match_live_python_numeric_categories() {
    let output = Command::new("python")
        .arg(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/rl_checkpoint_locale_contract.py"),
        )
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(cases.len(), 16340);
    let mut differences = Vec::new();
    for case in cases {
        let text = case["value"][1].as_str().unwrap();
        let value = match case["value"][0].as_str().unwrap() {
            "int" => TrainingMetricScalar::Integer(text.parse().unwrap()),
            "float" => TrainingMetricScalar::Float(text.parse().unwrap()),
            "bool" => TrainingMetricScalar::Boolean(text == "true"),
            "text" => TrainingMetricScalar::Text(text.into()),
            "null" => TrainingMetricScalar::Null,
            other => panic!("bad tag {other}"),
        };
        let result = format_rl_checkpoint_scalar_with_locale(
            &value,
            case["format"].as_str().unwrap(),
            &metadata(&case["metadata"]),
        );
        if case["error"].is_null() {
            if result.as_deref() != Ok(case["output"].as_str().unwrap()) {
                differences.push(format!("{case}: {result:?}"));
            }
        } else if result.is_ok() {
            differences.push(format!("{case}: {result:?}"));
        }
    }
    assert!(differences.is_empty(), "{}", differences.join("\n"));
}

#[test]
fn grouping_terminators_validation_and_long_prefixes_are_explicit() {
    assert!(RlCheckpointNumericLocale::new(String::new(), ",".into(), &[3, 0]).is_err());
    assert!(RlCheckpointNumericLocale::new(".".into(), ",".into(), &[128]).is_err());
    for (groups, expected) in [
        (&[3, 0, 255][..], "123,456,789"),
        (&[3, 127, 255], "123456,789"),
        (&[127], "123456789"),
        (&[0], "123456789"),
        (&[3, 2, 127], "1234,56,789"),
    ] {
        let locale = RlCheckpointNumericLocale::new(".".into(), ",".into(), groups).unwrap();
        assert_eq!(locale.grouped("123456789"), expected);
        assert_eq!(locale.grouped("12"), "12");
        assert_eq!(locale.grouped(""), "");
    }
    let locale = RlCheckpointNumericLocale::new(".".into(), "😀::".into(), &[3, 127]).unwrap();
    assert_eq!(
        locale.grouped(&"1".repeat(1000)),
        format!("{}😀::111", "1".repeat(997))
    );
    assert_eq!(locale.zero_grouped("1", 8), "00😀::001");
    assert_eq!(RlCheckpointNumericLocale::default().grouped("1234"), "1234");
}

struct Failure(AtomicUsize);
impl RlCheckpointLocaleProvider for Failure {
    fn numeric_locale(&self) -> Result<RlCheckpointNumericLocale, String> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Err("locale unavailable".into())
    }
}

#[test]
fn locale_queries_follow_type_validation_and_never_affect_other_presentations() {
    let provider = Failure(AtomicUsize::new(0));
    let value = TrainingMetricScalar::Float(1234.5);
    assert_eq!(
        format_rl_checkpoint_scalar_with_locale(&value, ".2f", &provider).unwrap(),
        "1234.50"
    );
    for spec in [
        ",n",
        "_n",
        "._n",
        ".2147483648n",
        "badn",
        ".٩٩٩٩٩٩٩٩٩٩٩٩٩٩٩٩٩٩٩٩n",
    ] {
        assert!(format_rl_checkpoint_scalar_with_locale(&value, spec, &provider).is_err());
    }
    assert!(
        format_rl_checkpoint_scalar_with_locale(
            &TrainingMetricScalar::Text("x".into()),
            "n",
            &provider
        )
        .is_err()
    );
    assert_eq!(provider.0.load(Ordering::SeqCst), 0);
    assert_eq!(
        format_rl_checkpoint_scalar_with_locale(&value, "n", &provider).unwrap_err(),
        "locale unavailable"
    );
    assert_eq!(provider.0.load(Ordering::SeqCst), 1);
}
