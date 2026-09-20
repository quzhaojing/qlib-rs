use std::{path::PathBuf, process::Command};

use domain_core::{
    RISK_ANALYSIS_FIELDS, RiskAnalysisError, RiskAnalysisWarning, RiskAnalysisWarningCategory,
    risk_analysis,
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct PythonWarning {
    category: String,
    message: String,
}

#[derive(Debug, Deserialize)]
struct PythonError {
    #[serde(rename = "type")]
    kind: String,
    message: String,
}

#[derive(Debug, Deserialize)]
struct PythonCase {
    name: String,
    #[serde(default)]
    schema: Vec<String>,
    #[serde(default)]
    column: Vec<String>,
    #[serde(default)]
    values: Vec<String>,
    #[serde(default)]
    warnings: Vec<PythonWarning>,
    error: Option<PythonError>,
}

#[derive(Debug, Deserialize)]
struct PythonSource {
    risk_analysis_sha256: String,
    freq_sha256: String,
}

#[derive(Debug, Deserialize)]
struct PythonPayload {
    source: PythonSource,
    cases: Vec<PythonCase>,
}

#[derive(Clone, Copy)]
struct RustCase {
    name: &'static str,
    returns: &'static [f64],
    n: Option<f64>,
    frequency: Option<&'static str>,
    mode: &'static str,
}

const NAN: f64 = f64::NAN;

const RUST_CASES: &[RustCase] = &[
    RustCase {
        name: "sum_day",
        returns: &[0.1, -0.2, 0.3, 0.0],
        n: None,
        frequency: Some("day"),
        mode: "sum",
    },
    RustCase {
        name: "sum_missing",
        returns: &[0.1, NAN, -0.2, 0.3],
        n: None,
        frequency: Some("2week"),
        mode: "sum",
    },
    RustCase {
        name: "sum_empty",
        returns: &[],
        n: None,
        frequency: Some("month"),
        mode: "sum",
    },
    RustCase {
        name: "sum_singleton",
        returns: &[0.25],
        n: Some(4.0),
        frequency: None,
        mode: "sum",
    },
    RustCase {
        name: "sum_all_nan",
        returns: &[NAN, NAN],
        n: Some(4.0),
        frequency: None,
        mode: "sum",
    },
    RustCase {
        name: "sum_positive_inf",
        returns: &[0.1, f64::INFINITY, -0.1],
        n: Some(4.0),
        frequency: None,
        mode: "sum",
    },
    RustCase {
        name: "sum_mixed_inf",
        returns: &[f64::INFINITY, f64::NEG_INFINITY],
        n: Some(4.0),
        frequency: None,
        mode: "sum",
    },
    RustCase {
        name: "precedence",
        returns: &[0.1, 0.2],
        n: Some(7.0),
        frequency: Some("not-a-freq"),
        mode: "sum",
    },
    RustCase {
        name: "missing_scaler",
        returns: &[0.1],
        n: None,
        frequency: None,
        mode: "sum",
    },
    RustCase {
        name: "invalid_freq",
        returns: &[0.1],
        n: None,
        frequency: Some("hour"),
        mode: "sum",
    },
    RustCase {
        name: "zero_freq",
        returns: &[0.1],
        n: None,
        frequency: Some("0day"),
        mode: "sum",
    },
    RustCase {
        name: "invalid_mode",
        returns: &[0.1],
        n: Some(4.0),
        frequency: None,
        mode: "median",
    },
    RustCase {
        name: "product_string_index",
        returns: &[0.1, -0.2, 0.3],
        n: Some(12.0),
        frequency: None,
        mode: "product",
    },
    RustCase {
        name: "product_missing_middle",
        returns: &[0.1, NAN, 0.2],
        n: Some(12.0),
        frequency: None,
        mode: "product",
    },
    RustCase {
        name: "product_missing_last",
        returns: &[0.1, 0.2, NAN],
        n: Some(12.0),
        frequency: None,
        mode: "product",
    },
    RustCase {
        name: "product_empty",
        returns: &[],
        n: Some(12.0),
        frequency: None,
        mode: "product",
    },
    RustCase {
        name: "product_minus_one",
        returns: &[0.1, -1.0, 0.2],
        n: Some(12.0),
        frequency: None,
        mode: "product",
    },
    RustCase {
        name: "product_below_minus_one",
        returns: &[0.1, -1.5, 0.2],
        n: Some(12.0),
        frequency: None,
        mode: "product",
    },
    RustCase {
        name: "product_positive_inf",
        returns: &[0.1, f64::INFINITY, 0.2],
        n: Some(12.0),
        frequency: None,
        mode: "product",
    },
    RustCase {
        name: "sum_zero_std",
        returns: &[0.25, 0.25],
        n: Some(4.0),
        frequency: None,
        mode: "sum",
    },
    RustCase {
        name: "sum_zero_over_zero",
        returns: &[0.0, 0.0],
        n: Some(4.0),
        frequency: None,
        mode: "sum",
    },
    RustCase {
        name: "sum_negative_scaler",
        returns: &[0.1, 0.2],
        n: Some(-4.0),
        frequency: None,
        mode: "sum",
    },
    RustCase {
        name: "product_cancellation",
        returns: &[-0.999_999_999_999_999_9, -0.999_999_999_999_999_9],
        n: Some(1.0),
        frequency: None,
        mode: "product",
    },
    RustCase {
        name: "product_overflow",
        returns: &[1e308, 1e308],
        n: Some(4.0),
        frequency: None,
        mode: "product",
    },
    RustCase {
        name: "product_invalid_accumulate",
        returns: &[f64::INFINITY, -1.0],
        n: Some(4.0),
        frequency: None,
        mode: "product",
    },
    RustCase {
        name: "product_negative_fractional_annual",
        returns: &[-1.5, 0.0],
        n: Some(3.0),
        frequency: None,
        mode: "product",
    },
    RustCase {
        name: "product_negative_singleton",
        returns: &[-1.5],
        n: Some(2.0),
        frequency: None,
        mode: "product",
    },
    RustCase {
        name: "product_negative_infinite_scaler",
        returns: &[-1.5, 0.0],
        n: Some(f64::INFINITY),
        frequency: None,
        mode: "product",
    },
    RustCase {
        name: "sum_pairwise_tail",
        returns: &[1e16, 1.0, -1e16, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        n: Some(4.0),
        frequency: None,
        mode: "sum",
    },
    RustCase {
        name: "sum_mean_overflow",
        returns: &[1e308, 1e308],
        n: Some(1.0),
        frequency: None,
        mode: "sum",
    },
];

fn python_payload() -> PythonPayload {
    let upstream = std::env::var_os("QLIB_PYTHON_SOURCE")
        .map_or_else(|| PathBuf::from(r"D:\code\github\qlib"), PathBuf::from);
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/risk_analysis_contract.py");
    let output = Command::new(std::env::var_os("PYTHON").unwrap_or_else(|| "python".into()))
        .arg(fixture)
        .arg(upstream)
        .env("PYTHONUTF8", "1")
        .output()
        .expect("Python characterization starts");
    assert!(
        output.status.success(),
        "Python characterization failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("fixture emits valid JSON")
}

fn parse_float(value: &str) -> f64 {
    match value {
        "nan" => f64::NAN,
        "+inf" => f64::INFINITY,
        "-inf" => f64::NEG_INFINITY,
        value => value.parse().expect("fixture float is valid"),
    }
}

fn assert_float_matches(actual: f64, expected: f64, context: &str) {
    if expected.is_nan() {
        assert!(actual.is_nan(), "{context}: expected NaN, got {actual:?}");
    } else if expected.is_infinite() {
        assert_eq!(actual.to_bits(), expected.to_bits(), "{context}");
    } else {
        let tolerance = 8.0 * f64::EPSILON * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{context}: expected {expected:.17}, got {actual:.17}, tolerance {tolerance:e}"
        );
    }
}

fn warning_pair(warning: &RiskAnalysisWarning) -> (&'static str, &str) {
    let category = match warning.category {
        RiskAnalysisWarningCategory::UserWarning => "UserWarning",
        RiskAnalysisWarningCategory::RuntimeWarning => "RuntimeWarning",
    };
    (category, &warning.message)
}

#[test]
fn source_pinned_risk_analysis_matches_values_schema_warnings_and_errors() {
    let payload = python_payload();
    assert_eq!(
        payload.source.risk_analysis_sha256,
        "858ff3db9498c8fd70e40f720525267374432ae8cde2af90da4595032e1ee909"
    );
    assert_eq!(
        payload.source.freq_sha256,
        "2e8a09395316e21b111e4ae8f6a8c9893c84ca4f9cb4bc9031d8513e20ca0733"
    );
    let python = payload.cases;
    assert_eq!(python.len(), RUST_CASES.len() + 1);

    for rust_case in RUST_CASES {
        let expected = python
            .iter()
            .find(|case| case.name == rust_case.name)
            .expect("each Rust case has a Python result");
        let actual = risk_analysis(
            rust_case.returns,
            rust_case.n,
            rust_case.frequency,
            rust_case.mode,
        );

        if let Some(error) = &expected.error {
            let actual = actual.expect_err("Python failure must remain a Rust failure");
            assert_eq!(actual.to_string(), error.message, "{}", rust_case.name);
            let expected_kind = match actual {
                RiskAnalysisError::MissingAnnualization
                | RiskAnalysisError::InvalidFrequency { .. }
                | RiskAnalysisError::UnsupportedMode { .. } => "ValueError",
                RiskAnalysisError::ZeroFrequency => "ZeroDivisionError",
                RiskAnalysisError::EmptyProduct => "IndexError",
            };
            assert_eq!(expected_kind, error.kind, "{}", rust_case.name);
            continue;
        }

        let actual = actual.expect("Python success must remain a Rust success");
        assert_eq!(expected.schema, RISK_ANALYSIS_FIELDS, "{}", rust_case.name);
        assert_eq!(expected.column, ["risk"], "{}", rust_case.name);
        for (index, (actual, expected)) in actual
            .result
            .ordered_values()
            .into_iter()
            .zip(expected.values.iter().map(|value| parse_float(value)))
            .enumerate()
        {
            assert_float_matches(
                actual,
                expected,
                &format!("{} value {index}", rust_case.name),
            );
        }
        let actual_warnings: Vec<_> = actual.warnings.iter().map(warning_pair).collect();
        let expected_warnings: Vec<_> = expected
            .warnings
            .iter()
            .map(|warning| (warning.category.as_str(), warning.message.as_str()))
            .collect();
        assert_eq!(actual_warnings, expected_warnings, "{}", rust_case.name);
    }
}

#[test]
fn numpy_pairwise_long_reductions_and_serde_roundtrip_match_contract() {
    let expected = python_payload()
        .cases
        .into_iter()
        .find(|case| case.name == "sum_long_cancellation")
        .expect("fixture includes the long reduction case");
    let returns: Vec<_> = [1e16, 1.0, -1e16, 1.0].repeat(1024);
    let actual = risk_analysis(&returns, Some(4.0), None, "sum").unwrap();
    for (index, (actual, expected)) in actual
        .result
        .ordered_values()
        .into_iter()
        .zip(expected.values.iter().map(|value| parse_float(value)))
        .enumerate()
    {
        assert_float_matches(actual, expected, &format!("long reduction value {index}"));
    }

    let serializable = risk_analysis(&[0.1, 0.2], Some(7.0), Some("ignored"), "sum").unwrap();
    let encoded = serde_json::to_string(&serializable).expect("finite output serializes");
    let decoded: domain_core::RiskAnalysisOutput =
        serde_json::from_str(&encoded).expect("output deserializes");
    for (index, (actual, expected)) in decoded
        .result
        .ordered_values()
        .into_iter()
        .zip(serializable.result.ordered_values())
        .enumerate()
    {
        assert_float_matches(actual, expected, &format!("serde value {index}"));
    }
    assert_eq!(serializable.warnings, decoded.warnings);
}

#[test]
fn frequency_aliases_scalers_and_large_counts_follow_existing_parser() {
    let one = [0.25, -0.125];
    let minute = risk_analysis(&one, None, Some("2MINUTE"), "sum").unwrap();
    let day = risk_analysis(&one, None, Some("2D"), "sum").unwrap();
    let week = risk_analysis(&one, None, Some("2W"), "sum").unwrap();
    let month = risk_analysis(&one, None, Some("2mon"), "sum").unwrap();
    assert_float_matches(minute.result.annualized_return, 0.0625 * 28_560.0, "minute");
    assert_float_matches(day.result.annualized_return, 0.0625 * 119.0, "day");
    assert_float_matches(week.result.annualized_return, 0.0625 * 25.0, "week");
    assert_float_matches(month.result.annualized_return, 0.0625 * 6.0, "month");

    let huge = format!("1{}day", "0".repeat(400));
    let result = risk_analysis(&one, None, Some(&huge), "sum").unwrap();
    assert_eq!(result.result.annualized_return.to_bits(), 0.0_f64.to_bits());
}

#[test]
fn ordered_values_preserve_public_field_order() {
    let output = risk_analysis(&[-0.0, 0.0], Some(1.0), None, "sum").unwrap();
    assert_eq!(RISK_ANALYSIS_FIELDS[0], "mean");
    assert_eq!(RISK_ANALYSIS_FIELDS[4], "max_drawdown");
    assert_eq!(
        output.result.ordered_values().len(),
        RISK_ANALYSIS_FIELDS.len()
    );
    assert_eq!(output.result.max_drawdown.to_bits(), 0.0_f64.to_bits());
}
