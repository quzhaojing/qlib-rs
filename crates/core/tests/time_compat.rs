use std::{path::PathBuf, process::Command, str::FromStr};

use arrow_schema::TimeUnit;
use chrono::{DateTime, NaiveDate, NaiveTime};
use domain_core::{
    EpsilonError, Frequency, FrequencyError, FrequencyUnit, MarketCalendarError,
    MinuteAlignmentError, Region,
    dataframe_append::{BuiltinFrameValue, TemporalFrameValue},
    time_calendar_cache::{TimeCalendarCache, TimeCalendarCacheError, TimeCalendarCall},
    time_compat::{
        CompatibleFrequency, TimeCompatError, align_sampled_minute_compatible,
        align_sampled_minute_with_cache, concat_date_time_compatible, epsilon_change_compatible,
        is_single_value_compatible, recent_frequency_compatible, time_delta_compatible,
    },
};
use num_bigint::{BigInt, BigUint};
use serde_json::{Value, json};

const UPSTREAM_TIME_SHA256: &str =
    "af7ac3709cac0d2a11a15aac478c7ceb68579d492aed322695f5f25261a69699";

fn timestamp(ticks: i64, unit: TimeUnit, timezone: Option<&str>) -> TemporalFrameValue {
    TemporalFrameValue::Timestamp {
        ticks,
        unit,
        timezone: timezone.map(Into::into),
    }
}

fn utc_nanos(text: &str) -> i64 {
    DateTime::parse_from_rfc3339(text)
        .expect("fixture timestamp is valid")
        .timestamp_nanos_opt()
        .expect("fixture timestamp fits nanoseconds")
}

fn source_snapshot() -> Value {
    source_snapshot_mode(None)
}

fn source_snapshot_mode(mode: Option<&str>) -> Value {
    let source = std::env::var_os("QLIB_PYTHON_TIME").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/utils/time.py"),
        PathBuf::from,
    );
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/time_compat_probe.py");
    let python = std::env::var_os("PYTHON").unwrap_or_else(|| "python".into());
    let output = Command::new(python)
        .arg(fixture)
        .arg(source)
        .args(mode)
        .output()
        .expect("Python interpreter starts");
    assert!(
        output.status.success(),
        "source characterization failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let actual: Value = serde_json::from_slice(&output.stdout).expect("fixture returns JSON");
    assert_eq!(actual["source_sha256"], UPSTREAM_TIME_SHA256);
    actual
}

#[test]
fn native_text_range_preserves_early_errors_and_reports_unimplemented_parsing() {
    use domain_core::time_compat::day_minute_text_range_with_cache;

    let snapshot = source_snapshot_mode(Some("--range-text"));
    let inputs = snapshot["inputs"].as_array().unwrap();
    let reference = NaiveDate::from_ymd_opt(2021, 1, 1).unwrap();
    let mut resolved = 0;
    for case in snapshot["cases"].as_array().unwrap() {
        let input = |key: &str| {
            inputs[usize::try_from(case[key].as_u64().unwrap()).unwrap()]["text"]
                .as_str()
                .unwrap()
        };
        let cache = TimeCalendarCache::new();
        if let Some(result) = day_minute_text_range_with_cache(
            input("start"),
            input("end"),
            case["frequency"].as_str().unwrap(),
            case["region"].as_str().unwrap(),
            reference,
            &cache,
        ) {
            assert_eq!(text_range_json(result), case["result"], "{case}");
            assert_eq!(
                json!(cache.cache_info().misses),
                case["cache_misses"],
                "{case}"
            );
            assert!(case["warnings"].as_array().unwrap().is_empty(), "{case}");
            resolved += 1;
        } else {
            assert_eq!(cache.cache_info().misses, 0, "{case}");
        }
    }
    // The remaining cases are explicitly deferred, not counted as passing parity.
    assert_eq!(resolved, 5264);
    let cache = TimeCalendarCache::new();
    let out_of_bounds = "1000-01-01T00:00:00.000000001";
    for (start, end) in [(out_of_bounds, "unknown"), ("9:30", out_of_bounds)] {
        let result =
            day_minute_text_range_with_cache(start, end, "bad", "CN", reference, &cache).unwrap();
        assert!(matches!(
            result,
            Err(TimeCompatError::TimestampTextBounds(_))
        ));
        assert_eq!(
            text_range_json(result),
            json!({"error": "OutOfBoundsDatetime",
            "message": "Out of bounds nanosecond timestamp: 1000-01-01 00:00:00"})
        );
    }
    assert_eq!(cache.cache_info().misses, 0);
}

#[test]
fn native_constructor_missing_markers_are_exact_and_precede_clock_validation() {
    use domain_core::timestamp_text::{TimestampTextValue, parse_timestamp_text_compatible};

    let snapshot = source_snapshot_mode(Some("--range-text"));
    let invalid_reference = NaiveDate::from_ymd_opt(0, 1, 1).unwrap();
    let mut missing = 0;
    for case in snapshot["markers"].as_array().unwrap() {
        let actual =
            parse_timestamp_text_compatible(case["text"].as_str().unwrap(), invalid_reference);
        if case["result"]["ok"]["kind"] == "NaT" {
            assert_eq!(actual, Some(Ok(TimestampTextValue::NotATime)), "{case}");
            missing += 1;
        } else {
            assert_eq!(actual, None, "{case}");
        }
    }
    assert_eq!(missing, 7);
}

fn text_range_json(result: Result<(i64, i64), TimeCompatError>) -> Value {
    match result {
        Ok((left, right)) => json!({"ok": [left, right]}),
        Err(error) => {
            let category = match error {
                TimeCompatError::TimestampTextBounds(_) => "OutOfBoundsDatetime",
                TimeCompatError::TimestampTextOffsetOverflow => "OverflowError",
                TimeCompatError::TimestampTextDateParse { .. } => "DateParseError",
                _ => "ValueError",
            };
            json!({"error": category, "message": error.to_string()})
        }
    }
}

fn assert_native_text_range(
    case: &Value,
    reference: impl Into<domain_core::timestamp_text::TimestampTextContext>,
) {
    let cache = TimeCalendarCache::new();
    let result = domain_core::time_compat::day_minute_text_range_with_cache(
        case["text"].as_str().unwrap(),
        "2021-01-01 14:59",
        "1min",
        "cn",
        reference,
        &cache,
    )
    .unwrap_or_else(|| panic!("missing constructor stage: {case}"));
    assert_eq!(text_range_json(result), case["range"], "{case}");
}

fn assert_native_date_token(
    case: &Value,
    reference: impl Into<domain_core::timestamp_text::TimestampTextContext>,
) {
    use domain_core::timestamp_text::parse_timestamp_text_compatible;

    let reference = reference.into();
    assert_native_text_range(case, reference);
    let result = parse_timestamp_text_compatible(case["text"].as_str().unwrap(), reference)
        .unwrap_or_else(|| panic!("date token should be recognized: {case}"));
    assert_constructor_result(case, result);
}

fn assert_constructor_result(
    case: &Value,
    result: Result<
        domain_core::timestamp_text::TimestampTextValue,
        domain_core::timestamp_text::TimestampTextError,
    >,
) {
    use chrono::{Datelike, Timelike};
    use domain_core::timestamp_text::{TimestampTextError, TimestampTextValue};
    match result {
        Ok(TimestampTextValue::Timestamp(value)) => {
            let expected = &case["result"]["ok"];
            assert_eq!(expected["kind"], "Timestamp", "{case}");
            let unit = match expected["unit"].as_str().unwrap() {
                "s" => TimeUnit::Second,
                "ms" => TimeUnit::Millisecond,
                "us" => TimeUnit::Microsecond,
                "ns" => TimeUnit::Nanosecond,
                other => panic!("unexpected unit {other}"),
            };
            assert_eq!(
                value.to_temporal(),
                timestamp(
                    expected["ticks"].as_i64().unwrap(),
                    unit,
                    expected["timezone"].as_str()
                ),
                "{case}"
            );
            let local = value.local_datetime;
            assert_eq!(
                json!([
                    local.year(),
                    local.month(),
                    local.day(),
                    local.hour(),
                    local.minute(),
                    local.second(),
                    local.nanosecond() / 1000,
                    local.nanosecond() % 1000
                ]),
                expected["components"],
                "{case}"
            );
        }
        Ok(TimestampTextValue::NotATime) => panic!("date token cannot produce NaT: {case}"),
        Err(error) => {
            let category = match error {
                TimestampTextError::OutOfBounds(_) => "OutOfBoundsDatetime",
                TimestampTextError::OffsetOverflow => "OverflowError",
                TimestampTextError::DateParse { .. } => "DateParseError",
                TimestampTextError::NotDateLike { .. }
                | TimestampTextError::InvalidOffset { .. }
                | TimestampTextError::InvalidCalendar { .. } => "ValueError",
            };
            assert_eq!(
                json!({"error": category, "message": error.to_string()}),
                case["result"],
                "{case}"
            );
        }
    }
}

#[test]
fn parser_initialization_year_is_independent_of_current_date() {
    use domain_core::timestamp_text::TimestampTextContext;

    let snapshot = source_snapshot_mode(Some("--parser-context"));
    assert_eq!(snapshot["pandas_version"], "2.3.3");
    let cases = snapshot["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 1632);
    for case in cases {
        let reference_date =
            NaiveDate::parse_from_str(case["reference_date"].as_str().unwrap(), "%Y-%m-%d")
                .unwrap();
        let parser_year = i32::try_from(case["parser_year"].as_i64().unwrap()).unwrap();
        assert_native_date_token(
            case,
            TimestampTextContext {
                reference_date,
                parser_initialized_date: NaiveDate::from_ymd_opt(parser_year, 1, 1).unwrap(),
            },
        );
    }
}

#[test]
fn date_token_reference_boundaries_and_compact_grammar_remain_distinct() {
    use chrono::Datelike;
    use domain_core::timestamp_text::{
        TimestampTextError, TimestampTextValue, parse_timestamp_text_compatible,
    };

    let future = NaiveDate::from_ymd_opt(2099, 9, 21).unwrap();
    let result = parse_timestamp_text_compatible("9:30:0,32", future)
        .unwrap()
        .unwrap();
    let TimestampTextValue::Timestamp(value) = result else {
        panic!("expected timestamp")
    };
    assert_eq!(value.local_datetime.year(), 2132);
    for (reference, text, message) in [
        (
            NaiveDate::from_ymd_opt(1, 1, 1).unwrap(),
            "9:30:0,99",
            "year must be in 1..9999, not -1: 9:30:0,99",
        ),
        (
            NaiveDate::from_ymd_opt(2024, 2, 29).unwrap(),
            "9:30:0,2023",
            "day 29 must be in range 1..28 for month 2 in year 2023: 9:30:0,2023",
        ),
        (
            NaiveDate::from_ymd_opt(9999, 12, 31).unwrap(),
            "9:30:0,0",
            "day 0 must be in range 1..31 for month 12 in year 9999: 9:30:0,0",
        ),
    ] {
        let error = parse_timestamp_text_compatible(text, reference)
            .unwrap()
            .unwrap_err();
        assert!(matches!(error, TimestampTextError::DateParse { .. }));
        assert_eq!(error.to_string(), message);
    }
    for text in ["24:30:0.1", "24:30:00,1", "24:30:0,"] {
        let error = parse_timestamp_text_compatible(text, future)
            .unwrap()
            .unwrap_err();
        assert!(matches!(error, TimestampTextError::DateParse { .. }));
        assert_eq!(
            error.to_string(),
            format!("hour must be in 0..23, not 24: {text}")
        );
    }
    assert_eq!(
        parse_timestamp_text_compatible("9:30:0,32", NaiveDate::from_ymd_opt(0, 1, 1).unwrap()),
        None
    );
}

#[test]
fn decimal_clock_day_period_preserves_existing_date_and_offset_fields() {
    let snapshot = source_snapshot_mode(Some("--decimal-day-period"));
    assert_eq!(snapshot["pandas_version"], "2.3.3");
    let cases = snapshot["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 9000);
    let year = i32::try_from(snapshot["parser_year"].as_i64().unwrap()).unwrap();
    for case in cases {
        assert_native_date_token(case, NaiveDate::from_ymd_opt(year, 1, 1).unwrap());
    }
}

#[test]
fn compact_pairs_preserve_clock_offset_and_decimal_dispatch() {
    let snapshot = source_snapshot_mode(Some("--compact-pair"));
    assert_eq!(snapshot["pandas_version"], "2.3.3");
    let cases = snapshot["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 8329);
    let year = i32::try_from(snapshot["parser_year"].as_i64().unwrap()).unwrap();
    for case in cases {
        assert_native_date_token(case, NaiveDate::from_ymd_opt(year, 1, 1).unwrap());
    }
}

#[test]
fn wide_year_month_preserves_lexical_dispatch_and_overflow_order() {
    let snapshot = source_snapshot_mode(Some("--wide-year-month"));
    assert_eq!(snapshot["pandas_version"], "2.3.3");
    let cases = snapshot["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 9198);
    let year = i32::try_from(snapshot["parser_year"].as_i64().unwrap()).unwrap();
    for case in cases {
        assert_native_date_token(case, NaiveDate::from_ymd_opt(year, 1, 1).unwrap());
    }
}

#[test]
fn hour_period_zones_preserve_defaults_offsets_and_error_order() {
    let snapshot = source_snapshot_mode(Some("--hour-period-zone"));
    assert_eq!(snapshot["pandas_version"], "2.3.3");
    let cases = snapshot["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 5040);
    let year = i32::try_from(snapshot["parser_year"].as_i64().unwrap()).unwrap();
    for case in cases {
        assert_native_date_token(case, NaiveDate::from_ymd_opt(year, 1, 1).unwrap());
    }
}

#[test]
fn year_month_preserves_iso_and_general_numeric_dispatch() {
    let snapshot = source_snapshot_mode(Some("--year-month"));
    assert_eq!(snapshot["pandas_version"], "2.3.3");
    let cases = snapshot["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 7398);
    let year = i32::try_from(snapshot["parser_year"].as_i64().unwrap()).unwrap();
    for case in cases {
        assert_native_date_token(case, NaiveDate::from_ymd_opt(year, 1, 1).unwrap());
    }
}

#[test]
fn compact_mixed_separators_preserve_iso_identity_and_decimal_clocks() {
    let snapshot = source_snapshot_mode(Some("--compact-mixed"));
    assert_eq!(snapshot["pandas_version"], "2.3.3");
    let cases = snapshot["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 14409);
    let year = i32::try_from(snapshot["parser_year"].as_i64().unwrap()).unwrap();
    for case in cases {
        assert_native_date_token(case, NaiveDate::from_ymd_opt(year, 1, 1).unwrap());
    }
}

#[test]
fn compact_following_tokens_preserve_clock_and_timezone_replacement() {
    let snapshot = source_snapshot_mode(Some("--compact-following"));
    assert_eq!(snapshot["pandas_version"], "2.3.3");
    let cases = snapshot["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 8721);
    let year = i32::try_from(snapshot["parser_year"].as_i64().unwrap()).unwrap();
    for case in cases {
        assert_native_date_token(case, NaiveDate::from_ymd_opt(year, 1, 1).unwrap());
    }
}

#[test]
fn wide_year_dates_preserve_lexical_calendar_and_overflow_precedence() {
    let snapshot = source_snapshot_mode(Some("--wide-year"));
    assert_eq!(snapshot["pandas_version"], "2.3.3");
    let cases = snapshot["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 8676);
    let year = i32::try_from(snapshot["parser_year"].as_i64().unwrap()).unwrap();
    for case in cases {
        assert_native_date_token(case, NaiveDate::from_ymd_opt(year, 1, 1).unwrap());
    }
}

#[test]
fn year_first_dates_preserve_iso_dispatch_precision_and_general_errors() {
    let snapshot = source_snapshot_mode(Some("--year-first"));
    assert_eq!(snapshot["pandas_version"], "2.3.3");
    let cases = snapshot["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 8676);
    let parser_year = i32::try_from(snapshot["parser_year"].as_i64().unwrap()).unwrap();
    for case in cases {
        assert_native_date_token(case, NaiveDate::from_ymd_opt(parser_year, 1, 1).unwrap());
    }
    // Source accepts mixed separators via a different general-token path; do
    // not incorrectly promote them into ISO or invent a rejection for them.
    assert_eq!(
        domain_core::timestamp_text::parse_timestamp_text_compatible(
            "2021/01-02",
            NaiveDate::from_ymd_opt(parser_year, 1, 1).unwrap(),
        ),
        None
    );
}

#[test]
fn pure_clock_errors_preserve_constructor_and_date_token_precedence() {
    let snapshot = source_snapshot_mode(Some("--clock-errors"));
    assert_eq!(snapshot["pandas_version"], "2.3.3");
    let cases = snapshot["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 3168);
    for case in cases {
        let reference =
            NaiveDate::parse_from_str(case["reference_date"].as_str().unwrap(), "%Y-%m-%d")
                .unwrap();
        assert_native_date_token(case, reference);
    }
}

#[test]
fn explicit_month_years_preserve_additional_day_and_duplicate_year_errors() {
    let snapshot = source_snapshot_mode(Some("--explicit-month-day"));
    assert_eq!(snapshot["pandas_version"], "2.3.3");
    let cases = snapshot["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 23760);
    let parser_year = i32::try_from(snapshot["parser_year"].as_i64().unwrap()).unwrap();
    for case in cases {
        assert_native_date_token(case, NaiveDate::from_ymd_opt(parser_year, 1, 1).unwrap());
    }
}

#[test]
fn month_date_compact_clocks_preserve_precision_period_and_error_order() {
    let snapshot = source_snapshot_mode(Some("--month-compact-clock"));
    assert_eq!(snapshot["pandas_version"], "2.3.3");
    let cases = snapshot["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 1179);
    let parser_year = i32::try_from(snapshot["parser_year"].as_i64().unwrap()).unwrap();
    for case in cases {
        assert_native_date_token(case, NaiveDate::from_ymd_opt(parser_year, 1, 1).unwrap());
    }
}

#[test]
fn partial_month_names_preserve_defaults_and_explicit_year_markers() {
    let snapshot = source_snapshot_mode(Some("--partial-month"));
    assert_eq!(snapshot["pandas_version"], "2.3.3");
    let cases = snapshot["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 3294);
    let parser_year = i32::try_from(snapshot["parser_year"].as_i64().unwrap()).unwrap();
    for case in cases {
        assert_native_date_token(case, NaiveDate::from_ymd_opt(parser_year, 1, 1).unwrap());
    }
}

#[test]
fn edge_month_names_preserve_token_order_and_century_markers() {
    let snapshot = source_snapshot_mode(Some("--edge-month"));
    assert_eq!(snapshot["pandas_version"], "2.3.3");
    let cases = snapshot["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 47250);
    let parser_year = i32::try_from(snapshot["parser_year"].as_i64().unwrap()).unwrap();
    for case in cases {
        assert_native_date_token(case, NaiveDate::from_ymd_opt(parser_year, 1, 1).unwrap());
    }
}

#[test]
fn decimal_month_following_numbers_preserve_year_day_and_clock_resolution() {
    let snapshot = source_snapshot_mode(Some("--decimal-month-following"));
    assert_eq!(snapshot["pandas_version"], "2.3.3");
    let cases = snapshot["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 5400);
    let parser_year = i32::try_from(snapshot["parser_year"].as_i64().unwrap()).unwrap();
    for case in cases {
        assert_native_date_token(case, NaiveDate::from_ymd_opt(parser_year, 1, 1).unwrap());
    }
}

#[test]
fn ordinal_month_dates_preserve_tokenization_century_and_errors() {
    let snapshot = source_snapshot_mode(Some("--ordinal-month"));
    assert_eq!(snapshot["pandas_version"], "2.3.3");
    let cases = snapshot["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 52092);
    let parser_year = i32::try_from(snapshot["parser_year"].as_i64().unwrap()).unwrap();
    for case in cases {
        assert_native_date_token(case, NaiveDate::from_ymd_opt(parser_year, 1, 1).unwrap());
    }
}

#[test]
fn middle_month_names_preserve_separator_centuries_and_date_order() {
    let snapshot = source_snapshot_mode(Some("--middle-month"));
    assert_eq!(snapshot["pandas_version"], "2.3.3");
    let cases = snapshot["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 21000);
    let parser_year = i32::try_from(snapshot["parser_year"].as_i64().unwrap()).unwrap();
    for case in cases {
        assert_native_date_token(case, NaiveDate::from_ymd_opt(parser_year, 1, 1).unwrap());
    }
}

#[test]
fn month_names_preserve_year_resolution_calendar_and_clock_errors() {
    let snapshot = source_snapshot_mode(Some("--month-date"));
    assert_eq!(snapshot["pandas_version"], "2.3.3");
    let cases = snapshot["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 16640);
    let parser_year = i32::try_from(snapshot["parser_year"].as_i64().unwrap()).unwrap();
    for case in cases {
        assert_native_date_token(case, NaiveDate::from_ymd_opt(parser_year, 1, 1).unwrap());
    }
    assert_eq!(
        domain_core::timestamp_text::parse_timestamp_text_compatible(
            "Notamonth 1, 2021",
            NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
        ),
        None
    );
}

#[test]
fn general_delimited_dates_preserve_mixed_separators_and_fallbacks() {
    let snapshot = source_snapshot_mode(Some("--general-delimited"));
    assert_eq!(snapshot["pandas_version"], "2.3.3");
    let cases = snapshot["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 4615);
    let parser_year = i32::try_from(snapshot["parser_year"].as_i64().unwrap()).unwrap();
    for case in cases {
        assert_native_date_token(case, NaiveDate::from_ymd_opt(parser_year, 1, 1).unwrap());
    }
}

#[test]
fn decimal_date_tokens_preserve_hour_replacement_and_lexical_errors() {
    let snapshot = source_snapshot_mode(Some("--decimal-date-token"));
    assert_eq!(snapshot["pandas_version"], "2.3.3");
    let cases = snapshot["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 1800);
    let parser_year = i32::try_from(snapshot["parser_year"].as_i64().unwrap()).unwrap();
    for case in cases {
        assert_native_date_token(case, NaiveDate::from_ymd_opt(parser_year, 1, 1).unwrap());
    }
}

#[test]
fn date_clock_composition_preserves_precision_and_error_order() {
    let snapshot = source_snapshot_mode(Some("--date-clock"));
    assert_eq!(snapshot["pandas_version"], "2.3.3");
    let cases = snapshot["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 7168);
    for case in cases {
        assert_native_date_token(case, NaiveDate::from_ymd_opt(2026, 1, 1).unwrap());
    }
    assert_eq!(
        domain_core::timestamp_text::parse_timestamp_text_compatible(
            "01/02/2021 bad",
            NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
        ),
        None,
    );
}

#[test]
fn delimited_dates_preserve_fast_parser_order_and_error_categories() {
    let snapshot = source_snapshot_mode(Some("--delimited-date"));
    assert_eq!(snapshot["pandas_version"], "2.3.3");
    let cases = snapshot["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 9300);
    for case in cases {
        assert_native_date_token(case, NaiveDate::from_ymd_opt(2026, 1, 1).unwrap());
    }
    assert_eq!(
        domain_core::timestamp_text::parse_timestamp_text_compatible(
            "02.2021",
            NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
        ),
        None,
    );
}

#[test]
fn standalone_numeric_dates_preserve_iso_precedence_and_general_errors() {
    let snapshot = source_snapshot_mode(Some("--numeric-date"));
    assert_eq!(snapshot["pandas_version"], "2.3.3");
    let cases = snapshot["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 2025);
    for case in cases {
        let reference =
            NaiveDate::parse_from_str(snapshot["reference_date"].as_str().unwrap(), "%Y-%m-%d")
                .unwrap();
        assert_native_date_token(case, reference);
    }
    // Four-digit numeric AM/PM forms need the general numeric-hour grammar.
    assert_eq!(
        domain_core::timestamp_text::parse_timestamp_text_compatible(
            "0001 PM",
            NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
        ),
        None,
    );
}

#[test]
fn native_compact_date_tokens_preserve_calendar_and_clock_replacement() {
    let snapshot = source_snapshot_mode(Some("--compact-token"));
    assert_eq!(snapshot["pandas_version"], "2.3.3");
    let cases = snapshot["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 2880);
    for case in cases {
        let reference =
            NaiveDate::parse_from_str(case["reference_date"].as_str().unwrap(), "%Y-%m-%d")
                .unwrap();
        assert_native_date_token(case, reference);
    }
}

#[test]
fn native_clock_date_tokens_preserve_numeric_dispatch_and_error_categories() {
    let snapshot = source_snapshot_mode(Some("--date-token"));
    assert_eq!(snapshot["pandas_version"], "2.3.3");
    let cases = snapshot["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 4176);
    for case in cases {
        let reference =
            NaiveDate::parse_from_str(case["reference_date"].as_str().unwrap(), "%Y-%m-%d")
                .unwrap();
        assert_native_date_token(case, reference);
    }
}

#[test]
fn clock_reference_date_bounds_do_not_erase_stored_calendar_fields() {
    use domain_core::timestamp_text::parse_clock_timestamp_compatible;

    let reference = NaiveDate::from_ymd_opt(9999, 12, 31).unwrap();
    let error = parse_clock_timestamp_compatible("9:30:00.000000001", reference)
        .unwrap()
        .unwrap_err();
    assert_eq!(error.to_string(), "Out of bounds nanosecond timestamp");
    assert_eq!(
        parse_clock_timestamp_compatible("9:30", NaiveDate::from_ymd_opt(10000, 1, 1).unwrap()),
        None
    );

    let parsed = parse_clock_timestamp_compatible("9:30:0.1", reference)
        .unwrap()
        .unwrap();
    assert_eq!(parsed.local_datetime.date(), reference);
    assert_eq!(
        parsed.time(),
        NaiveTime::from_hms_micro_opt(9, 30, 0, 100_000).unwrap()
    );
    let TemporalFrameValue::Timestamp {
        ticks,
        unit,
        timezone,
    } = parsed.to_temporal()
    else {
        panic!("clock constructor must produce a timestamp");
    };
    assert_eq!(unit, TimeUnit::Second);
    assert_eq!(
        ticks,
        reference
            .and_hms_opt(9, 30, 0)
            .unwrap()
            .and_utc()
            .timestamp()
    );
    assert_eq!(timezone, None);
}

#[test]
fn native_clock_constructor_preserves_default_date_and_fractional_resolution() {
    use chrono::{Datelike, Timelike};
    use domain_core::{
        time_compat::day_minute_index_range_with_cache,
        timestamp_text::parse_clock_timestamp_compatible,
    };

    let snapshot = source_snapshot_mode(Some("--clock-text"));
    assert_eq!(snapshot["pandas_version"], "2.3.3");
    let cases = snapshot["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 5148);
    let end = NaiveTime::from_hms_opt(14, 59, 0).unwrap();
    let frequency: Frequency = "1min".parse().unwrap();
    let cache = TimeCalendarCache::new();
    let mut resolved_date_tokens = 0;
    for case in cases {
        let text = case["text"].as_str().unwrap();
        let reference =
            NaiveDate::parse_from_str(case["reference_date"].as_str().unwrap(), "%Y-%m-%d")
                .unwrap();
        if case["general_date_token"] == true {
            assert_eq!(
                parse_clock_timestamp_compatible(text, reference),
                None,
                "{case}"
            );
            assert_native_date_token(case, reference);
            resolved_date_tokens += 1;
            continue;
        }
        assert_native_text_range(case, reference);
        let actual = parse_clock_timestamp_compatible(text, reference)
            .unwrap_or_else(|| panic!("clock should be recognized: {case}"));
        match actual {
            Ok(value) => {
                let expected = &case["result"]["ok"];
                assert_eq!(expected["kind"], "Timestamp", "{case}");
                let unit = match expected["unit"].as_str().unwrap() {
                    "s" => TimeUnit::Second,
                    "ms" => TimeUnit::Millisecond,
                    "us" => TimeUnit::Microsecond,
                    "ns" => TimeUnit::Nanosecond,
                    _ => unreachable!(),
                };
                assert_eq!(
                    value.to_temporal(),
                    timestamp(expected["ticks"].as_i64().unwrap(), unit, None),
                    "{case}"
                );
                assert!(expected["timezone"].is_null(), "{case}");
                let local = value.local_datetime;
                assert_eq!(
                    json!([
                        local.year(),
                        local.month(),
                        local.day(),
                        local.hour(),
                        local.minute(),
                        local.second(),
                        local.nanosecond() / 1000,
                        local.nanosecond() % 1000
                    ]),
                    expected["components"],
                    "{case}"
                );
                let range =
                    day_minute_index_range_with_cache(value.time(), end, &frequency, "cn", &cache)
                        .unwrap();
                assert_eq!(json!({"ok": [range.0, range.1]}), case["range"], "{case}");
            }
            Err(error) => {
                let actual = json!({"error": "OutOfBoundsDatetime", "message": error.to_string()});
                assert_eq!(actual, case["result"], "{case}");
                assert_eq!(actual, case["range"], "{case}");
            }
        }
    }
    assert_eq!(resolved_date_tokens, 396);
    let reference = NaiveDate::from_ymd_opt(2021, 1, 1).unwrap();
    for text in ["13:30 PM", "24:00", "09:60", "9:30:60", "", "NaT", "9:30Z"] {
        assert_eq!(
            parse_clock_timestamp_compatible(text, reference),
            None,
            "{text}"
        );
    }
    assert_eq!(
        parse_clock_timestamp_compatible("9:30", NaiveDate::from_ymd_opt(0, 1, 1).unwrap()),
        None
    );
}

#[test]
fn native_iso_constructor_preserves_resolution_offsets_bounds_and_range_behavior() {
    assert_iso_constructor_snapshot("--iso-text", 3392);
}

#[test]
fn iso_offsets_preserve_variable_width_whitespace_and_overflow() {
    assert_iso_constructor_snapshot("--iso-offset", 12803);
}

#[test]
fn named_zone_range_warnings_preserve_endpoint_and_frequency_order() {
    use domain_core::time_compat::day_minute_text_range_with_warnings;
    use domain_core::timestamp_text::{
        TimestampTextContext, TimestampTextZoneContext, parse_timestamp_text_with_warnings,
    };
    let snapshot = source_snapshot_mode(Some("--named-zone-range"));
    let names: Vec<&str> = snapshot["local_timezone_names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|name| name.as_str().unwrap())
        .collect();
    let year = i32::try_from(snapshot["parser_year"].as_i64().unwrap()).unwrap();
    let context = TimestampTextZoneContext {
        timestamp: TimestampTextContext {
            reference_date: NaiveDate::parse_from_str(
                snapshot["reference_date"].as_str().unwrap(),
                "%Y-%m-%d",
            )
            .unwrap(),
            parser_initialized_date: NaiveDate::from_ymd_opt(year, 1, 1).unwrap(),
        },
        local_timezone_names: &names,
    };
    let cases = snapshot["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 100);
    for case in cases {
        let result = day_minute_text_range_with_warnings(
            case["start"].as_str().unwrap(),
            case["end"].as_str().unwrap(),
            case["frequency"].as_str().unwrap(),
            case["region"].as_str().unwrap(),
            context,
            &TimeCalendarCache::new(),
        );
        assert_eq!(
            text_range_json(result.result.unwrap()),
            case["result"],
            "{case}"
        );
        assert_eq!(json!(result.future_warnings), case["warnings"], "{case}");
    }
    let local = parse_timestamp_text_with_warnings(
        "9:30 XYZ",
        TimestampTextZoneContext {
            timestamp: context.timestamp,
            local_timezone_names: &["XYZ"],
        },
    );
    assert!(local.result.is_none());
    assert!(local.future_warnings.is_empty());
    for (start, end, expected_warnings) in
        [("not parsed", "9:30 XYZ", 0), ("9:30 XYZ", "not parsed", 1)]
    {
        let outcome = day_minute_text_range_with_warnings(
            start,
            end,
            "1min",
            "cn",
            context,
            &TimeCalendarCache::new(),
        );
        assert!(outcome.result.is_none());
        assert_eq!(outcome.future_warnings.len(), expected_warnings);
    }
}

#[test]
fn named_timezones_preserve_warnings_results_and_range_errors() {
    use domain_core::timestamp_text::{TimestampTextContext, TimestampTextZoneContext};
    let snapshot = source_snapshot_mode(Some("--named-zone"));
    assert_eq!(snapshot["pandas_version"], "2.3.3");
    let cases = snapshot["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 3603);
    let names: Vec<&str> = snapshot["local_timezone_names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|name| name.as_str().unwrap())
        .collect();
    let year = i32::try_from(snapshot["parser_year"].as_i64().unwrap()).unwrap();
    for case in cases {
        let context = TimestampTextZoneContext {
            timestamp: TimestampTextContext {
                reference_date: NaiveDate::parse_from_str(
                    case["reference_date"].as_str().unwrap(),
                    "%Y-%m-%d",
                )
                .unwrap(),
                parser_initialized_date: NaiveDate::from_ymd_opt(year, 1, 1).unwrap(),
            },
            local_timezone_names: &names,
        };
        assert_warning_case(case, context);
    }
}

fn assert_warning_case(
    case: &Value,
    context: domain_core::timestamp_text::TimestampTextZoneContext<'_>,
) {
    let text = case["text"].as_str().unwrap();
    let outcome = domain_core::timestamp_text::parse_timestamp_text_with_warnings(text, context);
    if let Some(Ok(domain_core::timestamp_text::TimestampTextValue::Timestamp(value))) =
        &outcome.result
    {
        let end = timestamp(
            utc_nanos("2021-01-01T14:59:00Z"),
            TimeUnit::Nanosecond,
            None,
        );
        let result = domain_core::time_compat::day_minute_timestamp_range_with_cache(
            &value.to_temporal(),
            &end,
            "1min",
            "cn",
            &TimeCalendarCache::new(),
        );
        assert_eq!(
            text_range_json(result),
            case["range"],
            "typed consumer: {case}"
        );
    }
    assert_constructor_result(
        case,
        outcome
            .result
            .unwrap_or_else(|| panic!("unrecognized: {case}")),
    );
    let expected: Vec<&Value> = case["constructor_warnings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|warning| {
            assert_eq!(warning["category"], "FutureWarning");
            &warning["message"]
        })
        .collect();
    assert_eq!(json!(outcome.future_warnings), json!(expected), "{case}");
    let range = domain_core::time_compat::day_minute_text_range_with_warnings(
        text,
        "2021-01-01 14:59",
        "1min",
        "cn",
        context,
        &TimeCalendarCache::new(),
    );
    assert_eq!(
        text_range_json(range.result.unwrap()),
        case["range"],
        "{case}"
    );
    let expected: Vec<&Value> = case["range_warnings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|warning| &warning["message"])
        .collect();
    assert_eq!(json!(range.future_warnings), json!(expected), "{case}");
}

#[test]
fn utc_aliases_preserve_general_precision_and_reversed_offset_signs() {
    use domain_core::timestamp_text::TimestampTextContext;
    let snapshot = source_snapshot_mode(Some("--utc-alias"));
    assert_eq!(snapshot["pandas_version"], "2.3.3");
    let cases = snapshot["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 1440);
    let parser_year = i32::try_from(snapshot["parser_year"].as_i64().unwrap()).unwrap();
    let names: Vec<&str> = snapshot["local_timezone_names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|name| name.as_str().unwrap())
        .collect();
    let mut warning_aware = 0;
    for case in cases {
        let context = TimestampTextContext {
            reference_date: NaiveDate::parse_from_str(
                case["reference_date"].as_str().unwrap(),
                "%Y-%m-%d",
            )
            .unwrap(),
            parser_initialized_date: NaiveDate::from_ymd_opt(parser_year, 1, 1).unwrap(),
        };
        if case["whole_zone_name_required"] == true {
            assert_eq!(
                domain_core::timestamp_text::parse_timestamp_text_compatible(
                    case["text"].as_str().unwrap(),
                    context
                ),
                None,
                "{case}"
            );
            assert_warning_case(
                case,
                domain_core::timestamp_text::TimestampTextZoneContext {
                    timestamp: context,
                    local_timezone_names: &names,
                },
            );
            warning_aware += 1;
        } else {
            assert_eq!(case["constructor_warnings"], json!([]), "{case}");
            assert_eq!(case["range_warnings"], json!([]), "{case}");
            assert_native_date_token(case, context);
        }
    }
    assert_eq!(warning_aware, 80);
}

#[test]
fn zoned_date_tokens_preserve_calendar_replacement_and_error_order() {
    use domain_core::timestamp_text::TimestampTextContext;
    let snapshot = source_snapshot_mode(Some("--zoned-date-token"));
    assert_eq!(snapshot["pandas_version"], "2.3.3");
    let cases = snapshot["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 5043);
    let parser_year = i32::try_from(snapshot["parser_year"].as_i64().unwrap()).unwrap();
    for case in cases {
        let context = TimestampTextContext {
            reference_date: NaiveDate::parse_from_str(
                case["reference_date"].as_str().unwrap(),
                "%Y-%m-%d",
            )
            .unwrap(),
            parser_initialized_date: NaiveDate::from_ymd_opt(parser_year, 1, 1).unwrap(),
        };
        assert_native_date_token(case, context);
    }
}

#[test]
fn numeric_timezones_preserve_date_formats_and_clock_context() {
    use domain_core::timestamp_text::TimestampTextContext;
    let snapshot = source_snapshot_mode(Some("--zone-formats"));
    assert_eq!(snapshot["pandas_version"], "2.3.3");
    let cases = snapshot["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 6806);
    let parser_year = i32::try_from(snapshot["parser_year"].as_i64().unwrap()).unwrap();
    for case in cases {
        let reference_date =
            NaiveDate::parse_from_str(case["reference_date"].as_str().unwrap(), "%Y-%m-%d")
                .unwrap();
        assert_native_date_token(
            case,
            TimestampTextContext {
                reference_date,
                parser_initialized_date: NaiveDate::from_ymd_opt(parser_year, 1, 1).unwrap(),
            },
        );
    }
    assert_eq!(
        domain_core::timestamp_text::parse_timestamp_text_compatible(
            "9:30+8",
            NaiveDate::from_ymd_opt(0, 1, 1).unwrap(),
        ),
        None
    );
}

#[test]
fn general_timezone_errors_preserve_calendar_clock_and_offset_order() {
    let snapshot = source_snapshot_mode(Some("--general-zone"));
    assert_eq!(snapshot["pandas_version"], "2.3.3");
    let cases = snapshot["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 398);
    for case in cases {
        assert_general_timezone_case(case);
    }
}

#[test]
fn malformed_native_dateutil_timezone_labels_are_not_silently_normalized() {
    for label in [
        "tzoffset(None, 3600",
        "tzoffset(None, x)",
        "tzoffset(None, 1)",
        "tzoffset(None, 86400)",
        "tzoffset('XYZ', 3600",
        "tzoffset(XYZ, 3600)",
        "tzoffset('xyz', 3600)",
        "tzoffset('ABCDEF', 3600)",
        "tzoffset('', 3600)",
        "tzoffset('XYZ', x)",
        "tzoffset('XYZ', 1)",
        "tzoffset('XYZ', 86400)",
    ] {
        let value = timestamp(0, TimeUnit::Second, Some(label));
        let error =
            domain_core::time_compat::timestamp_to_day_index_compatible(&value, "cn").unwrap_err();
        assert!(
            matches!(error, TimeCompatError::InvalidTimezone { timezone, .. } if timezone == label)
        );
    }
    assert_eq!(
        domain_core::timestamp_text::parse_timestamp_text_compatible(
            "2021-01-01 9+8 ",
            NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
        ),
        None,
    );
}

fn assert_general_timezone_case(case: &Value) {
    use domain_core::timestamp_text::{TimestampTextValue, parse_timestamp_text_compatible};
    let reference = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
    assert_native_date_token(case, reference);
    if let Some(Ok(TimestampTextValue::Timestamp(value))) =
        parse_timestamp_text_compatible(case["text"].as_str().unwrap(), reference)
    {
        let end = timestamp(
            utc_nanos("2021-01-01T14:59:00Z"),
            TimeUnit::Nanosecond,
            None,
        );
        let result = domain_core::time_compat::day_minute_timestamp_range_with_cache(
            &value.to_temporal(),
            &end,
            "1min",
            "cn",
            &TimeCalendarCache::new(),
        );
        assert_eq!(text_range_json(result), case["range"], "{case}");
    }
}

fn assert_iso_constructor_snapshot(mode: &str, count: usize) {
    use domain_core::{
        time_compat::day_minute_timestamp_range_with_cache,
        timestamp_text::parse_iso_timestamp_compatible,
    };

    let snapshot = source_snapshot_mode(Some(mode));
    assert_eq!(snapshot["pandas_version"], "2.3.3");
    let cases = snapshot["cases"].as_array().unwrap();
    assert_eq!(cases.len(), count);
    let end = parse_iso_timestamp_compatible("2021-01-01 14:59")
        .unwrap()
        .unwrap();
    let cache = TimeCalendarCache::new();
    let mut failures = Vec::new();
    let mut general_timezone_cases = 0;
    for case in cases {
        let text = case["text"].as_str().unwrap();
        if case["general_timezone_required"] == true {
            // These captured source cases require dateutil timezone metadata
            // and precision. The narrow ISO entry defers; the complete staged
            // entry must now match the source, including downstream consumers.
            assert_eq!(parse_iso_timestamp_compatible(text), None, "{case}");
            assert_general_timezone_case(case);
            general_timezone_cases += 1;
            continue;
        }
        assert_native_text_range(case, NaiveDate::from_ymd_opt(2021, 1, 1).unwrap());
        let actual = parse_iso_timestamp_compatible(text).expect("canonical ISO is recognized");
        let (description, range) = match actual {
            Ok(value) => {
                let description = match &value {
                    TemporalFrameValue::Timestamp {
                        ticks,
                        unit,
                        timezone,
                    } => {
                        let unit = match unit {
                            TimeUnit::Second => "s",
                            TimeUnit::Millisecond => "ms",
                            TimeUnit::Microsecond => "us",
                            TimeUnit::Nanosecond => "ns",
                        };
                        json!({"ok": {"kind": "Timestamp", "ticks": ticks,
                            "unit": unit, "timezone": timezone.as_deref()}})
                    }
                    TemporalFrameValue::Builtin(BuiltinFrameValue::NotATime) => {
                        json!({"ok": {"kind": "NaT"}})
                    }
                    other => panic!("unexpected constructor value {other:?}"),
                };
                let range =
                    match day_minute_timestamp_range_with_cache(&value, &end, "1min", "cn", &cache)
                    {
                        Ok((left, right)) => json!({"ok": [left, right]}),
                        Err(error) => json!({"error": "ValueError", "message": error.to_string()}),
                    };
                (description, range)
            }
            Err(error) => {
                let result = json!({"error": "OutOfBoundsDatetime", "message": error.to_string()});
                (result.clone(), result)
            }
        };
        let mut expected = case["result"].clone();
        if let Some(value) = expected.get_mut("ok").and_then(Value::as_object_mut) {
            value.remove("text");
        }
        if description != expected || range != case["range"] {
            failures.push(format!(
                "{text}: native={description}, range={range}, source={case}"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert_eq!(
        general_timezone_cases,
        if mode == "--iso-offset" { 1280 } else { 0 }
    );
    assert_iso_defers_general_inputs();
}

fn assert_iso_defers_general_inputs() {
    use domain_core::timestamp_text::parse_iso_timestamp_compatible;

    // No-match must not claim that Pandas rejects a general/dateutil spelling.
    for text in [
        "",
        "NaT",
        "9:30",
        "Jan 1, 2021 9:30",
        "20210101",
        "2021/01-02",
        "2021-01-01 9",
        "2021-01-01 093",
        "2021-01-01 09300",
        "2021-01-01 09:3000",
        "2021-01-01 0930:00",
        "2021-02-29",
        "2021-01-01T09:30:60",
        "2021-01-01T24:00",
        "2021-01-01T09:60",
        "2021-01-01T09:30+24:00",
        "2021-01-01T09:30+00:60",
        "2021-01-01T09:30:00.1234567890123456789",
        "2021-01-01 09:30 XYZ",
    ] {
        assert_eq!(parse_iso_timestamp_compatible(text), None, "{text}");
    }
}

#[test]
fn range_text_constructor_contract_and_native_post_conversion_parity() {
    use domain_core::time_compat::day_minute_timestamp_range_with_cache;

    let snapshot = source_snapshot_mode(Some("--range-text"));
    assert_eq!(snapshot["pandas_version"], "2.3.3");
    let inputs = snapshot["inputs"].as_array().unwrap();
    let cases = snapshot["cases"].as_array().unwrap();
    assert_eq!(inputs.len(), 22);
    assert_eq!(cases.len(), 7744);
    // Native text parsing is not implemented by this test. Conversion stays in the
    // source oracle; only the downstream range calculation is compared to Rust.
    let mut converted_cases = 0;
    for case in cases {
        let mut expected_warnings = Vec::new();
        let mut values = Vec::new();
        let mut constructor_error = None;
        for key in ["start", "end"] {
            let input = &inputs[usize::try_from(case[key].as_u64().unwrap()).unwrap()];
            expected_warnings.extend(input["warnings"].as_array().unwrap().iter().cloned());
            let result = &input["result"];
            if result.get("error").is_some() {
                constructor_error = Some(result.clone());
                break;
            }
            let value = &result["ok"];
            if value["kind"] == "NaT" {
                constructor_error = Some(json!({
                    "error": "ValueError", "message": "NaTType does not support time"
                }));
                break;
            }
            let unit = match value["unit"].as_str().unwrap() {
                "s" => TimeUnit::Second,
                "ms" => TimeUnit::Millisecond,
                "us" => TimeUnit::Microsecond,
                "ns" => TimeUnit::Nanosecond,
                _ => unreachable!(),
            };
            values.push(timestamp(
                value["ticks"].as_i64().unwrap(),
                unit,
                value["timezone"].as_str(),
            ));
        }
        assert_eq!(case["warnings"], json!(expected_warnings), "{case}");
        if let Some(error) = constructor_error {
            assert_eq!(case["result"], error, "{case}");
            assert_eq!(case["cache_misses"], 0, "{case}");
            continue;
        }
        let cache = TimeCalendarCache::new();
        let actual = match day_minute_timestamp_range_with_cache(
            &values[0],
            &values[1],
            case["frequency"].as_str().unwrap(),
            case["region"].as_str().unwrap(),
            &cache,
        ) {
            Ok((left, right)) => json!({"ok": [left, right]}),
            Err(error) => json!({"error": "ValueError", "message": error.to_string()}),
        };
        assert_eq!(actual, case["result"], "{case}");
        assert_eq!(
            json!(cache.cache_info().misses),
            case["cache_misses"],
            "{case}"
        );
        converted_cases += 1;
    }
    assert_eq!(converted_cases, 1600);
    // These distinguish Pandas' accepted grammar from RFC3339 or a clock parser.
    assert_eq!(inputs[8]["result"]["ok"]["text"], "0000-01-01 09:30:00");
    assert_eq!(inputs[9]["warnings"].as_array().unwrap().len(), 1);
    assert_eq!(inputs[19]["result"]["error"], "DateParseError");
    assert_eq!(inputs[20]["result"]["error"], "DateParseError");
}

#[test]
fn intraday_and_range_propagate_invalid_native_timezone_before_cache_access() {
    use domain_core::time_compat::{
        day_minute_timestamp_range_with_cache, timestamp_to_day_index_compatible,
    };
    let invalid = timestamp(0, TimeUnit::Second, Some("Not/A_Real_Zone"));
    let valid = timestamp(0, TimeUnit::Second, None);
    let cache = TimeCalendarCache::new();
    let error = timestamp_to_day_index_compatible(&invalid, "cn").unwrap_err();
    assert!(
        matches!(&error, TimeCompatError::InvalidTimezone { timezone, .. } if timezone == "Not/A_Real_Zone")
    );
    assert_eq!(
        day_minute_timestamp_range_with_cache(&invalid, &valid, "bad", "CN", &cache),
        Err(error.clone())
    );
    assert_eq!(
        day_minute_timestamp_range_with_cache(&valid, &invalid, "bad", "CN", &cache),
        Err(error)
    );
    assert_eq!(cache.cache_info().misses, 0);
}

#[test]
fn timestamp_range_preserves_clock_precision_and_input_error_order() {
    use domain_core::time_compat::day_minute_timestamp_range_with_cache;
    fn decode(input: &Value) -> TemporalFrameValue {
        if input["kind"] == "NaT" {
            return TemporalFrameValue::Builtin(BuiltinFrameValue::NotATime);
        }
        let unit = match input["unit"].as_str().unwrap() {
            "s" => TimeUnit::Second,
            "ms" => TimeUnit::Millisecond,
            "us" => TimeUnit::Microsecond,
            "ns" => TimeUnit::Nanosecond,
            _ => unreachable!(),
        };
        timestamp(
            input["ticks"].as_i64().unwrap(),
            unit,
            input["timezone"].as_str(),
        )
    }
    let snapshot = source_snapshot();
    let cases = snapshot["range_timestamp_cases"].as_array().unwrap();
    assert_eq!(cases.len(), 3600);
    let cache = TimeCalendarCache::new();
    let mut mismatches = Vec::new();
    for case in cases {
        let actual = match day_minute_timestamp_range_with_cache(
            &decode(&case["start"]),
            &decode(&case["end"]),
            case["frequency"].as_str().unwrap(),
            case["region"].as_str().unwrap(),
            &cache,
        ) {
            Ok(pair) => json!({"ok": [pair.0, pair.1]}),
            Err(error) => {
                assert!(
                    matches!(
                        error,
                        TimeCompatError::NaTDoesNotSupportTime
                            | TimeCompatError::RangeFrequencyFormat
                            | TimeCompatError::CalendarCache(
                                TimeCalendarCacheError::UnsupportedRegion { .. }
                            )
                            | TimeCompatError::Alignment(MinuteAlignmentError::ZeroSamplingStep)
                    ),
                    "unexpected error for {case}: {error}"
                );
                json!({"error": "ValueError", "message": error.to_string()})
            }
        };
        if actual != case["result"] {
            mismatches.push(format!("{case}: actual={actual}"));
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} mismatches:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
    let empty_cache = TimeCalendarCache::new();
    let nat = TemporalFrameValue::Builtin(BuiltinFrameValue::NotATime);
    let valid = timestamp(0, TimeUnit::Second, None);
    assert_eq!(
        day_minute_timestamp_range_with_cache(&valid, &nat, "bad", "CN", &empty_cache),
        Err(TimeCompatError::NaTDoesNotSupportTime)
    );
    assert_eq!(
        day_minute_timestamp_range_with_cache(&valid, &valid, "bad", "CN", &empty_cache),
        Err(TimeCompatError::RangeFrequencyFormat)
    );
    let invalid = TemporalFrameValue::Duration {
        ticks: 0,
        unit: TimeUnit::Second,
    };
    assert!(matches!(
        day_minute_timestamp_range_with_cache(&invalid, &nat, "bad", "cn", &empty_cache),
        Err(TimeCompatError::UnsupportedTemporalValue {
            operation: "get_day_min_idx_range"
        })
    ));
    assert_eq!(empty_cache.cache_info().misses, 0);
}

#[test]
fn intraday_timestamps_preserve_nanoseconds_missingness_and_source_errors() {
    use domain_core::time_compat::timestamp_to_day_index_compatible;
    let snapshot = source_snapshot();
    let cases = snapshot["intraday_timestamp_cases"].as_array().unwrap();
    assert_eq!(cases.len(), 775);
    let mut mismatches = Vec::new();
    for case in cases {
        let input = &case["input"];
        let value = if input["kind"] == "NaT" {
            TemporalFrameValue::Builtin(BuiltinFrameValue::NotATime)
        } else {
            let unit = match input["unit"].as_str().unwrap() {
                "s" => TimeUnit::Second,
                "ms" => TimeUnit::Millisecond,
                "us" => TimeUnit::Microsecond,
                "ns" => TimeUnit::Nanosecond,
                _ => unreachable!(),
            };
            timestamp(
                input["ticks"].as_i64().unwrap(),
                unit,
                input["timezone"].as_str(),
            )
        };
        let actual =
            match timestamp_to_day_index_compatible(&value, case["region"].as_str().unwrap()) {
                Ok(index) => json!({"ok": index}),
                Err(error) => {
                    let category = match error {
                        TimeCompatError::AwareIntradayDatetime
                        | TimeCompatError::AwareIntradayTimestamp => "TypeError",
                        TimeCompatError::IntradayTimestampYearOverflow => "OverflowError",
                        TimeCompatError::IntradayTimestampYear { .. }
                        | TimeCompatError::UnsupportedIntradayRegion { .. }
                        | TimeCompatError::OutsideIntradayDatetime { .. } => "ValueError",
                        _ => panic!("unexpected error {error}"),
                    };
                    json!({"error": category, "message": error.to_string()})
                }
            };
        if actual != case["result"] {
            mismatches.push(format!("{case}: actual={actual}"));
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} mismatches:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
    assert!(matches!(
        timestamp_to_day_index_compatible(
            &TemporalFrameValue::Duration {
                ticks: 1,
                unit: TimeUnit::Second
            },
            "cn"
        ),
        Err(TimeCompatError::UnsupportedTemporalValue {
            operation: "time_to_day_index"
        })
    ));
}

#[test]
fn intraday_string_errors_and_region_precedence_match_source() {
    use domain_core::time_compat::time_to_day_index_compatible;
    let snapshot = source_snapshot();
    let cases = snapshot["intraday_string_cases"].as_array().unwrap();
    assert_eq!(cases.len(), 3360);
    for case in cases {
        let actual = time_to_day_index_compatible(
            case["input"].as_str().unwrap(),
            case["region"].as_str().unwrap(),
        );
        let expected = &case["result"];
        if let Some(index) = expected["ok"].as_i64() {
            assert_eq!(actual.unwrap(), index, "{case}");
        } else {
            let error = actual.unwrap_err();
            assert!(
                matches!(
                    error,
                    TimeCompatError::MarketClockFormat { .. }
                        | TimeCompatError::MarketClockRemainder { .. }
                        | TimeCompatError::UnsupportedIntradayRegion { .. }
                        | TimeCompatError::OutsideIntradayDatetime { .. }
                ),
                "{case}: {error}"
            );
            assert_eq!(expected["error"], "ValueError", "{case}");
            assert_eq!(error.to_string(), expected["message"], "{case}");
        }
    }
}

#[test]
fn intraday_datetime_preserves_date_awareness_and_region_precedence() {
    use domain_core::time_compat::datetime_to_day_index_compatible;
    let snapshot = source_snapshot();
    let cases = snapshot["intraday_datetime_cases"].as_array().unwrap();
    assert_eq!(cases.len(), 300);
    for case in cases {
        let local = chrono::NaiveDateTime::parse_from_str(
            case["local"].as_str().unwrap(),
            "%Y-%m-%dT%H:%M:%S%.f",
        )
        .unwrap();
        let offset = case["aware"]
            .as_bool()
            .unwrap()
            .then(|| chrono::FixedOffset::east_opt(0).unwrap());
        let actual =
            datetime_to_day_index_compatible(local, offset, case["region"].as_str().unwrap());
        let expected = &case["result"];
        if let Some(index) = expected["ok"].as_i64() {
            assert_eq!(actual.unwrap(), index, "{case}");
        } else {
            let error = actual.unwrap_err();
            let category = match error {
                TimeCompatError::AwareIntradayDatetime => "TypeError",
                TimeCompatError::UnsupportedIntradayRegion { .. }
                | TimeCompatError::OutsideIntradayDatetime { .. } => "ValueError",
                _ => panic!("unexpected error: {error}"),
            };
            assert_eq!(category, expected["error"], "{case}");
            assert_eq!(error.to_string(), expected["message"], "{case}");
        }
    }
    let invalid_year = NaiveDate::from_ymd_opt(0, 1, 1)
        .unwrap()
        .and_hms_opt(10, 0, 0)
        .unwrap();
    assert!(matches!(
        datetime_to_day_index_compatible(invalid_year, None, "cn"),
        Err(TimeCompatError::DateYearOutOfRange { year: 0 })
    ));
    let date = NaiveDate::from_ymd_opt(1900, 1, 1).unwrap();
    assert_eq!(
        datetime_to_day_index_compatible(date.and_hms_nano_opt(10, 0, 0, 1).unwrap(), None, "cn"),
        Err(TimeCompatError::SubmicrosecondTime)
    );
    assert_eq!(
        datetime_to_day_index_compatible(
            date.and_hms_nano_opt(10, 0, 59, 1_000_000_000).unwrap(),
            None,
            "cn"
        ),
        Err(TimeCompatError::DatetimeLeapSecond)
    );
}

#[test]
fn single_value_temporal_boundary_matches_actual_source() {
    fn input(value: &Value) -> TemporalFrameValue {
        match value["kind"].as_str().unwrap() {
            "NaT" => TemporalFrameValue::Builtin(BuiltinFrameValue::NotATime),
            "Timedelta" => TemporalFrameValue::Duration {
                ticks: value["nanoseconds"].as_i64().unwrap(),
                unit: TimeUnit::Nanosecond,
            },
            "Timestamp" => timestamp(
                value["ticks"].as_i64().unwrap(),
                match value["unit"].as_str().unwrap() {
                    "s" => TimeUnit::Second,
                    "ms" => TimeUnit::Millisecond,
                    "us" => TimeUnit::Microsecond,
                    "ns" => TimeUnit::Nanosecond,
                    unit => panic!("unexpected unit {unit}"),
                },
                value["timezone"].as_str(),
            ),
            kind => panic!("unexpected kind {kind}"),
        }
    }
    let snapshot = source_snapshot();
    let cases = snapshot["single_value_cases"].as_array().unwrap();
    assert_eq!(cases.len(), 11_940);
    let mut failures = Vec::new();
    for case in cases {
        let actual = match is_single_value_compatible(
            &input(&case["start"]),
            &input(&case["end"]),
            &input(&case["frequency"]),
            case["region"].as_str().unwrap(),
        ) {
            Ok(result) => json!({"ok": result}),
            Err(error) => {
                let kind = match &error {
                    TimeCompatError::UnsupportedSingleValueRegion { .. } => "NotImplementedError",
                    TimeCompatError::MixedTimestampAwareness => "TypeError",
                    TimeCompatError::TimestampSubtractionSentinel => "AssertionError",
                    TimeCompatError::TimestampPromotionOverflow { .. }
                    | TimeCompatError::TimestampSubtractionOverflow => "OutOfBoundsDatetime",
                    error => panic!("unexpected error: {error:?}"),
                };
                json!({"error": kind, "message": error.to_string()})
            }
        };
        if actual != case["result"] {
            failures.push(format!("{case}: actual={actual}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{} mismatches: {:?}",
        failures.len(),
        &failures[..failures.len().min(8)]
    );
}

#[test]
fn single_value_rejects_invalid_typed_inputs_at_the_operation_that_uses_them() {
    let nat = TemporalFrameValue::Builtin(BuiltinFrameValue::NotATime);
    let duration = TemporalFrameValue::Duration {
        ticks: 1,
        unit: TimeUnit::Second,
    };
    let ordinary = timestamp(0, TimeUnit::Second, None);
    for (start, end) in [(&duration, &ordinary), (&ordinary, &duration)] {
        assert_eq!(
            is_single_value_compatible(start, end, &duration, "cn"),
            Err(TimeCompatError::UnsupportedTemporalValue {
                operation: "is_single_value"
            })
        );
    }
    assert_eq!(
        is_single_value_compatible(&ordinary, &ordinary, &ordinary, "cn"),
        Err(TimeCompatError::UnsupportedSingleValueFrequency)
    );
    assert_eq!(
        is_single_value_compatible(&nat, &ordinary, &ordinary, "cn"),
        Ok(false)
    );
    let invalid_zone = timestamp(0, TimeUnit::Second, Some("Invalid/Zone"));
    assert_eq!(
        is_single_value_compatible(&invalid_zone, &invalid_zone, &duration, "cn"),
        Ok(true)
    );
    for (start, end) in [
        (&invalid_zone, &nat),
        (
            &timestamp(
                32_503_680_000_000_000,
                TimeUnit::Microsecond,
                Some("Invalid/Zone"),
            ),
            &timestamp(0, TimeUnit::Nanosecond, Some("UTC")),
        ),
    ] {
        assert!(matches!(is_single_value_compatible(start, end, &nat, "cn"),
            Err(TimeCompatError::InvalidTimezone { timezone, .. }) if timezone == "Invalid/Zone"));
    }
    let beyond_chrono = timestamp(i64::MAX, TimeUnit::Second, None);
    assert_eq!(
        is_single_value_compatible(&beyond_chrono, &nat, &nat, "cn"),
        Ok(false)
    );
    let beyond_chrono = timestamp(i64::MAX - 172_800, TimeUnit::Second, Some("Asia/Shanghai"));
    assert_eq!(
        is_single_value_compatible(&beyond_chrono, &nat, &nat, "cn"),
        Ok(false)
    );
    assert_eq!(
        is_single_value_compatible(
            &beyond_chrono,
            &timestamp(0, TimeUnit::Nanosecond, Some("UTC")),
            &nat,
            "cn"
        ),
        Err(TimeCompatError::TimestampPromotionOverflow {
            timestamp: "292277026596-12-02 23:30:07+08:00".into(),
            unit: "ns",
        })
    );
}

#[test]
fn alignment_timezone_history_and_wide_date_errors_match_actual_source() {
    let snapshot = source_snapshot();
    let cases = snapshot["alignment_timezones"].as_array().unwrap();
    assert_eq!(cases.len(), 2_904);
    let cache = TimeCalendarCache::new();
    let mut failures = Vec::new();
    for case in cases {
        let input = &case["input"];
        let unit = match input["unit"].as_str().unwrap() {
            "s" => TimeUnit::Second,
            "ms" => TimeUnit::Millisecond,
            "us" => TimeUnit::Microsecond,
            "ns" => TimeUnit::Nanosecond,
            value => panic!("unexpected unit {value}"),
        };
        let input = timestamp(
            input["ticks"].as_i64().unwrap(),
            unit,
            input["timezone"].as_str(),
        );
        let region = Region::from_str(case["region"].as_str().unwrap()).unwrap();
        let result = align_sampled_minute_with_cache(
            &input,
            &BigInt::from(1),
            &BigInt::from(0),
            region,
            &cache,
        );
        let matches = match &result {
            Ok(actual) => {
                let expected = &case["result"]["ok"];
                expected["kind"] == "Timestamp"
                    && expected["unit"] == "us"
                    && expected["timezone"].is_null()
                    && *actual
                        == timestamp(
                            expected["ticks"].as_i64().unwrap(),
                            TimeUnit::Microsecond,
                            None,
                        )
            }
            Err(error) => {
                *error == TimeCompatError::TimestampDateNotSupported
                    && case["result"]["error"] == "NotImplementedError"
                    && case["result"]["message"] == error.to_string()
            }
        };
        if !matches {
            failures.push(format!("{case}: actual={result:?}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{} mismatches: {:?}",
        failures.len(),
        &failures[..failures.len().min(5)]
    );
}

#[test]
fn market_clock_parser_matches_source_whitespace_and_unicode_grammar() {
    let snapshot = source_snapshot();
    let cases = snapshot["market_clock_cases"].as_array().unwrap();
    assert_eq!(cases.len(), 660);
    let mut failures = Vec::new();
    for case in cases {
        let input = case["input"].as_str().unwrap();
        let parsed = domain_core::parse_market_time(input);
        let matches = if case["parse"]["error"].is_string() {
            parsed.is_err()
        } else {
            let expected = &case["parse"]["ok"];
            assert_eq!(expected["unit"], "us");
            let clock = DateTime::from_timestamp_micros(expected["ticks"].as_i64().unwrap())
                .unwrap()
                .time();
            parsed == Ok(clock)
        };
        let indexed = domain_core::time_to_day_index_str(input, Region::Cn);
        let index_matches = if let Some(expected) = case["index"]["ok"].as_i64() {
            indexed == Ok(expected)
        } else {
            indexed.is_err()
        };
        if !matches || !index_matches {
            failures.push(format!("{case}: parse={parsed:?}, index={indexed:?}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{} mismatches: {:?}",
        failures.len(),
        &failures[..failures.len().min(5)]
    );
}

#[test]
fn generated_duration_transitions_match_actual_source() {
    let snapshot = source_snapshot_mode(Some("--generated-duration"));
    let cases = snapshot["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 22_143);
    let mut failures = Vec::new();
    for case in cases {
        let count = BigInt::from_str(case["count"].as_str().unwrap()).unwrap();
        let outcome = time_delta_compatible(&count, case["unit"].as_str().unwrap());
        let actual = match outcome.result {
            Ok(TemporalFrameValue::Duration { ticks, unit }) => {
                assert_eq!(unit, TimeUnit::Nanosecond);
                json!({"ok": {"kind": "Timedelta", "nanoseconds": ticks}})
            }
            Ok(TemporalFrameValue::Builtin(BuiltinFrameValue::NotATime)) => {
                json!({"ok": {"kind": "NaT", "text": "NaT"}})
            }
            Ok(value) => panic!("unexpected duration value {value:?}"),
            Err(error) => json!({"error": "ValueError", "message": error.to_string()}),
        };
        if actual != case["result"] || json!(outcome.future_warnings) != case["warnings"] {
            failures.push(format!(
                "{case}: actual={actual}, warnings={:?}",
                outcome.future_warnings
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} mismatches; first examples: {:?}",
        failures.len(),
        &failures[..failures.len().min(12)]
    );
}

#[test]
fn duration_compatibility_entry_preserves_value_kind_precision_and_warnings() {
    let snapshot = source_snapshot();
    for key in ["compound_duration_cases", "decimal_duration_cases"] {
        for case in snapshot[key].as_array().unwrap() {
            let count = BigInt::from_str(case["count"].as_str().unwrap()).unwrap();
            let outcome = time_delta_compatible(&count, case["unit"].as_str().unwrap());
            assert_eq!(json!(outcome.future_warnings), case["warnings"], "{case}");
            match outcome.result {
                Ok(value) => {
                    let expected = if case["result"]["ok"]["kind"] == "NaT" {
                        TemporalFrameValue::Builtin(BuiltinFrameValue::NotATime)
                    } else {
                        TemporalFrameValue::Duration {
                            ticks: case["result"]["ok"]["nanoseconds"].as_i64().unwrap(),
                            unit: TimeUnit::Nanosecond,
                        }
                    };
                    assert_eq!(value, expected, "{case}");
                }
                Err(error) => {
                    assert_eq!(
                        case["result"]["error"],
                        error.python_exception_name(),
                        "{case}"
                    );
                    assert_eq!(case["result"]["message"], error.to_string(), "{case}");
                }
            }
        }
    }
}

#[test]
fn frequency_terminal_newline_matches_python_regex_anchor() {
    let snapshot = source_snapshot();
    let cases = snapshot["frequency_newlines"].as_array().unwrap();
    assert_eq!(cases.len(), 8);
    for case in cases {
        let input = case["input"].as_str().unwrap();
        match Frequency::from_str(input) {
            Ok(value) => assert_eq!(
                value.to_string(),
                case["result"]["ok"]["text"].as_str().unwrap()
            ),
            Err(FrequencyError::UnsupportedFormat { input: rejected }) => {
                assert_eq!(case["result"]["error"], "ValueError");
                assert_eq!(rejected, input);
            }
            Err(error) => panic!("unexpected frequency error {error:?}"),
        }
    }
}

#[test]
fn duration_units_rounding_and_range_match_actual_source() {
    let snapshot = source_snapshot();
    let cases = snapshot["duration_cases"].as_array().unwrap();
    assert_eq!(cases.len(), 504);
    for case in cases {
        let count = BigInt::from_str(case["count"].as_str().unwrap()).unwrap();
        let unit = case["unit"].as_str().unwrap();
        let outcome = Frequency::time_delta_with_warnings(count.clone(), unit);
        let warnings: Vec<Value> = outcome
            .future_warning
            .into_iter()
            .map(|message| json!({"category": "FutureWarning", "message": message}))
            .collect();
        assert_eq!(json!(warnings), case["warnings"], "{case}");
        assert_eq!(outcome.result, Frequency::time_delta(count.clone(), unit));
        if let Err(error) = &outcome.result {
            assert_eq!(
                case["result"]["error"],
                error.python_exception_name(),
                "{case}"
            );
            assert_eq!(case["result"]["message"], error.to_string(), "{case}");
        }
        match outcome.result {
            Ok(value) => assert_eq!(
                json!(value.num_nanoseconds().unwrap()),
                case["result"]["ok"]["nanoseconds"],
                "{case}"
            ),
            Err(
                error @ (FrequencyError::UnsupportedDurationUnit { .. }
                | FrequencyError::AmbiguousDurationUnit),
            ) => {
                assert_eq!(case["result"]["error"], "ValueError", "{case}");
                assert_eq!(case["result"]["message"], error.to_string(), "{case}");
            }
            Err(FrequencyError::DurationOutOfRange {
                count: rejected,
                unit: rejected_unit,
            }) => {
                assert!(
                    matches!(
                        case["result"]["error"].as_str(),
                        Some("OutOfBoundsDatetime" | "OverflowError")
                    ),
                    "{case}"
                );
                assert_eq!(rejected, count);
                assert_eq!(rejected_unit, unit);
            }
            Err(error) => panic!("unexpected error {error:?} for {case}"),
        }
    }
}

#[test]
fn compound_integer_durations_preserve_sign_concatenation_and_warning_order() {
    let snapshot = source_snapshot();
    let cases = snapshot["compound_duration_cases"].as_array().unwrap();
    assert_eq!(cases.len(), 400);
    for case in cases {
        let count = BigInt::from_str(case["count"].as_str().unwrap()).unwrap();
        let outcome = Frequency::compound_time_delta(&count, case["unit"].as_str().unwrap());
        assert_eq!(json!(outcome.future_warnings), case["warnings"], "{case}");
        match outcome.result {
            Ok(Some(value)) => assert_eq!(
                json!(value.num_nanoseconds().unwrap()),
                case["result"]["ok"]["nanoseconds"],
                "{case}"
            ),
            Ok(None) => assert_eq!(case["result"]["ok"]["kind"], "NaT", "{case}"),
            Err(error @ FrequencyError::ClockIntegerOverflow) => {
                assert_eq!(case["result"]["error"], "OverflowError", "{case}");
                assert_eq!(case["result"]["message"], error.to_string(), "{case}");
            }
            Err(FrequencyError::DurationOutOfRange { count, unit }) => {
                assert_eq!(case["result"]["error"], "OutOfBoundsDatetime", "{case}");
                assert_eq!(count, BigInt::from_str("9223372036854775808").unwrap());
                assert_eq!(unit, "ns");
            }
            Err(error) => {
                assert_eq!(case["result"]["error"], "ValueError", "{case}");
                assert_eq!(case["result"]["message"], error.to_string(), "{case}");
            }
        }
    }
}

#[test]
fn decimal_duration_rounding_and_bounds_match_actual_source() {
    let snapshot = source_snapshot();
    let cases = snapshot["decimal_duration_cases"].as_array().unwrap();
    assert_eq!(cases.len(), 704);
    for case in cases {
        let count = BigInt::from_str(case["count"].as_str().unwrap()).unwrap();
        let outcome = Frequency::compound_time_delta(&count, case["unit"].as_str().unwrap());
        assert_eq!(json!(outcome.future_warnings), case["warnings"], "{case}");
        match outcome.result {
            Ok(Some(value)) => assert_eq!(
                json!(value.num_nanoseconds().unwrap()),
                case["result"]["ok"]["nanoseconds"],
                "{case}"
            ),
            Ok(None) => assert_eq!(case["result"]["ok"]["kind"], "NaT", "{case}"),
            Err(FrequencyError::DecimalDurationOutOfRange { input }) => {
                assert!(
                    matches!(
                        case["result"]["error"].as_str(),
                        Some("OutOfBoundsDatetime" | "OverflowError")
                    ),
                    "{case}"
                );
                assert_eq!(
                    input,
                    format!("{}{}", count.magnitude(), case["unit"].as_str().unwrap())
                );
            }
            Err(error) => panic!("unexpected error {error:?} for {case}"),
        }
    }
}

#[test]
fn alignment_combined_failures_follow_pinned_source_evaluation_order() {
    let snapshot = source_snapshot();
    let cases = snapshot["alignment_precedence"].as_array().unwrap();
    assert_eq!(cases.len(), 54);
    for case in cases {
        let region = Region::from_str(case["region"].as_str().unwrap()).unwrap();
        let shift = BigInt::from_str(case["shift"].as_str().unwrap()).unwrap();
        let step = BigInt::from(case["step"].as_i64().unwrap());
        let input = if case["is_nat"].as_bool().unwrap() {
            TemporalFrameValue::Builtin(BuiltinFrameValue::NotATime)
        } else {
            timestamp(
                utc_nanos("2021-01-01T10:38:00Z"),
                TimeUnit::Nanosecond,
                None,
            )
        };
        let actual = align_sampled_minute_compatible(&input, &step, &shift, region);
        let expected = &case["result"];
        match actual {
            Ok(value) => {
                assert_eq!(expected["ok"]["kind"], "Timestamp", "{case}");
                assert_eq!(expected["ok"]["unit"], "us", "{case}");
                assert!(expected["ok"]["timezone"].is_null(), "{case}");
                assert_eq!(
                    value,
                    timestamp(
                        expected["ok"]["ticks"].as_i64().unwrap(),
                        TimeUnit::Microsecond,
                        None
                    ),
                    "{case}"
                );
            }
            Err(TimeCompatError::CalendarCache(TimeCalendarCacheError::Calendar(
                MarketCalendarError::ShiftOutOfRange { shift: rejected },
            ))) => {
                assert_eq!(expected["error"], "OutOfBoundsTimedelta", "{case}");
                assert_eq!(rejected, shift, "{case}");
            }
            Err(
                error @ (TimeCompatError::NaTDoesNotSupportTime
                | TimeCompatError::Alignment(MinuteAlignmentError::ZeroSamplingStep)),
            ) => {
                assert_eq!(expected["error"], "ValueError", "{case}");
                assert_eq!(expected["message"], error.to_string(), "{case}");
            }
            Err(error) => panic!("unexpected error {error:?} for {case}"),
        }
    }
}

#[test]
fn source_cache_mutation_reaches_alignment_consumers() {
    let snapshot = source_snapshot();
    let cache = &snapshot["downstream_cache"];
    assert_eq!(cache["before"]["ok"]["text"], "2021-01-01 10:38:00");
    assert_eq!(cache["mutated"]["ok"]["text"], "2021-01-01 09:00:00");
    assert_eq!(cache["empty"]["error"], "IndexError");
    assert_eq!(cache["empty"]["message"], "list index out of range");
    let native = TimeCalendarCache::new();
    let input = timestamp(
        utc_nanos("2021-01-01T10:38:00Z"),
        TimeUnit::Nanosecond,
        None,
    );
    let step = BigInt::from(1);
    let shift = BigInt::from(0);
    let align = |value: &TemporalFrameValue, sample: &BigInt| {
        align_sampled_minute_with_cache(value, sample, &shift, Region::Cn, &native)
    };
    assert_eq!(
        align(&input, &step),
        Ok(timestamp(
            cache["before"]["ok"]["ticks"].as_i64().unwrap(),
            TimeUnit::Microsecond,
            None
        ))
    );
    let calendar = native
        .get(&TimeCalendarCall::new(
            vec![0_i64.into(), "cn".into()],
            Vec::new(),
        ))
        .unwrap();
    *calendar.lock().unwrap() = vec![NaiveTime::from_hms_opt(9, 0, 0).unwrap()];
    assert_eq!(
        align(&input, &step),
        Ok(timestamp(
            cache["mutated"]["ok"]["ticks"].as_i64().unwrap(),
            TimeUnit::Microsecond,
            None
        ))
    );
    calendar.lock().unwrap().clear();
    assert_eq!(align(&input, &step), Err(TimeCompatError::EmptyCalendar));
    let nat = TemporalFrameValue::Builtin(BuiltinFrameValue::NotATime);
    assert_eq!(
        align(&nat, &step).unwrap_err().to_string(),
        cache["empty_nat"]["message"].as_str().unwrap()
    );
    assert_eq!(
        align(&nat, &BigInt::from(0)).unwrap_err().to_string(),
        cache["empty_zero"]["message"].as_str().unwrap()
    );
    native.cache_clear();
    assert_eq!(
        align(&input, &step),
        Ok(timestamp(
            cache["before"]["ok"]["ticks"].as_i64().unwrap(),
            TimeUnit::Microsecond,
            None
        ))
    );
}

#[test]
fn range_queries_follow_source_keyword_cache_mutations_and_bisect_order() {
    use domain_core::{
        time_calendar_cache::TimeCalendarKeyword, time_compat::day_minute_index_range_with_cache,
    };
    let snapshot = source_snapshot();
    let cases = snapshot["range_cache"].as_array().unwrap();
    assert_eq!(cases.len(), 144);
    let clock = |minute: &Value| {
        let minute = u32::try_from(minute.as_u64().unwrap()).unwrap();
        NaiveTime::from_hms_opt(minute / 60, minute % 60, 0).unwrap()
    };
    for case in cases {
        let cache = TimeCalendarCache::new();
        let region_text = case["region"].as_str().unwrap();
        let region = Region::from_str(region_text).unwrap();
        let calendar = cache
            .get(&TimeCalendarCall::new(
                Vec::new(),
                vec![TimeCalendarKeyword::new("region", region_text)],
            ))
            .unwrap();
        *calendar.lock().unwrap() = case["minutes"]
            .as_array()
            .unwrap()
            .iter()
            .map(clock)
            .collect();
        let positional = cache
            .get(&TimeCalendarCall::new(
                vec![0_i64.into(), region_text.into()],
                Vec::new(),
            ))
            .unwrap();
        *positional.lock().unwrap() = vec![NaiveTime::from_hms_opt(23, 59, 0).unwrap()];
        let frequency =
            Frequency::from_str(&format!("{}min", case["step"].as_str().unwrap())).unwrap();
        let result = day_minute_index_range_with_cache(
            clock(&case["start"]),
            clock(&case["end"]),
            &frequency,
            region.code(),
            &cache,
        );
        match result {
            Ok((left, right)) => assert_eq!(json!([left, right]), case["result"]["ok"], "{case}"),
            Err(error) => {
                assert_eq!(case["result"]["error"], "ValueError", "{case}");
                assert_eq!(case["result"]["message"], error.to_string(), "{case}");
            }
        }
    }
}

#[test]
fn default_alignment_observes_shared_cache_and_explicit_cache_reports_poison() {
    use domain_core::time_calendar_cache::default_time_calendar_cache;

    // A unique raw key avoids mutating entries used by concurrently running tests.
    let shift = BigInt::from(7_777);
    let call = TimeCalendarCall::new(vec![shift.clone().into(), "tw".into()], Vec::new());
    let calendar = default_time_calendar_cache().get(&call).unwrap();
    let saved = calendar.lock().unwrap().clone();
    *calendar.lock().unwrap() = vec![NaiveTime::from_hms_opt(9, 0, 0).unwrap()];
    let input = timestamp(
        utc_nanos("2021-01-01T10:38:00Z"),
        TimeUnit::Nanosecond,
        None,
    );
    let actual = align_sampled_minute_compatible(&input, &BigInt::from(1), &shift, Region::Tw);
    *calendar.lock().unwrap() = saved;
    assert_eq!(
        actual,
        Ok(timestamp(
            1_609_491_600_000_000,
            TimeUnit::Microsecond,
            None
        ))
    );

    let cache = TimeCalendarCache::new();
    let poisoned = cache.get(&call).unwrap();
    assert!(
        std::thread::spawn(move || {
            let _guard = poisoned.lock().unwrap();
            panic!("simulate a consumer panic while editing its calendar");
        })
        .join()
        .is_err()
    );
    assert_eq!(
        align_sampled_minute_with_cache(&input, &BigInt::from(1), &shift, Region::Tw, &cache),
        Err(TimeCompatError::CalendarLockPoisoned)
    );
}

#[test]
fn default_range_uses_keyword_cache_and_explicit_range_reports_poison() {
    use domain_core::{
        time_calendar_cache::{TimeCalendarKeyword, default_time_calendar_cache},
        time_compat::{day_minute_index_range_compatible, day_minute_index_range_with_cache},
    };
    let call = TimeCalendarCall::new(Vec::new(), vec![TimeCalendarKeyword::new("region", "tw")]);
    let calendar = default_time_calendar_cache().get(&call).unwrap();
    let saved = calendar.lock().unwrap().clone();
    let clock = NaiveTime::from_hms_opt(9, 0, 0).unwrap();
    *calendar.lock().unwrap() = vec![clock, clock];
    let frequency = Frequency::new(1, FrequencyUnit::Day);
    let result = day_minute_index_range_compatible(clock, clock, &frequency, Region::Tw);
    *calendar.lock().unwrap() = saved;
    assert_eq!(result, Ok((0, 1)));
    let cache = TimeCalendarCache::new();
    let poisoned = cache.get(&call).unwrap();
    assert!(
        std::thread::spawn(move || {
            let _guard = poisoned.lock().unwrap();
            panic!("simulate a calendar consumer panic");
        })
        .join()
        .is_err()
    );
    assert_eq!(
        day_minute_index_range_with_cache(clock, clock, &frequency, "tw", &cache),
        Err(TimeCompatError::CalendarLockPoisoned)
    );
}

#[test]
fn raw_range_region_errors_precede_zero_sampling_and_do_not_cache_failures() {
    use domain_core::time_compat::day_minute_index_range_with_cache;
    let snapshot = source_snapshot();
    let cases = snapshot["range_errors"].as_array().unwrap();
    assert_eq!(cases.len(), 8);
    let cache = TimeCalendarCache::new();
    for case in cases {
        let region = case["region"].as_str().unwrap();
        let frequency = Frequency::new(case["step"].as_u64().unwrap(), FrequencyUnit::Minute);
        let error = day_minute_index_range_with_cache(
            NaiveTime::from_hms_opt(9, 30, 0).unwrap(),
            NaiveTime::from_hms_opt(10, 0, 0).unwrap(),
            &frequency,
            region,
            &cache,
        )
        .unwrap_err();
        assert_eq!(
            error,
            TimeCompatError::CalendarCache(TimeCalendarCacheError::UnsupportedRegion {
                region: region.to_owned()
            })
        );
        assert_eq!(case["result"]["error"], "ValueError");
        assert_eq!(case["result"]["message"], error.to_string());
    }
    assert_eq!(cache.cache_info().misses, 8);
    assert_eq!(cache.cache_info().currsize, 0);
}

#[test]
fn timestamp_date_errors_precede_empty_calendar_but_follow_sampling() {
    let snapshot = source_snapshot();
    let cases = snapshot["date_precedence"].as_array().unwrap();
    assert_eq!(cases.len(), 8);
    for case in cases {
        let cache = TimeCalendarCache::new();
        if case["empty"].as_bool().unwrap() {
            cache
                .get(&TimeCalendarCall::new(
                    vec![0_i64.into(), "cn".into()],
                    Vec::new(),
                ))
                .unwrap()
                .lock()
                .unwrap()
                .clear();
        }
        let step = case["step"].as_i64().unwrap();
        let error = align_sampled_minute_with_cache(
            &timestamp(case["ticks"].as_i64().unwrap(), TimeUnit::Second, None),
            &BigInt::from(step),
            &BigInt::from(0),
            Region::Cn,
            &cache,
        )
        .unwrap_err();
        if step == 0 {
            assert_eq!(
                error,
                TimeCompatError::Alignment(MinuteAlignmentError::ZeroSamplingStep)
            );
            assert_eq!(case["result"]["error"], "ValueError");
        } else {
            assert_eq!(error, TimeCompatError::TimestampDateNotSupported);
            assert_eq!(case["result"]["error"], "NotImplementedError");
        }
        assert_eq!(case["result"]["message"], error.to_string());
    }
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "the pinned source snapshot is clearer beside its exact compatibility assertions"
)]
fn pinned_source_characterizes_frequency_and_temporal_boundaries() {
    let actual = source_snapshot();
    assert_eq!(
        actual["pandas_version"], "2.3.3",
        "this snapshot targets the migration's Pandas 2.3.3 baseline; Pandas 3 changes Timedelta resolution"
    );
    assert_eq!(
        actual["frequency"]["first_text"]["ok"],
        json!({"kind": "str", "text": "01MIN"})
    );
    assert_eq!(
        actual["frequency"]["first_freq"]["ok"],
        json!({"kind": "str", "text": "1min"})
    );
    assert_eq!(
        actual["frequency"]["later_text"]["ok"],
        json!({"kind": "str", "text": "02MIN"})
    );
    assert_eq!(
        actual["frequency"]["later_freq"]["ok"],
        json!({"kind": "Freq", "text": "2min", "count": 2, "base": "min"})
    );
    assert_eq!(
        actual["frequency"]["tie"]["ok"],
        json!({"kind": "str", "text": "2min"})
    );
    assert!(actual["frequency"]["none"]["ok"].is_null());
    assert_eq!(
        actual["frequency"]["invalid_after_eligible"]["error"],
        "ValueError"
    );
    assert_eq!(actual["concat"]["minimum"]["unit"], "us");
    assert_eq!(
        actual["concat"]["minimum"]["ticks"],
        -62_135_593_076_543_211_i64
    );
    assert_eq!(
        actual["concat"]["maximum"]["ticks"],
        253_402_300_799_999_999_i64
    );
    assert_eq!(actual["epsilon"]["seconds_forward"]["ok"]["unit"], "ns");
    assert_eq!(
        actual["epsilon"]["microseconds_backward"]["ok"]["ticks"],
        1_609_459_199_123_456_000_i64
    );
    assert_eq!(
        actual["epsilon"]["iana_forward"]["ok"]["timezone"],
        "Asia/Shanghai"
    );
    assert_eq!(
        actual["epsilon"]["nat_forward"]["ok"],
        json!({"kind": "NaT", "text": "NaT"})
    );
    assert_eq!(
        actual["epsilon"]["nat_invalid_direction"]["message"],
        "Wrong input"
    );
    assert_eq!(
        actual["epsilon"]["minimum_backward"]["error"],
        "OutOfBoundsDatetime"
    );
    assert_eq!(
        actual["epsilon"]["maximum_forward"]["error"],
        "OutOfBoundsDatetime"
    );
    for name in [
        "naive",
        "utc",
        "fixed",
        "new_york_summer",
        "new_york_winter",
    ] {
        assert_eq!(actual["alignment"][name]["ok"]["unit"], "us");
        assert!(actual["alignment"][name]["ok"]["timezone"].is_null());
    }
    assert_eq!(
        actual["alignment"]["new_york_summer"]["ok"]["text"],
        "2021-07-01 10:35:00"
    );
    assert_eq!(
        actual["alignment"]["new_york_winter"]["ok"]["text"],
        "2021-01-01 10:35:00"
    );
    assert_eq!(
        actual["alignment"]["nat"]["message"],
        "NaTType does not support time"
    );
    assert_eq!(
        actual["alignment"]["zero_step"]["message"],
        "slice step cannot be zero"
    );
}

#[test]
fn recent_frequency_preserves_source_result_kind_spelling_and_big_counts() {
    let base: CompatibleFrequency = "day".into();
    assert_eq!(
        recent_frequency_compatible(&base, &["01MIN".into()]),
        Ok(Some(CompatibleFrequency::Text("01MIN".into())))
    );
    assert_eq!(
        recent_frequency_compatible(
            &base,
            &[CompatibleFrequency::Frequency(Frequency::new(
                1,
                FrequencyUnit::Minute
            ))]
        ),
        Ok(Some(CompatibleFrequency::Text("1min".into())))
    );
    assert_eq!(
        recent_frequency_compatible(&base, &["1min".into(), "02MIN".into()]),
        Ok(Some(CompatibleFrequency::Text("02MIN".into())))
    );
    let parsed_two = Frequency::new(2, FrequencyUnit::Minute);
    assert_eq!(
        recent_frequency_compatible(
            &base,
            &[
                "1min".into(),
                CompatibleFrequency::Frequency(parsed_two.clone())
            ]
        ),
        Ok(Some(CompatibleFrequency::Frequency(parsed_two)))
    );
    assert_eq!(
        recent_frequency_compatible(&base, &["2min".into(), "02MIN".into()]),
        Ok(Some(CompatibleFrequency::Text("2min".into())))
    );
    assert_eq!(
        recent_frequency_compatible(&"1min".into(), &["day".into(), "week".into()]),
        Ok(None)
    );
    let huge_count = BigUint::from_str(&"9".repeat(80)).expect("fixture integer parses");
    let huge = CompatibleFrequency::Frequency(Frequency {
        count: huge_count,
        unit: FrequencyUnit::Minute,
    });
    assert_eq!(
        recent_frequency_compatible(&huge, &["1min".into(), "2min".into()]),
        Ok(Some(CompatibleFrequency::Text("2min".into())))
    );
    assert!(matches!(
        recent_frequency_compatible(&base, &["1min".into(), "bad".into()]),
        Err(FrequencyError::UnsupportedFormat { input }) if input == "bad"
    ));
    assert_eq!(
        CompatibleFrequency::from(String::from("D")),
        CompatibleFrequency::Text("D".into())
    );
    let converted = Frequency::new(3, FrequencyUnit::Day);
    assert_eq!(
        CompatibleFrequency::from(converted.clone()),
        CompatibleFrequency::Frequency(converted)
    );
    assert!(matches!(
        recent_frequency_compatible(&"bad".into(), &["1min".into()]),
        Err(FrequencyError::UnsupportedFormat { input }) if input == "bad"
    ));
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one table-driven compatibility test keeps all temporal unit and boundary cases together"
)]
fn concat_and_epsilon_preserve_units_sentinels_zones_and_errors() {
    let ordinary_date = NaiveDate::from_ymd_opt(2020, 2, 29).expect("fixture date is valid");
    let ordinary_time =
        NaiveTime::from_hms_micro_opt(1, 2, 3, 456_789).expect("fixture time is valid");
    assert_eq!(
        concat_date_time_compatible(ordinary_date, ordinary_time),
        Ok(timestamp(
            1_582_938_123_456_789,
            TimeUnit::Microsecond,
            None
        ))
    );
    for (year, expected) in [
        (1, -62_135_593_076_543_211_i64),
        (9_999, 253_370_768_523_456_789_i64),
    ] {
        let date = NaiveDate::from_ymd_opt(year, 1, 1).expect("fixture date is valid");
        assert_eq!(
            concat_date_time_compatible(date, ordinary_time),
            Ok(timestamp(expected, TimeUnit::Microsecond, None))
        );
    }
    let year_zero = NaiveDate::from_ymd_opt(0, 1, 1).expect("Chrono supports year zero");
    assert_eq!(
        concat_date_time_compatible(year_zero, ordinary_time),
        Err(TimeCompatError::DateYearOutOfRange { year: 0 })
    );
    let submicro = NaiveTime::from_hms_nano_opt(1, 2, 3, 456_789_001)
        .expect("Chrono supports nanosecond time");
    assert_eq!(
        concat_date_time_compatible(ordinary_date, submicro),
        Err(TimeCompatError::SubmicrosecondTime)
    );
    let leap_second = NaiveTime::from_hms_nano_opt(23, 59, 59, 1_500_000_000)
        .expect("Chrono represents leap seconds");
    assert_eq!(
        concat_date_time_compatible(ordinary_date, leap_second),
        Err(TimeCompatError::SubmicrosecondTime)
    );

    let seconds = timestamp(1_609_459_200, TimeUnit::Second, None);
    assert_eq!(
        epsilon_change_compatible(&seconds, "forward"),
        Ok(timestamp(
            1_609_459_201_000_000_000,
            TimeUnit::Nanosecond,
            None
        ))
    );
    let micros = timestamp(1_609_459_200_123_456, TimeUnit::Microsecond, None);
    assert_eq!(
        epsilon_change_compatible(&micros, "backward"),
        Ok(timestamp(
            1_609_459_199_123_456_000,
            TimeUnit::Nanosecond,
            None
        ))
    );
    let millis = timestamp(1_609_459_200_123, TimeUnit::Millisecond, None);
    assert_eq!(
        epsilon_change_compatible(&millis, "forward"),
        Ok(timestamp(
            1_609_459_201_123_000_000,
            TimeUnit::Nanosecond,
            None
        ))
    );
    let aware = timestamp(
        1_609_464_645_123_456_789,
        TimeUnit::Nanosecond,
        Some("Asia/Shanghai"),
    );
    assert_eq!(
        epsilon_change_compatible(&aware, "forward"),
        Ok(timestamp(
            1_609_464_646_123_456_789,
            TimeUnit::Nanosecond,
            Some("Asia/Shanghai")
        ))
    );
    let nat = TemporalFrameValue::Builtin(BuiltinFrameValue::NotATime);
    assert_eq!(epsilon_change_compatible(&nat, "backward"), Ok(nat.clone()));
    assert!(matches!(
        epsilon_change_compatible(&nat, "Backward"),
        Err(TimeCompatError::Epsilon(EpsilonError::InvalidDirection { direction })) if direction == "Backward"
    ));
    assert!(matches!(
        epsilon_change_compatible(&timestamp(i64::MAX, TimeUnit::Second, None), "forward"),
        Err(TimeCompatError::TimestampOutOfRange { .. })
    ));
    for unit in [TimeUnit::Millisecond, TimeUnit::Microsecond] {
        assert!(matches!(
            epsilon_change_compatible(&timestamp(i64::MAX, unit, None), "forward"),
            Err(TimeCompatError::TimestampOutOfRange { .. })
        ));
    }
    assert!(matches!(
        epsilon_change_compatible(&timestamp(i64::MAX, TimeUnit::Nanosecond, None), "forward"),
        Err(TimeCompatError::Epsilon(
            EpsilonError::ResultOutOfRange { .. }
        ))
    ));
    for unsupported in [
        TemporalFrameValue::Builtin(BuiltinFrameValue::None),
        TemporalFrameValue::Duration {
            ticks: 1,
            unit: TimeUnit::Second,
        },
    ] {
        assert_eq!(
            epsilon_change_compatible(&unsupported, "forward"),
            Err(TimeCompatError::UnsupportedTemporalValue {
                operation: "epsilon_change"
            })
        );
    }
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one table-driven compatibility test keeps timezone and timestamp-unit cases together"
)]
fn alignment_resolves_utc_fixed_and_iana_local_time_then_strips_zone() {
    let sample = BigInt::from(5);
    let shift = BigInt::from(0);
    let cases = [
        (
            timestamp(1_609_497_525, TimeUnit::Second, None),
            1_609_497_300_000_000_i64,
        ),
        (
            timestamp(1_609_497_525_123, TimeUnit::Millisecond, None),
            1_609_497_300_000_000_i64,
        ),
        (
            timestamp(1_609_497_525_123_456, TimeUnit::Microsecond, None),
            1_609_497_300_000_000_i64,
        ),
        (
            timestamp(
                utc_nanos("2021-01-01T10:38:45.123456789Z"),
                TimeUnit::Nanosecond,
                None,
            ),
            1_609_497_300_000_000_i64,
        ),
        (
            timestamp(
                utc_nanos("2021-01-01T10:38:45.123456789Z"),
                TimeUnit::Nanosecond,
                Some("UTC"),
            ),
            1_609_497_300_000_000_i64,
        ),
        (
            timestamp(
                utc_nanos("2021-01-01T02:38:45.123456789Z"),
                TimeUnit::Nanosecond,
                Some("+08:00"),
            ),
            1_609_497_300_000_000_i64,
        ),
        (
            timestamp(
                utc_nanos("2021-07-01T14:38:45.123456789Z"),
                TimeUnit::Nanosecond,
                Some("America/New_York"),
            ),
            1_625_135_700_000_000_i64,
        ),
        (
            timestamp(
                utc_nanos("2021-01-01T15:38:45.123456789Z"),
                TimeUnit::Nanosecond,
                Some("America/New_York"),
            ),
            1_609_497_300_000_000_i64,
        ),
    ];
    for (input, ticks) in cases {
        assert_eq!(
            align_sampled_minute_compatible(&input, &sample, &shift, Region::Cn),
            Ok(timestamp(ticks, TimeUnit::Microsecond, None))
        );
    }
    assert_eq!(
        align_sampled_minute_compatible(
            &TemporalFrameValue::Builtin(BuiltinFrameValue::NotATime),
            &sample,
            &shift,
            Region::Cn
        ),
        Err(TimeCompatError::NaTDoesNotSupportTime)
    );
    assert_eq!(
        align_sampled_minute_compatible(
            &TemporalFrameValue::Builtin(BuiltinFrameValue::None),
            &sample,
            &shift,
            Region::Cn
        ),
        Err(TimeCompatError::UnsupportedTemporalValue {
            operation: "cal_sam_minute"
        })
    );
    assert!(matches!(
        align_sampled_minute_compatible(
            &timestamp(0, TimeUnit::Second, Some("Mars/Olympus")),
            &sample,
            &shift,
            Region::Cn
        ),
        Err(TimeCompatError::InvalidTimezone { timezone, .. }) if timezone == "Mars/Olympus"
    ));
    assert_eq!(
        align_sampled_minute_compatible(
            &timestamp(0, TimeUnit::Second, None),
            &BigInt::from(0),
            &shift,
            Region::Cn
        ),
        Err(TimeCompatError::Alignment(
            MinuteAlignmentError::ZeroSamplingStep
        ))
    );
    assert!(matches!(
        align_sampled_minute_compatible(
            &timestamp(i64::MAX, TimeUnit::Second, None),
            &sample,
            &shift,
            Region::Cn
        ),
        Err(TimeCompatError::TimestampDateNotSupported)
    ));
}
