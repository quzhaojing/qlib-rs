use std::{path::PathBuf, process::Command, str::FromStr, sync::Arc};

use chrono::{NaiveDate, NaiveDateTime, NaiveTime, TimeDelta};
use domain_core::{
    EpsilonDirection, Frequency, FrequencyUnit, IntradayIndexError, MinuteAlignmentError, Region,
    align_sampled_minute, day_minute_index_range, epsilon_change, epsilon_change_backward,
    is_single_market_value, minute_calendar, time_to_day_index_str,
};
use num_bigint::{BigInt, BigUint};
use serde_json::{Value, json};

const UPSTREAM_TIME_SHA256: &str =
    "af7ac3709cac0d2a11a15aac478c7ceb68579d492aed322695f5f25261a69699";

fn source_path() -> PathBuf {
    std::env::var_os("QLIB_PYTHON_TIME").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/utils/time.py"),
        PathBuf::from,
    )
}

fn timestamp(text: &str) -> NaiveDateTime {
    NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S%.f").expect("fixture timestamp is valid")
}

fn clock(hour: u32, minute: u32) -> NaiveTime {
    NaiveTime::from_hms_opt(hour, minute, 0).expect("fixture clock time is valid")
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "the whole-file closure keeps one source snapshot and its native counterpart assertions together"
)]
fn time_whole_file_source_and_native_contracts_close_together() {
    let source = source_path();
    assert!(
        source.is_file(),
        "Python source not found: {}",
        source.display()
    );
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/time_whole_file_probe.py");
    let python = std::env::var_os("PYTHON").unwrap_or_else(|| "python".into());
    let output = Command::new(python)
        .arg(&fixture)
        .arg(&source)
        .output()
        .expect("Python interpreter starts");
    assert!(
        output.status.success(),
        "whole-file characterization failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let actual: Value =
        serde_json::from_slice(&output.stdout).expect("whole-file probe returns JSON");

    assert_eq!(actual["source_sha256"], UPSTREAM_TIME_SHA256);
    assert_eq!(
        actual["module_symbols"],
        json!([
            "CN_TIME",
            "Freq",
            "TW_TIME",
            "US_TIME",
            "cal_sam_minute",
            "concat_date_time",
            "epsilon_change",
            "get_day_min_idx_range",
            "get_min_cal",
            "is_single_value",
            "time_to_day_index"
        ])
    );
    assert_eq!(
        actual["freq_symbols"],
        json!([
            "NORM_FREQ_DAY",
            "NORM_FREQ_MINUTE",
            "NORM_FREQ_MONTH",
            "NORM_FREQ_WEEK",
            "SUPPORT_CAL_LIST",
            "__eq__",
            "__init__",
            "__repr__",
            "__str__",
            "get_min_delta",
            "get_recent_freq",
            "get_timedelta",
            "parse"
        ])
    );
    assert_eq!(
        actual["defaults"],
        json!({
            "get_min_cal": "(shift: 'int' = 0, region: 'str' = 'cn') -> 'List[time]'",
            "is_single_value": "(start_time, end_time, freq, region: 'str' = 'cn')",
            "time_to_day_index": "(time_obj: 'Union[str, datetime]', region: 'str' = 'cn')",
            "get_day_min_idx_range": "(start: 'str', end: 'str', freq: 'str', region: 'str') -> 'Tuple[int, int]'",
            "concat_date_time": "(date_obj: 'date', time_obj: 'time') -> 'pd.Timestamp'",
            "cal_sam_minute": "(x: 'pd.Timestamp', sam_minutes: 'int', region: 'str' = 'cn') -> 'pd.Timestamp'",
            "epsilon_change": "(date_time: 'pd.Timestamp', direction: 'str' = 'backward') -> 'pd.Timestamp'"
        })
    );
    assert_eq!(
        actual["constants"],
        json!({
            "CN_TIME": {"type": "list", "element_type": "datetime", "values": ["09:30:00", "11:30:00", "13:00:00", "15:00:00"]},
            "US_TIME": {"type": "list", "element_type": "datetime", "values": ["09:30:00", "16:00:00"]},
            "TW_TIME": {"type": "list", "element_type": "datetime", "values": ["09:00:00", "13:30:00"]}
        })
    );
    assert_eq!(actual["calendar"]["cache_identity"], true);
    assert_eq!(
        actual["calendar"]["mutated_cache"],
        json!({"same": true, "length": 241})
    );
    assert_eq!(
        actual["calendar"]["call_shapes"],
        json!({
            "identities": [false, false, false, false, false, false],
            "lengths_after_default_mutation": [241, 240, 240, 240],
        "info": [0, 4, 240, 4]
        })
    );
    assert_eq!(actual["calendar"]["post_clear_length"], 240);
    assert_eq!(actual["calendar"]["unsupported"]["error"], "ValueError");
    assert_eq!(
        actual["calendar"]["unsupported"]["message"],
        "xx is not supported"
    );
    assert_eq!(
        actual["frequency"]["copy"],
        json!({"base": "min", "count": 5, "text": "5min"})
    );
    assert_eq!(
        actual["frequency"]["recent_strings"],
        json!({"type": "str", "value": "2min"})
    );
    assert_eq!(
        actual["frequency"]["recent_objects"],
        json!({"type": "Freq", "value": "2min"})
    );
    assert_eq!(
        actual["frequency"]["invalid_init"]["error"],
        "NotImplementedError"
    );
    assert_eq!(
        actual["frequency"]["invalid_equal"]["error"],
        "NotImplementedError"
    );
    assert_eq!(actual["frequency"]["timedelta"]["type"], "Timedelta");
    assert_eq!(
        actual["frequency"]["timedelta"]["nanoseconds"],
        120_000_000_000_i64
    );
    assert_eq!(actual["index"]["closed"], json!([0, 23]));
    assert_eq!(actual["index"]["ignored_unit"], json!([0, 5]));
    assert_eq!(actual["index"]["zero_step"]["error"], "ValueError");
    assert_eq!(
        actual["single_value"]["unsupported"],
        json!({
            "error": "NotImplementedError", "message": "please implement the is_single_value func for xx"
        })
    );
    assert_eq!(actual["concat"]["type"], "Timestamp");
    assert_eq!(actual["concat"]["value"], "2020-02-29 01:02:03.456789");
    assert_eq!(
        actual["alignment"],
        json!({
            "type": "Timestamp", "value": "2021-03-03 09:29:00", "timezone": "None",
            "zero_step": {"error": "ValueError", "message": "slice step cannot be zero"}
        })
    );
    assert_eq!(
        actual["epsilon"]["nat"],
        json!({"ok": "NaT", "type": "NaTType"})
    );
    assert_eq!(
        actual["epsilon"]["invalid"],
        json!({"error": "ValueError", "message": "Wrong input"})
    );

    let five_minutes = Frequency::from_str("5MIN").expect("source-compatible frequency parses");
    assert_eq!(five_minutes.count, BigUint::from(5_u8));
    assert_eq!(five_minutes.unit, FrequencyUnit::Minute);
    assert_eq!(five_minutes.to_string(), "5min");
    assert_eq!(format!("{five_minutes:?}"), "Freq(5min)");
    assert_eq!(
        Frequency::from_str("D").expect("alias parses"),
        Frequency::new(1, FrequencyUnit::Day)
    );
    assert_eq!(
        Frequency::nearest_resample_source(
            &Frequency::new(1, FrequencyUnit::Day),
            &[
                Frequency::new(1, FrequencyUnit::Minute),
                Frequency::new(2, FrequencyUnit::Minute)
            ]
        ),
        Some(Frequency::new(2, FrequencyUnit::Minute))
    );
    assert_eq!(
        Frequency::time_delta(BigInt::from(2), "min").expect("fixed duration is valid"),
        TimeDelta::minutes(2)
    );

    for (region, length, first, last) in [
        (Region::Cn, 240, clock(9, 30), clock(14, 59)),
        (Region::Us, 390, clock(9, 30), clock(15, 59)),
        (Region::Tw, 270, clock(9, 0), clock(13, 29)),
    ] {
        let calendar = minute_calendar(&BigInt::from(0), region).expect("zero shift is valid");
        assert_eq!(
            (calendar.len(), calendar[0], calendar[length - 1]),
            (length, first, last)
        );
        let cached = minute_calendar(&BigInt::from(0), region).expect("cached call succeeds");
        assert!(Arc::ptr_eq(&calendar, &cached));
    }

    assert!(is_single_market_value(
        clock(11, 29),
        TimeDelta::hours(1),
        TimeDelta::minutes(1),
        Region::Cn
    ));
    assert_eq!(time_to_day_index_str("9:30", Region::Cn), Ok(0));
    assert!(matches!(
        time_to_day_index_str("11:30", Region::Cn),
        Err(IntradayIndexError::OutsideTradingSessions { .. })
    ));
    assert_eq!(
        day_minute_index_range(
            clock(8, 30),
            clock(14, 59),
            &Frequency::new(10, FrequencyUnit::Minute),
            Region::Cn
        ),
        Ok((0, 23))
    );
    assert_eq!(
        day_minute_index_range(
            clock(9, 30),
            clock(9, 40),
            &Frequency::new(2, FrequencyUnit::Day),
            Region::Cn
        ),
        Ok((0, 5))
    );

    let aligned = align_sampled_minute(
        timestamp("2021-03-03 09:30:45"),
        &BigInt::from(5),
        &BigInt::from(1),
        Region::Cn,
    )
    .expect("alignment succeeds");
    assert_eq!(aligned, timestamp("2021-03-03 09:29:00"));
    assert_eq!(
        align_sampled_minute(
            timestamp("2021-03-03 09:30:45"),
            &BigInt::from(0),
            &BigInt::from(0),
            Region::Cn,
        ),
        Err(MinuteAlignmentError::ZeroSamplingStep)
    );

    let leap = NaiveDate::from_ymd_opt(2020, 2, 29)
        .expect("fixture date is valid")
        .and_hms_micro_opt(1, 2, 3, 456_789)
        .expect("fixture time is valid");
    assert_eq!(leap, timestamp("2020-02-29 01:02:03.456789"));
    let epsilon_input = timestamp("2021-01-01 00:00:00.123456789");
    assert_eq!(
        epsilon_change_backward(epsilon_input),
        Ok(timestamp("2020-12-31 23:59:59.123456789"))
    );
    assert_eq!(
        epsilon_change(timestamp("2021-01-01 00:00:00"), EpsilonDirection::Forward),
        Ok(timestamp("2021-01-01 00:00:01"))
    );
}
