use std::{path::PathBuf, process::Command, str::FromStr};

use arrow_schema::TimeUnit;
use chrono::{DateTime, NaiveDate, NaiveTime};
use domain_core::{
    EpsilonError, Frequency, FrequencyError, FrequencyUnit, MinuteAlignmentError, Region,
    dataframe_append::{BuiltinFrameValue, TemporalFrameValue},
    time_compat::{
        CompatibleFrequency, TimeCompatError, align_sampled_minute_compatible,
        concat_date_time_compatible, epsilon_change_compatible, recent_frequency_compatible,
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

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "the pinned source snapshot is clearer beside its exact compatibility assertions"
)]
fn pinned_source_characterizes_frequency_and_temporal_boundaries() {
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
        .output()
        .expect("Python interpreter starts");
    assert!(
        output.status.success(),
        "source characterization failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let actual: Value = serde_json::from_slice(&output.stdout).expect("fixture returns JSON");
    assert_eq!(actual["source_sha256"], UPSTREAM_TIME_SHA256);
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
        Err(TimeCompatError::TimestampOutOfRange { .. })
    ));
}
