use std::{path::PathBuf, process::Command, str::FromStr};

use chrono::TimeDelta;
use domain_core::{Frequency, FrequencyError, FrequencyUnit, SUPPORTED_CALENDAR_UNITS};
use num_bigint::{BigInt, BigUint};
use num_traits::ToPrimitive;
use serde_json::{Value, json};

#[test]
fn parsing_normalizes_every_python_alias() {
    let cases = [
        ("day", 1, FrequencyUnit::Day, "day"),
        ("1day", 1, FrequencyUnit::Day, "day"),
        ("D", 1, FrequencyUnit::Day, "day"),
        ("2d", 2, FrequencyUnit::Day, "2day"),
        ("week", 1, FrequencyUnit::Week, "1week"),
        ("3W", 3, FrequencyUnit::Week, "3week"),
        ("month", 1, FrequencyUnit::Month, "1month"),
        ("2mon", 2, FrequencyUnit::Month, "2month"),
        ("minute", 1, FrequencyUnit::Minute, "1min"),
        ("15min", 15, FrequencyUnit::Minute, "15min"),
        ("0min", 0, FrequencyUnit::Minute, "0min"),
        ("0002MIN", 2, FrequencyUnit::Minute, "2min"),
    ];

    for (input, count, unit, normalized) in cases {
        let frequency: Frequency = input.parse().expect("supported frequency parses");
        assert_eq!(frequency, Frequency::new(count, unit));
        assert_eq!(frequency.to_string(), normalized);
        assert_eq!(format!("{frequency:?}"), format!("Freq({normalized})"));
    }

    assert_eq!(
        SUPPORTED_CALENDAR_UNITS,
        [FrequencyUnit::Minute, FrequencyUnit::Day]
    );
}

#[test]
fn parsing_rejects_invalid_shapes_and_units_without_limiting_integer_size() {
    for input in ["", "-1day", "1hour", "day "] {
        assert_eq!(
            Frequency::from_str(input),
            Err(FrequencyError::UnsupportedFormat {
                input: input.to_owned()
            })
        );
    }

    let arbitrary_precision = format!("{}0min", u64::MAX);
    let parsed = Frequency::from_str(&arbitrary_precision)
        .expect("Python-compatible arbitrary precision count parses");
    assert_eq!(parsed.count.to_string(), format!("{}0", u64::MAX));
    assert_eq!(
        FrequencyUnit::from_str("hour")
            .expect_err("unsupported unit is rejected")
            .to_string(),
        "Matching variant not found"
    );
}

#[test]
fn frequency_serializes_as_the_normalized_config_string() {
    for frequency in [
        Frequency::new(1, FrequencyUnit::Day),
        Frequency::new(15, FrequencyUnit::Minute),
        Frequency::new(2, FrequencyUnit::Week),
        Frequency::new(3, FrequencyUnit::Month),
    ] {
        let encoded = serde_json::to_string(&frequency).expect("frequency serializes");
        assert_eq!(encoded, format!("\"{frequency}\""));
        assert_eq!(
            serde_json::from_str::<Frequency>(&encoded).expect("frequency deserializes"),
            frequency
        );
    }

    assert_eq!(
        serde_json::from_str::<Frequency>("\"MINUTE\"").expect("aliases deserialize"),
        Frequency::new(1, FrequencyUnit::Minute)
    );
    assert!(serde_json::from_str::<Frequency>("\"hour\"").is_err());
}

#[test]
fn approximate_minutes_and_deltas_preserve_upstream_weights() {
    assert_eq!(
        Frequency::new(1, FrequencyUnit::Minute).approximate_minutes(),
        BigUint::from(1_u8)
    );
    assert_eq!(
        Frequency::new(1, FrequencyUnit::Day).approximate_minutes(),
        BigUint::from(1_440_u16)
    );
    assert_eq!(
        Frequency::new(1, FrequencyUnit::Week).approximate_minutes(),
        BigUint::from(10_080_u16)
    );
    assert_eq!(
        Frequency::new(1, FrequencyUnit::Month).approximate_minutes(),
        BigUint::from(302_400_u32)
    );

    let day = Frequency::new(1, FrequencyUnit::Day);
    let minute = Frequency::new(1, FrequencyUnit::Minute);
    assert_eq!(day.minute_delta(&minute), BigInt::from(1_439));
    assert_eq!(minute.minute_delta(&day), BigInt::from(-1_439));
    assert_eq!(day.minute_delta(&day), BigInt::from(0));
}

#[test]
fn nearest_resample_source_preserves_order_and_filters_coarser_values() {
    let minute = Frequency::new(1, FrequencyUnit::Minute);
    let three_minutes = Frequency::new(3, FrequencyUnit::Minute);
    let five_minutes = Frequency::new(5, FrequencyUnit::Minute);
    let day = Frequency::new(1, FrequencyUnit::Day);
    let week = Frequency::new(1, FrequencyUnit::Week);

    assert_eq!(Frequency::nearest_resample_source(&day, &[]), None);
    assert_eq!(
        Frequency::nearest_resample_source(&minute, &[day.clone(), week]),
        None
    );
    assert_eq!(
        Frequency::nearest_resample_source(
            &five_minutes,
            &[day.clone(), minute, three_minutes.clone()]
        ),
        Some(three_minutes)
    );
    assert_eq!(
        Frequency::nearest_resample_source(
            &Frequency::new(2, FrequencyUnit::Day),
            &[day.clone(), Frequency::new(1_440, FrequencyUnit::Minute)]
        ),
        Some(day)
    );
}

#[test]
fn fixed_time_delta_matches_pandas_supported_spellings_and_errors() {
    let cases = [
        (1, "min", TimeDelta::minutes(1)),
        (2, "minute", TimeDelta::minutes(2)),
        (1, "day", TimeDelta::days(1)),
        (-1, "D", TimeDelta::days(-1)),
        (1, "w", TimeDelta::weeks(1)),
        (0, "MIN", TimeDelta::zero()),
    ];
    for (count, unit, expected) in cases {
        assert_eq!(
            Frequency::time_delta(BigInt::from(count), unit).expect("supported duration converts"),
            expected
        );
    }

    for unit in ["week", "month", "mon", "hour"] {
        assert_eq!(
            Frequency::time_delta(BigInt::from(1), unit),
            Err(FrequencyError::UnsupportedDurationUnit {
                unit: unit.to_owned()
            })
        );
    }
    assert_eq!(
        Frequency::time_delta(BigInt::from(i64::MAX), "day"),
        Err(FrequencyError::DurationOutOfRange {
            count: BigInt::from(i64::MAX),
            unit: "day".to_owned()
        })
    );
    let beyond_i64 = BigInt::from(i64::MAX) + BigInt::from(1);
    assert_eq!(
        Frequency::time_delta(beyond_i64.clone(), "day"),
        Err(FrequencyError::DurationOutOfRange {
            count: beyond_i64,
            unit: "day".to_owned()
        })
    );
}

#[test]
fn rust_behavior_matches_a_live_python_freq_snapshot() {
    let source = std::env::var_os("QLIB_PYTHON_TIME").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/utils/time.py"),
        PathBuf::from,
    );
    assert!(
        source.is_file(),
        "Python source not found: {}",
        source.display()
    );

    let script = r#"
import ast
import json
import re
import sys
import pandas as pd

tree = ast.parse(open(sys.argv[1], encoding="utf-8").read(), filename=sys.argv[1])
freq_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Freq")
module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), freq_class], type_ignores=[])
ast.fix_missing_locations(module)
namespace = {"re": re, "pd": pd}
exec(compile(module, sys.argv[1], "exec"), namespace)
Freq = namespace["Freq"]

parse_inputs = ["day", "1day", "D", "2d", "week", "3W", "month", "2mon", "minute", "15min", "0min", "0002MIN"]
parsed = {value: {"count": Freq(value).count, "base": Freq(value).base, "text": str(Freq(value)), "debug": repr(Freq(value))} for value in parse_inputs}
invalid = {}
for value in ["", "-1day", "1hour", "day "]:
    try:
        Freq(value)
    except Exception as error:
        invalid[value] = type(error).__name__

deltas = [
    Freq.get_min_delta("day", "1min"),
    Freq.get_min_delta("week", "day"),
    Freq.get_min_delta("month", "week"),
    Freq.get_min_delta("1min", "day"),
]
recent = []
for base, candidates in [("day", []), ("day", ["1min", "day", "week"]), ("5min", ["day", "1min", "3min"]), ("1min", ["day", "week"])]:
    result = Freq.get_recent_freq(base, candidates)
    recent.append(None if result is None else str(result))

timedeltas = []
for count, unit in [(1, "min"), (2, "minute"), (1, "day"), (-1, "d"), (1, "w"), (0, "min"), (1, "week"), (1, "month")]:
    try:
        timedeltas.append({"ns": Freq.get_timedelta(count, unit).value})
    except Exception as error:
        timedeltas.append({"error": type(error).__name__})

print(json.dumps({"parsed": parsed, "invalid": invalid, "deltas": deltas, "recent": recent, "timedeltas": timedeltas}, sort_keys=True))
"#;
    let python = std::env::var_os("PYTHON").unwrap_or_else(|| "python".into());
    let output = Command::new(python)
        .arg("-c")
        .arg(script)
        .arg(&source)
        .output()
        .expect("Python interpreter starts");
    assert!(
        output.status.success(),
        "Python snapshot failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let actual: Value =
        serde_json::from_slice(&output.stdout).expect("Python returns a JSON snapshot");

    let mut parsed = serde_json::Map::new();
    for input in [
        "day", "1day", "D", "2d", "week", "3W", "month", "2mon", "minute", "15min", "0min",
        "0002MIN",
    ] {
        let frequency: Frequency = input.parse().expect("fixture input parses");
        parsed.insert(
            input.to_owned(),
            json!({
                "count": frequency.count.to_u64().expect("fixture count fits u64"),
                "base": frequency.unit.to_string(),
                "text": frequency.to_string(),
                "debug": format!("{frequency:?}")
            }),
        );
    }
    let expected = json!({
        "parsed": parsed,
        "invalid": {"": "ValueError", "-1day": "ValueError", "1hour": "ValueError", "day ": "ValueError"},
        "deltas": [1439, 8640, 292_320, -1439],
        "recent": [null, "day", "3min", null],
        "timedeltas": [
            {"ns": 60_000_000_000_i64},
            {"ns": 120_000_000_000_i64},
            {"ns": 86_400_000_000_000_i64},
            {"ns": -86_400_000_000_000_i64},
            {"ns": 604_800_000_000_000_i64},
            {"ns": 0},
            {"error": "ValueError"},
            {"error": "ValueError"}
        ]
    });
    assert_eq!(actual, expected);
}
