use std::{path::PathBuf, process::Command, str::FromStr, sync::Arc};

use chrono::NaiveTime;
use domain_core::{
    Frequency, IntradayIndexError, Region, day_minute_index_range, minute_calendar,
    parse_market_time, regular_minute_calendar, time_to_day_index, time_to_day_index_str,
};
use num_bigint::BigInt;
use serde_json::{Value, json};

fn clock(hour: u32, minute: u32) -> NaiveTime {
    NaiveTime::from_hms_opt(hour, minute, 0).expect("fixture clock time is valid")
}

#[test]
fn clock_parsing_and_every_regular_minute_have_python_indices() {
    assert_eq!(parse_market_time("09:30"), Ok(clock(9, 30)));
    assert_eq!(parse_market_time("9:30"), Ok(clock(9, 30)));
    assert_eq!(parse_market_time("9:3"), Ok(clock(9, 3)));
    assert_eq!(parse_market_time(" 9:30"), Ok(clock(9, 30)));
    assert_eq!(
        parse_market_time("09:30:00"),
        Err(IntradayIndexError::InvalidTime {
            input: "09:30:00".to_owned()
        })
    );
    assert_eq!(time_to_day_index_str("13:00", Region::Cn), Ok(120));
    assert_eq!(
        time_to_day_index_str("bad", Region::Cn),
        Err(IntradayIndexError::InvalidTime {
            input: "bad".to_owned()
        })
    );

    for region in [Region::Cn, Region::Us, Region::Tw] {
        let calendar = regular_minute_calendar(region);
        assert!(Arc::ptr_eq(
            &calendar,
            &minute_calendar(&BigInt::from(0), region).expect("zero shift is valid")
        ));
        for (expected, time) in calendar.iter().copied().enumerate() {
            assert_eq!(
                time_to_day_index(time, region),
                Ok(i64::try_from(expected).expect("market calendar length fits i64"))
            );
        }
    }

    assert_eq!(
        time_to_day_index(
            NaiveTime::from_hms_micro_opt(9, 30, 59, 999_999).expect("fixture clock time is valid"),
            Region::Cn
        ),
        Ok(0)
    );
    assert_eq!(
        time_to_day_index(
            NaiveTime::from_hms_opt(11, 29, 59).expect("fixture clock time is valid"),
            Region::Cn
        ),
        Ok(119)
    );
    assert_eq!(
        time_to_day_index(
            NaiveTime::from_hms_opt(13, 0, 30).expect("fixture clock time is valid"),
            Region::Cn
        ),
        Ok(120)
    );

    for (time, region) in [
        (clock(11, 30), Region::Cn),
        (clock(12, 0), Region::Cn),
        (clock(15, 0), Region::Cn),
        (clock(16, 0), Region::Us),
        (clock(13, 30), Region::Tw),
    ] {
        assert_eq!(
            time_to_day_index(time, region),
            Err(IntradayIndexError::OutsideTradingSessions { time, region })
        );
    }
}

#[test]
fn sampled_closed_ranges_preserve_bisect_and_stride_edge_cases() {
    let cases = [
        ("08:30", "14:59", "10min", Region::Cn, (0, 23)),
        ("09:30", "14:59", "1min", Region::Cn, (0, 239)),
        ("11:29", "13:00", "1min", Region::Cn, (119, 120)),
        ("11:30", "12:59", "1min", Region::Cn, (120, 119)),
        ("08:00", "09:00", "1min", Region::Cn, (0, -1)),
        ("15:00", "16:00", "1min", Region::Cn, (240, 239)),
        ("14:00", "10:00", "1min", Region::Cn, (180, 30)),
        ("11:29", "13:03", "7min", Region::Cn, (17, 17)),
        ("09:30", "15:59", "5min", Region::Us, (0, 77)),
        ("09:00", "13:29", "5min", Region::Tw, (0, 53)),
        ("09:30", "09:31", "day", Region::Cn, (0, 1)),
    ];
    for (start, end, frequency, region, expected) in cases {
        let frequency = Frequency::from_str(frequency).expect("fixture frequency is valid");
        assert_eq!(
            day_minute_index_range(
                parse_market_time(start).expect("fixture start is valid"),
                parse_market_time(end).expect("fixture end is valid"),
                &frequency,
                region
            ),
            Ok(expected)
        );
    }

    let huge = format!("{}min", "1".repeat(100));
    let huge = Frequency::from_str(&huge).expect("arbitrary-size step parses");
    assert_eq!(
        day_minute_index_range(clock(9, 30), clock(15, 0), &huge, Region::Cn),
        Ok((0, 0))
    );
    let zero = Frequency::from_str("0min").expect("zero frequency parses");
    assert_eq!(
        day_minute_index_range(clock(9, 30), clock(10, 0), &zero, Region::Cn),
        Err(IntradayIndexError::ZeroFrequencyStep)
    );
}

#[test]
fn rust_indices_match_a_live_python_snapshot() {
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
import bisect
import functools
import json
import re
import sys
from datetime import date, datetime, time, timedelta
import pandas as pd

names = {"CN_TIME", "US_TIME", "TW_TIME", "get_min_cal", "Freq", "time_to_day_index", "get_day_min_idx_range"}
tree = ast.parse(open(sys.argv[1], encoding="utf-8").read(), filename=sys.argv[1])
body = []
for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in names for target in node.targets):
        body.append(node)
    elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
        body.append(node)
module = ast.Module(body=body, type_ignores=[])
ast.fix_missing_locations(module)
namespace = {
    "bisect": bisect, "date": date, "datetime": datetime, "functools": functools,
    "json": json, "pd": pd, "re": re, "time": time, "timedelta": timedelta,
    "List": list, "Optional": object, "Tuple": tuple, "Union": object,
    "REG_CN": "cn", "REG_US": "us", "REG_TW": "tw",
}
exec(compile(module, sys.argv[1], "exec"), namespace)

time_cases = [
    ("cn_open", "09:30", "cn"), ("cn_am_last", "11:29", "cn"),
    ("cn_pm_open", "13:00", "cn"), ("cn_last", "14:59", "cn"),
    ("us_last", "15:59", "us"), ("tw_last", "13:29", "tw"),
]
times = {name: namespace["time_to_day_index"](value, region) for name, value, region in time_cases}
times["seconds"] = namespace["time_to_day_index"](datetime(1900, 1, 1, 9, 30, 59, 999999), "cn")

range_cases = [
    ("all", "08:30", "14:59", "10min", "cn"),
    ("lunch", "11:30", "12:59", "1min", "cn"),
    ("before", "08:00", "09:00", "1min", "cn"),
    ("reverse", "14:00", "10:00", "1min", "cn"),
    ("cross_7", "11:29", "13:03", "7min", "cn"),
    ("us", "09:30", "15:59", "5min", "us"),
    ("tw", "09:00", "13:29", "5min", "tw"),
    ("unit_ignored", "09:30", "09:31", "day", "cn"),
    ("dated_tz", "2024-01-01 09:30+08:00", "2024-01-01 10:00+08:00", "5min", "cn"),
]
ranges = {name: namespace["get_day_min_idx_range"](start, end, freq, region) for name, start, end, freq, region in range_cases}
ranges["huge"] = namespace["get_day_min_idx_range"]("09:30", "15:00", "1" * 100 + "min", "cn")
try:
    namespace["get_day_min_idx_range"]("09:30", "10:00", "0min", "cn")
except Exception as error:
    zero_error = type(error).__name__
print(json.dumps({"times": times, "ranges": ranges, "zero_error": zero_error}, sort_keys=True))
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

    let expected = json!({
        "times": {
            "cn_open": 0, "cn_am_last": 119, "cn_pm_open": 120, "cn_last": 239,
            "us_last": 389, "tw_last": 269, "seconds": 0,
        },
        "ranges": {
            "all": [0, 23], "lunch": [120, 119], "before": [0, -1],
            "reverse": [180, 30], "cross_7": [17, 17], "us": [0, 77],
            "tw": [0, 53], "unit_ignored": [0, 1], "dated_tz": [0, 6],
            "huge": [0, 0],
        },
        "zero_error": "ValueError",
    });
    assert_eq!(actual, expected);
}
