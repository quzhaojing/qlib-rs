use std::{path::PathBuf, process::Command, str::FromStr};

use chrono::NaiveDateTime;
use domain_core::{
    CalendarResampleError, Frequency, MarketCalendarError, MinuteAlignmentError, Region,
    resample_calendar,
};
use num_bigint::BigInt;
use serde_json::{Value, json};

fn timestamp(text: &str) -> NaiveDateTime {
    NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S").expect("fixture timestamp is valid")
}

fn frequency(text: &str) -> Frequency {
    Frequency::from_str(text).expect("fixture frequency is valid")
}

fn sample(values: &[&str], raw: &str, sampled: &str) -> Vec<String> {
    let calendar: Vec<_> = values.iter().map(|value| timestamp(value)).collect();
    resample_calendar(
        &calendar,
        &frequency(raw),
        &frequency(sampled),
        Region::Cn,
        &BigInt::from(0),
    )
    .expect("fixture resampling succeeds")
    .into_iter()
    .map(|value| value.format("%Y-%m-%dT%H:%M:%S").to_string())
    .collect()
}

const CALENDAR: [&str; 7] = [
    "2024-02-05 09:31:00",
    "2024-01-31 15:00:00",
    "2024-02-01 09:30:00",
    "2024-02-05 09:30:00",
    "2024-02-29 15:00:00",
    "2024-03-01 09:30:00",
    "2024-02-01 09:30:00",
];

#[test]
fn minute_resampling_aligns_then_sorts_and_deduplicates() {
    assert_eq!(
        sample(&CALENDAR, "1min", "5min"),
        [
            "2024-01-31T14:55:00",
            "2024-02-01T09:30:00",
            "2024-02-05T09:30:00",
            "2024-02-29T14:55:00",
            "2024-03-01T09:30:00",
        ]
    );

    let shifted = resample_calendar(
        &[timestamp("2024-01-02 09:30:00")],
        &frequency("0min"),
        &frequency("5min"),
        Region::Us,
        &BigInt::from(1),
    )
    .expect("zero raw frequency and explicit shift are supported");
    assert_eq!(shifted, [timestamp("2024-01-02 09:29:00")]);
}

#[test]
fn day_week_and_month_sampling_use_period_first_trading_days() {
    assert_eq!(
        sample(&CALENDAR, "1min", "2day"),
        [
            "2024-01-31T00:00:00",
            "2024-02-05T00:00:00",
            "2024-03-01T00:00:00",
        ]
    );
    assert_eq!(
        sample(&CALENDAR, "9month", "1week"),
        ["2024-01-31T00:00:00", "2024-02-05T00:00:00"]
    );
    assert_eq!(
        sample(&CALENDAR, "9month", "2week"),
        ["2024-01-31T00:00:00"]
    );
    assert_eq!(
        sample(&CALENDAR, "9month", "1month"),
        [
            "2024-01-31T00:00:00",
            "2024-02-01T00:00:00",
            "2024-03-01T00:00:00",
        ]
    );
    assert_eq!(
        sample(&CALENDAR, "9month", "2month"),
        ["2024-01-31T00:00:00", "2024-03-01T00:00:00"]
    );

    let huge = format!("{}day", BigInt::from(10_u8).pow(100));
    assert_eq!(sample(&CALENDAR, "day", &huge), ["2024-01-31T00:00:00"]);
}

#[test]
fn empty_input_short_circuits_and_failures_remain_typed() {
    assert_eq!(
        resample_calendar(
            &[],
            &frequency("day"),
            &frequency("0min"),
            Region::Cn,
            &BigInt::from(0),
        ),
        Ok(Vec::new())
    );

    let value = [timestamp("2024-01-02 09:30:00")];
    assert_eq!(
        resample_calendar(
            &value,
            &frequency("day"),
            &frequency("5min"),
            Region::Cn,
            &BigInt::from(0),
        ),
        Err(CalendarResampleError::MinuteSampleFromNonMinuteRaw)
    );
    assert_eq!(
        resample_calendar(
            &value,
            &frequency("6min"),
            &frequency("5min"),
            Region::Cn,
            &BigInt::from(0),
        ),
        Err(CalendarResampleError::RawFrequencyCoarser)
    );
    assert_eq!(
        resample_calendar(
            &value,
            &frequency("0min"),
            &frequency("0min"),
            Region::Cn,
            &BigInt::from(0),
        ),
        Err(CalendarResampleError::MinuteAlignment(
            MinuteAlignmentError::ZeroSamplingStep
        ))
    );
    for sampled in ["0day", "0week", "0month"] {
        assert_eq!(
            resample_calendar(
                &value,
                &frequency("day"),
                &frequency(sampled),
                Region::Cn,
                &BigInt::from(0),
            ),
            Err(CalendarResampleError::ZeroSamplingStep)
        );
    }

    let shift = BigInt::from(116_906_958_i64);
    assert_eq!(
        resample_calendar(
            &value,
            &frequency("1min"),
            &frequency("5min"),
            Region::Cn,
            &shift,
        ),
        Err(CalendarResampleError::MinuteAlignment(
            MinuteAlignmentError::Calendar(MarketCalendarError::ShiftOutOfRange { shift })
        ))
    );
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "the live-Python differential fixture is kept beside its assertions"
)]
fn calendar_resampling_matches_live_python_source() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/utils");
    let time_source = root.join("time.py");
    let resam_source = root.join("resam.py");
    assert!(time_source.is_file() && resam_source.is_file());

    let script = r#"
import ast, bisect, functools, json, re, sys
from datetime import date, datetime, time, timedelta
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd

def selected(path, names):
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets): body.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names: body.append(node)
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    return compile(module, path, "exec")

class Conf(dict): min_data_shift = 0
ns = {"ast": ast, "bisect": bisect, "functools": functools, "re": re,
      "date": date, "datetime": datetime, "time": time, "timedelta": timedelta,
      "List": List, "Optional": Optional, "Tuple": Tuple, "Union": Union,
      "np": np, "pd": pd, "REG_CN": "cn", "REG_US": "us", "REG_TW": "tw", "C": Conf(region="cn")}
exec(selected(sys.argv[1], {"CN_TIME", "US_TIME", "TW_TIME", "get_min_cal", "Freq", "concat_date_time", "cal_sam_minute"}), ns)
exec(selected(sys.argv[2], {"resam_calendar"}), ns)
calendar = np.array(list(map(pd.Timestamp, [
    "2024-02-05 09:31:00", "2024-01-31 15:00:00", "2024-02-01 09:30:00",
    "2024-02-05 09:30:00", "2024-02-29 15:00:00", "2024-03-01 09:30:00",
    "2024-02-01 09:30:00"])))
result = {}
for sampled in ["5min", "2day", "1week", "2week", "1month", "2month"]:
    values = ns["resam_calendar"](calendar, "1min", sampled, "cn")
    result[sampled] = [value.isoformat() for value in values]
print(json.dumps(result, sort_keys=True))
"#;
    let output = Command::new(std::env::var_os("PYTHON").unwrap_or_else(|| "python".into()))
        .arg("-c")
        .arg(script)
        .arg(&time_source)
        .arg(&resam_source)
        .output()
        .expect("Python interpreter starts");
    assert!(
        output.status.success(),
        "Python snapshot failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let actual: Value = serde_json::from_slice(&output.stdout).expect("valid Python JSON");
    let expected = json!({
        "5min": sample(&CALENDAR, "1min", "5min"),
        "2day": sample(&CALENDAR, "1min", "2day"),
        "1week": sample(&CALENDAR, "1min", "1week"),
        "2week": sample(&CALENDAR, "1min", "2week"),
        "1month": sample(&CALENDAR, "1min", "1month"),
        "2month": sample(&CALENDAR, "1min", "2month"),
    });
    assert_eq!(actual, expected);
}
