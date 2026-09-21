use std::{path::PathBuf, process::Command};

use chrono::{NaiveDate, NaiveDateTime};
use domain_core::{
    MarketCalendarError, MinuteAlignmentError, Region, align_sampled_minute,
    regular_minute_calendar,
};
use num_bigint::BigInt;
use serde_json::{Value, json};

fn timestamp(text: &str) -> NaiveDateTime {
    NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S%.f").expect("fixture timestamp is valid")
}

fn iso(value: NaiveDateTime) -> String {
    value.format("%Y-%m-%dT%H:%M:%S%.f").to_string()
}

#[test]
fn calendar_failure_precedes_zero_sampling_step() {
    let shift = BigInt::from(10_u8).pow(100);
    for region in [Region::Cn, Region::Us, Region::Tw] {
        assert_eq!(
            align_sampled_minute(
                timestamp("2021-01-01 10:38:00"),
                &BigInt::from(0),
                &shift,
                region,
            ),
            Err(MinuteAlignmentError::Calendar(
                MarketCalendarError::ShiftOutOfRange {
                    shift: shift.clone()
                }
            ))
        );
    }
}

fn align(text: &str, sample: i64, shift: i64, region: Region) -> NaiveDateTime {
    align_sampled_minute(
        timestamp(text),
        &BigInt::from(sample),
        &BigInt::from(shift),
        region,
    )
    .expect("fixture alignment succeeds")
}

#[test]
fn alignment_preserves_python_boundaries_and_unusual_slice_steps() {
    let cases = [
        (
            "2021-03-03 10:38:45.123456",
            5,
            0,
            Region::Cn,
            "2021-03-03T10:35:00",
        ),
        (
            "2021-03-03 10:38:00",
            10,
            0,
            Region::Cn,
            "2021-03-03T10:30:00",
        ),
        (
            "2021-03-03 11:29:59",
            5,
            0,
            Region::Cn,
            "2021-03-03T11:25:00",
        ),
        (
            "2021-03-03 11:30:00",
            5,
            0,
            Region::Cn,
            "2021-03-03T11:25:00",
        ),
        (
            "2021-03-03 12:30:00",
            5,
            0,
            Region::Cn,
            "2021-03-03T11:25:00",
        ),
        (
            "2021-03-03 13:03:00",
            7,
            0,
            Region::Cn,
            "2021-03-03T11:29:00",
        ),
        (
            "2021-03-03 08:00:00",
            5,
            0,
            Region::Cn,
            "2021-03-03T14:55:00",
        ),
        (
            "2021-03-03 16:00:00",
            5,
            0,
            Region::Cn,
            "2021-03-03T14:55:00",
        ),
        (
            "2021-03-03 15:59:00",
            6,
            0,
            Region::Us,
            "2021-03-03T15:54:00",
        ),
        (
            "2021-03-03 13:29:00",
            6,
            0,
            Region::Tw,
            "2021-03-03T13:24:00",
        ),
    ];
    for (input, sample, shift, region, expected) in cases {
        assert_eq!(iso(align(input, sample, shift, region)), expected);
    }
}

#[test]
fn alignment_preserves_unusual_slice_steps_and_shifts() {
    let cases = [
        (
            "2021-03-03 14:00:00",
            1_000,
            0,
            Region::Cn,
            "2021-03-03T09:30:00",
        ),
        (
            "2021-03-03 10:00:00",
            -5,
            0,
            Region::Cn,
            "2021-03-03T09:34:00",
        ),
        (
            "2021-03-03 08:00:00",
            -5,
            0,
            Region::Cn,
            "2021-03-03T09:34:00",
        ),
        (
            "2021-03-03 16:00:00",
            -5,
            0,
            Region::Cn,
            "2021-03-03T09:34:00",
        ),
        (
            "2021-03-03 09:30:00",
            5,
            1,
            Region::Cn,
            "2021-03-03T09:29:00",
        ),
        (
            "2021-03-03 09:30:00",
            5,
            -1,
            Region::Cn,
            "2021-03-03T14:56:00",
        ),
        (
            "2021-03-03 10:00:00",
            5,
            600,
            Region::Cn,
            "2021-03-03T04:55:00",
        ),
        (
            "2021-03-03 10:00:00",
            5,
            -600,
            Region::Cn,
            "2021-03-03T00:55:00",
        ),
        (
            "2000-02-29 09:30:00",
            1,
            0,
            Region::Cn,
            "2000-02-29T09:30:00",
        ),
    ];
    for (input, sample, shift, region, expected) in cases {
        assert_eq!(iso(align(input, sample, shift, region)), expected);
    }

    let huge = BigInt::from(10_u8).pow(100);
    assert_eq!(
        iso(align_sampled_minute(
            timestamp("2021-03-03 10:00:00"),
            &huge,
            &BigInt::from(0),
            Region::Cn
        )
        .expect("huge positive stride succeeds")),
        "2021-03-03T09:30:00"
    );
    assert_eq!(
        iso(align_sampled_minute(
            timestamp("2021-03-03 10:00:00"),
            &-huge,
            &BigInt::from(0),
            Region::Cn
        )
        .expect("huge negative stride succeeds")),
        "2021-03-03T14:59:00"
    );
}

#[test]
fn alignment_reports_zero_stride_and_calendar_shift_failures() {
    let value = timestamp("2021-03-03 10:00:00");
    assert_eq!(
        align_sampled_minute(value, &BigInt::from(0), &BigInt::from(0), Region::Cn),
        Err(MinuteAlignmentError::ZeroSamplingStep)
    );
    let shift = BigInt::from(116_906_958_i64);
    assert_eq!(
        align_sampled_minute(value, &BigInt::from(5), &shift, Region::Cn),
        Err(MinuteAlignmentError::Calendar(
            MarketCalendarError::ShiftOutOfRange { shift }
        ))
    );
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "the embedded live-Python differential fixture is clearer kept beside its assertions"
)]
fn every_upstream_calendar_case_matches_a_live_python_snapshot() {
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
import sys
from datetime import date, datetime, time, timedelta
from types import SimpleNamespace
import pandas as pd

names = {"CN_TIME", "US_TIME", "TW_TIME", "get_min_cal", "concat_date_time", "cal_sam_minute"}
tree = ast.parse(open(sys.argv[1], encoding="utf-8").read(), filename=sys.argv[1])
body = []
for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in names for target in node.targets):
        body.append(node)
    elif isinstance(node, ast.FunctionDef) and node.name in names:
        body.append(node)
module = ast.Module(body=body, type_ignores=[])
ast.fix_missing_locations(module)
C = SimpleNamespace(min_data_shift=0)
namespace = {
    "bisect": bisect, "date": date, "datetime": datetime, "functools": functools,
    "pd": pd, "time": time, "timedelta": timedelta, "List": list, "C": C,
    "REG_CN": "cn", "REG_US": "us", "REG_TW": "tw",
}
exec(compile(module, sys.argv[1], "exec"), namespace)

bulk = []
for region in ["cn", "us", "tw"]:
    for sample in range(1, 7):
        for clock in namespace["get_min_cal"](region=region):
            value = pd.Timestamp(datetime(2021, 3, 3, clock.hour, clock.minute))
            bulk.append(namespace["cal_sam_minute"](value, sample, region).isoformat())

edges = [
    ("cn_1038_5", "2021-03-03 10:38:45.123456", 5, 0, "cn"),
    ("cn_lunch", "2021-03-03 12:30", 5, 0, "cn"),
    ("cn_before", "2021-03-03 08:00", 5, 0, "cn"),
    ("cn_pm_7", "2021-03-03 13:03", 7, 0, "cn"),
    ("negative", "2021-03-03 10:00", -5, 0, "cn"),
    ("shift_1", "2021-03-03 09:30", 5, 1, "cn"),
    ("shift_neg_1", "2021-03-03 09:30", 5, -1, "cn"),
    ("shift_600", "2021-03-03 10:00", 5, 600, "cn"),
    ("shift_neg_600", "2021-03-03 10:00", 5, -600, "cn"),
    ("tz_projection", "2021-03-03 10:38+08:00", 5, 0, "cn"),
]
edge_values = {}
for name, text, sample, shift, region in edges:
    C.min_data_shift = shift
    edge_values[name] = namespace["cal_sam_minute"](pd.Timestamp(text), sample, region).isoformat()
print(json.dumps({"bulk": bulk, "edges": edge_values}, sort_keys=True))
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

    let date = NaiveDate::from_ymd_opt(2021, 3, 3).expect("fixture date is valid");
    let mut bulk = Vec::new();
    for region in [Region::Cn, Region::Us, Region::Tw] {
        for sample in 1..=6 {
            for clock in regular_minute_calendar(region).iter().copied() {
                bulk.push(iso(align_sampled_minute(
                    date.and_time(clock),
                    &BigInt::from(sample),
                    &BigInt::from(0),
                    region,
                )
                .expect("upstream calendar fixture aligns")));
            }
        }
    }
    let expected = json!({
        "bulk": bulk,
        "edges": {
            "cn_1038_5": iso(align("2021-03-03 10:38:45.123456", 5, 0, Region::Cn)),
            "cn_lunch": iso(align("2021-03-03 12:30:00", 5, 0, Region::Cn)),
            "cn_before": iso(align("2021-03-03 08:00:00", 5, 0, Region::Cn)),
            "cn_pm_7": iso(align("2021-03-03 13:03:00", 7, 0, Region::Cn)),
            "negative": iso(align("2021-03-03 10:00:00", -5, 0, Region::Cn)),
            "shift_1": iso(align("2021-03-03 09:30:00", 5, 1, Region::Cn)),
            "shift_neg_1": iso(align("2021-03-03 09:30:00", 5, -1, Region::Cn)),
            "shift_600": iso(align("2021-03-03 10:00:00", 5, 600, Region::Cn)),
            "shift_neg_600": iso(align("2021-03-03 10:00:00", 5, -600, Region::Cn)),
            "tz_projection": iso(align("2021-03-03 10:38:00", 5, 0, Region::Cn)),
        },
    });
    assert_eq!(actual, expected);
}
