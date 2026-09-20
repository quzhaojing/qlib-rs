use std::{path::PathBuf, process::Command};

use chrono::{NaiveTime, TimeDelta};
use domain_core::{Region, is_single_market_value};
use serde_json::{Value, json};

fn clock(hour: u32, minute: u32, second: u32) -> NaiveTime {
    NaiveTime::from_hms_opt(hour, minute, second).expect("fixture clock time is valid")
}

#[test]
fn duration_boundaries_and_every_region_closing_match_python() {
    let minute = TimeDelta::minutes(1);
    let hour = TimeDelta::hours(1);
    let cases = [
        (
            clock(10, 0, 0),
            TimeDelta::nanoseconds(59_999_999_999),
            minute,
            Region::Cn,
            true,
        ),
        (clock(10, 0, 0), minute, minute, Region::Cn, false),
        (
            clock(10, 0, 0),
            TimeDelta::seconds(61),
            minute,
            Region::Cn,
            false,
        ),
        (
            clock(10, 0, 0),
            TimeDelta::seconds(-1),
            minute,
            Region::Cn,
            true,
        ),
        (clock(10, 0, 0), TimeDelta::zero(), minute, Region::Cn, true),
        (clock(11, 29, 0), hour, minute, Region::Cn, true),
        (clock(11, 28, 0), hour, minute, Region::Cn, false),
        (clock(14, 59, 0), hour, minute, Region::Cn, true),
        (clock(14, 58, 0), hour, minute, Region::Cn, false),
        (clock(11, 29, 1), hour, minute, Region::Cn, false),
        (clock(13, 24, 0), hour, minute, Region::Tw, false),
        (clock(13, 25, 0), hour, minute, Region::Tw, true),
        (clock(13, 59, 0), hour, minute, Region::Tw, true),
        (clock(13, 59, 1), hour, minute, Region::Tw, false),
        (clock(14, 0, 0), hour, minute, Region::Tw, false),
        (clock(15, 59, 0), hour, minute, Region::Us, true),
        (clock(15, 58, 0), hour, minute, Region::Us, false),
        (clock(14, 59, 0), hour, minute, Region::Us, false),
        (clock(15, 59, 1), hour, minute, Region::Us, false),
        (
            clock(10, 0, 0),
            TimeDelta::nanoseconds(-1),
            TimeDelta::zero(),
            Region::Cn,
            true,
        ),
        (
            clock(10, 0, 0),
            TimeDelta::zero(),
            TimeDelta::zero(),
            Region::Cn,
            false,
        ),
        (
            clock(10, 0, 0),
            TimeDelta::minutes(-2),
            TimeDelta::minutes(-1),
            Region::Cn,
            true,
        ),
        (
            clock(10, 0, 0),
            TimeDelta::minutes(-1),
            TimeDelta::minutes(-1),
            Region::Cn,
            false,
        ),
    ];
    for (start, elapsed, frequency, region, expected) in cases {
        assert_eq!(
            is_single_market_value(start, elapsed, frequency, region),
            expected
        );
    }

    let subsecond_close =
        NaiveTime::from_hms_micro_opt(11, 29, 0, 999_999).expect("fixture clock time is valid");
    assert!(is_single_market_value(
        subsecond_close,
        hour,
        minute,
        Region::Cn
    ));
    let subsecond_us =
        NaiveTime::from_hms_milli_opt(15, 59, 0, 500).expect("fixture clock time is valid");
    assert!(is_single_market_value(
        subsecond_us,
        hour,
        minute,
        Region::Us
    ));
}

#[test]
fn rust_predicate_matches_a_live_python_snapshot() {
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
import sys
import pandas as pd

tree = ast.parse(open(sys.argv[1], encoding="utf-8").read(), filename=sys.argv[1])
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "is_single_value")
module = ast.Module(body=[function], type_ignores=[])
ast.fix_missing_locations(module)
namespace = {"REG_CN": "cn", "REG_US": "us", "REG_TW": "tw"}
exec(compile(module, sys.argv[1], "exec"), namespace)
predicate = namespace["is_single_value"]

cases = [
    ("cn_lt", "2021-01-01 10:00:00", "59.999999999s", "1min", "cn"),
    ("cn_eq", "2021-01-01 10:00:00", "1min", "1min", "cn"),
    ("cn_gt", "2021-01-01 10:00:00", "61s", "1min", "cn"),
    ("cn_negative", "2021-01-01 10:00:00", "-1s", "1min", "cn"),
    ("cn_am_close", "2021-01-01 11:29:00", "1h", "1min", "cn"),
    ("cn_am_close_micro", "2021-01-01 11:29:00.999999", "1h", "1min", "cn"),
    ("cn_am_close_second", "2021-01-01 11:29:01", "1h", "1min", "cn"),
    ("cn_pm_close", "2021-01-01 14:59:00", "1h", "1min", "cn"),
    ("tw_before", "2021-01-01 13:24:00", "1h", "1min", "tw"),
    ("tw_close", "2021-01-01 13:25:00", "1h", "1min", "tw"),
    ("tw_late", "2021-01-01 13:59:00", "1h", "1min", "tw"),
    ("tw_second", "2021-01-01 13:59:01", "1h", "1min", "tw"),
    ("us_close", "2021-01-01 15:59:00", "1h", "1min", "us"),
    ("us_close_micro", "2021-01-01 15:59:00.500", "1h", "1min", "us"),
    ("us_second", "2021-01-01 15:59:01", "1h", "1min", "us"),
    ("zero_negative", "2021-01-01 10:00:00", "-1ns", "0s", "cn"),
    ("zero_equal", "2021-01-01 10:00:00", "0s", "0s", "cn"),
    ("negative_lt", "2021-01-01 10:00:00", "-2min", "-1min", "cn"),
    ("negative_eq", "2021-01-01 10:00:00", "-1min", "-1min", "cn"),
    ("aware_equal", "2021-03-14 01:59:30-05:00", "1min", "1min", "cn"),
]
result = {}
for name, start_text, elapsed_text, frequency_text, region in cases:
    start = pd.Timestamp(start_text)
    result[name] = predicate(start, start + pd.Timedelta(elapsed_text), pd.Timedelta(frequency_text), region)
result["nat_start"] = predicate(pd.NaT, pd.Timestamp("2021-01-01"), pd.Timedelta("1min"), "cn")
result["nat_frequency"] = predicate(pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-01 00:00:01"), pd.NaT, "cn")
print(json.dumps(result, sort_keys=True))
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
        "cn_lt": true, "cn_eq": false, "cn_gt": false, "cn_negative": true,
        "cn_am_close": true, "cn_am_close_micro": true, "cn_am_close_second": false,
        "cn_pm_close": true, "tw_before": false, "tw_close": true, "tw_late": true,
        "tw_second": false, "us_close": true, "us_close_micro": true,
        "us_second": false, "zero_negative": true, "zero_equal": false,
        "negative_lt": true, "negative_eq": false, "aware_equal": false,
        "nat_start": false, "nat_frequency": false,
    });
    assert_eq!(actual, expected);
}
