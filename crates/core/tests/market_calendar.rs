use std::{
    path::PathBuf,
    process::Command,
    sync::{Arc, LazyLock, Mutex},
};

use chrono::NaiveTime;
use domain_core::{
    CN_SESSIONS, MINUTE_CALENDAR_CACHE_CAPACITY, MarketCalendarError, Region, TW_SESSIONS,
    TradingSession, US_SESSIONS, minute_calendar,
};
use num_bigint::BigInt;
use serde_json::{Value, json};

static TEST_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

fn clock(hour: u32, minute: u32) -> NaiveTime {
    NaiveTime::from_hms_opt(hour, minute, 0).expect("fixture clock time is valid")
}

#[test]
fn sessions_calendars_boundaries_and_lru_match_the_contract() {
    let _guard = TEST_LOCK.lock().expect("calendar test lock is healthy");
    assert_eq!(
        CN_SESSIONS,
        [
            TradingSession {
                start: clock(9, 30),
                end: clock(11, 30),
            },
            TradingSession {
                start: clock(13, 0),
                end: clock(15, 0),
            },
        ]
    );
    assert_eq!(
        US_SESSIONS,
        [TradingSession {
            start: clock(9, 30),
            end: clock(16, 0),
        }]
    );
    assert_eq!(
        TW_SESSIONS,
        [TradingSession {
            start: clock(9, 0),
            end: clock(13, 30),
        }]
    );
    assert_eq!(Region::Cn.trading_sessions(), &CN_SESSIONS);
    assert_eq!(Region::Us.trading_sessions(), &US_SESSIONS);
    assert_eq!(Region::Tw.trading_sessions(), &TW_SESSIONS);

    for (region, length, first, middle, last) in [
        (Region::Cn, 240, clock(9, 30), clock(13, 0), clock(14, 59)),
        (Region::Us, 390, clock(9, 30), clock(12, 45), clock(15, 59)),
        (Region::Tw, 270, clock(9, 0), clock(11, 15), clock(13, 29)),
    ] {
        let calendar = minute_calendar(&BigInt::from(0), region).expect("zero shift is valid");
        assert_eq!(calendar.len(), length);
        assert_eq!(calendar[0], first);
        assert_eq!(calendar[length / 2], middle);
        assert_eq!(calendar[length - 1], last);
    }

    let shifted = minute_calendar(&BigInt::from(600), Region::Cn).expect("shift is valid");
    assert_eq!(shifted[0], clock(23, 30));
    assert_eq!(shifted[120], clock(3, 0));
    assert_eq!(shifted[239], clock(4, 59));
    let forward = minute_calendar(&BigInt::from(-600), Region::Cn).expect("shift is valid");
    assert_eq!(forward[0], clock(19, 30));
    assert_eq!(forward[239], clock(0, 59));
    assert_eq!(
        minute_calendar(&BigInt::from(1_440), Region::Cn).expect("one-day shift is valid"),
        minute_calendar(&BigInt::from(0), Region::Cn).expect("zero shift is valid")
    );

    for (region, accepted, rejected) in [
        (Region::Cn, 116_906_957_i64, 116_906_958_i64),
        (Region::Us, 116_906_957_i64, 116_906_958_i64),
        (Region::Tw, 116_906_927_i64, 116_906_928_i64),
    ] {
        assert!(minute_calendar(&BigInt::from(accepted), region).is_ok());
        let rejected = BigInt::from(rejected);
        assert_eq!(
            minute_calendar(&rejected, region),
            Err(MarketCalendarError::ShiftOutOfRange { shift: rejected })
        );
    }
    assert!(minute_calendar(&BigInt::from(-153_722_867_i64), Region::Cn).is_ok());
    for rejected in [
        BigInt::from(-153_722_868_i64),
        BigInt::from(i64::MAX) + BigInt::from(1),
    ] {
        assert_eq!(
            minute_calendar(&rejected, Region::Cn),
            Err(MarketCalendarError::ShiftOutOfRange {
                shift: rejected.clone()
            })
        );
    }

    assert_eq!(MINUTE_CALENDAR_CACHE_CAPACITY, 240);
    let cached = minute_calendar(&BigInt::from(7), Region::Cn).expect("shift is valid");
    let cached_again = minute_calendar(&BigInt::from(7), Region::Cn).expect("shift is valid");
    assert!(Arc::ptr_eq(&cached, &cached_again));
    for shift in 10_000..10_000 + MINUTE_CALENDAR_CACHE_CAPACITY {
        minute_calendar(&BigInt::from(shift), Region::Cn).expect("fixture shift is valid");
    }
    let rebuilt = minute_calendar(&BigInt::from(7), Region::Cn).expect("shift is valid");
    assert!(!Arc::ptr_eq(&cached, &rebuilt));
}

#[test]
fn rust_calendars_match_a_live_python_snapshot() {
    let _guard = TEST_LOCK.lock().expect("calendar test lock is healthy");
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
import functools
import json
import sys
from datetime import date, datetime, time, timedelta
import pandas as pd

tree = ast.parse(open(sys.argv[1], encoding="utf-8").read(), filename=sys.argv[1])
names = {"CN_TIME", "US_TIME", "TW_TIME", "get_min_cal"}
body = []
for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in names for target in node.targets):
        body.append(node)
    elif isinstance(node, ast.FunctionDef) and node.name in names:
        body.append(node)
module = ast.Module(body=body, type_ignores=[])
ast.fix_missing_locations(module)
namespace = {
    "date": date, "datetime": datetime, "time": time, "timedelta": timedelta,
    "functools": functools, "pd": pd, "List": list,
    "REG_CN": "cn", "REG_US": "us", "REG_TW": "tw",
}
exec(compile(module, sys.argv[1], "exec"), namespace)

def describe(calendar):
    return {
        "length": len(calendar),
        "first": str(calendar[0]),
        "middle": str(calendar[len(calendar) // 2]),
        "last": str(calendar[-1]),
        "sorted": calendar == sorted(calendar),
    }

shifts = [0, 1, -1, 600, -600, 1440]
calendars = {
    region: {str(shift): describe(namespace["get_min_cal"](shift, region)) for shift in shifts}
    for region in ["cn", "us", "tw"]
}
sessions = {
    region: [value.strftime("%H:%M:%S") for value in namespace[name]]
    for region, name in [("cn", "CN_TIME"), ("us", "US_TIME"), ("tw", "TW_TIME")]
}
errors = []
for shift, region in [(116906958, "cn"), (116906928, "tw"), (-153722868, "us")]:
    try:
        namespace["get_min_cal"](shift, region)
    except Exception as error:
        errors.append(type(error).__name__)
print(json.dumps({"calendars": calendars, "sessions": sessions, "errors": errors}, sort_keys=True))
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

    let mut calendars = serde_json::Map::new();
    for region in [Region::Cn, Region::Us, Region::Tw] {
        let mut shifts = serde_json::Map::new();
        for shift in [0, 1, -1, 600, -600, 1_440] {
            let calendar = minute_calendar(&BigInt::from(shift), region).expect("shift is valid");
            shifts.insert(
                shift.to_string(),
                json!({
                    "length": calendar.len(),
                    "first": calendar[0].to_string(),
                    "middle": calendar[calendar.len() / 2].to_string(),
                    "last": calendar[calendar.len() - 1].to_string(),
                    "sorted": calendar.windows(2).all(|pair| pair[0] <= pair[1]),
                }),
            );
        }
        calendars.insert(region.code().to_owned(), Value::Object(shifts));
    }
    let expected = json!({
        "calendars": calendars,
        "sessions": {
            "cn": ["09:30:00", "11:30:00", "13:00:00", "15:00:00"],
            "us": ["09:30:00", "16:00:00"],
            "tw": ["09:00:00", "13:30:00"],
        },
        "errors": ["OverflowError", "OverflowError", "OutOfBoundsTimedelta"],
    });
    assert_eq!(actual, expected);
}
