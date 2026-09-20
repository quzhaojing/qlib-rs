use std::{path::PathBuf, process::Command, sync::Mutex};

use chrono::{NaiveDateTime, NaiveTime};
use domain_core::{
    IdxTradeRange, TradeCalendarRange, TradeCalendarRangeError, TradeRange, TradeRangeByTime,
    TradeRangeError,
};
use serde_json::{Value, json};

fn timestamp(text: &str) -> NaiveDateTime {
    NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S%.f").unwrap()
}

fn clock(text: &str) -> NaiveTime {
    NaiveTime::parse_from_str(text, "%H:%M:%S%.f").unwrap()
}

#[test]
fn parsing_accepts_clock_timestamp_and_offset_forms_and_rejects_invalid_bounds() {
    let direct = TradeRangeByTime::new(clock("09:30:00"), clock("14:30:00")).unwrap();
    assert_eq!(direct.start_time(), clock("09:30:00"));
    assert_eq!(direct.end_time(), clock("14:30:00"));

    for (start, end, expected_start, expected_end) in [
        ("9:30", "14:30", "09:30:00", "14:30:00"),
        (
            "10:15:30.123456",
            "12:00:00.654321",
            "10:15:30.123456",
            "12:00:00.654321",
        ),
        (
            "2024-06-01 10:15",
            "2024-06-01 12:00",
            "10:15:00",
            "12:00:00",
        ),
        (
            "2024-06-01T10:15:30.123456",
            "2024-06-01T12:00:00.654321",
            "10:15:30.123456",
            "12:00:00.654321",
        ),
        (
            "2024-06-01T09:30:00+08:00",
            "2024-06-01T14:30:00+08:00",
            "09:30:00",
            "14:30:00",
        ),
        ("2024-06-01", "01:00", "00:00:00", "01:00:00"),
    ] {
        let parsed = TradeRangeByTime::parse(start, end).unwrap();
        assert_eq!(parsed.start_time(), clock(expected_start));
        assert_eq!(parsed.end_time(), clock(expected_end));
    }

    assert!(matches!(
        TradeRangeByTime::new(clock("10:00:00"), clock("10:00:00")),
        Err(TradeRangeError::InvalidBounds { start, end }) if start == end
    ));
    assert!(matches!(
        TradeRangeByTime::new(clock("11:00:00"), clock("10:00:00")),
        Err(TradeRangeError::InvalidBounds { start, end }) if start > end
    ));
    assert_eq!(
        TradeRangeByTime::parse("bad", "10:00"),
        Err(TradeRangeError::InvalidTime {
            input: "bad".to_owned()
        })
    );
    assert_eq!(
        TradeRangeByTime::parse("10:00", "bad"),
        Err(TradeRangeError::InvalidTime {
            input: "bad".to_owned()
        })
    );
}

struct Calendar {
    start_time: NaiveDateTime,
    result: Result<(i64, i64), TradeCalendarRangeError>,
    calls: Mutex<Vec<(NaiveDateTime, NaiveDateTime)>>,
}

impl TradeCalendarRange for Calendar {
    fn start_time(&self) -> Result<NaiveDateTime, TradeCalendarRangeError> {
        Ok(self.start_time)
    }

    fn get_range_idx(
        &self,
        start_time: NaiveDateTime,
        end_time: NaiveDateTime,
    ) -> Result<(i64, i64), TradeCalendarRangeError> {
        self.calls.lock().unwrap().push((start_time, end_time));
        self.result.clone()
    }
}

#[test]
fn index_resolution_clipping_and_plugin_failures_match_python_boundaries() {
    let idx = IdxTradeRange::new(-2, -5);
    assert_eq!(idx.start_idx(), -2);
    assert_eq!(idx.end_idx(), -5);
    assert_eq!(idx.range_indices(None), Ok((-2, -5)));
    assert!(matches!(
        idx.clip_time_range(
            timestamp("2024-01-02 09:00:00"),
            timestamp("2024-01-02 15:00:00")
        ),
        Err(TradeRangeError::IndexTimeClippingUnsupported)
    ));

    let by_time = TradeRangeByTime::parse("10:45", "14:44").unwrap();
    let successful = Calendar {
        start_time: timestamp("2024-01-02 08:00:00"),
        result: Ok((7, 9)),
        calls: Mutex::new(Vec::new()),
    };
    let ranges: [&dyn TradeRange; 2] = [&idx, &by_time];
    assert_eq!(ranges[0].time_bounds(), None);
    assert_eq!(
        ranges[1].time_bounds(),
        Some((clock("10:45:00"), clock("14:44:00")))
    );
    assert_eq!(ranges[0].range_indices(Some(&successful)), Ok((-2, -5)));
    assert_eq!(ranges[1].range_indices(Some(&successful)), Ok((7, 9)));
    assert_eq!(
        successful.calls.into_inner().unwrap(),
        [(
            timestamp("2024-01-02 10:45:00"),
            timestamp("2024-01-02 14:44:00")
        )]
    );
    assert_eq!(
        by_time.range_indices(None),
        Err(TradeRangeError::MissingCalendar)
    );

    for (start, end, expected_start, expected_end) in [
        (
            "2024-01-02 09:00:00",
            "2024-01-02 15:00:00",
            "2024-01-02 10:45:00",
            "2024-01-02 14:44:00",
        ),
        (
            "2024-01-02 11:00:00.123456",
            "2024-01-02 14:00:00.654321",
            "2024-01-02 11:00:00.123456",
            "2024-01-02 14:00:00.654321",
        ),
        (
            "2024-01-02 08:00:00",
            "2024-01-02 09:00:00",
            "2024-01-02 10:45:00",
            "2024-01-02 09:00:00",
        ),
        (
            "2024-01-02 14:00:00",
            "2024-01-03 10:00:00",
            "2024-01-02 14:00:00",
            "2024-01-02 14:44:00",
        ),
    ] {
        assert_eq!(
            by_time.clip_time_range(timestamp(start), timestamp(end)),
            Ok((timestamp(expected_start), timestamp(expected_end)))
        );
    }

    let failure = TradeCalendarRangeError::Provider {
        message: "offline".to_owned(),
    };
    let failed = Calendar {
        start_time: timestamp("2024-01-02 08:00:00"),
        result: Err(failure.clone()),
        calls: Mutex::new(Vec::new()),
    };
    assert_eq!(
        by_time.range_indices(Some(&failed)),
        Err(TradeRangeError::Calendar(failure))
    );
}

#[test]
fn trade_ranges_match_live_python_source() {
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/backtest/decision.py");
    let script = r"
import ast,json,sys
from abc import abstractmethod
from datetime import time
from typing import *
import pandas as pd
t=ast.parse(open(sys.argv[1],encoding='utf-8').read(),filename=sys.argv[1]);names=['TradeRange','IdxTradeRange','TradeRangeByTime'];nodes=[next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name==x) for x in names]
m=ast.Module(body=nodes,type_ignores=[]);ast.fix_missing_locations(m);TradeCalendarManager=object;concat_date_time=lambda d,t:pd.Timestamp.combine(d,t);exec(compile(m,sys.argv[1],'exec'),globals())
iso=lambda x:x.isoformat();out={}
class Cal:
 def __init__(self):self.start_time=pd.Timestamp('2024-01-02 08:00');self.calls=[]
 def get_range_idx(self,s,e):self.calls.append((s,e));return 7,9
for key,s,e in [('minute','9:30','14:30'),('seconds','10:15:30.123456','12:00:00.654321'),('datetime','2024-06-01 10:15','2024-06-01 12:00'),('offset','2024-06-01T09:30:00+08:00','2024-06-01T14:30:00+08:00')]:
 x=TradeRangeByTime(s,e);c=Cal();out[key]={'times':[x.start_time.isoformat(),x.end_time.isoformat()],'idx':x(c),'call':[iso(v) for v in c.calls[0]]}
x=TradeRangeByTime('10:45','14:44');out['clip_normal']=[iso(v) for v in x.clip_time_range(pd.Timestamp('2024-01-02 09:00'),pd.Timestamp('2024-01-02 15:00'))];out['clip_inner']=[iso(v) for v in x.clip_time_range(pd.Timestamp('2024-01-02 11:00:00.123456'),pd.Timestamp('2024-01-02 14:00:00.654321'))];out['clip_inverted']=[iso(v) for v in x.clip_time_range(pd.Timestamp('2024-01-02 08:00'),pd.Timestamp('2024-01-02 09:00'))];out['clip_cross_day']=[iso(v) for v in x.clip_time_range(pd.Timestamp('2024-01-02 14:00'),pd.Timestamp('2024-01-03 10:00'))]
for key,s,e in [('equal','10:00','10:00'),('reverse','11:00','10:00'),('invalid','bad','10:00')]:
 try:TradeRangeByTime(s,e);out[key]='ok'
 except Exception as error:out[key]=type(error).__name__
try:x(None);out['none']='ok'
except Exception as error:out['none']=type(error).__name__
i=IdxTradeRange(-2,-5);out['idx']=[i(None),i(Cal())]
try:i.clip_time_range(pd.Timestamp('2024-01-01'),pd.Timestamp('2024-01-02'));out['idx_clip']='ok'
except Exception as error:out['idx_clip']=type(error).__name__
print(json.dumps(out,sort_keys=True))
";
    let output = Command::new(std::env::var_os("PYTHON").unwrap_or_else(|| "python".into()))
        .arg("-c")
        .arg(script)
        .arg(source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let actual: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(actual["minute"]["times"], json!(["09:30:00", "14:30:00"]));
    assert_eq!(actual["minute"]["idx"], json!([7, 9]));
    assert_eq!(
        actual["minute"]["call"],
        json!(["2024-01-02T09:30:00", "2024-01-02T14:30:00"])
    );
    assert_eq!(
        actual["seconds"]["times"],
        json!(["10:15:30.123456", "12:00:00.654321"])
    );
    assert_eq!(actual["datetime"]["times"], json!(["10:15:00", "12:00:00"]));
    assert_eq!(actual["offset"]["times"], json!(["09:30:00", "14:30:00"]));
    assert_eq!(
        actual["clip_normal"],
        json!(["2024-01-02T10:45:00", "2024-01-02T14:44:00"])
    );
    assert_eq!(
        actual["clip_inner"],
        json!(["2024-01-02T11:00:00.123456", "2024-01-02T14:00:00.654321"])
    );
    assert_eq!(
        actual["clip_inverted"],
        json!(["2024-01-02T10:45:00", "2024-01-02T09:00:00"])
    );
    assert_eq!(
        actual["clip_cross_day"],
        json!(["2024-01-02T14:00:00", "2024-01-02T14:44:00"])
    );
    assert_eq!(actual["equal"], json!("AssertionError"));
    assert_eq!(actual["reverse"], json!("AssertionError"));
    assert_eq!(actual["invalid"], json!("DateParseError"));
    assert_eq!(actual["none"], json!("NotImplementedError"));
    assert_eq!(actual["idx"], json!([[-2, -5], [-2, -5]]));
    assert_eq!(actual["idx_clip"], json!("NotImplementedError"));
}
