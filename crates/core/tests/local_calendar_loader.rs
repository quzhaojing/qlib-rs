use chrono::NaiveDateTime;
use domain_core::{
    CachedCalendarProvider, CalendarBackendSource, CalendarCache, CalendarLoadError as Error,
    CalendarLoader, CalendarRows, CalendarTimestampDecoder, CalendarValue, CalendarWarningSink,
    ExecutionCalendar, ExecutionCalendarError, IsoCalendarTimestampDecoder, LocalCalendarLoader,
    TracingCalendarWarnings,
};
use serde_json::{Value, json};
use std::{
    process::Command,
    sync::{Arc, Mutex},
};

#[derive(Default)]
struct Trace {
    events: Vec<Value>,
    warnings: Vec<String>,
}

struct Backend {
    mode: &'static str,
    original_future: bool,
    trace: Arc<Mutex<Trace>>,
    cache: CalendarCache,
}

impl CalendarBackendSource for Backend {
    fn data(&self, frequency: &str, future: bool) -> Result<CalendarRows, Error> {
        self.trace
            .lock()
            .unwrap()
            .events
            .push(json!(["backend", frequency, future]));
        if self.mode == "construct" && future {
            return Err(Error::Value("construct".to_owned()));
        }
        self.trace
            .lock()
            .unwrap()
            .events
            .push(json!(["data", future]));
        if future && matches!(self.mode, "unicode" | "missing") {
            self.cache.get_or_load_raw("future", &mut || {
                Err(if self.mode == "unicode" {
                    Error::Value("invalid encoded bytes".to_owned())
                } else {
                    Error::Other("gone".to_owned())
                })
            })?;
        }
        if self.mode == "retry"
            || (matches!(self.mode, "value" | "warning1" | "warning2")
                && future == self.original_future)
        {
            return Err(Error::Value("backend".to_owned()));
        }
        if self.mode == "other" {
            return Err(Error::Other("backend".to_owned()));
        }
        let mode = self.mode;
        let trace = Arc::clone(&self.trace);
        Ok(Box::new((0..if mode == "empty" { 0 } else { 2 }).map(
            move |index| {
                trace.lock().unwrap().events.push(json!(["next", index]));
                if mode == "lazy" && index == 1 {
                    return Err(Error::Value("lazy".to_owned()));
                }
                Ok(CalendarValue::Text(
                    if mode == "parse" {
                        "bad-date"
                    } else {
                        "2024-01-02"
                    }
                    .to_owned(),
                ))
            },
        )))
    }
}

struct Decoder(Arc<Mutex<Trace>>);
impl CalendarTimestampDecoder for Decoder {
    fn decode(&self, value: CalendarValue) -> Result<NaiveDateTime, Error> {
        let CalendarValue::Text(text) = &value else {
            panic!("source comparison expects text")
        };
        self.0.lock().unwrap().events.push(json!(["decode", text]));
        IsoCalendarTimestampDecoder.decode(value)
    }
}

struct Warnings {
    mode: &'static str,
    trace: Arc<Mutex<Trace>>,
}
impl CalendarWarningSink for Warnings {
    fn warning(&self, message: &str) -> Result<(), Error> {
        let mut trace = self.trace.lock().unwrap();
        trace.warnings.push(message.to_owned());
        let count = trace.warnings.len();
        trace.events.push(json!(["warning", count]));
        if self.mode == format!("warning{count}") {
            return Err(Error::Other("warning".to_owned()));
        }
        Ok(())
    }
}

fn loader(mode: &'static str, future: bool, trace: &Arc<Mutex<Trace>>) -> LocalCalendarLoader {
    LocalCalendarLoader::new(
        Arc::new(Backend {
            mode,
            original_future: future,
            trace: Arc::clone(trace),
            cache: CalendarCache::new(None),
        }),
        Arc::new(Decoder(Arc::clone(trace))),
        Arc::new(Warnings {
            mode,
            trace: Arc::clone(trace),
        }),
    )
}

#[test]
fn local_loader_matches_upstream_acquisition_warnings_iteration_and_decode_boundaries() {
    let output = Command::new("python")
        .args([
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/local_calendar_loader_contract.py"
            ),
            r"D:\code\github\qlib\qlib\data\data.py",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let expected: Value = serde_json::from_slice(&output.stdout).unwrap();
    let mut actual = Vec::new();
    for (mode, future) in [
        ("ok", true),
        ("value", false),
        ("value", true),
        ("other", true),
        ("parse", true),
        ("retry", true),
        ("construct", true),
        ("lazy", true),
        ("empty", true),
        ("warning1", true),
        ("warning2", true),
        ("unicode", true),
        ("missing", true),
    ] {
        let trace = Arc::new(Mutex::new(Trace::default()));
        let loader = loader(mode, future, &trace);
        let (result, error) = match loader.load("1min", future) {
            Ok(timestamps) => (
                Some(
                    timestamps
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>(),
                ),
                None,
            ),
            Err(Error::Value(_)) => (None, Some("Value")),
            Err(Error::Other(_)) => (None, Some("Other")),
        };
        let trace = trace.lock().unwrap();
        actual.push(json!({"mode": mode, "future": future, "events": trace.events, "warnings": trace.warnings, "result": result, "error": error}));
    }
    assert_eq!(json!(actual), expected);
}

#[test]
fn canonical_decoder_preserves_nanoseconds_and_explicitly_rejects_other_policies() {
    let expected =
        NaiveDateTime::parse_from_str("2024-01-02 09:30:01.123456789", "%Y-%m-%d %H:%M:%S%.f")
            .unwrap();
    for text in [
        "2024-01-02 09:30:01.123456789",
        "2024-01-02T09:30:01.123456789",
    ] {
        assert_eq!(
            IsoCalendarTimestampDecoder
                .decode(CalendarValue::Text(text.to_owned()))
                .unwrap(),
            expected
        );
    }
    assert_eq!(
        IsoCalendarTimestampDecoder
            .decode(CalendarValue::Timestamp(expected))
            .unwrap(),
        expected
    );
    for text in ["2024-01-02 09:30", "2024-01-02T09:30"] {
        assert_eq!(
            IsoCalendarTimestampDecoder
                .decode(CalendarValue::Text(text.to_owned()))
                .unwrap()
                .to_string(),
            "2024-01-02 09:30:00"
        );
    }
    for text in ["bad-date", "2024-01-02T09:30:00+08:00", "NaT"] {
        assert!(matches!(
            IsoCalendarTimestampDecoder.decode(CalendarValue::Text(text.to_owned())),
            Err(Error::Value(_))
        ));
    }
}

#[test]
fn loader_trait_preserves_errors_and_composes_with_real_cache_and_cursor() {
    let trace = Arc::new(Mutex::new(Trace::default()));
    let failed = loader("other", true, &trace);
    assert_eq!(
        failed.load_calendar("1min", true).unwrap_err(),
        ExecutionCalendarError::Provider("backend".to_owned())
    );
    let backend = Arc::new(Backend {
        mode: "value",
        original_future: true,
        trace: Arc::clone(&trace),
        cache: CalendarCache::new(None),
    });
    let local = Arc::new(LocalCalendarLoader::new(
        backend,
        Arc::new(IsoCalendarTimestampDecoder),
        Arc::new(TracingCalendarWarnings),
    ));
    assert_eq!(local.load_calendar("1min", true).unwrap().len(), 2);
    let provider = Arc::new(CachedCalendarProvider::new(
        local,
        Arc::new(CalendarCache::new(None)),
    ));
    let timestamp = IsoCalendarTimestampDecoder
        .decode(CalendarValue::Text("2024-01-02".to_owned()))
        .unwrap();
    let calendar = ExecutionCalendar::new(
        provider,
        "1min".to_owned(),
        Some(timestamp),
        Some(timestamp),
    )
    .unwrap();
    assert_eq!(calendar.indices(), (1, 1));
    assert_eq!(calendar.trade_len(), 1);
    assert_eq!(
        calendar.step_time(None, 0),
        Err(ExecutionCalendarError::Index(1))
    );
}
