use chrono::{NaiveDateTime, TimeDelta};
use domain_core::{
    ExecutionCalendar, ExecutionCalendarContext, ExecutionCalendarError as Error,
    ExecutionCalendarProvider,
};
use domain_core::{
    ExecutorLifecycleCalendar, NestedCalendar, ResettableNestedCalendar, SaoeCalendar,
    SaoeDecisionCalendar, SharedExecutionCalendar, SimulatorCalendar, TradeCalendarRange,
    TradeCalendarRangeError, TradeRange, TradeRangeByTime, TradeRangeError,
};
use serde_json::{Value, json};
use std::{
    process::Command,
    sync::{Arc, Mutex},
};

fn time(seconds: i64) -> NaiveDateTime {
    NaiveDateTime::parse_from_str("2024-01-02 09:30:00", "%Y-%m-%d %H:%M:%S").unwrap()
        + TimeDelta::seconds(seconds)
}

struct Source {
    values: Vec<NaiveDateTime>,
    calls: Vec<String>,
    fail: Option<usize>,
    indices: Option<(i64, i64)>,
    scripted_indices: std::collections::VecDeque<(i64, i64)>,
}

impl Source {
    fn record(&mut self, event: String) -> Result<(), Error> {
        self.calls.push(event);
        if self.fail == Some(self.calls.len()) {
            return Err(Error::Provider("injected".to_owned()));
        }
        Ok(())
    }
}

struct Provider(Arc<Mutex<Source>>);
impl ExecutionCalendarProvider for Provider {
    fn calendar(&self, frequency: &str, future: bool) -> Result<Arc<[NaiveDateTime]>, Error> {
        let mut source = self.0.lock().unwrap();
        source.record(format!("load:{frequency}:{future}"))?;
        Ok(source.values.clone().into())
    }
    fn locate_index(
        &self,
        start: Option<NaiveDateTime>,
        end: Option<NaiveDateTime>,
        frequency: &str,
        future: bool,
    ) -> Result<(i64, i64), Error> {
        let mut source = self.0.lock().unwrap();
        source.record(format!("locate:{frequency}:{future}"))?;
        if let Some(indices) = source.scripted_indices.pop_front().or(source.indices) {
            return Ok(indices);
        }
        let start = start.map_or(0, |start| {
            source.values.partition_point(|value| *value < start)
        });
        let end = end.map_or(source.values.len(), |end| {
            source.values.partition_point(|value| *value <= end)
        });
        if start >= source.values.len() {
            return Err(Error::Provider("future start".to_owned()));
        }
        Ok((
            i64::try_from(start).unwrap(),
            i64::try_from(if end == 0 {
                source.values.len() - 1
            } else {
                end - 1
            })
            .unwrap(),
        ))
    }
}

struct Context(Result<String, Error>);
impl ExecutionCalendarContext for Context {
    fn data_frequency(&self) -> Result<String, Error> {
        self.0.clone()
    }
}

fn source() -> Arc<Mutex<Source>> {
    Arc::new(Mutex::new(Source {
        values: (0..5).map(|index| time(index * 60)).collect(),
        calls: Vec::new(),
        fail: None,
        indices: None,
        scripted_indices: std::collections::VecDeque::new(),
    }))
}

fn calendar(source: &Arc<Mutex<Source>>, start: i64, end: i64) -> ExecutionCalendar {
    ExecutionCalendar::new(
        Arc::new(Provider(Arc::clone(source))),
        "1min".to_owned(),
        Some(time(start)),
        Some(time(end)),
    )
    .unwrap()
}

#[test]
fn execution_window_matches_actual_upstream_manager_and_locator() {
    let output = Command::new("python")
        .args([
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/execution_calendar_contract.py"
            ),
            r"D:\code\github\qlib\qlib",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let oracle: Value = serde_json::from_slice(&output.stdout).unwrap();
    let mut rows = Vec::new();
    for (start, end) in [(0, 120), (30, 150), (180, 60), (-60, -60), (240, 240)] {
        let source = source();
        let mut calendar = calendar(&source, start, end);
        assert_eq!(calendar.frequency(), "1min");
        assert_eq!(calendar.all_time(), (Some(time(start)), Some(time(end))));
        let intervals: Vec<_> = [0, 1, -1, 6, -6]
            .into_iter()
            .map(|shift| match calendar.step_time(None, shift) {
                Ok((start, end)) => json!([start.to_string(), end.to_string()]),
                Err(Error::Index(_)) => json!("IndexError"),
                Err(error) => panic!("unexpected error: {error}"),
            })
            .collect();
        while !calendar.finished() {
            calendar.step().unwrap();
        }
        assert_eq!(calendar.step(), Err(Error::Finished));
        let indices = calendar.indices();
        let shared = SharedExecutionCalendar::new(
            Arc::new(Mutex::new(calendar)),
            Arc::new(Context(Err(Error::Provider(
                "must not query exchange".to_owned(),
            )))),
        );
        let time_range = TradeRangeByTime::parse("09:31", "09:33")
            .unwrap()
            .range_indices(Some(&shared))
            .unwrap();
        let calendar = shared.clone();
        // Inspect through the public execution interfaces after range evaluation.
        rows.push(
            json!({"indices": indices, "length": NestedCalendar::trade_len(&calendar).unwrap(),
            "range": calendar.get_range_idx(time(-60), time(600)).unwrap(), "intervals": intervals,
            "steps": NestedCalendar::trade_step(&calendar).unwrap(), "finished": NestedCalendar::finished(&calendar).unwrap(),
            "time_range": time_range}),
        );
        assert_eq!(
            source.lock().unwrap().calls,
            ["load:1min:true", "locate:1min:true"]
        );
    }
    assert_eq!(json!(rows), oracle);
}

#[test]
fn empty_missing_extreme_and_exhausted_calendar_errors_remain_explicit() {
    let source = source();
    let provider: Arc<dyn ExecutionCalendarProvider> = Arc::new(Provider(Arc::clone(&source)));
    source.lock().unwrap().fail = Some(1);
    assert!(matches!(
        ExecutionCalendar::new(Arc::clone(&provider), "1min".to_owned(), None, None),
        Err(Error::Provider(_))
    ));
    source.lock().unwrap().fail = None;
    let mut calendar =
        ExecutionCalendar::new(Arc::clone(&provider), "1min".to_owned(), None, None).unwrap();
    let context = Context(Ok("1min".to_owned()));
    assert_eq!(
        calendar.data_range("full", &context),
        Err(Error::MissingStart)
    );
    assert_eq!(
        calendar.step_time(Some(i64::MIN), i64::MAX),
        Err(Error::Index(i128::from(i64::MIN) - i128::from(i64::MAX)))
    );
    {
        let mut state = source.lock().unwrap();
        state.values.clear();
        state.indices = Some((0, -1));
    }
    calendar
        .reset("1min".to_owned(), Some(time(0)), Some(time(1)))
        .unwrap();
    assert!(calendar.finished());
    assert_eq!(calendar.step(), Err(Error::Finished));
    assert_eq!(calendar.step_time(None, 0), Err(Error::Index(0)));
    assert_eq!(calendar.range_indices(time(0), time(1)).unwrap(), (-1, -1));
    assert_eq!(calendar.data_range("step", &context), Err(Error::Index(0)));
    source.lock().unwrap().indices = Some((0, 0));
    calendar
        .reset("1min".to_owned(), Some(NaiveDateTime::MAX), None)
        .unwrap();
    assert_eq!(
        calendar.data_range("full", &context),
        Err(Error::DayOverflow)
    );
    calendar
        .reset("1min".to_owned(), Some(NaiveDateTime::MIN), None)
        .unwrap();
    assert!(matches!(
        calendar.data_range("full", &context),
        Err(Error::Epsilon(_))
    ));
    source.lock().unwrap().values = vec![time(0), NaiveDateTime::MIN];
    calendar
        .reset("1min".to_owned(), Some(time(0)), None)
        .unwrap();
    assert!(matches!(
        calendar.step_time(None, 0),
        Err(Error::Epsilon(_))
    ));
    assert_eq!(
        Error::Finished.to_string(),
        "The calendar is finished, please reset it if you want to call it!"
    );
}

#[test]
fn native_index_limits_do_not_wrap_or_hide_partial_reset_writes() {
    let source = source();
    let mut calendar = calendar(&source, 0, 120);
    calendar.step().unwrap();
    source.lock().unwrap().indices = Some((i64::MIN, i64::MAX));
    assert_eq!(
        calendar.reset("1min".to_owned(), Some(time(0)), None),
        Err(Error::IndexOverflow)
    );
    assert_eq!(calendar.indices(), (i64::MIN, i64::MAX));
    assert_eq!(calendar.trade_len(), 3);
    assert_eq!(calendar.trade_step(), 1);
    source.lock().unwrap().indices = Some((1, i64::MIN));
    calendar
        .reset("1min".to_owned(), Some(time(0)), None)
        .unwrap();
    assert_eq!(calendar.trade_len(), i64::MIN);
    assert_eq!(
        calendar.range_indices(time(0), time(0)),
        Err(Error::IndexOverflow)
    );
    source.lock().unwrap().indices = Some((i64::MIN, i64::MIN + 2));
    calendar
        .reset("1min".to_owned(), Some(time(0)), None)
        .unwrap();
    assert_eq!(calendar.range_indices(time(0), time(120)).unwrap(), (2, 2));
    let context = Context(Ok("1min".to_owned()));
    for indices in [(i64::MAX, i64::MIN), (i64::MIN, i64::MAX)] {
        source.lock().unwrap().scripted_indices = [(i64::MIN, 0), indices].into();
        assert_eq!(
            calendar.data_range("full", &context),
            Err(Error::IndexOverflow)
        );
    }
}

#[test]
fn reset_failures_keep_reached_metadata_array_and_cursor_writes() {
    for failed_call in [1, 2] {
        let source = source();
        let mut calendar = calendar(&source, 0, 120);
        calendar.step().unwrap();
        {
            let mut state = source.lock().unwrap();
            state.calls.clear();
            state.fail = Some(failed_call);
            state.values[0] = time(1);
        }
        assert_eq!(
            calendar.reset("5min".to_owned(), Some(time(60)), Some(time(180))),
            Err(Error::Provider("injected".to_owned()))
        );
        assert_eq!(calendar.frequency(), "5min");
        assert_eq!(calendar.all_time(), (Some(time(60)), Some(time(180))));
        assert_eq!(calendar.indices(), (0, 2));
        assert_eq!(calendar.trade_len(), 3);
        assert_eq!(calendar.trade_step(), 1);
        assert_eq!(
            calendar.step_time(Some(0), 0).unwrap().0,
            time(i64::from(failed_call == 2))
        );
        source.lock().unwrap().fail = None;
        calendar
            .reset("1min".to_owned(), Some(time(60)), Some(time(180)))
            .unwrap();
        assert_eq!(calendar.indices(), (1, 3));
        assert_eq!(calendar.trade_step(), 0);
    }
}

#[test]
fn data_ranges_use_live_frequency_and_stop_at_the_exact_failure() {
    let source = source();
    let calendar = calendar(&source, 0, 120);
    for frequency in ["1min", "5min"] {
        let context = Context(Ok(frequency.to_owned()));
        source.lock().unwrap().calls.clear();
        assert_eq!(calendar.data_range("full", &context).unwrap(), (0, 2));
        assert_eq!(calendar.data_range("step", &context).unwrap(), (0, 0));
        assert_eq!(
            source.lock().unwrap().calls,
            vec![format!("locate:{frequency}:false"); 4]
        );
    }
    let context = Context(Ok("1min".to_owned()));
    for failed_call in [1, 2] {
        {
            let mut state = source.lock().unwrap();
            state.calls.clear();
            state.fail = Some(failed_call);
        }
        assert_eq!(
            calendar.data_range("full", &context),
            Err(Error::Provider("injected".to_owned()))
        );
        assert_eq!(source.lock().unwrap().calls.len(), failed_call);
    }
    source.lock().unwrap().fail = None;
    assert_eq!(
        calendar.data_range("bad", &context),
        Err(Error::InvalidRangeType("bad".to_owned()))
    );
    source.lock().unwrap().calls.clear();
    assert_eq!(
        calendar.data_range("full", &Context(Err(Error::Provider("context".to_owned())))),
        Err(Error::Provider("context".to_owned()))
    );
    assert!(source.lock().unwrap().calls.is_empty());
}

#[test]
fn execution_interfaces_share_the_same_live_cursor_and_reset_window() {
    let source = source();
    let manager = Arc::new(Mutex::new(calendar(&source, 0, 120)));
    let shared = SharedExecutionCalendar::new(
        Arc::clone(&manager),
        Arc::new(Context(Ok("1min".to_owned()))),
    );
    let clone = shared.clone();
    assert_eq!(NestedCalendar::trade_len(&shared).unwrap(), 3);
    assert_eq!(NestedCalendar::trade_step(&shared).unwrap(), 0);
    assert!(!NestedCalendar::finished(&shared).unwrap());
    assert_eq!(SaoeDecisionCalendar::frequency(&shared).unwrap(), "1min");
    assert_eq!(SaoeCalendar::available_step_range(&shared).unwrap(), (0, 0));
    assert_eq!(
        SimulatorCalendar::step_time(&shared).unwrap(),
        (time(0), time(59))
    );
    ExecutorLifecycleCalendar::step(&clone).unwrap();
    assert_eq!(NestedCalendar::trade_step(&shared).unwrap(), 1);
    assert_eq!(
        ExecutorLifecycleCalendar::step_time(&shared).unwrap(),
        (time(60), time(119))
    );
    assert_eq!(
        SaoeCalendar::step_time(&shared).unwrap(),
        (time(60), time(119))
    );
    assert_eq!(SaoeCalendar::available_step_range(&shared).unwrap(), (1, 1));
    ResettableNestedCalendar::reset_window(&shared, time(120), time(180)).unwrap();
    assert_eq!(
        manager.lock().unwrap().all_time(),
        (Some(time(120)), Some(time(180)))
    );
    assert_eq!(NestedCalendar::trade_len(&clone).unwrap(), 2);
    assert_eq!(NestedCalendar::trade_step(&clone).unwrap(), 0);
    assert_eq!(
        NestedCalendar::step_time(&clone).unwrap(),
        (time(120), time(179))
    );
}

#[test]
fn execution_interface_failures_preserve_original_errors_and_partial_reset() {
    let source = source();
    let manager = Arc::new(Mutex::new(calendar(&source, 240, 240)));
    let shared = SharedExecutionCalendar::new(
        Arc::clone(&manager),
        Arc::new(Context(Ok("1min".to_owned()))),
    );
    assert!(
        NestedCalendar::step_time(&shared)
            .unwrap_err()
            .message
            .contains("index out of range")
    );
    assert!(
        ExecutorLifecycleCalendar::step_time(&shared)
            .unwrap_err()
            .message
            .contains("index out of range")
    );
    assert!(
        SimulatorCalendar::step_time(&shared)
            .unwrap_err()
            .message
            .contains("index out of range")
    );
    assert!(
        SaoeCalendar::step_time(&shared)
            .unwrap_err()
            .message
            .contains("index out of range")
    );
    assert!(
        SaoeCalendar::available_step_range(&shared)
            .unwrap_err()
            .message
            .contains("index out of range")
    );
    NestedCalendar::step(&shared).unwrap();
    assert!(NestedCalendar::finished(&shared).unwrap());
    assert!(
        ExecutorLifecycleCalendar::step(&shared)
            .unwrap_err()
            .message
            .contains("finished")
    );
    {
        let mut state = source.lock().unwrap();
        state.calls.clear();
        state.fail = Some(1);
    }
    assert!(
        ResettableNestedCalendar::reset_window(&shared, time(0), time(120))
            .unwrap_err()
            .message
            .contains("injected")
    );
    assert_eq!(manager.lock().unwrap().trade_step(), 1);
    assert_eq!(
        manager.lock().unwrap().all_time(),
        (Some(time(0)), Some(time(120)))
    );
}

#[test]
fn poisoned_cursor_is_reported_by_every_execution_interface_without_recovery() {
    let source = source();
    let manager = Arc::new(Mutex::new(calendar(&source, 0, 120)));
    let shared = SharedExecutionCalendar::new(
        Arc::clone(&manager),
        Arc::new(Context(Ok("1min".to_owned()))),
    );
    assert!(
        std::thread::spawn(move || {
            let _guard = manager.lock().unwrap();
            panic!("poison calendar");
        })
        .join()
        .is_err()
    );
    let messages = [
        NestedCalendar::finished(&shared).unwrap_err().message,
        NestedCalendar::trade_len(&shared).unwrap_err().message,
        NestedCalendar::trade_step(&shared).unwrap_err().message,
        NestedCalendar::step_time(&shared).unwrap_err().message,
        NestedCalendar::step(&shared).unwrap_err().message,
        ResettableNestedCalendar::reset_window(&shared, time(0), time(120))
            .unwrap_err()
            .message,
        ExecutorLifecycleCalendar::step_time(&shared)
            .unwrap_err()
            .message,
        ExecutorLifecycleCalendar::step(&shared)
            .unwrap_err()
            .message,
        SimulatorCalendar::step_time(&shared).unwrap_err().message,
        SaoeCalendar::available_step_range(&shared)
            .unwrap_err()
            .message,
        SaoeCalendar::step_time(&shared).unwrap_err().message,
        SaoeDecisionCalendar::frequency(&shared)
            .unwrap_err()
            .message,
    ];
    for message in messages {
        assert_eq!(
            message,
            "execution calendar provider error: execution calendar lock poisoned"
        );
    }
    let expected = TradeCalendarRangeError::Provider {
        message: "execution calendar provider error: execution calendar lock poisoned".to_owned(),
    };
    assert_eq!(
        TradeCalendarRange::start_time(&shared),
        Err(expected.clone())
    );
    assert_eq!(
        shared.get_range_idx(time(0), time(60)),
        Err(expected.clone())
    );
    assert_eq!(
        TradeRangeByTime::parse("09:31", "09:33")
            .unwrap()
            .range_indices(Some(&shared)),
        Err(TradeRangeError::Calendar(expected))
    );
}

#[test]
fn live_range_calendar_observes_requested_start_and_failed_reset_metadata() {
    let source = source();
    let manager = Arc::new(Mutex::new(calendar(&source, 30, 180)));
    let shared = SharedExecutionCalendar::new(
        Arc::clone(&manager),
        Arc::new(Context(Err(Error::Provider("unused".to_owned())))),
    );
    let range: Arc<dyn TradeCalendarRange> = Arc::new(shared.clone());
    assert_eq!(range.start_time(), Ok(time(30)));
    assert_eq!(NestedCalendar::step_time(&shared).unwrap().0, time(60));
    NestedCalendar::step(&shared).unwrap();
    assert_eq!(range.start_time(), Ok(time(30)));
    ResettableNestedCalendar::reset_window(&shared, time(120), time(180)).unwrap();
    assert_eq!(range.start_time(), Ok(time(120)));
    assert_eq!(range.get_range_idx(time(60), time(600)), Ok((0, 1)));
    {
        let mut state = source.lock().unwrap();
        state.calls.clear();
        state.fail = Some(1);
    }
    let next_day = time(0) + TimeDelta::days(1);
    assert!(ResettableNestedCalendar::reset_window(&shared, next_day, next_day).is_err());
    assert_eq!(range.start_time(), Ok(next_day));
    // Failed load retains the old index array, but the rule must use the new requested date.
    assert_eq!(
        TradeRangeByTime::parse("09:30", "09:31")
            .unwrap()
            .range_indices(Some(&*range)),
        Ok((1, 1))
    );
    source.lock().unwrap().fail = None;
    manager
        .lock()
        .unwrap()
        .reset("1min".to_owned(), None, None)
        .unwrap();
    let missing = TradeCalendarRangeError::from(Error::MissingStart);
    assert_eq!(range.start_time(), Err(missing.clone()));
    assert_eq!(
        TradeRangeByTime::parse("09:30", "09:31")
            .unwrap()
            .range_indices(Some(&*range)),
        Err(TradeRangeError::Calendar(missing))
    );
    // Direct index queries do not need a date anchor.
    assert_eq!(range.get_range_idx(time(0), time(60)), Ok((0, 1)));
    source.lock().unwrap().indices = Some((1, i64::MIN));
    manager
        .lock()
        .unwrap()
        .reset("1min".to_owned(), Some(time(0)), Some(time(60)))
        .unwrap();
    assert_eq!(
        range.get_range_idx(time(0), time(60)),
        Err(TradeCalendarRangeError::from(Error::IndexOverflow))
    );
}
