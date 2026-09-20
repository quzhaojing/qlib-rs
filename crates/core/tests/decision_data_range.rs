use std::{collections::VecDeque, process::Command};

use chrono::NaiveDateTime;
use domain_core::{
    DataCalendarLocation, DataCalendarLocator, DataCalendarLocatorError, DecisionDataRangeError,
    IdxTradeRange, SharedTradeRange, TradeDecision, TradeRangeByTime, TradeRangeError,
    data_calendar_range_limit,
};
use serde_json::{Value, json};

fn at(value: &str) -> NaiveDateTime {
    NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S").unwrap()
}

#[derive(Default)]
struct Locator {
    results: VecDeque<Result<DataCalendarLocation, DataCalendarLocatorError>>,
    calls: Vec<(NaiveDateTime, NaiveDateTime, String)>,
}

impl DataCalendarLocator for Locator {
    fn locate(
        &mut self,
        start_time: NaiveDateTime,
        end_time: NaiveDateTime,
        frequency: &str,
    ) -> Result<DataCalendarLocation, DataCalendarLocatorError> {
        self.calls
            .push((start_time, end_time, frequency.to_owned()));
        self.results.pop_front().expect("test result is configured")
    }
}

fn location(start_index: i64, end_index: i64) -> DataCalendarLocation {
    DataCalendarLocation {
        start_index,
        end_index,
    }
}

fn decision(range: Option<SharedTradeRange>) -> TradeDecision<String> {
    TradeDecision::from_items(
        Vec::new(),
        at("2024-01-02 09:30:00"),
        at("2024-01-02 10:00:00"),
        range,
    )
}

#[test]
fn actual_source_freezes_order_bounds_results_and_failures() {
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/decision_data_range_contract.py"
        ))
        .output()
        .expect("Python characterization fixture runs");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Value = serde_json::from_slice(&output.stdout).expect("fixture emits JSON");
    let cases = cases.as_array().unwrap();
    assert_eq!(cases.len(), 8);
    assert_eq!(cases[0]["result"], json!([0, 287]));
    assert_eq!(
        cases[0]["events"][0],
        json!([
            "locate",
            "2024-01-02T00:00:00",
            "2024-01-02T23:59:59",
            "5min"
        ])
    );
    assert_eq!(
        cases[1]["error"],
        "NotImplementedError:There is no trade_range in this case"
    );
    assert_eq!(
        cases[2]["events"][1],
        json!(["clip", "2024-01-02T00:00:00", "2024-01-02T23:59:59"])
    );
    assert_eq!(cases[2]["result"], json!([10, 267]));
    assert_eq!(
        cases[3]["events"][1],
        json!(["clip", "2024-01-02T09:30:00", "2024-01-02T10:00:00"])
    );
    assert_eq!(
        cases[4]["error"],
        "ValueError:This type of input bad is not supported"
    );
    assert_eq!(cases[4]["events"].as_array().unwrap().len(), 1);
    assert_eq!(cases[5]["error"], "RuntimeError:locate-1");
    assert_eq!(cases[6]["error"], "RuntimeError:locate-2");
    assert_eq!(cases[7]["error"], "RuntimeError:clip");
}

#[test]
fn native_projection_preserves_lookup_and_clipping_order() {
    let timed: SharedTradeRange = Arc::new(TradeRangeByTime::parse("09:40", "09:50").unwrap());
    for (range_type, expected_clip, expected) in [
        (
            "full",
            (at("2024-01-02 09:40:00"), at("2024-01-02 09:50:00")),
            (10, 267),
        ),
        (
            "step",
            (at("2024-01-02 09:40:00"), at("2024-01-02 09:50:00")),
            (10, 267),
        ),
    ] {
        let mut locator = Locator {
            results: VecDeque::from([Ok(location(100, 387)), Ok(location(110, 367))]),
            calls: Vec::new(),
        };
        assert_eq!(
            data_calendar_range_limit(
                &decision(Some(Arc::clone(&timed))),
                range_type,
                false,
                "5min",
                &mut locator
            ),
            Ok(expected)
        );
        assert_eq!(
            locator.calls[0],
            (
                at("2024-01-02 00:00:00"),
                at("2024-01-02 23:59:59"),
                "5min".to_owned()
            )
        );
        assert_eq!((locator.calls[1].0, locator.calls[1].1), expected_clip);
    }
}

#[test]
fn native_projection_covers_missing_invalid_and_every_failure_stage() {
    let mut locator = Locator {
        results: VecDeque::from([Ok(location(100, 387))]),
        calls: Vec::new(),
    };
    assert_eq!(
        data_calendar_range_limit(&decision(None), "full", false, "day", &mut locator),
        Ok((0, 287))
    );
    locator.results.push_back(Ok(location(100, 387)));
    assert_eq!(
        data_calendar_range_limit(&decision(None), "full", true, "day", &mut locator),
        Err(DecisionDataRangeError::MissingRange)
    );

    let index: SharedTradeRange = Arc::new(IdxTradeRange::new(1, 2));
    locator.results.push_back(Ok(location(100, 387)));
    assert_eq!(
        data_calendar_range_limit(
            &decision(Some(Arc::clone(&index))),
            "bad",
            false,
            "day",
            &mut locator
        ),
        Err(DecisionDataRangeError::UnsupportedRangeType(
            "bad".to_owned()
        ))
    );
    locator.results.push_back(Ok(location(100, 387)));
    assert_eq!(
        data_calendar_range_limit(&decision(Some(index)), "full", false, "day", &mut locator),
        Err(DecisionDataRangeError::TradeRange(
            TradeRangeError::IndexTimeClippingUnsupported
        ))
    );
    let index: SharedTradeRange = Arc::new(IdxTradeRange::new(1, 2));
    locator.results.push_back(Ok(location(100, 387)));
    assert_eq!(
        data_calendar_range_limit(&decision(Some(index)), "step", false, "day", &mut locator),
        Err(DecisionDataRangeError::TradeRange(
            TradeRangeError::IndexTimeClippingUnsupported
        ))
    );
}

#[test]
fn native_projection_covers_locator_failures_and_index_overflow() {
    let failure = DataCalendarLocatorError {
        message: "offline".to_owned(),
    };
    let mut first = Locator {
        results: VecDeque::from([Err(failure.clone())]),
        calls: Vec::new(),
    };
    assert_eq!(
        data_calendar_range_limit(&decision(None), "full", false, "day", &mut first),
        Err(DecisionDataRangeError::Locator(failure.clone()))
    );
    let timed: SharedTradeRange = Arc::new(TradeRangeByTime::parse("09:40", "09:50").unwrap());
    let mut second = Locator {
        results: VecDeque::from([Ok(location(100, 387)), Err(failure.clone())]),
        calls: Vec::new(),
    };
    assert_eq!(
        data_calendar_range_limit(&decision(Some(timed)), "full", false, "day", &mut second),
        Err(DecisionDataRangeError::Locator(failure))
    );

    let mut overflow = Locator {
        results: VecDeque::from([Ok(location(i64::MIN, i64::MAX))]),
        calls: Vec::new(),
    };
    assert_eq!(
        data_calendar_range_limit(&decision(None), "full", false, "day", &mut overflow),
        Err(DecisionDataRangeError::IndexOverflow)
    );

    let timed: SharedTradeRange = Arc::new(TradeRangeByTime::parse("09:40", "09:50").unwrap());
    let mut start_overflow = Locator {
        results: VecDeque::from([
            Ok(location(i64::MAX, i64::MAX)),
            Ok(location(i64::MIN, i64::MAX)),
        ]),
        calls: Vec::new(),
    };
    assert_eq!(
        data_calendar_range_limit(
            &decision(Some(Arc::clone(&timed))),
            "full",
            false,
            "day",
            &mut start_overflow
        ),
        Err(DecisionDataRangeError::IndexOverflow)
    );
    let mut end_overflow = Locator {
        results: VecDeque::from([
            Ok(location(i64::MIN, i64::MIN)),
            Ok(location(i64::MIN, i64::MAX)),
        ]),
        calls: Vec::new(),
    };
    assert_eq!(
        data_calendar_range_limit(
            &decision(Some(timed)),
            "full",
            false,
            "day",
            &mut end_overflow
        ),
        Err(DecisionDataRangeError::IndexOverflow)
    );
}

use std::sync::Arc;
