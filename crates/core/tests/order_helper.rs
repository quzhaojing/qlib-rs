use std::{
    process::Command,
    sync::{Arc, Mutex},
};

use chrono::NaiveDateTime;
use domain_core::{
    OrderDir, OrderHelper, OrderHelperError, OrderTimeInput, OrderTimestampParseError,
    OrderTimestampParser, create_order,
};
use serde_json::{Value, json};

fn at(value: &str) -> NaiveDateTime {
    NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S").unwrap()
}

#[derive(Clone)]
struct Parser {
    calls: Arc<Mutex<Vec<OrderTimeInput>>>,
    fail_on: Option<String>,
}

impl OrderTimestampParser for Parser {
    fn parse(&self, input: OrderTimeInput) -> Result<NaiveDateTime, OrderTimestampParseError> {
        self.calls.lock().unwrap().push(input.clone());
        let key = match &input {
            OrderTimeInput::Timestamp(value) => value.to_string(),
            OrderTimeInput::Text(value) => value.clone(),
        };
        if self.fail_on.as_deref() == Some(&key) {
            return Err(OrderTimestampParseError { message: key });
        }
        Ok(match input {
            OrderTimeInput::Timestamp(value) => value,
            OrderTimeInput::Text(value) if value == "start" => at("2024-01-02 09:30:00"),
            OrderTimeInput::Text(_) => at("2024-01-02 10:00:00"),
        })
    }
}

fn parser(fail_on: Option<&str>) -> Parser {
    Parser {
        calls: Arc::new(Mutex::new(Vec::new())),
        fail_on: fail_on.map(str::to_owned),
    }
}

#[test]
fn helper_retains_exchange_and_creates_orders_in_timestamp_order() {
    let exchange = Arc::new(String::from("exchange"));
    let timestamps = parser(None);
    let helper = OrderHelper::new(Arc::clone(&exchange), timestamps.clone());
    assert!(Arc::ptr_eq(helper.exchange(), &exchange));

    let order = helper
        .create(
            "股票/α",
            -0.0,
            OrderDir::Buy,
            Some(OrderTimeInput::from("start")),
            Some(OrderTimeInput::from("end")),
        )
        .unwrap();
    assert_eq!(order.stock_id(), "股票/α");
    assert_eq!(order.amount().to_bits(), (-0.0_f64).to_bits());
    assert_eq!(order.direction(), OrderDir::Buy);
    assert_eq!(order.start_time(), Some(at("2024-01-02 09:30:00")));
    assert_eq!(order.end_time(), Some(at("2024-01-02 10:00:00")));
    assert_eq!(order.deal_amount().to_bits(), 0.0_f64.to_bits());
    assert_eq!(order.factor(), None);
    assert_eq!(
        *timestamps.calls.lock().unwrap(),
        [OrderTimeInput::from("start"), OrderTimeInput::from("end")]
    );
}

#[test]
fn null_and_materialized_times_follow_the_same_static_constructor() {
    assert_eq!(
        OrderTimeInput::from(String::from("owned")),
        OrderTimeInput::Text(String::from("owned"))
    );
    let timestamps = parser(None);
    let no_times = create_order("A", f64::NAN, OrderDir::Sell, None, None, &timestamps).unwrap();
    assert!(no_times.amount().is_nan());
    assert_eq!(no_times.start_time(), None);
    assert_eq!(no_times.end_time(), None);
    assert!(timestamps.calls.lock().unwrap().is_empty());

    let start = at("2024-02-03 04:05:06");
    let end = at("2024-02-03 07:08:09");
    let typed = create_order(
        "B",
        f64::INFINITY,
        OrderDir::Buy,
        Some(start.into()),
        Some(end.into()),
        &timestamps,
    )
    .unwrap();
    assert_eq!(typed.start_time(), Some(start));
    assert_eq!(typed.end_time(), Some(end));
    assert_eq!(
        *timestamps.calls.lock().unwrap(),
        [
            OrderTimeInput::Timestamp(start),
            OrderTimeInput::Timestamp(end)
        ]
    );
}

#[test]
fn parser_failures_stop_before_end_or_order_construction() {
    let timestamps = parser(Some("bad-start"));
    let error = create_order(
        "A",
        1.0,
        OrderDir::Buy,
        Some("bad-start".into()),
        Some("unreached".into()),
        &timestamps,
    )
    .unwrap_err();
    assert_eq!(
        error,
        OrderHelperError::StartTime(OrderTimestampParseError {
            message: "bad-start".to_owned()
        })
    );
    assert_eq!(
        *timestamps.calls.lock().unwrap(),
        [OrderTimeInput::from("bad-start")]
    );

    let timestamps = parser(Some("bad-end"));
    let error = create_order(
        "A",
        1.0,
        OrderDir::Buy,
        Some("start".into()),
        Some("bad-end".into()),
        &timestamps,
    )
    .unwrap_err();
    assert_eq!(
        error,
        OrderHelperError::EndTime(OrderTimestampParseError {
            message: "bad-end".to_owned()
        })
    );
    assert_eq!(
        *timestamps.calls.lock().unwrap(),
        [
            OrderTimeInput::from("start"),
            OrderTimeInput::from("bad-end")
        ]
    );
}

#[test]
fn unchanged_source_ast_freezes_identity_conversion_and_failure_order() {
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/order_helper_contract.py"
        ))
        .output()
        .expect("Python characterization fixture runs");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Value = serde_json::from_slice(&output.stdout).expect("fixture emits JSON");
    assert_eq!(cases.as_array().unwrap().len(), 5);
    assert_eq!(cases[0]["exchange_identity"], true);
    assert_eq!(cases[0]["events"][0][0], "order");
    assert_eq!(cases[0]["result"]["stock_id"], "股票/α");
    assert_eq!(
        cases[0]["result"]["amount"].as_f64().unwrap().to_bits(),
        (-0.0_f64).to_bits()
    );
    assert_eq!(cases[0]["result"]["direction"], 1);
    assert_eq!(cases[0]["result"]["start_time"], Value::Null);
    assert_eq!(cases[1]["events"][0], json!(["timestamp", "2024-01-02"]));
    assert_eq!(
        cases[1]["events"][1],
        json!(["timestamp", "2024-01-03 09:30"])
    );
    assert_eq!(cases[1]["events"][2][0], "order");
    assert_eq!(cases[2]["result"]["start_time"], "existing-start");
    assert_eq!(cases[2]["result"]["end_time"], "existing-end");
    assert_eq!(cases[3]["error"], "ValueError:bad-start");
    assert_eq!(cases[3]["events"], json!([["timestamp", "bad-start"]]));
    assert_eq!(cases[4]["error"], "ValueError:bad-end");
    assert_eq!(
        cases[4]["events"],
        json!([["timestamp", "ok"], ["timestamp", "bad-end"]])
    );
}
