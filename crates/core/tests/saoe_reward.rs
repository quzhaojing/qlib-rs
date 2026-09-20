#![allow(clippy::float_cmp)]

use std::{
    path::PathBuf,
    process::Command,
    sync::{Arc, Mutex},
};

use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch, TimestampNanosecondArray};
use arrow_schema::Schema;
use chrono::NaiveDateTime;
use domain_core::{
    Order, OrderDir, PaPenaltyReward, PpoReward, RewardCombination, SaoeBacktestData, SaoeReward,
    SaoeRewardError, SaoeRewardLogError, SaoeRewardLogSink, SaoeState, SaoeStateParts,
};
use indexmap::IndexMap;
use ndarray::Array1;
use serde_json::Value;

fn nanos(minute: i64) -> i64 {
    (NaiveDateTime::parse_from_str("2024-01-02 09:30:00", "%Y-%m-%d %H:%M:%S").unwrap()
        + chrono::TimeDelta::minutes(minute))
    .and_utc()
    .timestamp_nanos_opt()
    .unwrap()
}

fn exec_batch(
    times: ArrayRef,
    prices: ArrayRef,
    deals: ArrayRef,
    amounts: ArrayRef,
) -> RecordBatch {
    RecordBatch::try_from_iter(vec![
        ("datetime", times),
        ("market_price", prices),
        ("deal_amount", deals),
        ("amount", amounts),
    ])
    .unwrap()
}

fn step_batch(times: ArrayRef, amounts: ArrayRef, pa: ArrayRef) -> RecordBatch {
    RecordBatch::try_from_iter(vec![("datetime", times), ("amount", amounts), ("pa", pa)]).unwrap()
}

fn f64s(values: Vec<Option<f64>>) -> ArrayRef {
    Arc::new(Float64Array::from(values))
}

fn times(values: Vec<Option<i64>>) -> ArrayRef {
    Arc::new(TimestampNanosecondArray::from(values))
}

fn state_with(
    direction: OrderDir,
    amount: f64,
    cur_step: i64,
    position: f64,
    history_exec: RecordBatch,
    history_steps: RecordBatch,
    deal_prices: Vec<f64>,
) -> SaoeState {
    SaoeState::new(SaoeStateParts {
        order: Order::new("A", amount, direction, None, None),
        cur_time: NaiveDateTime::default(),
        cur_step,
        position,
        history_exec,
        history_steps,
        metrics: None,
        backtest_data: SaoeBacktestData {
            ticks_index: Vec::new(),
            ticks_for_order: Vec::new(),
            deal_prices: Array1::from_vec(deal_prices),
            market_volumes: Array1::zeros(0),
            features: RecordBatch::new_empty(Arc::new(Schema::empty())),
        },
        ticks_per_step: 1,
        ticks_index: Vec::new(),
        ticks_for_order: Vec::new(),
    })
}

fn state(direction: OrderDir, cur_step: i64, position: f64, deal_prices: Vec<f64>) -> SaoeState {
    state_with(
        direction,
        10.0,
        cur_step,
        position,
        exec_batch(
            times(vec![Some(nanos(0)), Some(nanos(1)), Some(nanos(2))]),
            f64s(vec![Some(9.0), Some(10.0), Some(20.0)]),
            f64s(vec![Some(0.0), Some(1.0), Some(3.0)]),
            f64s(vec![Some(9.0), Some(2.0), Some(1.0)]),
        ),
        step_batch(
            times(vec![Some(nanos(1))]),
            f64s(vec![Some(4.0)]),
            f64s(vec![Some(2.0)]),
        ),
        deal_prices,
    )
}

struct Logger {
    values: SharedLogs,
    fail_at: Option<usize>,
}

impl SaoeRewardLogSink for Logger {
    fn log_scalar(&self, name: &str, value: f64) -> Result<(), SaoeRewardLogError> {
        let mut values = self.values.lock().unwrap();
        values.push((name.to_owned(), value));
        if self.fail_at == Some(values.len()) {
            return Err(SaoeRewardLogError {
                message: name.to_owned(),
            });
        }
        Ok(())
    }
}

type SharedLogs = Arc<Mutex<Vec<(String, f64)>>>;

fn logger(fail_at: Option<usize>) -> (Arc<dyn SaoeRewardLogSink>, SharedLogs) {
    let values = Arc::new(Mutex::new(Vec::new()));
    (
        Arc::new(Logger {
            values: Arc::clone(&values),
            fail_at,
        }),
        values,
    )
}

#[test]
fn reward_values_and_logging_match_live_python_source() {
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/saoe_reward_contract.py");
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../qlib/qlib/rl/order_execution/reward.py");
    let output = Command::new("python")
        .arg(fixture)
        .arg(source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let expected: Value = serde_json::from_slice(&output.stdout).unwrap();

    let (sink, values) = logger(None);
    let reward = PaPenaltyReward::new(100.0, 2.0).with_logger(sink);
    assert_eq!(
        reward
            .reward(&state(OrderDir::Buy, 0, 10.0, vec![12.0]))
            .unwrap(),
        expected["pa"].as_f64().unwrap()
    );
    let actual = values.lock().unwrap();
    for (actual, expected) in actual.iter().zip(expected["logs"].as_array().unwrap()) {
        assert_eq!(actual.0, expected[0].as_str().unwrap());
        assert_eq!(actual.1, expected[1].as_f64().unwrap());
    }

    let ppo = PpoReward::new(4, 7, 239);
    assert_eq!(ppo.max_step(), 4);
    assert_eq!(ppo.start_time_index(), 7);
    assert_eq!(ppo.end_time_index(), 239);
    let actual = [
        ppo.reward(&state(OrderDir::Buy, 1, 10.0, vec![12.0]))
            .unwrap(),
        ppo.reward(&state(OrderDir::Buy, 3, 10.0, vec![24.0]))
            .unwrap(),
        ppo.reward(&state(OrderDir::Sell, 0, 0.0, vec![20.0]))
            .unwrap(),
    ];
    for (actual, expected) in actual.iter().zip(&expected["ppo"].as_array().unwrap()[..3]) {
        assert_eq!(*actual, expected.as_f64().unwrap());
    }
    assert!(
        expected["assertions"]
            .as_object()
            .unwrap()
            .values()
            .all(|value| value.as_bool() == Some(true))
    );

    let (sink, _) = logger(None);
    let mut attached = PaPenaltyReward::new(100.0, 1.0);
    SaoeReward::set_logger(&mut attached, Some(sink));
    assert!(
        attached
            .reward(&state(OrderDir::Buy, 0, 10.0, vec![12.0]))
            .is_ok()
    );
    let mut combination = RewardCombination::new(IndexMap::new());
    SaoeReward::set_logger(&mut combination, None);
    assert_eq!(
        combination
            .reward(&state(OrderDir::Buy, 0, 10.0, vec![12.0]))
            .unwrap(),
        0.0
    );
}

#[test]
fn pa_penalty_covers_empty_slices_validation_nonfinite_scale_and_logger_failures() {
    let valid = state(OrderDir::Buy, 0, 10.0, vec![12.0]);
    assert!(matches!(
        PaPenaltyReward::default().reward(&valid),
        Err(SaoeRewardError::MissingLogger)
    ));
    for amount in [0.0, -1.0, f64::NAN] {
        let invalid = state_with(
            OrderDir::Buy,
            amount,
            0,
            1.0,
            valid.parts().history_exec.clone(),
            valid.parts().history_steps.clone(),
            vec![1.0],
        );
        assert!(matches!(
            PaPenaltyReward::default().reward(&invalid),
            Err(SaoeRewardError::InvalidOrderAmount(_))
        ));
    }
    let empty_steps = state_with(
        OrderDir::Buy,
        10.0,
        0,
        1.0,
        valid.parts().history_exec.clone(),
        step_batch(times(Vec::new()), f64s(Vec::new()), f64s(Vec::new())),
        vec![1.0],
    );
    assert!(matches!(
        PaPenaltyReward::default().reward(&empty_steps),
        Err(SaoeRewardError::EmptyStepHistory)
    ));

    let empty_exec = state_with(
        OrderDir::Buy,
        10.0,
        0,
        1.0,
        exec_batch(
            times(Vec::new()),
            f64s(Vec::new()),
            f64s(Vec::new()),
            f64s(Vec::new()),
        ),
        valid.parts().history_steps.clone(),
        vec![1.0],
    );
    let (sink, _) = logger(None);
    assert_eq!(
        PaPenaltyReward::new(100.0, f64::INFINITY)
            .with_logger(sink)
            .reward(&empty_exec)
            .unwrap(),
        f64::INFINITY
    );

    let nan_steps = step_batch(
        times(vec![Some(nanos(1))]),
        f64s(vec![Some(4.0)]),
        f64s(vec![Some(f64::NAN)]),
    );
    let nan_state = state_with(
        OrderDir::Buy,
        10.0,
        0,
        1.0,
        valid.parts().history_exec.clone(),
        nan_steps,
        vec![1.0],
    );
    let (sink, _) = logger(None);
    assert!(matches!(
        PaPenaltyReward::default()
            .with_logger(sink)
            .reward(&nan_state),
        Err(SaoeRewardError::NonFinite(_))
    ));

    for fail_at in [1, 2] {
        let (sink, values) = logger(Some(fail_at));
        assert!(matches!(
            PaPenaltyReward::default().with_logger(sink).reward(&valid),
            Err(SaoeRewardError::Log(_))
        ));
        assert_eq!(values.lock().unwrap().len(), fail_at);
    }
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one schema matrix covers every reward extraction boundary"
)]
fn malformed_reward_history_columns_and_nulls_are_typed() {
    let valid = state(OrderDir::Buy, 3, 10.0, vec![12.0]);
    let missing = RecordBatch::try_from_iter(vec![("other", f64s(vec![Some(1.0)]))]).unwrap();
    let bad_time = step_batch(
        Arc::new(Int64Array::from(vec![1])) as ArrayRef,
        f64s(vec![Some(1.0)]),
        f64s(vec![Some(1.0)]),
    );
    let bad_float = step_batch(
        times(vec![Some(nanos(0))]),
        Arc::new(Int64Array::from(vec![1])) as ArrayRef,
        f64s(vec![Some(1.0)]),
    );
    let null_time = step_batch(
        times(vec![None]),
        f64s(vec![Some(1.0)]),
        f64s(vec![Some(1.0)]),
    );
    let null_float = step_batch(
        times(vec![Some(nanos(0))]),
        f64s(vec![Some(1.0)]),
        f64s(vec![None]),
    );
    for steps in [missing, bad_time, bad_float, null_time, null_float] {
        let malformed = state_with(
            OrderDir::Buy,
            10.0,
            3,
            10.0,
            valid.parts().history_exec.clone(),
            steps,
            vec![12.0],
        );
        let (sink, _) = logger(None);
        assert!(
            PaPenaltyReward::default()
                .with_logger(sink)
                .reward(&malformed)
                .is_err()
        );
    }

    let missing_exec = RecordBatch::try_from_iter(vec![("other", f64s(vec![Some(1.0)]))]).unwrap();
    let bad_market = exec_batch(
        times(vec![Some(nanos(0))]),
        Arc::new(Int64Array::from(vec![1])) as ArrayRef,
        f64s(vec![Some(1.0)]),
        f64s(vec![Some(1.0)]),
    );
    let null_deal = exec_batch(
        times(vec![Some(nanos(0))]),
        f64s(vec![Some(1.0)]),
        f64s(vec![None]),
        f64s(vec![Some(1.0)]),
    );
    for exec in [missing_exec, bad_market, null_deal] {
        let malformed = state_with(
            OrderDir::Buy,
            10.0,
            3,
            10.0,
            exec,
            valid.parts().history_steps.clone(),
            vec![12.0],
        );
        assert!(PpoReward::new(4, 0, 239).reward(&malformed).is_err());
    }

    let null_exec_time = exec_batch(
        times(vec![None]),
        f64s(vec![Some(1.0)]),
        f64s(vec![Some(1.0)]),
        f64s(vec![Some(1.0)]),
    );
    let malformed = state_with(
        OrderDir::Buy,
        10.0,
        0,
        10.0,
        null_exec_time,
        valid.parts().history_steps.clone(),
        vec![1.0],
    );
    let (sink, _) = logger(None);
    assert!(matches!(
        PaPenaltyReward::default()
            .with_logger(sink)
            .reward(&malformed),
        Err(SaoeRewardError::NullValue { .. })
    ));

    for exec in [
        RecordBatch::try_from_iter(vec![("amount", f64s(vec![Some(1.0)]))]).unwrap(),
        RecordBatch::try_from_iter(vec![("datetime", times(vec![Some(nanos(0))]))]).unwrap(),
        exec_batch(
            times(vec![Some(nanos(0))]),
            f64s(vec![Some(1.0)]),
            f64s(vec![Some(1.0)]),
            f64s(vec![None]),
        ),
    ] {
        let malformed = state_with(
            OrderDir::Buy,
            10.0,
            0,
            10.0,
            exec,
            valid.parts().history_steps.clone(),
            vec![1.0],
        );
        let (sink, _) = logger(None);
        assert!(
            PaPenaltyReward::default()
                .with_logger(sink)
                .reward(&malformed)
                .is_err()
        );
    }

    for exec in [
        RecordBatch::try_from_iter(vec![("market_price", f64s(vec![Some(1.0)]))]).unwrap(),
        exec_batch(
            times(vec![Some(nanos(0))]),
            f64s(vec![None]),
            f64s(vec![Some(1.0)]),
            f64s(vec![Some(1.0)]),
        ),
    ] {
        let malformed = state_with(
            OrderDir::Buy,
            10.0,
            3,
            10.0,
            exec,
            valid.parts().history_steps.clone(),
            vec![1.0],
        );
        assert!(PpoReward::new(4, 0, 239).reward(&malformed).is_err());
    }
}

#[test]
fn ppo_preserves_zero_denominators_thresholds_nan_empty_and_step_overflow() {
    let one = |direction, price, twap| {
        state_with(
            direction,
            10.0,
            3,
            10.0,
            exec_batch(
                times(vec![Some(nanos(0))]),
                f64s(vec![Some(price)]),
                f64s(vec![Some(0.0)]),
                f64s(vec![Some(1.0)]),
            ),
            step_batch(times(Vec::new()), f64s(Vec::new()), f64s(Vec::new())),
            vec![twap],
        )
    };
    let reward = PpoReward::new(4, 0, 239);
    assert_eq!(reward.reward(&one(OrderDir::Buy, 10.0, 10.0)).unwrap(), 0.0);
    assert_eq!(reward.reward(&one(OrderDir::Buy, 10.0, 11.0)).unwrap(), 1.0);
    assert_eq!(reward.reward(&one(OrderDir::Buy, 0.0, 0.0)).unwrap(), 0.0);
    assert_eq!(reward.reward(&one(OrderDir::Sell, 10.0, 0.0)).unwrap(), 0.0);
    assert_eq!(
        reward
            .reward(&one(OrderDir::Buy, f64::NAN, f64::NAN))
            .unwrap(),
        1.0
    );

    let empty = state_with(
        OrderDir::Buy,
        10.0,
        3,
        10.0,
        exec_batch(
            times(Vec::new()),
            f64s(Vec::new()),
            f64s(Vec::new()),
            f64s(Vec::new()),
        ),
        step_batch(times(Vec::new()), f64s(Vec::new()), f64s(Vec::new())),
        Vec::new(),
    );
    assert_eq!(reward.reward(&empty).unwrap(), 1.0);

    let overflow = state(OrderDir::Buy, i64::MAX, 10.0, vec![1.0]);
    assert_eq!(
        PpoReward::new(i64::MIN, 0, 0).reward(&overflow).unwrap(),
        0.0
    );
}
