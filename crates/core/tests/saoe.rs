use std::{
    process::Command,
    sync::{Arc, Mutex},
};

use arrow_array::{ArrayRef, Float64Array, RecordBatch};
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{DataType, Field, Schema};
use chrono::{NaiveDateTime, TimeDelta};
use domain_core::{
    NestedCalendar, NestedDecisionUpdate, NestedOuterDecision, NestedOuterDecisionError,
    NestedStrategy, NestedStrategyProgress, NestedStrategyPrompt, Order, OrderDecision, OrderDir,
    OrderTradeDecision, OwnedOrderExecution, ProxySaoeStrategy, SAOE_PROXY_PROMPT_KIND,
    SAOE_STATE_SCHEMA_VERSION, SaoeBacktestData, SaoeCalendar, SaoeError, SaoeMetrics, SaoeNumeric,
    SaoeOrderFactory, SaoePluginError, SaoeState, SaoeStateParts, SaoeStateProvider, SaoeTime,
    SharedOrderExecution, TradeRange,
};
use ndarray::arr1;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Debug, Deserialize, PartialEq)]
struct StateModuleSnapshot {
    source_sha256: String,
    module_doc: Option<String>,
    module_body: Vec<String>,
    imports: Vec<String>,
    type_checking_imports: Vec<String>,
    public_names: Vec<String>,
    metrics_annotations: Vec<[String; 2]>,
    metrics_required_keys: Vec<String>,
    metrics_optional_keys: Vec<String>,
    metrics_total: bool,
    state_annotations: Vec<[String; 2]>,
    state_fields: Vec<String>,
    state_signature: String,
    state_defaults: Option<Vec<Value>>,
    facts: Vec<String>,
}

fn strings<const N: usize>(values: [&str; N]) -> Vec<String> {
    values.map(str::to_owned).to_vec()
}

fn annotations<const N: usize>(values: [(&str, &str); N]) -> Vec<[String; 2]> {
    values
        .map(|(name, annotation)| [name.to_owned(), annotation.to_owned()])
        .to_vec()
}

#[test]
fn complete_state_module_surface_matches_live_python() {
    let output = Command::new("python")
        .args([
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/rl_order_execution_state_contract.py"
            ),
            r"D:\code\github\qlib\qlib\rl\order_execution\state.py",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let actual: StateModuleSnapshot = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(actual, expected_state_module_snapshot());
}

fn metric_annotations() -> Vec<[String; 2]> {
    annotations([
        ("stock_id", "str"),
        ("datetime", "pd.Timestamp | pd.DatetimeIndex"),
        ("direction", "int"),
        ("market_volume", "np.ndarray | float"),
        ("market_price", "np.ndarray | float"),
        ("amount", "np.ndarray | float"),
        ("inner_amount", "np.ndarray | float"),
        ("deal_amount", "np.ndarray | float"),
        ("trade_price", "np.ndarray | float"),
        ("trade_value", "np.ndarray | float"),
        ("position", "np.ndarray | float"),
        ("ffr", "np.ndarray | float"),
        ("pa", "np.ndarray | float"),
    ])
}

fn state_annotations() -> Vec<[String; 2]> {
    annotations([
        ("order", "Order"),
        ("cur_time", "pd.Timestamp"),
        ("cur_step", "int"),
        ("position", "float"),
        ("history_exec", "pd.DataFrame"),
        ("history_steps", "pd.DataFrame"),
        ("metrics", "Optional[SAOEMetrics]"),
        ("backtest_data", "BaseIntradayBacktestData"),
        ("ticks_per_step", "int"),
        ("ticks_index", "pd.DatetimeIndex"),
        ("ticks_for_order", "pd.DatetimeIndex"),
    ])
}

fn expected_state_module_snapshot() -> StateModuleSnapshot {
    let metric_names = [
        "amount",
        "datetime",
        "deal_amount",
        "direction",
        "ffr",
        "inner_amount",
        "market_price",
        "market_volume",
        "pa",
        "position",
        "stock_id",
        "trade_price",
        "trade_value",
    ];
    StateModuleSnapshot {
        source_sha256: "545958e4a969314d1e12cfa22ac375af3eefe1345125399f47e40df539513538"
            .to_owned(),
        module_doc: None,
        module_body: strings([
            "ImportFrom",
            "Import",
            "ImportFrom",
            "Import",
            "Import",
            "ImportFrom",
            "ImportFrom",
            "If",
            "ClassDef",
            "ClassDef",
        ]),
        imports: strings([
            "__future__:annotations",
            "typing",
            "typing:NamedTuple,Optional",
            "numpy as np",
            "pandas as pd",
            "qlib.backtest:Order",
            "qlib.typehint:TypedDict",
        ]),
        type_checking_imports: strings(["qlib.rl.data.base:BaseIntradayBacktestData"]),
        public_names: strings([
            "annotations",
            "typing",
            "NamedTuple",
            "Optional",
            "np",
            "pd",
            "Order",
            "TypedDict",
            "SAOEMetrics",
            "SAOEState",
        ]),
        metrics_annotations: metric_annotations(),
        metrics_required_keys: strings(metric_names),
        metrics_optional_keys: Vec::new(),
        metrics_total: true,
        state_annotations: state_annotations(),
        state_fields: strings([
            "order",
            "cur_time",
            "cur_step",
            "position",
            "history_exec",
            "history_steps",
            "metrics",
            "backtest_data",
            "ticks_per_step",
            "ticks_index",
            "ticks_for_order",
        ]),
        state_signature: "(order: Order, cur_time: pd.Timestamp, cur_step: int, position: float, history_exec: pd.DataFrame, history_steps: pd.DataFrame, metrics: Optional[SAOEMetrics], backtest_data: BaseIntradayBacktestData, ticks_per_step: int, ticks_index: pd.DatetimeIndex, ticks_for_order: pd.DatetimeIndex)".to_owned(),
        state_defaults: Some(Vec::new()),
        facts: strings([
            "metrics_construct_plain_dict",
            "metrics_all_values_retain_identity",
            "metrics_partial_runtime_construction",
            "metrics_extra_runtime_key",
            "state_is_tuple",
            "state_values_retain_identity",
            "state_asdict_order_and_identity",
            "state_replace_is_nonmutating",
            "state_make_preserves_identity",
            "type_checking_import_absent",
            "no_explicit_all",
            "metrics_doc_present",
            "state_doc_present",
        ]),
    }
}

fn time(minute: i64) -> NaiveDateTime {
    NaiveDateTime::parse_from_str("2024-01-02 09:30:00", "%Y-%m-%d %H:%M:%S").unwrap()
        + TimeDelta::minutes(minute)
}

fn batch(name: &str, values: Vec<f64>) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            name,
            DataType::Float64,
            false,
        )])),
        vec![Arc::new(Float64Array::from(values)) as ArrayRef],
    )
    .unwrap()
}

fn ipc_unsupported_batch() -> RecordBatch {
    let nested_dictionary = DataType::Dictionary(
        Box::new(DataType::Int32),
        Box::new(DataType::Dictionary(
            Box::new(DataType::Int32),
            Box::new(DataType::Utf8),
        )),
    );
    RecordBatch::new_empty(Arc::new(Schema::new(vec![Field::new(
        "unsupported",
        nested_dictionary,
        false,
    )])))
}

fn metrics() -> SaoeMetrics {
    SaoeMetrics {
        stock_id: "A".to_owned(),
        datetime: SaoeTime::Array(vec![time(0), time(1)]),
        direction: OrderDir::Buy,
        market_volume: SaoeNumeric::Array(arr1(&[10.0, 20.0])),
        market_price: SaoeNumeric::Array(arr1(&[2.0, 3.0])),
        amount: SaoeNumeric::Scalar(4.0),
        inner_amount: SaoeNumeric::Scalar(4.0),
        deal_amount: SaoeNumeric::Array(arr1(&[1.0, 3.0])),
        trade_price: SaoeNumeric::Scalar(2.5),
        trade_value: SaoeNumeric::Scalar(10.0),
        position: SaoeNumeric::Array(arr1(&[3.0, 0.0])),
        ffr: SaoeNumeric::Scalar(1.0),
        pa: SaoeNumeric::Scalar(f64::NAN),
    }
}

fn sample_state(with_metrics: bool) -> SaoeState {
    SaoeState::new(SaoeStateParts {
        order: Order::new("A", 4.0, OrderDir::Buy, Some(time(0)), Some(time(4))),
        cur_time: time(1),
        cur_step: 1,
        position: 3.0,
        history_exec: batch("exec", vec![1.0, 2.0]),
        history_steps: batch("step", vec![3.0]),
        metrics: with_metrics.then(metrics),
        backtest_data: SaoeBacktestData {
            ticks_index: vec![time(0), time(1), time(2)],
            ticks_for_order: vec![time(0), time(1)],
            deal_prices: arr1(&[10.0, 11.0]),
            market_volumes: arr1(&[100.0, 200.0]),
            features: batch("feature", vec![5.0, 6.0, 7.0]),
        },
        ticks_per_step: 2,
        ticks_index: vec![time(0), time(1), time(2)],
        ticks_for_order: vec![time(0), time(1)],
    })
}

fn python_contract() -> Value {
    let output = Command::new("python")
        .args([
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/saoe_contract.py"
            ),
            r"D:\code\github\qlib\qlib\rl\order_execution\state.py",
            r"D:\code\github\qlib\qlib\rl\order_execution\strategy.py",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[derive(Serialize)]
struct WireReplica {
    order: Order,
    cur_time: NaiveDateTime,
    cur_step: i64,
    position: f64,
    history_exec: Vec<u8>,
    history_steps: Vec<u8>,
    metrics: Option<SaoeMetrics>,
    ticks_index: Vec<NaiveDateTime>,
    ticks_for_order: Vec<NaiveDateTime>,
    deal_prices: Vec<f64>,
    market_volumes: Vec<f64>,
    features: Vec<u8>,
    ticks_per_step: usize,
}

#[derive(Serialize)]
struct WireReplicaV1 {
    order: Order,
    cur_time: NaiveDateTime,
    cur_step: i64,
    position: f64,
    history_exec: Vec<u8>,
    history_steps: Vec<u8>,
    metrics: Option<SaoeMetrics>,
    ticks_index: Vec<NaiveDateTime>,
    ticks_for_order: Vec<NaiveDateTime>,
    features: Vec<u8>,
    ticks_per_step: usize,
}

fn empty_ipc() -> Vec<u8> {
    let mut payload = Vec::new();
    let schema = Schema::new(vec![Field::new("x", DataType::Float64, false)]);
    let mut writer = StreamWriter::try_new(&mut payload, &schema).unwrap();
    writer.finish().unwrap();
    drop(writer);
    payload
}

fn encode_test_batch(batch: &RecordBatch) -> Vec<u8> {
    let mut payload = Vec::new();
    let mut writer = StreamWriter::try_new(&mut payload, batch.schema().as_ref()).unwrap();
    writer.write(batch).unwrap();
    writer.finish().unwrap();
    drop(writer);
    payload
}

fn replica() -> WireReplica {
    let encoded = sample_state(false).encode().unwrap();
    let decoded = SaoeState::decode(&encoded).unwrap();
    let state = sample_state(false);
    let valid = encode_test_batch(&state.parts().history_steps);
    WireReplica {
        order: decoded.parts().order.clone(),
        cur_time: decoded.parts().cur_time,
        cur_step: decoded.parts().cur_step,
        position: decoded.parts().position,
        history_exec: valid.clone(),
        history_steps: valid.clone(),
        metrics: None,
        ticks_index: vec![time(0)],
        ticks_for_order: vec![time(0)],
        deal_prices: vec![10.0],
        market_volumes: vec![100.0],
        features: valid,
        ticks_per_step: 1,
    }
}

#[test]
fn owned_state_round_trips_scalar_array_arrow_and_python_field_order() {
    let original = sample_state(true);
    let decoded = SaoeState::decode(&original.encode().unwrap()).unwrap();
    assert_eq!(decoded.parts().order, original.parts().order);
    assert_eq!(decoded.parts().cur_time, time(1));
    assert_eq!(decoded.parts().cur_step, 1);
    assert_eq!(decoded.parts().position.to_bits(), 3.0_f64.to_bits());
    assert_eq!(decoded.parts().history_exec, original.parts().history_exec);
    assert_eq!(
        decoded.parts().history_steps,
        original.parts().history_steps
    );
    assert_eq!(
        decoded.parts().backtest_data.features,
        original.parts().backtest_data.features
    );
    assert_eq!(decoded.parts().backtest_data.ticks_index.len(), 3);
    assert_eq!(
        decoded.parts().backtest_data.deal_prices,
        original.parts().backtest_data.deal_prices
    );
    assert_eq!(
        decoded.parts().backtest_data.market_volumes,
        original.parts().backtest_data.market_volumes
    );
    assert_eq!(decoded.parts().ticks_per_step, 2);
    let legacy_batch = encode_test_batch(&original.parts().history_steps);
    let legacy = WireReplicaV1 {
        order: original.parts().order.clone(),
        cur_time: original.parts().cur_time,
        cur_step: original.parts().cur_step,
        position: original.parts().position,
        history_exec: legacy_batch.clone(),
        history_steps: legacy_batch.clone(),
        metrics: original.parts().metrics.clone(),
        ticks_index: original.parts().backtest_data.ticks_index.clone(),
        ticks_for_order: original.parts().backtest_data.ticks_for_order.clone(),
        features: legacy_batch,
        ticks_per_step: original.parts().ticks_per_step,
    };
    let legacy_decoded = SaoeState::decode(&bincode::serialize(&legacy).unwrap()).unwrap();
    assert!(legacy_decoded.parts().backtest_data.deal_prices.is_empty());
    assert!(
        legacy_decoded
            .parts()
            .backtest_data
            .market_volumes
            .is_empty()
    );
    assert_eq!(legacy_decoded.parts().cur_time, original.parts().cur_time);
    assert_eq!(
        legacy_decoded.parts().ticks_index,
        legacy_decoded.parts().backtest_data.ticks_index
    );
    assert_eq!(
        legacy_decoded.parts().ticks_for_order,
        legacy_decoded.parts().backtest_data.ticks_for_order
    );
    let decoded_metrics = decoded.parts().metrics.as_ref().unwrap();
    assert!(matches!(decoded_metrics.datetime, SaoeTime::Array(_)));
    assert!(matches!(
        decoded_metrics.market_volume,
        SaoeNumeric::Array(_)
    ));
    let SaoeNumeric::Scalar(pa) = decoded_metrics.pa else {
        panic!("expected scalar PA")
    };
    assert!(pa.is_nan());
    let mut scalar_parts = sample_state(true).parts().clone();
    scalar_parts.metrics.as_mut().unwrap().datetime = SaoeTime::Scalar(time(2));
    let scalar_time_state = SaoeState::new(scalar_parts);
    let scalar_time_state = SaoeState::decode(&scalar_time_state.encode().unwrap()).unwrap();
    assert_eq!(
        scalar_time_state.parts().metrics.as_ref().unwrap().datetime,
        SaoeTime::Scalar(time(2))
    );
    let python = python_contract();
    assert_eq!(
        python["fields"],
        json!([
            "order",
            "cur_time",
            "cur_step",
            "position",
            "history_exec",
            "history_steps",
            "metrics",
            "backtest_data",
            "ticks_per_step",
            "ticks_index",
            "ticks_for_order",
        ])
    );
    assert!(
        python["tuple_identity"]
            .as_array()
            .unwrap()
            .iter()
            .all(|value| value.as_bool() == Some(true))
    );
}

#[test]
fn owned_state_round_trip_preserves_rebound_direct_tick_fields() {
    let mut rebound_parts = sample_state(false).parts().clone();
    rebound_parts.ticks_index = vec![time(3), time(4)];
    rebound_parts.ticks_for_order = vec![time(4)];
    let rebound = SaoeState::decode(&SaoeState::new(rebound_parts).encode().unwrap()).unwrap();
    assert_eq!(rebound.parts().backtest_data.ticks_index[0], time(0));
    assert_eq!(rebound.parts().ticks_index, vec![time(3), time(4)]);
    assert_eq!(rebound.parts().ticks_for_order, vec![time(4)]);
}

#[test]
fn state_decode_reports_bincode_arrow_and_empty_stream_failures() {
    assert!(matches!(
        SaoeState::decode(b"bad"),
        Err(SaoeError::Bincode(_))
    ));
    for field in ["history_exec", "history_steps", "features"] {
        let mut wire = replica();
        match field {
            "history_exec" => wire.history_exec = vec![1, 2, 3],
            "history_steps" => wire.history_steps = vec![1, 2, 3],
            "features" => wire.features = vec![1, 2, 3],
            _ => unreachable!(),
        }
        assert!(matches!(
            SaoeState::decode(&bincode::serialize(&wire).unwrap()),
            Err(SaoeError::Arrow(_))
        ));
    }
    let valid_ipc = replica().history_exec;
    let mut saw_truncated_batch_error = false;
    for length in 1..valid_ipc.len() {
        let mut wire = replica();
        wire.history_exec = valid_ipc[..length].to_vec();
        if matches!(
            SaoeState::decode(&bincode::serialize(&wire).unwrap()),
            Err(SaoeError::Arrow(_))
        ) {
            saw_truncated_batch_error = true;
        }
    }
    assert!(saw_truncated_batch_error);
    for field in ["history_exec", "history_steps", "features"] {
        let mut parts = sample_state(false).parts().clone();
        match field {
            "history_exec" => parts.history_exec = ipc_unsupported_batch(),
            "history_steps" => parts.history_steps = ipc_unsupported_batch(),
            "features" => parts.backtest_data.features = ipc_unsupported_batch(),
            _ => unreachable!(),
        }
        assert!(matches!(
            SaoeState::new(parts).encode(),
            Err(SaoeError::Arrow(_))
        ));
    }
    for field in ["history_exec", "history_steps", "features"] {
        let mut wire = replica();
        match field {
            "history_exec" => wire.history_exec = empty_ipc(),
            "history_steps" => wire.history_steps = empty_ipc(),
            "features" => wire.features = empty_ipc(),
            _ => unreachable!(),
        }
        assert!(matches!(
            SaoeState::decode(&bincode::serialize(&wire).unwrap()),
            Err(SaoeError::MissingBatch)
        ));
    }
    assert!(
        SaoeState::decode(&sample_state(false).encode().unwrap())
            .unwrap()
            .parts()
            .metrics
            .is_none()
    );
}

#[derive(Default)]
struct Log {
    events: Vec<String>,
    fail: Option<&'static str>,
    range: (i64, i64),
    update_lengths: Vec<usize>,
}

type SharedLog = Arc<Mutex<Log>>;

fn plugin(log: &SharedLog, stage: &'static str) -> Result<(), SaoePluginError> {
    let mut log = log.lock().unwrap();
    log.events.push(stage.to_owned());
    if log.fail == Some(stage) {
        return Err(SaoePluginError {
            message: stage.to_owned(),
        });
    }
    Ok(())
}

struct Calendar(SharedLog);

impl SaoeCalendar for Calendar {
    fn available_step_range(&self) -> Result<(i64, i64), SaoePluginError> {
        plugin(&self.0, "range")?;
        Ok(self.0.lock().unwrap().range)
    }

    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), SaoePluginError> {
        plugin(&self.0, "time")?;
        Ok((time(2), time(3)))
    }
}

struct States {
    log: SharedLog,
    state: SaoeState,
}

impl SaoeStateProvider for States {
    fn reset(&mut self, _order: &Order) -> Result<(), SaoePluginError> {
        plugin(&self.log, "reset")
    }

    fn state(&self, _order: &Order) -> Result<SaoeState, SaoePluginError> {
        plugin(&self.log, "state")?;
        Ok(self.state.clone())
    }

    fn update(
        &mut self,
        executions: &[SharedOrderExecution],
        _step_range: (i64, i64),
    ) -> Result<(), SaoePluginError> {
        plugin(&self.log, "update")?;
        self.log
            .lock()
            .unwrap()
            .update_lengths
            .push(executions.len());
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), SaoePluginError> {
        plugin(&self.log, "finalize")
    }
}

struct Factory(SharedLog);

impl SaoeOrderFactory for Factory {
    fn create(
        &mut self,
        stock_id: &str,
        amount: Option<f64>,
        direction: OrderDir,
    ) -> Result<Order, SaoePluginError> {
        plugin(&self.0, "create")?;
        let amount = amount.ok_or_else(|| SaoePluginError {
            message: "missing amount".to_owned(),
        })?;
        Ok(Order::new(stock_id, amount, direction, None, None))
    }
}

struct Outer(OrderTradeDecision);

impl NestedOuterDecision for Outer {
    fn order_decision(&self) -> &dyn OrderDecision {
        &self.0
    }

    fn order_decision_mut(&mut self) -> &mut dyn OrderDecision {
        &mut self.0
    }

    fn update(
        &mut self,
        _calendar: &dyn NestedCalendar,
    ) -> Result<NestedDecisionUpdate, NestedOuterDecisionError> {
        Ok(NestedDecisionUpdate::Unchanged)
    }

    fn is_empty(&self) -> Result<bool, NestedOuterDecisionError> {
        Ok(false)
    }

    fn range_limit(
        &self,
        _calendar: &dyn NestedCalendar,
    ) -> Result<Option<(i64, i64)>, NestedOuterDecisionError> {
        Ok(None)
    }

    fn modify_inner_decision(
        &self,
        _decision: &mut dyn OrderDecision,
    ) -> Result<(), NestedOuterDecisionError> {
        Ok(())
    }
}

struct RawDecision {
    orders: Vec<Order>,
}

impl OrderDecision for RawDecision {
    fn orders(&self) -> &[Order] {
        &self.orders
    }

    fn orders_mut(&mut self) -> &mut [Order] {
        &mut self.orders
    }

    fn start_time(&self) -> NaiveDateTime {
        time(0)
    }

    fn end_time(&self) -> NaiveDateTime {
        time(9)
    }

    fn trade_range(&self) -> Option<&dyn TradeRange> {
        None
    }
}

struct RawOuter(RawDecision);

impl NestedOuterDecision for RawOuter {
    fn order_decision(&self) -> &dyn OrderDecision {
        &self.0
    }

    fn order_decision_mut(&mut self) -> &mut dyn OrderDecision {
        &mut self.0
    }

    fn update(
        &mut self,
        _calendar: &dyn NestedCalendar,
    ) -> Result<NestedDecisionUpdate, NestedOuterDecisionError> {
        Ok(NestedDecisionUpdate::Unchanged)
    }

    fn is_empty(&self) -> Result<bool, NestedOuterDecisionError> {
        Ok(false)
    }

    fn range_limit(
        &self,
        _calendar: &dyn NestedCalendar,
    ) -> Result<Option<(i64, i64)>, NestedOuterDecisionError> {
        Ok(None)
    }

    fn modify_inner_decision(
        &self,
        _decision: &mut dyn OrderDecision,
    ) -> Result<(), NestedOuterDecisionError> {
        Ok(())
    }
}

fn outer(orders: Vec<Order>) -> Outer {
    Outer(OrderTradeDecision::from_orders(
        orders,
        time(0),
        time(9),
        None,
    ))
}

fn execution(order: Order) -> SharedOrderExecution {
    OwnedOrderExecution {
        order,
        trade_value: 10.0,
        trade_cost: 0.0,
        trade_price: 2.0,
    }
    .into_shared()
}

fn strategy_error<T>(result: Result<T, domain_core::NestedStrategyError>) -> String {
    match result {
        Ok(_) => panic!("expected strategy failure"),
        Err(error) => error.message,
    }
}

#[test]
fn proxy_close_preserves_state_and_range_without_finalizing_or_creating_orders() {
    let log = Arc::new(Mutex::new(Log {
        range: (3, 5),
        ..Log::default()
    }));
    let mut strategy = ProxySaoeStrategy::new(
        Arc::new(Calendar(Arc::clone(&log))),
        Box::new(States {
            log: Arc::clone(&log),
            state: sample_state(true),
        }),
        Box::new(Factory(Arc::clone(&log))),
    );
    strategy.close_trade_decision().unwrap();
    assert!(matches!(
        strategy.current_state(),
        Err(SaoeError::MissingOuterOrder)
    ));
    strategy
        .reset(&outer(vec![Order::new(
            "A",
            4.0,
            OrderDir::Buy,
            Some(time(0)),
            Some(time(9)),
        )]))
        .unwrap();
    strategy.close_trade_decision().unwrap();
    for _ in 0..2 {
        assert!(matches!(
            strategy.begin_trade_decision(None).unwrap(),
            NestedStrategyProgress::Suspended(_)
        ));
        let events_before = log.lock().unwrap().events.clone();
        strategy.close_trade_decision().unwrap();
        strategy.close_trade_decision().unwrap();
        assert_eq!(log.lock().unwrap().events, events_before);
        assert_eq!(strategy.last_step_range(), (3, 5));
        assert_eq!(strategy.current_state().unwrap().parts().cur_step, 1);
        assert!(
            strategy_error(strategy.resume_trade_decision(Some(1.0))).contains("not suspended")
        );
    }
}

#[test]
fn proxy_strategy_preserves_prompt_action_factory_grouping_and_python_order() {
    let log = Arc::new(Mutex::new(Log {
        range: (3, 5),
        ..Log::default()
    }));
    let calendar = Arc::new(Calendar(Arc::clone(&log)));
    let states = Box::new(States {
        log: Arc::clone(&log),
        state: sample_state(true),
    });
    let factory = Box::new(Factory(Arc::clone(&log)));
    let mut strategy = ProxySaoeStrategy::new(calendar, states, factory);
    assert_eq!(strategy.last_step_range(), (0, 0));
    assert!(matches!(
        strategy.current_state(),
        Err(SaoeError::MissingOuterOrder)
    ));
    assert!(strategy_error(strategy.generate_trade_decision(None)).contains("resumable"));
    assert!(strategy_error(strategy.begin_trade_decision(None)).contains("not been reset"));
    assert!(strategy_error(strategy.resume_trade_decision(Some(1.0))).contains("not suspended"));
    let mut outer = outer(vec![Order::new(
        "A",
        4.0,
        OrderDir::Buy,
        Some(time(0)),
        Some(time(9)),
    )]);
    strategy.reset(&outer).unwrap();
    assert_eq!(strategy.last_step_range(), (0, 0));
    strategy.alter_outer_decision(&mut outer).unwrap();
    assert_eq!(strategy.current_state().unwrap().parts().cur_step, 1);
    let NestedStrategyProgress::Suspended(prompt) = strategy.begin_trade_decision(None).unwrap()
    else {
        panic!("expected proxy prompt")
    };
    assert_eq!(strategy.last_step_range(), (3, 5));
    assert_eq!(prompt.kind, SAOE_PROXY_PROMPT_KIND);
    assert_eq!(prompt.schema_version, SAOE_STATE_SCHEMA_VERSION);
    let locator = ProxySaoeStrategy::decode_prompt(&prompt).unwrap();
    assert_eq!(locator.stock_id, "A");
    assert_eq!(locator.day, time(0).date().and_hms_opt(0, 0, 0).unwrap());
    assert_eq!(locator.direction, OrderDir::Buy);
    assert_eq!(locator.step_range, (3, 5));
    assert!(strategy_error(strategy.begin_trade_decision(None)).contains("already suspended"));
    let NestedStrategyProgress::Ready(decision) =
        strategy.resume_trade_decision(Some(7.5)).unwrap()
    else {
        panic!("expected proxy decision")
    };
    assert_eq!(decision.orders()[0].amount().to_bits(), 7.5_f64.to_bits());
    assert_eq!(decision.orders()[0].start_time(), Some(time(2)));
    assert_eq!(decision.orders()[0].end_time(), Some(time(3)));
    let matching = execution(Order::new(
        "A",
        1.0,
        OrderDir::Buy,
        Some(time(4)),
        Some(time(5)),
    ));
    let foreign = execution(Order::new(
        "B",
        1.0,
        OrderDir::Buy,
        Some(time(4)),
        Some(time(5)),
    ));
    strategy.post_execute(&[matching, foreign]).unwrap();
    strategy.post_upper_level().unwrap();
    assert_eq!(log.lock().unwrap().update_lengths, [1]);
    assert_eq!(
        log.lock().unwrap().events,
        [
            "reset", "state", "range", "create", "time", "update", "finalize"
        ]
    );
    let python = python_contract();
    assert_eq!(python["events"][1], json!(["range", "step"]));
    assert_eq!(python["events"][3], json!(["helper"]));
    assert_eq!(python["events"][4], json!(["create", "A", 7.5, 1]));
}

#[test]
fn proxy_validates_outer_prompt_zero_range_and_missing_execution_times() {
    let log = Arc::new(Mutex::new(Log::default()));
    let calendar = Arc::new(Calendar(Arc::clone(&log)));
    let states = Box::new(States {
        log: Arc::clone(&log),
        state: sample_state(false),
    });
    let factory = Box::new(Factory(Arc::clone(&log)));
    let mut strategy = ProxySaoeStrategy::new(calendar, states, factory);
    let missing_day = RawOuter(RawDecision {
        orders: vec![Order::new("A", 1.0, OrderDir::Buy, None, None)],
    });
    assert!(strategy_error(strategy.reset(&missing_day)).contains("start time"));
    assert!(strategy_error(strategy.reset(&outer(Vec::new()))).contains("got 0"));
    assert!(
        strategy_error(strategy.reset(&outer(vec![
            sample_state(false).parts().order.clone(),
            sample_state(false).parts().order.clone(),
        ])))
        .contains("got 2")
    );
    let single = outer(vec![sample_state(false).parts().order.clone()]);
    strategy.reset(&single).unwrap();
    let prompt = NestedStrategyPrompt {
        kind: "wrong".to_owned(),
        schema_version: 1,
        payload: Vec::new(),
    };
    assert!(matches!(
        ProxySaoeStrategy::decode_prompt(&prompt),
        Err(SaoeError::Plugin(_))
    ));
    let prompt = NestedStrategyPrompt {
        kind: SAOE_PROXY_PROMPT_KIND.to_owned(),
        schema_version: 2,
        payload: Vec::new(),
    };
    assert!(matches!(
        ProxySaoeStrategy::decode_prompt(&prompt),
        Err(SaoeError::Plugin(_))
    ));
    let prompt = NestedStrategyPrompt {
        kind: SAOE_PROXY_PROMPT_KIND.to_owned(),
        schema_version: 1,
        payload: b"bad".to_vec(),
    };
    assert!(matches!(
        ProxySaoeStrategy::decode_prompt(&prompt),
        Err(SaoeError::Json(_))
    ));
    strategy.begin_trade_decision(None).unwrap();
    strategy.resume_trade_decision(Some(1.0)).unwrap();
    strategy.post_execute(&[]).unwrap();
    let valid = execution(sample_state(false).parts().order.clone());
    assert!(strategy_error(strategy.post_execute(&[valid])).contains("zero-length"));
}

#[test]
fn proxy_propagates_plugin_failures_and_covers_every_phase_edge() {
    let make_log = |fail, range| {
        Arc::new(Mutex::new(Log {
            fail,
            range,
            ..Log::default()
        }))
    };
    let sole_outer = || outer(vec![sample_state(false).parts().order.clone()]);

    for stage in [
        "reset", "state", "range", "create", "time", "update", "finalize",
    ] {
        let log = make_log(Some(stage), (1, 2));
        let calendar = Arc::new(Calendar(Arc::clone(&log)));
        let states = Box::new(States {
            log: Arc::clone(&log),
            state: sample_state(false),
        });
        let factory = Box::new(Factory(Arc::clone(&log)));
        let mut strategy = ProxySaoeStrategy::new(calendar, states, factory);
        let failure = match stage {
            "reset" => strategy_error(strategy.reset(&sole_outer())),
            "state" => {
                strategy.reset(&sole_outer()).unwrap();
                match strategy.current_state() {
                    Ok(_) => panic!("expected state plugin failure"),
                    Err(error) => error.to_string(),
                }
            }
            "range" => {
                strategy.reset(&sole_outer()).unwrap();
                strategy_error(strategy.begin_trade_decision(None))
            }
            "create" | "time" => {
                strategy.reset(&sole_outer()).unwrap();
                strategy.begin_trade_decision(None).unwrap();
                strategy_error(strategy.resume_trade_decision(Some(1.0)))
            }
            "update" => {
                strategy.reset(&sole_outer()).unwrap();
                strategy.begin_trade_decision(None).unwrap();
                assert_eq!(strategy.current_state().unwrap().parts().cur_step, 1);
                strategy_error(strategy.post_execute(&[]))
            }
            "finalize" => strategy_error(strategy.post_upper_level()),
            _ => unreachable!(),
        };
        assert!(failure.contains(stage), "{stage}: {failure}");
    }

    let log = make_log(None, (1, 2));
    let calendar = Arc::new(Calendar(Arc::clone(&log)));
    let states = Box::new(States {
        log: Arc::clone(&log),
        state: sample_state(false),
    });
    let factory = Box::new(Factory(Arc::clone(&log)));
    let mut strategy = ProxySaoeStrategy::new(calendar, states, factory);
    assert_eq!(strategy.last_step_range(), (0, 0));
    strategy.post_execute(&[]).unwrap();
    let valid = execution(sample_state(false).parts().order.clone());
    assert!(strategy_error(strategy.post_execute(&[Arc::clone(&valid)])).contains("zero-length"));
    strategy.reset(&sole_outer()).unwrap();
    strategy.begin_trade_decision(None).unwrap();
    assert!(strategy_error(strategy.resume_trade_decision(None)).contains("missing amount"));
    let malformed = execution(Order::new("A", 1.0, OrderDir::Buy, None, None));
    assert!(strategy_error(strategy.post_execute(&[malformed])).contains("start time"));
    let events = log.lock().unwrap().events.clone();
    assert!(
        std::panic::catch_unwind(|| {
            let _guard = valid.order.write().unwrap();
            panic!("poison execution order");
        })
        .is_err()
    );
    assert!(
        strategy_error(strategy.post_execute(&[valid])).contains("execution order lock poisoned")
    );
    assert_eq!(log.lock().unwrap().events, events);
}
