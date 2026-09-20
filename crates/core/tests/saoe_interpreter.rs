use std::{
    process::Command,
    sync::{Arc, Mutex},
};

use arrow_array::{Float32Array, Float64Array, Int32Array, RecordBatch};
use arrow_schema::Schema;
use chrono::{NaiveDateTime, TimeDelta};
use domain_core::{
    AllOnePolicy, CategoricalActionInterpreter, CurrentStepObservation,
    CurrentStepStateInterpreter, DummyStateInterpreter, FullHistoryObservation,
    FullHistoryStateInterpreter, Order, OrderDir, ProcessedSaoeData, SaoeActionInterpreter,
    SaoeActionSpace, SaoeBacktestData, SaoeInterpreterError, SaoeObservation, SaoePolicy,
    SaoePolicyAction, SaoePolicyPipeline, SaoeProcessedDataProvider, SaoeState,
    SaoeStateInterpreter, SaoeStateParts, TwapRelativeActionInterpreter,
};
use ndarray::{Array2, arr1};
use serde_json::{Value, json};

#[path = "support/saoe_pull_pipeline.rs"]
mod pull_pipeline;

fn time(minute: i64) -> NaiveDateTime {
    NaiveDateTime::parse_from_str("2024-01-02 09:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        + TimeDelta::minutes(minute)
}

fn state(
    direction: OrderDir,
    cur_step: i64,
    position: f64,
    tick_count: usize,
    ticks_per_step: usize,
) -> SaoeState {
    let ticks: Vec<_> = (0..tick_count)
        .map(|index| time(i64::try_from(index).unwrap()))
        .collect();
    let empty = RecordBatch::new_empty(Arc::new(Schema::empty()));
    SaoeState::new(SaoeStateParts {
        order: Order::new("A", 10.0, direction, Some(time(0)), Some(time(9))),
        cur_time: time(0),
        cur_step,
        position,
        history_exec: empty.clone(),
        history_steps: empty.clone(),
        metrics: None,
        backtest_data: SaoeBacktestData {
            ticks_index: ticks.clone(),
            ticks_for_order: ticks.clone(),
            deal_prices: arr1(&[]),
            market_volumes: arr1(&[]),
            features: empty,
        },
        ticks_per_step,
        ticks_index: ticks.clone(),
        ticks_for_order: ticks,
    })
}

fn full_state(
    direction: OrderDir,
    cur_step: i64,
    cur_time: NaiveDateTime,
    history_steps: RecordBatch,
    start_time: Option<NaiveDateTime>,
) -> SaoeState {
    let ticks: Vec<_> = (0..5).map(time).collect();
    let empty = RecordBatch::new_empty(Arc::new(Schema::empty()));
    SaoeState::new(SaoeStateParts {
        order: Order::new("A", 10.0, direction, start_time, Some(time(9))),
        cur_time,
        cur_step,
        position: 4.0,
        history_exec: empty.clone(),
        history_steps,
        metrics: None,
        backtest_data: SaoeBacktestData {
            ticks_index: ticks.clone(),
            ticks_for_order: ticks.clone(),
            deal_prices: arr1(&[]),
            market_volumes: arr1(&[]),
            features: empty,
        },
        ticks_per_step: 2,
        ticks_index: ticks.clone(),
        ticks_for_order: ticks,
    })
}

fn positions_f64(values: Vec<f64>) -> RecordBatch {
    RecordBatch::try_from_iter([(
        "position",
        Arc::new(Float64Array::from(values)) as Arc<dyn arrow_array::Array>,
    )])
    .unwrap()
}

#[derive(Clone, Debug, PartialEq)]
struct ProviderCall {
    stock_id: String,
    date: chrono::NaiveDate,
    feature_dim: usize,
    ticks: usize,
}

struct Provider {
    data: ProcessedSaoeData,
    failure: Option<String>,
    calls: Arc<Mutex<Vec<ProviderCall>>>,
}

struct FixedPolicy {
    actions: Vec<SaoePolicyAction>,
    fail: bool,
    observations: Arc<Mutex<Vec<SaoeObservation>>>,
}

impl SaoePolicy for FixedPolicy {
    fn actions(
        &mut self,
        observations: &[SaoeObservation],
    ) -> Result<Vec<SaoePolicyAction>, SaoeInterpreterError> {
        self.observations
            .lock()
            .unwrap()
            .extend_from_slice(observations);
        if self.fail {
            return Err(SaoeInterpreterError::ProcessedDataPlugin(
                "policy".to_owned(),
            ));
        }
        Ok(self.actions.clone())
    }
}

impl SaoeProcessedDataProvider for Provider {
    fn get_data(
        &self,
        stock_id: &str,
        date: chrono::NaiveDate,
        feature_dim: usize,
        time_index: &[NaiveDateTime],
    ) -> Result<ProcessedSaoeData, SaoeInterpreterError> {
        self.calls.lock().unwrap().push(ProviderCall {
            stock_id: stock_id.to_owned(),
            date,
            feature_dim,
            ticks: time_index.len(),
        });
        if let Some(message) = &self.failure {
            return Err(SaoeInterpreterError::ProcessedDataPlugin(message.clone()));
        }
        Ok(self.data.clone())
    }
}

#[allow(clippy::cast_precision_loss)]
fn processed(rows: usize, columns: usize, index_rows: usize) -> ProcessedSaoeData {
    let values: Vec<_> = (0..rows * columns).map(|value| value as f32).collect();
    ProcessedSaoeData {
        today: Array2::from_shape_vec((rows, columns), values).unwrap(),
        yesterday: Array2::from_shape_vec(
            (rows, columns),
            (10..10 + rows * columns)
                .map(|value| value as f32)
                .collect(),
        )
        .unwrap(),
        today_index: (0..index_rows)
            .map(|index| time(i64::try_from(index).unwrap()))
            .collect(),
    }
}

fn full_interpreter(
    data: ProcessedSaoeData,
    failure: Option<&str>,
    calls: Arc<Mutex<Vec<ProviderCall>>>,
) -> FullHistoryStateInterpreter {
    FullHistoryStateInterpreter::new(
        3,
        5,
        2,
        Arc::new(Provider {
            data,
            failure: failure.map(str::to_owned),
            calls,
        }),
    )
    .unwrap()
}

#[test]
fn environment_interpreter_adapters_preserve_full_observations_and_failures() {
    use domain_core::{
        EnvironmentStateInterpreter, EnvironmentStatus, SaoeEnvironmentStateInterpreter,
    };

    let snapshot = full_state(
        OrderDir::Buy,
        1,
        time(2),
        positions_f64(vec![4.0]),
        Some(time(0)),
    );
    let calls = Arc::new(Mutex::new(Vec::new()));
    let interpreter = full_interpreter(processed(5, 2, 5), None, calls.clone());
    let expected = SaoeStateInterpreter::interpret(&interpreter, &snapshot).unwrap();
    let mut adapter = SaoeEnvironmentStateInterpreter::new(Box::new(interpreter));
    let mut status = EnvironmentStatus::<String, SaoeObservation, SaoePolicyAction>::empty(Some(
        "original seed".to_owned(),
    ));
    status.cur_step = 99.into();
    status.done = true;
    let before = status.clone();
    let actual = adapter.interpret(&snapshot, &status).unwrap();
    assert_eq!(actual, expected);
    let SaoeObservation::FullHistory(full) = actual else {
        panic!("full observation lost")
    };
    assert_eq!(full.data_processed_prev, processed(5, 2, 5).yesterday);
    assert_eq!(full.position.to_bits(), 4.0_f32.to_bits());
    assert_eq!(status, before);
    assert_eq!(calls.lock().unwrap().len(), 2);

    let mut failing = SaoeEnvironmentStateInterpreter::new(Box::new(full_interpreter(
        processed(5, 2, 5),
        Some("provider unavailable"),
        calls.clone(),
    )));
    let error = failing.interpret(&snapshot, &status).unwrap_err();
    assert_eq!(
        error.message,
        SaoeInterpreterError::ProcessedDataPlugin("provider unavailable".to_owned()).to_string()
    );
    assert!(!error.is_stop_iteration());
    assert_eq!(status, before);
    assert_eq!(calls.lock().unwrap().len(), 3);
}

#[test]
fn environment_action_adapter_delegates_both_action_types_without_mutating_history() {
    use domain_core::{
        EnvironmentActionInterpreter, EnvironmentStatus, SaoeEnvironmentActionInterpreter,
    };

    let snapshot = state(OrderDir::Buy, 1, 6.0, 10, 2);
    let status = EnvironmentStatus::<(), SaoeObservation, SaoePolicyAction>::empty(None);
    let before = status.clone();
    let categorical = CategoricalActionInterpreter::from_count(3, Some(5)).unwrap();
    let twap = TwapRelativeActionInterpreter;
    let cases: Vec<(Box<dyn SaoeActionInterpreter>, Vec<SaoePolicyAction>)> = vec![
        (
            Box::new(categorical),
            vec![
                SaoePolicyAction::Discrete(1),
                SaoePolicyAction::Discrete(99),
                SaoePolicyAction::Continuous(1.0),
            ],
        ),
        (
            Box::new(twap),
            vec![
                SaoePolicyAction::Continuous(0.5),
                SaoePolicyAction::Continuous(-1.0),
                SaoePolicyAction::Discrete(1),
            ],
        ),
    ];
    for (interpreter, actions) in cases {
        let expected: Vec<_> = actions
            .iter()
            .map(|action| interpreter.interpret(&snapshot, *action))
            .collect();
        let mut adapter = SaoeEnvironmentActionInterpreter::new(interpreter);
        for (action, expected) in actions.iter().zip(expected) {
            let actual = adapter.interpret(&snapshot, action, &status);
            match expected {
                Ok(volume) => exact(actual.unwrap(), volume),
                Err(error) => {
                    let actual = actual.unwrap_err();
                    assert_eq!(actual.message, error.to_string());
                    assert!(!actual.is_stop_iteration());
                }
            }
            assert_eq!(status, before);
        }
    }
}

fn python_contract() -> Value {
    let output = Command::new("python")
        .args([
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/saoe_interpreter_contract.py"
            ),
            r"D:\code\github\qlib\qlib\rl\order_execution\interpreter.py",
            r"D:\code\github\qlib\qlib\rl\order_execution\policy.py",
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

fn exact(actual: f64, expected: f64) {
    assert_eq!(actual.to_bits(), expected.to_bits());
}

#[test]
fn simple_state_interpreters_match_python_and_validate_episode_bounds() {
    let buy = state(OrderDir::Buy, 0, 7.0, 5, 2);
    assert_eq!(
        DummyStateInterpreter.interpret(&buy).unwrap(),
        SaoeObservation::Dummy { dummy: 1 }
    );
    assert_eq!(
        CurrentStepStateInterpreter::new(0).unwrap_err(),
        SaoeInterpreterError::InvalidMaxStep(0)
    );
    let interpreter = CurrentStepStateInterpreter::new(3).unwrap();
    let sell = state(OrderDir::Sell, 2, 7.0, 5, 2);
    assert_eq!(
        interpreter.interpret(&sell).unwrap(),
        SaoeObservation::CurrentStep(CurrentStepObservation {
            acquiring: false,
            cur_step: 2,
            num_step: 3,
            target: 10.0,
            position: 7.0,
        })
    );
    assert_eq!(
        interpreter.interpret(&state(OrderDir::Buy, 4, 1.0, 1, 1)),
        Err(SaoeInterpreterError::InvalidMaxStep(3))
    );
    let python = python_contract();
    assert_eq!(python["dummy"], 1);
    assert_eq!(
        python["current"],
        json!({
            "acquiring": false,
            "cur_step": 2,
            "num_step": 3,
            "target": 10.0,
            "position": 7.0,
        })
    );
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn full_history_interpreter_matches_python_masks_shapes_dtypes_and_plugins() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let interpreter = full_interpreter(processed(5, 2, 5), None, Arc::clone(&calls));
    let state = full_state(
        OrderDir::Sell,
        4,
        time(2),
        positions_f64(vec![7.0, 4.0]),
        Some(time(0)),
    );
    let SaoeObservation::FullHistory(observation) = interpreter.interpret(&state).unwrap() else {
        panic!("expected full-history observation")
    };
    assert_eq!(
        observation,
        FullHistoryObservation {
            data_processed: Array2::from_shape_vec(
                (5, 2),
                vec![0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            )
            .unwrap(),
            data_processed_prev: Array2::from_shape_vec(
                (5, 2),
                (10..20).map(|value| value as f32).collect(),
            )
            .unwrap(),
            acquiring: 0,
            cur_tick: 2,
            cur_step: 2,
            num_step: 3,
            target: 10.0,
            position: 4.0,
            position_history: ndarray::arr1(&[10.0, 7.0, 4.0]),
        }
    );
    assert_eq!(
        calls.lock().unwrap().as_slice(),
        [ProviderCall {
            stock_id: "A".to_owned(),
            date: time(0).date(),
            feature_dim: 2,
            ticks: 5,
        }]
    );
    let python = python_contract();
    assert_eq!(
        python["full"],
        json!({
            "data_processed": [[0.0, 1.0], [2.0, 3.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            "data_processed_prev": [[10.0, 11.0], [12.0, 13.0], [14.0, 15.0], [16.0, 17.0], [18.0, 19.0]],
            "acquiring": 0,
            "cur_tick": 2,
            "cur_step": 2,
            "num_step": 3,
            "target": 10.0,
            "position": 4.0,
            "position_history": [10.0, 7.0, 4.0],
        })
    );
    assert_eq!(
        python["full_dtypes"],
        json!({
            "data_processed": "float32",
            "data_processed_prev": "float32",
            "acquiring": "int32",
            "cur_tick": "int32",
            "cur_step": "int32",
            "num_step": "int32",
            "target": "float32",
            "position": "float32",
            "position_history": "float32",
        })
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn full_history_interpreter_rejects_every_dimension_history_and_provider_failure() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    for (max_step, data_ticks, data_dim, name, value) in [
        (0, 5, 2, "max_step", 0),
        (3, 0, 2, "data_ticks", 0),
        (3, 5, 0, "data_dim", 0),
        (
            usize::try_from(i32::MAX).unwrap() + 1,
            5,
            2,
            "max_step",
            usize::try_from(i32::MAX).unwrap() + 1,
        ),
    ] {
        let error = FullHistoryStateInterpreter::new(
            max_step,
            data_ticks,
            data_dim,
            Arc::new(Provider {
                data: processed(5, 2, 5),
                failure: None,
                calls: Arc::clone(&calls),
            }),
        )
        .err()
        .unwrap();
        assert_eq!(
            error,
            SaoeInterpreterError::InvalidObservationDimension { name, value }
        );
    }
    let valid = full_state(
        OrderDir::Buy,
        1,
        time(9),
        positions_f64(vec![7.0]),
        Some(time(0)),
    );
    assert_eq!(
        full_interpreter(processed(5, 2, 5), Some("offline"), Arc::clone(&calls)).interpret(&valid),
        Err(SaoeInterpreterError::ProcessedDataPlugin(
            "offline".to_owned()
        ))
    );
    assert_eq!(
        full_interpreter(processed(4, 2, 5), None, Arc::clone(&calls)).interpret(&valid),
        Err(SaoeInterpreterError::ProcessedShape {
            dataset: "today",
            expected: (5, 2),
            actual: (4, 2),
        })
    );
    let mut wrong_yesterday = processed(5, 2, 5);
    wrong_yesterday.yesterday = Array2::zeros((5, 1));
    assert_eq!(
        full_interpreter(wrong_yesterday, None, Arc::clone(&calls)).interpret(&valid),
        Err(SaoeInterpreterError::ProcessedShape {
            dataset: "yesterday",
            expected: (5, 2),
            actual: (5, 1),
        })
    );
    assert_eq!(
        full_interpreter(processed(5, 2, 4), None, Arc::clone(&calls)).interpret(&valid),
        Err(SaoeInterpreterError::ProcessedIndexLength {
            expected: 5,
            actual: 4,
        })
    );
    let no_start = full_state(OrderDir::Buy, 0, time(0), positions_f64(vec![]), None);
    assert_eq!(
        full_interpreter(processed(5, 2, 5), None, Arc::clone(&calls)).interpret(&no_start),
        Err(SaoeInterpreterError::MissingOrderStartTime)
    );
    let missing = full_state(
        OrderDir::Buy,
        0,
        time(0),
        RecordBatch::new_empty(Arc::new(Schema::empty())),
        Some(time(0)),
    );
    assert_eq!(
        full_interpreter(processed(5, 2, 5), None, Arc::clone(&calls)).interpret(&missing),
        Err(SaoeInterpreterError::MissingHistoryPosition)
    );
    let invalid_type = RecordBatch::try_from_iter([(
        "position",
        Arc::new(Int32Array::from(vec![1])) as Arc<dyn arrow_array::Array>,
    )])
    .unwrap();
    assert_eq!(
        full_interpreter(processed(5, 2, 5), None, Arc::clone(&calls)).interpret(&full_state(
            OrderDir::Buy,
            0,
            time(0),
            invalid_type,
            Some(time(0)),
        )),
        Err(SaoeInterpreterError::InvalidHistoryPositionType(
            "Int32".to_owned()
        ))
    );
    let nulls = RecordBatch::try_from_iter([(
        "position",
        Arc::new(Float64Array::from(vec![Some(1.0), None])) as Arc<dyn arrow_array::Array>,
    )])
    .unwrap();
    assert_eq!(
        full_interpreter(processed(5, 2, 5), None, Arc::clone(&calls)).interpret(&full_state(
            OrderDir::Buy,
            0,
            time(0),
            nulls,
            Some(time(0)),
        )),
        Err(SaoeInterpreterError::NullHistoryPosition)
    );
    assert_eq!(
        full_interpreter(processed(5, 2, 5), None, Arc::clone(&calls)).interpret(&full_state(
            OrderDir::Buy,
            0,
            time(0),
            positions_f64(vec![1.0, 2.0, 3.0, 4.0]),
            Some(time(0)),
        )),
        Err(SaoeInterpreterError::HistoryTooLong {
            actual: 4,
            max_step: 3,
        })
    );
    let float32 = RecordBatch::try_from_iter([(
        "position",
        Arc::new(Float32Array::from(vec![8.0])) as Arc<dyn arrow_array::Array>,
    )])
    .unwrap();
    let SaoeObservation::FullHistory(unclipped) = full_interpreter(processed(5, 2, 5), None, calls)
        .interpret(&full_state(
            OrderDir::Buy,
            1,
            time(9),
            float32,
            Some(time(0)),
        ))
        .unwrap()
    else {
        panic!("expected float32 full-history observation")
    };
    assert_eq!(unclipped.acquiring, 1);
    assert_eq!(unclipped.cur_tick, 4);
    assert_eq!(unclipped.cur_step, 1);
    assert_eq!(unclipped.position_history, ndarray::arr1(&[10.0, 8.0, 0.0]));
}

#[test]
fn categorical_actions_match_python_grid_caps_final_step_and_reject_types() {
    assert_eq!(
        CategoricalActionInterpreter::from_count(0, None).unwrap_err(),
        SaoeInterpreterError::EmptyCategoricalCount
    );
    assert_eq!(
        CategoricalActionInterpreter::from_values(vec![], None).unwrap_err(),
        SaoeInterpreterError::EmptyCategoricalValues
    );
    assert_eq!(
        CategoricalActionInterpreter::from_values(vec![0.0], Some(0)).unwrap_err(),
        SaoeInterpreterError::InvalidMaxStep(0)
    );
    let generated = CategoricalActionInterpreter::from_count(4, None).unwrap();
    assert_eq!(
        generated.action_space(),
        SaoeActionSpace::Discrete { size: 5 }
    );
    let current = state(OrderDir::Buy, 0, 7.0, 5, 2);
    exact(
        generated
            .interpret(&current, SaoePolicyAction::Discrete(2))
            .unwrap(),
        5.0,
    );
    exact(
        generated
            .interpret(&current, SaoePolicyAction::Discrete(4))
            .unwrap(),
        7.0,
    );
    assert_eq!(
        generated.interpret(&current, SaoePolicyAction::Continuous(1.0)),
        Err(SaoeInterpreterError::ExpectedDiscrete)
    );
    assert_eq!(
        generated.interpret(&current, SaoePolicyAction::Discrete(-1)),
        Err(SaoeInterpreterError::DiscreteOutOfRange {
            action: -1,
            size: 5,
        })
    );
    assert_eq!(
        generated.interpret(&current, SaoePolicyAction::Discrete(5)),
        Err(SaoeInterpreterError::DiscreteOutOfRange { action: 5, size: 5 })
    );
    let final_step =
        CategoricalActionInterpreter::from_values(vec![0.0, 0.25, 0.5, 1.0], Some(3)).unwrap();
    exact(
        final_step
            .interpret(
                &state(OrderDir::Buy, 2, 7.0, 5, 2),
                SaoePolicyAction::Discrete(0),
            )
            .unwrap(),
        7.0,
    );
    let python = python_contract();
    assert_eq!(
        python["categorical_values"],
        json!([0.0, 0.25, 0.5, 0.75, 1.0])
    );
    assert_eq!(python["categorical"], json!([5.0, 7.0, 7.0]));
    assert_eq!(python["categorical_error"], "AssertionError");
}

#[test]
fn twap_actions_and_all_one_policy_match_python_and_cover_protocol_edges() {
    let interpreter = TwapRelativeActionInterpreter;
    assert_eq!(
        interpreter.action_space(),
        SaoeActionSpace::NonNegativeContinuous
    );
    let current = state(OrderDir::Buy, 1, 8.0, 5, 2);
    exact(
        interpreter
            .interpret(&current, SaoePolicyAction::Continuous(1.5))
            .unwrap(),
        6.0,
    );
    let mut rebound_parts = current.parts().clone();
    rebound_parts.backtest_data.ticks_for_order.clear();
    exact(
        interpreter
            .interpret(
                &SaoeState::new(rebound_parts),
                SaoePolicyAction::Continuous(1.5),
            )
            .unwrap(),
        6.0,
    );
    assert_eq!(
        interpreter.interpret(&current, SaoePolicyAction::Discrete(1)),
        Err(SaoeInterpreterError::ExpectedContinuous)
    );
    assert_eq!(
        interpreter.interpret(&current, SaoePolicyAction::Continuous(-0.1)),
        Err(SaoeInterpreterError::NegativeContinuous)
    );
    assert_eq!(
        interpreter.interpret(
            &state(OrderDir::Buy, 0, 1.0, 1, 0),
            SaoePolicyAction::Continuous(1.0),
        ),
        Err(SaoeInterpreterError::ZeroTicksPerStep)
    );
    assert_eq!(
        interpreter.interpret(
            &state(OrderDir::Buy, 3, 1.0, 5, 2),
            SaoePolicyAction::Continuous(1.0),
        ),
        Err(SaoeInterpreterError::NoRemainingSteps)
    );
    exact(
        interpreter
            .interpret(
                &state(OrderDir::Buy, 4, 8.0, 5, 2),
                SaoePolicyAction::Continuous(1.5),
            )
            .unwrap(),
        -12.0,
    );

    let observations = [
        SaoeObservation::Dummy { dummy: 1 },
        SaoeObservation::Dummy { dummy: 1 },
    ];
    let mut default_policy = AllOnePolicy::default();
    assert_eq!(
        default_policy.actions(&observations).unwrap(),
        [
            SaoePolicyAction::Continuous(1.0),
            SaoePolicyAction::Continuous(1.0),
        ]
    );
    let mut discrete_policy = AllOnePolicy::new(SaoePolicyAction::Discrete(2));
    assert_eq!(
        discrete_policy.actions(&observations).unwrap(),
        [SaoePolicyAction::Discrete(2), SaoePolicyAction::Discrete(2)]
    );
    assert!(discrete_policy.actions(&[]).unwrap().is_empty());

    let encoded = serde_json::to_vec(&(
        observations.clone(),
        SaoePolicyAction::Continuous(1.5),
        SaoeActionSpace::NonNegativeContinuous,
    ))
    .unwrap();
    let decoded: ([SaoeObservation; 2], SaoePolicyAction, SaoeActionSpace) =
        serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded.0, observations);
    let python = python_contract();
    assert_eq!(python["twap"], 6.0);
    assert_eq!(python["policy_default"], json!([1.0, 1.0, 1.0]));
    assert_eq!(python["policy_discrete"], json!([2, 2]));
}

#[test]
fn policy_pipeline_batches_in_stable_order_and_propagates_every_stage_failure() {
    let states = [
        state(OrderDir::Buy, 0, 7.0, 5, 2),
        state(OrderDir::Sell, 0, 4.0, 5, 2),
    ];
    let observations = Arc::new(Mutex::new(Vec::new()));
    let mut pipeline = SaoePolicyPipeline::new(
        Box::new(DummyStateInterpreter),
        Box::new(FixedPolicy {
            actions: vec![SaoePolicyAction::Discrete(2); 2],
            fail: false,
            observations: Arc::clone(&observations),
        }),
        Box::new(CategoricalActionInterpreter::from_count(4, None).unwrap()),
    );
    assert_eq!(pipeline.execution_volumes(&states).unwrap(), [5.0, 4.0]);
    assert_eq!(
        observations.lock().unwrap().as_slice(),
        [
            SaoeObservation::Dummy { dummy: 1 },
            SaoeObservation::Dummy { dummy: 1 },
        ]
    );
    assert_eq!(python_contract()["pipeline"], json!([5.0, 4.0]));

    let retained = pipeline.decisions(&states).unwrap();
    assert_eq!(retained[0].action, SaoePolicyAction::Discrete(2));
    exact(retained[0].execution_volume, 5.0);

    let mut empty = SaoePolicyPipeline::new(
        Box::new(DummyStateInterpreter),
        Box::new(AllOnePolicy::new(SaoePolicyAction::Discrete(1))),
        Box::new(CategoricalActionInterpreter::from_count(1, None).unwrap()),
    );
    assert!(empty.execution_volumes(&[]).unwrap().is_empty());

    let mut wrong_length = SaoePolicyPipeline::new(
        Box::new(DummyStateInterpreter),
        Box::new(FixedPolicy {
            actions: vec![],
            fail: false,
            observations: Arc::new(Mutex::new(Vec::new())),
        }),
        Box::new(CategoricalActionInterpreter::from_count(1, None).unwrap()),
    );
    assert_eq!(
        wrong_length.execution_volumes(&states),
        Err(SaoeInterpreterError::PolicyBatchLength {
            expected: 2,
            actual: 0,
        })
    );

    let mut policy_failure = SaoePolicyPipeline::new(
        Box::new(DummyStateInterpreter),
        Box::new(FixedPolicy {
            actions: vec![],
            fail: true,
            observations: Arc::new(Mutex::new(Vec::new())),
        }),
        Box::new(CategoricalActionInterpreter::from_count(1, None).unwrap()),
    );
    assert_eq!(
        policy_failure.execution_volumes(&states),
        Err(SaoeInterpreterError::ProcessedDataPlugin(
            "policy".to_owned()
        ))
    );

    let mut state_failure = SaoePolicyPipeline::new(
        Box::new(CurrentStepStateInterpreter::new(1).unwrap()),
        Box::new(AllOnePolicy::new(SaoePolicyAction::Discrete(1))),
        Box::new(CategoricalActionInterpreter::from_count(1, None).unwrap()),
    );
    assert_eq!(
        state_failure.execution_volumes(&[state(OrderDir::Buy, 2, 1.0, 1, 1)]),
        Err(SaoeInterpreterError::InvalidMaxStep(1))
    );

    let mut action_failure = SaoePolicyPipeline::new(
        Box::new(DummyStateInterpreter),
        Box::new(AllOnePolicy::default()),
        Box::new(CategoricalActionInterpreter::from_count(1, None).unwrap()),
    );
    assert_eq!(
        action_failure.execution_volumes(&states[..1]),
        Err(SaoeInterpreterError::ExpectedDiscrete)
    );
}
