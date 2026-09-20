use std::{
    process::Command,
    sync::{Arc, Mutex},
};

use chrono::NaiveDateTime;
use domain_core::{
    BacktestLoopBackend, BacktestLoopError, BacktestLoopEvent, CollectDataReportTarget,
    ConfiguredBacktestError, ConfiguredBacktestRequest, ConfiguredCollectData,
    ConfiguredCollectDataError, GetStrategyExecutorError, NestedControlEvent, NestedExecutorResume,
    NestedInnerProgress, NestedStrategyPrompt, OrderDecision, OrderTradeDecision,
    StrategyExecutorAssembler, StrategyExecutorInfrastructure,
    StrategyExecutorInfrastructureTarget, StrategyExecutorPluginError, run_configured_backtest,
};
use indexmap::IndexMap;
use serde_json::{Value, json};
use thiserror::Error;

fn time(hour: u32) -> NaiveDateTime {
    chrono::NaiveDate::from_ymd_opt(2024, 1, 2)
        .unwrap()
        .and_hms_opt(hour, 0, 0)
        .unwrap()
}

#[derive(Clone)]
enum Argument {
    Text(&'static str),
    Time(NaiveDateTime),
}

impl From<NaiveDateTime> for Argument {
    fn from(value: NaiveDateTime) -> Self {
        Self::Time(value)
    }
}

fn argument_value(value: &Argument) -> Value {
    match value {
        Argument::Text(value) => json!(value),
        Argument::Time(value) => json!(value.to_string()),
    }
}

struct Component {
    name: &'static str,
    events: Arc<Mutex<Vec<Value>>>,
}

impl StrategyExecutorInfrastructureTarget<&'static str, &'static str> for Component {
    fn reset_common_infrastructure(
        &mut self,
        infrastructure: Arc<StrategyExecutorInfrastructure<&'static str, &'static str>>,
    ) -> Result<(), StrategyExecutorPluginError> {
        self.events.lock().unwrap().push(json!([
            self.name,
            "reset",
            infrastructure.account,
            infrastructure.exchange,
        ]));
        Ok(())
    }
}

struct Assembler {
    events: Arc<Mutex<Vec<Value>>>,
    fail: bool,
}

impl StrategyExecutorAssembler<NaiveDateTime, bool, &'static str, &'static str, Argument>
    for Assembler
{
    type Account = &'static str;
    type Exchange = &'static str;
    type Strategy = Component;
    type Executor = Component;

    fn create_account(
        &mut self,
        start_time: &NaiveDateTime,
        end_time: &NaiveDateTime,
        benchmark: Option<&str>,
        account: &mut bool,
        position_type: &str,
    ) -> Result<Self::Account, StrategyExecutorPluginError> {
        self.events.lock().unwrap().push(json!([
            "account",
            start_time.to_string(),
            end_time.to_string(),
            benchmark,
            position_type,
        ]));
        *account = true;
        if self.fail {
            Err(StrategyExecutorPluginError {
                message: "assembly".to_owned(),
            })
        } else {
            Ok("account")
        }
    }

    fn create_exchange(
        &mut self,
        arguments: IndexMap<String, Argument>,
    ) -> Result<Self::Exchange, StrategyExecutorPluginError> {
        self.events.lock().unwrap().push(json!([
            "exchange",
            arguments
                .iter()
                .map(|(key, value)| (key, argument_value(value)))
                .collect::<IndexMap<_, _>>(),
        ]));
        Ok("exchange")
    }

    fn resolve_strategy(
        &mut self,
        configuration: &'static str,
    ) -> Result<Self::Strategy, StrategyExecutorPluginError> {
        self.events
            .lock()
            .unwrap()
            .push(json!(["strategy", "resolve", configuration]));
        Ok(Component {
            name: "strategy",
            events: Arc::clone(&self.events),
        })
    }

    fn resolve_executor(
        &mut self,
        configuration: &'static str,
    ) -> Result<Self::Executor, StrategyExecutorPluginError> {
        self.events
            .lock()
            .unwrap()
            .push(json!(["executor", "resolve", configuration]));
        Ok(Component {
            name: "executor",
            events: Arc::clone(&self.events),
        })
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
#[error("{0}")]
struct LoopFailure(&'static str);

struct Backend {
    events: Arc<Mutex<Vec<Value>>>,
    fail: bool,
    suspend: bool,
    resumed: bool,
}

impl BacktestLoopBackend for Backend {
    type Error = LoopFailure;
    type Reports = &'static str;

    fn reset_executor(
        &mut self,
        start: NaiveDateTime,
        end: NaiveDateTime,
    ) -> Result<(), Self::Error> {
        self.events.lock().unwrap().push(json!([
            "loop",
            "reset",
            start.to_string(),
            end.to_string()
        ]));
        if self.fail {
            Err(LoopFailure("loop"))
        } else {
            Ok(())
        }
    }

    fn reset_strategy(&mut self) -> Result<(), Self::Error> {
        self.events
            .lock()
            .unwrap()
            .push(json!(["loop", "strategy"]));
        Ok(())
    }

    fn trade_len(&mut self) -> Result<i64, Self::Error> {
        self.events.lock().unwrap().push(json!(["loop", "length"]));
        Ok(0)
    }

    fn enter_progress(&mut self, total: i64) -> Result<(), Self::Error> {
        self.events
            .lock()
            .unwrap()
            .push(json!(["loop", "enter", total]));
        Ok(())
    }

    fn finished(&mut self) -> Result<bool, Self::Error> {
        self.events
            .lock()
            .unwrap()
            .push(json!(["loop", "finished"]));
        Ok(!self.suspend || self.resumed)
    }

    fn generate(
        &mut self,
        _: Option<&domain_core::nested_executor::SharedNestedResult>,
    ) -> Result<Box<dyn OrderDecision>, Self::Error> {
        Ok(Box::new(OrderTradeDecision::from_orders(
            Vec::new(),
            time(9),
            time(16),
            None,
        )))
    }

    fn begin_collection(
        &mut self,
        _: Box<dyn OrderDecision>,
    ) -> Result<NestedInnerProgress, Self::Error> {
        Ok(NestedInnerProgress::Suspended(
            NestedControlEvent::StrategyPrompt(NestedStrategyPrompt {
                kind: "prompt".to_owned(),
                schema_version: 1,
                payload: Vec::new(),
            }),
        ))
    }

    fn resume_collection(
        &mut self,
        input: NestedExecutorResume,
    ) -> Result<NestedInnerProgress, Self::Error> {
        self.events
            .lock()
            .unwrap()
            .push(json!(["loop", "resume", input]));
        self.resumed = true;
        Ok(NestedInnerProgress::Complete {
            decision: Box::new(OrderTradeDecision::from_orders(
                Vec::new(),
                time(9),
                time(16),
                None,
            )),
            executions: Arc::default(),
        })
    }

    fn close_collection(&mut self) -> Result<(), Self::Error> {
        panic!("zero-step backtest has no open collection")
    }

    fn post_execute(
        &mut self,
        _: &domain_core::nested_executor::SharedNestedResult,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn update_progress(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    fn finalize_strategy(&mut self) -> Result<(), Self::Error> {
        self.events
            .lock()
            .unwrap()
            .push(json!(["loop", "finalize"]));
        Ok(())
    }

    fn close_progress(&mut self) -> Result<(), Self::Error> {
        self.events.lock().unwrap().push(json!(["loop", "close"]));
        Ok(())
    }

    fn collect_reports(&mut self) -> Result<Self::Reports, Self::Error> {
        self.events.lock().unwrap().push(json!(["loop", "reports"]));
        Ok("reports")
    }
}

type RunResult = (
    Result<&'static str, ConfiguredBacktestError<LoopFailure>>,
    Arc<Mutex<Vec<Value>>>,
    bool,
);

fn run(assembly_failure: bool, loop_failure: bool) -> RunResult {
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut account = false;
    let arguments = IndexMap::from([("freq".to_owned(), Argument::Text("1min"))]);
    let request: ConfiguredBacktestRequest<'_, _, _, _, _> = ConfiguredBacktestRequest {
        start_time: time(9),
        end_time: time(16),
        strategy: "strategy-config",
        executor: "executor-config",
        benchmark: Some("BENCH".to_owned()),
        account: &mut account,
        exchange_arguments: &arguments,
        position_type: "PositionX".to_owned(),
    };
    let mut assembler = Assembler {
        events: Arc::clone(&events),
        fail: assembly_failure,
    };
    let backend_events = Arc::clone(&events);
    let result = run_configured_backtest(request, &mut assembler, |pair| {
        backend_events.lock().unwrap().push(json!([
            "make_backend",
            pair.strategy.name,
            pair.executor.name,
        ]));
        Backend {
            events: Arc::clone(&backend_events),
            fail: loop_failure,
            suspend: false,
            resumed: false,
        }
    });
    (result, events, account)
}

#[test]
fn unchanged_source_forwards_defaults_values_identity_and_failures() {
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/configured_backtest_contract.py"
        ))
        .output()
        .expect("Python characterization fixture runs");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let actual: Value = serde_json::from_slice(&output.stdout).expect("fixture emits JSON");
    let cases = actual.as_array().expect("fixture emits cases");
    assert_eq!(cases.len(), 4);
    assert_eq!(
        cases[0]["events"][0],
        json!([
            "assemble",
            "start",
            "end",
            true,
            true,
            "BENCH",
            true,
            true,
            "PositionX"
        ])
    );
    assert_eq!(
        cases[0]["events"][1],
        json!(["loop", "start", "end", true, true])
    );
    assert_eq!(cases[0]["returned"], true);
    assert_eq!(cases[1]["events"][0][5], "SH000300");
    assert_eq!(cases[1]["events"][0][6], 1_000_000_000.0);
    assert_eq!(cases[1]["events"][0][7], 0);
    assert_eq!(cases[1]["events"][0][8], "Position");
    assert_eq!(cases[2]["events"].as_array().unwrap().len(), 1);
    assert_eq!(cases[2]["error"], "RuntimeError:assembly");
    assert_eq!(cases[3]["events"].as_array().unwrap().len(), 2);
    assert_eq!(cases[3]["error"], "RuntimeError:loop");
}

#[test]
fn configured_pair_is_adapted_once_and_native_loop_returns_reports() {
    let (result, events, account_mutated) = run(false, false);
    assert_eq!(result.unwrap(), "reports");
    assert!(account_mutated);
    let events = events.lock().unwrap();
    assert_eq!(events[0][0], "account");
    assert_eq!(events[6], json!(["make_backend", "strategy", "executor"]));
    assert_eq!(
        events[7],
        json!([
            "loop",
            "reset",
            "2024-01-02 09:00:00",
            "2024-01-02 16:00:00"
        ])
    );
    assert_eq!(events.last().unwrap(), &json!(["loop", "reports"]));
}

#[test]
fn construction_and_loop_failures_remain_distinct() {
    let (construction, events, account_mutated) = run(true, false);
    assert!(matches!(
        construction,
        Err(ConfiguredBacktestError::Construction(
            GetStrategyExecutorError::Account(_)
        ))
    ));
    assert!(account_mutated);
    assert_eq!(
        events.lock().unwrap().as_slice(),
        &[json!([
            "account",
            "2024-01-02 09:00:00",
            "2024-01-02 16:00:00",
            "BENCH",
            "PositionX"
        ])]
    );

    let (loop_result, events, account_mutated) = run(false, true);
    assert!(matches!(
        loop_result,
        Err(ConfiguredBacktestError::Loop(BacktestLoopError::Backend(
            LoopFailure("loop")
        )))
    ));
    assert!(account_mutated);
    assert_eq!(events.lock().unwrap().last().unwrap()[0], "loop");
}

#[derive(Default)]
struct ReportTarget(Option<&'static str>);

impl CollectDataReportTarget<&'static str> for ReportTarget {
    fn publish(&mut self, reports: &'static str) {
        self.0 = Some(reports);
    }
}

#[test]
fn actual_collect_data_is_lazy_forwards_defaults_values_and_send() {
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/configured_collect_data_contract.py"
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
    assert_eq!(cases.len(), 5);
    assert!(cases.iter().all(|case| case["before"] == json!([])));
    assert_eq!(
        cases[0]["events"][0],
        json!(["assemble", 7, "PositionX", "BENCH", {"cash": 7}, true])
    );
    assert_eq!(
        cases[0]["events"][1],
        json!(["loop", "start", "end", true, true, true])
    );
    assert_eq!(cases[0]["events"][2], json!(["received", true]));
    assert_eq!(cases[0]["yielded"], json!([true, true]));
    assert_eq!(cases[0]["kept"], true);
    assert_eq!(cases[0]["published"], true);
    assert_eq!(
        cases[1]["events"][0],
        json!([
            "assemble",
            7,
            "Position",
            "SH000300",
            1_000_000_000.0,
            false
        ])
    );
    assert_eq!(cases[2]["invalid"], "TypeError");
    assert_eq!(cases[2]["events"].as_array().unwrap().len(), 3);
    assert_eq!(cases[3]["error"], "RuntimeError:assembly");
    assert_eq!(cases[3]["events"].as_array().unwrap().len(), 1);
    assert_eq!(cases[4]["error"], "RuntimeError:loop");
    assert_eq!(cases[4]["events"].as_array().unwrap().len(), 2);
}

#[test]
fn actual_module_surface_and_initialization_are_frozen() {
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/backtest_init_surface_contract.py"
        ))
        .output()
        .expect("Python module-surface fixture runs");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let actual: Value = serde_json::from_slice(&output.stdout).expect("fixture emits JSON");
    assert_eq!(
        actual["functions"],
        json!([
            "get_exchange",
            "create_account_instance",
            "get_strategy_executor",
            "backtest",
            "collect_data",
            "format_decisions"
        ])
    );
    assert_eq!(
        actual["all"],
        json!(["Order", "backtest", "get_strategy_executor"])
    );
    assert_eq!(actual["logger_calls"], json!(["backtest caller"]));
    assert_eq!(
        actual["type_checking_imports"],
        json!(["BaseStrategy", "BaseExecutor", "BaseTradeDecision"])
    );
    for name in [
        "Account",
        "C",
        "get_module_logger",
        "init_instance_by_config",
        "INDICATOR_METRIC",
        "PORT_METRIC",
        "backtest_loop",
        "collect_data_loop",
        "Order",
        "Exchange",
        "CommonInfrastructure",
    ] {
        assert!(
            actual["runtime_imports"]
                .as_array()
                .unwrap()
                .contains(&json!(name))
        );
    }
}

type BackendFactory<'a> =
    Box<dyn FnOnce(domain_core::StrategyExecutorPair<Component, Component>) -> Backend + 'a>;
type ConfiguredGenerator<'a> = ConfiguredCollectData<
    'a,
    bool,
    &'static str,
    &'static str,
    Argument,
    Assembler,
    Backend,
    BackendFactory<'a>,
>;

fn configured_generator<'a>(
    assembler: &'a mut Assembler,
    account: &'a mut bool,
    arguments: &'a IndexMap<String, Argument>,
    target: Option<&'a mut dyn CollectDataReportTarget<&'static str>>,
    loop_failure: bool,
    suspend: bool,
) -> ConfiguredGenerator<'a> {
    let events = Arc::clone(&assembler.events);
    ConfiguredCollectData::new(
        ConfiguredBacktestRequest {
            start_time: time(9),
            end_time: time(16),
            strategy: "strategy-config",
            executor: "executor-config",
            benchmark: Some("BENCH".to_owned()),
            account,
            exchange_arguments: arguments,
            position_type: "PositionX".to_owned(),
        },
        assembler,
        Box::new(move |pair| {
            events.lock().unwrap().push(json!([
                "make_backend",
                pair.strategy.name,
                pair.executor.name
            ]));
            Backend {
                events,
                fail: loop_failure,
                suspend,
                resumed: false,
            }
        }),
        target,
    )
}

#[test]
fn configured_collect_data_is_lazy_retries_initial_protocol_and_publishes() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut assembler = Assembler {
        events: Arc::clone(&events),
        fail: false,
    };
    let mut account = false;
    let arguments = IndexMap::new();
    let mut target = ReportTarget::default();
    let mut generator = configured_generator(
        &mut assembler,
        &mut account,
        &arguments,
        Some(&mut target),
        false,
        false,
    );
    assert!(events.lock().unwrap().is_empty());
    assert!(matches!(
        generator.resume(NestedExecutorResume::Action(Some(3.0))),
        Err(ConfiguredCollectDataError::Loop(
            BacktestLoopError::InvalidInitialAction
        ))
    ));
    assert!(events.lock().unwrap().is_empty());
    assert!(matches!(
        generator.resume(NestedExecutorResume::Continue),
        Ok(BacktestLoopEvent::Complete)
    ));
    drop(generator);
    assert!(account);
    assert_eq!(target.0, Some("reports"));
    assert_eq!(
        events.lock().unwrap().last().unwrap(),
        &json!(["loop", "reports"])
    );
}

#[test]
fn configured_collect_data_keeps_failure_stages_and_optional_reports() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut assembler = Assembler {
        events: Arc::clone(&events),
        fail: true,
    };
    let mut account = false;
    let arguments = IndexMap::new();
    let mut generator =
        configured_generator(&mut assembler, &mut account, &arguments, None, false, false);
    assert!(matches!(
        generator.resume(NestedExecutorResume::Continue),
        Err(ConfiguredCollectDataError::Construction(
            GetStrategyExecutorError::Account(_)
        ))
    ));
    assert!(matches!(
        generator.resume(NestedExecutorResume::Continue),
        Err(ConfiguredCollectDataError::Loop(BacktestLoopError::Failed))
    ));
    drop(generator);
    assert!(account);

    let mut assembler = Assembler {
        events,
        fail: false,
    };
    let mut account = false;
    let mut generator =
        configured_generator(&mut assembler, &mut account, &arguments, None, true, false);
    assert!(matches!(
        generator.resume(NestedExecutorResume::Continue),
        Err(ConfiguredCollectDataError::Loop(
            BacktestLoopError::Backend(LoopFailure("loop"))
        ))
    ));
    assert!(matches!(
        generator.resume(NestedExecutorResume::Continue),
        Err(ConfiguredCollectDataError::Loop(BacktestLoopError::Failed))
    ));
}

#[test]
fn configured_collect_data_forwards_nested_resume_values() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut assembler = Assembler {
        events: Arc::clone(&events),
        fail: false,
    };
    let mut account = false;
    let arguments = IndexMap::new();
    let mut generator =
        configured_generator(&mut assembler, &mut account, &arguments, None, false, true);
    assert!(matches!(
        generator.resume(NestedExecutorResume::Continue),
        Ok(BacktestLoopEvent::Suspended(
            NestedControlEvent::StrategyPrompt(_)
        ))
    ));
    assert!(matches!(
        generator.resume(NestedExecutorResume::Action(Some(4.0))),
        Ok(BacktestLoopEvent::Complete)
    ));
    assert!(
        events
            .lock()
            .unwrap()
            .contains(&json!(["loop", "resume", {"Action": 4.0}]))
    );
}
