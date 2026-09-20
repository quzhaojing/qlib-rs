use std::{
    process::Command,
    sync::{Arc, Mutex},
};

use domain_core::{
    GetStrategyExecutorError, StrategyExecutorAssembler, StrategyExecutorConstructionRequest,
    StrategyExecutorInfrastructure, StrategyExecutorInfrastructureTarget,
    StrategyExecutorPluginError, get_strategy_executor,
};
use indexmap::IndexMap;
use serde_json::{Value, json};

#[derive(Clone, Debug)]
struct Marker(&'static str);

#[derive(Clone, Debug)]
struct Time(&'static str);

#[derive(Clone, Debug)]
enum Argument {
    Text(&'static str),
    Null,
    Opaque(Arc<Marker>),
}

impl From<Time> for Argument {
    fn from(value: Time) -> Self {
        Self::Text(value.0)
    }
}

fn argument_value(argument: &Argument, nested: &Arc<Marker>) -> Value {
    match argument {
        Argument::Text(value) => json!(value),
        Argument::Null => Value::Null,
        Argument::Opaque(value) => json!(["opaque", Arc::ptr_eq(value, nested)]),
    }
}

#[derive(Debug)]
struct AccountInput {
    cash: Option<f64>,
    positions: Vec<&'static str>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Failure {
    None,
    Account,
    Exchange,
    StrategyResolution,
    StrategyReset,
    ExecutorResolution,
    ExecutorReset,
}

type Infrastructure = StrategyExecutorInfrastructure<Arc<Marker>, Arc<Marker>>;

struct Component {
    name: &'static str,
    events: Arc<Mutex<Vec<Value>>>,
    first_infrastructure: Arc<Mutex<Option<Arc<Infrastructure>>>>,
    fail: bool,
}

impl StrategyExecutorInfrastructureTarget<Arc<Marker>, Arc<Marker>> for Component {
    fn reset_common_infrastructure(
        &mut self,
        infrastructure: Arc<Infrastructure>,
    ) -> Result<(), StrategyExecutorPluginError> {
        let same_infrastructure = {
            let mut first = self.first_infrastructure.lock().unwrap();
            if let Some(first) = first.as_ref() {
                Arc::ptr_eq(first, &infrastructure)
            } else {
                *first = Some(Arc::clone(&infrastructure));
                true
            }
        };
        self.events.lock().unwrap().push(json!([
            self.name,
            "reset",
            infrastructure.account.0,
            infrastructure.exchange.0,
            same_infrastructure,
        ]));
        if self.fail {
            Err(plugin_error(&format!("{}_reset", self.name)))
        } else {
            Ok(())
        }
    }
}

struct Assembler {
    events: Arc<Mutex<Vec<Value>>>,
    nested: Arc<Marker>,
    first_infrastructure: Arc<Mutex<Option<Arc<Infrastructure>>>>,
    failure: Failure,
}

fn plugin_error(stage: &str) -> StrategyExecutorPluginError {
    StrategyExecutorPluginError {
        message: stage.to_owned(),
    }
}

impl StrategyExecutorAssembler<Time, AccountInput, &'static str, &'static str, Argument>
    for Assembler
{
    type Account = Arc<Marker>;
    type Exchange = Arc<Marker>;
    type Strategy = Component;
    type Executor = Component;

    fn create_account(
        &mut self,
        start_time: &Time,
        end_time: &Time,
        benchmark: Option<&str>,
        account: &mut AccountInput,
        position_type: &str,
    ) -> Result<Self::Account, StrategyExecutorPluginError> {
        self.events.lock().unwrap().push(json!([
            "account",
            start_time.0,
            end_time.0,
            benchmark,
            position_type,
            account.positions,
        ]));
        account.cash.take();
        if self.failure == Failure::Account {
            Err(plugin_error("account"))
        } else {
            Ok(Arc::new(Marker("account")))
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
                .map(|(key, value)| (key, argument_value(value, &self.nested)))
                .collect::<IndexMap<_, _>>(),
        ]));
        if self.failure == Failure::Exchange {
            Err(plugin_error("exchange"))
        } else {
            Ok(Arc::new(Marker("exchange")))
        }
    }

    fn resolve_strategy(
        &mut self,
        configuration: &'static str,
    ) -> Result<Self::Strategy, StrategyExecutorPluginError> {
        self.events
            .lock()
            .unwrap()
            .push(json!(["strategy", "resolve", configuration]));
        if self.failure == Failure::StrategyResolution {
            return Err(plugin_error("strategy_resolution"));
        }
        Ok(Component {
            name: "strategy",
            events: Arc::clone(&self.events),
            first_infrastructure: Arc::clone(&self.first_infrastructure),
            fail: self.failure == Failure::StrategyReset,
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
        if self.failure == Failure::ExecutorResolution {
            return Err(plugin_error("executor_resolution"));
        }
        Ok(Component {
            name: "executor",
            events: Arc::clone(&self.events),
            first_infrastructure: Arc::clone(&self.first_infrastructure),
            fail: self.failure == Failure::ExecutorReset,
        })
    }
}

type ConstructionRequest = StrategyExecutorConstructionRequest<
    'static,
    Time,
    AccountInput,
    &'static str,
    &'static str,
    Argument,
>;
type Setup = (
    ConstructionRequest,
    Assembler,
    &'static IndexMap<String, Argument>,
);

fn setup(failure: Failure, provided_times: bool) -> Setup {
    let events = Arc::new(Mutex::new(Vec::new()));
    let nested = Arc::new(Marker("nested"));
    let mut arguments = IndexMap::from([
        ("freq".to_owned(), Argument::Text("1min")),
        ("nested".to_owned(), Argument::Opaque(Arc::clone(&nested))),
    ]);
    if provided_times {
        arguments.insert("start_time".to_owned(), Argument::Null);
        arguments.insert("end_time".to_owned(), Argument::Text("exchange-end"));
    }
    let arguments = Box::leak(Box::new(arguments));
    let account = Box::leak(Box::new(AccountInput {
        cash: Some(10.0),
        positions: vec!["A"],
    }));
    (
        StrategyExecutorConstructionRequest {
            start_time: Time("outer-start"),
            end_time: Time("outer-end"),
            strategy: "strategy-config",
            executor: "executor-config",
            benchmark: Some("BENCH".to_owned()),
            account,
            exchange_arguments: arguments,
            position_type: "PositionX".to_owned(),
        },
        Assembler {
            events,
            nested,
            first_infrastructure: Arc::new(Mutex::new(None)),
            failure,
        },
        arguments,
    )
}

fn take_error(
    result: Result<
        domain_core::StrategyExecutorPair<Component, Component>,
        GetStrategyExecutorError,
    >,
) -> GetStrategyExecutorError {
    match result {
        Ok(_) => panic!("operation should fail"),
        Err(error) => error,
    }
}

#[test]
fn unchanged_source_order_copy_identity_and_failures_are_frozen() {
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/get_strategy_executor_contract.py"
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
    assert_eq!(cases.len(), 8);
    assert_eq!(cases[0]["events"][0][0], "account");
    assert_eq!(cases[0]["events"][1], json!(["copy_exchange_kwargs"]));
    assert_eq!(
        cases[0]["events"][2],
        json!(["exchange", "outer-start", "outer-end", "1min", true])
    );
    assert_eq!(cases[0]["events"][4], json!(["strategy_resolve", true]));
    assert_eq!(cases[0]["events"][7], json!(["executor_reset", true]));
    assert_eq!(cases[0]["returned"], json!([true, true]));
    assert_eq!(
        cases[0]["original_exchange_keys"],
        json!(["freq", "nested"])
    );
    assert_eq!(cases[1]["events"][2][1], Value::Null);
    assert_eq!(cases[1]["events"][2][2], "exchange-end");
    for case in cases {
        assert_eq!(case["account_keys"], json!(["A"]));
    }
    for (index, stage) in [
        "account",
        "exchange",
        "strategy_resolve",
        "strategy_reset",
        "executor_resolve",
        "executor_reset",
    ]
    .into_iter()
    .enumerate()
    {
        assert_eq!(cases[index + 2]["error"], format!("RuntimeError:{stage}"));
        assert_eq!(
            cases[index + 2]["events"]
                .as_array()
                .unwrap()
                .last()
                .unwrap()[0],
            stage
        );
    }
}

#[test]
fn missing_times_are_inserted_into_a_private_shallow_copy() {
    let (request, mut assembler, original) = setup(Failure::None, false);
    let result = get_strategy_executor(request, &mut assembler).unwrap();
    assert_eq!(result.strategy.name, "strategy");
    assert_eq!(result.executor.name, "executor");
    assert_eq!(
        original.keys().map(String::as_str).collect::<Vec<_>>(),
        vec!["freq", "nested"]
    );
    assert_eq!(
        assembler.events.lock().unwrap().as_slice(),
        &[
            json!([
                "account",
                "outer-start",
                "outer-end",
                "BENCH",
                "PositionX",
                ["A"]
            ]),
            json!(["exchange", {
                "freq":"1min",
                "nested":["opaque", true],
                "start_time":"outer-start",
                "end_time":"outer-end",
            }]),
            json!(["strategy", "resolve", "strategy-config"]),
            json!(["strategy", "reset", "account", "exchange", true]),
            json!(["executor", "resolve", "executor-config"]),
            json!(["executor", "reset", "account", "exchange", true]),
        ]
    );
}

#[test]
fn present_null_and_custom_times_are_not_replaced() {
    let (request, mut assembler, original) = setup(Failure::None, true);
    get_strategy_executor(request, &mut assembler).unwrap();
    assert!(matches!(original["start_time"], Argument::Null));
    assert!(matches!(
        original["end_time"],
        Argument::Text("exchange-end")
    ));
    assert_eq!(
        assembler.events.lock().unwrap()[1],
        json!(["exchange", {
            "freq":"1min",
            "nested":["opaque", true],
            "start_time":null,
            "end_time":"exchange-end",
        }])
    );
}

#[test]
fn every_failure_keeps_its_stage_and_completed_side_effects() {
    let cases = [
        (Failure::Account, "account", "Account"),
        (Failure::Exchange, "exchange", "Exchange"),
        (
            Failure::StrategyResolution,
            "strategy_resolution",
            "StrategyResolution",
        ),
        (Failure::StrategyReset, "strategy_reset", "StrategyReset"),
        (
            Failure::ExecutorResolution,
            "executor_resolution",
            "ExecutorResolution",
        ),
        (Failure::ExecutorReset, "executor_reset", "ExecutorReset"),
    ];
    for (failure, message, variant) in cases {
        let (request, mut assembler, original) = setup(failure, false);
        let error = take_error(get_strategy_executor(request, &mut assembler));
        let (actual_variant, source) = match error {
            GetStrategyExecutorError::Account(source) => ("Account", source),
            GetStrategyExecutorError::Exchange(source) => ("Exchange", source),
            GetStrategyExecutorError::StrategyResolution(source) => ("StrategyResolution", source),
            GetStrategyExecutorError::StrategyReset(source) => ("StrategyReset", source),
            GetStrategyExecutorError::ExecutorResolution(source) => ("ExecutorResolution", source),
            GetStrategyExecutorError::ExecutorReset(source) => ("ExecutorReset", source),
        };
        assert_eq!(actual_variant, variant);
        assert_eq!(source.message, message);
        assert_eq!(
            original.keys().map(String::as_str).collect::<Vec<_>>(),
            vec!["freq", "nested"]
        );
    }
}
