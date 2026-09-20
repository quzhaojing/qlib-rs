use std::{
    process::Command,
    sync::{Arc, Mutex},
};

use chrono::NaiveDateTime;
use domain_core::{
    AccountConstructionError, AccountConstructionInput, AccountConstructionPluginError,
    AccountDataRequest, AccountDataResolver, AccountPositionFactory, AccountPositionRequest,
    BenchmarkReturnSampler, BenchmarkReturnSamplerError, InitialPositionValue,
    InitialStockPriceProvider, InitialStockPriceProviderError, InitialStockPriceRequest,
    NativeAccountPositionFactory, PositionHolding, ResolvedAccountData, create_account_instance,
};
use indexmap::IndexMap;
use serde_json::{Value, json};

fn time(text: &str) -> NaiveDateTime {
    NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S").unwrap()
}

struct Benchmark;

impl BenchmarkReturnSampler for Benchmark {
    fn sample_return(
        &self,
        _trade_start_time: NaiveDateTime,
        _trade_end_time: NaiveDateTime,
    ) -> Result<Option<f64>, BenchmarkReturnSamplerError> {
        Ok(None)
    }
}

struct Prices {
    events: Arc<Mutex<Vec<Value>>>,
    values: IndexMap<String, f64>,
}

impl InitialStockPriceProvider for Prices {
    fn latest_close_prices(
        &self,
        request: InitialStockPriceRequest<'_>,
    ) -> Result<IndexMap<String, f64>, InitialStockPriceProviderError> {
        self.events.lock().unwrap().push(json!({
            "stocks": request.stocks,
            "start": request.start_time.format("%Y-%m-%d %H:%M:%S").to_string(),
            "end": request.end_time.format("%Y-%m-%d %H:%M:%S").to_string(),
            "frequency": request.frequency,
            "disk_cache": request.disk_cache,
        }));
        Ok(self.values.clone())
    }
}

struct Resolver {
    events: Arc<Mutex<Vec<Value>>>,
    prices: Option<Arc<dyn InitialStockPriceProvider>>,
    failure: bool,
}

impl AccountDataResolver for Resolver {
    fn resolve(
        &self,
        request: AccountDataRequest<'_>,
    ) -> Result<ResolvedAccountData, AccountConstructionPluginError> {
        self.events.lock().unwrap().push(json!({
            "benchmark": request.benchmark,
            "start": request.start_time.map(|value| value.format("%Y-%m-%d %H:%M:%S").to_string()),
            "end": request.end_time.map(|value| value.format("%Y-%m-%d %H:%M:%S").to_string()),
            "frequency": request.frequency,
        }));
        if self.failure {
            return Err(AccountConstructionPluginError {
                message: "data".to_owned(),
            });
        }
        Ok(ResolvedAccountData {
            benchmark: Arc::new(Benchmark),
            initial_price_provider: self.prices.clone(),
        })
    }
}

struct Positions {
    events: Arc<Mutex<Vec<Value>>>,
    failure: bool,
}

impl AccountPositionFactory for Positions {
    fn create(
        &self,
        request: AccountPositionRequest<'_>,
    ) -> Result<Box<dyn domain_core::AccountPosition>, AccountConstructionPluginError> {
        self.events.lock().unwrap().push(json!([
            request.position_type,
            request.initial_cash,
            request.positions.keys().collect::<Vec<_>>(),
        ]));
        if self.failure {
            Err(AccountConstructionPluginError {
                message: "position".to_owned(),
            })
        } else {
            NativeAccountPositionFactory.create(request)
        }
    }
}

fn resolver(
    events: Arc<Mutex<Vec<Value>>>,
    prices: Option<Arc<dyn InitialStockPriceProvider>>,
) -> Resolver {
    Resolver {
        events,
        prices,
        failure: false,
    }
}

fn construction_error(
    result: Result<domain_core::Account, AccountConstructionError>,
) -> AccountConstructionError {
    match result {
        Ok(_) => panic!("construction should fail"),
        Err(error) => error,
    }
}

#[test]
fn unchanged_source_wrapper_preserves_shapes_mutation_identity_and_errors() {
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/account_construction_contract.py"
        ))
        .output()
        .expect("Python characterization fixture runs");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let actual: Value = serde_json::from_slice(&output.stdout).expect("fixture emits JSON");
    assert_eq!(
        actual,
        json!([
            {"name":"integer","events":[["account",7,{},"Position",{},false]],"remaining":7,"returned_marker":true,"error":null},
            {"name":"float","events":[["account",2.5,{},"Position",{"benchmark":"CUSTOM","start_time":"2024-01-02","end_time":"2024-01-31"},false]],"remaining":2.5,"returned_marker":true,"error":null},
            {"name":"dictionary","events":[["account",11.0,{"A":2,"B":{"amount":3.0,"price":4.0}},"InfPosition",{"benchmark":"BM","start_time":"2024-01-02","end_time":"2024-01-31"},true]],"remaining":{"A":2,"B":{"amount":3.0,"price":4.0}},"returned_marker":true,"error":null},
            {"name":"construction_failure","events":[["account",13.0,{"A":1},"Fail",{"benchmark":"BM","start_time":"2024-01-02","end_time":"2024-01-31"},true]],"remaining":{"A":1},"returned_marker":false,"error":"RuntimeError:account-construction"},
            {"name":"missing_cash","events":[],"remaining":{"A":1},"returned_marker":false,"error":"KeyError:'cash'"},
            {"name":"unsupported","events":[],"remaining":"7","returned_marker":false,"error":"ValueError:account must be in (int, float, dict)"},
        ])
    );
}

#[test]
fn native_cash_account_uses_default_benchmark_without_mutating_input() {
    let start = time("2024-01-02 00:00:00");
    let end = time("2024-01-31 00:00:00");
    let position_events = Arc::new(Mutex::new(Vec::new()));
    let data_events = Arc::new(Mutex::new(Vec::new()));
    let positions = Positions {
        events: Arc::clone(&position_events),
        failure: false,
    };
    let default_resolver = resolver(Arc::clone(&data_events), None);
    let mut cash = AccountConstructionInput::Cash(7.0);
    let account = create_account_instance(
        start,
        end,
        None,
        &mut cash,
        "Position",
        &positions,
        &default_resolver,
    )
    .unwrap();
    assert_eq!(cash, AccountConstructionInput::Cash(7.0));
    assert_eq!(account.initial_cash().to_bits(), 7.0_f64.to_bits());
    assert_eq!(
        account
            .current_position()
            .available_cash()
            .unwrap()
            .to_bits(),
        7.0_f64.to_bits()
    );
    assert!(account.current_position().stock_ids().unwrap().is_empty());
    assert_eq!(account.portfolio_metrics().unwrap().frequency(), "day");
    assert!(account.report_config().benchmark.is_some());
    assert_eq!(account.report_config().benchmark_name, None);
    assert_eq!(account.report_config().start_time, None);
    assert_eq!(account.report_config().end_time, None);
    assert_eq!(
        data_events.lock().unwrap().as_slice(),
        &[json!({"benchmark":"SH000300","start":null,"end":null,"frequency":"day"})]
    );
    assert_eq!(position_events.lock().unwrap().len(), 1);
}

#[test]
fn native_dictionary_account_retains_config_and_fills_missing_prices() {
    let start = time("2024-01-02 00:00:00");
    let end = time("2024-01-31 00:00:00");
    let position_events = Arc::new(Mutex::new(Vec::new()));
    let data_events = Arc::new(Mutex::new(Vec::new()));
    let price_events = Arc::new(Mutex::new(Vec::new()));
    let positions = Positions {
        events: Arc::clone(&position_events),
        failure: false,
    };
    let price_provider: Arc<dyn InitialStockPriceProvider> = Arc::new(Prices {
        events: Arc::clone(&price_events),
        values: IndexMap::from([("A".to_owned(), 5.0)]),
    });
    let named_resolver = resolver(Arc::clone(&data_events), Some(price_provider));
    let mut dictionary = AccountConstructionInput::Dictionary {
        cash: Some(11.0),
        positions: IndexMap::from([
            ("A".to_owned(), InitialPositionValue::Amount(2.0)),
            (
                "B".to_owned(),
                InitialPositionValue::Holding(PositionHolding::restored(3.0, Some(4.0), None)),
            ),
        ]),
    };
    let account = create_account_instance(
        start,
        end,
        Some("BM"),
        &mut dictionary,
        "Position",
        &positions,
        &named_resolver,
    )
    .unwrap();
    let AccountConstructionInput::Dictionary { cash, positions } = &dictionary else {
        panic!("dictionary input remains a dictionary")
    };
    assert!(cash.is_none());
    assert_eq!(
        positions.keys().map(String::as_str).collect::<Vec<_>>(),
        ["A", "B"]
    );
    assert_eq!(account.current_position().stock_ids().unwrap(), ["A", "B"]);
    assert_eq!(
        account
            .current_position()
            .stock_price("A")
            .unwrap()
            .to_bits(),
        5.0_f64.to_bits()
    );
    assert_eq!(
        account
            .current_position()
            .stock_price("B")
            .unwrap()
            .to_bits(),
        4.0_f64.to_bits()
    );
    assert_eq!(
        account.report_config().benchmark_name.as_deref(),
        Some("BM")
    );
    assert_eq!(account.report_config().start_time, Some(start));
    assert_eq!(account.report_config().end_time, Some(end));
    assert_eq!(
        data_events.lock().unwrap().as_slice(),
        &[json!({
            "benchmark":"BM",
            "start":"2024-01-02 00:00:00",
            "end":"2024-01-31 00:00:00",
            "frequency":"day"
        })]
    );
    assert_eq!(
        price_events.lock().unwrap().as_slice(),
        &[
            json!({"stocks":["A"],"start":"2023-12-03 00:00:00","end":"2024-01-02 00:00:00","frequency":"day","disk_cache":true})
        ]
    );
    assert_eq!(position_events.lock().unwrap().len(), 1);
}

#[test]
fn infinite_position_skips_data_but_retains_explicit_cash_and_metadata() {
    let start = time("2024-01-02 00:00:00");
    let end = time("2024-01-31 00:00:00");
    let data_events = Arc::new(Mutex::new(Vec::new()));
    let positions = Positions {
        events: Arc::new(Mutex::new(Vec::new())),
        failure: false,
    };
    let resolver = resolver(Arc::clone(&data_events), None);
    let mut input = AccountConstructionInput::Cash(1.0e12);
    let account = create_account_instance(
        start,
        end,
        Some("BM"),
        &mut input,
        "InfPosition",
        &positions,
        &resolver,
    )
    .unwrap();
    assert_eq!(account.initial_cash().to_bits(), 1.0e12_f64.to_bits());
    assert!(!account.is_portfolio_metrics_enabled());
    assert!(account.portfolio_metrics().is_none());
    assert_eq!(
        account.report_config().benchmark_name.as_deref(),
        Some("BM")
    );
    assert_eq!(account.report_config().start_time, Some(start));
    assert_eq!(account.report_config().end_time, Some(end));
    assert!(account.report_config().benchmark.is_none());
    assert!(data_events.lock().unwrap().is_empty());
}

#[test]
fn input_and_plugin_failures_preserve_the_source_mutation_boundary() {
    let start = time("2024-01-02 00:00:00");
    let resolver = resolver(Arc::new(Mutex::new(Vec::new())), None);
    let positions = Positions {
        events: Arc::new(Mutex::new(Vec::new())),
        failure: false,
    };
    let mut unsupported = AccountConstructionInput::Unsupported;
    assert_eq!(
        construction_error(create_account_instance(
            start,
            start,
            None,
            &mut unsupported,
            "Position",
            &positions,
            &resolver
        )),
        AccountConstructionError::UnsupportedInput
    );
    let mut missing = AccountConstructionInput::Dictionary {
        cash: None,
        positions: IndexMap::new(),
    };
    assert_eq!(
        construction_error(create_account_instance(
            start,
            start,
            None,
            &mut missing,
            "Position",
            &positions,
            &resolver
        )),
        AccountConstructionError::MissingCash
    );

    let failing_positions = Positions {
        events: Arc::new(Mutex::new(Vec::new())),
        failure: true,
    };
    let mut position_failure = AccountConstructionInput::Dictionary {
        cash: Some(3.0),
        positions: IndexMap::new(),
    };
    assert_eq!(
        construction_error(create_account_instance(
            start,
            start,
            None,
            &mut position_failure,
            "Position",
            &failing_positions,
            &resolver
        )),
        AccountConstructionError::Position(AccountConstructionPluginError {
            message: "position".to_owned()
        })
    );
    assert!(matches!(
        position_failure,
        AccountConstructionInput::Dictionary { cash: None, .. }
    ));
}

#[test]
fn data_and_account_failures_retain_the_consumed_dictionary_cash() {
    let start = time("2024-01-02 00:00:00");
    let positions = Positions {
        events: Arc::new(Mutex::new(Vec::new())),
        failure: false,
    };
    let failing_resolver = Resolver {
        events: Arc::new(Mutex::new(Vec::new())),
        prices: None,
        failure: true,
    };
    let mut data_failure = AccountConstructionInput::Dictionary {
        cash: Some(4.0),
        positions: IndexMap::new(),
    };
    assert_eq!(
        construction_error(create_account_instance(
            start,
            start,
            None,
            &mut data_failure,
            "Position",
            &positions,
            &failing_resolver
        )),
        AccountConstructionError::Data(AccountConstructionPluginError {
            message: "data".to_owned()
        })
    );
    assert!(matches!(
        data_failure,
        AccountConstructionInput::Dictionary { cash: None, .. }
    ));

    let resolver = resolver(Arc::new(Mutex::new(Vec::new())), None);
    let mut reset_failure = AccountConstructionInput::Dictionary {
        cash: Some(5.0),
        positions: IndexMap::from([("A".to_owned(), InitialPositionValue::Amount(1.0))]),
    };
    let error = construction_error(create_account_instance(
        start,
        start,
        Some("BM"),
        &mut reset_failure,
        "Position",
        &positions,
        &resolver,
    ));
    assert_eq!(
        error,
        AccountConstructionError::Account(
            domain_core::AccountResetError::MissingInitialPriceProvider
        )
    );
    assert!(matches!(
        reset_failure,
        AccountConstructionInput::Dictionary { cash: None, .. }
    ));
}

#[test]
fn native_position_factory_rejects_unknown_dynamic_names() {
    let positions = IndexMap::new();
    let Err(error) = NativeAccountPositionFactory.create(AccountPositionRequest {
        position_type: "Unknown",
        initial_cash: 1.0,
        positions: &positions,
    }) else {
        panic!("unknown position type should fail")
    };
    assert_eq!(error.message, "unknown account position type Unknown");
}
