use std::{
    collections::VecDeque,
    path::PathBuf,
    process::Command,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use arrow_array::Float64Array;
use chrono::NaiveDateTime;
use domain_core::{
    Account, AccountBarEndMode, AccountBarEndUpdate, AccountBarMarket, AccountBarMarketError,
    AccountError, AccountIndicator, AccountIndicatorError, AccountIndicatorMode,
    AccountIndicatorOutput, AccountIndicatorOutputError, AccountIndicatorUpdate, AccountPosition,
    AccountPositionError, AccountReportConfig, AccountReportFactory, AccountReportFactoryError,
    AccountResetError, AccountResetUpdate, AccumulatedInfo, BenchmarkReturnSampler,
    BenchmarkReturnSamplerError, DealPriceFields, ExchangeQuoteProvider, ExecutionPosition,
    ExecutionPositionError, ExecutionTarget, Indicator, IndicatorConfig, IndicatorWeightMethod,
    InfinitePosition, InitialPositionValue, InitialStockPriceProvider,
    InitialStockPriceProviderError, InitialStockPriceRequest, NestedAccountIndicatorUpdate,
    NumpyOrderIndicator, Order, OrderDir, OrderExecution, OrderIndicatorAggregationConfig,
    OrderTradeDecision, PortfolioMetrics, Position, PositionError, PositionHolding, Quote,
    QuoteData, QuoteError, QuoteMethod, StdoutAccountIndicatorOutput, TimeRange,
    format_account_indicator_output,
};
use indexmap::IndexMap;
use serde_json::{Value, json};

fn order(direction: OrderDir) -> Order {
    Order::new("A", 1.0, direction, None, None)
}

fn report_account(id: f64, enabled: bool) -> Account {
    let at = datetime("2024-01-02 09:30:00");
    let mut account = Account::new(finite(id * 100.0, 0.0, Some(5.0)), enabled);
    let mut indicator = ProbeIndicator::new(None);
    indicator.history.insert(
        at,
        Arc::new(std::sync::RwLock::new(IndexMap::from([(
            "id".to_owned(),
            id,
        )]))),
    );
    account.replace_indicator(Box::new(indicator));
    if enabled {
        account.update_portfolio_metrics(at, at).unwrap();
    }
    account.update_historical_positions(at).unwrap();
    account
}

fn update_empty_indicator(account: &mut Account, at: NaiveDateTime) -> Result<(), AccountError> {
    account.update_indicator(AccountIndicatorUpdate {
        trade_start_time: at,
        mode: AccountIndicatorMode::Atomic(&[]),
        calculation: IndicatorConfig::default(),
        show_indicator: false,
    })
}

#[test]
fn owned_indicator_reports_keep_original_engine_across_replacement_and_account_drop() {
    let first = datetime("2024-01-02 09:30:00");
    let second = datetime("2024-01-02 09:31:00");
    let third = datetime("2024-01-02 09:32:00");
    let mut account = Account::new(finite(100.0, 0.0, Some(5.0)), false);
    update_empty_indicator(&mut account, first).unwrap();
    let reports = domain_core::collect_backtest_reports([("day", &account)]).unwrap();
    assert!(Arc::ptr_eq(
        &reports.indicators["1day"].indicator,
        account.indicator()
    ));
    assert!(account.indicator().try_write().is_ok());
    update_empty_indicator(&mut account, second).unwrap();
    assert_eq!(reports.indicators["1day"].table.timestamps, [first]);
    assert_eq!(
        reports.indicators["1day"]
            .indicator
            .read()
            .unwrap()
            .trade_indicator_report()
            .unwrap()
            .timestamps,
        [first, second]
    );
    let old = account.replace_indicator(Box::new(Indicator::new()));
    assert!(Arc::ptr_eq(&old, &reports.indicators["1day"].indicator));
    assert!(!Arc::ptr_eq(&old, account.indicator()));
    update_empty_indicator(&mut account, third).unwrap();
    let new = account.indicator().clone();
    drop(account);
    // Owned aggregate reports can leave the account's lifetime and move across threads.
    std::thread::spawn(move || {
        let report = &reports.indicators["1day"];
        report.indicator.write().unwrap().record(third).unwrap();
        assert_eq!(report.table.timestamps, [first]);
        assert_eq!(
            old.read()
                .unwrap()
                .trade_indicator_report()
                .unwrap()
                .timestamps,
            [first, second, third]
        );
        assert_eq!(
            new.read()
                .unwrap()
                .trade_indicator_report()
                .unwrap()
                .timestamps,
            [third]
        );
        assert!(old.try_write().is_ok());
    })
    .join()
    .unwrap();
}

#[test]
fn poisoned_indicator_fails_before_plugin_calls_and_old_reports_remain_poisoned() {
    let at = datetime("2024-01-02 09:30:00");
    let mut account = report_account(1.0, false);
    let reports = domain_core::collect_backtest_reports([("day", &account)]).unwrap();
    let old = account.indicator().clone();
    let poison = old.clone();
    assert!(
        std::thread::spawn(move || {
            let _guard = poison.write().unwrap();
            panic!("poison retained indicator");
        })
        .join()
        .is_err()
    );
    let expected = "account indicator plugin error: account indicator lock poisoned";
    assert_eq!(
        account.order_indicator_snapshot().unwrap_err().to_string(),
        expected
    );
    assert_eq!(
        update_empty_indicator(&mut account, at)
            .unwrap_err()
            .to_string(),
        expected
    );
    let error = domain_core::collect_backtest_reports([("day", &account)])
        .err()
        .unwrap();
    assert_eq!(error.to_string(), expected);
    assert_eq!(reports.indicators["1day"].table.timestamps, [at]);
    account.replace_indicator(Box::new(Indicator::new()));
    update_empty_indicator(&mut account, at).unwrap();
    assert!(account.order_indicator_snapshot().is_ok());
    assert!(reports.indicators["1day"].indicator.read().is_err());
    assert!(Arc::ptr_eq(&old, &reports.indicators["1day"].indicator));
}

#[test]
fn source_report_ownership_contract_retains_old_objects_across_reset_and_failures() {
    // Source characterization and native differential for the complete reset matrix.
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/backtest_report_ownership_contract.py"
        ))
        .arg("D:/code/github/qlib/qlib")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let rows: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(rows.len(), 6);
    for (row, (mode, same_metrics, same_indicator, error, events)) in rows.iter().zip([
        (
            "enabled",
            false,
            false,
            None,
            json!([
                ["metrics", "2min"],
                ["fill", "2024-01-02T00:00:00", "2min"],
                ["indicator"]
            ]),
        ),
        ("disabled", true, false, None, json!([["indicator"]])),
        ("skip", true, false, None, json!([["indicator"]])),
        (
            "portfolio_failure",
            true,
            true,
            Some("metrics"),
            json!([["metrics", "2min"]]),
        ),
        (
            "fill_failure",
            false,
            true,
            Some("fill"),
            json!([["metrics", "2min"], ["fill", "2024-01-02T00:00:00", "2min"]]),
        ),
        (
            "indicator_failure",
            false,
            true,
            Some("indicator"),
            json!([
                ["metrics", "2min"],
                ["fill", "2024-01-02T00:00:00", "2min"],
                ["indicator"]
            ]),
        ),
    ]) {
        assert_eq!(
            row,
            &json!({
                "mode": mode, "events": events, "error": error, "frequency": "2min",
                "same_metrics": same_metrics, "same_positions": same_metrics,
                "same_indicator": same_indicator, "same_position": true, "same_accumulated": true,
                "old_positions": 2, "old_indicator_rows": 2, "portfolio_snapshot": 100.0,
                "indicator_snapshot": 1.0,
            })
        );
        let native = native_reset_row(mode);
        for key in [
            "mode",
            "events",
            "error",
            "frequency",
            "same_metrics",
            "same_positions",
            "same_indicator",
            "same_position",
            "same_accumulated",
        ] {
            assert_eq!(native[key], row[key], "reset differential field {key}");
        }
    }
}

#[test]
fn finite_position_fills_missing_prices_with_the_upstream_query_contract() {
    let mut position = Position::from_initial(
        10.0,
        IndexMap::from([
            ("A".to_owned(), InitialPositionValue::Amount(1.0)),
            ("B".to_owned(), InitialPositionValue::Amount(2.0)),
            (
                "C".to_owned(),
                InitialPositionValue::Holding(PositionHolding::restored(1.0, Some(3.0), None)),
            ),
        ]),
    );
    let calls = Arc::new(Mutex::new(Vec::new()));
    let provider = PriceProvider {
        calls: calls.clone(),
        prices: Ok(IndexMap::from([
            ("B".to_owned(), 7.0),
            ("A".to_owned(), 5.0),
            ("EXTRA".to_owned(), 99.0),
        ])),
    };
    AccountPosition::fill_stock_value(
        &mut position,
        datetime("2024-01-02 00:00:00"),
        "2min",
        &provider,
    )
    .unwrap();
    assert_eq!(position.holding("A").unwrap().price(), Some(5.0));
    assert_eq!(position.holding("B").unwrap().price(), Some(7.0));
    assert_eq!(position.holding("C").unwrap().price(), Some(3.0));
    assert_eq!(position.account_value(), Some(32.0));
    assert_eq!(
        *calls.lock().unwrap(),
        [json!({
            "stocks": ["A", "B"],
            "start": "2023-12-03T00:00:00",
            "end": "2024-01-02T00:00:00",
            "frequency": "2min",
            "disk_cache": true,
        })]
    );
}

#[test]
fn initial_price_fill_is_atomic_and_reports_provider_and_range_failures() {
    let at = datetime("2024-01-02 00:00:00");
    let mut missing = Position::from_initial(
        10.0,
        IndexMap::from([
            ("A".to_owned(), InitialPositionValue::Amount(1.0)),
            ("B".to_owned(), InitialPositionValue::Amount(1.0)),
            ("C".to_owned(), InitialPositionValue::Amount(1.0)),
        ]),
    );
    let provider = PriceProvider {
        calls: Arc::new(Mutex::new(Vec::new())),
        prices: Ok(IndexMap::from([
            ("A".to_owned(), 5.0),
            ("B".to_owned(), f64::NAN),
        ])),
    };
    assert_eq!(
        AccountPosition::fill_stock_value(&mut missing, at, "day", &provider)
            .unwrap_err()
            .to_string(),
        "account position error: {'B', 'C'} doesn't have close price in qlib in the latest 30 days"
    );
    assert_eq!(missing.holding("A").unwrap().price(), None);
    assert_eq!(missing.holding("B").unwrap().price(), None);
    assert_eq!(missing.holding("C").unwrap().price(), None);
    assert_eq!(missing.account_value(), None);

    let mut provider_failure = Position::from_initial(
        1.0,
        IndexMap::from([("A".to_owned(), InitialPositionValue::Amount(1.0))]),
    );
    let failing = PriceProvider {
        calls: Arc::new(Mutex::new(Vec::new())),
        prices: Err(InitialStockPriceProviderError {
            message: "offline".to_owned(),
        }),
    };
    assert_eq!(
        AccountPosition::fill_stock_value(&mut provider_failure, at, "day", &failing)
            .unwrap_err()
            .to_string(),
        "account position error: initial stock price provider error: offline"
    );

    let mut underflow = Position::from_initial(
        1.0,
        IndexMap::from([("A".to_owned(), InitialPositionValue::Amount(1.0))]),
    );
    assert_eq!(
        AccountPosition::fill_stock_value(&mut underflow, NaiveDateTime::MIN, "day", &provider,)
            .unwrap_err()
            .to_string(),
        "account position error: initial stock price lookback is outside the supported datetime range"
    );
}

#[test]
fn priced_and_base_positions_do_not_query_initial_prices() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let provider = PriceProvider {
        calls: calls.clone(),
        prices: Err(InitialStockPriceProviderError {
            message: "must not run".to_owned(),
        }),
    };
    let mut priced = finite(10.0, 1.0, Some(5.0));
    AccountPosition::fill_stock_value(
        &mut priced,
        datetime("2024-01-02 00:00:00"),
        "day",
        &provider,
    )
    .unwrap();
    let mut infinite = InfinitePosition;
    AccountPosition::fill_stock_value(
        &mut infinite,
        datetime("2024-01-02 00:00:00"),
        "day",
        &provider,
    )
    .unwrap();
    assert!(calls.lock().unwrap().is_empty());
}

#[test]
fn built_in_reset_retains_optional_metadata_and_exposes_missing_provider_stage() {
    let at = datetime("2024-01-02 00:00:00");
    let mut account = Account::new(finite(10.0, 1.0, Some(5.0)), true);
    account.update_historical_positions(at).unwrap();
    let old_positions = account.historical_positions().clone();
    let old_indicator = account.indicator().clone();
    account
        .reset(AccountResetUpdate {
            frequency: Some("2min".to_owned()),
            report_config: Some(AccountReportConfig::default()),
            portfolio_metrics_enabled: Some(true),
        })
        .unwrap();
    assert_eq!(account.frequency(), "2min");
    assert!(account.report_config().start_time.is_none());
    assert!(!Arc::ptr_eq(account.historical_positions(), &old_positions));
    assert!(!Arc::ptr_eq(account.indicator(), &old_indicator));

    let reached_indicator = account.indicator().clone();
    let error = account
        .reset(AccountResetUpdate {
            frequency: None,
            report_config: Some(AccountReportConfig {
                benchmark: None,
                start_time: Some(at),
                initial_price_provider: None,
                ..AccountReportConfig::default()
            }),
            portfolio_metrics_enabled: None,
        })
        .unwrap_err();
    assert_eq!(error, AccountResetError::MissingInitialPriceProvider);
    assert!(account.historical_positions().read().unwrap().is_empty());
    assert!(Arc::ptr_eq(account.indicator(), &reached_indicator));
    assert_eq!(account.frequency(), "2min");

    account
        .reset(AccountResetUpdate {
            frequency: None,
            report_config: Some(AccountReportConfig::default()),
            portfolio_metrics_enabled: None,
        })
        .unwrap();
    account.reset(AccountResetUpdate::default()).unwrap();
}

#[test]
fn portfolio_history_objects_survive_replacement_and_account_drop_without_cloning() {
    let mut account = report_account(1.0, true);
    let report = account.portfolio_report().unwrap();
    let first = datetime("2024-01-02 09:30:00");
    let second = datetime("2024-01-03 09:30:00");
    assert!(Arc::ptr_eq(
        &report.positions,
        account.historical_positions()
    ));
    account.update_historical_positions(second).unwrap();
    account.update_portfolio_metrics(second, second).unwrap();
    assert_eq!(report.positions.read().unwrap().len(), 2);
    assert_eq!(report.metrics.num_rows(), 1);
    assert_eq!(account.portfolio_report().unwrap().metrics.num_rows(), 2);
    // Same-object mutations through a report remain visible to the account.
    report
        .positions
        .write()
        .unwrap()
        .shift_remove(&first)
        .unwrap();
    assert!(
        !account
            .historical_positions()
            .read()
            .unwrap()
            .contains_key(&first)
    );
    let replacement = Arc::new(std::sync::RwLock::new(IndexMap::new()));
    let old = account.replace_historical_positions(replacement.clone());
    assert!(Arc::ptr_eq(&old, &report.positions));
    assert!(Arc::ptr_eq(account.historical_positions(), &replacement));
    account.update_historical_positions(first).unwrap();
    assert!(!report.positions.read().unwrap().contains_key(&first));
    drop(account);
    assert_eq!(report.positions.read().unwrap().len(), 1);
    assert_eq!(replacement.read().unwrap().len(), 1);
    assert_same(
        report.positions.read().unwrap()[&second].account_value(),
        100.0,
    );
}

#[test]
fn poisoned_history_retains_reached_position_updates_and_can_be_replaced() {
    let at = datetime("2024-01-02 09:30:00");
    let mut account = Account::new(finite(100.0, 2.0, Some(5.0)), true);
    let history = account.historical_positions().clone();
    let poison = history.clone();
    assert!(
        std::thread::spawn(move || {
            let _guard = poison.write().unwrap();
            panic!("intentional history poison");
        })
        .join()
        .is_err()
    );
    let error = account.update_historical_positions(at).unwrap_err();
    assert!(matches!(error, AccountError::HistoryPoisoned));
    assert_eq!(error.to_string(), "historical positions lock poisoned");
    assert_same(
        account.current_position().stored_account_value().unwrap(),
        110.0,
    );
    assert_same(
        account.current_position().stock_weight("A").unwrap(),
        10.0 / 110.0,
    );
    // Export returns identity, not history contents, and need not read/recover the poisoned map.
    let report = account.portfolio_report().unwrap();
    assert!(Arc::ptr_eq(&report.positions, &history));
    assert!(report.positions.read().is_err());
    let old =
        account.replace_historical_positions(Arc::new(std::sync::RwLock::new(IndexMap::new())));
    assert!(Arc::ptr_eq(&old, &history));
    account.update_historical_positions(at).unwrap();
    assert_eq!(account.historical_positions().read().unwrap().len(), 1);
    assert!(report.positions.read().is_err());
}

#[test]
fn final_report_aggregation_preserves_normalized_keys_overwrites_and_live_objects() {
    use domain_core::collect_backtest_reports;
    let empty = collect_backtest_reports([]).unwrap();
    assert!(empty.portfolio.is_empty());
    assert!(empty.indicators.is_empty());
    for enabled in [false, true] {
        let first = report_account(1.0, true);
        let middle = report_account(2.0, true);
        let last = report_account(3.0, enabled);
        let reports =
            collect_backtest_reports([("day", &first), ("min", &middle), ("1D", &last)]).unwrap();
        assert_eq!(
            reports
                .portfolio
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            ["1day", "1min"]
        );
        assert_eq!(
            reports
                .indicators
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            ["1day", "1min"]
        );
        let portfolio_owner = if enabled { &last } else { &first };
        assert!(Arc::ptr_eq(
            &reports.portfolio["1day"].positions,
            portfolio_owner.historical_positions()
        ));
        assert!(Arc::ptr_eq(
            &reports.indicators["1day"].indicator,
            last.indicator()
        ));
        assert!(Arc::ptr_eq(
            &reports.indicators["1min"].indicator,
            middle.indicator()
        ));
        let value = reports.indicators["1day"]
            .table
            .metrics
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(value.value(0).to_bits(), 3.0_f64.to_bits());
        let portfolio = reports.portfolio["1day"]
            .metrics
            .column_by_name("account")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(
            portfolio.value(0).to_bits(),
            (if enabled { 300.0_f64 } else { 100.0_f64 }).to_bits()
        );
    }
}

#[test]
fn final_report_failures_stop_traversal_and_do_not_publish_partial_maps() {
    use domain_core::{BacktestReportError, collect_backtest_reports};
    let good = report_account(1.0, true);
    let mut failing = report_account(2.0, false);
    let indicator = ProbeIndicator::new(Some("report"));
    let events = Arc::clone(&indicator.events);
    failing.replace_indicator(Box::new(indicator));
    let visited = std::cell::Cell::new(0);
    let result = collect_backtest_reports(
        [("day", &good), ("min", &failing), ("week", &good)]
            .into_iter()
            .inspect(|_| visited.set(visited.get() + 1)),
    );
    assert!(
        matches!(result, Err(BacktestReportError::Indicator(ref error)) if error.message == "report")
    );
    assert_eq!(visited.get(), 2);
    assert_eq!(
        result.err().unwrap().to_string(),
        "account indicator plugin error: report"
    );
    assert_eq!(*events.lock().unwrap(), ["report"]);
    events.lock().unwrap().clear();
    let mut published = collect_backtest_reports([("week", &good)]).unwrap();
    let result = collect_backtest_reports([("day", &good), ("bad", &failing)]);
    assert!(matches!(result, Err(BacktestReportError::Frequency(_))));
    assert_eq!(
        result.as_ref().err().unwrap().to_string(),
        "freq format is not supported, the freq should be like (n)month/mon, (n)week/w, (n)day/d, (n)minute/min: bad"
    );
    if let Ok(reports) = result {
        published = reports;
    }
    assert_eq!(
        published
            .portfolio
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["1week"]
    );
    assert!(events.lock().unwrap().is_empty());
    let mut position = ProbePosition::new(false, Ok(5.0), None);
    position.supports_metrics = false;
    let mut inconsistent = Account::new(position, true);
    let indicator = ProbeIndicator::new(None);
    let events = Arc::clone(&indicator.events);
    inconsistent.replace_indicator(Box::new(indicator));
    let result = collect_backtest_reports([("day", &inconsistent)]);
    assert!(matches!(
        result,
        Err(BacktestReportError::Account(
            AccountError::PortfolioMetricsDisabled
        ))
    ));
    assert_eq!(
        result.err().unwrap().to_string(),
        AccountError::PortfolioMetricsDisabled.to_string()
    );
    assert!(events.lock().unwrap().is_empty());
}

#[test]
fn final_report_source_contract_preserves_publication_and_accessor_order() {
    let output = Command::new("python")
        .args([
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/backtest_report_contract.py"
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
    let cases: Value = serde_json::from_slice(&output.stdout).unwrap();
    let mut prefix = vec!["finalize", "close"];
    for name in ["first", "middle"] {
        // The final source accessor is intentionally distinct from the accessor used for export.
        let additions = match name {
            "first" => [
                "enabled:first",
                "portfolio:first",
                "indicator:first",
                "export:first",
                "indicator:first",
            ],
            _ => [
                "enabled:middle",
                "portfolio:middle",
                "indicator:middle",
                "export:middle",
                "indicator:middle",
            ],
        };
        prefix.extend(additions);
    }
    for (case, enabled) in [("disabled", false), ("enabled", true)] {
        let mut events = prefix.clone();
        events.push("enabled:last");
        if enabled {
            events.push("portfolio:last");
        }
        events.extend(["indicator:last", "export:last", "indicator:last"]);
        assert_eq!(
            cases[case],
            json!({"events": events, "error": null, "result": {
                "prior": 42, "portfolio_dict": {"1day": if enabled { "last" } else { "first" }, "1min": "middle"},
                "indicator_dict": {"1day": ["last", "last"], "1min": ["middle", "middle"]}
            }})
        );
    }
    let mut events = prefix.clone();
    events.extend(["enabled:last", "indicator:last", "export:last"]);
    assert_eq!(
        cases["export_failure"],
        json!({"events": events, "error": "export failure", "result": {"prior": 42}})
    );
    assert_eq!(
        cases["frequency_failure"],
        json!({"events": prefix, "error": "freq format is not supported, the freq should be like (n)month/mon, (n)week/w, (n)day/d, (n)minute/min", "result": {"prior": 42}})
    );
    assert_eq!(
        cases["no_reports"],
        json!({"events": ["finalize", "close"], "error": null, "result": {"prior": 42}})
    );
}

fn finite(cash: f64, amount: f64, price: Option<f64>) -> Position {
    Position::from_initial(
        cash,
        IndexMap::from([(
            "A".to_owned(),
            InitialPositionValue::Holding(PositionHolding::restored(amount, price, Some(0.0))),
        )]),
    )
}

fn assert_same(actual: f64, expected: f64) {
    assert_eq!(actual.to_bits(), expected.to_bits());
}

#[test]
fn accumulated_info_adds_special_values_and_resets_exactly() {
    let mut info = AccumulatedInfo::default();
    assert_eq!(info, AccumulatedInfo::new());
    info.add_return_value(2.0);
    info.add_return_value(f64::NAN);
    info.add_cost(f64::INFINITY);
    info.add_turnover(f64::NEG_INFINITY);
    assert!(info.return_value().is_nan());
    assert_same(info.cost(), f64::INFINITY);
    assert_same(info.turnover(), f64::NEG_INFINITY);
    info.reset();
    assert_same(info.return_value(), 0.0);
    assert_same(info.cost(), 0.0);
    assert_same(info.turnover(), 0.0);
}

#[test]
fn finite_account_composes_directional_position_and_metric_updates() {
    let mut account = Account::new(finite(100.0, 2.0, Some(5.0)), true);
    assert!(account.is_portfolio_metrics_enabled());
    account
        .update_order(&order(OrderDir::Sell), 10.0, 1.0, 10.0)
        .unwrap();
    assert_same(account.accumulated_info().return_value(), 5.0);
    assert_same(account.accumulated_info().cost(), 1.0);
    assert_same(account.accumulated_info().turnover(), 10.0);
    assert_same(
        account
            .execution_position()
            .unwrap()
            .stock_amount("A")
            .unwrap(),
        1.0,
    );
    assert_same(account.execution_position().unwrap().cash().unwrap(), 109.0);

    account
        .update_order(&order(OrderDir::Buy), 20.0, 2.0, 10.0)
        .unwrap();
    assert_same(account.accumulated_info().return_value(), -5.0);
    assert_same(account.accumulated_info().cost(), 3.0);
    assert_same(account.accumulated_info().turnover(), 30.0);
    assert_same(account.current_position().stock_price("A").unwrap(), 5.0);
    assert_same(
        account
            .execution_position()
            .unwrap()
            .stock_amount("A")
            .unwrap(),
        3.0,
    );

    let target: &mut dyn ExecutionTarget = &mut account;
    assert_same(target.position().unwrap().cash().unwrap(), 87.0);
    target
        .update_order(&order(OrderDir::Buy), 10.0, 1.0, 10.0)
        .unwrap();
    assert_same(target.position().unwrap().stock_amount("A").unwrap(), 4.0);
}

#[test]
fn disabled_metrics_and_infinite_positions_preserve_short_circuits() {
    let mut disabled = Account::new(finite(100.0, 2.0, Some(5.0)), false);
    assert!(!disabled.is_portfolio_metrics_enabled());
    disabled
        .update_order(&order(OrderDir::Sell), 10.0, 1.0, 10.0)
        .unwrap();
    assert_eq!(*disabled.accumulated_info(), AccumulatedInfo::new());
    assert_same(
        disabled
            .execution_position()
            .unwrap()
            .stock_amount("A")
            .unwrap(),
        1.0,
    );

    let mut infinite = Account::new(InfinitePosition, true);
    assert!(!infinite.is_portfolio_metrics_enabled());
    infinite
        .update_order(&order(OrderDir::Buy), f64::NAN, f64::INFINITY, 0.0)
        .unwrap();
    assert_eq!(*infinite.accumulated_info(), AccumulatedInfo::new());
    assert!(
        infinite
            .execution_position()
            .unwrap()
            .stock_amount("anything")
            .unwrap()
            .is_infinite()
    );
    assert!(
        infinite
            .current_position()
            .stock_price("anything")
            .unwrap()
            .is_nan()
    );
    infinite
        .current_position_mut()
        .update_order(&order(OrderDir::Sell), f64::NAN, f64::NAN, 0.0)
        .unwrap();
}

#[test]
fn real_position_failures_retain_only_reached_account_state() {
    let mut sell_zero = Account::new(finite(100.0, 2.0, Some(5.0)), true);
    assert_eq!(
        sell_zero.update_order(&order(OrderDir::Sell), 10.0, 1.0, 0.0),
        Err(AccountError::ZeroTradePrice)
    );
    assert_same(sell_zero.accumulated_info().turnover(), 10.0);
    assert_same(sell_zero.accumulated_info().cost(), 1.0);
    assert_same(sell_zero.accumulated_info().return_value(), 0.0);
    assert_same(
        sell_zero
            .execution_position()
            .unwrap()
            .stock_amount("A")
            .unwrap(),
        2.0,
    );

    let mut target_error = Account::new(finite(100.0, 2.0, Some(5.0)), true);
    let target: &mut dyn ExecutionTarget = &mut target_error;
    let Err(error) = target.update_order(&order(OrderDir::Sell), 10.0, 1.0, 0.0) else {
        panic!("expected the account target update to fail");
    };
    assert_eq!(error.message, "trade price cannot be zero");

    let mut buy_zero = Account::new(finite(100.0, 2.0, Some(5.0)), true);
    assert_eq!(
        buy_zero.update_order(&order(OrderDir::Buy), 10.0, 1.0, 0.0),
        Err(AccountError::Position(AccountPositionError {
            message: PositionError::ZeroTradePrice.to_string(),
        }))
    );
    assert_eq!(*buy_zero.accumulated_info(), AccumulatedInfo::new());

    let mut missing_price = Account::new(finite(100.0, 2.0, None), true);
    assert_eq!(
        missing_price.update_order(&order(OrderDir::Sell), 10.0, 1.0, 10.0),
        Err(AccountError::Position(AccountPositionError {
            message: "stock A has no price".to_owned(),
        }))
    );
    assert_same(missing_price.accumulated_info().turnover(), 10.0);
    assert_same(missing_price.accumulated_info().cost(), 1.0);
    assert_same(missing_price.accumulated_info().return_value(), 0.0);

    let mut oversell = Account::new(finite(100.0, 1.0, Some(5.0)), true);
    assert!(matches!(
        oversell.update_order(&order(OrderDir::Sell), 20.0, 2.0, 10.0),
        Err(AccountError::Position(_))
    ));
    assert_same(oversell.accumulated_info().return_value(), 10.0);
    assert_same(oversell.accumulated_info().cost(), 2.0);
    assert_same(oversell.accumulated_info().turnover(), 20.0);
    assert_same(
        oversell
            .execution_position()
            .unwrap()
            .stock_amount("A")
            .unwrap(),
        -1.0,
    );
    assert_same(
        oversell.execution_position().unwrap().cash().unwrap(),
        100.0,
    );
}

#[derive(Clone)]
struct ProbePosition {
    events: Arc<Mutex<Vec<String>>>,
    skip: bool,
    supports_metrics: bool,
    price: Result<f64, String>,
    update_error: Option<String>,
    view_error: Option<String>,
    reset_events: Option<Arc<Mutex<Vec<Value>>>>,
    fill_error: Option<String>,
    reset_skip: Option<Arc<AtomicBool>>,
}

impl ProbePosition {
    fn new(skip: bool, price: Result<f64, &str>, update_error: Option<&str>) -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
            skip,
            supports_metrics: !skip,
            price: price.map_err(str::to_owned),
            update_error: update_error.map(str::to_owned),
            view_error: None,
            reset_events: None,
            fill_error: None,
            reset_skip: None,
        }
    }

    fn record(&self, event: impl Into<String>) {
        self.events.lock().unwrap().push(event.into());
    }
}

impl ExecutionPosition for ProbePosition {
    fn check_stock(&self, _stock: &str) -> Result<bool, ExecutionPositionError> {
        Ok(true)
    }

    fn stock_amount(&self, _stock: &str) -> Result<f64, ExecutionPositionError> {
        Ok(1.0)
    }

    fn cash(&self) -> Result<f64, ExecutionPositionError> {
        Ok(100.0)
    }
}

impl AccountPosition for ProbePosition {
    fn portfolio_metrics_supported(&self) -> bool {
        self.supports_metrics
    }

    fn initial_cash(&self) -> f64 {
        100.0
    }

    fn total_value(&self) -> Result<f64, AccountPositionError> {
        self.record("total");
        Ok(105.0)
    }

    fn stock_value(&self) -> Result<f64, AccountPositionError> {
        self.record("stock_value");
        Ok(5.0)
    }

    fn available_cash(&self) -> Result<f64, AccountPositionError> {
        self.record("cash");
        Ok(100.0)
    }

    fn stored_account_value(&self) -> Option<f64> {
        None
    }

    fn stock_weight(&self, _stock: &str) -> Result<f64, AccountPositionError> {
        Ok(0.0)
    }

    fn set_account_value(&mut self, value: f64) -> Result<(), AccountPositionError> {
        self.record(format!("account:{value:?}"));
        Ok(())
    }

    fn update_weights(&mut self) -> Result<(), AccountPositionError> {
        self.record("weights");
        Ok(())
    }

    fn history_snapshot(&self) -> Result<Box<dyn AccountPosition>, AccountPositionError> {
        self.record("snapshot");
        Ok(Box::new(self.clone()))
    }

    fn fill_stock_value(
        &mut self,
        start_time: NaiveDateTime,
        frequency: &str,
        _provider: &dyn InitialStockPriceProvider,
    ) -> Result<(), AccountPositionError> {
        if let Some(events) = &self.reset_events {
            events.lock().unwrap().push(json!([
                "fill",
                start_time.format("%Y-%m-%dT%H:%M:%S").to_string(),
                frequency
            ]));
        }
        self.fill_error
            .clone()
            .map_or(Ok(()), |message| Err(AccountPositionError { message }))
    }

    fn skip_update(&self) -> bool {
        self.record("skip");
        self.reset_skip
            .as_ref()
            .map_or(self.skip, |skip| skip.load(Ordering::SeqCst))
    }

    fn execution_position(&self) -> Result<&dyn ExecutionPosition, AccountPositionError> {
        self.record("view");
        if let Some(message) = &self.view_error {
            return Err(AccountPositionError {
                message: message.clone(),
            });
        }
        Ok(self)
    }

    fn stock_ids(&self) -> Result<Vec<String>, AccountPositionError> {
        self.record("stocks");
        Ok(vec!["A".to_owned()])
    }

    fn stock_price(&self, stock: &str) -> Result<f64, AccountPositionError> {
        self.record(format!("price:{stock}"));
        self.price
            .clone()
            .map_err(|message| AccountPositionError { message })
    }

    fn update_stock_price(&mut self, stock: &str, price: f64) -> Result<(), AccountPositionError> {
        self.record(format!("mark:{stock}:{price:?}"));
        Ok(())
    }

    fn add_count_all(&mut self, bar: &str) -> Result<(), AccountPositionError> {
        self.record(format!("count:{bar}"));
        Ok(())
    }

    fn settle_start(&mut self, _settlement_type: &str) -> Result<(), AccountPositionError> {
        Ok(())
    }

    fn settle_commit(&mut self) -> Result<(), AccountPositionError> {
        Ok(())
    }

    fn update_order(
        &mut self,
        order: &Order,
        _trade_value: f64,
        _trade_cost: f64,
        _trade_price: f64,
    ) -> Result<(), AccountPositionError> {
        let direction = match order.direction() {
            OrderDir::Sell => 0,
            OrderDir::Buy => 1,
        };
        self.record(format!("update:{direction}"));
        if let Some(message) = &self.update_error {
            return Err(AccountPositionError {
                message: message.clone(),
            });
        }
        Ok(())
    }
}

fn events(events: &Arc<Mutex<Vec<String>>>) -> Vec<String> {
    events.lock().unwrap().clone()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResetFailure {
    None,
    Portfolio,
    Fill,
    Indicator,
}

struct ResetFactory {
    events: Arc<Mutex<Vec<Value>>>,
    failure: ResetFailure,
}

impl AccountReportFactory for ResetFactory {
    fn portfolio_metrics(
        &self,
        frequency: &str,
        config: &AccountReportConfig,
    ) -> Result<PortfolioMetrics, AccountReportFactoryError> {
        self.events
            .lock()
            .unwrap()
            .push(json!(["metrics", frequency]));
        if self.failure == ResetFailure::Portfolio {
            return Err(AccountReportFactoryError {
                message: "metrics".to_owned(),
            });
        }
        Ok(PortfolioMetrics::new(frequency, config.benchmark.clone()))
    }

    fn indicator(&self) -> Result<Box<dyn AccountIndicator>, AccountReportFactoryError> {
        self.events.lock().unwrap().push(json!(["indicator"]));
        if self.failure == ResetFailure::Indicator {
            return Err(AccountReportFactoryError {
                message: "indicator".to_owned(),
            });
        }
        Ok(Box::new(Indicator::new()))
    }
}

#[derive(Clone)]
struct PriceProvider {
    calls: Arc<Mutex<Vec<Value>>>,
    prices: Result<IndexMap<String, f64>, InitialStockPriceProviderError>,
}

impl InitialStockPriceProvider for PriceProvider {
    fn latest_close_prices(
        &self,
        request: InitialStockPriceRequest<'_>,
    ) -> Result<IndexMap<String, f64>, InitialStockPriceProviderError> {
        self.calls.lock().unwrap().push(json!({
            "stocks": request.stocks,
            "start": request.start_time.format("%Y-%m-%dT%H:%M:%S").to_string(),
            "end": request.end_time.format("%Y-%m-%dT%H:%M:%S").to_string(),
            "frequency": request.frequency,
            "disk_cache": request.disk_cache,
        }));
        self.prices.clone()
    }
}

fn reset_failure(mode: &str) -> ResetFailure {
    match mode {
        "portfolio_failure" => ResetFailure::Portfolio,
        "fill_failure" => ResetFailure::Fill,
        "indicator_failure" => ResetFailure::Indicator,
        _ => ResetFailure::None,
    }
}

fn native_reset_row(mode: &str) -> Value {
    let reset_events = Arc::new(Mutex::new(Vec::new()));
    let skip = Arc::new(AtomicBool::new(false));
    let mut position = ProbePosition::new(false, Ok(5.0), None);
    position.reset_events = Some(reset_events.clone());
    position.reset_skip = Some(skip.clone());
    if mode == "fill_failure" {
        position.fill_error = Some("fill".to_owned());
    }
    let mut account = Account::new(position, true);
    let start = datetime("2024-01-02 00:00:00");
    account.update_portfolio_metrics(start, start).unwrap();
    account.update_historical_positions(start).unwrap();
    let old_positions = account.historical_positions().clone();
    let old_indicator = account.indicator().clone();
    let old_position = std::ptr::from_ref(account.current_position()).cast::<()>();
    account.accumulated_info_mut().add_return_value(3.0);
    skip.store(mode == "skip", Ordering::SeqCst);

    let provider = Arc::new(PriceProvider {
        calls: Arc::new(Mutex::new(Vec::new())),
        prices: Ok(IndexMap::new()),
    });
    let factory = ResetFactory {
        events: reset_events.clone(),
        failure: reset_failure(mode),
    };
    let result = account.reset_with_factory(
        AccountResetUpdate {
            frequency: Some("2min".to_owned()),
            report_config: Some(AccountReportConfig {
                benchmark: None,
                start_time: Some(start),
                initial_price_provider: Some(provider),
                ..AccountReportConfig::default()
            }),
            portfolio_metrics_enabled: Some(mode != "disabled"),
        },
        &factory,
    );
    let error = result.err().map(|error| match error {
        AccountResetError::Position(error) => error.message,
        other => other.to_string(),
    });
    json!({
        "mode": mode,
        "events": reset_events.lock().unwrap().clone(),
        "error": error,
        "frequency": account.frequency(),
        "same_metrics": account.portfolio_metrics().is_some_and(|metrics| !metrics.is_empty()),
        "same_positions": Arc::ptr_eq(account.historical_positions(), &old_positions),
        "same_indicator": Arc::ptr_eq(account.indicator(), &old_indicator),
        "same_position": old_position == std::ptr::from_ref(account.current_position()).cast::<()>(),
        "same_accumulated": account.accumulated_info().return_value().to_bits() == 3.0f64.to_bits(),
    })
}

fn datetime(text: &str) -> NaiveDateTime {
    NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S").unwrap()
}

struct ScriptMarket {
    events: Arc<Mutex<Vec<String>>>,
    suspended: Mutex<VecDeque<Result<bool, AccountBarMarketError>>>,
    closes: Mutex<VecDeque<Result<f64, AccountBarMarketError>>>,
}

impl ScriptMarket {
    fn new(suspended: Vec<Result<bool, &str>>, closes: Vec<Result<f64, &str>>) -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
            suspended: Mutex::new(
                suspended
                    .into_iter()
                    .map(|result| {
                        result.map_err(|message| AccountBarMarketError {
                            message: message.to_owned(),
                        })
                    })
                    .collect(),
            ),
            closes: Mutex::new(
                closes
                    .into_iter()
                    .map(|result| {
                        result.map_err(|message| AccountBarMarketError {
                            message: message.to_owned(),
                        })
                    })
                    .collect(),
            ),
        }
    }
}

impl AccountBarMarket for ScriptMarket {
    fn is_suspended(&self, stock: &str, range: TimeRange) -> Result<bool, AccountBarMarketError> {
        self.events.lock().unwrap().push(format!(
            "suspended:{stock}:{}:{}",
            range.start.unwrap(),
            range.end.unwrap()
        ));
        self.suspended.lock().unwrap().pop_front().unwrap()
    }

    fn close(&self, stock: &str, range: TimeRange) -> Result<f64, AccountBarMarketError> {
        self.events.lock().unwrap().push(format!(
            "close:{stock}:{}:{}",
            range.start.unwrap(),
            range.end.unwrap()
        ));
        self.closes.lock().unwrap().pop_front().unwrap()
    }
}

#[derive(Clone)]
struct BarPosition {
    events: Arc<Mutex<Vec<String>>>,
    skip: bool,
    stocks: Result<Vec<String>, String>,
    mark_error: Option<String>,
    count_error: Option<String>,
}

impl BarPosition {
    fn new(stocks: Result<Vec<&str>, &str>) -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
            skip: false,
            stocks: stocks
                .map(|stocks| stocks.into_iter().map(str::to_owned).collect())
                .map_err(str::to_owned),
            mark_error: None,
            count_error: None,
        }
    }

    fn record(&self, event: impl Into<String>) {
        self.events.lock().unwrap().push(event.into());
    }
}

impl ExecutionPosition for BarPosition {
    fn check_stock(&self, _stock: &str) -> Result<bool, ExecutionPositionError> {
        Ok(true)
    }

    fn stock_amount(&self, _stock: &str) -> Result<f64, ExecutionPositionError> {
        Ok(1.0)
    }

    fn cash(&self) -> Result<f64, ExecutionPositionError> {
        Ok(0.0)
    }
}

impl AccountPosition for BarPosition {
    fn portfolio_metrics_supported(&self) -> bool {
        !self.skip
    }

    fn initial_cash(&self) -> f64 {
        0.0
    }

    fn total_value(&self) -> Result<f64, AccountPositionError> {
        Ok(0.0)
    }

    fn stock_value(&self) -> Result<f64, AccountPositionError> {
        Ok(0.0)
    }

    fn available_cash(&self) -> Result<f64, AccountPositionError> {
        Ok(0.0)
    }

    fn stored_account_value(&self) -> Option<f64> {
        None
    }

    fn stock_weight(&self, _stock: &str) -> Result<f64, AccountPositionError> {
        Ok(0.0)
    }

    fn set_account_value(&mut self, _value: f64) -> Result<(), AccountPositionError> {
        Ok(())
    }

    fn update_weights(&mut self) -> Result<(), AccountPositionError> {
        Ok(())
    }

    fn history_snapshot(&self) -> Result<Box<dyn AccountPosition>, AccountPositionError> {
        Ok(Box::new(self.clone()))
    }

    fn skip_update(&self) -> bool {
        self.record("skip");
        self.skip
    }

    fn execution_position(&self) -> Result<&dyn ExecutionPosition, AccountPositionError> {
        Ok(self)
    }

    fn stock_ids(&self) -> Result<Vec<String>, AccountPositionError> {
        self.record("stocks");
        self.stocks
            .clone()
            .map_err(|message| AccountPositionError { message })
    }

    fn stock_price(&self, _stock: &str) -> Result<f64, AccountPositionError> {
        Ok(0.0)
    }

    fn update_stock_price(&mut self, stock: &str, price: f64) -> Result<(), AccountPositionError> {
        self.record(format!("mark:{stock}:{price:?}"));
        if let Some(message) = &self.mark_error {
            return Err(AccountPositionError {
                message: message.clone(),
            });
        }
        Ok(())
    }

    fn add_count_all(&mut self, bar: &str) -> Result<(), AccountPositionError> {
        self.record(format!("count:{bar}"));
        if let Some(message) = &self.count_error {
            return Err(AccountPositionError {
                message: message.clone(),
            });
        }
        Ok(())
    }

    fn settle_start(&mut self, _settlement_type: &str) -> Result<(), AccountPositionError> {
        Ok(())
    }

    fn settle_commit(&mut self) -> Result<(), AccountPositionError> {
        Ok(())
    }

    fn update_order(
        &mut self,
        _order: &Order,
        _trade_value: f64,
        _trade_cost: f64,
        _trade_price: f64,
    ) -> Result<(), AccountPositionError> {
        Ok(())
    }
}

#[test]
fn bar_update_preserves_order_ranges_suspension_and_frequency() {
    let position = BarPosition::new(Ok(vec!["A", "B"]));
    let position_events = Arc::clone(&position.events);
    let market = ScriptMarket::new(vec![Ok(false), Ok(true)], vec![Ok(7.0)]);
    let market_events = Arc::clone(&market.events);
    let start = datetime("2024-01-02 09:30:00");
    let end = datetime("2024-01-02 09:31:00");
    let mut account = Account::with_frequency(position, true, "1min");
    assert_eq!(account.frequency(), "1min");
    account
        .update_current_position(start, end, &market)
        .unwrap();
    assert_eq!(
        events(&position_events),
        ["skip", "stocks", "mark:A:7.0", "count:1min"]
    );
    assert_eq!(
        events(&market_events),
        [
            "suspended:A:2024-01-02 09:30:00:2024-01-02 09:31:00",
            "close:A:2024-01-02 09:30:00:2024-01-02 09:31:00",
            "suspended:B:2024-01-02 09:30:00:2024-01-02 09:31:00",
        ]
    );
}

#[test]
fn bar_update_stops_at_each_plugin_failure_without_counting() {
    let start = datetime("2024-01-02 09:30:00");
    let end = datetime("2024-01-02 09:31:00");

    let position = BarPosition::new(Err("stocks"));
    let log = Arc::clone(&position.events);
    let mut account = Account::new(position, true);
    assert!(matches!(
        account.update_current_position(start, end, &ScriptMarket::new(vec![], vec![])),
        Err(AccountError::Position(_))
    ));
    assert_eq!(events(&log), ["skip", "stocks"]);

    let position = BarPosition::new(Ok(vec!["A"]));
    let log = Arc::clone(&position.events);
    let mut account = Account::new(position, true);
    assert!(matches!(
        account.update_current_position(
            start,
            end,
            &ScriptMarket::new(vec![Err("suspended")], vec![])
        ),
        Err(AccountError::Market(_))
    ));
    assert_eq!(events(&log), ["skip", "stocks"]);

    let position = BarPosition::new(Ok(vec!["A"]));
    let log = Arc::clone(&position.events);
    let mut account = Account::new(position, true);
    assert!(matches!(
        account.update_current_position(
            start,
            end,
            &ScriptMarket::new(vec![Ok(false)], vec![Err("close")])
        ),
        Err(AccountError::Market(_))
    ));
    assert_eq!(events(&log), ["skip", "stocks"]);

    let mut position = BarPosition::new(Ok(vec!["A"]));
    position.mark_error = Some("mark".to_owned());
    let log = Arc::clone(&position.events);
    let mut account = Account::new(position, true);
    assert!(matches!(
        account.update_current_position(
            start,
            end,
            &ScriptMarket::new(vec![Ok(false)], vec![Ok(3.0)])
        ),
        Err(AccountError::Position(_))
    ));
    assert_eq!(events(&log), ["skip", "stocks", "mark:A:3.0"]);

    let mut position = BarPosition::new(Ok(vec![]));
    position.count_error = Some("count".to_owned());
    let log = Arc::clone(&position.events);
    let mut account = Account::from_boxed_with_frequency(Box::new(position), true, "week");
    assert!(matches!(
        account.update_current_position(start, end, &ScriptMarket::new(vec![], vec![])),
        Err(AccountError::Position(_))
    ));
    assert_eq!(events(&log), ["skip", "stocks", "count:week"]);
}

struct ScriptQuote {
    stocks: Vec<String>,
    responses: Mutex<VecDeque<Result<Option<QuoteData>, QuoteError>>>,
}

impl Quote for ScriptQuote {
    fn get_all_stock(&self) -> Vec<String> {
        self.stocks.clone()
    }

    fn get_data(
        &self,
        _stock: &str,
        _range: TimeRange,
        _field: &str,
        _method: QuoteMethod,
    ) -> Result<Option<QuoteData>, QuoteError> {
        self.responses.lock().unwrap().pop_front().unwrap()
    }
}

fn scalar(value: f64) -> QuoteData {
    QuoteData::Scalar(Arc::new(Float64Array::from(vec![value])))
}

fn exchange(
    stocks: &[&str],
    responses: Vec<Result<Option<QuoteData>, QuoteError>>,
) -> ExchangeQuoteProvider {
    ExchangeQuoteProvider::new(
        Arc::new(ScriptQuote {
            stocks: stocks.iter().map(|stock| (*stock).to_owned()).collect(),
            responses: Mutex::new(responses.into()),
        }),
        DealPriceFields::directional("$close", "$close"),
    )
}

#[test]
fn exchange_quote_provider_is_a_bar_market_plugin() {
    let range = TimeRange::default();
    let provider = exchange(&["A"], vec![Ok(Some(scalar(1.0))), Ok(Some(scalar(9.0)))]);
    assert!(!AccountBarMarket::is_suspended(&provider, "A", range).unwrap());
    assert_same(AccountBarMarket::close(&provider, "A", range).unwrap(), 9.0);

    let absent = exchange(&["A"], vec![Ok(None)]);
    assert_eq!(
        AccountBarMarket::close(&absent, "A", range),
        Err(AccountBarMarketError {
            message: "stock A has no close price in the requested bar".to_owned(),
        })
    );

    let quote_error = QuoteError::MissingStock {
        stock: "A".to_owned(),
    };
    let failed = exchange(&["A"], vec![Err(quote_error)]);
    assert!(AccountBarMarket::is_suspended(&failed, "A", range).is_err());

    let quote_error = QuoteError::MissingStock {
        stock: "A".to_owned(),
    };
    let failed = exchange(&["A"], vec![Err(quote_error)]);
    assert!(AccountBarMarket::close(&failed, "A", range).is_err());
}

#[test]
fn built_in_position_adapters_cover_finite_and_infinite_bar_operations() {
    let mut position = finite(100.0, 2.0, Some(5.0));
    assert!(AccountPosition::portfolio_metrics_supported(&position));
    assert_same(AccountPosition::initial_cash(&position), 100.0);
    assert_same(AccountPosition::total_value(&position).unwrap(), 110.0);
    assert_same(AccountPosition::stock_value(&position).unwrap(), 10.0);
    assert_same(AccountPosition::available_cash(&position).unwrap(), 100.0);
    assert_eq!(AccountPosition::stored_account_value(&position), None);
    assert_same(AccountPosition::stock_weight(&position, "A").unwrap(), 0.0);
    AccountPosition::set_account_value(&mut position, 110.0).unwrap();
    AccountPosition::update_weights(&mut position).unwrap();
    assert_eq!(
        AccountPosition::stored_account_value(&position),
        Some(110.0)
    );
    assert_same(
        AccountPosition::stock_weight(&position, "A").unwrap(),
        10.0 / 110.0,
    );
    let snapshot = AccountPosition::history_snapshot(&position).unwrap();
    assert_eq!(snapshot.stored_account_value(), Some(110.0));
    assert_eq!(AccountPosition::stock_ids(&position).unwrap(), ["A"]);
    AccountPosition::update_stock_price(&mut position, "A", f64::NAN).unwrap();
    assert!(position.stock_price("A").unwrap().is_nan());
    AccountPosition::add_count_all(&mut position, "day").unwrap();
    assert_same(
        *position.holding("A").unwrap().counts().get("day").unwrap(),
        1.0,
    );
    assert!(AccountPosition::update_stock_price(&mut position, "Z", 1.0).is_err());

    let missing_price = finite(100.0, 2.0, None);
    assert!(AccountPosition::total_value(&missing_price).is_err());
    assert!(AccountPosition::stock_value(&missing_price).is_err());
    assert!(AccountPosition::stock_weight(&missing_price, "Z").is_err());

    let mut infinite = InfinitePosition;
    assert!(!AccountPosition::portfolio_metrics_supported(&infinite));
    assert!(AccountPosition::initial_cash(&infinite).is_infinite());
    assert!(AccountPosition::total_value(&infinite).is_err());
    assert!(
        AccountPosition::stock_value(&infinite)
            .unwrap()
            .is_infinite()
    );
    assert!(
        AccountPosition::available_cash(&infinite)
            .unwrap()
            .is_infinite()
    );
    assert_eq!(AccountPosition::stored_account_value(&infinite), None);
    assert!(AccountPosition::stock_weight(&infinite, "A").is_err());
    assert!(AccountPosition::set_account_value(&mut infinite, 1.0).is_err());
    assert!(AccountPosition::update_weights(&mut infinite).is_err());
    assert!(AccountPosition::history_snapshot(&infinite).is_ok());
    assert!(AccountPosition::stock_ids(&infinite).is_err());
    AccountPosition::update_stock_price(&mut infinite, "A", f64::NAN).unwrap();
    assert!(AccountPosition::add_count_all(&mut infinite, "day").is_err());
    let mut account = Account::new(infinite, true);
    account
        .update_current_position(
            datetime("2024-01-02 09:30:00"),
            datetime("2024-01-02 09:31:00"),
            &ScriptMarket::new(vec![], vec![]),
        )
        .unwrap();
}

#[derive(Clone)]
struct FixedBenchmark {
    result: Result<Option<f64>, BenchmarkReturnSamplerError>,
    events: Arc<Mutex<Vec<String>>>,
}

impl BenchmarkReturnSampler for FixedBenchmark {
    fn sample_return(
        &self,
        _trade_start_time: NaiveDateTime,
        _trade_end_time: NaiveDateTime,
    ) -> Result<Option<f64>, BenchmarkReturnSamplerError> {
        self.events.lock().unwrap().push("benchmark".to_owned());
        self.result.clone()
    }
}

#[test]
fn portfolio_metrics_use_initial_then_latest_account_baselines() {
    let benchmark = FixedBenchmark {
        result: Ok(Some(0.03)),
        events: Arc::new(Mutex::new(Vec::new())),
    };
    let benchmark_events = Arc::clone(&benchmark.events);
    let mut account = Account::with_portfolio_metrics(
        finite(100.0, 2.0, Some(5.0)),
        true,
        "day",
        Some(Arc::new(benchmark)),
    );
    assert_same(account.initial_cash(), 100.0);
    assert_eq!(account.portfolio_metrics().unwrap().frequency(), "day");
    account.accumulated_info_mut().add_cost(2.0);
    account.accumulated_info_mut().add_turnover(20.0);
    let first_start = datetime("2024-01-02 00:00:00");
    let first_end = datetime("2024-01-02 23:59:59");
    account
        .update_portfolio_metrics(first_start, first_end)
        .unwrap();
    let first = account
        .portfolio_metrics()
        .unwrap()
        .latest_record()
        .unwrap();
    assert_same(first.account_value, 110.0);
    assert_same(first.stock_value, 10.0);
    assert_same(first.cash, 100.0);
    assert_same(first.return_rate, 0.12);
    assert_same(first.turnover_rate, 0.2);
    assert_same(first.cost_rate, 0.02);
    assert_eq!(first.bench_value, Some(0.03));

    account
        .current_position_mut()
        .update_stock_price("A", 6.0)
        .unwrap();
    account.accumulated_info_mut().add_cost(1.0);
    account.accumulated_info_mut().add_turnover(12.0);
    let second_start = datetime("2024-01-03 00:00:00");
    account
        .update_portfolio_metrics(second_start, datetime("2024-01-03 23:59:59"))
        .unwrap();
    let second = account
        .portfolio_metrics()
        .unwrap()
        .latest_record()
        .unwrap();
    assert_same(second.account_value, 112.0);
    assert_same(second.return_rate, 3.0 / 110.0);
    assert_same(second.turnover_rate, 12.0 / 110.0);
    assert_same(second.cost_rate, 1.0 / 110.0);
    assert_eq!(events(&benchmark_events), ["benchmark", "benchmark"]);
    account.portfolio_metrics_mut().unwrap().clear();
    assert!(account.portfolio_metrics().unwrap().is_empty());
}

#[derive(Clone)]
struct MetricsPosition {
    events: Arc<Mutex<Vec<String>>>,
    initial_cash: f64,
    total: Result<f64, String>,
    stock: Result<f64, String>,
    cash: Result<f64, String>,
    set_error: Option<String>,
    weights_error: Option<String>,
    snapshot_error: Option<String>,
    count_error: Option<String>,
    stored_account_value: Option<f64>,
}

impl MetricsPosition {
    fn result(&self, name: &str, value: &Result<f64, String>) -> Result<f64, AccountPositionError> {
        self.events.lock().unwrap().push(name.to_owned());
        value
            .clone()
            .map_err(|message| AccountPositionError { message })
    }
}

impl ExecutionPosition for MetricsPosition {
    fn check_stock(&self, _stock: &str) -> Result<bool, ExecutionPositionError> {
        Ok(true)
    }
    fn stock_amount(&self, _stock: &str) -> Result<f64, ExecutionPositionError> {
        Ok(0.0)
    }
    fn cash(&self) -> Result<f64, ExecutionPositionError> {
        Ok(0.0)
    }
}

impl AccountPosition for MetricsPosition {
    fn portfolio_metrics_supported(&self) -> bool {
        true
    }
    fn initial_cash(&self) -> f64 {
        self.initial_cash
    }
    fn total_value(&self) -> Result<f64, AccountPositionError> {
        self.result("total", &self.total)
    }
    fn stock_value(&self) -> Result<f64, AccountPositionError> {
        self.result("stock", &self.stock)
    }
    fn available_cash(&self) -> Result<f64, AccountPositionError> {
        self.result("cash", &self.cash)
    }
    fn stored_account_value(&self) -> Option<f64> {
        self.stored_account_value
    }
    fn stock_weight(&self, _stock: &str) -> Result<f64, AccountPositionError> {
        Ok(0.0)
    }
    fn set_account_value(&mut self, value: f64) -> Result<(), AccountPositionError> {
        self.events
            .lock()
            .unwrap()
            .push(format!("account:{value:?}"));
        if let Some(message) = &self.set_error {
            return Err(AccountPositionError {
                message: message.clone(),
            });
        }
        self.stored_account_value = Some(value);
        Ok(())
    }
    fn update_weights(&mut self) -> Result<(), AccountPositionError> {
        self.events.lock().unwrap().push("weights".to_owned());
        self.weights_error
            .clone()
            .map_or(Ok(()), |message| Err(AccountPositionError { message }))
    }
    fn history_snapshot(&self) -> Result<Box<dyn AccountPosition>, AccountPositionError> {
        self.events.lock().unwrap().push("snapshot".to_owned());
        if let Some(message) = &self.snapshot_error {
            return Err(AccountPositionError {
                message: message.clone(),
            });
        }
        Ok(Box::new(self.clone()))
    }
    fn skip_update(&self) -> bool {
        false
    }
    fn execution_position(&self) -> Result<&dyn ExecutionPosition, AccountPositionError> {
        Ok(self)
    }
    fn stock_ids(&self) -> Result<Vec<String>, AccountPositionError> {
        Ok(Vec::new())
    }
    fn stock_price(&self, _stock: &str) -> Result<f64, AccountPositionError> {
        Ok(0.0)
    }
    fn update_stock_price(
        &mut self,
        _stock: &str,
        _price: f64,
    ) -> Result<(), AccountPositionError> {
        Ok(())
    }
    fn add_count_all(&mut self, bar: &str) -> Result<(), AccountPositionError> {
        self.events.lock().unwrap().push(format!("count:{bar}"));
        self.count_error
            .clone()
            .map_or(Ok(()), |message| Err(AccountPositionError { message }))
    }
    fn settle_start(&mut self, _settlement_type: &str) -> Result<(), AccountPositionError> {
        Ok(())
    }
    fn settle_commit(&mut self) -> Result<(), AccountPositionError> {
        Ok(())
    }
    fn update_order(
        &mut self,
        _order: &Order,
        _trade_value: f64,
        _trade_cost: f64,
        _trade_price: f64,
    ) -> Result<(), AccountPositionError> {
        Ok(())
    }
}

fn metrics_position(initial_cash: f64) -> MetricsPosition {
    MetricsPosition {
        events: Arc::new(Mutex::new(Vec::new())),
        initial_cash,
        total: Ok(105.0),
        stock: Ok(5.0),
        cash: Ok(100.0),
        set_error: None,
        weights_error: None,
        snapshot_error: None,
        count_error: None,
        stored_account_value: None,
    }
}

#[test]
fn portfolio_metric_failures_stop_before_ledger_mutation() {
    let start = datetime("2024-01-02 00:00:00");
    let end = datetime("2024-01-02 23:59:59");
    let mut disabled = Account::from_boxed_with_portfolio_metrics(
        Box::new(metrics_position(100.0)),
        false,
        "day",
        None,
    );
    assert!(disabled.portfolio_metrics().is_none());
    assert_eq!(
        disabled.update_portfolio_metrics(start, end),
        Err(AccountError::PortfolioMetricsDisabled)
    );

    for (field, expected) in [
        ("total", vec!["total"]),
        ("stock", vec!["total", "stock"]),
        ("cash", vec!["total", "stock", "cash"]),
    ] {
        let mut position = metrics_position(100.0);
        match field {
            "total" => position.total = Err(field.to_owned()),
            "stock" => position.stock = Err(field.to_owned()),
            "cash" => position.cash = Err(field.to_owned()),
            _ => unreachable!(),
        }
        let log = Arc::clone(&position.events);
        let mut account = Account::new(position, true);
        assert!(matches!(
            account.update_portfolio_metrics(start, end),
            Err(AccountError::Position(_))
        ));
        assert_eq!(events(&log), expected);
        assert!(account.portfolio_metrics().unwrap().is_empty());
    }

    let zero = metrics_position(0.0);
    let zero_log = Arc::clone(&zero.events);
    let mut account = Account::new(zero, true);
    assert_eq!(
        account.update_portfolio_metrics(start, end),
        Err(AccountError::ZeroPreviousAccountValue)
    );
    assert_eq!(events(&zero_log), ["total", "stock", "cash"]);
    assert!(account.portfolio_metrics().unwrap().is_empty());

    let position = metrics_position(100.0);
    let position_log = Arc::clone(&position.events);
    let benchmark = FixedBenchmark {
        result: Err(BenchmarkReturnSamplerError {
            message: "bench".to_owned(),
        }),
        events: Arc::new(Mutex::new(Vec::new())),
    };
    let benchmark_log = Arc::clone(&benchmark.events);
    let mut account =
        Account::with_portfolio_metrics(position, true, "day", Some(Arc::new(benchmark)));
    assert_eq!(
        account.update_portfolio_metrics(start, end),
        Err(AccountError::PortfolioMetrics {
            message: "benchmark return sampler error: bench".to_owned()
        })
    );
    assert_eq!(events(&position_log), ["total", "stock", "cash"]);
    assert_eq!(events(&benchmark_log), ["benchmark"]);
    assert!(account.portfolio_metrics().unwrap().is_empty());
}

#[test]
fn portfolio_metrics_preserve_ieee_special_values() {
    let mut position = metrics_position(100.0);
    position.total = Ok(f64::NAN);
    position.stock = Ok(f64::INFINITY);
    position.cash = Ok(f64::NEG_INFINITY);
    let mut account = Account::new(position, true);
    account.accumulated_info_mut().add_cost(f64::INFINITY);
    account
        .accumulated_info_mut()
        .add_turnover(f64::NEG_INFINITY);
    account
        .update_portfolio_metrics(
            datetime("2024-01-02 00:00:00"),
            datetime("2024-01-02 23:59:59"),
        )
        .unwrap();
    let row = account
        .portfolio_metrics()
        .unwrap()
        .latest_record()
        .unwrap();
    assert!(row.account_value.is_nan());
    assert!(row.return_rate.is_nan());
    assert!(row.stock_value.is_infinite() && row.stock_value.is_sign_positive());
    assert!(row.cash.is_infinite() && row.cash.is_sign_negative());
    assert!(row.turnover_rate.is_infinite() && row.turnover_rate.is_sign_negative());
    assert!(row.cost_rate.is_infinite() && row.cost_rate.is_sign_positive());
}

#[test]
fn historical_positions_refresh_weights_overwrite_stably_and_are_independent() {
    let first_time = datetime("2024-01-02 00:00:00");
    let second_time = datetime("2024-01-03 00:00:00");
    let mut account = Account::new(finite(100.0, 2.0, Some(5.0)), true);
    account.update_historical_positions(first_time).unwrap();
    assert_eq!(
        account.current_position().stored_account_value(),
        Some(110.0)
    );
    assert_same(
        account.current_position().stock_weight("A").unwrap(),
        10.0 / 110.0,
    );
    let history = account.historical_positions().read().unwrap();
    let first = history.get(&first_time).unwrap();
    assert_same(first.account_value(), 110.0);
    assert_same(first.position().stock_price("A").unwrap(), 5.0);
    assert_same(first.position().stock_weight("A").unwrap(), 10.0 / 110.0);
    drop(history);

    account
        .current_position_mut()
        .update_stock_price("A", 6.0)
        .unwrap();
    account.update_historical_positions(second_time).unwrap();
    assert_same(
        account
            .historical_positions()
            .read()
            .unwrap()
            .get(&first_time)
            .unwrap()
            .position()
            .stock_price("A")
            .unwrap(),
        5.0,
    );
    account
        .current_position_mut()
        .update_stock_price("A", 7.0)
        .unwrap();
    account.update_historical_positions(first_time).unwrap();
    assert_eq!(
        account
            .historical_positions()
            .read()
            .unwrap()
            .keys()
            .copied()
            .collect::<Vec<_>>(),
        [first_time, second_time]
    );
    assert_same(
        account
            .historical_positions()
            .read()
            .unwrap()
            .get(&first_time)
            .unwrap()
            .account_value(),
        114.0,
    );
}

#[test]
fn historical_position_failures_retain_only_reached_current_state() {
    let time = datetime("2024-01-02 00:00:00");
    for (stage, expected, stored) in [
        ("total", vec!["total"], None),
        ("account", vec!["total", "account:105.0"], None),
        (
            "weights",
            vec!["total", "account:105.0", "weights"],
            Some(105.0),
        ),
        (
            "snapshot",
            vec!["total", "account:105.0", "weights", "snapshot"],
            Some(105.0),
        ),
    ] {
        let mut position = metrics_position(100.0);
        match stage {
            "total" => position.total = Err(stage.to_owned()),
            "account" => position.set_error = Some(stage.to_owned()),
            "weights" => position.weights_error = Some(stage.to_owned()),
            "snapshot" => position.snapshot_error = Some(stage.to_owned()),
            _ => unreachable!(),
        }
        let log = Arc::clone(&position.events);
        let mut account = Account::new(position, true);
        assert!(matches!(
            account.update_historical_positions(time),
            Err(AccountError::Position(_))
        ));
        assert_eq!(events(&log), expected);
        assert_eq!(account.current_position().stored_account_value(), stored);
        assert!(account.historical_positions().read().unwrap().is_empty());
    }

    let mut account = Account::new(finite(0.0, 0.0, Some(5.0)), true);
    assert!(matches!(
        account.update_historical_positions(time),
        Err(AccountError::Position(_))
    ));
    assert_eq!(account.current_position().stored_account_value(), Some(0.0));
    assert!(account.historical_positions().read().unwrap().is_empty());

    let mut infinite = Account::new(InfinitePosition, true);
    assert!(matches!(
        infinite.update_historical_positions(time),
        Err(AccountError::Position(_))
    ));
    assert!(infinite.historical_positions().read().unwrap().is_empty());
}

fn live_python_history_snapshot() -> Value {
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/backtest/account.py");
    let script = r#"
import ast,copy,json,sys
p=sys.argv[1];t=ast.parse(open(p,encoding='utf-8').read());c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='Account');f=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='update_hist_positions');f.returns=None
for a in f.args.args:a.annotation=None
K=ast.fix_missing_locations(ast.ClassDef('Account',[],[],[f],[]));ns={'copy':copy};exec(compile(ast.Module([K],[]),p,'exec'),ns);A=ns['Account']
class M:
 def __init__(self,o):self.o=o;self.d={}
 def __setitem__(self,k,v):self.o.events.append('account:'+repr(v));exec('raise RuntimeError("account")') if self.o.se else None;self.d[k]=v
class P:
 def __init__(self,stage=None):self.events=[];self.te=stage=='total';self.se=stage=='account';self.we=stage=='weights';self.de=stage=='snapshot';self.position=M(self)
 def calculate_value(self):self.events.append('total');exec('raise RuntimeError("total")') if self.te else None;return 105.
 def update_weight_all(self):self.events.append('weights');exec('raise RuntimeError("weights")') if self.we else None
 def __deepcopy__(self,memo):self.events.append('snapshot');exec('raise RuntimeError("snapshot")') if self.de else None;q=P();q.position.d=copy.deepcopy(self.position.d);return q
def run(stage):
 a=A();a.current_position=P(stage);a.hist_positions={}
 try:a.update_hist_positions('t');e=None
 except Exception as x:e=type(x).__name__+':'+str(x)
 return [e,a.current_position.events,a.current_position.position.d.get('now_account_value'),[[k,v.position.d.get('now_account_value')] for k,v in a.hist_positions.items()]]
print(json.dumps([run(None),run('total'),run('account'),run('weights'),run('snapshot')],separators=(',',':')))
"#;
    let output = Command::new("python")
        .arg("-c")
        .arg(script)
        .arg(source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

fn rust_history_row(stage: Option<&str>) -> Value {
    let mut position = metrics_position(100.0);
    match stage {
        Some("total") => position.total = Err("total".to_owned()),
        Some("account") => position.set_error = Some("account".to_owned()),
        Some("weights") => position.weights_error = Some("weights".to_owned()),
        Some("snapshot") => position.snapshot_error = Some("snapshot".to_owned()),
        None => {}
        _ => unreachable!(),
    }
    let log = Arc::clone(&position.events);
    let mut account = Account::new(position, true);
    let error = match account.update_historical_positions(datetime("2024-01-02 00:00:00")) {
        Ok(()) => Value::Null,
        Err(AccountError::Position(error)) => json!(format!("RuntimeError:{}", error.message)),
        Err(_) => unreachable!(),
    };
    json!([
        error,
        events(&log),
        account.current_position().stored_account_value(),
        account
            .historical_positions()
            .read()
            .unwrap()
            .values()
            .map(|snapshot| json!(["t", snapshot.account_value()]))
            .collect::<Vec<_>>()
    ])
}

#[test]
fn account_history_update_matches_live_python_source() {
    assert_eq!(
        Value::Array(vec![
            rust_history_row(None),
            rust_history_row(Some("total")),
            rust_history_row(Some("account")),
            rust_history_row(Some("weights")),
            rust_history_row(Some("snapshot")),
        ]),
        live_python_history_snapshot()
    );
}

#[derive(Clone)]
struct ProbeIndicator {
    events: Arc<Mutex<Vec<String>>>,
    fail: Option<String>,
    trade: domain_core::SharedTradeIndicator,
    order: domain_core::SharedOrderIndicator<NumpyOrderIndicator>,
    history: IndexMap<NaiveDateTime, domain_core::SharedTradeIndicator>,
}

#[test]
fn unsupported_live_indicator_resets_then_rejects_without_reading_decision_or_recording() {
    use domain_core::decision_update::{
        LiveDecisionHandle, SharedDecisionUpdateStrategy, SharedLiveDecision,
    };
    struct Origin;
    impl SharedDecisionUpdateStrategy<()> for Origin {
        fn update_trade_decision(
            &self,
            _: &SharedLiveDecision<Self, ()>,
            _: &dyn domain_core::DecisionUpdateCalendar,
        ) -> Result<Option<SharedLiveDecision<Self, ()>>, domain_core::DecisionUpdateStrategyError>
        {
            panic!("unsupported plugin must not update decision")
        }
    }
    let state = domain_core::decision_construction::SharedOrderDecisionConstruction::<_, ()>::new(
        Arc::new(Origin),
    );
    // Deliberately uninitialized: default rejection must precede any decision access.
    let outer: LiveDecisionHandle = Arc::new(std::sync::RwLock::new(state));
    let at = datetime("2024-01-02 09:30:00");
    let provider = exchange(&[], Vec::new());
    let mut account = Account::new(InfinitePosition, false);
    let indicator = ProbeIndicator::new(None);
    let events = indicator.events.clone();
    account.replace_indicator(Box::new(indicator));
    let failure = account
        .update_indicator(AccountIndicatorUpdate {
            trade_start_time: at,
            mode: AccountIndicatorMode::LiveNested(
                domain_core::account::LiveNestedAccountIndicatorUpdate {
                    inner: &[],
                    outer_decision: &outer,
                    steps: &[],
                    provider: &provider,
                    config: OrderIndicatorAggregationConfig::default(),
                },
            ),
            calculation: IndicatorConfig::default(),
            show_indicator: true,
        })
        .unwrap_err();
    assert_eq!(
        failure.to_string(),
        "account indicator plugin error: indicator backend does not support live nested decisions"
    );
    assert_eq!(*events.lock().unwrap(), ["reset"]);
    assert!(
        account
            .indicator()
            .read()
            .unwrap()
            .recorded_trade_indicator(at)
            .is_none()
    );
}

impl ProbeIndicator {
    fn new(fail: Option<&str>) -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
            fail: fail.map(str::to_owned),
            trade: Arc::default(),
            order: Arc::default(),
            history: IndexMap::new(),
        }
    }

    fn stage(&self, name: &str) -> Result<(), AccountIndicatorError> {
        self.events.lock().unwrap().push(name.to_owned());
        if self.fail.as_deref() == Some(name) {
            return Err(AccountIndicatorError {
                message: name.to_owned(),
            });
        }
        Ok(())
    }
}

impl AccountIndicator for ProbeIndicator {
    fn reset(&mut self) -> Result<(), AccountIndicatorError> {
        self.stage("reset")?;
        self.trade = Arc::default();
        self.order = Arc::default();
        Ok(())
    }

    fn update_atomic(
        &mut self,
        _executions: &[OrderExecution<'_>],
    ) -> Result<(), AccountIndicatorError> {
        self.stage("atomic")
    }

    fn update_nested(
        &mut self,
        _update: NestedAccountIndicatorUpdate<'_>,
    ) -> Result<(), AccountIndicatorError> {
        self.stage("nested")
    }

    fn calculate(&mut self, _config: IndicatorConfig) -> Result<(), AccountIndicatorError> {
        self.stage("calculate")?;
        self.trade.write().unwrap().extend([
            ("ffr".to_owned(), 0.5),
            ("pa".to_owned(), 0.25),
            ("pos".to_owned(), 1.0),
        ]);
        if self.fail.as_deref() == Some("row_poison") {
            let row = self.trade.clone();
            assert!(
                std::thread::spawn(move || {
                    let _guard = row.write().unwrap();
                    panic!("poison calculated trade row");
                })
                .join()
                .is_err()
            );
        }
        Ok(())
    }

    fn record(&mut self, trade_start_time: NaiveDateTime) -> Result<(), AccountIndicatorError> {
        self.stage("record")?;
        self.history.insert(trade_start_time, self.trade.clone());
        Ok(())
    }

    fn trade_indicator(&self) -> &domain_core::SharedTradeIndicator {
        &self.trade
    }

    fn order_indicator_snapshot(&self) -> Result<NumpyOrderIndicator, AccountIndicatorError> {
        self.stage("snapshot")?;
        Ok(NumpyOrderIndicator::default())
    }

    fn order_indicator_handle(
        &self,
    ) -> Result<domain_core::SharedOrderIndicator<NumpyOrderIndicator>, AccountIndicatorError> {
        self.stage("handle")?;
        Ok(self.order.clone())
    }

    fn recorded_trade_indicator(
        &self,
        time: NaiveDateTime,
    ) -> Option<&domain_core::SharedTradeIndicator> {
        self.history.get(&time)
    }

    fn trade_indicator_report(
        &self,
    ) -> Result<domain_core::TradeIndicatorReport, AccountIndicatorError> {
        self.stage("report")?;
        Ok(domain_core::TradeIndicatorReport::from_shared_history(&self.history).unwrap())
    }
}

struct ProbeIndicatorOutput {
    events: Arc<Mutex<Vec<String>>>,
    fail: bool,
}

#[test]
fn poisoned_calculated_row_stops_before_output_and_record_and_releases_engine_lock() {
    let at = datetime("2024-01-02 09:30:00");
    let probe = ProbeIndicator::new(Some("row_poison"));
    let log = probe.events.clone();
    let output = Arc::new(Mutex::new(Vec::new()));
    let mut account = Account::new(InfinitePosition, false);
    account.replace_indicator(Box::new(probe));
    account.replace_indicator_output(Box::new(ProbeIndicatorOutput {
        events: output.clone(),
        fail: false,
    }));
    let error = account
        .update_indicator(AccountIndicatorUpdate {
            trade_start_time: at,
            mode: AccountIndicatorMode::Atomic(&[]),
            calculation: IndicatorConfig::default(),
            show_indicator: true,
        })
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "account indicator plugin error: trade indicator row lock poisoned"
    );
    assert_eq!(events(&log), ["reset", "atomic", "calculate"]);
    assert!(output.lock().unwrap().is_empty());
    let indicator = account.indicator().try_write().unwrap();
    assert!(indicator.recorded_trade_indicator(at).is_none());
    assert!(indicator.trade_indicator().read().is_err());
}

impl AccountIndicatorOutput for ProbeIndicatorOutput {
    fn write(
        &self,
        frequency: &str,
        _trade_start_time: NaiveDateTime,
        _values: &IndexMap<String, f64>,
    ) -> Result<(), AccountIndicatorOutputError> {
        self.events
            .lock()
            .unwrap()
            .push(format!("output:{frequency}"));
        if self.fail {
            return Err(AccountIndicatorOutputError {
                message: "output".to_owned(),
            });
        }
        Ok(())
    }
}

fn indicator_order() -> Order {
    let mut order = Order::new("A", 10.0, OrderDir::Buy, None, None);
    order.set_deal_amount(4.0);
    order
}

#[test]
fn built_in_account_indicator_runs_atomic_nested_display_and_records() {
    let time = datetime("2024-01-02 09:30:00");
    let order = indicator_order();
    let execution = OrderExecution {
        order: &order,
        trade_value: 40.0,
        trade_cost: 1.0,
        trade_price: 10.0,
    };
    let mut atomic = Account::with_frequency(finite(100.0, 0.0, Some(5.0)), false, "1min");
    atomic
        .update_indicator(AccountIndicatorUpdate {
            trade_start_time: time,
            mode: AccountIndicatorMode::Atomic(std::slice::from_ref(&execution)),
            calculation: IndicatorConfig::default(),
            show_indicator: true,
        })
        .unwrap();
    let indicator_guard = atomic.indicator().read().unwrap();
    let current = indicator_guard.trade_indicator().read().unwrap();
    assert_same(current["ffr"], 0.4);
    assert_same(current["pa"], 0.0);
    assert_same(current["pos"], 0.0);
    assert_same(current["deal_amount"], 4.0);
    assert_same(current["value"], 40.0);
    assert_same(current["count"], 1.0);
    assert!(Arc::ptr_eq(
        indicator_guard.recorded_trade_indicator(time).unwrap(),
        indicator_guard.trade_indicator()
    ));

    let mut nested = Account::new(finite(100.0, 0.0, Some(5.0)), false);
    let decision = OrderTradeDecision::from_orders(Vec::new(), time, time, None);
    let provider = exchange(&[], Vec::new());
    let inner: Vec<domain_core::SharedOrderIndicator<NumpyOrderIndicator>> = Vec::new();
    nested
        .update_indicator(AccountIndicatorUpdate {
            trade_start_time: time,
            mode: AccountIndicatorMode::Nested(NestedAccountIndicatorUpdate {
                inner: &inner,
                outer_decision: &decision,
                steps: &[],
                provider: &provider,
                config: OrderIndicatorAggregationConfig::default(),
            }),
            calculation: IndicatorConfig {
                fulfill_rate: IndicatorWeightMethod::AmountWeighted,
                price_advantage: IndicatorWeightMethod::ValueWeighted,
            },
            show_indicator: false,
        })
        .unwrap();
    assert!(
        nested
            .indicator()
            .read()
            .unwrap()
            .recorded_trade_indicator(time)
            .is_some()
    );

    let mut invalid = Indicator::new();
    assert!(AccountIndicator::calculate(&mut invalid, IndicatorConfig::default()).is_err());
    let invalid_inner = vec![Arc::new(std::sync::RwLock::new(
        NumpyOrderIndicator::default(),
    ))];
    assert!(
        AccountIndicator::update_nested(
            &mut invalid,
            NestedAccountIndicatorUpdate {
                inner: &invalid_inner,
                outer_decision: &decision,
                steps: &[],
                provider: &provider,
                config: OrderIndicatorAggregationConfig::default(),
            }
        )
        .is_err()
    );
}

#[test]
fn account_indicator_plugins_preserve_every_failure_boundary() {
    let time = datetime("2024-01-02 09:30:00");
    let order = indicator_order();
    let execution = OrderExecution {
        order: &order,
        trade_value: 40.0,
        trade_cost: 1.0,
        trade_price: 10.0,
    };
    for (failure, expected) in [
        ("reset", vec!["reset"]),
        ("atomic", vec!["reset", "atomic"]),
        ("calculate", vec!["reset", "atomic", "calculate"]),
        ("record", vec!["reset", "atomic", "calculate", "record"]),
    ] {
        let probe = ProbeIndicator::new(Some(failure));
        let log = Arc::clone(&probe.events);
        let mut account = Account::new(finite(100.0, 0.0, Some(5.0)), false);
        let previous = account.replace_indicator(Box::new(probe));
        let previous = previous.read().unwrap();
        assert!(previous.trade_indicator().read().unwrap().is_empty());
        assert!(matches!(
            account.update_indicator(AccountIndicatorUpdate {
                trade_start_time: time,
                mode: AccountIndicatorMode::Atomic(std::slice::from_ref(&execution)),
                calculation: IndicatorConfig::default(),
                show_indicator: false,
            }),
            Err(AccountError::Indicator(_))
        ));
        assert_eq!(events(&log), expected);
        assert!(
            account
                .indicator()
                .read()
                .unwrap()
                .recorded_trade_indicator(time)
                .is_none()
        );
    }

    let probe = ProbeIndicator::new(None);
    let indicator_log = Arc::clone(&probe.events);
    let output_log = Arc::new(Mutex::new(Vec::new()));
    let mut account = Account::with_frequency(finite(100.0, 0.0, Some(5.0)), false, "week");
    account.replace_indicator(Box::new(probe));
    let old_output = account.replace_indicator_output(Box::new(ProbeIndicatorOutput {
        events: Arc::clone(&output_log),
        fail: true,
    }));
    let values = IndexMap::from([
        ("ffr".to_owned(), 1.0),
        ("pa".to_owned(), 2.0),
        ("pos".to_owned(), 3.0),
    ]);
    old_output.write("day", time, &values).unwrap();
    assert!(matches!(
        account.update_indicator(AccountIndicatorUpdate {
            trade_start_time: time,
            mode: AccountIndicatorMode::Atomic(std::slice::from_ref(&execution)),
            calculation: IndicatorConfig::default(),
            show_indicator: true,
        }),
        Err(AccountError::IndicatorOutput(_))
    ));
    assert_eq!(events(&indicator_log), ["reset", "atomic", "calculate"]);
    assert_eq!(events(&output_log), ["output:week"]);
    assert!(
        account
            .indicator()
            .read()
            .unwrap()
            .recorded_trade_indicator(time)
            .is_none()
    );

    let probe = ProbeIndicator::new(Some("nested"));
    let log = Arc::clone(&probe.events);
    account.replace_indicator(Box::new(probe));
    let decision = OrderTradeDecision::from_orders(Vec::new(), time, time, None);
    let provider = exchange(&[], Vec::new());
    let mut inner = Vec::new();
    assert!(matches!(
        account.update_indicator(AccountIndicatorUpdate {
            trade_start_time: time,
            mode: AccountIndicatorMode::Nested(NestedAccountIndicatorUpdate {
                inner: &mut inner,
                outer_decision: &decision,
                steps: &[],
                provider: &provider,
                config: OrderIndicatorAggregationConfig::default(),
            }),
            calculation: IndicatorConfig::default(),
            show_indicator: false,
        }),
        Err(AccountError::Indicator(_))
    ));
    assert_eq!(events(&log), ["reset", "nested"]);
}

#[test]
fn account_indicator_output_formats_python_special_values_and_missing_metrics() {
    let time = datetime("2024-01-02 09:30:00");
    let values = IndexMap::from([
        ("ffr".to_owned(), f64::NAN),
        ("pa".to_owned(), f64::INFINITY),
        ("pos".to_owned(), f64::NEG_INFINITY),
    ]);
    assert_eq!(
        format_account_indicator_output("1min", time, &values).unwrap(),
        "[Indicator(1min) 2024-01-02 09:30:00]: FFR: nan, PA: inf, POS: -inf"
    );
    assert!(format_account_indicator_output("day", time, &IndexMap::new()).is_err());
    assert!(
        format_account_indicator_output("day", time, &IndexMap::from([("ffr".to_owned(), 1.0)]))
            .is_err()
    );
    assert!(
        format_account_indicator_output(
            "day",
            time,
            &IndexMap::from([("ffr".to_owned(), 1.0), ("pa".to_owned(), 2.0)])
        )
        .is_err()
    );
    assert_eq!(
        format_account_indicator_output(
            "day",
            time,
            &IndexMap::from([
                ("ffr".to_owned(), 1.0),
                ("pa".to_owned(), 2.5),
                ("pos".to_owned(), 3.0),
            ])
        )
        .unwrap(),
        "[Indicator(day) 2024-01-02 09:30:00]: FFR: 1.0, PA: 2.5, POS: 3.0"
    );
    StdoutAccountIndicatorOutput
        .write(
            "day",
            time,
            &IndexMap::from([
                ("ffr".to_owned(), 1.0),
                ("pa".to_owned(), 2.0),
                ("pos".to_owned(), 3.0),
            ]),
        )
        .unwrap();
    assert!(
        StdoutAccountIndicatorOutput
            .write("day", time, &IndexMap::new())
            .is_err()
    );
}

fn live_python_indicator_update_snapshot() -> Value {
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/backtest/account.py");
    let script = r#"
import ast,json,sys
p=sys.argv[1];t=ast.parse(open(p,encoding='utf-8').read());c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='Account');f=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='update_indicator');f.returns=None
for a in f.args.args:a.annotation=None
K=ast.fix_missing_locations(ast.ClassDef('Account',[],[],[f],[]));ns={};exec(compile(ast.Module([K],[]),p,'exec'),ns);A=ns['Account']
class I:
 def __init__(self,fail=None):self.fail=fail;self.events=[]
 def s(self,n):self.events.append(n);exec('raise RuntimeError("'+n+'")') if self.fail==n else None
 def reset(self):self.s('reset')
 def update_order_indicators(self,x):self.s('atomic')
 def agg_order_indicators(self,*a,**k):self.s('nested')
 def cal_trade_indicators(self,*a,**k):self.s('calculate')
 def record(self,t):self.s('record')
def run(nested,fail=None):
 a=A();a.indicator=I(fail);a.freq='day'
 try:a.update_indicator(1,'exchange',not nested,'decision',['execution'],['inner'],['step'],{});e=None
 except Exception as x:e=type(x).__name__+':'+str(x)
 return [e,a.indicator.events]
print(json.dumps([run(False),run(True),run(False,'reset'),run(False,'atomic'),run(True,'nested'),run(False,'calculate'),run(False,'record')],separators=(',',':')))
"#;
    let output = Command::new("python")
        .arg("-c")
        .arg(script)
        .arg(source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

fn rust_indicator_update_row(nested: bool, failure: Option<&str>) -> Value {
    let time = datetime("2024-01-02 09:30:00");
    let probe = ProbeIndicator::new(failure);
    let log = Arc::clone(&probe.events);
    let mut account = Account::new(finite(100.0, 0.0, Some(5.0)), false);
    account.replace_indicator(Box::new(probe));
    let result = if nested {
        let decision = OrderTradeDecision::from_orders(Vec::new(), time, time, None);
        let provider = exchange(&[], Vec::new());
        let mut inner = Vec::new();
        account.update_indicator(AccountIndicatorUpdate {
            trade_start_time: time,
            mode: AccountIndicatorMode::Nested(NestedAccountIndicatorUpdate {
                inner: &mut inner,
                outer_decision: &decision,
                steps: &[],
                provider: &provider,
                config: OrderIndicatorAggregationConfig::default(),
            }),
            calculation: IndicatorConfig::default(),
            show_indicator: false,
        })
    } else {
        account.update_indicator(AccountIndicatorUpdate {
            trade_start_time: time,
            mode: AccountIndicatorMode::Atomic(&[]),
            calculation: IndicatorConfig::default(),
            show_indicator: false,
        })
    };
    let error = match result {
        Ok(()) => Value::Null,
        Err(AccountError::Indicator(error)) => json!(format!("RuntimeError:{}", error.message)),
        Err(_) => unreachable!(),
    };
    json!([error, events(&log)])
}

#[test]
fn account_indicator_update_matches_live_python_source() {
    assert_eq!(
        Value::Array(vec![
            rust_indicator_update_row(false, None),
            rust_indicator_update_row(true, None),
            rust_indicator_update_row(false, Some("reset")),
            rust_indicator_update_row(false, Some("atomic")),
            rust_indicator_update_row(true, Some("nested")),
            rust_indicator_update_row(false, Some("calculate")),
            rust_indicator_update_row(false, Some("record")),
        ]),
        live_python_indicator_update_snapshot()
    );
}

fn live_python_portfolio_metric_snapshot() -> Value {
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/backtest/account.py");
    let script = r"
import ast,json,sys
p=sys.argv[1];t=ast.parse(open(p,encoding='utf-8').read());c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='Account');f=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='update_portfolio_metrics');f.returns=None
for a in f.args.args:a.annotation=None
K=ast.fix_missing_locations(ast.ClassDef('Account',[],[],[f],[]));ns={};exec(compile(ast.Module([K],[]),p,'exec'),ns);A=ns['Account']
class PM:
 def __init__(self):self.rows=[]
 def is_empty(self):return not self.rows
 def get_latest_account_value(self):return self.rows[-1]['account_value']
 def get_latest_total_cost(self):return self.rows[-1]['total_cost']
 def get_latest_total_turnover(self):return self.rows[-1]['total_turnover']
 def update_portfolio_metrics_record(self,**kw):self.rows.append(kw)
class P:
 def __init__(self):self.position={'cash':100.};self.price=5.
 def calculate_stock_value(self):return 2.*self.price
 def calculate_value(self):return self.position['cash']+self.calculate_stock_value()
class I:
 def __init__(self,c,t):self.get_cost=c;self.get_turnover=t
a=A();a.init_cash=100.;a.current_position=P();a.portfolio_metrics=PM();a.accum_info=I(2.,20.);a.update_portfolio_metrics('s1','e1');a.current_position.price=6.;a.accum_info=I(3.,32.);a.update_portfolio_metrics('s2','e2')
keys=['account_value','cash','return_rate','total_turnover','turnover_rate','total_cost','cost_rate','stock_value'];print(json.dumps([[r[k] for k in keys] for r in a.portfolio_metrics.rows],separators=(',',':')))
";
    let output = Command::new("python")
        .arg("-c")
        .arg(script)
        .arg(source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

fn rust_portfolio_metric_snapshot() -> Value {
    let mut account = Account::new(finite(100.0, 2.0, Some(5.0)), true);
    account.accumulated_info_mut().add_cost(2.0);
    account.accumulated_info_mut().add_turnover(20.0);
    account
        .update_portfolio_metrics(
            datetime("2024-01-02 00:00:00"),
            datetime("2024-01-02 23:59:59"),
        )
        .unwrap();
    account
        .current_position_mut()
        .update_stock_price("A", 6.0)
        .unwrap();
    account.accumulated_info_mut().add_cost(1.0);
    account.accumulated_info_mut().add_turnover(12.0);
    account
        .update_portfolio_metrics(
            datetime("2024-01-03 00:00:00"),
            datetime("2024-01-03 23:59:59"),
        )
        .unwrap();
    Value::Array(
        account
            .portfolio_metrics()
            .unwrap()
            .records()
            .values()
            .map(|row| {
                json!([
                    row.account_value,
                    row.cash,
                    row.return_rate,
                    row.total_turnover,
                    row.turnover_rate,
                    row.total_cost,
                    row.cost_rate,
                    row.stock_value
                ])
            })
            .collect(),
    )
}

#[test]
fn account_portfolio_metrics_match_live_python_source() {
    assert_eq!(
        rust_portfolio_metric_snapshot(),
        live_python_portfolio_metric_snapshot()
    );
}

fn live_python_bar_snapshot() -> Value {
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/backtest/account.py");
    let script = r#"
import ast,json,sys
p=sys.argv[1];t=ast.parse(open(p,encoding='utf-8').read());c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='Account');f=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='update_current_position');f.returns=None
for a in f.args.args:a.annotation=None
K=ast.fix_missing_locations(ast.ClassDef('Account',[],[],[f],[]));ns={'cast':lambda _,v:v};exec(compile(ast.Module([K],[]),p,'exec'),ns);A=ns['Account']
class P:
 def __init__(self,skip=False,stocks=None,se=None,me=None,ce=None):self.s=skip;self.stocks=['A','B'] if stocks is None else stocks;self.se=se;self.me=me;self.ce=ce;self.events=[]
 def skip_update(self):self.events.append('skip');return self.s
 def get_stock_list(self):self.events.append('stocks');exec('raise RuntimeError("stocks")') if self.se else None;return self.stocks
 def update_stock_price(self,stock_id,price):self.events.append('mark:'+stock_id+':'+str(price));exec('raise RuntimeError("mark")') if self.me else None
 def add_count_all(self,bar):self.events.append('count:'+bar);exec('raise RuntimeError("count")') if self.ce else None
class X:
 def __init__(self,suspended,closes=None):self.s=list(suspended);self.c=list(closes or []);self.events=[]
 def check_stock_suspended(self,code,start,end):self.events.append('suspended:'+code);v=self.s.pop(0);exec('raise RuntimeError("suspended")') if v=='E' else None;return v
 def get_close(self,code,start,end):self.events.append('close:'+code);v=self.c.pop(0);exec('raise RuntimeError("close")') if v=='E' else None;return v
def run(pos,market):
 a=A();a.current_position=pos;a.freq='1min'
 try:a.update_current_position(1,2,market);e=None
 except Exception as x:e=type(x).__name__+':'+str(x)
 return [e,pos.events,market.events]
r=[run(P(),X([False,True],[7.])),run(P(skip=True),X([])),run(P(se=True),X([])),run(P(stocks=['A']),X(['E'])),run(P(stocks=['A']),X([False],['E'])),run(P(stocks=['A'],me=True),X([False],[3.])),run(P(stocks=[],ce=True),X([]))];print(json.dumps(r,separators=(',',':')))
"#;
    let output = Command::new("python")
        .arg("-c")
        .arg(script)
        .arg(source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

fn rust_bar_row(
    position: BarPosition,
    market: &ScriptMarket,
    start: NaiveDateTime,
    end: NaiveDateTime,
) -> Value {
    let position_events = Arc::clone(&position.events);
    let market_events = Arc::clone(&market.events);
    let mut account = Account::with_frequency(position, true, "1min");
    let error = match account.update_current_position(start, end, market) {
        Ok(()) => Value::Null,
        Err(AccountError::Position(error)) => json!(format!("RuntimeError:{}", error.message)),
        Err(AccountError::Market(error)) => json!(format!("RuntimeError:{}", error.message)),
        Err(
            AccountError::HistoryPoisoned
            | AccountError::ZeroTradePrice
            | AccountError::MissingAtomicTradeInfo
            | AccountError::MissingInnerOrderIndicators
            | AccountError::PortfolioMetricsDisabled
            | AccountError::PortfolioReportDisabled
            | AccountError::PortfolioMetrics { .. }
            | AccountError::ZeroPreviousAccountValue
            | AccountError::Indicator { .. }
            | AccountError::IndicatorOutput { .. },
        ) => unreachable!(),
    };
    let market_events = events(&market_events)
        .into_iter()
        .map(|event| {
            let mut parts = event.splitn(3, ':');
            format!("{}:{}", parts.next().unwrap(), parts.next().unwrap())
        })
        .collect::<Vec<_>>();
    json!([error, events(&position_events), market_events])
}

fn rust_bar_snapshot() -> Value {
    let start = datetime("2024-01-02 09:30:00");
    let end = datetime("2024-01-02 09:31:00");
    let success = BarPosition::new(Ok(vec!["A", "B"]));
    let mut skipped = BarPosition::new(Ok(vec![]));
    skipped.skip = true;
    let stocks_error = BarPosition::new(Err("stocks"));
    let suspended_error = BarPosition::new(Ok(vec!["A"]));
    let close_error = BarPosition::new(Ok(vec!["A"]));
    let mut mark_error = BarPosition::new(Ok(vec!["A"]));
    mark_error.mark_error = Some("mark".to_owned());
    let mut count_error = BarPosition::new(Ok(vec![]));
    count_error.count_error = Some("count".to_owned());
    Value::Array(vec![
        rust_bar_row(
            success,
            &ScriptMarket::new(vec![Ok(false), Ok(true)], vec![Ok(7.0)]),
            start,
            end,
        ),
        rust_bar_row(skipped, &ScriptMarket::new(vec![], vec![]), start, end),
        rust_bar_row(stocks_error, &ScriptMarket::new(vec![], vec![]), start, end),
        rust_bar_row(
            suspended_error,
            &ScriptMarket::new(vec![Err("suspended")], vec![]),
            start,
            end,
        ),
        rust_bar_row(
            close_error,
            &ScriptMarket::new(vec![Ok(false)], vec![Err("close")]),
            start,
            end,
        ),
        rust_bar_row(
            mark_error,
            &ScriptMarket::new(vec![Ok(false)], vec![Ok(3.0)]),
            start,
            end,
        ),
        rust_bar_row(count_error, &ScriptMarket::new(vec![], vec![]), start, end),
    ])
}

#[test]
fn account_bar_update_matches_live_python_source() {
    assert_eq!(rust_bar_snapshot(), live_python_bar_snapshot());
}

#[test]
fn plugin_sequence_and_failures_match_account_mutation_boundaries() {
    let probe = ProbePosition::new(false, Ok(5.0), None);
    let log = Arc::clone(&probe.events);
    let mut sell = Account::new(probe, true);
    sell.update_order(&order(OrderDir::Sell), 10.0, 1.0, 10.0)
        .unwrap();
    assert_eq!(events(&log), ["skip", "skip", "price:A", "update:0"]);

    let probe = ProbePosition::new(false, Ok(5.0), None);
    let log = Arc::clone(&probe.events);
    let mut buy = Account::from_boxed(Box::new(probe), true);
    buy.update_order(&order(OrderDir::Buy), 20.0, 2.0, 10.0)
        .unwrap();
    assert_eq!(events(&log), ["skip", "update:1", "skip", "price:A"]);

    let probe = ProbePosition::new(false, Ok(5.0), None);
    let log = Arc::clone(&probe.events);
    let mut disabled = Account::new(probe, false);
    disabled
        .update_order(&order(OrderDir::Sell), 10.0, 1.0, 10.0)
        .unwrap();
    assert_eq!(events(&log), ["skip", "update:0"]);

    let probe = ProbePosition::new(true, Ok(5.0), None);
    let log = Arc::clone(&probe.events);
    let mut skipped = Account::new(probe, true);
    skipped
        .update_order(&order(OrderDir::Buy), 20.0, 2.0, 10.0)
        .unwrap();
    assert_eq!(events(&log), ["skip"]);

    let probe = ProbePosition::new(false, Err("price"), None);
    let log = Arc::clone(&probe.events);
    let mut price_error = Account::new(probe, true);
    assert!(matches!(
        price_error.update_order(&order(OrderDir::Sell), 10.0, 1.0, 10.0),
        Err(AccountError::Position(_))
    ));
    assert_eq!(events(&log), ["skip", "skip", "price:A"]);
    assert_same(price_error.accumulated_info().turnover(), 10.0);
    assert_same(price_error.accumulated_info().cost(), 1.0);

    let probe = ProbePosition::new(false, Ok(5.0), Some("update"));
    let mut sell_update_error = Account::new(probe, true);
    assert!(matches!(
        sell_update_error.update_order(&order(OrderDir::Sell), 10.0, 1.0, 10.0),
        Err(AccountError::Position(_))
    ));
    assert_same(sell_update_error.accumulated_info().return_value(), 5.0);

    let probe = ProbePosition::new(false, Ok(5.0), Some("update"));
    let mut buy_update_error = Account::new(probe, true);
    assert!(matches!(
        buy_update_error.update_order(&order(OrderDir::Buy), 20.0, 2.0, 10.0),
        Err(AccountError::Position(_))
    ));
    assert_eq!(*buy_update_error.accumulated_info(), AccumulatedInfo::new());

    let probe = ProbePosition::new(false, Ok(5.0), None);
    let mut buy_zero = Account::new(probe, true);
    assert_eq!(
        buy_zero.update_order(&order(OrderDir::Buy), 20.0, 2.0, 0.0),
        Err(AccountError::ZeroTradePrice)
    );
    assert_same(buy_zero.accumulated_info().turnover(), 20.0);
    assert_same(buy_zero.accumulated_info().cost(), 2.0);

    let mut view_probe = ProbePosition::new(false, Ok(5.0), None);
    view_probe.view_error = Some("view".to_owned());
    let mut view_error = Account::new(view_probe, true);
    assert!(matches!(
        view_error.execution_position(),
        Err(AccountError::Position(_))
    ));
    let target: &mut dyn ExecutionTarget = &mut view_error;
    let Err(error) = target.position() else {
        panic!("expected the execution-position adapter to fail");
    };
    assert_eq!(error.message, "account position error: view");

    assert!(!view_error.current_position_mut().skip_update());
    view_error.accumulated_info_mut().add_cost(3.0);
    assert_same(view_error.accumulated_info().cost(), 3.0);
}

fn special(value: f64) -> Value {
    if value.is_nan() {
        json!("NaN")
    } else if value == f64::INFINITY {
        json!("Infinity")
    } else if value == f64::NEG_INFINITY {
        json!("-Infinity")
    } else {
        json!(value)
    }
}

fn live_python_account_snapshot() -> Value {
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/backtest/account.py");
    let script = r#"
import ast,json,math,sys
class O:SELL=0;BUY=1
p=sys.argv[1];t=ast.parse(open(p,encoding='utf-8').read())
def extract(name,names):
 c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name==name);body=[]
 for n in c.body:
  if isinstance(n,ast.FunctionDef) and n.name in names:
   n.returns=None
   for a in n.args.args:a.annotation=None
   body.append(n)
 return ast.ClassDef(name,[],[],body,[])
m=ast.fix_missing_locations(ast.Module([extract('AccumulatedInfo',{'__init__','reset','add_return_value','add_cost','add_turnover','get_return','get_cost','get_turnover'}),extract('Account',{'is_port_metr_enabled','_update_state_from_order','update_order'})],[]));ns={'Order':O};exec(compile(m,p,'exec'),ns);Info,A=ns['AccumulatedInfo'],ns['Account']
class P:
 def __init__(self,skip=False,price=5.,pe=False,ue=False):self.skip=skip;self.price=price;self.pe=pe;self.ue=ue;self.events=[]
 def skip_update(self):self.events.append('skip');return self.skip
 def get_stock_price(self,s):self.events.append('price:'+s);exec('raise RuntimeError("price")') if self.pe else None;return self.price
 def update_order(self,o,v,c,p):self.events.append('update:'+str(o.direction));exec('raise RuntimeError("update")') if self.ue else None
def acc(pos,enabled=True):a=A();a.current_position=pos;a._port_metr_enabled=enabled;a.accum_info=Info();return a
def order(d):o=O();o.direction=d;o.stock_id='A';return o
def v(x):
 if isinstance(x,float):
  if math.isnan(x):return 'NaN'
  if x==float('inf'):return 'Infinity'
  if x==float('-inf'):return '-Infinity'
 return x
def run(d,tv,c,tp,**kw):
 a=acc(P(kw.get('skip',False),kw.get('price',5.),kw.get('pe',False),kw.get('ue',False)),kw.get('enabled',True))
 try:a.update_order(order(d),tv,c,tp);e=None
 except Exception as x:e=type(x).__name__+':'+str(x)
 return [e,{'events':a.current_position.events,'return':v(a.accum_info.get_return),'cost':v(a.accum_info.get_cost),'turnover':v(a.accum_info.get_turnover)}]
r=[run(O.SELL,10.,1.,10.),run(O.BUY,20.,2.,10.),run(O.SELL,10.,1.,10.,enabled=False),run(O.BUY,20.,2.,10.,skip=True),run(O.SELL,10.,1.,10.,pe=True),run(O.SELL,10.,1.,10.,ue=True),run(O.BUY,20.,2.,10.,ue=True),run(O.SELL,10.,1.,0.),run(O.BUY,20.,2.,0.),run(O.SELL,float('nan'),1.,10.),run(O.SELL,10.,1.,float('inf')),run(O.SELL,10.,1.,float('nan'))]
i=Info();i.add_return_value(float('nan'));i.add_cost(float('inf'));i.add_turnover(float('-inf'));r.append([v(i.get_return),v(i.get_cost),v(i.get_turnover)]);i.reset();r.append([v(i.get_return),v(i.get_cost),v(i.get_turnover)]);print(json.dumps(r,separators=(',',':')))
"#;
    let output = Command::new("python")
        .arg("-c")
        .arg(script)
        .arg(source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[allow(clippy::too_many_arguments)]
fn probe_row(
    direction: OrderDir,
    trade_value: f64,
    cost: f64,
    trade_price: f64,
    enabled: bool,
    skip: bool,
    price: Result<f64, &str>,
    update_error: Option<&str>,
) -> Value {
    let probe = ProbePosition::new(skip, price, update_error);
    let log = Arc::clone(&probe.events);
    let mut account = Account::new(probe, enabled);
    let result = account.update_order(&order(direction), trade_value, cost, trade_price);
    let error = match result {
        Ok(()) => Value::Null,
        Err(AccountError::ZeroTradePrice) => json!("ZeroDivisionError:division by zero"),
        Err(AccountError::Position(error)) => json!(format!("RuntimeError:{}", error.message)),
        Err(AccountError::Market(error)) => json!(format!("RuntimeError:{}", error.message)),
        Err(
            AccountError::HistoryPoisoned
            | AccountError::PortfolioMetricsDisabled
            | AccountError::PortfolioReportDisabled
            | AccountError::MissingAtomicTradeInfo
            | AccountError::MissingInnerOrderIndicators
            | AccountError::PortfolioMetrics { .. }
            | AccountError::ZeroPreviousAccountValue
            | AccountError::Indicator { .. }
            | AccountError::IndicatorOutput { .. },
        ) => unreachable!(),
    };
    json!([
        error,
        {
            "events": events(&log),
            "return": special(account.accumulated_info().return_value()),
            "cost": special(account.accumulated_info().cost()),
            "turnover": special(account.accumulated_info().turnover()),
        }
    ])
}

fn rust_account_snapshot() -> Value {
    let mut rows = vec![
        probe_row(OrderDir::Sell, 10.0, 1.0, 10.0, true, false, Ok(5.0), None),
        probe_row(OrderDir::Buy, 20.0, 2.0, 10.0, true, false, Ok(5.0), None),
        probe_row(OrderDir::Sell, 10.0, 1.0, 10.0, false, false, Ok(5.0), None),
        probe_row(OrderDir::Buy, 20.0, 2.0, 10.0, true, true, Ok(5.0), None),
        probe_row(
            OrderDir::Sell,
            10.0,
            1.0,
            10.0,
            true,
            false,
            Err("price"),
            None,
        ),
        probe_row(
            OrderDir::Sell,
            10.0,
            1.0,
            10.0,
            true,
            false,
            Ok(5.0),
            Some("update"),
        ),
        probe_row(
            OrderDir::Buy,
            20.0,
            2.0,
            10.0,
            true,
            false,
            Ok(5.0),
            Some("update"),
        ),
        probe_row(OrderDir::Sell, 10.0, 1.0, 0.0, true, false, Ok(5.0), None),
        probe_row(OrderDir::Buy, 20.0, 2.0, 0.0, true, false, Ok(5.0), None),
        probe_row(
            OrderDir::Sell,
            f64::NAN,
            1.0,
            10.0,
            true,
            false,
            Ok(5.0),
            None,
        ),
        probe_row(
            OrderDir::Sell,
            10.0,
            1.0,
            f64::INFINITY,
            true,
            false,
            Ok(5.0),
            None,
        ),
        probe_row(
            OrderDir::Sell,
            10.0,
            1.0,
            f64::NAN,
            true,
            false,
            Ok(5.0),
            None,
        ),
    ];
    let mut info = AccumulatedInfo::new();
    info.add_return_value(f64::NAN);
    info.add_cost(f64::INFINITY);
    info.add_turnover(f64::NEG_INFINITY);
    rows.push(json!([
        special(info.return_value()),
        special(info.cost()),
        special(info.turnover())
    ]));
    info.reset();
    rows.push(json!([
        special(info.return_value()),
        special(info.cost()),
        special(info.turnover())
    ]));
    Value::Array(rows)
}

#[test]
fn account_order_core_matches_live_python_source() {
    assert_eq!(rust_account_snapshot(), live_python_account_snapshot());
}

#[test]
fn account_bar_end_validates_before_mutation_and_accepts_empty_atomic_input() {
    let start = datetime("2024-01-02 09:30:00");
    let end = datetime("2024-01-02 09:30:59");
    let market = ScriptMarket::new(Vec::new(), Vec::new());

    for (mode, error) in [
        (
            AccountBarEndMode::Atomic(None),
            AccountError::MissingAtomicTradeInfo,
        ),
        (
            AccountBarEndMode::Nested(None),
            AccountError::MissingInnerOrderIndicators,
        ),
    ] {
        let position = metrics_position(100.0);
        let log = Arc::clone(&position.events);
        let probe = ProbeIndicator {
            events: Arc::clone(&log),
            fail: None,
            trade: Arc::default(),
            order: Arc::default(),
            history: IndexMap::new(),
        };
        let mut account = Account::new(position, true);
        account.replace_indicator(Box::new(probe));
        assert_eq!(
            account.update_bar_end(AccountBarEndUpdate {
                trade_start_time: start,
                trade_end_time: end,
                market: &market,
                mode,
                calculation: IndicatorConfig::default(),
                show_indicator: false,
            }),
            Err(error)
        );
        assert!(events(&log).is_empty());
        assert!(account.portfolio_metrics().unwrap().is_empty());
        assert!(account.historical_positions().read().unwrap().is_empty());
    }

    let position = metrics_position(100.0);
    let log = Arc::clone(&position.events);
    let probe = ProbeIndicator {
        events: Arc::clone(&log),
        fail: None,
        trade: Arc::default(),
        order: Arc::default(),
        history: IndexMap::new(),
    };
    let mut account = Account::new(position, true);
    account.replace_indicator(Box::new(probe));
    account.replace_indicator_output(Box::new(ProbeIndicatorOutput {
        events: Arc::clone(&log),
        fail: false,
    }));
    account
        .update_bar_end(AccountBarEndUpdate {
            trade_start_time: start,
            trade_end_time: end,
            market: &market,
            mode: AccountBarEndMode::Atomic(Some(&[])),
            calculation: IndicatorConfig::default(),
            show_indicator: true,
        })
        .unwrap();
    assert_eq!(
        events(&log),
        [
            "count:day",
            "total",
            "stock",
            "cash",
            "total",
            "account:105.0",
            "weights",
            "snapshot",
            "reset",
            "atomic",
            "calculate",
            "output:day",
            "record",
        ]
    );
    assert_eq!(account.portfolio_metrics().unwrap().records().len(), 1);
    assert_eq!(account.historical_positions().read().unwrap().len(), 1);
    assert!(
        account
            .indicator()
            .read()
            .unwrap()
            .recorded_trade_indicator(start)
            .is_some()
    );
}

#[test]
fn account_bar_end_preserves_nested_disabled_path() {
    let start = datetime("2024-01-02 09:30:00");
    let end = datetime("2024-01-02 09:30:59");
    let market = ScriptMarket::new(Vec::new(), Vec::new());
    let decision = OrderTradeDecision::from_orders(Vec::new(), start, end, None);
    let provider = exchange(&[], Vec::new());
    let mut inner = Vec::new();
    let position = metrics_position(100.0);
    let log = Arc::clone(&position.events);
    let mut account = Account::new(position, false);
    account.replace_indicator(Box::new(ProbeIndicator {
        events: Arc::clone(&log),
        fail: None,
        trade: Arc::default(),
        order: Arc::default(),
        history: IndexMap::new(),
    }));
    account
        .update_bar_end(AccountBarEndUpdate {
            trade_start_time: start,
            trade_end_time: end,
            market: &market,
            mode: AccountBarEndMode::Nested(Some(NestedAccountIndicatorUpdate {
                inner: &mut inner,
                outer_decision: &decision,
                steps: &[],
                provider: &provider,
                config: OrderIndicatorAggregationConfig::default(),
            })),
            calculation: IndicatorConfig::default(),
            show_indicator: false,
        })
        .unwrap();
    assert_eq!(
        events(&log),
        ["count:day", "reset", "nested", "calculate", "record"]
    );
    assert!(account.portfolio_metrics().is_none());
    assert!(account.historical_positions().read().unwrap().is_empty());
}

#[test]
fn account_bar_end_preserves_every_stage_failure_boundary() {
    let start = datetime("2024-01-02 09:30:00");
    let end = datetime("2024-01-02 09:30:59");
    let market = ScriptMarket::new(Vec::new(), Vec::new());
    for (failure, expected, metrics, history) in [
        ("current", vec!["count:day"], 0, 0),
        ("metrics", vec!["count:day", "total", "stock"], 0, 0),
        (
            "history",
            vec![
                "count:day",
                "total",
                "stock",
                "cash",
                "total",
                "account:105.0",
                "weights",
                "snapshot",
            ],
            1,
            0,
        ),
        (
            "indicator",
            vec![
                "count:day",
                "total",
                "stock",
                "cash",
                "total",
                "account:105.0",
                "weights",
                "snapshot",
                "reset",
            ],
            1,
            1,
        ),
    ] {
        let mut position = metrics_position(100.0);
        match failure {
            "current" => position.count_error = Some(failure.to_owned()),
            "metrics" => position.stock = Err(failure.to_owned()),
            "history" => position.snapshot_error = Some(failure.to_owned()),
            "indicator" => {}
            _ => unreachable!(),
        }
        let log = Arc::clone(&position.events);
        let mut account = Account::new(position, true);
        account.replace_indicator(Box::new(ProbeIndicator {
            events: Arc::clone(&log),
            fail: (failure == "indicator").then_some("reset".to_owned()),
            trade: Arc::default(),
            order: Arc::default(),
            history: IndexMap::new(),
        }));
        assert!(
            account
                .update_bar_end(AccountBarEndUpdate {
                    trade_start_time: start,
                    trade_end_time: end,
                    market: &market,
                    mode: AccountBarEndMode::Atomic(Some(&[])),
                    calculation: IndicatorConfig::default(),
                    show_indicator: false,
                })
                .is_err()
        );
        assert_eq!(events(&log), expected);
        assert_eq!(
            account.portfolio_metrics().unwrap().records().len(),
            metrics
        );
        assert_eq!(
            account.historical_positions().read().unwrap().len(),
            history
        );
        assert!(
            account
                .indicator()
                .read()
                .unwrap()
                .recorded_trade_indicator(start)
                .is_none()
        );
    }
}

fn live_python_bar_end_snapshot() -> Value {
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/backtest/account.py");
    let script = r"
import ast,json,sys
p=sys.argv[1];t=ast.parse(open(p,encoding='utf-8').read());c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='Account');f=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='update_bar_end');f.returns=None
for a in f.args.args:a.annotation=None
K=ast.fix_missing_locations(ast.ClassDef('Account',[],[],[f],[]));ns={};exec(compile(ast.Module([K],[]),p,'exec'),ns);B=ns['Account']
class A(B):
 def __init__(self,enabled,fail):self.enabled=enabled;self.fail=fail;self.events=[];self.metrics=0;self.history=0
 def update_current_position(self,s,e,x):
  self.events.append('count:day')
  if self.fail=='current':raise RuntimeError('current')
 def is_port_metr_enabled(self):return self.enabled
 def update_portfolio_metrics(self,s,e):
  self.events+=['total','stock']
  if self.fail=='metrics':raise RuntimeError('metrics')
  self.events.append('cash');self.metrics+=1
 def update_hist_positions(self,s):
  self.events+=['total','account:105.0','weights','snapshot']
  if self.fail=='history':raise RuntimeError('history')
  self.history+=1
 def update_indicator(self,**kw):
  self.events.append('reset')
  if self.fail=='indicator':raise RuntimeError('reset')
  self.events.append('atomic' if kw['atomic'] else 'nested');self.events.append('calculate')
  if kw['indicator_config'].get('show'):self.events.append('output:day')
  self.events.append('record')
def run(mode,missing,enabled,fail,show):
 a=A(enabled,fail);trade=None if mode=='atomic' and missing else [];inner=None if mode=='nested' and missing else []
 try:a.update_bar_end('s','e','exchange',mode=='atomic','decision',trade,inner,[],{'show':show});err=None
 except Exception as x:err=str(x)
 return [err,a.events,a.metrics,a.history]
cases=[('atomic',True,True,None,False),('nested',True,True,None,False),('atomic',False,True,None,True),('nested',False,False,None,False),('atomic',False,True,'current',False),('atomic',False,True,'metrics',False),('atomic',False,True,'history',False),('atomic',False,True,'indicator',False)]
print(json.dumps([run(*x) for x in cases],separators=(',',':')))
";
    let output = Command::new("python")
        .arg("-c")
        .arg(script)
        .arg(source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

fn rust_bar_end_row(
    mode: &str,
    missing: bool,
    enabled: bool,
    failure: Option<&str>,
    show: bool,
) -> Value {
    let start = datetime("2024-01-02 09:30:00");
    let end = datetime("2024-01-02 09:30:59");
    let mut position = metrics_position(100.0);
    match failure {
        Some("current") => position.count_error = Some("current".to_owned()),
        Some("metrics") => position.stock = Err("metrics".to_owned()),
        Some("history") => position.snapshot_error = Some("history".to_owned()),
        Some("indicator") | None => {}
        _ => unreachable!(),
    }
    let log = Arc::clone(&position.events);
    let mut account = Account::new(position, enabled);
    account.replace_indicator(Box::new(ProbeIndicator {
        events: Arc::clone(&log),
        fail: (failure == Some("indicator")).then_some("reset".to_owned()),
        trade: Arc::default(),
        order: Arc::default(),
        history: IndexMap::new(),
    }));
    account.replace_indicator_output(Box::new(ProbeIndicatorOutput {
        events: Arc::clone(&log),
        fail: false,
    }));
    let market = ScriptMarket::new(Vec::new(), Vec::new());
    let result = if mode == "atomic" {
        account.update_bar_end(AccountBarEndUpdate {
            trade_start_time: start,
            trade_end_time: end,
            market: &market,
            mode: AccountBarEndMode::Atomic((!missing).then_some(&[][..])),
            calculation: IndicatorConfig::default(),
            show_indicator: show,
        })
    } else {
        let decision = OrderTradeDecision::from_orders(Vec::new(), start, end, None);
        let provider = exchange(&[], Vec::new());
        let mut inner = Vec::new();
        let nested = (!missing).then_some(NestedAccountIndicatorUpdate {
            inner: &mut inner,
            outer_decision: &decision,
            steps: &[],
            provider: &provider,
            config: OrderIndicatorAggregationConfig::default(),
        });
        account.update_bar_end(AccountBarEndUpdate {
            trade_start_time: start,
            trade_end_time: end,
            market: &market,
            mode: AccountBarEndMode::Nested(nested),
            calculation: IndicatorConfig::default(),
            show_indicator: show,
        })
    };
    let error = match result {
        Ok(()) => Value::Null,
        Err(AccountError::Position(error)) => json!(error.message),
        Err(AccountError::Indicator(error)) => json!(error.message),
        Err(error) => json!(error.to_string()),
    };
    json!([
        error,
        events(&log),
        account
            .portfolio_metrics()
            .map_or(0, |metrics| metrics.records().len()),
        account.historical_positions().read().unwrap().len(),
    ])
}

#[test]
fn account_bar_end_matches_live_python_source() {
    let rows = [
        rust_bar_end_row("atomic", true, true, None, false),
        rust_bar_end_row("nested", true, true, None, false),
        rust_bar_end_row("atomic", false, true, None, true),
        rust_bar_end_row("nested", false, false, None, false),
        rust_bar_end_row("atomic", false, true, Some("current"), false),
        rust_bar_end_row("atomic", false, true, Some("metrics"), false),
        rust_bar_end_row("atomic", false, true, Some("history"), false),
        rust_bar_end_row("atomic", false, true, Some("indicator"), false),
    ];
    assert_eq!(Value::Array(rows.into()), live_python_bar_end_snapshot());
}

#[test]
fn account_portfolio_report_exports_arrow_history_and_every_disabled_state() {
    let start = datetime("2024-01-02 00:00:00");
    let end = datetime("2024-01-02 23:59:59");
    let mut disabled = Account::new(finite(100.0, 1.0, Some(5.0)), false);
    assert!(matches!(
        disabled.portfolio_report(),
        Err(AccountError::PortfolioReportDisabled)
    ));

    let mut inconsistent_position = ProbePosition::new(false, Ok(5.0), None);
    inconsistent_position.supports_metrics = false;
    let inconsistent = Account::new(inconsistent_position, true);
    assert!(matches!(
        inconsistent.portfolio_report(),
        Err(AccountError::PortfolioMetricsDisabled)
    ));

    let mut enabled = Account::new(finite(100.0, 1.0, Some(5.0)), true);
    enabled.update_portfolio_metrics(start, end).unwrap();
    enabled.update_historical_positions(start).unwrap();
    let report = enabled.portfolio_report().unwrap();
    assert_eq!(report.metrics.num_rows(), 1);
    assert_eq!(
        report
            .metrics
            .schema()
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .collect::<Vec<_>>(),
        [
            "datetime",
            "account",
            "return",
            "total_turnover",
            "turnover",
            "total_cost",
            "cost",
            "value",
            "cash",
            "bench",
        ]
    );
    assert!(Arc::ptr_eq(
        &report.positions,
        enabled.historical_positions()
    ));
    assert_eq!(report.positions.read().unwrap().len(), 1);

    disabled.update_historical_positions(start).unwrap();
    assert_eq!(disabled.historical_positions().read().unwrap().len(), 1);
    assert!(matches!(
        disabled.portfolio_report(),
        Err(AccountError::PortfolioReportDisabled)
    ));
}

#[test]
fn owned_order_indicator_snapshots_are_independent_and_plugin_failures_are_typed() {
    let first_time = datetime("2024-01-02 09:30:00");
    let second_time = datetime("2024-01-02 09:31:00");
    let first_order = indicator_order();
    let first_execution = OrderExecution {
        order: &first_order,
        trade_value: 40.0,
        trade_cost: 1.0,
        trade_price: 10.0,
    };
    let mut account = Account::new(finite(100.0, 0.0, Some(5.0)), false);
    account
        .update_indicator(AccountIndicatorUpdate {
            trade_start_time: first_time,
            mode: AccountIndicatorMode::Atomic(std::slice::from_ref(&first_execution)),
            calculation: IndicatorConfig::default(),
            show_indicator: false,
        })
        .unwrap();
    let first_snapshot = account.order_indicator_snapshot().unwrap();

    let mut second_order = Order::new("B", 5.0, OrderDir::Buy, None, None);
    second_order.set_deal_amount(2.0);
    let second_execution = OrderExecution {
        order: &second_order,
        trade_value: 14.0,
        trade_cost: 0.5,
        trade_price: 7.0,
    };
    account
        .update_indicator(AccountIndicatorUpdate {
            trade_start_time: second_time,
            mode: AccountIndicatorMode::Atomic(std::slice::from_ref(&second_execution)),
            calculation: IndicatorConfig::default(),
            show_indicator: false,
        })
        .unwrap();
    let second_snapshot = account.order_indicator_snapshot().unwrap();
    let first_deals = first_snapshot.get_metric_series("deal_amount").unwrap();
    let second_deals = second_snapshot.get_metric_series("deal_amount").unwrap();
    assert_eq!(first_deals.get("A").unwrap().to_bits(), 4.0_f64.to_bits());
    assert_eq!(first_deals.get("B"), None);
    assert_eq!(second_deals.get("A"), None);
    assert_eq!(second_deals.get("B").unwrap().to_bits(), 2.0_f64.to_bits());

    let probe = ProbeIndicator::new(None);
    let log = Arc::clone(&probe.events);
    account.replace_indicator(Box::new(probe));
    assert!(
        account
            .order_indicator_snapshot()
            .unwrap()
            .to_series()
            .is_empty()
    );
    assert_eq!(events(&log), ["snapshot"]);

    let probe = ProbeIndicator::new(Some("snapshot"));
    let log = Arc::clone(&probe.events);
    account.replace_indicator(Box::new(probe));
    assert!(matches!(
        account.order_indicator_snapshot(),
        Err(AccountError::Indicator(_))
    ));
    assert_eq!(events(&log), ["snapshot"]);
}

fn live_python_portfolio_report_snapshot() -> Value {
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/backtest/account.py");
    let script = r"
import ast,json,sys
p=sys.argv[1];t=ast.parse(open(p,encoding='utf-8').read());c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='Account');f=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='get_portfolio_metrics');f.returns=None
for a in f.args.args:a.annotation=None
K=ast.fix_missing_locations(ast.ClassDef('Account',[],[],[f],[]));ns={};exec(compile(ast.Module([K],[]),p,'exec'),ns);B=ns['Account']
class PM:
 def generate_portfolio_metrics_dataframe(self):return [{'account':105.}]
class A(B):
 def __init__(self,enabled):self.enabled=enabled;self.portfolio_metrics=PM()
 def is_port_metr_enabled(self):return self.enabled
 def get_hist_positions(self):return {'t':'position'}
def run(enabled):
 a=A(enabled)
 try:m,p=a.get_portfolio_metrics();return [None,len(m),len(p)]
 except Exception as e:return [str(e),0,0]
print(json.dumps([run(True),run(False)],separators=(',',':')))
";
    let output = Command::new("python")
        .arg("-c")
        .arg(script)
        .arg(source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn account_portfolio_report_matches_live_python_source() {
    let start = datetime("2024-01-02 00:00:00");
    let end = datetime("2024-01-02 23:59:59");
    let mut enabled = Account::new(finite(100.0, 1.0, Some(5.0)), true);
    enabled.update_portfolio_metrics(start, end).unwrap();
    enabled.update_historical_positions(start).unwrap();
    let report = enabled.portfolio_report().unwrap();
    let success = json!([
        Value::Null,
        report.metrics.num_rows(),
        report.positions.read().unwrap().len()
    ]);
    let disabled = Account::new(finite(100.0, 1.0, Some(5.0)), false);
    let error = match disabled.portfolio_report() {
        Err(error) => error.to_string(),
        Ok(_) => unreachable!(),
    };
    assert_eq!(
        Value::Array(vec![success, json!([error, 0, 0])]),
        live_python_portfolio_report_snapshot()
    );
}

#[test]
fn account_trade_indicator_accessor_matches_live_python_identity() {
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/backtest/account.py");
    let script = r"
import ast,json,sys
p=sys.argv[1];t=ast.parse(open(p,encoding='utf-8').read());c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='Account');f=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='get_trade_indicator');f.returns=None
for a in f.args.args:a.annotation=None
K=ast.fix_missing_locations(ast.ClassDef('Account',[],[],[f],[]));ns={};exec(compile(ast.Module([K],[]),p,'exec'),ns);A=ns['Account'];a=A();a.indicator=object();print(json.dumps(a.get_trade_indicator() is a.indicator))
";
    let output = Command::new("python")
        .arg("-c")
        .arg(script)
        .arg(source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let account = Account::new(finite(100.0, 0.0, Some(5.0)), false);
    assert_eq!(
        json!(Arc::ptr_eq(account.indicator(), account.indicator())),
        serde_json::from_slice::<Value>(&output.stdout).unwrap()
    );
}
