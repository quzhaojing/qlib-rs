#![allow(clippy::float_cmp)]

use std::{path::PathBuf, process::Command, sync::Arc};

use chrono::{NaiveDate, NaiveDateTime};
use domain_core::{
    Indicator, IndicatorAggregationMode, IndicatorConfig, IndicatorError, IndicatorStore,
    IndicatorStoreAccess, IndicatorWeightMethod, MetricSnapshot, NumpyOrderIndicator, Order,
    OrderDir, OrderExecution, PandasOrderIndicator,
};
use indexmap::IndexMap;
use serde_json::{Value, json};

fn time(hour: u32) -> NaiveDateTime {
    NaiveDate::from_ymd_opt(2024, 1, 2)
        .unwrap()
        .and_hms_opt(hour, 0, 0)
        .unwrap()
}

fn native_aggregation_failure<S: IndicatorStore>(omissions: &[&[&str]], shared: bool) -> Value {
    let metrics = [
        "inner_amount",
        "deal_amount",
        "trade_price",
        "trade_value",
        "trade_cost",
        "trade_dir",
    ];
    let mut inner: Vec<S> = omissions
        .iter()
        .map(|omitted| {
            let mut store = S::default();
            for (name, value) in metrics.into_iter().zip([4.0, 2.0, 3.0, 6.0, 1.0, 1.0]) {
                if !omitted.contains(&name) {
                    store_assign(&mut store, name, &[("A", value)]);
                }
            }
            store
        })
        .collect();
    let mut output = Indicator::<S>::default();
    for name in metrics {
        assign(&mut output, name, &[("OLD", 9.0)]);
    }
    let result = if shared {
        let stores: Vec<_> = inner
            .into_iter()
            .map(|store| Arc::new(std::sync::RwLock::new(store)))
            .collect();
        let result = output.aggregate_shared_order_trade_info(&stores);
        inner = stores
            .iter()
            .map(|store| store.read().unwrap().clone())
            .collect();
        result
    } else {
        output.aggregate_order_trade_info(&mut inner)
    };
    let IndicatorError::MissingMetric(error) = result.unwrap_err() else {
        panic!("expected the first missing source metric")
    };
    let out: serde_json::Map<String, Value> = output
        .order_snapshot()
        .unwrap()
        .into_iter()
        .map(|(name, metric)| {
            (
                name,
                json!({"index":metric.index(), "values":metric.values()}),
            )
        })
        .collect();
    let prices: Vec<_> = inner
        .iter()
        .map(|store| {
            let price = store.metric_snapshot("trade_price").unwrap();
            json!({"index":price.index(), "values":price.values()})
        })
        .collect();
    json!({"error":error,"out":out,"prices":prices})
}

#[test]
fn raw_alias_source_contract_and_native_aggregation_failure_order() {
    let output = Command::new(std::env::var_os("PYTHON").unwrap_or_else(|| "python".into()))
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/inner_indicator_alias_contract.py"
        ))
        .arg(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let source: Value = serde_json::from_slice(&output.stdout).unwrap();
    for backend in ["numpy", "pandas"] {
        // Shared aggregation is native; executor/account raw-handle transport remains pending.
        assert_eq!(
            source[backend]["duplicate"]["frozen"]["trade_price"]["values"],
            json!([3.0])
        );
        assert_eq!(
            source[backend]["duplicate"]["old"]["trade_price"]["values"],
            json!([12.0])
        );
        assert_eq!(
            source[backend]["duplicate"]["out"]["trade_price"]["values"],
            json!([6.0])
        );
        assert_eq!(source[backend]["duplicate"]["new_empty"], true);
        assert_eq!(
            source[backend]["self_alias"],
            source[backend]["duplicate"]["out"]
        );
        let (duplicate, self_alias) = if backend == "numpy" {
            native_shared_aliases::<NumpyOrderIndicator>()
        } else {
            native_shared_aliases::<PandasOrderIndicator>()
        };
        assert_eq!(duplicate, source[backend]["duplicate"]);
        assert_eq!(self_alias, source[backend]["self_alias"]);
        for (name, omissions) in [
            ("late_missing", vec![&["trade_cost"][..]]),
            (
                "metric_order",
                vec![&["trade_cost"][..], &["trade_value"][..]],
            ),
            ("early_missing", vec![&[][..], &["deal_amount"][..]]),
        ] {
            for shared in [false, true] {
                let native = if backend == "numpy" {
                    native_aggregation_failure::<NumpyOrderIndicator>(&omissions, shared)
                } else {
                    native_aggregation_failure::<PandasOrderIndicator>(&omissions, shared)
                };
                assert_eq!(
                    native, source[backend][name],
                    "{backend}/{name}/shared={shared}"
                );
            }
        }
    }
}

fn alias_store<S: IndicatorStore>() -> S {
    let mut store = S::default();
    for (name, value) in [
        ("inner_amount", 4.0),
        ("deal_amount", 2.0),
        ("trade_price", 3.0),
        ("trade_value", 6.0),
        ("trade_cost", 1.0),
        ("trade_dir", 1.0),
    ] {
        store_assign(&mut store, name, &[("A", value)]);
    }
    store
}

fn store_json(store: &dyn IndicatorStoreAccess) -> Value {
    Value::Object(
        store
            .metric_names()
            .map(|name| {
                let metric = store.metric_snapshot(name).unwrap();
                (
                    name.to_owned(),
                    json!({"index":metric.index(),"values":metric.values()}),
                )
            })
            .collect(),
    )
}

fn native_shared_aliases<S: IndicatorStore + 'static>() -> (Value, Value) {
    let mut child = Indicator::with_store(alias_store::<S>());
    child.record(time(9));
    let retained = child.order_indicator().clone();
    let frozen = store_json(&*retained.read().unwrap());
    child.reset();
    let mut parent = Indicator::<S>::default();
    parent.aggregate_shared_order_trade_info(&[]).unwrap();
    parent
        .aggregate_shared_order_trade_info(&[retained.clone(), retained.clone()])
        .unwrap();
    assert!(Arc::ptr_eq(
        &retained,
        &child.order_indicator_history()[&time(9)]
    ));
    assert!(!Arc::ptr_eq(&retained, child.order_indicator()));
    assert!(retained.try_write().is_ok());
    let duplicate = json!({
        "old":store_json(&*retained.read().unwrap()),
        "out":store_json(&*parent.order_indicator().read().unwrap()),
        "frozen":frozen,
        "new_empty":child.order_snapshot().unwrap().is_empty(),
    });
    drop(child);
    drop(parent);
    assert_eq!(
        retained
            .read()
            .unwrap()
            .metric_snapshot("trade_price")
            .unwrap()
            .values(),
        [12.0]
    );
    let mut output = Indicator::with_store(alias_store::<S>());
    let self_raw = output.order_indicator().clone();
    output
        .aggregate_shared_order_trade_info(&[self_raw.clone(), self_raw.clone()])
        .unwrap();
    assert!(Arc::ptr_eq(&self_raw, output.order_indicator()));
    let self_result = store_json(&*self_raw.try_read().expect("self-alias guard is released"));
    let poisoned = Arc::new(std::sync::RwLock::new(alias_store::<S>()));
    poison_store(poisoned.clone());
    assert_eq!(
        output.aggregate_shared_order_trade_info(&[poisoned]),
        Err(IndicatorError::OrderStorePoisoned)
    );
    assert_eq!(store_json(&*self_raw.try_read().unwrap()), self_result);
    (duplicate, self_result)
}

#[derive(Clone, Default)]
struct PoisonOnAssignStore {
    values: NumpyOrderIndicator,
    poison_target: Option<domain_core::SharedOrderIndicator<Self>>,
}

fn poison_store<S: IndicatorStore + 'static>(store: domain_core::SharedOrderIndicator<S>) {
    assert!(
        std::thread::spawn(move || {
            let _guard = store.write().unwrap();
            panic!("poison shared aggregation participant");
        })
        .join()
        .is_err()
    );
}

impl IndicatorStoreAccess for PoisonOnAssignStore {
    fn metric_names(&self) -> Box<dyn Iterator<Item = &str> + '_> {
        self.values.metric_names()
    }
    fn metric_snapshot(&self, name: &str) -> Option<MetricSnapshot> {
        self.values.metric_snapshot(name)
    }
    fn assign_snapshot(&mut self, name: &str, metric: MetricSnapshot) {
        self.values.assign_snapshot(name, metric);
        if let Some(target) = self.poison_target.take() {
            poison_store(target);
        }
    }
}

#[test]
fn shared_aggregation_poison_keeps_reached_mutations_and_releases_every_guard() {
    let first = Arc::new(std::sync::RwLock::new(alias_store::<PoisonOnAssignStore>()));
    let second = Arc::new(std::sync::RwLock::new(alias_store::<PoisonOnAssignStore>()));
    poison_store(second.clone());
    let mut output = Indicator::<PoisonOnAssignStore>::default();
    assert_eq!(
        output.aggregate_shared_order_trade_info(&[first.clone(), second]),
        Err(IndicatorError::OrderStorePoisoned)
    );
    assert_eq!(
        first
            .try_read()
            .unwrap()
            .metric_snapshot("trade_price")
            .unwrap()
            .values(),
        [6.0]
    );
    assert!(output.order_snapshot().unwrap().is_empty());

    // A later input transformation poisons an earlier input before the column-read stage.
    let earlier = Arc::new(std::sync::RwLock::new(alias_store::<PoisonOnAssignStore>()));
    let later = Arc::new(std::sync::RwLock::new(alias_store::<PoisonOnAssignStore>()));
    later.write().unwrap().poison_target = Some(earlier.clone());
    assert_eq!(
        output.aggregate_shared_order_trade_info(&[earlier.clone(), later.clone()]),
        Err(IndicatorError::OrderStorePoisoned)
    );
    assert_eq!(
        earlier
            .read()
            .err()
            .unwrap()
            .into_inner()
            .metric_snapshot("trade_price")
            .unwrap()
            .values(),
        [6.0]
    );
    assert_eq!(
        later
            .try_read()
            .unwrap()
            .metric_snapshot("trade_price")
            .unwrap()
            .values(),
        [6.0]
    );
    assert!(output.order_snapshot().unwrap().is_empty());

    // Input mutation can finish before publishing fails on the now-poisoned output.
    assign(&mut output, "sentinel", &[("OLD", 9.0)]);
    let original_output = output.order_indicator().clone();
    let input = Arc::new(std::sync::RwLock::new(alias_store::<PoisonOnAssignStore>()));
    input.write().unwrap().poison_target = Some(original_output.clone());
    assert_eq!(
        output.aggregate_shared_order_trade_info(std::slice::from_ref(&input)),
        Err(IndicatorError::OrderStorePoisoned)
    );
    assert_eq!(
        input
            .try_read()
            .unwrap()
            .metric_snapshot("trade_price")
            .unwrap()
            .values(),
        [6.0]
    );
    let old = original_output.read().err().unwrap().into_inner();
    assert_eq!(old.metric_names().collect::<Vec<_>>(), ["sentinel"]);
    assert_eq!(old.metric_snapshot("sentinel").unwrap().values(), [9.0]);
}

#[test]
fn source_history_identity_covers_recalculation_reset_failure_and_retained_rows() {
    // Source characterization, not a claim that the native cloned-row gap is fixed.
    let output = Command::new(std::env::var_os("PYTHON").unwrap_or_else(|| "python".into()))
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/indicator_history_identity_contract.py"
        ))
        .arg(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let actual: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        actual,
        json!({
            "same_generation_aliases": true, "recalculation_preserves_identity": true,
            "keys": ["custom", "ffr", "pa", "pos", "deal_amount", "value", "count"],
            "frozen_pa": 0.25, "live_pa": 0.5, "failed_calculation_retains_map": true,
            "reset_detaches": true, "old_order_pa": 0.75, "old_custom": 9.0,
            "replacement_keeps_key_order": true
        })
    );
}

fn shared_trade_history<S: IndicatorStore>() {
    let mut indicator = Indicator::<S>::default();
    let mut order = Order::new("A", 10.0, OrderDir::Buy, None, None);
    order.set_deal_amount(4.0);
    indicator
        .update_order_indicators(&[OrderExecution {
            order: &order,
            trade_value: 40.0,
            trade_cost: 1.0,
            trade_price: 10.0,
        }])
        .unwrap();
    assign(&mut indicator, "pa", &[("A", 0.25)]);
    let old = indicator.trade_indicator().clone();
    old.write().unwrap().insert("custom".into(), 7.0);
    indicator
        .calculate_trade_indicators(IndicatorConfig::default())
        .unwrap();
    indicator.record(time(9));
    indicator.record(time(10));
    let frozen = indicator.trade_indicator_report().unwrap();
    for row in indicator.trade_indicator_history().values() {
        assert!(Arc::ptr_eq(row, &old));
    }
    assign(&mut indicator, "pa", &[("A", 0.5)]);
    indicator
        .calculate_trade_indicators(IndicatorConfig::default())
        .unwrap();
    assert!(Arc::ptr_eq(indicator.trade_indicator(), &old));
    assert_eq!(old.read().unwrap()["custom"], 7.0);
    assert_eq!(old.read().unwrap()["pa"], 0.5);
    let before = old.read().unwrap().clone();
    assert!(
        indicator
            .calculate_trade_indicators_str("mean", "bad")
            .is_err()
    );
    assert_eq!(*old.read().unwrap(), before);
    assert_eq!(
        before.keys().map(String::as_str).collect::<Vec<_>>(),
        [
            "custom",
            "ffr",
            "pa",
            "pos",
            "deal_amount",
            "value",
            "count"
        ]
    );
    let pa = frozen
        .metrics
        .column_by_name("pa")
        .unwrap()
        .as_any()
        .downcast_ref::<arrow_array::Float64Array>()
        .unwrap();
    assert_eq!(pa.values().as_ref(), [0.25, 0.25]);
    indicator.reset();
    assert!(!Arc::ptr_eq(indicator.trade_indicator(), &old));
    assert!(indicator.trade_indicator().read().unwrap().is_empty());
    old.write().unwrap().insert("custom".into(), 9.0);
    indicator.record(time(9));
    assert_eq!(
        indicator
            .trade_indicator_history()
            .keys()
            .copied()
            .collect::<Vec<_>>(),
        [time(9), time(10)]
    );
    assert!(Arc::ptr_eq(
        &indicator.trade_indicator_history()[&time(9)],
        indicator.trade_indicator()
    ));
    assert!(Arc::ptr_eq(
        &indicator.trade_indicator_history()[&time(10)],
        &old
    ));
    assert_eq!(
        indicator.trade_indicator_report().unwrap().timestamps,
        [time(10)]
    );
    drop(indicator);
    assert_eq!(old.read().unwrap()["custom"], 9.0);
}

#[test]
fn both_backends_retain_trade_row_aliases_recalculate_in_place_and_detach_on_reset() {
    shared_trade_history::<NumpyOrderIndicator>();
    shared_trade_history::<PandasOrderIndicator>();
}

fn shared_order_history<S: IndicatorStore>() {
    let mut indicator = Indicator::<S>::default();
    update(&mut indicator);
    let old = indicator.order_indicator().clone();
    indicator.record(time(9));
    indicator.record(time(10));
    for stored in indicator.order_indicator_history().values() {
        assert!(Arc::ptr_eq(stored, &old));
    }
    let frozen = indicator.order_snapshot().unwrap();
    assign(&mut indicator, "pa", &[("A", 0.5)]);
    assert_eq!(
        indicator.order_indicator_history()[&time(9)]
            .read()
            .unwrap()
            .metric_snapshot("pa")
            .unwrap()
            .values(),
        [0.5]
    );
    assert_eq!(frozen["pa"].values(), [0.0, 0.0]);
    indicator.reset();
    assert!(!Arc::ptr_eq(indicator.order_indicator(), &old));
    assert!(indicator.order_snapshot().unwrap().is_empty());
    old.write().unwrap().assign_snapshot(
        "pa",
        MetricSnapshot::try_new(vec!["A".into()], vec![0.75]).unwrap(),
    );
    assert_eq!(
        indicator.order_indicator_history()[&time(10)]
            .read()
            .unwrap()
            .metric_snapshot("pa")
            .unwrap()
            .values(),
        [0.75]
    );
    indicator.record(time(9));
    assert_eq!(
        indicator
            .order_indicator_history()
            .keys()
            .copied()
            .collect::<Vec<_>>(),
        [time(9), time(10)]
    );
    assert!(Arc::ptr_eq(
        &indicator.order_indicator_history()[&time(9)],
        indicator.order_indicator()
    ));
    assert!(Arc::ptr_eq(
        &indicator.order_indicator_history()[&time(10)],
        &old
    ));
    drop(indicator);
    assert_eq!(
        old.read().unwrap().metric_snapshot("pa").unwrap().values(),
        [0.75]
    );
}

#[test]
fn both_backends_keep_order_history_identity_and_frozen_snapshots_across_reset() {
    shared_order_history::<NumpyOrderIndicator>();
    shared_order_history::<PandasOrderIndicator>();
}

#[derive(Clone, Default)]
struct GuardObservedStore {
    values: NumpyOrderIndicator,
    identity: std::sync::Weak<std::sync::RwLock<Self>>,
    events: Arc<std::sync::Mutex<Vec<(String, bool)>>>,
}

impl GuardObservedStore {
    fn observe(&self, operation: &str, name: &str) {
        if let Some(identity) = self.identity.upgrade() {
            let exclusively_held = matches!(
                identity.try_read(),
                Err(std::sync::TryLockError::WouldBlock)
            );
            self.events
                .lock()
                .unwrap()
                .push((format!("{operation}:{name}"), exclusively_held));
        }
    }
}

impl IndicatorStoreAccess for GuardObservedStore {
    fn metric_names(&self) -> Box<dyn Iterator<Item = &str> + '_> {
        self.values.metric_names()
    }

    fn metric_snapshot(&self, name: &str) -> Option<MetricSnapshot> {
        self.observe("read", name);
        self.values.metric_snapshot(name)
    }

    fn assign_snapshot(&mut self, name: &str, metric: MetricSnapshot) {
        self.observe("write", name);
        self.values.assign_snapshot(name, metric);
    }
}

#[test]
fn compound_order_calculations_hold_exclusive_access_and_release_on_return() {
    let mut indicator = Indicator::<GuardObservedStore>::default();
    assign(&mut indicator, "deal_amount", &[("A", 2.0)]);
    assign(&mut indicator, "amount", &[("A", 4.0)]);
    assign(&mut indicator, "trade_price", &[("A", 110.0)]);
    assign(&mut indicator, "trade_dir", &[("A", 0.0)]);
    assign(&mut indicator, "base_price", &[("A", 100.0)]);
    let identity = indicator.order_indicator().clone();
    let events = {
        let mut store = identity.write().unwrap();
        store.identity = Arc::downgrade(&identity);
        store.events.clone()
    };
    indicator.update_order_fulfill_rate().unwrap();
    indicator.aggregate_order_price_advantage().unwrap();
    assert_eq!(
        *events.lock().unwrap(),
        [
            "read:deal_amount",
            "read:amount",
            "write:ffr",
            "read:trade_price",
            "read:trade_dir",
            "read:base_price",
            "write:pa"
        ]
        .map(|name| (name.to_owned(), true))
    );
    let store = identity
        .try_write()
        .expect("no guard escapes either calculation");
    assert_eq!(store.values.metric_snapshot("ffr").unwrap().values(), [0.5]);
    assert!((store.values.metric_snapshot("pa").unwrap().values()[0] - 0.1).abs() < 1e-14);
}

#[test]
fn poisoned_order_store_rejects_reads_updates_and_snapshots_but_reset_detaches_it() {
    let mut indicator = Indicator::new();
    let old = indicator.order_indicator().clone();
    let poison = old.clone();
    assert!(
        std::thread::spawn(move || {
            let _guard = poison.write().unwrap();
            panic!("poison order store");
        })
        .join()
        .is_err()
    );
    indicator.record(time(9));
    assert!(Arc::ptr_eq(
        &indicator.order_indicator_history()[&time(9)],
        &old
    ));
    assert_eq!(
        indicator.order_snapshot().unwrap_err(),
        IndicatorError::OrderStorePoisoned
    );
    assert_eq!(
        indicator.order_indicator_mut().unwrap_err(),
        IndicatorError::OrderStorePoisoned
    );
    assert_eq!(
        indicator.update_order_indicators(&[]).unwrap_err(),
        IndicatorError::OrderStorePoisoned
    );
    assert_eq!(
        indicator.update_trade_amount(&[]).unwrap_err(),
        IndicatorError::OrderStorePoisoned
    );
    assert_eq!(
        indicator.aggregate_order_trade_info(&mut []).unwrap_err(),
        IndicatorError::OrderStorePoisoned
    );
    assert_eq!(
        indicator
            .aggregate_shared_order_trade_info(&[])
            .unwrap_err(),
        IndicatorError::OrderStorePoisoned
    );
    assert_eq!(
        indicator.update_order_fulfill_rate().unwrap_err(),
        IndicatorError::OrderStorePoisoned
    );
    assert_eq!(
        indicator.aggregate_order_price_advantage().unwrap_err(),
        IndicatorError::OrderStorePoisoned
    );
    assert_eq!(
        indicator
            .calculate_trade_indicators(IndicatorConfig::default())
            .unwrap_err(),
        IndicatorError::OrderStorePoisoned
    );
    let plugin: &mut dyn domain_core::AccountIndicator = &mut indicator;
    assert_eq!(
        plugin.update_atomic(&[]).unwrap_err().message,
        "order indicator store lock poisoned"
    );
    assert_eq!(
        plugin.order_indicator_snapshot().unwrap_err().message,
        "order indicator store lock poisoned"
    );
    indicator.reset();
    assert!(old.read().is_err());
    indicator.update_order_indicators(&[]).unwrap();
    assert!(indicator.order_snapshot().is_ok());
    assert!(
        indicator.order_indicator_history()[&time(9)]
            .read()
            .is_err()
    );
    indicator.record(time(9));
    assert!(indicator.order_indicator_history()[&time(9)].read().is_ok());
}

#[test]
fn poisoned_trade_rows_fail_export_and_calculation_without_erasing_history() {
    let mut indicator = Indicator::new();
    update(&mut indicator);
    assign(&mut indicator, "pa", &[("A", 0.25), ("B", 0.5)]);
    let row = indicator.trade_indicator().clone();
    let poison = row.clone();
    assert!(
        std::thread::spawn(move || {
            let _guard = poison.write().unwrap();
            panic!("poison trade row");
        })
        .join()
        .is_err()
    );
    indicator.record(time(9)); // Publishing row identity does not read its contents.
    assert!(Arc::ptr_eq(
        &indicator.trade_indicator_history()[&time(9)],
        &row
    ));
    assert_eq!(
        indicator
            .calculate_trade_indicators(IndicatorConfig::default())
            .unwrap_err(),
        IndicatorError::TradeRowPoisoned
    );
    assert_eq!(
        indicator.trade_indicator_report().unwrap_err().to_string(),
        "trade indicator row lock poisoned"
    );
    let plugin: &dyn domain_core::AccountIndicator = &indicator;
    assert_eq!(
        plugin.trade_indicator_report().unwrap_err().message,
        "trade indicator row lock poisoned"
    );
    indicator.reset();
    assert!(indicator.trade_indicator().read().is_ok());
    assert!(indicator.trade_indicator_report().is_err());
    indicator.record(time(9));
    assert!(
        indicator
            .trade_indicator_report()
            .unwrap()
            .timestamps
            .is_empty()
    );
    assert!(row.read().is_err());
}

fn orders() -> (Order, Order) {
    let mut buy = Order::new("A", 10.0, OrderDir::Buy, Some(time(9)), Some(time(10)));
    buy.set_deal_amount(4.0);
    let mut sell = Order::new("B", 5.0, OrderDir::Sell, Some(time(9)), Some(time(10)));
    sell.set_deal_amount(1.0);
    (buy, sell)
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1.0e-12,
        "{actual} != {expected}"
    );
}

fn close_one_ulp(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= f64::EPSILON,
        "{actual} != {expected} within one f64 epsilon"
    );
}

fn assign<S: IndicatorStore>(indicator: &mut Indicator<S>, name: &str, rows: &[(&str, f64)]) {
    indicator.order_indicator_mut().unwrap().assign_snapshot(
        name,
        MetricSnapshot::try_new(
            rows.iter().map(|(stock, _)| (*stock).to_owned()).collect(),
            rows.iter().map(|(_, value)| *value).collect(),
        )
        .unwrap(),
    );
}

fn store_assign<S: IndicatorStore>(store: &mut S, name: &str, rows: &[(&str, f64)]) {
    store.assign_snapshot(
        name,
        MetricSnapshot::try_new(
            rows.iter().map(|(stock, _)| (*stock).to_owned()).collect(),
            rows.iter().map(|(_, value)| *value).collect(),
        )
        .unwrap(),
    );
}

fn inner_stores<S: IndicatorStore>() -> [S; 2] {
    let mut first = S::default();
    for (name, rows) in [
        ("inner_amount", vec![("B", 2.0), ("A", 10.0)]),
        ("deal_amount", vec![("B", -1.0), ("A", 4.0)]),
        ("trade_price", vec![("B", 3.0), ("A", 10.0)]),
        ("trade_value", vec![("B", -3.0), ("A", 40.0)]),
        ("trade_cost", vec![("B", 2.0), ("A", 1.0)]),
        ("trade_dir", vec![("B", 0.0), ("A", 1.0)]),
    ] {
        store_assign(&mut first, name, &rows);
    }
    let mut second = S::default();
    for (name, rows) in [
        ("inner_amount", vec![("C", 6.0), ("A", 5.0)]),
        ("deal_amount", vec![("C", 0.0), ("A", 1.0)]),
        ("trade_price", vec![("C", 7.0), ("A", 20.0)]),
        ("trade_value", vec![("C", 0.0), ("A", 20.0)]),
        ("trade_cost", vec![("C", 0.0), ("A", 3.0)]),
        ("trade_dir", vec![("C", 1.0), ("A", 0.0)]),
    ] {
        store_assign(&mut second, name, &rows);
    }
    [first, second]
}

fn update<S: IndicatorStore>(indicator: &mut Indicator<S>) {
    let (buy, sell) = orders();
    indicator
        .update_order_indicators(&[
            OrderExecution {
                order: &buy,
                trade_value: 40.0,
                trade_cost: 1.0,
                trade_price: 10.0,
            },
            OrderExecution {
                order: &sell,
                trade_value: 15.0,
                trade_cost: 2.0,
                trade_price: 3.0,
            },
        ])
        .unwrap();
}

#[test]
fn snapshots_validate_and_both_backend_adapters_round_trip() {
    assert!(MetricSnapshot::default().is_empty());
    assert_eq!(MetricSnapshot::empty().index(), &[] as &[String]);
    assert!(matches!(
        MetricSnapshot::try_new(vec!["a".into()], vec![]),
        Err(IndicatorError::LengthMismatch {
            index: 1,
            values: 0
        })
    ));
    assert!(matches!(
        MetricSnapshot::try_new(vec!["a".into(), "a".into()], vec![1.0, 2.0]),
        Err(IndicatorError::DuplicateStock(stock)) if stock == "a"
    ));

    let metric =
        MetricSnapshot::try_new(vec!["a".into(), "b".into()], vec![1.0, f64::NAN]).unwrap();
    assert_eq!(metric.index(), ["a", "b"]);
    assert_eq!(metric.values()[0], 1.0);
    assert!(metric.values()[1].is_nan());

    let mut dense = NumpyOrderIndicator::default();
    dense.assign_snapshot("x", metric.clone());
    assert_eq!(dense.metric_names().collect::<Vec<_>>(), ["x"]);
    let dense_metric = dense.metric_snapshot("x").unwrap();
    assert_eq!(dense_metric.index(), ["a", "b"]);
    assert!(dense.metric_snapshot("missing").is_none());

    let mut pandas = PandasOrderIndicator::default();
    pandas.assign_snapshot("x", metric);
    assert_eq!(pandas.metric_names().collect::<Vec<_>>(), ["x"]);
    let pandas_metric = pandas.metric_snapshot("x").unwrap();
    assert_eq!(pandas_metric.index(), ["a", "b"]);
    assert!(pandas_metric.values()[1].is_nan());
    assert!(pandas.metric_snapshot("missing").is_none());
}

#[test]
fn update_matches_qlib_rows_duplicate_and_nan_rules() {
    let mut indicator = Indicator::new();
    update(&mut indicator);
    let snapshot = indicator.order_snapshot().unwrap();
    assert_eq!(
        snapshot.keys().map(String::as_str).collect::<Vec<_>>(),
        [
            "amount",
            "inner_amount",
            "deal_amount",
            "trade_price",
            "trade_value",
            "trade_cost",
            "trade_dir",
            "pa",
            "ffr"
        ]
    );
    assert_eq!(snapshot["amount"].values(), [10.0, -5.0]);
    assert_eq!(snapshot["inner_amount"].values(), [10.0, -5.0]);
    assert_eq!(snapshot["deal_amount"].values(), [4.0, -1.0]);
    assert_eq!(snapshot["trade_price"].values(), [10.0, 3.0]);
    assert_eq!(snapshot["trade_value"].values(), [40.0, -15.0]);
    assert_eq!(snapshot["trade_cost"].values(), [1.0, 2.0]);
    assert_eq!(snapshot["trade_dir"].values(), [1.0, 0.0]);
    assert_eq!(snapshot["pa"].values(), [0.0, 0.0]);
    assert_eq!(snapshot["ffr"].values(), [0.4, 0.2]);

    let mut duplicate = Order::new("A", 20.0, OrderDir::Sell, None, None);
    duplicate.set_deal_amount(2.0);
    let (buy, _) = orders();
    indicator
        .update_order_indicators(&[
            OrderExecution {
                order: &buy,
                trade_value: 40.0,
                trade_cost: 1.0,
                trade_price: 10.0,
            },
            OrderExecution {
                order: &duplicate,
                trade_value: 6.0,
                trade_cost: 3.0,
                trade_price: 4.0,
            },
        ])
        .unwrap();
    let snapshot = indicator.order_snapshot().unwrap();
    assert_eq!(snapshot["amount"].index(), ["A"]);
    assert_eq!(snapshot["amount"].values(), [-20.0]);
    assert_eq!(snapshot["deal_amount"].values(), [-2.0]);
    assert_eq!(snapshot["ffr"].values(), [0.1]);
    assert_eq!(snapshot["trade_value"].values(), [-6.0]);

    let mut zero = Order::new("Z", 0.0, OrderDir::Buy, None, None);
    zero.set_deal_amount(f64::NAN);
    indicator
        .update_order_indicators(&[OrderExecution {
            order: &zero,
            trade_value: f64::NAN,
            trade_cost: 0.0,
            trade_price: f64::NAN,
        }])
        .unwrap();
    assert!(indicator.order_snapshot().unwrap()["ffr"].values()[0].is_nan());

    indicator.update_order_indicators(&[]).unwrap();
    assert!(indicator.order_snapshot().unwrap()["amount"].is_empty());
}

#[test]
fn statistics_weighting_record_and_reset_are_stable() {
    let mut indicator = Indicator::new();
    update(&mut indicator);
    assign(&mut indicator, "pa", &[("A", 0.1), ("B", -0.2)]);

    indicator
        .calculate_trade_indicators(IndicatorConfig::default())
        .unwrap();
    let trade = indicator.trade_indicator().read().unwrap();
    assert_eq!(
        trade.keys().map(String::as_str).collect::<Vec<_>>(),
        ["ffr", "pa", "pos", "deal_amount", "value", "count"]
    );
    close(trade["ffr"], 0.3);
    close(trade["pa"], -0.05);
    close(trade["pos"], 0.5);
    assert_eq!(trade["deal_amount"], 5.0);
    assert_eq!(trade["value"], 55.0);
    assert_eq!(trade["count"], 2.0);
    drop(trade);

    indicator
        .calculate_trade_indicators_str("amount_weighted", "value_weighted")
        .unwrap();
    close(indicator.trade_indicator().read().unwrap()["ffr"], 0.36);
    close(
        indicator.trade_indicator().read().unwrap()["pa"],
        1.0 / 55.0,
    );
    indicator
        .calculate_trade_indicators(IndicatorConfig {
            fulfill_rate: IndicatorWeightMethod::ValueWeighted,
            price_advantage: IndicatorWeightMethod::AmountWeighted,
        })
        .unwrap();
    close(
        indicator.trade_indicator().read().unwrap()["ffr"],
        19.0 / 55.0,
    );
    close(indicator.trade_indicator().read().unwrap()["pa"], 0.04);
    assert_eq!(
        IndicatorWeightMethod::ValueWeighted.to_string(),
        "value_weighted"
    );

    let at = time(9);
    indicator.record(at);
    indicator.record(at);
    let report = indicator.trade_indicator_report().unwrap();
    assert_eq!(report.timestamps, [at]);
    assert_eq!(report.metrics.num_rows(), 1);
    for (column, (name, value)) in indicator
        .trade_indicator()
        .read()
        .unwrap()
        .iter()
        .enumerate()
    {
        assert_eq!(report.metrics.schema().field(column).name(), name);
        let array = report
            .metrics
            .column(column)
            .as_any()
            .downcast_ref::<arrow_array::Float64Array>()
            .unwrap();
        assert_eq!(array.value(0).to_bits(), value.to_bits());
    }
    assert_eq!(indicator.order_indicator_history().len(), 1);
    assert_eq!(indicator.trade_indicator_history().len(), 1);
    assign(&mut indicator, "pa", &[("A", 9.0)]);
    assert_eq!(
        indicator.order_indicator_history()[&at]
            .read()
            .unwrap()
            .metric_snapshot("pa")
            .unwrap()
            .values(),
        [9.0]
    );
    indicator.reset();
    assert!(indicator.order_snapshot().unwrap().is_empty());
    assert!(indicator.trade_indicator().read().unwrap().is_empty());
    assert_eq!(indicator.order_indicator_history().len(), 1);
}

#[derive(Clone, Default, Debug)]
struct TestStore {
    names: Vec<String>,
    metrics: IndexMap<String, MetricSnapshot>,
}

impl IndicatorStoreAccess for TestStore {
    fn metric_names(&self) -> Box<dyn Iterator<Item = &str> + '_> {
        Box::new(self.names.iter().map(String::as_str))
    }

    fn metric_snapshot(&self, name: &str) -> Option<MetricSnapshot> {
        self.metrics.get(name).cloned()
    }

    fn assign_snapshot(&mut self, name: &str, metric: MetricSnapshot) {
        if !self.metrics.contains_key(name) {
            self.names.push(name.to_owned());
        }
        self.metrics.insert(name.to_owned(), metric);
    }
}

#[test]
fn typed_failures_cover_plugins_missing_metrics_and_alignment() {
    let advertised = Indicator::with_store(TestStore {
        names: vec!["ghost".into()],
        ..TestStore::default()
    });
    assert_eq!(
        advertised
            .order_indicator()
            .read()
            .unwrap()
            .metric_names()
            .collect::<Vec<_>>(),
        ["ghost"]
    );
    assert!(
        matches!(advertised.order_snapshot(), Err(IndicatorError::MissingMetric(name)) if name == "ghost")
    );

    let mut missing = Indicator::<NumpyOrderIndicator>::default();
    assert!(
        matches!(missing.calculate_trade_indicators(IndicatorConfig::default()), Err(IndicatorError::MissingMetric(name)) if name == "ffr")
    );
    for (name, rows) in [
        ("ffr", vec![("a", 0.5)]),
        ("pa", vec![("a", 0.1)]),
        ("deal_amount", vec![("a", 2.0)]),
        ("trade_value", vec![("a", 3.0)]),
    ] {
        assign(&mut missing, name, &rows);
        let expected = match name {
            "ffr" => "pa",
            "pa" => "deal_amount",
            "deal_amount" => "trade_value",
            _ => "amount",
        };
        assert!(
            matches!(missing.calculate_trade_indicators(IndicatorConfig::default()), Err(IndicatorError::MissingMetric(found)) if found == expected)
        );
    }
    assign(&mut missing, "amount", &[("a", 1.0)]);
    assign(&mut missing, "deal_amount", &[("a", 2.0), ("b", 1.0)]);
    assert!(
        matches!(missing.calculate_trade_indicators(IndicatorConfig { fulfill_rate: IndicatorWeightMethod::AmountWeighted, price_advantage: IndicatorWeightMethod::Mean }), Err(IndicatorError::IndexMismatch { metric, weight }) if metric == "ffr" && weight == "deal_amount")
    );
    assign(&mut missing, "deal_amount", &[("b", 2.0)]);
    assert!(matches!(
        missing.calculate_trade_indicators(IndicatorConfig {
            fulfill_rate: IndicatorWeightMethod::AmountWeighted,
            price_advantage: IndicatorWeightMethod::Mean
        }),
        Err(IndicatorError::IndexMismatch { .. })
    ));
    assign(&mut missing, "deal_amount", &[("a", 2.0)]);
    assign(&mut missing, "trade_value", &[("b", 3.0)]);
    assert!(
        matches!(missing.calculate_trade_indicators(IndicatorConfig { fulfill_rate: IndicatorWeightMethod::Mean, price_advantage: IndicatorWeightMethod::ValueWeighted }), Err(IndicatorError::IndexMismatch { metric, weight }) if metric == "pa" && weight == "trade_value")
    );

    assert!(
        matches!(missing.calculate_trade_indicators_str("bad", "mean"), Err(IndicatorError::UnsupportedWeightMethod(method)) if method == "bad")
    );
    assert!(
        matches!(missing.calculate_trade_indicators_str("mean", "bad"), Err(IndicatorError::UnsupportedWeightMethod(method)) if method == "bad")
    );

    for error in [
        IndicatorError::LengthMismatch {
            index: 1,
            values: 2,
        },
        IndicatorError::DuplicateStock("x".into()),
        IndicatorError::MissingMetric("x".into()),
        IndicatorError::IndexMismatch {
            metric: "x".into(),
            weight: "w".into(),
        },
        IndicatorError::UnsupportedWeightMethod("x".into()),
    ] {
        assert!(!error.to_string().is_empty());
    }
}

#[test]
fn pandas_backend_and_empty_statistics_follow_dense_semantics() {
    let mut pandas = Indicator::with_store(PandasOrderIndicator::default());
    update(&mut pandas);
    assign(&mut pandas, "pa", &[("A", 0.1), ("B", -0.2)]);
    pandas
        .calculate_trade_indicators(IndicatorConfig::default())
        .unwrap();
    close(pandas.trade_indicator().read().unwrap()["ffr"], 0.3);
    let mut zero = Order::new("Z", 0.0, OrderDir::Buy, None, None);
    zero.set_deal_amount(f64::NAN);
    pandas
        .update_order_indicators(&[OrderExecution {
            order: &zero,
            trade_value: f64::NAN,
            trade_cost: 0.0,
            trade_price: f64::NAN,
        }])
        .unwrap();
    assert!(pandas.order_snapshot().unwrap()["ffr"].values()[0].is_nan());

    let mut empty = Indicator::<NumpyOrderIndicator>::default();
    for name in ["ffr", "pa", "deal_amount", "trade_value", "amount"] {
        assign(&mut empty, name, &[]);
    }
    empty
        .calculate_trade_indicators(IndicatorConfig {
            fulfill_rate: IndicatorWeightMethod::AmountWeighted,
            price_advantage: IndicatorWeightMethod::Mean,
        })
        .unwrap();
    assert!(empty.trade_indicator().read().unwrap()["ffr"].is_nan());
    assert!(empty.trade_indicator().read().unwrap()["pa"].is_nan());
    assert!(empty.trade_indicator().read().unwrap()["pos"].is_nan());
    assert_eq!(empty.trade_indicator().read().unwrap()["deal_amount"], 0.0);

    assign(&mut empty, "ffr", &[("a", f64::NAN), ("b", 0.5)]);
    assign(&mut empty, "pa", &[("a", f64::NAN), ("b", 0.1)]);
    assign(&mut empty, "deal_amount", &[("a", 2.0), ("b", f64::NAN)]);
    assign(&mut empty, "trade_value", &[("a", 3.0), ("b", 4.0)]);
    assign(&mut empty, "amount", &[("a", f64::NAN), ("b", 1.0)]);
    empty
        .calculate_trade_indicators(IndicatorConfig {
            fulfill_rate: IndicatorWeightMethod::AmountWeighted,
            price_advantage: IndicatorWeightMethod::ValueWeighted,
        })
        .unwrap();
    assert_eq!(empty.trade_indicator().read().unwrap()["ffr"], 0.0);
    close(empty.trade_indicator().read().unwrap()["pa"], 0.4 / 7.0);
    assert_eq!(empty.trade_indicator().read().unwrap()["count"], 1.0);
}

fn assert_normal_aggregation<S: IndicatorStore>() {
    let mut inner = inner_stores::<S>();
    let mut output = Indicator::with_store(S::default());
    output.aggregate_order_trade_info(&mut inner).unwrap();
    let snapshot = output.order_snapshot().unwrap();
    assert_eq!(
        snapshot.keys().map(String::as_str).collect::<Vec<_>>(),
        [
            "inner_amount",
            "deal_amount",
            "trade_price",
            "trade_value",
            "trade_cost",
            "trade_dir"
        ]
    );
    for metric in snapshot.values() {
        assert_eq!(metric.index(), ["A", "B", "C"]);
    }
    assert_eq!(snapshot["inner_amount"].values(), [15.0, 2.0, 6.0]);
    assert_eq!(snapshot["deal_amount"].values(), [5.0, -1.0, 0.0]);
    assert_eq!(snapshot["trade_price"].values()[..2], [12.0, 3.0]);
    assert!(snapshot["trade_price"].values()[2].is_nan());
    assert_eq!(snapshot["trade_value"].values(), [60.0, -3.0, 0.0]);
    assert_eq!(snapshot["trade_cost"].values(), [4.0, 2.0, 0.0]);
    assert_eq!(snapshot["trade_dir"].values(), [1.0, 0.0, 1.0]);
    assert_eq!(
        inner[0].metric_snapshot("trade_price").unwrap().values(),
        [-3.0, 40.0]
    );
    assert_eq!(
        inner[1].metric_snapshot("trade_price").unwrap().values(),
        [0.0, 20.0]
    );
}

#[test]
fn cross_layer_aggregation_matches_both_backends_and_mutates_inner_price() {
    assert_eq!(
        NumpyOrderIndicator::default().aggregation_mode(),
        IndicatorAggregationMode::DenseZeroFill
    );
    assert_eq!(
        PandasOrderIndicator::default().aggregation_mode(),
        IndicatorAggregationMode::PandasFillValue
    );
    assert_eq!(
        TestStore::default().aggregation_mode(),
        IndicatorAggregationMode::DenseZeroFill
    );
    assert_normal_aggregation::<NumpyOrderIndicator>();
    assert_normal_aggregation::<PandasOrderIndicator>();
}

fn assert_outer_trade_amount<S: IndicatorStore>() {
    let first = Order::new("B", 1.0, OrderDir::Buy, None, None);
    let second = Order::new("A", 2.0, OrderDir::Sell, None, None);
    let replacement = Order::new("B", 9.0, OrderDir::Sell, None, None);
    let missing = Order::new("C", f64::NAN, OrderDir::Buy, None, None);
    let mut indicator = Indicator::<S>::default();

    indicator
        .update_trade_amount(&[first, second, replacement, missing])
        .unwrap();
    let amount = &indicator.order_snapshot().unwrap()["amount"];
    assert_eq!(amount.index(), ["B", "A", "C"]);
    assert_eq!(amount.values()[..2], [-9.0, -2.0]);
    assert!(amount.values()[2].is_nan());

    indicator.update_trade_amount(&[]).unwrap();
    assert!(indicator.order_snapshot().unwrap()["amount"].is_empty());
}

#[test]
fn outer_trade_amount_preserves_sign_duplicates_empty_and_both_backends() {
    assert_outer_trade_amount::<NumpyOrderIndicator>();
    assert_outer_trade_amount::<PandasOrderIndicator>();
}

fn assert_price_advantage<S: IndicatorStore>(expected_index: &[&str]) {
    let mut indicator = Indicator::<S>::default();
    assign(
        &mut indicator,
        "trade_dir",
        &[
            ("B", 0.0),
            ("A", 1.0),
            ("C", 0.0),
            ("D", 1.0),
            ("E", 0.0),
            ("F", f64::NAN),
        ],
    );
    assign(
        &mut indicator,
        "trade_price",
        &[
            ("F", 2.0),
            ("E", f64::INFINITY),
            ("D", 10.0),
            ("C", 0.0),
            ("B", 110.0),
            ("A", 90.0),
        ],
    );
    assign(
        &mut indicator,
        "base_price",
        &[
            ("A", 100.0),
            ("B", 100.0),
            ("C", 0.0),
            ("D", 0.0),
            ("E", f64::INFINITY),
            ("F", 2.0),
        ],
    );

    indicator.aggregate_order_price_advantage().unwrap();
    let pa = &indicator.order_snapshot().unwrap()["pa"];
    assert_eq!(pa.index(), expected_index);
    let values: IndexMap<_, _> = pa
        .index()
        .iter()
        .zip(pa.values())
        .map(|(stock, value)| (stock.as_str(), *value))
        .collect();
    assert_eq!(values["A"], -(90.0 / 100.0 - 1.0));
    assert_eq!(values["B"], 110.0 / 100.0 - 1.0);
    assert!(values["C"].is_nan());
    assert_eq!(values["D"], f64::NEG_INFINITY);
    assert!(values["E"].is_nan());
    assert!(values["F"].is_nan());
}

#[test]
fn price_advantage_preserves_backend_order_and_ieee_special_values() {
    assert_price_advantage::<NumpyOrderIndicator>(&["B", "A", "C", "D", "E", "F"]);
    assert_price_advantage::<PandasOrderIndicator>(&["A", "B", "C", "D", "E", "F"]);
}

#[test]
fn price_advantage_preserves_empty_missing_and_alignment_rules() {
    let mut empty = Indicator::new();
    assign(&mut empty, "trade_price", &[]);
    assign(&mut empty, "pa", &[("old", 1.0)]);
    empty.aggregate_order_price_advantage().unwrap();
    assert!(empty.order_snapshot().unwrap()["pa"].is_empty());

    let mut missing = Indicator::new();
    assign(&mut missing, "pa", &[("OLD", 7.0)]);
    assert!(matches!(
        missing.aggregate_order_price_advantage(),
        Err(IndicatorError::MissingMetric(name)) if name == "trade_price"
    ));
    assign(&mut missing, "trade_price", &[("A", 1.0)]);
    assert!(matches!(
        missing.aggregate_order_price_advantage(),
        Err(IndicatorError::MissingMetric(name)) if name == "trade_dir"
    ));
    assign(&mut missing, "trade_dir", &[("A", 1.0)]);
    assert!(matches!(
        missing.aggregate_order_price_advantage(),
        Err(IndicatorError::MissingMetric(name)) if name == "base_price"
    ));
    assert_eq!(missing.order_snapshot().unwrap()["pa"].values(), [7.0]);
    assert!(missing.order_indicator().try_write().is_ok());

    let mut dense = Indicator::new();
    assign(&mut dense, "trade_price", &[("A", 1.0), ("B", 2.0)]);
    assign(&mut dense, "trade_dir", &[("A", 1.0), ("B", 0.0)]);
    assign(&mut dense, "base_price", &[("A", 1.0)]);
    assert!(matches!(
        dense.aggregate_order_price_advantage(),
        Err(IndicatorError::IndexMismatch { metric, weight }) if metric == "trade_price" && weight == "base_price"
    ));
    assign(&mut dense, "base_price", &[("B", 2.0), ("A", 1.0)]);
    assign(&mut dense, "trade_dir", &[("A", 1.0), ("C", 0.0)]);
    assert!(matches!(
        dense.aggregate_order_price_advantage(),
        Err(IndicatorError::IndexMismatch { metric, weight }) if metric == "trade_dir" && weight == "trade_price"
    ));

    let mut pandas = Indicator::with_store(PandasOrderIndicator::default());
    assign(&mut pandas, "trade_dir", &[("A", 1.0), ("B", 0.0)]);
    assign(&mut pandas, "trade_price", &[("B", 110.0), ("C", 90.0)]);
    assign(&mut pandas, "base_price", &[("C", 100.0), ("D", 100.0)]);
    pandas.aggregate_order_price_advantage().unwrap();
    let pa = &pandas.order_snapshot().unwrap()["pa"];
    assert_eq!(pa.index(), ["A", "B", "C", "D"]);
    assert!(pa.values().iter().all(|value| value.is_nan()));
}

fn all_nan_store<S: IndicatorStore>() -> S {
    let mut store = S::default();
    for name in [
        "inner_amount",
        "deal_amount",
        "trade_price",
        "trade_value",
        "trade_cost",
        "trade_dir",
    ] {
        store_assign(&mut store, name, &[("A", f64::NAN)]);
    }
    store
}

#[test]
fn aggregation_preserves_empty_missing_nan_and_alignment_backend_rules() {
    for mut output in [Indicator::new(), Indicator::new()] {
        output.aggregate_order_trade_info(&mut []).unwrap();
        let snapshot = output.order_snapshot().unwrap();
        assert_eq!(snapshot.len(), 6);
        assert!(snapshot.values().all(MetricSnapshot::is_empty));
    }
    let mut pandas_empty = Indicator::with_store(PandasOrderIndicator::default());
    pandas_empty.aggregate_order_trade_info(&mut []).unwrap();
    assert!(
        pandas_empty
            .order_snapshot()
            .unwrap()
            .values()
            .all(MetricSnapshot::is_empty)
    );

    let mut dense_inner = [all_nan_store(), all_nan_store()];
    let mut dense = Indicator::new();
    dense.aggregate_order_trade_info(&mut dense_inner).unwrap();
    let dense_snapshot = dense.order_snapshot().unwrap();
    for name in ["inner_amount", "deal_amount", "trade_value", "trade_cost"] {
        assert_eq!(dense_snapshot[name].values(), [0.0]);
    }
    assert_eq!(dense_snapshot["trade_dir"].values(), [0.0]);
    assert!(dense_snapshot["trade_price"].values()[0].is_nan());

    let mut pandas_inner = [
        all_nan_store::<PandasOrderIndicator>(),
        all_nan_store::<PandasOrderIndicator>(),
    ];
    let mut pandas = Indicator::with_store(PandasOrderIndicator::default());
    pandas
        .aggregate_order_trade_info(&mut pandas_inner)
        .unwrap();
    let pandas_snapshot = pandas.order_snapshot().unwrap();
    for name in [
        "inner_amount",
        "deal_amount",
        "trade_price",
        "trade_value",
        "trade_cost",
    ] {
        assert!(pandas_snapshot[name].values()[0].is_nan());
    }
    assert_eq!(pandas_snapshot["trade_dir"].values(), [0.0]);

    let mut pandas_misaligned = PandasOrderIndicator::default();
    for (name, rows) in [
        ("inner_amount", vec![("A", 1.0)]),
        ("deal_amount", vec![("A", 1.0)]),
        ("trade_price", vec![("B", 2.0), ("C", 3.0)]),
        ("trade_value", vec![("A", 2.0)]),
        ("trade_cost", vec![("A", 0.0)]),
        ("trade_dir", vec![("A", 1.0)]),
    ] {
        store_assign(&mut pandas_misaligned, name, &rows);
    }
    let mut pandas_misaligned_output = Indicator::with_store(PandasOrderIndicator::default());
    pandas_misaligned_output
        .aggregate_order_trade_info(&mut [pandas_misaligned.clone()])
        .unwrap();
    let misaligned_price =
        pandas_misaligned_output.order_snapshot().unwrap()["trade_price"].clone();
    assert_eq!(misaligned_price.index(), ["A", "B", "C"]);
    assert!(misaligned_price.values().iter().all(|value| value.is_nan()));
    pandas_misaligned_output
        .aggregate_order_trade_info(&mut [pandas_misaligned])
        .unwrap();
}

#[test]
fn aggregation_reports_each_missing_and_dense_alignment_failure() {
    let mut dense_bad = NumpyOrderIndicator::default();
    store_assign(&mut dense_bad, "deal_amount", &[("A", 1.0)]);
    store_assign(&mut dense_bad, "trade_price", &[("B", 2.0)]);
    let mut dense_output = Indicator::new();
    assert!(matches!(
        dense_output.aggregate_order_trade_info(&mut [dense_bad]),
        Err(IndicatorError::IndexMismatch { metric, weight }) if metric == "deal_amount" && weight == "trade_price"
    ));
    let mut dense_bad_length = NumpyOrderIndicator::default();
    store_assign(&mut dense_bad_length, "deal_amount", &[("A", 1.0)]);
    store_assign(
        &mut dense_bad_length,
        "trade_price",
        &[("A", 2.0), ("B", 3.0)],
    );
    assert!(matches!(
        Indicator::new().aggregate_order_trade_info(&mut [dense_bad_length]),
        Err(IndicatorError::IndexMismatch { metric, weight }) if metric == "deal_amount" && weight == "trade_price"
    ));

    for mut bad in [
        NumpyOrderIndicator::default(),
        NumpyOrderIndicator::default(),
    ] {
        store_assign(&mut bad, "inner_amount", &[("A", 1.0)]);
        let mut output = Indicator::new();
        assert!(matches!(
            output.aggregate_order_trade_info(&mut [bad]),
            Err(IndicatorError::MissingMetric(name)) if name == "deal_amount"
        ));
    }
    let mut missing_price = NumpyOrderIndicator::default();
    store_assign(&mut missing_price, "deal_amount", &[("A", 1.0)]);
    assert!(matches!(
        Indicator::new().aggregate_order_trade_info(&mut [missing_price]),
        Err(IndicatorError::MissingMetric(name)) if name == "trade_price"
    ));
    let mut missing_aggregate_metric = NumpyOrderIndicator::default();
    store_assign(&mut missing_aggregate_metric, "deal_amount", &[("A", 1.0)]);
    store_assign(&mut missing_aggregate_metric, "trade_price", &[("A", 2.0)]);
    assert!(matches!(
        Indicator::new().aggregate_order_trade_info(&mut [missing_aggregate_metric]),
        Err(IndicatorError::MissingMetric(name)) if name == "inner_amount"
    ));
    let mut pandas_bad = PandasOrderIndicator::default();
    store_assign(&mut pandas_bad, "inner_amount", &[("A", 1.0)]);
    let mut pandas_output = Indicator::with_store(PandasOrderIndicator::default());
    assert!(matches!(
        pandas_output.aggregate_order_trade_info(&mut [pandas_bad]),
        Err(IndicatorError::MissingMetric(name)) if name == "deal_amount"
    ));
}

#[test]
fn contract_matches_live_python_report_indicator() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib");
    let report_source = root.join("qlib/backtest/report.py");
    let decision_source = root.join("qlib/backtest/decision.py");
    let indicator_source = root.join("qlib/backtest/high_performance_ds.py");
    let index_source = root.join("qlib/utils/index_data.py");
    let script = r"
import ast,importlib.util,inspect,json,sys
from collections import OrderedDict
from dataclasses import dataclass
from enum import IntEnum
from typing import *
import numpy as np,pandas as pd
spec=importlib.util.spec_from_file_location('index_data_live',sys.argv[4]);idd=importlib.util.module_from_spec(spec);spec.loader.exec_module(idd);SingleData=idd.SingleData
class Logger: pass
def get_module_logger(_): return Logger()
def classes(path,names):
 t=ast.parse(open(path,encoding='utf-8').read(),filename=path);return [next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name==name) for name in names]
nodes=classes(sys.argv[2],['OrderDir','Order'])+classes(sys.argv[3],['BaseOrderIndicator','NumpyOrderIndicator'])
m=ast.Module(body=nodes,type_ignores=[]);ast.fix_missing_locations(m);exec(compile(m,'live','exec'),globals())
class BaseTradeDecision: pass
nodes=classes(sys.argv[1],['Indicator'])
m=ast.Module(body=nodes,type_ignores=[]);ast.fix_missing_locations(m);exec(compile(m,'live','exec'),globals())
def snap(x):return {'index':x.index.tolist(),'values':[None if np.isnan(v) else float(v) for v in x.data.tolist()]}
a=Order('A',10,1,None,None);a.deal_amount=4;b=Order('B',5,0,None,None);b.deal_amount=1
i=Indicator();i._update_order_trade_info([(a,40,1,10),(b,15,2,3)]);i._update_order_fulfill_rate();i.order_indicator.assign('pa',{'A':.1,'B':-.2});i.cal_trade_indicators('','')
out={'order':{k:snap(v) for k,v in i.order_indicator.data.items()},'trade':dict(i.trade_indicator)}
i.cal_trade_indicators('','',{'ffr_config':{'weight_method':'amount_weighted'},'pa_config':{'weight_method':'value_weighted'}});out['weighted']=dict(i.trade_indicator)
i.record(pd.Timestamp('2024-01-02 09:00'));i.reset();out['history']=[len(i.order_indicator_his),len(i.trade_indicator_his),len(i.order_indicator.data)]
d=Order('A',20,0,None,None);d.deal_amount=2;i._update_order_trade_info([(a,40,1,10),(d,6,3,4)]);i._update_order_fulfill_rate();out['duplicate']={k:snap(v) for k,v in i.order_indicator.data.items()}
z=Order('Z',0,1,None,None);z.deal_amount=np.nan;i._update_order_trade_info([(z,np.nan,0,np.nan)]);i._update_order_fulfill_rate();out['zero']=snap(i.order_indicator.get_index_data('ffr'))
class Decision:
 def __init__(self,orders):self.orders=orders
 def get_decision(self):return self.orders
outer=Indicator();outer._update_trade_amount(Decision([Order('B',1,1,None,None),Order('A',2,0,None,None),Order('B',9,0,None,None),Order('C',np.nan,1,None,None)]));out['outer_amount']=snap(outer.order_indicator.get_index_data('amount'))
outer._update_trade_amount(Decision([]));out['outer_empty']=snap(outer.order_indicator.get_index_data('amount'))
for key,config in [('bad_ffr',{'ffr_config':{'weight_method':'bad'}}),('bad_pa',{'pa_config':{'weight_method':'bad'}})]:
 try:i.cal_trade_indicators('','',config)
 except Exception as error:out[key]=type(error).__name__
print(json.dumps(out,sort_keys=True))
";
    let output = Command::new(std::env::var_os("PYTHON").unwrap_or_else(|| "python".into()))
        .arg("-c")
        .arg(script)
        .arg(report_source)
        .arg(decision_source)
        .arg(indicator_source)
        .arg(index_source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let actual: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        actual["order"]["amount"],
        json!({"index":["A","B"],"values":[10.0,-5.0]})
    );
    assert_eq!(
        actual["order"]["ffr"],
        json!({"index":["A","B"],"values":[0.4,0.2]})
    );
    assert_eq!(actual["trade"]["deal_amount"], json!(5.0));
    close(actual["trade"]["ffr"].as_f64().unwrap(), 0.3);
    close(actual["weighted"]["ffr"].as_f64().unwrap(), 0.36);
    close(actual["weighted"]["pa"].as_f64().unwrap(), 1.0 / 55.0);
    assert_eq!(actual["history"], json!([1, 1, 0]));
    assert_eq!(
        actual["duplicate"]["amount"],
        json!({"index":["A"],"values":[-20.0]})
    );
    assert_eq!(actual["zero"], json!({"index":["Z"],"values":[null]}));
    assert_eq!(
        actual["outer_amount"],
        json!({"index":["B","A","C"],"values":[-9.0,-2.0,null]})
    );
    assert_eq!(actual["outer_empty"], json!({"index":[],"values":[]}));
    assert_eq!(actual["bad_ffr"], json!("ValueError"));
    assert_eq!(actual["bad_pa"], json!("ValueError"));
}

#[test]
fn cross_layer_aggregation_matches_live_python_backends() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib");
    let report_source = root.join("qlib/backtest/report.py");
    let decision_source = root.join("qlib/backtest/decision.py");
    let indicator_source = root.join("qlib/backtest/high_performance_ds.py");
    let index_source = root.join("qlib/utils/index_data.py");
    let script = r"
import ast,importlib.util,inspect,json,sys
from collections import OrderedDict
from dataclasses import dataclass
from enum import IntEnum
from typing import *
import numpy as np,pandas as pd
spec=importlib.util.spec_from_file_location('index_data_live',sys.argv[4]);idd=importlib.util.module_from_spec(spec);spec.loader.exec_module(idd);SingleData=idd.SingleData
class Logger: pass
def get_module_logger(_): return Logger()
def classes(path,names):
 t=ast.parse(open(path,encoding='utf-8').read(),filename=path);return [next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name==name) for name in names]
nodes=classes(sys.argv[2],['OrderDir','Order'])+classes(sys.argv[3],['BaseSingleMetric','BaseOrderIndicator','SingleMetric','PandasSingleMetric','PandasOrderIndicator','NumpyOrderIndicator'])
m=ast.Module(body=nodes,type_ignores=[]);ast.fix_missing_locations(m);exec(compile(m,'live','exec'),globals())
class BaseTradeDecision: pass
nodes=classes(sys.argv[1],['Indicator']);m=ast.Module(body=nodes,type_ignores=[]);ast.fix_missing_locations(m);exec(compile(m,'live','exec'),globals())
def vals(x):
 try:idx=x.index.tolist();data=x.data.tolist()
 except:idx=list(x.index);data=x.tolist()
 return {'index':idx,'values':[None if pd.isna(v) else ('inf' if v==np.inf else ('-inf' if v==-np.inf else float(v))) for v in data]}
def snap(ind):return {k:vals(ind.get_index_data(k)) for k in ind.data}
def make(cls,rows):
 x=cls()
 for k,v in rows.items():x.assign(k,v)
 return x
def run(cls):
 r1={'inner_amount':{'B':2,'A':10},'deal_amount':{'B':-1,'A':4},'trade_price':{'B':3,'A':10},'trade_value':{'B':-3,'A':40},'trade_cost':{'B':2,'A':1},'trade_dir':{'B':0,'A':1}}
 r2={'inner_amount':{'C':6,'A':5},'deal_amount':{'C':0,'A':1},'trade_price':{'C':7,'A':20},'trade_value':{'C':0,'A':20},'trade_cost':{'C':0,'A':3},'trade_dir':{'C':1,'A':0}}
 a,b=make(cls,r1),make(cls,r2);out=Indicator(cls);out._agg_order_trade_info([a,b])
 result={'out':snap(out.order_indicator),'mutated':[vals(a.get_index_data('trade_price')),vals(b.get_index_data('trade_price'))]}
 empty=Indicator(cls);empty._agg_order_trade_info([]);result['empty']=snap(empty.order_indicator)
 missing={k:{'A':np.nan} for k in r1};c,d=make(cls,missing),make(cls,missing);nanout=Indicator(cls);nanout._agg_order_trade_info([c,d]);result['nan']=snap(nanout.order_indicator)
 bad=make(cls,{'inner_amount':{'A':1}})
 try:Indicator(cls)._agg_order_trade_info([bad])
 except Exception as error:result['bad']=type(error).__name__
 return result
print(json.dumps({'numpy':run(NumpyOrderIndicator),'pandas':run(PandasOrderIndicator)},sort_keys=True))
";
    let output = Command::new(std::env::var_os("PYTHON").unwrap_or_else(|| "python".into()))
        .arg("-c")
        .arg(script)
        .arg(report_source)
        .arg(decision_source)
        .arg(indicator_source)
        .arg(index_source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let actual: Value = serde_json::from_slice(&output.stdout).unwrap();
    for backend in ["numpy", "pandas"] {
        assert_eq!(
            actual[backend]["out"]["inner_amount"],
            json!({"index":["A","B","C"],"values":[15.0,2.0,6.0]})
        );
        assert_eq!(
            actual[backend]["out"]["deal_amount"],
            json!({"index":["A","B","C"],"values":[5.0,-1.0,0.0]})
        );
        assert_eq!(
            actual[backend]["out"]["trade_price"],
            json!({"index":["A","B","C"],"values":[12.0,3.0,null]})
        );
        assert_eq!(
            actual[backend]["out"]["trade_dir"],
            json!({"index":["A","B","C"],"values":[1.0,0.0,1.0]})
        );
        assert_eq!(
            actual[backend]["mutated"],
            json!([
                {"index":["B","A"],"values":[-3.0,40.0]},
                {"index":["C","A"],"values":[0.0,20.0]}
            ])
        );
        assert_eq!(actual[backend]["empty"].as_object().unwrap().len(), 6);
        assert_eq!(actual[backend]["bad"], json!("KeyError"));
    }
    assert_eq!(
        actual["numpy"]["nan"]["inner_amount"]["values"],
        json!([0.0])
    );
    assert_eq!(actual["numpy"]["nan"]["trade_dir"]["values"], json!([0.0]));
    assert_eq!(
        actual["pandas"]["nan"]["inner_amount"]["values"],
        json!([null])
    );
    assert_eq!(actual["pandas"]["nan"]["trade_dir"]["values"], json!([0.0]));
}

fn assert_live_price_advantage(actual: &Value) {
    for backend in ["numpy", "pandas"] {
        assert_eq!(actual[backend]["empty"]["pa"]["index"], json!([]));
        assert_eq!(actual[backend]["missing_trade_price"], json!("KeyError"));
        assert_eq!(actual[backend]["missing_trade_dir"], json!("KeyError"));
        assert_eq!(actual[backend]["missing_base_price"], json!("KeyError"));
    }
    let expected_tail = json!([null, "-inf", null, null]);
    let numpy = &actual["numpy"]["normal"]["pa"];
    assert_eq!(numpy["index"], json!(["B", "A", "C", "D", "E", "F"]));
    close_one_ulp(numpy["values"][0].as_f64().unwrap(), 110.0 / 100.0 - 1.0);
    close_one_ulp(numpy["values"][1].as_f64().unwrap(), -(90.0 / 100.0 - 1.0));
    assert_eq!(
        &numpy["values"].as_array().unwrap()[2..],
        expected_tail.as_array().unwrap()
    );
    assert_eq!(actual["numpy"]["misaligned"], json!("ValueError"));

    let pandas = &actual["pandas"]["normal"]["pa"];
    assert_eq!(pandas["index"], json!(["A", "B", "C", "D", "E", "F"]));
    close_one_ulp(pandas["values"][0].as_f64().unwrap(), -(90.0 / 100.0 - 1.0));
    close_one_ulp(pandas["values"][1].as_f64().unwrap(), 110.0 / 100.0 - 1.0);
    assert_eq!(
        &pandas["values"].as_array().unwrap()[2..],
        expected_tail.as_array().unwrap()
    );
    assert_eq!(
        actual["pandas"]["misaligned"]["pa"],
        json!({"index":["A","B","C","D"],"values":[null,null,null,null]})
    );
}

#[test]
fn price_advantage_matches_live_python_backends() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib");
    let script = r"
import ast,importlib.util,inspect,json,sys
from collections import OrderedDict
from dataclasses import dataclass
from enum import IntEnum
from typing import *
import numpy as np,pandas as pd
spec=importlib.util.spec_from_file_location('index_data_live',sys.argv[4]);idd=importlib.util.module_from_spec(spec);spec.loader.exec_module(idd);SingleData=idd.SingleData
class Logger:pass
def get_module_logger(_):return Logger()
def classes(path,names):
 t=ast.parse(open(path,encoding='utf-8').read(),filename=path);return [next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name==name) for name in names]
nodes=classes(sys.argv[2],['OrderDir','Order'])+classes(sys.argv[3],['BaseSingleMetric','BaseOrderIndicator','SingleMetric','PandasSingleMetric','PandasOrderIndicator','NumpyOrderIndicator'])
m=ast.Module(body=nodes,type_ignores=[]);ast.fix_missing_locations(m);exec(compile(m,'live','exec'),globals())
class BaseTradeDecision:pass
m=ast.Module(body=classes(sys.argv[1],['Indicator']),type_ignores=[]);ast.fix_missing_locations(m);exec(compile(m,'live','exec'),globals())
def vals(x):
 try:idx=x.index.tolist();data=x.data.tolist()
 except:idx=list(x.index);data=x.tolist()
 return {'index':idx,'values':[None if pd.isna(v) else ('inf' if v==np.inf else ('-inf' if v==-np.inf else float(v))) for v in data]}
def snap(ind):return {k:vals(ind.get_index_data(k)) for k in ind.data}
def make(cls,rows):
 x=cls()
 for k,v in rows.items():x.assign(k,v)
 return x
def run(cls):
 normal={'trade_dir':{'B':0,'A':1,'C':0,'D':1,'E':0,'F':np.nan},'trade_price':{'F':2,'E':np.inf,'D':10,'C':0,'B':110,'A':90},'base_price':{'A':100,'B':100,'C':0,'D':0,'E':np.inf,'F':2}}
 out=Indicator(cls);out.order_indicator=make(cls,normal);out._agg_order_price_advantage();result={'normal':snap(out.order_indicator)}
 empty=Indicator(cls);empty.order_indicator.assign('trade_price',{});empty._agg_order_price_advantage();result['empty']=snap(empty.order_indicator)
 bad=Indicator(cls);bad.order_indicator=make(cls,{'trade_dir':{'A':1,'B':0},'trade_price':{'B':110,'C':90},'base_price':{'C':100,'D':100}})
 try:bad._agg_order_price_advantage();result['misaligned']=snap(bad.order_indicator)
 except Exception as error:result['misaligned']=type(error).__name__
 for key,rows in [('missing_trade_price',{}),('missing_trade_dir',{'trade_price':{'A':1}}),('missing_base_price',{'trade_price':{'A':1},'trade_dir':{'A':1}})]:
  missing=Indicator(cls);missing.order_indicator=make(cls,rows)
  try:missing._agg_order_price_advantage();result[key]='ok'
  except Exception as error:result[key]=type(error).__name__
 return result
print(json.dumps({'numpy':run(NumpyOrderIndicator),'pandas':run(PandasOrderIndicator)},sort_keys=True))
";
    let output = Command::new(std::env::var_os("PYTHON").unwrap_or_else(|| "python".into()))
        .arg("-c")
        .arg(script)
        .arg(root.join("qlib/backtest/report.py"))
        .arg(root.join("qlib/backtest/decision.py"))
        .arg(root.join("qlib/backtest/high_performance_ds.py"))
        .arg(root.join("qlib/utils/index_data.py"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_live_price_advantage(&serde_json::from_slice(&output.stdout).unwrap());
}
