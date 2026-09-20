use std::{
    process::Command,
    sync::{Arc, Mutex, RwLock},
};

use chrono::{NaiveDate, NaiveDateTime, TimeDelta};
use domain_core::{
    IdxTradeRange, Order, OrderDir, SAOE_BACKTEST_DATA_CACHE_CAPACITY, SaoeBacktestDataLoadError,
    SaoeBacktestDataLoader, SaoeBacktestDataSource, SaoePluginError, TradeRangeByTime,
};
use ndarray::arr1;
use serde_json::{Value, json};

fn time(day: u32, minute: i64) -> NaiveDateTime {
    NaiveDate::from_ymd_opt(2024, 1, day)
        .unwrap()
        .and_hms_opt(9, 29, 0)
        .unwrap()
        + TimeDelta::minutes(minute)
}

fn order(stock: &str, day: u32, start: i64, end: i64, direction: OrderDir) -> Order {
    Order::new(
        stock,
        10.0,
        direction,
        Some(time(day, start)),
        Some(time(day, end)),
    )
}

struct Source {
    timestamps: Vec<NaiveDateTime>,
    events: Arc<Mutex<Vec<String>>>,
    fail: Option<&'static str>,
}

impl Source {
    fn result<T>(&self, stage: &'static str, value: T) -> Result<T, SaoePluginError> {
        self.events.lock().unwrap().push(stage.to_owned());
        if self.fail == Some(stage) {
            return Err(SaoePluginError {
                message: stage.to_owned(),
            });
        }
        Ok(value)
    }
}

impl SaoeBacktestDataSource for Source {
    fn quote_timestamps(&self) -> Result<Vec<NaiveDateTime>, SaoePluginError> {
        self.result("quote", self.timestamps.clone())
    }

    fn deal_prices(
        &self,
        stock_id: &str,
        start: NaiveDateTime,
        end: NaiveDateTime,
        direction: OrderDir,
    ) -> Result<ndarray::Array1<f64>, SaoePluginError> {
        self.events.lock().unwrap().push(format!(
            "deal:{stock_id}:{start}:{end}:{}",
            direction.value()
        ));
        if self.fail == Some("deal") {
            return Err(SaoePluginError {
                message: "deal".to_owned(),
            });
        }
        Ok(arr1(&[10.0, 11.0]))
    }

    fn market_volumes(
        &self,
        stock_id: &str,
        start: NaiveDateTime,
        end: NaiveDateTime,
    ) -> Result<ndarray::Array1<f64>, SaoePluginError> {
        self.events
            .lock()
            .unwrap()
            .push(format!("volume:{stock_id}:{start}:{end}"));
        if self.fail == Some("volume") {
            return Err(SaoePluginError {
                message: "volume".to_owned(),
            });
        }
        Ok(arr1(&[100.0, 110.0]))
    }
}

fn loader(
    timestamps: Vec<NaiveDateTime>,
    events: Arc<Mutex<Vec<String>>>,
    fail: Option<&'static str>,
) -> SaoeBacktestDataLoader {
    SaoeBacktestDataLoader::new(Arc::new(Source {
        timestamps,
        events,
        fail,
    }))
}

fn python_contract() -> Value {
    let output = Command::new("python")
        .args([
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/saoe_backtest_data_contract.py"
            ),
            r"D:\code\github\qlib\qlib\rl\data\native.py",
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

#[test]
fn loader_matches_python_order_time_slice_eager_queries_and_cache_key() {
    let python = python_contract();
    assert_eq!(python["cache_identity"], true);
    assert_eq!(python["cached_mutation"], 77.0);
    assert_eq!(python["cached_order_is_first"], true);
    assert_eq!(python["cached_exchange_is_first"], true);
    assert_eq!(python["unsupported"], "TypeError");
    assert_eq!(python["empty"], "IndexError");
    assert_eq!(python["second_events"], json!([]));

    let timestamps = vec![time(2, 2), time(2, 0), time(2, 1), time(2, 1), time(2, 3)];
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut loader = loader(timestamps.clone(), Arc::clone(&events), None);
    let range = TradeRangeByTime::parse("09:30", "09:31").unwrap();
    let first = loader
        .load(&order("A", 2, 0, 3, OrderDir::Buy), &range)
        .unwrap();
    assert_eq!(first.ticks_index, timestamps);
    assert_eq!(
        first.ticks_for_order,
        vec![time(2, 2), time(2, 1), time(2, 1)]
    );
    assert_eq!(first.deal_prices, arr1(&[10.0, 11.0]));
    assert_eq!(first.market_volumes, arr1(&[100.0, 110.0]));
    assert_eq!(first.features.num_columns(), 0);
    assert_eq!(first.features.num_rows(), 0);
    assert_eq!(
        *events.lock().unwrap(),
        [
            "quote",
            "deal:A:2024-01-02 09:31:00:2024-01-02 09:30:00:1",
            "volume:A:2024-01-02 09:31:00:2024-01-02 09:30:00"
        ]
    );

    let cached = loader
        .load(
            &order("A", 2, 1, 1, OrderDir::Buy),
            &IdxTradeRange::new(0, 0),
        )
        .unwrap();
    assert_eq!(cached.ticks_index, first.ticks_index);
    assert_eq!(events.lock().unwrap().len(), 3);
    assert_eq!(loader.cache_len(), 1);
}

#[test]
fn loader_reports_every_input_range_and_source_failure() {
    let range = TradeRangeByTime::parse("09:30", "09:31").unwrap();
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut missing_start_loader = loader(vec![time(2, 1)], Arc::clone(&events), None);
    let missing_start = Order::new("A", 1.0, OrderDir::Buy, None, Some(time(2, 2)));
    assert!(matches!(
        missing_start_loader.load(&missing_start, &range),
        Err(SaoeBacktestDataLoadError::Order(_))
    ));
    let missing_end = Order::new("A", 1.0, OrderDir::Buy, Some(time(2, 1)), None);
    assert!(matches!(
        missing_start_loader.load(&missing_end, &range),
        Err(SaoeBacktestDataLoadError::MissingEndTime)
    ));
    assert!(events.lock().unwrap().is_empty());

    let mut empty_interval_loader = loader(vec![time(2, 1)], Arc::clone(&events), None);
    assert!(matches!(
        empty_interval_loader.load(&order("A", 3, 0, 2, OrderDir::Buy), &range),
        Err(SaoeBacktestDataLoadError::EmptyOrderInterval)
    ));
    let mut unsupported_loader = loader(vec![time(2, 1)], Arc::clone(&events), None);
    assert!(matches!(
        unsupported_loader.load(
            &order("A", 2, 0, 2, OrderDir::Buy),
            &IdxTradeRange::new(0, 1)
        ),
        Err(SaoeBacktestDataLoadError::UnsupportedTradeRange)
    ));
    let late = TradeRangeByTime::parse("10:00", "10:01").unwrap();
    let mut empty_range_loader = loader(vec![time(2, 1)], Arc::clone(&events), None);
    assert!(matches!(
        empty_range_loader.load(&order("A", 2, 0, 2, OrderDir::Buy), &late),
        Err(SaoeBacktestDataLoadError::EmptyTradeRange)
    ));

    for (failure, expected_events) in [
        ("quote", vec!["quote"]),
        (
            "deal",
            vec!["quote", "deal:A:2024-01-02 09:30:00:2024-01-02 09:30:00:1"],
        ),
        (
            "volume",
            vec![
                "quote",
                "deal:A:2024-01-02 09:30:00:2024-01-02 09:30:00:1",
                "volume:A:2024-01-02 09:30:00:2024-01-02 09:30:00",
            ],
        ),
    ] {
        let stage_events = Arc::new(Mutex::new(Vec::new()));
        let mut failing = loader(vec![time(2, 1)], Arc::clone(&stage_events), Some(failure));
        let error = failing
            .load(&order("A", 2, 0, 2, OrderDir::Buy), &range)
            .unwrap_err();
        assert!(matches!(error, SaoeBacktestDataLoadError::Source(_)));
        assert_eq!(*stage_events.lock().unwrap(), expected_events);
        assert_eq!(failing.cache_len(), 0);
    }
}

#[test]
fn loader_enforces_the_hundred_entry_lru_capacity() {
    assert_eq!(SAOE_BACKTEST_DATA_CACHE_CAPACITY, 100);
    let timestamps = vec![time(2, 1)];
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut loader = loader(timestamps, Arc::clone(&events), None);
    let range = TradeRangeByTime::parse("09:30", "09:31").unwrap();
    for index in 0..101 {
        loader
            .load(&order(&format!("A{index}"), 2, 0, 2, OrderDir::Buy), &range)
            .unwrap();
    }
    assert_eq!(loader.cache_len(), 100);
    assert_eq!(
        events
            .lock()
            .unwrap()
            .iter()
            .filter(|event| event.as_str() == "quote")
            .count(),
        101
    );
    loader
        .load(&order("A0", 2, 0, 2, OrderDir::Buy), &range)
        .unwrap();
    assert_eq!(loader.cache_len(), 100);
    assert_eq!(
        events
            .lock()
            .unwrap()
            .iter()
            .filter(|event| event.as_str() == "quote")
            .count(),
        102
    );
}

struct MutatingSharedSource {
    order: Arc<RwLock<Order>>,
    events: Arc<Mutex<Vec<String>>>,
    poison_at: Option<&'static str>,
    clear_start_on_quote: bool,
}

impl MutatingSharedSource {
    fn poison_order(&self) {
        let order = Arc::clone(&self.order);
        assert!(
            std::thread::spawn(move || {
                let _guard = order.write().unwrap();
                panic!("poison shared loader order");
            })
            .join()
            .is_err()
        );
    }
}

impl SaoeBacktestDataSource for MutatingSharedSource {
    fn quote_timestamps(&self) -> Result<Vec<NaiveDateTime>, SaoePluginError> {
        self.events.lock().unwrap().push("quote".to_owned());
        if self.poison_at == Some("quote") {
            self.poison_order();
        } else if self.clear_start_on_quote {
            *self.order.write().unwrap() =
                Order::new("B", 1.0, OrderDir::Sell, None, Some(time(2, 3)));
        } else {
            *self.order.write().unwrap() = order("B", 2, 1, 3, OrderDir::Sell);
        }
        Ok(vec![time(2, 0), time(2, 1), time(2, 2), time(2, 3)])
    }

    fn deal_prices(
        &self,
        stock_id: &str,
        start: NaiveDateTime,
        end: NaiveDateTime,
        direction: OrderDir,
    ) -> Result<ndarray::Array1<f64>, SaoePluginError> {
        self.events.lock().unwrap().push(format!(
            "deal:{stock_id}:{start}:{end}:{}",
            direction.value()
        ));
        if self.poison_at == Some("deal") {
            self.poison_order();
        } else {
            *self.order.write().unwrap() = order("C", 2, 1, 3, OrderDir::Buy);
        }
        Ok(arr1(&[20.0, 21.0, 22.0]))
    }

    fn market_volumes(
        &self,
        stock_id: &str,
        start: NaiveDateTime,
        end: NaiveDateTime,
    ) -> Result<ndarray::Array1<f64>, SaoePluginError> {
        self.events
            .lock()
            .unwrap()
            .push(format!("volume:{stock_id}:{start}:{end}"));
        Ok(arr1(&[200.0, 210.0, 220.0]))
    }
}

#[test]
fn shared_loader_releases_guards_and_rereads_the_original_order_at_source_stages() {
    let shared = Arc::new(RwLock::new(order("A", 2, 0, 3, OrderDir::Buy)));
    let events = Arc::new(Mutex::new(Vec::new()));
    let source = Arc::new(MutatingSharedSource {
        order: Arc::clone(&shared),
        events: Arc::clone(&events),
        poison_at: None,
        clear_start_on_quote: false,
    });
    let mut loader = SaoeBacktestDataLoader::new(source);
    let range = TradeRangeByTime::parse("09:30", "09:32").unwrap();
    let data = loader.load_shared(&shared, &range).unwrap();

    let handles = data.read().unwrap().clone();
    assert!(format!("{handles:?}").contains("<data source>"));
    assert_eq!(
        *handles.ticks_index.read().unwrap(),
        vec![time(2, 1), time(2, 2), time(2, 3)]
    );
    assert_eq!(
        *handles.deal_prices.read().unwrap(),
        arr1(&[20.0, 21.0, 22.0])
    );
    assert_eq!(
        *handles.market_volumes.read().unwrap(),
        arr1(&[200.0, 210.0, 220.0])
    );
    assert_eq!(
        *events.lock().unwrap(),
        [
            "quote",
            "deal:B:2024-01-02 09:30:00:2024-01-02 09:32:00:0",
            "volume:C:2024-01-02 09:30:00:2024-01-02 09:32:00"
        ]
    );
    assert_eq!(shared.read().unwrap().stock_id(), "C");

    let same_key = Arc::new(RwLock::new(order("A", 2, 2, 2, OrderDir::Buy)));
    let cached = loader
        .load_shared(&same_key, &IdxTradeRange::new(0, 0))
        .unwrap();
    assert!(Arc::ptr_eq(&cached, &data));
    let source_order = cached_handles_source_order(&cached);
    assert!(Arc::ptr_eq(&source_order, &shared));
    assert!(!Arc::ptr_eq(&source_order, &same_key));
    *handles.deal_prices.write().unwrap() = arr1(&[99.0]);
    let cached_handles = cached.read().unwrap().clone();
    assert_eq!(*cached_handles.deal_prices.read().unwrap(), arr1(&[99.0]));
    assert_eq!(events.lock().unwrap().len(), 3);
}

fn cached_handles_source_order(data: &domain_core::SharedSaoeBacktestData) -> Arc<RwLock<Order>> {
    data.read()
        .unwrap()
        .source_order
        .as_ref()
        .map(Arc::clone)
        .unwrap()
}

fn poison_cached_field<T: Send + Sync + 'static>(field: Arc<RwLock<T>>) {
    assert!(
        std::thread::spawn(move || {
            let _guard = field.write().unwrap();
            panic!("poison cached backtest field");
        })
        .join()
        .is_err()
    );
}

#[test]
fn owned_cache_reads_observe_shared_mutation_and_report_every_poisoned_identity() {
    let range = TradeRangeByTime::parse("09:30", "09:31").unwrap();
    let make_loader = || {
        loader(
            vec![time(2, 1), time(2, 2)],
            Arc::new(Mutex::new(Vec::new())),
            None,
        )
    };
    let shared_order = Arc::new(RwLock::new(order("A", 2, 1, 2, OrderDir::Buy)));
    let mut mutated_loader = make_loader();
    let shared = mutated_loader.load_shared(&shared_order, &range).unwrap();
    let deal_prices = Arc::clone(&shared.read().unwrap().deal_prices);
    *deal_prices.write().unwrap() = arr1(&[77.0]);
    assert_eq!(
        mutated_loader
            .load(&order("A", 2, 2, 2, OrderDir::Buy), &range)
            .unwrap()
            .deal_prices,
        arr1(&[77.0])
    );

    let mut parent_loader = make_loader();
    let parent = parent_loader.load_shared(&shared_order, &range).unwrap();
    poison_cached_field(Arc::clone(&parent));
    assert!(matches!(
        parent_loader.load(&order("A", 2, 2, 2, OrderDir::Buy), &range),
        Err(SaoeBacktestDataLoadError::CachedDataPoisoned)
    ));

    for (name, select) in [
        ("ticks_index", 0_u8),
        ("ticks_for_order", 1),
        ("deal_prices", 2),
        ("market_volumes", 3),
        ("features", 4),
    ] {
        let mut field_loader = make_loader();
        let parent = field_loader.load_shared(&shared_order, &range).unwrap();
        let handles = parent.read().unwrap().clone();
        match select {
            0 => poison_cached_field(handles.ticks_index),
            1 => poison_cached_field(handles.ticks_for_order),
            2 => poison_cached_field(handles.deal_prices),
            3 => poison_cached_field(handles.market_volumes),
            4 => poison_cached_field(handles.features),
            _ => unreachable!(),
        }
        assert!(matches!(
            field_loader.load(&order("A", 2, 2, 2, OrderDir::Buy), &range),
            Err(SaoeBacktestDataLoadError::CachedFieldPoisoned(field)) if field == name
        ));
    }
}

#[test]
fn shared_loader_reports_order_poison_at_the_first_reached_reread() {
    for (stage, expected) in [
        ("quote", vec!["quote"]),
        (
            "deal",
            vec!["quote", "deal:B:2024-01-02 09:30:00:2024-01-02 09:32:00:0"],
        ),
    ] {
        let shared = Arc::new(RwLock::new(order("A", 2, 0, 3, OrderDir::Buy)));
        let events = Arc::new(Mutex::new(Vec::new()));
        let source = Arc::new(MutatingSharedSource {
            order: Arc::clone(&shared),
            events: Arc::clone(&events),
            poison_at: Some(stage),
            clear_start_on_quote: false,
        });
        let mut loader = SaoeBacktestDataLoader::new(source);
        let range = TradeRangeByTime::parse("09:30", "09:32").unwrap();
        assert!(matches!(
            loader.load_shared(&shared, &range),
            Err(SaoeBacktestDataLoadError::OrderPoisoned)
        ));
        assert_eq!(*events.lock().unwrap(), expected);
        assert_eq!(loader.cache_len(), 0);
    }
}

#[test]
fn shared_loader_preserves_all_pre_market_failure_boundaries() {
    let range = TradeRangeByTime::parse("09:30", "09:32").unwrap();

    let poisoned = Arc::new(RwLock::new(order("A", 2, 0, 3, OrderDir::Buy)));
    let poison_target = Arc::clone(&poisoned);
    assert!(
        std::thread::spawn(move || {
            let _guard = poison_target.write().unwrap();
            panic!("poison initial shared order");
        })
        .join()
        .is_err()
    );
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut poisoned_loader = loader(vec![time(2, 1)], Arc::clone(&events), None);
    assert!(matches!(
        poisoned_loader.load_shared(&poisoned, &range),
        Err(SaoeBacktestDataLoadError::OrderPoisoned)
    ));
    assert!(events.lock().unwrap().is_empty());

    let missing_start = Arc::new(RwLock::new(Order::new(
        "A",
        1.0,
        OrderDir::Buy,
        None,
        Some(time(2, 3)),
    )));
    let mut missing_start_loader = loader(vec![time(2, 1)], Arc::clone(&events), None);
    assert!(matches!(
        missing_start_loader.load_shared(&missing_start, &range),
        Err(SaoeBacktestDataLoadError::Order(_))
    ));
    assert!(events.lock().unwrap().is_empty());

    let changed = Arc::new(RwLock::new(order("A", 2, 0, 3, OrderDir::Buy)));
    let changed_events = Arc::new(Mutex::new(Vec::new()));
    let changed_source = Arc::new(MutatingSharedSource {
        order: Arc::clone(&changed),
        events: Arc::clone(&changed_events),
        poison_at: None,
        clear_start_on_quote: true,
    });
    let mut changed_loader = SaoeBacktestDataLoader::new(changed_source);
    assert!(matches!(
        changed_loader.load_shared(&changed, &range),
        Err(SaoeBacktestDataLoadError::Order(_))
    ));
    assert_eq!(*changed_events.lock().unwrap(), ["quote"]);

    for (candidate, candidate_range, expected) in [
        (
            Arc::new(RwLock::new(Order::new(
                "A",
                1.0,
                OrderDir::Buy,
                Some(time(2, 0)),
                None,
            ))),
            &range as &dyn domain_core::TradeRange,
            "missing-end",
        ),
        (
            Arc::new(RwLock::new(order("A", 3, 0, 2, OrderDir::Buy))),
            &range,
            "empty-interval",
        ),
        (
            Arc::new(RwLock::new(order("A", 2, 0, 2, OrderDir::Buy))),
            &IdxTradeRange::new(0, 1),
            "unsupported",
        ),
        (
            Arc::new(RwLock::new(order("A", 2, 0, 2, OrderDir::Buy))),
            &TradeRangeByTime::parse("10:00", "10:01").unwrap(),
            "empty-range",
        ),
    ] {
        let case_events = Arc::new(Mutex::new(Vec::new()));
        let mut case_loader = loader(vec![time(2, 1)], case_events, None);
        let error = case_loader
            .load_shared(&candidate, candidate_range)
            .unwrap_err();
        assert!(match expected {
            "missing-end" => matches!(error, SaoeBacktestDataLoadError::MissingEndTime),
            "empty-interval" => matches!(error, SaoeBacktestDataLoadError::EmptyOrderInterval),
            "unsupported" => matches!(error, SaoeBacktestDataLoadError::UnsupportedTradeRange),
            "empty-range" => matches!(error, SaoeBacktestDataLoadError::EmptyTradeRange),
            _ => unreachable!(),
        });
    }
}

#[test]
fn shared_loader_stops_at_each_market_source_failure() {
    let range = TradeRangeByTime::parse("09:30", "09:32").unwrap();
    for (failure, expected) in [
        ("quote", vec!["quote"]),
        (
            "deal",
            vec!["quote", "deal:A:2024-01-02 09:30:00:2024-01-02 09:30:00:1"],
        ),
        (
            "volume",
            vec![
                "quote",
                "deal:A:2024-01-02 09:30:00:2024-01-02 09:30:00:1",
                "volume:A:2024-01-02 09:30:00:2024-01-02 09:30:00",
            ],
        ),
    ] {
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut failing = loader(vec![time(2, 1)], Arc::clone(&events), Some(failure));
        let shared = Arc::new(RwLock::new(order("A", 2, 0, 2, OrderDir::Buy)));
        assert!(matches!(
            failing.load_shared(&shared, &range),
            Err(SaoeBacktestDataLoadError::Source(_))
        ));
        assert_eq!(*events.lock().unwrap(), expected);
        assert_eq!(failing.cache_len(), 0);
    }
}
