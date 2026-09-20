use super::*;

struct NoCallbacks;
impl SaoeAdapterMarket for NoCallbacks {
    fn market_slice(
        &self,
        _: &str,
        _: NaiveDateTime,
        _: NaiveDateTime,
        _: OrderDir,
    ) -> Result<SaoeMarketSlice, SaoePluginError> {
        panic!("time advancement must not call the market")
    }
}
impl SaoeAdapterContext for NoCallbacks {
    fn current_trade_step(&self) -> Result<i64, SaoePluginError> {
        panic!("time advancement must not call the runtime step")
    }
    fn latest_price_advantage(&self) -> Result<f64, SaoePluginError> {
        panic!("time advancement must not call the indicator")
    }
    fn warn_overfill(&self, _: f64, _: f64) -> Result<(), SaoePluginError> {
        panic!("time advancement must not emit an overfill warning")
    }
}

fn test_adapter(start: NaiveDateTime, end: NaiveDateTime) -> ConcreteSaoeStateAdapter {
    ConcreteSaoeStateAdapter::new(
        Arc::new(NoCallbacks),
        Arc::new(NoCallbacks),
        SaoeAdapterConfig {
            backtest_data: SaoeBacktestData {
                ticks_index: vec![start, end],
                ticks_for_order: vec![start, end],
                deal_prices: Array1::zeros(2),
                market_volumes: Array1::zeros(2),
                features: RecordBatch::new_empty(Arc::new(Schema::empty())),
            },
            deal_prices: Array1::zeros(2),
            ticks_per_step: 1,
            data_granularity: 1,
            start_step: 0,
        },
    )
    .unwrap()
}

#[test]
fn next_time_reports_order_poison_after_resolving_the_current_tick() {
    let start = NaiveDateTime::parse_from_str("2024-01-02 09:30:00", "%Y-%m-%d %H:%M:%S").unwrap();
    let end = start + TimeDelta::minutes(2);
    let adapter = test_adapter(start, end);
    let order = Arc::new(RwLock::new(Order::new(
        "A",
        10.0,
        OrderDir::Buy,
        Some(start),
        Some(end),
    )));
    assert_eq!(adapter.next_time(start, &order).unwrap(), end);
    assert_eq!(adapter.next_time(end, &order).unwrap(), end);
    *order.write().unwrap() = Order::new(
        "A",
        10.0,
        OrderDir::Buy,
        Some(start),
        Some(end + TimeDelta::minutes(1)),
    );
    assert_eq!(adapter.next_time(start, &order).unwrap(), end);
    *order.write().unwrap() = Order::new("A", 10.0, OrderDir::Buy, Some(start), None);
    assert!(matches!(
        adapter.next_time(start, &order),
        Err(SaoeAdapterError::MissingEndTime)
    ));
    let ticks_index = Arc::clone(&adapter.backtest_data.read().unwrap().ticks_index);
    *ticks_index.write().unwrap() = vec![start, start, end];
    assert!(
        matches!(adapter.next_time(start, &order), Err(SaoeAdapterError::MissingCurrentTime(value)) if value == start)
    );
    *ticks_index.write().unwrap() = vec![start, end];
    let writer = order.clone();
    assert!(
        std::thread::spawn(move || {
            let _guard = writer.try_write().unwrap();
            panic!("deliberate time-advance order poison");
        })
        .join()
        .is_err()
    );
    assert!(matches!(
        adapter.next_time(start, &order),
        Err(SaoeAdapterError::AdapterOrderPoisoned)
    ));
    // Resolve the missing current tick before ever consulting the poisoned order.
    assert!(
        matches!(adapter.next_time(start + TimeDelta::minutes(1), &order),
        Err(SaoeAdapterError::MissingCurrentTime(value)) if value == start + TimeDelta::minutes(1))
    );
    assert!(adapter.session.is_none());
    assert!(adapter.history_exec.read().unwrap().is_empty());
    assert!(adapter.history_steps.read().unwrap().is_empty());
}

#[test]
fn next_time_reports_backtest_object_and_tick_field_poison() {
    let start = NaiveDateTime::parse_from_str("2024-01-02 09:30:00", "%Y-%m-%d %H:%M:%S").unwrap();
    let end = start + TimeDelta::minutes(2);
    let order = Arc::new(RwLock::new(Order::new(
        "A",
        10.0,
        OrderDir::Buy,
        Some(start),
        Some(end),
    )));

    let adapter = test_adapter(start, end);
    let backtest_data = Arc::clone(&adapter.backtest_data);
    assert!(
        std::thread::spawn(move || {
            let _guard = backtest_data.write().unwrap();
            panic!("deliberate backtest object poison");
        })
        .join()
        .is_err()
    );
    assert!(matches!(
        adapter.next_time(start, &order),
        Err(SaoeAdapterError::BacktestDataPoisoned)
    ));

    let adapter = test_adapter(start, end);
    let ticks_index = Arc::clone(&adapter.backtest_data.read().unwrap().ticks_index);
    assert!(
        std::thread::spawn(move || {
            let _guard = ticks_index.write().unwrap();
            panic!("deliberate backtest ticks poison");
        })
        .join()
        .is_err()
    );
    assert!(matches!(
        adapter.next_time(start, &order),
        Err(SaoeAdapterError::BacktestDataFieldPoisoned("ticks_index"))
    ));
}
