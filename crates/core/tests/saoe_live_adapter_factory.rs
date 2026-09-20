use std::sync::{Arc, Mutex, RwLock};

use arrow_array::RecordBatch;
use arrow_schema::Schema;
use chrono::{NaiveDate, NaiveDateTime, TimeDelta};
use domain_core::decision_construction::{
    ConstructedDecisionBase, DecisionOrderItem, DecisionTotalStep, SharedDecisionOrders,
    SharedOrderDecisionConstruction,
};
use domain_core::decision_update::{
    LiveDecisionHandle, SharedDecisionUpdateStrategy, SharedLiveDecision,
};
use domain_core::saoe_live_registry::LiveSaoeAdapterFactory;
use domain_core::{
    ConcreteSaoeStateAdapter, ConfiguredLiveSaoeAdapterFactory, ExecutionCalendar,
    ExecutionCalendarContext, ExecutionCalendarError, ExecutionCalendarProvider,
    LiveSaoeAdapterRuntime, LiveSaoeBacktestData, Order, OrderDir, SaoeAdapterContext,
    SaoeAdapterError, SaoeBacktestData, SaoeBacktestDataLoader, SaoeBacktestDataSource,
    SaoePluginError, SharedExecutionCalendar, SharedSaoeBacktestData, SharedTradeRange,
    TradeRangeByTime,
};
use ndarray::{Array1, arr1};

fn time(minute: i64) -> NaiveDateTime {
    NaiveDate::from_ymd_opt(2024, 1, 2)
        .unwrap()
        .and_hms_opt(9, 29, 0)
        .unwrap()
        + TimeDelta::minutes(minute)
}

fn plugin_error(message: &str) -> SaoePluginError {
    SaoePluginError {
        message: message.to_owned(),
    }
}

struct Origin;

impl SharedDecisionUpdateStrategy<()> for Origin {
    fn update_trade_decision(
        &self,
        _: &SharedLiveDecision<Self, ()>,
        _: &dyn domain_core::DecisionUpdateCalendar,
    ) -> Result<Option<SharedLiveDecision<Self, ()>>, domain_core::DecisionUpdateStrategyError>
    {
        Ok(None)
    }
}

fn decision(order: &Arc<RwLock<Order>>, range: &SharedTradeRange) -> LiveDecisionHandle {
    decision_with_range(order, Some(Arc::clone(range)))
}

fn decision_with_range(
    order: &Arc<RwLock<Order>>,
    range: Option<SharedTradeRange>,
) -> LiveDecisionHandle {
    let orders: SharedDecisionOrders = Arc::new(RwLock::new(vec![DecisionOrderItem::Order(
        Arc::clone(order),
    )]));
    Arc::new(RwLock::new(SharedOrderDecisionConstruction {
        strategy: Arc::new(Origin),
        base: Some(ConstructedDecisionBase {
            start_time: time(1),
            end_time: time(3),
            trade_range: range,
        }),
        total_step: DecisionTotalStep::Value(3),
        orders: Some(orders),
        details: Some(()),
    }))
}

struct CalendarProvider;

impl ExecutionCalendarProvider for CalendarProvider {
    fn calendar(&self, _: &str, _: bool) -> Result<Arc<[NaiveDateTime]>, ExecutionCalendarError> {
        Ok(vec![time(1), time(2), time(3)].into())
    }

    fn locate_index(
        &self,
        _: Option<NaiveDateTime>,
        _: Option<NaiveDateTime>,
        _: &str,
        _: bool,
    ) -> Result<(i64, i64), ExecutionCalendarError> {
        Ok((0, 2))
    }
}

struct CalendarContext;

impl ExecutionCalendarContext for CalendarContext {
    fn data_frequency(&self) -> Result<String, ExecutionCalendarError> {
        Ok("1min".to_owned())
    }
}

fn runtime_calendar(frequency: &str) -> SharedExecutionCalendar {
    SharedExecutionCalendar::new(
        Arc::new(Mutex::new(
            ExecutionCalendar::new(
                Arc::new(CalendarProvider),
                frequency.to_owned(),
                Some(time(1)),
                Some(time(3)),
            )
            .unwrap(),
        )),
        Arc::new(CalendarContext),
    )
}

fn poisoned_runtime_calendar() -> SharedExecutionCalendar {
    let cursor = Arc::new(Mutex::new(
        ExecutionCalendar::new(
            Arc::new(CalendarProvider),
            "30min".to_owned(),
            Some(time(1)),
            Some(time(3)),
        )
        .unwrap(),
    ));
    let poison_target = Arc::clone(&cursor);
    assert!(
        std::thread::spawn(move || {
            let _guard = poison_target.lock().unwrap();
            panic!("poison live runtime calendar");
        })
        .join()
        .is_err()
    );
    SharedExecutionCalendar::new(cursor, Arc::new(CalendarContext))
}

struct Source {
    events: Arc<Mutex<Vec<&'static str>>>,
    fail: Option<&'static str>,
}

impl Source {
    fn reached<T>(&self, stage: &'static str, value: T) -> Result<T, SaoePluginError> {
        self.events.lock().unwrap().push(stage);
        if self.fail == Some(stage) {
            return Err(plugin_error(stage));
        }
        Ok(value)
    }
}

impl SaoeBacktestDataSource for Source {
    fn quote_timestamps(&self) -> Result<Vec<NaiveDateTime>, SaoePluginError> {
        self.reached("quote", vec![time(1), time(2), time(3)])
    }

    fn deal_prices(
        &self,
        _: &str,
        _: NaiveDateTime,
        _: NaiveDateTime,
        _: OrderDir,
    ) -> Result<Array1<f64>, SaoePluginError> {
        self.reached("deal", arr1(&[10.0, 11.0, 12.0]))
    }

    fn market_volumes(
        &self,
        _: &str,
        _: NaiveDateTime,
        _: NaiveDateTime,
    ) -> Result<Array1<f64>, SaoePluginError> {
        self.reached("volume", arr1(&[100.0, 110.0, 120.0]))
    }
}

struct Context;

impl SaoeAdapterContext for Context {
    fn current_trade_step(&self) -> Result<i64, SaoePluginError> {
        Ok(7)
    }

    fn latest_price_advantage(&self) -> Result<f64, SaoePluginError> {
        Ok(0.0)
    }

    fn warn_overfill(&self, _: f64, _: f64) -> Result<(), SaoePluginError> {
        Ok(())
    }
}

struct Runtime {
    events: Arc<Mutex<Vec<&'static str>>>,
    order: Arc<RwLock<Order>>,
    fail: Option<&'static str>,
}

impl LiveSaoeAdapterRuntime for Runtime {
    fn ticks_per_step(&self) -> Result<usize, SaoePluginError> {
        self.events.lock().unwrap().push("frequency");
        if self.fail == Some("frequency") {
            return Err(plugin_error("frequency"));
        }
        Ok(2)
    }

    fn start_step(&self, _: &LiveDecisionHandle) -> Result<i64, SaoePluginError> {
        self.events.lock().unwrap().push("start");
        if self.fail == Some("start") {
            return Err(plugin_error("start"));
        }
        *self.order.write().unwrap() =
            Order::new("A", 20.0, OrderDir::Sell, Some(time(2)), Some(time(3)));
        Ok(1)
    }
}

fn factory(
    events: Arc<Mutex<Vec<&'static str>>>,
    order: Arc<RwLock<Order>>,
    source_fail: Option<&'static str>,
    runtime_fail: Option<&'static str>,
) -> ConfiguredLiveSaoeAdapterFactory {
    ConfiguredLiveSaoeAdapterFactory::new(
        SaoeBacktestDataLoader::new(Arc::new(Source {
            events: Arc::clone(&events),
            fail: source_fail,
        })),
        Arc::new(Context),
        Arc::new(Runtime {
            events,
            order,
            fail: runtime_fail,
        }),
        1,
    )
}

fn cached_data() -> SharedSaoeBacktestData {
    LiveSaoeBacktestData::from_owned(SaoeBacktestData {
        ticks_index: vec![time(1), time(2), time(3)],
        ticks_for_order: vec![time(1), time(2), time(3)],
        deal_prices: arr1(&[10.0, 11.0, 12.0]),
        market_volumes: arr1(&[100.0, 110.0, 120.0]),
        features: RecordBatch::new_empty(Arc::new(Schema::empty())),
    })
    .into_shared()
}

fn poison_field<T: Send + Sync + 'static>(field: Arc<RwLock<T>>) {
    assert!(
        std::thread::spawn(move || {
            let _guard = field.write().unwrap();
            panic!("poison cached constructor field");
        })
        .join()
        .is_err()
    );
}

fn cached_constructor(
    data: SharedSaoeBacktestData,
    order: &Arc<RwLock<Order>>,
    ticks_per_step: usize,
    granularity: usize,
    start: &mut dyn FnMut() -> Result<i64, SaoePluginError>,
) -> Result<ConcreteSaoeStateAdapter, SaoeAdapterError> {
    ConcreteSaoeStateAdapter::new_live_cached(
        Arc::new(Source {
            events: Arc::new(Mutex::new(Vec::new())),
            fail: None,
        }),
        Arc::new(Context),
        data,
        ticks_per_step,
        granularity,
        order,
        start,
    )
}

#[test]
fn configured_live_factory_preserves_load_frequency_and_constructor_order() {
    let order = Arc::new(RwLock::new(Order::new(
        "A",
        10.0,
        OrderDir::Buy,
        Some(time(1)),
        Some(time(3)),
    )));
    let range: SharedTradeRange = Arc::new(TradeRangeByTime::parse("09:30", "09:32").unwrap());
    let outer = decision(&order, &range);
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut factory = factory(Arc::clone(&events), Arc::clone(&order), None, None);

    let adapter = factory.create(&order, &outer, &range).unwrap();
    assert_eq!(
        *events.lock().unwrap(),
        ["quote", "deal", "volume", "frequency", "start"]
    );
    assert_eq!(factory.backtest_cache_len(), 1);
    let live = adapter.live_state().unwrap();
    assert!(Arc::ptr_eq(live.order(), &order));
    let cached_backtest_data = Arc::clone(live.backtest_data());
    let state = adapter.state().unwrap();
    assert_eq!(state.parts().position.to_bits(), 10.0_f64.to_bits());
    assert_eq!(state.parts().order.amount().to_bits(), 20.0_f64.to_bits());
    assert_eq!(state.parts().order.direction(), OrderDir::Sell);
    assert_eq!(state.parts().cur_time, time(2));
    assert_eq!(state.parts().cur_step, 6);
    assert_eq!(state.parts().ticks_per_step, 2);
    assert_eq!(
        state.parts().backtest_data.deal_prices,
        arr1(&[10.0, 11.0, 12.0])
    );

    let deal_prices = Arc::clone(&cached_backtest_data.read().unwrap().deal_prices);
    *deal_prices.write().unwrap() = arr1(&[99.0, 98.0, 97.0]);
    *order.write().unwrap() = Order::new("A", 10.0, OrderDir::Buy, Some(time(1)), Some(time(3)));
    let cached_adapter = factory.create(&order, &outer, &range).unwrap();
    let cached_live = cached_adapter.live_state().unwrap();
    assert!(Arc::ptr_eq(
        cached_live.backtest_data(),
        &cached_backtest_data
    ));
    assert_eq!(
        cached_adapter
            .state()
            .unwrap()
            .parts()
            .backtest_data
            .deal_prices,
        arr1(&[99.0, 98.0, 97.0])
    );
    assert_eq!(
        *events.lock().unwrap(),
        [
            "quote",
            "deal",
            "volume",
            "frequency",
            "start",
            "frequency",
            "start"
        ]
    );
}

#[test]
fn cached_constructor_reports_identity_field_order_and_validation_failures() {
    let fresh_order = || {
        Arc::new(RwLock::new(Order::new(
            "A",
            10.0,
            OrderDir::Buy,
            Some(time(1)),
            Some(time(3)),
        )))
    };

    let poisoned_order = fresh_order();
    poison_field(Arc::clone(&poisoned_order));
    assert!(matches!(
        cached_constructor(cached_data(), &poisoned_order, 2, 1, &mut || Ok(0)),
        Err(SaoeAdapterError::AdapterOrderPoisoned)
    ));

    let poisoned_parent = cached_data();
    poison_field(Arc::clone(&poisoned_parent));
    assert!(matches!(
        cached_constructor(poisoned_parent, &fresh_order(), 2, 1, &mut || Ok(0)),
        Err(SaoeAdapterError::BacktestDataPoisoned)
    ));

    for (field_name, selected) in [
        ("deal_prices", 0_u8),
        ("ticks_for_order", 1),
        ("ticks_index", 2),
        ("market_volumes", 3),
        ("features", 4),
    ] {
        let data = cached_data();
        let handles = data.read().unwrap().clone();
        match selected {
            0 => poison_field(handles.deal_prices),
            1 => poison_field(handles.ticks_for_order),
            2 => poison_field(handles.ticks_index),
            3 => poison_field(handles.market_volumes),
            4 => poison_field(handles.features),
            _ => unreachable!(),
        }
        assert!(matches!(
            cached_constructor(data, &fresh_order(), 2, 1, &mut || Ok(0)),
            Err(SaoeAdapterError::BacktestDataFieldPoisoned(field)) if field == field_name
        ));
    }

    let empty_ticks = cached_data();
    let empty_handle = Arc::clone(&empty_ticks.read().unwrap().ticks_for_order);
    empty_handle.write().unwrap().clear();
    assert!(matches!(
        cached_constructor(empty_ticks, &fresh_order(), 2, 1, &mut || Ok(0)),
        Err(SaoeAdapterError::EmptyOrderTicks)
    ));

    let post_start_poison = fresh_order();
    let poison_target = Arc::clone(&post_start_poison);
    assert!(matches!(
        cached_constructor(cached_data(), &post_start_poison, 2, 1, &mut move || {
            poison_field(Arc::clone(&poison_target));
            Ok(0)
        }),
        Err(SaoeAdapterError::AdapterOrderPoisoned)
    ));

    let missing_start = fresh_order();
    let missing_target = Arc::clone(&missing_start);
    let result = cached_constructor(cached_data(), &missing_start, 2, 1, &mut move || {
        *missing_target.write().unwrap() =
            Order::new("A", 10.0, OrderDir::Buy, None, Some(time(3)));
        Ok(0)
    });
    assert!(matches!(
        result,
        Err(SaoeAdapterError::Order(
            domain_core::OrderError::MissingStartTime
        ))
    ));

    assert!(matches!(
        cached_constructor(cached_data(), &fresh_order(), 3, 2, &mut || Ok(0)),
        Err(SaoeAdapterError::IncompatibleGranularity {
            ticks_per_step: 3,
            data_granularity: 2
        })
    ));

    let debug = format!("{:?}", cached_data().read().unwrap());
    assert!(debug.contains("LiveSaoeBacktestData"));
    assert!(!debug.contains("<data source>"));
}

#[test]
fn configured_live_factory_stops_at_the_first_loading_or_runtime_failure() {
    for (source_fail, runtime_fail, expected) in [
        (Some("quote"), None, vec!["quote"]),
        (
            None,
            Some("frequency"),
            vec!["quote", "deal", "volume", "frequency"],
        ),
        (
            None,
            Some("start"),
            vec!["quote", "deal", "volume", "frequency", "start"],
        ),
    ] {
        let order = Arc::new(RwLock::new(Order::new(
            "A",
            10.0,
            OrderDir::Buy,
            Some(time(1)),
            Some(time(3)),
        )));
        let range: SharedTradeRange = Arc::new(TradeRangeByTime::parse("09:30", "09:32").unwrap());
        let outer = decision(&order, &range);
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut factory = factory(
            Arc::clone(&events),
            Arc::clone(&order),
            source_fail,
            runtime_fail,
        );
        assert!(factory.create(&order, &outer, &range).is_err());
        assert_eq!(*events.lock().unwrap(), expected);
    }
}

#[test]
fn shared_execution_calendar_supplies_live_frequency_and_start_step() {
    let order = Arc::new(RwLock::new(Order::new(
        "A",
        10.0,
        OrderDir::Buy,
        Some(time(1)),
        Some(time(3)),
    )));
    let range: SharedTradeRange = Arc::new(TradeRangeByTime::parse("09:30", "09:32").unwrap());
    let outer = decision(&order, &range);
    let calendar = runtime_calendar("30min");
    assert_eq!(
        LiveSaoeAdapterRuntime::ticks_per_step(&calendar).unwrap(),
        30
    );
    assert_eq!(
        LiveSaoeAdapterRuntime::start_step(&calendar, &outer).unwrap(),
        0
    );

    let missing = decision_with_range(&order, None);
    assert!(LiveSaoeAdapterRuntime::start_step(&calendar, &missing).is_err());
    assert!(LiveSaoeAdapterRuntime::ticks_per_step(&runtime_calendar("bad-frequency")).is_err());
    assert!(LiveSaoeAdapterRuntime::ticks_per_step(&runtime_calendar("month")).is_err());
    let huge = format!("{}min", "9".repeat(100));
    assert!(LiveSaoeAdapterRuntime::ticks_per_step(&runtime_calendar(&huge)).is_err());
    assert!(LiveSaoeAdapterRuntime::ticks_per_step(&poisoned_runtime_calendar()).is_err());
}
