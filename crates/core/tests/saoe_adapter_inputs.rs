use std::sync::{Arc, Mutex};

use chrono::{NaiveDateTime, TimeDelta};
use domain_core::{
    ConfiguredSaoeAdapterInputsProvider, ConfiguredSaoeStateAdapterFactory, NestedCalendar,
    NestedDecisionUpdate, NestedOuterDecision, NestedOuterDecisionError, Order, OrderDecision,
    OrderDir, OrderTradeDecision, SaoeAdapterContext, SaoeAdapterInputsProvider, SaoeAdapterMarket,
    SaoeAdapterRuntime, SaoeBacktestDataLoader, SaoeBacktestDataSource, SaoeMarketSlice,
    SaoePluginError, SaoeStateAdapterFactory, SharedTradeRange, TradeRangeByTime,
};
use ndarray::arr1;

fn time(minute: i64) -> NaiveDateTime {
    NaiveDateTime::parse_from_str("2024-01-02 09:30:00", "%Y-%m-%d %H:%M:%S").unwrap()
        + TimeDelta::minutes(minute)
}

fn order() -> Order {
    Order::new("A", 10.0, OrderDir::Buy, Some(time(0)), Some(time(2)))
}

struct Outer {
    decision: OrderTradeDecision,
}

impl Outer {
    fn new(range: Option<SharedTradeRange>) -> Self {
        Self {
            decision: OrderTradeDecision::from_orders(vec![order()], time(0), time(2), range),
        }
    }
}

impl NestedOuterDecision for Outer {
    fn order_decision(&self) -> &dyn OrderDecision {
        &self.decision
    }

    fn order_decision_mut(&mut self) -> &mut dyn OrderDecision {
        &mut self.decision
    }

    fn update(
        &mut self,
        _calendar: &dyn NestedCalendar,
    ) -> Result<NestedDecisionUpdate, NestedOuterDecisionError> {
        Ok(NestedDecisionUpdate::Unchanged)
    }

    fn is_empty(&self) -> Result<bool, NestedOuterDecisionError> {
        Ok(self.decision.is_empty())
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

struct Source {
    events: Arc<Mutex<Vec<String>>>,
    fail: bool,
}

impl SaoeBacktestDataSource for Source {
    fn quote_timestamps(&self) -> Result<Vec<NaiveDateTime>, SaoePluginError> {
        self.events.lock().unwrap().push("quote".to_owned());
        if self.fail {
            return Err(error("quote"));
        }
        Ok(vec![time(0), time(1), time(2)])
    }

    fn deal_prices(
        &self,
        _stock_id: &str,
        _start: NaiveDateTime,
        _end: NaiveDateTime,
        _direction: OrderDir,
    ) -> Result<ndarray::Array1<f64>, SaoePluginError> {
        self.events.lock().unwrap().push("deal".to_owned());
        Ok(arr1(&[10.0, 11.0, 12.0]))
    }

    fn market_volumes(
        &self,
        _stock_id: &str,
        _start: NaiveDateTime,
        _end: NaiveDateTime,
    ) -> Result<ndarray::Array1<f64>, SaoePluginError> {
        self.events.lock().unwrap().push("volume".to_owned());
        Ok(arr1(&[100.0, 110.0, 120.0]))
    }
}

struct Market;

impl SaoeAdapterMarket for Market {
    fn market_slice(
        &self,
        _stock_id: &str,
        _start: NaiveDateTime,
        _end: NaiveDateTime,
        _direction: OrderDir,
    ) -> Result<SaoeMarketSlice, SaoePluginError> {
        unreachable!("adapter construction and state snapshots do not read market slices")
    }
}

struct Context;

impl SaoeAdapterContext for Context {
    fn current_trade_step(&self) -> Result<i64, SaoePluginError> {
        Ok(7)
    }

    fn latest_price_advantage(&self) -> Result<f64, SaoePluginError> {
        unreachable!("adapter construction and state snapshots do not read indicators")
    }

    fn warn_overfill(&self, _execution_volume: f64, _position: f64) -> Result<(), SaoePluginError> {
        unreachable!("adapter construction and state snapshots do not warn")
    }
}

struct Runtime {
    events: Arc<Mutex<Vec<String>>>,
    fail: Option<&'static str>,
}

impl SaoeAdapterRuntime for Runtime {
    fn ticks_per_step(&self) -> Result<usize, SaoePluginError> {
        self.events.lock().unwrap().push("frequency".to_owned());
        if self.fail == Some("frequency") {
            return Err(error("frequency"));
        }
        Ok(2)
    }

    fn start_step(&self, outer: &dyn NestedOuterDecision) -> Result<i64, SaoePluginError> {
        self.events
            .lock()
            .unwrap()
            .push(format!("start:{}", outer.order_decision().orders().len()));
        if self.fail == Some("start") {
            return Err(error("start"));
        }
        Ok(1)
    }
}

fn error(message: &str) -> SaoePluginError {
    SaoePluginError {
        message: message.to_owned(),
    }
}

fn outer() -> Outer {
    Outer::new(Some(Arc::new(
        TradeRangeByTime::parse("09:30", "09:32").unwrap(),
    )))
}

fn provider(
    source_events: Arc<Mutex<Vec<String>>>,
    runtime_events: Arc<Mutex<Vec<String>>>,
    source_fail: bool,
    runtime_fail: Option<&'static str>,
) -> ConfiguredSaoeAdapterInputsProvider {
    ConfiguredSaoeAdapterInputsProvider::new(
        SaoeBacktestDataLoader::new(Arc::new(Source {
            events: source_events,
            fail: source_fail,
        })),
        Arc::new(Market),
        Arc::new(Context),
        Arc::new(Runtime {
            events: runtime_events,
            fail: runtime_fail,
        }),
        1,
    )
}

#[test]
#[allow(clippy::float_cmp)]
fn configured_inputs_load_cache_and_initialize_a_concrete_adapter_in_python_order() {
    let source_events = Arc::new(Mutex::new(Vec::new()));
    let runtime_events = Arc::new(Mutex::new(Vec::new()));
    let mut provider = provider(
        Arc::clone(&source_events),
        Arc::clone(&runtime_events),
        false,
        None,
    );
    let outer = outer();
    let first = provider.load(&order(), &outer).unwrap();
    assert_eq!(first.config.ticks_per_step, 2);
    assert_eq!(first.config.data_granularity, 1);
    assert_eq!(first.config.start_step, 1);
    assert_eq!(first.config.deal_prices, arr1(&[10.0, 11.0, 12.0]));
    assert_eq!(
        first.config.backtest_data.deal_prices,
        first.config.deal_prices
    );
    assert_eq!(provider.backtest_cache_len(), 1);
    let mut first_prices = first.config.backtest_data.deal_prices;
    first_prices[0] = 99.0;

    let second = provider.load(&order(), &outer).unwrap();
    assert_eq!(second.config.backtest_data.deal_prices[0], 10.0);
    assert_eq!(
        *source_events.lock().unwrap(),
        ["quote".to_owned(), "deal".to_owned(), "volume".to_owned()]
    );
    assert_eq!(
        *runtime_events.lock().unwrap(),
        [
            "frequency".to_owned(),
            "start:1".to_owned(),
            "frequency".to_owned(),
            "start:1".to_owned()
        ]
    );

    let mut factory = ConfiguredSaoeStateAdapterFactory::new(Box::new(provider));
    let adapter = factory.create(&order(), &outer).unwrap();
    let state = adapter.state(&order()).unwrap();
    assert_eq!(state.parts().cur_step, 6);
    assert_eq!(state.parts().ticks_per_step, 2);
    assert_eq!(
        state.parts().backtest_data.market_volumes,
        arr1(&[100.0, 110.0, 120.0])
    );
}

#[test]
fn configured_inputs_stop_at_missing_range_loader_and_runtime_failures() {
    let source_events = Arc::new(Mutex::new(Vec::new()));
    let runtime_events = Arc::new(Mutex::new(Vec::new()));
    let mut missing = provider(
        Arc::clone(&source_events),
        Arc::clone(&runtime_events),
        false,
        None,
    );
    assert_eq!(
        missing
            .load(&order(), &Outer::new(None))
            .err()
            .unwrap()
            .message,
        "SAOE adapter construction requires an outer trade range"
    );
    assert!(source_events.lock().unwrap().is_empty());
    assert!(runtime_events.lock().unwrap().is_empty());
    assert_eq!(missing.backtest_cache_len(), 0);

    let mut source_failure = provider(
        Arc::clone(&source_events),
        Arc::clone(&runtime_events),
        true,
        None,
    );
    assert!(
        source_failure
            .load(&order(), &outer())
            .err()
            .unwrap()
            .message
            .contains("quote")
    );
    assert_eq!(*source_events.lock().unwrap(), ["quote".to_owned()]);
    assert!(runtime_events.lock().unwrap().is_empty());
    assert_eq!(source_failure.backtest_cache_len(), 0);

    for (failure, expected_runtime) in [
        ("frequency", vec!["frequency"]),
        ("start", vec!["frequency", "start:1"]),
    ] {
        let stage_source = Arc::new(Mutex::new(Vec::new()));
        let stage_runtime = Arc::new(Mutex::new(Vec::new()));
        let mut failing = provider(
            Arc::clone(&stage_source),
            Arc::clone(&stage_runtime),
            false,
            Some(failure),
        );
        assert_eq!(
            failing.load(&order(), &outer()).err().unwrap().message,
            failure
        );
        assert_eq!(
            *stage_source.lock().unwrap(),
            ["quote".to_owned(), "deal".to_owned(), "volume".to_owned()]
        );
        assert_eq!(*stage_runtime.lock().unwrap(), expected_runtime);
        assert_eq!(failing.backtest_cache_len(), 1);
    }
}
