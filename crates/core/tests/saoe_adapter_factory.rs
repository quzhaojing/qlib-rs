use std::{
    collections::VecDeque,
    process::Command,
    sync::{Arc, Mutex},
};

use arrow_array::RecordBatch;
use arrow_schema::Schema;
use chrono::{NaiveDateTime, TimeDelta};
use domain_core::{
    ConfiguredSaoeStateAdapterFactory, NestedCalendar, NestedDecisionUpdate, NestedOuterDecision,
    NestedOuterDecisionError, Order, OrderDecision, OrderDir, SaoeAdapterConfig,
    SaoeAdapterContext, SaoeAdapterInputs, SaoeAdapterInputsProvider, SaoeAdapterMarket,
    SaoeBacktestData, SaoeMarketSlice, SaoePluginError, SaoeStateAdapterFactory,
};
use ndarray::arr1;
use serde_json::{Value, json};

fn time(minute: i64) -> NaiveDateTime {
    NaiveDateTime::parse_from_str("2024-01-02 09:30:00", "%Y-%m-%d %H:%M:%S").unwrap()
        + TimeDelta::minutes(minute)
}

fn order(stock: &str, amount: f64) -> Order {
    Order::new(stock, amount, OrderDir::Buy, Some(time(0)), Some(time(3)))
}

fn config(start_step: i64) -> SaoeAdapterConfig {
    let ticks: Vec<_> = (0..4).map(time).collect();
    SaoeAdapterConfig {
        backtest_data: SaoeBacktestData {
            ticks_index: ticks.clone(),
            ticks_for_order: ticks,
            deal_prices: arr1(&[10.0, 11.0, 12.0, 13.0]),
            market_volumes: arr1(&[100.0, 110.0, 120.0, 130.0]),
            features: RecordBatch::new_empty(Arc::new(Schema::empty())),
        },
        deal_prices: arr1(&[10.0, 11.0, 12.0, 13.0]),
        ticks_per_step: 2,
        data_granularity: 1,
        start_step,
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
        unreachable!("state snapshots do not read the market")
    }
}

struct Context(i64);

impl SaoeAdapterContext for Context {
    fn current_trade_step(&self) -> Result<i64, SaoePluginError> {
        Ok(self.0)
    }

    fn latest_price_advantage(&self) -> Result<f64, SaoePluginError> {
        unreachable!("state snapshots do not read indicators")
    }

    fn warn_overfill(&self, _execution_volume: f64, _position: f64) -> Result<(), SaoePluginError> {
        unreachable!("state snapshots do not warn")
    }
}

struct Inputs {
    events: Arc<Mutex<Vec<String>>>,
    values: VecDeque<Result<SaoeAdapterInputs, SaoePluginError>>,
}

impl SaoeAdapterInputsProvider for Inputs {
    fn load(
        &mut self,
        order: &Order,
        outer: &dyn NestedOuterDecision,
    ) -> Result<SaoeAdapterInputs, SaoePluginError> {
        self.events.lock().unwrap().push(format!(
            "load:{}:{}",
            order.stock_id(),
            outer.order_decision().orders().len()
        ));
        self.values
            .pop_front()
            .expect("test supplies one input result per factory call")
    }
}

struct Outer {
    decision: Decision,
}

struct Decision(Vec<Order>);

impl OrderDecision for Decision {
    fn orders(&self) -> &[Order] {
        &self.0
    }

    fn orders_mut(&mut self) -> &mut [Order] {
        &mut self.0
    }

    fn start_time(&self) -> NaiveDateTime {
        time(0)
    }

    fn end_time(&self) -> NaiveDateTime {
        time(3)
    }

    fn trade_range(&self) -> Option<&dyn domain_core::TradeRange> {
        None
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

fn inputs(config: SaoeAdapterConfig, step: i64) -> SaoeAdapterInputs {
    SaoeAdapterInputs {
        market: Arc::new(Market),
        context: Arc::new(Context(step)),
        config,
    }
}

fn factory(
    events: Arc<Mutex<Vec<String>>>,
    values: impl IntoIterator<Item = Result<SaoeAdapterInputs, SaoePluginError>>,
) -> ConfiguredSaoeStateAdapterFactory {
    ConfiguredSaoeStateAdapterFactory::new(Box::new(Inputs {
        events,
        values: values.into_iter().collect(),
    }))
}

fn python_contract() -> Value {
    let output = Command::new("python")
        .args([
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/saoe_adapter_factory_contract.py"
            ),
            r"D:\code\github\qlib\qlib\rl\order_execution\strategy.py",
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
#[allow(clippy::float_cmp)]
fn configured_factory_loads_and_initializes_fresh_adapters_per_order() {
    assert_eq!(
        python_contract(),
        json!({
            "events": [
                ["load", "A", "exchange", "range"],
                ["frequency", "30min"],
                ["start", 4],
                ["adapter", "A", "decision", "executor", "exchange", 30, "backtest", 5]
            ],
            "type": "Adapter"
        })
    );

    let events = Arc::new(Mutex::new(Vec::new()));
    let orders = vec![order("A", 10.0), order("B", 20.0)];
    let outer = Outer {
        decision: Decision(orders.clone()),
    };
    let mut factory = factory(
        Arc::clone(&events),
        [Ok(inputs(config(1), 5)), Ok(inputs(config(2), 8))],
    );

    let first = factory.create(&orders[0], &outer).unwrap();
    let second = factory.create(&orders[1], &outer).unwrap();
    let first_state = first.state(&orders[0]).unwrap();
    let second_state = second.state(&orders[1]).unwrap();

    assert_eq!(first_state.parts().position, 10.0);
    assert_eq!(first_state.parts().cur_step, 4);
    assert_eq!(second_state.parts().position, 20.0);
    assert_eq!(second_state.parts().cur_step, 6);
    assert_eq!(
        *events.lock().unwrap(),
        ["load:A:2".to_owned(), "load:B:2".to_owned()]
    );
}

#[test]
fn configured_factory_maps_provider_constructor_and_reset_failures() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let valid = order("A", 10.0);
    let outer = Outer {
        decision: Decision(vec![valid.clone()]),
    };
    let mut provider_failure = factory(
        Arc::clone(&events),
        [Err(SaoePluginError {
            message: "provider".to_owned(),
        })],
    );
    assert_eq!(
        provider_failure
            .create(&valid, &outer)
            .err()
            .unwrap()
            .message,
        "provider"
    );

    let mut invalid = Vec::new();
    let mut empty_ticks = config(0);
    empty_ticks.backtest_data.ticks_index.clear();
    invalid.push((empty_ticks, "SAOE backtest ticks are empty"));
    let mut empty_order_ticks = config(0);
    empty_order_ticks.backtest_data.ticks_for_order.clear();
    invalid.push((empty_order_ticks, "SAOE order ticks are empty"));
    let mut zero = config(0);
    zero.data_granularity = 0;
    invalid.push((zero, "SAOE data granularity must be positive"));
    let mut incompatible = config(0);
    incompatible.data_granularity = 3;
    invalid.push((
        incompatible,
        "ticks_per_step 2 is not divisible by data granularity 3",
    ));
    let mut huge = config(0);
    huge.ticks_per_step = usize::MAX;
    huge.data_granularity = usize::MAX;
    invalid.push((huge, "SAOE data granularity exceeds Chrono's minute range"));

    for (config, expected) in invalid {
        let mut factory = factory(Arc::clone(&events), [Ok(inputs(config, 0))]);
        assert_eq!(
            factory.create(&valid, &outer).err().unwrap().message,
            expected
        );
    }

    for (candidate, expected) in [
        (
            Order::new("A", 10.0, OrderDir::Buy, None, Some(time(3))),
            "order start time is required for its day key",
        ),
        (
            Order::new("A", 10.0, OrderDir::Buy, Some(time(0)), None),
            "order end time is required by the SAOE adapter",
        ),
    ] {
        let mut factory = factory(Arc::clone(&events), [Ok(inputs(config(0), 0))]);
        assert_eq!(
            factory.create(&candidate, &outer).err().unwrap().message,
            expected
        );
    }
}
