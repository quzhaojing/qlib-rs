use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use arrow_array::{ArrayRef, Float64Array, RecordBatch, TimestampNanosecondArray};
use chrono::{NaiveDateTime, TimeDelta};
use domain_core::{
    Account, AccountIndicator, AccountIndicatorError, ConfiguredSaoeAdapterInputsProvider,
    ConfiguredSaoeStateAdapterFactory, ExchangeQuoteProvider, ExchangeSaoeMarket, InfinitePosition,
    NestedAccountIndicatorUpdate, NestedAccountSaoeContext, NestedCalendar, NestedCalendarError,
    NestedDecisionUpdate, NestedOuterDecision, NestedOuterDecisionError, NumpyOrderIndicator,
    Order, OrderDecision, OrderDir, OrderExecution, OrderTradeDecision, Quote, QuoteData,
    QuoteError, QuoteMethod, SaoeAdapterContext, SaoeAdapterMarket, SaoeAdapterRuntime,
    SaoeBacktestDataLoader, SaoeBacktestDataSource, SaoeStateAdapterFactory, SharedSaoeAccount,
    SharedSaoeCalendar, TimeRange, TradeRangeByTime,
};
use indexmap::IndexMap;

fn time(minute: i64) -> NaiveDateTime {
    NaiveDateTime::parse_from_str("2024-01-02 09:30:00", "%Y-%m-%d %H:%M:%S").unwrap()
        + TimeDelta::minutes(minute)
}

#[test]
fn indicator_poison_is_distinct_from_account_poison_and_replacement_restores_context() {
    let (context, _, account) = infrastructure("1min", Ok(1), Some(0.25));
    let indicator = account.lock().unwrap().indicator().clone();
    assert!(
        std::thread::spawn(move || {
            let _guard = indicator.write().unwrap();
            panic!("poison indicator only");
        })
        .join()
        .is_err()
    );
    assert_eq!(
        context.latest_price_advantage().unwrap_err().message,
        "account indicator lock poisoned"
    );
    assert!(account.try_lock().is_ok());
    account
        .lock()
        .unwrap()
        .replace_indicator(Box::new(IndicatorView {
            order: Arc::default(),
            values: Arc::new(std::sync::RwLock::new(IndexMap::from([(
                "pa".to_owned(),
                0.5,
            )]))),
        }));
    assert_eq!(
        context.latest_price_advantage().unwrap().to_bits(),
        0.5_f64.to_bits()
    );
}

#[test]
fn trade_row_poison_does_not_poison_its_indicator_or_account() {
    let (context, _, account) = infrastructure("1min", Ok(1), Some(0.25));
    let indicator = account.lock().unwrap().indicator().clone();
    let row = indicator.read().unwrap().trade_indicator().clone();
    assert!(
        std::thread::spawn(move || {
            let _guard = row.write().unwrap();
            panic!("poison trade row");
        })
        .join()
        .is_err()
    );
    assert_eq!(
        context.latest_price_advantage().unwrap_err().message,
        "trade indicator row lock poisoned"
    );
    assert!(indicator.try_write().is_ok());
    assert!(account.try_lock().is_ok());
}

fn series(values: &[Option<f64>]) -> QuoteData {
    let timestamps = (0..values.len())
        .map(|minute| {
            time(i64::try_from(minute).unwrap())
                .and_utc()
                .timestamp_nanos_opt()
                .unwrap()
        })
        .collect::<Vec<_>>();
    QuoteData::Series(
        RecordBatch::try_from_iter(vec![
            (
                "datetime",
                Arc::new(TimestampNanosecondArray::from(timestamps)) as ArrayRef,
            ),
            (
                "value",
                Arc::new(Float64Array::from(values.to_vec())) as ArrayRef,
            ),
        ])
        .unwrap(),
    )
}

struct ScriptQuote {
    responses: Mutex<VecDeque<Result<Option<QuoteData>, QuoteError>>>,
    calls: Arc<Mutex<Vec<String>>>,
}

impl Quote for ScriptQuote {
    fn get_all_stock(&self) -> Vec<String> {
        vec!["A".to_owned()]
    }

    fn get_data(
        &self,
        _stock: &str,
        _range: TimeRange,
        field: &str,
        method: QuoteMethod,
    ) -> Result<Option<QuoteData>, QuoteError> {
        assert_eq!(method, QuoteMethod::Selection);
        self.calls.lock().unwrap().push(field.to_owned());
        self.responses.lock().unwrap().pop_front().unwrap()
    }
}

fn market(
    responses: Vec<Result<Option<QuoteData>, QuoteError>>,
) -> (ExchangeSaoeMarket, Arc<Mutex<Vec<String>>>) {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let quote = Arc::new(ScriptQuote {
        responses: Mutex::new(responses.into()),
        calls: Arc::clone(&calls),
    });
    let exchange = ExchangeQuoteProvider::new(
        quote,
        domain_core::DealPriceFields::directional("$ask", "$bid"),
    );
    (
        ExchangeSaoeMarket::new(exchange, vec![time(0), time(1), time(1), time(2)]),
        calls,
    )
}

#[test]
fn live_market_reader_uses_real_exchange_bridge_without_a_new_provider_adapter() {
    let (bridge, calls) = market(vec![
        Ok(Some(series(&[Some(100.0), Some(200.0)]))),
        Ok(Some(series(&[Some(11.0), Some(12.0)]))),
    ]);
    let order = Arc::new(std::sync::RwLock::new(Order::new(
        "A",
        10.0,
        OrderDir::Sell,
        Some(time(0)),
        Some(time(1)),
    )));
    let result =
        domain_core::saoe_live_market::read_live_saoe_market(&bridge, &order, time(0), time(1))
            .unwrap();
    assert_eq!(result.volume, ndarray::arr1(&[100.0, 200.0]));
    assert_eq!(result.price, ndarray::arr1(&[11.0, 12.0]));
    assert_eq!(*calls.lock().unwrap(), ["$volume", "$bid"]);
    assert!(order.try_write().is_ok());
}

#[test]
#[allow(clippy::float_cmp)]
fn exchange_bridge_preserves_timeline_shapes_nulls_direction_and_python_query_order() {
    let (bridge, calls) = market(vec![
        Ok(Some(series(&[Some(10.0), None]))),
        Ok(Some(series(&[Some(100.0), Some(200.0)]))),
        Ok(Some(series(&[Some(300.0), Some(400.0)]))),
        Ok(Some(series(&[Some(11.0), Some(12.0)]))),
    ]);
    let cloned = bridge.clone();
    assert_eq!(
        SaoeBacktestDataSource::quote_timestamps(&cloned).unwrap(),
        [time(0), time(1), time(1), time(2)]
    );
    let prices =
        SaoeBacktestDataSource::deal_prices(&bridge, "A", time(0), time(1), OrderDir::Buy).unwrap();
    assert_eq!(prices[0], 10.0);
    assert!(prices[1].is_nan());
    assert_eq!(
        SaoeBacktestDataSource::market_volumes(&bridge, "A", time(0), time(1)).unwrap(),
        ndarray::arr1(&[100.0, 200.0])
    );
    let slice = bridge
        .market_slice("A", time(0), time(1), OrderDir::Buy)
        .unwrap();
    assert_eq!(slice.volume, ndarray::arr1(&[300.0, 400.0]));
    assert_eq!(slice.price, ndarray::arr1(&[11.0, 12.0]));
    assert_eq!(
        *calls.lock().unwrap(),
        ["$ask", "$volume", "$volume", "$ask"]
    );
}

#[test]
fn exchange_bridge_reports_absent_scalar_and_provider_failures_at_the_exact_stage() {
    for response in [
        Ok(None),
        Ok(Some(QuoteData::Scalar(Arc::new(Float64Array::from(vec![
            1.0,
        ]))))),
        Err(QuoteError::MissingStock {
            stock: "A".to_owned(),
        }),
    ] {
        let (bridge, _) = market(vec![response]);
        let message =
            SaoeBacktestDataSource::deal_prices(&bridge, "A", time(0), time(1), OrderDir::Sell)
                .unwrap_err()
                .message;
        assert!(
            message.contains("no data")
                || message.contains("scalar instead of a series")
                || message.contains("not present")
        );
    }

    let (volume_failure, calls) = market(vec![Ok(None)]);
    assert!(
        volume_failure
            .market_slice("A", time(0), time(1), OrderDir::Buy)
            .unwrap_err()
            .message
            .contains("volume")
    );
    assert_eq!(*calls.lock().unwrap(), ["$volume"]);

    let (price_failure, calls) = market(vec![
        Ok(Some(series(&[Some(1.0)]))),
        Ok(Some(QuoteData::Scalar(Arc::new(Float64Array::from(vec![
            2.0,
        ]))))),
    ]);
    assert!(
        price_failure
            .market_slice("A", time(0), time(1), OrderDir::Buy)
            .unwrap_err()
            .message
            .contains("deal price")
    );
    assert_eq!(*calls.lock().unwrap(), ["$volume", "$ask"]);
}

struct Calendar {
    step: Result<i64, NestedCalendarError>,
}

impl NestedCalendar for Calendar {
    fn finished(&self) -> Result<bool, NestedCalendarError> {
        Ok(false)
    }

    fn trade_len(&self) -> Result<i64, NestedCalendarError> {
        Ok(10)
    }

    fn trade_step(&self) -> Result<i64, NestedCalendarError> {
        self.step.clone()
    }

    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), NestedCalendarError> {
        Ok((time(0), time(1)))
    }

    fn step(&self) -> Result<(), NestedCalendarError> {
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum RangeMode {
    Some,
    None,
    Error,
}

struct Outer {
    decision: OrderTradeDecision,
    range: RangeMode,
}

impl Outer {
    fn new(range: RangeMode) -> Self {
        Self {
            decision: OrderTradeDecision::from_orders(Vec::new(), time(0), time(1), None),
            range,
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
        Ok(true)
    }

    fn range_limit(
        &self,
        _calendar: &dyn NestedCalendar,
    ) -> Result<Option<(i64, i64)>, NestedOuterDecisionError> {
        match self.range {
            RangeMode::Some => Ok(Some((3, 7))),
            RangeMode::None => Ok(None),
            RangeMode::Error => Err(NestedOuterDecisionError {
                message: "range".to_owned(),
            }),
        }
    }

    fn modify_inner_decision(
        &self,
        _decision: &mut dyn OrderDecision,
    ) -> Result<(), NestedOuterDecisionError> {
        Ok(())
    }
}

struct IndicatorView {
    values: domain_core::SharedTradeIndicator,
    order: domain_core::SharedOrderIndicator<NumpyOrderIndicator>,
}

impl AccountIndicator for IndicatorView {
    fn reset(&mut self) -> Result<(), AccountIndicatorError> {
        Ok(())
    }

    fn update_atomic(
        &mut self,
        _executions: &[OrderExecution<'_>],
    ) -> Result<(), AccountIndicatorError> {
        Ok(())
    }

    fn update_nested(
        &mut self,
        _update: NestedAccountIndicatorUpdate<'_>,
    ) -> Result<(), AccountIndicatorError> {
        Ok(())
    }

    fn calculate(
        &mut self,
        _config: domain_core::IndicatorConfig,
    ) -> Result<(), AccountIndicatorError> {
        Ok(())
    }

    fn record(&mut self, _trade_start_time: NaiveDateTime) -> Result<(), AccountIndicatorError> {
        Ok(())
    }

    fn trade_indicator(&self) -> &domain_core::SharedTradeIndicator {
        &self.values
    }

    fn order_indicator_snapshot(&self) -> Result<NumpyOrderIndicator, AccountIndicatorError> {
        Ok(NumpyOrderIndicator::default())
    }

    fn order_indicator_handle(
        &self,
    ) -> Result<domain_core::SharedOrderIndicator<NumpyOrderIndicator>, AccountIndicatorError> {
        Ok(self.order.clone())
    }

    fn recorded_trade_indicator(
        &self,
        _time: NaiveDateTime,
    ) -> Option<&domain_core::SharedTradeIndicator> {
        None
    }

    fn trade_indicator_report(
        &self,
    ) -> Result<domain_core::TradeIndicatorReport, AccountIndicatorError> {
        Err(AccountIndicatorError {
            message: "test view has no history export".to_owned(),
        })
    }
}

fn infrastructure(
    frequency: &str,
    calendar_result: Result<i64, NestedCalendarError>,
    pa: Option<f64>,
) -> (
    NestedAccountSaoeContext,
    SharedSaoeCalendar,
    SharedSaoeAccount,
) {
    let calendar: SharedSaoeCalendar = Arc::new(Mutex::new(Box::new(Calendar {
        step: calendar_result,
    })));
    let mut account = Account::new(InfinitePosition, false);
    account.replace_indicator(Box::new(IndicatorView {
        order: Arc::default(),
        values: Arc::new(std::sync::RwLock::new(
            pa.map(|value| IndexMap::from([("pa".to_owned(), value)]))
                .unwrap_or_default(),
        )),
    }));
    let account = Arc::new(Mutex::new(account));
    (
        NestedAccountSaoeContext::new(Arc::clone(&calendar), Arc::clone(&account), frequency),
        calendar,
        account,
    )
}

#[test]
#[allow(clippy::float_cmp)]
fn runtime_context_converts_frequencies_ranges_steps_account_metrics_and_warning() {
    for (frequency, minutes) in [("30min", 30), ("2day", 2880), ("3w", 30240)] {
        let (context, _, _) = infrastructure(frequency, Ok(8), Some(0.25));
        assert_eq!(context.ticks_per_step().unwrap(), minutes);
        assert_eq!(context.start_step(&Outer::new(RangeMode::Some)).unwrap(), 3);
        assert_eq!(context.start_step(&Outer::new(RangeMode::None)).unwrap(), 0);
        assert_eq!(context.current_trade_step().unwrap(), 8);
        assert_eq!(context.latest_price_advantage().unwrap(), 0.25);
        context.warn_overfill(12.0, 10.0).unwrap();
    }
}

#[test]
fn concrete_bridges_compose_into_a_ready_configured_adapter() {
    let (market, _) = market(vec![
        Ok(Some(series(&[Some(10.0), Some(11.0)]))),
        Ok(Some(series(&[Some(100.0), Some(110.0)]))),
    ]);
    let market = Arc::new(market);
    let (context, _, _) = infrastructure("2min", Ok(5), Some(0.1));
    let context = Arc::new(context);
    let mut outer = Outer::new(RangeMode::Some);
    outer.decision = OrderTradeDecision::from_orders(
        vec![Order::new(
            "A",
            10.0,
            OrderDir::Buy,
            Some(time(0)),
            Some(time(1)),
        )],
        time(0),
        time(1),
        Some(Arc::new(TradeRangeByTime::parse("09:30", "09:31").unwrap())),
    );
    let order = outer.decision.orders()[0].clone();
    let inputs = ConfiguredSaoeAdapterInputsProvider::new(
        SaoeBacktestDataLoader::new(market.clone()),
        market,
        context.clone(),
        context,
        1,
    );
    let mut factory = ConfiguredSaoeStateAdapterFactory::new(Box::new(inputs));
    let adapter = factory.create(&order, &outer).unwrap();
    let state = adapter.state(&order).unwrap();
    assert_eq!(state.parts().cur_step, 2);
    assert_eq!(state.parts().ticks_per_step, 2);
    assert_eq!(state.parts().backtest_data.deal_prices.len(), 2);
}

#[test]
fn runtime_context_reports_frequency_calendar_range_account_and_lock_failures() {
    for (frequency, expected) in [
        ("bad", "freq format is not supported"),
        ("month", "cannot use calendar months"),
        ("999999999999999999999999999999min", "too large"),
    ] {
        let (context, _, _) = infrastructure(frequency, Ok(1), Some(0.0));
        assert!(
            context
                .ticks_per_step()
                .unwrap_err()
                .message
                .contains(expected)
        );
    }

    let calendar_error = NestedCalendarError {
        message: "step".to_owned(),
    };
    let (context, _, _) = infrastructure("1min", Err(calendar_error), None);
    assert!(
        context
            .current_trade_step()
            .unwrap_err()
            .message
            .contains("step")
    );
    assert!(
        context
            .latest_price_advantage()
            .unwrap_err()
            .message
            .contains("no current price advantage")
    );
    assert!(
        context
            .start_step(&Outer::new(RangeMode::Error))
            .unwrap_err()
            .message
            .contains("range")
    );

    let (poisoned_calendar_context, calendar, _) = infrastructure("1min", Ok(1), Some(0.0));
    let calendar_thread = Arc::clone(&calendar);
    assert!(
        std::thread::spawn(move || {
            let _guard = calendar_thread.lock().unwrap();
            panic!("poison calendar");
        })
        .join()
        .is_err()
    );
    assert!(
        poisoned_calendar_context
            .current_trade_step()
            .unwrap_err()
            .message
            .contains("calendar lock is poisoned")
    );
    assert!(
        poisoned_calendar_context
            .start_step(&Outer::new(RangeMode::Some))
            .unwrap_err()
            .message
            .contains("calendar lock is poisoned")
    );

    let (poisoned_account_context, _, account) = infrastructure("1min", Ok(1), Some(0.0));
    let account_thread = Arc::clone(&account);
    assert!(
        std::thread::spawn(move || {
            let _guard = account_thread.lock().unwrap();
            panic!("poison account");
        })
        .join()
        .is_err()
    );
    assert!(
        poisoned_account_context
            .latest_price_advantage()
            .unwrap_err()
            .message
            .contains("account lock is poisoned")
    );
}
