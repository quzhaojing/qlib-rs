#![allow(
    clippy::float_cmp,
    reason = "Qlib differential fixtures require exact IEEE values, signed infinities, and zeros"
)]

use std::{collections::HashMap, path::PathBuf, process::Command, sync::Mutex};

use chrono::NaiveDateTime;
use domain_core::{
    AggregateBasePriceError, AggregateOrderIndicatorsError, BasePriceAggregation, BasePriceConfig,
    BasePriceDataProvider, BasePriceError, BasePriceProviderError, BasePriceRequest,
    BasePriceSource, BasePriceStep, BaseVolumePrice, IdxTradeRange, Indicator, IndicatorError,
    IndicatorStore, IndicatorStoreAccess, MIN_BASE_PRICE, MarketDataSeries, MarketDataValue,
    MetricSnapshot, NumpyOrderIndicator, Order, OrderDir, OrderIndicatorAggregationConfig,
    PandasOrderIndicator, TimeRange, TradeRangeByTime,
};
use serde_json::{Value, json};

#[path = "base_price/shared.rs"]
mod shared_tests;

#[path = "base_price/live.rs"]
mod live_tests;

fn timestamp(text: &str) -> NaiveDateTime {
    NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S%.f").unwrap()
}

fn series(timestamps: &[&str], values: &[f64]) -> MarketDataValue {
    MarketDataValue::Series(
        MarketDataSeries::try_new(
            timestamps.iter().map(|value| timestamp(value)).collect(),
            values.to_vec(),
        )
        .unwrap(),
    )
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Call {
    Price(String, TimeRange, OrderDir),
    Volume(String, TimeRange),
}

struct Provider {
    price: Result<Option<MarketDataValue>, BasePriceProviderError>,
    volume: Result<Option<MarketDataValue>, BasePriceProviderError>,
    calls: Mutex<Vec<Call>>,
}

#[test]
fn poisoned_order_store_aborts_base_price_before_provider_access() {
    let mut indicator = Indicator::new();
    let store = indicator.order_indicator().clone();
    assert!(
        std::thread::spawn(move || {
            let _guard = store.write().unwrap();
            panic!("poison base-price order store");
        })
        .join()
        .is_err()
    );
    let provider = Provider::new(None, None);
    let error = indicator
        .aggregate_base_price(&[], &[], &provider, BasePriceConfig::default())
        .unwrap_err();
    assert!(matches!(
        error,
        AggregateBasePriceError::Indicator(IndicatorError::OrderStorePoisoned)
    ));
    assert_eq!(error.to_string(), "order indicator store lock poisoned");
    let mut inner = orchestration_inner();
    assert_eq!(
        indicator.aggregate_order_indicators(
            &mut inner,
            &outer_orders(),
            &orchestration_steps(),
            &provider,
            OrderIndicatorAggregationConfig::default(),
        ),
        Err(AggregateOrderIndicatorsError::Indicator(
            IndicatorError::OrderStorePoisoned
        ))
    );
    assert_eq!(
        inner[0].metric_snapshot("trade_price").unwrap().values(),
        [3.0, 10.0]
    );
    assert!(provider.calls.lock().unwrap().is_empty());
}

impl Provider {
    fn new(price: Option<MarketDataValue>, volume: Option<MarketDataValue>) -> Self {
        Self {
            price: Ok(price),
            volume: Ok(volume),
            calls: Mutex::new(Vec::new()),
        }
    }
}

struct StoreCallbackProvider {
    store: domain_core::SharedOrderIndicator<NumpyOrderIndicator>,
    poison: bool,
}

impl BasePriceDataProvider for StoreCallbackProvider {
    fn deal_price(
        &self,
        _: &str,
        _: TimeRange,
        _: OrderDir,
    ) -> Result<Option<MarketDataValue>, BasePriceProviderError> {
        // try_write fails immediately rather than hanging if the caller holds a guard.
        let mut store = self.store.try_write().expect("provider must run unlocked");
        assign_metric(&mut *store, "callback", &[("A", 9.0)]);
        drop(store);
        if self.poison {
            let shared = self.store.clone();
            assert!(
                std::thread::spawn(move || {
                    let _guard = shared.write().unwrap();
                    panic!("provider poisons store before publication");
                })
                .join()
                .is_err()
            );
        }
        Ok(Some(MarketDataValue::Scalar(100.0)))
    }

    fn volume(
        &self,
        _: &str,
        _: TimeRange,
    ) -> Result<Option<MarketDataValue>, BasePriceProviderError> {
        panic!("TWAP must not request volume")
    }
}

#[test]
fn base_price_callback_runs_unlocked_and_poison_prevents_both_output_writes() {
    for poison in [false, true] {
        let mut indicator = Indicator::new();
        {
            let mut store = indicator.order_indicator_mut().unwrap();
            assign_metric(&mut *store, "trade_dir", &[("A", 1.0)]);
            assign_metric(&mut *store, "base_price", &[("OLD", 7.0)]);
            assign_metric(&mut *store, "base_volume", &[("OLD", 8.0)]);
        }
        let provider = StoreCallbackProvider {
            store: indicator.order_indicator().clone(),
            poison,
        };
        let inner = NumpyOrderIndicator::default();
        let steps = [BasePriceStep {
            start_time: timestamp("2024-01-02 09:30:00"),
            end_time: timestamp("2024-01-02 10:30:00"),
            trade_range: None,
        }];
        let result = indicator.aggregate_base_price(
            &[&inner],
            &steps,
            &provider,
            BasePriceConfig::default(),
        );
        if poison {
            assert!(matches!(
                result,
                Err(AggregateBasePriceError::Indicator(
                    IndicatorError::OrderStorePoisoned
                ))
            ));
        } else {
            result.unwrap();
        }
        // Inspect poisoned contents only in the test; production never recovers this lock.
        let store = provider
            .store
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(store.metric_snapshot("callback").unwrap().values(), [9.0]);
        let price = store.metric_snapshot("base_price").unwrap();
        let volume = store.metric_snapshot("base_volume").unwrap();
        if poison {
            assert_eq!(price.values(), [7.0]);
            assert_eq!(volume.values(), [8.0]);
        } else {
            assert_eq!(price.values(), [100.0]);
            assert_eq!(volume.values(), [1.0]);
        }
    }
}

impl BasePriceDataProvider for Provider {
    fn deal_price(
        &self,
        stock: &str,
        range: TimeRange,
        direction: OrderDir,
    ) -> Result<Option<MarketDataValue>, BasePriceProviderError> {
        self.calls
            .lock()
            .unwrap()
            .push(Call::Price(stock.to_owned(), range, direction));
        self.price.clone()
    }

    fn volume(
        &self,
        stock: &str,
        range: TimeRange,
    ) -> Result<Option<MarketDataValue>, BasePriceProviderError> {
        self.calls
            .lock()
            .unwrap()
            .push(Call::Volume(stock.to_owned(), range));
        self.volume.clone()
    }
}

struct MappingProvider {
    prices: HashMap<(NaiveDateTime, String), Option<f64>>,
    failure: Option<(NaiveDateTime, String)>,
    calls: Mutex<Vec<Call>>,
}

impl MappingProvider {
    fn new(values: &[(NaiveDateTime, &str, Option<f64>)]) -> Self {
        Self {
            prices: values
                .iter()
                .map(|(time, stock, value)| ((*time, (*stock).to_owned()), *value))
                .collect(),
            failure: None,
            calls: Mutex::new(Vec::new()),
        }
    }
}

impl BasePriceDataProvider for MappingProvider {
    fn deal_price(
        &self,
        stock: &str,
        range: TimeRange,
        direction: OrderDir,
    ) -> Result<Option<MarketDataValue>, BasePriceProviderError> {
        self.calls
            .lock()
            .unwrap()
            .push(Call::Price(stock.to_owned(), range, direction));
        let start = range.start.unwrap();
        if self.failure.as_ref() == Some(&(start, stock.to_owned())) {
            return Err(BasePriceProviderError::Provider {
                message: "boom".to_owned(),
            });
        }
        Ok(self
            .prices
            .get(&(start, stock.to_owned()))
            .copied()
            .flatten()
            .map(MarketDataValue::Scalar))
    }

    fn volume(
        &self,
        stock: &str,
        range: TimeRange,
    ) -> Result<Option<MarketDataValue>, BasePriceProviderError> {
        self.calls
            .lock()
            .unwrap()
            .push(Call::Volume(stock.to_owned(), range));
        Ok(None)
    }
}

fn assign_metric(store: &mut dyn IndicatorStoreAccess, name: &str, rows: &[(&str, f64)]) {
    store.assign_snapshot(
        name,
        MetricSnapshot::try_new(
            rows.iter().map(|(stock, _)| (*stock).to_owned()).collect(),
            rows.iter().map(|(_, value)| *value).collect(),
        )
        .unwrap(),
    );
}

#[test]
fn outer_local_stages_release_guard_before_provider_and_retain_partial_writes() {
    for poison in [false, true] {
        let mut indicator = Indicator::new();
        assign_metric(
            &mut *indicator.order_indicator_mut().unwrap(),
            "pa",
            &[("OLD", 7.0)],
        );
        let provider = StoreCallbackProvider {
            store: indicator.order_indicator().clone(),
            poison,
        };
        let mut inner = [NumpyOrderIndicator::default()];
        for (name, value) in [
            ("inner_amount", 4.0),
            ("deal_amount", 2.0),
            ("trade_price", 110.0),
            ("trade_value", 220.0),
            ("trade_cost", 1.0),
            ("trade_dir", 1.0),
        ] {
            assign_metric(&mut inner[0], name, &[("A", value)]);
        }
        let result = indicator.aggregate_order_indicators(
            &mut inner,
            &[Order::new("A", 4.0, OrderDir::Buy, None, None)],
            &orchestration_steps()[..1],
            &provider,
            OrderIndicatorAggregationConfig::default(),
        );
        if poison {
            assert_eq!(
                result,
                Err(AggregateOrderIndicatorsError::BasePrice(
                    AggregateBasePriceError::Indicator(IndicatorError::OrderStorePoisoned)
                ))
            );
        } else {
            result.unwrap();
        }
        assert_eq!(
            inner[0].metric_snapshot("trade_price").unwrap().values(),
            [220.0]
        );
        let store = provider
            .store
            .try_read()
            .unwrap_or_else(|error| match error {
                std::sync::TryLockError::Poisoned(error) => error.into_inner(),
                std::sync::TryLockError::WouldBlock => panic!("operation retained its guard"),
            });
        assert_eq!(store.metric_snapshot("amount").unwrap().values(), [4.0]);
        assert_eq!(store.metric_snapshot("ffr").unwrap().values(), [0.5]);
        assert_eq!(store.metric_snapshot("callback").unwrap().values(), [9.0]);
        if poison {
            assert_eq!(store.metric_snapshot("pa").unwrap().values(), [7.0]);
            assert!(store.metric_snapshot("base_price").is_none());
            assert!(store.metric_snapshot("base_volume").is_none());
        } else {
            assert!((store.metric_snapshot("pa").unwrap().values()[0] + 0.1).abs() < 1e-14);
        }
    }
}

fn orchestration_inner<S: IndicatorStore>() -> [S; 2] {
    let mut first = S::default();
    for (name, rows) in [
        ("inner_amount", vec![("B", 2.0), ("A", 10.0)]),
        ("deal_amount", vec![("B", -1.0), ("A", 4.0)]),
        ("trade_price", vec![("B", 3.0), ("A", 10.0)]),
        ("trade_value", vec![("B", -3.0), ("A", 40.0)]),
        ("trade_cost", vec![("B", 2.0), ("A", 1.0)]),
        ("trade_dir", vec![("B", 0.0), ("A", 1.0)]),
        ("base_price", vec![("B", f64::NAN), ("A", 100.0)]),
        ("base_volume", vec![("B", 9.0), ("A", 2.0)]),
    ] {
        assign_metric(&mut first, name, &rows);
    }
    let mut second = S::default();
    for (name, rows) in [
        ("inner_amount", vec![("C", 6.0), ("A", 5.0)]),
        ("deal_amount", vec![("C", 0.0), ("A", 1.0)]),
        ("trade_price", vec![("C", 7.0), ("A", 20.0)]),
        ("trade_value", vec![("C", 0.0), ("A", 20.0)]),
        ("trade_cost", vec![("C", 0.0), ("A", 3.0)]),
        ("trade_dir", vec![("C", 1.0), ("A", 0.0)]),
        ("base_price", vec![("C", f64::NAN), ("A", 110.0)]),
        ("base_volume", vec![("C", 9.0), ("A", 3.0)]),
    ] {
        assign_metric(&mut second, name, &rows);
    }
    [first, second]
}

fn outer_orders() -> [Order; 3] {
    [
        Order::new("A", 20.0, OrderDir::Buy, None, None),
        Order::new("B", 2.0, OrderDir::Sell, None, None),
        Order::new("D", 4.0, OrderDir::Buy, None, None),
    ]
}

fn orchestration_steps() -> [BasePriceStep<'static>; 2] {
    [
        BasePriceStep {
            start_time: timestamp("2024-01-02 09:30:00"),
            end_time: timestamp("2024-01-02 10:30:00"),
            trade_range: None,
        },
        BasePriceStep {
            start_time: timestamp("2024-01-02 10:30:00"),
            end_time: timestamp("2024-01-02 11:30:00"),
            trade_range: None,
        },
    ]
}

fn calculate(
    provider: &Provider,
    aggregation: BasePriceAggregation,
) -> Result<Option<BaseVolumePrice>, BasePriceError> {
    Indicator::new().get_base_volume_price(
        BasePriceRequest {
            stock: "A",
            start_time: timestamp("2024-01-02 09:30:00"),
            end_time: timestamp("2024-01-02 10:30:00"),
            direction: OrderDir::Buy,
            trade_range: None,
            config: BasePriceConfig {
                aggregation,
                source: BasePriceSource::DealPrice,
            },
        },
        provider,
    )
}

#[test]
fn configuration_series_and_object_safe_provider_are_typed() {
    assert_eq!(MIN_BASE_PRICE, 1.0e-8);
    assert_eq!(
        BasePriceConfig::default(),
        BasePriceConfig::parse(None, None).unwrap()
    );
    assert_eq!(
        BasePriceConfig::parse(Some("VwAp"), Some("DEAL_PRICE")).unwrap(),
        BasePriceConfig {
            aggregation: BasePriceAggregation::Vwap,
            source: BasePriceSource::DealPrice,
        }
    );
    assert_eq!(
        BasePriceConfig::parse(Some("median"), None),
        Err(BasePriceError::UnsupportedAggregation {
            aggregation: "median".to_owned()
        })
    );
    assert_eq!(
        BasePriceConfig::parse(Some("median"), Some("close")),
        Err(BasePriceError::UnsupportedPriceSource {
            price_source: "close".to_owned()
        })
    );

    let valid =
        MarketDataSeries::try_new(vec![timestamp("2024-01-02 09:30:00")], vec![2.0]).unwrap();
    assert_eq!(valid.timestamps(), [timestamp("2024-01-02 09:30:00")]);
    assert_eq!(valid.values(), [2.0]);
    assert_eq!(
        MarketDataSeries::try_new(vec![timestamp("2024-01-02 09:30:00")], vec![2.0, 3.0]),
        Err(BasePriceError::SeriesLengthMismatch {
            timestamps: 1,
            values: 2
        })
    );

    let provider = Provider::new(Some(MarketDataValue::Scalar(2.0)), None);
    let plugin: &dyn BasePriceDataProvider = &provider;
    assert!(
        plugin
            .deal_price("A", TimeRange::default(), OrderDir::Sell)
            .unwrap()
            .is_some()
    );
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one cohesive TWAP behavior matrix covers filtering, clipping, and provider failures"
)]
fn twap_filters_prices_clips_ranges_and_propagates_failures() {
    let start = timestamp("2024-01-02 09:30:00");
    let end = timestamp("2024-01-02 10:30:00");
    let provider = Provider::new(
        Some(series(
            &[
                "2024-01-02 09:30:00",
                "2024-01-02 09:31:00",
                "2024-01-02 09:32:00",
                "2024-01-02 09:33:00",
                "2024-01-02 09:34:00",
                "2024-01-02 09:35:00",
            ],
            &[
                0.0,
                -1.0,
                MIN_BASE_PRICE,
                f64::from_bits(MIN_BASE_PRICE.to_bits() + 1),
                f64::NAN,
                2.0,
            ],
        )),
        None,
    );
    let result = calculate(&provider, BasePriceAggregation::Twap)
        .unwrap()
        .unwrap();
    assert_eq!(result.base_volume, 2.0);
    assert_eq!(
        result.base_price,
        f64::midpoint(f64::from_bits(MIN_BASE_PRICE.to_bits() + 1), 2.0)
    );
    assert_eq!(provider.calls.lock().unwrap().len(), 1);

    let scalar = Provider::new(Some(MarketDataValue::Scalar(5.5)), None);
    assert_eq!(
        calculate(&scalar, BasePriceAggregation::Twap),
        Ok(Some(BaseVolumePrice {
            base_price: 5.5,
            base_volume: 1.0
        }))
    );
    for price in [
        None,
        Some(MarketDataValue::Scalar(f64::NAN)),
        Some(MarketDataValue::Scalar(0.0)),
    ] {
        let empty = Provider::new(price, None);
        assert_eq!(calculate(&empty, BasePriceAggregation::Twap), Ok(None));
    }
    let empty = Provider::new(Some(series(&[], &[])), None);
    assert_eq!(calculate(&empty, BasePriceAggregation::Twap), Ok(None));

    let clipped = Provider::new(Some(MarketDataValue::Scalar(2.0)), None);
    let trade_range = TradeRangeByTime::parse("09:45", "10:15").unwrap();
    let result = Indicator::new().get_base_volume_price(
        BasePriceRequest {
            stock: "CLIP",
            start_time: start,
            end_time: end,
            direction: OrderDir::Sell,
            trade_range: Some(&trade_range),
            config: BasePriceConfig::default(),
        },
        &clipped,
    );
    assert!(result.unwrap().is_some());
    assert_eq!(
        *clipped.calls.lock().unwrap(),
        [Call::Price(
            "CLIP".to_owned(),
            TimeRange {
                start: Some(timestamp("2024-01-02 09:45:00")),
                end: Some(timestamp("2024-01-02 10:15:00"))
            },
            OrderDir::Sell
        )]
    );
    assert_eq!(
        Indicator::new().get_base_volume_price(
            BasePriceRequest {
                stock: "A",
                start_time: start,
                end_time: end,
                direction: OrderDir::Buy,
                trade_range: Some(&IdxTradeRange::new(0, 1)),
                config: BasePriceConfig::default(),
            },
            &clipped,
        ),
        Err(BasePriceError::TradeRange(
            domain_core::TradeRangeError::IndexTimeClippingUnsupported
        ))
    );

    let failure = BasePriceProviderError::Provider {
        message: "price boom".to_owned(),
    };
    let failed = Provider {
        price: Err(failure.clone()),
        volume: Ok(None),
        calls: Mutex::new(Vec::new()),
    };
    assert_eq!(
        calculate(&failed, BasePriceAggregation::Twap),
        Err(BasePriceError::Provider(failure))
    );
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one cohesive VWAP behavior matrix covers alignment, duplicates, IEEE edges, and failures"
)]
fn vwap_matches_single_data_alignment_nan_and_duplicate_semantics() {
    let t1 = "2024-01-02 09:31:00";
    let t2 = "2024-01-02 09:32:00";
    for (price, volume, expected) in [
        (
            series(&[t1, t2], &[2.0, 4.0]),
            series(&[t2, t1], &[3.0, 1.0]),
            BaseVolumePrice {
                base_price: 3.5,
                base_volume: 4.0,
            },
        ),
        (
            series(&[t1, t2], &[2.0, 4.0]),
            series(&[t1], &[3.0]),
            BaseVolumePrice {
                base_price: 2.0,
                base_volume: 3.0,
            },
        ),
        (
            series(&[t1, t1], &[2.0, 4.0]),
            series(&[t1, t1], &[3.0, 1.0]),
            BaseVolumePrice {
                base_price: 2.5,
                base_volume: 4.0,
            },
        ),
        (
            series(&[t1, t1], &[2.0, 4.0]),
            series(&[t1, t2], &[3.0, 1.0]),
            BaseVolumePrice {
                base_price: 3.0,
                base_volume: 6.0,
            },
        ),
    ] {
        let provider = Provider::new(Some(price), Some(volume));
        assert_eq!(
            calculate(&provider, BasePriceAggregation::Vwap),
            Ok(Some(expected))
        );
        assert!(matches!(
            provider.calls.lock().unwrap().as_slice(),
            [Call::Price(..), Call::Volume(..)]
        ));
    }

    let scalar_volume = Provider::new(
        Some(series(&["2024-01-02 09:30:00", t2], &[2.0, 4.0])),
        Some(MarketDataValue::Scalar(3.0)),
    );
    assert_eq!(
        calculate(&scalar_volume, BasePriceAggregation::Vwap),
        Ok(Some(BaseVolumePrice {
            base_price: 2.0,
            base_volume: 3.0
        }))
    );

    let nan_volume = Provider::new(
        Some(series(&[t1, t2], &[2.0, 4.0])),
        Some(series(&[t1, t2], &[f64::NAN, f64::NAN])),
    );
    let result = calculate(&nan_volume, BasePriceAggregation::Vwap)
        .unwrap()
        .unwrap();
    assert_eq!(result.base_volume, 0.0);
    assert!(result.base_price.is_nan());

    let zero_sum = Provider::new(
        Some(series(&[t1, t2], &[2.0, 4.0])),
        Some(series(&[t1, t2], &[1.0, -1.0])),
    );
    let result = calculate(&zero_sum, BasePriceAggregation::Vwap)
        .unwrap()
        .unwrap();
    assert_eq!(result.base_volume, 0.0);
    assert_eq!(result.base_price, f64::NEG_INFINITY);

    let infinite = Provider::new(
        Some(series(&[t1, t2], &[f64::INFINITY, 2.0])),
        Some(series(&[t1, t2], &[0.0, 1.0])),
    );
    assert_eq!(
        calculate(&infinite, BasePriceAggregation::Vwap),
        Ok(Some(BaseVolumePrice {
            base_price: 2.0,
            base_volume: 1.0
        }))
    );

    let missing = Provider::new(Some(MarketDataValue::Scalar(2.0)), None);
    assert_eq!(
        calculate(&missing, BasePriceAggregation::Vwap),
        Err(BasePriceError::MissingVolume)
    );
    let failure = BasePriceProviderError::Provider {
        message: "volume boom".to_owned(),
    };
    let failed = Provider {
        price: Ok(Some(MarketDataValue::Scalar(2.0))),
        volume: Err(failure.clone()),
        calls: Mutex::new(Vec::new()),
    };
    assert_eq!(
        calculate(&failed, BasePriceAggregation::Vwap),
        Err(BasePriceError::Provider(failure))
    );
}

fn run_cross_step_aggregation<S: IndicatorStore>(
    mut output: Indicator<S>,
) -> (MetricSnapshot, MetricSnapshot, Vec<Call>) {
    assign_metric(
        &mut *output.order_indicator_mut().unwrap(),
        "trade_dir",
        &[("B", 1.0), ("A", 0.0), ("D", 0.0), ("C", 1.0), ("E", 0.0)],
    );
    let mut first = S::default();
    assign_metric(
        &mut first,
        "base_price",
        &[
            ("A", 100.0),
            ("B", f64::NAN),
            ("C", f64::INFINITY),
            ("E", 50.0),
        ],
    );
    assign_metric(
        &mut first,
        "base_volume",
        &[("A", 2.0), ("B", 9.0), ("C", 0.0)],
    );
    let mut second = S::default();
    assign_metric(&mut second, "base_price", &[("A", 110.0)]);
    assign_metric(&mut second, "base_volume", &[("A", 3.0)]);

    let provider = MappingProvider::new(&[
        (timestamp("2024-01-02 09:31:00"), "B", Some(80.0)),
        (timestamp("2024-01-02 09:31:00"), "D", None),
        (timestamp("2024-01-02 10:30:00"), "B", Some(120.0)),
        (timestamp("2024-01-02 10:30:00"), "D", Some(40.0)),
        (timestamp("2024-01-02 10:30:00"), "C", None),
        (timestamp("2024-01-02 10:30:00"), "E", None),
    ]);
    let range = TradeRangeByTime::parse("09:31", "10:29").unwrap();
    let steps = [
        BasePriceStep {
            start_time: timestamp("2024-01-02 09:30:00"),
            end_time: timestamp("2024-01-02 10:30:00"),
            trade_range: Some(&range),
        },
        BasePriceStep {
            start_time: timestamp("2024-01-02 10:30:00"),
            end_time: timestamp("2024-01-02 11:30:00"),
            trade_range: None,
        },
    ];
    let inner: [&dyn IndicatorStoreAccess; 2] = [&first, &second];
    output
        .aggregate_base_price(&inner, &steps, &provider, BasePriceConfig::default())
        .unwrap();
    (
        output
            .order_indicator()
            .read()
            .unwrap()
            .metric_snapshot("base_price")
            .unwrap(),
        output
            .order_indicator()
            .read()
            .unwrap()
            .metric_snapshot("base_volume")
            .unwrap(),
        provider.calls.into_inner().unwrap(),
    )
}

#[test]
fn cross_step_aggregation_backfills_and_weights_both_storage_plugins() {
    for (price, volume, calls) in [
        run_cross_step_aggregation(Indicator::<NumpyOrderIndicator>::default()),
        run_cross_step_aggregation(Indicator::<PandasOrderIndicator>::default()),
    ] {
        assert_eq!(price.index(), ["A", "B", "C", "D", "E"]);
        assert_eq!(price.values()[0..2], [106.0, 100.0]);
        assert!(price.values()[2].is_nan());
        assert_eq!(price.values()[3], 40.0);
        assert!(price.values()[4].is_nan());
        assert_eq!(volume.index(), ["A", "B", "C", "D", "E"]);
        assert_eq!(volume.values(), [5.0, 2.0, 0.0, 1.0, 0.0]);
        assert_eq!(calls.len(), 6);
        assert_eq!(
            calls[0],
            Call::Price(
                "B".to_owned(),
                TimeRange {
                    start: Some(timestamp("2024-01-02 09:31:00")),
                    end: Some(timestamp("2024-01-02 10:29:00"))
                },
                OrderDir::Buy
            )
        );
        assert!(matches!(calls[1], Call::Price(ref stock, _, OrderDir::Sell) if stock == "D"));
        assert!(matches!(calls[2], Call::Price(ref stock, _, OrderDir::Buy) if stock == "B"));
        assert!(matches!(calls[3], Call::Price(ref stock, _, OrderDir::Sell) if stock == "D"));
        assert!(matches!(calls[4], Call::Price(ref stock, _, OrderDir::Buy) if stock == "C"));
        assert!(matches!(calls[5], Call::Price(ref stock, _, OrderDir::Sell) if stock == "E"));
    }
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one cohesive edge matrix verifies Python zip, empty, lazy-direction, and atomic failure rules"
)]
fn cross_step_aggregation_preserves_empty_zip_and_atomic_failure_rules() {
    let provider = MappingProvider::new(&[]);
    let mut empty = Indicator::<NumpyOrderIndicator>::default();
    assign_metric(
        &mut *empty.order_indicator_mut().unwrap(),
        "base_price",
        &[("OLD", 7.0)],
    );
    assign_metric(
        &mut *empty.order_indicator_mut().unwrap(),
        "base_volume",
        &[("OLD", 8.0)],
    );
    empty
        .aggregate_base_price(&[], &[], &provider, BasePriceConfig::default())
        .unwrap();
    assert_eq!(
        empty
            .order_indicator()
            .read()
            .unwrap()
            .metric_snapshot("base_price")
            .unwrap()
            .values(),
        [7.0]
    );
    assign_metric(&mut *empty.order_indicator_mut().unwrap(), "trade_dir", &[]);
    empty
        .aggregate_base_price(&[], &[], &provider, BasePriceConfig::default())
        .unwrap();
    assert_eq!(
        empty
            .order_indicator()
            .read()
            .unwrap()
            .metric_snapshot("base_volume")
            .unwrap()
            .values(),
        [8.0]
    );

    assign_metric(
        &mut *empty.order_indicator_mut().unwrap(),
        "trade_dir",
        &[("A", 1.0)],
    );
    let extra_step = [BasePriceStep {
        start_time: timestamp("2024-01-02 09:30:00"),
        end_time: timestamp("2024-01-02 10:30:00"),
        trade_range: None,
    }];
    empty
        .aggregate_base_price(&[], &extra_step, &provider, BasePriceConfig::default())
        .unwrap();
    assert!(
        empty
            .order_indicator()
            .read()
            .unwrap()
            .metric_snapshot("base_price")
            .unwrap()
            .is_empty()
    );

    let mut first = NumpyOrderIndicator::default();
    assign_metric(&mut first, "base_price", &[("A", 100.0)]);
    assign_metric(&mut first, "base_volume", &[("A", 2.0)]);
    let second = NumpyOrderIndicator::default();
    let inner: [&dyn IndicatorStoreAccess; 2] = [&first, &second];
    empty
        .aggregate_base_price(&inner, &extra_step, &provider, BasePriceConfig::default())
        .unwrap();
    assert_eq!(
        empty
            .order_indicator()
            .read()
            .unwrap()
            .metric_snapshot("base_price")
            .unwrap()
            .values(),
        [100.0]
    );

    let mut lazy = Indicator::<NumpyOrderIndicator>::default();
    assign_metric(
        &mut *lazy.order_indicator_mut().unwrap(),
        "trade_dir",
        &[("A", f64::NAN)],
    );
    let mut existing = NumpyOrderIndicator::default();
    assign_metric(&mut existing, "base_price", &[("A", 5.0)]);
    assign_metric(&mut existing, "base_volume", &[("A", 2.0)]);
    lazy.aggregate_base_price(
        &[&existing],
        &extra_step,
        &provider,
        BasePriceConfig::default(),
    )
    .unwrap();
    assert_eq!(
        lazy.order_indicator()
            .read()
            .unwrap()
            .metric_snapshot("base_price")
            .unwrap()
            .values(),
        [5.0]
    );

    assign_metric(
        &mut *lazy.order_indicator_mut().unwrap(),
        "base_price",
        &[("OLD", 7.0)],
    );
    assign_metric(
        &mut *lazy.order_indicator_mut().unwrap(),
        "base_volume",
        &[("OLD", 8.0)],
    );
    let missing = NumpyOrderIndicator::default();
    let error = lazy
        .aggregate_base_price(
            &[&missing],
            &extra_step,
            &provider,
            BasePriceConfig::default(),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        AggregateBasePriceError::InvalidDirection { stock, direction }
            if stock == "A" && direction.is_nan()
    ));
    assert_eq!(
        lazy.order_indicator()
            .read()
            .unwrap()
            .metric_snapshot("base_price")
            .unwrap()
            .index(),
        ["OLD"]
    );

    assign_metric(
        &mut *lazy.order_indicator_mut().unwrap(),
        "trade_dir",
        &[("A", 1.0)],
    );
    let mut failed_provider = MappingProvider::new(&[]);
    failed_provider.failure = Some((timestamp("2024-01-02 09:30:00"), "A".to_owned()));
    assert!(matches!(
        lazy.aggregate_base_price(
            &[&missing],
            &extra_step,
            &failed_provider,
            BasePriceConfig::default()
        ),
        Err(AggregateBasePriceError::BasePrice(BasePriceError::Provider(
            BasePriceProviderError::Provider { message }
        ))) if message == "boom"
    ));
    assert_eq!(
        lazy.order_indicator()
            .read()
            .unwrap()
            .metric_snapshot("base_volume")
            .unwrap()
            .index(),
        ["OLD"]
    );

    assert!(matches!(
        lazy.aggregate_base_price(
            &[&missing],
            &[BasePriceStep {
                trade_range: Some(&IdxTradeRange::new(0, 1)),
                ..extra_step[0]
            }],
            &provider,
            BasePriceConfig::default()
        ),
        Err(AggregateBasePriceError::BasePrice(
            BasePriceError::TradeRange(domain_core::TradeRangeError::IndexTimeClippingUnsupported)
        ))
    ));

    let fixed = Provider::new(Some(MarketDataValue::Scalar(2.0)), None);
    assert_eq!(
        lazy.aggregate_base_price(
            &[&missing],
            &extra_step,
            &fixed,
            BasePriceConfig {
                aggregation: BasePriceAggregation::Vwap,
                source: BasePriceSource::DealPrice
            }
        ),
        Err(AggregateBasePriceError::BasePrice(
            BasePriceError::MissingVolume
        ))
    );
}

fn run_full_orchestration<S: IndicatorStore>(mut output: Indicator<S>, shared: bool) {
    let mut inner = orchestration_inner();
    let handles: Vec<_> = inner
        .iter()
        .cloned()
        .map(|store| std::sync::Arc::new(std::sync::RwLock::new(store)))
        .collect();
    let provider = MappingProvider::new(&[
        (timestamp("2024-01-02 09:30:00"), "B", Some(80.0)),
        (timestamp("2024-01-02 09:30:00"), "C", None),
        (timestamp("2024-01-02 10:30:00"), "B", None),
        (timestamp("2024-01-02 10:30:00"), "C", Some(40.0)),
    ]);
    if shared {
        output
            .aggregate_shared_order_indicators(
                &handles,
                &outer_orders(),
                &orchestration_steps(),
                &provider,
                OrderIndicatorAggregationConfig::default(),
            )
            .unwrap();
        for (store, handle) in inner.iter_mut().zip(&handles) {
            *store = handle.read().unwrap().clone();
        }
    } else {
        output
            .aggregate_order_indicators(
                &mut inner,
                &outer_orders(),
                &orchestration_steps(),
                &provider,
                OrderIndicatorAggregationConfig::default(),
            )
            .unwrap();
    }
    let snapshot = output.order_snapshot().unwrap();
    assert_eq!(snapshot["inner_amount"].values(), [15.0, 2.0, 6.0]);
    assert_eq!(snapshot["deal_amount"].values(), [5.0, -1.0, 0.0]);
    assert_eq!(snapshot["trade_price"].values()[0..2], [12.0, 3.0]);
    assert!(snapshot["trade_price"].values()[2].is_nan());
    assert_eq!(snapshot["amount"].index(), ["A", "B", "D"]);
    assert_eq!(snapshot["amount"].values(), [20.0, -2.0, 4.0]);
    assert_eq!(snapshot["ffr"].index(), ["A", "B", "D"]);
    assert_eq!(snapshot["ffr"].values(), [0.25, 0.5, 0.0]);
    assert_eq!(snapshot["base_price"].values(), [106.0, 80.0, 40.0]);
    assert_eq!(snapshot["base_volume"].values(), [5.0, 1.0, 1.0]);
    assert!((snapshot["pa"].values()[0] - (1.0 - 12.0 / 106.0)).abs() < f64::EPSILON);
    assert_eq!(snapshot["pa"].values()[1], 3.0 / 80.0 - 1.0);
    assert!(snapshot["pa"].values()[2].is_nan());
    assert_eq!(
        inner[0].metric_snapshot("trade_price").unwrap().values(),
        [-3.0, 40.0]
    );
    assert_eq!(
        inner[1].metric_snapshot("trade_price").unwrap().values(),
        [0.0, 20.0]
    );
    assert_eq!(provider.calls.into_inner().unwrap().len(), 4);
}

#[test]
fn full_order_orchestration_matches_both_storage_plugins() {
    for shared in [false, true] {
        run_full_orchestration(Indicator::<NumpyOrderIndicator>::default(), shared);
        run_full_orchestration(Indicator::<PandasOrderIndicator>::default(), shared);
    }
}

#[test]
fn fulfillment_rate_aligns_labels_and_reports_missing_inputs() {
    let mut indicator = Indicator::<NumpyOrderIndicator>::default();
    assign_metric(
        &mut *indicator.order_indicator_mut().unwrap(),
        "ffr",
        &[("OLD", 9.0)],
    );
    assert_eq!(
        indicator.update_order_fulfill_rate(),
        Err(IndicatorError::MissingMetric("deal_amount".to_owned()))
    );
    assign_metric(
        &mut *indicator.order_indicator_mut().unwrap(),
        "deal_amount",
        &[("B", 1.0), ("A", f64::NAN)],
    );
    assert_eq!(
        indicator.update_order_fulfill_rate(),
        Err(IndicatorError::MissingMetric("amount".to_owned()))
    );
    assert_eq!(indicator.order_snapshot().unwrap()["ffr"].values(), [9.0]);
    assert!(indicator.order_indicator().try_write().is_ok());
    assign_metric(
        &mut *indicator.order_indicator_mut().unwrap(),
        "amount",
        &[("A", 2.0), ("B", 0.0), ("C", f64::NAN)],
    );
    indicator.update_order_fulfill_rate().unwrap();
    let ffr = indicator
        .order_indicator()
        .read()
        .unwrap()
        .metric_snapshot("ffr")
        .unwrap();
    assert_eq!(ffr.index(), ["A", "B", "C"]);
    assert_eq!(ffr.values()[0], 0.0);
    assert!(ffr.values()[1].is_infinite());
    assert!(ffr.values()[2].is_nan());
    let retained = indicator.order_indicator().clone();
    let poison = retained.clone();
    assert!(
        std::thread::spawn(move || {
            let _guard = poison.write().unwrap();
            panic!("poison previously calculated fulfillment store");
        })
        .join()
        .is_err()
    );
    assert_eq!(
        indicator.update_order_fulfill_rate(),
        Err(IndicatorError::OrderStorePoisoned)
    );
    let store = retained.read().unwrap_err().into_inner();
    let unchanged = store.metric_snapshot("ffr").unwrap();
    assert_eq!(unchanged.index(), ffr.index());
    assert_eq!(unchanged.values()[0], 0.0);
    assert!(unchanged.values()[1].is_infinite());
    assert!(unchanged.values()[2].is_nan());
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one stage-order matrix verifies non-atomic orchestration semantics"
)]
fn full_order_orchestration_preserves_empty_and_partial_failure_state() {
    let provider = MappingProvider::new(&[]);
    let mut empty = Indicator::<NumpyOrderIndicator>::default();
    assign_metric(
        &mut *empty.order_indicator_mut().unwrap(),
        "base_price",
        &[("OLD", 7.0)],
    );
    assign_metric(
        &mut *empty.order_indicator_mut().unwrap(),
        "base_volume",
        &[("OLD", 8.0)],
    );
    assign_metric(
        &mut *empty.order_indicator_mut().unwrap(),
        "pa",
        &[("OLD", 9.0)],
    );
    empty
        .aggregate_order_indicators(
            &mut [],
            &[],
            &[],
            &provider,
            OrderIndicatorAggregationConfig::default(),
        )
        .unwrap();
    let snapshot = empty.order_snapshot().unwrap();
    assert_eq!(snapshot["base_price"].values(), [7.0]);
    assert_eq!(snapshot["base_volume"].values(), [8.0]);
    assert!(snapshot["pa"].is_empty());
    assert!(snapshot["amount"].is_empty());
    assert!(snapshot["ffr"].is_empty());

    let mut bad_inner = [NumpyOrderIndicator::default()];
    assign_metric(&mut bad_inner[0], "trade_price", &[("A", 2.0)]);
    let mut stage_one = Indicator::<NumpyOrderIndicator>::default();
    assert_eq!(
        stage_one.aggregate_order_indicators(
            &mut bad_inner,
            &[],
            &[],
            &provider,
            OrderIndicatorAggregationConfig::default(),
        ),
        Err(AggregateOrderIndicatorsError::Indicator(
            IndicatorError::MissingMetric("deal_amount".to_owned())
        ))
    );
    assert!(stage_one.order_snapshot().unwrap().is_empty());

    let mut inner = orchestration_inner();
    let mut failed_provider = MappingProvider::new(&[]);
    failed_provider.failure = Some((timestamp("2024-01-02 09:30:00"), "B".to_owned()));
    let mut base_failure = Indicator::<NumpyOrderIndicator>::default();
    assign_metric(
        &mut *base_failure.order_indicator_mut().unwrap(),
        "base_price",
        &[("OLD", 7.0)],
    );
    assign_metric(
        &mut *base_failure.order_indicator_mut().unwrap(),
        "base_volume",
        &[("OLD", 8.0)],
    );
    assign_metric(
        &mut *base_failure.order_indicator_mut().unwrap(),
        "pa",
        &[("OLD", 9.0)],
    );
    assert!(matches!(
        base_failure.aggregate_order_indicators(
            &mut inner,
            &outer_orders(),
            &orchestration_steps(),
            &failed_provider,
            OrderIndicatorAggregationConfig::default(),
        ),
        Err(AggregateOrderIndicatorsError::BasePrice(
            AggregateBasePriceError::BasePrice(BasePriceError::Provider(
                BasePriceProviderError::Provider { message }
            ))
        )) if message == "boom"
    ));
    let partial = base_failure.order_snapshot().unwrap();
    assert_eq!(partial["ffr"].values(), [0.25, 0.5, 0.0]);
    assert_eq!(partial["base_price"].index(), ["OLD"]);
    assert_eq!(partial["base_volume"].index(), ["OLD"]);
    assert_eq!(partial["pa"].index(), ["OLD"]);
    assert_eq!(
        inner[0].metric_snapshot("trade_price").unwrap().values(),
        [-3.0, 40.0]
    );

    let mut dense_inner = orchestration_inner();
    let mut dense = Indicator::<NumpyOrderIndicator>::default();
    assert!(matches!(
        dense.aggregate_order_indicators(
            &mut dense_inner,
            &outer_orders(),
            &orchestration_steps(),
            &provider,
            OrderIndicatorAggregationConfig::default(),
        ),
        Err(AggregateOrderIndicatorsError::Indicator(
            IndicatorError::IndexMismatch { metric, weight }
        )) if metric == "trade_price" && weight == "base_price"
    ));
    assert_eq!(
        dense
            .order_indicator()
            .read()
            .unwrap()
            .metric_snapshot("base_price")
            .unwrap()
            .index(),
        ["A"]
    );

    let mut pandas_inner = orchestration_inner();
    let mut pandas = Indicator::<PandasOrderIndicator>::default();
    pandas
        .aggregate_order_indicators(
            &mut pandas_inner,
            &outer_orders(),
            &orchestration_steps(),
            &provider,
            OrderIndicatorAggregationConfig::default(),
        )
        .unwrap();
    let pa = pandas
        .order_indicator()
        .read()
        .unwrap()
        .metric_snapshot("pa")
        .unwrap();
    assert_eq!(pa.index(), ["A", "B", "C"]);
    assert!(pa.values()[1].is_nan());
    assert!(pa.values()[2].is_nan());
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "the self-contained differential executes the complete original Python pipeline"
)]
fn full_order_orchestration_matches_live_python_stage_order() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib");
    let script = r"
import ast,importlib.util,inspect,json,sys,warnings
from collections import OrderedDict
from dataclasses import dataclass
from enum import IntEnum
from types import SimpleNamespace
from typing import *
import numpy as np,pandas as pd
spec=importlib.util.spec_from_file_location('index_data_live',sys.argv[4]);idd=importlib.util.module_from_spec(spec);spec.loader.exec_module(idd);SingleData=idd.SingleData
class Logger:pass
def get_module_logger(_):return Logger()
def classes(path,names):
 t=ast.parse(open(path,encoding='utf-8').read(),filename=path);return [next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name==x) for x in names]
nodes=classes(sys.argv[2],['OrderDir','Order'])+classes(sys.argv[3],['BaseSingleMetric','BaseOrderIndicator','SingleMetric','PandasSingleMetric','PandasOrderIndicator','NumpyOrderIndicator']);m=ast.Module(body=nodes,type_ignores=[]);ast.fix_missing_locations(m);exec(compile(m,'live','exec'),globals())
t=ast.parse(open(sys.argv[1],encoding='utf-8').read(),filename=sys.argv[1]);c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='Indicator');names=['_agg_order_trade_info','_update_trade_amount','_update_order_fulfill_rate','_get_base_vol_pri','_agg_base_price','_agg_order_price_advantage','agg_order_indicators'];nodes=[next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==x) for x in names];BaseTradeDecision=Exchange=object;exec(compile(ast.Module(body=nodes,type_ignores=[]),'live','exec'),globals())
class I:
 _agg_order_trade_info=_agg_order_trade_info;_update_trade_amount=_update_trade_amount;_update_order_fulfill_rate=_update_order_fulfill_rate;_get_base_vol_pri=_get_base_vol_pri;_agg_base_price=_agg_base_price;_agg_order_price_advantage=_agg_order_price_advantage;agg_order_indicators=agg_order_indicators
 def __init__(self,cls,rows={}):self.order_indicator_cls=cls;self.order_indicator=make(cls,rows)
class D:
 def __init__(self,orders=[],trade_range=None):self.orders=orders;self.trade_range=trade_range
 def get_decision(self):return self.orders
class E:
 def __init__(self,fail=False,missing=False):self.fail=fail;self.missing=missing;self.calls=[]
 def get_deal_price(self,inst,start,end,direction,method):
  self.calls.append([inst,start.isoformat(),float(direction)])
  if self.fail:raise RuntimeError('boom')
  if self.missing:return None
  return {(9,'B'):80,(10,'C'):40}.get((start.hour,inst))
 def get_volume(self,*a,**k):raise AssertionError('unused')
def make(cls,rows):
 x=cls()
 for k,v in rows.items():x.assign(k,v)
 return x
def vals(x):return {'index':[str(v) for v in x.index.tolist()],'values':[None if np.isnan(v) else ('inf' if np.isposinf(v) else ('-inf' if np.isneginf(v) else float(v))) for v in x.data]}
def snap(i):return {k:vals(i.order_indicator.get_index_data(k)) for k in i.order_indicator.data}
def fixtures(cls):
 r1={'inner_amount':{'B':2,'A':10},'deal_amount':{'B':-1,'A':4},'trade_price':{'B':3,'A':10},'trade_value':{'B':-3,'A':40},'trade_cost':{'B':2,'A':1},'trade_dir':{'B':0,'A':1},'base_price':{'B':np.nan,'A':100},'base_volume':{'B':9,'A':2}}
 r2={'inner_amount':{'C':6,'A':5},'deal_amount':{'C':0,'A':1},'trade_price':{'C':7,'A':20},'trade_value':{'C':0,'A':20},'trade_cost':{'C':0,'A':3},'trade_dir':{'C':1,'A':0},'base_price':{'C':np.nan,'A':110},'base_volume':{'C':9,'A':3}}
 return [make(cls,r1),make(cls,r2)]
def run(cls):
 t0=pd.Timestamp('2024-01-02 09:30');t1=pd.Timestamp('2024-01-02 10:30');steps=[(D(),t0,t1),(D(),t1,t1+pd.Timedelta(hours=1))];outer=D([SimpleNamespace(stock_id='A',amount_delta=20),SimpleNamespace(stock_id='B',amount_delta=-2),SimpleNamespace(stock_id='D',amount_delta=4)])
 inner=fixtures(cls);e=E();out=I(cls);out.agg_order_indicators(inner,steps,outer,e,{'pa_config':{'agg':'TWAP','price':'DEAL_PRICE'},'ignored':1});result={'normal':snap(out),'mutated':[vals(x.get_index_data('trade_price')) for x in inner],'calls':e.calls}
 empty=I(cls,{'base_price':{'OLD':7},'base_volume':{'OLD':8},'pa':{'OLD':9}});empty.agg_order_indicators([],[],D(),E());result['empty']=snap(empty)
 inner=fixtures(cls);failed=I(cls,{'base_price':{'OLD':7},'base_volume':{'OLD':8},'pa':{'OLD':9}})
 try:failed.agg_order_indicators(inner,steps,outer,E(True));kind='ok'
 except Exception as error:kind=type(error).__name__
 result['failure']=[kind,snap(failed)]
 inner=fixtures(cls);missing=I(cls)
 try:missing.agg_order_indicators(inner,steps,outer,E(missing=True));result['missing']=['ok',snap(missing)]
 except Exception as error:result['missing']=[type(error).__name__,snap(missing)]
 return result
with warnings.catch_warnings():warnings.simplefilter('ignore');print(json.dumps({'numpy':run(NumpyOrderIndicator),'pandas':run(PandasOrderIndicator)},sort_keys=True,separators=(',',':')))
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
    let actual: Value = serde_json::from_slice(&output.stdout).unwrap();
    for backend in ["numpy", "pandas"] {
        assert_eq!(
            actual[backend]["normal"]["ffr"],
            json!({"index":["A","B","D"],"values":[0.25,0.5,0.0]})
        );
        assert_eq!(
            actual[backend]["normal"]["base_price"],
            json!({"index":["A","B","C"],"values":[106.0,80.0,40.0]})
        );
        assert_eq!(actual[backend]["calls"].as_array().unwrap().len(), 4);
        assert_eq!(
            actual[backend]["mutated"],
            json!([
                {"index":["B","A"],"values":[-3.0,40.0]},
                {"index":["C","A"],"values":[0.0,20.0]}
            ])
        );
        assert_eq!(
            actual[backend]["empty"]["base_price"],
            json!({"index":["OLD"],"values":[7.0]})
        );
        assert_eq!(actual[backend]["empty"]["pa"]["index"], json!([]));
        assert_eq!(actual[backend]["failure"][0], json!("RuntimeError"));
        assert_eq!(
            actual[backend]["failure"][1]["ffr"],
            json!({"index":["A","B","D"],"values":[0.25,0.5,0.0]})
        );
        assert_eq!(
            actual[backend]["failure"][1]["base_price"],
            json!({"index":["OLD"],"values":[7.0]})
        );
    }
    assert_eq!(actual["numpy"]["missing"][0], json!("ValueError"));
    assert_eq!(actual["pandas"]["missing"][0], json!("ok"));
}

#[test]
fn base_volume_price_matches_live_python_source() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib");
    let script = r"
import ast,importlib.util,json,sys,warnings
from types import SimpleNamespace
from typing import *
import numpy as np,pandas as pd
spec=importlib.util.spec_from_file_location('index_data_live',sys.argv[2]);idd=importlib.util.module_from_spec(spec);spec.loader.exec_module(idd)
t=ast.parse(open(sys.argv[1],encoding='utf-8').read(),filename=sys.argv[1]);c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='Indicator');f=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='_get_base_vol_pri');ns={'np':np,'pd':pd,'idd':idd,'OrderDir':object,'BaseTradeDecision':object,'Exchange':object,'Optional':Optional,'Tuple':Tuple};exec(compile(ast.Module(body=[f],type_ignores=[]),sys.argv[1],'exec'),ns);f=ns['_get_base_vol_pri']
class E:
 def __init__(self,p,v=None):self.p=p;self.v=v;self.calls=[]
 def get_deal_price(self,*a,**k):self.calls.append(['p',a[1].isoformat(),a[2].isoformat(),k['direction'],k['method']]);return self.p
 def get_volume(self,*a,**k):self.calls.append(['v',a[1].isoformat(),a[2].isoformat(),k['method']]);return self.v
class R:
 def clip_time_range(self,start_time,end_time):return start_time+pd.Timedelta(minutes=1),end_time-pd.Timedelta(minutes=1)
def enc(v):
 if v is None:return None
 if np.isnan(v):return 'nan'
 if np.isposinf(v):return 'inf'
 if np.isneginf(v):return '-inf'
 return float(v)
def run(p,v=None,cfg={},r=None):
 e=E(p,v);d=SimpleNamespace(trade_range=r)
 try:x=f(None,'A',pd.Timestamp('2024-01-02 09:30'),pd.Timestamp('2024-01-02 10:30'),1,d,e,cfg);result=['ok',[enc(x[0]),enc(x[1])]]
 except Exception as error:result=[type(error).__name__,str(error)]
 return {'result':result,'calls':e.calls}
t0=pd.Timestamp('2024-01-02 09:30');t1=pd.Timestamp('2024-01-02 09:31');t2=pd.Timestamp('2024-01-02 09:32')
with warnings.catch_warnings():
 warnings.simplefilter('ignore')
 out={'scalar':run(np.float32(5.5)),'none':run(None),'threshold':run(idd.SingleData([0,1e-8,np.nextafter(1e-8,np.inf),2,np.nan],[t0,t1,t2,t1+pd.Timedelta(minutes=2),t1+pd.Timedelta(minutes=3)])),'vwap':run(idd.SingleData([2,4],[t1,t2]),idd.SingleData([3,1],[t2,t1]),{'agg':'VwAp','price':'DEAL_PRICE'}),'missing':run(idd.SingleData([2,4],[t1,t2]),idd.SingleData([3],[t1]),{'agg':'vwap'}),'zero':run(idd.SingleData([2,4],[t1,t2]),idd.SingleData([1,-1],[t1,t2]),{'agg':'vwap'}),'range':run(idd.SingleData([2],[t1]),r=R()),'bad_price':run(idd.SingleData([2],[t1]),cfg={'price':'close'}),'bad_agg_empty':run(idd.SingleData([],[]),cfg={'agg':'median'}),'no_volume':run(idd.SingleData([2],[t1]),None,{'agg':'vwap'})}
 print(json.dumps(out,sort_keys=True,separators=(',',':')))
";
    let output = Command::new(std::env::var_os("PYTHON").unwrap_or_else(|| "python".into()))
        .arg("-c")
        .arg(script)
        .arg(root.join("qlib/backtest/report.py"))
        .arg(root.join("qlib/utils/index_data.py"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let actual: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(actual["scalar"]["result"], json!(["ok", [5.5, 1.0]]));
    assert_eq!(actual["none"]["result"], json!(["ok", [null, null]]));
    let next = f64::from_bits(MIN_BASE_PRICE.to_bits() + 1);
    assert_eq!(
        actual["threshold"]["result"][1][0].as_f64().unwrap(),
        f64::midpoint(next, 2.0)
    );
    assert_eq!(actual["threshold"]["result"][1][1], json!(2.0));
    assert_eq!(actual["vwap"]["result"], json!(["ok", [3.5, 4.0]]));
    assert_eq!(actual["missing"]["result"], json!(["ok", [2.0, 3.0]]));
    assert_eq!(actual["zero"]["result"], json!(["ok", ["-inf", 0.0]]));
    assert_eq!(
        actual["range"]["calls"],
        json!([["p", "2024-01-02T09:31:00", "2024-01-02T10:29:00", 1, null]])
    );
    assert_eq!(
        actual["bad_price"]["result"][0],
        json!("NotImplementedError")
    );
    assert_eq!(
        actual["bad_agg_empty"]["result"],
        json!(["ok", [null, null]])
    );
    assert_eq!(actual["no_volume"]["result"][0], json!("AssertionError"));
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "the self-contained live Python differential loads both original storage backends"
)]
fn cross_step_aggregation_matches_live_python_storage_backends() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib");
    let script = r"
import ast,importlib.util,json,sys,warnings
from collections import OrderedDict
from dataclasses import dataclass
from enum import IntEnum
from typing import *
import numpy as np,pandas as pd
spec=importlib.util.spec_from_file_location('index_data_live',sys.argv[4]);idd=importlib.util.module_from_spec(spec);spec.loader.exec_module(idd);SingleData=idd.SingleData
class Logger:pass
def get_module_logger(_):return Logger()
def classes(path,names):
 t=ast.parse(open(path,encoding='utf-8').read(),filename=path);return [next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name==x) for x in names]
nodes=classes(sys.argv[2],['OrderDir','Order'])+classes(sys.argv[3],['BaseSingleMetric','BaseOrderIndicator','SingleMetric','PandasSingleMetric','PandasOrderIndicator','NumpyOrderIndicator']);m=ast.Module(body=nodes,type_ignores=[]);ast.fix_missing_locations(m);exec(compile(m,'live','exec'),globals())
t=ast.parse(open(sys.argv[1],encoding='utf-8').read(),filename=sys.argv[1]);c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='Indicator');nodes=[next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==x) for x in ['_get_base_vol_pri','_agg_base_price']];BaseTradeDecision=Exchange=object;exec(compile(ast.Module(body=nodes,type_ignores=[]),'live','exec'),globals())
class I:
 _get_base_vol_pri=_get_base_vol_pri;_agg_base_price=_agg_base_price
 def __init__(self,cls,rows={}):self.order_indicator=make(cls,rows)
class R:
 def clip_time_range(self,start_time,end_time):return start_time+pd.Timedelta(minutes=1),end_time-pd.Timedelta(minutes=1)
class E:
 def __init__(self,fail=False):self.fail=fail;self.calls=[]
 def get_deal_price(self,inst,start,end,direction,method):
  self.calls.append([inst,start.isoformat(),end.isoformat(),float(direction)])
  if self.fail:raise RuntimeError('boom')
  if direction not in (0,1):raise NotImplementedError('direction')
  return {(31,'B'):80,(30,'B'):120,(30,'D'):40}.get((start.minute,inst))
 def get_volume(self,*a,**k):raise AssertionError('unused')
def make(cls,rows):
 x=cls()
 for k,v in rows.items():x.assign(k,v)
 return x
def vals(x):return {'index':[str(v) for v in x.index.tolist()],'values':[None if np.isnan(v) else ('inf' if np.isposinf(v) else float(v)) for v in x.data]}
def snap(i):return {k:vals(i.order_indicator.get_index_data(k)) for k in i.order_indicator.data}
def run(cls):
 t0=pd.Timestamp('2024-01-02 09:30');t1=pd.Timestamp('2024-01-02 10:30');steps=[(type('D',(),{'trade_range':R()})(),t0,t1),(type('D',(),{'trade_range':None})(),t1,t1+pd.Timedelta(hours=1))]
 a=make(cls,{'base_price':{'A':100,'B':np.nan,'C':np.inf,'E':50},'base_volume':{'A':2,'B':9,'C':0}});b=make(cls,{'base_price':{'A':110},'base_volume':{'A':3}});e=E();out=I(cls,{'trade_dir':{'B':1,'A':0,'D':0,'C':1,'E':0}});out._agg_base_price([a,b],steps,e)
 empty=I(cls,{'trade_dir':{},'base_price':{'OLD':7},'base_volume':{'OLD':8}});empty._agg_base_price([a],steps,E())
 no_pairs=I(cls,{'trade_dir':{'A':1},'base_price':{'OLD':7}});no_pairs._agg_base_price([],steps,E())
 short=I(cls,{'trade_dir':{'A':1}});short._agg_base_price([a,b],steps[:1],E())
 failed=I(cls,{'trade_dir':{'A':1},'base_price':{'OLD':7},'base_volume':{'OLD':8}})
 try:failed._agg_base_price([make(cls,{})],steps[1:],E(True));failure='ok'
 except Exception as error:failure=type(error).__name__
 return {'normal':snap(out),'calls':e.calls,'empty':snap(empty),'no_pairs':snap(no_pairs),'short':snap(short),'failure':[failure,snap(failed)]}
with warnings.catch_warnings():warnings.simplefilter('ignore');print(json.dumps({'numpy':run(NumpyOrderIndicator),'pandas':run(PandasOrderIndicator)},sort_keys=True,separators=(',',':')))
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
    let actual: Value = serde_json::from_slice(&output.stdout).unwrap();
    for backend in ["numpy", "pandas"] {
        assert_eq!(
            actual[backend]["normal"]["base_price"],
            json!({"index":["A","B","C","D","E"],"values":[106.0,100.0,null,40.0,null]})
        );
        assert_eq!(
            actual[backend]["normal"]["base_volume"],
            json!({"index":["A","B","C","D","E"],"values":[5.0,2.0,0.0,1.0,0.0]})
        );
        assert_eq!(actual[backend]["calls"].as_array().unwrap().len(), 6);
        assert_eq!(
            actual[backend]["empty"]["base_price"],
            json!({"index":["OLD"],"values":[7.0]})
        );
        assert_eq!(
            actual[backend]["no_pairs"]["base_price"]["index"],
            json!([])
        );
        assert_eq!(
            actual[backend]["short"]["base_price"],
            json!({"index":["A"],"values":[100.0]})
        );
        assert_eq!(actual[backend]["failure"][0], json!("RuntimeError"));
        assert_eq!(
            actual[backend]["failure"][1]["base_price"],
            json!({"index":["OLD"],"values":[7.0]})
        );
    }
}
