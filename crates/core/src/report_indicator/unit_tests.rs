use super::{IndicatorAggregationMode, IndicatorError, MetricSnapshot, aggregate_columns};

#[test]
fn poisoned_output_between_trade_and_target_keeps_completed_trade_stage() {
    use super::{Indicator, NumpyOrderIndicator, OrderIndicatorAggregationConfig};
    use crate::{
        AggregateOrderIndicatorsError, BasePriceDataProvider, BasePriceProviderError,
        BasePriceStep, MarketDataValue, OrderDir, TimeRange,
    };

    struct UnusedProvider(std::sync::atomic::AtomicUsize);
    impl BasePriceDataProvider for UnusedProvider {
        fn deal_price(
            &self,
            _: &str,
            _: TimeRange,
            _: OrderDir,
        ) -> Result<Option<MarketDataValue>, BasePriceProviderError> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Err(BasePriceProviderError::Provider {
                message: "price unavailable".into(),
            })
        }
        fn volume(
            &self,
            _: &str,
            _: TimeRange,
        ) -> Result<Option<MarketDataValue>, BasePriceProviderError> {
            panic!("target lock failure must precede volume lookup")
        }
    }

    let provider = UnusedProvider(std::sync::atomic::AtomicUsize::new(0));
    let config = OrderIndicatorAggregationConfig::default();
    let mut success = Indicator::<NumpyOrderIndicator>::default();
    success.aggregate_shared_order_trade_info(&[]).unwrap();
    success
        .finish_shared_order_indicators(&[], &[], &[], &provider, config)
        .unwrap();
    assert!(success.order_snapshot().unwrap()["pa"].is_empty());

    // Seed the completed trade stage and verify subsequent failure boundaries too.
    for name in ["trade_dir", "trade_price", "deal_amount"] {
        success
            .assign(
                name,
                MetricSnapshot::try_new(vec!["A".into()], vec![1.0]).unwrap(),
            )
            .unwrap();
    }
    let start = chrono::NaiveDate::from_ymd_opt(2024, 1, 2)
        .unwrap()
        .and_hms_opt(9, 30, 0)
        .unwrap();
    let steps = [BasePriceStep {
        start_time: start,
        end_time: start,
        trade_range: None,
    }];
    let input = std::sync::Arc::new(std::sync::RwLock::new(NumpyOrderIndicator::default()));
    assert!(matches!(
        success.finish_shared_order_indicators(&[input], &[], &steps, &provider, config),
        Err(AggregateOrderIndicatorsError::BasePrice(_))
    ));
    assert_eq!(provider.0.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert!(matches!(
        success.finish_shared_order_indicators(&[], &[], &[], &provider, config),
        Err(AggregateOrderIndicatorsError::Indicator(
            IndicatorError::IndexMismatch { .. }
        ))
    ));
    assert!(success.order_snapshot().unwrap()["base_price"].is_empty());

    let mut output = Indicator::<NumpyOrderIndicator>::default();
    output.aggregate_shared_order_trade_info(&[]).unwrap();
    let completed = output.order_snapshot().unwrap();
    assert_eq!(completed.len(), 6);
    let retained = output.order_indicator().clone();
    let poison = retained.clone();
    assert!(
        std::thread::spawn(move || {
            let _guard = poison.write().unwrap();
            panic!("poison output at the explicit post-trade boundary");
        })
        .join()
        .is_err()
    );
    assert_eq!(
        output.finish_shared_order_indicators(
            &[],
            &[],
            &[],
            &provider,
            OrderIndicatorAggregationConfig::default()
        ),
        Err(AggregateOrderIndicatorsError::Indicator(
            IndicatorError::OrderStorePoisoned
        ))
    );
    let store = retained.read().unwrap_err().into_inner();
    for (name, expected) in completed {
        assert_eq!(
            super::IndicatorStoreAccess::metric_snapshot(&*store, &name),
            Some(expected)
        );
    }
    assert!(super::IndicatorStoreAccess::metric_snapshot(&*store, "amount").is_none());
    assert!(super::IndicatorStoreAccess::metric_snapshot(&*store, "ffr").is_none());
    assert_eq!(provider.0.load(std::sync::atomic::Ordering::SeqCst), 1);
}

#[test]
fn column_writer_failure_keeps_exactly_the_preceding_assignments() {
    let names = [
        "inner_amount",
        "deal_amount",
        "trade_price",
        "trade_value",
        "trade_cost",
        "trade_dir",
        "trade_price",
        "trade_dir",
    ];
    let cases = [
        IndicatorAggregationMode::DenseZeroFill,
        IndicatorAggregationMode::PandasFillValue,
    ]
    .into_iter()
    .flat_map(|mode| {
        let initial_read = usize::from(mode == IndicatorAggregationMode::DenseZeroFill);
        (0..8)
            .map(move |index| (mode, None, Some(index)))
            .chain((0..6 + initial_read).map(move |index| (mode, Some(index), None)))
            .chain(std::iter::once((mode, None, None)))
    });
    for (mode, fail_read, fail_write) in cases {
        let mut writes = Vec::new();
        let mut reads = 0;
        let result = aggregate_columns(
            mode,
            1,
            |_, name| {
                if fail_read == Some(reads) {
                    return Err(IndicatorError::MissingMetric(name.to_owned()));
                }
                reads += 1;
                MetricSnapshot::try_new(vec!["A".to_owned()], vec![2.0])
            },
            |name, metric| {
                if fail_write == Some(writes.len()) {
                    return Err(IndicatorError::OrderStorePoisoned);
                }
                writes.push((name.to_owned(), metric));
                Ok(())
            },
        );
        let expected_writes = if let Some(index) = fail_read {
            let column =
                index.saturating_sub(usize::from(mode == IndicatorAggregationMode::DenseZeroFill));
            assert_eq!(
                result,
                Err(IndicatorError::MissingMetric(names[column].to_owned()))
            );
            column
        } else if let Some(index) = fail_write {
            assert_eq!(result, Err(IndicatorError::OrderStorePoisoned));
            index
        } else {
            assert_eq!(result, Ok(()));
            names.len()
        };
        assert_eq!(
            writes
                .iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>(),
            names[..expected_writes]
        );
        if expected_writes >= 7 {
            assert_eq!(writes[2].1.values(), &[2.0]);
            assert_eq!(writes[6].1.values(), &[1.0]);
        }
    }
}
