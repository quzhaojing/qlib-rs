use super::*;
use domain_core::SharedOrderIndicator;
use std::sync::{Arc, RwLock};

fn shared<S>(store: S) -> SharedOrderIndicator<S> {
    Arc::new(RwLock::new(store))
}

fn metrics(store: &dyn IndicatorStoreAccess) -> Value {
    let rows: serde_json::Map<_, _> = store
        .metric_names()
        .map(|name| {
            let value = store.metric_snapshot(name).unwrap();
            (
                name.to_owned(),
                json!({"index": value.index(), "values": value.values()}),
            )
        })
        .collect();
    Value::Object(rows)
}

struct LaterStepProvider<S> {
    input: SharedOrderIndicator<S>,
    output: SharedOrderIndicator<S>,
    calls: Mutex<usize>,
}

impl<S: IndicatorStore> BasePriceDataProvider for LaterStepProvider<S> {
    fn deal_price(
        &self,
        _: &str,
        _: TimeRange,
        _: OrderDir,
    ) -> Result<Option<MarketDataValue>, BasePriceProviderError> {
        *self.calls.lock().unwrap() += 1;
        assert!(
            self.output.try_write().is_ok(),
            "output guard escaped into callback"
        );
        let mut input = self
            .input
            .try_write()
            .expect("input guard escaped into callback");
        assign_metric(&mut *input, "base_price", &[("A", 200.0)]);
        assign_metric(&mut *input, "base_volume", &[("A", 3.0)]);
        Ok(Some(MarketDataValue::Scalar(100.0)))
    }

    fn volume(
        &self,
        _: &str,
        _: TimeRange,
    ) -> Result<Option<MarketDataValue>, BasePriceProviderError> {
        panic!("TWAP does not query volume")
    }
}

fn pipeline_contract<S: IndicatorStore>() -> Value {
    let mut result = serde_json::Map::new();
    for self_alias in [false, true] {
        let mut output = Indicator::<S>::default();
        let raw = if self_alias {
            output.order_indicator().clone()
        } else {
            shared(S::default())
        };
        for (name, value) in [
            ("inner_amount", 4.0),
            ("deal_amount", 2.0),
            ("trade_price", 3.0),
            ("trade_value", 6.0),
            ("trade_cost", 1.0),
            ("trade_dir", 1.0),
            ("base_price", 100.0),
            ("base_volume", 2.0),
        ] {
            assign_metric(&mut *raw.write().unwrap(), name, &[("A", value)]);
        }
        let frozen = metrics(&*raw.read().unwrap());
        let provider = Provider::new(None, None);
        output
            .aggregate_shared_order_indicators(
                &[raw.clone(), raw.clone()],
                &[Order::new("A", 8.0, OrderDir::Buy, None, None)],
                &orchestration_steps(),
                &provider,
                OrderIndicatorAggregationConfig::default(),
            )
            .unwrap();
        assert!(provider.calls.lock().unwrap().is_empty());
        assert_eq!(frozen["trade_price"]["values"], json!([3.0]));
        if !self_alias {
            assert_eq!(
                raw.read()
                    .unwrap()
                    .metric_snapshot("trade_price")
                    .unwrap()
                    .values(),
                [12.0]
            );
        }
        result.insert(
            if self_alias { "self" } else { "duplicate" }.into(),
            metrics(&*output.order_indicator().read().unwrap()),
        );
        drop(output);
        assert!(raw.try_write().is_ok());
    }
    for self_alias in [false, true] {
        let mut output = Indicator::<S>::default();
        assign_metric(
            &mut *output.order_indicator_mut().unwrap(),
            "trade_dir",
            &[("A", 1.0)],
        );
        let raw = if self_alias {
            output.order_indicator().clone()
        } else {
            shared(S::default())
        };
        assign_metric(&mut *raw.write().unwrap(), "base_price", &[("A", f64::NAN)]);
        assign_metric(&mut *raw.write().unwrap(), "base_volume", &[("A", 9.0)]);
        let provider = LaterStepProvider {
            input: raw.clone(),
            output: output.order_indicator().clone(),
            calls: Mutex::new(0),
        };
        output
            .aggregate_shared_base_price(
                &[raw.clone(), raw],
                &orchestration_steps(),
                &provider,
                BasePriceConfig::default(),
            )
            .unwrap();
        let values = output.order_snapshot().unwrap();
        assert_eq!(values["base_price"].values(), [175.0]);
        assert_eq!(values["base_volume"].values(), [4.0]);
        result.insert(if self_alias { "callback_self" } else { "callback" }.into(), json!({
            "price": {"index": values["base_price"].index(), "values": values["base_price"].values()},
            "volume": {"index": values["base_volume"].index(), "values": values["base_volume"].values()},
            "calls": *provider.calls.lock().unwrap(),
        }));
    }
    Value::Object(result)
}

#[test]
fn shared_full_pipeline_and_lazy_step_reads_match_actual_python_aliases() {
    let output = Command::new("python")
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
    assert_eq!(
        pipeline_contract::<NumpyOrderIndicator>(),
        source["numpy"]["shared_pipeline"]
    );
    assert_eq!(
        pipeline_contract::<PandasOrderIndicator>(),
        source["pandas"]["shared_pipeline"]
    );
}

fn poison<S: IndicatorStore + 'static>(store: &SharedOrderIndicator<S>) {
    let store = store.clone();
    assert!(
        std::thread::spawn(move || {
            let _guard = store.write().unwrap();
            panic!("intentional input poison");
        })
        .join()
        .is_err()
    );
}

fn shared_failure_contract<S: IndicatorStore + 'static>() {
    let provider = Provider::new(None, None);
    let poisoned = shared(S::default());
    poison(&poisoned);
    let mut output = Indicator::<S>::default();
    // Empty direction and zip truncation never read an unused poisoned input.
    output
        .aggregate_shared_base_price(
            std::slice::from_ref(&poisoned),
            &orchestration_steps(),
            &provider,
            BasePriceConfig::default(),
        )
        .unwrap();
    assign_metric(
        &mut *output.order_indicator_mut().unwrap(),
        "trade_dir",
        &[("A", 1.0)],
    );
    output
        .aggregate_shared_base_price(
            std::slice::from_ref(&poisoned),
            &[],
            &provider,
            BasePriceConfig::default(),
        )
        .unwrap();
    assert!(output.order_snapshot().unwrap()["base_price"].is_empty());
    assert_eq!(
        output.aggregate_shared_base_price(
            std::slice::from_ref(&poisoned),
            &orchestration_steps(),
            &provider,
            BasePriceConfig::default()
        ),
        Err(AggregateBasePriceError::Indicator(
            IndicatorError::OrderStorePoisoned
        ))
    );
    assert!(output.order_indicator().try_write().is_ok());
    assert!(provider.calls.lock().unwrap().is_empty());
    assert_eq!(
        output.aggregate_shared_order_indicators(
            &[poisoned],
            &[],
            &[],
            &provider,
            OrderIndicatorAggregationConfig::default()
        ),
        Err(AggregateOrderIndicatorsError::Indicator(
            IndicatorError::OrderStorePoisoned
        ))
    );
    poison(output.order_indicator());
    assert_eq!(
        output.aggregate_shared_base_price(&[], &[], &provider, BasePriceConfig::default()),
        Err(AggregateBasePriceError::Indicator(
            IndicatorError::OrderStorePoisoned
        ))
    );

    let inner: Vec<_> = orchestration_inner::<S>().into_iter().map(shared).collect();
    let mut output = Indicator::<S>::default();
    let mut failed = MappingProvider::new(&[]);
    failed.failure = Some((timestamp("2024-01-02 09:30:00"), "B".into()));
    assert!(matches!(
        output.aggregate_shared_order_indicators(
            &inner,
            &outer_orders(),
            &orchestration_steps(),
            &failed,
            OrderIndicatorAggregationConfig::default()
        ),
        Err(AggregateOrderIndicatorsError::BasePrice(_))
    ));
    assert_eq!(
        output.order_snapshot().unwrap()["ffr"].values(),
        [0.25, 0.5, 0.0]
    );
    assert_eq!(
        inner[0]
            .read()
            .unwrap()
            .metric_snapshot("trade_price")
            .unwrap()
            .values(),
        [-3.0, 40.0]
    );
    assert!(output.order_snapshot().unwrap().get("pa").is_none());
    assert!(inner.iter().all(|store| store.try_write().is_ok()));
    let mut empty = Indicator::<S>::default();
    empty
        .aggregate_shared_order_indicators(
            &[],
            &[],
            &[],
            &provider,
            OrderIndicatorAggregationConfig::default(),
        )
        .unwrap();
    assert!(empty.order_snapshot().unwrap()["pa"].is_empty());
}

#[test]
fn shared_pipeline_preserves_lazy_poison_errors_and_completed_stages() {
    shared_failure_contract::<NumpyOrderIndicator>();
    shared_failure_contract::<PandasOrderIndicator>();
    let mut output = Indicator::new();
    let inner: Vec<_> = orchestration_inner::<NumpyOrderIndicator>()
        .into_iter()
        .map(shared)
        .collect();
    let provider = MappingProvider::new(&[]);
    assert!(matches!(
        output.aggregate_shared_order_indicators(
            &inner,
            &outer_orders(),
            &orchestration_steps(),
            &provider,
            OrderIndicatorAggregationConfig::default()
        ),
        Err(AggregateOrderIndicatorsError::Indicator(
            IndicatorError::IndexMismatch { .. }
        ))
    ));
    assert_eq!(
        output.order_snapshot().unwrap()["base_price"].index(),
        ["A"]
    );
}
