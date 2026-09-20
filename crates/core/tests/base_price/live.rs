use super::*;
use domain_core::base_price::LiveBasePriceStep;
use domain_core::decision_construction::{
    ConstructedDecisionBase, SharedOrderDecisionConstruction,
};
use domain_core::decision_update::{SharedDecisionUpdateStrategy, SharedLiveDecision};
use std::sync::{Arc, RwLock};

#[path = "live_failures.rs"]
mod failures;

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

fn outer_decision() -> SharedLiveDecision<Origin, ()> {
    let mut state = SharedOrderDecisionConstruction::new(Arc::new(Origin));
    state.details = Some(());
    state.base = Some(ConstructedDecisionBase {
        start_time: timestamp("2024-01-02 09:30:00"),
        end_time: timestamp("2024-01-02 11:30:00"),
        trade_range: None,
    });
    state.orders = Some(Arc::new(RwLock::new(
        outer_orders()
            .into_iter()
            .map(|order| {
                domain_core::decision_construction::DecisionOrderItem::Order(Arc::new(RwLock::new(
                    order,
                )))
            })
            .collect(),
    )));
    Arc::new(RwLock::new(state))
}

fn compare_live_pipeline<S: IndicatorStore>() {
    let mut reference = Indicator::<S>::default();
    let mut expected_inner = orchestration_inner::<S>();
    let provider = Provider::new(Some(MarketDataValue::Scalar(10.0)), None);
    reference
        .aggregate_order_indicators(
            &mut expected_inner,
            &outer_orders(),
            &orchestration_steps(),
            &provider,
            OrderIndicatorAggregationConfig::default(),
        )
        .unwrap();
    let inner: Vec<_> = orchestration_inner::<S>()
        .into_iter()
        .map(|store| Arc::new(RwLock::new(store)))
        .collect();
    let outer: domain_core::decision_update::LiveDecisionHandle = outer_decision();
    let steps: Vec<_> = orchestration_steps()
        .into_iter()
        .map(|step| LiveBasePriceStep {
            decision: outer.clone(),
            start_time: step.start_time,
            end_time: step.end_time,
        })
        .collect();
    let mut actual = Indicator::<S>::default();
    actual
        .aggregate_live_order_indicators(
            &inner,
            &outer,
            &steps,
            &provider,
            OrderIndicatorAggregationConfig::default(),
        )
        .unwrap();
    let actual = actual.order_snapshot().unwrap();
    let expected = reference.order_snapshot().unwrap();
    assert_eq!(
        actual.keys().collect::<Vec<_>>(),
        expected.keys().collect::<Vec<_>>()
    );
    for (name, value) in expected {
        assert_eq!(actual[&name].index(), value.index());
        assert_eq!(
            actual[&name]
                .values()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            value
                .values()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn full_live_pipeline_matches_both_existing_storage_backends() {
    compare_live_pipeline::<NumpyOrderIndicator>();
    compare_live_pipeline::<PandasOrderIndicator>();
}

#[test]
fn live_target_duplicates_keep_first_key_order_and_last_signed_amount() {
    use domain_core::decision_construction::DecisionOrderItem;
    let typed = outer_decision();
    let orders = typed.read().unwrap().orders.clone().unwrap();
    orders
        .write()
        .unwrap()
        .push(DecisionOrderItem::Order(Arc::new(RwLock::new(Order::new(
            "A",
            3.0,
            OrderDir::Sell,
            None,
            None,
        )))));
    let outer: domain_core::decision_update::LiveDecisionHandle = typed;
    let provider = Provider::new(None, None);
    let mut output = Indicator::new();
    output
        .aggregate_live_order_indicators(
            &[],
            &outer,
            &[],
            &provider,
            OrderIndicatorAggregationConfig::default(),
        )
        .unwrap();
    let snapshot = output.order_snapshot().unwrap();
    assert_eq!(snapshot["amount"].index(), ["A", "B", "D"]);
    assert_eq!(snapshot["amount"].values(), [-3.0, -2.0, 4.0]);
    orders.write().unwrap().clear();
    output
        .aggregate_live_order_indicators(
            &[],
            &outer,
            &[],
            &provider,
            OrderIndicatorAggregationConfig::default(),
        )
        .unwrap();
    assert!(output.order_snapshot().unwrap()["amount"].is_empty());
    assert!(provider.calls.lock().unwrap().is_empty());
}

#[test]
fn account_maps_live_decision_failure_without_recording_the_failed_bar() {
    use domain_core::account::LiveNestedAccountIndicatorUpdate;
    use domain_core::{
        Account, AccountError, AccountIndicatorMode, AccountIndicatorUpdate, IndicatorConfig,
        InfinitePosition,
    };
    let typed = outer_decision();
    typed.write().unwrap().orders = None;
    let outer: domain_core::decision_update::LiveDecisionHandle = typed;
    let provider = Provider::new(None, None);
    let at = timestamp("2024-01-02 09:30:00");
    let mut account = Account::new(InfinitePosition, false);
    let error = account
        .update_indicator(AccountIndicatorUpdate {
            trade_start_time: at,
            mode: AccountIndicatorMode::LiveNested(LiveNestedAccountIndicatorUpdate {
                inner: &[],
                outer_decision: &outer,
                steps: &[],
                provider: &provider,
                config: OrderIndicatorAggregationConfig::default(),
            }),
            calculation: IndicatorConfig::default(),
            show_indicator: false,
        })
        .unwrap_err();
    assert!(
        matches!(error, AccountError::Indicator(ref error) if error.message == "decision order list has not been initialized")
    );
    assert!(
        account
            .indicator()
            .read()
            .unwrap()
            .recorded_trade_indicator(at)
            .is_none()
    );
    assert!(provider.calls.lock().unwrap().is_empty());
}

#[test]
fn live_outer_read_failure_keeps_trade_stage_but_does_not_publish_target() {
    use domain_core::decision_construction::{DecisionAccessError, DecisionOrderItem};
    use domain_core::decision_update::{LiveDecisionAccessError, LiveDecisionHandle};
    for case in ["missing", "invalid", "list_poison", "order_poison"] {
        let typed = outer_decision();
        let list = typed.read().unwrap().orders.clone().unwrap();
        match case {
            "missing" => typed.write().unwrap().orders = None,
            "invalid" => list
                .write()
                .unwrap()
                .push(DecisionOrderItem::Other(Arc::new(1))),
            "list_poison" => {
                assert!(
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let _guard = list.write().unwrap();
                        panic!("poison list");
                    }))
                    .is_err()
                );
            }
            "order_poison" => {
                let items = list.read().unwrap();
                let DecisionOrderItem::Order(order) = &items[0] else {
                    panic!("order");
                };
                assert!(
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let _guard = order.write().unwrap();
                        panic!("poison order");
                    }))
                    .is_err()
                );
            }
            _ => unreachable!(),
        }
        let outer: LiveDecisionHandle = typed;
        let inner: Vec<_> = orchestration_inner::<NumpyOrderIndicator>()
            .into_iter()
            .map(|store| Arc::new(RwLock::new(store)))
            .collect();
        let mut output = Indicator::new();
        let provider = Provider::new(None, None);
        let error = output
            .aggregate_live_order_indicators(
                &inner,
                &outer,
                &[],
                &provider,
                OrderIndicatorAggregationConfig::default(),
            )
            .unwrap_err();
        let expected = match case {
            "missing" => DecisionAccessError::MissingOrders,
            "invalid" => DecisionAccessError::InvalidOrder(3),
            "list_poison" => DecisionAccessError::ListPoisoned,
            "order_poison" => DecisionAccessError::OrderPoisoned(0),
            _ => unreachable!(),
        };
        assert_eq!(
            error,
            AggregateOrderIndicatorsError::Decision(LiveDecisionAccessError::Access(expected))
        );
        let metrics = output.order_snapshot().unwrap();
        assert!(metrics.contains_key("trade_price"));
        assert!(!metrics.contains_key("amount"));
        assert!(!metrics.contains_key("ffr"));
        assert!(provider.calls.lock().unwrap().is_empty());
    }
}

struct NoMarket;
impl domain_core::AccountBarMarket for NoMarket {
    fn is_suspended(
        &self,
        _: &str,
        _: TimeRange,
    ) -> Result<bool, domain_core::AccountBarMarketError> {
        panic!("infinite account");
    }
    fn close(&self, _: &str, _: TimeRange) -> Result<f64, domain_core::AccountBarMarketError> {
        panic!("infinite account");
    }
}

#[test]
fn account_live_nested_dispatch_validates_missing_inputs_and_records_native_metrics() {
    use domain_core::NestedExecutorAccount;
    use domain_core::{
        Account, AccountBarEndMode, AccountBarEndUpdate, AccountError, IndicatorConfig,
        InfinitePosition,
    };
    let at = timestamp("2024-01-02 09:30:00");
    let mut account = Account::new(InfinitePosition, false);
    assert!(matches!(
        account.update_bar_end(AccountBarEndUpdate {
            trade_start_time: at,
            trade_end_time: at,
            market: &NoMarket,
            mode: AccountBarEndMode::LiveNested(None),
            calculation: IndicatorConfig::default(),
            show_indicator: false,
        }),
        Err(AccountError::MissingInnerOrderIndicators)
    ));
    let outer: domain_core::decision_update::LiveDecisionHandle = outer_decision();
    let inner: Vec<_> = orchestration_inner::<NumpyOrderIndicator>()
        .into_iter()
        .map(|store| Arc::new(RwLock::new(store)))
        .collect();
    let provider = Provider::new(Some(MarketDataValue::Scalar(10.0)), None);
    let steps: Vec<_> = orchestration_steps()
        .into_iter()
        .map(|step| LiveBasePriceStep {
            decision: outer.clone(),
            start_time: step.start_time,
            end_time: step.end_time,
        })
        .collect();
    domain_core::NestedAccountAdapter::new(&mut account, &NoMarket, &provider, false)
        .update_live_bar_end(domain_core::nested_executor_lifecycle::LiveNestedBarEnd {
            trade_start_time: at,
            trade_end_time: at,
            inner_order_indicators: &inner,
            outer_decision: &outer,
            steps: &steps,
            aggregation_config: OrderIndicatorAggregationConfig::default(),
            indicator_config: IndicatorConfig::default(),
        })
        .unwrap();
    assert!(
        account
            .indicator()
            .read()
            .unwrap()
            .recorded_trade_indicator(at)
            .is_some()
    );
    assert_eq!(
        account
            .order_indicator_snapshot()
            .unwrap()
            .metric_snapshot("amount")
            .unwrap()
            .values(),
        [20.0, -2.0, 4.0]
    );
}

struct MutatingProvider {
    decision: SharedLiveDecision<Origin, ()>,
    calls: Mutex<Vec<Value>>,
}
impl BasePriceDataProvider for MutatingProvider {
    fn deal_price(
        &self,
        stock: &str,
        range: TimeRange,
        _: OrderDir,
    ) -> Result<Option<MarketDataValue>, BasePriceProviderError> {
        self.calls.lock().unwrap().push(json!([
            stock,
            range.start.unwrap().to_string(),
            range.end.unwrap().to_string()
        ]));
        self.decision
            .try_write()
            .expect("decision guard must not span quote callback")
            .base
            .as_mut()
            .unwrap()
            .trade_range = Some(Arc::new(TradeRangeByTime::parse("10:00", "11:00").unwrap()));
        Ok(None)
    }
    fn volume(
        &self,
        _: &str,
        _: TimeRange,
    ) -> Result<Option<MarketDataValue>, BasePriceProviderError> {
        panic!("TWAP");
    }
}

#[test]
fn live_baseline_rereads_range_per_missing_stock_and_skips_unneeded_decisions() {
    let source = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/live_base_range_contract.py"
        ))
        .output()
        .unwrap();
    assert!(
        source.status.success(),
        "{}",
        String::from_utf8_lossy(&source.stderr)
    );
    let expected: Value = serde_json::from_slice(&source.stdout).unwrap();
    let start = timestamp("2024-01-02 09:00:00");
    let end = timestamp("2024-01-02 11:00:00");
    let mut state = SharedOrderDecisionConstruction::new(Arc::new(Origin));
    state.details = Some(());
    state.base = Some(ConstructedDecisionBase {
        start_time: start,
        end_time: end,
        trade_range: None,
    });
    let decision = Arc::new(RwLock::new(state));
    let provider = MutatingProvider {
        decision: decision.clone(),
        calls: Mutex::default(),
    };
    let step = LiveBasePriceStep {
        decision: decision.clone(),
        start_time: start,
        end_time: end,
    };
    let raw = Arc::new(RwLock::new(NumpyOrderIndicator::default()));
    let mut output = Indicator::new();
    assign_metric(
        &mut *output.order_indicator_mut().unwrap(),
        "trade_dir",
        &[("A", 1.0), ("B", 1.0)],
    );
    output
        .aggregate_live_base_price(
            std::slice::from_ref(&raw),
            &[step],
            &provider,
            BasePriceConfig::default(),
        )
        .unwrap();
    assert_eq!(json!(*provider.calls.lock().unwrap()), expected);
    assert!(decision.try_write().is_ok());
    // Fully observed baselines must not access an invalid decision's metadata.
    decision.write().unwrap().base = None;
    assign_metric(
        &mut *raw.write().unwrap(),
        "base_price",
        &[("A", 7.0), ("B", 9.0)],
    );
    assign_metric(
        &mut *raw.write().unwrap(),
        "base_volume",
        &[("A", 1.0), ("B", 2.0)],
    );
    let steps = [LiveBasePriceStep {
        decision: decision.clone(),
        start_time: start,
        end_time: end,
    }];
    output
        .aggregate_live_base_price(
            std::slice::from_ref(&raw),
            &steps,
            &provider,
            BasePriceConfig::default(),
        )
        .unwrap();
    assert_eq!(
        output.order_snapshot().unwrap()["base_price"].values(),
        [7.0, 9.0]
    );
    assert_eq!(provider.calls.lock().unwrap().len(), 2);
    raw.write()
        .unwrap()
        .assign_snapshot("base_price", MetricSnapshot::default());
    assert!(matches!(
        output.aggregate_live_base_price(&[raw], &steps, &provider, BasePriceConfig::default()),
        Err(AggregateBasePriceError::BasePrice(
            BasePriceError::Decision(_)
        ))
    ));
    assert_eq!(provider.calls.lock().unwrap().len(), 2);
}
