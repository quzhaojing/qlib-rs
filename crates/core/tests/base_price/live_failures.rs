use super::*;
use domain_core::decision_update::{
    LiveDecision, LiveDecisionAccessError, LiveDecisionHandle, SharedDecisionUpdateError,
};
use domain_core::nested_executor_lifecycle::LiveNestedBarEnd;
use domain_core::{
    Account, IndicatorConfig, InfinitePosition, NestedBarEnd, NestedExecutorAccount,
    NestedExecutorAccountError,
};

struct LegacyAccount;
impl NestedExecutorAccount for LegacyAccount {
    fn settle_start(&mut self, _: &str) -> Result<(), NestedExecutorAccountError> {
        panic!("unsupported update must not settle")
    }
    fn settle_commit(&mut self) -> Result<(), NestedExecutorAccountError> {
        panic!("unsupported update must not commit")
    }
    fn update_bar_end(&mut self, _: NestedBarEnd<'_>) -> Result<(), NestedExecutorAccountError> {
        panic!("must not convert to legacy decisions")
    }
}

#[test]
fn live_account_adapters_preserve_errors_and_reject_unsupported_capabilities() {
    let at = timestamp("2024-01-02 09:30:00");
    let typed = outer_decision();
    typed.write().unwrap().orders = None;
    let outer: LiveDecisionHandle = typed;
    let bar = || LiveNestedBarEnd {
        trade_start_time: at,
        trade_end_time: at,
        outer_decision: &outer,
        inner_order_indicators: &[],
        steps: &[],
        indicator_config: IndicatorConfig::default(),
        aggregation_config: OrderIndicatorAggregationConfig::default(),
    };
    assert_eq!(
        LegacyAccount
            .update_live_bar_end(bar())
            .unwrap_err()
            .message,
        "nested account does not support live decisions"
    );
    let provider = Provider::new(None, None);
    let mut account = Account::new(InfinitePosition, false);
    let failure = domain_core::NestedAccountAdapter::new(&mut account, &NoMarket, &provider, false)
        .update_live_bar_end(bar())
        .unwrap_err();
    assert_eq!(
        failure.message,
        "account indicator plugin error: decision order list has not been initialized"
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
    let account = Arc::new(Mutex::new(account));
    let mut adapter = domain_core::SharedNestedAccountAdapter::new(
        account.clone(),
        Arc::new(NoMarket),
        Arc::new(Provider::new(None, None)),
        false,
    );
    assert_eq!(adapter.update_live_bar_end(bar()).unwrap_err(), failure);
    assert!(account.try_lock().is_ok());
    let poisoned = account.clone();
    assert!(
        std::thread::spawn(move || {
            let _guard = poisoned.lock().unwrap();
            panic!("poison shared account");
        })
        .join()
        .is_err()
    );
    assert_eq!(
        adapter.update_live_bar_end(bar()).unwrap_err().message,
        "shared nested account lock poisoned"
    );
}

struct PoisonOnOrders {
    inner: LiveDecisionHandle,
    store: domain_core::SharedOrderIndicator<NumpyOrderIndicator>,
}
impl LiveDecision for PoisonOnOrders {
    fn total_step(
        &self,
    ) -> Result<domain_core::decision_construction::DecisionTotalStep, LiveDecisionAccessError>
    {
        self.inner.total_step()
    }
    fn base(&self) -> Result<ConstructedDecisionBase, LiveDecisionAccessError> {
        self.inner.base()
    }
    fn inherit_range(
        &self,
        range: Option<domain_core::SharedTradeRange>,
    ) -> Result<(), LiveDecisionAccessError> {
        self.inner.inherit_range(range)
    }
    fn is_empty(&self) -> Result<bool, LiveDecisionAccessError> {
        self.inner.is_empty()
    }
    fn update(
        self: Arc<Self>,
        calendar: &dyn domain_core::DecisionUpdateCalendar,
    ) -> Result<Option<LiveDecisionHandle>, SharedDecisionUpdateError> {
        self.inner.clone().update(calendar)
    }
    fn orders(
        &self,
    ) -> Result<domain_core::decision_construction::SharedDecisionOrders, LiveDecisionAccessError>
    {
        let store = self.store.clone();
        assert!(
            std::thread::spawn(move || {
                let _guard = store
                    .try_write()
                    .expect("no aggregation guard at decision callback");
                panic!("poison between aggregation stages");
            })
            .join()
            .is_err()
        );
        self.inner.orders()
    }
}

#[test]
fn live_full_aggregation_stops_at_trade_target_and_baseline_failures() {
    for stage in ["trade", "target", "baseline"] {
        let mut output = Indicator::new();
        let inner: Vec<_> = orchestration_inner::<NumpyOrderIndicator>()
            .into_iter()
            .map(|store| Arc::new(RwLock::new(store)))
            .collect();
        let outer: LiveDecisionHandle = outer_decision();
        let outer: LiveDecisionHandle = if stage == "trade" {
            let store = inner[0].clone();
            assert!(
                std::thread::spawn(move || {
                    let _guard = store.write().unwrap();
                    panic!("poison initial inner store");
                })
                .join()
                .is_err()
            );
            outer
        } else {
            Arc::new(PoisonOnOrders {
                inner: outer,
                store: if stage == "target" {
                    output.order_indicator().clone()
                } else {
                    inner[0].clone()
                },
            })
        };
        let steps: Vec<_> = orchestration_steps()
            .into_iter()
            .map(|step| LiveBasePriceStep {
                decision: outer.clone(),
                start_time: step.start_time,
                end_time: step.end_time,
            })
            .collect();
        let provider = Provider::new(Some(MarketDataValue::Scalar(10.0)), None);
        let failure = output
            .aggregate_live_order_indicators(
                &inner,
                &outer,
                &steps,
                &provider,
                OrderIndicatorAggregationConfig::default(),
            )
            .unwrap_err();
        if stage == "baseline" {
            assert!(matches!(
                failure,
                AggregateOrderIndicatorsError::BasePrice(AggregateBasePriceError::Indicator(
                    IndicatorError::OrderStorePoisoned
                ))
            ));
            let metrics = output.order_snapshot().unwrap();
            assert!(metrics.contains_key("amount") && metrics.contains_key("ffr"));
            assert!(!metrics.contains_key("base_price") && !metrics.contains_key("pa"));
        } else {
            assert_eq!(
                failure,
                AggregateOrderIndicatorsError::Indicator(IndicatorError::OrderStorePoisoned)
            );
        }
        assert!(provider.calls.lock().unwrap().is_empty());
    }
}

struct ClearingProvider(domain_core::SharedOrderIndicator<NumpyOrderIndicator>);
impl BasePriceDataProvider for ClearingProvider {
    fn deal_price(
        &self,
        _: &str,
        _: TimeRange,
        _: OrderDir,
    ) -> Result<Option<MarketDataValue>, BasePriceProviderError> {
        *self
            .0
            .try_write()
            .expect("provider must run without output guard") = NumpyOrderIndicator::default();
        Ok(Some(MarketDataValue::Scalar(10.0)))
    }
    fn volume(
        &self,
        _: &str,
        _: TimeRange,
    ) -> Result<Option<MarketDataValue>, BasePriceProviderError> {
        panic!("TWAP")
    }
}

#[test]
fn live_full_aggregation_preserves_completed_baselines_when_advantage_fails() {
    let mut output = Indicator::new();
    let provider = ClearingProvider(output.order_indicator().clone());
    let inner: Vec<_> = orchestration_inner::<NumpyOrderIndicator>()
        .into_iter()
        .map(|store| Arc::new(RwLock::new(store)))
        .collect();
    let outer: LiveDecisionHandle = outer_decision();
    let steps: Vec<_> = orchestration_steps()
        .into_iter()
        .map(|step| LiveBasePriceStep {
            decision: outer.clone(),
            start_time: step.start_time,
            end_time: step.end_time,
        })
        .collect();
    assert!(matches!(
        output.aggregate_live_order_indicators(
            &inner,
            &outer,
            &steps,
            &provider,
            OrderIndicatorAggregationConfig::default()
        ),
        Err(AggregateOrderIndicatorsError::Indicator(_))
    ));
    let metrics = output.order_snapshot().unwrap();
    assert!(metrics.contains_key("base_price") && metrics.contains_key("base_volume"));
    assert!(!metrics.contains_key("trade_price") && !metrics.contains_key("pa"));
}
