//! Concrete bridge from the account core to the nested executor lifecycle.

use crate::{
    Account, AccountBarEndMode, AccountBarEndUpdate, AccountBarMarket, BasePriceDataProvider,
    NestedAccountIndicatorUpdate, NestedBarEnd, NestedExecutorAccount, NestedExecutorAccountError,
};

/// Binds one mutable account to market and base-price services required at nested bar end.
pub struct NestedAccountAdapter<'a> {
    account: &'a mut Account,
    market: &'a dyn AccountBarMarket,
    base_price_provider: &'a dyn BasePriceDataProvider,
    show_indicator: bool,
}

impl<'a> NestedAccountAdapter<'a> {
    #[must_use]
    pub const fn new(
        account: &'a mut Account,
        market: &'a dyn AccountBarMarket,
        base_price_provider: &'a dyn BasePriceDataProvider,
        show_indicator: bool,
    ) -> Self {
        Self {
            account,
            market,
            base_price_provider,
            show_indicator,
        }
    }
}

impl NestedExecutorAccount for NestedAccountAdapter<'_> {
    fn update_live_bar_end(
        &mut self,
        bar: crate::nested_executor_lifecycle::LiveNestedBarEnd<'_>,
    ) -> Result<(), NestedExecutorAccountError> {
        self.account
            .update_bar_end(AccountBarEndUpdate {
                trade_start_time: bar.trade_start_time,
                trade_end_time: bar.trade_end_time,
                market: self.market,
                mode: AccountBarEndMode::LiveNested(Some(
                    crate::account::LiveNestedAccountIndicatorUpdate {
                        inner: bar.inner_order_indicators,
                        outer_decision: bar.outer_decision,
                        steps: bar.steps,
                        provider: self.base_price_provider,
                        config: bar.aggregation_config,
                    },
                )),
                calculation: bar.indicator_config,
                show_indicator: self.show_indicator,
            })
            .map_err(|error| NestedExecutorAccountError {
                message: error.to_string(),
            })
    }

    fn settle_start(&mut self, settle_type: &str) -> Result<(), NestedExecutorAccountError> {
        self.account
            .current_position_mut()
            .settle_start(settle_type)
            .map_err(position_error)
    }

    fn update_bar_end(&mut self, bar: NestedBarEnd<'_>) -> Result<(), NestedExecutorAccountError> {
        self.account
            .update_bar_end(AccountBarEndUpdate {
                trade_start_time: bar.trade_start_time,
                trade_end_time: bar.trade_end_time,
                market: self.market,
                mode: AccountBarEndMode::Nested(Some(NestedAccountIndicatorUpdate {
                    inner: bar.inner_order_indicators,
                    outer_decision: bar.outer_decision,
                    steps: bar.steps,
                    provider: self.base_price_provider,
                    config: bar.aggregation_config,
                })),
                calculation: bar.indicator_config,
                show_indicator: self.show_indicator,
            })
            .map_err(|error| NestedExecutorAccountError {
                message: error.to_string(),
            })
    }

    fn settle_commit(&mut self) -> Result<(), NestedExecutorAccountError> {
        self.account
            .current_position_mut()
            .settle_commit()
            .map_err(position_error)
    }
}

fn position_error(error: crate::AccountPositionError) -> NestedExecutorAccountError {
    let crate::AccountPositionError { message } = error;
    NestedExecutorAccountError {
        message: format!("account position error: {message}"),
    }
}
