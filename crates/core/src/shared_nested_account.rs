//! Shared account ownership for nested executors that suspend between lifecycle stages.

use std::sync::{Arc, MutexGuard};

use crate::{
    Account, AccountBarMarket, BasePriceDataProvider, NestedAccountAdapter, NestedBarEnd,
    NestedExecutorAccount, NestedExecutorAccountError, SharedSaoeAccount,
};

/// Retains the original account and services without holding a guard across a yield.
/// Each lifecycle operation borrows the shared account only until that call returns.
/// Settlement state remains in the account between calls; failures do not roll it back.
/// Market/base-price callbacks must not reenter the same account while bar end holds its lock.
pub struct SharedNestedAccountAdapter {
    account: SharedSaoeAccount,
    market: Arc<dyn AccountBarMarket>,
    base_price: Arc<dyn BasePriceDataProvider>,
    show_indicator: bool,
}

impl SharedNestedAccountAdapter {
    #[must_use]
    pub fn new(
        account: SharedSaoeAccount,
        market: Arc<dyn AccountBarMarket>,
        base_price: Arc<dyn BasePriceDataProvider>,
        show_indicator: bool,
    ) -> Self {
        Self {
            account,
            market,
            base_price,
            show_indicator,
        }
    }

    fn lock(&self) -> Result<MutexGuard<'_, Account>, NestedExecutorAccountError> {
        self.account.lock().map_err(|_| NestedExecutorAccountError {
            message: "shared nested account lock poisoned".into(),
        })
    }

    fn adapter<'a>(&'a self, account: &'a mut Account) -> NestedAccountAdapter<'a> {
        NestedAccountAdapter::new(
            account,
            &*self.market,
            &*self.base_price,
            self.show_indicator,
        )
    }
}

impl NestedExecutorAccount for SharedNestedAccountAdapter {
    fn update_live_bar_end(
        &mut self,
        bar: crate::nested_executor_lifecycle::LiveNestedBarEnd<'_>,
    ) -> Result<(), NestedExecutorAccountError> {
        let mut account = self.lock()?;
        self.adapter(&mut account).update_live_bar_end(bar)
    }

    fn settle_start(&mut self, settle_type: &str) -> Result<(), NestedExecutorAccountError> {
        let mut account = self.lock()?;
        self.adapter(&mut account).settle_start(settle_type)
    }

    fn update_bar_end(&mut self, bar: NestedBarEnd<'_>) -> Result<(), NestedExecutorAccountError> {
        let mut account = self.lock()?;
        self.adapter(&mut account).update_bar_end(bar)
    }

    fn settle_commit(&mut self) -> Result<(), NestedExecutorAccountError> {
        let mut account = self.lock()?;
        self.adapter(&mut account).settle_commit()
    }
}
