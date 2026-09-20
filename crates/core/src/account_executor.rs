//! Concrete bridge from the account core to the atomic executor lifecycle.

use crate::{
    Account, AccountBarEndMode, AccountBarEndUpdate, AccountBarMarket, AtomicBarEnd,
    AtomicExecutorAccount, AtomicExecutorAccountError, ExecutionTarget, NumpyOrderIndicator,
};

/// Binds one mutable account to the market required by executor bar-end accounting.
pub struct AtomicAccountAdapter<'a> {
    account: &'a mut Account,
    market: &'a dyn AccountBarMarket,
    show_indicator: bool,
}

impl<'a> AtomicAccountAdapter<'a> {
    #[must_use]
    pub const fn new(
        account: &'a mut Account,
        market: &'a dyn AccountBarMarket,
        show_indicator: bool,
    ) -> Self {
        Self {
            account,
            market,
            show_indicator,
        }
    }

    fn update_retained_bar_end(
        &mut self,
        bar: &crate::shared_executor_lifecycle::SharedAtomicBarEnd,
    ) -> Result<(), AtomicExecutorAccountError> {
        self.account
            .update_bar_end(AccountBarEndUpdate {
                trade_start_time: bar.trade_start_time,
                trade_end_time: bar.trade_end_time,
                market: self.market,
                mode: AccountBarEndMode::SharedAtomic(&bar.trade_info),
                calculation: bar.indicator_config,
                show_indicator: self.show_indicator,
            })
            .map_err(|error| AtomicExecutorAccountError {
                message: error.to_string(),
            })
    }
}

impl AtomicExecutorAccount for AtomicAccountAdapter<'_> {
    fn execution_target(&mut self) -> Result<&mut dyn ExecutionTarget, AtomicExecutorAccountError> {
        Ok(self.account)
    }

    fn settle_start(&mut self, settle_type: &str) -> Result<(), AtomicExecutorAccountError> {
        self.account
            .current_position_mut()
            .settle_start(settle_type)
            .map_err(position_error)
    }

    fn update_bar_end(&mut self, bar: AtomicBarEnd<'_>) -> Result<(), AtomicExecutorAccountError> {
        if !bar.atomic {
            return Err(AtomicExecutorAccountError {
                message: "atomic account adapter requires atomic bar-end input".to_owned(),
            });
        }
        self.account
            .update_bar_end(AccountBarEndUpdate {
                trade_start_time: bar.trade_start_time,
                trade_end_time: bar.trade_end_time,
                market: self.market,
                mode: AccountBarEndMode::Atomic(Some(bar.trade_info)),
                calculation: bar.indicator_config,
                show_indicator: self.show_indicator,
            })
            .map_err(|error| AtomicExecutorAccountError {
                message: error.to_string(),
            })
    }

    fn settle_commit(&mut self) -> Result<(), AtomicExecutorAccountError> {
        self.account
            .current_position_mut()
            .settle_commit()
            .map_err(position_error)
    }

    fn order_indicator_snapshot(&self) -> Result<NumpyOrderIndicator, AtomicExecutorAccountError> {
        self.account
            .order_indicator_snapshot()
            .map_err(|error| AtomicExecutorAccountError {
                message: error.to_string(),
            })
    }

    fn order_indicator_handle(
        &self,
    ) -> Result<crate::SharedOrderIndicator<NumpyOrderIndicator>, AtomicExecutorAccountError> {
        self.account
            .order_indicator_handle()
            .map_err(|error| AtomicExecutorAccountError {
                message: error.to_string(),
            })
    }
}

fn position_error(error: crate::AccountPositionError) -> AtomicExecutorAccountError {
    let crate::AccountPositionError { message } = error;
    AtomicExecutorAccountError {
        message: format!("account position error: {message}"),
    }
}

impl<S, D> crate::shared_executor_lifecycle::SharedAtomicExecutorAccount<S, D>
    for AtomicAccountAdapter<'_>
{
    fn update_shared_bar_end(
        &mut self,
        bar: crate::shared_executor_lifecycle::SharedAtomicBarEnd,
        _decision: &mut crate::decision_construction::SharedOrderDecisionConstruction<S, D>,
    ) -> Result<(), AtomicExecutorAccountError> {
        self.update_retained_bar_end(&bar)
    }
}

impl crate::shared_executor_lifecycle::LiveAtomicExecutorAccount for AtomicAccountAdapter<'_> {
    fn update_live_bar_end(
        &mut self,
        bar: crate::shared_executor_lifecycle::SharedAtomicBarEnd,
        _decision: &crate::decision_update::LiveDecisionHandle,
    ) -> Result<(), AtomicExecutorAccountError> {
        self.update_retained_bar_end(&bar)
    }
}
