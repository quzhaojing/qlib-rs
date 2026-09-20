//! Order extraction and iteration rules used by Qlib's simulator executor.

use std::{cmp::Reverse, collections::HashMap, sync::Arc};

use chrono::{NaiveDate, NaiveDateTime};
use strum::{AsRefStr, Display, EnumString, VariantArray};
use thiserror::Error;

use crate::{
    ExchangeDealError, ExchangeDealExecutor, ExecutionPositionError, ExecutionTarget,
    ExecutionTargetError, Order, OrderDealResult, OrderDecision, OrderDir, OrderExecution,
};

/// Simulator order-processing mode.
#[derive(Clone, Copy, Debug, Display, EnumString, AsRefStr, VariantArray, PartialEq, Eq, Hash)]
#[strum(serialize_all = "lowercase")]
pub enum SimulatorTradeType {
    /// Preserve the strategy's order sequence.
    Serial,
    /// Use Qlib's stable buy-first sequence before simulated execution.
    Parallel,
}

impl SimulatorTradeType {
    /// Parse the exact case-sensitive Python configuration spelling.
    ///
    /// # Errors
    ///
    /// Returns a typed unsupported-mode error for whitespace, case, or value mismatches.
    pub fn parse_text(value: &str) -> Result<Self, SimulatorExecutorError> {
        value
            .parse()
            .map_err(|_| SimulatorExecutorError::UnsupportedTradeType {
                trade_type: value.to_owned(),
            })
    }
}

/// Failures emitted before the simulator processes an order batch.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum SimulatorExecutorError {
    /// Only Qlib's exact `serial` and `parallel` values are accepted.
    #[error("unsupported simulator trade type: {trade_type}")]
    UnsupportedTradeType {
        /// Original rejected configuration value.
        trade_type: String,
    },
}

/// Copy the decision's order references into an independently reorderable batch.
///
/// Mutating an order through a returned reference updates the decision-owned order, while
/// reordering the returned vector never changes the decision's original sequence.
pub fn retrieve_orders_from_decision(decision: &mut dyn OrderDecision) -> Vec<&mut Order> {
    decision.orders_mut().iter_mut().collect()
}

/// Produce the order batch consumed by one simulator step.
///
/// Order extraction deliberately happens before mode validation, matching Python's observable
/// call/error order. Parallel mode uses Rust's stable slice sort so buys precede sells while the
/// relative sequence within each direction remains unchanged.
///
/// # Errors
///
/// Returns [`SimulatorExecutorError::UnsupportedTradeType`] unless `trade_type` is exactly
/// `serial` or `parallel`.
pub fn simulator_order_iterator<'a>(
    decision: &'a mut dyn OrderDecision,
    trade_type: &str,
) -> Result<Vec<&'a mut Order>, SimulatorExecutorError> {
    let mut orders = retrieve_orders_from_decision(decision);
    match SimulatorTradeType::parse_text(trade_type)? {
        SimulatorTradeType::Serial => {}
        SimulatorTradeType::Parallel => {
            orders.sort_by_key(|order| Reverse(order.direction().value()));
        }
    }
    Ok(orders)
}

/// Diagnostic returned by a replaceable simulator calendar.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("simulator calendar error: {message}")]
pub struct SimulatorCalendarError {
    pub message: String,
}

/// Narrow calendar boundary required by one simulator collection step.
pub trait SimulatorCalendar: Send + Sync {
    /// Return the current closed execution interval.
    ///
    /// # Errors
    ///
    /// Returns a calendar state or retrieval failure.
    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), SimulatorCalendarError>;
}

/// Order-dealing boundary used by the simulator loop.
pub trait SimulatorDealProvider: Send + Sync {
    /// Deal one order against the mutable account and current intraday fills.
    ///
    /// # Errors
    ///
    /// Returns the underlying Exchange dealing failure.
    fn deal_order(
        &self,
        order: &mut Order,
        account: &mut dyn ExecutionTarget,
        dealt_order_amount: &HashMap<String, f64>,
    ) -> Result<OrderDealResult, ExchangeDealError>;
}

impl SimulatorDealProvider for ExchangeDealExecutor {
    fn deal_order(
        &self,
        order: &mut Order,
        account: &mut dyn ExecutionTarget,
        dealt_order_amount: &HashMap<String, f64>,
    ) -> Result<OrderDealResult, ExchangeDealError> {
        self.deal_order(order, Some(account), None, dealt_order_amount)
    }
}

/// Structured payload corresponding to one Python verbose execution line.
#[derive(Clone, Copy, Debug)]
pub struct SimulatorExecutionLog<'a> {
    pub trade_start_time: NaiveDateTime,
    pub execution: OrderExecution<'a>,
    pub cash: f64,
}

/// Diagnostic returned by a replaceable verbose-output sink.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("simulator execution reporter error: {message}")]
pub struct SimulatorReporterError {
    pub message: String,
}

/// Replaceable sink for verbose simulator execution output.
pub trait SimulatorExecutionReporter: Send + Sync {
    /// Emit one completed execution after cumulative fill accounting.
    ///
    /// # Errors
    ///
    /// Returns an output or transport failure.
    fn report(&self, log: SimulatorExecutionLog<'_>) -> Result<(), SimulatorReporterError>;
}

/// Format the stable console line produced for one verbose execution.
#[must_use]
pub fn format_simulator_execution(log: SimulatorExecutionLog<'_>) -> String {
    let direction = match log.execution.order.direction() {
        OrderDir::Sell => "sell",
        OrderDir::Buy => "buy",
    };
    let factor = log
        .execution
        .order
        .factor()
        .map_or_else(|| "None".to_owned(), |factor| format!("{factor:?}"));
    format!(
        "[I {}]: {} {}, price {:.2}, amount {:?}, deal_amount {:?}, factor {}, value {:.2}, cash {:.2}.",
        log.trade_start_time.format("%Y-%m-%d %H:%M:%S"),
        direction,
        log.execution.order.stock_id(),
        log.execution.trade_price,
        log.execution.order.amount(),
        log.execution.order.deal_amount(),
        factor,
        log.execution.trade_value,
        log.cash,
    )
}

/// Failures from one simulator execution-collection step.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SimulatorCollectionError {
    #[error(transparent)]
    Calendar(#[from] SimulatorCalendarError),
    #[error(transparent)]
    Iterator(#[from] SimulatorExecutorError),
    #[error(transparent)]
    Deal(#[from] ExchangeDealError),
    #[error(transparent)]
    Target(#[from] ExecutionTargetError),
    #[error(transparent)]
    Position(#[from] ExecutionPositionError),
    #[error(transparent)]
    Reporter(#[from] SimulatorReporterError),
}

/// One collection result exposed both as the execution list and `trade_info` payload.
#[derive(Debug)]
pub struct SimulatorCollection<'a> {
    executions: Vec<OrderExecution<'a>>,
}

impl<'a> SimulatorCollection<'a> {
    /// Primary execution-result list.
    #[must_use]
    pub fn execution_result(&self) -> &[OrderExecution<'a>] {
        &self.executions
    }

    /// `trade_info` view backed by the exact same result allocation.
    #[must_use]
    pub fn trade_info(&self) -> &[OrderExecution<'a>] {
        &self.executions
    }
}

/// Stateful core of `SimulatorExecutor._collect_data`.
pub struct SimulatorCollector {
    trade_type: String,
    calendar: Arc<dyn SimulatorCalendar>,
    dealer: Arc<dyn SimulatorDealProvider>,
    reporter: Arc<dyn SimulatorExecutionReporter>,
    verbose: bool,
    deal_day: Option<NaiveDate>,
    dealt_order_amount: HashMap<String, f64>,
}

impl SimulatorCollector {
    #[must_use]
    pub fn new(
        trade_type: impl Into<String>,
        calendar: Arc<dyn SimulatorCalendar>,
        dealer: Arc<dyn SimulatorDealProvider>,
        reporter: Arc<dyn SimulatorExecutionReporter>,
        verbose: bool,
    ) -> Self {
        Self {
            trade_type: trade_type.into(),
            calendar,
            dealer,
            reporter,
            verbose,
            deal_day: None,
            dealt_order_amount: HashMap::new(),
        }
    }

    /// Restore private intraday state from a checkpoint.
    pub fn restore_intraday_state(
        &mut self,
        deal_day: Option<NaiveDate>,
        dealt_order_amount: HashMap<String, f64>,
    ) {
        self.deal_day = deal_day;
        self.dealt_order_amount = dealt_order_amount;
    }

    #[must_use]
    pub const fn deal_day(&self) -> Option<NaiveDate> {
        self.deal_day
    }

    #[must_use]
    pub fn dealt_order_amount(&self) -> &HashMap<String, f64> {
        &self.dealt_order_amount
    }

    /// Execute one decision batch and return its aliased execution/trade-info view.
    ///
    /// # Errors
    ///
    /// Returns the first calendar, iterator, deal, account, position, or reporter failure after
    /// retaining every state mutation already reached by the Python method.
    pub fn collect_data<'a>(
        &mut self,
        decision: &'a mut dyn OrderDecision,
        account: &mut dyn ExecutionTarget,
        _level: usize,
    ) -> Result<SimulatorCollection<'a>, SimulatorCollectionError> {
        let (trade_start_time, _) = self.calendar.step_time()?;
        let orders = simulator_order_iterator(decision, &self.trade_type)?;
        let mut executions = Vec::with_capacity(orders.len());

        for order in orders {
            self.start_order_day()?;
            let result = self.deal_and_report(order, account, trade_start_time)?;
            executions.push(OrderExecution {
                order,
                trade_value: result.trade_value,
                trade_cost: result.trade_cost,
                trade_price: result.trade_price,
            });
        }

        Ok(SimulatorCollection { executions })
    }

    /// Execute shared orders and return live order handles in both result views.
    ///
    /// Duplicate references are executed sequentially, so later fills update the order visible
    /// through earlier result rows. Reordering or replacing the original list does not reorder
    /// the already extracted batch. The legacy dealer/report callbacks borrow each locked order;
    /// they must not recursively acquire that same order lock. A handle-native callback boundary
    /// is still needed for reentrant Python-style plugins.
    ///
    /// # Errors
    /// Returns extraction/lock failures or the same ordered execution errors as `collect_data`.
    /// Mutations made by earlier executions and by a failing dealer remain in the original order.
    pub fn collect_shared_data(
        &mut self,
        decision: &crate::decision_construction::SharedDecisionOrders,
        account: &mut dyn ExecutionTarget,
        _level: usize,
    ) -> Result<
        crate::shared_simulator::SharedSimulatorCollection,
        crate::shared_simulator::SharedSimulatorError,
    > {
        use crate::shared_simulator::{
            SharedSimulatorCollection, SharedSimulatorError, SharedSimulatorExecution,
            shared_simulator_order_iterator,
        };

        let (trade_start_time, _) = self
            .calendar
            .step_time()
            .map_err(SimulatorCollectionError::from)?;
        let orders = shared_simulator_order_iterator(decision, &self.trade_type)?;
        let mut executions = Vec::with_capacity(orders.len());
        for (index, handle) in orders.into_iter().enumerate() {
            self.start_order_day()?;
            let result = {
                let mut order = handle
                    .write()
                    .map_err(|_| SharedSimulatorError::OrderPoisoned(index))?;
                self.deal_and_report(&mut order, account, trade_start_time)?
            };
            executions.push(Arc::new(SharedSimulatorExecution {
                order: handle,
                trade_value: result.trade_value,
                trade_cost: result.trade_cost,
                trade_price: result.trade_price,
            }));
        }
        Ok(SharedSimulatorCollection { executions })
    }

    fn start_order_day(&mut self) -> Result<(), SimulatorCollectionError> {
        let (current_start, _) = self.calendar.step_time()?;
        let now_deal_day = current_start.date();
        if self.deal_day.is_none_or(|deal_day| now_deal_day > deal_day) {
            self.dealt_order_amount = HashMap::new();
            self.deal_day = Some(now_deal_day);
        }
        Ok(())
    }

    fn deal_and_report(
        &mut self,
        order: &mut Order,
        account: &mut dyn ExecutionTarget,
        trade_start_time: NaiveDateTime,
    ) -> Result<OrderDealResult, SimulatorCollectionError> {
        let result = self
            .dealer
            .deal_order(order, account, &self.dealt_order_amount)?;
        *self
            .dealt_order_amount
            .entry(order.stock_id().to_owned())
            .or_default() += order.deal_amount();
        if self.verbose {
            let cash = account.position()?.cash()?;
            self.reporter.report(SimulatorExecutionLog {
                trade_start_time,
                execution: OrderExecution {
                    order,
                    trade_value: result.trade_value,
                    trade_cost: result.trade_cost,
                    trade_price: result.trade_price,
                },
                cash,
            })?;
        }
        Ok(result)
    }
}
