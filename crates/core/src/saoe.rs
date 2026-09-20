//! Owned single-asset order-execution state and proxy strategy boundary.

use std::{
    io::Cursor,
    sync::{Arc, Mutex, MutexGuard},
};

use arrow_array::RecordBatch;
use arrow_ipc::{reader::StreamReader, writer::StreamWriter};
use arrow_schema::ArrowError;
use chrono::NaiveDateTime;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    NestedOuterDecision, NestedStrategy, NestedStrategyError, NestedStrategyProgress,
    NestedStrategyPrompt, Order, OrderDecision, OrderDir, OrderError, OrderTradeDecision,
    SharedOrderExecution,
};

pub const SAOE_PROXY_PROMPT_KIND: &str = "qlib.saoe.proxy";
pub const SAOE_STATE_SCHEMA_VERSION: u32 = 1;

/// Python's `float_or_ndarray` contract represented with ndarray-owned storage.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SaoeNumeric {
    Scalar(f64),
    Array(Array1<f64>),
}

/// Scalar timestamp or vectorized timestamp column from `SAOEMetrics`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SaoeTime {
    Scalar(NaiveDateTime),
    Array(Vec<NaiveDateTime>),
}

/// One scalar or vectorized `SAOEMetrics` value set.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SaoeMetrics {
    pub stock_id: String,
    pub datetime: SaoeTime,
    pub direction: OrderDir,
    pub market_volume: SaoeNumeric,
    pub market_price: SaoeNumeric,
    pub amount: SaoeNumeric,
    pub inner_amount: SaoeNumeric,
    pub deal_amount: SaoeNumeric,
    pub trade_price: SaoeNumeric,
    pub trade_value: SaoeNumeric,
    pub position: SaoeNumeric,
    pub ffr: SaoeNumeric,
    pub pa: SaoeNumeric,
}

/// Owned intraday backtest data retained in an RL state snapshot.
#[derive(Clone, Debug)]
pub struct SaoeBacktestData {
    pub ticks_index: Vec<NaiveDateTime>,
    pub ticks_for_order: Vec<NaiveDateTime>,
    pub deal_prices: Array1<f64>,
    pub market_volumes: Array1<f64>,
    pub features: RecordBatch,
}

/// Constructor fields for one owned `SAOEState` snapshot.
#[derive(Clone, Debug)]
pub struct SaoeStateParts {
    pub order: Order,
    pub cur_time: NaiveDateTime,
    pub cur_step: i64,
    pub position: f64,
    pub history_exec: RecordBatch,
    pub history_steps: RecordBatch,
    pub metrics: Option<SaoeMetrics>,
    pub backtest_data: SaoeBacktestData,
    pub ticks_per_step: usize,
    pub ticks_index: Vec<NaiveDateTime>,
    pub ticks_for_order: Vec<NaiveDateTime>,
}

/// Fully owned equivalent of Python's `SAOEState` named tuple.
#[derive(Clone, Debug)]
pub struct SaoeState {
    parts: SaoeStateParts,
}

impl SaoeState {
    #[must_use]
    pub const fn new(parts: SaoeStateParts) -> Self {
        Self { parts }
    }

    #[must_use]
    pub const fn parts(&self) -> &SaoeStateParts {
        &self.parts
    }

    /// Encode the complete state using Bincode and Arrow IPC streams.
    ///
    /// # Errors
    /// Returns an Arrow IPC or Bincode serialization failure.
    ///
    /// # Panics
    /// Panics only if Bincode rejects the fixed, fully owned wire DTO or an in-memory Arrow
    /// writer fails after accepting the batch schema; both indicate a violated library invariant.
    pub fn encode(&self) -> Result<Vec<u8>, SaoeError> {
        let wire = SaoeStateWire {
            order: self.parts.order.clone(),
            cur_time: self.parts.cur_time,
            cur_step: self.parts.cur_step,
            position: self.parts.position,
            history_exec: encode_batch(&self.parts.history_exec)?,
            history_steps: encode_batch(&self.parts.history_steps)?,
            metrics: self.parts.metrics.clone(),
            ticks_index: self.parts.backtest_data.ticks_index.clone(),
            ticks_for_order: self.parts.backtest_data.ticks_for_order.clone(),
            deal_prices: self.parts.backtest_data.deal_prices.to_vec(),
            market_volumes: self.parts.backtest_data.market_volumes.to_vec(),
            features: encode_batch(&self.parts.backtest_data.features)?,
            ticks_per_step: self.parts.ticks_per_step,
            state_ticks_index: self.parts.ticks_index.clone(),
            state_ticks_for_order: self.parts.ticks_for_order.clone(),
        };
        Ok(bincode::serialize(&wire)
            .expect("the fixed owned SAOE wire DTO is always Bincode serializable"))
    }

    /// Decode a complete state produced by [`Self::encode`].
    ///
    /// # Errors
    /// Returns an Arrow IPC or Bincode deserialization failure.
    pub fn decode(payload: &[u8]) -> Result<Self, SaoeError> {
        let wire: SaoeStateWire = match bincode::deserialize(payload) {
            Ok(wire) => wire,
            Err(current_error) => match bincode::deserialize::<SaoeStateWireV2>(payload) {
                Ok(legacy) => legacy.into(),
                Err(_) => match bincode::deserialize::<SaoeStateWireV1>(payload) {
                    Ok(legacy) => legacy.into(),
                    Err(_) => return Err(current_error.into()),
                },
            },
        };
        Ok(Self::new(SaoeStateParts {
            order: wire.order,
            cur_time: wire.cur_time,
            cur_step: wire.cur_step,
            position: wire.position,
            history_exec: decode_batch(&wire.history_exec)?,
            history_steps: decode_batch(&wire.history_steps)?,
            metrics: wire.metrics,
            backtest_data: SaoeBacktestData {
                ticks_index: wire.ticks_index,
                ticks_for_order: wire.ticks_for_order,
                deal_prices: Array1::from_vec(wire.deal_prices),
                market_volumes: Array1::from_vec(wire.market_volumes),
                features: decode_batch(&wire.features)?,
            },
            ticks_per_step: wire.ticks_per_step,
            ticks_index: wire.state_ticks_index,
            ticks_for_order: wire.state_ticks_for_order,
        }))
    }
}

#[derive(Serialize, Deserialize)]
struct SaoeStateWire {
    order: Order,
    cur_time: NaiveDateTime,
    cur_step: i64,
    position: f64,
    history_exec: Vec<u8>,
    history_steps: Vec<u8>,
    metrics: Option<SaoeMetrics>,
    ticks_index: Vec<NaiveDateTime>,
    ticks_for_order: Vec<NaiveDateTime>,
    deal_prices: Vec<f64>,
    market_volumes: Vec<f64>,
    features: Vec<u8>,
    ticks_per_step: usize,
    state_ticks_index: Vec<NaiveDateTime>,
    state_ticks_for_order: Vec<NaiveDateTime>,
}

#[derive(Deserialize)]
struct SaoeStateWireV2 {
    order: Order,
    cur_time: NaiveDateTime,
    cur_step: i64,
    position: f64,
    history_exec: Vec<u8>,
    history_steps: Vec<u8>,
    metrics: Option<SaoeMetrics>,
    ticks_index: Vec<NaiveDateTime>,
    ticks_for_order: Vec<NaiveDateTime>,
    deal_prices: Vec<f64>,
    market_volumes: Vec<f64>,
    features: Vec<u8>,
    ticks_per_step: usize,
}

impl From<SaoeStateWireV2> for SaoeStateWire {
    fn from(legacy: SaoeStateWireV2) -> Self {
        Self {
            order: legacy.order,
            cur_time: legacy.cur_time,
            cur_step: legacy.cur_step,
            position: legacy.position,
            history_exec: legacy.history_exec,
            history_steps: legacy.history_steps,
            metrics: legacy.metrics,
            state_ticks_index: legacy.ticks_index.clone(),
            state_ticks_for_order: legacy.ticks_for_order.clone(),
            ticks_index: legacy.ticks_index,
            ticks_for_order: legacy.ticks_for_order,
            deal_prices: legacy.deal_prices,
            market_volumes: legacy.market_volumes,
            features: legacy.features,
            ticks_per_step: legacy.ticks_per_step,
        }
    }
}

#[derive(Deserialize)]
struct SaoeStateWireV1 {
    order: Order,
    cur_time: NaiveDateTime,
    cur_step: i64,
    position: f64,
    history_exec: Vec<u8>,
    history_steps: Vec<u8>,
    metrics: Option<SaoeMetrics>,
    ticks_index: Vec<NaiveDateTime>,
    ticks_for_order: Vec<NaiveDateTime>,
    features: Vec<u8>,
    ticks_per_step: usize,
}

impl From<SaoeStateWireV1> for SaoeStateWire {
    fn from(legacy: SaoeStateWireV1) -> Self {
        SaoeStateWireV2 {
            order: legacy.order,
            cur_time: legacy.cur_time,
            cur_step: legacy.cur_step,
            position: legacy.position,
            history_exec: legacy.history_exec,
            history_steps: legacy.history_steps,
            metrics: legacy.metrics,
            ticks_index: legacy.ticks_index,
            ticks_for_order: legacy.ticks_for_order,
            deal_prices: Vec::new(),
            market_volumes: Vec::new(),
            features: legacy.features,
            ticks_per_step: legacy.ticks_per_step,
        }
        .into()
    }
}

fn encode_batch(batch: &RecordBatch) -> Result<Vec<u8>, ArrowError> {
    let mut payload = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut payload, batch.schema().as_ref())?;
        writer
            .write(batch)
            .expect("a fresh Arrow stream accepts a batch with the same validated schema");
        writer
            .finish()
            .expect("finishing an in-memory Arrow stream is infallible");
    }
    Ok(payload)
}

fn decode_batch(payload: &[u8]) -> Result<RecordBatch, SaoeError> {
    let mut reader = StreamReader::try_new(Cursor::new(payload), None)?;
    reader.next().transpose()?.ok_or(SaoeError::MissingBatch)
}

/// Owned prompt locator corresponding to the yielded Python proxy strategy object.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SaoePromptLocator {
    pub stock_id: String,
    pub day: NaiveDateTime,
    pub direction: OrderDir,
    pub step_range: (i64, i64),
}

/// Typed diagnostic from an injected SAOE calendar, state, or order plugin.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("SAOE plugin error: {message}")]
pub struct SaoePluginError {
    pub message: String,
}

/// Calendar information observed immediately before proxy suspension and after action delivery.
pub trait SaoeCalendar: Send + Sync {
    /// Return the current inclusive data-calendar step range.
    ///
    /// # Errors
    /// Returns a calendar or transport failure.
    fn available_step_range(&self) -> Result<(i64, i64), SaoePluginError>;

    /// Return the current closed decision interval.
    ///
    /// # Errors
    /// Returns a calendar or transport failure.
    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), SaoePluginError>;
}

/// State adapter corresponding to Python's per-order `SAOEStateAdapter`.
pub trait SaoeStateProvider: Send {
    /// Reset state for the sole outer order.
    ///
    /// # Errors
    /// Returns an adapter construction or transport failure.
    fn reset(&mut self, order: &Order) -> Result<(), SaoePluginError>;

    /// Materialize the current owned state snapshot.
    ///
    /// # Errors
    /// Returns a state conversion or transport failure.
    fn state(&self, order: &Order) -> Result<SaoeState, SaoePluginError>;

    /// Apply results belonging to this order for the last step range.
    ///
    /// # Errors
    /// Returns an update or transport failure.
    fn update(
        &mut self,
        executions: &[SharedOrderExecution],
        step_range: (i64, i64),
    ) -> Result<(), SaoePluginError>;

    /// Generate final metrics after the upper level completes.
    ///
    /// # Errors
    /// Returns a metric or transport failure.
    fn finalize(&mut self) -> Result<(), SaoePluginError>;
}

/// Shared in-process access to the same live adapter from a strategy and simulator.
/// Each call locks only for the delegated operation; providers must not reenter
/// their own shared handle. Poisoned state is reported, never silently recovered.
impl<P: SaoeStateProvider + ?Sized> SaoeStateProvider for Arc<Mutex<P>> {
    fn reset(&mut self, order: &Order) -> Result<(), SaoePluginError> {
        shared_state_lock(self)?.reset(order)
    }

    fn state(&self, order: &Order) -> Result<SaoeState, SaoePluginError> {
        shared_state_lock(self)?.state(order)
    }

    fn update(
        &mut self,
        executions: &[SharedOrderExecution],
        step_range: (i64, i64),
    ) -> Result<(), SaoePluginError> {
        shared_state_lock(self)?.update(executions, step_range)
    }

    fn finalize(&mut self) -> Result<(), SaoePluginError> {
        shared_state_lock(self)?.finalize()
    }
}

fn shared_state_lock<P: SaoeStateProvider + ?Sized>(
    provider: &Mutex<P>,
) -> Result<MutexGuard<'_, P>, SaoePluginError> {
    provider.lock().map_err(|error| SaoePluginError {
        message: format!("shared SAOE state provider lock poisoned: {error}"),
    })
}

/// Exchange order-helper boundary used by outer and child SAOE strategies.
pub trait SaoeOrderFactory: Send {
    /// Create one child order from the outer instrument, optional action, and direction.
    ///
    /// # Errors
    /// Returns an invalid action or exchange-helper failure.
    fn create(
        &mut self,
        stock_id: &str,
        amount: Option<f64>,
        direction: OrderDir,
    ) -> Result<Order, SaoePluginError>;
}

/// State and wire-format failures for the concrete SAOE boundary.
#[derive(Debug, Error)]
pub enum SaoeError {
    #[error(transparent)]
    Arrow(#[from] ArrowError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Bincode(#[from] Box<bincode::ErrorKind>),
    #[error("Arrow IPC stream contained no record batch")]
    MissingBatch,
    #[error("proxy SAOE strategy requires exactly one outer order, got {0}")]
    OuterOrderCount(usize),
    #[error("proxy SAOE strategy has not been reset with an outer order")]
    MissingOuterOrder,
    #[error("proxy SAOE strategy is already suspended")]
    AlreadySuspended,
    #[error("proxy SAOE strategy is not suspended")]
    NotSuspended,
    #[error("a zero-length SAOE step range cannot receive executions")]
    ExecutionsForZeroRange,
    #[error(transparent)]
    Order(#[from] OrderError),
    #[error(transparent)]
    Plugin(#[from] SaoePluginError),
}

fn strategy_error(error: impl std::fmt::Display) -> NestedStrategyError {
    NestedStrategyError {
        message: error.to_string(),
    }
}

/// Concrete proxy strategy preserving Qlib's yield/action/order-helper sequence.
pub struct ProxySaoeStrategy {
    calendar: Arc<dyn SaoeCalendar>,
    states: Box<dyn SaoeStateProvider>,
    orders: Box<dyn SaoeOrderFactory>,
    phase: ProxyPhase,
}

enum ProxyPhase {
    Uninitialized,
    Ready {
        order: Order,
        step_range: (i64, i64),
    },
    Suspended {
        order: Order,
        step_range: (i64, i64),
    },
}

impl ProxySaoeStrategy {
    #[must_use]
    pub fn new(
        calendar: Arc<dyn SaoeCalendar>,
        states: Box<dyn SaoeStateProvider>,
        orders: Box<dyn SaoeOrderFactory>,
    ) -> Self {
        Self {
            calendar,
            states,
            orders,
            phase: ProxyPhase::Uninitialized,
        }
    }

    #[must_use]
    pub fn last_step_range(&self) -> (i64, i64) {
        match self.phase {
            ProxyPhase::Uninitialized => (0, 0),
            ProxyPhase::Ready { step_range, .. } | ProxyPhase::Suspended { step_range, .. } => {
                step_range
            }
        }
    }

    /// Return the state that Python exposes after yielding this proxy strategy.
    ///
    /// # Errors
    /// Returns a missing-order or state-provider failure.
    pub fn current_state(&self) -> Result<SaoeState, SaoeError> {
        let order = match &self.phase {
            ProxyPhase::Ready { order, .. } | ProxyPhase::Suspended { order, .. } => order,
            ProxyPhase::Uninitialized => return Err(SaoeError::MissingOuterOrder),
        };
        Ok(self.states.state(order)?)
    }

    /// Decode and validate this strategy's versioned prompt locator.
    ///
    /// # Errors
    /// Returns a kind, version, or JSON failure.
    pub fn decode_prompt(prompt: &NestedStrategyPrompt) -> Result<SaoePromptLocator, SaoeError> {
        if prompt.kind != SAOE_PROXY_PROMPT_KIND {
            return Err(SaoeError::Plugin(SaoePluginError {
                message: format!("unsupported prompt kind: {}", prompt.kind),
            }));
        }
        if prompt.schema_version != SAOE_STATE_SCHEMA_VERSION {
            return Err(SaoeError::Plugin(SaoePluginError {
                message: format!(
                    "unsupported prompt schema version: {}",
                    prompt.schema_version
                ),
            }));
        }
        Ok(serde_json::from_slice(&prompt.payload)?)
    }

    fn prompt(order: &Order, step_range: (i64, i64)) -> NestedStrategyPrompt {
        let locator = SaoePromptLocator {
            stock_id: order.stock_id().to_owned(),
            day: order
                .day_timestamp()
                .expect("reset validates the outer order timestamp"),
            direction: order.direction(),
            step_range,
        };
        let payload = serde_json::to_vec(&locator)
            .expect("the fixed SAOE prompt locator is always JSON serializable");
        NestedStrategyPrompt {
            kind: SAOE_PROXY_PROMPT_KIND.to_owned(),
            schema_version: SAOE_STATE_SCHEMA_VERSION,
            payload,
        }
    }
}

impl NestedStrategy for ProxySaoeStrategy {
    fn reset(&mut self, outer: &dyn NestedOuterDecision) -> Result<(), NestedStrategyError> {
        let orders = outer.order_decision().orders();
        if orders.len() != 1 {
            return Err(strategy_error(SaoeError::OuterOrderCount(orders.len())));
        }
        orders[0]
            .day_timestamp()
            .map_err(SaoeError::from)
            .map_err(strategy_error)?;
        self.states
            .reset(&orders[0])
            .map_err(|error| strategy_error(SaoeError::Plugin(error)))?;
        self.phase = ProxyPhase::Ready {
            order: orders[0].clone(),
            step_range: (0, 0),
        };
        Ok(())
    }

    fn alter_outer_decision(
        &mut self,
        _outer: &mut dyn NestedOuterDecision,
    ) -> Result<(), NestedStrategyError> {
        Ok(())
    }

    fn generate_trade_decision(
        &mut self,
        _previous: Option<&[SharedOrderExecution]>,
    ) -> Result<Box<dyn OrderDecision>, NestedStrategyError> {
        Err(NestedStrategyError {
            message: "proxy SAOE strategy requires the resumable protocol".to_owned(),
        })
    }

    fn begin_trade_decision(
        &mut self,
        _previous: Option<&[SharedOrderExecution]>,
    ) -> Result<NestedStrategyProgress, NestedStrategyError> {
        let order = match &self.phase {
            ProxyPhase::Uninitialized => {
                return Err(strategy_error(SaoeError::MissingOuterOrder));
            }
            ProxyPhase::Ready { order, .. } => order.clone(),
            ProxyPhase::Suspended { .. } => {
                return Err(strategy_error(SaoeError::AlreadySuspended));
            }
        };
        let step_range = self
            .calendar
            .available_step_range()
            .map_err(|error| strategy_error(SaoeError::Plugin(error)))?;
        let prompt = Self::prompt(&order, step_range);
        self.phase = ProxyPhase::Suspended { order, step_range };
        Ok(NestedStrategyProgress::Suspended(prompt))
    }

    fn resume_trade_decision(
        &mut self,
        execution_volume: Option<f64>,
    ) -> Result<NestedStrategyProgress, NestedStrategyError> {
        let outer = match &self.phase {
            ProxyPhase::Suspended { order, step_range } => (order.clone(), *step_range),
            ProxyPhase::Uninitialized | ProxyPhase::Ready { .. } => {
                return Err(strategy_error(SaoeError::NotSuspended));
            }
        };
        let order = self
            .orders
            .create(outer.0.stock_id(), execution_volume, outer.0.direction())
            .map_err(|error| strategy_error(SaoeError::Plugin(error)))?;
        let (start_time, end_time) = self
            .calendar
            .step_time()
            .map_err(|error| strategy_error(SaoeError::Plugin(error)))?;
        self.phase = ProxyPhase::Ready {
            order: outer.0,
            step_range: outer.1,
        };
        Ok(NestedStrategyProgress::Ready(Box::new(
            OrderTradeDecision::from_orders(vec![order], start_time, end_time, None),
        )))
    }

    fn close_trade_decision(&mut self) -> Result<(), NestedStrategyError> {
        self.phase = match std::mem::replace(&mut self.phase, ProxyPhase::Uninitialized) {
            ProxyPhase::Suspended { order, step_range } => ProxyPhase::Ready { order, step_range },
            phase => phase,
        };
        Ok(())
    }

    fn post_execute(
        &mut self,
        executions: &[SharedOrderExecution],
    ) -> Result<(), NestedStrategyError> {
        let (outer, step_range) = match &self.phase {
            ProxyPhase::Uninitialized => {
                if executions.is_empty() {
                    return Ok(());
                }
                return Err(strategy_error(SaoeError::ExecutionsForZeroRange));
            }
            ProxyPhase::Ready { order, step_range }
            | ProxyPhase::Suspended { order, step_range } => (order, *step_range),
        };
        if step_range.1 - step_range.0 <= 0 {
            if executions.is_empty() {
                return Ok(());
            }
            return Err(strategy_error(SaoeError::ExecutionsForZeroRange));
        }
        let outer_key = outer
            .key_by_day()
            .expect("reset validates the outer order timestamp");
        let mut matching = Vec::new();
        for execution in executions {
            let order = execution.order.read().map_err(|_| {
                strategy_error(SaoeError::Plugin(SaoePluginError {
                    message: "SAOE execution order lock poisoned".to_owned(),
                }))
            })?;
            let key = order
                .key_by_day()
                .map_err(SaoeError::from)
                .map_err(strategy_error)?;
            if key == outer_key {
                matching.push(Arc::clone(execution));
            }
        }
        self.states
            .update(&matching, step_range)
            .map_err(|error| strategy_error(SaoeError::Plugin(error)))
    }

    fn post_upper_level(&mut self) -> Result<(), NestedStrategyError> {
        self.states
            .finalize()
            .map_err(|error| strategy_error(SaoeError::Plugin(error)))
    }
}
