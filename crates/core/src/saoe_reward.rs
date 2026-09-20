//! Reward functions for Qlib's single-asset order-execution state.

use std::{cmp::Ordering, sync::Arc};

use arrow_array::{Array, Float64Array, RecordBatch, TimestampNanosecondArray};
use arrow_schema::DataType;
use indexmap::IndexMap;
use num_traits::ToPrimitive;
use thiserror::Error;

use crate::{OrderDir, SaoeState};

#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("SAOE reward log error: {message}")]
pub struct SaoeRewardLogError {
    pub message: String,
}

pub trait SaoeRewardLogSink: Send + Sync {
    /// # Errors
    /// Returns transport, storage, or lifecycle failures.
    fn log_scalar(&self, name: &str, value: f64) -> Result<(), SaoeRewardLogError>;
}

#[derive(Debug, Error)]
pub enum SaoeRewardError {
    #[error("SAOE reward requires a positive whole-order amount, got {0}")]
    InvalidOrderAmount(f64),
    #[error("SAOE reward requires at least one step-history row")]
    EmptyStepHistory,
    #[error("{history} history column not found: {column}")]
    MissingColumn {
        history: &'static str,
        column: &'static str,
    },
    #[error("{history} history column {column} must be {expected}, got {actual}")]
    InvalidColumnType {
        history: &'static str,
        column: &'static str,
        expected: &'static str,
        actual: DataType,
    },
    #[error("{history} history column {column} contains null at row {row}")]
    NullValue {
        history: &'static str,
        column: &'static str,
        row: usize,
    },
    #[error("PAPenaltyReward produced a non-finite value: {0}")]
    NonFinite(f64),
    #[error("PAPenaltyReward is not attached to a reward logger")]
    MissingLogger,
    #[error("RewardCombination is not attached to a reward logger")]
    MissingCombinationLogger,
    #[error(transparent)]
    Log(#[from] SaoeRewardLogError),
}

pub trait SaoeReward: Send + Sync {
    fn set_logger(&mut self, _logger: Option<Arc<dyn SaoeRewardLogSink>>) {}

    /// # Errors
    /// Returns malformed-state or plugin-side-effect failures.
    fn reward(&self, state: &SaoeState) -> Result<f64, SaoeRewardError>;
}

pub type WeightedSaoeReward = (Arc<dyn SaoeReward>, f64);

/// Insertion-ordered weighted composition of reward plugins.
pub struct RewardCombination {
    rewards: IndexMap<String, WeightedSaoeReward>,
    logger: Option<Arc<dyn SaoeRewardLogSink>>,
}

impl RewardCombination {
    #[must_use]
    pub fn new(rewards: IndexMap<String, WeightedSaoeReward>) -> Self {
        Self {
            rewards,
            logger: None,
        }
    }

    #[must_use]
    pub fn with_logger(mut self, logger: Arc<dyn SaoeRewardLogSink>) -> Self {
        self.logger = Some(logger);
        self
    }

    pub fn set_logger(&mut self, logger: Option<Arc<dyn SaoeRewardLogSink>>) {
        self.logger = logger;
    }
}

impl SaoeReward for RewardCombination {
    fn set_logger(&mut self, logger: Option<Arc<dyn SaoeRewardLogSink>>) {
        self.logger = logger;
    }

    fn reward(&self, state: &SaoeState) -> Result<f64, SaoeRewardError> {
        let mut total_reward = 0.0;
        for (name, (reward_fn, weight)) in &self.rewards {
            let reward = reward_fn.reward(state)? * weight;
            total_reward += reward;
            self.logger
                .as_ref()
                .ok_or(SaoeRewardError::MissingCombinationLogger)?
                .log_scalar(name, reward)?;
        }
        Ok(total_reward)
    }
}

pub struct PaPenaltyReward {
    penalty: f64,
    scale: f64,
    logger: Option<Arc<dyn SaoeRewardLogSink>>,
}

impl Default for PaPenaltyReward {
    fn default() -> Self {
        Self::new(100.0, 1.0)
    }
}

impl PaPenaltyReward {
    #[must_use]
    pub const fn new(penalty: f64, scale: f64) -> Self {
        Self {
            penalty,
            scale,
            logger: None,
        }
    }

    #[must_use]
    pub fn with_logger(mut self, logger: Arc<dyn SaoeRewardLogSink>) -> Self {
        self.logger = Some(logger);
        self
    }
}

impl SaoeReward for PaPenaltyReward {
    fn set_logger(&mut self, logger: Option<Arc<dyn SaoeRewardLogSink>>) {
        self.logger = logger;
    }

    fn reward(&self, state: &SaoeState) -> Result<f64, SaoeRewardError> {
        let parts = state.parts();
        let whole_order = parts.order.amount();
        if whole_order.partial_cmp(&0.0) != Some(Ordering::Greater) {
            return Err(SaoeRewardError::InvalidOrderAmount(whole_order));
        }
        let last_row = parts
            .history_steps
            .num_rows()
            .checked_sub(1)
            .ok_or(SaoeRewardError::EmptyStepHistory)?;
        let last_time = timestamp_value(&parts.history_steps, "steps", "datetime", last_row)?;
        let pa = float_value(&parts.history_steps, "steps", "pa", last_row)?
            * float_value(&parts.history_steps, "steps", "amount", last_row)?
            / whole_order;
        let times = timestamp_column(&parts.history_exec, "executions", "datetime")?;
        let amounts = float_column(&parts.history_exec, "executions", "amount")?;
        let mut squared = 0.0;
        for row in 0..parts.history_exec.num_rows() {
            let timestamp = timestamp_array_value(times, "executions", "datetime", row)?;
            let amount = float_array_value(amounts, "executions", "amount", row)?;
            if timestamp >= last_time {
                squared += (amount / whole_order).powi(2);
            }
        }
        let penalty = -self.penalty * squared;
        let reward = pa + penalty;
        if !reward.is_finite() {
            return Err(SaoeRewardError::NonFinite(reward));
        }
        let logger = self.logger.as_ref().ok_or(SaoeRewardError::MissingLogger)?;
        logger.log_scalar("reward/pa", pa)?;
        logger.log_scalar("reward/penalty", penalty)?;
        Ok(reward * self.scale)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PpoReward {
    max_step: i64,
    start_time_index: i64,
    end_time_index: i64,
}

impl PpoReward {
    #[must_use]
    pub const fn new(max_step: i64, start_time_index: i64, end_time_index: i64) -> Self {
        Self {
            max_step,
            start_time_index,
            end_time_index,
        }
    }

    #[must_use]
    pub const fn max_step(self) -> i64 {
        self.max_step
    }

    #[must_use]
    pub const fn start_time_index(self) -> i64 {
        self.start_time_index
    }

    #[must_use]
    pub const fn end_time_index(self) -> i64 {
        self.end_time_index
    }
}

impl SaoeReward for PpoReward {
    fn reward(&self, state: &SaoeState) -> Result<f64, SaoeRewardError> {
        let parts = state.parts();
        let terminal =
            parts.cur_step.checked_add(1) == Some(self.max_step) || parts.position < 1.0e-6;
        if !terminal {
            return Ok(0.0);
        }
        let market_prices = float_column(&parts.history_exec, "executions", "market_price")?;
        let deal_amounts = float_column(&parts.history_exec, "executions", "deal_amount")?;
        let mut prices = Vec::with_capacity(parts.history_exec.num_rows());
        let mut weights = Vec::with_capacity(parts.history_exec.num_rows());
        for row in 0..parts.history_exec.num_rows() {
            prices.push(float_array_value(
                market_prices,
                "executions",
                "market_price",
                row,
            )?);
            weights.push(float_array_value(
                deal_amounts,
                "executions",
                "deal_amount",
                row,
            )?);
        }
        let weight_sum: f64 = weights.iter().sum();
        let vwap_price = if weight_sum == 0.0 {
            mean(&prices)
        } else {
            prices
                .iter()
                .zip(&weights)
                .map(|(price, weight)| price * weight)
                .sum::<f64>()
                / weight_sum
        };
        let twap_price = mean(parts.backtest_data.deal_prices.as_slice().unwrap_or(&[]));
        let ratio = match parts.order.direction() {
            OrderDir::Sell => {
                if twap_price == 0.0 {
                    1.0
                } else {
                    vwap_price / twap_price
                }
            }
            OrderDir::Buy => {
                if vwap_price == 0.0 {
                    1.0
                } else {
                    twap_price / vwap_price
                }
            }
        };
        Ok(if ratio < 1.0 {
            -1.0
        } else if ratio < 1.1 {
            0.0
        } else {
            1.0
        })
    }
}

fn float_column<'a>(
    batch: &'a RecordBatch,
    history: &'static str,
    column: &'static str,
) -> Result<&'a Float64Array, SaoeRewardError> {
    let index = batch
        .schema_ref()
        .index_of(column)
        .map_err(|_| SaoeRewardError::MissingColumn { history, column })?;
    batch
        .column(index)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| SaoeRewardError::InvalidColumnType {
            history,
            column,
            expected: "Float64",
            actual: batch.column(index).data_type().clone(),
        })
}

fn timestamp_column<'a>(
    batch: &'a RecordBatch,
    history: &'static str,
    column: &'static str,
) -> Result<&'a TimestampNanosecondArray, SaoeRewardError> {
    let index = batch
        .schema_ref()
        .index_of(column)
        .map_err(|_| SaoeRewardError::MissingColumn { history, column })?;
    batch
        .column(index)
        .as_any()
        .downcast_ref::<TimestampNanosecondArray>()
        .ok_or_else(|| SaoeRewardError::InvalidColumnType {
            history,
            column,
            expected: "Timestamp(Nanosecond)",
            actual: batch.column(index).data_type().clone(),
        })
}

fn float_value(
    batch: &RecordBatch,
    history: &'static str,
    column: &'static str,
    row: usize,
) -> Result<f64, SaoeRewardError> {
    float_array_value(float_column(batch, history, column)?, history, column, row)
}

fn timestamp_value(
    batch: &RecordBatch,
    history: &'static str,
    column: &'static str,
    row: usize,
) -> Result<i64, SaoeRewardError> {
    timestamp_array_value(
        timestamp_column(batch, history, column)?,
        history,
        column,
        row,
    )
}

fn float_array_value(
    values: &Float64Array,
    history: &'static str,
    column: &'static str,
    row: usize,
) -> Result<f64, SaoeRewardError> {
    if values.is_null(row) {
        return Err(SaoeRewardError::NullValue {
            history,
            column,
            row,
        });
    }
    Ok(values.value(row))
}

fn timestamp_array_value(
    values: &TimestampNanosecondArray,
    history: &'static str,
    column: &'static str,
    row: usize,
) -> Result<i64, SaoeRewardError> {
    if values.is_null(row) {
        return Err(SaoeRewardError::NullValue {
            history,
            column,
            row,
        });
    }
    Ok(values.value(row))
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        f64::NAN
    } else {
        values.iter().sum::<f64>() / values.len().to_f64().expect("slice length fits f64")
    }
}
