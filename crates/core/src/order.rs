use chrono::{NaiveDateTime, Timelike};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use strum::{AsRefStr, Display, EnumString, VariantArray};
use thiserror::Error;

use crate::{DenseIndicatorTransform, DenseIndicatorValue, DenseMetric, SingleData};

/// Buy/sell direction used throughout Qlib's backtest domain.
#[derive(
    Clone,
    Copy,
    Debug,
    Display,
    EnumString,
    AsRefStr,
    VariantArray,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
)]
#[repr(i8)]
#[strum(serialize_all = "lowercase", ascii_case_insensitive)]
pub enum OrderDir {
    Sell = 0,
    Buy = 1,
}

impl OrderDir {
    #[must_use]
    pub const fn value(self) -> i8 {
        self as i8
    }

    #[must_use]
    pub const fn sign(self) -> i8 {
        match self {
            Self::Sell => -1,
            Self::Buy => 1,
        }
    }

    /// Parses Qlib's trimmed, ASCII-case-insensitive string representation.
    ///
    /// # Errors
    ///
    /// Returns an error unless the value is exactly `buy` or `sell` after trimming.
    pub fn parse_text(value: &str) -> Result<Self, OrderError> {
        value
            .trim()
            .parse()
            .map_err(|_| OrderError::UnsupportedDirection(value.to_owned()))
    }

    /// Implements scalar `Order.parse_dir`: every positive number buys; all others sell.
    #[must_use]
    pub fn parse_number(value: f64) -> Self {
        if value > 0.0 { Self::Buy } else { Self::Sell }
    }

    /// Implements NumPy-array `Order.parse_dir`, preserving NaN values.
    #[must_use]
    pub fn parse_values(values: &[f64]) -> Vec<f64> {
        values
            .iter()
            .map(|value| {
                if *value > 0.0 {
                    f64::from(Self::Buy.value())
                } else if *value <= 0.0 {
                    f64::from(Self::Sell.value())
                } else {
                    *value
                }
            })
            .collect()
    }
}

impl TryFrom<i64> for OrderDir {
    type Error = OrderError;

    fn try_from(value: i64) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Sell),
            1 => Ok(Self::Buy),
            _ => Err(OrderError::InvalidDirectionCode(value)),
        }
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum OrderError {
    #[error("direction not supported: {0}")]
    UnsupportedDirection(String),
    #[error("direction code must be 0 (sell) or 1 (buy), got {0}")]
    InvalidDirectionCode(i64),
    #[error("order start time is required for its day key")]
    MissingStartTime,
}

pub type OrderKey<'a> = (
    &'a str,
    Option<NaiveDateTime>,
    Option<NaiveDateTime>,
    OrderDir,
);
pub type OrderDayKey<'a> = (&'a str, NaiveDateTime, OrderDir);

/// Core order value and mutable execution result state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Order {
    stock_id: String,
    amount: f64,
    direction: OrderDir,
    start_time: Option<NaiveDateTime>,
    end_time: Option<NaiveDateTime>,
    deal_amount: f64,
    factor: Option<f64>,
}

impl Order {
    pub const SELL: OrderDir = OrderDir::Sell;
    pub const BUY: OrderDir = OrderDir::Buy;

    #[must_use]
    pub fn new(
        stock_id: impl Into<String>,
        amount: f64,
        direction: OrderDir,
        start_time: Option<NaiveDateTime>,
        end_time: Option<NaiveDateTime>,
    ) -> Self {
        Self {
            stock_id: stock_id.into(),
            amount,
            direction,
            start_time,
            end_time,
            deal_amount: 0.0,
            factor: None,
        }
    }

    /// Constructs an order from Python-compatible integer direction input.
    ///
    /// # Errors
    ///
    /// Returns an error unless the direction code is exactly zero or one.
    pub fn try_new(
        stock_id: impl Into<String>,
        amount: f64,
        direction: i64,
        start_time: Option<NaiveDateTime>,
        end_time: Option<NaiveDateTime>,
    ) -> Result<Self, OrderError> {
        Ok(Self::new(
            stock_id,
            amount,
            OrderDir::try_from(direction)?,
            start_time,
            end_time,
        ))
    }

    #[must_use]
    pub fn stock_id(&self) -> &str {
        &self.stock_id
    }

    #[must_use]
    pub fn amount(&self) -> f64 {
        self.amount
    }

    #[must_use]
    pub const fn direction(&self) -> OrderDir {
        self.direction
    }

    #[must_use]
    pub const fn start_time(&self) -> Option<NaiveDateTime> {
        self.start_time
    }

    #[must_use]
    pub const fn end_time(&self) -> Option<NaiveDateTime> {
        self.end_time
    }

    pub(crate) fn fill_missing_interval(
        &mut self,
        default_start: NaiveDateTime,
        default_end: NaiveDateTime,
    ) {
        if self.start_time.is_none() {
            self.start_time = Some(default_start);
        }
        if self.end_time.is_none() {
            self.end_time = Some(default_end);
        }
    }

    #[must_use]
    pub fn deal_amount(&self) -> f64 {
        self.deal_amount
    }

    #[must_use]
    pub fn factor(&self) -> Option<f64> {
        self.factor
    }

    pub fn set_deal_amount(&mut self, deal_amount: f64) {
        self.deal_amount = deal_amount;
    }

    pub fn set_factor(&mut self, factor: Option<f64>) {
        self.factor = factor;
    }

    pub fn reset_results(&mut self) {
        self.deal_amount = 0.0;
        self.factor = None;
    }

    #[must_use]
    pub const fn sign(&self) -> i8 {
        self.direction.sign()
    }

    #[must_use]
    pub fn amount_delta(&self) -> f64 {
        self.amount * f64::from(self.sign())
    }

    #[must_use]
    pub fn deal_amount_delta(&self) -> f64 {
        self.deal_amount * f64::from(self.sign())
    }

    #[must_use]
    pub fn key(&self) -> OrderKey<'_> {
        (
            &self.stock_id,
            self.start_time,
            self.end_time,
            self.direction,
        )
    }

    /// Returns the Python `date` property, which is midnight with subsecond precision retained.
    ///
    /// # Errors
    ///
    /// Returns an error when the order has no start time.
    ///
    /// # Panics
    ///
    /// Panics only if Chrono reports an invalid nanosecond value from an existing
    /// `NaiveDateTime`, which would violate Chrono's type invariant.
    pub fn day_timestamp(&self) -> Result<NaiveDateTime, OrderError> {
        let start = self.start_time.ok_or(OrderError::MissingStartTime)?;
        Ok(start
            .date()
            .and_hms_nano_opt(0, 0, 0, start.nanosecond())
            .expect("an existing NaiveDateTime has a valid nanosecond component"))
    }

    /// Returns the stock/day/direction key.
    ///
    /// # Errors
    ///
    /// Returns an error when the order has no start time.
    pub fn key_by_day(&self) -> Result<OrderDayKey<'_>, OrderError> {
        Ok((&self.stock_id, self.day_timestamp()?, self.direction))
    }
}

/// Dense transform adapter for `Order.parse_dir`'s NumPy-array path.
#[derive(Clone, Debug)]
pub struct ParseOrderDirectionTransform {
    inputs: [String; 1],
}

impl ParseOrderDirectionTransform {
    #[must_use]
    pub fn new(input: impl Into<String>) -> Self {
        Self {
            inputs: [input.into()],
        }
    }
}

impl DenseIndicatorTransform for ParseOrderDirectionTransform {
    fn input_names(&self) -> &[String] {
        &self.inputs
    }

    fn apply(&self, inputs: &[&dyn DenseMetric]) -> Result<DenseIndicatorValue, String> {
        let [input] = inputs else {
            return Err(format!(
                "Order.parse_dir expects one dense metric, got {}",
                inputs.len()
            ));
        };
        SingleData::try_new(
            input.index().to_vec(),
            Array1::from_vec(OrderDir::parse_values(input.values())),
        )
        .map(DenseIndicatorValue::Metric)
        .map_err(|error| error.to_string())
    }
}
