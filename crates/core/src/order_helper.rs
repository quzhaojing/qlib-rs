//! Exchange-bound helper for source-compatible order construction.

use chrono::NaiveDateTime;
use thiserror::Error;

use crate::{Order, OrderDir};

/// Input accepted by the typed `OrderHelper.create` boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OrderTimeInput {
    /// An already materialized timestamp, still passed through the parser just like
    /// `pandas.Timestamp(existing_timestamp)`.
    Timestamp(NaiveDateTime),
    /// Text interpreted by the configured Pandas-compatible parser plugin.
    Text(String),
}

impl From<NaiveDateTime> for OrderTimeInput {
    fn from(value: NaiveDateTime) -> Self {
        Self::Timestamp(value)
    }
}

impl From<String> for OrderTimeInput {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for OrderTimeInput {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

/// Replaceable timestamp conversion boundary corresponding to `pandas.Timestamp`.
pub trait OrderTimestampParser {
    /// Normalize one non-null order timestamp input.
    ///
    /// # Errors
    ///
    /// Returns the parser's conversion or range failure.
    fn parse(&self, input: OrderTimeInput) -> Result<NaiveDateTime, OrderTimestampParseError>;
}

/// Failure returned by an order timestamp parser plugin.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("order timestamp parsing failed: {message}")]
pub struct OrderTimestampParseError {
    pub message: String,
}

/// Ordered failures from `OrderHelper.create`.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum OrderHelperError {
    #[error("order start time is invalid: {0}")]
    StartTime(#[source] OrderTimestampParseError),
    #[error("order end time is invalid: {0}")]
    EndTime(#[source] OrderTimestampParseError),
}

/// Helper retaining the exact exchange handle from which it was obtained.
///
/// The source `create` method is static and does not inspect this handle. Keeping it here preserves
/// helper/exchange identity while allowing concrete exchange implementations to choose `Arc`, a
/// borrow, or an owned handle.
pub struct OrderHelper<E, P> {
    exchange: E,
    timestamps: P,
}

impl<E, P> OrderHelper<E, P> {
    #[must_use]
    pub const fn new(exchange: E, timestamps: P) -> Self {
        Self {
            exchange,
            timestamps,
        }
    }

    /// Return the retained exchange handle without cloning or resolving it.
    #[must_use]
    pub const fn exchange(&self) -> &E {
        &self.exchange
    }
}

impl<E, P: OrderTimestampParser> OrderHelper<E, P> {
    /// Create an order through this cached helper.
    ///
    /// # Errors
    ///
    /// Returns the first start/end timestamp conversion failure in source order.
    pub fn create(
        &self,
        code: impl Into<String>,
        amount: f64,
        direction: OrderDir,
        start_time: Option<OrderTimeInput>,
        end_time: Option<OrderTimeInput>,
    ) -> Result<Order, OrderHelperError> {
        create_order(
            code,
            amount,
            direction,
            start_time,
            end_time,
            &self.timestamps,
        )
    }
}

/// Static `OrderHelper.create` equivalent for callers that do not need an exchange-bound helper.
///
/// # Errors
///
/// Returns the first start/end timestamp conversion failure in source order.
pub fn create_order<P: OrderTimestampParser + ?Sized>(
    code: impl Into<String>,
    amount: f64,
    direction: OrderDir,
    start_time: Option<OrderTimeInput>,
    end_time: Option<OrderTimeInput>,
    timestamps: &P,
) -> Result<Order, OrderHelperError> {
    let start_time = start_time
        .map(|value| timestamps.parse(value).map_err(OrderHelperError::StartTime))
        .transpose()?;
    let end_time = end_time
        .map(|value| timestamps.parse(value).map_err(OrderHelperError::EndTime))
        .transpose()?;
    Ok(Order::new(code, amount, direction, start_time, end_time))
}
