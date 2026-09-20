//! Construction dispatcher for `qlib.backtest.get_exchange`.

use std::{path::PathBuf, sync::Arc};

use chrono::NaiveDateTime;
use indexmap::IndexMap;
use serde_json::Value;
use thiserror::Error;

use crate::Region;

/// Typed price-limit forms accepted by the Exchange constructor.
#[derive(Clone, Debug, PartialEq)]
pub enum ExchangeLimitThreshold {
    Rate(f64),
    Expressions { buy: String, sell: String },
}

/// Raw deal-price input passed through to the Exchange constructor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExchangeDealPriceInput {
    Shared(String),
    Sequence(Vec<String>),
}

/// Instrument universe selector or an already resolved stable list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExchangeCodes {
    Universe(String),
    Instruments(Vec<String>),
}

/// Timestamp forms accepted by the Python wrapper and normalized by the factory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExchangeTimeInput {
    Timestamp(NaiveDateTime),
    Text(String),
}

/// All arguments forwarded when `exchange=None` selects new construction.
#[derive(Clone, Debug, PartialEq)]
pub struct ExchangeConstructionRequest<K = IndexMap<String, Value>> {
    pub frequency: String,
    pub start_time: Option<ExchangeTimeInput>,
    pub end_time: Option<ExchangeTimeInput>,
    pub codes: ExchangeCodes,
    pub subscribe_fields: Vec<String>,
    pub open_cost: f64,
    pub close_cost: f64,
    pub min_cost: f64,
    pub limit_threshold: Option<ExchangeLimitThreshold>,
    pub deal_price: Option<ExchangeDealPriceInput>,
    /// Caller-chosen opaque representation of the wrapper's complete `**kwargs`.
    pub extra_arguments: K,
}

/// Serializable configuration forms accepted by `init_instance_by_config`.
#[derive(Clone, Debug, PartialEq)]
pub enum ExchangeConfiguration {
    Name(String),
    Mapping(IndexMap<String, Value>),
    Path(PathBuf),
}

/// Object-safe identity retained for constructed or pre-existing exchanges.
pub trait ExchangeInstance: Send + Sync {}

impl<T: Send + Sync> ExchangeInstance for T {}

/// The optional wrapper argument selecting new, existing, or configured exchange handling.
pub enum ExchangeSource {
    New,
    Existing(Arc<dyn ExchangeInstance>),
    Configuration(ExchangeConfiguration),
}

/// Failure returned by configurable defaults or exchange factories.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("{message}")]
pub struct ExchangeConstructionPluginError {
    pub message: String,
}

/// Live global configuration boundary read before source dispatch.
pub trait ExchangeDefaults {
    /// Return the current global price-limit default.
    ///
    /// # Errors
    ///
    /// Returns a configuration acquisition or synchronization failure.
    fn limit_threshold(
        &self,
    ) -> Result<Option<ExchangeLimitThreshold>, ExchangeConstructionPluginError>;
}

/// Region-backed equivalent of Qlib's current `C.limit_threshold` setting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegionExchangeDefaults {
    pub region: Region,
}

impl ExchangeDefaults for RegionExchangeDefaults {
    fn limit_threshold(
        &self,
    ) -> Result<Option<ExchangeLimitThreshold>, ExchangeConstructionPluginError> {
        Ok(self
            .region
            .defaults()
            .limit_threshold
            .map(ExchangeLimitThreshold::Rate))
    }
}

/// Pluggable native constructor and configured-instance resolver.
pub trait ExchangeFactory<K = IndexMap<String, Value>> {
    /// Construct a fresh exchange using every forwarded wrapper argument.
    ///
    /// # Errors
    ///
    /// Returns data, configuration, quote, or component-construction failures.
    fn create(
        &self,
        request: ExchangeConstructionRequest<K>,
    ) -> Result<Arc<dyn ExchangeInstance>, ExchangeConstructionPluginError>;

    /// Resolve an existing instance or serialized/name/mapping configuration.
    ///
    /// # Errors
    ///
    /// Returns type validation, loading, deserialization, or factory failures.
    fn resolve(
        &self,
        source: ExchangeSource,
    ) -> Result<Arc<dyn ExchangeInstance>, ExchangeConstructionPluginError>;
}

/// Typed errors preserving the failing wrapper stage.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum GetExchangeError {
    #[error("exchange default resolution failed: {0}")]
    Defaults(#[source] ExchangeConstructionPluginError),
    #[error("exchange construction failed: {0}")]
    Factory(#[source] ExchangeConstructionPluginError),
}

/// Apply `get_exchange` default lookup and dispatch in source order.
///
/// Parameters other than the limit threshold are intentionally ignored for a
/// configured source, matching the Python wrapper. The factory still receives
/// existing objects so it can enforce the source `accept_types=Exchange` rule.
///
/// # Errors
///
/// Returns a live-default failure before dispatch, or the selected factory error.
pub fn get_exchange<K>(
    source: ExchangeSource,
    mut request: ExchangeConstructionRequest<K>,
    defaults: &dyn ExchangeDefaults,
    factory: &dyn ExchangeFactory<K>,
) -> Result<Arc<dyn ExchangeInstance>, GetExchangeError> {
    if request.limit_threshold.is_none() {
        request.limit_threshold = defaults
            .limit_threshold()
            .map_err(GetExchangeError::Defaults)?;
    }

    match source {
        ExchangeSource::New => {
            tracing::info!(target: "backtest caller", "Create new exchange");
            factory.create(request).map_err(GetExchangeError::Factory)
        }
        configured => factory
            .resolve(configured)
            .map_err(GetExchangeError::Factory),
    }
}
