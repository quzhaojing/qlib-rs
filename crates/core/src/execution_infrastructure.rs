//! Live, typed exchange bindings used by execution-calendar data-range queries.

use std::sync::{Arc, RwLock};

use crate::{ExchangeQuoteProvider, ExecutionCalendarContext, ExecutionCalendarError};

/// One shared exchange's data frequency and its quote service. Frequency spelling is
/// retained verbatim; querying it does not parse, normalize or rebuild quote data.
/// This is the calendar/quote projection, not the complete exchange execution engine.
pub struct ExecutionExchange {
    pub frequency: String,
    pub quotes: ExchangeQuoteProvider,
}

/// The exchange binding shared by execution levels. Account ownership remains in the
/// existing account adapters. `None` is an unbound typed slot, not a Python null object.
#[derive(Default)]
pub struct ExecutionCommonBindings {
    pub exchange: Option<Arc<RwLock<ExecutionExchange>>>,
}

/// Level-local reference to the current common bindings. Replacing this reference,
/// replacing its exchange, or editing exchange frequency is visible on the next call.
#[derive(Default)]
pub struct ExecutionLevelBindings {
    pub common: Option<Arc<RwLock<ExecutionCommonBindings>>>,
}

impl ExecutionLevelBindings {
    /// Acquire the currently bound exchange without retaining infrastructure guards.
    /// Callers can use the same exchange for quote access and calendar frequency.
    ///
    /// # Errors
    /// Returns missing-binding or poisoned-lock failures without implicit recovery.
    pub fn exchange(
        level: &RwLock<Self>,
    ) -> Result<Arc<RwLock<ExecutionExchange>>, ExecutionCalendarError> {
        let common = level.read().map_err(|_| poisoned("level"))?.common.clone();
        let common = common.ok_or_else(|| missing("common_infra"))?;
        let exchange = common
            .read()
            .map_err(|_| poisoned("common"))?
            .exchange
            .clone();
        let exchange = exchange.ok_or_else(|| missing("trade_exchange"))?;
        Ok(exchange)
    }
}

impl ExecutionCalendarContext for RwLock<ExecutionLevelBindings> {
    fn data_frequency(&self) -> Result<String, ExecutionCalendarError> {
        let exchange = ExecutionLevelBindings::exchange(self)?;
        let frequency = exchange
            .read()
            .map_err(|_| poisoned("exchange"))?
            .frequency
            .clone();
        Ok(frequency)
    }
}

fn poisoned(binding: &str) -> ExecutionCalendarError {
    ExecutionCalendarError::Provider(format!("execution {binding} binding lock poisoned"))
}

fn missing(binding: &str) -> ExecutionCalendarError {
    tracing::warn!("infra {binding} is not found!");
    ExecutionCalendarError::Provider(format!("missing execution infrastructure: {binding}"))
}
