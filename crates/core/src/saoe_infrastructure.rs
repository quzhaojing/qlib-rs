//! Concrete bridges from migrated Exchange, calendar, and account infrastructure into SAOE.

use std::sync::{Arc, Mutex};

use chrono::NaiveDateTime;
use ndarray::Array1;
use num_traits::ToPrimitive;

use crate::{
    Account, BasePriceDataProvider, ExchangeQuoteProvider, Frequency, FrequencyUnit,
    MarketDataValue, NestedCalendar, NestedOuterDecision, OrderDir, SaoeAdapterContext,
    SaoeAdapterMarket, SaoeAdapterRuntime, SaoeBacktestDataSource, SaoeMarketSlice,
    SaoePluginError, TimeRange,
};

/// Shared mutable calendar used by the nested executor and SAOE observations.
pub type SharedSaoeCalendar = Arc<Mutex<Box<dyn NestedCalendar>>>;

/// Shared mutable account used by the executor and SAOE indicator observations.
pub type SharedSaoeAccount = Arc<Mutex<Account>>;

/// Exchange-backed implementation of both eager backtest loading and per-step market access.
#[derive(Clone)]
pub struct ExchangeSaoeMarket {
    exchange: ExchangeQuoteProvider,
    quote_timestamps: Arc<[NaiveDateTime]>,
}

impl ExchangeSaoeMarket {
    #[must_use]
    pub fn new(
        exchange: ExchangeQuoteProvider,
        quote_timestamps: impl Into<Arc<[NaiveDateTime]>>,
    ) -> Self {
        Self {
            exchange,
            quote_timestamps: quote_timestamps.into(),
        }
    }

    fn deal_prices(
        &self,
        stock_id: &str,
        start: NaiveDateTime,
        end: NaiveDateTime,
        direction: OrderDir,
    ) -> Result<Array1<f64>, SaoePluginError> {
        self.exchange
            .deal_price(
                stock_id,
                TimeRange {
                    start: Some(start),
                    end: Some(end),
                },
                direction,
            )
            .map_err(plugin_error)
            .and_then(|value| market_array("deal price", value))
    }

    fn market_volumes(
        &self,
        stock_id: &str,
        start: NaiveDateTime,
        end: NaiveDateTime,
    ) -> Result<Array1<f64>, SaoePluginError> {
        self.exchange
            .volume(
                stock_id,
                TimeRange {
                    start: Some(start),
                    end: Some(end),
                },
            )
            .map_err(plugin_error)
            .and_then(|value| market_array("volume", value))
    }
}

impl SaoeBacktestDataSource for ExchangeSaoeMarket {
    fn quote_timestamps(&self) -> Result<Vec<NaiveDateTime>, SaoePluginError> {
        Ok(self.quote_timestamps.to_vec())
    }

    fn deal_prices(
        &self,
        stock_id: &str,
        start: NaiveDateTime,
        end: NaiveDateTime,
        direction: OrderDir,
    ) -> Result<Array1<f64>, SaoePluginError> {
        self.deal_prices(stock_id, start, end, direction)
    }

    fn market_volumes(
        &self,
        stock_id: &str,
        start: NaiveDateTime,
        end: NaiveDateTime,
    ) -> Result<Array1<f64>, SaoePluginError> {
        self.market_volumes(stock_id, start, end)
    }
}

impl SaoeAdapterMarket for ExchangeSaoeMarket {
    fn market_slice(
        &self,
        stock_id: &str,
        start: NaiveDateTime,
        end: NaiveDateTime,
        direction: OrderDir,
    ) -> Result<SaoeMarketSlice, SaoePluginError> {
        let volume = self.market_volumes(stock_id, start, end)?;
        let price = self.deal_prices(stock_id, start, end, direction)?;
        Ok(SaoeMarketSlice { volume, price })
    }
}

/// Nested-calendar and Account bridge shared by adapter construction and live state updates.
pub struct NestedAccountSaoeContext {
    calendar: SharedSaoeCalendar,
    account: SharedSaoeAccount,
    frequency: String,
}

impl NestedAccountSaoeContext {
    #[must_use]
    pub fn new(
        calendar: SharedSaoeCalendar,
        account: SharedSaoeAccount,
        frequency: impl Into<String>,
    ) -> Self {
        Self {
            calendar,
            account,
            frequency: frequency.into(),
        }
    }

    fn calendar(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, Box<dyn NestedCalendar>>, SaoePluginError> {
        self.calendar.lock().map_err(|_| SaoePluginError {
            message: "SAOE nested calendar lock is poisoned".to_owned(),
        })
    }
}

impl SaoeAdapterRuntime for NestedAccountSaoeContext {
    fn ticks_per_step(&self) -> Result<usize, SaoePluginError> {
        let frequency: Frequency = self.frequency.parse().map_err(plugin_error)?;
        if frequency.unit == FrequencyUnit::Month {
            return Err(SaoePluginError {
                message: "SAOE executor frequency cannot use calendar months".to_owned(),
            });
        }
        frequency
            .approximate_minutes()
            .to_usize()
            .ok_or_else(|| SaoePluginError {
                message: format!("SAOE executor frequency is too large: {}", self.frequency),
            })
    }

    fn start_step(&self, outer: &dyn NestedOuterDecision) -> Result<i64, SaoePluginError> {
        let calendar = self.calendar()?;
        outer
            .range_limit(&**calendar)
            .map(|range| range.map_or(0, |(start, _)| start))
            .map_err(plugin_error)
    }
}

impl SaoeAdapterContext for NestedAccountSaoeContext {
    fn current_trade_step(&self) -> Result<i64, SaoePluginError> {
        self.calendar()?.trade_step().map_err(plugin_error)
    }

    fn latest_price_advantage(&self) -> Result<f64, SaoePluginError> {
        let account = self.account.lock().map_err(|_| SaoePluginError {
            message: "SAOE account lock is poisoned".to_owned(),
        })?;
        account
            .indicator()
            .read()
            .map_err(|_| SaoePluginError {
                message: "account indicator lock poisoned".to_owned(),
            })?
            .trade_indicator()
            .read()
            .map_err(|_| SaoePluginError {
                message: "trade indicator row lock poisoned".to_owned(),
            })?
            .get("pa")
            .copied()
            .ok_or_else(|| SaoePluginError {
                message: "SAOE account indicator has no current price advantage".to_owned(),
            })
    }

    fn warn_overfill(&self, execution_volume: f64, position: f64) -> Result<(), SaoePluginError> {
        tracing::warn!(
            execution_volume,
            position,
            "SAOE execution volume exceeds position by more than one unit"
        );
        Ok(())
    }
}

fn market_array(
    kind: &'static str,
    value: Option<MarketDataValue>,
) -> Result<Array1<f64>, SaoePluginError> {
    match value {
        Some(MarketDataValue::Series(series)) => Ok(Array1::from_vec(series.values().to_vec())),
        Some(MarketDataValue::Scalar(_)) => Err(SaoePluginError {
            message: format!("SAOE {kind} query returned a scalar instead of a series"),
        }),
        None => Err(SaoePluginError {
            message: format!("SAOE {kind} query returned no data"),
        }),
    }
}

fn plugin_error(error: impl std::fmt::Display) -> SaoePluginError {
    SaoePluginError {
        message: error.to_string(),
    }
}
