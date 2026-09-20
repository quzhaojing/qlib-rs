//! Source-ordered SAOE market queries over the original mutable order.
use std::sync::{Arc, RwLock};

use chrono::NaiveDateTime;
use thiserror::Error;

use crate::{Order, SaoeBacktestDataSource, SaoeMarketSlice, SaoePluginError};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum LiveSaoeMarketError {
    #[error("SAOE order lock poisoned before volume query")]
    VolumeOrderPoisoned,
    #[error("SAOE volume query failed: {0}")]
    Volume(SaoePluginError),
    #[error("SAOE order lock poisoned before price query")]
    PriceOrderPoisoned,
    #[error("SAOE price query failed: {0}")]
    Price(SaoePluginError),
}

/// Reuse the existing separate exchange-data operations, including `ExchangeSaoeMarket`.
/// The volume callback may modify the original order; price arguments are read afterwards.
/// No order guard spans a plugin callback. No timeline lookup, filling, clipping or shape
/// validation occurs here: the numerical adapter must perform those in source order later.
///
/// # Errors
/// Returns the first reached order-access or market-plugin failure without rollback.
pub fn read_live_saoe_market(
    source: &dyn SaoeBacktestDataSource,
    order: &Arc<RwLock<Order>>,
    start: NaiveDateTime,
    end: NaiveDateTime,
) -> Result<SaoeMarketSlice, LiveSaoeMarketError> {
    let stock = order
        .read()
        .map_err(|_| LiveSaoeMarketError::VolumeOrderPoisoned)?
        .stock_id()
        .to_owned();
    let volume = source
        .market_volumes(&stock, start, end)
        .map_err(LiveSaoeMarketError::Volume)?;
    let (stock, direction) = {
        let order = order
            .read()
            .map_err(|_| LiveSaoeMarketError::PriceOrderPoisoned)?;
        (order.stock_id().to_owned(), order.direction())
    };
    let price = source
        .deal_prices(&stock, start, end, direction)
        .map_err(LiveSaoeMarketError::Price)?;
    Ok(SaoeMarketSlice { volume, price })
}
