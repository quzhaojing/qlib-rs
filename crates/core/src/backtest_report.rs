//! Final, ordered report aggregation for the outer backtest loop.

use indexmap::IndexMap;
use thiserror::Error;

use crate::{
    Account, AccountError, AccountIndicatorError, AccountPortfolioReport, Frequency,
    FrequencyError, SharedAccountIndicator, SharedSaoeAccount, TradeIndicatorReport,
};

/// A frozen numeric table and an independently retained original live indicator object.
pub struct BacktestIndicatorReport {
    pub table: TradeIndicatorReport,
    pub indicator: SharedAccountIndicator,
}

/// Reports keyed by explicit count and normalized frequency unit (including `1day`).
pub struct BacktestReports {
    pub portfolio: IndexMap<String, AccountPortfolioReport>,
    pub indicators: IndexMap<String, BacktestIndicatorReport>,
}

#[derive(Debug, Error)]
pub enum BacktestReportError {
    #[error(transparent)]
    Frequency(#[from] FrequencyError),
    #[error(transparent)]
    Account(#[from] AccountError),
    #[error(transparent)]
    Indicator(#[from] AccountIndicatorError),
    #[error("shared report account lock poisoned at frequency {frequency}")]
    AccountLock { frequency: String },
}

/// Collect reports in executor traversal order, normally outermost to innermost.
///
/// Later enabled portfolios and all later indicators overwrite duplicate keys without
/// changing key order. A disabled portfolio does not remove an earlier portfolio.
/// Tables are owned; historical-position handles retain their original mapping, while
/// indicator handles retain their original engine. Neither engine nor positions are cloned.
/// This function does not publish partial maps on failure. The caller must invoke it only
/// after the outer loop's strategy finalization and progress-scope exit have succeeded.
///
/// # Errors
/// Returns the first frequency, portfolio, or indicator-export error, retaining its type.
pub fn collect_backtest_reports<'a>(
    levels: impl IntoIterator<Item = (&'a str, &'a Account)>,
) -> Result<BacktestReports, BacktestReportError> {
    collect_reports(&mut levels.into_iter())
}

/// Collect reports from shared accounts without retaining more than one account lock.
/// Duplicate account handles are processed independently in traversal order.
///
/// # Errors
/// Returns the first lock, frequency, account, or indicator failure and publishes no maps.
pub fn collect_shared_backtest_reports<'a>(
    levels: impl IntoIterator<Item = (&'a str, &'a SharedSaoeAccount)>,
) -> Result<BacktestReports, BacktestReportError> {
    let mut portfolio = IndexMap::new();
    let mut indicators = IndexMap::new();
    for (frequency, account) in levels {
        let account = account
            .lock()
            .map_err(|_| BacktestReportError::AccountLock {
                frequency: frequency.to_owned(),
            })?;
        append_report(frequency, &account, &mut portfolio, &mut indicators)?;
    }
    Ok(BacktestReports {
        portfolio,
        indicators,
    })
}

// Keep the fallible aggregation in one implementation for all input iterator types.
// This avoids duplicating the report-building pipeline for arrays, vectors and adapters.
fn collect_reports<'a>(
    levels: &mut dyn Iterator<Item = (&'a str, &'a Account)>,
) -> Result<BacktestReports, BacktestReportError> {
    let mut portfolio = IndexMap::new();
    let mut indicators = IndexMap::new();
    for (frequency, account) in levels {
        append_report(frequency, account, &mut portfolio, &mut indicators)?;
    }
    Ok(BacktestReports {
        portfolio,
        indicators,
    })
}

fn append_report(
    frequency: &str,
    account: &Account,
    portfolio: &mut IndexMap<String, AccountPortfolioReport>,
    indicators: &mut IndexMap<String, BacktestIndicatorReport>,
) -> Result<(), BacktestReportError> {
    let frequency: Frequency = frequency.parse()?;
    let key = format!("{}{}", frequency.count, frequency.unit);
    if account.is_portfolio_metrics_enabled() {
        portfolio.insert(key.clone(), account.portfolio_report()?);
    }
    let indicator = account.indicator().clone();
    let table = indicator
        .read()
        .map_err(|_| AccountIndicatorError {
            message: "account indicator lock poisoned".to_owned(),
        })?
        .trade_indicator_report()?;
    indicators.insert(key, BacktestIndicatorReport { table, indicator });
    Ok(())
}
