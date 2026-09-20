//! Direction-aware Exchange volume-limit clipping.

use std::{collections::HashMap, sync::Arc};

use thiserror::Error;

use crate::{ExchangeQuoteProvider, Order, OrderDir, QuoteMethod, TimeRange};

/// Aggregation semantics configured by Qlib's volume-threshold tuple.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VolumeLimitKind {
    /// A per-interval capacity, queried with Pandas `sum`.
    Current,
    /// A cumulative capacity, queried with `ts_data_last` and reduced by prior fills.
    Cumulative,
}

impl VolumeLimitKind {
    /// Parse the exact strings accepted by `_clip_amount_by_volume`.
    ///
    /// # Errors
    ///
    /// Returns a typed error for any spelling other than `current` or `cum`.
    pub fn from_python_name(value: &str) -> Result<Self, ExchangeVolumeError> {
        match value {
            "current" => Ok(Self::Current),
            "cum" => Ok(Self::Cumulative),
            _ => Err(ExchangeVolumeError::UnsupportedLimitKind {
                kind: value.to_owned(),
            }),
        }
    }

    /// Return Qlib's stable configuration spelling.
    #[must_use]
    pub const fn python_name(self) -> &'static str {
        match self {
            Self::Current => "current",
            Self::Cumulative => "cum",
        }
    }

    const fn quote_method(self) -> QuoteMethod {
        match self {
            Self::Current => QuoteMethod::BuiltIn(crate::BuiltInAggregation::Sum),
            Self::Cumulative => QuoteMethod::LastValid,
        }
    }
}

/// One typed `(mode, expression)` volume limit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VolumeLimit {
    kind: VolumeLimitKind,
    field: String,
}

impl VolumeLimit {
    #[must_use]
    pub fn new(kind: VolumeLimitKind, field: impl Into<String>) -> Self {
        Self {
            kind,
            field: field.into(),
        }
    }

    #[must_use]
    pub const fn kind(&self) -> VolumeLimitKind {
        self.kind
    }

    #[must_use]
    pub fn field(&self) -> &str {
        &self.field
    }
}

/// Diagnostic returned by a replaceable volume-limit data provider.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("{message}")]
pub struct VolumeLimitProviderError {
    pub message: String,
}

/// Object-safe market-data seam for Exchange volume limits.
pub trait VolumeLimitProvider: Send + Sync {
    /// Return the scalar capacity for one expression and interval.
    ///
    /// `None` preserves an absent Python quote result for typed validation by the limiter.
    ///
    /// # Errors
    ///
    /// Returns a provider-specific diagnostic when the market-data lookup or conversion fails.
    fn volume_limit(
        &self,
        stock: &str,
        range: TimeRange,
        limit: &VolumeLimit,
    ) -> Result<Option<f64>, VolumeLimitProviderError>;
}

impl VolumeLimitProvider for ExchangeQuoteProvider {
    fn volume_limit(
        &self,
        stock: &str,
        range: TimeRange,
        limit: &VolumeLimit,
    ) -> Result<Option<f64>, VolumeLimitProviderError> {
        self.get_aggregated_scalar(stock, range, limit.field(), limit.kind().quote_method())
            .map_err(|error| VolumeLimitProviderError {
                message: error.to_string(),
            })
    }
}

/// Typed failures from volume-limit configuration, lookup, and cumulative accounting.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ExchangeVolumeError {
    #[error("unsupported volume-limit kind: {kind}")]
    UnsupportedLimitKind { kind: String },
    #[error("{direction} volume-limit list is configured but empty")]
    EmptyLimits { direction: OrderDir },
    #[error("volume limits require a configured provider")]
    MissingProvider,
    #[error("volume-limit provider returned no value for field {field}")]
    MissingLimitValue { field: String },
    #[error("cumulative volume limit has no prior dealt amount for stock {stock}")]
    MissingDealtAmount { stock: String },
    #[error(transparent)]
    Provider(#[from] VolumeLimitProviderError),
}

/// Direction-aware, state-free implementation of `Exchange._clip_amount_by_volume`.
#[derive(Clone)]
pub struct ExchangeVolumeLimiter {
    buy_limits: Option<Vec<VolumeLimit>>,
    sell_limits: Option<Vec<VolumeLimit>>,
    provider: Option<Arc<dyn VolumeLimitProvider>>,
}

impl ExchangeVolumeLimiter {
    #[must_use]
    pub fn new(
        buy_limits: Option<Vec<VolumeLimit>>,
        sell_limits: Option<Vec<VolumeLimit>>,
        provider: Option<Arc<dyn VolumeLimitProvider>>,
    ) -> Self {
        Self {
            buy_limits,
            sell_limits,
            provider,
        }
    }

    #[must_use]
    pub fn buy_limits(&self) -> Option<&[VolumeLimit]> {
        self.buy_limits.as_deref()
    }

    #[must_use]
    pub fn sell_limits(&self) -> Option<&[VolumeLimit]> {
        self.sell_limits.as_deref()
    }

    /// Clip `order.deal_amount` against every configured directional capacity.
    ///
    /// The `Option` mirrors Python's return contract: `Some(original_amount)` means no
    /// directional limits were configured; `None` means the configured limits were applied.
    /// Provider access and cumulative validation complete before the order is mutated.
    ///
    /// # Errors
    ///
    /// Returns typed configuration, provider, missing-value, or cumulative-state failures.
    ///
    /// # Panics
    ///
    /// Does not panic: the first-value assertion follows the explicit non-empty check and every
    /// configured limit contributes exactly one collected entry or returns an error first.
    pub fn clip_amount_by_volume(
        &self,
        order: &mut Order,
        dealt_order_amount: &HashMap<String, f64>,
    ) -> Result<Option<f64>, ExchangeVolumeError> {
        let limits = match order.direction() {
            OrderDir::Buy => &self.buy_limits,
            OrderDir::Sell => &self.sell_limits,
        };
        let Some(limits) = limits else {
            return Ok(Some(order.deal_amount()));
        };
        if limits.is_empty() {
            return Err(ExchangeVolumeError::EmptyLimits {
                direction: order.direction(),
            });
        }
        let provider = self
            .provider
            .as_ref()
            .ok_or(ExchangeVolumeError::MissingProvider)?;
        let range = TimeRange {
            start: order.start_time(),
            end: order.end_time(),
        };
        let mut values = Vec::with_capacity(limits.len());
        for limit in limits {
            let value = provider.volume_limit(order.stock_id(), range, limit)?;
            match limit.kind() {
                VolumeLimitKind::Current => values.push((limit, value)),
                VolumeLimitKind::Cumulative => {
                    let value = value.ok_or_else(|| ExchangeVolumeError::MissingLimitValue {
                        field: limit.field().to_owned(),
                    })?;
                    let dealt = dealt_order_amount.get(order.stock_id()).ok_or_else(|| {
                        ExchangeVolumeError::MissingDealtAmount {
                            stock: order.stock_id().to_owned(),
                        }
                    })?;
                    values.push((limit, Some(value - dealt)));
                }
            }
        }
        let mut values = values.into_iter().map(|(limit, value)| {
            value.ok_or_else(|| ExchangeVolumeError::MissingLimitValue {
                field: limit.field().to_owned(),
            })
        });
        let mut volume_min = values
            .next()
            .expect("a non-empty limit list has a first value")?;
        for value in values {
            let value = value?;
            if value < volume_min {
                volume_min = value;
            }
        }

        let original = order.deal_amount();
        let clipped_to_limit = if original < volume_min {
            original
        } else {
            volume_min
        };
        let clipped = if 0.0 > clipped_to_limit {
            0.0
        } else {
            clipped_to_limit
        };
        order.set_deal_amount(clipped);
        if volume_min < original {
            tracing::debug!(
                stock = order.stock_id(),
                ?range,
                original,
                clipped,
                "order clipped due to volume limitation"
            );
        }
        Ok(None)
    }
}
