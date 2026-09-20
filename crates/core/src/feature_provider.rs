//! Injectable feature-provider fallback compatible with `get_higher_eq_freq_feature`.

use arrow_array::RecordBatch;
use chrono::NaiveDateTime;
use num_bigint::BigInt;
use thiserror::Error;

use crate::{Frequency, FrequencyError, FrequencyUnit};

/// One provider request using the typed subset required by current Qlib callers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureQuery {
    /// Explicit instrument symbols.
    pub instruments: Vec<String>,
    /// Qlib feature expressions.
    pub fields: Vec<String>,
    /// Inclusive start timestamp.
    pub start_time: Option<NaiveDateTime>,
    /// Inclusive end timestamp.
    pub end_time: Option<NaiveDateTime>,
    /// Requested frequency spelling, retained exactly until a fallback is needed.
    pub frequency: String,
    /// Qlib disk-cache policy (`0` skip, `1` use, `2` replace).
    pub disk_cache: BigInt,
}

impl FeatureQuery {
    /// Construct a query with Qlib's `day`, no-bound, and disk-cache `1` defaults.
    #[must_use]
    pub fn new(instruments: Vec<String>, fields: Vec<String>) -> Self {
        Self {
            instruments,
            fields,
            start_time: None,
            end_time: None,
            frequency: "day".to_owned(),
            disk_cache: BigInt::from(1),
        }
    }

    fn with_frequency(&self, frequency: &str) -> Self {
        let mut query = self.clone();
        frequency.clone_into(&mut query.frequency);
        query
    }
}

/// Provider failures, separated by the two upstream retryable exception kinds.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum FeatureProviderError {
    /// Equivalent to an upstream `ValueError`.
    #[error("provider value error: {message}")]
    Value {
        /// Provider diagnostic.
        message: String,
    },
    /// Equivalent to an upstream `KeyError`.
    #[error("provider key error: {message}")]
    Key {
        /// Provider diagnostic.
        message: String,
    },
    /// Any non-retryable provider failure.
    #[error("provider error: {message}")]
    Other {
        /// Provider diagnostic.
        message: String,
    },
}

impl FeatureProviderError {
    const fn is_frequency_unavailable(&self) -> bool {
        matches!(self, Self::Value { .. } | Self::Key { .. })
    }
}

/// Object-safe boundary implemented by local, remote, or plugin-backed providers.
pub trait FeatureProvider: Send + Sync {
    /// Fetch Arrow batches for one query.
    ///
    /// # Errors
    ///
    /// Returns a classified provider failure. Only [`FeatureProviderError::Value`]
    /// and [`FeatureProviderError::Key`] participate in frequency fallback.
    fn features(&self, query: &FeatureQuery) -> Result<Vec<RecordBatch>, FeatureProviderError>;
}

/// Feature data together with the frequency that actually produced it.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedFeatures {
    /// Arrow batches returned by the provider, unchanged and in provider order.
    pub batches: Vec<RecordBatch>,
    /// Exact requested spelling on direct success, or `day`/`1min` after fallback.
    pub frequency: String,
}

/// Failures produced by feature-frequency resolution.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum FeatureResolutionError {
    /// The requested frequency is parsed only after a retryable initial failure.
    #[error(transparent)]
    Frequency(#[from] FrequencyError),
    /// A provider failure that is not handled by another fallback.
    #[error(transparent)]
    Provider(#[from] FeatureProviderError),
}

/// Fetch features at the requested frequency or the nearest upstream fallback.
///
/// This preserves Qlib's exact call order: direct request first; month/week/day
/// then try `day` followed by `1min`; minute requests retry at `1min`. The
/// frequency string is not parsed when the direct provider request succeeds.
///
/// # Errors
///
/// Returns [`FeatureResolutionError::Frequency`] when a rejected request has an
/// invalid frequency spelling, or [`FeatureResolutionError::Provider`] when the
/// provider emits a non-retryable or final-attempt failure.
pub fn get_higher_eq_frequency_features(
    provider: &dyn FeatureProvider,
    query: &FeatureQuery,
) -> Result<ResolvedFeatures, FeatureResolutionError> {
    match provider.features(query) {
        Ok(batches) => {
            return Ok(ResolvedFeatures {
                batches,
                frequency: query.frequency.clone(),
            });
        }
        Err(error) if error.is_frequency_unavailable() => {}
        Err(error) => return Err(error.into()),
    }

    let frequency: Frequency = query.frequency.parse()?;
    match frequency.unit {
        FrequencyUnit::Minute => fetch_at(provider, query, "1min"),
        FrequencyUnit::Month | FrequencyUnit::Week | FrequencyUnit::Day => {
            let day_query = query.with_frequency("day");
            match provider.features(&day_query) {
                Ok(batches) => Ok(ResolvedFeatures {
                    batches,
                    frequency: "day".to_owned(),
                }),
                Err(error) if error.is_frequency_unavailable() => fetch_at(provider, query, "1min"),
                Err(error) => Err(error.into()),
            }
        }
    }
}

fn fetch_at(
    provider: &dyn FeatureProvider,
    query: &FeatureQuery,
    frequency: &str,
) -> Result<ResolvedFeatures, FeatureResolutionError> {
    let fallback_query = query.with_frequency(frequency);
    let batches = provider.features(&fallback_query)?;
    Ok(ResolvedFeatures {
        batches,
        frequency: frequency.to_owned(),
    })
}
