//! `LocalCalendarProvider.load_calendar` fallback and deferred timestamp conversion.

use std::sync::Arc;

use chrono::{NaiveDate, NaiveDateTime, NaiveTime};
use thiserror::Error;

use crate::{CalendarLoader, ExecutionCalendarError};

/// The exception class, not just its text, determines whether a backend may fall back.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CalendarLoadError {
    #[error("{0}")]
    Value(String),
    #[error("{0}")]
    Other(String),
}

/// Calendar storage returns text lines or already parsed/resampled timestamps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalendarValue {
    Text(String),
    Timestamp(NaiveDateTime),
}

/// Errors during lazy iteration happen outside the backend acquisition fallback scope.
pub type CalendarRows = Box<dyn Iterator<Item = Result<CalendarValue, CalendarLoadError>> + Send>;

pub trait CalendarBackendSource: Send + Sync {
    /// Construct the requested backend and obtain its `data` iterable on every call.
    ///
    /// # Errors
    /// Returns construction/data-access failures; only Value errors can trigger fallback.
    fn data(&self, frequency: &str, future: bool) -> Result<CalendarRows, CalendarLoadError>;
}

pub trait CalendarTimestampDecoder: Send + Sync {
    /// Convert one row after acquiring the backend; preserve its typed failure.
    ///
    /// # Errors
    /// Returns unsupported timestamp, time-zone or decoding failures.
    fn decode(&self, value: CalendarValue) -> Result<NaiveDateTime, CalendarLoadError>;
}

pub trait CalendarWarningSink: Send + Sync {
    /// # Errors
    /// Returns logger failures before fallback proceeds to the next operation.
    fn warning(&self, message: &str) -> Result<(), CalendarLoadError>;
}

/// Default native logging adapter, using the workspace's existing tracing subscriber.
pub struct TracingCalendarWarnings;

impl CalendarWarningSink for TracingCalendarWarnings {
    fn warning(&self, message: &str) -> Result<(), CalendarLoadError> {
        tracing::warn!(target: "data", "{message}");
        Ok(())
    }
}

/// Decoder for canonical timezone-naive calendar files: ISO dates and date-times
/// with `T` or space separators, minute/second precision and fractional seconds.
/// This is explicitly not the full dynamic `pandas.Timestamp` grammar or a time-zone
/// conversion policy. Callers requiring another input contract provide a decoder.
pub struct IsoCalendarTimestampDecoder;

impl CalendarTimestampDecoder for IsoCalendarTimestampDecoder {
    fn decode(&self, value: CalendarValue) -> Result<NaiveDateTime, CalendarLoadError> {
        match value {
            CalendarValue::Timestamp(value) => Ok(value),
            CalendarValue::Text(text) => decode_iso(&text),
        }
    }
}

fn decode_iso(text: &str) -> Result<NaiveDateTime, CalendarLoadError> {
    for format in [
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M",
    ] {
        if let Ok(value) = NaiveDateTime::parse_from_str(text, format) {
            return Ok(value);
        }
    }
    NaiveDate::parse_from_str(text, "%Y-%m-%d")
        .map(|date| date.and_time(NaiveTime::MIN))
        .map_err(|_| {
            CalendarLoadError::Value(format!(
                "invalid timezone-naive ISO calendar timestamp: {text}"
            ))
        })
}

/// Owned local-provider orchestration, reusable with native file or custom storage.
pub struct LocalCalendarLoader {
    backend: Arc<dyn CalendarBackendSource>,
    decoder: Arc<dyn CalendarTimestampDecoder>,
    warnings: Arc<dyn CalendarWarningSink>,
}

impl LocalCalendarLoader {
    #[must_use]
    pub fn new(
        backend: Arc<dyn CalendarBackendSource>,
        decoder: Arc<dyn CalendarTimestampDecoder>,
        warnings: Arc<dyn CalendarWarningSink>,
    ) -> Self {
        Self {
            backend,
            decoder,
            warnings,
        }
    }

    /// Load with the source's ValueError-only, future-to-current retry boundary.
    /// Conversion/iteration happens after that boundary and cannot trigger another read.
    ///
    /// # Errors
    /// Returns the first backend, warning, iterator or decoder failure in source order.
    pub fn load(
        &self,
        frequency: &str,
        future: bool,
    ) -> Result<Arc<[NaiveDateTime]>, CalendarLoadError> {
        let rows = match self.backend.data(frequency, future) {
            Ok(rows) => rows,
            Err(CalendarLoadError::Value(_)) if future => {
                self.warnings.warning(&format!(
                    "load calendar error: freq={frequency}, future=True; return current calendar!"
                ))?;
                self.warnings.warning("You can get future calendar by referring to the following document: https://github.com/microsoft/qlib/blob/main/scripts/data_collector/contrib/README.md")?;
                self.backend.data(frequency, false)?
            }
            Err(error) => return Err(error),
        };
        rows.map(|row| self.decoder.decode(row?))
            .collect::<Result<Vec<_>, _>>()
            .map(Arc::from)
    }
}

impl CalendarLoader for LocalCalendarLoader {
    fn load_calendar(
        &self,
        frequency: &str,
        future: bool,
    ) -> Result<Arc<[NaiveDateTime]>, ExecutionCalendarError> {
        self.load(frequency, future)
            .map_err(|error| ExecutionCalendarError::Provider(error.to_string()))
    }
}
