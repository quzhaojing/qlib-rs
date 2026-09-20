//! Captured region and live minute-shift policy for configured calendar storage.

use std::sync::{Arc, RwLock};

use chrono::NaiveDateTime;
use num_bigint::BigInt;

use crate::{
    CalendarLoadError, Frequency, FrequencyUnit, Region, align_sampled_minute,
    calendar_resample::{sort_unique, validate_minute_frequencies},
    resample_calendar,
};

/// Replaceable resampling policy. Static region callers retain their explicit shift.
pub trait CalendarResampling: Send + Sync {
    /// Resample decoded timestamps, preserving configuration and numerical failures.
    ///
    /// # Errors
    /// Returns configuration lookup, invalid frequency/region or alignment failures.
    fn resample(
        &self,
        values: &[NaiveDateTime],
        source: &Frequency,
        requested: &Frequency,
        static_shift: &BigInt,
    ) -> Result<Vec<NaiveDateTime>, CalendarLoadError>;
}

impl CalendarResampling for Region {
    fn resample(
        &self,
        values: &[NaiveDateTime],
        source: &Frequency,
        requested: &Frequency,
        static_shift: &BigInt,
    ) -> Result<Vec<NaiveDateTime>, CalendarLoadError> {
        resample_calendar(values, source, requested, *self, static_shift)
            .map_err(|error| CalendarLoadError::Value(error.to_string()))
    }
}

/// Live global lookups; implementations retain missing keys as errors rather than defaults.
/// No backend lock is held while these callbacks run.
pub trait CalendarRuntimeConfiguration: Send + Sync {
    /// Read the raw region. A configured null is distinct from a missing key.
    ///
    /// # Errors
    /// Returns a missing or inaccessible configuration entry.
    fn region(&self) -> Result<Option<String>, CalendarLoadError>;

    /// Read the minute shift at each timestamp, not once for a whole resampling call.
    ///
    /// # Errors
    /// Returns a missing, invalid or inaccessible shift entry.
    fn minute_shift(&self) -> Result<BigInt, CalendarLoadError>;
}

/// Native shared values. Outer `None` is a missing key; the region may itself
/// be null. No implicit global defaults are installed.
pub struct CalendarRuntimeValues {
    pub region: Option<Option<String>>,
    pub minute_shift: Option<BigInt>,
}

impl CalendarRuntimeConfiguration for RwLock<CalendarRuntimeValues> {
    fn region(&self) -> Result<Option<String>, CalendarLoadError> {
        self.read()
            .map_err(|_| runtime_poisoned())?
            .region
            .clone()
            .ok_or_else(|| CalendarLoadError::Other("missing configuration key: region".into()))
    }

    fn minute_shift(&self) -> Result<BigInt, CalendarLoadError> {
        self.read()
            .map_err(|_| runtime_poisoned())?
            .minute_shift
            .clone()
            .ok_or_else(|| {
                CalendarLoadError::Other("missing configuration key: min_data_shift".into())
            })
    }
}

fn runtime_poisoned() -> CalendarLoadError {
    CalendarLoadError::Other("calendar runtime configuration lock poisoned".into())
}

/// Region is captured without validation during construction. A captured null
/// triggers a global region lookup at each resampling entry, including empty or
/// non-minute calendars. Minute shifts remain live per input timestamp.
pub struct LiveCalendarResampling {
    pub captured_region: Option<String>,
    pub configuration: Arc<dyn CalendarRuntimeConfiguration>,
}

impl CalendarResampling for LiveCalendarResampling {
    fn resample(
        &self,
        values: &[NaiveDateTime],
        source: &Frequency,
        requested: &Frequency,
        _static_shift: &BigInt,
    ) -> Result<Vec<NaiveDateTime>, CalendarLoadError> {
        let region = match &self.captured_region {
            Some(region) => Some(region.clone()),
            None => self.configuration.region()?,
        };
        if values.is_empty() || requested.unit != FrequencyUnit::Minute {
            // The period algorithm does not use region or shift. In particular,
            // an unknown region must not prevent day/week/month resampling.
            return Region::Cn.resample(values, source, requested, &BigInt::from(0));
        }
        validate_minute_frequencies(source, requested)
            .map_err(|error| CalendarLoadError::Value(error.to_string()))?;
        let sample_minutes = BigInt::from(requested.count.clone());
        let mut result = Vec::with_capacity(values.len());
        for &value in values {
            // Python evaluates C.min_data_shift before get_min_cal validates region.
            let shift = self.configuration.minute_shift()?;
            let parsed = match region.as_deref() {
                Some("cn") => Region::Cn,
                Some("us") => Region::Us,
                Some("tw") => Region::Tw,
                other => {
                    return Err(CalendarLoadError::Value(format!(
                        "{} is not supported",
                        other.unwrap_or("None")
                    )));
                }
            };
            result.push(
                align_sampled_minute(value, &sample_minutes, &shift, parsed)
                    .map_err(|error| CalendarLoadError::Value(error.to_string()))?,
            );
        }
        sort_unique(&mut result);
        Ok(result)
    }
}
