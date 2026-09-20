//! Stateful execution windows from `TradeCalendarManager`, over a replaceable calendar source.

use std::sync::Arc;

use chrono::{NaiveDateTime, TimeDelta};
use thiserror::Error;

use crate::{EpsilonError, epsilon_change_backward};

/// Calendar access and cursor failures. No failed operation silently resets the cursor.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ExecutionCalendarError {
    #[error("execution calendar provider error: {0}")]
    Provider(String),
    #[error("The calendar is finished, please reset it if you want to call it!")]
    Finished,
    #[error("calendar index out of range: {0}")]
    Index(i128),
    #[error("calendar index arithmetic exceeds the native integer range")]
    IndexOverflow,
    #[error("data calendar lookup requires a start timestamp")]
    MissingStart,
    #[error("This type of input {0} is not supported")]
    InvalidRangeType(String),
    #[error("next calendar day is outside the timestamp range")]
    DayOverflow,
    #[error(transparent)]
    Epsilon(#[from] EpsilonError),
}

/// The upstream `Cal` boundary. Implementations preserve sorted calendar order and
/// return original absolute indices, including reversed windows. Loading and locating
/// are separate calls because reset exposes the loaded array even when locating fails.
pub trait ExecutionCalendarProvider: Send + Sync {
    /// # Errors
    /// Returns data loading failures.
    fn calendar(
        &self,
        frequency: &str,
        future: bool,
    ) -> Result<Arc<[NaiveDateTime]>, ExecutionCalendarError>;

    /// # Errors
    /// Returns timestamp lookup failures; absent bounds retain the provider's semantics.
    fn locate_index(
        &self,
        start: Option<NaiveDateTime>,
        end: Option<NaiveDateTime>,
        frequency: &str,
        future: bool,
    ) -> Result<(i64, i64), ExecutionCalendarError>;
}

/// Live exchange-frequency lookup used by data-calendar range queries.
pub trait ExecutionCalendarContext: Send + Sync {
    /// # Errors
    /// Returns missing infrastructure or exchange access failures.
    fn data_frequency(&self) -> Result<String, ExecutionCalendarError>;
}

/// Typed timestamp boundary for Qlib's mutable trading-window manager.
/// Parsing strings/time zones belongs to the caller. Calendar loading and indexing remain
/// provider operations, so this does not replace the data provider or its cache.
pub struct ExecutionCalendar {
    provider: Arc<dyn ExecutionCalendarProvider>,
    frequency: String,
    start: Option<NaiveDateTime>,
    end: Option<NaiveDateTime>,
    calendar: Arc<[NaiveDateTime]>,
    start_index: i64,
    end_index: i64,
    trade_len: i64,
    trade_step: i64,
}

impl ExecutionCalendar {
    /// # Errors
    /// Returns the same errors as [`Self::reset`].
    pub fn new(
        provider: Arc<dyn ExecutionCalendarProvider>,
        frequency: String,
        start: Option<NaiveDateTime>,
        end: Option<NaiveDateTime>,
    ) -> Result<Self, ExecutionCalendarError> {
        let mut calendar = Self {
            provider,
            frequency: String::new(),
            start: None,
            end: None,
            calendar: Arc::from([]),
            start_index: 0,
            end_index: 0,
            trade_len: 0,
            trade_step: 0,
        };
        calendar.reset(frequency, start, end)?;
        Ok(calendar)
    }

    /// Reset in source mutation order: metadata, loaded array, indices, length, step.
    /// # Errors
    /// Returns provider errors or unrepresentable index arithmetic, retaining reached writes.
    pub fn reset(
        &mut self,
        frequency: String,
        start: Option<NaiveDateTime>,
        end: Option<NaiveDateTime>,
    ) -> Result<(), ExecutionCalendarError> {
        self.frequency = frequency;
        self.start = start;
        self.end = end;
        self.calendar = self.provider.calendar(&self.frequency, true)?;
        let (start_index, end_index) =
            self.provider
                .locate_index(start, end, &self.frequency, true)?;
        self.start_index = start_index;
        self.end_index = end_index;
        self.trade_len = native_index(i128::from(end_index) - i128::from(start_index) + 1)?;
        self.trade_step = 0;
        Ok(())
    }

    #[must_use]
    pub fn frequency(&self) -> &str {
        &self.frequency
    }

    #[must_use]
    pub const fn all_time(&self) -> (Option<NaiveDateTime>, Option<NaiveDateTime>) {
        (self.start, self.end)
    }

    #[must_use]
    pub const fn indices(&self) -> (i64, i64) {
        (self.start_index, self.end_index)
    }

    #[must_use]
    pub const fn trade_len(&self) -> i64 {
        self.trade_len
    }

    #[must_use]
    pub const fn trade_step(&self) -> i64 {
        self.trade_step
    }

    #[must_use]
    pub const fn finished(&self) -> bool {
        self.trade_step >= self.trade_len
    }

    /// # Errors
    /// Returns [`ExecutionCalendarError::Finished`] after exhaustion without advancing.
    pub fn step(&mut self) -> Result<(), ExecutionCalendarError> {
        if self.finished() {
            return Err(ExecutionCalendarError::Finished);
        }
        self.trade_step += 1;
        Ok(())
    }

    /// Return array[index] and array[index + 1] minus exactly one second. Each array
    /// access independently follows Python's negative-index convention; do not clamp.
    /// # Errors
    /// Returns missing endpoint or timestamp-shift failures.
    pub fn step_time(
        &self,
        step: Option<i64>,
        shift: i64,
    ) -> Result<(NaiveDateTime, NaiveDateTime), ExecutionCalendarError> {
        let index = i128::from(self.start_index) + i128::from(step.unwrap_or(self.trade_step))
            - i128::from(shift);
        let start = self.at(index)?;
        let end = epsilon_change_backward(self.at(index + 1)?)?;
        Ok((start, end))
    }

    fn at(&self, index: i128) -> Result<NaiveDateTime, ExecutionCalendarError> {
        let normalized = if index < 0 {
            self.calendar.len() as i128 + index
        } else {
            index
        };
        usize::try_from(normalized)
            .ok()
            .and_then(|index| self.calendar.get(index))
            .copied()
            .ok_or(ExecutionCalendarError::Index(index))
    }

    /// Resolve closed times using `NumPy`'s right-insertion index minus one, then source clipping.
    /// Reversed or empty windows retain their negative upper bound instead of becoming empty.
    /// # Errors
    /// Returns unrepresentable native index arithmetic.
    pub fn range_indices(
        &self,
        start: NaiveDateTime,
        end: NaiveDateTime,
    ) -> Result<(i64, i64), ExecutionCalendarError> {
        let upper = self
            .trade_len
            .checked_sub(1)
            .ok_or(ExecutionCalendarError::IndexOverflow)?;
        let clip = |time| {
            let index = self.calendar.partition_point(|value| *value <= time) as i128
                - 1
                - i128::from(self.start_index);
            // The subtraction cannot fall below i64::MIN. Positive overflow is
            // necessarily above the native upper bound and is clipped to it.
            i64::try_from(index).unwrap_or(i64::MAX).max(0).min(upper)
        };
        Ok((clip(start), clip(end)))
    }

    /// Resolve full/step data indices relative to midnight of the requested start date.
    /// The context is queried on each call, before range-type dispatch, as in upstream.
    /// # Errors
    /// Returns missing start, infrastructure, provider, type or endpoint failures.
    pub fn data_range(
        &self,
        range_type: &str,
        context: &dyn ExecutionCalendarContext,
    ) -> Result<(i64, i64), ExecutionCalendarError> {
        let day_start = self
            .start
            .ok_or(ExecutionCalendarError::MissingStart)?
            .date()
            .and_time(chrono::NaiveTime::MIN);
        let day_end = epsilon_change_backward(
            day_start
                .checked_add_signed(TimeDelta::days(1))
                .ok_or(ExecutionCalendarError::DayOverflow)?,
        )?;
        let frequency = context.data_frequency()?;
        let (day_start_index, _) =
            self.provider
                .locate_index(Some(day_start), Some(day_end), &frequency, false)?;
        let (start, end) = match range_type {
            "full" => (self.start, self.end),
            "step" => {
                let (start, end) = self.step_time(None, 0)?;
                (Some(start), Some(end))
            }
            _ => {
                return Err(ExecutionCalendarError::InvalidRangeType(
                    range_type.to_owned(),
                ));
            }
        };
        let (start, end) = self.provider.locate_index(start, end, &frequency, false)?;
        Ok((
            native_index(i128::from(start) - i128::from(day_start_index))?,
            native_index(i128::from(end) - i128::from(day_start_index))?,
        ))
    }
}

fn native_index(index: i128) -> Result<i64, ExecutionCalendarError> {
    i64::try_from(index).map_err(|_| ExecutionCalendarError::IndexOverflow)
}
