//! Shared live execution calendar for nested, atomic, simulator and SAOE collaborators.

use std::sync::{Arc, Mutex, MutexGuard};

use chrono::NaiveDateTime;

use crate::{
    ExecutionCalendar, ExecutionCalendarContext, ExecutionCalendarError, ExecutorLifecycleCalendar,
    ExecutorLifecycleCalendarError, NestedCalendar, NestedCalendarError, ResettableNestedCalendar,
    SaoeCalendar, SaoeDecisionCalendar, SaoePluginError, SimulatorCalendar, SimulatorCalendarError,
    TradeCalendarRange, TradeCalendarRangeError,
};

/// Clones share the original cursor, including failed reset mutations. Provider/context
/// callbacks must not reenter this calendar while its mutex is held. Poisoning is an error,
/// never implicit recovery of potentially partially mutated state.
#[derive(Clone)]
pub struct SharedExecutionCalendar {
    calendar: Arc<Mutex<ExecutionCalendar>>,
    context: Arc<dyn ExecutionCalendarContext>,
}

impl SharedExecutionCalendar {
    #[must_use]
    pub fn new(
        calendar: Arc<Mutex<ExecutionCalendar>>,
        context: Arc<dyn ExecutionCalendarContext>,
    ) -> Self {
        Self { calendar, context }
    }

    fn lock(&self) -> Result<MutexGuard<'_, ExecutionCalendar>, ExecutionCalendarError> {
        self.calendar.lock().map_err(|_| {
            ExecutionCalendarError::Provider("execution calendar lock poisoned".to_owned())
        })
    }
}

impl From<ExecutionCalendarError> for NestedCalendarError {
    fn from(error: ExecutionCalendarError) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}

impl From<ExecutionCalendarError> for TradeCalendarRangeError {
    fn from(error: ExecutionCalendarError) -> Self {
        Self::Provider {
            message: error.to_string(),
        }
    }
}

impl TradeCalendarRange for SharedExecutionCalendar {
    fn start_time(&self) -> Result<NaiveDateTime, TradeCalendarRangeError> {
        self.lock()?
            .all_time()
            .0
            .ok_or(ExecutionCalendarError::MissingStart)
            .map_err(Into::into)
    }

    fn get_range_idx(
        &self,
        start_time: NaiveDateTime,
        end_time: NaiveDateTime,
    ) -> Result<(i64, i64), TradeCalendarRangeError> {
        self.lock()?
            .range_indices(start_time, end_time)
            .map_err(Into::into)
    }
}

impl NestedCalendar for SharedExecutionCalendar {
    fn finished(&self) -> Result<bool, NestedCalendarError> {
        Ok(self.lock().map_err(NestedCalendarError::from)?.finished())
    }
    fn trade_len(&self) -> Result<i64, NestedCalendarError> {
        Ok(self.lock().map_err(NestedCalendarError::from)?.trade_len())
    }
    fn trade_step(&self) -> Result<i64, NestedCalendarError> {
        Ok(self.lock().map_err(NestedCalendarError::from)?.trade_step())
    }
    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), NestedCalendarError> {
        self.lock()
            .map_err(NestedCalendarError::from)?
            .step_time(None, 0)
            .map_err(NestedCalendarError::from)
    }
    fn step(&self) -> Result<(), NestedCalendarError> {
        self.lock()
            .map_err(NestedCalendarError::from)?
            .step()
            .map_err(NestedCalendarError::from)
    }
}

impl ResettableNestedCalendar for SharedExecutionCalendar {
    fn reset_window(
        &self,
        start_time: NaiveDateTime,
        end_time: NaiveDateTime,
    ) -> Result<(), NestedCalendarError> {
        let mut calendar = self.lock().map_err(NestedCalendarError::from)?;
        let frequency = calendar.frequency().to_owned();
        calendar
            .reset(frequency, Some(start_time), Some(end_time))
            .map_err(NestedCalendarError::from)
    }
}

impl ExecutorLifecycleCalendar for SharedExecutionCalendar {
    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), ExecutorLifecycleCalendarError> {
        NestedCalendar::step_time(self).map_err(|error| ExecutorLifecycleCalendarError {
            message: error.message,
        })
    }
    fn step(&self) -> Result<(), ExecutorLifecycleCalendarError> {
        NestedCalendar::step(self).map_err(|error| ExecutorLifecycleCalendarError {
            message: error.message,
        })
    }
}

impl SimulatorCalendar for SharedExecutionCalendar {
    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), SimulatorCalendarError> {
        NestedCalendar::step_time(self).map_err(|error| SimulatorCalendarError {
            message: error.message,
        })
    }
}

impl SaoeCalendar for SharedExecutionCalendar {
    fn available_step_range(&self) -> Result<(i64, i64), SaoePluginError> {
        self.lock()
            .map_err(SaoePluginError::from)?
            .data_range("step", &*self.context)
            .map_err(SaoePluginError::from)
    }
    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), SaoePluginError> {
        self.lock()
            .map_err(SaoePluginError::from)?
            .step_time(None, 0)
            .map_err(SaoePluginError::from)
    }
}

impl SaoeDecisionCalendar for SharedExecutionCalendar {
    fn frequency(&self) -> Result<String, SaoePluginError> {
        Ok(self
            .lock()
            .map_err(SaoePluginError::from)?
            .frequency()
            .to_owned())
    }
}

impl From<ExecutionCalendarError> for SaoePluginError {
    fn from(error: ExecutionCalendarError) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}
