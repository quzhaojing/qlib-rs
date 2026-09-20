//! Fresh file storage acquisition from an isolated provider template per request.

use std::sync::{Arc, RwLock};

use crate::{
    CalendarBackendSource, CalendarConstructionError, CalendarLoadError, CalendarProviderInput,
    CalendarRows, CalendarStorageFactory,
};

/// Native file-backend selection for `LocalCalendarLoader`. Every acquisition,
/// including its future-to-current retry, copies template values and constructs
/// a new store. Global configuration and the raw cache remain shared.
/// This is the file backend, not an arbitrary Python dynamic class/deepcopy engine.
pub struct CalendarFileSource {
    pub factory: Arc<CalendarStorageFactory>,
    pub provider_template: CalendarProviderInput,
}

impl CalendarFileSource {
    fn copy_provider(&self) -> Result<CalendarProviderInput, CalendarLoadError> {
        match &self.provider_template {
            CalendarProviderInput::Scalar(value) => {
                Ok(CalendarProviderInput::Scalar(value.clone()))
            }
            CalendarProviderInput::Mapping(values) => {
                let copied = values
                    .read()
                    .map_err(|_| {
                        CalendarLoadError::Other("calendar provider template lock poisoned".into())
                    })?
                    .clone();
                Ok(CalendarProviderInput::Mapping(Arc::new(RwLock::new(
                    copied,
                ))))
            }
        }
    }
}

impl CalendarBackendSource for CalendarFileSource {
    fn data(&self, frequency: &str, future: bool) -> Result<CalendarRows, CalendarLoadError> {
        let provider = self.copy_provider()?;
        let mut created = self
            .factory
            .create(frequency.into(), future, provider, ())
            .map_err(|error| match error {
                CalendarConstructionError::Configuration(error) => error,
                CalendarConstructionError::Path(error) => {
                    CalendarLoadError::Other(error.to_string())
                }
            })?;
        let values = created.storage.data()?;
        Ok(Box::new(values.into_iter().map(Ok)))
    }
}
