//! Live provider/override paths without freezing mutable configuration at construction.

use std::{
    ffi::OsString,
    path::PathBuf,
    sync::{Arc, RwLock},
};

use indexmap::IndexMap;

use crate::{CalendarLoadError, DataPathManager};

/// Path policy for both persistent storage and fresh local-provider requests.
/// Each query is independent; do not cache roots behind this boundary.
pub trait CalendarPathProvider: Send + Sync {
    /// # Errors
    /// Preserves configuration acquisition failures before discovery.
    fn provider_keys(&self) -> Result<Vec<String>, CalendarLoadError>;

    /// # Errors
    /// Preserves configuration acquisition and data-root lookup failures.
    fn data_uri(&self, frequency: &str, system: &str) -> Result<PathBuf, CalendarLoadError>;
}

impl CalendarPathProvider for DataPathManager {
    fn provider_keys(&self) -> Result<Vec<String>, CalendarLoadError> {
        Ok(self.provider_uri.keys().cloned().collect())
    }

    fn data_uri(&self, frequency: &str, system: &str) -> Result<PathBuf, CalendarLoadError> {
        self.get_data_uri(Some(frequency), system)
            .map_err(|error| CalendarLoadError::Other(error.to_string()))
    }
}

/// Shared already-normalized provider and mount maps. An instance override
/// replaces only providers; mount paths remain live in the global configuration.
/// No callback is invoked while a read lock is held. Poisoning is an explicit
/// native configuration failure, never silently recovered or cached.
pub struct LiveCalendarPaths {
    pub configuration: Arc<RwLock<DataPathManager>>,
    pub provider_override: Option<Arc<RwLock<IndexMap<String, OsString>>>>,
}

fn poisoned() -> CalendarLoadError {
    CalendarLoadError::Other("calendar path configuration lock poisoned".into())
}

impl CalendarPathProvider for LiveCalendarPaths {
    fn provider_keys(&self) -> Result<Vec<String>, CalendarLoadError> {
        match &self.provider_override {
            Some(providers) => Ok(providers
                .read()
                .map_err(|_| poisoned())?
                .keys()
                .cloned()
                .collect()),
            None => self
                .configuration
                .read()
                .map_err(|_| poisoned())?
                .provider_keys(),
        }
    }

    fn data_uri(&self, frequency: &str, system: &str) -> Result<PathBuf, CalendarLoadError> {
        let configuration = self.configuration.read().map_err(|_| poisoned())?;
        match &self.provider_override {
            Some(providers) => {
                let providers = providers.read().map_err(|_| poisoned())?;
                configuration
                    .get_data_uri_with_provider(&providers, Some(frequency), system)
                    .map_err(|error| CalendarLoadError::Other(error.to_string()))
            }
            None => configuration.data_uri(frequency, system),
        }
    }
}
