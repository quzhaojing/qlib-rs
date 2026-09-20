//! Native calendar construction with shared override normalization and region capture.

use std::{
    path::PathBuf,
    sync::{Arc, RwLock},
};

use indexmap::IndexMap;
use thiserror::Error;

use crate::{
    CalendarCache, CalendarLoadError, CalendarPathProvider, CalendarRuntimeConfiguration,
    CalendarTextDecoder, CalendarTimestampDecoder, DEFAULT_DATA_FREQUENCY, DataPathManager,
    FileCalendarBackend, FileCalendarStorage, LiveCalendarResampling,
    path_initialization::{
        ConfigPathOperations, ConfigPathValue, PathInitializationError, normalize_provider_map,
    },
    provider_uri_kind,
};

/// Shared identity is retained across normalization, storage creation and later updates.
pub type CalendarProviderMap = Arc<RwLock<IndexMap<String, ConfigPathValue>>>;

pub enum CalendarProviderInput {
    /// Null selects global providers; text/path values become a default-frequency map.
    Scalar(ConfigPathValue),
    Mapping(CalendarProviderMap),
}

#[derive(Debug, Error)]
pub enum CalendarConstructionError {
    #[error(transparent)]
    Path(#[from] PathInitializationError),
    #[error("{0}")]
    Configuration(CalendarLoadError),
}

/// Override values remain typed and shared, rather than being copied into an
/// independent normalized map. External mutations are observed by later queries.
pub struct ConfiguredCalendarPaths {
    pub global: Arc<RwLock<DataPathManager>>,
    pub provider_override: Option<CalendarProviderMap>,
    pub operations: Arc<dyn ConfigPathOperations + Send + Sync>,
}

fn poisoned() -> CalendarLoadError {
    CalendarLoadError::Other("calendar construction configuration lock poisoned".into())
}

impl CalendarPathProvider for ConfiguredCalendarPaths {
    fn provider_keys(&self) -> Result<Vec<String>, CalendarLoadError> {
        match &self.provider_override {
            Some(mapping) => Ok(mapping
                .read()
                .map_err(|_| poisoned())?
                .keys()
                .cloned()
                .collect()),
            None => self.global.read().map_err(|_| poisoned())?.provider_keys(),
        }
    }

    fn data_uri(&self, frequency: &str, system: &str) -> Result<PathBuf, CalendarLoadError> {
        let Some(mapping) = &self.provider_override else {
            return self
                .global
                .read()
                .map_err(|_| poisoned())?
                .data_uri(frequency, system);
        };
        let (key, value) = {
            let values = mapping.read().map_err(|_| poisoned())?;
            let key = if values.contains_key(frequency) {
                frequency
            } else {
                DEFAULT_DATA_FREQUENCY
            };
            let value = values.get(key).ok_or_else(|| {
                CalendarLoadError::Other(format!("missing provider URI key: {key}"))
            })?;
            (key.to_owned(), value.clone())
        };
        // Release the map lock before any path plugin runs. Resolve only the
        // selected value: unrelated invalid entries do not affect a root lookup.
        let (uri, kind) = match value {
            ConfigPathValue::Text(text) => {
                let kind = provider_uri_kind(&text.to_string_lossy());
                (text, kind)
            }
            ConfigPathValue::Path(path) => {
                let expanded = self
                    .operations
                    .expand_home(&path)
                    .map_err(|error| path_load_error(&error))?;
                let resolved = self
                    .operations
                    .resolve(&expanded)
                    .map_err(|error| path_load_error(&error))?;
                let kind = provider_uri_kind(&resolved.to_string_lossy());
                (path.into_os_string(), kind)
            }
            ConfigPathValue::Null => {
                return Err(path_load_error(
                    &PathInitializationError::InvalidProviderEntry("NoneType".into()),
                ));
            }
            ConfigPathValue::Unsupported(kind) => {
                return Err(path_load_error(
                    &PathInitializationError::InvalidProviderEntry(kind),
                ));
            }
        };
        self.global
            .read()
            .map_err(|_| poisoned())?
            .get_data_uri_for_value(&key, &uri, kind, system)
            .map_err(|error| CalendarLoadError::Other(error.to_string()))
    }
}

fn path_load_error(error: &PathInitializationError) -> CalendarLoadError {
    CalendarLoadError::Other(error.to_string())
}

/// Dependencies shared by constructed stores. Global providers/mounts and runtime
/// lookups stay live; codecs/cache are injected infrastructure, not Python bridges.
pub struct CalendarStorageFactory {
    pub global_paths: Arc<RwLock<DataPathManager>>,
    pub runtime: Arc<dyn CalendarRuntimeConfiguration>,
    pub operations: Arc<dyn ConfigPathOperations + Send + Sync>,
    pub system: String,
    pub text_decoder: Arc<dyn CalendarTextDecoder>,
    pub timestamp_decoder: Arc<dyn CalendarTimestampDecoder>,
    pub cache: Arc<CalendarCache>,
}

/// Opaque constructor kwargs are retained without interpreting or serializing them.
/// The typed Rust boundary returns no usable storage on construction failure.
pub struct ConstructedCalendarStorage<K> {
    pub storage: FileCalendarStorage<ConfiguredCalendarPaths, LiveCalendarResampling>,
    pub kwargs: K,
}

impl CalendarStorageFactory {
    /// Normalize the override in place, then capture the raw region. Frequency
    /// validation is deferred; read caching is enabled regardless of opaque kwargs.
    /// Earlier map mutations remain visible if normalization or region lookup fails.
    /// Path callbacks must not reenter the override map during normalization.
    ///
    /// # Errors
    /// Returns the first normalization, lock or region lookup failure without rollback.
    pub fn create<K>(
        &self,
        frequency: String,
        future: bool,
        provider: CalendarProviderInput,
        kwargs: K,
    ) -> Result<ConstructedCalendarStorage<K>, CalendarConstructionError> {
        Ok(ConstructedCalendarStorage {
            storage: self.create_storage(frequency, future, provider)?,
            kwargs,
        })
    }

    fn create_storage(
        &self,
        frequency: String,
        future: bool,
        provider: CalendarProviderInput,
    ) -> Result<
        FileCalendarStorage<ConfiguredCalendarPaths, LiveCalendarResampling>,
        CalendarConstructionError,
    > {
        let mapping =
            match provider {
                CalendarProviderInput::Scalar(ConfigPathValue::Null) => None,
                CalendarProviderInput::Scalar(ConfigPathValue::Unsupported(kind)) => {
                    return Err(PathInitializationError::UnsupportedProvider(kind).into());
                }
                CalendarProviderInput::Scalar(value) => Some(Arc::new(RwLock::new(
                    IndexMap::from([(DEFAULT_DATA_FREQUENCY.into(), value)]),
                ))),
                CalendarProviderInput::Mapping(mapping) => Some(mapping),
            };
        if let Some(mapping) = &mapping {
            let mut values = mapping
                .write()
                .map_err(|_| CalendarConstructionError::Configuration(poisoned()))?;
            normalize_provider_map(&mut values, self.operations.as_ref())?;
        }
        let region = self
            .runtime
            .region()
            .map_err(CalendarConstructionError::Configuration)?;
        let backend = FileCalendarBackend {
            paths: ConfiguredCalendarPaths {
                global: self.global_paths.clone(),
                provider_override: mapping,
                operations: self.operations.clone(),
            },
            system: self.system.clone(),
            region: LiveCalendarResampling {
                captured_region: region,
                configuration: self.runtime.clone(),
            },
            minute_shift: 0.into(),
            text_decoder: self.text_decoder.clone(),
            timestamp_decoder: self.timestamp_decoder.clone(),
            cache: self.cache.clone(),
            enable_read_cache: true,
        };
        Ok(FileCalendarStorage::new(backend, frequency, future))
    }
}
