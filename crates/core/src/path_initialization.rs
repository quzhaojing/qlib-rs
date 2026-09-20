//! Ordered provider/mount normalization from `QlibConfig.resolve_path`.

use std::{
    ffi::OsString,
    path::{Path, PathBuf},
};

use indexmap::IndexMap;
use thiserror::Error;

use crate::{DEFAULT_DATA_FREQUENCY, ProviderUriKind, provider_uri_kind};

/// Text and already constructed path values are intentionally distinct: source
/// URI classification expands/resolves path objects, but not text objects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigPathValue {
    Text(OsString),
    Path(PathBuf),
    Null,
    /// A rejected dynamic value, identified by its type at the input boundary.
    Unsupported(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathSetting {
    Scalar(ConfigPathValue),
    Mapping(IndexMap<String, ConfigPathValue>),
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PathInitializationError {
    #[error("provider_uri cannot be None")]
    NoneProvider,
    #[error("provider_uri does not support {0}")]
    UnsupportedProvider(String),
    #[error("provider URI entry has no expanduser method: {0}")]
    InvalidProviderEntry(String),
    #[error("mount path is not path-like: {0}")]
    InvalidMountEntry(String),
    #[error("mount_path is missing freq: {0:?}")]
    MissingMount(Vec<String>),
    #[error("{0}")]
    Operation(String),
    /// Retain native failure category/code instead of flattening it into text.
    #[cfg(windows)]
    #[error(transparent)]
    NativePath(path::PathQueryError),
}

/// Replaceable platform path operations. Implementations must preserve source
/// non-strict filesystem resolution; lexical absolute paths are not a substitute.
/// Callbacks run synchronously in source order and must not mutate this configuration.
pub trait ConfigPathOperations {
    /// Construct/normalize a native path and expand its home prefix, without resolving.
    ///
    /// # Errors
    /// Returns home lookup or platform path failures unchanged to the caller.
    fn expand_home(&self, path: &Path) -> Result<PathBuf, PathInitializationError>;

    /// Resolve symlinks and missing suffixes using the source's non-strict semantics.
    ///
    /// # Errors
    /// Returns filesystem/platform failures not ignored by that policy.
    fn resolve(&self, path: &Path) -> Result<PathBuf, PathInitializationError>;
}

/// Owned settings retain partial mapping updates on failure. Scalar settings
/// remain scalar until both provider and mount normalization finish successfully.
/// This models values/state, not arbitrary Python dictionary aliases/subclasses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathConfiguration {
    pub provider_uri: PathSetting,
    pub mount_path: PathSetting,
}

impl PathConfiguration {
    /// Normalize provider roots before validating/expanding mounts. Missing mount
    /// keys are checked before any mount expansion; extra mount keys are untouched.
    /// Provider entries are processed in insertion order, then mounts in provider order.
    ///
    /// # Errors
    /// Preserves earlier mapping mutations and returns the first failure in source order.
    pub fn resolve_paths(
        &mut self,
        operations: &dyn ConfigPathOperations,
    ) -> Result<(), PathInitializationError> {
        let mut temporary_provider;
        let provider = match &mut self.provider_uri {
            PathSetting::Mapping(mapping) => mapping,
            PathSetting::Scalar(ConfigPathValue::Null) => {
                return Err(PathInitializationError::NoneProvider);
            }
            PathSetting::Scalar(ConfigPathValue::Unsupported(kind)) => {
                return Err(PathInitializationError::UnsupportedProvider(kind.clone()));
            }
            PathSetting::Scalar(value) => {
                temporary_provider =
                    IndexMap::from([(DEFAULT_DATA_FREQUENCY.into(), value.clone())]);
                &mut temporary_provider
            }
        };
        normalize_provider_map(provider, operations)?;
        let mut temporary_mount;
        let mounts = match &mut self.mount_path {
            PathSetting::Mapping(mapping) => mapping,
            PathSetting::Scalar(value) => {
                temporary_mount = provider
                    .keys()
                    .map(|key| (key.clone(), value.clone()))
                    .collect();
                &mut temporary_mount
            }
        };
        let missing: Vec<_> = provider
            .keys()
            .filter(|key| !mounts.contains_key(*key))
            .cloned()
            .collect();
        if !missing.is_empty() {
            return Err(PathInitializationError::MissingMount(missing));
        }
        for key in provider.keys() {
            let value = &mut mounts[key];
            let path = match value {
                ConfigPathValue::Null => continue,
                ConfigPathValue::Unsupported(kind) => {
                    return Err(PathInitializationError::InvalidMountEntry(kind.clone()));
                }
                ConfigPathValue::Text(text) => Path::new(text),
                ConfigPathValue::Path(path) => path.as_path(),
            };
            *value = ConfigPathValue::Text(operations.expand_home(path)?.into_os_string());
        }
        let provider = std::mem::take(provider);
        let mounts = std::mem::take(mounts);
        self.provider_uri = PathSetting::Mapping(provider);
        self.mount_path = PathSetting::Mapping(mounts);
        Ok(())
    }
}

/// Normalize an existing provider map in-place, retaining earlier entries on error.
///
/// # Errors
/// Returns invalid entry types or platform operation failures without rolling back.
pub fn normalize_provider_map(
    values: &mut IndexMap<String, ConfigPathValue>,
    operations: &dyn ConfigPathOperations,
) -> Result<(), PathInitializationError> {
    for value in values.values_mut() {
        let is_path_object = matches!(value, ConfigPathValue::Path(_));
        let path = match value {
            ConfigPathValue::Text(text) => Path::new(text),
            ConfigPathValue::Path(path) => path.as_path(),
            ConfigPathValue::Null => {
                return Err(PathInitializationError::InvalidProviderEntry(
                    "NoneType".into(),
                ));
            }
            ConfigPathValue::Unsupported(kind) => {
                return Err(PathInitializationError::InvalidProviderEntry(kind.clone()));
            }
        };
        let kind = if is_path_object {
            let expanded = operations.expand_home(path)?;
            let resolved = operations.resolve(&expanded)?;
            provider_uri_kind(&resolved.to_string_lossy())
        } else {
            provider_uri_kind(&path.to_string_lossy())
        };
        if kind == ProviderUriKind::Local {
            let expanded = operations.expand_home(path)?;
            *value = ConfigPathValue::Text(operations.resolve(&expanded)?.into_os_string());
        }
    }
    Ok(())
}
