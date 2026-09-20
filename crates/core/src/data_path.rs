//! Data-root lookup from `QlibConfig.DataPathManager.get_data_uri`.
//!
//! Maps are already configured inputs. Home expansion, filesystem resolution and
//! `format_provider_uri` mutation are separate initialization steps, not performed here.

use std::{
    ffi::{OsStr, OsString},
    path::{Path, PathBuf},
    sync::LazyLock,
};

use indexmap::IndexMap;
use regex::Regex;
use thiserror::Error;

pub const DEFAULT_DATA_FREQUENCY: &str = "__DEFAULT_FREQ";

static WINDOWS_URI: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[a-zA-Z]:.*").expect("static drive-prefix regex is valid"));
static NFS_URI: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[^/]+:.+").expect("static source NFS regex is valid"));

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderUriKind {
    Local,
    Nfs,
}

/// Classify a string with Qlib's regexes, not general URL parsing. In particular,
/// an HTTP-looking string can be classified as NFS. No filesystem access occurs.
#[must_use]
pub fn provider_uri_kind(uri: &str) -> ProviderUriKind {
    if NFS_URI.is_match(uri) && !WINDOWS_URI.is_match(uri) {
        ProviderUriKind::Nfs
    } else {
        ProviderUriKind::Local
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DataPathError {
    #[error("missing provider URI key: {0}")]
    MissingProvider(String),
    #[error("missing mount path key: {0}")]
    MissingMount(String),
    #[error("mount path must be text, not None: {0}")]
    NullMount(String),
}

/// Typed configurable maps. Frequency keys remain exact strings, including aliases.
/// Values may be updated between calls; no root or platform result is memoized.
pub struct DataPathManager {
    pub provider_uri: IndexMap<String, OsString>,
    pub mount_path: IndexMap<String, Option<OsString>>,
}

impl DataPathManager {
    #[must_use]
    pub fn new(
        provider_uri: IndexMap<String, String>,
        mount_path: IndexMap<String, Option<String>>,
    ) -> Self {
        Self::from_native(
            provider_uri
                .into_iter()
                .map(|(key, value)| (key, value.into()))
                .collect(),
            mount_path
                .into_iter()
                .map(|(key, value)| (key, value.map(Into::into)))
                .collect(),
        )
    }

    /// Build from already configured native paths without a Unicode conversion.
    /// Home expansion and resolution remain explicit initialization operations.
    #[must_use]
    pub fn from_native(
        provider_uri: IndexMap<String, OsString>,
        mount_path: IndexMap<String, Option<OsString>>,
    ) -> Self {
        Self {
            provider_uri,
            mount_path,
        }
    }

    /// Resolve a data root using the caller's current `platform.system()` value.
    /// The source's substring test deliberately includes `Darwin`, not just Windows.
    /// Mounting, file creation, existence checks and canonicalization do not occur.
    ///
    /// # Errors
    /// Reports missing provider/mount keys or a null mount in the non-Windows branch.
    pub fn get_data_uri(
        &self,
        frequency: Option<&str>,
        system: &str,
    ) -> Result<PathBuf, DataPathError> {
        self.get_data_uri_with_provider(&self.provider_uri, frequency, system)
    }

    pub(crate) fn get_data_uri_with_provider(
        &self,
        provider_uri: &IndexMap<String, OsString>,
        frequency: Option<&str>,
        system: &str,
    ) -> Result<PathBuf, DataPathError> {
        let key = frequency
            .filter(|key| provider_uri.contains_key(*key))
            .unwrap_or(DEFAULT_DATA_FREQUENCY);
        let uri = provider_uri
            .get(key)
            .ok_or_else(|| DataPathError::MissingProvider(key.to_owned()))?;
        // Only classification uses replacement text: both source regexes depend
        // on ASCII drive letters, colon, slash and newline. Unpaired native units
        // and U+FFFD are equivalent for these predicates. Never use this text as a path.
        self.get_data_uri_for_value(key, uri, provider_uri_kind(&uri.to_string_lossy()), system)
    }

    pub(crate) fn get_data_uri_for_value(
        &self,
        key: &str,
        uri: &OsStr,
        kind: ProviderUriKind,
        system: &str,
    ) -> Result<PathBuf, DataPathError> {
        if kind == ProviderUriKind::Local {
            return Ok(native_path(uri));
        }
        let mount = self
            .mount_path
            .get(key)
            .ok_or_else(|| DataPathError::MissingMount(key.to_owned()))?;
        let path = if system.to_lowercase().contains("win") {
            let mut mount = mount.as_deref().unwrap_or(OsStr::new("None")).to_owned();
            if !mount.as_encoded_bytes().contains(&b':') {
                mount.push(":\\");
            }
            mount
        } else {
            mount
                .clone()
                .ok_or_else(|| DataPathError::NullMount(key.to_owned()))?
        };
        Ok(native_path(&path))
    }
}

// Pathlib removes redundant separators and dot components, but not parent components.
// The filesystem's native path syntax is used, independently of the supplied system label.
#[cfg(windows)]
fn native_path(text: &OsStr) -> PathBuf {
    crate::windows_home::lexical_path(Path::new(text))
}

#[cfg(not(windows))]
fn native_path(text: &OsStr) -> PathBuf {
    let path: PathBuf = Path::new(text)
        .components()
        .filter(|component| !matches!(component, std::path::Component::CurDir))
        .collect();
    if path.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        path
    }
}
