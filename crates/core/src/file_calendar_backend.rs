//! Configured native file acquisition for the local calendar provider.

use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    sync::Arc,
};

use num_bigint::BigInt;

use crate::{
    CalendarBackendSource, CalendarCache, CalendarLoadError, CalendarPathProvider,
    CalendarResampling, CalendarRows, CalendarTextDecoder, CalendarTimestampDecoder, CalendarValue,
    DEFAULT_DATA_FREQUENCY, DataPathManager, Frequency, Region, read_calendar_file,
};

/// Each `data` call constructs a fresh storage request, like `backend_obj` in the
/// upstream local provider. Frequency discovery is local to that request; raw
/// lines are shared across requests through `cache`. Roots are already configured
/// (home expansion and filesystem resolution belong to configuration setup).
///
/// Encoding and timestamp policies are explicit plugins, not implicit UTF-8 or
/// partial pandas grammar assumptions. The same timestamp policy should be used
/// by the enclosing local loader. Read callbacks must not reenter the cache.
pub struct FileCalendarBackend<P = DataPathManager, R = Region> {
    pub paths: P,
    pub system: String,
    pub region: R,
    pub minute_shift: BigInt,
    pub text_decoder: Arc<dyn CalendarTextDecoder>,
    pub timestamp_decoder: Arc<dyn CalendarTimestampDecoder>,
    pub cache: Arc<CalendarCache>,
    pub enable_read_cache: bool,
}

impl<P: CalendarPathProvider, R: CalendarResampling> FileCalendarBackend<P, R> {
    /// Discover supported frequencies in source order, without sorting or deduplication.
    /// A single default root scans `calendars/*.txt`; otherwise all map keys are used.
    /// Like pathlib glob, inaccessible/nonexistent scan directories produce no entries.
    ///
    /// # Errors
    /// Returns invalid frequency names, unsupported non-Unicode filenames, or root errors.
    pub fn supported_frequencies(&self) -> Result<Vec<Frequency>, CalendarLoadError> {
        discover_frequencies(&self.paths, &self.system)
    }

    fn root(&self, frequency: &str) -> Result<PathBuf, CalendarLoadError> {
        self.paths.data_uri(frequency, &self.system)
    }

    /// Resolve the selected file frequency and filename for one fresh request.
    ///
    /// # Errors
    /// Returns invalid or unavailable frequencies and path configuration failures.
    pub fn resolve_file(
        &self,
        frequency: &str,
        future: bool,
    ) -> Result<(Frequency, PathBuf), CalendarLoadError> {
        self.resolve_request(frequency, future)
            .map(|(_, selected, path)| (selected, path))
    }

    fn resolve_request(
        &self,
        frequency: &str,
        future: bool,
    ) -> Result<(Frequency, Frequency, PathBuf), CalendarLoadError> {
        let requested = parse_frequency(frequency)?;
        let supported = self.supported_frequencies()?;
        let selected = select_frequency(&requested, frequency, &supported)?;
        let path = self.path_for_frequency(&selected, future)?;
        Ok((requested, selected, path))
    }

    pub(crate) fn path_for_frequency(
        &self,
        selected: &Frequency,
        future: bool,
    ) -> Result<PathBuf, CalendarLoadError> {
        let normalized = selected.to_string();
        let suffix = if future { "_future.txt" } else { ".txt" };
        Ok(self
            .root(&normalized)?
            .join("calendars")
            .join(format!("{normalized}{suffix}")))
    }
}

// One discovery implementation serves owned maps and fallible live plugins.
// Keep policy acquisition/error propagation at this shared dynamic boundary,
// rather than specializing the filesystem scan for every provider type.
fn discover_frequencies(
    paths: &dyn CalendarPathProvider,
    system: &str,
) -> Result<Vec<Frequency>, CalendarLoadError> {
    let provider_keys = paths.provider_keys()?;
    let names = if provider_keys.len() == 1 && provider_keys[0] == DEFAULT_DATA_FREQUENCY {
        let directory = paths
            .data_uri(DEFAULT_DATA_FREQUENCY, system)?
            .join("calendars");
        let mut names = Vec::new();
        // Python materializes scandir before yielding any entries.
        if let Ok(entries) =
            std::fs::read_dir(directory).and_then(Iterator::collect::<Result<Vec<_>, _>>)
        {
            for entry in entries {
                let filename = entry.file_name();
                // Replacement text is used only to filter the ASCII suffix.
                // Selected invalid-Unicode names are rejected, never persisted.
                let name = filename.to_string_lossy();
                let matches = name.ends_with(".txt")
                    || (cfg!(windows) && name.to_ascii_lowercase().ends_with(".txt"));
                if matches {
                    let stem = &name[..name.len() - 4];
                    if !stem.ends_with("_future") {
                        if filename.to_str().is_none() {
                            return Err(CalendarLoadError::Value(
                                "invalid calendar frequency filename: non-Unicode text".into(),
                            ));
                        }
                        names.push(stem.to_owned());
                    }
                }
            }
        }
        names
    } else {
        provider_keys
    };
    names.iter().map(|name| parse_frequency(name)).collect()
}

impl<P: CalendarPathProvider, R: CalendarResampling> CalendarBackendSource
    for FileCalendarBackend<P, R>
{
    fn data(&self, frequency: &str, future: bool) -> Result<CalendarRows, CalendarLoadError> {
        let (requested, selected, path) = self.resolve_request(frequency, future)?;
        let raw = self.raw_at(&path)?;
        let values = self.values_from_raw(&raw, &selected, &requested)?;
        Ok(Box::new(values.into_iter().map(Ok)))
    }
}

impl<P: CalendarPathProvider, R: CalendarResampling> FileCalendarBackend<P, R> {
    pub(crate) fn raw_at(&self, path: &Path) -> Result<Arc<[String]>, CalendarLoadError> {
        check_calendar_path(path)?;
        let mut read = || read_calendar_file(path, self.text_decoder.as_ref()).map(Arc::from);
        if self.enable_read_cache {
            // Match Python's raw string key without replacing unpaired native
            // units. Display is diagnostic-only and can merge distinct filenames.
            let mut key = OsString::from("orig_file");
            key.push(path.as_os_str());
            self.cache.get_or_load_raw_native(&key, &mut read)
        } else {
            read()
        }
    }

    pub(crate) fn values_from_raw(
        &self,
        raw: &[String],
        selected: &Frequency,
        requested: &Frequency,
    ) -> Result<Vec<CalendarValue>, CalendarLoadError> {
        let values = if selected == requested {
            raw.iter()
                .cloned()
                .map(CalendarValue::Text)
                .collect::<Vec<_>>()
        } else {
            let timestamps = raw
                .iter()
                .map(|line| {
                    self.timestamp_decoder
                        .decode(CalendarValue::Text(line.clone()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.region
                .resample(&timestamps, selected, requested, &self.minute_shift)?
                .into_iter()
                .map(CalendarValue::Timestamp)
                .collect()
        };
        Ok(values)
    }
}

pub(crate) fn check_calendar_path(path: &Path) -> Result<(), CalendarLoadError> {
    if path.exists() {
        Ok(())
    } else {
        Err(CalendarLoadError::Value(format!(
            "calendar not exists: {}",
            path.display()
        )))
    }
}

pub(crate) fn parse_frequency(text: &str) -> Result<Frequency, CalendarLoadError> {
    text.parse::<Frequency>()
        .map_err(|error| CalendarLoadError::Value(error.to_string()))
}

pub(crate) fn select_frequency(
    requested: &Frequency,
    original: &str,
    supported: &[Frequency],
) -> Result<Frequency, CalendarLoadError> {
    if supported.contains(requested) {
        Ok(requested.clone())
    } else {
        Frequency::nearest_resample_source(requested, supported).ok_or_else(|| {
            CalendarLoadError::Value(format!(
                "can't find a freq from {supported:?} that can resample to {original}!"
            ))
        })
    }
}
