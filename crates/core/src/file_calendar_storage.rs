//! Persistent configured calendar requests and their source frequency caches.

use std::path::PathBuf;

use crate::{
    CalendarFileValues, CalendarLoadError, CalendarPathProvider, CalendarResampling,
    CalendarTextArray, CalendarValue, CalendarWriteMode, DataPathManager, FileCalendarBackend,
    Frequency, Region,
    file_calendar_backend::{parse_frequency, select_frequency},
    read_calendar_file, write_calendar_file,
};

/// One persistent `FileCalendarStorage` instance, unlike the fresh requests made
/// by `FileCalendarBackend`. The path policy can own configured roots or access
/// live shared maps. Root changes do not invalidate the two lazy frequency caches.
/// Failed discovery/selection is never cached, but successful discovery survives
/// a later selection failure. Paths themselves are recomputed on every call.
/// This constructor does not initialize global Qlib configuration or overrides.
pub struct FileCalendarStorage<P = DataPathManager, R = Region> {
    pub backend: FileCalendarBackend<P, R>,
    pub frequency: String,
    pub future: bool,
    supported: Option<Vec<Frequency>>,
    selected: Option<Frequency>,
}

impl<P: CalendarPathProvider, R: CalendarResampling> FileCalendarStorage<P, R> {
    #[must_use]
    pub fn new(backend: FileCalendarBackend<P, R>, frequency: String, future: bool) -> Self {
        Self {
            backend,
            frequency,
            future,
            supported: None,
            selected: None,
        }
    }

    /// Discover once, publishing the complete frequency list only on success.
    ///
    /// # Errors
    /// Preserves root/discovery/frequency errors; a subsequent call may retry.
    pub fn supported_frequencies(&mut self) -> Result<&[Frequency], CalendarLoadError> {
        match self.supported {
            Some(ref frequencies) => Ok(frequencies),
            None => Ok(self.supported.insert(self.backend.supported_frequencies()?)),
        }
    }

    /// Select once independently of the discovery cache. Later request changes
    /// affect resampling, but never reselect the source file automatically.
    ///
    /// # Errors
    /// Returns parsing/discovery/unavailable-source failures without caching them.
    pub fn file_frequency(&mut self) -> Result<Frequency, CalendarLoadError> {
        if let Some(selected) = &self.selected {
            return Ok(selected.clone());
        }
        // Source parses the request before it attempts frequency discovery.
        let frequency = self.frequency.clone();
        let requested = parse_frequency(&frequency)?;
        let supported = self.supported_frequencies()?;
        let selected = select_frequency(&requested, &frequency, supported)?;
        self.selected = Some(selected.clone());
        Ok(selected)
    }

    /// Resolve the current configured root and future flag, retaining file frequency.
    ///
    /// # Errors
    /// Preserves selection and root lookup errors.
    pub fn uri(&mut self) -> Result<PathBuf, CalendarLoadError> {
        let selected = self.file_frequency()?;
        self.backend.path_for_frequency(&selected, self.future)
    }

    /// Check existence and load/cache raw lines before parsing the current request
    /// for resampling. An invalid changed frequency can therefore populate cache.
    ///
    /// # Errors
    /// Returns file, decoder, cache, frequency or resampling failures.
    pub fn data(&mut self) -> Result<Vec<CalendarValue>, CalendarLoadError> {
        let selected = self.file_frequency()?;
        let path = self.backend.path_for_frequency(&selected, self.future)?;
        let raw = self.backend.raw_at(&path)?;
        let requested = parse_frequency(&self.frequency)?;
        self.backend.values_from_raw(&raw, &selected, &requested)
    }

    /// Read directly without the public data existence gate or shared cache.
    /// A missing file is created empty; its parent directory is not created.
    ///
    /// # Errors
    /// Returns selection/root, file or decoder failures.
    pub fn read_calendar(&mut self) -> Result<Vec<String>, CalendarLoadError> {
        let path = self.uri()?;
        read_calendar_file(&path, self.backend.text_decoder.as_ref())
    }

    /// Write values after resolving the current file, without cache invalidation.
    ///
    /// # Errors
    /// Preserves selection/root, open, encoding and flush failures.
    pub fn write_calendar(
        &mut self,
        values: &dyn CalendarFileValues,
        mode: CalendarWriteMode,
    ) -> Result<(), CalendarLoadError> {
        write_calendar_file(&self.uri()?, values, mode)
    }

    /// Append to the selected file, retaining any shared cached data.
    ///
    /// # Errors
    /// Returns the same failures as `write_calendar`.
    pub fn extend(&mut self, values: &dyn CalendarFileValues) -> Result<(), CalendarLoadError> {
        self.write_calendar(values, CalendarWriteMode::Append)
    }

    /// Truncate the selected file, retaining any shared cached data.
    ///
    /// # Errors
    /// Returns the same failures as `write_calendar`.
    pub fn clear(&mut self) -> Result<(), CalendarLoadError> {
        self.write_calendar(
            &CalendarTextArray {
                shape: vec![0],
                values: vec![],
            },
            CalendarWriteMode::Overwrite,
        )
    }

    /// Length comes from data (including cache and resampling), not a fresh raw read.
    ///
    /// # Errors
    /// Preserves data acquisition failures.
    pub fn len(&mut self) -> Result<usize, CalendarLoadError> {
        Ok(self.data()?.len())
    }

    /// Whether the data view is empty, subject to the same cache policy as `len`.
    ///
    /// # Errors
    /// Returns data acquisition failures.
    pub fn is_empty(&mut self) -> Result<bool, CalendarLoadError> {
        self.len().map(|length| length == 0)
    }
}
