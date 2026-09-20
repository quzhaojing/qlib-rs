//! Raw calendar text acquisition, before frequency selection, caching and resampling.

use std::path::Path;

use crate::{CalendarLoadError, CalendarTextDecoder};

/// Read the raw records of `FileCalendarStorage._read_calendar`.
///
/// A missing file is created empty; missing parent directories are not created.
/// The caller supplies the text encoding policy, with no BOM detection. This
/// direct read deliberately does not perform the storage `.data` existence gate
/// or cache lookup. Both belong before this operation in the storage layer.
///
/// # Errors
/// Filesystem failures have Other classification; decoder failures retain their
/// classification (including Value for malformed text). No partial rows escape.
pub fn read_calendar_file(
    path: &Path,
    decoder: &dyn CalendarTextDecoder,
) -> Result<Vec<String>, CalendarLoadError> {
    if !path.exists() {
        std::fs::File::create(path).map_err(|error| file_error(&error))?;
    }
    let bytes = std::fs::read(path).map_err(|error| file_error(&error))?;
    let text = decoder.decode_text(&bytes)?;
    Ok(calendar_text_records(&text))
}

/// Universal CR/LF/CRLF record boundaries and Python `str.strip` semantics.
/// Other Unicode line separators remain inside records, just as in text files.
#[must_use]
pub fn calendar_text_records(text: &str) -> Vec<String> {
    text.split(['\r', '\n'])
        .map(|line| line.trim_matches(python_whitespace))
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
        .collect()
}

fn python_whitespace(character: char) -> bool {
    character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
}

fn file_error(error: &std::io::Error) -> CalendarLoadError {
    CalendarLoadError::Other(error.to_string())
}
