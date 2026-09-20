//! Lossless Windows storage-frequency names, distinct from supported frequencies.

use std::{
    collections::BTreeMap,
    ffi::{OsStr, OsString},
    io,
    os::windows::ffi::{OsStrExt, OsStringExt},
};

use crate::{CalendarLoadError, FileCalendarStorage};

fn frequency_name(name: &OsStr) -> Option<(Vec<u32>, OsString)> {
    // ASCII suffix comparison does not decode or replace the native name.
    let bytes = name.as_encoded_bytes();
    if bytes.len() < 4 || !bytes[bytes.len() - 4..].eq_ignore_ascii_case(b".txt") {
        return None;
    }
    let units: Vec<_> = name.encode_wide().collect();
    let suffix_start = units.len() - 4;
    // Python 3.14 keeps the entire name when removing the suffix would leave
    // only dots (including the empty stem). Rust Path::file_stem differs here.
    let stem = if units[..suffix_start]
        .iter()
        .any(|&unit| unit != u16::from(b'.'))
    {
        &units[..suffix_start]
    } else {
        &units
    };
    let stem: Vec<_> = stem
        .iter()
        .copied()
        .take_while(|&unit| unit != u16::from(b'_'))
        .collect();
    // Python compares Unicode code points, not UTF-16 code units. Decode pairs,
    // but retain unpaired surrogates as their original Python code points.
    let ordering = char::decode_utf16(stem.iter().copied())
        .map(|point| point.map_or_else(|error| u32::from(error.unpaired_surrogate()), u32::from))
        .collect();
    Some((ordering, OsString::from_wide(&stem)))
}

fn collect_frequencies(
    entries: io::Result<impl Iterator<Item = io::Result<OsString>>>,
) -> Vec<OsString> {
    // pathlib materializes scandir before matching. A late enumeration error
    // discards earlier names too; directories are not filtered out.
    let Ok(names) = entries.and_then(Iterator::collect::<Result<Vec<_>, _>>) else {
        return Vec::new();
    };
    names
        .iter()
        .filter_map(|name| frequency_name(name))
        .collect::<BTreeMap<_, _>>()
        .into_values()
        .collect()
}

impl<P: crate::CalendarPathProvider, R: crate::CalendarResampling> FileCalendarStorage<P, R> {
    /// Enumerate current Windows storage names like `_get_storage_freq`.
    /// Returns sorted, deduplicated raw names, including unknown frequencies and
    /// the prefix of future files. Results are not cached. Native names retain
    /// lone surrogates and are sorted in Python code-point order.
    ///
    /// # Errors
    /// Preserves URI selection/configuration failures. Directory scan failures
    /// instead produce an empty list, without creating the directory or file.
    pub fn storage_frequencies(&mut self) -> Result<Vec<OsString>, CalendarLoadError> {
        let mut directory = self.uri()?;
        directory.pop();
        Ok(collect_frequencies(std::fs::read_dir(directory).map(
            |entries| entries.map(|entry| entry.map(|entry| entry.file_name())),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::collect_frequencies;
    use std::{ffi::OsString, io};

    #[test]
    fn failed_or_partially_failed_scans_never_return_partial_frequency_names() {
        let opened: io::Result<std::vec::IntoIter<io::Result<OsString>>> =
            Err(io::Error::other("open failed"));
        assert!(collect_frequencies(opened).is_empty());
        let entries = vec![
            Ok(OsString::from("day.txt")),
            Err(io::Error::other("scan failed")),
        ];
        assert!(collect_frequencies(Ok(entries.into_iter())).is_empty());
    }
}
