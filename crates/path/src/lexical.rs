//! Windows Python lexical normalization, before filesystem resolution.

use std::{
    ffi::OsString,
    os::windows::ffi::{OsStrExt, OsStringExt},
    path::{Path, PathBuf},
};

const SEP: u16 = b'\\' as u16;
const DOT: u16 = b'.' as u16;

/// Split Windows path spelling into drive, root separator and remaining tail.
/// Retains original slash spelling and malformed device/UNC prefixes, unlike Rust
/// component iteration. The three strings concatenate to the original input.
#[must_use]
pub fn split_root(path: &Path) -> (OsString, OsString, OsString) {
    let units: Vec<_> = path.as_os_str().encode_wide().collect();
    let separators: Vec<_> = units
        .iter()
        .map(|unit| if *unit == 47 { SEP } else { *unit })
        .collect();
    let (end, rooted) = prefix(&separators);
    let drive_end = end - usize::from(rooted);
    (
        OsString::from_wide(&units[..drive_end]),
        OsString::from_wide(&units[drive_end..end]),
        OsString::from_wide(&units[end..]),
    )
}

/// Split a path into its directory spelling and final name, as `ntpath.split`.
/// Trailing separators produce an empty name; the directory loses excess trailing
/// separators but retains its root. Dot components are not normalized.
#[must_use]
pub fn split(path: &Path) -> (PathBuf, OsString) {
    let (mut drive, root, tail) = split_root(path);
    let units: Vec<_> = tail.encode_wide().collect();
    let boundary = units
        .iter()
        .rposition(|unit| matches!(*unit, 47 | 92))
        .map_or(0, |index| index + 1);
    let mut end = boundary;
    while end > 0 && matches!(units[end - 1], 47 | 92) {
        end -= 1;
    }
    drive.push(root);
    drive.push(OsString::from_wide(&units[..end]));
    (
        PathBuf::from(drive),
        OsString::from_wide(&units[boundary..]),
    )
}

/// Python 3.14 Windows absolute-path predicate: a rooted drive or two leading
/// separators. A single leading separator is not absolute. Reads native UTF-16
/// as Unicode characters, preserving unpaired units as non-delimiter characters.
#[must_use]
pub fn is_absolute(path: &Path) -> bool {
    let head: Vec<_> = char::decode_utf16(path.as_os_str().encode_wide())
        .take(3)
        .map(|unit| match unit {
            Ok('/' | '\\') => 92,
            Ok(':') => 58,
            _ => 0,
        })
        .collect();
    matches!(head.as_slice(), [92, 92, ..] | [_, 58, 92])
}

/// Normalize separators and dot components using Windows Python path semantics.
/// No filesystem access, home expansion, case folding or absolute-path conversion
/// occurs. Unpaired UTF-16 units and embedded NULs are preserved as path data.
#[must_use]
pub fn normalize(path: &Path) -> PathBuf {
    let units: Vec<_> = path
        .as_os_str()
        .encode_wide()
        .map(|unit| if unit == u16::from(b'/') { SEP } else { unit })
        .collect();
    let (prefix_end, rooted) = prefix(&units);
    let mut components: Vec<&[u16]> = Vec::new();
    for component in units[prefix_end..].split(|unit| *unit == SEP) {
        match component {
            [] | [DOT] => {}
            [DOT, DOT] => match components.last() {
                Some(previous) if *previous != [DOT, DOT] => {
                    components.pop();
                }
                None if rooted => {}
                _ => components.push(component),
            },
            _ => components.push(component),
        }
    }
    let mut result = units[..prefix_end].to_vec();
    result.extend(components.join(&SEP));
    if result.is_empty() {
        result.push(DOT);
    }
    PathBuf::from(OsString::from_wide(&result))
}

// Source splitroot differs from Rust Prefix parsing for malformed UNC/device
// names and non-letter drive designators. Keep this narrow compatibility policy.
fn prefix(units: &[u16]) -> (usize, bool) {
    if units.starts_with(&[SEP, SEP]) {
        let start = if matches!(
            units,
            [SEP, SEP, 63, SEP, 85 | 117, 78 | 110, 67 | 99, SEP, ..]
        ) {
            8
        } else {
            2
        };
        let mut separators = units[start..]
            .iter()
            .enumerate()
            .filter(|(_, unit)| **unit == SEP);
        match separators.nth(1) {
            Some((index, _)) => (start + index + 1, true),
            None => (units.len(), false),
        }
    } else if units.first() == Some(&SEP) {
        (1, true)
    } else if units.get(1) == Some(&u16::from(b':')) {
        if units.get(2) == Some(&SEP) {
            (3, true)
        } else {
            (2, false)
        }
    } else {
        (0, false)
    }
}
