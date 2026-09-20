//! Source path-joining policy; Unicode mappings are supplied by ICU4X.
use crate::split_root;
use icu_casemap::CaseMapper;
use std::{
    ffi::{OsStr, OsString},
    os::windows::ffi::{OsStrExt, OsStringExt},
    path::{Path, PathBuf},
};

/// Join path spellings with Windows Python drive/root rules. Rooted paths replace
/// the tail; different drives replace the accumulated path. Equal drives are
/// compared using Unicode-16 full lowercase, not native Windows case mapping.
/// Dots, slash spelling, trailing separators and non-Unicode units are retained.
#[must_use]
pub fn join(base: &Path, paths: &[&Path]) -> PathBuf {
    let (mut drive, mut root, mut tail) = split_root(base);
    for path in paths {
        let (next_drive, next_root, next_tail) = split_root(path);
        if !next_root.is_empty() {
            if !next_drive.is_empty() || drive.is_empty() {
                drive = next_drive;
            }
            root = next_root;
            tail = next_tail;
            continue;
        }
        if !next_drive.is_empty() && next_drive != drive {
            if lowercase(&next_drive) != lowercase(&drive) {
                drive = next_drive;
                root = next_root;
                tail = next_tail;
                continue;
            }
            drive = next_drive;
        }
        if !tail.is_empty() && !matches!(tail.as_encoded_bytes().last(), Some(b'/' | b'\\')) {
            tail.push("\\");
        }
        tail.push(next_tail);
    }
    if !tail.is_empty()
        && root.is_empty()
        && !drive.is_empty()
        && !matches!(drive.as_encoded_bytes().last(), Some(b':' | b'/' | b'\\'))
    {
        drive.push("\\");
    }
    drive.push(root);
    drive.push(tail);
    PathBuf::from(drive)
}

// Unpaired surrogates are uncased/non-case-ignorable boundaries in Python, not
// replacement characters. Map valid spans with ICU and copy boundary units intact.
fn lowercase(input: &OsStr) -> OsString {
    let mapper = CaseMapper::new();
    let locale = icu_locale_core::LanguageIdentifier::UNKNOWN;
    let mut output = Vec::new();
    let mut span = String::new();
    for value in char::decode_utf16(input.encode_wide()) {
        match value {
            Ok(character) => span.push(character),
            Err(error) => {
                output.extend(mapper.lowercase_to_string(&span, &locale).encode_utf16());
                span.clear();
                output.push(error.unpaired_surrogate());
            }
        }
    }
    output.extend(mapper.lowercase_to_string(&span, &locale).encode_utf16());
    OsString::from_wide(&output)
}

#[cfg(test)]
#[path = "../tests/support/join_lowercase.rs"]
mod tests;
