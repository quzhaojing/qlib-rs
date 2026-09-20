//! Complete source symbolic-link classification over native query stages.
use crate::{
    DirectoryAttributes, LinkOperations, PathQueryError, lstat_attributes, normalize_case,
    read_link, symbolic_link_by_name, symbolic_link_by_open,
};
use std::{
    ffi::OsString,
    path::{Path, PathBuf},
};
use windows_sys::Win32::{
    Storage::FileSystem::FILE_ATTRIBUTE_REPARSE_POINT,
    System::SystemServices::IO_REPARSE_TAG_SYMLINK,
};

struct Queries {
    by_name: fn(&Path) -> Result<bool, PathQueryError>,
    by_open: fn(&Path) -> Result<bool, PathQueryError>,
    lstat: fn(&Path) -> Result<DirectoryAttributes, PathQueryError>,
}
const NATIVE: Queries = Queries {
    by_name: symbolic_link_by_name,
    by_open: symbolic_link_by_open,
    lstat: lstat_attributes,
};

/// Windows Python `islink` predicate. Does not classify junctions as symlinks.
/// Query/invalid-input failures return false with the source-specific retry policy.
#[must_use]
pub fn is_symbolic_link(path: &Path) -> bool {
    classify(path, &NATIVE)
}

fn classify(path: &Path, queries: &Queries) -> bool {
    match (queries.by_name)(path) {
        Ok(value) => return value,
        Err(PathQueryError::Windows {
            code: 2 | 3 | 21 | 53 | 67 | 123 | 161 | 206,
            ..
        }) => return false,
        Err(PathQueryError::Windows { .. }) => {}
        Err(_) => return false,
    }
    match (queries.by_open)(path) {
        Ok(value) => value,
        Err(PathQueryError::Windows {
            operation: "CreateFileW",
            code: 5 | 32 | 87 | 1920,
        }) => match (queries.lstat)(path) {
            Ok(info) => {
                info.attributes & FILE_ATTRIBUTE_REPARSE_POINT != 0
                    && info.reparse_tag == IO_REPARSE_TAG_SYMLINK
            }
            Err(_) => false,
        },
        Err(_) => false,
    }
}

/// Native implementation for complete deep-link traversal, with actual source
/// classification instead of caller-supplied fixture metadata.
#[derive(Debug, Default, Clone, Copy)]
pub struct NativeLinkOperations;
impl LinkOperations for NativeLinkOperations {
    fn case_key(&self, path: &Path) -> Result<OsString, PathQueryError> {
        normalize_case(path)
    }
    fn read_link(&self, path: &Path) -> Result<PathBuf, PathQueryError> {
        read_link(path)
    }
    fn is_symbolic_link(&self, path: &Path) -> bool {
        is_symbolic_link(path)
    }
}

#[cfg(test)]
#[path = "../tests/support/classification.rs"]
mod tests;
