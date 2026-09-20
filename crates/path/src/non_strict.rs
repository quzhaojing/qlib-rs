//! Preserve the unresolved suffix while resolving the longest available prefix.
use crate::{
    NativeLinkOperations, PathQueryError, final_path, find_name, join, read_link_deep, split,
};
use std::path::{Path, PathBuf};

/// Replaceable query boundary for non-strict Windows resolution.
pub trait FinalPathOperations {
    /// Query the final native path, retaining its verbatim prefix.
    /// # Errors
    /// Preserve native error codes and invalid-input/allocation errors.
    fn final_path(&self, path: &Path) -> Result<PathBuf, PathQueryError>;
    /// Follow the source link-chain policy, including cycle detection.
    /// # Errors
    /// Preserve native and non-native failure categories.
    fn read_link_deep(&self, path: &Path) -> Result<PathBuf, PathQueryError>;
    /// Find the real final-component name without requiring metadata access.
    /// # Errors
    /// Preserve native search errors and invalid inputs.
    fn find_name(&self, path: &Path) -> Result<PathBuf, PathQueryError>;
}

/// Production query composition; performs no filesystem writes.
#[derive(Debug, Default, Clone, Copy)]
pub struct NativeFinalPathOperations;
impl FinalPathOperations for NativeFinalPathOperations {
    fn final_path(&self, path: &Path) -> Result<PathBuf, PathQueryError> {
        final_path(path)
    }
    fn read_link_deep(&self, path: &Path) -> Result<PathBuf, PathQueryError> {
        read_link_deep(path, &NativeLinkOperations)
    }
    fn find_name(&self, path: &Path) -> Result<PathBuf, PathQueryError> {
        find_name(path).map(PathBuf::from)
    }
}

/// Windows Python `_getfinalpathname_nonstrict` with default `OSError` handling.
/// Does not normalize/absolutize input or remove the verbatim prefix: those are
/// the outer `realpath` contract. Missing suffixes retain their source spelling.
/// # Errors
/// Unexpected native final-path errors propagate. Link/search native failures
/// permit further traversal; non-native errors always propagate.
pub fn final_path_non_strict(
    path: &Path,
    operations: &dyn FinalPathOperations,
) -> Result<PathBuf, PathQueryError> {
    let mut current = path.to_path_buf();
    let mut tail = PathBuf::new();
    while !current.as_os_str().is_empty() {
        let code = match operations.final_path(&current) {
            Ok(value) => return Ok(append_tail(value, &tail)),
            Err(PathQueryError::Windows {
                code:
                    code @ (1 | 2 | 3 | 5 | 21 | 32 | 50 | 53 | 65 | 67 | 87 | 123 | 161 | 1005 | 1920
                    | 1921),
                ..
            }) => code,
            Err(error) => return Err(error),
        };
        match operations.read_link_deep(&current) {
            Ok(value) => {
                // Path equality erases spelling distinctions significant to Python.
                if value.as_os_str() != current.as_os_str() {
                    return Ok(append_tail(value, &tail));
                }
            }
            Err(PathQueryError::Windows { .. }) => {}
            Err(error) => return Err(error),
        }
        let (parent, mut name) = split(&current);
        if matches!(code, 1 | 5 | 32 | 50 | 87 | 1920 | 1921) {
            match operations.find_name(&current) {
                Ok(value) => name = value.into_os_string(),
                Err(PathQueryError::Windows { .. }) => {}
                Err(error) => return Err(error),
            }
        }
        current = parent;
        if !current.as_os_str().is_empty() && name.as_os_str().is_empty() {
            // Source concatenates here, not path-joins (including drive roots).
            let mut value = current.into_os_string();
            value.push(tail.as_os_str());
            return Ok(value.into());
        }
        tail = append_tail(name.into(), &tail);
    }
    Ok(tail)
}

fn append_tail(path: PathBuf, tail: &Path) -> PathBuf {
    if tail.as_os_str().is_empty() {
        path
    } else {
        join(&path, &[tail])
    }
}
