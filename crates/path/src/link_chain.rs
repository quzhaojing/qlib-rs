//! Source-compatible link traversal policy, independent of query acquisition.
use crate::{PathQueryError, is_absolute, join, normalize, split};
use std::{
    collections::HashSet,
    ffi::OsString,
    path::{Path, PathBuf},
};

/// Query boundary for link-chain resolution. Native case/read queries are already
/// available in this crate. Implementations must classify symbolic links narrowly:
/// Windows junctions are not symbolic links for the relative-target rule.
pub trait LinkOperations {
    /// Compute Windows invariant lowercase with separator normalization.
    /// # Errors
    /// Native mapping/length/allocation failures propagate outside the read fallback.
    fn case_key(&self, path: &Path) -> Result<OsString, PathQueryError>;
    /// Read exactly one substitute target without following the chain.
    /// # Errors
    /// Preserve native error codes and non-symbolic-link/NUL categories.
    fn read_link(&self, path: &Path) -> Result<PathBuf, PathQueryError>;
    /// Test the original path for the symbolic-link tag. Acquisition failures and
    /// invalid inputs return false, matching Python's predicate behavior.
    fn is_symbolic_link(&self, path: &Path) -> bool;
}

/// Follow links as Windows Python `_readlink_deep`, stopping at case-key cycles,
/// allowed native read errors or relative targets of non-symbolic reparse points.
/// Returns the last path spelling, not necessarily an existing/canonical path.
///
/// # Errors
/// Propagates case-key failures, allocation/length errors and non-ignored native
/// read errors. No arbitrary depth limit or filesystem-mutation side effects.
pub fn read_link_deep(
    path: &Path,
    operations: &dyn LinkOperations,
) -> Result<PathBuf, PathQueryError> {
    let mut current = path.to_path_buf();
    let mut seen = HashSet::new();
    // Keep source query ordering: test a key, then compute the inserted key again.
    // OsString equality is intentional: Path equality discards spelling details.
    while !seen.contains(&operations.case_key(&current)?) {
        seen.insert(operations.case_key(&current)?);
        let target = match operations.read_link(&current) {
            Ok(target) => target,
            Err(
                PathQueryError::EmbeddedNul
                | PathQueryError::NotSymbolicLink
                | PathQueryError::Windows {
                    code: 1 | 2 | 3 | 5 | 21 | 32 | 50 | 67 | 87 | 4390 | 4392 | 4393,
                    ..
                },
            ) => break,
            Err(error) => return Err(error),
        };
        if is_absolute(&target) {
            current = target;
        } else if operations.is_symbolic_link(&current) {
            current = normalize(&join(&split(&current).0, &[&target]));
        } else {
            break;
        }
    }
    Ok(current)
}
