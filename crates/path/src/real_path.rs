//! Outer Windows realpath policy, retaining default non-strict behavior.
use crate::{
    NativeFinalPathOperations, PathQueryError, current_directory, final_path,
    final_path_non_strict, is_absolute, join, normalize, normalize_case,
};
use std::{
    ffi::OsString,
    os::windows::ffi::{OsStrExt, OsStringExt},
    path::{Path, PathBuf},
};

/// Replaceable operations for the outer resolver; callbacks run in source order.
pub trait RealPathOperations {
    /// Read current directory even when the input is absolute or names NUL.
    /// # Errors
    /// Preserve current-directory acquisition failures.
    fn current_directory(&self) -> Result<PathBuf, PathQueryError>;
    /// Native invariant lowercase and separator normalization.
    /// # Errors
    /// Preserve mapping/allocation errors.
    fn case_key(&self, path: &Path) -> Result<OsString, PathQueryError>;
    /// Query the existing final path with native verbatim prefix.
    /// # Errors
    /// Preserve native and invalid-input failures.
    fn final_path(&self, path: &Path) -> Result<PathBuf, PathQueryError>;
    /// Run the complete non-strict inner fallback.
    /// # Errors
    /// Preserve errors not ignored by that policy.
    fn non_strict(&self, path: &Path) -> Result<PathBuf, PathQueryError>;
}

/// Real native Windows operations, without a Python runtime dependency.
#[derive(Debug, Default, Clone, Copy)]
pub struct NativeRealPathOperations;
impl RealPathOperations for NativeRealPathOperations {
    fn current_directory(&self) -> Result<PathBuf, PathQueryError> {
        current_directory()
    }
    fn case_key(&self, path: &Path) -> Result<OsString, PathQueryError> {
        normalize_case(path)
    }
    fn final_path(&self, path: &Path) -> Result<PathBuf, PathQueryError> {
        final_path(path)
    }
    fn non_strict(&self, path: &Path) -> Result<PathBuf, PathQueryError> {
        final_path_non_strict(path, &NativeFinalPathOperations)
    }
}

/// Resolve a Windows path using Python's default (strict=False) realpath policy.
/// Input/output are native path strings, not pathlib object construction.
/// # Errors
/// Propagates current-directory/mapping errors and failures not ignored by the
/// source fallback. Missing paths and embedded NUL retain non-strict semantics.
pub fn real_path(path: &Path) -> Result<PathBuf, PathQueryError> {
    real_path_with(path, &NativeRealPathOperations)
}

/// Resolve with replaceable query operations, preserving query order.
/// # Errors
/// Same policy as [`real_path`], including prefix verification failures.
pub fn real_path_with(
    path: &Path,
    operations: &dyn RealPathOperations,
) -> Result<PathBuf, PathQueryError> {
    let mut path = normalize(path);
    let cwd = operations.current_directory()?;
    if operations.case_key(&path)? == "nul" {
        return Ok(PathBuf::from(r"\\.\NUL"));
    }
    let had_prefix = path.as_os_str().as_encoded_bytes().starts_with(br"\\?\");
    if !had_prefix && !is_absolute(&path) {
        path = join(&cwd, &[&path]);
    }
    let initial_error = match operations.final_path(&path) {
        Ok(value) => {
            path = value;
            Some(0)
        }
        Err(PathQueryError::EmbeddedNul | PathQueryError::NotSymbolicLink) => {
            path = normalize(&path);
            None
        }
        Err(PathQueryError::Windows { code, .. }) => {
            path = operations.non_strict(&path)?;
            Some(code)
        }
        Err(error) => return Err(error),
    };
    let units: Vec<_> = path.as_os_str().encode_wide().collect();
    if !had_prefix && units.starts_with(&[92, 92, 63, 92]) {
        let candidate = if units.starts_with(&[92, 92, 63, 92, 85, 78, 67, 92]) {
            let mut value = OsString::from(r"\\");
            value.push(OsString::from_wide(&units[8..]));
            PathBuf::from(value)
        } else {
            PathBuf::from(OsString::from_wide(&units[4..]))
        };
        match operations.final_path(&candidate) {
            Ok(value) => {
                if value.as_os_str() == path.as_os_str() {
                    path = candidate;
                }
            }
            Err(PathQueryError::EmbeddedNul | PathQueryError::NotSymbolicLink) => {}
            Err(PathQueryError::Windows { code, .. }) => {
                if Some(code) == initial_error {
                    path = candidate;
                }
            }
            Err(error) => return Err(error),
        }
    }
    Ok(path)
}
