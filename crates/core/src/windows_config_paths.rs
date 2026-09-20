//! Production Windows path operations for ordered configuration initialization.
use crate::{
    path_initialization::{ConfigPathOperations, PathConfiguration, PathInitializationError},
    windows_home::{WindowsHomeEnvironment, lexical_path},
};
use path::{NativeRealPathOperations, RealPathOperations, real_path_with};
use std::path::{Path, PathBuf};

/// Uses the current process home environment and native non-strict resolution.
/// Does not mutate the environment, cwd, filesystem, or unrelated configuration.
#[derive(Debug, Default, Clone, Copy)]
pub struct WindowsConfigPathOperations;

impl ConfigPathOperations for WindowsConfigPathOperations {
    fn expand_home(&self, path: &Path) -> Result<PathBuf, PathInitializationError> {
        expand_with(path, &WindowsHomeEnvironment::capture())
    }
    fn resolve(&self, path: &Path) -> Result<PathBuf, PathInitializationError> {
        resolve_with(path, &NativeRealPathOperations)
    }
}

impl PathConfiguration {
    /// Normalize provider and mount settings using real Windows path operations.
    /// Keeps the existing ordered/partial-update contract and NFS distinctions.
    /// # Errors
    /// Preserves validation/home/native failures without rolling back earlier map entries.
    pub fn resolve_paths_native(&mut self) -> Result<(), PathInitializationError> {
        self.resolve_paths(&WindowsConfigPathOperations)
    }
}

fn expand_with(
    path: &Path,
    environment: &WindowsHomeEnvironment,
) -> Result<PathBuf, PathInitializationError> {
    environment
        .expand(path)
        .map_err(|error| PathInitializationError::Operation(error.to_string()))
}

fn resolve_with(
    path: &Path,
    operations: &dyn RealPathOperations,
) -> Result<PathBuf, PathInitializationError> {
    // Path.resolve constructs a Path before realpath, then constructs another
    // from the result. These lexical boundaries do not themselves collapse '..'.
    let input = lexical_path(path);
    let resolved =
        real_path_with(&input, operations).map_err(PathInitializationError::NativePath)?;
    Ok(lexical_path(&resolved))
}

#[cfg(test)]
#[path = "../tests/support/windows_config_paths.rs"]
mod tests;
