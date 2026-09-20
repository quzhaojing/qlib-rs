use super::*;
use path::PathQueryError;
use std::ffi::OsString;

struct FailedCwd(PathQueryError);
impl RealPathOperations for FailedCwd {
    fn current_directory(&self) -> Result<PathBuf, PathQueryError> {
        Err(self.0.clone())
    }
    fn case_key(&self, _: &Path) -> Result<OsString, PathQueryError> {
        panic!("cwd failure stops before mapping")
    }
    fn final_path(&self, _: &Path) -> Result<PathBuf, PathQueryError> {
        panic!("cwd failure stops before queries")
    }
    fn non_strict(&self, _: &Path) -> Result<PathBuf, PathQueryError> {
        panic!("cwd failure stops before fallback")
    }
}

#[test]
fn native_categories_and_home_failures_survive_the_configuration_boundary() {
    // Exercise successful native resolution in the same unit boundary as the
    // injected failures, retaining the device-path construction regression.
    assert_eq!(
        resolve_with(Path::new("nul"), &NativeRealPathOperations)
            .unwrap()
            .as_os_str(),
        std::ffi::OsStr::new(r"\\.\NUL")
    );
    assert_eq!(
        expand_with(Path::new("~"), &WindowsHomeEnvironment::default()),
        Err(PathInitializationError::Operation(
            "Could not determine home directory.".into()
        ))
    );
    for error in [
        PathQueryError::EmbeddedNul,
        PathQueryError::NotSymbolicLink,
        PathQueryError::Allocation,
        PathQueryError::InputTooLong,
        PathQueryError::Windows {
            operation: "GetCurrentDirectoryW",
            code: 5,
        },
    ] {
        let failure =
            resolve_with(Path::new("C:/absolute"), &FailedCwd(error.clone())).unwrap_err();
        assert_eq!(failure.to_string(), error.to_string());
        assert_eq!(failure, PathInitializationError::NativePath(error));
    }
}
