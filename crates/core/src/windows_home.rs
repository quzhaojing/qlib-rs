//! Native Windows home expansion used by Python pathlib configuration setup.
//! No filesystem resolution or environment mutation is performed here.

use std::{
    ffi::{OsStr, OsString},
    os::windows::ffi::{OsStrExt, OsStringExt},
    path::{Component, Path, PathBuf},
};

use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("Could not determine home directory.")]
pub struct HomeExpansionError;

/// An explicit environment snapshot. Missing and empty values are distinct.
/// Python's Windows expansion uses these variables, not HOME or the OS profile API.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WindowsHomeEnvironment {
    pub user_profile: Option<OsString>,
    pub home_drive: Option<OsString>,
    pub home_path: Option<OsString>,
    pub user_name: Option<OsString>,
}

impl WindowsHomeEnvironment {
    #[must_use]
    pub fn capture() -> Self {
        Self {
            user_profile: std::env::var_os("USERPROFILE"),
            home_drive: std::env::var_os("HOMEDRIVE"),
            home_path: std::env::var_os("HOMEPATH"),
            user_name: std::env::var_os("USERNAME"),
        }
    }

    /// Expand the first unrooted `~` component, then retain pathlib's lexical
    /// separators/dot normalization without collapsing parents or checking existence.
    /// UTF-16 path data is retained, including unpaired surrogates.
    ///
    /// # Errors
    /// Returns the source `RuntimeError` equivalent when home cannot be determined.
    pub fn expand(&self, input: &Path) -> Result<PathBuf, HomeExpansionError> {
        let path = lexical_path(input);
        let mut parts = path.components();
        let Some(Component::Normal(first)) = parts.next() else {
            return Ok(path);
        };
        if !starts_tilde(first) {
            return Ok(path);
        }
        let token: Vec<_> = first.encode_wide().skip(1).collect();
        let target_user = OsString::from_wide(&token);
        let mut home = if let Some(profile) = &self.user_profile {
            PathBuf::from(profile)
        } else {
            PathBuf::from(self.home_drive.as_deref().unwrap_or_default())
                .join(self.home_path.as_deref().ok_or(HomeExpansionError)?)
        };
        if !target_user.is_empty() && self.user_name.as_deref() != Some(target_user.as_os_str()) {
            let (directory, basename) = split_home(home.as_os_str());
            if self.user_name.as_deref() != Some(basename.as_os_str()) {
                return Err(HomeExpansionError);
            }
            home = PathBuf::from(directory).join(target_user);
        }
        if starts_tilde(home.as_os_str()) {
            return Err(HomeExpansionError);
        }
        home.extend(parts);
        Ok(lexical_path(&home))
    }
}

fn starts_tilde(value: &OsStr) -> bool {
    value.as_encoded_bytes().first() == Some(&b'~')
}

// ntpath.split preserves a trailing separator as an EMPTY basename, unlike
// Path::file_name. Prefix bytes belong to the directory, even without a slash.
fn split_home(home: &OsStr) -> (OsString, OsString) {
    let units: Vec<_> = home.encode_wide().collect();
    let prefix = match Path::new(home).components().next() {
        Some(Component::Prefix(prefix)) => prefix.as_os_str().encode_wide().count(),
        _ => 0,
    };
    let index = units
        .iter()
        .rposition(|unit| matches!(unit, 47 | 92))
        .map_or(prefix, |index| (index + 1).max(prefix));
    (
        OsString::from_wide(&units[..index]),
        OsString::from_wide(&units[index..]),
    )
}

pub(crate) fn lexical_path(path: &Path) -> PathBuf {
    // Pathlib construction differs from Rust component collection for device
    // prefixes (e.g. \\.\NUL must not acquire a trailing separator).
    let units: Vec<_> = path
        .as_os_str()
        .encode_wide()
        .map(|unit| if unit == 47 { 92 } else { unit })
        .collect();
    let spelling = OsString::from_wide(&units);
    let (mut drive, mut root, tail) = path::split_root(Path::new(&spelling));
    let drive_units: Vec<_> = drive.encode_wide().collect();
    if root.is_empty() && drive_units.starts_with(&[92]) && !drive_units.ends_with(&[92]) {
        let parts: Vec<_> = drive_units.split(|unit| *unit == 92).collect();
        if (parts.len() == 4 && !matches!(parts[2], [] | [63 | 46] | [63, 46])) || parts.len() == 6
        {
            root = OsString::from("\\");
        }
    }
    let tail_units: Vec<_> = tail.encode_wide().collect();
    let parts: Vec<_> = tail_units
        .split(|unit| *unit == 92)
        .filter(|part| !part.is_empty() && *part != [46])
        .collect();
    if drive.is_empty() && root.is_empty() {
        if let Some(first) = parts.first() {
            if !path::split_root(Path::new(&OsString::from_wide(first)))
                .0
                .is_empty()
            {
                drive.push(".\\");
            }
        }
    }
    drive.push(root);
    drive.push(OsString::from_wide(&parts.join(&92)));
    if drive.is_empty() {
        drive.push(".");
    }
    drive.into()
}
