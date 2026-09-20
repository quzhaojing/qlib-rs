//! Native current-directory acquisition without changing process state.
use super::{Api, PathQueryError, WINDOWS, last_error};
use std::{ffi::OsString, os::windows::ffi::OsStringExt, path::PathBuf};
use windows_sys::{Win32::System::Environment::GetCurrentDirectoryW, core::PWSTR};

type Query = unsafe extern "system" fn(u32, PWSTR) -> u32;

/// Return the process current directory as lossless native UTF-16.
/// # Errors
/// Preserves native query errors and fallible growth-allocation failure.
pub fn current_directory() -> Result<PathBuf, PathQueryError> {
    query_with(&WINDOWS, GetCurrentDirectoryW)
}

fn query_with(api: &Api, query: Query) -> Result<PathBuf, PathQueryError> {
    // CPython's Windows osdefs.h sets MAXPATHLEN to 256.
    let mut stack = [0_u16; 256];
    let mut heap;
    let mut buffer = &mut stack[..];
    loop {
        // SAFETY: exclusive initialized UTF-16 output, with its exact accessible
        // length. Native query reports required capacity including NUL on growth.
        let written = unsafe {
            query(
                u32::try_from(buffer.len()).expect("native DWORD capacity"),
                buffer.as_mut_ptr(),
            )
        };
        if written == 0 {
            return Err(last_error(api, "GetCurrentDirectoryW"));
        }
        if (written as usize) < buffer.len() {
            return Ok(OsString::from_wide(&buffer[..written as usize]).into());
        }
        // Retry safely if another thread changes the directory length again.
        heap = (api.allocate)(written as usize)?;
        buffer = &mut heap;
    }
}

#[cfg(test)]
#[path = "../../tests/support/current_directory.rs"]
mod tests;
