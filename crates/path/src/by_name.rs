//! Optional Windows by-name acquisition. The missing SDK layout is checked by
//! compiling an independent C layout oracle against the installed Windows SDK.
use crate::{DirectoryAttributes, PathQueryError, windows::wide_path};
use libloading::os::windows::{LOAD_LIBRARY_SEARCH_SYSTEM32, Library};
use std::{ffi::c_void, path::Path, sync::OnceLock};
use windows_sys::{
    Win32::{
        Foundation::GetLastError,
        Storage::FileSystem::{FILE_ATTRIBUTE_REPARSE_POINT, FILE_ID_128},
        System::SystemServices::IO_REPARSE_TAG_SYMLINK,
    },
    core::{BOOL, PCWSTR},
};

// windows-sys 0.61.2 does not expose FILE_STAT_BASIC_INFORMATION or this API.
// Exact SDK winnt.h layout, not a truncated prefix or guessed byte buffer.
// All fields, size/alignment and enum value are independently SDK-tested.
#[repr(C)]
#[derive(Default)]
struct BasicInformation {
    file_id: i64,
    creation_time: i64,
    last_access_time: i64,
    last_write_time: i64,
    change_time: i64,
    allocation_size: i64,
    end_of_file: i64,
    attributes: u32,
    tag: u32,
    number_of_links: u32,
    device_type: u32,
    device_characteristics: u32,
    reserved: u32,
    volume_serial_number: i64,
    file_id_128: FILE_ID_128,
}

type Query = unsafe extern "system" fn(PCWSTR, i32, *mut c_void, u32) -> BOOL;
const BASIC_BY_NAME: i32 = 3;
const MODULE: &str = "api-ms-win-core-file-l2-1-4.dll";
const SYMBOL: &[u8] = b"GetFileInformationByName\0";

struct NameQuery {
    // Own the module for at least as long as the copied function pointer.
    _library: Library,
    query: Query,
}

// SAFETY contract: callers only request this SDK function with Query's signature,
// or intentionally nonexistent names in failure tests. Never an arbitrary plugin.
fn load_query(module: &str, symbol: &[u8]) -> Option<NameQuery> {
    // SAFETY: trusted Windows system DLL/API-set, resolved only in System32.
    let library = unsafe { Library::load_with_flags(module, LOAD_LIBRARY_SEARCH_SYSTEM32) }.ok()?;
    // SAFETY: the fixed symbol has the SDK Query ABI. Ownership stays in NameQuery.
    let query = *unsafe { library.get::<Query>(symbol) }.ok()?;
    Some(NameQuery {
        _library: library,
        query,
    })
}

/// Test the exact symlink tag using the optional native by-name API, without
/// opening a file handle. This is the first stage, not the full source predicate.
///
/// # Errors
/// Retains native failures for the caller's error-specific fallback. An unavailable
/// DLL/export becomes Windows error 50 (not supported), matching source policy.
/// NUL input is rejected before any native query.
pub fn symbolic_link_by_name(path: &Path) -> Result<bool, PathQueryError> {
    let info = attributes_by_name(path)?;
    Ok(info.attributes & FILE_ATTRIBUTE_REPARSE_POINT != 0
        && info.reparse_tag == IO_REPARSE_TAG_SYMLINK)
}

pub(crate) fn attributes_by_name(path: &Path) -> Result<DirectoryAttributes, PathQueryError> {
    static QUERY: OnceLock<Option<NameQuery>> = OnceLock::new();
    let query = QUERY.get_or_init(|| load_query(MODULE, SYMBOL));
    query_with(path, query.as_ref().map(|loaded| loaded.query))
}

fn query_with(path: &Path, query: Option<Query>) -> Result<DirectoryAttributes, PathQueryError> {
    let name = wide_path(path)?;
    let query = query.ok_or(PathQueryError::Windows {
        operation: "GetFileInformationByName",
        code: 50,
    })?;
    let mut info = BasicInformation::default();
    // SAFETY: live NUL-terminated input and exclusively writable, initialized SDK
    // layout with matching class and complete buffer extent; function owner lives.
    let success = unsafe {
        query(
            name.as_ptr(),
            BASIC_BY_NAME,
            (&raw mut info).cast(),
            u32::try_from(size_of::<BasicInformation>()).expect("SDK structure fits DWORD"),
        )
    };
    if success == 0 {
        return Err(PathQueryError::Windows {
            operation: "GetFileInformationByName",
            // SAFETY: immediate thread-local error capture; no intervening call.
            code: unsafe { GetLastError() },
        });
    }
    Ok(DirectoryAttributes {
        attributes: info.attributes,
        reparse_tag: info.tag,
    })
}

#[cfg(test)]
#[path = "../tests/support/by_name.rs"]
mod tests;
