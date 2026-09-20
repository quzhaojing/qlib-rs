use std::{
    ffi::{OsString, c_void},
    os::windows::ffi::{OsStrExt, OsStringExt},
    path::{Path, PathBuf},
    ptr,
};

use thiserror::Error;
use widestring::U16CString;
use windows_sys::{
    Win32::{
        Foundation::{CloseHandle, GetLastError, HANDLE, INVALID_HANDLE_VALUE},
        Globalization::{LCMAP_LOWERCASE, LCMapStringEx, LOCALE_NAME_INVARIANT, NLSVERSIONINFO},
        Security::SECURITY_ATTRIBUTES,
        Storage::FileSystem::{
            CreateFileW, FILE_ATTRIBUTE_REPARSE_POINT, FILE_ATTRIBUTE_TAG_INFO,
            FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_OPEN_REPARSE_POINT, FILE_INFO_BY_HANDLE_CLASS,
            FILE_READ_ATTRIBUTES, FileAttributeTagInfo, FindClose, FindFirstFileW,
            GetFileInformationByHandleEx, GetFinalPathNameByHandleW,
            MAXIMUM_REPARSE_DATA_BUFFER_SIZE, OPEN_EXISTING, WIN32_FIND_DATAW,
        },
        System::{
            IO::{DeviceIoControl, OVERLAPPED},
            Ioctl::FSCTL_GET_REPARSE_POINT,
            SystemServices::IO_REPARSE_TAG_SYMLINK,
        },
    },
    core::{BOOL, PCWSTR, PWSTR},
};

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PathQueryError {
    #[error("embedded null character")]
    EmbeddedNul,
    #[error("{operation} failed with Windows error {code}")]
    Windows { operation: &'static str, code: u32 },
    #[error("could not allocate the native path buffer")]
    Allocation,
    #[error("not a symbolic link")]
    NotSymbolicLink,
    #[error("input string is too long")]
    InputTooLong,
}

/// Owned attribute/tag projection shared by native query and stat boundaries.
/// Directory-search acquisition clears the tag on non-reparse entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectoryAttributes {
    pub attributes: u32,
    pub reparse_tag: u32,
}

/// Query source `attributes_from_dir` attributes/tag through `FindFirstFileW`.
/// Retains its wildcard semantics and trailing-separator trimming rules.
/// `None` means trimming left nothing queryable; callers must retain their prior
/// failure in that case, not manufacture a new native error.
///
/// # Errors
/// Rejects NUL, preserves search errors, and maps trailing-name copy allocation
/// failure to native error 8. Every acquired search handle is closed exactly once.
pub fn directory_attributes(path: &Path) -> Result<Option<DirectoryAttributes>, PathQueryError> {
    directory_attributes_with(path, &WINDOWS)
}

fn directory_attributes_with(
    path: &Path,
    api: &Api,
) -> Result<Option<DirectoryAttributes>, PathQueryError> {
    let name = wide_path(path)?;
    let mut trimmed;
    let query_name = if matches!(name.as_slice().last(), Some(92 | 47)) {
        trimmed = (api.allocate)(name.len() + 1).map_err(|_| PathQueryError::Windows {
            operation: "attributes_from_dir",
            code: 8,
        })?;
        trimmed.copy_from_slice(name.as_slice_with_nul());
        let mut index = name.len() - 1;
        while index > 0 && matches!(trimmed[index], 92 | 47) {
            trimmed[index] = 0;
            index -= 1;
        }
        // The source tests the last remaining index, not the remaining length.
        // Thus a one-character relative name with a trailing separator also skips.
        if index == 0 || (index == 1 && trimmed[1] == 58) {
            return Ok(None);
        }
        trimmed.as_ptr()
    } else {
        name.as_ptr()
    };
    let mut data = WIN32_FIND_DATAW::default();
    // SAFETY: either live buffer is terminated, and data is exclusive SDK storage.
    let raw = unsafe { (api.find)(query_name, &raw mut data) };
    if raw == INVALID_HANDLE_VALUE {
        return Err(last_error(api, "FindFirstFileW"));
    }
    let _handle = OwnedNativeHandle {
        raw,
        close: api.close_search,
    };
    Ok(Some(DirectoryAttributes {
        attributes: data.dwFileAttributes,
        reparse_tag: if data.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            data.dwReserved0
        } else {
            0
        },
    }))
}

type Open = unsafe extern "system" fn(
    PCWSTR,
    u32,
    u32,
    *const SECURITY_ATTRIBUTES,
    u32,
    u32,
    HANDLE,
) -> HANDLE;
type Query = unsafe extern "system" fn(HANDLE, PWSTR, u32, u32) -> u32;
type Find = unsafe extern "system" fn(PCWSTR, *mut WIN32_FIND_DATAW) -> HANDLE;
type Close = unsafe extern "system" fn(HANDLE) -> BOOL;
type TagQuery =
    unsafe extern "system" fn(HANDLE, FILE_INFO_BY_HANDLE_CLASS, *mut c_void, u32) -> BOOL;
type MapCase = unsafe extern "system" fn(
    PCWSTR,
    u32,
    PCWSTR,
    i32,
    PWSTR,
    i32,
    *const NLSVERSIONINFO,
    *const c_void,
    isize,
) -> i32;
type Ioctl = unsafe extern "system" fn(
    HANDLE,
    u32,
    *const c_void,
    u32,
    *mut c_void,
    u32,
    *mut u32,
    *mut OVERLAPPED,
) -> BOOL;

// Private ABI seam. Functions must honor the Windows buffer/handle contracts.
struct Api {
    open: Open,
    query: Query,
    find: Find,
    close_file: Close,
    close_search: Close,
    last_error: unsafe extern "system" fn() -> u32,
    allocate: fn(usize) -> Result<Vec<u16>, PathQueryError>,
    // A successful conversion must preserve the exact source length. Validators
    // may reject inputs, but must never expand the accessible input buffer extent.
    native_length: fn(usize) -> Result<i32, PathQueryError>,
}

const WINDOWS: Api = Api {
    open: CreateFileW,
    query: GetFinalPathNameByHandleW,
    find: FindFirstFileW,
    close_file: CloseHandle,
    close_search: FindClose,
    last_error: GetLastError,
    allocate,
    native_length,
};

struct OwnedNativeHandle {
    raw: HANDLE,
    close: Close,
}

impl Drop for OwnedNativeHandle {
    fn drop(&mut self) {
        // SAFETY: Created only after successful acquisition; uniquely owned and
        // paired with the matching Windows closer. Close is called exactly once.
        unsafe { (self.close)(self.raw) };
    }
}

/// Open the reparse point with attribute-read access and test its exact symlink tag.
/// Junctions and other name-surrogate tags return false. This supplies only the
/// open/handle-query stage of Python's classification, not its complete predicate:
/// the caller still needs by-name acquisition and source-specific stat fallbacks.
///
/// # Errors
/// Preserves NUL and native open/query failures separately, so the caller can apply
/// the appropriate fallback. Acquired handles close on success and query failure.
pub fn symbolic_link_by_open(path: &Path) -> Result<bool, PathQueryError> {
    symbolic_link_by_open_with(path, &WINDOWS, GetFileInformationByHandleEx)
}

fn symbolic_link_by_open_with(
    path: &Path,
    api: &Api,
    query: TagQuery,
) -> Result<bool, PathQueryError> {
    let name = wide_path(path)?;
    // SAFETY: validated live terminated path; no security/template pointers.
    // Match the source classification open, not the zero-access target reader.
    let raw = unsafe {
        (api.open)(
            name.as_ptr(),
            FILE_READ_ATTRIBUTES,
            0,
            ptr::null(),
            OPEN_EXISTING,
            FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT,
            ptr::null_mut(),
        )
    };
    if raw == INVALID_HANDLE_VALUE {
        return Err(last_error(api, "CreateFileW"));
    }
    let handle = OwnedNativeHandle {
        raw,
        close: api.close_file,
    };
    let mut info = FILE_ATTRIBUTE_TAG_INFO::default();
    // SAFETY: live owned handle and exclusive, aligned SDK output structure;
    // the class and size describe exactly this initialized allocation.
    let success = unsafe {
        query(
            handle.raw,
            FileAttributeTagInfo,
            (&raw mut info).cast(),
            u32::try_from(size_of::<FILE_ATTRIBUTE_TAG_INFO>()).expect("SDK structure fits DWORD"),
        )
    };
    if success == 0 {
        return Err(last_error(api, "GetFileInformationByHandleEx"));
    }
    Ok(info.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT != 0
        && info.ReparseTag == IO_REPARSE_TAG_SYMLINK)
}

/// Equivalent to Windows Python `_getfinalpathname`: open existing file/directory
/// with desired access zero, share mode zero and backup semantics; query the DOS
/// volume final path. Retains the verbatim prefix and original Windows error code.
///
/// # Errors
/// Returns embedded-NUL, native open/query or allocation failures. No path fallback.
pub fn final_path(path: &Path) -> Result<PathBuf, PathQueryError> {
    final_path_with(path, &WINDOWS)
}

fn final_path_with(path: &Path, api: &Api) -> Result<PathBuf, PathQueryError> {
    let name = wide_path(path)?;
    // SAFETY: name is live, terminated and contains no interior NUL. No security
    // or template pointers are supplied; constants match the source open policy.
    let raw = unsafe {
        (api.open)(
            name.as_ptr(),
            0,
            0,
            ptr::null(),
            OPEN_EXISTING,
            FILE_FLAG_BACKUP_SEMANTICS,
            ptr::null_mut(),
        )
    };
    if raw == INVALID_HANDLE_VALUE {
        return Err(last_error(api, "CreateFileW"));
    }
    let handle = OwnedNativeHandle {
        raw,
        close: api.close_file,
    };
    let mut capacity = 260_u32;
    loop {
        let mut buffer = (api.allocate)(capacity as usize)?;
        // SAFETY: handle stays owned/live and buffer has capacity initialized u16
        // entries. API receives exactly that length; flags zero select DOS names.
        let length = unsafe { (api.query)(handle.raw, buffer.as_mut_ptr(), capacity, 0) };
        if length == 0 {
            return Err(last_error(api, "GetFinalPathNameByHandleW"));
        }
        if length < capacity {
            return Ok(PathBuf::from(OsString::from_wide(
                &buffer[..length as usize],
            )));
        }
        // Insufficient-buffer result includes the terminator; retry may grow again
        // if a concurrent rename changes the path length, as in CPython's loop.
        capacity = length;
    }
}

/// Query the real matching filename using `FindFirstFileW`, without opening the
/// file itself. Wildcards retain native first-match behavior; no sorting is added.
///
/// # Errors
/// Returns embedded-NUL or native lookup failures; always closes acquired search handles.
pub fn find_name(path: &Path) -> Result<OsString, PathQueryError> {
    find_name_with(path, &WINDOWS)
}

/// Read one Windows symbolic link or junction, retaining the substitute name.
/// Does not follow chains, normalize the target or remove a verbatim prefix.
///
/// # Errors
/// Returns native open/query errors, embedded NUL, unsupported reparse tags or
/// invalid reparse data. Every acquired handle is closed, including on failure.
pub fn read_link(path: &Path) -> Result<PathBuf, PathQueryError> {
    read_link_with(path, &WINDOWS, DeviceIoControl)
}

/// Windows Python `ntpath.normcase`: replace forward slashes and lowercase using
/// the native invariant locale. Preserves embedded NUL and unpaired UTF-16 data.
/// Does not collapse dots, access the filesystem or normalize Unicode composition.
///
/// # Errors
/// Returns native mapping failures, oversized-input or allocation errors.
pub fn normalize_case(path: &Path) -> Result<OsString, PathQueryError> {
    normalize_case_with(path, &WINDOWS, LCMapStringEx)
}

fn normalize_case_with(path: &Path, api: &Api, map: MapCase) -> Result<OsString, PathQueryError> {
    let input: Vec<_> = path
        .as_os_str()
        .encode_wide()
        .map(|unit| if unit == 47 { 92 } else { unit })
        .collect();
    if input.is_empty() {
        return Ok(OsString::new());
    }
    let length = (api.native_length)(input.len())?;
    // SAFETY: Locale is a static terminated SDK string; input is live for its
    // explicit UTF-16 length (NUL data is allowed). Null destination queries size.
    let capacity = unsafe {
        map(
            LOCALE_NAME_INVARIANT,
            LCMAP_LOWERCASE,
            input.as_ptr(),
            length,
            ptr::null_mut(),
            0,
            ptr::null(),
            ptr::null(),
            0,
        )
    };
    if capacity <= 0 {
        return Err(last_error(api, "LCMapStringEx"));
    }
    let mut output = (api.allocate)(capacity.unsigned_abs() as usize)?;
    // SAFETY: Separate initialized output holds exactly capacity UTF-16 units.
    // The mapper must honor the native contract and return no more than capacity.
    let written = unsafe {
        map(
            LOCALE_NAME_INVARIANT,
            LCMAP_LOWERCASE,
            input.as_ptr(),
            length,
            output.as_mut_ptr(),
            capacity,
            ptr::null(),
            ptr::null(),
            0,
        )
    };
    if written <= 0 {
        return Err(last_error(api, "LCMapStringEx"));
    }
    Ok(OsString::from_wide(
        &output[..written.unsigned_abs() as usize],
    ))
}

fn native_length(length: usize) -> Result<i32, PathQueryError> {
    i32::try_from(length).map_err(|_| PathQueryError::InputTooLong)
}

fn read_link_with(path: &Path, api: &Api, ioctl: Ioctl) -> Result<PathBuf, PathQueryError> {
    let name = wide_path(path)?;
    // SAFETY: name is a validated, live terminated string; no other input pointers
    // are supplied. Opening the reparse point itself prevents following the link.
    let raw = unsafe {
        (api.open)(
            name.as_ptr(),
            0,
            0,
            ptr::null(),
            OPEN_EXISTING,
            FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS,
            ptr::null_mut(),
        )
    };
    if raw == INVALID_HANDLE_VALUE {
        return Err(last_error(api, "CreateFileW"));
    }
    let handle = OwnedNativeHandle {
        raw,
        close: api.close_file,
    };
    let mut buffer = [0_u8; MAXIMUM_REPARSE_DATA_BUFFER_SIZE as usize];
    let mut returned = 0;
    // SAFETY: handle is live; output is an initialized byte buffer of the specified
    // size. The synchronous call receives writable byte-count storage, no input
    // buffer and no OVERLAPPED structure. No typed/aligned buffer cast is used.
    let success = unsafe {
        ioctl(
            handle.raw,
            FSCTL_GET_REPARSE_POINT,
            ptr::null(),
            0,
            buffer.as_mut_ptr().cast(),
            MAXIMUM_REPARSE_DATA_BUFFER_SIZE,
            &raw mut returned,
            ptr::null_mut(),
        )
    };
    if success == 0 {
        return Err(last_error(api, "DeviceIoControl"));
    }
    let data = buffer
        .get(..returned as usize)
        .ok_or(crate::reparse::INVALID_DATA)?;
    crate::reparse::target(data)
}

fn find_name_with(path: &Path, api: &Api) -> Result<OsString, PathQueryError> {
    let name = wide_path(path)?;
    let mut data = WIN32_FIND_DATAW::default();
    // SAFETY: name is terminated and live; data is a correctly aligned, initialized
    // Windows SDK structure, exclusively writable for the duration of the call.
    let raw = unsafe { (api.find)(name.as_ptr(), &raw mut data) };
    if raw == INVALID_HANDLE_VALUE {
        return Err(last_error(api, "FindFirstFileW"));
    }
    let _handle = OwnedNativeHandle {
        raw,
        close: api.close_search,
    };
    let length = data
        .cFileName
        .iter()
        .position(|unit| *unit == 0)
        .unwrap_or(data.cFileName.len());
    Ok(OsString::from_wide(&data.cFileName[..length]))
}

pub(crate) fn wide_path(path: &Path) -> Result<U16CString, PathQueryError> {
    // Add our own terminator before validation: an input trailing NUL is invalid
    // Python path data too, not an already-supplied C-string terminator.
    let mut units: Vec<_> = path.as_os_str().encode_wide().collect();
    units.push(0);
    U16CString::from_vec(units).map_err(|_| PathQueryError::EmbeddedNul)
}

fn last_error(api: &Api, operation: &'static str) -> PathQueryError {
    // SAFETY: This no-argument native getter is called immediately after failure,
    // before closing a handle can overwrite the thread-local Windows error.
    PathQueryError::Windows {
        operation,
        code: unsafe { (api.last_error)() },
    }
}

fn allocate(capacity: usize) -> Result<Vec<u16>, PathQueryError> {
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(capacity)
        .map_err(|_| PathQueryError::Allocation)?;
    buffer.resize(capacity, 0);
    Ok(buffer)
}

#[cfg(test)]
#[path = "../tests/support/query_failures.rs"]
mod query_failures;

mod stat;
pub use stat::lstat_attributes;

mod current_directory;
pub use current_directory::current_directory;
