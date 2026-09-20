//! Attribute/tag projection of source Windows lstat, retaining its query gates.
use super::{
    Api, DirectoryAttributes, FILE_ATTRIBUTE_REPARSE_POINT, FILE_ATTRIBUTE_TAG_INFO,
    FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_OPEN_REPARSE_POINT, FILE_INFO_BY_HANDLE_CLASS,
    FILE_READ_ATTRIBUTES, FileAttributeTagInfo, GetFileInformationByHandleEx, INVALID_HANDLE_VALUE,
    OPEN_EXISTING, OwnedNativeHandle, PathQueryError, TagQuery, WINDOWS, directory_attributes,
    last_error, wide_path,
};
use std::mem::ManuallyDrop;
use std::{path::Path, ptr};
use windows_sys::Win32::{
    Foundation::GENERIC_READ,
    Storage::FileSystem::{
        BY_HANDLE_FILE_INFORMATION, FILE_ATTRIBUTE_NORMAL, FILE_BASIC_INFO, FILE_ID_INFO,
        FILE_SHARE_READ, FILE_SHARE_WRITE, FILE_TYPE_DISK, FILE_TYPE_UNKNOWN, FileBasicInfo,
        FileIdInfo, GetFileAttributesW, GetFileInformationByHandle, GetFileType,
    },
};
use windows_sys::{
    Win32::Foundation::HANDLE,
    core::{BOOL, PCWSTR},
};

const EMPTY: DirectoryAttributes = DirectoryAttributes {
    attributes: 0,
    reparse_tag: 0,
};
struct StatApi {
    base: &'static Api,
    query: TagQuery,
    information: unsafe extern "system" fn(HANDLE, *mut BY_HANDLE_FILE_INFORMATION) -> BOOL,
    file_type: unsafe extern "system" fn(HANDLE) -> u32,
    attributes: unsafe extern "system" fn(PCWSTR) -> u32,
    by_name: fn(&Path) -> Result<DirectoryAttributes, PathQueryError>,
    directory: fn(&Path) -> Result<Option<DirectoryAttributes>, PathQueryError>,
}
const NATIVE: StatApi = StatApi {
    base: &WINDOWS,
    query: GetFileInformationByHandleEx,
    information: GetFileInformationByHandle,
    file_type: GetFileType,
    attributes: GetFileAttributesW,
    by_name: crate::by_name::attributes_by_name,
    directory: directory_attributes,
};

/// Return the attribute/tag projection of Windows Python `lstat`.
/// Retains its fast-path, fallback, query and close-error semantics, without
/// claiming the complete timestamp/inode/mode stat-result representation.
/// # Errors
/// Propagates invalid input and native stat failures, including final close failure.
pub fn lstat_attributes(path: &Path) -> Result<DirectoryAttributes, PathQueryError> {
    stat_with(path, &NATIVE)
}

fn name_surrogate(info: DirectoryAttributes) -> bool {
    info.reparse_tag & 0x2000_0000 != 0
}

fn stat_with(path: &Path, api: &StatApi) -> Result<DirectoryAttributes, PathQueryError> {
    match (api.by_name)(path) {
        Ok(info) if info.attributes & FILE_ATTRIBUTE_REPARSE_POINT == 0 || name_surrogate(info) => {
            Ok(info)
        }
        Err(
            error @ PathQueryError::Windows {
                code: 2 | 3 | 21 | 67,
                ..
            },
        ) => Err(error),
        Ok(_) | Err(PathQueryError::Windows { .. }) => slow(path, false, api),
        Err(error) => Err(error),
    }
}

fn open(
    name: PCWSTR,
    follow: bool,
    console: bool,
    api: &StatApi,
) -> Result<OwnedNativeHandle, PathQueryError> {
    let flags = FILE_FLAG_BACKUP_SEMANTICS
        | if follow {
            0
        } else {
            FILE_FLAG_OPEN_REPARSE_POINT
        };
    // SAFETY: caller retains the validated terminated name. Other pointers null.
    let raw = unsafe {
        (api.base.open)(
            name,
            FILE_READ_ATTRIBUTES | if console { GENERIC_READ } else { 0 },
            if console {
                FILE_SHARE_READ | FILE_SHARE_WRITE
            } else {
                0
            },
            ptr::null(),
            OPEN_EXISTING,
            flags,
            ptr::null_mut(),
        )
    };
    if raw == INVALID_HANDLE_VALUE {
        Err(last_error(api.base, "CreateFileW"))
    } else {
        Ok(OwnedNativeHandle {
            raw,
            close: api.base.close_file,
        })
    }
}

fn finish(
    handle: OwnedNativeHandle,
    result: Result<DirectoryAttributes, PathQueryError>,
    api: &StatApi,
) -> Result<DirectoryAttributes, PathQueryError> {
    let handle = ManuallyDrop::new(handle);
    // SAFETY: uniquely owned live handle; suppress Drop because this explicit close
    // is its sole release, even on failure. Capture close error before other calls.
    if unsafe { (handle.close)(handle.raw) } == 0 {
        Err(last_error(api.base, "CloseHandle"))
    } else {
        result
    }
}

fn slow(
    path: &Path,
    mut follow: bool,
    api: &StatApi,
) -> Result<DirectoryAttributes, PathQueryError> {
    let name = wide_path(path)?;
    let mut unhandled = false;
    let handle = match open(name.as_ptr(), follow, false, api) {
        Ok(handle) => handle,
        Err(original @ PathQueryError::Windows { code: 5 | 32, .. }) => {
            return match (api.directory)(path) {
                Ok(Some(info))
                    if info.attributes & FILE_ATTRIBUTE_REPARSE_POINT == 0
                        || (!follow && name_surrogate(info)) =>
                {
                    Ok(info)
                }
                Err(
                    error @ PathQueryError::Windows {
                        code: 2 | 3 | 21 | 67,
                        ..
                    },
                ) => Err(error),
                _ => Err(original),
            };
        }
        Err(original @ PathQueryError::Windows { code: 87, .. }) => {
            open(name.as_ptr(), follow, true, api).map_err(|_| original)?
        }
        Err(original @ PathQueryError::Windows { code: 1920, .. }) if follow => {
            follow = false;
            unhandled = true;
            open(name.as_ptr(), false, false, api).map_err(|_| original)?
        }
        Err(error) => return Err(error),
    };
    match inspect_handle(&handle, name.as_ptr(), follow, unhandled, api) {
        Ok(None) => {
            // Source ignores this intermediate close result before retrying.
            drop(handle);
            slow(path, true, api)
        }
        result => finish(
            handle,
            result.map(|value| value.expect("non-retry outcome contains attributes")),
            api,
        ),
    }
}

fn query_info<T: Default>(
    handle: HANDLE,
    class: FILE_INFO_BY_HANDLE_CLASS,
    api: &StatApi,
) -> Result<T, PathQueryError> {
    let mut info = T::default();
    // SAFETY: private callers pair class with its exact SDK structure; exclusive
    // aligned initialized output lives throughout the synchronous native call.
    if unsafe {
        (api.query)(
            handle,
            class,
            (&raw mut info).cast(),
            u32::try_from(size_of::<T>()).expect("SDK structure fits DWORD"),
        )
    } == 0
    {
        Err(last_error(api.base, "GetFileInformationByHandleEx"))
    } else {
        Ok(info)
    }
}

fn inspect_handle(
    handle: &OwnedNativeHandle,
    name: PCWSTR,
    follow: bool,
    unhandled: bool,
    api: &StatApi,
) -> Result<Option<DirectoryAttributes>, PathQueryError> {
    // SAFETY: the handle remains owned and live through this inspection.
    let file_type = unsafe { (api.file_type)(handle.raw) };
    if file_type != FILE_TYPE_DISK {
        let error = last_error(api.base, "GetFileType");
        if file_type == FILE_TYPE_UNKNOWN
            && !matches!(error, PathQueryError::Windows { code: 0, .. })
        {
            return Err(error);
        }
        // SAFETY: caller retains the live validated name. Source queries these
        // attributes for mode selection, but its returned attribute/tag fields stay zero.
        unsafe { (api.attributes)(name) };
        return Ok(Some(EMPTY));
    }
    let mut tag = FILE_ATTRIBUTE_TAG_INFO::default();
    if !follow {
        tag = match query_info(handle.raw, FileAttributeTagInfo, api) {
            Ok(info) => info,
            Err(PathQueryError::Windows {
                code: 1 | 50 | 87, ..
            }) => FILE_ATTRIBUTE_TAG_INFO {
                FileAttributes: FILE_ATTRIBUTE_NORMAL,
                ReparseTag: 0,
            },
            Err(error) => return Err(error),
        };
        if tag.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            if tag.ReparseTag & 0x2000_0000 != 0 {
                if unhandled {
                    return Err(PathQueryError::Windows {
                        operation: "lstat",
                        code: 1920,
                    });
                }
            } else if !unhandled {
                return Ok(None);
            }
        }
    }
    let mut info = BY_HANDLE_FILE_INFORMATION::default();
    // SAFETY: owned live handle and exclusive initialized SDK storage.
    let metadata = if unsafe { (api.information)(handle.raw, &raw mut info) } == 0 {
        Err(last_error(api.base, "GetFileInformationByHandle"))
    } else {
        query_info::<FILE_BASIC_INFO>(handle.raw, FileBasicInfo, api).map(|_| ())
    };
    match metadata {
        Ok(()) => {}
        Err(PathQueryError::Windows {
            code: 1 | 50 | 87, ..
        }) => return Ok(Some(EMPTY)),
        Err(error) => return Err(error),
    }
    // FileIdInfo is optional in the source; failure does not invalidate stat.
    let _ = query_info::<FILE_ID_INFO>(handle.raw, FileIdInfo, api);
    Ok(Some(DirectoryAttributes {
        attributes: info.dwFileAttributes,
        reparse_tag: tag.ReparseTag,
    }))
}

#[cfg(test)]
#[path = "../../tests/support/stat.rs"]
mod tests;
