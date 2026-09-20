//! Checked parsing of the native reparse response, using SDK field offsets.

use crate::PathQueryError;
use byteorder::{ByteOrder, LittleEndian};
use std::{ffi::OsString, mem::offset_of, os::windows::ffi::OsStringExt, path::PathBuf};
use windows_sys::{
    Wdk::Storage::FileSystem::{
        REPARSE_DATA_BUFFER, REPARSE_DATA_BUFFER_0_0, REPARSE_DATA_BUFFER_0_1,
    },
    Win32::{
        Foundation::ERROR_INVALID_REPARSE_DATA,
        System::SystemServices::{IO_REPARSE_TAG_MOUNT_POINT, IO_REPARSE_TAG_SYMLINK},
    },
};

pub(crate) const INVALID_DATA: PathQueryError = PathQueryError::Windows {
    operation: "FSCTL_GET_REPARSE_POINT data",
    code: ERROR_INVALID_REPARSE_DATA,
};
const HEADER: usize = offset_of!(REPARSE_DATA_BUFFER, Anonymous);

pub(crate) fn target(data: &[u8]) -> Result<PathBuf, PathQueryError> {
    let header = data.get(..HEADER).ok_or(INVALID_DATA)?;
    let tag = LittleEndian::read_u32(header);
    let body_length = usize::from(LittleEndian::read_u16(&header[4..]));
    let name_base = match tag {
        IO_REPARSE_TAG_SYMLINK => offset_of!(REPARSE_DATA_BUFFER_0_0, PathBuffer),
        IO_REPARSE_TAG_MOUNT_POINT => offset_of!(REPARSE_DATA_BUFFER_0_1, PathBuffer),
        _ => return Err(PathQueryError::NotSymbolicLink),
    };
    let body = data.get(HEADER..HEADER + body_length).ok_or(INVALID_DATA)?;
    let fields = body.get(..name_base).ok_or(INVALID_DATA)?;
    let offset = usize::from(LittleEndian::read_u16(fields));
    let length = usize::from(LittleEndian::read_u16(&fields[2..]));
    if offset % 2 != 0 || length % 2 != 0 {
        return Err(INVALID_DATA);
    }
    let bytes = body
        .get(name_base + offset..name_base + offset + length)
        .ok_or(INVALID_DATA)?;
    let mut units: Vec<_> = bytes.chunks_exact(2).map(LittleEndian::read_u16).collect();
    // Source changes the NT namespace prefix only for names longer than four
    // units, irrespective of the reparse flag, and otherwise preserves spelling.
    if units.len() > 4 && units.starts_with(&[92, 63, 63, 92]) {
        units[1] = 92;
    }
    Ok(PathBuf::from(OsString::from_wide(&units)))
}

#[cfg(test)]
#[path = "../tests/support/reparse.rs"]
mod tests;
