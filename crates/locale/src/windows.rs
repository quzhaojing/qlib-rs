use std::{
    ffi::{CStr, c_char, c_int, c_void},
    marker::PhantomData,
    ptr::NonNull,
    rc::Rc,
};
use thiserror::Error;
use widestring::{U16CStr, error::Utf16Error};

/// Owned Unicode numeric fields, independent of the lifetime of CRT storage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NumericLocale {
    /// Decimal mark from UCRT's wide-character field.
    pub decimal: String,
    /// Thousands separator from UCRT's wide-character field.
    pub separator: String,
    /// Raw grouping sequence including its terminating zero. `CHAR_MAX` is 127.
    pub grouping: Vec<u8>,
}

/// Acquisition/decoding failures; no fallback locale is silently substituted.
#[derive(Debug, Error)]
pub enum LocaleError {
    #[error("UCRT locale data unavailable at {0}")]
    Unavailable(&'static str),
    #[error("UCRT returned invalid thread locale mode {0}")]
    ThreadMode(c_int),
    #[error("UCRT {field} is not valid UTF-16: {source}")]
    InvalidUtf16 {
        field: &'static str,
        #[source]
        source: Utf16Error,
    },
}

// SDK locale.h order: ten char pointers, eight chars, eight wchar_t pointers.
// Consecutive same-typed members are represented as arrays, preserving C layout.
// native_locale_abi.c independently checks size/alignment and accessed offsets.
#[repr(C)]
struct Lconv {
    narrow: [*const c_char; 10],
    flags: [c_char; 8],
    wide: [*const u16; 8],
}

#[link(name = "ucrt")]
unsafe extern "C" {
    fn localeconv() -> *const Lconv;
    fn _configthreadlocale(flag: c_int) -> c_int;
    fn _get_current_locale() -> *mut c_void;
    fn _free_locale(locale: *mut c_void);
}

// Private call seam for verifying real cleanup under acquisition failures.
// Implementations must honor UCRT's pointer ownership and thread semantics.
struct Crt {
    locale: unsafe extern "C" fn() -> *const Lconv,
    config: unsafe extern "C" fn(c_int) -> c_int,
    acquire: unsafe extern "C" fn() -> *mut c_void,
    free: unsafe extern "C" fn(*mut c_void),
}

const UCRT: Crt = Crt {
    locale: localeconv,
    config: _configthreadlocale,
    acquire: _get_current_locale,
    free: _free_locale,
};

struct FrozenThread<'a> {
    api: &'a Crt,
    previous: c_int,
    _same_thread: PhantomData<Rc<()>>,
}

impl Drop for FrozenThread<'_> {
    fn drop(&mut self) {
        // SAFETY: Previous is validated as a supported UCRT flag, and this guard
        // cannot move to another thread. Valid flags cannot fail validation.
        unsafe { (self.api.config)(self.previous) };
    }
}

struct OwnedLocale<'a> {
    api: &'a Crt,
    pointer: NonNull<c_void>,
}

impl Drop for OwnedLocale<'_> {
    fn drop(&mut self) {
        // SAFETY: This unique handle came from the matching acquire function.
        unsafe { (self.api.free)(self.pointer.as_ptr()) };
    }
}

/// Copy the calling thread's current dynamically linked MSVC UCRT numeric locale.
///
/// Temporarily freezes and then restores the thread's locale mode. The current
/// thread reference is refreshed first, and an owned CRT handle pins its data
/// during copying. No borrowed pointers escape, no process locale is changed,
/// and no application callback or async suspension occurs within this scope.
/// This observes the same CRT instance, not the locale of another process or a
/// separately statically linked runtime. Do not call from an asynchronous signal
/// handler; locale mutation from such a handler is not supported by UCRT.
/// # Errors
/// Reports unavailable CRT data, an invalid thread mode, or invalid UTF-16.
pub fn current_numeric_locale() -> Result<NumericLocale, LocaleError> {
    snapshot(&UCRT)
}

fn snapshot(api: &Crt) -> Result<NumericLocale, LocaleError> {
    // SAFETY: No-argument lookup refreshes this thread's retained reference. Its
    // pointer is not dereferenced or retained across the following calls.
    NonNull::new(unsafe { (api.locale)() }.cast_mut())
        .ok_or(LocaleError::Unavailable("refresh"))?;
    // SAFETY: 1 enables per-thread locale; unlike localeconv it does not refresh.
    let previous = unsafe { (api.config)(1) };
    if !matches!(previous, 1 | 2) {
        return Err(LocaleError::ThreadMode(previous));
    }
    let _thread = FrozenThread {
        api,
        previous,
        _same_thread: PhantomData,
    };
    // SAFETY: Freeze prevents a global update between ownership acquisition and
    // the subsequent lookup. CRT allocation does not invoke Rust allocators.
    let pointer =
        NonNull::new(unsafe { (api.acquire)() }).ok_or(LocaleError::Unavailable("acquire"))?;
    let _owned = OwnedLocale { api, pointer };
    // SAFETY: The private UCRT implementation returns its thread's lconv. The
    // owning handle pins this exact data, including during Rust allocations.
    let raw = unsafe { (api.locale)().as_ref() }.ok_or(LocaleError::Unavailable("snapshot"))?;
    // SAFETY: The pinned lconv's selected fields are immutable, valid C strings.
    // Only owned Rust values escape; both RAII guards also run on every error.
    unsafe {
        let decimal = wide_string(raw.wide[0], "decimal")?;
        let separator = wide_string(raw.wide[1], "separator")?;
        let grouping = CStr::from_ptr(raw.narrow[2]).to_bytes_with_nul().to_vec();
        Ok(NumericLocale {
            decimal,
            separator,
            grouping,
        })
    }
}

// SAFETY: Caller pins a valid immutable NUL-terminated UTF-16 allocation until
// this function returns. Malformed code units are errors, not lossy replacement.
unsafe fn wide_string(pointer: *const u16, field: &'static str) -> Result<String, LocaleError> {
    // SAFETY: Same lifetime and termination preconditions as this function.
    unsafe { U16CStr::from_ptr_str(pointer) }
        .to_string()
        .map_err(|source| LocaleError::InvalidUtf16 { field, source })
}

#[cfg(test)]
#[path = "../tests/support/windows.rs"]
mod tests;
