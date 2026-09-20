//! Current-runtime locale acquisition. No OS-default or environment fallback.
//!
//! The Windows implementation targets dynamically linked MSVC UCRT. Other
//! platforms retain core's explicit locale-provider boundary.

#[cfg(all(windows, target_env = "msvc", not(target_feature = "crt-static")))]
mod windows;
#[cfg(all(windows, target_env = "msvc", not(target_feature = "crt-static")))]
pub use windows::{LocaleError, NumericLocale, current_numeric_locale};
