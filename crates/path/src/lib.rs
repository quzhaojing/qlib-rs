//! Source-compatible native Windows path queries and non-strict resolution.

#[cfg(windows)]
mod windows;
#[cfg(windows)]
pub use windows::{
    DirectoryAttributes, PathQueryError, directory_attributes, final_path, find_name,
    normalize_case, read_link, symbolic_link_by_open,
};

#[cfg(windows)]
mod reparse;

#[cfg(windows)]
mod lexical;
#[cfg(windows)]
pub use lexical::{is_absolute, normalize, split, split_root};

#[cfg(windows)]
mod joining;
#[cfg(windows)]
pub use joining::join;

#[cfg(windows)]
mod link_chain;
#[cfg(windows)]
pub use link_chain::{LinkOperations, read_link_deep};

#[cfg(windows)]
mod by_name;
#[cfg(windows)]
pub use by_name::symbolic_link_by_name;

#[cfg(windows)]
pub use windows::lstat_attributes;

#[cfg(windows)]
mod classification;
#[cfg(windows)]
pub use classification::{NativeLinkOperations, is_symbolic_link};

#[cfg(windows)]
mod non_strict;
#[cfg(windows)]
pub use non_strict::{FinalPathOperations, NativeFinalPathOperations, final_path_non_strict};

#[cfg(windows)]
pub use windows::current_directory;
#[cfg(windows)]
mod real_path;
#[cfg(windows)]
pub use real_path::{NativeRealPathOperations, RealPathOperations, real_path, real_path_with};
