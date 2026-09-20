//! Native error categories corresponding to `qlib.utils.exceptions`.

use std::{error::Error, fmt};

/// Shared category implemented by `QlibException` and its two upstream subclasses.
pub trait QlibError: Error {
    /// Return the optional single string argument used by current Qlib call sites.
    fn message(&self) -> Option<&str>;
}

/// Base exception for Qlib-specific failures.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct QlibException {
    message: Option<String>,
}

/// Re-initialization was attempted while starting an experiment.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RecorderInitializationError {
    message: Option<String>,
}

/// A recorder could not load an object.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoadObjectError {
    message: Option<String>,
}

/// An experiment with the requested identity already exists.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExpAlreadyExistError {
    message: Option<String>,
}

macro_rules! impl_exception {
    ($name:ident) => {
        impl $name {
            /// Construct the exception with the single string argument used upstream.
            pub fn new(message: impl Into<String>) -> Self {
                Self {
                    message: Some(message.into()),
                }
            }

            /// Return the optional string argument; `None` represents empty Python `args`.
            #[must_use]
            pub fn message(&self) -> Option<&str> {
                self.message.as_deref()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(self.message.as_deref().unwrap_or_default())
            }
        }

        impl Error for $name {}
    };
}

impl_exception!(QlibException);
impl_exception!(RecorderInitializationError);
impl_exception!(LoadObjectError);
impl_exception!(ExpAlreadyExistError);

impl QlibError for QlibException {
    fn message(&self) -> Option<&str> {
        self.message()
    }
}

impl QlibError for RecorderInitializationError {
    fn message(&self) -> Option<&str> {
        self.message()
    }
}

impl QlibError for LoadObjectError {
    fn message(&self) -> Option<&str> {
        self.message()
    }
}
