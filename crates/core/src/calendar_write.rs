//! Binary UTF-8 calendar persistence from `FileCalendarStorage._write_calendar`.

use std::{
    fs::OpenOptions,
    io::{BufWriter, Write},
    path::Path,
};

use ndarray::{ArrayView, Axis, IxDyn};

use crate::CalendarLoadError;

/// The two binary modes used by the source calendar mutation methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalendarWriteMode {
    Overwrite,
    Append,
}

/// Values are converted only AFTER the destination is opened. An implementation
/// may fail after producing a prefix; neither conversion nor writes are atomic.
/// This is an encoding boundary, not a filesystem or cache implementation.
pub trait CalendarFileValues {
    /// Emit the source's UTF-8 representation, preserving any partial effects.
    ///
    /// # Errors
    /// Value failures retain Value classification; I/O failures are Other.
    fn write_values(&self, output: &mut dyn Write) -> Result<(), CalendarLoadError>;
}

/// Owned string-array input to `NumPy`'s `%s` writer. Shape validation is delayed
/// until writing, like `np.asarray` inside `np.savetxt`. One-dimensional values
/// write one line each; two-dimensional rows use a single space as delimiter.
/// No quoting, escaping, whitespace trimming, BOM or newline translation is applied.
/// Trailing NUL padding is removed when extracting each `NumPy` Unicode scalar;
/// embedded NULs remain. Object-array/custom scalar codecs use the plugin boundary.
/// Scalar and higher-dimensional arrays are rejected, not silently flattened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalendarTextArray {
    pub shape: Vec<usize>,
    pub values: Vec<String>,
}

impl CalendarFileValues for CalendarTextArray {
    fn write_values(&self, output: &mut dyn Write) -> Result<(), CalendarLoadError> {
        let array = ArrayView::from_shape(IxDyn(&self.shape), self.values.as_slice())
            .map_err(|error| CalendarLoadError::Value(error.to_string()))?;
        if !(1..=2).contains(&array.ndim()) {
            return Err(CalendarLoadError::Value(format!(
                "Expected 1D or 2D array, got {}D array instead",
                array.ndim()
            )));
        }
        for row in array.axis_iter(Axis(0)) {
            for (index, value) in row.iter().enumerate() {
                if index != 0 {
                    output.write_all(b" ").map_err(|error| io_error(&error))?;
                }
                output
                    .write_all(value.trim_end_matches('\0').as_bytes())
                    .map_err(|error| io_error(&error))?;
            }
            output.write_all(b"\n").map_err(|error| io_error(&error))?;
        }
        Ok(())
    }
}

/// Open before conversion, then overwrite or append UTF-8 calendar bytes.
/// Parent directories are never created. Existing shared read caches are not
/// invalidated: the upstream mutation methods leave those entries untouched.
/// Flush buffered partial output even after an encoder failure; a flush failure
/// replaces that failure, as context-manager close does in the source.
///
/// # Errors
/// Returns open/write/flush failures as Other, retaining encoder error classes.
pub fn write_calendar_file(
    path: &Path,
    values: &dyn CalendarFileValues,
    mode: CalendarWriteMode,
) -> Result<(), CalendarLoadError> {
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(mode == CalendarWriteMode::Overwrite)
        .append(mode == CalendarWriteMode::Append)
        .open(path)
        .map_err(|error| io_error(&error))?;
    let mut output = BufWriter::new(file);
    let result = write_and_flush(values, &mut output);
    // A failed flush must not be retried implicitly by BufWriter::drop. The
    // source closes its stream after that failure instead of replaying bytes.
    let (_file, _unwritten) = output.into_parts();
    result
}

fn write_and_flush(
    values: &dyn CalendarFileValues,
    output: &mut dyn Write,
) -> Result<(), CalendarLoadError> {
    let result = values.write_values(output);
    output.flush().map_err(|error| io_error(&error))?;
    result
}

fn io_error(error: &std::io::Error) -> CalendarLoadError {
    CalendarLoadError::Other(error.to_string())
}

#[cfg(test)]
#[path = "../tests/support/calendar_write_failures.rs"]
mod tests;
