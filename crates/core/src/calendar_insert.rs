//! Source `NumPy` insertion conversion for string-calendar rows.

use num_bigint::BigInt;
use num_traits::{Signed, ToPrimitive};

use crate::{CalendarLoadError, CalendarTextArray, CalendarWriteMode, FileCalendarStorage};

fn insertion_position(length: usize, index: &BigInt) -> Result<usize, CalendarLoadError> {
    let adjusted = if index.is_negative() {
        index + BigInt::from(length)
    } else {
        index.clone()
    };
    adjusted
        .to_usize()
        .filter(|&position| position <= length)
        .ok_or_else(|| CalendarLoadError::Other("calendar insertion index out of bounds".into()))
}

fn empty_array_value(value: &str) -> Result<String, CalendarLoadError> {
    // CPython's float conversion accepts Unicode decimal digits. The existing
    // RustPython parser supplies decimal grammar/underscores/rounding, but only
    // after this Unicode normalization at its ASCII parsing boundary.
    let normalized: String = value
        .chars()
        .map(|ch| {
            crate::rl_checkpoint_numeric::decimal_digit(ch)
                .map_or(ch, |digit| char::from(b"0123456789"[digit]))
        })
        .collect();
    rustpython_literal::float::parse_str(&normalized)
        .map(rustpython_literal::float::to_string)
        .ok_or_else(|| CalendarLoadError::Value("cannot convert calendar value to float".into()))
}

impl<P: crate::CalendarPathProvider, R: crate::CalendarResampling> FileCalendarStorage<P, R> {
    /// Insert a string using the source array's inferred dtype, not list insertion.
    /// Nonempty arrays truncate the value to the longest existing Unicode row;
    /// empty arrays convert it to a floating-point value before writing.
    /// No cached data is invalidated. A missing file is created by the initial read.
    ///
    /// # Errors
    /// Preserves read/write failures, rejects indices outside `[-len, len]` as
    /// Other, and reports invalid empty-array numeric conversion as Value.
    pub fn insert(&mut self, index: &BigInt, value: &str) -> Result<(), CalendarLoadError> {
        let mut rows = self.read_calendar()?;
        let position = insertion_position(rows.len(), index)?;
        let value = match rows.iter().map(|row| row.chars().count()).max() {
            Some(width) => value.chars().take(width).collect(),
            None => empty_array_value(value)?,
        };
        rows.insert(position, value);
        self.write_calendar(
            &CalendarTextArray {
                shape: vec![rows.len()],
                values: rows,
            },
            CalendarWriteMode::Overwrite,
        )
    }
}
