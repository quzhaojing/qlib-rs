//! String-calendar sequence access and mutation with Python slice boundaries.

use std::io::Write;

use ndarray::{ArrayView1, Axis};
use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive, Zero};

use crate::{
    CalendarFileValues, CalendarLoadError, CalendarTextArray, CalendarWriteMode,
    FileCalendarStorage,
};

/// Unbounded integer slice fields; omitted values are different from explicit -1.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CalendarSliceSpec {
    pub start: Option<BigInt>,
    pub stop: Option<BigInt>,
    pub step: Option<BigInt>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalendarSelection {
    Index(BigInt),
    Slice(CalendarSliceSpec),
}

/// A scalar lookup and a slice retain different result shapes, even for one row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalendarSelectionValue {
    One(String),
    Many(Vec<String>),
}

/// Strings assigned to slices are iterated by Unicode scalar, while a sequence
/// assigned to an integer position becomes a nested row before array conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalendarAssignment {
    Text(String),
    Sequence(Vec<String>),
}

struct SlicePlan {
    start: BigInt,
    stop: BigInt,
    step: BigInt,
    indices: Vec<usize>,
}

fn adjusted_bound(value: &BigInt, length: &BigInt, lower: &BigInt, upper: &BigInt) -> BigInt {
    let adjusted = if value.is_negative() {
        value + length
    } else {
        value.clone()
    };
    adjusted.clamp(lower.clone(), upper.clone())
}

fn slice_plan(length: usize, spec: &CalendarSliceSpec) -> Result<SlicePlan, CalendarLoadError> {
    let step = spec.step.clone().unwrap_or_else(BigInt::one);
    if step.is_zero() {
        return Err(CalendarLoadError::Value("slice step cannot be zero".into()));
    }
    let backwards = step.is_negative();
    let length = BigInt::from(length);
    let lower = if backwards {
        -BigInt::one()
    } else {
        BigInt::zero()
    };
    let upper = &length + &lower;
    let start = spec.start.as_ref().map_or_else(
        || {
            if backwards {
                upper.clone()
            } else {
                lower.clone()
            }
        },
        |value| adjusted_bound(value, &length, &lower, &upper),
    );
    let end = spec.stop.as_ref().map_or_else(
        || {
            if backwards {
                lower.clone()
            } else {
                upper.clone()
            }
        },
        |value| adjusted_bound(value, &length, &lower, &upper),
    );
    let mut indices = Vec::new();
    let mut current = start.clone();
    while if backwards {
        current > end
    } else {
        current < end
    } {
        indices.push(
            current
                .to_usize()
                .expect("selected index is within the input length"),
        );
        current += &step;
    }
    Ok(SlicePlan {
        start,
        stop: end,
        step,
        indices,
    })
}

fn list_index(length: usize, index: &BigInt) -> Result<usize, CalendarLoadError> {
    let adjusted = if index.is_negative() {
        index + BigInt::from(length)
    } else {
        index.clone()
    };
    adjusted
        .to_usize()
        .filter(|&index| index < length)
        .ok_or_else(|| CalendarLoadError::Other("list index out of range".into()))
}

fn select_rows(rows: &[String], indices: &[usize]) -> Vec<String> {
    ArrayView1::from(rows)
        .select(Axis(0), indices)
        .into_iter()
        .collect()
}

fn without_indices(rows: &[String], indices: &[usize]) -> Vec<String> {
    let mut keep = vec![true; rows.len()];
    for &index in indices {
        keep[index] = false;
    }
    let indices: Vec<_> = keep
        .into_iter()
        .enumerate()
        .filter_map(|(i, keep)| keep.then_some(i))
        .collect();
    select_rows(rows, &indices)
}

fn find_index(rows: &[String], value: &str) -> Result<usize, CalendarLoadError> {
    rows.iter()
        .position(|row| row == value)
        .ok_or_else(|| CalendarLoadError::Value(format!("calendar value not in list: {value}")))
}

// A list containing both a nested sequence and scalar strings is rejected by
// np.asarray inside savetxt, AFTER wb has truncated the destination.
struct RaggedCalendarRows;

impl CalendarFileValues for RaggedCalendarRows {
    fn write_values(&self, _output: &mut dyn Write) -> Result<(), CalendarLoadError> {
        Err(CalendarLoadError::Value(
            "setting an array element with a sequence: inhomogeneous shape".into(),
        ))
    }
}

impl<P: crate::CalendarPathProvider, R: crate::CalendarResampling> FileCalendarStorage<P, R> {
    /// Enforce the source's existence gate without loading or creating a file.
    ///
    /// # Errors
    /// Returns selection/root errors or Value when the selected file is absent.
    pub fn check(&mut self) -> Result<(), CalendarLoadError> {
        crate::file_calendar_backend::check_calendar_path(&self.uri()?)
    }

    /// Find the first equal raw string, bypassing the shared data cache.
    ///
    /// # Errors
    /// Returns acquisition failures or Value for a missing string.
    pub fn index(&mut self, value: &str) -> Result<usize, CalendarLoadError> {
        self.check()?;
        find_index(&self.read_calendar()?, value)
    }

    /// Fetch a raw scalar or slice, retaining negative/omitted/large boundaries.
    ///
    /// # Errors
    /// Returns acquisition failures, Other for an invalid scalar index, or Value
    /// for a zero slice step. Reading a missing file never creates it here.
    pub fn get_item(
        &mut self,
        selection: &CalendarSelection,
    ) -> Result<CalendarSelectionValue, CalendarLoadError> {
        self.check()?;
        let rows = self.read_calendar()?;
        match selection {
            CalendarSelection::Index(index) => Ok(CalendarSelectionValue::One(
                rows[list_index(rows.len(), index)?].clone(),
            )),
            CalendarSelection::Slice(spec) => Ok(CalendarSelectionValue::Many(select_rows(
                &rows,
                &slice_plan(rows.len(), spec)?.indices,
            ))),
        }
    }

    /// Delete raw positions, then rewrite without invalidating the shared cache.
    ///
    /// # Errors
    /// Preserves acquisition/write errors and index/zero-step errors before opening
    /// the destination for writing.
    pub fn delete_item(&mut self, selection: &CalendarSelection) -> Result<(), CalendarLoadError> {
        self.check()?;
        let rows = self.read_calendar()?;
        let indices = match selection {
            CalendarSelection::Index(index) => vec![list_index(rows.len(), index)?],
            CalendarSelection::Slice(spec) => slice_plan(rows.len(), spec)?.indices,
        };
        self.replace_rows(without_indices(&rows, &indices))
    }

    /// Remove the first equal string. Preserve the source's index/read/read order
    /// instead of pinning an earlier snapshot across potentially changing files.
    ///
    /// # Errors
    /// Returns acquisition/write failures, Value for missing values, or Other if
    /// the subsequent read is too short for the earlier index.
    pub fn remove(&mut self, value: &str) -> Result<(), CalendarLoadError> {
        self.check()?;
        let index = self.index(value)?;
        let rows = self.read_calendar()?;
        let index = list_index(rows.len(), &BigInt::from(index))?;
        self.replace_rows(without_indices(&rows, &[index]))
    }

    /// Assign a raw scalar or slice using list semantics, then the source array
    /// writer. Contiguous slices may resize; extended slices require equal length.
    /// A missing file can be created by the initial raw read before index validation.
    ///
    /// # Errors
    /// Preserves acquisition/write failures, Other for scalar bounds, and Value
    /// for zero steps, mismatched extended slices or ragged post-assignment arrays.
    ///
    /// # Panics
    /// Panics if an internal normalized forward bound cannot fit in `usize`.
    /// Bounds are clamped to the input length, so valid inputs cannot cause this.
    pub fn set_item(
        &mut self,
        selection: &CalendarSelection,
        assignment: &CalendarAssignment,
    ) -> Result<(), CalendarLoadError> {
        let mut rows = self.read_calendar()?;
        match selection {
            CalendarSelection::Index(index) => {
                let index = list_index(rows.len(), index)?;
                match assignment {
                    CalendarAssignment::Text(value) => rows[index].clone_from(value),
                    CalendarAssignment::Sequence(values) => {
                        if rows.len() != 1 {
                            return self
                                .write_calendar(&RaggedCalendarRows, CalendarWriteMode::Overwrite);
                        }
                        return self.write_calendar(
                            &CalendarTextArray {
                                shape: vec![1, values.len()],
                                values: values.clone(),
                            },
                            CalendarWriteMode::Overwrite,
                        );
                    }
                }
            }
            CalendarSelection::Slice(spec) => {
                let plan = slice_plan(rows.len(), spec)?;
                let values = match assignment {
                    CalendarAssignment::Text(text) => {
                        text.chars().map(|ch| ch.to_string()).collect()
                    }
                    CalendarAssignment::Sequence(values) => values.clone(),
                };
                if plan.step.is_one() {
                    let start = plan
                        .start
                        .to_usize()
                        .expect("forward slice bound is nonnegative");
                    let stop = plan
                        .stop
                        .max(plan.start)
                        .to_usize()
                        .expect("forward slice bound is nonnegative");
                    rows.splice(start..stop, values);
                } else {
                    if values.len() != plan.indices.len() {
                        return Err(CalendarLoadError::Value(format!(
                            "attempt to assign sequence of size {} to extended slice of size {}",
                            values.len(),
                            plan.indices.len()
                        )));
                    }
                    for (index, value) in plan.indices.into_iter().zip(values) {
                        rows[index] = value;
                    }
                }
            }
        }
        self.replace_rows(rows)
    }

    fn replace_rows(&mut self, rows: Vec<String>) -> Result<(), CalendarLoadError> {
        self.write_calendar(
            &CalendarTextArray {
                shape: vec![rows.len()],
                values: rows,
            },
            CalendarWriteMode::Overwrite,
        )
    }
}
