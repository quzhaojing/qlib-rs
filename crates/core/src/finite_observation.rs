//! NumPy-compatible invalid observation sentinels used by finite RL environments.

use half::f16;
use indexmap::IndexMap;
use ndarray::ArrayD;
use thiserror::Error;

use crate::{EnvironmentObservationSpace, EnvironmentPluginError};

#[derive(Clone, Debug, PartialEq)]
pub enum FiniteObservation {
    Float16(ArrayD<f16>),
    Float32(ArrayD<f32>),
    Float64(ArrayD<f64>),
    Int8(ArrayD<i8>),
    Int16(ArrayD<i16>),
    Int32(ArrayD<i32>),
    Int64(ArrayD<i64>),
    UInt8(ArrayD<u8>),
    UInt16(ArrayD<u16>),
    UInt32(ArrayD<u32>),
    UInt64(ArrayD<u64>),
    Bool(ArrayD<bool>),
    Map(IndexMap<String, Self>),
    List(Vec<Self>),
    Tuple(Vec<Self>),
    UnsupportedArray { dtype: String },
    Opaque { description: String },
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum FiniteObservationError {
    #[error("unsupported NumPy dtype for invalid sentinel: {0}")]
    UnsupportedDtype(String),
    #[error("unsupported value to fill with invalid sentinel: {0}")]
    UnsupportedValue(String),
}

macro_rules! filled_array {
    ($array:expr, $value:expr) => {
        ArrayD::from_elem($array.raw_dim(), $value)
    };
}

/// Recursively creates the exhausted-environment sentinel while preserving shape and dtype.
///
/// # Errors
/// Returns an error for bool/unsupported arrays and opaque values, matching `NumPy` `iinfo` and
/// Qlib's final unsupported-value branch.
pub fn fill_invalid(
    observation: &FiniteObservation,
) -> Result<FiniteObservation, FiniteObservationError> {
    match observation {
        FiniteObservation::Float16(array) => {
            Ok(FiniteObservation::Float16(filled_array!(array, f16::NAN)))
        }
        FiniteObservation::Float32(array) => {
            Ok(FiniteObservation::Float32(filled_array!(array, f32::NAN)))
        }
        FiniteObservation::Float64(array) => {
            Ok(FiniteObservation::Float64(filled_array!(array, f64::NAN)))
        }
        FiniteObservation::Int8(array) => {
            Ok(FiniteObservation::Int8(filled_array!(array, i8::MAX)))
        }
        FiniteObservation::Int16(array) => {
            Ok(FiniteObservation::Int16(filled_array!(array, i16::MAX)))
        }
        FiniteObservation::Int32(array) => {
            Ok(FiniteObservation::Int32(filled_array!(array, i32::MAX)))
        }
        FiniteObservation::Int64(array) => {
            Ok(FiniteObservation::Int64(filled_array!(array, i64::MAX)))
        }
        FiniteObservation::UInt8(array) => {
            Ok(FiniteObservation::UInt8(filled_array!(array, u8::MAX)))
        }
        FiniteObservation::UInt16(array) => {
            Ok(FiniteObservation::UInt16(filled_array!(array, u16::MAX)))
        }
        FiniteObservation::UInt32(array) => {
            Ok(FiniteObservation::UInt32(filled_array!(array, u32::MAX)))
        }
        FiniteObservation::UInt64(array) => {
            Ok(FiniteObservation::UInt64(filled_array!(array, u64::MAX)))
        }
        FiniteObservation::Bool(_) => {
            Err(FiniteObservationError::UnsupportedDtype("bool".to_owned()))
        }
        FiniteObservation::Map(values) => values
            .iter()
            .map(|(name, value)| Ok((name.clone(), fill_invalid(value)?)))
            .collect::<Result<IndexMap<_, _>, _>>()
            .map(FiniteObservation::Map),
        FiniteObservation::List(values) => values
            .iter()
            .map(fill_invalid)
            .collect::<Result<Vec<_>, _>>()
            .map(FiniteObservation::List),
        FiniteObservation::Tuple(values) => values
            .iter()
            .map(fill_invalid)
            .collect::<Result<Vec<_>, _>>()
            .map(FiniteObservation::Tuple),
        FiniteObservation::UnsupportedArray { dtype } => {
            Err(FiniteObservationError::UnsupportedDtype(dtype.clone()))
        }
        FiniteObservation::Opaque { description } => Err(FiniteObservationError::UnsupportedValue(
            description.clone(),
        )),
    }
}

/// Checks whether every leaf contains its dtype-specific invalid sentinel.
///
/// # Errors
/// Returns an error when `NumPy` `iinfo(dtype)` would reject an array dtype.
pub fn is_invalid(observation: &FiniteObservation) -> Result<bool, FiniteObservationError> {
    match observation {
        FiniteObservation::Float16(array) => Ok(array.iter().all(|value| value.is_nan())),
        FiniteObservation::Float32(array) => Ok(array.iter().all(|value| value.is_nan())),
        FiniteObservation::Float64(array) => Ok(array.iter().all(|value| value.is_nan())),
        FiniteObservation::Int8(array) => Ok(array.iter().all(|value| *value == i8::MAX)),
        FiniteObservation::Int16(array) => Ok(array.iter().all(|value| *value == i16::MAX)),
        FiniteObservation::Int32(array) => Ok(array.iter().all(|value| *value == i32::MAX)),
        FiniteObservation::Int64(array) => Ok(array.iter().all(|value| *value == i64::MAX)),
        FiniteObservation::UInt8(array) => Ok(array.iter().all(|value| *value == u8::MAX)),
        FiniteObservation::UInt16(array) => Ok(array.iter().all(|value| *value == u16::MAX)),
        FiniteObservation::UInt32(array) => Ok(array.iter().all(|value| *value == u32::MAX)),
        FiniteObservation::UInt64(array) => Ok(array.iter().all(|value| *value == u64::MAX)),
        FiniteObservation::Bool(_) => {
            Err(FiniteObservationError::UnsupportedDtype("bool".to_owned()))
        }
        FiniteObservation::Map(values) => {
            for value in values.values() {
                if !is_invalid(value)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        FiniteObservation::List(values) | FiniteObservation::Tuple(values) => {
            for value in values {
                if !is_invalid(value)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        FiniteObservation::UnsupportedArray { dtype } => {
            Err(FiniteObservationError::UnsupportedDtype(dtype.clone()))
        }
        FiniteObservation::Opaque { .. } => Ok(true),
    }
}

/// Alias matching Qlib's public exhausted-observation predicate.
///
/// # Errors
/// Returns unsupported `NumPy` dtype errors from [`is_invalid`].
pub fn check_nan_observation(
    observation: &FiniteObservation,
) -> Result<bool, FiniteObservationError> {
    is_invalid(observation)
}

pub struct SampledFiniteObservationSpace {
    sampler: Box<dyn FnMut() -> Result<FiniteObservation, EnvironmentPluginError> + Send + 'static>,
}

impl SampledFiniteObservationSpace {
    #[must_use]
    pub fn new(
        sampler: impl FnMut() -> Result<FiniteObservation, EnvironmentPluginError> + Send + 'static,
    ) -> Self {
        Self {
            sampler: Box::new(sampler),
        }
    }
}

impl EnvironmentObservationSpace<FiniteObservation> for SampledFiniteObservationSpace {
    fn invalid_observation(&mut self) -> Result<FiniteObservation, EnvironmentPluginError> {
        let sample = (self.sampler)()?;
        fill_invalid(&sample).map_err(|error| EnvironmentPluginError::new(error.to_string()))
    }
}
