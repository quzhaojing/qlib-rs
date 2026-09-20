//! Ordered training-vessel metric reduction and diagnostic delivery.

use std::{borrow::Cow, sync::Arc};

use arrow_arith::aggregate::sum;
use arrow_array::{Array, ArrayRef, Float32Array, Float64Array};
use arrow_cast::cast;
use arrow_schema::DataType;
use half::f16;
use indexmap::IndexMap;
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use thiserror::Error;

use crate::{TrainingVesselBinding, TrainingVesselBindingError};

/// Delayed formatting for non-array values, including opaque model-specific diagnostics.
pub trait TrainingMetricDisplay: Send + Sync {
    /// # Errors
    /// Returns a value-specific formatting failure.
    fn render(&self) -> Result<String, String>;
}

#[derive(Clone)]
pub enum TrainingMetricScalar {
    Text(String),
    Float(f64),
    Integer(BigInt),
    Boolean(bool),
    Null,
    Custom(Arc<dyn TrainingMetricDisplay>),
}

#[derive(Clone)]
pub enum TrainingMetricValue {
    Scalar(TrainingMetricScalar),
    /// A numeric ndarray or rectangular numeric list, flattened by its input adapter.
    /// Preserve the source dtype; Boolean/integer inputs accumulate in Float64.
    Numeric(ArrayRef),
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum TrainingMetricReductionError {
    #[error("unsupported training metric dtype: {0}")]
    UnsupportedDtype(DataType),
    #[error("training metric contains Arrow nulls; provide numeric NaNs explicitly")]
    NullValues,
    #[error("training metric reduction plugin failed: {0}")]
    Plugin(String),
}

/// Replacement seam for alternate dtype support or backend-specific reduction behavior.
pub trait TrainingMetricReducer: Send {
    /// # Errors
    /// Returns unsupported dtype, null, or backend-specific failures.
    fn mean(&mut self, values: &dyn Array) -> Result<f64, TrainingMetricReductionError>;
}

#[derive(Debug, Default)]
pub struct ArrowTrainingMetricReducer;

impl TrainingMetricReducer for ArrowTrainingMetricReducer {
    #[allow(clippy::cast_possible_truncation)]
    fn mean(&mut self, values: &dyn Array) -> Result<f64, TrainingMetricReductionError> {
        if !matches!(
            values.data_type(),
            DataType::Boolean
                | DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Float16
                | DataType::Float32
                | DataType::Float64
        ) {
            return Err(TrainingMetricReductionError::UnsupportedDtype(
                values.data_type().clone(),
            ));
        }
        if values.null_count() != 0 {
            return Err(TrainingMetricReductionError::NullValues);
        }
        let count = values
            .len()
            .to_f64()
            .expect("Arrow array length fits a Float64 denominator");
        if matches!(values.data_type(), DataType::Float16 | DataType::Float32) {
            let floats = cast(values, &DataType::Float32).expect("Float16/32 casts to Float32");
            let floats = floats
                .as_any()
                .downcast_ref::<Float32Array>()
                .expect("Float32 cast has Float32 storage");
            let mean = f64::from(sum(floats).unwrap_or(f32::NAN)) / count;
            return Ok(if values.data_type() == &DataType::Float16 {
                f16::from_f64(mean).to_f64()
            } else {
                f64::from(mean as f32)
            });
        }
        let floats =
            cast(values, &DataType::Float64).expect("Boolean/integer/Float64 casts to Float64");
        let floats = floats
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("Float64 cast has Float64 storage");
        Ok(sum(floats).unwrap_or(f64::NAN) / count)
    }
}

pub trait TrainingMetricSink: Send {
    /// # Errors
    /// Returns a diagnostic-delivery failure.
    fn info(&mut self, message: &str) -> Result<(), String>;
}

#[derive(Debug, Default)]
pub struct TracingTrainingMetricSink;

impl TrainingMetricSink for TracingTrainingMetricSink {
    fn info(&mut self, message: &str) -> Result<(), String> {
        tracing::info!("{message}");
        Ok(())
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum TrainingVesselLogError {
    #[error(transparent)]
    Reduction(#[from] TrainingMetricReductionError),
    #[error(transparent)]
    Binding(#[from] TrainingVesselBindingError),
    #[error("training metric formatting failed: {0}")]
    Format(String),
    #[error("training metric delivery failed: {0}")]
    Sink(String),
}

pub struct TrainingVesselLog {
    reducer: Box<dyn TrainingMetricReducer>,
    sink: Box<dyn TrainingMetricSink>,
}

impl Default for TrainingVesselLog {
    fn default() -> Self {
        Self::with_plugins(
            Box::new(ArrowTrainingMetricReducer),
            Box::new(TracingTrainingMetricSink),
        )
    }
}

impl TrainingVesselLog {
    #[must_use]
    pub fn with_plugins(
        reducer: Box<dyn TrainingMetricReducer>,
        sink: Box<dyn TrainingMetricSink>,
    ) -> Self {
        Self { reducer, sink }
    }

    /// Reduces first, then reads the live iteration, formats the value, and delivers one message.
    ///
    /// # Errors
    /// Returns the first reduction, trainer, formatting, or delivery failure.
    pub fn log(
        &mut self,
        binding: &TrainingVesselBinding,
        name: &str,
        value: &TrainingMetricValue,
    ) -> Result<(), TrainingVesselLogError> {
        let scalar = match value {
            TrainingMetricValue::Scalar(value) => Cow::Borrowed(value),
            TrainingMetricValue::Numeric(values) => Cow::Owned(TrainingMetricScalar::Float(
                self.reducer.mean(values.as_ref())?,
            )),
        };
        let iteration = binding.current_iteration()? + 1;
        let text = render_scalar(&scalar)?;
        self.sink
            .info(&format!("[Iter {iteration}] {name} = {text}"))
            .map_err(TrainingVesselLogError::Sink)
    }

    /// # Errors
    /// Stops at the first failure; preceding messages are not rolled back.
    pub fn log_dict(
        &mut self,
        binding: &TrainingVesselBinding,
        values: &IndexMap<String, TrainingMetricValue>,
    ) -> Result<(), TrainingVesselLogError> {
        for (name, value) in values {
            self.log(binding, name, value)?;
        }
        Ok(())
    }
}

pub(crate) fn render_scalar(
    value: &TrainingMetricScalar,
) -> Result<String, TrainingVesselLogError> {
    Ok(match value {
        TrainingMetricScalar::Text(text) => text.clone(),
        TrainingMetricScalar::Float(value) => python_float(*value),
        TrainingMetricScalar::Integer(value) => value.to_string(),
        TrainingMetricScalar::Boolean(value) => if *value { "True" } else { "False" }.to_owned(),
        TrainingMetricScalar::Null => "None".to_owned(),
        TrainingMetricScalar::Custom(value) => {
            value.render().map_err(TrainingVesselLogError::Format)?
        }
    })
}

pub(crate) fn python_float(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_owned();
    }
    // Rust's shortest Debug float rendering has Python's fixed/scientific cutoffs;
    // normalize Python's explicit exponent sign and two-digit minimum exponent.
    let text = format!("{value:?}");
    if let Some((mantissa, exponent)) = text.split_once('e') {
        let exponent: i32 = exponent
            .parse()
            .expect("standard float formatter emits an integer exponent");
        format!("{mantissa}e{exponent:+03}")
    } else {
        text
    }
}

#[cfg(test)]
#[path = "../tests/support/training_vessel_log.rs"]
mod tests;
