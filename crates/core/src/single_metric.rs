use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use arrow_arith::numeric::{add, div, mul, sub};
use arrow_array::{Array, ArrayRef, BooleanArray, Float64Array, StringArray};
use arrow_cast::{CastOptions, cast_with_options};
use arrow_ord::cmp::{eq, gt, lt};
use arrow_schema::DataType;
use thiserror::Error;

/// Read-only storage boundary for one stock-indexed metric.
///
/// Implementations may use Arrow memory, an IPC snapshot, or a remote adapter. Qlib's
/// alignment and arithmetic semantics remain framework-owned and consume this interface.
pub trait SingleMetric: Send + Sync {
    fn index(&self) -> &StringArray;
    fn values(&self) -> &dyn Array;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetricBinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Equal,
    Greater,
    Less,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MetricValue {
    Number(f64),
    Boolean(bool),
    Missing,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MetricReplacement {
    pub from: MetricValue,
    pub to: MetricValue,
}

#[derive(Debug, Error)]
pub enum SingleMetricError {
    #[error("metric index and values must have equal lengths, got {index} and {values}")]
    LengthMismatch { index: usize, values: usize },
    #[error("metric stock identifiers cannot be null")]
    NullIndex,
    #[error("duplicate metric stock identifier: {0}")]
    DuplicateIndex(String),
    #[error("metric values must be numeric or Boolean, got {0}")]
    UnsupportedDataType(DataType),
    #[error("metric comparisons require identically labelled indices")]
    ComparisonIndexMismatch,
    #[error("a metric transform cannot mix numeric and Boolean results")]
    MixedTransformTypes,
    #[error("metric transform failed: {0}")]
    Transform(String),
}

#[derive(Clone, Debug)]
pub struct PandasSingleMetric {
    index: StringArray,
    values: ArrayRef,
}

impl PandasSingleMetric {
    #[must_use]
    pub fn empty() -> Self {
        Self {
            index: StringArray::from(Vec::<String>::new()),
            values: Arc::new(Float64Array::from(Vec::<Option<f64>>::new())),
        }
    }

    /// Creates a validated Arrow-backed metric.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid indices, unsupported value types, or Arrow cast failures.
    pub fn try_new(index: StringArray, values: ArrayRef) -> Result<Self, SingleMetricError> {
        validate_index(&index, values.len())?;
        let values = normalize_values(values)?;
        Ok(Self { index, values })
    }

    /// Creates a Float64 metric from stock/value pairs.
    ///
    /// # Errors
    ///
    /// Returns an error when stock identifiers are duplicated.
    pub fn from_f64<I, S>(values: I) -> Result<Self, SingleMetricError>
    where
        I: IntoIterator<Item = (S, Option<f64>)>,
        S: Into<String>,
    {
        let (index, values): (Vec<_>, Vec<_>) = values
            .into_iter()
            .map(|(key, value)| (key.into(), value))
            .unzip();
        Self::try_new(
            StringArray::from(index),
            Arc::new(Float64Array::from(values)),
        )
    }

    #[must_use]
    pub fn values_ref(&self) -> &ArrayRef {
        &self.values
    }

    /// Returns the `SingleData`-compatible Float64 representation.
    #[must_use]
    pub fn to_f64(&self) -> Self {
        Self {
            index: self.index.clone(),
            values: as_f64(self.values()),
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.index.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    #[must_use]
    pub fn count(&self) -> usize {
        metric_values(self)
            .into_iter()
            .filter(|value| !matches!(value, MetricValue::Missing))
            .count()
    }

    pub fn sum(&self) -> f64 {
        metric_values(self)
            .into_iter()
            .filter_map(numeric_value)
            .sum()
    }

    pub fn mean(&self) -> f64 {
        let (sum, count) = metric_values(self)
            .into_iter()
            .filter_map(numeric_value)
            .fold((0.0, 0.0), |(sum, count), value| (sum + value, count + 1.0));
        sum / count
    }

    /// Applies a scalar operation with this metric on the left.
    ///
    /// # Errors
    ///
    /// Returns an error if Arrow cannot execute the requested numeric operation.
    pub fn scalar(
        &self,
        operation: MetricBinaryOp,
        scalar: f64,
    ) -> Result<Self, SingleMetricError> {
        Ok(self.scalar_ordered(operation, scalar, false))
    }

    /// Applies a scalar operation with the scalar on the left.
    ///
    /// # Errors
    ///
    /// Returns an error if Arrow cannot execute the requested numeric operation.
    pub fn reverse_scalar(
        &self,
        operation: MetricBinaryOp,
        scalar: f64,
    ) -> Result<Self, SingleMetricError> {
        Ok(self.scalar_ordered(operation, scalar, true))
    }

    /// Applies an index-aware operation to two metrics.
    ///
    /// # Errors
    ///
    /// Returns an error for differently labelled comparisons or Arrow kernel failures.
    pub fn binary(
        &self,
        operation: MetricBinaryOp,
        other: &dyn SingleMetric,
    ) -> Result<Self, SingleMetricError> {
        if matches!(
            operation,
            MetricBinaryOp::Equal | MetricBinaryOp::Greater | MetricBinaryOp::Less
        ) {
            if !same_index(self.index(), other.index()) {
                return Err(SingleMetricError::ComparisonIndexMismatch);
            }
            let lhs = as_f64(self.values());
            let rhs = as_f64(other.values());
            return Ok(Self {
                index: self.index.clone(),
                values: compute(operation, &lhs.as_ref(), &rhs.as_ref()),
            });
        }

        let (index, lhs, rhs) = align_numeric(self, other);
        Ok(Self {
            index,
            values: compute(operation, &lhs, &rhs),
        })
    }

    /// Adds aligned metrics, optionally filling one-sided missing values.
    ///
    /// # Errors
    ///
    /// Returns an error if Arrow cannot add the aligned values.
    pub fn add_filled(
        &self,
        other: &dyn SingleMetric,
        fill_value: Option<f64>,
    ) -> Result<Self, SingleMetricError> {
        let (index, lhs, rhs) = align_numeric(self, other);
        let (lhs, rhs) = match fill_value {
            Some(fill) => fill_one_sided_missing(&lhs, &rhs, fill),
            None => (lhs, rhs),
        };
        Ok(Self {
            index,
            values: compute(MetricBinaryOp::Add, &lhs, &rhs),
        })
    }

    /// Computes element-wise absolute values.
    ///
    /// # Errors
    ///
    /// Returns an error if rebuilding the typed metric fails.
    pub fn abs(&self) -> Result<Self, SingleMetricError> {
        let values = metric_values(self)
            .into_iter()
            .map(|value| match value {
                MetricValue::Number(value) => MetricValue::Number(value.abs()),
                other => other,
            })
            .collect();
        metric_from_values(self.index.clone(), values)
    }

    /// Replaces matching values in declaration order.
    ///
    /// # Errors
    ///
    /// Returns an error if replacements mix numeric and Boolean output values.
    pub fn replace(&self, replacements: &[MetricReplacement]) -> Result<Self, SingleMetricError> {
        let values = metric_values(self)
            .into_iter()
            .map(|value| {
                replacements
                    .iter()
                    .find(|replacement| metric_value_eq(value, replacement.from))
                    .map_or(value, |replacement| replacement.to)
            })
            .collect();
        metric_from_values(self.index.clone(), values)
    }

    /// Applies a fallible transform to every metric value.
    ///
    /// # Errors
    ///
    /// Returns the callback error or an error for mixed numeric and Boolean output values.
    pub fn apply(
        &self,
        transform: &dyn Fn(MetricValue) -> Result<MetricValue, String>,
    ) -> Result<Self, SingleMetricError> {
        let values = metric_values(self)
            .into_iter()
            .map(transform)
            .collect::<Result<Vec<_>, _>>()
            .map_err(SingleMetricError::Transform)?;
        metric_from_values(self.index.clone(), values)
    }

    /// Reorders/subsets the metric and fills labels absent from the source.
    ///
    /// # Errors
    ///
    /// Returns an error when the requested index contains duplicates.
    pub fn reindex<S: AsRef<str>>(
        &self,
        index: &[S],
        fill_value: Option<f64>,
    ) -> Result<Self, SingleMetricError> {
        let positions = index_positions(self.index());
        let source = metric_values(self);
        let values = index
            .iter()
            .map(|key| {
                positions.get(key.as_ref()).map_or_else(
                    || fill_value.map_or(MetricValue::Missing, MetricValue::Number),
                    |row| source[*row],
                )
            })
            .collect();
        metric_from_values(
            StringArray::from(
                index
                    .iter()
                    .map(|key| key.as_ref().to_owned())
                    .collect::<Vec<_>>(),
            ),
            values,
        )
    }

    fn scalar_ordered(&self, operation: MetricBinaryOp, scalar: f64, reverse: bool) -> Self {
        let values = as_f64(self.values());
        let scalar = Float64Array::new_scalar(scalar);
        let output = if matches!(
            operation,
            MetricBinaryOp::Equal | MetricBinaryOp::Greater | MetricBinaryOp::Less
        ) {
            if reverse {
                compute(operation, &scalar, &values.as_ref())
            } else {
                compute(operation, &values.as_ref(), &scalar)
            }
        } else if reverse {
            compute(operation, &scalar, &values.as_ref())
        } else {
            compute(operation, &values.as_ref(), &scalar)
        };
        Self {
            index: self.index.clone(),
            values: output,
        }
    }
}

impl Default for PandasSingleMetric {
    fn default() -> Self {
        Self::empty()
    }
}

impl SingleMetric for PandasSingleMetric {
    fn index(&self) -> &StringArray {
        &self.index
    }

    fn values(&self) -> &dyn Array {
        self.values.as_ref()
    }
}

fn validate_index(index: &StringArray, values_len: usize) -> Result<(), SingleMetricError> {
    if index.len() != values_len {
        return Err(SingleMetricError::LengthMismatch {
            index: index.len(),
            values: values_len,
        });
    }
    if index.null_count() != 0 {
        return Err(SingleMetricError::NullIndex);
    }
    let mut positions = HashSet::with_capacity(index.len());
    for key in index.iter().flatten() {
        if !positions.insert(key) {
            return Err(SingleMetricError::DuplicateIndex(key.to_owned()));
        }
    }
    Ok(())
}

fn normalize_values(values: ArrayRef) -> Result<ArrayRef, SingleMetricError> {
    match values.data_type() {
        DataType::Boolean => Ok(values),
        data_type if data_type.is_numeric() => {
            let cast = cast_with_options(
                values.as_ref(),
                &DataType::Float64,
                &CastOptions {
                    safe: false,
                    ..CastOptions::default()
                },
            )
            .expect("every Arrow numeric type casts to Float64 with unsafe options");
            let floats = cast
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("Float64 cast returns Float64Array");
            Ok(Arc::new(Float64Array::from_iter(floats.iter().map(
                |value| match value {
                    Some(value) if !value.is_nan() => Some(value),
                    _ => None,
                },
            ))))
        }
        data_type => Err(SingleMetricError::UnsupportedDataType(data_type.clone())),
    }
}

fn as_f64(values: &dyn Array) -> ArrayRef {
    normalize_values(
        cast_with_options(
            values,
            &DataType::Float64,
            &CastOptions {
                safe: false,
                ..CastOptions::default()
            },
        )
        .expect("Boolean and Float64 metric values cast to Float64"),
    )
    .expect("SingleMetric implementations expose only Boolean or Float64 values")
}

fn metric_values(metric: &dyn SingleMetric) -> Vec<MetricValue> {
    if let Some(values) = metric.values().as_any().downcast_ref::<BooleanArray>() {
        return values
            .iter()
            .map(|value| value.map_or(MetricValue::Missing, MetricValue::Boolean))
            .collect();
    }
    let values = metric
        .values()
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("SingleMetric invariant permits Float64 or Boolean");
    values
        .iter()
        .map(|value| value.map_or(MetricValue::Missing, MetricValue::Number))
        .collect()
}

fn numeric_value(value: MetricValue) -> Option<f64> {
    match value {
        MetricValue::Number(value) => Some(value),
        MetricValue::Boolean(value) => Some(f64::from(value)),
        MetricValue::Missing => None,
    }
}

fn metric_from_values(
    index: StringArray,
    values: Vec<MetricValue>,
) -> Result<PandasSingleMetric, SingleMetricError> {
    let saw_number = values
        .iter()
        .any(|value| matches!(value, MetricValue::Number(_)));
    let saw_boolean = values
        .iter()
        .any(|value| matches!(value, MetricValue::Boolean(_)));
    if saw_number && saw_boolean {
        return Err(SingleMetricError::MixedTransformTypes);
    }
    let values: ArrayRef = if saw_boolean {
        Arc::new(BooleanArray::from_iter(values.into_iter().map(|value| {
            if let MetricValue::Boolean(value) = value {
                Some(value)
            } else {
                None
            }
        })))
    } else {
        Arc::new(Float64Array::from_iter(values.into_iter().map(|value| {
            if let MetricValue::Number(value) = value {
                Some(value)
            } else {
                None
            }
        })))
    };
    PandasSingleMetric::try_new(index, values)
}

fn same_index(lhs: &StringArray, rhs: &StringArray) -> bool {
    lhs.len() == rhs.len() && lhs.iter().zip(rhs.iter()).all(|(lhs, rhs)| lhs == rhs)
}

fn index_positions(index: &StringArray) -> HashMap<&str, usize> {
    index
        .iter()
        .flatten()
        .enumerate()
        .map(|(row, key)| (key, row))
        .collect()
}

fn align_numeric(
    lhs: &dyn SingleMetric,
    rhs: &dyn SingleMetric,
) -> (StringArray, Float64Array, Float64Array) {
    let lhs_positions = index_positions(lhs.index());
    let rhs_positions = index_positions(rhs.index());
    let mut index: Vec<_> = if same_index(lhs.index(), rhs.index()) {
        lhs.index().iter().flatten().map(str::to_owned).collect()
    } else {
        lhs_positions
            .keys()
            .chain(rhs_positions.keys())
            .map(|key| (*key).to_owned())
            .collect()
    };
    if !same_index(lhs.index(), rhs.index()) {
        index.sort_unstable();
        index.dedup();
    }
    let lhs_values = metric_values(lhs);
    let rhs_values = metric_values(rhs);
    let lhs = Float64Array::from_iter(index.iter().map(|key| {
        lhs_positions
            .get(key.as_str())
            .and_then(|row| numeric_value(lhs_values[*row]))
    }));
    let rhs = Float64Array::from_iter(index.iter().map(|key| {
        rhs_positions
            .get(key.as_str())
            .and_then(|row| numeric_value(rhs_values[*row]))
    }));
    (StringArray::from(index), lhs, rhs)
}

fn compute(
    operation: MetricBinaryOp,
    lhs: &dyn arrow_array::Datum,
    rhs: &dyn arrow_array::Datum,
) -> ArrayRef {
    let output = match operation {
        MetricBinaryOp::Add => add(lhs, rhs),
        MetricBinaryOp::Subtract => sub(lhs, rhs),
        MetricBinaryOp::Multiply => mul(lhs, rhs),
        MetricBinaryOp::Divide => div(lhs, rhs),
        MetricBinaryOp::Equal => {
            return comparison_output(&eq(lhs, rhs).expect("Float64 equality is supported"));
        }
        MetricBinaryOp::Greater => {
            return comparison_output(&gt(lhs, rhs).expect("Float64 ordering is supported"));
        }
        MetricBinaryOp::Less => {
            return comparison_output(&lt(lhs, rhs).expect("Float64 ordering is supported"));
        }
    };
    output.expect("Float64 arithmetic is supported")
}

fn comparison_output(compared: &BooleanArray) -> ArrayRef {
    Arc::new(BooleanArray::from_iter(
        compared.iter().map(|value| Some(value.unwrap_or(false))),
    ))
}

fn fill_one_sided_missing(
    lhs: &Float64Array,
    rhs: &Float64Array,
    fill: f64,
) -> (Float64Array, Float64Array) {
    let lhs_filled = Float64Array::from_iter(
        lhs.iter()
            .zip(rhs.iter())
            .map(|(lhs, rhs)| lhs.or_else(|| rhs.map(|_| fill))),
    );
    let rhs_filled = Float64Array::from_iter(
        lhs.iter()
            .zip(rhs.iter())
            .map(|(lhs, rhs)| rhs.or_else(|| lhs.map(|_| fill))),
    );
    (lhs_filled, rhs_filled)
}

fn metric_value_eq(lhs: MetricValue, rhs: MetricValue) -> bool {
    match (lhs, rhs) {
        (MetricValue::Missing, MetricValue::Missing) => true,
        (MetricValue::Number(lhs), MetricValue::Number(rhs)) => lhs == rhs,
        (MetricValue::Boolean(lhs), MetricValue::Boolean(rhs)) => lhs == rhs,
        _ => false,
    }
}
