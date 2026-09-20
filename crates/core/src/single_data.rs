use std::collections::{BTreeSet, HashMap, HashSet};

use ndarray::Array1;
use thiserror::Error;

use crate::MetricBinaryOp;

pub type DenseTransform = dyn Fn(&[f64]) -> Result<Vec<f64>, String>;

/// Read-only, ABI-neutral boundary for a dense stock-indexed metric.
pub trait DenseMetric: Send + Sync {
    fn index(&self) -> &[String];
    fn values(&self) -> &[f64];
}

#[derive(Debug, Error)]
pub enum SingleDataError {
    #[error("single data index and values must have equal lengths, got {index} and {values}")]
    LengthMismatch { index: usize, values: usize },
    #[error("duplicate single data index: {0}")]
    DuplicateIndex(String),
    #[error("single data arithmetic requires equal index sets")]
    IndexMismatch,
    #[error("single data transform failed: {0}")]
    Transform(String),
    #[error("single data transform returned {actual} values for an index of length {expected}")]
    TransformLengthMismatch { expected: usize, actual: usize },
}

/// NumPy-compatible one-dimensional Float64 data with stable labels.
#[derive(Clone, Debug)]
pub struct SingleData {
    index: Vec<String>,
    positions: HashMap<String, usize>,
    values: Array1<f64>,
}

impl SingleData {
    #[must_use]
    pub fn empty() -> Self {
        Self {
            index: Vec::new(),
            positions: HashMap::new(),
            values: Array1::from_vec(Vec::new()),
        }
    }

    /// Creates validated dense data.
    ///
    /// # Errors
    ///
    /// Returns an error for unequal lengths or duplicate labels.
    pub fn try_new(index: Vec<String>, values: Array1<f64>) -> Result<Self, SingleDataError> {
        if index.len() != values.len() {
            return Err(SingleDataError::LengthMismatch {
                index: index.len(),
                values: values.len(),
            });
        }
        let mut positions = HashMap::with_capacity(index.len());
        for (row, key) in index.iter().enumerate() {
            if positions.insert(key.clone(), row).is_some() {
                return Err(SingleDataError::DuplicateIndex(key.clone()));
            }
        }
        Ok(Self {
            index,
            positions,
            values,
        })
    }

    /// Creates dense data from stock/value pairs, converting missing values to IEEE NaN.
    ///
    /// # Errors
    ///
    /// Returns an error for duplicate stock identifiers.
    pub fn from_f64<I, S>(values: I) -> Result<Self, SingleDataError>
    where
        I: IntoIterator<Item = (S, Option<f64>)>,
        S: Into<String>,
    {
        let (index, values): (Vec<_>, Vec<_>) = values
            .into_iter()
            .map(|(key, value)| (key.into(), value.unwrap_or(f64::NAN)))
            .unzip();
        Self::try_new(index, Array1::from_vec(values))
    }

    /// Broadcasts one scalar over an explicit index.
    ///
    /// # Errors
    ///
    /// Returns an error for duplicate labels.
    pub fn broadcast<S: AsRef<str>>(value: f64, index: &[S]) -> Result<Self, SingleDataError> {
        Self::try_new(
            index.iter().map(|key| key.as_ref().to_owned()).collect(),
            Array1::from_elem(index.len(), value),
        )
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    #[must_use]
    pub fn get(&self, key: &str) -> Option<f64> {
        self.positions.get(key).map(|row| self.values[*row])
    }

    #[must_use]
    pub fn count(&self) -> usize {
        self.values.iter().filter(|value| !value.is_nan()).count()
    }

    #[must_use]
    pub fn sum(&self) -> f64 {
        self.values.iter().filter(|value| !value.is_nan()).sum()
    }

    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn mean(&self) -> f64 {
        let count = self.count();
        if count == 0 {
            f64::NAN
        } else {
            self.sum() / count as f64
        }
    }

    #[must_use]
    pub fn all(&self) -> bool {
        self.values.iter().all(|value| *value != 0.0)
    }

    #[must_use]
    pub fn scalar(&self, operation: MetricBinaryOp, scalar: f64) -> Self {
        self.scalar_ordered(operation, scalar, false)
    }

    #[must_use]
    pub fn reverse_scalar(&self, operation: MetricBinaryOp, scalar: f64) -> Self {
        self.scalar_ordered(operation, scalar, true)
    }

    /// Applies index-aligned arithmetic or comparison.
    ///
    /// # Errors
    ///
    /// Returns an error unless both metrics contain the same unique label set.
    pub fn binary(
        &self,
        operation: MetricBinaryOp,
        other: &dyn DenseMetric,
    ) -> Result<Self, SingleDataError> {
        let rhs = self.align_other(other)?;
        let values = self
            .values()
            .iter()
            .zip(rhs)
            .map(|(lhs, rhs)| compute(operation, *lhs, rhs))
            .collect();
        Self::try_new(self.index.clone(), Array1::from_vec(values))
    }

    /// Adds two metrics over their sorted label union, replacing absent and NaN values first.
    ///
    /// # Errors
    ///
    /// Returns an error only if the generated sorted index violates the unique-index invariant.
    pub fn add(&self, other: &dyn DenseMetric, fill_value: f64) -> Result<Self, SingleDataError> {
        let index: BTreeSet<_> = self.index().iter().chain(other.index()).cloned().collect();
        let other_positions: HashMap<_, _> = other
            .index()
            .iter()
            .enumerate()
            .map(|(row, key)| (key.as_str(), row))
            .collect();
        let values = index
            .iter()
            .map(|key| {
                let lhs = self
                    .get(key)
                    .filter(|value| !value.is_nan())
                    .unwrap_or(fill_value);
                let rhs = other_positions
                    .get(key.as_str())
                    .map(|row| other.values()[*row])
                    .filter(|value| !value.is_nan())
                    .unwrap_or(fill_value);
                lhs + rhs
            })
            .collect();
        Self::try_new(index.into_iter().collect(), Array1::from_vec(values))
    }

    /// Reorders/subsets and fills absent labels.
    ///
    /// # Errors
    ///
    /// Returns an error when the requested index contains duplicates.
    pub fn reindex<S: AsRef<str>>(
        &self,
        index: &[S],
        fill_value: f64,
    ) -> Result<Self, SingleDataError> {
        if index
            .iter()
            .map(AsRef::as_ref)
            .eq(self.index.iter().map(String::as_str))
        {
            return Ok(self.clone());
        }
        Self::try_new(
            index.iter().map(|key| key.as_ref().to_owned()).collect(),
            Array1::from_iter(
                index
                    .iter()
                    .map(|key| self.get(key.as_ref()).unwrap_or(fill_value)),
            ),
        )
    }

    #[must_use]
    pub fn abs(&self) -> Self {
        Self::from_validated(self.index.clone(), self.values.mapv(f64::abs))
    }

    #[must_use]
    #[allow(clippy::float_cmp)]
    pub fn replace(&self, replacements: &[(f64, f64)]) -> Self {
        let values = self.values.mapv(|mut value| {
            for (from, to) in replacements {
                if value == *from {
                    value = *to;
                    break;
                }
            }
            value
        });
        Self::from_validated(self.index.clone(), values)
    }

    /// Applies one vector transform, matching `SingleData.apply` rather than Pandas element mapping.
    ///
    /// # Errors
    ///
    /// Returns the callback failure or an output-length error.
    pub fn apply(&self, transform: &DenseTransform) -> Result<Self, SingleDataError> {
        let values = transform(self.values()).map_err(SingleDataError::Transform)?;
        if values.len() != self.len() {
            return Err(SingleDataError::TransformLengthMismatch {
                expected: self.len(),
                actual: values.len(),
            });
        }
        Ok(Self::from_validated(
            self.index.clone(),
            Array1::from_vec(values),
        ))
    }

    #[must_use]
    pub fn isna(&self) -> Self {
        Self::from_validated(
            self.index.clone(),
            self.values.mapv(|value| f64::from(value.is_nan())),
        )
    }

    #[must_use]
    pub fn fillna(&self, value: f64) -> Self {
        Self::from_validated(
            self.index.clone(),
            self.values
                .mapv(|item| if item.is_nan() { value } else { item }),
        )
    }

    fn scalar_ordered(&self, operation: MetricBinaryOp, scalar: f64, reverse: bool) -> Self {
        let values = self.values.mapv(|value| {
            if reverse {
                compute(operation, scalar, value)
            } else {
                compute(operation, value, scalar)
            }
        });
        Self::from_validated(self.index.clone(), values)
    }

    fn align_other(&self, other: &dyn DenseMetric) -> Result<Vec<f64>, SingleDataError> {
        if self.index() == other.index() {
            return Ok(other.values().to_vec());
        }
        let self_set: HashSet<_> = self.index.iter().map(String::as_str).collect();
        let other_positions: HashMap<_, _> = other
            .index()
            .iter()
            .enumerate()
            .map(|(row, key)| (key.as_str(), row))
            .collect();
        if self_set.len() != other_positions.len()
            || !self_set.iter().all(|key| other_positions.contains_key(key))
        {
            return Err(SingleDataError::IndexMismatch);
        }
        Ok(self
            .index
            .iter()
            .map(|key| other.values()[other_positions[key.as_str()]])
            .collect())
    }

    fn from_validated(index: Vec<String>, values: Array1<f64>) -> Self {
        Self::try_new(index, values).expect("an existing validated index remains valid")
    }
}

impl Default for SingleData {
    fn default() -> Self {
        Self::empty()
    }
}

impl DenseMetric for SingleData {
    fn index(&self) -> &[String] {
        &self.index
    }

    fn values(&self) -> &[f64] {
        self.values
            .as_slice()
            .expect("owned one-dimensional ndarray storage is contiguous")
    }
}

#[allow(clippy::float_cmp)]
fn compute(operation: MetricBinaryOp, lhs: f64, rhs: f64) -> f64 {
    match operation {
        MetricBinaryOp::Add => lhs + rhs,
        MetricBinaryOp::Subtract => lhs - rhs,
        MetricBinaryOp::Multiply => lhs * rhs,
        MetricBinaryOp::Divide => lhs / rhs,
        MetricBinaryOp::Equal => f64::from(lhs == rhs),
        MetricBinaryOp::Greater => f64::from(lhs > rhs),
        MetricBinaryOp::Less => f64::from(lhs < rhs),
    }
}
