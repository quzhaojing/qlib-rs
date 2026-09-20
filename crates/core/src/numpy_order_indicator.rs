use std::collections::BTreeSet;

use indexmap::IndexMap;
use thiserror::Error;

use crate::{DenseMetric, SingleData, SingleDataError};

/// Mutable storage boundary for NumPy-style dense order indicators.
pub trait DenseOrderIndicator: Send + Sync {
    fn metric_names(&self) -> Box<dyn Iterator<Item = &str> + '_>;
    fn metric(&self, name: &str) -> Option<&dyn DenseMetric>;
    fn assign_metric(&mut self, name: &str, metric: SingleData);
}

#[derive(Clone, Debug)]
pub enum DenseIndicatorValue {
    Metric(SingleData),
    Number(f64),
    Integer(i64),
    Boolean(bool),
    Missing,
}

pub trait DenseIndicatorTransform: Send + Sync {
    fn input_names(&self) -> &[String];

    /// # Errors
    ///
    /// Returns a plugin-defined error when the transform cannot produce a result.
    fn apply(&self, inputs: &[&dyn DenseMetric]) -> Result<DenseIndicatorValue, String>;
}

#[derive(Debug, Error)]
pub enum NumpyOrderIndicatorError {
    #[error("NumPy order indicator metric not found: {0}")]
    MissingMetric(String),
    #[error("NumPy order indicator transform failed: {0}")]
    Transform(String),
    #[error("only a dense metric transform result can be assigned to a new column")]
    AssignedNonMetric,
    #[error("at least one metric name is required when indicators are present")]
    EmptyMetrics,
    #[error(transparent)]
    SingleData(#[from] SingleDataError),
}

/// Executes a dense transform after resolving its declared inputs by name.
///
/// # Errors
///
/// Returns an error for missing inputs, callback failures, or scalar assignment.
pub fn transfer_dense(
    indicator: &mut dyn DenseOrderIndicator,
    transform: &dyn DenseIndicatorTransform,
    new_column: Option<&str>,
) -> Result<Option<DenseIndicatorValue>, NumpyOrderIndicatorError> {
    let result = {
        let inputs = transform
            .input_names()
            .iter()
            .map(|name| {
                indicator
                    .metric(name)
                    .ok_or_else(|| NumpyOrderIndicatorError::MissingMetric(name.clone()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        transform
            .apply(&inputs)
            .map_err(NumpyOrderIndicatorError::Transform)?
    };
    if let Some(name) = new_column {
        if let DenseIndicatorValue::Metric(metric) = result {
            indicator.assign_metric(name, metric);
            Ok(None)
        } else {
            Err(NumpyOrderIndicatorError::AssignedNonMetric)
        }
    } else {
        Ok(Some(result))
    }
}

#[derive(Clone, Debug, Default)]
pub struct NumpyOrderIndicator {
    data: IndexMap<String, SingleData>,
}

impl NumpyOrderIndicator {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn assign(&mut self, name: &str, metric: SingleData) {
        self.data.insert(name.to_owned(), metric);
    }

    #[must_use]
    pub fn get_index_data(&self, name: &str) -> SingleData {
        self.data.get(name).cloned().unwrap_or_default()
    }

    /// Returns a cheap owned dense-series snapshot.
    ///
    /// # Errors
    ///
    /// Returns an error when the metric is absent.
    pub fn get_metric_series(&self, name: &str) -> Result<SingleData, NumpyOrderIndicatorError> {
        self.data
            .get(name)
            .cloned()
            .ok_or_else(|| NumpyOrderIndicatorError::MissingMetric(name.to_owned()))
    }

    #[must_use]
    pub fn to_series(&self) -> IndexMap<String, SingleData> {
        self.data.clone()
    }

    /// Sums requested metrics over the sorted stock union determined by the first metric.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty metric list with inputs, missing metrics, or invalid data.
    ///
    /// # Panics
    ///
    /// Panics only if the internally generated `BTreeSet` stock index contains duplicates,
    /// which would violate the standard-library collection invariant.
    pub fn sum_all_indicators(
        output: &mut dyn DenseOrderIndicator,
        indicators: &[&dyn DenseOrderIndicator],
        metrics: &[&str],
        fill_value: f64,
    ) -> Result<(), NumpyOrderIndicatorError> {
        if indicators.is_empty() {
            for name in metrics {
                output.assign_metric(name, SingleData::empty());
            }
            return Ok(());
        }
        let first = metrics
            .first()
            .ok_or(NumpyOrderIndicatorError::EmptyMetrics)?;
        let mut stocks = BTreeSet::new();
        for indicator in indicators {
            let metric = indicator
                .metric(first)
                .ok_or_else(|| NumpyOrderIndicatorError::MissingMetric((*first).to_owned()))?;
            stocks.extend(metric.index().iter().cloned());
        }
        let stocks: Vec<_> = stocks.into_iter().collect();
        for name in metrics {
            let data = indicators
                .iter()
                .map(|indicator| {
                    indicator
                        .metric(name)
                        .ok_or_else(|| NumpyOrderIndicatorError::MissingMetric((*name).to_owned()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            output.assign_metric(
                name,
                sum_by_index(&data, &stocks, fill_value)
                    .expect("a BTreeSet-derived stock index is unique"),
            );
        }
        Ok(())
    }
}

impl DenseOrderIndicator for NumpyOrderIndicator {
    fn metric_names(&self) -> Box<dyn Iterator<Item = &str> + '_> {
        Box::new(self.data.keys().map(String::as_str))
    }

    fn metric(&self, name: &str) -> Option<&dyn DenseMetric> {
        self.data.get(name).map(|metric| metric as &dyn DenseMetric)
    }

    fn assign_metric(&mut self, name: &str, metric: SingleData) {
        self.assign(name, metric);
    }
}

/// NumPy-compatible summation over an explicit output index.
///
/// # Errors
///
/// Returns an error when the generated output violates the dense metric invariant.
pub fn sum_by_index(
    data: &[&dyn DenseMetric],
    index: &[String],
    fill_value: f64,
) -> Result<SingleData, SingleDataError> {
    let values = index.iter().map(|key| {
        data.iter()
            .map(|metric| {
                metric
                    .index()
                    .iter()
                    .position(|candidate| candidate == key)
                    .map(|row| metric.values()[row])
                    .filter(|value| !value.is_nan())
                    .unwrap_or(fill_value)
            })
            .sum()
    });
    SingleData::try_new(index.to_vec(), ndarray::Array1::from_iter(values))
}
