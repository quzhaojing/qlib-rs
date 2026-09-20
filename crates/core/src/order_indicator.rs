use indexmap::IndexMap;
use thiserror::Error;

use crate::{PandasSingleMetric, SingleMetric, SingleMetricError};

/// Mutable storage boundary for a named collection of stock-indexed metrics.
///
/// Implementations can keep metrics in memory, proxy them to another process, or adapt a
/// future physical plugin. Ordering is observable and must follow first insertion order.
pub trait OrderIndicator: Send + Sync {
    fn metric_names(&self) -> Box<dyn Iterator<Item = &str> + '_>;
    fn metric(&self, name: &str) -> Option<&dyn SingleMetric>;
    fn assign_metric(&mut self, name: &str, metric: PandasSingleMetric);
}

/// Typed result of an order-indicator transform.
#[derive(Clone, Debug)]
pub enum IndicatorValue {
    Metric(PandasSingleMetric),
    Number(f64),
    Integer(i64),
    Boolean(bool),
    Missing,
}

/// Plugin-ready replacement for Python's callback-signature reflection.
pub trait OrderIndicatorTransform: Send + Sync {
    fn input_names(&self) -> &[String];

    /// # Errors
    ///
    /// Returns a plugin-defined error string when the transform cannot produce a value.
    fn apply(&self, inputs: &[&dyn SingleMetric]) -> Result<IndicatorValue, String>;
}

#[derive(Debug, Error)]
pub enum OrderIndicatorError {
    #[error("order indicator metric not found: {0}")]
    MissingMetric(String),
    #[error("order indicator transform failed: {0}")]
    Transform(String),
    #[error("only a metric transform result can be assigned to a new column")]
    AssignedNonMetric,
    #[error(transparent)]
    SingleMetric(#[from] SingleMetricError),
}

/// Executes a transform after resolving its declared inputs by metric name.
///
/// # Errors
///
/// Returns an error for a missing input, transform failure, or scalar assignment.
pub fn transfer(
    indicator: &mut dyn OrderIndicator,
    transform: &dyn OrderIndicatorTransform,
    new_column: Option<&str>,
) -> Result<Option<IndicatorValue>, OrderIndicatorError> {
    let result = {
        let inputs = transform
            .input_names()
            .iter()
            .map(|name| {
                indicator
                    .metric(name)
                    .ok_or_else(|| OrderIndicatorError::MissingMetric(name.clone()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        transform
            .apply(&inputs)
            .map_err(OrderIndicatorError::Transform)?
    };

    if let Some(name) = new_column {
        if let IndicatorValue::Metric(metric) = result {
            indicator.assign_metric(name, metric);
            Ok(None)
        } else {
            Err(OrderIndicatorError::AssignedNonMetric)
        }
    } else {
        Ok(Some(result))
    }
}

#[derive(Clone, Debug, Default)]
pub struct PandasOrderIndicator {
    data: IndexMap<String, PandasSingleMetric>,
}

impl PandasOrderIndicator {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn assign(&mut self, name: &str, metric: PandasSingleMetric) {
        self.data.insert(name.to_owned(), metric);
    }

    /// Returns the raw Pandas-Series-compatible metric or an empty metric when absent.
    #[must_use]
    pub fn get_metric_series(&self, name: &str) -> PandasSingleMetric {
        self.data.get(name).cloned().unwrap_or_default()
    }

    /// Returns the Float64 `SingleData`-compatible metric or an empty metric when absent.
    #[must_use]
    pub fn get_index_data(&self, name: &str) -> PandasSingleMetric {
        self.data
            .get(name)
            .map_or_else(PandasSingleMetric::empty, PandasSingleMetric::to_f64)
    }

    /// Returns an ordered, cheap Arrow-buffer-sharing snapshot.
    #[must_use]
    pub fn to_series(&self) -> IndexMap<String, PandasSingleMetric> {
        self.data.clone()
    }

    /// Adds the requested metrics across all indicators and assigns each result to `output`.
    ///
    /// # Errors
    ///
    /// Returns an error when any input indicator lacks a requested metric.
    ///
    /// # Panics
    ///
    /// Panics if an external `SingleMetric` implementation violates the documented numeric or
    /// Boolean value invariant required by the metric arithmetic boundary.
    pub fn sum_all_indicators(
        output: &mut dyn OrderIndicator,
        indicators: &[&dyn OrderIndicator],
        metrics: &[&str],
        fill_value: Option<f64>,
    ) -> Result<(), OrderIndicatorError> {
        for name in metrics {
            let mut sum = PandasSingleMetric::empty();
            for indicator in indicators {
                let metric = indicator
                    .metric(name)
                    .ok_or_else(|| OrderIndicatorError::MissingMetric((*name).to_owned()))?;
                sum = sum
                    .add_filled(metric, fill_value)
                    .expect("validated SingleMetric values support filled addition");
            }
            output.assign_metric(name, sum);
        }
        Ok(())
    }
}

impl OrderIndicator for PandasOrderIndicator {
    fn metric_names(&self) -> Box<dyn Iterator<Item = &str> + '_> {
        Box::new(self.data.keys().map(String::as_str))
    }

    fn metric(&self, name: &str) -> Option<&dyn SingleMetric> {
        self.data
            .get(name)
            .map(|metric| metric as &dyn SingleMetric)
    }

    fn assign_metric(&mut self, name: &str, metric: PandasSingleMetric) {
        self.assign(name, metric);
    }
}
