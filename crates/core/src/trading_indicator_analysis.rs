use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Float64Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use chrono::NaiveDateTime;
use num_traits::ToPrimitive;
use thiserror::Error;

const OUTPUT_INDEX: [&str; 3] = ["ffr", "pa", "pos"];

/// Numeric indicator table with its pandas datetime index kept outside Arrow columns.
#[derive(Clone, Debug)]
pub struct TradingIndicatorTable {
    index: Vec<NaiveDateTime>,
    columns: RecordBatch,
}

impl TradingIndicatorTable {
    /// Builds a table without reordering its rows or index.
    ///
    /// # Errors
    /// Returns an error when the explicit index and Arrow rows are not aligned.
    pub fn try_new(
        index: Vec<NaiveDateTime>,
        columns: RecordBatch,
    ) -> Result<Self, IndicatorAnalysisError> {
        if index.len() != columns.num_rows() {
            return Err(IndicatorAnalysisError::LengthMismatch {
                index: index.len(),
                rows: columns.num_rows(),
            });
        }
        Ok(Self { index, columns })
    }

    #[must_use]
    pub fn index(&self) -> &[NaiveDateTime] {
        &self.index
    }

    #[must_use]
    pub fn columns(&self) -> &RecordBatch {
        &self.columns
    }
}

/// One-column result retaining pandas' row labels separately from Arrow storage.
#[derive(Clone, Debug)]
pub struct IndicatorAnalysis {
    index: [&'static str; 3],
    values: RecordBatch,
}

impl IndicatorAnalysis {
    #[must_use]
    pub const fn index(&self) -> &[&'static str; 3] {
        &self.index
    }

    #[must_use]
    pub fn values(&self) -> &RecordBatch {
        &self.values
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum IndicatorAnalysisError {
    #[error("indicator index has {index} entries but its columns have {rows} rows")]
    LengthMismatch { index: usize, rows: usize },
    #[error("indicator column `{column}` is missing")]
    MissingColumn { column: &'static str },
    #[error("{message}")]
    MissingIndicatorColumns {
        columns: Vec<&'static str>,
        message: String,
    },
    #[error("indicator column `{column}` must be Float64 or Int64, got {actual}")]
    InvalidColumnType {
        column: &'static str,
        actual: DataType,
    },
    #[error("indicator_analysis method {method} is not supported!")]
    UnsupportedMethod { method: String },
}

#[derive(Clone, Copy)]
enum NumericColumn<'a> {
    Float64(&'a Float64Array),
    Int64(&'a Int64Array),
}

impl NumericColumn<'_> {
    fn value(self, row: usize) -> f64 {
        match self {
            Self::Float64(values) => values.is_valid(row).then(|| values.value(row)),
            Self::Int64(values) => values.is_valid(row).then(|| {
                values
                    .value(row)
                    .to_f64()
                    .expect("every i64 converts to f64")
            }),
        }
        .unwrap_or(f64::NAN)
    }

    fn len(self) -> usize {
        match self {
            Self::Float64(values) => values.len(),
            Self::Int64(values) => values.len(),
        }
    }
}

fn column<'a>(
    table: &'a TradingIndicatorTable,
    name: &'static str,
) -> Result<NumericColumn<'a>, IndicatorAnalysisError> {
    let array = table
        .columns
        .column_by_name(name)
        .ok_or(IndicatorAnalysisError::MissingColumn { column: name })?;
    if let Some(values) = array.as_any().downcast_ref::<Float64Array>() {
        Ok(NumericColumn::Float64(values))
    } else if let Some(values) = array.as_any().downcast_ref::<Int64Array>() {
        Ok(NumericColumn::Int64(values))
    } else {
        Err(IndicatorAnalysisError::InvalidColumnType {
            column: name,
            actual: array.data_type().clone(),
        })
    }
}

fn pairwise_sum(values: &[f64]) -> f64 {
    const BLOCK: usize = 128;
    match values.len() {
        0 => 0.0,
        1..=7 => values.iter().fold(-0.0, |sum, value| sum + value),
        8..=BLOCK => {
            let mut partial = <[f64; 8]>::try_from(&values[..8]).expect("eight-value prefix");
            let mut offset = 8;
            while offset + 8 <= values.len() {
                for lane in 0..8 {
                    partial[lane] += values[offset + lane];
                }
                offset += 8;
            }
            let mut sum = ((partial[0] + partial[1]) + (partial[2] + partial[3]))
                + ((partial[4] + partial[5]) + (partial[6] + partial[7]));
            for value in &values[offset..] {
                sum += value;
            }
            sum
        }
        _ => {
            let midpoint = values.len() / 2 / 8 * 8;
            pairwise_sum(&values[..midpoint]) + pairwise_sum(&values[midpoint..])
        }
    }
}

fn pandas_sum(values: impl Iterator<Item = f64>) -> f64 {
    let values = values
        .map(|value| if value.is_nan() { 0.0 } else { value })
        .collect::<Vec<_>>();
    pairwise_sum(&values)
}

fn weighted_mean(values: NumericColumn<'_>, weights: NumericColumn<'_>, absolute: bool) -> f64 {
    let rows = 0..values.len();
    let numerator = match (values, weights) {
        (NumericColumn::Int64(values), NumericColumn::Int64(weights)) => rows
            .clone()
            .filter(|row| values.is_valid(*row) && weights.is_valid(*row))
            .map(|row| {
                let weight = if absolute {
                    weights.value(row).wrapping_abs()
                } else {
                    weights.value(row)
                };
                values.value(row).wrapping_mul(weight)
            })
            .fold(0_i64, i64::wrapping_add)
            .to_f64()
            .expect("every i64 converts to f64"),
        _ => pandas_sum(rows.clone().map(|row| {
            let weight = weights.value(row);
            values.value(row) * if absolute { weight.abs() } else { weight }
        })),
    };
    let denominator = match weights {
        NumericColumn::Int64(weights) => rows
            .filter(|row| weights.is_valid(*row))
            .map(|row| {
                if absolute {
                    weights.value(row).wrapping_abs()
                } else {
                    weights.value(row)
                }
            })
            .fold(0_i64, i64::wrapping_add)
            .to_f64()
            .expect("every i64 converts to f64"),
        NumericColumn::Float64(_) => pandas_sum(rows.map(|row| {
            let weight = weights.value(row);
            if absolute { weight.abs() } else { weight }
        })),
    };
    numerator / denominator
}

fn pandas_series_nan(value: f64) -> f64 {
    if value.is_nan() {
        f64::from_bits(0xfff8_0000_0000_0000)
    } else {
        value
    }
}

fn pandas_scalar_nan(value: f64) -> f64 {
    if value.is_nan() { f64::NAN } else { value }
}

fn missing_indicator_columns(columns: Vec<&'static str>) -> IndicatorAnalysisError {
    let message = if columns.len() == 2 {
        "\"None of [Index(['ffr', 'pa'], dtype='object')] are in the [columns]\"".to_owned()
    } else {
        format!("\"['{}'] not in index\"", columns[0])
    };
    IndicatorAnalysisError::MissingIndicatorColumns { columns, message }
}

/// Analyze `ffr`, `pa`, and `pos` exactly in Qlib's stable output order.
///
/// The three possible weight columns are deliberately accessed before method validation,
/// matching the eager Python dictionary literal. `pos` always uses raw `count` weights.
/// Float64 and Int64 Arrow nulls follow pandas nullable-numeric behavior and participate as NaN.
///
/// # Errors
/// Returns the first eagerly accessed missing/type-invalid column, an unsupported method,
/// or the first subsequently required indicator column.
///
/// # Panics
/// Panics only if the internally constructed three-row Float64 output violates its fixed schema.
pub fn indicator_analysis(
    table: &TradingIndicatorTable,
    method: &str,
) -> Result<IndicatorAnalysis, IndicatorAnalysisError> {
    let count = column(table, "count")?;
    let deal_amount = column(table, "deal_amount")?;
    let value = column(table, "value")?;
    let (weights, absolute) = match method {
        "mean" => (count, false),
        "amount_weighted" => (deal_amount, true),
        "value_weighted" => (value, true),
        method => {
            return Err(IndicatorAnalysisError::UnsupportedMethod {
                method: method.to_owned(),
            });
        }
    };
    let missing = ["ffr", "pa"]
        .into_iter()
        .filter(|name| table.columns.column_by_name(name).is_none())
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(missing_indicator_columns(missing));
    }
    let ffr = column(table, "ffr")?;
    let pa = column(table, "pa")?;
    let ffr = pandas_series_nan(weighted_mean(ffr, weights, absolute));
    let pa = pandas_series_nan(weighted_mean(pa, weights, absolute));
    let pos = column(table, "pos")?;
    let pos = pandas_scalar_nan(weighted_mean(pos, count, false));
    let values: ArrayRef = Arc::new(Float64Array::from(vec![ffr, pa, pos]));
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Float64,
        false,
    )]));
    let values = RecordBatch::try_new(schema, vec![values])
        .expect("one fixed-length output column always matches its schema");
    Ok(IndicatorAnalysis {
        index: OUTPUT_INDEX,
        values,
    })
}
