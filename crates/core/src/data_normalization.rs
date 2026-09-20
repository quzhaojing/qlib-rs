//! Numeric normalization compatible with `qlib.utils.data`.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Float16Array, Float32Array, Float64Array};
use arrow_schema::DataType;
use half::f16;
use thiserror::Error;

/// A Pandas numeric dtype paired with Arrow storage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PandasNumericDtype {
    physical: DataType,
    nullable: bool,
}

impl PandasNumericDtype {
    /// Describes a NumPy-backed Pandas numeric dtype.
    ///
    /// # Errors
    ///
    /// Returns [`NormalizationError::UnsupportedDtype`] for non-numeric Arrow types.
    pub fn native(physical: DataType) -> Result<Self, NormalizationError> {
        Self::new(physical, false)
    }

    /// Describes a nullable Pandas extension numeric dtype.
    ///
    /// # Errors
    ///
    /// Returns [`NormalizationError::UnsupportedDtype`] for non-numeric types and `Float16`,
    /// which has no corresponding Pandas nullable extension dtype.
    pub fn nullable(physical: DataType) -> Result<Self, NormalizationError> {
        Self::new(physical, true)
    }

    fn new(physical: DataType, nullable: bool) -> Result<Self, NormalizationError> {
        if !is_supported(&physical) || nullable && physical == DataType::Float16 {
            return Err(NormalizationError::UnsupportedDtype(physical));
        }
        Ok(Self { physical, nullable })
    }

    #[must_use]
    pub fn physical(&self) -> &DataType {
        &self.physical
    }

    #[must_use]
    pub fn is_nullable(&self) -> bool {
        self.nullable
    }
}

/// One numeric column with explicit Pandas dtype identity.
#[derive(Clone, Debug)]
pub struct NumericColumn {
    values: ArrayRef,
    dtype: PandasNumericDtype,
}

impl NumericColumn {
    /// Attaches a Pandas dtype to Arrow values.
    ///
    /// # Errors
    ///
    /// Rejects a physical dtype mismatch and Arrow nulls in a NumPy-backed dtype. Native
    /// floating missing values are IEEE NaNs; nullable extension dtypes use Arrow nulls.
    pub fn try_new(
        values: ArrayRef,
        dtype: PandasNumericDtype,
    ) -> Result<Self, NormalizationError> {
        if values.data_type() != dtype.physical() {
            return Err(NormalizationError::DtypeMismatch {
                declared: dtype.physical().clone(),
                actual: values.data_type().clone(),
            });
        }
        if !dtype.is_nullable() && values.null_count() != 0 {
            return Err(NormalizationError::NativeNulls);
        }
        Ok(Self { values, dtype })
    }

    #[must_use]
    pub fn values(&self) -> &ArrayRef {
        &self.values
    }

    #[must_use]
    pub fn dtype(&self) -> &PandasNumericDtype {
        &self.dtype
    }
}

/// A Pandas-Series-shaped numeric value with stable labels.
#[derive(Clone, Debug)]
pub struct NumericSeries {
    index: ArrayRef,
    index_name: Option<String>,
    name: Option<String>,
    column: NumericColumn,
}

impl NumericSeries {
    /// Builds a labeled numeric series.
    ///
    /// # Errors
    ///
    /// Rejects unequal index and value lengths.
    pub fn try_new(
        index: ArrayRef,
        index_name: Option<String>,
        name: Option<String>,
        column: NumericColumn,
    ) -> Result<Self, NormalizationError> {
        if index.len() != column.values().len() {
            return Err(NormalizationError::IndexLength {
                index: index.len(),
                values: column.values().len(),
            });
        }
        Ok(Self {
            index,
            index_name,
            name,
            column,
        })
    }

    #[must_use]
    pub fn index(&self) -> &ArrayRef {
        &self.index
    }

    #[must_use]
    pub fn index_name(&self) -> Option<&str> {
        self.index_name.as_deref()
    }

    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    #[must_use]
    pub fn column(&self) -> &NumericColumn {
        &self.column
    }
}

/// A numeric `DataFrame` shape. Column labels may repeat, matching Pandas.
#[derive(Clone, Debug)]
pub struct NumericFrame {
    index: ArrayRef,
    index_name: Option<String>,
    columns: Vec<(String, NumericColumn)>,
}

impl NumericFrame {
    /// Builds a labeled numeric frame while preserving column order.
    ///
    /// # Errors
    ///
    /// Rejects any column whose length differs from the index length.
    pub fn try_new(
        index: ArrayRef,
        index_name: Option<String>,
        columns: Vec<(String, NumericColumn)>,
    ) -> Result<Self, NormalizationError> {
        if let Some((_, column)) = columns
            .iter()
            .find(|(_, column)| column.values().len() != index.len())
        {
            return Err(NormalizationError::IndexLength {
                index: index.len(),
                values: column.values().len(),
            });
        }
        Ok(Self {
            index,
            index_name,
            columns,
        })
    }

    #[must_use]
    pub fn index(&self) -> &ArrayRef {
        &self.index
    }

    #[must_use]
    pub fn index_name(&self) -> Option<&str> {
        self.index_name.as_deref()
    }

    #[must_use]
    pub fn columns(&self) -> &[(String, NumericColumn)] {
        &self.columns
    }
}

/// Input shapes accepted by [`zscore`] and [`robust_zscore`].
pub trait NormalizationInput: sealed::Sealed + Sized {
    #[doc(hidden)]
    fn qlib_zscore(&self) -> Result<Self, NormalizationError>;
    #[doc(hidden)]
    fn qlib_robust_zscore(&self, post_zscore: bool) -> Result<Self, NormalizationError>;
    #[doc(hidden)]
    fn qlib_zscore_warnings(&self) -> Vec<NormalizationWarning>;
    #[doc(hidden)]
    fn qlib_robust_zscore_warnings(&self) -> Vec<NormalizationWarning>;
}

/// One ordered Python warning emitted by the source normalization operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NormalizationWarning {
    category: &'static str,
    message: &'static str,
}

impl NormalizationWarning {
    #[must_use]
    pub fn category(self) -> &'static str {
        self.category
    }

    #[must_use]
    pub fn message(self) -> &'static str {
        self.message
    }
}

/// A normalized value plus source-ordered warnings.
#[derive(Clone, Debug)]
pub struct NormalizationReport<T> {
    output: T,
    warnings: Vec<NormalizationWarning>,
}

impl<T> NormalizationReport<T> {
    #[must_use]
    pub fn output(&self) -> &T {
        &self.output
    }

    #[must_use]
    pub fn warnings(&self) -> &[NormalizationWarning] {
        &self.warnings
    }

    #[must_use]
    pub fn into_output(self) -> T {
        self.output
    }
}

/// Failures at the explicit Arrow/Pandas compatibility boundary.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum NormalizationError {
    #[error("unsupported Pandas numeric dtype backed by {0:?}")]
    UnsupportedDtype(DataType),
    #[error("declared dtype {declared:?} does not match Arrow storage {actual:?}")]
    DtypeMismatch {
        declared: DataType,
        actual: DataType,
    },
    #[error("NumPy-backed numeric columns cannot contain Arrow nulls")]
    NativeNulls,
    #[error("series/frame index has {index} rows but a value column has {values}")]
    IndexLength { index: usize, values: usize },
}

/// Column-wise sample z-score using Pandas' default `ddof=1` and missing-value handling.
///
/// # Errors
///
/// Returns an error only when an input violates the validated numeric shape contract.
pub fn zscore<T: NormalizationInput>(input: &T) -> Result<T, NormalizationError> {
    zscore_with_warnings(input).map(NormalizationReport::into_output)
}

/// Column-wise sample z-score together with ordered source warnings.
///
/// # Errors
///
/// Returns an error only when an input violates the validated numeric shape contract.
pub fn zscore_with_warnings<T: NormalizationInput>(
    input: &T,
) -> Result<NormalizationReport<T>, NormalizationError> {
    Ok(NormalizationReport {
        output: input.qlib_zscore()?,
        warnings: input.qlib_zscore_warnings(),
    })
}

/// Median/MAD normalization, clipped to `[-3, 3]`, with optional post-zscore.
///
/// # Errors
///
/// Returns an error only when an input violates the validated numeric shape contract.
pub fn robust_zscore<T: NormalizationInput>(
    input: &T,
    post_zscore: bool,
) -> Result<T, NormalizationError> {
    robust_zscore_with_warnings(input, post_zscore).map(NormalizationReport::into_output)
}

/// Median/MAD normalization together with ordered source warnings.
///
/// # Errors
///
/// Returns an error only when an input violates the validated numeric shape contract.
pub fn robust_zscore_with_warnings<T: NormalizationInput>(
    input: &T,
    post_zscore: bool,
) -> Result<NormalizationReport<T>, NormalizationError> {
    Ok(NormalizationReport {
        output: input.qlib_robust_zscore(post_zscore)?,
        warnings: input.qlib_robust_zscore_warnings(),
    })
}

impl NormalizationInput for NumericSeries {
    fn qlib_zscore(&self) -> Result<Self, NormalizationError> {
        let precision = series_precision(&self.column.dtype);
        Ok(Self {
            index: Arc::clone(&self.index),
            index_name: self.index_name.clone(),
            name: self.name.clone(),
            column: normalize_column(&self.column, precision, Method::Zscore),
        })
    }

    fn qlib_robust_zscore(&self, post_zscore: bool) -> Result<Self, NormalizationError> {
        let precision = series_precision(&self.column.dtype);
        Ok(Self {
            index: Arc::clone(&self.index),
            index_name: self.index_name.clone(),
            name: self.name.clone(),
            column: normalize_column(&self.column, precision, Method::Robust(post_zscore)),
        })
    }

    fn qlib_zscore_warnings(&self) -> Vec<NormalizationWarning> {
        column_zscore_warnings(&self.column)
    }

    fn qlib_robust_zscore_warnings(&self) -> Vec<NormalizationWarning> {
        column_robust_warnings(&self.column)
    }
}

impl NormalizationInput for NumericFrame {
    fn qlib_zscore(&self) -> Result<Self, NormalizationError> {
        Ok(self.normalized(Method::Zscore))
    }

    fn qlib_robust_zscore(&self, post_zscore: bool) -> Result<Self, NormalizationError> {
        Ok(self.normalized(Method::Robust(post_zscore)))
    }

    fn qlib_zscore_warnings(&self) -> Vec<NormalizationWarning> {
        frame_warnings(&self.columns, WarningMethod::Zscore)
    }

    fn qlib_robust_zscore_warnings(&self) -> Vec<NormalizationWarning> {
        frame_warnings(&self.columns, WarningMethod::Robust)
    }
}

impl NumericFrame {
    fn normalized(&self, method: Method) -> Self {
        let native_precision = frame_native_precision(&self.columns);
        let columns = self
            .columns
            .iter()
            .map(|(name, column)| {
                let precision = if column.dtype.is_nullable() {
                    series_precision(&column.dtype)
                } else {
                    native_precision.expect("a native column establishes frame precision")
                };
                (name.clone(), normalize_column(column, precision, method))
            })
            .collect();
        Self {
            index: Arc::clone(&self.index),
            index_name: self.index_name.clone(),
            columns,
        }
    }
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::NumericSeries {}
    impl Sealed for super::NumericFrame {}
}

#[derive(Clone, Copy)]
enum Method {
    Zscore,
    Robust(bool),
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Precision {
    F16,
    F32,
    F64,
}

#[derive(Clone, Copy)]
enum WarningMethod {
    Zscore,
    Robust,
}

#[derive(Clone, Copy, Default)]
struct ReductionWarnings {
    overflow: bool,
    invalid: bool,
}

#[derive(Clone, Copy, Default)]
struct ZscoreWarnings {
    mean: ReductionWarnings,
    subtract_invalid: bool,
    deviation_mean: ReductionWarnings,
    square_overflow: bool,
    deviation_sum: ReductionWarnings,
    cast_overflow: bool,
}

#[derive(Clone, Copy, Default)]
struct RobustWarnings {
    center: ReductionWarnings,
    mad: ReductionWarnings,
}

fn runtime_warning(message: &'static str) -> NormalizationWarning {
    NormalizationWarning {
        category: "RuntimeWarning",
        message,
    }
}

fn reduction_warnings(values: &[f64], result: f64) -> ReductionWarnings {
    let has_value = !values.is_empty();
    let all_finite = values.iter().all(|value| value.is_finite());
    let has_nan = values.iter().any(|value| value.is_nan());
    ReductionWarnings {
        overflow: has_value && all_finite && !result.is_finite(),
        invalid: has_value && !has_nan && result.is_nan(),
    }
}

#[allow(clippy::cast_precision_loss)]
fn zscore_warning_events(values: &[Option<f64>], precision: Precision) -> ZscoreWarnings {
    let count = values.iter().filter(|value| !is_missing(**value)).count();
    if count == 0 {
        return ZscoreWarnings::default();
    }
    let replaced = values
        .iter()
        .map(|value| value.filter(|item| !item.is_nan()).unwrap_or(0.0))
        .collect::<Vec<_>>();
    let mean_sum = pairwise_sum(&replaced, precision);
    let mean = quantize(mean_sum / quantize(count as f64, precision), precision);
    let mean_warnings = reduction_warnings(&replaced, mean_sum);
    let subtract_invalid = mean.is_infinite()
        && values
            .iter()
            .flatten()
            .any(|value| value.to_bits() == mean.to_bits());
    if count < 2 {
        return ZscoreWarnings {
            mean: mean_warnings,
            subtract_invalid,
            ..ZscoreWarnings::default()
        };
    }
    let deviation_sum = pairwise_sum(&replaced, Precision::F64);
    let average = deviation_sum / count as f64;
    let squares = values
        .iter()
        .map(|value| {
            value.map_or(0.0, |item| {
                if item.is_nan() {
                    0.0
                } else {
                    (average - item).powi(2)
                }
            })
        })
        .collect::<Vec<_>>();
    let square_sum = pairwise_sum(&squares, Precision::F64);
    let variance = square_sum / (count - 1) as f64;
    let cast = quantize(variance, precision);
    ZscoreWarnings {
        mean: mean_warnings,
        subtract_invalid,
        deviation_mean: reduction_warnings(&replaced, deviation_sum),
        square_overflow: values.iter().flatten().any(|item| {
            let difference = average - item;
            difference.is_finite() && difference.powi(2).is_infinite()
        }),
        deviation_sum: reduction_warnings(&squares, square_sum),
        cast_overflow: variance.is_finite() && cast.is_infinite(),
    }
}

fn append_reduction_warnings(output: &mut Vec<NormalizationWarning>, warnings: ReductionWarnings) {
    if warnings.overflow {
        output.push(runtime_warning("overflow encountered in reduce"));
    }
    if warnings.invalid {
        output.push(runtime_warning("invalid value encountered in reduce"));
    }
}

fn render_zscore_warnings(events: &[ZscoreWarnings]) -> Vec<NormalizationWarning> {
    let mut output = Vec::new();
    for event in events {
        append_reduction_warnings(&mut output, event.mean);
    }
    for event in events {
        if event.subtract_invalid {
            output.push(runtime_warning("invalid value encountered in subtract"));
        }
    }
    for event in events {
        append_reduction_warnings(&mut output, event.deviation_mean);
    }
    for event in events {
        if event.square_overflow {
            output.push(runtime_warning("overflow encountered in square"));
        }
    }
    for event in events {
        append_reduction_warnings(&mut output, event.deviation_sum);
    }
    for event in events {
        if event.cast_overflow {
            output.push(runtime_warning("overflow encountered in cast"));
        }
    }
    output
}

fn median_warning(values: &[Option<f64>], precision: Precision) -> ReductionWarnings {
    let mut values = values
        .iter()
        .flatten()
        .copied()
        .filter(|value| !value.is_nan())
        .collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() < 2 || values.len() % 2 != 0 {
        return ReductionWarnings::default();
    }
    let pair = [values[middle - 1], values[middle]];
    reduction_warnings(&pair, quantize(pair[0] + pair[1], precision))
}

fn robust_warning_events(column: &NumericColumn) -> RobustWarnings {
    let precision = series_precision(&column.dtype);
    let values = values_as_f64(column.values());
    let center_warning = median_warning(&values, precision);
    let center = median(&values, precision);
    let absolute = values
        .iter()
        .map(|value| value.map(|item| quantize((item - center).abs(), precision)))
        .collect::<Vec<_>>();
    RobustWarnings {
        center: center_warning,
        mad: median_warning(&absolute, precision),
    }
}

fn column_zscore_warnings(column: &NumericColumn) -> Vec<NormalizationWarning> {
    let mut event = zscore_warning_events(
        &values_as_f64(column.values()),
        series_precision(&column.dtype),
    );
    if column.dtype.is_nullable() {
        event.subtract_invalid = false;
        event.deviation_mean = ReductionWarnings::default();
        event.square_overflow = false;
        event.deviation_sum = ReductionWarnings::default();
        event.cast_overflow = false;
    }
    render_zscore_warnings(&[event])
}

fn column_robust_warnings(column: &NumericColumn) -> Vec<NormalizationWarning> {
    render_robust_warnings(&[robust_warning_events(column)])
}

fn warning_groups(columns: &[(String, NumericColumn)]) -> Vec<Vec<&NumericColumn>> {
    let mut groups: Vec<Vec<&NumericColumn>> = Vec::new();
    for (_, column) in columns {
        if column.dtype.is_nullable() {
            groups.push(vec![column]);
        } else if let Some(group) = groups.iter_mut().find(|group| {
            !group[0].dtype.is_nullable() && group[0].dtype.physical() == column.dtype.physical()
        }) {
            group.push(column);
        } else {
            groups.push(vec![column]);
        }
    }
    groups
}

fn frame_warnings(
    columns: &[(String, NumericColumn)],
    method: WarningMethod,
) -> Vec<NormalizationWarning> {
    let groups = warning_groups(columns);
    match method {
        WarningMethod::Zscore => render_zscore_warnings(
            &groups
                .iter()
                .map(|group| {
                    group
                        .iter()
                        .fold(ZscoreWarnings::default(), |mut all, column| {
                            let event = zscore_warning_events(
                                &values_as_f64(column.values()),
                                series_precision(&column.dtype),
                            );
                            all.mean.overflow |= event.mean.overflow;
                            all.mean.invalid |= event.mean.invalid;
                            all.subtract_invalid |= event.subtract_invalid;
                            all.deviation_mean.overflow |= event.deviation_mean.overflow;
                            all.deviation_mean.invalid |= event.deviation_mean.invalid;
                            all.square_overflow |= event.square_overflow;
                            all.deviation_sum.overflow |= event.deviation_sum.overflow;
                            all.deviation_sum.invalid |= event.deviation_sum.invalid;
                            all.cast_overflow |= event.cast_overflow;
                            if column.dtype.is_nullable() {
                                all.subtract_invalid = false;
                                all.deviation_mean = ReductionWarnings::default();
                                all.square_overflow = false;
                                all.deviation_sum = ReductionWarnings::default();
                                all.cast_overflow = false;
                            }
                            all
                        })
                })
                .collect::<Vec<_>>(),
        ),
        WarningMethod::Robust => render_robust_warnings(
            &groups
                .iter()
                .map(|group| {
                    group
                        .iter()
                        .fold(RobustWarnings::default(), |mut all, column| {
                            let event = robust_warning_events(column);
                            all.center.overflow |= event.center.overflow;
                            all.center.invalid |= event.center.invalid;
                            all.mad.overflow |= event.mad.overflow;
                            all.mad.invalid |= event.mad.invalid;
                            all
                        })
                })
                .collect::<Vec<_>>(),
        ),
    }
}

fn render_robust_warnings(events: &[RobustWarnings]) -> Vec<NormalizationWarning> {
    let mut output = Vec::new();
    for event in events {
        append_reduction_warnings(&mut output, event.center);
    }
    for event in events {
        append_reduction_warnings(&mut output, event.mad);
    }
    output
}

fn is_supported(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::Int8
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
    )
}

fn series_precision(dtype: &PandasNumericDtype) -> Precision {
    match dtype.physical() {
        DataType::Float16 => Precision::F16,
        DataType::Float32 => Precision::F32,
        _ => Precision::F64,
    }
}

fn frame_native_precision(columns: &[(String, NumericColumn)]) -> Option<Precision> {
    columns
        .iter()
        .filter(|(_, column)| !column.dtype.is_nullable())
        .map(|(_, column)| series_precision(&column.dtype))
        .max()
}

#[allow(clippy::cast_possible_truncation)]
fn normalize_column(column: &NumericColumn, precision: Precision, method: Method) -> NumericColumn {
    let input = values_as_f64(column.values());
    let mut output = match method {
        Method::Zscore => standardize(&input, precision),
        Method::Robust(post_zscore) => robust_standardize(&input, precision, post_zscore),
    };
    let nullable = column.dtype.is_nullable();
    if nullable {
        for value in &mut output {
            if value.is_some_and(f64::is_nan) {
                *value = None;
            }
        }
    }
    let physical = match precision {
        Precision::F16 => DataType::Float16,
        Precision::F32 => DataType::Float32,
        Precision::F64 => DataType::Float64,
    };
    let values: ArrayRef = match precision {
        Precision::F16 => Arc::new(Float16Array::from(
            output
                .into_iter()
                .map(|value| value.map(f16::from_f64))
                .collect::<Vec<_>>(),
        )),
        Precision::F32 => Arc::new(Float32Array::from(
            output
                .into_iter()
                .map(|value| value.map(|item| item as f32))
                .collect::<Vec<_>>(),
        )),
        Precision::F64 => Arc::new(Float64Array::from(output)),
    };
    NumericColumn {
        values,
        dtype: PandasNumericDtype { physical, nullable },
    }
}

fn values_as_f64(array: &ArrayRef) -> Vec<Option<f64>> {
    let cast = arrow_cast::cast(array, &DataType::Float64)
        .expect("validated primitive numeric arrays always cast to Float64");
    cast.as_any()
        .downcast_ref::<Float64Array>()
        .expect("Float64 cast produces Float64Array")
        .iter()
        .collect()
}

#[allow(clippy::cast_possible_truncation)]
fn quantize(value: f64, precision: Precision) -> f64 {
    match precision {
        Precision::F16 => f16::from_f64(value).to_f64(),
        Precision::F32 => f64::from(value as f32),
        Precision::F64 => value,
    }
}

fn is_missing(value: Option<f64>) -> bool {
    value.is_none_or(f64::is_nan)
}

fn pairwise_sum(values: &[f64], precision: Precision) -> f64 {
    const BLOCK: usize = 128;
    if values.len() < 8 {
        return values.iter().fold(-0.0, |sum, value| {
            quantize(sum + quantize(*value, precision), precision)
        });
    }
    if values.len() <= BLOCK {
        let mut sums = [0.0; 8];
        for (slot, value) in sums.iter_mut().zip(values) {
            *slot = quantize(*value, precision);
        }
        let mut consumed = 8;
        while consumed + 8 <= values.len() {
            for (slot, value) in sums.iter_mut().zip(&values[consumed..consumed + 8]) {
                *slot = quantize(*slot + quantize(*value, precision), precision);
            }
            consumed += 8;
        }
        let left = quantize(
            quantize(sums[0] + sums[1], precision) + quantize(sums[2] + sums[3], precision),
            precision,
        );
        let right = quantize(
            quantize(sums[4] + sums[5], precision) + quantize(sums[6] + sums[7], precision),
            precision,
        );
        return values[consumed..]
            .iter()
            .fold(quantize(left + right, precision), |sum, value| {
                quantize(sum + quantize(*value, precision), precision)
            });
    }
    let mut middle = values.len() / 2;
    middle -= middle % 8;
    quantize(
        pairwise_sum(&values[..middle], precision) + pairwise_sum(&values[middle..], precision),
        precision,
    )
}

#[allow(clippy::cast_precision_loss)]
fn mean(values: &[Option<f64>], precision: Precision) -> f64 {
    let count = values.iter().filter(|value| !is_missing(**value)).count();
    if count == 0 {
        return f64::NAN;
    }
    let replaced = values
        .iter()
        .map(|value| value.filter(|item| !item.is_nan()).unwrap_or(0.0))
        .collect::<Vec<_>>();
    let count = quantize(count as f64, precision);
    quantize(pairwise_sum(&replaced, precision) / count, precision)
}

#[allow(clippy::cast_precision_loss)]
fn standard_deviation(values: &[Option<f64>], precision: Precision) -> f64 {
    let count = values.iter().filter(|value| !is_missing(**value)).count();
    if count < 2 {
        return f64::NAN;
    }
    let replaced = values
        .iter()
        .map(|value| value.filter(|item| !item.is_nan()).unwrap_or(0.0))
        .collect::<Vec<_>>();
    let average = pairwise_sum(&replaced, Precision::F64) / count as f64;
    let squares = values
        .iter()
        .map(|value| {
            value.map_or(0.0, |item| {
                if item.is_nan() {
                    0.0
                } else {
                    (average - item).powi(2)
                }
            })
        })
        .collect::<Vec<_>>();
    let variance = pairwise_sum(&squares, Precision::F64) / (count - 1) as f64;
    quantize(quantize(variance, precision).sqrt(), precision)
}

fn standardize(values: &[Option<f64>], precision: Precision) -> Vec<Option<f64>> {
    let mean = mean(values, precision);
    let deviation = standard_deviation(values, precision);
    values
        .iter()
        .map(|value| {
            value.map(|item| quantize(quantize(item - mean, precision) / deviation, precision))
        })
        .collect()
}

#[allow(clippy::manual_midpoint)]
fn median(values: &[Option<f64>], precision: Precision) -> f64 {
    let mut values = values
        .iter()
        .flatten()
        .copied()
        .filter(|value| !value.is_nan())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        quantize(
            quantize(values[middle - 1] + values[middle], precision) / 2.0,
            precision,
        )
    } else {
        values[middle]
    }
}

fn robust_standardize(
    values: &[Option<f64>],
    precision: Precision,
    post_zscore: bool,
) -> Vec<Option<f64>> {
    let center = median(values, precision);
    let centered: Vec<_> = values
        .iter()
        .map(|value| value.map(|item| quantize(item - center, precision)))
        .collect();
    let absolute: Vec<_> = centered
        .iter()
        .map(|value| value.map(|item| quantize(item.abs(), precision)))
        .collect();
    let mad = median(&absolute, precision);
    let clipped: Vec<_> = centered
        .into_iter()
        .map(|value| {
            value.map(|item| {
                let scaled = quantize(
                    quantize(item / mad, precision) / quantize(1.4826, precision),
                    precision,
                );
                quantize(scaled.clamp(-3.0, 3.0), precision)
            })
        })
        .collect();
    if post_zscore {
        standardize(&clipped, precision)
    } else {
        clipped
    }
}
