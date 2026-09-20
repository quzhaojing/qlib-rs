//! Long-short alpha-return calculation compatible with Qlib's evaluator.

use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    sync::Arc,
};

use arrow_array::{
    Array, ArrayRef, Float64Array, Int64Array, StringArray, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray, UInt64Array,
    new_empty_array,
};
use arrow_schema::{DataType, TimeUnit};
use arrow_select::concat::concat;
use num_traits::ToPrimitive;
use thiserror::Error;

/// A two-level, floating-point series accepted by [`calc_long_short_return`].
///
/// Index levels accept Arrow UTF-8, signed/unsigned 64-bit integer, and
/// timestamp arrays. Null level values model a missing `MultiIndex` label;
/// null numeric values and `NaN` both model Pandas floating missing values.
#[derive(Clone, Debug)]
pub struct AlphaSeries {
    level_names: [String; 2],
    levels: [AlphaIndexLevel; 2],
    values: Arc<Float64Array>,
}

#[derive(Clone, Debug)]
enum AlphaIndexLevel {
    Utf8(Arc<StringArray>),
    Int64(Arc<Int64Array>),
    UInt64(Arc<UInt64Array>),
    TimestampSecond(Arc<TimestampSecondArray>),
    TimestampMillisecond(Arc<TimestampMillisecondArray>),
    TimestampMicrosecond(Arc<TimestampMicrosecondArray>),
    TimestampNanosecond(Arc<TimestampNanosecondArray>),
}

impl AlphaIndexLevel {
    fn from_array(array: &ArrayRef) -> Result<Self, AlphaReturnError> {
        let unsupported = || AlphaReturnError::UnsupportedIndexType(array.data_type().clone());
        Ok(match array.data_type() {
            DataType::Utf8 => Self::Utf8(Arc::new(
                array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(unsupported)?
                    .clone(),
            )),
            DataType::Int64 => Self::Int64(Arc::new(
                array
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(unsupported)?
                    .clone(),
            )),
            DataType::UInt64 => Self::UInt64(Arc::new(
                array
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(unsupported)?
                    .clone(),
            )),
            DataType::Timestamp(TimeUnit::Second, _) => Self::TimestampSecond(Arc::new(
                array
                    .as_any()
                    .downcast_ref::<TimestampSecondArray>()
                    .ok_or_else(unsupported)?
                    .clone(),
            )),
            DataType::Timestamp(TimeUnit::Millisecond, _) => Self::TimestampMillisecond(Arc::new(
                array
                    .as_any()
                    .downcast_ref::<TimestampMillisecondArray>()
                    .ok_or_else(unsupported)?
                    .clone(),
            )),
            DataType::Timestamp(TimeUnit::Microsecond, _) => Self::TimestampMicrosecond(Arc::new(
                array
                    .as_any()
                    .downcast_ref::<TimestampMicrosecondArray>()
                    .ok_or_else(unsupported)?
                    .clone(),
            )),
            DataType::Timestamp(TimeUnit::Nanosecond, _) => Self::TimestampNanosecond(Arc::new(
                array
                    .as_any()
                    .downcast_ref::<TimestampNanosecondArray>()
                    .ok_or_else(unsupported)?
                    .clone(),
            )),
            _ => return Err(unsupported()),
        })
    }

    fn data_type(&self) -> &DataType {
        match self {
            Self::Utf8(array) => array.data_type(),
            Self::Int64(array) => array.data_type(),
            Self::UInt64(array) => array.data_type(),
            Self::TimestampSecond(array) => array.data_type(),
            Self::TimestampMillisecond(array) => array.data_type(),
            Self::TimestampMicrosecond(array) => array.data_type(),
            Self::TimestampNanosecond(array) => array.data_type(),
        }
    }

    fn is_null(&self, row: usize) -> bool {
        match self {
            Self::Utf8(array) => array.is_null(row),
            Self::Int64(array) => array.is_null(row),
            Self::UInt64(array) => array.is_null(row),
            Self::TimestampSecond(array) => array.is_null(row),
            Self::TimestampMillisecond(array) => array.is_null(row),
            Self::TimestampMicrosecond(array) => array.is_null(row),
            Self::TimestampNanosecond(array) => array.is_null(row),
        }
    }

    fn slice(&self, row: usize) -> ArrayRef {
        match self {
            Self::Utf8(array) => Arc::new(array.slice(row, 1)),
            Self::Int64(array) => Arc::new(array.slice(row, 1)),
            Self::UInt64(array) => Arc::new(array.slice(row, 1)),
            Self::TimestampSecond(array) => Arc::new(array.slice(row, 1)),
            Self::TimestampMillisecond(array) => Arc::new(array.slice(row, 1)),
            Self::TimestampMicrosecond(array) => Arc::new(array.slice(row, 1)),
            Self::TimestampNanosecond(array) => Arc::new(array.slice(row, 1)),
        }
    }
}

impl AlphaSeries {
    /// Builds a validated immutable indexed series.
    ///
    /// # Errors
    /// Returns a shape, duplicate-name, or unsupported-index-type error before
    /// publishing a partially validated series.
    ///
    pub fn try_new(
        level_names: [String; 2],
        levels: [ArrayRef; 2],
        values: Arc<Float64Array>,
    ) -> Result<Self, AlphaReturnError> {
        let length = values.len();
        if levels.iter().any(|level| level.len() != length) {
            return Err(AlphaReturnError::LengthMismatch);
        }
        if level_names[0] == level_names[1] {
            return Err(AlphaReturnError::DuplicateLevelName(level_names[0].clone()));
        }
        let [first, second] = levels.map(|level| AlphaIndexLevel::from_array(&level));
        let levels = [first?, second?];
        Ok(Self {
            level_names,
            levels,
            values,
        })
    }

    /// Returns the level names in physical index order.
    #[must_use]
    pub fn level_names(&self) -> &[String; 2] {
        &self.level_names
    }

    /// Returns the immutable numeric storage.
    #[must_use]
    pub fn values(&self) -> &Arc<Float64Array> {
        &self.values
    }
}

/// A date-indexed floating-point result series.
#[derive(Clone, Debug)]
pub struct AlphaReturnSeries {
    index_name: String,
    name: Option<String>,
    dates: ArrayRef,
    values: Arc<Float64Array>,
}

impl AlphaReturnSeries {
    /// Name of the source level used for grouping.
    #[must_use]
    pub fn index_name(&self) -> &str {
        &self.index_name
    }

    /// Pandas-compatible series name (`None` for long-short, `label` for average).
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Sorted, non-missing group keys.
    #[must_use]
    pub fn dates(&self) -> &ArrayRef {
        &self.dates
    }

    /// Group results. Empty selections and all-missing means are `NaN`.
    #[must_use]
    pub fn values(&self) -> &Arc<Float64Array> {
        &self.values
    }
}

/// Pair returned by Qlib's `calc_long_short_return`.
#[derive(Clone, Debug)]
pub struct AlphaReturns {
    /// Half of the long-minus-short return for every date.
    pub long_short: AlphaLongShortReturn,
    /// Mean label return for every date.
    pub average: AlphaReturnSeries,
}

/// Dynamic first return value produced by the upstream Pandas implementation.
#[derive(Clone, Debug)]
pub enum AlphaLongShortReturn {
    /// The ordinary date-indexed long-short series.
    Series(AlphaReturnSeries),
    /// The anomalous empty DataFrame returned when no non-missing group exists.
    EmptyFrame {
        /// Source DataFrame columns, in their observed order.
        columns: [String; 2],
        /// Arrow dtypes of the empty source columns.
        column_types: [DataType; 2],
        /// Typed empty grouping index retained by Pandas.
        index: ArrayRef,
        /// Name of the grouping index.
        index_name: String,
    },
}

impl AlphaLongShortReturn {
    /// Returns the ordinary series representation, when groups exist.
    #[must_use]
    pub const fn as_series(&self) -> Option<&AlphaReturnSeries> {
        match self {
            Self::Series(series) => Some(series),
            Self::EmptyFrame { .. } => None,
        }
    }

    /// Returns the empty DataFrame columns for the no-group source edge.
    #[must_use]
    pub fn empty_frame_columns(&self) -> Option<&[String; 2]> {
        match self {
            Self::Series(_) => None,
            Self::EmptyFrame { columns, .. } => Some(columns),
        }
    }

    /// Returns the empty DataFrame column dtypes for the no-group source edge.
    #[must_use]
    pub fn empty_frame_column_types(&self) -> Option<&[DataType; 2]> {
        match self {
            Self::Series(_) => None,
            Self::EmptyFrame { column_types, .. } => Some(column_types),
        }
    }

    /// Returns the named grouping index for the empty DataFrame source edge.
    #[must_use]
    pub fn empty_frame_index(&self) -> Option<(&str, &ArrayRef)> {
        match self {
            Self::Series(_) => None,
            Self::EmptyFrame {
                index, index_name, ..
            } => Some((index_name, index)),
        }
    }
}

/// Errors at the deliberately narrow native boundary.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum AlphaReturnError {
    /// Index level and value arrays do not have identical lengths.
    #[error("index levels and values must have identical lengths")]
    LengthMismatch,
    /// Pandas cannot resolve a level name that occurs more than once.
    #[error("index level name is duplicated: {0}")]
    DuplicateLevelName(String),
    /// Prediction and label index schemas must agree at this boundary.
    #[error("prediction and label index level names differ")]
    IndexNamesDiffer,
    /// Corresponding levels must use the same lossless Arrow representation.
    #[error("prediction and label index level types differ")]
    IndexTypesDiffer,
    /// The native boundary does not silently coerce arbitrary Python objects.
    #[error("unsupported index level type: {0}")]
    UnsupportedIndexType(DataType),
    /// The requested grouping level is not present.
    #[error("index level not found: {0}")]
    DateLevelNotFound(String),
    /// Pandas cannot align two incompatible non-unique `MultiIndex` values.
    #[error("cannot handle incompatible non-unique indexes")]
    NonUniqueAlignment,
    /// Non-finite or unrepresentably large selection sizes fail.
    #[error("quantile must produce a finite non-negative selection size")]
    InvalidQuantile,
}

#[derive(Clone, Debug, Hash, Ord, PartialOrd, PartialEq, Eq)]
enum IndexValue {
    Text(String),
    Signed(i64),
    Unsigned(u64),
    Timestamp(i64),
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct IndexKey([Option<IndexValue>; 2]);

#[derive(Clone, Copy, Debug)]
struct Row {
    pred: f64,
    label: f64,
}

fn index_value(level: &AlphaIndexLevel, row: usize) -> Option<IndexValue> {
    if level.is_null(row) {
        return None;
    }
    Some(match level {
        AlphaIndexLevel::Utf8(array) => IndexValue::Text(array.value(row).to_owned()),
        AlphaIndexLevel::Int64(array) => IndexValue::Signed(array.value(row)),
        AlphaIndexLevel::UInt64(array) => IndexValue::Unsigned(array.value(row)),
        AlphaIndexLevel::TimestampSecond(array) => IndexValue::Timestamp(array.value(row)),
        AlphaIndexLevel::TimestampMillisecond(array) => IndexValue::Timestamp(array.value(row)),
        AlphaIndexLevel::TimestampMicrosecond(array) => IndexValue::Timestamp(array.value(row)),
        AlphaIndexLevel::TimestampNanosecond(array) => IndexValue::Timestamp(array.value(row)),
    })
}

fn key(series: &AlphaSeries, row: usize) -> IndexKey {
    IndexKey(std::array::from_fn(|level| {
        index_value(&series.levels[level], row)
    }))
}

fn value(series: &AlphaSeries, row: usize) -> f64 {
    if series.values.is_null(row) {
        f64::NAN
    } else {
        series.values.value(row)
    }
}

fn indexed_values(series: &AlphaSeries) -> Vec<(IndexKey, f64)> {
    (0..series.values.len())
        .map(|row| (key(series, row), value(series, row)))
        .collect()
}

fn is_unique(rows: &[(IndexKey, f64)]) -> bool {
    let mut keys = HashSet::with_capacity(rows.len());
    rows.iter().all(|(key, _)| keys.insert(key))
}

fn missing_value() -> f64 {
    f64::NAN
}

fn align_unique(pred: &[(IndexKey, f64)], label: &[(IndexKey, f64)]) -> Vec<(IndexKey, Row)> {
    let mut keys = pred
        .iter()
        .chain(label)
        .map(|row| row.0.clone())
        .collect::<Vec<_>>();
    let pred = pred.iter().cloned().collect::<HashMap<_, _>>();
    let label = label.iter().cloned().collect::<HashMap<_, _>>();
    keys.sort_by(compare_keys);
    keys.dedup();
    keys.into_iter()
        .map(|key| {
            let row = Row {
                pred: pred.get(&key).copied().unwrap_or_else(missing_value),
                label: label.get(&key).copied().unwrap_or_else(missing_value),
            };
            (key, row)
        })
        .collect()
}

fn align_to_non_unique(
    target: &[(IndexKey, f64)],
    unique: &[(IndexKey, f64)],
    target_is_pred: bool,
) -> Result<Vec<(IndexKey, Row)>, AlphaReturnError> {
    let unique = unique.iter().cloned().collect::<HashMap<_, _>>();
    if unique
        .keys()
        .any(|key| !target.iter().any(|row| &row.0 == key))
    {
        return Err(AlphaReturnError::NonUniqueAlignment);
    }
    Ok(target
        .iter()
        .map(|(key, target_value)| {
            let other = unique.get(key).copied().unwrap_or_else(missing_value);
            let row = if target_is_pred {
                Row {
                    pred: *target_value,
                    label: other,
                }
            } else {
                Row {
                    pred: other,
                    label: *target_value,
                }
            };
            (key.clone(), row)
        })
        .collect())
}

fn compare_keys(left: &IndexKey, right: &IndexKey) -> Ordering {
    left.0
        .iter()
        .zip(&right.0)
        .map(|(left, right)| match (left, right) {
            (Some(left), Some(right)) => left.cmp(right),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => Ordering::Equal,
        })
        .find(|ordering| *ordering != Ordering::Equal)
        .unwrap_or(Ordering::Equal)
}

fn align(
    pred: &AlphaSeries,
    label: &AlphaSeries,
) -> Result<Vec<(IndexKey, Row)>, AlphaReturnError> {
    let pred = indexed_values(pred);
    let label = indexed_values(label);
    if pred
        .iter()
        .map(|row| &row.0)
        .eq(label.iter().map(|row| &row.0))
    {
        return Ok(pred
            .into_iter()
            .zip(label)
            .map(|((key, pred), (_, label))| (key, Row { pred, label }))
            .collect());
    }
    match (is_unique(&pred), is_unique(&label)) {
        (true, true) => Ok(align_unique(&pred, &label)),
        (false, true) => align_to_non_unique(&pred, &label, true),
        (true, false) => align_to_non_unique(&label, &pred, false),
        (false, false) => Err(AlphaReturnError::NonUniqueAlignment),
    }
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let (sum, count) = values
        .filter(|value| !value.is_nan())
        .fold((0.0, 0_usize), |(sum, count), value| {
            (sum + value, count + 1)
        });
    if count == 0 {
        f64::NAN
    } else {
        sum / count
            .to_f64()
            .expect("array length is representable as f64")
    }
}

fn ranked_mean(rows: &[Row], count: usize, largest: bool) -> f64 {
    let mut order = (0..rows.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| {
        let left = rows[left].pred;
        let right = rows[right].pred;
        match (left.is_nan(), right.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) if largest => right.partial_cmp(&left).unwrap_or(Ordering::Equal),
            (false, false) => left.partial_cmp(&right).unwrap_or(Ordering::Equal),
        }
    });
    mean(order.into_iter().take(count).map(|row| rows[row].label))
}

fn selection_size(length: usize, quantile: f64) -> Result<usize, AlphaReturnError> {
    let size = length
        .to_f64()
        .expect("array length is representable as f64")
        * quantile;
    if !size.is_finite() {
        return Err(AlphaReturnError::InvalidQuantile);
    }
    if size.is_sign_negative() {
        return Ok(0);
    }
    size.trunc()
        .to_usize()
        .ok_or(AlphaReturnError::InvalidQuantile)
}

fn date_cells(
    pred: &AlphaSeries,
    label: &AlphaSeries,
    date_level: usize,
) -> HashMap<IndexValue, ArrayRef> {
    let mut cells = HashMap::new();
    for series in [pred, label] {
        for row in 0..series.values.len() {
            if let Some(date) = index_value(&series.levels[date_level], row) {
                cells
                    .entry(date)
                    .or_insert_with(|| series.levels[date_level].slice(row));
            }
        }
    }
    cells
}

/// Calculates Qlib's daily long-short and average label returns.
///
/// Alignment follows the relevant Pandas constructor rules: equal indexes are
/// paired positionally, unique unequal indexes form a sorted outer union, and
/// a unique index may broadcast onto an otherwise-compatible duplicated one.
/// Group ordering is sorted by the selected date level. Ranking is
/// stable, includes missing predictions only after every finite prediction,
/// and Pandas-style means skip missing labels.
///
/// # Errors
/// Returns typed schema, alignment, grouping-level, empty-input, or quantile
/// errors without mutating either input.
///
/// # Panics
/// Panics only if Arrow fails to concatenate same-typed, validated one-element
/// slices taken from the input date level.
pub fn calc_long_short_return(
    pred: &AlphaSeries,
    label: &AlphaSeries,
    date_col: &str,
    quantile: f64,
    dropna: bool,
) -> Result<AlphaReturns, AlphaReturnError> {
    if pred.level_names != label.level_names {
        return Err(AlphaReturnError::IndexNamesDiffer);
    }
    if pred
        .levels
        .iter()
        .zip(&label.levels)
        .any(|(pred, label)| pred.data_type() != label.data_type())
    {
        return Err(AlphaReturnError::IndexTypesDiffer);
    }
    let date_level = pred
        .level_names
        .iter()
        .position(|name| name == date_col)
        .ok_or_else(|| AlphaReturnError::DateLevelNotFound(date_col.to_owned()))?;
    if !quantile.is_finite() {
        return Err(AlphaReturnError::InvalidQuantile);
    }

    let date_cells = date_cells(pred, label, date_level);
    let mut groups = HashMap::<IndexValue, Vec<Row>>::new();
    for (index, row) in align(pred, label)? {
        let Some(date) = index.0[date_level].clone() else {
            continue;
        };
        if dropna && (row.pred.is_nan() || row.label.is_nan()) {
            continue;
        }
        groups.entry(date).or_default().push(row);
    }
    let mut dates = groups.keys().cloned().collect::<Vec<_>>();
    dates.sort();
    let mut long_short = Vec::with_capacity(dates.len());
    let mut average = Vec::with_capacity(dates.len());
    for date in &dates {
        let rows = &groups[date];
        let count = selection_size(rows.len(), quantile)?;
        let long = ranked_mean(rows, count, true);
        let short = ranked_mean(rows, count, false);
        long_short.push((long - short) / 2.0);
        average.push(mean(rows.iter().map(|row| row.label)));
    }
    let dates = if dates.is_empty() {
        new_empty_array(pred.levels[date_level].data_type())
    } else {
        let arrays = dates
            .iter()
            .map(|date| date_cells[date].as_ref())
            .collect::<Vec<_>>();
        concat(&arrays).expect("same-typed date cells always concatenate")
    };
    let series = |name: Option<&str>, values| AlphaReturnSeries {
        index_name: date_col.to_owned(),
        name: name.map(str::to_owned),
        dates: Arc::clone(&dates),
        values: Arc::new(Float64Array::from(values)),
    };
    let average = series(Some("label"), average);
    let long_short = if long_short.is_empty() {
        AlphaLongShortReturn::EmptyFrame {
            columns: ["pred".to_owned(), "label".to_owned()],
            column_types: [DataType::Float64, DataType::Float64],
            index: Arc::clone(&dates),
            index_name: date_col.to_owned(),
        }
    } else {
        AlphaLongShortReturn::Series(series(None, long_short))
    };
    Ok(AlphaReturns {
        long_short,
        average,
    })
}
