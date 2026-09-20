//! Validated explicit levels/codes and row values for Arrow-representable `MultiIndex` levels.
use super::{
    BuiltinFrameValue as B, CategoricalIndexDescriptor, DataframeAppendError,
    NullableIndexDescriptor, StringIndexDescriptor, StringMissing, TemporalFrameValue as T,
    TupleFrameValue as V, index, object_index_is_unique, temporal_cast, tuple_index,
};
use arrow_array::ArrayRef;
use arrow_schema::{ArrowError, DataType};
use std::{cmp::Ordering, collections::HashSet};
use thiserror::Error;

/// Errors at the typed explicit-level import boundary, before any descriptor is published.
#[derive(Debug, Error)]
pub enum MultiIndexError {
    #[error("Must pass non-zero number of levels/codes")]
    EmptyLevels,
    #[error("Length of levels and codes must be the same.")]
    LevelCount,
    #[error("Length of names must match number of levels in MultiIndex.")]
    NameCount,
    #[error("Unequal code lengths: {0:?}")]
    CodeLengths(Vec<usize>),
    #[error(
        "On level {level}, code max ({max}) >= length of level ({len}). NOTE: this index is in an inconsistent state"
    )]
    CodeTooLarge { level: usize, max: i64, len: usize },
    #[error("On level {level}, code value ({min}) < -1")]
    CodeTooNegative { level: usize, min: i64 },
    #[error("Level values must be unique on level {0}")]
    DuplicateLevel(usize),
    #[error(
        "Value for sortorder must be inferior or equal to actual lexsort_depth: sortorder {order} with lexsort_depth {depth}"
    )]
    SortOrder { order: i64, depth: usize },
    #[error(transparent)]
    Index(#[from] DataframeAppendError),
    #[error(transparent)]
    Arrow(#[from] ArrowError),
}

/// Explicit level values and normalized row codes, retaining unused levels and names.
/// Tuple-factorizing construction is performed by the frame append adapter.
/// Arrow dtype metadata is retained; extension-family/frequency metadata needs its
/// own index adapter rather than being guessed from the value buffers.
#[derive(Clone, Debug)]
pub struct MultiIndexDescriptor {
    levels: Vec<ArrayRef>,
    categorical_levels: Vec<Option<CategoricalIndexDescriptor>>,
    nullable_levels: Vec<Option<NullableIndexDescriptor>>,
    string_levels: Vec<Option<StringIndexDescriptor>>,
    codes: Vec<Vec<Option<usize>>>,
    names: Vec<V>,
    sortorder: Option<i64>,
    materialized: Vec<Vec<V>>,
}

/// Explicit level identity; dictionary ordering is never guessed from Arrow buffers.
#[derive(Clone, Debug)]
pub enum MultiIndexLevel {
    Array(ArrayRef),
    Categorical(CategoricalIndexDescriptor),
    Nullable(NullableIndexDescriptor),
    String(StringIndexDescriptor),
}

// Metadata equality is structural, not elementwise Python index comparison.
// Floating payload bits (including NaNs and signed zero) remain significant.
fn same_name(left: &V, right: &V) -> bool {
    match (left, right) {
        (V::Scalar(T::Builtin(B::Float(left))), V::Scalar(T::Builtin(B::Float(right)))) => {
            left.to_bits() == right.to_bits()
        }
        (V::Tuple(left), V::Tuple(right)) => {
            left.len() == right.len() && left.iter().zip(right).all(|(l, r)| same_name(l, r))
        }
        _ => left == right,
    }
}

impl PartialEq for MultiIndexDescriptor {
    fn eq(&self, other: &Self) -> bool {
        self.codes == other.codes
            && self.categorical_levels == other.categorical_levels
            && self.nullable_levels == other.nullable_levels
            && self.string_levels == other.string_levels
            && self.sortorder == other.sortorder
            && self.names.len() == other.names.len()
            && self
                .names
                .iter()
                .zip(&other.names)
                .all(|(l, r)| same_name(l, r))
            && self.levels.len() == other.levels.len()
            && self
                .levels
                .iter()
                .zip(&other.levels)
                .all(|(l, r)| l.to_data() == r.to_data())
    }
}

impl Eq for MultiIndexDescriptor {}

fn depth(codes: &[Vec<i64>]) -> usize {
    for prefix in (1..=codes.len()).rev() {
        if (1..codes[0].len()).all(|row| {
            codes[..prefix]
                .iter()
                .map(|c| c[row - 1])
                .cmp(codes[..prefix].iter().map(|c| c[row]))
                != Ordering::Greater
        }) {
            return prefix;
        }
    }
    0
}

fn missing(value: &V) -> bool {
    match value {
        V::Scalar(value) => temporal_cast::missing(value),
        V::Tuple(_) | V::PythonTemporal(_) => false,
    }
}

fn validate_level(
    level: usize,
    array: &ArrayRef,
    codes: &[i64],
    rows: usize,
    lengths: &[usize],
    category: Option<&CategoricalIndexDescriptor>,
    extension_values: Option<Vec<V>>,
) -> Result<Vec<V>, MultiIndexError> {
    if codes.len() != rows {
        return Err(MultiIndexError::CodeLengths(lengths.to_vec()));
    }
    if let Some(&max) = codes.iter().max() {
        if i128::from(max) >= i128::try_from(array.len()).expect("array length fits i128") {
            return Err(MultiIndexError::CodeTooLarge {
                level,
                max,
                len: array.len(),
            });
        }
    }
    if let Some(&min) = codes.iter().min() {
        if min < -1 {
            return Err(MultiIndexError::CodeTooNegative { level, min });
        }
    }
    let values = if let Some(category) = category {
        category.values()
    } else if let Some(values) = extension_values {
        values
    } else {
        index::validate(array.data_type())?;
        tuple_index::values(array)?
    };
    if let Some(category) = category {
        // A valid masked NaN category and the missing code are distinct entries,
        // even though iteration materializes both as NaN. Still validate cells.
        object_index_is_unique(&values)?;
        let codes = category.codes();
        if codes.iter().collect::<HashSet<_>>().len() != codes.len() {
            return Err(MultiIndexError::DuplicateLevel(level));
        }
        Ok(values)
    } else {
        unique_level(level, values)
    }
}

// Keep cell validation independent of Arrow decoding: it is also the boundary
// needed by level constructors that start with owned object cells.
fn unique_level(level: usize, values: Vec<V>) -> Result<Vec<V>, MultiIndexError> {
    if !object_index_is_unique(&values)? {
        return Err(MultiIndexError::DuplicateLevel(level));
    }
    Ok(values)
}

fn extension_level_values(
    nullable: Option<&NullableIndexDescriptor>,
    string: Option<&StringIndexDescriptor>,
) -> Option<Vec<V>> {
    nullable
        .map(|v| v.values().iter().cloned().map(V::Scalar).collect())
        .or_else(|| {
            string.map(|v| {
                v.values()
                    .iter()
                    .cloned()
                    .map(|v| V::Scalar(T::Builtin(v)))
                    .collect()
            })
        })
}

impl MultiIndexDescriptor {
    // Pandas's all-empty-tuple constructor keeps repeated empty tuple levels
    // with every code equal to zero and bypasses ordinary level uniqueness.
    pub(super) fn empty_tuple_rows(
        values: &[V],
        array: ArrayRef,
    ) -> Result<Self, DataframeAppendError> {
        let materialized = tuple_index::values(&array)?;
        if materialized != values {
            return Err(DataframeAppendError::IndexMetadata(
                "empty-tuple backend changed level values",
            ));
        }
        Ok(Self {
            levels: vec![array],
            categorical_levels: vec![None],
            nullable_levels: vec![None],
            string_levels: vec![None],
            codes: vec![vec![Some(0); values.len()]],
            names: vec![V::Scalar(T::Builtin(B::None))],
            sortorder: None,
            materialized: vec![materialized],
        })
    }

    /// `MultiIndex.append([])` retains levels/codes/names but resets sortorder.
    pub(super) fn appended_without_other(&self) -> Self {
        let mut result = self.clone();
        result.sortorder = None;
        result
    }

    /// Import already-typed levels, integral codes (-1 for missing), and hashable names.
    /// Checks code shape/bounds, per-level uniqueness and declared sort order before
    /// normalizing codes pointing at missing level entries. Inputs are never mutated.
    /// # Errors
    /// Returns structural/level errors or an Arrow conversion failure atomically.
    pub fn new(
        levels: Vec<ArrayRef>,
        codes: Vec<Vec<i64>>,
        names: Vec<V>,
        sortorder: Option<i64>,
    ) -> Result<Self, MultiIndexError> {
        Self::build(levels, codes, names, sortorder, &mut |array| {
            arrow_cast::cast(array, &DataType::Float64)
        })
    }

    /// Import ordinary, categorical, masked and string levels while preserving extension
    /// identity, unused values and ordering independently from row codes.
    /// # Errors
    /// Returns the same structural and uniqueness errors as [`Self::new`].
    pub fn from_levels(
        levels: Vec<MultiIndexLevel>,
        codes: Vec<Vec<i64>>,
        names: Vec<V>,
        sortorder: Option<i64>,
    ) -> Result<Self, MultiIndexError> {
        Self::build_typed(levels, codes, names, sortorder, &mut |array| {
            arrow_cast::cast(array, &DataType::Float64)
        })
    }

    fn build(
        levels: Vec<ArrayRef>,
        codes: Vec<Vec<i64>>,
        names: Vec<V>,
        sortorder: Option<i64>,
        cast_integer: &mut dyn FnMut(&ArrayRef) -> Result<ArrayRef, ArrowError>,
    ) -> Result<Self, MultiIndexError> {
        Self::build_typed(
            levels.into_iter().map(MultiIndexLevel::Array).collect(),
            codes,
            names,
            sortorder,
            cast_integer,
        )
    }

    fn build_typed(
        levels: Vec<MultiIndexLevel>,
        codes: Vec<Vec<i64>>,
        names: Vec<V>,
        sortorder: Option<i64>,
        cast_integer: &mut dyn FnMut(&ArrayRef) -> Result<ArrayRef, ArrowError>,
    ) -> Result<Self, MultiIndexError> {
        let (levels, metadata): (Vec<_>, Vec<_>) = levels
            .into_iter()
            .map(|level| match level {
                MultiIndexLevel::Array(array) => (array, (None, (None, None))),
                MultiIndexLevel::Categorical(category) => {
                    (category.storage(), (Some(category), (None, None)))
                }
                MultiIndexLevel::Nullable(nullable) => {
                    (nullable.storage(), (None, (Some(nullable), None)))
                }
                MultiIndexLevel::String(string) => (string.storage(), (None, (None, Some(string)))),
            })
            .unzip();
        let (categorical_levels, extensions): (Vec<_>, Vec<_>) = metadata.into_iter().unzip();
        let (nullable_levels, string_levels): (Vec<_>, Vec<_>) = extensions.into_iter().unzip();
        if levels.len() != codes.len() {
            return Err(MultiIndexError::LevelCount);
        }
        if levels.is_empty() {
            return Err(MultiIndexError::EmptyLevels);
        }
        if levels.len() != names.len() {
            return Err(MultiIndexError::NameCount);
        }
        // The owned name type is hashable; validate temporal payloads as for levels.
        object_index_is_unique(&names)?;
        let lengths = codes.iter().map(Vec::len).collect::<Vec<_>>();
        let mut materialized = levels
            .iter()
            .zip(&codes)
            .enumerate()
            .map(|(i, (level, code))| {
                validate_level(
                    i,
                    level,
                    code,
                    lengths[0],
                    &lengths,
                    categorical_levels[i].as_ref(),
                    extension_level_values(nullable_levels[i].as_ref(), string_levels[i].as_ref()),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let actual_depth = depth(&codes);
        if let Some(order) = sortorder {
            if i128::from(order) > i128::try_from(actual_depth).expect("level count fits i128") {
                return Err(MultiIndexError::SortOrder {
                    order,
                    depth: actual_depth,
                });
            }
        }
        let codes = codes
            .into_iter()
            .zip(&materialized)
            .enumerate()
            .map(|(level, (codes, values))| {
                codes
                    .into_iter()
                    .map(|code| {
                        usize::try_from(code).ok().filter(|&i| {
                            if nullable_levels[level].is_some()
                                || categorical_levels[level].is_some()
                            {
                                !levels[level].is_null(i)
                            } else {
                                !missing(&values[i])
                            }
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for (i, level) in levels.iter().enumerate() {
            // Pandas promotes the whole extracted integer level when a row has -1,
            // including otherwise-exact u64 values. Stored level metadata stays intact.
            let integer = categorical_levels[i].as_ref().map_or_else(
                || nullable_levels[i].is_none() && level.data_type().is_integer(),
                |category| {
                    // Inner missing codes force object extraction, preserving exact
                    // integers; otherwise outer missing codes promote to Float64.
                    category.categories().data_type().is_integer()
                        && !category.codes().contains(&-1)
                },
            );
            if integer && codes[i].contains(&None) {
                materialized[i] = tuple_index::values(&cast_integer(level)?)?;
            }
        }
        Ok(Self {
            levels,
            categorical_levels,
            nullable_levels,
            string_levels,
            codes,
            names,
            sortorder,
            materialized,
        })
    }

    #[must_use]
    pub fn levels(&self) -> &[ArrayRef] {
        &self.levels
    }
    /// Category descriptors aligned one-to-one with [`Self::levels`].
    #[must_use]
    pub fn categorical_levels(&self) -> &[Option<CategoricalIndexDescriptor>] {
        &self.categorical_levels
    }
    /// Masked descriptors aligned one-to-one with [`Self::levels`].
    #[must_use]
    pub fn nullable_levels(&self) -> &[Option<NullableIndexDescriptor>] {
        &self.nullable_levels
    }
    /// String descriptors aligned one-to-one with [`Self::levels`], retaining storage and NA identity.
    #[must_use]
    pub fn string_levels(&self) -> &[Option<StringIndexDescriptor>] {
        &self.string_levels
    }
    #[must_use]
    pub fn codes(&self) -> &[Vec<Option<usize>>] {
        &self.codes
    }
    #[must_use]
    pub fn names(&self) -> &[V] {
        &self.names
    }
    #[must_use]
    pub fn sortorder(&self) -> Option<i64> {
        self.sortorder
    }

    /// Materialize tuple rows while retaining dtype-specific missing values and
    /// integer-to-float promotion; the descriptor itself retains its original levels.
    #[must_use]
    pub fn values(&self) -> Vec<V> {
        (0..self.codes[0].len())
            .map(|row| {
                V::Tuple(
                    self.levels
                        .iter()
                        .enumerate()
                        .map(|(i, level)| {
                            self.codes[i][row].map_or_else(
                                || {
                                    V::Scalar(T::Builtin(
                                        if self.nullable_levels[i].is_some()
                                            || self.string_levels[i].as_ref().is_some_and(|s| {
                                                s.missing_kind() == StringMissing::PandasNa
                                            })
                                        {
                                            B::PandasNa
                                        } else if temporal_cast::is_temporal(level.data_type()) {
                                            B::NotATime
                                        } else {
                                            B::Float(f64::NAN)
                                        },
                                    ))
                                },
                                |code| self.materialized[i][code].clone(),
                            )
                        })
                        .collect(),
                )
            })
            .collect()
    }
}

#[cfg(test)]
#[path = "multi_index_tests.rs"]
mod tests;
