//! Explicit categorical identity over canonical Arrow dictionary storage.
use super::{
    BuiltinFrameValue as B, DataframeAppendError, NullableIndexDescriptor, TemporalFrameValue as T,
    TupleFrameValue as V, index,
    object_factorization::{Key, factorize_with_keys},
    temporal_cast, tuple_index,
};
use arrow_array::{Array, ArrayRef, DictionaryArray, Int64Array, types::Int64Type};
use arrow_schema::{ArrowError, Field};
use std::sync::Arc;
use thiserror::Error;

const MASKED_CATEGORIES: &str = "domain.pandas.category.masked_dtype";

/// Validation failures before a categorical descriptor is published.
#[derive(Debug, Error)]
pub enum CategoricalIndexError {
    #[error("Categorical categories cannot be null")]
    NullCategories,
    #[error("Categorical categories must be unique")]
    DuplicateCategories,
    #[error("codes need to be between -1 and len(categories)-1")]
    Codes,
    #[error("invalid categorical storage: {0}")]
    Storage(&'static str),
    #[error(transparent)]
    Index(#[from] DataframeAppendError),
    #[error(transparent)]
    Arrow(#[from] ArrowError),
    #[error(transparent)]
    Nullable(#[from] super::NullableIndexError),
}

/// Categories (including unused entries), codes and ordering retained together.
/// Canonical storage uses i64 Arrow dictionary keys; null keys represent code -1.
/// The category dtype is never inferred from the materialized row values.
#[derive(Clone, Debug)]
pub struct CategoricalIndexDescriptor {
    array: DictionaryArray<Int64Type>,
    ordered: bool,
    materialized_categories: Vec<V>,
    category_keys: Vec<Key>,
    nullable_categories: Option<NullableIndexDescriptor>,
}

impl PartialEq for CategoricalIndexDescriptor {
    fn eq(&self, other: &Self) -> bool {
        self.ordered == other.ordered
            && self.nullable_categories == other.nullable_categories
            && self.array.to_data() == other.array.to_data()
    }
}

impl Eq for CategoricalIndexDescriptor {}

// Keep validation available at the owned-cell boundary as well as after Arrow
// decoding, so future category constructors share identical error ordering.
fn validate_values(values: &[V]) -> Result<Vec<Key>, CategoricalIndexError> {
    if values
        .iter()
        .any(|v| matches!(v, V::Scalar(v) if temporal_cast::missing(v)))
    {
        return Err(CategoricalIndexError::NullCategories);
    }
    let (factorization, keys) = factorize_with_keys(values, false)?;
    if factorization.uniques.len() != values.len() {
        return Err(CategoricalIndexError::DuplicateCategories);
    }
    Ok(keys)
}

impl CategoricalIndexDescriptor {
    /// Import explicit categories/codes in source validation order.
    /// # Errors
    /// Rejects missing/duplicate categories before checking code bounds; rejects
    /// malformed or unsupported category arrays through the existing adapters.
    pub fn new(
        categories: ArrayRef,
        codes: Vec<i64>,
        ordered: bool,
    ) -> Result<Self, CategoricalIndexError> {
        index::validate(categories.data_type())?;
        let values = tuple_index::values(&categories)?;
        let category_keys = validate_values(&values)?;
        Self::assemble(categories, codes, ordered, values, category_keys, None)
    }

    /// Import explicitly masked numeric/bool categories. A valid NaN is a category,
    /// while a masked slot is rejected before uniqueness and code validation.
    /// # Errors
    /// Rejects null/duplicate categories, invalid scalar keys and out-of-range codes.
    pub fn from_nullable_categories(
        categories: NullableIndexDescriptor,
        codes: Vec<i64>,
        ordered: bool,
    ) -> Result<Self, CategoricalIndexError> {
        if categories.storage().null_count() > 0 {
            return Err(CategoricalIndexError::NullCategories);
        }
        let values = categories
            .values()
            .iter()
            .cloned()
            .map(V::Scalar)
            .collect::<Vec<_>>();
        let (factored, keys) = factorize_with_keys(&values, false)?;
        if factored.uniques.len() != values.len() {
            return Err(CategoricalIndexError::DuplicateCategories);
        }
        Self::assemble(
            categories.storage(),
            codes,
            ordered,
            values,
            keys,
            Some(categories),
        )
    }

    fn assemble(
        categories: ArrayRef,
        codes: Vec<i64>,
        ordered: bool,
        values: Vec<V>,
        category_keys: Vec<Key>,
        nullable_categories: Option<NullableIndexDescriptor>,
    ) -> Result<Self, CategoricalIndexError> {
        let keys =
            Int64Array::from_iter(codes.into_iter().map(|code| (code != -1).then_some(code)));
        Ok(Self {
            // Arrow's only constructor failure here is an out-of-range key.
            // Reuse that validation and retain the source's public error text.
            array: DictionaryArray::try_new(keys, categories)
                .map_err(|_| CategoricalIndexError::Codes)?,
            ordered,
            materialized_categories: values,
            category_keys,
            nullable_categories,
        })
    }

    /// Validate a canonical dictionary array and its ordering-bearing field.
    /// # Errors
    /// Rejects mismatched fields, non-i64 dictionary keys and invalid categories.
    pub fn from_storage(array: &ArrayRef, field: &Field) -> Result<Self, CategoricalIndexError> {
        if array.data_type() != field.data_type() {
            return Err(CategoricalIndexError::Storage(
                "field dtype does not match array",
            ));
        }
        let dictionary = array
            .as_any()
            .downcast_ref::<DictionaryArray<Int64Type>>()
            .ok_or(CategoricalIndexError::Storage(
                "expected canonical Int64 dictionary",
            ))?;
        let codes = dictionary
            .keys()
            .iter()
            .map(|code| code.unwrap_or(-1))
            .collect();
        let ordered = field.dict_is_ordered().unwrap_or(false);
        let mut result = if let Some(tag) = field.metadata().get(MASKED_CATEGORIES) {
            let categories = NullableIndexDescriptor::new(dictionary.values().clone())?;
            if categories.dtype_name() != tag {
                return Err(CategoricalIndexError::Storage(
                    "masked category dtype does not match array",
                ));
            }
            Self::from_nullable_categories(categories, codes, ordered)?
        } else {
            Self::new(dictionary.values().clone(), codes, ordered)?
        };
        // Preserve physical null slots and slice offsets of a validated import.
        result.array = dictionary.clone();
        Ok(result)
    }

    #[must_use]
    pub fn categories(&self) -> &ArrayRef {
        self.array.values()
    }

    /// Explicit masked category identity, never inferred from dictionary buffers.
    #[must_use]
    pub fn nullable_categories(&self) -> Option<&NullableIndexDescriptor> {
        self.nullable_categories.as_ref()
    }

    pub(super) fn category_values(&self) -> &[V] {
        &self.materialized_categories
    }

    pub(super) fn keys(&self) -> &[Key] {
        &self.category_keys
    }

    #[must_use]
    pub fn codes(&self) -> Vec<i64> {
        self.array
            .keys()
            .iter()
            .map(|code| code.unwrap_or(-1))
            .collect()
    }

    #[must_use]
    pub const fn ordered(&self) -> bool {
        self.ordered
    }

    /// Scalar iteration preserves category identity; it is not `NumPy` boxing.
    #[must_use]
    pub fn values(&self) -> Vec<V> {
        let missing = if self.nullable_categories.is_some()
            && self.categories().data_type() == &arrow_schema::DataType::Boolean
        {
            B::PandasNa
        } else if temporal_cast::is_temporal(self.categories().data_type()) {
            B::NotATime
        } else {
            B::Float(f64::NAN)
        };
        self.array
            .keys_iter()
            .map(|code| {
                code.map_or_else(
                    || V::Scalar(T::Builtin(missing.clone())),
                    |code| self.materialized_categories[code].clone(),
                )
            })
            .collect()
    }

    #[must_use]
    pub fn storage(&self) -> ArrayRef {
        Arc::new(self.array.clone())
    }

    /// Keep this field alongside storage when serializing ordered categories.
    #[must_use]
    pub fn field(&self, name: &str) -> Field {
        let field = Field::new(name, self.array.data_type().clone(), true)
            .with_dict_is_ordered(self.ordered);
        self.nullable_categories
            .as_ref()
            .map_or(field.clone(), |categories| {
                field.with_metadata(
                    [(MASKED_CATEGORIES.into(), categories.dtype_name().into())].into(),
                )
            })
    }
}

#[cfg(test)]
#[path = "categorical_index_tests.rs"]
mod tests;
