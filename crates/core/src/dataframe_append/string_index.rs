//! Explicit `StringDtype` identity using the existing lossless Arrow text codec.
use super::{BuiltinFrameValue as V, builtin_frame_values};
use arrow_array::ArrayRef;
use arrow_schema::{ArrowError, Field};
use thiserror::Error;

pub(super) const DTYPE_KEY: &str = "domain.pandas.string_dtype";

/// Source storage identity, independent of this adapter's Arrow interchange codec.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StringStorage {
    Python,
    PyArrow,
}

/// `StringDtype` has two observably different scalar missing-value conventions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StringMissing {
    PandasNa,
    NaN,
}

#[derive(Debug, Error)]
pub enum StringIndexError {
    #[error("invalid string index storage: {0}")]
    Storage(&'static str),
    #[error("{reason}")]
    Unicode { row: usize, reason: String },
    #[error(transparent)]
    Arrow(#[from] ArrowError),
}

/// Typed string index with lossless Python text and explicit missing semantics.
/// Canonical storage uses the existing built-in object union: Text or None only.
/// This is an interchange representation, not an inference from an object index.
#[derive(Clone, Debug)]
pub struct StringIndexDescriptor {
    array: ArrayRef,
    storage: StringStorage,
    missing: StringMissing,
    values: Vec<V>,
}

impl PartialEq for StringIndexDescriptor {
    fn eq(&self, other: &Self) -> bool {
        self.storage == other.storage
            && self.missing == other.missing
            && self.array.to_data() == other.array.to_data()
    }
}
impl Eq for StringIndexDescriptor {}

impl StringIndexDescriptor {
    /// Import already-typed text or missing cells, without Python string coercion.
    /// # Errors
    /// Rejects non-text cells, malformed Arrow payloads and PyArrow-incompatible text.
    pub fn new(
        array: ArrayRef,
        storage: StringStorage,
        missing: StringMissing,
    ) -> Result<Self, StringIndexError> {
        Self::build(array, storage, missing, &mut builtin_frame_values)
    }

    fn build(
        array: ArrayRef,
        storage: StringStorage,
        missing: StringMissing,
        decode: &mut dyn FnMut(&ArrayRef) -> Result<Vec<V>, ArrowError>,
    ) -> Result<Self, StringIndexError> {
        let mut values = decode(&array)?;
        if values.len() != array.len() {
            return Err(StringIndexError::Storage("decoder changed row count"));
        }
        for (row, value) in values.iter_mut().enumerate() {
            match value {
                V::Text(text) => {
                    if storage == StringStorage::PyArrow {
                        text.to_utf8().map_err(|_| unicode_error(text, row))?;
                    }
                }
                V::None => {
                    *value = match missing {
                        StringMissing::PandasNa => V::PandasNa,
                        StringMissing::NaN => V::Float(f64::NAN),
                    };
                }
                _ => return Err(StringIndexError::Storage("expected text or None cell")),
            }
        }
        Ok(Self {
            array,
            storage,
            missing,
            values,
        })
    }

    /// Restore explicit identity from this adapter's Arrow field metadata.
    /// # Errors
    /// Rejects absent/unknown metadata, schema mismatch and invalid canonical cells.
    pub fn from_storage(array: &ArrayRef, field: &Field) -> Result<Self, StringIndexError> {
        if array.data_type() != field.data_type() {
            return Err(StringIndexError::Storage(
                "field dtype does not match array",
            ));
        }
        let tag = field
            .metadata()
            .get(DTYPE_KEY)
            .ok_or(StringIndexError::Storage("missing string dtype metadata"))?;
        let (storage, missing) = match tag.as_str() {
            "python:NA" => (StringStorage::Python, StringMissing::PandasNa),
            "python:nan" => (StringStorage::Python, StringMissing::NaN),
            "pyarrow:NA" => (StringStorage::PyArrow, StringMissing::PandasNa),
            "pyarrow:nan" => (StringStorage::PyArrow, StringMissing::NaN),
            _ => return Err(StringIndexError::Storage("unknown string dtype metadata")),
        };
        let result = Self::new(array.clone(), storage, missing)?;
        // Arrow unions have no top-level null bitmap; inspect canonical cells.
        if !field.is_nullable() && result.values.iter().any(|v| !matches!(v, V::Text(_))) {
            return Err(StringIndexError::Storage(
                "non-nullable field contains missing cells",
            ));
        }
        Ok(result)
    }

    #[must_use]
    pub const fn storage_kind(&self) -> StringStorage {
        self.storage
    }

    #[must_use]
    pub const fn missing_kind(&self) -> StringMissing {
        self.missing
    }

    #[must_use]
    pub const fn dtype_name(&self) -> &'static str {
        match self.missing {
            StringMissing::PandasNa => "string",
            StringMissing::NaN => "str",
        }
    }

    #[must_use]
    pub fn values(&self) -> &[V] {
        &self.values
    }

    #[must_use]
    pub fn storage(&self) -> ArrayRef {
        self.array.clone()
    }

    /// Retain this tag when exchanging the canonical union through Arrow IPC.
    #[must_use]
    pub fn field(&self, name: &str) -> Field {
        let tag = match (self.storage, self.missing) {
            (StringStorage::Python, StringMissing::PandasNa) => "python:NA",
            (StringStorage::Python, StringMissing::NaN) => "python:nan",
            (StringStorage::PyArrow, StringMissing::PandasNa) => "pyarrow:NA",
            (StringStorage::PyArrow, StringMissing::NaN) => "pyarrow:nan",
        };
        Field::new(name, self.array.data_type().clone(), true)
            .with_metadata([(DTYPE_KEY.into(), tag.into())].into())
    }
}

pub(super) fn unicode_error(text: &crate::RlCheckpointText, row: usize) -> StringIndexError {
    let points = text.as_code_points();
    let start = points
        .iter()
        .position(|p| (0xd800..=0xdfff).contains(p))
        .expect("validated Python text only fails UTF-8 conversion on surrogates");
    let count = points[start..]
        .iter()
        .take_while(|p| (0xd800..=0xdfff).contains(*p))
        .count();
    let description = if count == 1 {
        format!("character '\\u{:04x}' in position {start}", points[start])
    } else {
        format!("characters in position {start}-{}", start + count - 1)
    };
    StringIndexError::Unicode {
        row,
        reason: format!("'utf-8' codec can't encode {description}: surrogates not allowed"),
    }
}

#[cfg(test)]
#[path = "string_index_tests.rs"]
mod tests;
