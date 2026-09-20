//! Explicit Pandas masked numeric/bool identity over Arrow validity buffers.
use super::{BuiltinFrameValue as B, TemporalFrameValue as T, temporal_cast};
use arrow_array::ArrayRef;
use arrow_schema::{ArrowError, DataType, Field};
use thiserror::Error;

pub(super) const DTYPE_KEY: &str = "domain.pandas.masked_dtype";

#[derive(Debug, Error)]
pub enum NullableIndexError {
    #[error("unsupported masked index storage: {0}")]
    Unsupported(DataType),
    #[error("invalid masked index storage: {0}")]
    Storage(&'static str),
    #[error(transparent)]
    Arrow(#[from] ArrowError),
}

fn dtype_name(dtype: &DataType) -> Result<&'static str, NullableIndexError> {
    Ok(match dtype {
        DataType::Int8 => "Int8",
        DataType::Int16 => "Int16",
        DataType::Int32 => "Int32",
        DataType::Int64 => "Int64",
        DataType::UInt8 => "UInt8",
        DataType::UInt16 => "UInt16",
        DataType::UInt32 => "UInt32",
        DataType::UInt64 => "UInt64",
        DataType::Float32 => "Float32",
        DataType::Float64 => "Float64",
        DataType::Boolean => "boolean",
        _ => return Err(NullableIndexError::Unsupported(dtype.clone())),
    })
}

/// A masked index, distinct from an ordinary nullable Arrow numeric array.
/// Valid NaNs remain valid values; only Arrow null bits represent `pd.NA`.
#[derive(Clone, Debug)]
pub struct NullableIndexDescriptor {
    array: ArrayRef,
    name: &'static str,
    values: Vec<T>,
}

impl PartialEq for NullableIndexDescriptor {
    fn eq(&self, other: &Self) -> bool {
        self.array.to_data() == other.array.to_data()
    }
}
impl Eq for NullableIndexDescriptor {}

impl NullableIndexDescriptor {
    /// Import already-typed physical values and validity, without NaN inference.
    /// # Errors
    /// Rejects unsupported dtypes and scalar conversion failures.
    pub fn new(array: ArrayRef) -> Result<Self, NullableIndexError> {
        Self::build(array, &mut temporal_cast::values)
    }

    fn build(
        array: ArrayRef,
        decode: &mut dyn FnMut(&ArrayRef) -> Result<Vec<T>, ArrowError>,
    ) -> Result<Self, NullableIndexError> {
        let name = dtype_name(array.data_type())?;
        let mut values = decode(&array)?;
        if values.len() != array.len() {
            return Err(NullableIndexError::Storage("decoder changed row count"));
        }
        for (i, value) in values.iter_mut().enumerate() {
            if array.is_null(i) {
                *value = T::Builtin(B::PandasNa);
            }
        }
        Ok(Self {
            array,
            name,
            values,
        })
    }

    /// Import storage with this adapter's explicit dtype-bearing field metadata.
    /// # Errors
    /// Rejects missing/mismatched tags, fields, nullability and unsupported types.
    pub fn from_storage(array: &ArrayRef, field: &Field) -> Result<Self, NullableIndexError> {
        if array.data_type() != field.data_type() {
            return Err(NullableIndexError::Storage(
                "field dtype does not match array",
            ));
        }
        if array.null_count() > 0 && !field.is_nullable() {
            return Err(NullableIndexError::Storage(
                "non-nullable field contains nulls",
            ));
        }
        let tag = field
            .metadata()
            .get(DTYPE_KEY)
            .ok_or(NullableIndexError::Storage("missing masked dtype metadata"))?;
        let result = Self::new(array.clone())?;
        if tag != result.name {
            return Err(NullableIndexError::Storage(
                "masked dtype metadata does not match array",
            ));
        }
        Ok(result)
    }

    #[must_use]
    pub const fn dtype_name(&self) -> &'static str {
        self.name
    }

    /// Scalar iteration preserves `pd.NA` separately from valid NaNs.
    #[must_use]
    pub fn values(&self) -> &[T] {
        &self.values
    }

    #[must_use]
    pub fn storage(&self) -> ArrayRef {
        self.array.clone()
    }

    /// Keep this owned-adapter tag with the array during Arrow IPC interchange.
    #[must_use]
    pub fn field(&self, name: &str) -> Field {
        Field::new(name, self.array.data_type().clone(), true)
            .with_metadata([(DTYPE_KEY.into(), self.name.into())].into())
    }
}

#[cfg(test)]
#[path = "nullable_index_tests.rs"]
pub(super) mod tests;
