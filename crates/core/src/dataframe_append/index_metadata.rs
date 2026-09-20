//! Index identity that cannot be inferred from an Arrow value buffer alone.
use super::{
    CategoricalIndexDescriptor, DataframeAppendError, MultiIndexDescriptor,
    NullableIndexDescriptor, StringIndexDescriptor, tuple_frame_array,
};
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch};
use arrow_schema::DataType;
use num_bigint::BigInt;
use num_traits::{Signed, ToPrimitive, Zero};
use std::sync::Arc;

/// Imported row-index metadata. Ordinary arrays retain the family implied by dtype.
/// Other index families require their own validated metadata adapters.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum IndexMetadata {
    /// No explicit range identity; an integer array is an ordinary Index.
    #[default]
    Array,
    /// Validated explicit levels/codes; row storage is the canonical tuple array.
    Multi(Arc<MultiIndexDescriptor>),
    /// Validated dictionary storage, including unused categories and ordering.
    Categorical(Arc<CategoricalIndexDescriptor>),
    /// Explicit masked numeric/bool identity, not inferred from null bits.
    Nullable(Arc<NullableIndexDescriptor>),
    /// Explicit string storage identity and missing semantics over canonical text.
    String(Arc<StringIndexDescriptor>),
    /// A Python-style half-open integer range, including its original empty bounds.
    Range {
        start: BigInt,
        stop: BigInt,
        step: BigInt,
    },
}

impl IndexMetadata {
    pub(super) fn from_column(
        batch: &RecordBatch,
        position: usize,
    ) -> Result<Self, DataframeAppendError> {
        let index = batch.column(position);
        if batch
            .schema()
            .field(position)
            .metadata()
            .contains_key(super::string_index::DTYPE_KEY)
        {
            return Ok(Self::String(Arc::new(StringIndexDescriptor::from_storage(
                index,
                batch.schema().field(position),
            )?)));
        }
        if batch
            .schema()
            .field(position)
            .metadata()
            .contains_key(super::nullable_index::DTYPE_KEY)
        {
            let descriptor =
                NullableIndexDescriptor::from_storage(index, batch.schema().field(position))
                    .map_err(|error| DataframeAppendError::NullableIndex(error.to_string()))?;
            return Ok(Self::Nullable(Arc::new(descriptor)));
        }
        if matches!(index.data_type(), DataType::Dictionary(_, _)) {
            let descriptor =
                CategoricalIndexDescriptor::from_storage(index, batch.schema().field(position))
                    .map_err(|error| DataframeAppendError::CategoricalIndex(error.to_string()))?;
            Ok(Self::Categorical(Arc::new(descriptor)))
        } else {
            Ok(Self::Array)
        }
    }

    pub(super) fn validate(&self, index: &ArrayRef) -> Result<(), DataframeAppendError> {
        if let Self::String(descriptor) = self {
            if descriptor.storage().to_data() != index.to_data() {
                return Err(DataframeAppendError::IndexMetadata(
                    "StringIndex descriptor does not match storage",
                ));
            }
            return Ok(());
        }
        if let Self::Nullable(descriptor) = self {
            if descriptor.storage().to_data() != index.to_data() {
                return Err(DataframeAppendError::IndexMetadata(
                    "NullableIndex descriptor does not match storage",
                ));
            }
            return Ok(());
        }
        if let Self::Categorical(descriptor) = self {
            if descriptor.storage().to_data() != index.to_data() {
                return Err(DataframeAppendError::IndexMetadata(
                    "CategoricalIndex descriptor does not match dictionary storage",
                ));
            }
            return Ok(());
        }
        if let Self::Multi(descriptor) = self {
            let expected = tuple_frame_array(&descriptor.values())?;
            if expected.to_data() != index.to_data() {
                return Err(DataframeAppendError::IndexMetadata(
                    "MultiIndex descriptor does not match tuple storage",
                ));
            }
            return Ok(());
        }
        let Self::Range { start, stop, step } = self else {
            return Ok(());
        };
        if step.is_zero() {
            return Err(DataframeAppendError::IndexMetadata(
                "range step must not be zero",
            ));
        }
        let values = index.as_any().downcast_ref::<Int64Array>().ok_or(
            DataframeAppendError::IndexMetadata("range requires Int64 storage"),
        )?;
        // Python range bounds can exceed i64 even when every stored value fits.
        let distance = if step.is_positive() {
            stop - start
        } else {
            start - stop
        };
        let count: BigInt = if distance <= BigInt::ZERO {
            BigInt::ZERO
        } else {
            1 + (distance - 1) / step.abs()
        };
        if count.to_usize() != Some(values.len()) {
            return Err(DataframeAppendError::IndexMetadata(
                "range bounds do not match index length",
            ));
        }
        if values
            .iter()
            .enumerate()
            .any(|(position, value)| value.map(BigInt::from) != Some(start + step * position))
        {
            return Err(DataframeAppendError::IndexMetadata(
                "range values do not match bounds",
            ));
        }
        Ok(())
    }
}
