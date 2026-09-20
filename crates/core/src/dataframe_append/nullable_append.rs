//! Masked axis concatenation; preserve mask identity separately from native NaNs.
use super::{
    BuiltinFrameValue as B, DataframeAppendError as Error, FrameOperations, IndexMetadata,
    NullableIndexDescriptor, TemporalFrameValue as T, TupleFrameValue as V, tuple_frame_dtype,
    tuple_index,
};
use arrow_array::{ArrayRef, BooleanArray};
use arrow_schema::DataType;
use std::sync::Arc;

fn masked(metadata: &IndexMetadata) -> bool {
    matches!(metadata, IndexMetadata::Nullable(_))
}

fn dtype<'a>(array: &'a ArrayRef, metadata: &'a IndexMetadata) -> &'a DataType {
    match metadata {
        IndexMetadata::Categorical(category) => category.categories().data_type(),
        _ => array.data_type(),
    }
}

fn category_codes(
    category: &super::CategoricalIndexDescriptor,
    nullable: &NullableIndexDescriptor,
    operations: &mut dyn FrameOperations,
) -> Result<Option<Vec<i64>>, Error> {
    let cd = category.categories().data_type();
    let storage = nullable.storage();
    let nd = storage.data_type();
    if category.nullable_categories().is_some()
        && cd == &DataType::Boolean
        && storage.null_count() > 0
    {
        return Ok(None);
    }
    if matches!(
        cd,
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64
    ) && matches!(
        nd,
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64
    ) && storage.null_count() > 0
    {
        if storage.null_count() == storage.len() {
            // The source's other.min() >= 0 evaluates pd.NA here and raises
            // TypeError, which categorical concatenation catches as fallback.
            return Ok(None);
        }
        if category.nullable_categories().is_none()
            && !nullable
                .values()
                .iter()
                .any(|v| matches!(v, T::Builtin(B::Int(i)) if *i < 0))
        {
            return Err(Error::CategoricalIndex(
                "cannot convert NA to integer".into(),
            ));
        }
    }
    let unsigned_mask = match cd {
        DataType::UInt8 => Some(u64::from(u8::MAX)),
        DataType::UInt16 => Some(u64::from(u16::MAX)),
        DataType::UInt32 => Some(u64::from(u32::MAX)),
        DataType::UInt64 => Some(u64::MAX),
        _ => None,
    };
    let downcast = unsigned_mask.filter(|_| {
        category.nullable_categories().is_some()
            && matches!(
                nd,
                DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64
            )
            && !nullable
                .values()
                .iter()
                .any(|v| matches!(v, T::Builtin(B::Int(i)) if *i < 0))
    });
    let original = nullable
        .values()
        .iter()
        .map(|v| super::object_factorization::key(&V::Scalar(v.clone())))
        .collect::<Result<Vec<_>, _>>()?;
    let incompatible = (super::categorical_append::boolean(category.category_values())
        && nd != &DataType::Boolean)
        || ((cd.is_integer() || cd.is_floating()) && nd == &DataType::Boolean);
    // Index lookup coerces both operands to its common dtype. The native
    // uint64/signed-extension case instead uses object equality; reversing
    // those dtypes does not take that special case in Pandas.
    // Both descriptors have already rejected unsupported numeric storage
    // (in particular Float16Index), unlike the raw-array join boundary.
    let common = common(cd, nd).expect("validated categorical/masked numeric dtypes");
    let float_lookup = matches!(common, Some(DataType::Float32 | DataType::Float64))
        && !(category.nullable_categories().is_none()
            && cd == &DataType::UInt64
            && matches!(
                nd,
                DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64
            ));
    let (keys, values) = if let Some(mask) = downcast {
        (
            category.keys().to_vec(),
            unsigned_lookup_keys(&storage, mask, operations)?,
        )
    } else if float_lookup {
        let dtype = common.as_ref().expect("floating lookup dtype");
        (
            cast_keys(category.categories(), dtype, operations)?,
            cast_keys(&storage, dtype, operations)?,
        )
    } else {
        (category.keys().to_vec(), original.clone())
    };
    let lookup: std::collections::HashMap<_, _> = keys
        .iter()
        .enumerate()
        .map(|(i, key)| (key, i64::try_from(i).expect("category length fits i64")))
        .collect();
    let mut codes = Vec::with_capacity(storage.len());
    for (i, value) in values.into_iter().enumerate() {
        let code = if incompatible || storage.is_null(i) {
            None
        } else {
            lookup.get(&value).copied()
        };
        match code {
            Some(_) if downcast.is_some() && original[i] != value => return Ok(None),
            Some(code) => codes.push(code),
            // Source membership recognizes masked NA; nullable comparisons
            // yield NA for unmatched category values and all() skips those.
            None if storage.null_count() > 0 => codes.push(-1),
            None => return Ok(None),
        }
    }
    Ok(Some(codes))
}

fn unsigned_lookup_keys(
    storage: &ArrayRef,
    mask: u64,
    operations: &mut dyn FrameOperations,
) -> Result<Vec<super::object_factorization::Key>, Error> {
    let wide = operations.cast(storage, &DataType::UInt64)?;
    let wide = wide
        .as_any()
        .downcast_ref::<arrow_array::UInt64Array>()
        .ok_or_else(|| Error::NullableIndex("unsigned lookup cast changed dtype".into()))?;
    if wide.len() != storage.len() {
        return Err(Error::NullableIndex(
            "unsigned lookup cast changed array length".into(),
        ));
    }
    // Masked astype uses NumPy's wrapping integer cast, preserving its mask.
    // Reuse Arrow's bitwise kernel after widening nonnegative signed values.
    let wrapped = arrow_arith::bitwise::bitwise_and_scalar(wide, mask)
        .expect("primitive bitwise-and has no failure path");
    Ok(wrapped
        .iter()
        .map(|value| {
            value.map_or(super::object_factorization::Key::Na, |value| {
                super::object_factorization::Key::Integer(value.into())
            })
        })
        .collect())
}

pub(super) fn cast_keys(
    array: &ArrayRef,
    dtype: &DataType,
    operations: &mut dyn FrameOperations,
) -> Result<Vec<super::object_factorization::Key>, Error> {
    cast_keys_with(array, dtype, operations, tuple_index::values)
}

fn cast_keys_with(
    array: &ArrayRef,
    dtype: &DataType,
    operations: &mut dyn FrameOperations,
    decode: fn(&ArrayRef) -> Result<Vec<V>, arrow_schema::ArrowError>,
) -> Result<Vec<super::object_factorization::Key>, Error> {
    let array = operations.cast(array, dtype)?;
    Ok(decode(&array)?
        .iter()
        .map(super::object_factorization::key)
        .collect::<Result<_, _>>()?)
}

fn common(left: &DataType, right: &DataType) -> Result<Option<DataType>, Error> {
    let numeric = |dtype: &DataType| dtype.is_integer() || dtype.is_floating();
    if left == &DataType::Boolean && right == &DataType::Boolean {
        Ok(Some(DataType::Boolean))
    } else if numeric(left) && numeric(right) {
        match super::numeric::common(left, right) {
            Some(dtype) => Ok(Some(dtype)),
            None => super::joined_type(left, right).map(Some),
        }
    } else {
        Ok(None)
    }
}

fn boxed(
    array: &ArrayRef,
    metadata: &IndexMetadata,
    operations: &mut dyn FrameOperations,
) -> Result<ArrayRef, Error> {
    let cells = match metadata {
        IndexMetadata::Nullable(descriptor) => {
            descriptor.values().iter().cloned().map(V::Scalar).collect()
        }
        IndexMetadata::Categorical(category) => category
            .values()
            .into_iter()
            .zip(category.codes())
            .map(|(value, code)| {
                if category.nullable_categories().is_some() && code == -1 {
                    V::Scalar(T::Builtin(B::PandasNa))
                } else {
                    value
                }
            })
            .collect(),
        _ => tuple_index::values(array)?,
    };
    Ok(operations.tuple_array(&cells)?)
}

fn recover_category_precision(
    category: &super::CategoricalIndexDescriptor,
    rounded: &ArrayRef,
) -> Result<bool, Error> {
    let values = tuple_index::values(rounded)?;
    let mut maximum: Option<(usize, f64)> = None;
    for (i, value) in values.iter().enumerate() {
        if rounded.is_null(i) {
            continue;
        }
        let V::Scalar(T::Builtin(B::Float(value))) = value else {
            return Err(Error::NullableIndex(
                "categorical numeric materialization changed dtype".into(),
            ));
        };
        if maximum.is_none_or(|(_, previous)| *value > previous) {
            maximum = Some((i, *value));
        }
    }
    let Some((i, value)) = maximum else {
        return Ok(false);
    };
    // Pandas checks only nanargmax (first maximum), then falls back to its
    // original object representation if that one integer lost precision.
    Ok(super::object_factorization::key(&category.values()[i])?
        != super::object_factorization::key(&V::Scalar(T::Builtin(B::Float(value))))
            .expect("a floating scalar always has a factorization key"))
}

fn cast(
    array: &ArrayRef,
    metadata: &IndexMetadata,
    dtype: &DataType,
    operations: &mut dyn FrameOperations,
) -> Result<ArrayRef, Error> {
    if let IndexMetadata::Categorical(category) = metadata {
        // Extension construction first materializes the categorical's NumPy
        // array. Missing integer codes promote that array to float64 before
        // the requested masked dtype is applied, including precision loss.
        let materialized =
            if category.categories().data_type().is_integer() && category.codes().contains(&-1) {
                &DataType::Float64
            } else {
                dtype
            };
        let values = operations.cast(category.categories(), materialized)?;
        let codes = arrow_array::Int64Array::from_iter(
            category.codes().into_iter().map(|c| (c >= 0).then_some(c)),
        );
        let values = arrow_select::take::take(
            values.as_ref(),
            &codes,
            Some(arrow_select::take::TakeOptions { check_bounds: true }),
        )?;
        let values = if materialized == dtype {
            values
        } else {
            let converted = if category.nullable_categories().is_some()
                && dtype.is_integer()
                && recover_category_precision(category, &values)?
            {
                let original = operations.cast(category.categories(), dtype)?;
                arrow_select::take::take(
                    original.as_ref(),
                    &codes,
                    Some(arrow_select::take::TakeOptions { check_bounds: true }),
                )?
            } else {
                operations.cast(&values, dtype)?
            };
            if dtype.is_integer() && converted.null_count() > values.null_count() {
                return Err(Error::NullableIndex("int too big to convert".into()));
            }
            converted
        };
        // Extension construction masks NumPy NaNs, including a valid NaN
        // category that only became missing during this materialization.
        return cast_with(
            &values,
            &IndexMetadata::Array,
            dtype,
            operations,
            super::temporal_cast::values,
        );
    }
    cast_with(
        array,
        metadata,
        dtype,
        operations,
        super::temporal_cast::values,
    )
}

fn cast_with(
    array: &ArrayRef,
    metadata: &IndexMetadata,
    dtype: &DataType,
    operations: &mut dyn FrameOperations,
    decode: fn(&ArrayRef) -> Result<Vec<T>, arrow_schema::ArrowError>,
) -> Result<ArrayRef, Error> {
    let converted = operations.cast(array, dtype)?;
    if !masked(metadata) && array.data_type().is_floating() {
        // NumPy -> masked-array construction masks NaN. A masked-array cast
        // instead preserves an existing valid NaN and its independent mask.
        let cells = decode(array)?;
        let validity = BooleanArray::from(
            cells
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    !array.is_null(i) && !matches!(v,T::Builtin(B::Float(f)) if f.is_nan())
                })
                .collect::<Vec<_>>(),
        );
        return Ok(arrow_array::make_array(
            converted
                .to_data()
                .into_builder()
                .nulls(Some(validity.values().clone().into()))
                .build()?,
        ));
    }
    Ok(converted)
}

pub(super) fn join(
    left: &ArrayRef,
    lm: &IndexMetadata,
    right: &ArrayRef,
    rm: &IndexMetadata,
    operations: &mut dyn FrameOperations,
    warning: &mut dyn FnMut(&str),
) -> Result<(ArrayRef, IndexMetadata), Error> {
    if !masked(lm) && !masked(rm) {
        return super::categorical_append::join(left, lm, right, rm, operations, warning);
    }
    if let (IndexMetadata::Categorical(category), IndexMetadata::Nullable(nullable)) = (lm, rm) {
        if let Some(codes) = category_codes(category, nullable, operations)? {
            return super::categorical_append::merged(category, codes);
        }
    }
    fallback(left, lm, right, rm, operations, warning)
}

pub(super) fn fallback(
    left: &ArrayRef,
    lm: &IndexMetadata,
    right: &ArrayRef,
    rm: &IndexMetadata,
    operations: &mut dyn FrameOperations,
    warning: &mut dyn FnMut(&str),
) -> Result<(ArrayRef, IndexMetadata), Error> {
    if matches!(lm, IndexMetadata::Multi(_)) {
        // NumPy materialization promotes masked integers with missing values
        // to float, whereas boolean arrays retain object-valued pd.NA.
        let values = if right.null_count() > 0 && right.data_type().is_integer() {
            operations.cast(right, &DataType::Float64)?
        } else if right.null_count() > 0 && right.data_type() == &DataType::Boolean {
            boxed(right, rm, operations)?
        } else {
            right.clone()
        };
        return super::multi_append::join(left, &values, operations);
    }
    let dtype = common(dtype(left, lm), dtype(right, rm))?;
    let equal = masked(lm) && masked(rm) && left.data_type() == right.data_type();
    if !equal && left.is_empty() != right.is_empty() {
        let (array, metadata) = if left.is_empty() {
            (right, rm)
        } else {
            (left, lm)
        };
        // A masked target differs from a native dtype even at the same width.
        // A single extension array uses the source's None target sentinel.
        if dtype.is_some()
            || masked(metadata)
            || !(super::blocks::logical_object(array.data_type())
                || array.data_type() == &tuple_frame_dtype())
        {
            warning(super::EMPTY_INDEX_WARNING);
        }
        return Ok((
            super::categorical_append::infer_fallback(
                array,
                matches!(lm, IndexMetadata::Categorical(_)),
                operations,
            )?,
            metadata.clone(),
        ));
    }
    if let Some(dtype) = dtype {
        let left = cast(left, lm, &dtype, operations)?;
        let right = cast(right, rm, &dtype, operations)?;
        let output = operations.concat(&left, &right)?;
        let descriptor = NullableIndexDescriptor::new(output.clone())
            .map_err(|error| Error::NullableIndex(error.to_string()))?;
        Ok((output, IndexMetadata::Nullable(Arc::new(descriptor))))
    } else {
        let left = boxed(left, lm, operations)?;
        let right = boxed(right, rm, operations)?;
        let output = operations.concat(&left, &right)?;
        Ok((
            super::categorical_append::infer_fallback(
                &output,
                matches!(lm, IndexMetadata::Categorical(_)),
                operations,
            )?,
            IndexMetadata::Array,
        ))
    }
}

#[cfg(test)]
#[path = "nullable_append_tests.rs"]
pub(super) mod tests;
