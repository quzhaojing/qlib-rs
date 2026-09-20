//! Categorical axis concatenation: recode compatible categories, otherwise materialize.
use super::{
    BuiltinFrameValue as B, CategoricalIndexDescriptor as Category, DataframeAppendError as Error,
    FrameOperations, IndexMetadata, TemporalFrameValue as T, TupleFrameValue as V, blocks, index,
    object_factorization::key, temporal_cast, tuple_frame_dtype, tuple_index,
};
use arrow_array::{ArrayRef, Int64Array};
use arrow_schema::DataType;
use std::{collections::HashMap, sync::Arc};

fn category(metadata: &IndexMetadata) -> Option<&Category> {
    match metadata {
        IndexMetadata::Categorical(value) => Some(value),
        _ => None,
    }
}

fn object(dtype: &DataType) -> bool {
    blocks::logical_object(dtype) || dtype == &tuple_frame_dtype()
}

fn dtype_equal(left: &DataType, right: &DataType) -> bool {
    left == right || (object(left) && object(right))
}

pub(super) fn boolean(values: &[V]) -> bool {
    !values.is_empty()
        && values
            .iter()
            .all(|value| matches!(value, V::Scalar(T::Builtin(B::Bool(_)))))
}

pub(super) fn merged(
    left: &Category,
    right_codes: Vec<i64>,
) -> Result<(ArrayRef, IndexMetadata), Error> {
    let mut codes = left.codes();
    codes.extend(right_codes);
    let result = if let Some(categories) = left.nullable_categories() {
        Category::from_nullable_categories(categories.clone(), codes, left.ordered())
    } else {
        Category::new(left.categories().clone(), codes, left.ordered())
    }
    .map_err(|error| Error::CategoricalIndex(error.to_string()))?;
    Ok((
        result.storage(),
        IndexMetadata::Categorical(Arc::new(result)),
    ))
}

fn compatible_codes(
    left: &Category,
    right: &ArrayRef,
    other: Option<&Category>,
    operations: &mut dyn FrameOperations,
) -> Result<Option<Vec<i64>>, Error> {
    let categories = left.category_values();
    let values = match other {
        Some(other) => other.values(),
        None => tuple_index::values(right)?,
    };
    let ld = left.categories().data_type();
    let rd = right.data_type();
    // Index._should_compare uses inferred boolean identity for whole indexes,
    // but real numeric identity from dtype. Mixed object cells still compare.
    if other.is_none()
        && !values.is_empty()
        && ((boolean(&values) && (ld.is_integer() || ld.is_floating()))
            || (boolean(categories) && (rd.is_integer() || rd.is_floating())))
    {
        return Ok(None);
    }
    if other.is_none()
        && !temporal_cast::is_temporal(ld)
        && values
            .iter()
            .any(|value| matches!(value, V::Scalar(T::Builtin(B::PandasNa))))
    {
        // Source elementwise comparison with pd.NA raises an ambiguous-truth
        // TypeError for these categories; CategoricalIndex falls back to Index.
        return Ok(None);
    }
    let keys = left.keys();
    if other.is_none() && left.nullable_categories().is_some() && object(rd) {
        // Object NaN can locate a valid NaN category but fails the source's
        // subsequent elementwise equality check. Boolean pd.NA comparisons
        // also reject NaN/NaT, while object None is accepted as missing.
        if values.iter().any(|value| match value {
            V::Scalar(T::Builtin(B::Float(value))) if value.is_nan() => {
                ld == &DataType::Boolean || keys.contains(&super::object_factorization::Key::Nan)
            }
            V::Scalar(T::Builtin(B::NotATime)) => ld == &DataType::Boolean,
            _ => false,
        }) {
            return Ok(None);
        }
    }
    // Numeric Index lookup finds a common floating dtype before matching.
    // Integer-only/object lookup must retain exact keys instead.
    let floating = if other.is_none()
        && (ld.is_integer() || ld.is_floating())
        && (rd.is_integer() || rd.is_floating())
        && (ld.is_floating() || rd.is_floating())
    {
        let dtype = common(ld, rd).expect("validated categorical/native numeric dtypes");
        let casted = (
            super::nullable_append::cast_keys(left.categories(), &dtype, operations)?,
            super::nullable_append::cast_keys(right, &dtype, operations)?,
        );
        if casted.0.len() != keys.len() || casted.1.len() != values.len() {
            return Err(Error::CategoricalIndex(
                "numeric lookup cast changed array length".into(),
            ));
        }
        Some(casted)
    } else {
        None
    };
    let keys = floating.as_ref().map_or(keys, |(keys, _)| keys);
    let lookup: HashMap<_, _> = keys
        .iter()
        .enumerate()
        .map(|(i, key)| (key, i64::try_from(i).expect("category length fits i64")))
        .collect();
    if let Some(other) = other {
        if left.ordered() != other.ordered()
            || left.nullable_categories().is_some() != other.nullable_categories().is_some()
            || !dtype_equal(
                left.categories().data_type(),
                other.categories().data_type(),
            )
        {
            return Ok(None);
        }
        let other_keys = other.keys();
        if keys.len() != other_keys.len() || (left.ordered() && keys != other_keys) {
            return Ok(None);
        }
        if other_keys.iter().any(|value| !lookup.contains_key(value)) {
            return Ok(None);
        }
    }
    let mut result = Vec::with_capacity(values.len());
    let other_codes = other.map(Category::codes);
    for (i, value) in values.into_iter().enumerate() {
        let key = key(&value)?;
        let key = floating.as_ref().map_or(&key, |(_, values)| &values[i]);
        if other_codes.as_ref().is_some_and(|codes| codes[i] == -1)
            || (other.is_none()
                && rd.is_floating()
                && matches!(&value, V::Scalar(T::Builtin(B::Float(value))) if value.is_nan()))
        {
            result.push(-1);
        } else if let Some(code) = lookup.get(key) {
            result.push(*code);
        } else if matches!(&value, V::Scalar(value) if temporal_cast::missing(value)) {
            result.push(-1);
        } else {
            return Ok(None);
        }
    }
    Ok(Some(result))
}

fn common(left: &DataType, right: &DataType) -> Result<DataType, Error> {
    if dtype_equal(left, right) {
        return Ok(left.clone());
    }
    if object(left) || object(right) || left == &DataType::Boolean || right == &DataType::Boolean {
        return Ok(tuple_frame_dtype());
    }
    if temporal_cast::is_temporal(left) || temporal_cast::is_temporal(right) {
        return Ok(match (left, right) {
            (DataType::Timestamp(_, _), DataType::Timestamp(_, _)) => {
                super::joined_type(left, right).unwrap_or_else(|_| tuple_frame_dtype())
            }
            (DataType::Duration(l), DataType::Duration(r)) => DataType::Duration((*l).max(*r)),
            _ => tuple_frame_dtype(),
        });
    }
    super::numeric::common(left, right).map_or_else(|| super::joined_type(left, right), Ok)
}

fn materialize(
    array: &ArrayRef,
    category: Option<&Category>,
    dtype: &DataType,
    operations: &mut dyn FrameOperations,
) -> Result<ArrayRef, Error> {
    if object(dtype) {
        let values = match category {
            Some(category) => category.values(),
            None => tuple_index::values(array)?,
        };
        return Ok(operations.tuple_array(&values)?);
    }
    if let Some(category) = category {
        // Categorical.astype uses NumPy's NaN-to-bool conversion when the
        // category set is empty, rather than take_nd's object missing value.
        if dtype == &DataType::Boolean && category.categories().is_empty() {
            let missing: ArrayRef =
                Arc::new(arrow_array::Float64Array::from(vec![
                    f64::NAN;
                    category.codes().len()
                ]));
            return Ok(operations.cast(&missing, dtype)?);
        }
        let values = operations.cast(category.categories(), dtype)?;
        let codes = Int64Array::from_iter(
            category
                .codes()
                .into_iter()
                .map(|code| (code >= 0).then_some(code)),
        );
        return Ok(arrow_select::take::take(
            values.as_ref(),
            &codes,
            Some(arrow_select::take::TakeOptions { check_bounds: true }),
        )?);
    }
    Ok(operations.cast(array, dtype)?)
}

pub(super) fn infer_fallback(
    array: &ArrayRef,
    categorical_left: bool,
    operations: &mut dyn FrameOperations,
) -> Result<ArrayRef, Error> {
    let inferred = index::infer(array, operations)?;
    // CategoricalIndex fallback calls Index, not Index._with_infer: only
    // temporal object inference remains, numeric object identity is retained.
    if categorical_left
        && object(array.data_type())
        && !temporal_cast::is_temporal(inferred.data_type())
    {
        Ok(array.clone())
    } else {
        Ok(inferred)
    }
}

fn ordered_categories_equal(
    left: &Category,
    right: &Category,
    operations: &mut dyn FrameOperations,
) -> Result<bool, Error> {
    if left.nullable_categories().is_some() || right.nullable_categories().is_some() {
        return Ok(
            left.nullable_categories().is_some() == right.nullable_categories().is_some()
                && left.categories().data_type() == right.categories().data_type()
                && left.keys() == right.keys(),
        );
    }
    if left.keys() != right.keys() {
        return Ok(false);
    }
    let ld = left.categories().data_type();
    let rd = right.categories().data_type();
    // DatetimeLikeIndex.equals requires equal temporal dtype, including units
    // and zone. It may first infer a temporal index from an object operand.
    if temporal_cast::is_temporal(ld) {
        let other = if object(rd) {
            index::infer(right.categories(), operations)?
        } else {
            right.categories().clone()
        };
        return Ok(ld == other.data_type());
    }
    if temporal_cast::is_temporal(rd) {
        let other = if object(ld) {
            index::infer(left.categories(), operations)?
        } else {
            left.categories().clone()
        };
        return Ok(rd == other.data_type());
    }
    Ok(true)
}

pub(super) fn fallback(
    left: &ArrayRef,
    lm: &IndexMetadata,
    right: &ArrayRef,
    rm: &IndexMetadata,
    operations: &mut dyn FrameOperations,
    warning: &mut dyn FnMut(&str),
) -> Result<(ArrayRef, IndexMetadata), Error> {
    let lc = category(lm);
    let rc = category(rm);
    let ld = lc.map_or(left.data_type(), |value| value.categories().data_type());
    let rd = rc.map_or(right.data_type(), |value| value.categories().data_type());
    if let (Some(l), Some(r)) = (lc, rc) {
        if l.ordered()
            && r.ordered()
            && !dtype_equal(ld, rd)
            && ordered_categories_equal(l, r, operations)?
        {
            return Err(Error::CategoricalIndex(
                "dtype of categories must be the same".into(),
            ));
        }
    }
    if [lc, rc]
        .into_iter()
        .flatten()
        .any(|c| c.nullable_categories().is_some())
    {
        return super::nullable_append::fallback(left, lm, right, rm, operations, warning);
    }
    let mut dtype = common(ld, rd)?;
    if [lc, rc]
        .into_iter()
        .flatten()
        .any(|c| c.codes().contains(&-1))
    {
        if dtype.is_integer() {
            dtype = DataType::Float64;
        } else if dtype == DataType::Boolean
            && [lc, rc]
                .into_iter()
                .flatten()
                .any(|c| !c.categories().is_empty() && c.codes().contains(&-1))
        {
            dtype = tuple_frame_dtype();
        }
    }
    if left.is_empty() != right.is_empty() {
        let (array, metadata) = if left.is_empty() {
            (right, rm)
        } else {
            (left, lm)
        };
        // A single extension array leaves concat_compat's target dtype None;
        // NumPy compares that sentinel equal to its default float64 dtype.
        let warn = if category(metadata).is_some() || temporal_cast::is_temporal(array.data_type())
        {
            dtype != DataType::Float64
        } else {
            !dtype_equal(&dtype, array.data_type())
        };
        if warn {
            warning(super::EMPTY_INDEX_WARNING);
        }
        return Ok((
            infer_fallback(array, lc.is_some(), operations)?,
            metadata.clone(),
        ));
    }
    if object(&dtype) {
        dtype = tuple_frame_dtype();
    }
    let left = materialize(left, lc, &dtype, operations)?;
    let right = materialize(right, rc, &dtype, operations)?;
    let joined = super::join_arrays(&left, &right, &dtype, operations)?;
    let output = infer_fallback(&joined, lc.is_some(), operations)?;
    Ok((output, IndexMetadata::Array))
}

pub(super) fn join(
    left: &ArrayRef,
    lm: &IndexMetadata,
    right: &ArrayRef,
    rm: &IndexMetadata,
    operations: &mut dyn FrameOperations,
    warning: &mut dyn FnMut(&str),
) -> Result<(ArrayRef, IndexMetadata), Error> {
    if matches!(lm, IndexMetadata::Multi(_)) {
        if let Some(category) = category(rm) {
            // MultiIndex.append concatenates NumPy values, not categorical
            // scalar iteration: missing integers promote the whole array.
            let mut dtype = category.categories().data_type().clone();
            if category.codes().contains(&-1) {
                if dtype.is_integer() {
                    dtype = DataType::Float64;
                } else if dtype == DataType::Boolean {
                    dtype = tuple_frame_dtype();
                }
            }
            let values = materialize(right, Some(category), &dtype, operations)?;
            return super::multi_append::join(left, &values, operations);
        }
        return super::multi_append::join(left, right, operations);
    }
    if let Some(category) = category(lm) {
        if let Some(codes) = compatible_codes(category, right, self::category(rm), operations)? {
            return merged(category, codes);
        }
    } else if category(rm).is_none() {
        return Ok((
            super::join_indexes(left, right, operations, warning)?,
            IndexMetadata::Array,
        ));
    }
    fallback(left, lm, right, rm, operations, warning)
}

#[cfg(test)]
#[path = "categorical_append_internal_tests.rs"]
mod tests;
