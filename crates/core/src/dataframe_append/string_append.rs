//! String extension identity and scalar boxing at the index append boundary.
use super::{
    DataframeAppendError as Error, FrameOperations, IndexMetadata, StringIndexDescriptor,
    StringMissing, StringStorage, TemporalFrameValue as T, TupleFrameValue as V,
};
use arrow_array::ArrayRef;
use std::{borrow::Cow, sync::Arc};

fn arrow_builtin_comparison_fails(values: &[V]) -> bool {
    use super::BuiltinFrameValue as B;
    // Pandas boxes the entire category dictionary, including unused entries.
    // ArrowTypeError escapes its comparison fallback, whereas ArrowInvalid
    // takes the pointwise path. Conversion follows dictionary order.
    match values.first() {
        Some(V::Scalar(T::Builtin(B::Text(_)))) => values
            .iter()
            .any(|v| !matches!(v, V::Scalar(T::Builtin(B::Text(_))))),
        Some(V::Scalar(T::Builtin(B::Int(_) | B::UInt(_)))) => matches!(
            values
                .iter()
                .find(|v| { !matches!(v, V::Scalar(T::Builtin(B::Int(_) | B::UInt(_)))) }),
            Some(V::Scalar(T::Builtin(B::Bool(_))))
        ),
        Some(V::Scalar(T::Timestamp { .. })) => values.iter().any(|v| {
            !matches!(
                v,
                V::Scalar(T::Timestamp { .. } | T::Builtin(B::Int(_) | B::UInt(_) | B::Float(_)))
            )
        }),
        Some(V::Scalar(T::Duration { .. })) => values.iter().any(|v| {
            !matches!(
                v,
                V::Scalar(T::Duration { .. } | T::Builtin(B::Int(_) | B::UInt(_) | B::Float(_)))
            )
        }),
        Some(V::Tuple(_)) => {
            // Arrow infers one child type across all list values, including
            // unused dictionary entries and children of unequal-length tuples.
            let mut children = Vec::new();
            for value in values {
                let V::Tuple(cells) = value else {
                    // Mixing lists and scalars raises ArrowInvalid, which
                    // takes the pointwise comparison path rather than escape.
                    return false;
                };
                children.extend(
                    cells
                        .iter()
                        .filter(|cell| {
                            !matches!(cell, V::Scalar(value) if super::temporal_cast::missing(value))
                        })
                        .cloned(),
                );
            }
            arrow_builtin_comparison_fails(&children)
        }
        _ => false,
    }
}

fn category_codes(
    category: &super::CategoricalIndexDescriptor,
    string: &StringIndexDescriptor,
) -> Option<Vec<i64>> {
    let categories = category.category_values();
    if string.storage_kind() == StringStorage::PyArrow && arrow_builtin_comparison_fails(categories)
    {
        return None;
    }
    let has_missing = string
        .values()
        .iter()
        .any(|v| !matches!(v, super::BuiltinFrameValue::Text(_)));
    // Python StringArray masks missing comparisons before comparing. Arrow
    // comparison against native numeric categories instead returns false for
    // an unsupported string/numeric pair, even when that category code is null.
    let skip_missing_comparison = string.missing_kind() == StringMissing::PandasNa
        && (string.storage_kind() == StringStorage::Python
            || ((super::blocks::logical_object(category.categories().data_type())
                || category.categories().data_type() == &super::tuple_frame_dtype())
                && categories.iter().all(|v| {
                    matches!(v, V::Scalar(T::Builtin(super::BuiltinFrameValue::Text(_))))
                })));
    let mut codes = Vec::with_capacity(string.values().len());
    for value in string.values() {
        if let super::BuiltinFrameValue::Text(text) = value {
            let key = super::object_factorization::Key::Text(text.clone());
            if let Some(code) = category.keys().iter().position(|k| k == &key) {
                codes.push(i64::try_from(code).expect("category length fits i64"));
            } else if has_missing && skip_missing_comparison {
                codes.push(-1);
            } else {
                return None;
            }
        } else {
            codes.push(-1);
        }
    }
    Some(codes)
}

fn descriptor(metadata: &IndexMetadata) -> Option<&StringIndexDescriptor> {
    if let IndexMetadata::String(value) = metadata {
        Some(value)
    } else {
        None
    }
}

fn validate_tuple_category_hash(metadata: &IndexMetadata) -> Result<(), Error> {
    let IndexMetadata::Categorical(category) = metadata else {
        return Ok(());
    };
    let values = category.category_values();
    if matches!(values.first(), Some(V::Tuple(_))) {
        // CategoricalDtype hashes tuple-first dictionaries through hash_tuples,
        // including unused entries. A later numeric or temporal scalar has no length;
        // concat_compat exposes that failure while collecting dtype identities.
        for value in values {
            let name = match value {
                V::Scalar(T::Timestamp { .. }) => "Timestamp",
                V::Scalar(T::Duration { .. }) => "Timedelta",
                V::Scalar(T::Builtin(
                    super::BuiltinFrameValue::Int(_) | super::BuiltinFrameValue::UInt(_),
                )) => "int",
                V::Scalar(T::Builtin(super::BuiltinFrameValue::Bool(_))) => "bool",
                V::Scalar(T::Builtin(super::BuiltinFrameValue::Float(_))) => "float",
                _ => continue,
            };
            return Err(Error::CategoricalIndex(format!(
                "cannot use 'pandas.core.dtypes.dtypes.CategoricalDtype' as a set element (object of type '{name}' has no len())"
            )));
        }
    }
    validate_category_hash_unicode(metadata)
}

fn category_hash_text(value: &V, column: usize) -> Option<Cow<'_, crate::RlCheckpointText>> {
    match value {
        V::Tuple(cells) => match cells.get(column) {
            Some(V::Scalar(T::Builtin(super::BuiltinFrameValue::Text(text)))) => {
                Some(Cow::Borrowed(text))
            }
            _ => None,
        },
        V::Scalar(T::Builtin(super::BuiltinFrameValue::Text(text))) => {
            text.as_code_points().get(column).map(|point| {
                Cow::Owned(
                    crate::RlCheckpointText::try_from_code_points([*point])
                        .expect("a code point from validated text remains valid"),
                )
            })
        }
        _ => None,
    }
}

fn validate_category_hash_unicode(metadata: &IndexMetadata) -> Result<(), Error> {
    if let IndexMetadata::Categorical(category) = metadata {
        let values = category.category_values();
        if matches!(values.first(), Some(V::Tuple(_))) {
            // A surviving single categorical axis still hashes its dictionary.
            // UnicodeEncodeError escapes the TypeError compatibility fallback.
            // MultiIndex conversion expands tuples once and scalar strings into
            // individual characters, then hashes columns before later columns.
            // Nested tuple objects use escaped repr, not recursive text hashing.
            let width = values.iter().fold(0, |width, value| {
                width.max(match value {
                    V::Tuple(cells) => cells.len(),
                    V::Scalar(T::Builtin(super::BuiltinFrameValue::Text(text))) => text.len(),
                    _ => 0,
                })
            });
            for column in 0..width {
                for (row, value) in values.iter().enumerate() {
                    if let Some(text) = category_hash_text(value, column) {
                        text.to_utf8()
                            .map_err(|_| super::string_index::unicode_error(&text, row))?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn validate_arrow_tuple_unicode(value: &V, row: usize) -> Result<(), Error> {
    match value {
        V::Tuple(cells) => {
            for cell in cells {
                validate_arrow_tuple_unicode(cell, row)?;
            }
        }
        V::Scalar(T::Builtin(super::BuiltinFrameValue::Text(text))) => {
            text.to_utf8()
                .map_err(|_| super::string_index::unicode_error(text, row))?;
        }
        _ => {}
    }
    Ok(())
}

pub(super) fn infer(
    array: &ArrayRef,
    metadata: &IndexMetadata,
    operations: &mut dyn FrameOperations,
) -> Result<ArrayRef, Error> {
    validate_category_hash_unicode(metadata)?;
    if descriptor(metadata).is_some() {
        Ok(array.clone())
    } else {
        super::index::infer(array, operations)
    }
}

fn boxed(
    array: &ArrayRef,
    metadata: &IndexMetadata,
    operations: &mut dyn FrameOperations,
) -> Result<(ArrayRef, IndexMetadata), Error> {
    if let Some(descriptor) = descriptor(metadata) {
        let cells = descriptor
            .values()
            .iter()
            .cloned()
            .map(|v| V::Scalar(T::Builtin(v)))
            .collect::<Vec<_>>();
        Ok((operations.tuple_array(&cells)?, IndexMetadata::Array))
    } else {
        Ok((array.clone(), metadata.clone()))
    }
}

pub(super) fn join(
    left: &ArrayRef,
    lm: &IndexMetadata,
    right: &ArrayRef,
    rm: &IndexMetadata,
    operations: &mut dyn FrameOperations,
    warning: &mut dyn FnMut(&str),
) -> Result<(ArrayRef, IndexMetadata), Error> {
    let ls = descriptor(lm);
    let rs = descriptor(rm);
    if ls.is_none() && rs.is_none() {
        return super::nullable_append::join(left, lm, right, rm, operations, warning);
    }
    validate_tuple_category_hash(lm)?;
    validate_tuple_category_hash(rm)?;
    if let (IndexMetadata::Categorical(category), Some(string)) = (lm, rs) {
        // isin runs before Arrow equality converts the category dictionary.
        // Unmatched text without a missing candidate takes TypeError fallback
        // before UTF-8 conversion; empty/missing candidates reach conversion.
        if string.storage_kind() == StringStorage::PyArrow {
            if let Some(
                first @ (V::Tuple(_) | V::Scalar(T::Builtin(super::BuiltinFrameValue::Text(_)))),
            ) = category.category_values().first()
            {
                // Track missing candidates and text membership independently
                // in one pass: a missing candidate reaches Arrow comparison
                // even when another candidate is unmatched text.
                let (has_missing, all_text_matches) = string.values().iter().fold(
                    (false, true),
                    |(has_missing, all_text_matches), value| match value {
                        super::BuiltinFrameValue::Text(value) => (
                            has_missing,
                            all_text_matches
                                && category.keys().contains(
                                    &super::object_factorization::Key::Text(value.clone()),
                                ),
                        ),
                        _ => (true, all_text_matches),
                    },
                );
                if has_missing || all_text_matches {
                    if matches!(first, V::Tuple(_)) {
                        // Arrow boxes the whole dictionary as nested lists,
                        // unlike the one-level Pandas hash_tuples conversion.
                        for (row, value) in category.category_values().iter().enumerate() {
                            validate_arrow_tuple_unicode(value, row)?;
                        }
                    } else {
                        validate_arrow_tuple_unicode(first, 0)?;
                    }
                }
            }
        }
        if let Some(codes) = category_codes(category, string) {
            return super::categorical_append::merged(category, codes);
        }
        if left.is_empty() && !right.is_empty() {
            warning(super::EMPTY_INDEX_WARNING);
            return Ok((right.clone(), rm.clone()));
        }
        let (right, rm) = boxed(right, rm, operations)?;
        return super::categorical_append::fallback(left, lm, &right, &rm, operations, warning);
    }
    // These left index subclasses own their directional append semantics.
    if matches!(lm, IndexMetadata::Multi(_) | IndexMetadata::Categorical(_)) {
        let (right, rm) = boxed(right, rm, operations)?;
        return super::nullable_append::join(left, lm, &right, &rm, operations, warning);
    }
    let equal = match (ls, rs) {
        (Some(l), Some(r)) => {
            l.storage_kind() == r.storage_kind() && l.missing_kind() == r.missing_kind()
        }
        _ => false,
    };
    if !equal && left.is_empty() != right.is_empty() {
        let (array, metadata) = if left.is_empty() {
            (right, rm)
        } else {
            (left, lm)
        };
        if !matches!(metadata, IndexMetadata::Array)
            || !(super::blocks::logical_object(array.data_type())
                || array.data_type() == &super::tuple_frame_dtype())
        {
            warning(super::EMPTY_INDEX_WARNING);
        }
        return Ok((infer(array, metadata, operations)?, metadata.clone()));
    }
    if let (Some(l), Some(r)) = (ls, rs) {
        let storage = if l.storage_kind() == StringStorage::PyArrow
            || r.storage_kind() == StringStorage::PyArrow
        {
            StringStorage::PyArrow
        } else {
            StringStorage::Python
        };
        let missing = if l.missing_kind() == StringMissing::PandasNa
            || r.missing_kind() == StringMissing::PandasNa
        {
            StringMissing::PandasNa
        } else {
            StringMissing::NaN
        };
        let array = operations.concat(left, right)?;
        if array.len() != left.len() + right.len() {
            return Err(Error::IndexMetadata(
                "string concatenation changed row count",
            ));
        }
        let descriptor = StringIndexDescriptor::new(array.clone(), storage, missing)?;
        return Ok((array, IndexMetadata::String(Arc::new(descriptor))));
    }
    let (left, lm) = boxed(left, lm, operations)?;
    let (right, rm) = boxed(right, rm, operations)?;
    super::nullable_append::join(&left, &lm, &right, &rm, operations, warning)
}

#[cfg(test)]
#[path = "string_append_tests.rs"]
pub(super) mod tests;
