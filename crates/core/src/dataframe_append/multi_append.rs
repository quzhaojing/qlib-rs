//! Reconstructing a multi-level index after appending ordinary index values.
use super::{
    BuiltinFrameValue as B, DataframeAppendError, FrameOperations, IndexMetadata,
    MultiIndexDescriptor, TemporalFrameValue as T, TupleFrameValue as V, factorize_level_objects,
    index, tuple_index,
};
use arrow_array::ArrayRef;
use arrow_schema::DataType;
use std::sync::Arc;

fn right_values(array: &ArrayRef) -> Result<Vec<V>, DataframeAppendError> {
    let values = tuple_index::values(array)?;
    if values.is_empty() {
        return Ok(values);
    }
    match array.data_type() {
        DataType::Timestamp(_, None) | DataType::Duration(_) => {
            // NumPy boxes ns values as integer ticks and coarser values as
            // Python temporals (or integers outside Python's range).
            // NaT becomes None, not pd.NaT.
            Ok(values
                .into_iter()
                .map(|value| match value {
                    V::Scalar(T::Timestamp { ticks, unit, .. }) => {
                        super::python_temporal::numpy_box(ticks, unit, true)
                    }
                    V::Scalar(T::Duration { ticks, unit }) => {
                        super::python_temporal::numpy_box(ticks, unit, false)
                    }
                    _ => V::Scalar(T::Builtin(B::None)),
                })
                .collect())
        }
        _ => Ok(values),
    }
}

fn reconstruct(
    values: &[V],
    operations: &mut dyn FrameOperations,
) -> Result<Option<MultiIndexDescriptor>, DataframeAppendError> {
    let Some(V::Tuple(first)) = values.first() else {
        return Ok(None);
    };
    if values
        .iter()
        .all(|v| matches!(v, V::Tuple(row) if row.is_empty()))
    {
        let array = operations.tuple_array(values)?;
        return MultiIndexDescriptor::empty_tuple_rows(values, array).map(Some);
    }
    let width = first.len();
    let rows = values
        .iter()
        .map(|value| match value {
            V::Tuple(row) if row.len() >= width => Some(row),
            _ => None,
        })
        .collect::<Option<Vec<_>>>();
    let Some(rows) = rows.filter(|_| width > 0) else {
        return Ok(None);
    };
    let mut levels = Vec::with_capacity(width);
    let mut codes = Vec::with_capacity(width);
    for column in 0..width {
        let cells = rows
            .iter()
            .map(|row| row[column].clone())
            .collect::<Vec<_>>();
        let factored = factorize_level_objects(&cells)?;
        let object = operations.tuple_array(&factored.uniques)?;
        levels.push(index::infer(&object, operations)?);
        codes.push(
            factored
                .codes
                .into_iter()
                .map(|c| c.map_or(-1, |c| i64::try_from(c).expect("allocated level fits i64")))
                .collect(),
        );
    }
    MultiIndexDescriptor::new(
        levels,
        codes,
        vec![V::Scalar(T::Builtin(B::None)); width],
        None,
    )
    .map(Some)
    .map_err(|e| DataframeAppendError::MultiIndexReconstruction(e.to_string()))
}

pub(super) fn join(
    left: &ArrayRef,
    right: &ArrayRef,
    operations: &mut dyn FrameOperations,
) -> Result<(ArrayRef, IndexMetadata), DataframeAppendError> {
    let mut values = tuple_index::values(left)?;
    values.extend(right_values(right)?);
    if let Some(descriptor) = reconstruct(&values, operations)? {
        let index = operations.tuple_array(&descriptor.values())?;
        Ok((index, IndexMetadata::Multi(Arc::new(descriptor))))
    } else {
        let object = operations.tuple_array(&values)?;
        // Index(object ndarray) keeps numeric objects, unlike Index.append([]).
        // Datetime inference is a distinct constructor behavior.
        let inferred = index::infer(&object, operations)?;
        let python_duration = values.iter().any(|v| {
            matches!(
                v,
                V::PythonTemporal(super::PythonTemporalValue::Timedelta { .. })
            )
        });
        let index = if matches!(inferred.data_type(), DataType::Timestamp(..))
            || (python_duration && matches!(inferred.data_type(), DataType::Duration(_)))
        {
            inferred
        } else {
            object
        };
        Ok((index, IndexMetadata::Array))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reconstruction_rejects_invalid_internal_level_values() {
        let invalid = V::Tuple(vec![V::Scalar(T::Duration {
            ticks: i64::MIN,
            unit: arrow_schema::TimeUnit::Second,
        })]);
        let before = invalid.clone();
        let error = reconstruct(
            std::slice::from_ref(&invalid),
            &mut super::super::ArrowOperations,
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "Invalid argument error: temporal object uses reserved NaT ticks"
        );
        assert_eq!(invalid, before);
    }
}
