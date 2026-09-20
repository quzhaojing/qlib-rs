//! Indexed Arrow frame append for `qlib.rl.order_execution.utils`.

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, RecordBatchOptions};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use arrow_select::concat::concat;
use indexmap::{IndexMap, IndexSet};
use thiserror::Error;

mod blocks;
mod builtin_objects;
mod categorical_append;
mod categorical_index;
mod constructor;
mod extended_plan;
mod index;
mod index_metadata;
mod inference;
mod multi_append;
mod multi_index;
mod nullable_append;
mod nullable_index;
mod numeric;
mod object_argsort;
mod object_factorization;
mod object_sort;
mod objects;
mod python_temporal;
mod string_append;
mod string_index;
mod temporal_cast;
mod temporal_inference;
mod temporal_objects;
mod temporal_plan;
mod tuple_index;
mod tuple_objects;
pub use builtin_objects::{
    BuiltinFrameValue, builtin_frame_array, builtin_frame_dtype, builtin_frame_values,
};
pub use categorical_index::{CategoricalIndexDescriptor, CategoricalIndexError};
pub use constructor::{
    FrameColumnInput, FrameConstructionError, frame_from_builtin_records, frame_from_columns,
    frame_from_temporal_records,
};
pub use index_metadata::IndexMetadata;
pub use inference::infer_frame_values;
pub use multi_index::{MultiIndexDescriptor, MultiIndexError, MultiIndexLevel};
pub use nullable_index::{NullableIndexDescriptor, NullableIndexError};
pub use object_factorization::{
    ObjectFactorization, factorize_index_objects, object_index_is_unique,
};
pub use object_sort::factorize_level_objects;
pub use objects::{numeric_object_array, numeric_object_dtype};
pub use python_temporal::PythonTemporalValue;
pub use string_index::{StringIndexDescriptor, StringIndexError, StringMissing, StringStorage};
pub use temporal_cast::temporal_object_array;
pub use temporal_inference::infer_temporal_frame_values;
pub use temporal_objects::{
    TemporalFrameValue, temporal_frame_array, temporal_frame_dtype, temporal_frame_values,
};
pub use tuple_objects::{
    TupleFrameValue, tuple_frame_array, tuple_frame_dtype, tuple_frame_values,
};

/// Empty column axes have observable union semantics in Pandas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmptyColumnAxis {
    /// The default axis of a frame constructed without columns.
    RangeIndex,
    /// An explicitly supplied empty sequence of string/object column labels.
    ObjectIndex,
}

/// A separate row index prevents collision with a real data column of the same name.
#[derive(Clone, Debug)]
pub struct IndexedFrame {
    index: ArrayRef,
    index_name: Option<String>,
    index_metadata: IndexMetadata,
    data: RecordBatch,
    empty_column_axis: EmptyColumnAxis,
    blocks: Vec<Vec<usize>>,
}

impl IndexedFrame {
    /// Attach explicit string storage identity and missing-value semantics.
    /// # Errors
    /// Rejects index/data row-count mismatches.
    pub fn from_string_index(
        descriptor: StringIndexDescriptor,
        index_name: Option<String>,
        data: RecordBatch,
    ) -> Result<Self, DataframeAppendError> {
        let mut frame = Self::new(descriptor.storage(), index_name, data)?;
        frame.index_metadata = IndexMetadata::String(Arc::new(descriptor));
        Ok(frame)
    }

    /// Construct a frame retaining explicit categories, codes and ordering.
    /// # Errors
    /// Rejects a data/index row-count mismatch.
    pub fn from_categorical_index(
        descriptor: CategoricalIndexDescriptor,
        index_name: Option<String>,
        data: RecordBatch,
    ) -> Result<Self, DataframeAppendError> {
        let mut frame = Self::new(descriptor.storage(), index_name, data)?;
        frame.index_metadata = IndexMetadata::Categorical(Arc::new(descriptor));
        Ok(frame)
    }

    /// Attach explicit Pandas masked numeric/bool index identity.
    /// # Errors
    /// Rejects index/data row-count mismatches.
    pub fn from_nullable_index(
        descriptor: NullableIndexDescriptor,
        index_name: Option<String>,
        data: RecordBatch,
    ) -> Result<Self, DataframeAppendError> {
        let mut frame = Self::new(descriptor.storage(), index_name, data)?;
        frame.index_metadata = IndexMetadata::Nullable(Arc::new(descriptor));
        Ok(frame)
    }

    /// Construct a frame from explicit multi-level metadata without losing names
    /// or unused levels.
    /// # Errors
    /// Rejects invalid tuple storage or a data/index row-count mismatch.
    pub fn from_multi_index(
        descriptor: MultiIndexDescriptor,
        data: RecordBatch,
    ) -> Result<Self, DataframeAppendError> {
        let index = tuple_frame_array(&descriptor.values())?;
        let mut frame = Self::new(index, None, data)?;
        frame.index_metadata = IndexMetadata::Multi(Arc::new(descriptor));
        Ok(frame)
    }

    /// # Errors
    /// Rejects an index whose length differs from the data's row count.
    pub fn new(
        index: ArrayRef,
        index_name: Option<String>,
        data: RecordBatch,
    ) -> Result<Self, DataframeAppendError> {
        if index.len() != data.num_rows() {
            return Err(DataframeAppendError::IndexLength);
        }
        index::validate(index.data_type())?;
        Ok(Self {
            index,
            index_name,
            index_metadata: IndexMetadata::Array,
            blocks: blocks::consolidated(&data),
            data,
            empty_column_axis: EmptyColumnAxis::RangeIndex,
        })
    }

    #[must_use]
    pub fn index(&self) -> &ArrayRef {
        &self.index
    }
    #[must_use]
    pub fn index_name(&self) -> Option<&str> {
        self.index_name.as_deref()
    }

    /// Attach validated row-index identity without changing the stored values.
    /// # Errors
    /// Rejects incompatible storage/values or invalid range bounds/step.
    pub fn with_index_metadata(
        mut self,
        metadata: IndexMetadata,
    ) -> Result<Self, DataframeAppendError> {
        metadata.validate(&self.index)?;
        if matches!(metadata, IndexMetadata::Multi(_)) {
            self.index_name = None;
        }
        self.index_metadata = metadata;
        Ok(self)
    }

    #[must_use]
    pub fn index_metadata(&self) -> &IndexMetadata {
        &self.index_metadata
    }
    #[must_use]
    pub fn data(&self) -> &RecordBatch {
        &self.data
    }

    /// Imported frames can retain their original (possibly fragmented) block partition.
    /// # Errors
    /// Rejects missing/repeated/out-of-range columns, empty blocks and mixed block dtypes.
    pub fn with_blocks(mut self, groups: Vec<Vec<usize>>) -> Result<Self, DataframeAppendError> {
        blocks::validate(&self.data, &groups)?;
        self.blocks = groups;
        Ok(self)
    }

    #[must_use]
    pub fn blocks(&self) -> &[Vec<usize>] {
        &self.blocks
    }

    /// Select the column-axis metadata used when the frame has no data columns.
    #[must_use]
    pub fn with_empty_column_axis(mut self, kind: EmptyColumnAxis) -> Self {
        self.empty_column_axis = kind;
        self
    }

    /// Nonempty Arrow schemas have string/object labels, not a numeric range.
    #[must_use]
    pub fn column_axis(&self) -> EmptyColumnAxis {
        if self.data.num_columns() == 0 {
            self.empty_column_axis
        } else {
            EmptyColumnAxis::ObjectIndex
        }
    }
}

#[derive(Debug, Error)]
pub enum DataframeAppendError {
    #[error("{0}")]
    CategoricalIndex(String),
    #[error("{0}")]
    NullableIndex(String),
    #[error("MultiIndex reconstruction failed: {0}")]
    MultiIndexReconstruction(String),
    #[error("invalid index metadata: {0}")]
    IndexMetadata(&'static str),
    #[error("float16 indexes are not supported")]
    Float16Index,
    #[error("frame index and row count differ")]
    IndexLength,
    #[error("None of ['datetime'] are in the columns")]
    MissingDatetime,
    #[error("Index data must be 1-dimensional")]
    DuplicateDatetime,
    #[error("Reindexing only valid with uniquely valued Index objects")]
    DuplicateColumns,
    #[error("invalid frame block layout: {0}")]
    BlockLayout(&'static str),
    #[error("native frame append needs a dtype adapter for {0:?} and {1:?}")]
    DtypeAdapter(DataType, DataType),
    #[error(transparent)]
    StringIndex(#[from] StringIndexError),
    #[error(transparent)]
    Arrow(#[from] ArrowError),
}

fn joined_type(left: &DataType, right: &DataType) -> Result<DataType, DataframeAppendError> {
    if left == right {
        return Ok(left.clone());
    }
    match (left, right) {
        (DataType::Timestamp(lu, lz), DataType::Timestamp(ru, rz)) if lz == rz => {
            Ok(DataType::Timestamp((*lu).max(*ru), lz.clone()))
        }
        (DataType::Null, value) | (value, DataType::Null) => Ok(value.clone()),
        (
            DataType::Int64 | DataType::Float32 | DataType::Float64,
            DataType::Int64 | DataType::Float32 | DataType::Float64,
        ) => Ok(DataType::Float64),
        _ => Err(DataframeAppendError::DtypeAdapter(
            left.clone(),
            right.clone(),
        )),
    }
}

trait FrameOperations {
    fn tuple_array(&mut self, values: &[TupleFrameValue]) -> Result<ArrayRef, ArrowError> {
        tuple_frame_array(values)
    }
    fn infer_index(&mut self, values: &[TemporalFrameValue]) -> Result<ArrayRef, ArrowError> {
        infer_temporal_frame_values(values)
    }
    fn fill(
        &mut self,
        array: &ArrayRef,
        value: Option<&BuiltinFrameValue>,
    ) -> Result<ArrayRef, ArrowError> {
        extended_plan::fill(array, value)
    }
    fn cast(&mut self, array: &ArrayRef, dtype: &DataType) -> Result<ArrayRef, ArrowError>;
    fn concat(&mut self, left: &ArrayRef, right: &ArrayRef) -> Result<ArrayRef, ArrowError>;
    fn batch(
        &mut self,
        fields: Vec<Field>,
        columns: Vec<ArrayRef>,
        rows: usize,
    ) -> Result<RecordBatch, ArrowError>;
}

struct ArrowOperations;

impl FrameOperations for ArrowOperations {
    fn cast(&mut self, array: &ArrayRef, dtype: &DataType) -> Result<ArrayRef, ArrowError> {
        objects::cast(array, dtype)
    }
    fn concat(&mut self, left: &ArrayRef, right: &ArrayRef) -> Result<ArrayRef, ArrowError> {
        concat(&[left.as_ref(), right.as_ref()])
    }
    fn batch(
        &mut self,
        fields: Vec<Field>,
        columns: Vec<ArrayRef>,
        rows: usize,
    ) -> Result<RecordBatch, ArrowError> {
        RecordBatch::try_new_with_options(
            Arc::new(Schema::new(fields)),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(rows)),
        )
    }
}

fn join_arrays(
    left: &ArrayRef,
    right: &ArrayRef,
    dtype: &DataType,
    operations: &mut dyn FrameOperations,
) -> Result<ArrayRef, ArrowError> {
    let left = operations.cast(left, dtype)?;
    let right = operations.cast(right, dtype)?;
    operations.concat(&left, &right)
}

fn missing_type(dtype: &DataType, aligned: bool) -> DataType {
    // NumPy-backed integer columns gain NaN when alignment inserts missing rows.
    if aligned && dtype.is_integer() {
        DataType::Float64
    } else if aligned && dtype == &DataType::Boolean {
        numeric_object_dtype()
    } else {
        dtype.clone()
    }
}

fn join_data(
    left: &ArrayRef,
    right: &ArrayRef,
    dtype: &DataType,
    plan: &blocks::Plan,
    operations: &mut dyn FrameOperations,
) -> Result<ArrayRef, ArrowError> {
    let left = operations.fill(left, plan.fill[0].as_ref())?;
    let right = operations.fill(right, plan.fill[1].as_ref())?;
    if plan.box_floats {
        let left = operations.cast(&left, &DataType::Float64)?;
        let right = operations.cast(&right, &DataType::Float64)?;
        join_arrays(&left, &right, dtype, operations)
    } else {
        join_arrays(&left, &right, dtype, operations)
    }
}

type ColumnPair = (String, (Option<usize>, Option<usize>));

fn column_pairs(
    frame: &IndexedFrame,
    other: &RecordBatch,
    datetime: usize,
) -> Result<Vec<ColumnPair>, DataframeAppendError> {
    let left = frame.data.schema_ref().fields();
    let right: Vec<_> = other
        .schema_ref()
        .fields()
        .iter()
        .enumerate()
        .filter(|&(position, _)| position != datetime)
        .collect();
    // A 0x0 side is ignored. An empty RangeIndex also preserves repeated labels
    // when it has rows; an explicit object-label axis instead requires reindexing.
    if left.is_empty()
        && (frame.data.num_rows() == 0 || frame.empty_column_axis == EmptyColumnAxis::RangeIndex)
    {
        return Ok(right
            .into_iter()
            .map(|(i, f)| (f.name().clone(), (None, Some(i))))
            .collect());
    }
    if other.num_rows() == 0 && right.is_empty() {
        return Ok(left
            .iter()
            .enumerate()
            .map(|(i, f)| (f.name().clone(), (Some(i), None)))
            .collect());
    }
    if left
        .iter()
        .map(|f| f.name())
        .eq(right.iter().map(|(_, f)| f.name()))
    {
        return Ok(right
            .into_iter()
            .enumerate()
            .map(|(i, (j, f))| (f.name().clone(), (Some(i), Some(j))))
            .collect());
    }
    let mut names = IndexMap::new();
    for (position, field) in left.iter().enumerate() {
        if names
            .insert(field.name().clone(), (Some(position), None))
            .is_some()
        {
            return Err(DataframeAppendError::DuplicateColumns);
        }
    }
    let mut right_names = IndexSet::new();
    for (position, field) in right {
        if !right_names.insert(field.name().clone()) {
            return Err(DataframeAppendError::DuplicateColumns);
        }
        names.entry(field.name().clone()).or_insert((None, None)).1 = Some(position);
    }
    Ok(names.into_iter().collect())
}

/// Append a preconstructed Arrow table after moving its `datetime` column to the index.
///
/// Supports equal Arrow dtypes, Arrow nulls, and native integer/floating data blocks.
/// Integer/bool combinations promote to integer; bool/float mixing follows the
/// source's asymmetric block rules and uses [`numeric_object_dtype`] when boxed.
/// Arrow nulls are a native typed-missing representation, not a Python object
/// column: callers must adapt object/extension dtypes before entering this API.
/// Other mixed dtypes report a missing native adapter rather than discarding type
/// information. Mapping/record construction, Pandas extension/object promotion,
/// and `MultiIndex` remain separate boundaries. Equal repeated data-column labels
/// append positionally; unequal nonempty labels require unique columns on both sides.
/// Empty column axes retain their range/object distinction during later appends.
/// Inputs retain their buffers and are never mutated. Index order and duplicates
/// are retained; index names must agree except when a 0x0 side is ignored.
/// This convenience entry point ignores warnings; compatibility callers should
/// use [`dataframe_append_with_warnings`] to observe them in source order.
///
/// # Errors
/// Reports missing/duplicate datetime, non-alignable duplicate labels, unsupported dtype
/// combinations, or an Arrow cast/concatenation failure.
///
/// # Panics
/// Only if the internal union-of-column-labels invariant is violated; every
/// constructed union entry has at least one source position.
pub fn dataframe_append(
    frame: &IndexedFrame,
    other: &RecordBatch,
) -> Result<IndexedFrame, DataframeAppendError> {
    dataframe_append_with_warnings(frame, other, &mut ignore_warning)
}

fn ignore_warning(_: &str) {}

/// Append with an ordered sink for Pandas-compatible `FutureWarning` messages.
/// Warnings emitted before a later Arrow failure remain observable to the sink.
/// # Errors
/// Propagates the same errors as [`dataframe_append`].
pub fn dataframe_append_with_warnings(
    frame: &IndexedFrame,
    other: &RecordBatch,
    warning: &mut dyn FnMut(&str),
) -> Result<IndexedFrame, DataframeAppendError> {
    append_with_operations(frame, other, &mut ArrowOperations, warning)
}

const ALL_NA_WARNING: &str = "The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.";
const EMPTY_INDEX_WARNING: &str = "The behavior of array concatenation with empty entries is deprecated. In a future version, this will no longer exclude empty items when determining the result dtype. To retain the old behavior, exclude the empty entries before the concat operation.";

fn join_indexes(
    left: &ArrayRef,
    right: &ArrayRef,
    operations: &mut dyn FrameOperations,
    warning: &mut dyn FnMut(&str),
) -> Result<ArrayRef, DataframeAppendError> {
    index::join(left, right, operations, warning)
}

fn datetime_column(other: &RecordBatch) -> Result<usize, DataframeAppendError> {
    let datetime: Vec<_> = other
        .schema_ref()
        .fields()
        .iter()
        .enumerate()
        .filter_map(|(position, field)| (field.name() == "datetime").then_some(position))
        .collect();
    match datetime.as_slice() {
        [] => Err(DataframeAppendError::MissingDatetime),
        [position] => Ok(*position),
        _ => Err(DataframeAppendError::DuplicateDatetime),
    }
}

struct AppendedIndex {
    index: ArrayRef,
    name: Option<String>,
    empty_column_axis: EmptyColumnAxis,
    metadata: IndexMetadata,
}

fn append_index(
    frame: &IndexedFrame,
    other: &RecordBatch,
    datetime: usize,
    operations: &mut dyn FrameOperations,
    warning: &mut dyn FnMut(&str),
) -> Result<AppendedIndex, DataframeAppendError> {
    let right_metadata = IndexMetadata::from_column(other, datetime)?;
    // Pandas constructs concatenated axes before reindexing or joining data blocks.
    let right_index = if matches!(right_metadata, IndexMetadata::String(_)) {
        other.column(datetime).clone()
    } else {
        index::from_column(other.column(datetime), operations, warning)?
    };
    let left_empty = frame.data.num_rows() == 0 && frame.data.num_columns() == 0;
    let right_empty = other.num_rows() == 0 && other.num_columns() == 1;
    let mut reconstructed_metadata = IndexMetadata::Array;
    let (index, index_name, empty_column_axis) = if left_empty && !right_empty {
        let inferred = string_append::infer(&right_index, &right_metadata, operations)?;
        reconstructed_metadata = right_metadata;
        (
            inferred,
            Some("datetime".to_owned()),
            EmptyColumnAxis::ObjectIndex,
        )
    } else if right_empty && !left_empty {
        (
            string_append::infer(&frame.index, &frame.index_metadata, operations)?,
            frame.index_name.clone(),
            frame.empty_column_axis,
        )
    } else {
        let (index, metadata) = string_append::join(
            &frame.index,
            &frame.index_metadata,
            &right_index,
            &right_metadata,
            operations,
            warning,
        )?;
        reconstructed_metadata = metadata;
        let name = frame
            .index_name
            .as_deref()
            .filter(|&name| name == "datetime")
            .map(str::to_owned);
        (index, name, EmptyColumnAxis::ObjectIndex)
    };
    Ok(AppendedIndex {
        index,
        name: index_name,
        empty_column_axis,
        metadata: reconstructed_metadata,
    })
}

fn append_with_operations(
    frame: &IndexedFrame,
    other: &RecordBatch,
    operations: &mut dyn FrameOperations,
    warning: &mut dyn FnMut(&str),
) -> Result<IndexedFrame, DataframeAppendError> {
    let datetime = datetime_column(other)?;
    let AppendedIndex {
        index,
        name: index_name,
        empty_column_axis,
        metadata: reconstructed_metadata,
    } = append_index(frame, other, datetime, operations, warning)?;
    let left_empty = frame.data.num_rows() == 0 && frame.data.num_columns() == 0;
    let right_empty = other.num_rows() == 0 && other.num_columns() == 1;
    let names = column_pairs(frame, other, datetime)?;
    let plans = blocks::plans(frame, other, &names, datetime)?;
    let mut fields = Vec::with_capacity(names.len());
    let mut columns = Vec::with_capacity(names.len());
    for plan in &plans {
        for (name, (left, right)) in &names[plan.positions.clone()] {
            let (left, right) = (*left, *right);
            let (left, right, dtype) = if let Some(position) = left {
                let left = frame.data.column(position);
                if let Some(position) = right {
                    let right = other.column(position);
                    (
                        left.clone(),
                        right.clone(),
                        match &plan.dtype {
                            Some(dtype) => dtype.clone(),
                            None => joined_type(left.data_type(), right.data_type())?,
                        },
                    )
                } else {
                    let dtype = missing_type(
                        left.data_type(),
                        other.num_rows() > 0 || other.num_columns() > 1,
                    );
                    (
                        left.clone(),
                        objects::missing(&dtype, other.num_rows()),
                        dtype,
                    )
                }
            } else {
                let right = other.column(right.expect("union entry has at least one source"));
                let dtype = missing_type(
                    right.data_type(),
                    frame.data.num_rows() > 0 || frame.data.num_columns() > 0,
                );
                (
                    objects::missing(&dtype, frame.data.num_rows()),
                    right.clone(),
                    dtype,
                )
            };
            fields.push(Field::new(name.clone(), dtype.clone(), true));
            columns.push(join_data(&left, &right, &dtype, plan, operations)?);
        }
        if plan.warn {
            warning(ALL_NA_WARNING);
        }
    }
    let data = operations.batch(fields, columns, index.len())?;
    let mut output =
        IndexedFrame::new(index, index_name, data)?.with_empty_column_axis(empty_column_axis);
    output.index_metadata = reconstructed_metadata;
    if right_empty && !left_empty {
        output.index_metadata = match &frame.index_metadata {
            IndexMetadata::Multi(descriptor) => {
                IndexMetadata::Multi(Arc::new(descriptor.appended_without_other()))
            }
            metadata => metadata.clone(),
        };
        output.blocks.clone_from(&frame.blocks);
    } else if !left_empty {
        output.blocks = plans
            .into_iter()
            .map(|plan| plan.positions.collect())
            .collect();
    }
    Ok(output)
}

#[cfg(test)]
mod categorical_append_tests;
#[cfg(test)]
mod differential_tests;
#[cfg(test)]
mod multi_append_tests;

#[cfg(test)]
mod block_tests;

#[cfg(test)]
mod builtin_object_tests;

#[cfg(test)]
mod temporal_tests;

#[cfg(test)]
mod missing_append_tests;

#[cfg(test)]
mod temporal_append_tests;

#[cfg(test)]
mod index_tests;

#[cfg(test)]
mod range_index_tests;

#[cfg(test)]
mod tuple_index_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int64Array;

    fn warning_sink(warnings: &mut Vec<String>) -> impl FnMut(&str) + '_ {
        |message| warnings.push(message.to_owned())
    }

    struct FailAt {
        fail: usize,
        calls: Vec<&'static str>,
    }
    impl FailAt {
        fn check(&mut self, name: &'static str) -> Result<(), ArrowError> {
            self.calls.push(name);
            if self.calls.len() == self.fail + 1 {
                Err(ArrowError::ComputeError(format!(
                    "failure at {}",
                    self.fail
                )))
            } else {
                Ok(())
            }
        }
    }
    impl FrameOperations for FailAt {
        fn cast(&mut self, a: &ArrayRef, dtype: &DataType) -> Result<ArrayRef, ArrowError> {
            self.check("cast")?;
            ArrowOperations.cast(a, dtype)
        }
        fn concat(&mut self, a: &ArrayRef, b: &ArrayRef) -> Result<ArrayRef, ArrowError> {
            self.check("concat")?;
            ArrowOperations.concat(a, b)
        }
        fn batch(
            &mut self,
            fields: Vec<Field>,
            columns: Vec<ArrayRef>,
            rows: usize,
        ) -> Result<RecordBatch, ArrowError> {
            self.check("batch")?;
            ArrowOperations.batch(fields, columns, rows)
        }
    }

    #[test]
    fn arrow_failure_short_circuits_without_mutating_inputs() {
        let a = Arc::new(Int64Array::from(vec![1])) as ArrayRef;
        let field = |name| Field::new(name, DataType::Int64, true);
        let data = ArrowOperations
            .batch(vec![field("a")], vec![a.clone()], 1)
            .unwrap();
        let frame = IndexedFrame::new(a.clone(), Some("datetime".into()), data).unwrap();
        let other = ArrowOperations
            .batch(vec![field("datetime"), field("a")], vec![a.clone(), a], 1)
            .unwrap();
        let expected = ["cast", "cast", "concat", "cast", "cast", "concat", "batch"];
        for fail in 0..expected.len() {
            let mut operations = FailAt {
                fail,
                calls: vec![],
            };
            let error =
                append_with_operations(&frame, &other, &mut operations, &mut ignore_warning)
                    .unwrap_err();
            assert!(error.to_string().ends_with(&format!("failure at {fail}")));
            assert_eq!(operations.calls, expected[..=fail]);
            assert_eq!(frame.data.num_rows(), 1);
            assert_eq!(other.num_rows(), 1);
            assert_eq!(
                frame
                    .index
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .value(0),
                1
            );
        }
        let mut operations = FailAt {
            fail: expected.len(),
            calls: vec![],
        };
        let result =
            append_with_operations(&frame, &other, &mut operations, &mut ignore_warning).unwrap();
        assert_eq!(operations.calls, expected);
        assert_eq!(result.data.num_rows(), 2);
        assert_eq!(result.index_name(), Some("datetime"));
    }

    #[test]
    fn warning_precedes_later_arrow_failure() {
        use arrow_array::{Float32Array, Float64Array};
        let index = Arc::new(Int64Array::from(vec![0])) as ArrayRef;
        let left = Arc::new(Float64Array::from(vec![f64::NAN])) as ArrayRef;
        let right = Arc::new(Float32Array::from(vec![1.0])) as ArrayRef;
        let frame = IndexedFrame::new(
            index.clone(),
            None,
            ArrowOperations
                .batch(
                    vec![Field::new("x", DataType::Float64, true)],
                    vec![left],
                    1,
                )
                .unwrap(),
        )
        .unwrap();
        let other = ArrowOperations
            .batch(
                vec![
                    Field::new("datetime", DataType::Int64, true),
                    Field::new("x", DataType::Float32, true),
                ],
                vec![index, right],
                1,
            )
            .unwrap();
        for fail in [5, 6] {
            let mut warnings = vec![];
            let mut operations = FailAt {
                fail,
                calls: vec![],
            };
            assert!(
                append_with_operations(
                    &frame,
                    &other,
                    &mut operations,
                    &mut warning_sink(&mut warnings)
                )
                .is_err()
            );
            assert_eq!(warnings.len(), usize::from(fail == 6));
            if fail == 6 {
                assert_eq!(warnings[0], ALL_NA_WARNING);
            }
        }
        let result = dataframe_append(&frame, &other).unwrap();
        assert_eq!(result.data().column(0).data_type(), &DataType::Float32);
    }

    #[test]
    fn object_boxing_failures_preserve_inputs_and_warning_order() {
        use arrow_array::{BooleanArray, Float32Array};
        let index = Arc::new(Int64Array::from(vec![0])) as ArrayRef;
        let boolean = Arc::new(BooleanArray::from(vec![true])) as ArrayRef;
        let float = Arc::new(Float32Array::from(vec![2.0])) as ArrayRef;
        let data = ArrowOperations
            .batch(
                vec![Field::new("x", DataType::Boolean, true)],
                vec![boolean.clone()],
                1,
            )
            .unwrap();
        let frame = IndexedFrame::new(index.clone(), None, data).unwrap();
        let other = ArrowOperations
            .batch(
                vec![
                    Field::new("datetime", DataType::Int64, true),
                    Field::new("x", DataType::Float32, true),
                ],
                vec![index, float.clone()],
                1,
            )
            .unwrap();
        let expected = [
            "cast", "cast", "concat", "cast", "cast", "cast", "cast", "concat", "batch",
        ];
        for fail in 0..expected.len() {
            let mut operations = FailAt {
                fail,
                calls: vec![],
            };
            let mut warnings = vec![];
            let error = append_with_operations(
                &frame,
                &other,
                &mut operations,
                &mut warning_sink(&mut warnings),
            )
            .unwrap_err();
            assert!(error.to_string().ends_with(&format!("failure at {fail}")));
            assert_eq!(operations.calls, expected[..=fail]);
            assert!(warnings.is_empty());
            assert!(Arc::ptr_eq(frame.data.column(0), &boolean));
            assert!(Arc::ptr_eq(other.column(1), &float));
        }
    }

    struct WrongRowCount;
    impl FrameOperations for WrongRowCount {
        fn cast(&mut self, a: &ArrayRef, dtype: &DataType) -> Result<ArrayRef, ArrowError> {
            ArrowOperations.cast(a, dtype)
        }
        fn concat(&mut self, a: &ArrayRef, b: &ArrayRef) -> Result<ArrayRef, ArrowError> {
            ArrowOperations.concat(a, b)
        }
        fn batch(
            &mut self,
            fields: Vec<Field>,
            columns: Vec<ArrayRef>,
            rows: usize,
        ) -> Result<RecordBatch, ArrowError> {
            ArrowOperations
                .batch(fields, columns, rows)
                .map(|batch| batch.slice(0, 0))
        }
    }

    #[test]
    fn malformed_backend_result_is_not_published() {
        let a = Arc::new(Int64Array::from(vec![1])) as ArrayRef;
        let data = ArrowOperations
            .batch(
                vec![Field::new("a", DataType::Int64, true)],
                vec![a.clone()],
                1,
            )
            .unwrap();
        let frame = IndexedFrame::new(a.clone(), None, data).unwrap();
        let other = ArrowOperations
            .batch(
                vec![
                    Field::new("datetime", DataType::Int64, true),
                    Field::new("a", DataType::Int64, true),
                ],
                vec![a.clone(), a],
                1,
            )
            .unwrap();
        let error = append_with_operations(&frame, &other, &mut WrongRowCount, &mut ignore_warning)
            .unwrap_err();
        assert_eq!(error.to_string(), "frame index and row count differ");
        assert_eq!(frame.data.num_rows(), 1);
    }
}
