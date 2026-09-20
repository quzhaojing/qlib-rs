//! Pandas block identity retained alongside Arrow buffers.
use arrow_array::{ArrayRef, Float16Array, Float32Array, Float64Array, RecordBatch};
use arrow_schema::DataType;

use super::DataframeAppendError;
use super::{ColumnPair, IndexedFrame};

pub(super) struct Plan {
    pub positions: std::ops::Range<usize>,
    pub dtype: Option<DataType>,
    pub warn: bool,
    pub box_floats: bool,
    pub fill: [Option<super::BuiltinFrameValue>; 2],
}

pub(super) fn plans(
    frame: &IndexedFrame,
    other: &RecordBatch,
    names: &[ColumnPair],
    datetime: usize,
) -> Result<Vec<Plan>, DataframeAppendError> {
    let left_ids = ids(&frame.blocks, frame.data.num_columns());
    // set_index drops datetime; Pandas splits its original block into column blocks.
    let right_groups = consolidated(other)
        .into_iter()
        .flat_map(|group| {
            if group.contains(&datetime) {
                group.into_iter().map(|i| vec![i]).collect()
            } else {
                vec![group]
            }
        })
        .collect::<Vec<_>>();
    let right_ids = ids(&right_groups, other.num_columns());
    if let Some(plan) = homogeneous_plan(frame, other, names.len(), datetime, &right_groups) {
        return Ok(vec![plan]);
    }
    let keys: Vec<_> = names
        .iter()
        .map(|(_, (left, right))| (left.map(|i| left_ids[i]), right.map(|i| right_ids[i])))
        .collect();
    let mut plans = Vec::new();
    let mut start = 0;
    while start < names.len() {
        let mut end = start + 1;
        while end < names.len() && keys[end] == keys[start] {
            end += 1;
        }
        let mut plan = numeric_plan(frame, other, names, start..end);
        configure_objects(&mut plan, frame, other, &names[start..end])?;
        plans.push(plan);
        start = end;
    }
    Ok(plans)
}

fn homogeneous_plan(
    frame: &IndexedFrame,
    other: &RecordBatch,
    columns: usize,
    datetime: usize,
    right_groups: &[Vec<usize>],
) -> Option<Plan> {
    let right_data_groups = right_groups
        .iter()
        .filter(|group| !group.contains(&datetime))
        .collect::<Vec<_>>();
    if let ([left], [right]) = (frame.blocks.as_slice(), right_data_groups.as_slice()) {
        let dtype = frame.data.column(left[0]).data_type();
        if matches!(dtype, DataType::Float32 | DataType::Float64)
            && dtype == other.column(right[0]).data_type()
            && left.windows(2).all(|pair| pair[1] == pair[0] + 1)
        {
            return Some(Plan {
                positions: 0..columns,
                dtype: Some(dtype.clone()),
                warn: false,
                box_floats: false,
                fill: [None, None],
            });
        }
    }
    None
}

fn numeric_plan(
    frame: &IndexedFrame,
    other: &RecordBatch,
    names: &[ColumnPair],
    positions: std::ops::Range<usize>,
) -> Plan {
    let start = positions.start;
    let end = positions.end;
    let mut dtype = None;
    let mut warn = false;
    let mut box_floats = false;
    if let (Some(left), Some(right)) = names[start].1 {
        let ld = frame.data.column(left).data_type();
        let rd = other.column(right).data_type();
        if let Some(common) = float_common(ld, rd).or_else(|| temporal_common(ld, rd)) {
            let left_na = names[start..end].iter().all(|(_, (left, _))| {
                nullable_all_na(frame.data.column(left.expect("same block pair")))
            });
            let right_na = names[start..end].iter().all(|(_, (_, right))| {
                nullable_all_na(other.column(right.expect("same block pair")))
            });
            let selected = match (left_na, right_na) {
                (true, false) => rd.clone(),
                (false, true) => ld.clone(),
                _ => common.clone(),
            };
            warn = selected != common;
            dtype = Some(selected);
        } else {
            dtype = super::numeric::common(ld, rd);
            let left_na = names[start..end].iter().all(|(_, (left, _))| {
                object_or_float_na(frame.data.column(left.expect("same block pair")))
            });
            let right_na = names[start..end].iter().all(|(_, (_, right))| {
                object_or_float_na(other.column(right.expect("same block pair")))
            });
            if (ld == &DataType::Boolean && rd.is_floating())
                || (ld.is_floating() && rd == &DataType::Boolean)
            {
                if ld.is_floating() && !left_na {
                    dtype = Some(ld.clone());
                } else {
                    dtype = Some(super::numeric_object_dtype());
                    box_floats = true;
                }
            } else if (super::objects::is_object(ld) || super::objects::is_object(rd))
                && native_or_object(ld)
                && native_or_object(rd)
            {
                let selected = if left_na && rd.is_floating() && !right_na {
                    rd.clone()
                } else if right_na && ld.is_floating() && !left_na {
                    ld.clone()
                } else {
                    super::numeric_object_dtype()
                };
                warn = !super::objects::is_object(&selected);
                dtype = Some(selected);
            }
        }
    }
    Plan {
        positions,
        dtype,
        warn,
        box_floats,
        fill: [None, None],
    }
}

fn configure_objects(
    plan: &mut Plan,
    frame: &IndexedFrame,
    other: &RecordBatch,
    names: &[ColumnPair],
) -> Result<(), DataframeAppendError> {
    // Pandas omits a 0x0 manager; it must not normalize the surviving object's sentinels.
    if (frame.data.num_rows() == 0 && frame.data.num_columns() == 0)
        || (other.num_rows() == 0 && other.num_columns() == 1)
    {
        return Ok(());
    }
    if let (Some(_), Some(_)) = names[0].1 {
        let left = names
            .iter()
            .map(|(_, (i, _))| frame.data.column(i.expect("same block pair")).clone())
            .collect::<Vec<_>>();
        let right = names
            .iter()
            .map(|(_, (_, i))| other.column(i.expect("same block pair")).clone())
            .collect::<Vec<_>>();
        if !super::temporal_plan::configure(plan, &left, &right)? {
            super::extended_plan::configure(plan, &left, &right)?;
        }
    } else {
        for (side, is_left) in [(0, true), (1, false)] {
            let arrays = names
                .iter()
                .filter_map(|(_, (left, right))| {
                    if is_left {
                        left.map(|i| frame.data.column(i).clone())
                    } else {
                        right.map(|i| other.column(i).clone())
                    }
                })
                .collect::<Vec<_>>();
            if !arrays.is_empty() {
                plan.fill[side] = super::temporal_plan::alignment_fill(&arrays)?;
            }
        }
    }
    Ok(())
}

fn native_or_object(dtype: &DataType) -> bool {
    dtype.is_integer()
        || dtype.is_floating()
        || dtype == &DataType::Boolean
        || super::objects::is_object(dtype)
}

fn temporal_common(left: &DataType, right: &DataType) -> Option<DataType> {
    match (left, right) {
        (DataType::Timestamp(lu, lz), DataType::Timestamp(ru, rz)) if lz == rz => {
            Some(DataType::Timestamp((*lu).max(*ru), lz.clone()))
        }
        _ => None,
    }
}

fn nullable_all_na(array: &ArrayRef) -> bool {
    match array.data_type() {
        // Pandas timezone-aware extension blocks retain their resolution even when empty/all-NaT.
        DataType::Timestamp(_, Some(_)) => false,
        DataType::Timestamp(_, None) => array.null_count() == array.len(),
        _ => float_all_na(array),
    }
}

fn object_or_float_na(array: &ArrayRef) -> bool {
    if super::objects::is_object(array.data_type()) {
        super::objects::all_na(array)
    } else {
        float_all_na(array)
    }
}

pub(super) fn float_common(left: &DataType, right: &DataType) -> Option<DataType> {
    use DataType::{Float16, Float32, Float64};
    match (left, right) {
        (Float16 | Float32 | Float64, Float16 | Float32 | Float64) => {
            if left == &Float64 || right == &Float64 {
                Some(Float64)
            } else if left == &Float32 || right == &Float32 {
                Some(Float32)
            } else {
                Some(Float16)
            }
        }
        _ => None,
    }
}

pub(super) fn consolidated(data: &RecordBatch) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (position, array) in data.columns().iter().enumerate() {
        if let Some(group) = groups.iter_mut().find(|group| {
            // Pandas timezone-aware arrays are non-consolidatable extension blocks.
            !matches!(array.data_type(), DataType::Timestamp(_, Some(_)))
                && logical_type_eq(data.column(group[0]).data_type(), array.data_type())
        }) {
            group.push(position);
        } else {
            groups.push(vec![position]);
        }
    }
    groups
}

pub(super) fn validate(
    data: &RecordBatch,
    groups: &[Vec<usize>],
) -> Result<(), DataframeAppendError> {
    let mut seen = vec![false; data.num_columns()];
    for group in groups {
        let Some(&first) = group.first() else {
            return Err(DataframeAppendError::BlockLayout("empty block"));
        };
        if first >= seen.len() {
            return Err(DataframeAppendError::BlockLayout("column out of bounds"));
        }
        for &position in group {
            let Some(slot) = seen.get_mut(position) else {
                return Err(DataframeAppendError::BlockLayout("column out of bounds"));
            };
            if *slot {
                return Err(DataframeAppendError::BlockLayout(
                    "column occurs more than once",
                ));
            }
            *slot = true;
            if !logical_type_eq(
                data.column(position).data_type(),
                data.column(first).data_type(),
            ) {
                return Err(DataframeAppendError::BlockLayout(
                    "mixed dtypes in one block",
                ));
            }
        }
    }
    if seen.contains(&false) {
        return Err(DataframeAppendError::BlockLayout(
            "column omitted from blocks",
        ));
    }
    Ok(())
}

pub(super) fn logical_type_eq(left: &DataType, right: &DataType) -> bool {
    left == right || (logical_object(left) && logical_object(right))
}

pub(super) fn logical_object(dtype: &DataType) -> bool {
    super::objects::is_object(dtype)
        || super::extended_plan::is_extended(dtype)
        || dtype == &super::temporal_frame_dtype()
}

pub(super) fn ids(groups: &[Vec<usize>], count: usize) -> Vec<usize> {
    let mut ids = vec![0; count];
    for (id, group) in groups.iter().enumerate() {
        for &position in group {
            ids[position] = id;
        }
    }
    ids
}

pub(super) fn float_all_na(array: &ArrayRef) -> bool {
    if let Some(array) = array.as_any().downcast_ref::<Float16Array>() {
        array
            .iter()
            .all(|value| value.is_none_or(half::f16::is_nan))
    } else if let Some(array) = array.as_any().downcast_ref::<Float32Array>() {
        array.iter().all(|value| value.is_none_or(f32::is_nan))
    } else if let Some(array) = array.as_any().downcast_ref::<Float64Array>() {
        array.iter().all(|value| value.is_none_or(f64::is_nan))
    } else {
        false
    }
}
