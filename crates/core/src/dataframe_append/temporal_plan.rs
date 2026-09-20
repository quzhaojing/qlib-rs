//! Missing-value validity and common dtype planning for temporal join units.
use super::{
    BuiltinFrameValue as B, TemporalFrameValue as V,
    blocks::{Plan, logical_object, logical_type_eq},
    temporal_cast, temporal_frame_dtype,
};
use arrow_array::ArrayRef;
use arrow_schema::{ArrowError, DataType};

struct State {
    all_na: bool,
    float_valid: bool,
    first_none: bool,
}

fn state(arrays: &[ArrayRef]) -> Result<State, ArrowError> {
    let mut result = State {
        all_na: true,
        float_valid: true,
        first_none: false,
    };
    if logical_object(arrays[0].data_type()) {
        for (i, array) in arrays.iter().enumerate() {
            let values = temporal_cast::values(array)?;
            if i == 0 {
                result.first_none = matches!(values.first(), Some(V::Builtin(B::None)));
            }
            result.all_na &= values.iter().all(temporal_cast::missing);
            result.float_valid &= !values.iter().any(|v| matches!(v, V::Builtin(B::NotATime)));
        }
    } else if temporal_cast::is_temporal(arrays[0].data_type()) {
        result.all_na = arrays.iter().all(|array| array.null_count() == array.len());
    } else {
        result.all_na = arrays.iter().all(super::blocks::float_all_na);
    }
    Ok(result)
}

fn common(left: &DataType, right: &DataType) -> DataType {
    if left == right && !logical_object(left) {
        return left.clone();
    }
    match (left, right) {
        (DataType::Timestamp(lu, lz), DataType::Timestamp(ru, rz)) if lz == rz => {
            DataType::Timestamp((*lu).max(*ru), lz.clone())
        }
        (DataType::Duration(left), DataType::Duration(right)) => {
            DataType::Duration((*left).max(*right))
        }
        _ => temporal_frame_dtype(),
    }
}

fn valid_na(state: &State, source: &DataType, target: &DataType) -> bool {
    if !state.all_na {
        return false;
    }
    if logical_object(source) {
        return logical_object(target)
            || temporal_cast::is_temporal(target)
            || (target.is_floating() && state.float_valid);
    }
    match source {
        DataType::Timestamp(_, Some(_)) => source == target,
        DataType::Timestamp(_, None) => {
            matches!(target, DataType::Timestamp(_, None)) || logical_object(target)
        }
        DataType::Duration(_) => matches!(target, DataType::Duration(_)) || logical_object(target),
        _ => target.is_floating() || temporal_cast::is_temporal(target) || logical_object(target),
    }
}

pub(super) fn configure(
    plan: &mut Plan,
    left: &[ArrayRef],
    right: &[ArrayRef],
) -> Result<bool, ArrowError> {
    if !left.iter().chain(right).any(|a| {
        temporal_cast::is_temporal(a.data_type()) || a.data_type() == &temporal_frame_dtype()
    }) {
        return Ok(false);
    }
    let supported = |dtype: &DataType| {
        logical_object(dtype)
            || temporal_cast::is_temporal(dtype)
            || dtype.is_integer()
            || dtype.is_floating()
            || dtype == &DataType::Boolean
    };
    let ld = left[0].data_type();
    let rd = right[0].data_type();
    if !supported(ld) || !supported(rd) {
        return Ok(false);
    }
    let l = state(left)?;
    let r = state(right)?;
    let future = common(ld, rd);
    let empty = match (l.all_na, r.all_na) {
        (true, false) => rd.clone(),
        (false, true) => ld.clone(),
        _ => future.clone(),
    };
    let mut selected = [ld.clone(), rd.clone()];
    for (i, (dtype, state)) in [(ld, &l), (rd, &r)].into_iter().enumerate() {
        if valid_na(state, dtype, &empty) {
            selected[i] = empty.clone();
            plan.fill[i] = Some(if logical_object(dtype) && state.first_none {
                B::None
            } else if temporal_cast::is_temporal(&empty) {
                B::NotATime
            } else {
                B::Float(f64::NAN)
            });
        }
    }
    let dtype = common(&selected[0], &selected[1]);
    plan.warn = !logical_type_eq(&empty, &future) && logical_type_eq(&empty, &dtype);
    plan.dtype = Some(dtype);
    plan.box_floats = false;
    Ok(true)
}

pub(super) fn alignment_fill(arrays: &[ArrayRef]) -> Result<Option<B>, ArrowError> {
    if !arrays
        .iter()
        .any(|a| a.data_type() == &temporal_frame_dtype())
    {
        return super::extended_plan::alignment_fill(arrays);
    }
    let state = state(arrays)?;
    Ok(state.all_na.then_some(if state.first_none {
        B::None
    } else {
        B::Float(f64::NAN)
    }))
}

#[cfg(test)]
#[path = "temporal_plan_tests.rs"]
mod tests;
