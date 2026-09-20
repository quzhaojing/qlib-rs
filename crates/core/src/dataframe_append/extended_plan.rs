//! Source-compatible planning when extended object payloads participate.
use super::objects::{ArrowObjects, ObjectOperations};
use super::{
    BuiltinFrameValue as V, blocks::Plan, builtin_frame_array, builtin_frame_dtype,
    builtin_frame_values,
};
use arrow_array::{Array, ArrayRef, Float64Array, UnionArray, new_null_array};
use arrow_schema::{ArrowError, DataType};
use std::sync::Arc;

pub(super) fn is_extended(dtype: &DataType) -> bool {
    dtype == &builtin_frame_dtype()
}
fn object(dtype: &DataType) -> bool {
    is_extended(dtype) || super::objects::is_object(dtype)
}

pub(super) fn promote(array: &ArrayRef) -> Result<ArrayRef, ArrowError> {
    if is_extended(array.data_type()) {
        return Ok(array.clone());
    }
    let boxed = super::numeric_object_array(array)?;
    let union = boxed
        .as_any()
        .downcast_ref::<UnionArray>()
        .expect("numeric object union");
    let mut ids = union.type_ids().to_vec();
    for (i, id) in ids.iter_mut().enumerate() {
        if union.child(*id).is_null(union.value_offset(i)) {
            *id = 3;
        }
    }
    let floats = union
        .child(3)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("float object child");
    let mut children = (0..3).map(|id| union.child(id).clone()).collect::<Vec<_>>();
    children.push(Arc::new(Float64Array::from_iter_values(
        floats.iter().map(|value| value.unwrap_or(f64::NAN)),
    )));
    let fields = super::builtin_objects::fields();
    children.extend(
        fields
            .iter()
            .skip(4)
            .map(|(_, field)| new_null_array(field.data_type(), union.len())),
    );
    ArrowObjects.build(fields, ids, children)
}

fn missing(value: &V) -> bool {
    match value {
        V::None | V::PandasNa | V::NotATime => true,
        V::Float(v) => v.is_nan(),
        _ => false,
    }
}

struct State {
    all_na: bool,
    float_valid: bool,
    first_none: bool,
}
fn state(arrays: &[ArrayRef]) -> Result<State, ArrowError> {
    if !object(arrays[0].data_type()) {
        return Ok(State {
            all_na: arrays.iter().all(super::blocks::float_all_na),
            float_valid: true,
            first_none: false,
        });
    }
    let mut result = State {
        all_na: true,
        float_valid: true,
        first_none: false,
    };
    for (i, array) in arrays.iter().enumerate() {
        let values = builtin_frame_values(&promote(array)?)?;
        if i == 0 {
            result.first_none = matches!(values.first(), Some(V::None));
        }
        result.all_na &= values.iter().all(missing);
        result.float_valid &= !values.iter().any(|v| matches!(v, V::NotATime));
    }
    Ok(result)
}

pub(super) fn configure(
    plan: &mut Plan,
    left: &[ArrayRef],
    right: &[ArrayRef],
) -> Result<(), ArrowError> {
    let ld = left[0].data_type();
    let rd = right[0].data_type();
    if !left
        .iter()
        .chain(right)
        .any(|array| is_extended(array.data_type()))
    {
        return Ok(());
    }
    let supported = |dtype: &DataType| {
        object(dtype) || dtype.is_integer() || dtype.is_floating() || dtype == &DataType::Boolean
    };
    if !supported(ld) || !supported(rd) {
        return Ok(());
    }
    let l = state(left)?;
    let r = state(right)?;
    let dtype = if l.all_na && l.float_valid && rd.is_floating() && !r.all_na {
        rd.clone()
    } else if r.all_na && r.float_valid && ld.is_floating() && !l.all_na {
        ld.clone()
    } else {
        builtin_frame_dtype()
    };
    plan.warn = dtype.is_floating();
    plan.dtype = Some(dtype);
    let empty_object = (object(ld) && !l.all_na)
        || (object(rd) && !r.all_na)
        || (l.all_na && r.all_na)
        || (object(ld) && object(rd));
    if empty_object {
        for (slot, (dtype, state)) in plan.fill.iter_mut().zip([(ld, l), (rd, r)]) {
            if object(dtype) && state.all_na {
                *slot = Some(if state.first_none {
                    V::None
                } else {
                    V::Float(f64::NAN)
                });
            }
        }
    }
    Ok(())
}

pub(super) fn unbox_missing(array: &ArrayRef, dtype: &DataType) -> Result<ArrayRef, ArrowError> {
    let s = state(std::slice::from_ref(array))?;
    if dtype.is_floating() && s.all_na && s.float_valid {
        Ok(new_null_array(dtype, array.len()))
    } else {
        Err(ArrowError::CastError(format!(
            "cannot unbox extended objects as {dtype}"
        )))
    }
}

pub(super) fn fill(array: &ArrayRef, value: Option<&V>) -> Result<ArrayRef, ArrowError> {
    match value {
        Some(value) => builtin_frame_array(&vec![value.clone(); array.len()]),
        None => Ok(array.clone()),
    }
}

pub(super) fn alignment_fill(arrays: &[ArrayRef]) -> Result<Option<V>, ArrowError> {
    if !arrays.iter().any(|array| is_extended(array.data_type())) {
        return Ok(None);
    }
    let state = state(arrays)?;
    Ok(state.all_na.then_some(if state.first_none {
        V::None
    } else {
        V::Float(f64::NAN)
    }))
}
