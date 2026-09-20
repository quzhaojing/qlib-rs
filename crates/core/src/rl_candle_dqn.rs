//! Native DQN numerical and prepared-batch learning boundary.
//! Candle owns tensors/autograd/Adam; ndarray and rand own exploration arrays.
//! Target-network lifecycle and replay scheduling are policy-level operations.

use crate::rl_candle_nstep::CandleNStepBatch;
use crate::rl_candle_optimizer::CandleAdam;
use candle_core::{CpuStorage, CustomOp2, DType, Error, Layout, Result, Shape, Tensor, WithDType};
use ndarray::{Array2, ArrayView2};
use num_traits::{One, WrappingSub};
use rand::Rng;

/// Raw model output is retained separately from masked action selection.
pub struct DqnForward<State> {
    pub logits: Tensor,
    pub actions: Tensor,
    pub state: State,
}

// NumPy integer masks use modular subtraction. Candle's ordinary integer Sub
// uses checked Rust arithmetic in debug builds, while float conversion would
// lose I64 bits. Keep this small compatibility boundary exact using std integer
// wrapping through num-traits, then return to the original tensor device/shape.
fn integer_mask_complement<T: WithDType + One + WrappingSub>(mask: &Tensor) -> Result<Tensor> {
    let values = mask
        .flatten_all()?
        .to_vec1::<T>()?
        .into_iter()
        .map(|value| T::one().wrapping_sub(&value))
        .collect::<Vec<_>>();
    Tensor::from_vec(values, mask.shape(), mask.device())
}

fn mask_complement(mask: &Tensor) -> Result<Tensor> {
    match (mask.dtype(), mask.rank() == 0) {
        // NumPy 1.x promotes zero-dimensional unsigned arrays before Python
        // scalar subtraction (U8 to signed native int, U32 to I64 on Windows).
        // Their complements both fit I64, unlike non-scalar modular results.
        (DType::U8 | DType::U32, true) => {
            integer_mask_complement::<i64>(&mask.to_dtype(DType::I64)?)
        }
        (DType::U8, false) => integer_mask_complement::<u8>(mask),
        (DType::U32, false) => integer_mask_complement::<u32>(mask),
        (DType::I64, _) => integer_mask_complement::<i64>(mask),
        (DType::F16 | DType::F32, true) => {
            let promoted = mask.to_dtype(DType::F64)?;
            promoted.ones_like()?.sub(&promoted)
        }
        // Subtract in the mask dtype BEFORE conversion to the logits dtype.
        // In particular this retains the source F16/BF16 rounding boundary.
        // BF16 has no NumPy representation and follows Torch tensor semantics.
        _ => mask.ones_like()?.sub(mask),
    }
}

/// Apply the source global min/max penalty, allowing broadcast numeric masks.
/// A mask need not be Boolean; all-masked rows still select a maximum.
/// Mask arithmetic follows `NumPy` 1.x arrays, including scalar promotion; BF16
/// follows Torch tensors because `NumPy` has no corresponding dtype.
/// # Errors
/// Returns empty masked reductions, incompatible broadcasting or backend errors.
pub fn dqn_q_values(logits: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
    let Some(mask) = mask else {
        return Ok(logits.clone());
    };
    let flat = logits.flatten_all()?;
    let penalty = flat.min(0)?.sub(&flat.max(0)?)?.affine(1., -1.)?;
    // Candle reductions do not propagate a NaN that follows a finite value.
    // Source global min/max do; keep the global penalty NaN in that case.
    let penalty = flat
        .ne(&flat)?
        .to_dtype(DType::F32)?
        .sum_all()?
        .gt(0.)?
        .where_cond(
            &Tensor::new(f64::NAN, logits.device())?.to_dtype(logits.dtype())?,
            &penalty,
        )?;
    let unavailable = mask_complement(mask)?
        .to_device(logits.device())?
        .to_dtype(logits.dtype())?;
    logits.broadcast_add(&unavailable.broadcast_mul(&penalty)?)
}

// Torch max chooses the first NaN, or the first maximum in a tie.
pub(crate) fn action_argmax(values: &Tensor) -> Result<Tensor> {
    let (rows, columns) = values.dims2()?;
    if columns == 0 {
        return Err(Error::Msg("DQN argmax requires actions".into()));
    }
    if rows == 0 {
        return Tensor::zeros(0, DType::I64, values.device());
    }
    let nan = values.ne(values)?;
    let any_nan = nan.to_dtype(DType::F32)?.sum(1)?.gt(0.)?;
    any_nan
        .where_cond(&nan.argmax(1)?, &values.argmax(1)?)?
        .to_dtype(DType::I64)
}

/// Select actions while returning the original logits and unchanged caller state.
/// # Errors
/// Returns non-matrix logits, invalid masks, empty action axes or backend errors.
pub fn dqn_forward<State>(
    logits: Tensor,
    state: State,
    mask: Option<&Tensor>,
) -> Result<DqnForward<State>> {
    logits.dims2()?;
    let actions = action_argmax(&dqn_q_values(&logits, mask)?)?;
    Ok(DqnForward {
        logits,
        actions,
        state,
    })
}

/// Batched Torch-style signed action indexing, including negative positions.
/// # Errors
/// Returns non-I64/non-vector actions, mismatched rows or out-of-range actions.
pub fn dqn_action_values(logits: &Tensor, actions: &Tensor) -> Result<Tensor> {
    let (rows, columns) = logits.dims2()?;
    if actions.dtype() != DType::I64 || actions.dims() != [rows] {
        return Err(Error::Msg("DQN actions must be an I64 batch vector".into()));
    }
    let columns =
        i64::try_from(columns).map_err(|_| Error::Msg("action count exceeds I64".into()))?;
    let offset = Tensor::new(columns, actions.device())?;
    let negative = actions.lt(0_i64)?;
    let shifted = negative
        .where_cond(actions, &actions.zeros_like()?)?
        .broadcast_add(&offset)?;
    let normalized = negative.where_cond(&shifted, actions)?;
    logits
        .gather(&normalized.unsqueeze(1)?.contiguous()?, 1)?
        .squeeze(1)
}

/// Double DQN gathers target logits using online masked actions; Nature DQN
/// instead maximizes raw target logits, even if the maximizing action is masked.
/// # Errors
/// Returns invalid action/target shapes or tensor backend failures.
pub fn dqn_target_values(
    online: &DqnForward<impl Sized>,
    target_logits: Option<&Tensor>,
    is_double: bool,
) -> Result<Tensor> {
    let target = target_logits.unwrap_or(&online.logits);
    if is_double {
        dqn_action_values(target, &online.actions)
    } else {
        // Gathering the first maximum also preserves source NaN behavior.
        dqn_action_values(target, &action_argmax(target)?)
    }
}

pub struct DqnLoss {
    pub loss: Tensor,
    /// Signed (returns - selected Q), not absolute or detached priorities.
    pub td_error: Tensor,
}

// Inactive square branches otherwise produce 0 * infinity during generic
// autodiff. Torch Huber has finite saturated slopes at infinities and NaN at
// NaN. The hook composes Candle operations, not custom numeric kernels.
struct SourceHuber;
impl CustomOp2 for SourceHuber {
    fn name(&self) -> &'static str {
        "source-dqn-huber"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        output: &CpuStorage,
        layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        Ok((output.clone(), layout.shape().clone()))
    }
    fn bwd(
        &self,
        error: &Tensor,
        _: &Tensor,
        _: &Tensor,
        gradient: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let ones = error.ones_like()?;
        let lower = error.lt(-1.)?.where_cond(&ones.neg()?, error)?;
        let slope = error.gt(1.)?.where_cond(&ones, &lower)?;
        Ok((Some(gradient.mul(&slope)?), None))
    }
}

fn product_dtype(left: &Tensor, right: &Tensor) -> DType {
    if right.rank() == 0 || !right.dtype().is_float() || left.dtype() == right.dtype() {
        left.dtype()
    } else if left.dtype() == DType::F64 || right.dtype() == DType::F64 {
        DType::F64
    } else {
        DType::F32
    }
}

/// Source prepared DQN loss. Returns flatten/cast to Q; MSE weights broadcast
/// without flattening (column weights produce an outer product). Huber ignores
/// weights entirely and uses delta one. Both retain automatic differentiation.
/// # Errors
/// Returns invalid action indices, incompatible broadcasting or backend errors.
pub fn dqn_loss(
    logits: &Tensor,
    actions: &Tensor,
    returns: &Tensor,
    weight: Option<&Tensor>,
    huber: bool,
) -> Result<DqnLoss> {
    let q = dqn_action_values(logits, actions)?;
    selected_loss(&q, returns, weight, huber)
}

fn selected_loss(
    q: &Tensor,
    returns: &Tensor,
    weight: Option<&Tensor>,
    huber: bool,
) -> Result<DqnLoss> {
    let returns = returns
        .flatten_all()?
        .to_device(q.device())?
        .to_dtype(q.dtype())?;
    let td_error = returns.broadcast_sub(q)?;
    let loss = if huber {
        let absolute = td_error.abs()?;
        let elementwise = absolute.lt(1.)?.where_cond(
            &td_error.sqr()?.affine(0.5, 0.)?,
            &absolute.affine(1., -0.5)?,
        )?;
        td_error
            .apply_op2(&elementwise.detach().copy()?, SourceHuber)?
            .mean_all()?
    } else {
        let squared = td_error.sqr()?;
        match weight {
            Some(weight) => {
                let dtype = product_dtype(&squared, weight);
                squared
                    .to_dtype(dtype)?
                    .broadcast_mul(&weight.to_dtype(dtype)?)?
                    .mean_all()?
            }
            None => squared.mean_all()?,
        }
    };
    Ok(DqnLoss { loss, td_error })
}

/// Learn one already-processed batch. Removes old priorities before evaluating
/// the model; publishes signed TD errors before backward/Adam. A failure retains
/// preceding mutations. Target synchronization/iteration counting belong to the
/// outer DQN policy and must occur before/after this operation respectively.
/// # Errors
/// Returns model, missing-return, loss, backward or optimizer failures.
pub fn learn_dqn_batch(
    batch: &mut CandleNStepBatch,
    actions: &Tensor,
    forward: impl FnOnce() -> Result<Tensor>,
    optimizer: &mut CandleAdam,
    huber: bool,
) -> Result<f64> {
    let weight = batch.weight.take();
    let logits = forward()?;
    let q = dqn_action_values(&logits, actions)?;
    let returns = batch
        .returns
        .as_ref()
        .ok_or_else(|| Error::Msg("DQN batch lacks returns".into()))?;
    let output = selected_loss(&q, returns, weight.as_ref(), huber)?;
    batch.weight = Some(output.td_error);
    let gradients = output.loss.backward()?;
    optimizer.step(&gradients)?;
    output.loss.to_dtype(DType::F64)?.to_scalar()
}

/// Epsilon-greedy exploration with caller-owned RNG. Near-zero epsilon skips all
/// draws; otherwise draws the whole row-selection vector before the score matrix,
/// adds the observation mask and mutates only selected actions. This preserves
/// source random-operation ordering, not `NumPy`'s seed-to-bitstream mapping.
/// # Errors
/// Returns an empty action space, invalid mask broadcasting or native index overflow.
pub fn dqn_exploration<R: Rng + ?Sized>(
    actions: &mut [i64],
    epsilon: f64,
    action_count: usize,
    mask: Option<&ArrayView2<'_, f64>>,
    rng: &mut R,
) -> Result<()> {
    if epsilon.abs() <= 1e-8 {
        return Ok(());
    }
    let selected: Vec<_> = (0..actions.len())
        .map(|_| rng.random::<f64>() < epsilon)
        .collect();
    let mut scores =
        Array2::from_shape_simple_fn((actions.len(), action_count), || rng.random::<f64>());
    if let Some(mask) = mask {
        let mask = mask
            .broadcast(scores.raw_dim())
            .ok_or_else(|| Error::Msg("DQN exploration mask does not broadcast".into()))?;
        scores += &mask;
    }
    if action_count == 0 {
        return Err(Error::Msg("DQN exploration requires actions".into()));
    }
    for ((action, selected), row) in actions.iter_mut().zip(selected).zip(scores.rows()) {
        // NumPy argmax chooses the first maximum and first NaN.
        let index = row.iter().enumerate().fold(0, |best, (index, &value)| {
            if !row[best].is_nan() && (value.is_nan() || value > row[best]) {
                index
            } else {
                best
            }
        });
        if selected {
            *action =
                i64::try_from(index).map_err(|_| Error::Msg("action index exceeds I64".into()))?;
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "../tests/support/rl_candle_dqn.rs"]
mod tests;
