//! Tianshou n-step return preparation for native value-based policies.
//! ndarray owns array arithmetic; Candle owns tensor reshaping/casts.

use candle_core::{DType, Device, Tensor};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use thiserror::Error;

/// Read-only replay seam. Physical columns include allocated, unwritten slots.
pub trait CandleNStepReplay {
    /// # Errors
    /// Returns missing reward storage or backend failures.
    fn rewards(&self) -> Result<Array1<f64>, String>;
    /// # Errors
    /// Returns missing navigation metadata or invalid physical indices.
    fn next_indices(&self, indices: &[usize]) -> Result<Vec<usize>, String>;
    /// # Errors
    /// Returns missing termination storage or invalid indices.
    fn bootstrap_mask(&self, indices: &[usize]) -> Result<Array1<bool>, String>;
    /// # Errors
    /// Returns missing end-flag storage or backend failures.
    fn end_flags(&self) -> Result<Array1<bool>, String>;
    /// # Errors
    /// Returns missing end-flag storage or backend failures.
    fn unfinished(&self) -> Result<Vec<usize>, String>;
}

#[derive(Clone, Copy, Debug)]
pub struct CandleNStepConfig {
    pub gamma: f64,
    pub steps: usize,
    pub reward_normalization: bool,
}
impl Default for CandleNStepConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            steps: 1,
            reward_normalization: false,
        }
    }
}

#[derive(Default)]
pub struct CandleNStepBatch {
    pub returns: Option<Tensor>,
    pub weight: Option<Tensor>,
}

#[derive(Debug, Error)]
pub enum CandleNStepError {
    #[error("reward normalization is unsupported for n-step returns")]
    RewardNormalization,
    #[error("n-step horizon must be positive")]
    Horizon,
    #[error("cannot infer the target width for an empty batch")]
    Empty,
    #[error("invalid n-step metadata shape or physical index")]
    Metadata,
    #[error("the source n-step path does not support target dtype {0:?}")]
    Dtype(DType),
    #[error("n-step {stage} adapter failed: {message}")]
    Adapter {
        stage: &'static str,
        message: String,
    },
    #[error(transparent)]
    Tensor(#[from] candle_core::Error),
}

fn adapter<T>(stage: &'static str, result: Result<T, String>) -> Result<T, CandleNStepError> {
    result.map_err(|message| CandleNStepError::Adapter { stage, message })
}

/// Reverse reward recurrence with source endpoint resets and sequential gamma powers.
/// Target values must already have the terminal bootstrap mask applied.
/// # Errors
/// Returns empty horizon or inconsistent physical metadata/target dimensions.
pub fn nstep_returns(
    rewards: &ArrayView1<'_, f64>,
    end_flags: &ArrayView1<'_, bool>,
    target: &ArrayView2<'_, f64>,
    indices: &[Vec<usize>],
    gamma: f64,
) -> Result<Array2<f64>, CandleNStepError> {
    if indices.is_empty() {
        return Err(CandleNStepError::Horizon);
    }
    if end_flags.len() != rewards.len()
        || indices.iter().any(|row| {
            row.len() != target.nrows() || row.iter().any(|&index| index >= rewards.len())
        })
    {
        return Err(CandleNStepError::Metadata);
    }
    let mut gamma_powers = vec![1.];
    for step in 0..indices.len() {
        gamma_powers.push(gamma_powers[step] * gamma);
    }
    let mut horizons = vec![indices.len(); target.nrows()];
    let mut returns = Array2::zeros(target.raw_dim());
    for (step, positions) in indices.iter().enumerate().rev() {
        for (row, &position) in positions.iter().enumerate() {
            let mut values = returns.row_mut(row);
            if end_flags[position] {
                horizons[row] = step + 1;
                values.fill(0.);
            }
            values.mapv_inplace(|value| rewards[position] + gamma * value);
        }
    }
    for ((mut values, target), horizon) in returns
        .rows_mut()
        .into_iter()
        .zip(target.rows())
        .zip(horizons)
    {
        values += &target.mapv(|value| value * gamma_powers[horizon]);
    }
    Ok(returns)
}

/// Read rewards → navigate → evaluate detached Q → mask → mark unfinished → compute.
/// Commits returns before converting existing replay weights, matching source order.
/// The target callback may update policy-local state but borrows replay read-only.
/// # Errors
/// Returns the first adapter, shape, dtype or arithmetic-boundary failure.
pub fn prepare_nstep_returns<B: CandleNStepReplay + ?Sized>(
    batch: &mut CandleNStepBatch,
    buffer: &B,
    indices: &[usize],
    target_q: impl FnOnce(&B, &[usize]) -> Result<Tensor, String>,
    config: CandleNStepConfig,
) -> Result<(), CandleNStepError> {
    if config.reward_normalization {
        return Err(CandleNStepError::RewardNormalization);
    }
    if config.steps == 0 {
        return Err(CandleNStepError::Horizon);
    }
    let rewards = adapter("rewards", buffer.rewards())?;
    let mut walk = vec![indices.to_vec()];
    for step in 1..config.steps {
        walk.push(adapter("next", buffer.next_indices(&walk[step - 1]))?);
    }
    let terminal = &walk[config.steps - 1];
    let target = adapter("target Q", target_q(buffer, terminal))?.detach();
    if indices.is_empty() {
        return Err(CandleNStepError::Empty);
    }
    // Source reshape flattens all non-batch dimensions even for rank-one targets.
    let target = target.reshape((indices.len(), target.elem_count() / indices.len()))?;
    if target.dtype() == DType::BF16 {
        return Err(CandleNStepError::Dtype(target.dtype()));
    }
    let mask = adapter("bootstrap", buffer.bootstrap_mask(terminal))?;
    if mask.len() != indices.len() {
        return Err(CandleNStepError::Metadata);
    }
    let mut end_flags = adapter("end flags", buffer.end_flags())?;
    for index in adapter("unfinished", buffer.unfinished())? {
        *end_flags.get_mut(index).ok_or(CandleNStepError::Metadata)? = true;
    }
    // NumPy conversion rejects BF16 earlier; source Numba rejects F16 here.
    if target.dtype() == DType::F16 {
        return Err(CandleNStepError::Dtype(target.dtype()));
    }
    let rows = target
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F64)?
        .to_vec2::<f64>()?;
    let mut values = Array2::zeros((indices.len(), target.dim(1)?));
    for ((mut row, source), valid) in values.rows_mut().into_iter().zip(rows).zip(mask) {
        row.assign(&Array1::from(source));
        row *= f64::from(u8::from(valid));
    }
    let returns = nstep_returns(
        &rewards.view(),
        &end_flags.view(),
        &values.view(),
        &walk,
        config.gamma,
    )?;
    let data: Vec<_> = returns.iter().copied().collect();
    batch.returns =
        Some(Tensor::from_vec(data, target.shape(), target.device())?.to_dtype(target.dtype())?);
    if let Some(weight) = &batch.weight {
        batch.weight = Some(
            weight
                .to_device(target.device())?
                .to_dtype(target.dtype())?,
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "../tests/support/rl_candle_nstep.rs"]
mod tests;
