//! A2C/PPO return preparation from evaluated critic tensors and replay metadata.
//! Candle handles tensors/casts and ndarray handles batch statistics. The scalar
//! merge and sequencing preserve Tianshou's return-normalization contract.

use crate::rl_episodic_return::{EpisodicReturnError, EpisodicReturnInput, episodic_returns};
use candle_core::{DType, Tensor};
use ndarray::{Array1, ArrayView1};
use num_traits::ToPrimitive;
use thiserror::Error;

/// Scalar return-distribution state, not the general vector observation normalizer.
#[derive(Clone, Debug, PartialEq)]
pub struct ReturnStatistics {
    pub mean: f64,
    pub variance: f64,
    pub count: usize,
}

impl Default for ReturnStatistics {
    fn default() -> Self {
        Self {
            mean: 0.,
            variance: 1.,
            count: 0,
        }
    }
}

impl ReturnStatistics {
    /// Source parallel-variance merge, including empty/nonfinite batch behavior.
    /// Does not skip the first merge: NaN * 0 remains NaN, as in `NumPy`.
    ///
    /// # Panics
    /// Panics without changing state if the cumulative count exceeds `usize::MAX`.
    pub fn update(&mut self, returns: &ArrayView1<'_, f64>) {
        let count = self
            .count
            .checked_add(returns.len())
            .expect("return count overflow");
        let batch_mean = returns.mean().unwrap_or(f64::NAN);
        let batch_var = returns
            .mapv(|value| (value - batch_mean).powi(2))
            .mean()
            .unwrap_or(f64::NAN);
        let batch_count = returns.len().to_f64().expect("array length fits f64");
        let old_count = self.count.to_f64().expect("count fits f64");
        let total_count = old_count + batch_count;
        let delta = batch_mean - self.mean;
        self.mean += delta * batch_count / total_count;
        let moment = self.variance * old_count
            + batch_var * batch_count
            + delta.powi(2) * old_count * batch_count / total_count;
        self.variance = moment / total_count;
        self.count = count;
    }
}

#[derive(Clone)]
pub struct CandleReturnInput<'a> {
    pub rewards: ArrayView1<'a, f64>,
    pub terminated: ArrayView1<'a, bool>,
    pub truncated: ArrayView1<'a, bool>,
    pub bootstrap_valid: ArrayView1<'a, bool>,
    pub indices: &'a [usize],
    pub unfinished_indices: &'a [usize],
    /// Evaluated critic outputs; flattened and detached during preparation.
    pub values: &'a Tensor,
    pub next_values: &'a Tensor,
}

#[derive(Debug)]
pub struct CandleReturns {
    pub old_values: Tensor,
    pub returns: Tensor,
    pub advantages: Tensor,
}

#[derive(Debug, Error)]
pub enum CandleReturnError {
    #[error("return preparation requires a nonempty batch")]
    Empty,
    #[error("critic values must both use the same F32 or F64 dtype")]
    Dtype,
    #[error(transparent)]
    Episodic(#[from] EpisodicReturnError),
    #[error(transparent)]
    Tensor(#[from] candle_core::Error),
}

fn estimates(values: &Tensor, scale: Option<f64>) -> Result<Array1<f64>, CandleReturnError> {
    let values = values.detach().flatten_all()?;
    let values = if let Some(scale) = scale {
        // NumPy 1.x scalar promotion retains F32 for a representable scalar;
        // multiply before widening so the intermediate F32 rounding is retained.
        // Its min-scalar-type threshold is 3.4e38 (not f32::MAX); nonfinite
        // scalars retain F32. Large finite scales instead promote the array.
        let values = if scale.is_finite() && scale.abs() >= 3.4e38 {
            values.to_dtype(DType::F64)?
        } else {
            values
        };
        let factor = Tensor::new(scale, values.device())?.to_dtype(values.dtype())?;
        values.broadcast_mul(&factor)?
    } else {
        values
    };
    Ok(Array1::from(values.to_dtype(DType::F64)?.to_vec1::<f64>()?))
}

/// Prepare detached PPO/A2C targets after critic evaluation. With normalization,
/// use the OLD variance plus 1e-8 to unnormalize estimates and normalize returns,
/// then update statistics from unnormalized F64 returns. Never subtract the mean.
/// Advantages remain unnormalized until the optional PPO minibatch step.
///
/// # Errors
/// Rejects empty batches, unsupported/mismatched critic dtypes and inconsistent
/// transition lengths. Propagates tensor backend errors. Network evaluation,
/// batch splitting, action conversion and old log probabilities are separate.
pub fn prepare_returns(
    input: &CandleReturnInput<'_>,
    gamma: f64,
    gae_lambda: f64,
    normalization: Option<&mut ReturnStatistics>,
) -> Result<CandleReturns, CandleReturnError> {
    if input.rewards.is_empty() {
        return Err(CandleReturnError::Empty);
    }
    if !matches!(input.values.dtype(), DType::F32 | DType::F64)
        || input.values.dtype() != input.next_values.dtype()
    {
        return Err(CandleReturnError::Dtype);
    }
    let scale = normalization
        .as_ref()
        .map(|stats| (stats.variance + 1e-8).sqrt());
    let values = estimates(input.values, scale)?;
    let next_values = estimates(input.next_values, scale)?;
    let mut prepared = episodic_returns(
        &EpisodicReturnInput {
            rewards: input.rewards,
            terminated: input.terminated,
            truncated: input.truncated,
            bootstrap_valid: input.bootstrap_valid,
            indices: input.indices,
            unfinished_indices: input.unfinished_indices,
            values: Some(values.view()),
            next_values: Some(next_values.view()),
        },
        gamma,
        gae_lambda,
    )?;
    if let Some(stats) = normalization {
        let normalized = &prepared.returns / (stats.variance + 1e-8).sqrt();
        stats.update(&prepared.returns.view());
        prepared.returns = normalized;
    }
    let old_values = input.values.detach().flatten_all()?;
    let returns = Tensor::new(prepared.returns.to_vec(), old_values.device())?
        .to_dtype(old_values.dtype())?;
    let advantages = Tensor::new(prepared.advantages.to_vec(), old_values.device())?
        .to_dtype(old_values.dtype())?;
    Ok(CandleReturns {
        old_values,
        returns,
        advantages,
    })
}

#[cfg(test)]
#[path = "../tests/support/rl_candle_returns.rs"]
mod tests;
