//! Differentiable discrete PPO minibatch loss used by Qlib/Tianshou.
//! Native Candle operators own arithmetic and differentiation; this module fixes
//! source-specific probability, clipping, dtype and reduction contracts.

use crate::rl_candle_categorical::{CandleCategorical, clamp};
use candle_core::{DType, Error, Result, Tensor};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct PpoLossConfig {
    pub eps_clip: f64,
    pub value_clip: bool,
    pub normalize_advantage: bool,
    pub value_weight: f64,
    pub entropy_weight: f64,
    pub dual_clip: Option<f64>,
}
impl Default for PpoLossConfig {
    fn default() -> Self {
        Self {
            eps_clip: 0.3,
            value_clip: true,
            normalize_advantage: true,
            value_weight: 1.,
            entropy_weight: 0.01,
            dual_clip: None,
        }
    }
}

/// Prepared discrete minibatch. Actor probabilities have shape [batch, actions];
/// all other fields are [batch], with integral action values and one floating
/// dtype for value/log-probability/advantage/return tensors. Actions may use an
/// integer or floating dtype, as produced by source PPO preprocessing.
/// Returns/advantages/old predictions come from the separate rollout/GAE stage.
pub struct PpoLossInput<'a> {
    pub probabilities: &'a Tensor,
    pub values: &'a Tensor,
    pub actions: &'a Tensor,
    pub old_log_prob: &'a Tensor,
    pub advantages: &'a Tensor,
    pub returns: &'a Tensor,
    /// Required only when value clipping is enabled; otherwise not inspected.
    pub old_values: Option<&'a Tensor>,
}

pub struct PpoLoss {
    pub total: Tensor,
    pub policy: Tensor,
    pub value: Tensor,
    pub entropy: Tensor,
    /// Source PPO normalizes the minibatch advantages before calculating loss.
    pub advantages: Tensor,
}

// Native min/max split ties correctly. Torch additionally propagates a NaN and
// passes the incoming gradient to BOTH operands when either operand is NaN.
fn extreme(left: &Tensor, right: &Tensor, maximum: bool) -> Result<Tensor> {
    let value = if maximum {
        left.maximum(right)?
    } else {
        left.minimum(right)?
    };
    let nan = left.ne(left)?.add(&right.ne(right)?)?.gt(0_u8)?;
    nan.where_cond(&left.add(right)?, &value)
}

// Candle's affine(0, _) prunes the input from its autograd graph. Torch's zero
// loss coefficient retains zero gradients, which still initialize/decay Adam.
fn weighted(input: &Tensor, weight: f64) -> Result<Tensor> {
    input.broadcast_mul(&Tensor::new(weight, input.device())?.to_dtype(input.dtype())?)
}

fn categorical(input: &PpoLossInput<'_>) -> Result<(Tensor, Tensor)> {
    let (batch, actions) = input.probabilities.dims2()?;
    if batch == 0 || actions == 0 {
        return Err(Error::Msg(
            "PPO requires a nonempty discrete minibatch".into(),
        ));
    }
    let dtype = input.probabilities.dtype();
    for tensor in [
        input.values,
        input.old_log_prob,
        input.advantages,
        input.returns,
    ] {
        if tensor.dims() != [batch]
            || tensor.dtype() != dtype
            || !tensor.device().same_device(input.probabilities.device())
        {
            return Err(Error::Msg(
                "PPO float fields must match batch, dtype and device".into(),
            ));
        }
    }
    if input.actions.dims() != [batch] {
        return Err(Error::Msg("PPO actions must be a batch vector".into()));
    }
    let distribution = CandleCategorical::from_probabilities(input.probabilities)?;
    let log_prob = distribution.log_prob(input.actions)?;
    let entropy = distribution.entropy()?.mean_all()?;
    Ok((log_prob, entropy))
}

impl PpoLossConfig {
    /// Computes source PPO policy/value/entropy terms and the weighted total.
    /// Does not sample rollouts, compute returns, shuffle/repeat minibatches, or
    /// mutate parameters. The returned total retains the actor/critic graph.
    /// # Errors
    /// Rejects invalid dual clipping, batch metadata, categorical probabilities
    /// and action indices; propagates native backend computation failures.
    pub fn loss(&self, input: &PpoLossInput<'_>) -> Result<PpoLoss> {
        if let Some(dual) = self.dual_clip {
            if dual <= 1. || dual.is_nan() {
                return Err(Error::Msg("PPO dual clip must be greater than one".into()));
            }
        }
        let (log_prob, entropy) = categorical(input)?;
        let dtype = input.probabilities.dtype();
        let advantages = if self.normalize_advantage {
            // No epsilon: constant or singleton advantages produce NaN in source.
            input
                .advantages
                .broadcast_sub(&input.advantages.mean_all()?)?
                .broadcast_div(&input.advantages.var(0)?.sqrt()?)?
        } else {
            input.advantages.clone()
        };
        let ratio = log_prob
            .sub(input.old_log_prob)?
            .exp()?
            .to_dtype(DType::F32)?;
        let clipped = clamp(&ratio, 1. - self.eps_clip, 1. + self.eps_clip)?;
        // Source ratio is always F32. Tensor multiplication then promotes against
        // advantages (F64 stays F64; F16/BF16 advantages promote to F32).
        let surrogate_dtype = if dtype == DType::F64 {
            DType::F64
        } else {
            DType::F32
        };
        let adv = advantages.to_dtype(surrogate_dtype)?;
        let surrogate = ratio.to_dtype(surrogate_dtype)?.mul(&adv)?;
        let clipped = clipped.to_dtype(surrogate_dtype)?.mul(&adv)?;
        let policy = extreme(&surrogate, &clipped, false)?;
        let policy = if let Some(dual) = self.dual_clip {
            let bound = extreme(&policy, &adv.affine(dual, 0.)?, true)?;
            adv.lt(0.)?.where_cond(&bound, &policy)?
        } else {
            policy
        };
        let policy = policy.mean_all()?.neg()?;
        let value_error = input.returns.sub(input.values)?.sqr()?;
        let value = if self.value_clip {
            let old_values = input
                .old_values
                .ok_or_else(|| Error::Msg("PPO value clipping requires old values".into()))?;
            if old_values.shape() != input.values.shape()
                || old_values.dtype() != dtype
                || !old_values.device().same_device(input.values.device())
            {
                return Err(Error::Msg(
                    "PPO old values must match values shape, dtype and device".into(),
                ));
            }
            let clipped = old_values.add(&clamp(
                &input.values.sub(old_values)?,
                -self.eps_clip,
                self.eps_clip,
            )?)?;
            extreme(&value_error, &input.returns.sub(&clipped)?.sqr()?, true)?.mean_all()?
        } else {
            value_error.mean_all()?
        };
        let total = policy
            .add(&weighted(&value, self.value_weight)?.to_dtype(surrogate_dtype)?)?
            .sub(&weighted(&entropy, self.entropy_weight)?.to_dtype(surrogate_dtype)?)?;
        Ok(PpoLoss {
            total,
            policy,
            value,
            entropy,
            advantages,
        })
    }
}

#[cfg(test)]
#[path = "../tests/support/rl_candle_ppo.rs"]
mod tests;
