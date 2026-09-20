//! Discrete probability distributions shared by native policy forward and loss.
//! Candle provides arithmetic/autograd; rand provides weighted sampling.

use crate::training_vessel_runner::TrainingPolicyMode;
use candle_core::{CpuStorage, CustomOp2, D, DType, Error, Layout, Result, Shape, Tensor};
use num_traits::ToPrimitive;
use rand::{
    Rng,
    distr::{Distribution, weighted::WeightedIndex},
};

// Torch scalar-clamp gradients are one at either bound and zero at NaN.
pub(crate) fn clamp(input: &Tensor, min: f64, max: f64) -> Result<Tensor> {
    let constant =
        |value| Tensor::full(value, input.shape(), input.device())?.to_dtype(input.dtype());
    if min.is_nan() || max.is_nan() {
        return input.eq(f64::NAN)?.where_cond(input, &constant(f64::NAN)?);
    }
    let lower = input.lt(min)?.where_cond(&constant(min)?, input)?;
    let bounded = lower.gt(max)?.where_cond(&constant(max)?, &lower)?;
    input
        .ge(min)?
        .mul(&input.le(max)?)?
        .where_cond(input, &bounded.detach())
}

pub(crate) fn epsilon(dtype: DType) -> Result<f64> {
    match dtype {
        DType::F16 => Ok(0.000_976_562_5),
        DType::BF16 => Ok(0.007_812_5),
        DType::F32 => Ok(f64::from(f32::EPSILON)),
        DType::F64 => Ok(f64::EPSILON),
        _ => Err(Error::Msg(
            "categorical probabilities must be floating point".into(),
        )),
    }
}

// Torch's denominator derivative is -grad * ((raw / sum) / sum), whereas
// Candle's generic division computes -(grad * raw) / sum.square(). The order
// changes rounded low-precision gradients. This hook composes Candle operations;
// its forward only returns the already computed, owned normalized tensor.
struct SourceNormalize;
impl CustomOp2 for SourceNormalize {
    fn name(&self) -> &'static str {
        "source-categorical-normalization"
    }
    fn cpu_fwd(
        &self,
        _raw: &CpuStorage,
        _raw_layout: &Layout,
        probabilities: &CpuStorage,
        layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        Ok((probabilities.clone(), layout.shape().clone()))
    }
    fn bwd(
        &self,
        raw: &Tensor,
        _saved: &Tensor,
        output: &Tensor,
        gradient: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let sum = raw.sum_keepdim(D::Minus1)?;
        let direct = gradient.broadcast_div(&sum)?;
        let denominator = gradient.neg()?.mul(&output.broadcast_div(&sum)?)?;
        let denominator = if matches!(raw.dtype(), DType::F16 | DType::BF16) {
            denominator
                .to_dtype(DType::F32)?
                .sum_keepdim(D::Minus1)?
                .to_dtype(raw.dtype())?
        } else {
            denominator.sum_keepdim(D::Minus1)?
        };
        Ok((Some(direct.broadcast_add(&denominator)?), None))
    }
}

#[derive(Debug)]
pub struct CandleCategorical {
    probabilities: Tensor,
    logits: Tensor,
    events: usize,
}

impl CandleCategorical {
    /// Normalize along the last dimension BEFORE checking the simplex, matching
    /// Torch's probabilities constructor. All-negative raw weights can therefore
    /// normalize into a valid distribution. Leading dimensions form the batch.
    /// # Errors
    /// Rejects scalar/non-floating inputs or invalid normalized probabilities;
    /// propagates tensor backend failures.
    pub fn from_probabilities(raw: &Tensor) -> Result<Self> {
        let events = raw.dim(D::Minus1)?;
        let eps = epsilon(raw.dtype())?;
        let probabilities = raw.broadcast_div(&raw.sum_keepdim(D::Minus1)?)?;
        let probabilities = raw.apply_op2(&probabilities.detach().copy()?, SourceNormalize)?;
        let nonnegative = probabilities.ge(0.)?.flatten_all()?.to_vec1::<u8>()?;
        let normalized = probabilities
            .sum(D::Minus1)?
            .affine(1., -1.)?
            .abs()?
            .lt(1e-6)?
            .flatten_all()?
            .to_vec1::<u8>()?;
        // all(empty) is true in Torch; empty leading batches have a valid
        // distribution even though some sampling/argmax operations reject them.
        if nonnegative.contains(&0) || normalized.contains(&0) {
            return Err(Error::Msg(
                "categorical probabilities are not a simplex".into(),
            ));
        }
        let logits = clamp(&probabilities, eps, 1. - eps)?.log()?;
        Ok(Self {
            probabilities,
            logits,
            events,
        })
    }

    #[must_use]
    pub fn probabilities(&self) -> &Tensor {
        &self.probabilities
    }

    #[must_use]
    pub fn logits(&self) -> &Tensor {
        &self.logits
    }

    /// Supports scalar, batched and broadcast sample indices, including integral
    /// floating values (PPO `process_fn` casts actions to its critic dtype).
    /// # Errors
    /// Rejects fractional/nonfinite/out-of-support actions, incompatible shapes,
    /// or tensor conversion/backend errors.
    pub fn log_prob(&self, actions: &Tensor) -> Result<Tensor> {
        let values = actions
            .to_dtype(DType::F64)?
            .flatten_all()?
            .to_vec1::<f64>()?;
        if values.iter().any(|value| {
            value.fract().abs() > 0. || value.to_usize().is_none_or(|index| index >= self.events)
        }) {
            return Err(Error::Msg(
                "categorical action is outside integer support".into(),
            ));
        }
        let indices = actions
            .to_dtype(DType::I64)?
            .to_device(self.logits.device())?
            .unsqueeze(D::Minus1)?;
        let shape = indices
            .shape()
            .broadcast_shape_binary_op(self.logits.shape(), "categorical log_prob")?;
        let indices = indices
            .broadcast_as(&shape)?
            .narrow(D::Minus1, 0, 1)?
            .contiguous()?;
        self.logits
            .broadcast_as(&shape)?
            .contiguous()?
            .gather(&indices, D::Minus1)?
            .squeeze(D::Minus1)
    }

    /// Per-distribution entropy (no batch reduction). Logits derived from valid
    /// probabilities are already finite after dtype-epsilon clamping.
    /// # Errors
    /// Propagates native tensor failures.
    pub fn entropy(&self) -> Result<Tensor> {
        self.probabilities.mul(&self.logits)?.sum(D::Minus1)?.neg()
    }

    /// Draw one I64 action per distribution with the caller-owned RNG. Sampling
    /// uses rand's weighted index, not Torch's generator or a custom RNG kernel;
    /// seeded replay is native-deterministic, not bitwise Torch RNG replay.
    /// # Errors
    /// Returns invalid sampling shape/weights, index conversion or backend errors.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<Tensor> {
        if self.events == 0 {
            return Err(Error::Msg(
                "cannot sample an empty categorical event space".into(),
            ));
        }
        let probabilities = self
            .probabilities
            .to_dtype(DType::F64)?
            .flatten_all()?
            .to_vec1::<f64>()?;
        let actions = probabilities
            .chunks(self.events)
            .map(|row| {
                let distribution = WeightedIndex::new(row).map_err(Error::wrap)?;
                i64::try_from(distribution.sample(rng)).map_err(Error::wrap)
            })
            .collect::<Result<Vec<_>>>()?;
        let shape = &self.probabilities.dims()[..self.probabilities.rank() - 1];
        Tensor::from_vec(actions, shape, self.probabilities.device())
    }
}

pub struct CategoricalForward<State> {
    /// Source forward's `logits` field is the RAW actor output, not log-probability.
    pub logits: Tensor,
    pub actions: Tensor,
    pub state: State,
    pub distribution: CandleCategorical,
}

/// Assemble Tianshou's discrete forward result after actor evaluation. Training
/// always samples, including forward calls made only to read log probabilities.
/// Deterministic evaluation uses raw actor argmax, not normalized distribution mode.
/// # Errors
/// Propagates distribution, argmax, sampling and tensor failures.
pub fn categorical_forward<State, R: Rng + ?Sized>(
    logits: Tensor,
    state: State,
    mode: TrainingPolicyMode,
    deterministic_eval: bool,
    rng: &mut R,
) -> Result<CategoricalForward<State>> {
    let distribution = CandleCategorical::from_probabilities(&logits)?;
    let actions = if deterministic_eval && mode == TrainingPolicyMode::Evaluation {
        logits.argmax(D::Minus1)?.to_dtype(DType::I64)?
    } else {
        distribution.sample(rng)?
    };
    Ok(CategoricalForward {
        logits,
        actions,
        state,
        distribution,
    })
}

#[cfg(test)]
#[path = "../tests/support/rl_candle_categorical.rs"]
mod tests;
