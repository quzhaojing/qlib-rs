//! Qlib actor, critic and unscaled attention over native differentiable layers.

use crate::rl_candle_categorical::{CategoricalForward, categorical_forward};
use crate::rl_candle_network::{
    CandleFeatureExtractor, Parameters, RecurrentObservation, empty_link, independent_parameters,
    linear_forward,
};
use crate::training_vessel_runner::TrainingPolicyMode;
use candle_core::{CpuStorage, CustomOp2, D, DType, Error, Layout, Result, Shape, Tensor};
use candle_nn::{Linear, VarBuilder};
use indexmap::IndexMap;
use std::sync::Arc;

// Torch saves the rounded F16/BF16 softmax output for its backward pass. Using
// the F32 forward graph instead changes cancellation-sensitive bias gradients.
// All arithmetic remains native Candle operations; this hook selects the saved
// value and its dtype/rounding boundary, not a new tensor kernel.
struct RoundedSoftmax;
impl CustomOp2 for RoundedSoftmax {
    fn name(&self) -> &'static str {
        "source-rounded-softmax"
    }
    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _input_layout: &Layout,
        output: &CpuStorage,
        layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        // Private caller supplies an owned contiguous copy with zero offset.
        Ok((output.clone(), layout.shape().clone()))
    }
    fn bwd(
        &self,
        input: &Tensor,
        _saved: &Tensor,
        output: &Tensor,
        gradient: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let output = output.to_dtype(DType::F32)?;
        let gradient = gradient.to_dtype(DType::F32)?;
        // The source CPU last-dimension kernel stores its reduction in scalar_t,
        // rounding this sum too before converting it back to accumulation dtype.
        let sum = (&gradient * &output)?
            .sum_keepdim(D::Minus1)?
            .to_dtype(input.dtype())?
            .to_dtype(DType::F32)?;
        let derivative = (gradient.broadcast_sub(&sum)? * output)?.to_dtype(input.dtype())?;
        Ok((Some(derivative), None))
    }
}

fn softmax_last(tensor: &Tensor) -> Result<Tensor> {
    if tensor.elem_count() == 0 {
        Ok(tensor.clone())
    } else if matches!(tensor.dtype(), DType::F16 | DType::BF16) {
        let rounded = candle_nn::ops::softmax(&tensor.to_dtype(DType::F32)?, D::Minus1)?
            .to_dtype(tensor.dtype())?;
        tensor.apply_op2(&rounded.detach().copy()?, RoundedSoftmax)
    } else {
        candle_nn::ops::softmax(tensor, D::Minus1)
    }
}

fn all_parameters(
    extractor: &dyn CandleFeatureExtractor,
    head: IndexMap<String, Tensor>,
) -> IndexMap<String, Tensor> {
    extractor
        .parameters()
        .iter()
        .map(|(name, tensor)| (format!("extractor.{name}"), tensor.clone()))
        .chain(head)
        .collect()
}

pub struct PpoActor {
    extractor: Arc<dyn CandleFeatureExtractor>,
    layer: Linear,
    parameters: IndexMap<String, Tensor>,
}
impl PpoActor {
    /// Build an independent target model, preserving parameter aliases within it.
    /// Feature plugins explicitly reconstruct around copied handles; no Python
    /// object serialization or graph codec is involved.
    /// # Errors
    /// Returns unsupported feature reconstruction or inconsistent rebuilt parameters.
    pub fn independent_copy(&self) -> Result<Self> {
        let copies = independent_parameters(&self.parameters)?;
        let extractor = self.extractor.rebuild(
            copies
                .iter()
                .filter_map(|(name, value)| {
                    name.strip_prefix("extractor.")
                        .map(|name| (name.to_owned(), value.clone()))
                })
                .collect(),
        )?;
        let builder = VarBuilder::from_tensors(
            copies.clone().into_iter().collect(),
            self.layer.weight().dtype(),
            self.layer.weight().device(),
        );
        let target = Self::new(extractor, self.layer.weight().dim(0)?, builder)?;
        if target.parameters.len() != copies.len()
            || target.parameters.iter().any(|(name, tensor)| {
                copies
                    .get(name)
                    .is_none_or(|expected| tensor.id() != expected.id())
            })
        {
            return Err(Error::Msg(
                "rebuilt feature extractor did not retain copied parameters".into(),
            ));
        }
        Ok(target)
    }

    pub fn set_mode(&self, mode: TrainingPolicyMode) {
        self.extractor.set_mode(mode);
    }

    /// # Errors
    /// Returns parameter-source, shape or backend initialization failures.
    pub fn new(
        extractor: Arc<dyn CandleFeatureExtractor>,
        action_dim: usize,
        builder: VarBuilder<'_>,
    ) -> Result<Self> {
        let mut parameters = Parameters::new(builder);
        let layer = parameters.linear("layer_out.0", extractor.output_dim(), action_dim)?;
        let parameters = all_parameters(extractor.as_ref(), parameters.tensors);
        Ok(Self {
            extractor,
            layer,
            parameters,
        })
    }
    #[must_use]
    pub fn parameters(&self) -> &IndexMap<String, Tensor> {
        &self.parameters
    }
    /// Keeps arbitrary caller-owned state unchanged, as Qlib's actor does.
    /// # Errors
    /// Returns extractor, shape, dtype or backend computation failures.
    pub fn forward<State>(
        &self,
        obs: &RecurrentObservation,
        state: State,
    ) -> Result<(Tensor, State)> {
        let features = self.extractor.forward(&obs.detached())?;
        Ok((
            softmax_last(&linear_forward(&self.layer, &features)?)?,
            state,
        ))
    }

    /// Native discrete policy forward with the source mode/sampling contract.
    /// Keeps caller state unchanged and uses the same differentiable actor head.
    /// # Errors
    /// Propagates extractor, tensor, distribution and sampler errors.
    pub fn policy_forward<State, R: rand::Rng + ?Sized>(
        &self,
        obs: &RecurrentObservation,
        state: State,
        mode: TrainingPolicyMode,
        deterministic_eval: bool,
        rng: &mut R,
    ) -> Result<CategoricalForward<State>> {
        let (logits, state) = self.forward(obs, state)?;
        categorical_forward(logits, state, mode, deterministic_eval, rng)
    }
}

/// Qlib DQN intentionally reuses its softmax actor, not an unconstrained Q head.
pub type DqnModel = PpoActor;

pub struct PpoCritic {
    extractor: Arc<dyn CandleFeatureExtractor>,
    layer: Linear,
    parameters: IndexMap<String, Tensor>,
}
impl PpoCritic {
    /// # Errors
    /// Returns parameter-source, shape or backend initialization failures.
    pub fn new(
        extractor: Arc<dyn CandleFeatureExtractor>,
        builder: VarBuilder<'_>,
    ) -> Result<Self> {
        let mut parameters = Parameters::new(builder);
        let layer = parameters.linear("value_out", extractor.output_dim(), 1)?;
        let parameters = all_parameters(extractor.as_ref(), parameters.tensors);
        Ok(Self {
            extractor,
            layer,
            parameters,
        })
    }
    #[must_use]
    pub fn parameters(&self) -> &IndexMap<String, Tensor> {
        &self.parameters
    }
    /// # Errors
    /// Returns extractor, shape, dtype or backend computation failures.
    pub fn forward(&self, obs: &RecurrentObservation) -> Result<Tensor> {
        linear_forward(&self.layer, &self.extractor.forward(&obs.detached())?)?.squeeze(D::Minus1)
    }
}

fn broadcast_extent(left: usize, right: usize) -> Result<usize> {
    if left == right || right == 1 {
        Ok(left)
    } else if left == 1 {
        Ok(right)
    } else {
        Err(Error::Msg(
            "incompatible attention einsum dimensions".into(),
        ))
    }
}
fn contract(left: &Tensor, right: &Tensor) -> Result<Tensor> {
    let (lb, lm, lk) = left.dims3()?;
    let (rb, rk, rn) = right.dims3()?;
    let batch = broadcast_extent(lb, rb)?;
    let inner = broadcast_extent(lk, rk)?;
    if left.dtype() != right.dtype() {
        return Err(Error::Msg("attention tensor dtype mismatch".into()));
    }
    if batch == 0 || lm == 0 || inner == 0 || rn == 0 {
        return empty_link(left, right)?.broadcast_as((batch, lm, rn));
    }
    let left = left.broadcast_as((batch, lm, inner))?.contiguous()?;
    let right = right.broadcast_as((batch, inner, rn))?.contiguous()?;
    if matches!(left.dtype(), DType::F16 | DType::BF16) {
        left.to_dtype(DType::F32)?
            .matmul(&right.to_dtype(DType::F32)?)?
            .to_dtype(left.dtype())
    } else {
        left.matmul(&right)
    }
}

pub struct Attention {
    query: Linear,
    key: Linear,
    value: Linear,
    parameters: IndexMap<String, Tensor>,
}
impl Attention {
    /// # Errors
    /// Returns parameter-source, shape or backend initialization failures.
    pub fn new(input_dim: usize, output_dim: usize, builder: VarBuilder<'_>) -> Result<Self> {
        let mut parameters = Parameters::new(builder);
        let query = parameters.linear("q_net", input_dim, output_dim)?;
        let key = parameters.linear("k_net", input_dim, output_dim)?;
        let value = parameters.linear("v_net", input_dim, output_dim)?;
        Ok(Self {
            query,
            key,
            value,
            parameters: parameters.tensors,
        })
    }
    #[must_use]
    pub fn parameters(&self) -> &IndexMap<String, Tensor> {
        &self.parameters
    }
    /// Qlib uses unscaled scores, with einsum singleton broadcasting, and no mask.
    /// # Errors
    /// Returns rank, shape, dtype or backend computation failures.
    pub fn forward(&self, query: &Tensor, key: &Tensor, value: &Tensor) -> Result<Tensor> {
        let q = linear_forward(&self.query, query)?;
        let k = linear_forward(&self.key, key)?;
        let v = linear_forward(&self.value, value)?;
        let scores = contract(&q, &k.transpose(1, 2)?)?;
        contract(&softmax_last(&scores)?, &v)
    }
}

#[cfg(test)]
#[path = "../tests/support/rl_candle_heads.rs"]
mod tests;
