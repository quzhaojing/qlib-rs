//! Qlib's dense Adam update boundary, backed by third-party optimizers.
//!
//! One lazy optimizer per unique variable preserves Torch's per-parameter clock:
//! an absent gradient must not advance it. No Adam arithmetic is implemented here.

use candle_core::{DType, Device, Error, Result, Tensor, TensorId, Var, backprop::GradStore};
use candle_nn::Optimizer;
use candle_optimisers::{
    Decay,
    adam::{Adam, ParamsAdam},
};
use indexmap::{IndexMap, IndexSet, map::Entry};

fn nonnegative(value: f64, name: &str) -> Result<()> {
    if value < 0. || value.is_nan() {
        Err(Error::Msg(format!("invalid {name}: {value}")))
    } else {
        Ok(())
    }
}

// Torch computes low-precision vector norms with float accumulation, then rounds
// each norm to the input dtype before the second (across-parameter) reduction.
fn l2_norm(tensor: &Tensor) -> Result<Tensor> {
    let accumulation = match tensor.dtype() {
        DType::F16 | DType::BF16 => DType::F32,
        dtype => dtype,
    };
    tensor
        .to_dtype(accumulation)?
        .sqr()?
        .sum_all()?
        .sqrt()?
        .to_dtype(tensor.dtype())
}

/// Dense native equivalent of the Adam configuration exposed by Qlib PPO/DQN.
/// Defaults for betas, epsilon and `AMSGrad` come from the verified Adam runtime.
pub struct CandleAdam {
    parameters: Vec<Var>,
    states: IndexMap<TensorId, Adam>,
    config: ParamsAdam,
}
impl CandleAdam {
    /// Deduplicates by live variable identity in first-seen order, like `chain_dedup`.
    /// Constant tensors are rejected rather than copied into unrelated trainable variables.
    /// # Errors
    /// Returns invalid hyperparameters, empty parameters, non-variable/non-float tensors,
    /// or backend failures while acquiring live variable handles.
    pub fn new(parameters: Vec<Tensor>, learning_rate: f64, weight_decay: f64) -> Result<Self> {
        nonnegative(learning_rate, "learning rate")?;
        nonnegative(weight_decay, "weight decay")?;
        if parameters.is_empty() {
            return Err(Error::Msg("optimizer requires parameters".into()));
        }
        let mut seen = IndexSet::new();
        let mut variables = Vec::new();
        for parameter in parameters {
            if !parameter.is_variable() || !parameter.dtype().is_float() {
                return Err(Error::Msg(
                    "Adam requires live floating-point variables".into(),
                ));
            }
            if seen.insert(parameter.id()) {
                variables.push(Var::from_tensor(&parameter)?);
            }
        }
        Ok(Self {
            parameters: variables,
            states: IndexMap::new(),
            config: ParamsAdam {
                lr: learning_rate,
                weight_decay: (weight_decay != 0.).then_some(Decay::WeightDecay(weight_decay)),
                ..ParamsAdam::default()
            },
        })
    }

    #[must_use]
    pub fn parameters(&self) -> &[Var] {
        &self.parameters
    }
    /// Number of variables which have received a gradient (including an empty/zero gradient).
    #[must_use]
    pub fn initialized_parameter_count(&self) -> usize {
        self.states.len()
    }
    #[must_use]
    pub fn learning_rate(&self) -> f64 {
        self.config.lr
    }

    /// Updates existing state and the defaults for parameters not yet initialized.
    /// # Errors
    /// Rejects negative or NaN rates without changing the optimizer.
    pub fn set_learning_rate(&mut self, learning_rate: f64) -> Result<()> {
        nonnegative(learning_rate, "learning rate")?;
        self.config.lr = learning_rate;
        for optimizer in self.states.values_mut() {
            optimizer.set_learning_rate(learning_rate);
        }
        Ok(())
    }

    /// Clips present gradients by their combined L2 norm, as used by Qlib PPO.
    /// Returns the pre-clipping norm; absent gradients remain absent and shared
    /// parameters are counted once. Does not initialize or advance Adam state.
    /// Like Torch's default, negative limits and nonfinite norms are not rejected.
    /// A zero limit explicitly clips to zero; PPO's separate zero-disables-clipping
    /// configuration rule belongs to the training loop, not this operation.
    /// # Errors
    /// Returns malformed gradient metadata or native tensor-operation failures.
    pub fn clip_grad_norm(&self, gradients: &mut GradStore, max_norm: f64) -> Result<Tensor> {
        self.validate_gradients(gradients)?;
        let present: Vec<_> = self
            .parameters
            .iter()
            .filter_map(|parameter| {
                gradients
                    .get(parameter)
                    .map(|grad| (parameter, grad.detach()))
            })
            .collect();
        let Some((_, first)) = present.first() else {
            return Tensor::new(0_f32, &Device::Cpu);
        };
        let dtype = present.iter().fold(first.dtype(), |dtype, (_, grad)| {
            match (dtype, grad.dtype()) {
                (DType::F64, _) | (_, DType::F64) => DType::F64,
                (left, right) if left == right => left,
                _ => DType::F32,
            }
        });
        let norms = present
            .iter()
            .map(|(_, grad)| l2_norm(grad)?.to_device(first.device())?.to_dtype(dtype))
            .collect::<Result<Vec<_>>>()?;
        let norm = l2_norm(&Tensor::stack(&norms, 0)?)?;
        let coefficient = norm.affine(1., 1e-6)?.recip()?.affine(max_norm, 0.)?;
        // An upper-only clamp must retain NaN (minimum may select its finite peer).
        let coefficient = coefficient
            .gt(1.)?
            .where_cond(&coefficient.ones_like()?, &coefficient)?;
        for (parameter, grad) in present {
            // Torch's scalar multiply keeps the coefficient in the gradient's
            // opmath dtype. Rounding a mixed F32 coefficient to BF16 first loses
            // information and can change the final rounded gradient by one ULP.
            let accumulation = match grad.dtype() {
                DType::F16 | DType::BF16 => DType::F32,
                dtype => dtype,
            };
            let coefficient = coefficient
                .to_device(grad.device())?
                .to_dtype(accumulation)?;
            gradients.insert(
                parameter,
                grad.to_dtype(accumulation)?
                    .broadcast_mul(&coefficient)?
                    .to_dtype(grad.dtype())?,
            );
        }
        Ok(norm)
    }

    fn validate_gradients(&self, gradients: &GradStore) -> Result<()> {
        for parameter in &self.parameters {
            if let Some(gradient) = gradients.get(parameter) {
                if gradient.shape() != parameter.shape()
                    || gradient.dtype() != parameter.dtype()
                    || !gradient.device().same_device(parameter.device())
                {
                    return Err(Error::Msg(
                        "gradient shape, dtype or device does not match parameter".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Performs one update for each present gradient, leaving absent ones untouched.
    /// Incoming gradients are validated before any update (Torch validates grad assignment).
    /// Once backend updates start, an error may retain prior successful updates/state.
    /// # Errors
    /// Returns incompatible dense gradient shape/dtype/device or backend update failures.
    pub fn step(&mut self, gradients: &GradStore) -> Result<()> {
        self.validate_gradients(gradients)?;
        for parameter in &self.parameters {
            if gradients.get(parameter).is_none() {
                continue;
            }
            let optimizer = match self.states.entry(parameter.id()) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    entry.insert(Adam::new(vec![parameter.clone()], self.config.clone())?)
                }
            };
            // Empty variables cannot change shape and have no values to update.
            // Candle's empty subtraction aliases theta, which its Var::set rejects.
            // Retain lazy state creation, but avoid that invalid no-op assignment.
            if parameter.elem_count() != 0 {
                optimizer.step(gradients)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../tests/support/rl_candle_optimizer.rs"]
mod tests;
