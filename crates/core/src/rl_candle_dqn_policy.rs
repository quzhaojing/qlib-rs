//! Native Qlib DQN: independently owned target model and source update lifecycle.

use crate::rl_candle_checkpoint::{CandlePolicySnapshot, CandlePolicyState};
use crate::rl_candle_dqn::{
    DqnForward, action_argmax, dqn_exploration, dqn_q_values, dqn_target_values, learn_dqn_batch,
};
use crate::rl_candle_heads::DqnModel;
use crate::rl_candle_network::{CandleFeatureExtractor, RecurrentObservation};
use crate::rl_candle_nstep::{
    CandleNStepBatch, CandleNStepConfig, CandleNStepError, CandleNStepReplay, prepare_nstep_returns,
};
use crate::rl_candle_optimizer::CandleAdam;
use crate::rl_candle_policy::{CandlePpoReplay, CandlePpoRollout};
use crate::rl_candle_replay::{CandleReplayBuffer, CandleVectorReplayBuffer};
use crate::rl_policy_weight::{
    PolicyWeightLoadError, PolicyWeightLoader, PolicyWeights, set_policy_weights,
};
use crate::{TrainingPolicyMode, TrainingPolicyState};
use candle_core::{Tensor, Var};
use candle_nn::VarBuilder;
use indexmap::IndexMap;
use ndarray::ArrayView2;
use rand::{Rng, RngCore};
use std::sync::Arc;
use thiserror::Error;

// This is the existing generic Adam scheduler interface, not a second scheduler.
pub use crate::rl_candle_policy::CandlePpoScheduler as CandleDqnScheduler;

#[derive(Clone, Copy, Debug)]
pub struct CandleDqnConfig {
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub gamma: f64,
    pub steps: usize,
    /// Nonpositive frequencies disable the target network, as in source.
    pub target_update_frequency: i64,
    pub reward_normalization: bool,
    pub is_double: bool,
    pub huber: bool,
}
impl CandleDqnConfig {
    #[must_use]
    pub const fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            weight_decay: 0.,
            gamma: 0.99,
            steps: 1,
            target_update_frequency: 0,
            reward_normalization: false,
            is_double: true,
            huber: false,
        }
    }
}

#[derive(Clone)]
pub struct CandleDqnObservation {
    pub observation: RecurrentObservation,
    pub mask: Option<Tensor>,
}
pub struct CandleDqnRollout {
    pub input: CandleDqnObservation,
    pub actions: Tensor,
    pub indices: Vec<usize>,
    pub targets: CandleNStepBatch,
}
impl From<CandlePpoRollout> for CandleDqnRollout {
    fn from(rollout: CandlePpoRollout) -> Self {
        Self {
            input: CandleDqnObservation {
                observation: rollout.observations,
                mask: None,
            },
            actions: rollout.actions,
            indices: rollout.indices,
            targets: CandleNStepBatch {
                returns: None,
                weight: rollout.replay_weights,
            },
        }
    }
}

/// Linked replay seam. Default concrete storage has no action masks/priorities;
/// specialized adapters provide them through the same policy lifecycle.
pub trait CandleDqnReplay: CandleNStepReplay {
    /// # Errors
    /// Returns sampling failures before updating is set.
    fn sample(&mut self, size: u64, rng: &mut dyn RngCore) -> Result<CandleDqnRollout, String>;
    /// # Errors
    /// Returns storage/index failures during target evaluation.
    fn next_observations(&self, indices: &[usize]) -> Result<CandleDqnObservation, String>;
    /// # Errors
    /// Returns priority errors after already completed learning.
    fn update_weights(&mut self, _indices: &[usize], _weights: &Tensor) -> Result<(), String> {
        Ok(())
    }
}

impl CandleDqnReplay for CandleReplayBuffer {
    fn sample(&mut self, size: u64, rng: &mut dyn RngCore) -> Result<CandleDqnRollout, String> {
        CandlePpoReplay::sample(self, size, rng).map(Into::into)
    }
    fn next_observations(&self, indices: &[usize]) -> Result<CandleDqnObservation, String> {
        self.get(indices)
            .map(|row| CandleDqnObservation {
                observation: row.next_observations,
                mask: None,
            })
            .map_err(|e| e.to_string())
    }
}
impl CandleDqnReplay for CandleVectorReplayBuffer {
    fn sample(&mut self, size: u64, rng: &mut dyn RngCore) -> Result<CandleDqnRollout, String> {
        CandlePpoReplay::sample(self, size, rng).map(Into::into)
    }
    fn next_observations(&self, indices: &[usize]) -> Result<CandleDqnObservation, String> {
        self.get(indices)
            .map(|row| CandleDqnObservation {
                observation: row.next_observations,
                mask: None,
            })
            .map_err(|e| e.to_string())
    }
}

#[derive(Debug, Error)]
pub enum CandleDqnError {
    #[error("invalid DQN configuration: {0}")]
    Configuration(&'static str),
    #[error("DQN has no target model")]
    MissingTarget,
    #[error("DQN exploration requires a preceding forward to discover action count")]
    MissingActionCount,
    #[error("DQN iteration exceeds native u64 counter after optimizer update")]
    Iteration,
    #[error("DQN {stage} adapter failed: {message}")]
    Adapter {
        stage: &'static str,
        message: String,
    },
    #[error(transparent)]
    Tensor(#[from] candle_core::Error),
    #[error(transparent)]
    Returns(#[from] CandleNStepError),
    #[error(transparent)]
    Weights(#[from] PolicyWeightLoadError),
}
fn adapter<T>(stage: &'static str, value: Result<T, String>) -> Result<T, CandleDqnError> {
    value.map_err(|message| CandleDqnError::Adapter { stage, message })
}
fn variables(model: &DqnModel) -> candle_core::Result<IndexMap<String, Var>> {
    model
        .parameters()
        .iter()
        .map(|(name, value)| {
            if !value.is_variable() {
                return Err(candle_core::Error::Msg(
                    "DQN state requires live variable handles".into(),
                ));
            }
            Ok((name.clone(), Var::from_tensor(value)?))
        })
        .collect()
}
fn model_forward<State>(
    model: &DqnModel,
    action_count: &mut Option<usize>,
    input: &CandleDqnObservation,
    state: State,
) -> candle_core::Result<DqnForward<State>> {
    let (logits, state) = model.forward(&input.observation, state)?;
    let q = dqn_q_values(&logits, input.mask.as_ref())?;
    if action_count.is_none() {
        *action_count = Some(q.dim(1)?);
    }
    Ok(DqnForward {
        actions: action_argmax(&q)?,
        logits,
        state,
    })
}

pub struct CandleDqn {
    pub model: DqnModel,
    pub optimizer: CandleAdam,
    target: Option<DqnModel>,
    config: CandleDqnConfig,
    frequency: Option<u64>,
    iteration: u64,
    action_count: Option<usize>,
    epsilon: f64,
    mode: TrainingPolicyMode,
    updating: bool,
}
impl CandleDqn {
    /// Construct online model/Adam, validate DQN options, then independently copy
    /// the target when enabled. Target features enter evaluation mode.
    /// # Errors
    /// Returns model, optimizer, configuration or unsupported reconstruction errors.
    pub fn new(
        extractor: Arc<dyn CandleFeatureExtractor>,
        action_dim: usize,
        builder: &VarBuilder<'_>,
        config: CandleDqnConfig,
    ) -> Result<Self, CandleDqnError> {
        let model = DqnModel::new(extractor, action_dim, builder.pp("model"))?;
        let optimizer = CandleAdam::new(
            model.parameters().values().cloned().collect(),
            config.learning_rate,
            config.weight_decay,
        )?;
        if !(0. ..=1.).contains(&config.gamma) {
            return Err(CandleDqnError::Configuration("gamma must be in [0,1]"));
        }
        if config.steps == 0 {
            return Err(CandleDqnError::Configuration("steps must be positive"));
        }
        let frequency = u64::try_from(config.target_update_frequency)
            .ok()
            .filter(|value| *value > 0);
        let target = if frequency.is_some() {
            let target = model.independent_copy()?;
            target.set_mode(TrainingPolicyMode::Evaluation);
            Some(target)
        } else {
            None
        };
        Ok(Self {
            model,
            optimizer,
            target,
            config,
            frequency,
            iteration: 0,
            action_count: None,
            epsilon: 0.,
            mode: TrainingPolicyMode::Train,
            updating: false,
        })
    }

    /// Construct before loading caller-decoded weights, including target state.
    /// Does not read Torch files or restore optimizer/RNG/iteration state.
    /// # Errors
    /// Returns construction or weight errors with the existing one-runtime-error retry.
    pub fn new_with_weights<M>(
        extractor: Arc<dyn CandleFeatureExtractor>,
        action_dim: usize,
        builder: &VarBuilder<'_>,
        config: CandleDqnConfig,
        weights: &mut PolicyWeights<Tensor, M>,
    ) -> Result<Self, CandleDqnError> {
        let mut result = Self::new(extractor, action_dim, builder, config)?;
        set_policy_weights(&mut result, weights)?;
        Ok(result)
    }
    #[must_use]
    pub fn target_model(&self) -> Option<&DqnModel> {
        self.target.as_ref()
    }
    #[must_use]
    pub fn iteration(&self) -> u64 {
        self.iteration
    }
    #[must_use]
    pub fn is_updating(&self) -> bool {
        self.updating
    }
    #[must_use]
    pub fn action_count(&self) -> Option<usize> {
        self.action_count
    }
    #[must_use]
    pub fn mode(&self) -> TrainingPolicyMode {
        self.mode
    }
    pub fn set_epsilon(&mut self, epsilon: f64) {
        self.epsilon = epsilon;
    }
    pub fn set_mode(&mut self, mode: TrainingPolicyMode) {
        self.mode = mode;
        self.model.set_mode(mode);
    }

    /// Register online then target parameters in source state-dict order.
    /// # Errors
    /// Returns invalid live variable registration.
    pub fn policy_state<M>(&self, metadata: M) -> candle_core::Result<CandlePolicyState<M>> {
        let mut registered = IndexMap::new();
        for (prefix, model) in std::iter::once(("model", &self.model))
            .chain(self.target.as_ref().map(|m| ("model_old", m)))
        {
            registered.extend(
                variables(model)?
                    .into_iter()
                    .map(|(name, var)| (format!("{prefix}.{name}"), var)),
            );
        }
        Ok(CandlePolicyState::new(registered, metadata))
    }

    /// Synchronize before scheduled learn calls. Preserves partial copies on error.
    /// # Errors
    /// Returns absent target, registration or source-compatible load errors.
    pub fn sync_weight(&mut self) -> Result<(), CandleDqnError> {
        let target = self.target.as_ref().ok_or(CandleDqnError::MissingTarget)?;
        let state = CandlePolicyState::new(variables(target)?, ());
        state.load_native_weights(&PolicyWeights {
            weights: self
                .model
                .parameters()
                .iter()
                .map(|(name, value)| (name.clone(), Arc::new(value.clone())))
                .collect(),
            metadata: (),
        })?;
        Ok(())
    }

    /// Online inference retains caller state and caches the first action count.
    /// # Errors
    /// Returns observation/model/mask/argmax failures with preceding cache mutation retained.
    pub fn forward<State>(
        &mut self,
        input: &CandleDqnObservation,
        state: State,
    ) -> Result<DqnForward<State>, CandleDqnError> {
        Ok(model_forward(
            &self.model,
            &mut self.action_count,
            input,
            state,
        )?)
    }
    /// # Errors
    /// Returns online or target inference failures in that order.
    pub fn target_q(&mut self, input: &CandleDqnObservation) -> Result<Tensor, CandleDqnError> {
        let online = self.forward(input, ())?;
        let target = self
            .target
            .as_ref()
            .map(|model| {
                model_forward(model, &mut self.action_count, input, ()).map(|out| out.logits)
            })
            .transpose()?;
        Ok(dqn_target_values(
            &online,
            target.as_ref(),
            self.config.is_double,
        )?)
    }
    /// # Errors
    /// Returns n-step metadata, target inference or conversion failures.
    pub fn process<B: CandleDqnReplay + ?Sized>(
        &mut self,
        batch: &mut CandleDqnRollout,
        buffer: &B,
    ) -> Result<(), CandleDqnError> {
        let config = self.config;
        prepare_nstep_returns(
            &mut batch.targets,
            buffer,
            &batch.indices,
            |buffer, indices| {
                let input = buffer.next_observations(indices)?;
                self.target_q(&input).map_err(|e| e.to_string())
            },
            CandleNStepConfig {
                gamma: config.gamma,
                steps: config.steps,
                reward_normalization: config.reward_normalization,
            },
        )?;
        Ok(())
    }

    /// # Errors
    /// Returns sync, forward/loss/optimizer or post-update counter-overflow failures.
    pub fn learn(&mut self, batch: &mut CandleDqnRollout) -> Result<f64, CandleDqnError> {
        if self
            .frequency
            .is_some_and(|frequency| self.iteration % frequency == 0)
        {
            self.sync_weight()?;
        }
        let loss = learn_dqn_batch(
            &mut batch.targets,
            &batch.actions,
            || {
                model_forward(&self.model, &mut self.action_count, &batch.input, ())
                    .map(|out| out.logits)
            },
            &mut self.optimizer,
            self.config.huber,
        )?;
        self.iteration = self
            .iteration
            .checked_add(1)
            .ok_or(CandleDqnError::Iteration)?;
        Ok(loss)
    }

    /// # Errors
    /// Returns absent action count after selection draws, or exploration errors.
    pub fn exploration<R: Rng + ?Sized>(
        &self,
        actions: &mut [i64],
        mask: Option<&ArrayView2<'_, f64>>,
        rng: &mut R,
    ) -> Result<(), CandleDqnError> {
        if self.action_count.is_none()
            && !matches!(
                self.epsilon.abs().partial_cmp(&1e-8),
                Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
            )
        {
            for _ in 0..actions.len() {
                let _ = rng.random::<f64>();
            }
            return Err(CandleDqnError::MissingActionCount);
        }
        Ok(dqn_exploration(
            actions,
            self.epsilon,
            self.action_count.unwrap_or(0),
            mask,
            rng,
        )?)
    }

    /// Source sample -> updating -> process -> learn -> priority -> scheduler.
    /// Only complete success clears updating; None leaves existing state untouched.
    /// # Errors
    /// Returns the first stage failure without rolling back preceding mutations.
    pub fn update(
        &mut self,
        buffer: Option<&mut dyn CandleDqnReplay>,
        scheduler: Option<&mut dyn CandleDqnScheduler>,
        sample_size: u64,
        rng: &mut dyn RngCore,
    ) -> Result<IndexMap<String, f64>, CandleDqnError> {
        let Some(buffer) = buffer else {
            return Ok(IndexMap::new());
        };
        let mut batch = adapter("sample", buffer.sample(sample_size, rng))?;
        self.updating = true;
        self.process(&mut batch, buffer)?;
        let loss = self.learn(&mut batch)?;
        if let Some(weight) = &batch.targets.weight {
            adapter("priority", buffer.update_weights(&batch.indices, weight))?;
        }
        if let Some(scheduler) = scheduler {
            adapter("scheduler", scheduler.step(&mut self.optimizer))?;
        }
        self.updating = false;
        Ok(IndexMap::from([("loss".into(), loss)]))
    }
}
impl TrainingPolicyState<CandlePolicySnapshot<()>> for CandleDqn {
    fn state_dict(&mut self) -> Result<CandlePolicySnapshot<()>, String> {
        self.policy_state(())
            .map_err(|error| error.to_string())?
            .snapshot()
    }

    fn load_state_dict(&mut self, state: &CandlePolicySnapshot<()>) -> Result<(), String> {
        self.policy_state(())
            .map_err(|error| error.to_string())?
            .restore(state)
    }
}

impl<M> PolicyWeightLoader<Tensor, M> for CandleDqn {
    fn load_weights(
        &mut self,
        weights: &mut PolicyWeights<Tensor, M>,
    ) -> Result<(), PolicyWeightLoadError> {
        self.policy_state(())
            .map_err(|e| PolicyWeightLoadError::Runtime(e.to_string()))?
            .load_native_weights(weights)
    }
}

#[cfg(test)]
#[path = "../tests/support/rl_candle_dqn_policy.rs"]
mod tests;
