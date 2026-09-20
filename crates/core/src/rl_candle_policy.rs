//! Native discrete PPO assembly and rollout training. Tensor, autodiff, sampling
//! and optimizer work is delegated to the existing Candle/rand adapters.

use crate::TrainingPolicyState;
use crate::rl_candle_categorical::CategoricalForward;
use crate::rl_candle_checkpoint::{CandlePolicySnapshot, CandlePolicyState};
use crate::rl_candle_heads::{PpoActor, PpoCritic};
use crate::rl_candle_network::{CandleFeatureExtractor, RecurrentObservation};
use crate::rl_candle_optimizer::CandleAdam;
use crate::rl_candle_ppo::{PpoLossConfig, PpoLossInput};
use crate::rl_candle_returns::{
    CandleReturnError, CandleReturnInput, CandleReturns, ReturnStatistics, prepare_returns,
};
use crate::rl_policy_batch::{InvalidBatchSize, minibatch_indices};
use crate::rl_policy_weight::{
    PolicyWeightLoadError, PolicyWeightLoader, PolicyWeights, set_policy_weights,
};
use crate::training_vessel_runner::TrainingPolicyMode;
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::VarBuilder;
use indexmap::IndexMap;
use ndarray::Array1;
use rand::Rng;
use std::sync::Arc;
use thiserror::Error;

#[derive(Clone, Copy, Debug)]
pub struct CandlePpoConfig {
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub gamma: f64,
    pub gae_lambda: f64,
    pub max_grad_norm: Option<f64>,
    pub reward_normalization: bool,
    pub max_batch_size: usize,
    pub deterministic_eval: bool,
    pub recompute_advantage: bool,
    pub loss: PpoLossConfig,
}

impl CandlePpoConfig {
    /// Qlib wrapper defaults; unlike generic Tianshou, learning rate is required.
    #[must_use]
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            weight_decay: 0.,
            gamma: 1.,
            gae_lambda: 1.,
            max_grad_norm: Some(100.),
            reward_normalization: true,
            max_batch_size: 256,
            deterministic_eval: true,
            recompute_advantage: false,
            loss: PpoLossConfig::default(),
        }
    }
}

/// Ordered sampled rollout and replay metadata. A collector/replay adapter owns
/// sampling and supplies bootstrap masks/unfinished indices; this is not itself
/// a replay buffer. Observations retain all full-history fields.
pub struct CandlePpoRollout {
    pub observations: RecurrentObservation,
    pub next_observations: RecurrentObservation,
    pub actions: Tensor,
    pub rewards: Array1<f64>,
    pub terminated: Array1<bool>,
    pub truncated: Array1<bool>,
    pub bootstrap_valid: Array1<bool>,
    pub indices: Vec<usize>,
    pub unfinished_indices: Vec<usize>,
    /// Optional replay priorities are passed through unchanged by PPO.
    pub replay_weights: Option<Tensor>,
}

/// In-process Candle replay adapter, not a stable external plugin ABI. Concrete
/// storage owns sampling and metadata extraction; size zero means all entries.
pub trait CandlePpoReplay {
    /// # Errors
    /// Returns replay sampling/metadata errors before the policy enters updating mode.
    fn sample(
        &mut self,
        size: u64,
        rng: &mut dyn rand::RngCore,
    ) -> Result<CandlePpoRollout, String>;

    /// Ordinary buffers have no priority update operation. Prioritized adapters
    /// override this method; it is invoked only when the sampled batch has weights.
    /// # Errors
    /// Returns priority update errors after already completed policy updates.
    fn update_weights(&mut self, _indices: &[usize], _weights: &Tensor) -> Result<(), String> {
        Ok(())
    }
}

pub trait CandlePpoScheduler {
    /// # Errors
    /// Returns scheduling errors, retaining any learning-rate changes already made.
    fn step(&mut self, optimizer: &mut CandleAdam) -> Result<(), String>;
}

#[derive(Clone, Copy, Debug)]
pub struct CandlePpoUpdate {
    pub sample_size: u64,
    pub batch_size: usize,
    pub repeat: usize,
}

/// Detached process results plus the original rollout for advantage recomputation.
pub struct PreparedPpoRollout<'a> {
    pub rollout: &'a CandlePpoRollout,
    pub actions: Tensor,
    pub targets: CandleReturns,
    pub old_log_prob: Tensor,
}

#[derive(Debug, Error)]
pub enum CandlePpoError {
    #[error("invalid PPO configuration: {0}")]
    Configuration(&'static str),
    #[error("batch position exceeds the native I64 index range")]
    IndexRange,
    #[error("PPO {stage} adapter failed: {message}")]
    Adapter {
        stage: &'static str,
        message: String,
    },
    #[error(transparent)]
    Batch(#[from] InvalidBatchSize),
    #[error(transparent)]
    Returns(#[from] CandleReturnError),
    #[error(transparent)]
    Tensor(#[from] candle_core::Error),
    #[error(transparent)]
    Weights(#[from] PolicyWeightLoadError),
}

fn positions_tensor(positions: &[usize], device: &Device) -> Result<Tensor, CandlePpoError> {
    let values = positions
        .iter()
        .map(|&index| i64::try_from(index).map_err(|_| CandlePpoError::IndexRange))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Tensor::new(values.as_slice(), device)?)
}

pub struct CandlePpo {
    pub actor: PpoActor,
    pub critic: PpoCritic,
    pub optimizer: CandleAdam,
    pub return_statistics: ReturnStatistics,
    // Tianshou's registered ActorCritic container keeps the original modules
    // even if the public actor/critic fields are later replaced.
    actor_critic_parameters: IndexMap<String, Tensor>,
    config: CandlePpoConfig,
    mode: TrainingPolicyMode,
    updating: bool,
}

impl CandlePpo {
    /// Register source state-dict order, including shared extractor and the
    /// `_actor_critic` aliases. Handles refer to live model/optimizer variables.
    /// Does not include optimizer moments, RNG, or return-normalization statistics.
    /// # Errors
    /// Returns an invalid native variable registration error.
    pub fn policy_state<M>(&self, metadata: M) -> candle_core::Result<CandlePolicyState<M>> {
        let mut variables = IndexMap::new();
        for (component, parameters) in [
            ("actor", self.actor.parameters()),
            ("critic", self.critic.parameters()),
            ("_actor_critic", &self.actor_critic_parameters),
        ] {
            for (name, tensor) in parameters {
                // Var::from_tensor copies constants into unrelated variables.
                // Checkpoint restoration must only acquire existing live handles.
                if !tensor.is_variable() {
                    return Err(candle_core::Error::Msg(format!(
                        "PPO state requires live variable handles: {component}.{name}"
                    )));
                }
                variables.insert(format!("{component}.{name}"), Var::from_tensor(tensor)?);
            }
        }
        Ok(CandlePolicyState::new(variables, metadata))
    }

    /// Construct model and optimizer before applying decoded constructor weights.
    /// Reuses Qlib's one-runtime-error retry, mutating the caller's legacy key map.
    /// A caller-owned file/Trainer adapter supplies decoded weights; this is not
    /// a Torch file reader or optimizer-state restoration operation.
    /// # Errors
    /// Returns construction or loading errors, retaining preceding shared-variable
    /// copies and input-map changes even when construction ultimately fails.
    pub fn new_with_weights<M>(
        extractor: Arc<dyn CandleFeatureExtractor>,
        action_dim: usize,
        builder: &VarBuilder<'_>,
        config: CandlePpoConfig,
        weights: &mut PolicyWeights<Tensor, M>,
    ) -> Result<Self, CandlePpoError> {
        let mut policy = Self::new(extractor, action_dim, builder, config)?;
        set_policy_weights(&mut policy, weights)?;
        Ok(policy)
    }

    /// Constructs shared-extractor actor/critic and deduplicated native Adam.
    /// Builder namespaces are `actor` and `critic`; the extractor already owns
    /// its variables. File decoding remains caller-owned; `new_with_weights`
    /// additionally applies already decoded constructor weights.
    /// # Errors
    /// Rejects invalid gamma/lambda/dual/value-normalization combinations and
    /// propagates model/optimizer initialization failures.
    pub fn new(
        extractor: Arc<dyn CandleFeatureExtractor>,
        action_dim: usize,
        builder: &VarBuilder<'_>,
        config: CandlePpoConfig,
    ) -> Result<Self, CandlePpoError> {
        if !(0. ..=1.).contains(&config.gamma) || !(0. ..=1.).contains(&config.gae_lambda) {
            return Err(CandlePpoError::Configuration(
                "gamma/lambda must be in [0,1]",
            ));
        }
        if config
            .loss
            .dual_clip
            .is_some_and(|dual| dual.partial_cmp(&1.) != Some(std::cmp::Ordering::Greater))
        {
            return Err(CandlePpoError::Configuration("dual clip must exceed one"));
        }
        if config.loss.value_clip && !config.reward_normalization {
            return Err(CandlePpoError::Configuration(
                "value clipping requires reward normalization",
            ));
        }
        let actor = PpoActor::new(extractor.clone(), action_dim, builder.pp("actor"))?;
        let critic = PpoCritic::new(extractor, builder.pp("critic"))?;
        let optimizer = CandleAdam::new(
            actor
                .parameters()
                .values()
                .chain(critic.parameters().values())
                .cloned()
                .collect(),
            config.learning_rate,
            config.weight_decay,
        )?;
        let actor_critic_parameters = [
            ("actor", actor.parameters()),
            ("critic", critic.parameters()),
        ]
        .into_iter()
        .flat_map(|(component, parameters)| {
            parameters
                .iter()
                .map(move |(name, tensor)| (format!("{component}.{name}"), tensor.clone()))
        })
        .collect();
        Ok(Self {
            actor,
            critic,
            optimizer,
            actor_critic_parameters,
            config,
            return_statistics: ReturnStatistics::default(),
            mode: TrainingPolicyMode::Train,
            updating: false,
        })
    }

    pub fn set_mode(&mut self, mode: TrainingPolicyMode) {
        self.mode = mode;
    }

    #[must_use]
    pub fn is_updating(&self) -> bool {
        self.updating
    }

    /// Source `BasePolicy.update` sequencing. None returns an empty mapping without
    /// resetting prior state. Sampling happens before setting updating; successful
    /// process/learn/priority/scheduler completion clears it. A later failure leaves
    /// updating true, exactly as source (there is deliberately no finally reset).
    /// Does not implicitly change train/evaluation mode.
    /// # Errors
    /// Propagates replay, process, learning, priority and scheduler errors without
    /// rolling back prior weight, optimizer, replay or normalization mutations.
    pub fn update(
        &mut self,
        buffer: Option<&mut dyn CandlePpoReplay>,
        scheduler: Option<&mut dyn CandlePpoScheduler>,
        options: CandlePpoUpdate,
        rng: &mut dyn rand::RngCore,
    ) -> Result<IndexMap<String, Vec<f64>>, CandlePpoError> {
        self.update_with_options(buffer, scheduler, options.sample_size, rng, |_, _| {
            Ok((options.batch_size, options.repeat))
        })
    }

    // The vessel's dynamic keyword adapter resolves learn arguments only AFTER
    // process. Typed callers reuse the same lifecycle with infallible resolution.
    pub(crate) fn update_with_options(
        &mut self,
        buffer: Option<&mut dyn CandlePpoReplay>,
        scheduler: Option<&mut dyn CandlePpoScheduler>,
        sample_size: u64,
        rng: &mut dyn rand::RngCore,
        resolve: impl FnOnce(usize, &mut dyn rand::RngCore) -> Result<(usize, usize), CandlePpoError>,
    ) -> Result<IndexMap<String, Vec<f64>>, CandlePpoError> {
        let Some(buffer) = buffer else {
            return Ok(IndexMap::new());
        };
        let rollout =
            buffer
                .sample(sample_size, rng)
                .map_err(|message| CandlePpoError::Adapter {
                    stage: "sample",
                    message,
                })?;
        self.updating = true;
        let mut batch = self.process(&rollout, rng)?;
        let (batch_size, repeat) = resolve(rollout.rewards.len(), rng)?;
        let result = self.learn(&mut batch, batch_size, repeat, rng)?;
        if let Some(weights) = &rollout.replay_weights {
            buffer
                .update_weights(&rollout.indices, weights)
                .map_err(|message| CandlePpoError::Adapter {
                    stage: "post-process",
                    message,
                })?;
        }
        if let Some(scheduler) = scheduler {
            scheduler
                .step(&mut self.optimizer)
                .map_err(|message| CandlePpoError::Adapter {
                    stage: "scheduler",
                    message,
                })?;
        }
        self.updating = false;
        Ok(result)
    }

    /// # Errors
    /// Propagates actor/distribution errors. Caller state passes through unchanged.
    pub fn forward<State, R: Rng + ?Sized>(
        &self,
        obs: &RecurrentObservation,
        state: State,
        rng: &mut R,
    ) -> Result<CategoricalForward<State>, CandlePpoError> {
        Ok(self
            .actor
            .policy_forward(obs, state, self.mode, self.config.deterministic_eval, rng)?)
    }

    fn compute_returns<R: Rng + ?Sized>(
        &mut self,
        rollout: &CandlePpoRollout,
        rng: &mut R,
    ) -> Result<CandleReturns, CandlePpoError> {
        let mut values = Vec::new();
        let mut next_values = Vec::new();
        for positions in minibatch_indices(
            rollout.rewards.len(),
            self.config.max_batch_size,
            false,
            true,
            rng,
        )? {
            let indices =
                positions_tensor(&positions, rollout.observations.data_processed.device())?;
            values.push(
                self.critic
                    .forward(&rollout.observations.select_batch(&indices)?)?
                    .detach(),
            );
            next_values.push(
                self.critic
                    .forward(&rollout.next_observations.select_batch(&indices)?)?
                    .detach(),
            );
        }
        Ok(prepare_returns(
            &CandleReturnInput {
                rewards: rollout.rewards.view(),
                terminated: rollout.terminated.view(),
                truncated: rollout.truncated.view(),
                bootstrap_valid: rollout.bootstrap_valid.view(),
                indices: &rollout.indices,
                unfinished_indices: &rollout.unfinished_indices,
                values: &Tensor::cat(&values, 0)?,
                next_values: &Tensor::cat(&next_values, 0)?,
            },
            self.config.gamma,
            self.config.gae_lambda,
            self.config
                .reward_normalization
                .then_some(&mut self.return_statistics),
        )?)
    }

    /// Ordered critic evaluation, normalized returns, action conversion and old
    /// log-probability capture. Training mode still samples during capture.
    /// On later failure, earlier running-statistic updates remain observable.
    /// # Errors
    /// Propagates batch/observation, return, action and backend failures.
    pub fn process<'a, R: Rng + ?Sized>(
        &mut self,
        rollout: &'a CandlePpoRollout,
        rng: &mut R,
    ) -> Result<PreparedPpoRollout<'a>, CandlePpoError> {
        let targets = self.compute_returns(rollout, rng)?;
        let actions = rollout
            .actions
            .to_device(targets.old_values.device())?
            .to_dtype(targets.old_values.dtype())?
            .detach();
        let mut old_log_prob = Vec::new();
        for positions in minibatch_indices(
            rollout.rewards.len(),
            self.config.max_batch_size,
            false,
            true,
            rng,
        )? {
            let indices =
                positions_tensor(&positions, rollout.observations.data_processed.device())?;
            let prediction =
                self.forward(&rollout.observations.select_batch(&indices)?, (), rng)?;
            old_log_prob.push(
                prediction
                    .distribution
                    .log_prob(&actions.index_select(&indices, 0)?)?
                    .detach(),
            );
        }
        Ok(PreparedPpoRollout {
            rollout,
            actions,
            targets,
            old_log_prob: Tensor::cat(&old_log_prob, 0)?,
        })
    }

    fn learn_batch<R: Rng + ?Sized>(
        &mut self,
        batch: &PreparedPpoRollout<'_>,
        positions: &[usize],
        rng: &mut R,
    ) -> Result<[f64; 4], CandlePpoError> {
        let indices = positions_tensor(positions, batch.actions.device())?;
        let observation = batch.rollout.observations.select_batch(&indices)?;
        let prediction = self.forward(&observation, (), rng)?;
        let loss = self.config.loss.loss(&PpoLossInput {
            probabilities: &prediction.logits,
            values: &self.critic.forward(&observation)?,
            actions: &batch.actions.index_select(&indices, 0)?,
            old_log_prob: &batch.old_log_prob.index_select(&indices, 0)?,
            advantages: &batch.targets.advantages.index_select(&indices, 0)?,
            returns: &batch.targets.returns.index_select(&indices, 0)?,
            old_values: Some(&batch.targets.old_values.index_select(&indices, 0)?),
        })?;
        let mut gradients = loss.total.backward()?;
        if let Some(limit) = self.config.max_grad_norm.filter(|limit| *limit != 0.) {
            self.optimizer.clip_grad_norm(&mut gradients, limit)?;
        }
        self.optimizer.step(&gradients)?;
        let mut metrics = [0.; 4];
        for (slot, tensor) in
            metrics
                .iter_mut()
                .zip([loss.total, loss.policy, loss.value, loss.entropy])
        {
            *slot = tensor.to_dtype(DType::F64)?.to_scalar::<f64>()?;
        }
        Ok(metrics)
    }

    /// Shuffle once per repeat, merge a short final minibatch, perform fresh
    /// backward/optional clipping/Adam, and return all four per-update loss lists.
    /// With recomputation, old log probabilities remain fixed while critic values,
    /// returns and advantages refresh after the first repeat. Zero repeats do not
    /// split, validate batch size, sample, update statistics or mutate parameters.
    /// # Errors
    /// Propagates preprocessing/minibatch/backend errors. Earlier successful
    /// optimizer updates and running-statistic updates are not rolled back.
    pub fn learn<R: Rng + ?Sized>(
        &mut self,
        batch: &mut PreparedPpoRollout<'_>,
        batch_size: usize,
        repeat: usize,
        rng: &mut R,
    ) -> Result<IndexMap<String, Vec<f64>>, CandlePpoError> {
        let mut metrics: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
        for step in 0..repeat {
            if self.config.recompute_advantage && step > 0 {
                batch.targets = self.compute_returns(batch.rollout, rng)?;
            }
            for positions in
                minibatch_indices(batch.rollout.rewards.len(), batch_size, true, true, rng)?
            {
                let losses = self.learn_batch(batch, &positions, rng)?;
                for (values, loss) in metrics.iter_mut().zip(losses) {
                    values.push(loss);
                }
            }
        }
        Ok(["loss", "loss/clip", "loss/vf", "loss/ent"]
            .into_iter()
            .map(str::to_owned)
            .zip(metrics)
            .collect())
    }
}

// Native checkpoint delegation is strict load_state_dict, not set_weight's
// legacy-name retry. Only model state is saved, as in the source policy.
impl TrainingPolicyState<CandlePolicySnapshot<()>> for CandlePpo {
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

impl<M> PolicyWeightLoader<Tensor, M> for CandlePpo {
    fn load_weights(
        &mut self,
        state: &mut PolicyWeights<Tensor, M>,
    ) -> Result<(), PolicyWeightLoadError> {
        self.policy_state(())
            .map_err(|error| PolicyWeightLoadError::Other(error.to_string()))?
            .load_native_weights(state)
    }
}

#[cfg(test)]
#[path = "../tests/support/rl_candle_policy.rs"]
mod tests;
