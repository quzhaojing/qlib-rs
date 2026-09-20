//! Default vessel collection/update orchestration, independent of a concrete ML runtime.

use indexmap::IndexMap;
use num_bigint::BigInt;
use serde_json::Value;
use std::sync::Arc;
use thiserror::Error;

use crate::{
    FiniteVectorEnv, FiniteVectorError, INF, TrainingMetricValue, TrainingPolicyState,
    TrainingTrainerView, TrainingVesselBinding, TrainingVesselBindingError,
    TrainingVesselCheckpoint, TrainingVesselLog, TrainingVesselLogError, TrainingVesselStateError,
};

pub type TrainingVesselMetrics = IndexMap<String, TrainingMetricValue>;
/// Keyword values are adapter-owned and forwarded by reference, without cloning or coercion.
/// JSON is a convenience default, not a restriction on tensors or other opaque values.
pub type TrainingUpdateOptions<V = Value> = IndexMap<String, V>;

#[derive(Clone, Debug, PartialEq)]
pub struct TrainingVesselRunConfig<V = Value> {
    pub buffer_size: i64,
    pub episode_per_iter: i64,
    pub update_kwargs: TrainingUpdateOptions<V>,
}

impl<V> Default for TrainingVesselRunConfig<V> {
    fn default() -> Self {
        Self {
            buffer_size: 20_000,
            episode_per_iter: 1_000,
            update_kwargs: IndexMap::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrainingPolicyMode {
    Train,
    Evaluation,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TrainingCollectLimit {
    /// A live trainer is read twice when development mode is enabled; its second
    /// read may be `None`, which is passed through rather than replaced with a default.
    Episodes(Option<i64>),
    Steps(BigInt),
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum TrainingVesselRunError {
    #[error(transparent)]
    Environment(#[from] FiniteVectorError),
    #[error(transparent)]
    Binding(#[from] TrainingVesselBindingError),
    #[error(transparent)]
    Log(#[from] TrainingVesselLogError),
    #[error("policy update received multiple values for keyword '{0}'")]
    DuplicateKeyword(String),
    #[error("training plugin failed at {stage}: {message}")]
    Plugin { stage: String, message: String },
}

pub trait TrainingRunPolicy<Buffer, V = Value>: Send {
    /// # Errors
    /// Returns a policy mode-switch failure, outside the environment guard.
    fn set_mode(&mut self, mode: TrainingPolicyMode) -> Result<(), TrainingVesselRunError>;

    /// `sample_size=0` requests the entire collector buffer. The collector owns that buffer;
    /// policies may mutate it, and a collector-supplied `None` is forwarded unchanged.
    ///
    /// # Errors
    /// Returns policy update failures without rolling back earlier mutations.
    fn update(
        &mut self,
        sample_size: u64,
        buffer: Option<&mut Buffer>,
        options: &TrainingUpdateOptions<V>,
    ) -> Result<TrainingVesselMetrics, TrainingVesselRunError>;
}

pub trait TrainingRunCollector<
    O,
    A,
    R,
    I,
    Buffer,
    V = Value,
    P: ?Sized = dyn TrainingRunPolicy<Buffer, V>,
>: Send
{
    /// Policy and environment are reborrowed per call, not retained as aliased mutable references.
    ///
    /// # Errors
    /// Returns collection failures; environment exhaustion is suppressed by the vessel guard.
    fn collect(
        &mut self,
        policy: &mut P,
        environment: &mut FiniteVectorEnv<O, A, R, I>,
        limit: TrainingCollectLimit,
    ) -> Result<TrainingVesselMetrics, TrainingVesselRunError>;

    /// # Errors
    /// Returns a buffer-access failure before update keyword validation occurs.
    fn buffer(&mut self) -> Result<Option<&mut Buffer>, TrainingVesselRunError>;
}

pub type BoxTrainingRunCollector<
    O,
    A,
    R,
    I,
    Buffer,
    V = Value,
    P = dyn TrainingRunPolicy<Buffer, V>,
> = Box<dyn TrainingRunCollector<O, A, R, I, Buffer, V, P>>;

pub type TrainingCollectorBuildResult<
    O,
    A,
    R,
    I,
    Buffer,
    V = Value,
    P = dyn TrainingRunPolicy<Buffer, V>,
> = Result<BoxTrainingRunCollector<O, A, R, I, Buffer, V, P>, TrainingVesselRunError>;

/// Linked adapter boundary, not a replay-buffer implementation or a stable native plugin ABI.
pub trait TrainingCollectorFactory<
    O,
    A,
    R,
    I,
    Buffer,
    V = Value,
    P: ?Sized = dyn TrainingRunPolicy<Buffer, V>,
>: Send
{
    /// # Errors
    /// Returns replay-buffer construction failures.
    fn create_buffer(
        &mut self,
        capacity: i64,
        environments: usize,
    ) -> Result<Buffer, TrainingVesselRunError>;

    /// A missing buffer and disabled exploration denote evaluation defaults.
    /// The adapter may retain shared handles, but cannot retain the borrowed policy/environment.
    ///
    /// # Errors
    /// Returns collector construction failures.
    fn create_collector(
        &mut self,
        policy: &mut P,
        environment: &mut FiniteVectorEnv<O, A, R, I>,
        buffer: Option<Buffer>,
        exploration_noise: bool,
    ) -> TrainingCollectorBuildResult<O, A, R, I, Buffer, V, P>;
}

pub struct TrainingVesselRunner<
    O,
    A,
    R,
    I,
    Buffer,
    V = Value,
    P: ?Sized = dyn TrainingRunPolicy<Buffer, V>,
> {
    policy: Box<P>,
    factory: Box<dyn TrainingCollectorFactory<O, A, R, I, Buffer, V, P>>,
    binding: TrainingVesselBinding,
    config: TrainingVesselRunConfig<V>,
    log: TrainingVesselLog,
}

impl<O, A, R, I, Buffer, V, P, State> crate::RlCheckpointState<TrainingVesselCheckpoint<State>>
    for TrainingVesselRunner<O, A, R, I, Buffer, V, P>
where
    O: Clone + 'static,
    A: Clone + 'static,
    R: Clone + 'static,
    I: Clone + 'static,
    Buffer: 'static,
    V: 'static,
    P: TrainingRunPolicy<Buffer, V> + TrainingPolicyState<State> + ?Sized + 'static,
{
    fn save_checkpoint(&mut self) -> Result<TrainingVesselCheckpoint<State>, String> {
        self.state_dict().map_err(|error| error.to_string())
    }

    fn load_checkpoint(&mut self, state: &TrainingVesselCheckpoint<State>) -> Result<(), String> {
        self.load_state_dict(state)
            .map_err(|error| error.to_string())
    }
}

impl<O, A, R, I, Buffer, V> TrainingVesselRunner<O, A, R, I, Buffer, V>
where
    O: Clone + 'static,
    A: Clone + 'static,
    R: Clone + 'static,
    I: Clone + 'static,
    Buffer: 'static,
    V: 'static,
{
    #[must_use]
    pub fn new(
        policy: Box<dyn TrainingRunPolicy<Buffer, V>>,
        factory: Box<dyn TrainingCollectorFactory<O, A, R, I, Buffer, V>>,
        binding: TrainingVesselBinding,
        config: TrainingVesselRunConfig<V>,
        log: TrainingVesselLog,
    ) -> Self {
        Self::with_policy(policy, factory, binding, config, log)
    }
}

impl<O, A, R, I, Buffer, V, P> TrainingVesselRunner<O, A, R, I, Buffer, V, P>
where
    O: Clone + 'static,
    A: Clone + 'static,
    R: Clone + 'static,
    I: Clone + 'static,
    Buffer: 'static,
    V: 'static,
    P: TrainingRunPolicy<Buffer, V> + ?Sized + 'static,
{
    /// Keeps a concrete or extended policy interface available to its collector,
    /// while collection and update borrow the same owned policy instance.
    #[must_use]
    pub fn with_policy(
        policy: Box<P>,
        factory: Box<dyn TrainingCollectorFactory<O, A, R, I, Buffer, V, P>>,
        binding: TrainingVesselBinding,
        config: TrainingVesselRunConfig<V>,
        log: TrainingVesselLog,
    ) -> Self {
        Self {
            policy,
            factory,
            binding,
            config,
            log,
        }
    }

    /// Rebinds the runner to the live Trainer without replacing policy, factory, or config.
    pub fn assign_trainer(&mut self, trainer: &Arc<dyn TrainingTrainerView>) {
        self.binding.assign_trainer(trainer);
    }

    /// Snapshot the same policy instance used by collection and training.
    /// Does not add collector, trainer, scheduler or RNG state to the envelope.
    /// # Errors
    /// Returns the policy's save failure without replacing the running policy.
    pub fn state_dict<State>(
        &mut self,
    ) -> Result<TrainingVesselCheckpoint<State>, TrainingVesselStateError>
    where
        P: TrainingPolicyState<State>,
    {
        self.policy
            .state_dict()
            .map(|policy| TrainingVesselCheckpoint { policy })
            .map_err(TrainingVesselStateError::Save)
    }

    /// Strictly restore the policy in place, retaining its loader's partial mutations.
    /// # Errors
    /// Returns policy restoration errors without applying legacy weight-name retries.
    pub fn load_state_dict<State>(
        &mut self,
        state: &TrainingVesselCheckpoint<State>,
    ) -> Result<(), TrainingVesselStateError>
    where
        P: TrainingPolicyState<State>,
    {
        self.policy
            .load_state_dict(&state.policy)
            .map_err(TrainingVesselStateError::Load)
    }

    /// Collects episodes, updates with the same buffer, merges update metrics over collection
    /// metrics without reordering existing keys, and logs before completing the guard.
    ///
    /// # Errors
    /// Returns the first policy, collector, trainer, logging, or non-exhaustion environment error.
    pub fn train(
        &mut self,
        environment: &mut FiniteVectorEnv<O, A, R, I>,
    ) -> Result<Option<TrainingVesselMetrics>, TrainingVesselRunError> {
        self.policy.set_mode(TrainingPolicyMode::Train)?;
        environment.collect_guarded_with(
            |environment| {
                let buffer = self
                    .factory
                    .create_buffer(self.config.buffer_size, environment.environment_count())?;
                let mut collector = self.factory.create_collector(
                    self.policy.as_mut(),
                    environment,
                    Some(buffer),
                    true,
                )?;
                let episodes = if self.binding.fast_dev_run()?.is_some() {
                    self.binding.fast_dev_run()?
                } else {
                    Some(self.config.episode_per_iter)
                };
                let mut metrics = collector.collect(
                    self.policy.as_mut(),
                    environment,
                    TrainingCollectLimit::Episodes(episodes),
                )?;
                let buffer = collector.buffer()?;
                for key in self.config.update_kwargs.keys() {
                    if matches!(key.as_str(), "sample_size" | "buffer") {
                        return Err(TrainingVesselRunError::DuplicateKeyword(key.clone()));
                    }
                }
                metrics.extend(self.policy.update(0, buffer, &self.config.update_kwargs)?);
                self.log.log_dict(&self.binding, &metrics)?;
                Ok(metrics)
            },
            is_exhausted,
        )
    }

    /// # Errors
    /// Returns evaluation errors; exhaustion inside the guard returns `Ok(None)`.
    pub fn validate(
        &mut self,
        environment: &mut FiniteVectorEnv<O, A, R, I>,
    ) -> Result<Option<TrainingVesselMetrics>, TrainingVesselRunError> {
        self.evaluate(environment)
    }

    /// # Errors
    /// Returns evaluation errors; exhaustion inside the guard returns `Ok(None)`.
    pub fn test(
        &mut self,
        environment: &mut FiniteVectorEnv<O, A, R, I>,
    ) -> Result<Option<TrainingVesselMetrics>, TrainingVesselRunError> {
        self.evaluate(environment)
    }

    fn evaluate(
        &mut self,
        environment: &mut FiniteVectorEnv<O, A, R, I>,
    ) -> Result<Option<TrainingVesselMetrics>, TrainingVesselRunError> {
        self.policy.set_mode(TrainingPolicyMode::Evaluation)?;
        environment.collect_guarded_with(
            |environment| {
                let mut collector = self.factory.create_collector(
                    self.policy.as_mut(),
                    environment,
                    None,
                    false,
                )?;
                let steps = BigInt::from(INF) * environment.environment_count();
                let metrics = collector.collect(
                    self.policy.as_mut(),
                    environment,
                    TrainingCollectLimit::Steps(steps),
                )?;
                self.log.log_dict(&self.binding, &metrics)?;
                Ok(metrics)
            },
            is_exhausted,
        )
    }
}

fn is_exhausted(error: &TrainingVesselRunError) -> bool {
    matches!(
        error,
        TrainingVesselRunError::Environment(FiniteVectorError::Exhausted)
    )
}

#[cfg(test)]
#[path = "../tests/support/training_vessel_runner.rs"]
mod tests;
