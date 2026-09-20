//! Synchronous full-history policy collection over the existing finite environments.

use crate::rl_candle_network::RecurrentObservation;
use crate::rl_candle_replay::{CandleReplayBatch, CandleVectorReplayBuffer, concatenate};
use crate::rl_candle_vessel::CandlePpoVessel;
use crate::rl_replay_index::ReplayEpisode;
use crate::training_vessel_runner::TrainingCollectorBuildResult;
use crate::{
    FiniteVectorEnv, TrainingCollectLimit, TrainingCollectorFactory, TrainingMetricScalar,
    TrainingMetricValue, TrainingRunCollector, TrainingRunPolicy, TrainingVesselMetrics,
    TrainingVesselRunError,
};
use arrow_array::{Float64Array, Int64Array};
use candle_core::Tensor;
use indexmap::IndexMap;
use ndarray::Array1;
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use rand::RngCore;
use serde_json::Value;
use std::sync::Arc;

pub type CandleCollectorEnvironment<I> = FiniteVectorEnv<RecurrentObservation, i64, f64, I>;
type CollectorBuild<I, P> = TrainingCollectorBuildResult<
    RecurrentObservation,
    i64,
    f64,
    I,
    CandleVectorReplayBuffer,
    Value,
    P,
>;

fn failure(stage: &str, message: impl Into<String>) -> TrainingVesselRunError {
    TrainingVesselRunError::Plugin {
        stage: stage.into(),
        message: message.into(),
    }
}

/// Linked full-history policy seam. Default discrete mapping/noise are identity.
/// Policy inference and update operate on the same instance; no downcast or copy.
pub trait CandleCollectionPolicy: TrainingRunPolicy<CandleVectorReplayBuffer> {
    /// # Errors
    /// Returns observation, model or action-distribution failures.
    fn actions(&mut self, observations: &RecurrentObservation) -> Result<Tensor, String>;

    /// # Errors
    /// An optional exploration adapter may fail after action inference.
    fn exploration_noise(
        &mut self,
        actions: Tensor,
        _observations: &RecurrentObservation,
    ) -> Result<Tensor, String> {
        Ok(actions)
    }

    /// Environment mapping does not change the actions committed to replay.
    /// # Errors
    /// Returns malformed/non-I64 action vectors or adapter failures.
    fn map_action(&mut self, actions: &Tensor) -> Result<Vec<i64>, String> {
        actions.to_vec1::<i64>().map_err(|error| error.to_string())
    }
}

impl<R: RngCore + Send> CandleCollectionPolicy for CandlePpoVessel<R> {
    fn actions(&mut self, observations: &RecurrentObservation) -> Result<Tensor, String> {
        self.forward(observations, ())
            .map(|result| result.actions)
            .map_err(|error| error.to_string())
    }
}

/// Interpret the legacy four-result environment's time-limit flag without
/// conflating termination with truncation. Domain info types can implement this.
pub trait CandleTerminationInfo: Clone + Send + 'static {
    /// # Errors
    /// Returns invalid time-limit flag representation.
    fn time_limit_truncated(&self) -> Result<bool, String>;
}
impl CandleTerminationInfo for () {
    fn time_limit_truncated(&self) -> Result<bool, String> {
        Ok(false)
    }
}
// The simulator wrapper's info contains only logs, not a top-level time-limit flag.
impl<V: Clone + Send + 'static> CandleTerminationInfo for crate::RlLogInfo<V> {
    fn time_limit_truncated(&self) -> Result<bool, String> {
        Ok(false)
    }
}
// EnvWrapper keeps auxiliary data nested; it never supplies the top-level flag.
impl<O, A, I> CandleTerminationInfo for crate::EnvironmentStepInfo<O, A, I>
where
    O: Clone + Send + 'static,
    A: Clone + Send + 'static,
    I: Clone + Send + 'static,
{
    fn time_limit_truncated(&self) -> Result<bool, String> {
        Ok(false)
    }
}
impl CandleTerminationInfo for Value {
    fn time_limit_truncated(&self) -> Result<bool, String> {
        match self.get("TimeLimit.truncated") {
            None => Ok(false),
            Some(Value::Bool(value)) => Ok(*value),
            Some(_) => Err("TimeLimit.truncated must be a boolean".into()),
        }
    }
}

pub trait CandleCollectorClock: Send {
    fn seconds(&mut self) -> f64;
}
#[derive(Default)]
pub struct CandleCollectorWallClock;
impl CandleCollectorClock for CandleCollectorWallClock {
    fn seconds(&mut self) -> f64 {
        let now = chrono::Utc::now();
        now.timestamp().to_f64().expect("i64 timestamps fit f64")
            + f64::from(now.timestamp_subsec_nanos()) * 1e-9
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CandleCollectionStatistics {
    pub steps: usize,
    pub episodes: usize,
    pub seconds: f64,
}

pub struct CandleCollector {
    replay: CandleVectorReplayBuffer,
    observations: Vec<Option<RecurrentObservation>>,
    environments: usize,
    exploration_noise: bool,
    statistics: CandleCollectionStatistics,
    clock: Box<dyn CandleCollectorClock>,
}

struct CollectedStep {
    next: Vec<Option<RecurrentObservation>>,
    dones: Vec<bool>,
    episodes: Vec<ReplayEpisode>,
}

enum Target {
    Episodes(usize),
    Steps(BigInt),
}
impl Target {
    fn parse(limit: TrainingCollectLimit) -> Result<Self, TrainingVesselRunError> {
        match limit {
            TrainingCollectLimit::Episodes(Some(count)) if count > 0 => usize::try_from(count)
                .map(Self::Episodes)
                .map_err(|error| failure("collect limit", error.to_string())),
            TrainingCollectLimit::Steps(count) if count > BigInt::from(0) => Ok(Self::Steps(count)),
            _ => Err(failure(
                "collect limit",
                "specify a positive episode or step count",
            )),
        }
    }
    fn reached(&self, progress: &Progress) -> bool {
        match self {
            Self::Episodes(count) => progress.rewards.len() >= *count,
            Self::Steps(count) => BigInt::from(progress.steps) >= *count,
        }
    }
}

#[derive(Default)]
struct Progress {
    steps: usize,
    rewards: Vec<f64>,
    lengths: Vec<i64>,
    starts: Vec<i64>,
}
impl Progress {
    fn record(
        &mut self,
        dones: &[bool],
        episodes: &[ReplayEpisode],
    ) -> Result<Vec<usize>, TrainingVesselRunError> {
        self.steps = self
            .steps
            .checked_add(dones.len())
            .ok_or_else(|| failure("collect statistics", "step count overflow"))?;
        let mut finished = Vec::new();
        for (index, (done, episode)) in dones.iter().zip(episodes).enumerate() {
            if *done {
                self.rewards.push(episode.reward);
                self.lengths.push(
                    i64::try_from(episode.length)
                        .map_err(|error| failure("episode length", error.to_string()))?,
                );
                self.starts.push(
                    i64::try_from(episode.start)
                        .map_err(|error| failure("episode start", error.to_string()))?,
                );
                finished.push(index);
            }
        }
        Ok(finished)
    }
    fn metrics(self) -> TrainingVesselMetrics {
        let rewards = Array1::from_vec(self.rewards.clone());
        let lengths = Array1::from_iter(
            self.lengths
                .iter()
                .map(|value| value.to_f64().expect("i64 fits f64")),
        );
        let (reward_mean, reward_std, length_mean, length_std) = if self.rewards.is_empty() {
            (0., 0., 0., 0.)
        } else {
            (
                rewards.mean().unwrap_or(0.),
                rewards.std(0.),
                lengths.mean().unwrap_or(0.),
                lengths.std(0.),
            )
        };
        IndexMap::from([
            (
                "n/ep".into(),
                TrainingMetricValue::Scalar(TrainingMetricScalar::Integer(
                    self.rewards.len().into(),
                )),
            ),
            (
                "n/st".into(),
                TrainingMetricValue::Scalar(TrainingMetricScalar::Integer(self.steps.into())),
            ),
            (
                "rews".into(),
                TrainingMetricValue::Numeric(Arc::new(Float64Array::from(self.rewards))),
            ),
            (
                "lens".into(),
                TrainingMetricValue::Numeric(Arc::new(Int64Array::from(self.lengths))),
            ),
            (
                "idxs".into(),
                TrainingMetricValue::Numeric(Arc::new(Int64Array::from(self.starts))),
            ),
            (
                "rew".into(),
                TrainingMetricValue::Scalar(TrainingMetricScalar::Float(reward_mean)),
            ),
            (
                "len".into(),
                TrainingMetricValue::Scalar(TrainingMetricScalar::Float(length_mean)),
            ),
            (
                "rew_std".into(),
                TrainingMetricValue::Scalar(TrainingMetricScalar::Float(reward_std)),
            ),
            (
                "len_std".into(),
                TrainingMetricValue::Scalar(TrainingMetricScalar::Float(length_std)),
            ),
        ])
    }
}

impl CandleCollector {
    /// Does not clear a supplied replay; None allocates one slot per environment.
    /// # Errors
    /// Returns buffer validation/allocation and initial environment reset failures.
    pub fn new<I: CandleTerminationInfo>(
        environment: &mut CandleCollectorEnvironment<I>,
        replay: Option<CandleVectorReplayBuffer>,
        exploration_noise: bool,
    ) -> Result<Self, TrainingVesselRunError> {
        let environments = environment.environment_count();
        let replay = match replay {
            Some(replay) => replay,
            None => CandleVectorReplayBuffer::new(environments, environments)
                .map_err(|error| failure("collector buffer", error.to_string()))?,
        };
        if replay.children().len() < environments {
            return Err(failure(
                "collector buffer",
                "fewer replay children than environments",
            ));
        }
        let mut collector = Self {
            replay,
            observations: Vec::new(),
            environments,
            exploration_noise,
            statistics: CandleCollectionStatistics::default(),
            clock: Box::new(CandleCollectorWallClock),
        };
        collector.reset(environment, false)?;
        Ok(collector)
    }
    #[must_use]
    pub fn replay(&self) -> &CandleVectorReplayBuffer {
        &self.replay
    }
    #[must_use]
    pub fn statistics(&self) -> &CandleCollectionStatistics {
        &self.statistics
    }
    pub fn set_clock(&mut self, clock: Box<dyn CandleCollectorClock>) {
        self.clock = clock;
    }
    pub fn reset_statistics(&mut self) {
        self.statistics = CandleCollectionStatistics::default();
    }
    pub fn reset_buffer(&mut self, keep_statistics: bool) {
        self.replay.reset(keep_statistics);
    }

    /// Clear observation state, reset environments, optionally clear replay, then statistics.
    /// # Errors
    /// Environment reset failures preserve the not-yet-reset replay/statistics.
    pub fn reset<I: CandleTerminationInfo>(
        &mut self,
        environment: &mut CandleCollectorEnvironment<I>,
        reset_buffer: bool,
    ) -> Result<(), TrainingVesselRunError> {
        self.observations.clear();
        self.observations = environment.reset(None)?.observations;
        if reset_buffer {
            self.reset_buffer(false);
        }
        self.reset_statistics();
        Ok(())
    }

    fn collect_step<I: CandleTerminationInfo, P: CandleCollectionPolicy + ?Sized>(
        &mut self,
        policy: &mut P,
        environment: &mut CandleCollectorEnvironment<I>,
        ready: &[usize],
    ) -> Result<CollectedStep, TrainingVesselRunError> {
        let observations = stack_observations(&self.observations)?;
        let mut actions = policy
            .actions(&observations)
            .map_err(|error| failure("collector forward", error))?;
        if self.exploration_noise {
            actions = policy
                .exploration_noise(actions, &observations)
                .map_err(|error| failure("collector exploration", error))?;
        }
        let mapped = policy
            .map_action(&actions)
            .map_err(|error| failure("collector map action", error))?;
        let transitions = environment.step(&mapped, Some(ready))?.transitions;
        let next: Vec<_> = transitions
            .iter()
            .map(|step| step.observation.clone())
            .collect();
        let rewards = transitions
            .iter()
            .map(|step| {
                step.reward
                    .ok_or_else(|| failure("collector reward", "missing reward"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let truncated = transitions
            .iter()
            .map(|step| {
                step.info
                    .as_ref()
                    .map_or(Ok(false), CandleTerminationInfo::time_limit_truncated)
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| failure("collector truncation", error))?;
        let terminated: Vec<_> = transitions
            .iter()
            .zip(&truncated)
            .map(|(step, truncated)| step.done && !truncated)
            .collect();
        let dones: Vec<_> = transitions.iter().map(|step| step.done).collect();
        let following = stack_observations(&next)?;
        let episodes = self
            .replay
            .add(
                &CandleReplayBatch {
                    observations: &observations,
                    next_observations: &following,
                    actions: &actions,
                    rewards: &rewards,
                    terminated: &terminated,
                    truncated: &truncated,
                },
                Some(ready),
            )
            .map_err(|error| failure("collector replay", error.to_string()))?;
        Ok(CollectedStep {
            next,
            dones,
            episodes,
        })
    }

    /// Default Qlib synchronous collection: inferred actions, optional exploration,
    /// legacy time-limit flags, replay writes before finished-environment resets.
    /// # Errors
    /// First policy/environment/storage/limit error, preserving preceding side effects.
    pub fn collect<I: CandleTerminationInfo, P: CandleCollectionPolicy + ?Sized>(
        &mut self,
        policy: &mut P,
        environment: &mut CandleCollectorEnvironment<I>,
        limit: TrainingCollectLimit,
    ) -> Result<TrainingVesselMetrics, TrainingVesselRunError> {
        if environment.environment_count() != self.environments {
            return Err(failure(
                "collector environment",
                "environment count changed",
            ));
        }
        let target = Target::parse(limit)?;
        let active = match &target {
            Target::Episodes(count) => {
                let active = self.environments.min(*count);
                self.observations.truncate(active);
                active
            }
            Target::Steps(count) => {
                if count % self.environments != BigInt::from(0) {
                    tracing::warn!(
                        "step count is not a multiple of environment count; collection may overshoot"
                    );
                }
                self.environments
            }
        };
        let mut ready: Vec<_> = (0..active).collect();
        let started = self.clock.seconds();
        let mut progress = Progress::default();
        loop {
            if self.observations.len() != ready.len() {
                return Err(failure(
                    "collector state",
                    "observation rows differ from ready environment ids",
                ));
            }
            let CollectedStep {
                mut next,
                dones,
                episodes,
            } = self.collect_step(policy, environment, &ready)?;
            let finished = progress.record(&dones, &episodes)?;
            if !finished.is_empty() {
                let ids: Vec<_> = finished.iter().map(|&index| ready[index]).collect();
                let reset = environment.reset(Some(&ids))?.observations;
                for (&index, observation) in finished.iter().zip(reset) {
                    next[index] = observation;
                }
                if let Target::Episodes(count) = &target {
                    let surplus = ready
                        .len()
                        .saturating_sub(count.saturating_sub(progress.rewards.len()));
                    let mut keep = vec![true; ready.len()];
                    for &index in finished.iter().take(surplus) {
                        keep[index] = false;
                    }
                    ready = ready
                        .into_iter()
                        .zip(&keep)
                        .filter_map(|(id, keep)| keep.then_some(id))
                        .collect();
                    next = next
                        .into_iter()
                        .zip(keep)
                        .filter_map(|(obs, keep)| keep.then_some(obs))
                        .collect();
                }
            }
            self.observations = next;
            if target.reached(&progress) {
                break;
            }
        }
        self.statistics.steps = self
            .statistics
            .steps
            .checked_add(progress.steps)
            .ok_or_else(|| failure("collector statistics", "step count overflow"))?;
        self.statistics.episodes = self
            .statistics
            .episodes
            .checked_add(progress.rewards.len())
            .ok_or_else(|| failure("collector statistics", "episode count overflow"))?;
        let elapsed = self.clock.seconds() - started;
        self.statistics.seconds += if elapsed < 1e-9 { 1e-9 } else { elapsed };
        if matches!(target, Target::Episodes(_)) {
            self.observations.clear();
            self.observations = environment.reset(None)?.observations;
        }
        Ok(progress.metrics())
    }
}

fn stack_observations(
    observations: &[Option<RecurrentObservation>],
) -> Result<RecurrentObservation, TrainingVesselRunError> {
    let rows = observations
        .iter()
        .map(|obs| {
            obs.as_ref()
                .ok_or_else(|| failure("collector observation", "missing observation"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let first = rows
        .first()
        .ok_or_else(|| failure("collector observation", "empty observation batch"))?;
    concatenate(&rows, first).map_err(|error| failure("collector observation", error.to_string()))
}

impl<I: CandleTerminationInfo, P: CandleCollectionPolicy + ?Sized>
    TrainingRunCollector<RecurrentObservation, i64, f64, I, CandleVectorReplayBuffer, Value, P>
    for CandleCollector
{
    fn collect(
        &mut self,
        policy: &mut P,
        environment: &mut CandleCollectorEnvironment<I>,
        limit: TrainingCollectLimit,
    ) -> Result<TrainingVesselMetrics, TrainingVesselRunError> {
        Self::collect(self, policy, environment, limit)
    }
    fn buffer(&mut self) -> Result<Option<&mut CandleVectorReplayBuffer>, TrainingVesselRunError> {
        Ok(Some(&mut self.replay))
    }
}

#[derive(Default)]
pub struct CandleCollectorFactory;
impl<I: CandleTerminationInfo, P: CandleCollectionPolicy + ?Sized + 'static>
    TrainingCollectorFactory<RecurrentObservation, i64, f64, I, CandleVectorReplayBuffer, Value, P>
    for CandleCollectorFactory
{
    fn create_buffer(
        &mut self,
        capacity: i64,
        environments: usize,
    ) -> Result<CandleVectorReplayBuffer, TrainingVesselRunError> {
        let capacity = usize::try_from(capacity)
            .map_err(|error| failure("collector buffer", error.to_string()))?;
        CandleVectorReplayBuffer::new(capacity, environments)
            .map_err(|error| failure("collector buffer", error.to_string()))
    }
    fn create_collector(
        &mut self,
        _policy: &mut P,
        environment: &mut CandleCollectorEnvironment<I>,
        buffer: Option<CandleVectorReplayBuffer>,
        exploration_noise: bool,
    ) -> CollectorBuild<I, P> {
        Ok(Box::new(CandleCollector::new(
            environment,
            buffer,
            exploration_noise,
        )?))
    }
}

#[cfg(test)]
#[path = "../tests/support/rl_candle_collector.rs"]
mod tests;
