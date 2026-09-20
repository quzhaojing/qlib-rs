//! DQN inference, exploration and training on the shared native collector/vessel.

use crate::TrainingPolicyState;
use crate::rl_candle_checkpoint::CandlePolicySnapshot;
use crate::rl_candle_collector::CandleCollectionPolicy;
use crate::rl_candle_dqn::DqnForward;
use crate::rl_candle_dqn_policy::{
    CandleDqn, CandleDqnError, CandleDqnObservation, CandleDqnReplay, CandleDqnScheduler,
};
use crate::rl_candle_network::RecurrentObservation;
use crate::{
    TrainingMetricScalar, TrainingMetricValue, TrainingPolicyMode, TrainingRunPolicy,
    TrainingUpdateOptions, TrainingVesselMetrics, TrainingVesselRunError,
};
use candle_core::Tensor;
use rand::RngCore;

pub struct CandleDqnVessel<R> {
    pub policy: CandleDqn,
    pub rng: R,
    pub scheduler: Option<Box<dyn CandleDqnScheduler + Send>>,
}
impl<R: RngCore> CandleDqnVessel<R> {
    #[must_use]
    pub fn new(policy: CandleDqn, rng: R) -> Self {
        Self {
            policy,
            rng,
            scheduler: None,
        }
    }
    /// # Errors
    /// Propagates model/mask/action errors without changing caller state.
    pub fn forward<State>(
        &mut self,
        input: &CandleDqnObservation,
        state: State,
    ) -> Result<DqnForward<State>, CandleDqnError> {
        self.policy.forward(input, state)
    }
}
impl<B: CandleDqnReplay, R: RngCore + Send> TrainingRunPolicy<B> for CandleDqnVessel<R> {
    fn set_mode(&mut self, mode: TrainingPolicyMode) -> Result<(), TrainingVesselRunError> {
        self.policy.set_mode(mode);
        Ok(())
    }
    fn update(
        &mut self,
        sample_size: u64,
        buffer: Option<&mut B>,
        options: &TrainingUpdateOptions,
    ) -> Result<TrainingVesselMetrics, TrainingVesselRunError> {
        for key in options.keys() {
            if matches!(key.as_str(), "sample_size" | "buffer") {
                return Err(TrainingVesselRunError::DuplicateKeyword(key.clone()));
            }
        }
        // DQN.learn accepts and ignores additional keyword arguments; unlike PPO
        // it does not require batch_size/repeat or turn its scalar loss into a list.
        let scheduler = self
            .scheduler
            .as_mut()
            .map(|s| s.as_mut() as &mut dyn CandleDqnScheduler);
        let metrics = self
            .policy
            .update(
                buffer.map(|b| b as &mut dyn CandleDqnReplay),
                scheduler,
                sample_size,
                &mut self.rng,
            )
            .map_err(|e| TrainingVesselRunError::Plugin {
                stage: "DQN update".into(),
                message: e.to_string(),
            })?;
        Ok(metrics
            .into_iter()
            .map(|(name, value)| {
                (
                    name,
                    TrainingMetricValue::Scalar(TrainingMetricScalar::Float(value)),
                )
            })
            .collect())
    }
}
impl<R: Send> TrainingPolicyState<CandlePolicySnapshot<()>> for CandleDqnVessel<R> {
    fn state_dict(&mut self) -> Result<CandlePolicySnapshot<()>, String> {
        self.policy.state_dict()
    }

    fn load_state_dict(&mut self, state: &CandlePolicySnapshot<()>) -> Result<(), String> {
        self.policy.load_state_dict(state)
    }
}

impl<R: RngCore + Send> CandleCollectionPolicy for CandleDqnVessel<R> {
    fn actions(&mut self, observations: &RecurrentObservation) -> Result<Tensor, String> {
        self.forward(
            &CandleDqnObservation {
                observation: observations.clone(),
                mask: None,
            },
            (),
        )
        .map(|out| out.actions)
        .map_err(|e| e.to_string())
    }
    fn exploration_noise(
        &mut self,
        actions: Tensor,
        _: &RecurrentObservation,
    ) -> Result<Tensor, String> {
        let mut values = actions.to_vec1::<i64>().map_err(|e| e.to_string())?;
        self.policy
            .exploration(&mut values, None, &mut self.rng)
            .map_err(|e| e.to_string())?;
        Tensor::new(values, actions.device()).map_err(|e| e.to_string())
    }
}
