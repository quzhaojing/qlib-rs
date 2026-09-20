//! Default vessel composition over the existing seed, policy and environment machinery.

use std::sync::{Arc, Mutex};

use serde_json::Value;

use crate::{
    DataQueue, FiniteVectorEnv, RlCheckpointState, RlTrainerControl, RlTrainerDriverError,
    RlTrainerPhase, RlTrainerRuntime, RlTrainerVessel, TrainingPolicyState, TrainingRunPolicy,
    TrainingSeedPhase, TrainingTrainerView, TrainingVesselBinding, TrainingVesselCheckpoint,
    TrainingVesselRunner, TrainingVesselSeeds,
};

pub type RlTrainingSeed<T> = Arc<Mutex<DataQueue<T>>>;
pub type RlTrainingEnvironmentResult<O, A, R, I> =
    Result<FiniteVectorEnv<O, A, R, I>, RlTrainerDriverError>;

/// Linked environment assembly, not a stable external ABI. Implementations can reuse
/// the finite backend registry and construct independent per-worker interpreters.
/// The Trainer has entered the seed context before invoking this factory.
pub trait RlTrainingEnvironmentFactory<T, O, A, R, I, M = f64>: Send {
    /// # Errors
    /// Returns construction failures; the Trainer remains responsible for queue cleanup.
    fn create(
        &mut self,
        seeds: &RlTrainingSeed<T>,
        control: &mut RlTrainerControl<M>,
    ) -> RlTrainingEnvironmentResult<O, A, R, I>;
}

impl<T, O, A, R, I, M, F> RlTrainingEnvironmentFactory<T, O, A, R, I, M> for F
where
    F: FnMut(
            &RlTrainingSeed<T>,
            &mut RlTrainerControl<M>,
        ) -> RlTrainingEnvironmentResult<O, A, R, I>
        + Send,
{
    fn create(
        &mut self,
        seeds: &RlTrainingSeed<T>,
        control: &mut RlTrainerControl<M>,
    ) -> RlTrainingEnvironmentResult<O, A, R, I> {
        self(seeds, control)
    }
}

/// Owns one runner and its seed collections. Collection, learning and checkpoint
/// restore continue using that runner's original policy, without a duplicate scheduler.
pub struct RlTrainingVessel<
    T,
    O,
    A,
    R,
    I,
    Buffer,
    V = Value,
    P: ?Sized = dyn TrainingRunPolicy<Buffer, V>,
    M = f64,
> {
    pub runner: TrainingVesselRunner<O, A, R, I, Buffer, V, P>,
    pub seeds: TrainingVesselSeeds<T>,
    environment_factory: Box<dyn RlTrainingEnvironmentFactory<T, O, A, R, I, M>>,
    binding: TrainingVesselBinding,
}

impl<T, O, A, R, I, Buffer, V, P: ?Sized, M> RlTrainingVessel<T, O, A, R, I, Buffer, V, P, M> {
    #[must_use]
    pub fn new(
        runner: TrainingVesselRunner<O, A, R, I, Buffer, V, P>,
        seeds: TrainingVesselSeeds<T>,
        environment_factory: Box<dyn RlTrainingEnvironmentFactory<T, O, A, R, I, M>>,
    ) -> Self {
        Self {
            runner,
            seeds,
            environment_factory,
            binding: TrainingVesselBinding::default(),
        }
    }
}

impl<T, O, A, R, I, Buffer, V, P, M> RlTrainerVessel<M>
    for RlTrainingVessel<T, O, A, R, I, Buffer, V, P, M>
where
    T: Clone + Send + Sync + 'static,
    O: Clone + 'static,
    A: Clone + 'static,
    R: Clone + 'static,
    I: Clone + 'static,
    Buffer: 'static,
    V: 'static,
    P: TrainingRunPolicy<Buffer, V> + ?Sized + 'static,
    M: Send + 'static,
{
    type Seed = RlTrainingSeed<T>;
    type Environment = FiniteVectorEnv<O, A, R, I>;

    fn assign_trainer(
        &mut self,
        runtime: &Arc<RlTrainerRuntime<M>>,
    ) -> Result<(), RlTrainerDriverError> {
        let view: Arc<dyn TrainingTrainerView> = runtime.clone();
        self.runner.assign_trainer(&view);
        self.binding.assign_trainer(&view);
        Ok(())
    }

    fn seeds(&mut self, phase: RlTrainerPhase) -> Result<Self::Seed, RlTrainerDriverError> {
        let phase = match phase {
            RlTrainerPhase::Train => TrainingSeedPhase::Train,
            RlTrainerPhase::Validation => TrainingSeedPhase::Val,
            RlTrainerPhase::Test => TrainingSeedPhase::Test,
        };
        self.seeds
            .seed_queue_with_trainer(phase, &self.binding)
            .map(|queue| Arc::new(Mutex::new(queue)))
            .map_err(|error| RlTrainerDriverError::Plugin {
                stage: "seed_factory".into(),
                message: error.to_string(),
            })
    }

    fn environment(
        &mut self,
        seeds: &mut Self::Seed,
        control: &mut RlTrainerControl<M>,
    ) -> Result<Self::Environment, RlTrainerDriverError> {
        self.environment_factory.create(seeds, control)
    }

    fn run(
        &mut self,
        phase: RlTrainerPhase,
        environment: &mut Self::Environment,
        _: &mut RlTrainerControl<M>,
    ) -> Result<(), RlTrainerDriverError> {
        let (stage, result) = match phase {
            RlTrainerPhase::Train => ("train", self.runner.train(environment)),
            RlTrainerPhase::Validation => ("validate", self.runner.validate(environment)),
            RlTrainerPhase::Test => ("test", self.runner.test(environment)),
        };
        result
            .map(|_| ())
            .map_err(|error| RlTrainerDriverError::Plugin {
                stage: stage.into(),
                message: error.to_string(),
            })
    }
}

impl<T, O, A, R, I, Buffer, V, P, M, State> RlCheckpointState<TrainingVesselCheckpoint<State>>
    for RlTrainingVessel<T, O, A, R, I, Buffer, V, P, M>
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
        self.runner.save_checkpoint()
    }

    fn load_checkpoint(&mut self, state: &TrainingVesselCheckpoint<State>) -> Result<(), String> {
        self.runner.load_checkpoint(state)
    }
}
