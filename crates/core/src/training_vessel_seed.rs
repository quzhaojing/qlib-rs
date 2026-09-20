//! Initial-state queue construction for Qlib's default reinforcement-learning training vessel.

use std::sync::Arc;

use strum::{Display, EnumString};
use thiserror::Error;

use crate::{
    DataQueue, DataQueueConfig, DataQueueDataset, DataQueueSampler, RandomDataQueueSampler,
    TrainingVesselBinding, TrainingVesselBindingError,
};

#[derive(Clone, Copy, Debug, Display, EnumString, Eq, PartialEq)]
#[strum(serialize_all = "snake_case")]
pub enum TrainingSeedPhase {
    Train,
    Val,
    Test,
}

#[derive(Clone, Copy, Debug, Display, Eq, PartialEq)]
#[strum(serialize_all = "snake_case")]
pub enum TrainingSeedLogEvent {
    CollectionSize,
    FastDevelopmentSubset,
}

/// Logging seam for seed-collection and fast-development subset diagnostics.
pub trait TrainingVesselSeedLogger: Send {
    /// # Errors
    /// Returns a logger-specific failure.
    fn collection_size(&mut self, phase: TrainingSeedPhase, size: usize) -> Result<(), String>;

    /// # Errors
    /// Returns a logger-specific failure.
    fn fast_development_subset(
        &mut self,
        phase: TrainingSeedPhase,
        original_size: usize,
        subset_size: usize,
    ) -> Result<(), String>;
}

#[derive(Debug, Default)]
pub struct NoopTrainingVesselSeedLogger;

/// Default diagnostic adapter; applications configure delivery with a tracing subscriber.
#[derive(Debug, Default)]
pub struct TracingTrainingVesselSeedLogger;

impl TrainingVesselSeedLogger for TracingTrainingVesselSeedLogger {
    fn collection_size(&mut self, phase: TrainingSeedPhase, size: usize) -> Result<(), String> {
        tracing::info!(%phase, size, "Initial states collection size");
        Ok(())
    }

    fn fast_development_subset(
        &mut self,
        phase: TrainingSeedPhase,
        original_size: usize,
        subset_size: usize,
    ) -> Result<(), String> {
        tracing::info!(%phase, original_size, subset_size, "Fast running in development mode");
        Ok(())
    }
}

impl TrainingVesselSeedLogger for NoopTrainingVesselSeedLogger {
    fn collection_size(&mut self, _phase: TrainingSeedPhase, _size: usize) -> Result<(), String> {
        Ok(())
    }

    fn fast_development_subset(
        &mut self,
        _phase: TrainingSeedPhase,
        _original_size: usize,
        _subset_size: usize,
    ) -> Result<(), String> {
        Ok(())
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum TrainingVesselSeedError {
    #[error("seed iterator for {phase} is not available")]
    SeedIteratorNotAvailable { phase: TrainingSeedPhase },
    #[error("training seed trainer access failed for {phase}: {source}")]
    Trainer {
        phase: TrainingSeedPhase,
        #[source]
        source: TrainingVesselBindingError,
    },
    #[error("training seed logger failed during {event} for {phase}: {message}")]
    Logger {
        phase: TrainingSeedPhase,
        event: TrainingSeedLogEvent,
        message: String,
    },
    #[error("fast-development sampler failed for {phase}: {message}")]
    Sampler {
        phase: TrainingSeedPhase,
        message: String,
    },
    #[error("fast-development sampler returned {actual} indices for {phase}; expected {expected}")]
    InvalidPermutationLength {
        phase: TrainingSeedPhase,
        expected: usize,
        actual: usize,
    },
    #[error(
        "fast-development sampler returned out-of-range index {index} for {phase} collection length {len}"
    )]
    InvalidPermutationIndex {
        phase: TrainingSeedPhase,
        index: usize,
        len: usize,
    },
    #[error("fast-development sampler returned duplicate index {index} for {phase}")]
    DuplicatePermutationIndex {
        phase: TrainingSeedPhase,
        index: usize,
    },
}

/// Owns train/validation/test seed collections and constructs fresh ephemeral queues.
pub struct TrainingVesselSeeds<T> {
    train: Option<Arc<Vec<T>>>,
    validation: Option<Arc<Vec<T>>>,
    test: Option<Arc<Vec<T>>>,
    fast_dev_run: Option<i64>,
    subset_sampler: Box<dyn DataQueueSampler>,
    logger: Box<dyn TrainingVesselSeedLogger>,
}

impl<T> TrainingVesselSeeds<T>
where
    T: Clone + Send + Sync + 'static,
{
    #[must_use]
    pub fn new(
        train: Option<Arc<Vec<T>>>,
        validation: Option<Arc<Vec<T>>>,
        test: Option<Arc<Vec<T>>>,
        fast_dev_run: Option<i64>,
    ) -> Self {
        Self::with_plugins(
            train,
            validation,
            test,
            fast_dev_run,
            Box::new(RandomDataQueueSampler),
            Box::new(TracingTrainingVesselSeedLogger),
        )
    }

    #[must_use]
    pub fn with_plugins(
        train: Option<Arc<Vec<T>>>,
        validation: Option<Arc<Vec<T>>>,
        test: Option<Arc<Vec<T>>>,
        fast_dev_run: Option<i64>,
        subset_sampler: Box<dyn DataQueueSampler>,
        logger: Box<dyn TrainingVesselSeedLogger>,
    ) -> Self {
        Self {
            train,
            validation,
            test,
            fast_dev_run,
            subset_sampler,
            logger,
        }
    }

    #[must_use]
    pub const fn fast_dev_run(&self) -> Option<i64> {
        self.fast_dev_run
    }

    pub const fn set_fast_dev_run(&mut self, size: Option<i64>) {
        self.fast_dev_run = size;
    }

    /// # Errors
    /// Returns missing collection, logger, or subset-sampler failures.
    pub fn train_seed_queue(&mut self) -> Result<DataQueue<T>, TrainingVesselSeedError> {
        self.seed_queue(TrainingSeedPhase::Train)
    }

    /// # Errors
    /// Returns missing collection, logger, or subset-sampler failures.
    pub fn validation_seed_queue(&mut self) -> Result<DataQueue<T>, TrainingVesselSeedError> {
        self.seed_queue(TrainingSeedPhase::Val)
    }

    /// # Errors
    /// Returns missing collection, logger, or subset-sampler failures.
    pub fn test_seed_queue(&mut self) -> Result<DataQueue<T>, TrainingVesselSeedError> {
        self.seed_queue(TrainingSeedPhase::Test)
    }

    fn seed_queue(
        &mut self,
        phase: TrainingSeedPhase,
    ) -> Result<DataQueue<T>, TrainingVesselSeedError> {
        let size = self.fast_dev_run;
        self.seed_queue_with_size(phase, || Ok(size))
    }

    /// Read the current Trainer setting after collection validation and logging,
    /// matching the default vessel's per-phase access rather than caching at bind time.
    /// The standalone configuration and trainer lifetime are unchanged.
    ///
    /// # Errors
    /// Returns missing collection, logging, live trainer access or subset failures
    /// in source order, before activating any queue producer.
    pub fn seed_queue_with_trainer(
        &mut self,
        phase: TrainingSeedPhase,
        trainer: &TrainingVesselBinding,
    ) -> Result<DataQueue<T>, TrainingVesselSeedError> {
        self.seed_queue_with_size(phase, || {
            trainer
                .fast_dev_run()
                .map_err(|source| TrainingVesselSeedError::Trainer { phase, source })
        })
    }

    fn seed_queue_with_size(
        &mut self,
        phase: TrainingSeedPhase,
        size: impl FnOnce() -> Result<Option<i64>, TrainingVesselSeedError>,
    ) -> Result<DataQueue<T>, TrainingVesselSeedError> {
        let collection = self
            .collection(phase)
            .ok_or(TrainingVesselSeedError::SeedIteratorNotAvailable { phase })?;
        self.logger
            .collection_size(phase, collection.len())
            .map_err(|message| TrainingVesselSeedError::Logger {
                phase,
                event: TrainingSeedLogEvent::CollectionSize,
                message,
            })?;
        let selected = self.random_subset(phase, collection, size()?)?;
        let repeat = if phase == TrainingSeedPhase::Train {
            -1
        } else {
            1
        };
        let dataset: Arc<dyn DataQueueDataset<T>> = selected;
        Ok(DataQueue::new(
            dataset,
            DataQueueConfig {
                repeat,
                ..DataQueueConfig::default()
            },
        ))
    }

    fn collection(&self, phase: TrainingSeedPhase) -> Option<Arc<Vec<T>>> {
        match phase {
            TrainingSeedPhase::Train => self.train.as_ref(),
            TrainingSeedPhase::Val => self.validation.as_ref(),
            TrainingSeedPhase::Test => self.test.as_ref(),
        }
        .map(Arc::clone)
    }

    fn random_subset(
        &mut self,
        phase: TrainingSeedPhase,
        collection: Arc<Vec<T>>,
        size: Option<i64>,
    ) -> Result<Arc<Vec<T>>, TrainingVesselSeedError> {
        let Some(size) = size else {
            return Ok(collection);
        };
        let permutation = self
            .subset_sampler
            .indices(collection.len(), 0)
            .map_err(|message| TrainingVesselSeedError::Sampler { phase, message })?;
        validate_permutation(phase, collection.len(), &permutation)?;
        let end = subset_end(collection.len(), size);
        let subset: Arc<Vec<T>> = Arc::new(
            permutation[..end]
                .iter()
                .map(|&index| collection[index].clone())
                .collect(),
        );
        self.logger
            .fast_development_subset(phase, collection.len(), subset.len())
            .map_err(|message| TrainingVesselSeedError::Logger {
                phase,
                event: TrainingSeedLogEvent::FastDevelopmentSubset,
                message,
            })?;
        Ok(subset)
    }
}

fn subset_end(len: usize, size: i64) -> usize {
    if size >= 0 {
        usize::try_from(size).unwrap_or(usize::MAX).min(len)
    } else {
        len.saturating_sub(usize::try_from(size.unsigned_abs()).unwrap_or(usize::MAX))
    }
}

fn validate_permutation(
    phase: TrainingSeedPhase,
    len: usize,
    permutation: &[usize],
) -> Result<(), TrainingVesselSeedError> {
    if permutation.len() != len {
        return Err(TrainingVesselSeedError::InvalidPermutationLength {
            phase,
            expected: len,
            actual: permutation.len(),
        });
    }
    let mut seen = vec![false; len];
    for &index in permutation {
        if index >= len {
            return Err(TrainingVesselSeedError::InvalidPermutationIndex { phase, index, len });
        }
        if seen[index] {
            return Err(TrainingVesselSeedError::DuplicatePermutationIndex { phase, index });
        }
        seen[index] = true;
    }
    Ok(())
}

#[cfg(test)]
#[path = "../tests/support/training_vessel_seed.rs"]
mod tests;
