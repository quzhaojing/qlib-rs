//! Typed native policy extraction after complete checkpoint decoding.
//!
//! Positional Bincode files do not identify their own schema. Configure the
//! projection/codec explicitly; never guess from a suffix or decode only a prefix.

use std::{marker::PhantomData, path::Path};
use thiserror::Error;

use crate::{
    FileRlCheckpointStorage, RlCheckpointField, RlCheckpointFileCodec, RlTrainerCheckpoint,
    TrainingVesselCheckpoint,
};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PolicyCheckpointError {
    #[error("policy checkpoint read failed: {0}")]
    Read(String),
    #[error("policy checkpoint extraction failed: {0}")]
    Extract(String),
}

/// A model-specific, complete decoded document can own arbitrary non-`Clone` state.
pub trait PolicyCheckpointProjection<Document> {
    type Policy;

    /// # Errors
    /// Returns an envelope/schema failure, after the entire document was decoded.
    fn extract(&mut self, document: Document) -> Result<Self::Policy, String>;
}

/// The configured file contains a policy state directly, without a Trainer wrapper.
pub struct DirectPolicyCheckpoint;

impl<Policy> PolicyCheckpointProjection<Policy> for DirectPolicyCheckpoint {
    type Policy = Policy;

    fn extract(&mut self, document: Policy) -> Result<Policy, String> {
        Ok(document)
    }
}

/// The configured file contains the existing complete native Trainer document.
pub struct TrainerPolicyCheckpoint;

impl<Policy, Callback, Logger, Metric>
    PolicyCheckpointProjection<
        RlTrainerCheckpoint<TrainingVesselCheckpoint<Policy>, Callback, Logger, Metric>,
    > for TrainerPolicyCheckpoint
{
    type Policy = Policy;

    fn extract(
        &mut self,
        document: RlTrainerCheckpoint<TrainingVesselCheckpoint<Policy>, Callback, Logger, Metric>,
    ) -> Result<Policy, String> {
        match document.vessel {
            RlCheckpointField::Present(vessel) => Ok(vessel.policy),
            RlCheckpointField::Missing => Err("checkpoint field is missing: vessel".into()),
        }
    }
}

/// Logical file-reader plugin. For native tensor DTOs, the selected model adapter
/// owns CPU materialization and validation; this boundary does not remap devices.
pub trait PolicyCheckpointReader<Policy> {
    /// # Errors
    /// Returns read/decode or extraction failure without touching any live policy.
    fn read_policy(&mut self, path: &Path) -> Result<Policy, PolicyCheckpointError>;
}

/// Reuses the existing file/codec boundary, then projects the complete decoded state.
/// The document type must match the configured codec; no format auto-detection occurs.
pub struct PolicyCheckpointFile<Codec, Projection, Document> {
    pub storage: FileRlCheckpointStorage<Codec>,
    pub projection: Projection,
    document: PhantomData<fn() -> Document>,
}

impl<Codec, Projection, Document> PolicyCheckpointFile<Codec, Projection, Document> {
    #[must_use]
    pub const fn new(codec: Codec, projection: Projection) -> Self {
        Self {
            storage: FileRlCheckpointStorage::new(codec),
            projection,
            document: PhantomData,
        }
    }
}

impl<Codec, Projection, Document> PolicyCheckpointReader<Projection::Policy>
    for PolicyCheckpointFile<Codec, Projection, Document>
where
    Codec: RlCheckpointFileCodec<Document>,
    Projection: PolicyCheckpointProjection<Document>,
{
    fn read_policy(&mut self, path: &Path) -> Result<Projection::Policy, PolicyCheckpointError> {
        self.storage
            .load(path)
            .map_err(PolicyCheckpointError::Read)
            .and_then(|document| {
                self.projection
                    .extract(document)
                    .map_err(PolicyCheckpointError::Extract)
            })
    }
}

#[cfg(test)]
#[path = "../tests/support/rl_policy_checkpoint.rs"]
mod tests;
