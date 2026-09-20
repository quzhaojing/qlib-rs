//! Shared-memory observation adapter over the finite subprocess control protocol.

use std::{future::Future, marker::PhantomData, pin::Pin, time::Duration};

use bincode::{DefaultOptions, Options};
use serde::{Serialize, de::DeserializeOwned};
use shmem::MappedObservationRegion;
use tempfile::TempDir;
use thiserror::Error;

use crate::{
    BoxedFiniteSubprocessWorker, EnvironmentPluginError, FiniteBackendStep,
    FiniteSubprocessBackend, FiniteSubprocessBackendBuildError, FiniteSubprocessBackendError,
    FiniteSubprocessExitStatus, FiniteSubprocessProgram, FiniteSubprocessReply,
    FiniteSubprocessResponse, FiniteSubprocessRuntimeError, FiniteSubprocessTransport,
    FiniteSubprocessWorker, FiniteVectorBackend, decode_finite_subprocess_response,
    encode_finite_subprocess_response,
};

pub const FINITE_SHMEM_PATH_ENV: &str = "QLIB_FINITE_SHMEM_PATH";
pub const FINITE_SHMEM_CAPACITY_ENV: &str = "QLIB_FINITE_SHMEM_CAPACITY";

#[derive(Debug, Error)]
pub enum FiniteShmemBuildError {
    #[error("failed to create finite shared-memory owner directory: {0}")]
    Directory(#[source] std::io::Error),
    #[error(
        "failed to create finite shared-memory region for environment {environment_id}: {source}"
    )]
    Region {
        environment_id: usize,
        #[source]
        source: shmem::MappedObservationError,
    },
    #[error("failed to spawn finite shared-memory worker {environment_id}: {source}")]
    Worker {
        environment_id: usize,
        #[source]
        source: FiniteSubprocessRuntimeError,
    },
    #[error("failed to construct finite shared-memory backend: {0}")]
    Backend(#[source] FiniteSubprocessBackendBuildError),
}

type ShmemTypes<Observation, Reward, Info, Payload> = fn() -> (Observation, Reward, Info, Payload);

/// Rehydrates observations from a fixed shared mapping while delegating the complete control and
/// lifecycle plane to an existing subprocess worker.
pub struct FiniteShmemWorker<Observation, Reward, Info, Payload> {
    inner: BoxedFiniteSubprocessWorker,
    observations: MappedObservationRegion,
    types: PhantomData<ShmemTypes<Observation, Reward, Info, Payload>>,
}

impl<Observation, Reward, Info, Payload> FiniteShmemWorker<Observation, Reward, Info, Payload> {
    #[must_use]
    pub const fn new(
        inner: BoxedFiniteSubprocessWorker,
        observations: MappedObservationRegion,
    ) -> Self {
        Self {
            inner,
            observations,
            types: PhantomData,
        }
    }

    #[must_use]
    pub fn observation_path(&self) -> &std::path::Path {
        self.observations.path()
    }
}

impl<Observation, Reward, Info, Payload> FiniteShmemWorker<Observation, Reward, Info, Payload>
where
    Observation: DeserializeOwned + Serialize,
    Reward: DeserializeOwned + Serialize,
    Info: DeserializeOwned + Serialize,
    Payload: DeserializeOwned + Serialize,
{
    fn hydrate(&self, frame: &[u8]) -> Result<Vec<u8>, FiniteSubprocessRuntimeError> {
        let limit = self.inner.frame_limit();
        let response: FiniteSubprocessResponse<(), Reward, Info, Payload> =
            decode_finite_subprocess_response(frame, limit)?;
        let request_id = response.request_id;
        let reply = match response.reply {
            FiniteSubprocessReply::Reset { observation, info } => {
                if observation.is_some() {
                    return Err(FiniteSubprocessRuntimeError::InlineSharedObservation);
                }
                FiniteSubprocessReply::Reset {
                    observation: Some(self.read_observation(request_id)?),
                    info,
                }
            }
            FiniteSubprocessReply::Step { transition } => {
                if transition.observation.is_some() {
                    return Err(FiniteSubprocessRuntimeError::InlineSharedObservation);
                }
                FiniteSubprocessReply::Step {
                    transition: FiniteBackendStep {
                        observation: Some(self.read_observation(request_id)?),
                        reward: transition.reward,
                        done: transition.done,
                        info: transition.info,
                    },
                }
            }
            FiniteSubprocessReply::Close { result } => FiniteSubprocessReply::Close { result },
            FiniteSubprocessReply::Render { result } => FiniteSubprocessReply::Render { result },
            FiniteSubprocessReply::Seed { result } => FiniteSubprocessReply::Seed { result },
            FiniteSubprocessReply::Attribute { value } => {
                FiniteSubprocessReply::Attribute { value }
            }
            FiniteSubprocessReply::Failure(failure) => FiniteSubprocessReply::Failure(failure),
        };
        encode_finite_subprocess_response(&FiniteSubprocessResponse::new(request_id, reply), limit)
            .map_err(Into::into)
    }

    fn read_observation(
        &self,
        request_id: u64,
    ) -> Result<Observation, FiniteSubprocessRuntimeError> {
        let frame = self.observations.read_frame()?;
        if frame.request_id != request_id {
            return Err(FiniteSubprocessRuntimeError::SharedObservationIdMismatch {
                expected: request_id,
                actual: frame.request_id,
            });
        }
        shared_wire_options()
            .deserialize(&frame.payload)
            .map_err(FiniteSubprocessRuntimeError::SharedObservationDecode)
    }
}

impl<Observation, Reward, Info, Payload> FiniteSubprocessWorker
    for FiniteShmemWorker<Observation, Reward, Info, Payload>
where
    Observation: DeserializeOwned + Serialize + Send,
    Reward: DeserializeOwned + Serialize + Send,
    Info: DeserializeOwned + Serialize + Send,
    Payload: DeserializeOwned + Serialize + Send,
{
    fn frame_limit(&self) -> usize {
        self.inner.frame_limit()
    }

    fn send_frame(
        &mut self,
        frame: Vec<u8>,
    ) -> Pin<Box<dyn Future<Output = Result<(), FiniteSubprocessRuntimeError>> + Send + '_>> {
        self.inner.send_frame(frame)
    }

    fn receive_frame(
        &mut self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<u8>, FiniteSubprocessRuntimeError>> + Send + '_>>
    {
        Box::pin(async move {
            let frame = self.inner.receive_frame().await?;
            self.hydrate(&frame)
        })
    }

    fn close_input(
        &mut self,
    ) -> Pin<Box<dyn Future<Output = Result<(), FiniteSubprocessRuntimeError>> + Send + '_>> {
        self.inner.close_input()
    }

    fn wait_for_exit(
        &mut self,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<FiniteSubprocessExitStatus, FiniteSubprocessRuntimeError>>
                + Send
                + '_,
        >,
    > {
        self.inner.wait_for_exit()
    }

    fn terminate(
        &mut self,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<FiniteSubprocessExitStatus, FiniteSubprocessRuntimeError>>
                + Send
                + '_,
        >,
    > {
        self.inner.terminate()
    }
}

fn shared_wire_options() -> impl Options {
    DefaultOptions::new()
        .with_fixint_encoding()
        .with_little_endian()
        .reject_trailing_bytes()
}

pub struct FiniteShmemBackend<Action, Observation, Reward, Info, Payload> {
    inner: FiniteSubprocessBackend<Action, Observation, Reward, Info, Payload>,
    owner_directory: TempDir,
}

impl<Action, Observation, Reward, Info, Payload>
    FiniteShmemBackend<Action, Observation, Reward, Info, Payload>
where
    Action: Clone + Serialize + Send + Sync + 'static,
    Observation: DeserializeOwned + Serialize + Send + 'static,
    Reward: DeserializeOwned + Serialize + Send + 'static,
    Info: DeserializeOwned + Serialize + Send + 'static,
    Payload: Default + DeserializeOwned + Serialize + Send + Sync + 'static,
{
    /// Creates one mapped observation region and one subprocess per program.
    ///
    /// # Errors
    /// Returns owner-directory, region, worker, or backend construction failures.
    pub fn from_programs(
        programs: Vec<FiniteSubprocessProgram>,
        observation_capacity: usize,
        frame_limit: usize,
        operation_timeout: Duration,
    ) -> Result<Self, FiniteShmemBuildError> {
        Self::from_programs_in(
            std::env::temp_dir(),
            programs,
            observation_capacity,
            frame_limit,
            operation_timeout,
        )
    }

    /// Creates the backend beneath an explicit owner-directory root.
    ///
    /// # Errors
    /// Returns owner-directory, region, worker, or backend construction failures.
    pub fn from_programs_in(
        directory_root: impl AsRef<std::path::Path>,
        programs: Vec<FiniteSubprocessProgram>,
        observation_capacity: usize,
        frame_limit: usize,
        operation_timeout: Duration,
    ) -> Result<Self, FiniteShmemBuildError> {
        let owner_directory = tempfile::Builder::new()
            .prefix("qlib-finite-shmem-")
            .tempdir_in(directory_root)
            .map_err(FiniteShmemBuildError::Directory)?;
        let mut workers = Vec::with_capacity(programs.len());
        for (environment_id, program) in programs.into_iter().enumerate() {
            let path = owner_directory
                .path()
                .join(format!("environment-{environment_id}.bin"));
            let observations = MappedObservationRegion::create(&path, observation_capacity)
                .map_err(|source| FiniteShmemBuildError::Region {
                    environment_id,
                    source,
                })?;
            let program = program
                .env(FINITE_SHMEM_PATH_ENV, path.as_os_str())
                .env(FINITE_SHMEM_CAPACITY_ENV, observation_capacity.to_string());
            let transport =
                FiniteSubprocessTransport::spawn(&program, frame_limit, operation_timeout)
                    .map_err(|source| FiniteShmemBuildError::Worker {
                        environment_id,
                        source,
                    })?;
            workers.push(Box::new(
                FiniteShmemWorker::<Observation, Reward, Info, Payload>::new(
                    Box::new(transport),
                    observations,
                ),
            ) as BoxedFiniteSubprocessWorker);
        }
        let inner =
            FiniteSubprocessBackend::new(workers).map_err(FiniteShmemBuildError::Backend)?;
        Ok(Self {
            inner,
            owner_directory,
        })
    }

    #[must_use]
    pub fn owner_directory(&self) -> &std::path::Path {
        self.owner_directory.path()
    }

    /// Cooperatively closes workers before releasing their mapped files.
    ///
    /// # Errors
    /// Returns the underlying subprocess close failure.
    pub fn close(
        &mut self,
    ) -> Result<Vec<FiniteSubprocessExitStatus>, FiniteSubprocessBackendError> {
        self.inner.close()
    }
}

impl<Action, Observation, Reward, Info, Payload>
    FiniteVectorBackend<Observation, Action, Reward, Info>
    for FiniteShmemBackend<Action, Observation, Reward, Info, Payload>
where
    Action: Clone + Serialize + Send + Sync + 'static,
    Observation: DeserializeOwned + Serialize + Send + 'static,
    Reward: DeserializeOwned + Serialize + Send + 'static,
    Info: DeserializeOwned + Serialize + Send + 'static,
    Payload: Default + DeserializeOwned + Serialize + Send + Sync + 'static,
{
    fn environment_count(&self) -> usize {
        self.inner.environment_count()
    }

    fn reset(
        &mut self,
        environment_ids: &[usize],
    ) -> Result<Vec<Option<Observation>>, EnvironmentPluginError> {
        FiniteVectorBackend::reset(&mut self.inner, environment_ids)
    }

    fn step(
        &mut self,
        actions: &[Action],
        environment_ids: &[usize],
    ) -> Result<Vec<FiniteBackendStep<Observation, Reward, Info>>, EnvironmentPluginError> {
        FiniteVectorBackend::step(&mut self.inner, actions, environment_ids)
    }
}
