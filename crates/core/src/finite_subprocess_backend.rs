//! Synchronous batched backend over one subprocess transport per finite environment.

use std::{future::Future, io, marker::PhantomData, pin::Pin};

use serde::{Serialize, de::DeserializeOwned};
use thiserror::Error;
use tokio::runtime::{Builder, Runtime};

use crate::{
    EnvironmentPluginError, FiniteBackendStep, FiniteSubprocessCommand, FiniteSubprocessExitStatus,
    FiniteSubprocessFailure, FiniteSubprocessReply, FiniteSubprocessReplyKind,
    FiniteSubprocessRequest, FiniteSubprocessResponse, FiniteSubprocessRuntimeError,
    FiniteSubprocessWireError, FiniteVectorBackend, decode_finite_subprocess_response,
    encode_finite_subprocess_request,
};

type WorkerFuture<'a, T> =
    Pin<Box<dyn Future<Output = Result<T, FiniteSubprocessRuntimeError>> + Send + 'a>>;

pub type BoxedFiniteSubprocessWorker = Box<dyn FiniteSubprocessWorker>;

/// Object-safe transport plugin used by the batched backend.
///
/// The stable boundary contains only qlib-rs protocol DTOs. Tokio remains an implementation
/// detail of the production transport and alternative in-memory or remote transports can be
/// supplied without exposing their concrete I/O types.
pub trait FiniteSubprocessWorker: Send {
    fn frame_limit(&self) -> usize;

    fn send_frame(&mut self, frame: Vec<u8>) -> WorkerFuture<'_, ()>;

    fn receive_frame(&mut self) -> WorkerFuture<'_, Vec<u8>>;

    fn close_input(&mut self) -> WorkerFuture<'_, ()>;

    fn wait_for_exit(&mut self) -> WorkerFuture<'_, FiniteSubprocessExitStatus>;

    fn terminate(&mut self) -> WorkerFuture<'_, FiniteSubprocessExitStatus>;
}

pub trait FiniteSubprocessWorkerFactory: Send {
    /// # Errors
    /// Returns a worker-construction or process-spawn failure.
    fn create(
        &mut self,
        environment_id: usize,
    ) -> Result<BoxedFiniteSubprocessWorker, FiniteSubprocessRuntimeError>;
}

/// Opaque Tokio runtime owned by the synchronous subprocess backend.
pub struct FiniteSubprocessRuntime(Runtime);

pub trait FiniteSubprocessRuntimeFactory: Send {
    /// # Errors
    /// Returns an executor or I/O-driver initialization failure.
    fn create(&mut self) -> Result<FiniteSubprocessRuntime, io::Error>;
}

impl<Factory> FiniteSubprocessRuntimeFactory for Factory
where
    Factory: FnMut() -> Result<FiniteSubprocessRuntime, io::Error> + Send,
{
    fn create(&mut self) -> Result<FiniteSubprocessRuntime, io::Error> {
        self()
    }
}

pub struct TokioFiniteSubprocessRuntimeFactory;

impl FiniteSubprocessRuntimeFactory for TokioFiniteSubprocessRuntimeFactory {
    fn create(&mut self) -> Result<FiniteSubprocessRuntime, io::Error> {
        Builder::new_current_thread()
            .enable_all()
            .build()
            .map(FiniteSubprocessRuntime)
    }
}

impl<Factory> FiniteSubprocessWorkerFactory for Factory
where
    Factory:
        FnMut(usize) -> Result<BoxedFiniteSubprocessWorker, FiniteSubprocessRuntimeError> + Send,
{
    fn create(
        &mut self,
        environment_id: usize,
    ) -> Result<BoxedFiniteSubprocessWorker, FiniteSubprocessRuntimeError> {
        self(environment_id)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FiniteSubprocessBackendOperation {
    Reset,
    Step,
    Close,
}

#[derive(Debug, Error)]
pub enum FiniteSubprocessBackendBuildError {
    #[error("a finite subprocess backend requires at least one environment")]
    Empty,
    #[error("failed to create the finite subprocess Tokio runtime: {0}")]
    Runtime(#[source] io::Error),
    #[error("finite subprocess worker {environment_id} construction failed: {source}")]
    Worker {
        environment_id: usize,
        #[source]
        source: FiniteSubprocessRuntimeError,
    },
}

#[derive(Debug, Error)]
pub enum FiniteSubprocessBackendError {
    #[error("finite subprocess backend is closed")]
    Closed,
    #[error("finite subprocess backend is poisoned by an incomplete batch")]
    Poisoned,
    #[error("environment id {id} is outside 0..{environment_count}")]
    InvalidEnvironmentId { id: usize, environment_count: usize },
    #[error("action batch has {actual} entries but exactly {required} are required")]
    ActionCount { required: usize, actual: usize },
    #[error(
        "finite subprocess {operation:?} send failed for environment {environment_id}: {source}"
    )]
    Send {
        operation: FiniteSubprocessBackendOperation,
        environment_id: usize,
        #[source]
        source: FiniteSubprocessRuntimeError,
    },
    #[error(
        "finite subprocess {operation:?} receive failed for environment {environment_id}: {source}"
    )]
    Receive {
        operation: FiniteSubprocessBackendOperation,
        environment_id: usize,
        #[source]
        source: FiniteSubprocessRuntimeError,
    },
    #[error(
        "finite subprocess {operation:?} failed remotely for environment {environment_id}: {failure:?}"
    )]
    Remote {
        operation: FiniteSubprocessBackendOperation,
        environment_id: usize,
        failure: FiniteSubprocessFailure,
    },
    #[error(
        "finite subprocess {operation:?} returned unexpected {actual:?} reply for environment {environment_id}"
    )]
    UnexpectedReply {
        operation: FiniteSubprocessBackendOperation,
        environment_id: usize,
        actual: FiniteSubprocessReplyKind,
    },
    #[error("failed to close input for finite subprocess environment {environment_id}: {source}")]
    CloseInput {
        environment_id: usize,
        #[source]
        source: FiniteSubprocessRuntimeError,
    },
    #[error("failed to wait for finite subprocess environment {environment_id}: {source}")]
    Wait {
        environment_id: usize,
        #[source]
        source: FiniteSubprocessRuntimeError,
    },
    #[error("failed to terminate finite subprocess environment {environment_id}: {source}")]
    Terminate {
        environment_id: usize,
        #[source]
        source: FiniteSubprocessRuntimeError,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BackendState {
    Open,
    Poisoned,
    Closed,
}

type BackendTypes<Action, Observation, Reward, Info, Payload> =
    fn() -> (Action, Observation, Reward, Info, Payload);

pub struct FiniteSubprocessBackend<Action, Observation, Reward, Info, Payload> {
    runtime: Runtime,
    workers: Vec<BoxedFiniteSubprocessWorker>,
    next_request_id: u64,
    state: BackendState,
    types: PhantomData<BackendTypes<Action, Observation, Reward, Info, Payload>>,
}

impl<Action, Observation, Reward, Info, Payload>
    FiniteSubprocessBackend<Action, Observation, Reward, Info, Payload>
{
    /// Builds a backend from transport plugins. The workers are kept in ascending environment-id
    /// order and one worker always corresponds to one environment.
    ///
    /// # Errors
    /// Returns an empty-pool or Tokio runtime-construction failure.
    pub fn new(
        workers: Vec<BoxedFiniteSubprocessWorker>,
    ) -> Result<Self, FiniteSubprocessBackendBuildError> {
        let mut runtime_factory = TokioFiniteSubprocessRuntimeFactory;
        Self::new_with_runtime_factory(workers, &mut runtime_factory)
    }

    /// Builds a backend with an injectable executor factory.
    ///
    /// # Errors
    /// Returns an empty-pool or runtime-construction failure.
    pub fn new_with_runtime_factory(
        workers: Vec<BoxedFiniteSubprocessWorker>,
        runtime_factory: &mut dyn FiniteSubprocessRuntimeFactory,
    ) -> Result<Self, FiniteSubprocessBackendBuildError> {
        if workers.is_empty() {
            return Err(FiniteSubprocessBackendBuildError::Empty);
        }
        let runtime = runtime_factory
            .create()
            .map_err(FiniteSubprocessBackendBuildError::Runtime)?;
        Ok(Self::with_runtime(runtime.0, workers))
    }

    /// Creates workers through an object-safe plugin factory in ascending id order.
    ///
    /// # Errors
    /// Returns an empty-pool, runtime, or first worker-construction failure.
    pub fn from_factory(
        environment_count: usize,
        factory: &mut dyn FiniteSubprocessWorkerFactory,
    ) -> Result<Self, FiniteSubprocessBackendBuildError> {
        let mut runtime_factory = TokioFiniteSubprocessRuntimeFactory;
        Self::from_factories(environment_count, factory, &mut runtime_factory)
    }

    /// Creates workers and the executor through separate object-safe plugin factories.
    ///
    /// # Errors
    /// Returns an empty-pool, runtime, or first worker-construction failure.
    pub fn from_factories(
        environment_count: usize,
        factory: &mut dyn FiniteSubprocessWorkerFactory,
        runtime_factory: &mut dyn FiniteSubprocessRuntimeFactory,
    ) -> Result<Self, FiniteSubprocessBackendBuildError> {
        if environment_count == 0 {
            return Err(FiniteSubprocessBackendBuildError::Empty);
        }
        let runtime = runtime_factory
            .create()
            .map_err(FiniteSubprocessBackendBuildError::Runtime)?;
        let runtime = runtime.0;
        let runtime_guard = runtime.enter();
        let workers = (0..environment_count)
            .map(|environment_id| {
                factory.create(environment_id).map_err(|source| {
                    FiniteSubprocessBackendBuildError::Worker {
                        environment_id,
                        source,
                    }
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        drop(runtime_guard);
        Ok(Self::with_runtime(runtime, workers))
    }

    fn with_runtime(runtime: Runtime, workers: Vec<BoxedFiniteSubprocessWorker>) -> Self {
        Self {
            runtime,
            workers,
            next_request_id: 0,
            state: BackendState::Open,
            types: PhantomData,
        }
    }

    #[must_use]
    pub fn environment_count(&self) -> usize {
        self.workers.len()
    }

    #[must_use]
    pub const fn is_poisoned(&self) -> bool {
        matches!(self.state, BackendState::Poisoned)
    }

    #[must_use]
    pub const fn is_closed(&self) -> bool {
        matches!(self.state, BackendState::Closed)
    }

    fn ensure_open(&self) -> Result<(), FiniteSubprocessBackendError> {
        match self.state {
            BackendState::Open => Ok(()),
            BackendState::Poisoned => Err(FiniteSubprocessBackendError::Poisoned),
            BackendState::Closed => Err(FiniteSubprocessBackendError::Closed),
        }
    }

    fn validate_ids(&self, environment_ids: &[usize]) -> Result<(), FiniteSubprocessBackendError> {
        for id in environment_ids.iter().copied() {
            if id >= self.workers.len() {
                return Err(FiniteSubprocessBackendError::InvalidEnvironmentId {
                    id,
                    environment_count: self.workers.len(),
                });
            }
        }
        Ok(())
    }

    fn request(
        &mut self,
        command: FiniteSubprocessCommand<Action, Payload>,
    ) -> FiniteSubprocessRequest<Action, Payload> {
        let request = FiniteSubprocessRequest::new(self.next_request_id, command);
        self.next_request_id = self.next_request_id.wrapping_add(1);
        request
    }
}

impl<Action, Observation, Reward, Info, Payload>
    FiniteSubprocessBackend<Action, Observation, Reward, Info, Payload>
where
    Action: Clone + Serialize + Send + Sync,
    Observation: DeserializeOwned + Send,
    Reward: DeserializeOwned + Send,
    Info: DeserializeOwned + Send,
    Payload: Default + Serialize + DeserializeOwned + Send + Sync,
{
    /// Sends every reset before receiving any response, preserving request order and duplicates.
    ///
    /// # Errors
    /// Returns state, id, transport, or structured remote failures.
    pub fn reset_environments(
        &mut self,
        environment_ids: &[usize],
    ) -> Result<Vec<Option<Observation>>, FiniteSubprocessBackendError> {
        self.ensure_open()?;
        self.validate_ids(environment_ids)?;
        let requests = environment_ids
            .iter()
            .map(|_| {
                self.request(FiniteSubprocessCommand::Reset {
                    options: Payload::default(),
                })
            })
            .collect::<Vec<_>>();
        self.send_batch(
            FiniteSubprocessBackendOperation::Reset,
            environment_ids,
            &requests,
        )?;
        let responses = self.receive_batch(
            FiniteSubprocessBackendOperation::Reset,
            environment_ids,
            &requests,
        )?;
        responses
            .into_iter()
            .zip(environment_ids.iter().copied())
            .map(|(response, environment_id)| match response.reply {
                FiniteSubprocessReply::Reset { observation, .. } => Ok(observation),
                FiniteSubprocessReply::Failure(failure) => {
                    self.state = BackendState::Poisoned;
                    Err(FiniteSubprocessBackendError::Remote {
                        operation: FiniteSubprocessBackendOperation::Reset,
                        environment_id,
                        failure,
                    })
                }
                reply => {
                    self.state = BackendState::Poisoned;
                    Err(FiniteSubprocessBackendError::UnexpectedReply {
                        operation: FiniteSubprocessBackendOperation::Reset,
                        environment_id,
                        actual: reply.kind(),
                    })
                }
            })
            .collect()
    }

    /// Sends every action before receiving any response, preserving request order and duplicates.
    ///
    /// # Errors
    /// Returns state, action-count, id, transport, or structured remote failures.
    pub fn step_environments(
        &mut self,
        actions: &[Action],
        environment_ids: &[usize],
    ) -> Result<Vec<FiniteBackendStep<Observation, Reward, Info>>, FiniteSubprocessBackendError>
    {
        self.ensure_open()?;
        if actions.len() != environment_ids.len() {
            return Err(FiniteSubprocessBackendError::ActionCount {
                required: environment_ids.len(),
                actual: actions.len(),
            });
        }
        self.validate_ids(environment_ids)?;
        let requests = actions
            .iter()
            .cloned()
            .map(|action| self.request(FiniteSubprocessCommand::Step { action }))
            .collect::<Vec<_>>();
        self.send_batch(
            FiniteSubprocessBackendOperation::Step,
            environment_ids,
            &requests,
        )?;
        let responses = self.receive_batch(
            FiniteSubprocessBackendOperation::Step,
            environment_ids,
            &requests,
        )?;
        responses
            .into_iter()
            .zip(environment_ids.iter().copied())
            .map(|(response, environment_id)| match response.reply {
                FiniteSubprocessReply::Step { transition } => Ok(transition),
                FiniteSubprocessReply::Failure(failure) => {
                    self.state = BackendState::Poisoned;
                    Err(FiniteSubprocessBackendError::Remote {
                        operation: FiniteSubprocessBackendOperation::Step,
                        environment_id,
                        failure,
                    })
                }
                reply => {
                    self.state = BackendState::Poisoned;
                    Err(FiniteSubprocessBackendError::UnexpectedReply {
                        operation: FiniteSubprocessBackendOperation::Step,
                        environment_id,
                        actual: reply.kind(),
                    })
                }
            })
            .collect()
    }

    fn send_batch(
        &mut self,
        operation: FiniteSubprocessBackendOperation,
        environment_ids: &[usize],
        requests: &[FiniteSubprocessRequest<Action, Payload>],
    ) -> Result<(), FiniteSubprocessBackendError> {
        for (environment_id, request) in environment_ids.iter().copied().zip(requests) {
            let frame_limit = self.workers[environment_id].frame_limit();
            let frame = match encode_finite_subprocess_request(request, frame_limit) {
                Ok(frame) => frame,
                Err(error) => {
                    self.state = BackendState::Poisoned;
                    return Err(FiniteSubprocessBackendError::Send {
                        operation,
                        environment_id,
                        source: FiniteSubprocessRuntimeError::Wire(error),
                    });
                }
            };
            if let Err(source) = self
                .runtime
                .block_on(self.workers[environment_id].send_frame(frame))
            {
                self.state = BackendState::Poisoned;
                return Err(FiniteSubprocessBackendError::Send {
                    operation,
                    environment_id,
                    source,
                });
            }
        }
        Ok(())
    }

    fn receive_batch(
        &mut self,
        operation: FiniteSubprocessBackendOperation,
        environment_ids: &[usize],
        requests: &[FiniteSubprocessRequest<Action, Payload>],
    ) -> Result<
        Vec<FiniteSubprocessResponse<Observation, Reward, Info, Payload>>,
        FiniteSubprocessBackendError,
    > {
        let mut responses = Vec::with_capacity(requests.len());
        for (environment_id, request) in environment_ids.iter().copied().zip(requests) {
            let frame = match self
                .runtime
                .block_on(self.workers[environment_id].receive_frame())
            {
                Ok(frame) => frame,
                Err(source) => {
                    self.state = BackendState::Poisoned;
                    return Err(FiniteSubprocessBackendError::Receive {
                        operation,
                        environment_id,
                        source,
                    });
                }
            };
            let frame_limit = self.workers[environment_id].frame_limit();
            let response = decode_finite_subprocess_response(&frame, frame_limit)
                .and_then(|response| {
                    if response.request_id != request.request_id {
                        return Err(FiniteSubprocessWireError::RequestIdMismatch {
                            expected: request.request_id,
                            actual: response.request_id,
                        });
                    }
                    Ok(response)
                })
                .map_err(FiniteSubprocessRuntimeError::Wire)
                .map_err(|source| FiniteSubprocessBackendError::Receive {
                    operation,
                    environment_id,
                    source,
                });
            match response {
                Ok(response) => responses.push(response),
                Err(error) => {
                    self.state = BackendState::Poisoned;
                    return Err(error);
                }
            }
        }
        Ok(responses)
    }

    /// Cooperatively closes a healthy pool one worker at a time in Tianshou worker order. A
    /// poisoned pool is force-terminated so queued replies cannot be mistaken for close
    /// acknowledgements. Forced cleanup continues across all workers and returns the first error.
    ///
    /// # Errors
    /// Returns closed-state, close exchange, input-close, wait, or forced-termination failures.
    pub fn close(
        &mut self,
    ) -> Result<Vec<FiniteSubprocessExitStatus>, FiniteSubprocessBackendError> {
        match self.state {
            BackendState::Closed => return Err(FiniteSubprocessBackendError::Closed),
            BackendState::Poisoned => return self.terminate_all(),
            BackendState::Open => {}
        }
        let mut statuses = Vec::with_capacity(self.workers.len());
        for environment_id in 0..self.workers.len() {
            statuses.push(self.close_environment(environment_id)?);
        }
        self.state = BackendState::Closed;
        Ok(statuses)
    }

    fn close_environment(
        &mut self,
        environment_id: usize,
    ) -> Result<FiniteSubprocessExitStatus, FiniteSubprocessBackendError> {
        let request = self.request(FiniteSubprocessCommand::Close);
        let frame_limit = self.workers[environment_id].frame_limit();
        let frame = match encode_finite_subprocess_request(&request, frame_limit) {
            Ok(frame) => frame,
            Err(error) => {
                return self.fail_close(FiniteSubprocessBackendError::Send {
                    operation: FiniteSubprocessBackendOperation::Close,
                    environment_id,
                    source: FiniteSubprocessRuntimeError::Wire(error),
                });
            }
        };
        if let Err(source) = self
            .runtime
            .block_on(self.workers[environment_id].send_frame(frame))
        {
            return self.fail_close(FiniteSubprocessBackendError::Send {
                operation: FiniteSubprocessBackendOperation::Close,
                environment_id,
                source,
            });
        }
        let response_frame = match self
            .runtime
            .block_on(self.workers[environment_id].receive_frame())
        {
            Ok(response) => response,
            Err(source) => {
                return self.fail_close(FiniteSubprocessBackendError::Receive {
                    operation: FiniteSubprocessBackendOperation::Close,
                    environment_id,
                    source,
                });
            }
        };
        let response = decode_finite_subprocess_response::<Observation, Reward, Info, Payload>(
            &response_frame,
            frame_limit,
        )
        .and_then(|response| {
            if response.request_id != request.request_id {
                return Err(FiniteSubprocessWireError::RequestIdMismatch {
                    expected: request.request_id,
                    actual: response.request_id,
                });
            }
            Ok(response)
        });
        let response = match response {
            Ok(response) => response,
            Err(error) => {
                return self.fail_close(FiniteSubprocessBackendError::Receive {
                    operation: FiniteSubprocessBackendOperation::Close,
                    environment_id,
                    source: FiniteSubprocessRuntimeError::Wire(error),
                });
            }
        };
        match response.reply {
            FiniteSubprocessReply::Close { .. } => {}
            FiniteSubprocessReply::Failure(failure) => {
                return self.fail_close(FiniteSubprocessBackendError::Remote {
                    operation: FiniteSubprocessBackendOperation::Close,
                    environment_id,
                    failure,
                });
            }
            reply => {
                return self.fail_close(FiniteSubprocessBackendError::UnexpectedReply {
                    operation: FiniteSubprocessBackendOperation::Close,
                    environment_id,
                    actual: reply.kind(),
                });
            }
        }
        if let Err(source) = self
            .runtime
            .block_on(self.workers[environment_id].close_input())
        {
            return self.fail_close(FiniteSubprocessBackendError::CloseInput {
                environment_id,
                source,
            });
        }
        match self
            .runtime
            .block_on(self.workers[environment_id].wait_for_exit())
        {
            Ok(status) => Ok(status),
            Err(source) => self.fail_close(FiniteSubprocessBackendError::Wait {
                environment_id,
                source,
            }),
        }
    }

    fn fail_close<T>(
        &mut self,
        error: FiniteSubprocessBackendError,
    ) -> Result<T, FiniteSubprocessBackendError> {
        self.state = BackendState::Poisoned;
        let _ = self.terminate_all();
        Err(error)
    }

    fn terminate_all(
        &mut self,
    ) -> Result<Vec<FiniteSubprocessExitStatus>, FiniteSubprocessBackendError> {
        let mut statuses = Vec::with_capacity(self.workers.len());
        let mut first_error = None;
        for (environment_id, worker) in self.workers.iter_mut().enumerate() {
            match self.runtime.block_on(worker.terminate()) {
                Ok(status) => statuses.push(status),
                Err(source) => {
                    if first_error.is_none() {
                        first_error = Some(FiniteSubprocessBackendError::Terminate {
                            environment_id,
                            source,
                        });
                    }
                }
            }
        }
        self.state = BackendState::Closed;
        first_error.map_or(Ok(statuses), Err)
    }
}

impl<Action, Observation, Reward, Info, Payload>
    FiniteVectorBackend<Observation, Action, Reward, Info>
    for FiniteSubprocessBackend<Action, Observation, Reward, Info, Payload>
where
    Action: Clone + Serialize + Send + Sync,
    Observation: DeserializeOwned + Send,
    Reward: DeserializeOwned + Send,
    Info: DeserializeOwned + Send,
    Payload: Default + Serialize + DeserializeOwned + Send + Sync,
{
    fn environment_count(&self) -> usize {
        self.environment_count()
    }

    fn reset(
        &mut self,
        environment_ids: &[usize],
    ) -> Result<Vec<Option<Observation>>, EnvironmentPluginError> {
        self.reset_environments(environment_ids)
            .map_err(|error| EnvironmentPluginError::new(error.to_string()))
    }

    fn step(
        &mut self,
        actions: &[Action],
        environment_ids: &[usize],
    ) -> Result<Vec<FiniteBackendStep<Observation, Reward, Info>>, EnvironmentPluginError> {
        self.step_environments(actions, environment_ids)
            .map_err(|error| EnvironmentPluginError::new(error.to_string()))
    }
}
