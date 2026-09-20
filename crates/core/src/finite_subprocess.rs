//! Tokio process transport and worker loop for finite environments.

use std::{
    any::Any,
    ffi::OsString,
    future::Future,
    io,
    panic::{AssertUnwindSafe, catch_unwind},
    path::{Path, PathBuf},
    pin::Pin,
    process::{ExitStatus, Stdio},
    time::Duration,
};

use bytes::Bytes;
use futures_util::{SinkExt, StreamExt};
use serde::{Serialize, de::DeserializeOwned};
use shmem::MappedObservationError;
use thiserror::Error;
use tokio::{
    io::{AsyncRead, AsyncWrite},
    process::{Child, ChildStdin, ChildStdout, Command},
    time::timeout,
};
use tokio_util::codec::{FramedRead, FramedWrite, LengthDelimitedCodec};

use crate::{
    BoxedFiniteSubprocessWorker, FiniteSubprocessCommand, FiniteSubprocessFailure,
    FiniteSubprocessFailureKind, FiniteSubprocessReply, FiniteSubprocessReplyKind,
    FiniteSubprocessRequest, FiniteSubprocessResponse, FiniteSubprocessWireError,
    FiniteSubprocessWorker, FiniteSubprocessWorkerFactory, decode_finite_subprocess_request,
    decode_finite_subprocess_response, encode_finite_subprocess_request,
    encode_finite_subprocess_response, validate_finite_subprocess_response,
};

impl FiniteSubprocessWorker for FiniteSubprocessTransport {
    fn frame_limit(&self) -> usize {
        FiniteSubprocessTransport::frame_limit(self)
    }

    fn send_frame(
        &mut self,
        frame: Vec<u8>,
    ) -> Pin<Box<dyn Future<Output = Result<(), FiniteSubprocessRuntimeError>> + Send + '_>> {
        Box::pin(async move { FiniteSubprocessTransport::send_frame(self, frame).await })
    }

    fn receive_frame(
        &mut self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<u8>, FiniteSubprocessRuntimeError>> + Send + '_>>
    {
        Box::pin(async move { FiniteSubprocessTransport::receive_frame(self).await })
    }

    fn close_input(
        &mut self,
    ) -> Pin<Box<dyn Future<Output = Result<(), FiniteSubprocessRuntimeError>> + Send + '_>> {
        Box::pin(async move { FiniteSubprocessTransport::close_input(self).await })
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
        Box::pin(async move { FiniteSubprocessTransport::wait_for_exit(self).await })
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
        Box::pin(async move { FiniteSubprocessTransport::terminate(self).await })
    }
}

/// Production worker factory backed by Tokio child-process transports.
pub struct FiniteSubprocessProgramFactory {
    programs: Vec<FiniteSubprocessProgram>,
    frame_limit: usize,
    operation_timeout: Duration,
}

impl FiniteSubprocessProgramFactory {
    #[must_use]
    pub const fn new(
        programs: Vec<FiniteSubprocessProgram>,
        frame_limit: usize,
        operation_timeout: Duration,
    ) -> Self {
        Self {
            programs,
            frame_limit,
            operation_timeout,
        }
    }
}

impl FiniteSubprocessWorkerFactory for FiniteSubprocessProgramFactory {
    fn create(
        &mut self,
        environment_id: usize,
    ) -> Result<BoxedFiniteSubprocessWorker, FiniteSubprocessRuntimeError> {
        let program = self.programs.get(environment_id).ok_or(
            FiniteSubprocessRuntimeError::MissingProgram {
                id: environment_id,
                program_count: self.programs.len(),
            },
        )?;
        FiniteSubprocessTransport::spawn(program, self.frame_limit, self.operation_timeout)
            .map(|transport| Box::new(transport) as BoxedFiniteSubprocessWorker)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FiniteSubprocessProgram {
    executable: PathBuf,
    arguments: Vec<OsString>,
    environment: Vec<(OsString, OsString)>,
    current_directory: Option<PathBuf>,
}

impl FiniteSubprocessProgram {
    #[must_use]
    pub fn new(executable: impl Into<PathBuf>) -> Self {
        Self {
            executable: executable.into(),
            arguments: Vec::new(),
            environment: Vec::new(),
            current_directory: None,
        }
    }

    #[must_use]
    pub fn arg(mut self, argument: impl Into<OsString>) -> Self {
        self.arguments.push(argument.into());
        self
    }

    #[must_use]
    pub fn args<I, S>(mut self, arguments: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<OsString>,
    {
        self.arguments.extend(arguments.into_iter().map(Into::into));
        self
    }

    #[must_use]
    pub fn env(mut self, key: impl Into<OsString>, value: impl Into<OsString>) -> Self {
        self.environment.push((key.into(), value.into()));
        self
    }

    #[must_use]
    pub fn envs<I, K, V>(mut self, environment: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<OsString>,
        V: Into<OsString>,
    {
        self.environment.extend(
            environment
                .into_iter()
                .map(|(key, value)| (key.into(), value.into())),
        );
        self
    }

    #[must_use]
    pub fn current_dir(mut self, directory: impl Into<PathBuf>) -> Self {
        self.current_directory = Some(directory.into());
        self
    }

    #[must_use]
    pub fn executable(&self) -> &Path {
        &self.executable
    }

    #[must_use]
    pub fn arguments(&self) -> &[OsString] {
        &self.arguments
    }

    #[must_use]
    pub fn environment(&self) -> &[(OsString, OsString)] {
        &self.environment
    }

    #[must_use]
    pub fn current_directory(&self) -> Option<&Path> {
        self.current_directory.as_deref()
    }

    fn command(&self) -> Command {
        let mut command = Command::new(&self.executable);
        command
            .args(&self.arguments)
            .envs(self.environment.iter().cloned())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .kill_on_drop(true);
        if let Some(directory) = &self.current_directory {
            command.current_dir(directory);
        }
        command
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FiniteSubprocessOperation {
    Write,
    Read,
    CloseInput,
    Wait,
    Kill,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FiniteSubprocessExitStatus {
    success: bool,
    code: Option<i32>,
}

impl FiniteSubprocessExitStatus {
    #[must_use]
    pub const fn new(success: bool, code: Option<i32>) -> Self {
        Self { success, code }
    }

    #[must_use]
    pub const fn success(self) -> bool {
        self.success
    }

    #[must_use]
    pub const fn code(self) -> Option<i32> {
        self.code
    }
}

impl From<ExitStatus> for FiniteSubprocessExitStatus {
    fn from(status: ExitStatus) -> Self {
        Self::new(status.success(), status.code())
    }
}

impl std::fmt::Display for FiniteSubprocessOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Write => formatter.write_str("write"),
            Self::Read => formatter.write_str("read"),
            Self::CloseInput => formatter.write_str("close-input"),
            Self::Wait => formatter.write_str("wait"),
            Self::Kill => formatter.write_str("kill"),
        }
    }
}

#[derive(Debug, Error)]
pub enum FiniteSubprocessRuntimeError {
    #[error("finite subprocess program id {id} is outside 0..{program_count}")]
    MissingProgram { id: usize, program_count: usize },
    #[error("failed to spawn finite subprocess: {0}")]
    Spawn(#[source] io::Error),
    #[error("finite subprocess {operation} timed out after {duration:?}")]
    Timeout {
        operation: FiniteSubprocessOperation,
        duration: Duration,
    },
    #[error("failed to write finite subprocess frame: {0}")]
    Write(#[source] io::Error),
    #[error("failed to read finite subprocess frame: {0}")]
    Read(#[source] io::Error),
    #[error("finite subprocess closed its response stream")]
    Eof,
    #[error("failed to close finite subprocess input: {0}")]
    CloseInput(#[source] io::Error),
    #[error("failed to wait for finite subprocess: {0}")]
    Wait(#[source] io::Error),
    #[error("failed to kill finite subprocess: {0}")]
    Kill(#[source] io::Error),
    #[error(transparent)]
    Wire(#[from] FiniteSubprocessWireError),
    #[error(transparent)]
    SharedMemory(#[from] MappedObservationError),
    #[error("failed to decode a finite shared-memory observation: {0}")]
    SharedObservationDecode(#[source] Box<bincode::ErrorKind>),
    #[error("finite shared-memory response unexpectedly carried an inline observation")]
    InlineSharedObservation,
    #[error(
        "finite shared-memory observation id {actual} does not match control response id {expected}"
    )]
    SharedObservationIdMismatch { expected: u64, actual: u64 },
}

pub struct FiniteSubprocessTransport {
    child: Child,
    requests: FramedWrite<ChildStdin, LengthDelimitedCodec>,
    responses: FramedRead<ChildStdout, LengthDelimitedCodec>,
    frame_limit: usize,
    operation_timeout: Duration,
}

impl FiniteSubprocessTransport {
    /// # Panics
    /// Panics only if Tokio violates the `Stdio::piped` post-spawn invariant by omitting a pipe.
    ///
    /// # Errors
    /// Returns a process spawn failure.
    pub fn spawn(
        program: &FiniteSubprocessProgram,
        frame_limit: usize,
        operation_timeout: Duration,
    ) -> Result<Self, FiniteSubprocessRuntimeError> {
        let mut child = program
            .command()
            .spawn()
            .map_err(FiniteSubprocessRuntimeError::Spawn)?;
        let stdin = child
            .stdin
            .take()
            .expect("a command configured with piped stdin owns stdin");
        let stdout = child
            .stdout
            .take()
            .expect("a command configured with piped stdout owns stdout");
        Ok(Self {
            child,
            requests: FramedWrite::new(stdin, length_codec(frame_limit)),
            responses: FramedRead::new(stdout, length_codec(frame_limit)),
            frame_limit,
            operation_timeout,
        })
    }

    #[must_use]
    pub fn process_id(&self) -> Option<u32> {
        self.child.id()
    }

    #[must_use]
    pub const fn frame_limit(&self) -> usize {
        self.frame_limit
    }

    #[must_use]
    pub const fn operation_timeout(&self) -> Duration {
        self.operation_timeout
    }

    /// # Errors
    /// Returns wire encoding, frame write, or timeout failures.
    pub async fn send<Action, Payload>(
        &mut self,
        request: &FiniteSubprocessRequest<Action, Payload>,
    ) -> Result<(), FiniteSubprocessRuntimeError>
    where
        Action: Serialize,
        Payload: Serialize,
    {
        let frame = encode_finite_subprocess_request(request, self.frame_limit)?;
        self.send_frame(frame).await
    }

    /// Sends one already encoded protocol frame.
    ///
    /// # Errors
    /// Returns frame write or timeout failures.
    pub async fn send_frame(&mut self, frame: Vec<u8>) -> Result<(), FiniteSubprocessRuntimeError> {
        timeout(
            self.operation_timeout,
            self.requests.send(Bytes::from(frame)),
        )
        .await
        .map_err(|_| self.timeout(FiniteSubprocessOperation::Write))?
        .map_err(FiniteSubprocessRuntimeError::Write)
    }

    /// # Errors
    /// Returns timeout, EOF, frame read/decode, version, correlation, or reply-kind failures.
    pub async fn receive<Action, Observation, Reward, Info, Payload>(
        &mut self,
        request: &FiniteSubprocessRequest<Action, Payload>,
    ) -> Result<
        FiniteSubprocessResponse<Observation, Reward, Info, Payload>,
        FiniteSubprocessRuntimeError,
    >
    where
        Observation: DeserializeOwned,
        Reward: DeserializeOwned,
        Info: DeserializeOwned,
        Payload: DeserializeOwned,
    {
        let item = self.receive_frame().await?;
        let response = decode_finite_subprocess_response(&item, self.frame_limit)?;
        validate_finite_subprocess_response(request, &response)?;
        Ok(response)
    }

    /// Receives one encoded protocol frame.
    ///
    /// # Errors
    /// Returns timeout, EOF, or frame-read failures.
    pub async fn receive_frame(&mut self) -> Result<Vec<u8>, FiniteSubprocessRuntimeError> {
        timeout(self.operation_timeout, self.responses.next())
            .await
            .map_err(|_| self.timeout(FiniteSubprocessOperation::Read))?
            .ok_or(FiniteSubprocessRuntimeError::Eof)?
            .map(|bytes| bytes.to_vec())
            .map_err(FiniteSubprocessRuntimeError::Read)
    }

    /// Sends one request and receives its reply unless the command is one-way.
    ///
    /// # Errors
    /// Returns any send or receive failure.
    pub async fn request<Action, Observation, Reward, Info, Payload>(
        &mut self,
        request: &FiniteSubprocessRequest<Action, Payload>,
    ) -> Result<
        Option<FiniteSubprocessResponse<Observation, Reward, Info, Payload>>,
        FiniteSubprocessRuntimeError,
    >
    where
        Action: Serialize,
        Observation: DeserializeOwned,
        Reward: DeserializeOwned,
        Info: DeserializeOwned,
        Payload: Serialize + DeserializeOwned,
    {
        self.send(request).await?;
        if request.command.expected_reply().is_none() {
            return Ok(None);
        }
        self.receive(request).await.map(Some)
    }

    /// Closes stdin, allowing a cooperative child to terminate.
    ///
    /// # Errors
    /// Returns close or timeout failures.
    pub async fn close_input(&mut self) -> Result<(), FiniteSubprocessRuntimeError> {
        timeout(self.operation_timeout, self.requests.close())
            .await
            .map_err(|_| self.timeout(FiniteSubprocessOperation::CloseInput))?
            .map_err(FiniteSubprocessRuntimeError::CloseInput)
    }

    /// Waits for a cooperative exit. On timeout, kills and reaps the child before returning the
    /// timeout error.
    ///
    /// # Errors
    /// Returns wait, timeout, or forced-cleanup failures.
    pub async fn wait_for_exit(
        &mut self,
    ) -> Result<FiniteSubprocessExitStatus, FiniteSubprocessRuntimeError> {
        wait_for_finite_subprocess_exit(&mut self.child, self.operation_timeout).await
    }

    /// Forces the child to exit and waits for resource collection.
    ///
    /// # Errors
    /// Returns kill, kill-timeout, wait, or wait-timeout failures.
    pub async fn terminate(
        &mut self,
    ) -> Result<FiniteSubprocessExitStatus, FiniteSubprocessRuntimeError> {
        terminate_finite_subprocess(&mut self.child, self.operation_timeout).await
    }

    const fn timeout(&self, operation: FiniteSubprocessOperation) -> FiniteSubprocessRuntimeError {
        FiniteSubprocessRuntimeError::Timeout {
            operation,
            duration: self.operation_timeout,
        }
    }
}

pub trait FiniteSubprocessLifecycle: Send {
    fn kill(&mut self) -> Pin<Box<dyn Future<Output = io::Result<()>> + Send + '_>>;

    fn wait(
        &mut self,
    ) -> Pin<Box<dyn Future<Output = io::Result<FiniteSubprocessExitStatus>> + Send + '_>>;
}

impl FiniteSubprocessLifecycle for Child {
    fn kill(&mut self) -> Pin<Box<dyn Future<Output = io::Result<()>> + Send + '_>> {
        Box::pin(async move { Child::kill(self).await })
    }

    fn wait(
        &mut self,
    ) -> Pin<Box<dyn Future<Output = io::Result<FiniteSubprocessExitStatus>> + Send + '_>> {
        Box::pin(async move { Child::wait(self).await.map(Into::into) })
    }
}

/// Waits for a lifecycle plugin and forces cleanup after a timeout.
///
/// # Errors
/// Returns wait, timeout, kill, or reap failures.
pub async fn wait_for_finite_subprocess_exit(
    child: &mut dyn FiniteSubprocessLifecycle,
    operation_timeout: Duration,
) -> Result<FiniteSubprocessExitStatus, FiniteSubprocessRuntimeError> {
    if let Ok(result) = timeout(operation_timeout, child.wait()).await {
        return result.map_err(FiniteSubprocessRuntimeError::Wait);
    }
    terminate_finite_subprocess(child, operation_timeout).await?;
    Err(timeout_error(
        FiniteSubprocessOperation::Wait,
        operation_timeout,
    ))
}

/// Kills and reaps a lifecycle plugin with separate bounded operations.
///
/// # Errors
/// Returns kill, kill-timeout, wait, or wait-timeout failures.
pub async fn terminate_finite_subprocess(
    child: &mut dyn FiniteSubprocessLifecycle,
    operation_timeout: Duration,
) -> Result<FiniteSubprocessExitStatus, FiniteSubprocessRuntimeError> {
    timeout(operation_timeout, child.kill())
        .await
        .map_err(|_| timeout_error(FiniteSubprocessOperation::Kill, operation_timeout))?
        .map_err(FiniteSubprocessRuntimeError::Kill)?;
    timeout(operation_timeout, child.wait())
        .await
        .map_err(|_| timeout_error(FiniteSubprocessOperation::Wait, operation_timeout))?
        .map_err(FiniteSubprocessRuntimeError::Wait)
}

const fn timeout_error(
    operation: FiniteSubprocessOperation,
    duration: Duration,
) -> FiniteSubprocessRuntimeError {
    FiniteSubprocessRuntimeError::Timeout {
        operation,
        duration,
    }
}

pub trait FiniteSubprocessCommandHandler<Action, Observation, Reward, Info, Payload>: Send {
    /// # Errors
    /// Returns a structured environment or protocol failure that will be sent to the parent.
    fn handle(
        &mut self,
        command: FiniteSubprocessCommand<Action, Payload>,
    ) -> Result<
        Option<FiniteSubprocessReply<Observation, Reward, Info, Payload>>,
        FiniteSubprocessFailure,
    >;
}

impl<Action, Observation, Reward, Info, Payload, Handler>
    FiniteSubprocessCommandHandler<Action, Observation, Reward, Info, Payload> for Handler
where
    Handler: FnMut(
            FiniteSubprocessCommand<Action, Payload>,
        ) -> Result<
            Option<FiniteSubprocessReply<Observation, Reward, Info, Payload>>,
            FiniteSubprocessFailure,
        > + Send,
{
    fn handle(
        &mut self,
        command: FiniteSubprocessCommand<Action, Payload>,
    ) -> Result<
        Option<FiniteSubprocessReply<Observation, Reward, Info, Payload>>,
        FiniteSubprocessFailure,
    > {
        self(command)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FiniteSubprocessWorkerExit {
    Eof,
    Closed,
    Failure(FiniteSubprocessFailureKind),
}

/// Runs a sequential worker over arbitrary Tokio byte streams.
///
/// # Errors
/// Returns framing, I/O, serialization, or protocol-validation failures that cannot be correlated
/// with a decoded request. Handler failures and panics are sent as structured responses and
/// returned as normal worker exit states.
pub async fn serve_finite_subprocess<Reader, Writer, Action, Observation, Reward, Info, Payload>(
    reader: Reader,
    writer: Writer,
    handler: &mut dyn FiniteSubprocessCommandHandler<Action, Observation, Reward, Info, Payload>,
    frame_limit: usize,
) -> Result<FiniteSubprocessWorkerExit, FiniteSubprocessRuntimeError>
where
    Reader: AsyncRead + Unpin,
    Writer: AsyncWrite + Unpin,
    Action: DeserializeOwned,
    Observation: Serialize,
    Reward: Serialize,
    Info: Serialize,
    Payload: Serialize + DeserializeOwned,
{
    let mut requests = FramedRead::new(reader, length_codec(frame_limit));
    let mut responses = FramedWrite::new(writer, length_codec(frame_limit));
    loop {
        let Some(item) = requests.next().await else {
            return Ok(FiniteSubprocessWorkerExit::Eof);
        };
        let item = item.map_err(FiniteSubprocessRuntimeError::Read)?;
        let request: FiniteSubprocessRequest<Action, Payload> =
            decode_finite_subprocess_request(&item, frame_limit)?;
        let request_id = request.request_id;
        let command_kind = request.command.kind();
        let expected_reply = request.command.expected_reply();
        let outcome = catch_unwind(AssertUnwindSafe(|| handler.handle(request.command)));
        let (reply, exit) = match outcome {
            Ok(Ok(reply)) => (reply, None),
            Ok(Err(failure)) => {
                let kind = failure.kind;
                (Some(FiniteSubprocessReply::Failure(failure)), Some(kind))
            }
            Err(payload) => {
                let failure = FiniteSubprocessFailure {
                    kind: FiniteSubprocessFailureKind::Panic,
                    message: panic_message(payload.as_ref()),
                };
                (
                    Some(FiniteSubprocessReply::Failure(failure)),
                    Some(FiniteSubprocessFailureKind::Panic),
                )
            }
        };
        let Some(reply) = reply else {
            if expected_reply.is_none() {
                continue;
            }
            let failure = FiniteSubprocessFailure {
                kind: FiniteSubprocessFailureKind::Protocol,
                message: format!("subprocess command {command_kind:?} produced no reply"),
            };
            write_response(
                &mut responses,
                FiniteSubprocessResponse::new(
                    request_id,
                    FiniteSubprocessReply::<Observation, Reward, Info, Payload>::Failure(failure),
                ),
                frame_limit,
            )
            .await?;
            return Ok(FiniteSubprocessWorkerExit::Failure(
                FiniteSubprocessFailureKind::Protocol,
            ));
        };
        let mut response = FiniteSubprocessResponse::new(request_id, reply);
        let actual_reply = response.reply.kind();
        if actual_reply != FiniteSubprocessReplyKind::Failure
            && expected_reply != Some(actual_reply)
        {
            response.reply = FiniteSubprocessReply::Failure(FiniteSubprocessFailure {
                kind: FiniteSubprocessFailureKind::Protocol,
                message: format!(
                    "subprocess command {command_kind:?} expected reply {expected_reply:?}, but received {actual_reply:?}"
                ),
            });
            write_response(&mut responses, response, frame_limit).await?;
            return Ok(FiniteSubprocessWorkerExit::Failure(
                FiniteSubprocessFailureKind::Protocol,
            ));
        }
        write_response(&mut responses, response, frame_limit).await?;
        if let Some(kind) = exit {
            return Ok(FiniteSubprocessWorkerExit::Failure(kind));
        }
        if command_kind == crate::FiniteSubprocessCommandKind::Close {
            return Ok(FiniteSubprocessWorkerExit::Closed);
        }
    }
}

async fn write_response<Writer, Observation, Reward, Info, Payload>(
    responses: &mut FramedWrite<Writer, LengthDelimitedCodec>,
    response: FiniteSubprocessResponse<Observation, Reward, Info, Payload>,
    frame_limit: usize,
) -> Result<(), FiniteSubprocessRuntimeError>
where
    Writer: AsyncWrite + Unpin,
    Observation: Serialize,
    Reward: Serialize,
    Info: Serialize,
    Payload: Serialize,
{
    let frame = encode_finite_subprocess_response(&response, frame_limit)?;
    responses
        .send(Bytes::from(frame))
        .await
        .map_err(FiniteSubprocessRuntimeError::Write)
}

fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "finite subprocess handler panicked with a non-string payload".to_owned()
    }
}

fn length_codec(frame_limit: usize) -> LengthDelimitedCodec {
    LengthDelimitedCodec::builder()
        .max_frame_length(frame_limit)
        .new_codec()
}
