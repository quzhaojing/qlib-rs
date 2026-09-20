//! Versioned wire protocol for finite subprocess environment workers.

use bincode::{DefaultOptions, Options};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use thiserror::Error;

use crate::FiniteBackendStep;

pub const FINITE_SUBPROCESS_PROTOCOL_VERSION: u16 = 1;
pub const DEFAULT_FINITE_SUBPROCESS_FRAME_LIMIT: usize = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FiniteSubprocessCommandKind {
    Reset,
    Step,
    Close,
    Render,
    Seed,
    GetAttribute,
    SetAttribute,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FiniteSubprocessCommand<Action, Payload> {
    Reset { options: Payload },
    Step { action: Action },
    Close,
    Render { options: Payload },
    Seed { seed: Option<Payload> },
    GetAttribute { key: String },
    SetAttribute { key: String, value: Payload },
}

impl<Action, Payload> FiniteSubprocessCommand<Action, Payload> {
    #[must_use]
    pub const fn kind(&self) -> FiniteSubprocessCommandKind {
        match self {
            Self::Reset { .. } => FiniteSubprocessCommandKind::Reset,
            Self::Step { .. } => FiniteSubprocessCommandKind::Step,
            Self::Close => FiniteSubprocessCommandKind::Close,
            Self::Render { .. } => FiniteSubprocessCommandKind::Render,
            Self::Seed { .. } => FiniteSubprocessCommandKind::Seed,
            Self::GetAttribute { .. } => FiniteSubprocessCommandKind::GetAttribute,
            Self::SetAttribute { .. } => FiniteSubprocessCommandKind::SetAttribute,
        }
    }

    #[must_use]
    pub const fn expected_reply(&self) -> Option<FiniteSubprocessReplyKind> {
        match self {
            Self::Reset { .. } => Some(FiniteSubprocessReplyKind::Reset),
            Self::Step { .. } => Some(FiniteSubprocessReplyKind::Step),
            Self::Close => Some(FiniteSubprocessReplyKind::Close),
            Self::Render { .. } => Some(FiniteSubprocessReplyKind::Render),
            Self::Seed { .. } => Some(FiniteSubprocessReplyKind::Seed),
            Self::GetAttribute { .. } => Some(FiniteSubprocessReplyKind::Attribute),
            Self::SetAttribute { .. } => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FiniteSubprocessReplyKind {
    Reset,
    Step,
    Close,
    Render,
    Seed,
    Attribute,
    Failure,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FiniteSubprocessFailureKind {
    Environment,
    Protocol,
    Panic,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FiniteSubprocessFailure {
    pub kind: FiniteSubprocessFailureKind,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FiniteSubprocessReply<Observation, Reward, Info, Payload> {
    Reset {
        observation: Option<Observation>,
        info: Option<Payload>,
    },
    Step {
        transition: FiniteBackendStep<Observation, Reward, Info>,
    },
    Close {
        result: Option<Payload>,
    },
    Render {
        result: Option<Payload>,
    },
    Seed {
        result: Option<Payload>,
    },
    Attribute {
        value: Option<Payload>,
    },
    Failure(FiniteSubprocessFailure),
}

impl<Observation, Reward, Info, Payload> FiniteSubprocessReply<Observation, Reward, Info, Payload> {
    #[must_use]
    pub const fn kind(&self) -> FiniteSubprocessReplyKind {
        match self {
            Self::Reset { .. } => FiniteSubprocessReplyKind::Reset,
            Self::Step { .. } => FiniteSubprocessReplyKind::Step,
            Self::Close { .. } => FiniteSubprocessReplyKind::Close,
            Self::Render { .. } => FiniteSubprocessReplyKind::Render,
            Self::Seed { .. } => FiniteSubprocessReplyKind::Seed,
            Self::Attribute { .. } => FiniteSubprocessReplyKind::Attribute,
            Self::Failure(_) => FiniteSubprocessReplyKind::Failure,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FiniteSubprocessRequest<Action, Payload> {
    pub version: u16,
    pub request_id: u64,
    pub command: FiniteSubprocessCommand<Action, Payload>,
}

impl<Action, Payload> FiniteSubprocessRequest<Action, Payload> {
    #[must_use]
    pub const fn new(request_id: u64, command: FiniteSubprocessCommand<Action, Payload>) -> Self {
        Self {
            version: FINITE_SUBPROCESS_PROTOCOL_VERSION,
            request_id,
            command,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FiniteSubprocessResponse<Observation, Reward, Info, Payload> {
    pub version: u16,
    pub request_id: u64,
    pub reply: FiniteSubprocessReply<Observation, Reward, Info, Payload>,
}

impl<Observation, Reward, Info, Payload>
    FiniteSubprocessResponse<Observation, Reward, Info, Payload>
{
    #[must_use]
    pub const fn new(
        request_id: u64,
        reply: FiniteSubprocessReply<Observation, Reward, Info, Payload>,
    ) -> Self {
        Self {
            version: FINITE_SUBPROCESS_PROTOCOL_VERSION,
            request_id,
            reply,
        }
    }
}

#[derive(Debug, Error)]
pub enum FiniteSubprocessWireError {
    #[error("subprocess frame is empty")]
    Empty,
    #[error("subprocess frame has {actual} bytes, exceeding limit {limit}")]
    FrameTooLarge { limit: usize, actual: usize },
    #[error("failed to encode subprocess frame: {0}")]
    Encode(#[source] Box<bincode::ErrorKind>),
    #[error("failed to decode subprocess frame: {0}")]
    Decode(#[source] Box<bincode::ErrorKind>),
    #[error("subprocess protocol version {actual} is unsupported; expected {expected}")]
    UnsupportedVersion { expected: u16, actual: u16 },
    #[error("subprocess response id {actual} does not match request id {expected}")]
    RequestIdMismatch { expected: u64, actual: u64 },
    #[error("subprocess command {command:?} expected reply {expected:?}, but received {actual:?}")]
    UnexpectedReply {
        command: FiniteSubprocessCommandKind,
        expected: Option<FiniteSubprocessReplyKind>,
        actual: FiniteSubprocessReplyKind,
    },
}

/// # Errors
/// Returns unsupported-version, serialization, or frame-size failures.
pub fn encode_finite_subprocess_request<Action, Payload>(
    request: &FiniteSubprocessRequest<Action, Payload>,
    frame_limit: usize,
) -> Result<Vec<u8>, FiniteSubprocessWireError>
where
    Action: Serialize,
    Payload: Serialize,
{
    validate_version(request.version)?;
    encode(request, frame_limit)
}

/// # Errors
/// Returns empty, oversized, malformed, trailing-data, or unsupported-version failures.
pub fn decode_finite_subprocess_request<Action, Payload>(
    frame: &[u8],
    frame_limit: usize,
) -> Result<FiniteSubprocessRequest<Action, Payload>, FiniteSubprocessWireError>
where
    Action: DeserializeOwned,
    Payload: DeserializeOwned,
{
    let request: FiniteSubprocessRequest<Action, Payload> = decode(frame, frame_limit)?;
    validate_version(request.version)?;
    Ok(request)
}

/// # Errors
/// Returns unsupported-version, serialization, or frame-size failures.
pub fn encode_finite_subprocess_response<Observation, Reward, Info, Payload>(
    response: &FiniteSubprocessResponse<Observation, Reward, Info, Payload>,
    frame_limit: usize,
) -> Result<Vec<u8>, FiniteSubprocessWireError>
where
    Observation: Serialize,
    Reward: Serialize,
    Info: Serialize,
    Payload: Serialize,
{
    validate_version(response.version)?;
    encode(response, frame_limit)
}

/// # Errors
/// Returns empty, oversized, malformed, trailing-data, or unsupported-version failures.
pub fn decode_finite_subprocess_response<Observation, Reward, Info, Payload>(
    frame: &[u8],
    frame_limit: usize,
) -> Result<FiniteSubprocessResponse<Observation, Reward, Info, Payload>, FiniteSubprocessWireError>
where
    Observation: DeserializeOwned,
    Reward: DeserializeOwned,
    Info: DeserializeOwned,
    Payload: DeserializeOwned,
{
    let response: FiniteSubprocessResponse<Observation, Reward, Info, Payload> =
        decode(frame, frame_limit)?;
    validate_version(response.version)?;
    Ok(response)
}

/// Validates request/response correlation and the command-specific reply kind.
///
/// A structured failure is valid for every command, including the normally no-reply attribute
/// setter.
///
/// # Errors
/// Returns unsupported versions, request-id mismatch, or an unexpected successful reply kind.
pub fn validate_finite_subprocess_response<Action, Observation, Reward, Info, Payload>(
    request: &FiniteSubprocessRequest<Action, Payload>,
    response: &FiniteSubprocessResponse<Observation, Reward, Info, Payload>,
) -> Result<(), FiniteSubprocessWireError> {
    validate_version(request.version)?;
    validate_version(response.version)?;
    if request.request_id != response.request_id {
        return Err(FiniteSubprocessWireError::RequestIdMismatch {
            expected: request.request_id,
            actual: response.request_id,
        });
    }
    let actual = response.reply.kind();
    if actual == FiniteSubprocessReplyKind::Failure {
        return Ok(());
    }
    let expected = request.command.expected_reply();
    if expected != Some(actual) {
        return Err(FiniteSubprocessWireError::UnexpectedReply {
            command: request.command.kind(),
            expected,
            actual,
        });
    }
    Ok(())
}

fn validate_version(version: u16) -> Result<(), FiniteSubprocessWireError> {
    if version != FINITE_SUBPROCESS_PROTOCOL_VERSION {
        return Err(FiniteSubprocessWireError::UnsupportedVersion {
            expected: FINITE_SUBPROCESS_PROTOCOL_VERSION,
            actual: version,
        });
    }
    Ok(())
}

fn encode<T: Serialize>(
    value: &T,
    frame_limit: usize,
) -> Result<Vec<u8>, FiniteSubprocessWireError> {
    let frame = wire_options()
        .serialize(value)
        .map_err(FiniteSubprocessWireError::Encode)?;
    validate_frame_size(frame.len(), frame_limit)?;
    Ok(frame)
}

fn decode<T: DeserializeOwned>(
    frame: &[u8],
    frame_limit: usize,
) -> Result<T, FiniteSubprocessWireError> {
    if frame.is_empty() {
        return Err(FiniteSubprocessWireError::Empty);
    }
    validate_frame_size(frame.len(), frame_limit)?;
    wire_options()
        .deserialize(frame)
        .map_err(FiniteSubprocessWireError::Decode)
}

fn validate_frame_size(actual: usize, limit: usize) -> Result<(), FiniteSubprocessWireError> {
    if actual > limit {
        return Err(FiniteSubprocessWireError::FrameTooLarge { limit, actual });
    }
    Ok(())
}

fn wire_options() -> impl Options {
    DefaultOptions::new()
        .with_fixint_encoding()
        .with_little_endian()
        .reject_trailing_bytes()
}
