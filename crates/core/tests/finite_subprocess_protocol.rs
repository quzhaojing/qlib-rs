use std::{path::PathBuf, process::Command};

use domain_core::{
    DEFAULT_FINITE_SUBPROCESS_FRAME_LIMIT, FINITE_SUBPROCESS_PROTOCOL_VERSION, FiniteBackendStep,
    FiniteSubprocessCommand, FiniteSubprocessCommandKind, FiniteSubprocessFailure,
    FiniteSubprocessFailureKind, FiniteSubprocessReply, FiniteSubprocessReplyKind,
    FiniteSubprocessRequest, FiniteSubprocessResponse, FiniteSubprocessWireError,
    decode_finite_subprocess_request, decode_finite_subprocess_response,
    encode_finite_subprocess_request, encode_finite_subprocess_response,
    validate_finite_subprocess_response,
};
use serde::{Deserialize, Serialize, Serializer, ser::Error as _};
use serde_json::{Value, json};

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
enum Payload {
    Integer(i64),
    Text(String),
    List(Vec<Payload>),
    Fields(Vec<(String, Payload)>),
    Refuse,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Info {
    env_id: i32,
}

type Request = FiniteSubprocessRequest<Vec<i32>, Payload>;
type Response = FiniteSubprocessResponse<i64, f64, Info, Payload>;
type Reply = FiniteSubprocessReply<i64, f64, Info, Payload>;

impl Serialize for Payload {
    fn serialize<SerializerType>(
        &self,
        serializer: SerializerType,
    ) -> Result<SerializerType::Ok, SerializerType::Error>
    where
        SerializerType: Serializer,
    {
        match self {
            Self::Integer(value) => {
                serializer.serialize_newtype_variant("Payload", 0, "Integer", value)
            }
            Self::Text(value) => serializer.serialize_newtype_variant("Payload", 1, "Text", value),
            Self::List(value) => serializer.serialize_newtype_variant("Payload", 2, "List", value),
            Self::Fields(value) => {
                serializer.serialize_newtype_variant("Payload", 3, "Fields", value)
            }
            Self::Refuse => Err(SerializerType::Error::custom("refuse")),
        }
    }
}

fn python_contract() -> Value {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/finite_subprocess_protocol_contract.py");
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/rl/utils/finite_env.py");
    let output = Command::new("python")
        .arg(fixture)
        .arg(source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

fn commands() -> Vec<FiniteSubprocessCommand<Vec<i32>, Payload>> {
    vec![
        FiniteSubprocessCommand::Reset {
            options: Payload::Fields(vec![("seed".to_owned(), Payload::Integer(7))]),
        },
        FiniteSubprocessCommand::Step { action: vec![1, 2] },
        FiniteSubprocessCommand::Close,
        FiniteSubprocessCommand::Render {
            options: Payload::Fields(vec![(
                "mode".to_owned(),
                Payload::Text("rgb_array".to_owned()),
            )]),
        },
        FiniteSubprocessCommand::Seed {
            seed: Some(Payload::Integer(9)),
        },
        FiniteSubprocessCommand::GetAttribute {
            key: "action_space".to_owned(),
        },
        FiniteSubprocessCommand::SetAttribute {
            key: "value".to_owned(),
            value: Payload::Integer(3),
        },
    ]
}

fn replies() -> Vec<Reply> {
    vec![
        Reply::Reset {
            observation: Some(4),
            info: Some(Payload::Text("reset".to_owned())),
        },
        Reply::Step {
            transition: FiniteBackendStep {
                observation: None,
                reward: Some(1.5),
                done: true,
                info: Some(Info { env_id: 2 }),
            },
        },
        Reply::Close { result: None },
        Reply::Render {
            result: Some(Payload::List(vec![
                Payload::Integer(1),
                Payload::Integer(2),
            ])),
        },
        Reply::Seed {
            result: Some(Payload::List(vec![Payload::Integer(9)])),
        },
        Reply::Attribute { value: None },
        Reply::Failure(FiniteSubprocessFailure {
            kind: FiniteSubprocessFailureKind::Environment,
            message: "worker".to_owned(),
        }),
    ]
}

fn successful_reply(kind: FiniteSubprocessCommandKind) -> Reply {
    match kind {
        FiniteSubprocessCommandKind::Reset => Reply::Reset {
            observation: None,
            info: None,
        },
        FiniteSubprocessCommandKind::Step => Reply::Step {
            transition: FiniteBackendStep::default(),
        },
        FiniteSubprocessCommandKind::Close => Reply::Close { result: None },
        FiniteSubprocessCommandKind::Render => Reply::Render { result: None },
        FiniteSubprocessCommandKind::Seed => Reply::Seed { result: None },
        FiniteSubprocessCommandKind::GetAttribute | FiniteSubprocessCommandKind::SetAttribute => {
            Reply::Attribute { value: None }
        }
    }
}

#[test]
fn qlib_surface_and_tianshou_command_contract_are_frozen() {
    let python = python_contract();
    assert_eq!(
        python["subproc_bases"],
        json!(["FiniteVectorEnv", "SubprocVectorEnv"])
    );
    assert_eq!(python["subproc_body"], json!(["Pass"]));
    assert_eq!(
        python["shmem_bases"],
        json!(["FiniteVectorEnv", "ShmemVectorEnv"])
    );
    assert_eq!(python["shmem_body"], json!(["Pass"]));
    assert_eq!(
        python["commands"],
        json!([
            "step", "reset", "close", "render", "seed", "getattr", "setattr"
        ])
    );
    assert_eq!(python["no_reply"], json!(["setattr"]));
    assert_eq!(python["missing_attribute"], Value::Null);
    assert_eq!(python["seed_fallback"], Value::Null);
    assert_eq!(python["unknown"], "NotImplementedError");
    assert_eq!(FINITE_SUBPROCESS_PROTOCOL_VERSION, 1);
    assert_eq!(DEFAULT_FINITE_SUBPROCESS_FRAME_LIMIT, 16 * 1024 * 1024);
}

#[test]
fn every_command_and_reply_round_trips_with_stable_kinds() {
    let expected_command_kinds = [
        FiniteSubprocessCommandKind::Reset,
        FiniteSubprocessCommandKind::Step,
        FiniteSubprocessCommandKind::Close,
        FiniteSubprocessCommandKind::Render,
        FiniteSubprocessCommandKind::Seed,
        FiniteSubprocessCommandKind::GetAttribute,
        FiniteSubprocessCommandKind::SetAttribute,
    ];
    let expected_replies = [
        Some(FiniteSubprocessReplyKind::Reset),
        Some(FiniteSubprocessReplyKind::Step),
        Some(FiniteSubprocessReplyKind::Close),
        Some(FiniteSubprocessReplyKind::Render),
        Some(FiniteSubprocessReplyKind::Seed),
        Some(FiniteSubprocessReplyKind::Attribute),
        None,
    ];
    for (index, ((command, kind), expected_reply)) in commands()
        .into_iter()
        .zip(expected_command_kinds)
        .zip(expected_replies)
        .enumerate()
    {
        assert_eq!(command.kind(), kind);
        assert_eq!(command.expected_reply(), expected_reply);
        let request = Request::new(u64::try_from(index).unwrap(), command);
        let frame =
            encode_finite_subprocess_request(&request, DEFAULT_FINITE_SUBPROCESS_FRAME_LIMIT)
                .unwrap();
        assert_eq!(
            decode_finite_subprocess_request::<Vec<i32>, Payload>(
                &frame,
                DEFAULT_FINITE_SUBPROCESS_FRAME_LIMIT
            )
            .unwrap(),
            request
        );
    }

    let expected_reply_kinds = [
        FiniteSubprocessReplyKind::Reset,
        FiniteSubprocessReplyKind::Step,
        FiniteSubprocessReplyKind::Close,
        FiniteSubprocessReplyKind::Render,
        FiniteSubprocessReplyKind::Seed,
        FiniteSubprocessReplyKind::Attribute,
        FiniteSubprocessReplyKind::Failure,
    ];
    for (index, (reply, kind)) in replies().into_iter().zip(expected_reply_kinds).enumerate() {
        assert_eq!(reply.kind(), kind);
        let response = Response::new(u64::try_from(index).unwrap(), reply);
        let frame =
            encode_finite_subprocess_response(&response, DEFAULT_FINITE_SUBPROCESS_FRAME_LIMIT)
                .unwrap();
        assert_eq!(
            decode_finite_subprocess_response::<i64, f64, Info, Payload>(
                &frame,
                DEFAULT_FINITE_SUBPROCESS_FRAME_LIMIT
            )
            .unwrap(),
            response
        );
    }

    for kind in [
        FiniteSubprocessFailureKind::Environment,
        FiniteSubprocessFailureKind::Protocol,
        FiniteSubprocessFailureKind::Panic,
    ] {
        let value = FiniteSubprocessFailure {
            kind,
            message: "failure".to_owned(),
        };
        assert_eq!(
            serde_json::from_value::<FiniteSubprocessFailure>(
                serde_json::to_value(&value).unwrap()
            )
            .unwrap(),
            value
        );
    }
}

#[test]
fn response_validation_covers_success_failure_versions_ids_and_kinds() {
    for (index, command) in commands().into_iter().enumerate() {
        let request = Request::new(u64::try_from(index).unwrap(), command);
        if request.command.expected_reply().is_some() {
            let response =
                Response::new(request.request_id, successful_reply(request.command.kind()));
            validate_finite_subprocess_response(&request, &response).unwrap();
        }
        let failure = Response::new(
            request.request_id,
            Reply::Failure(FiniteSubprocessFailure {
                kind: FiniteSubprocessFailureKind::Protocol,
                message: "failure".to_owned(),
            }),
        );
        validate_finite_subprocess_response(&request, &failure).unwrap();
    }

    let request = Request::new(7, FiniteSubprocessCommand::Close);
    let wrong_id = Response::new(8, Reply::Close { result: None });
    assert!(matches!(
        validate_finite_subprocess_response(&request, &wrong_id),
        Err(FiniteSubprocessWireError::RequestIdMismatch {
            expected: 7,
            actual: 8
        })
    ));
    let wrong_kind = Response::new(7, Reply::Render { result: None });
    assert!(matches!(
        validate_finite_subprocess_response(&request, &wrong_kind),
        Err(FiniteSubprocessWireError::UnexpectedReply {
            command: FiniteSubprocessCommandKind::Close,
            expected: Some(FiniteSubprocessReplyKind::Close),
            actual: FiniteSubprocessReplyKind::Render,
        })
    ));
    let set = Request::new(
        9,
        FiniteSubprocessCommand::SetAttribute {
            key: "x".to_owned(),
            value: Payload::Integer(1),
        },
    );
    let unexpected = Response::new(9, Reply::Attribute { value: None });
    assert!(matches!(
        validate_finite_subprocess_response(&set, &unexpected),
        Err(FiniteSubprocessWireError::UnexpectedReply { expected: None, .. })
    ));

    let mut bad_request = request.clone();
    bad_request.version = 2;
    assert!(matches!(
        validate_finite_subprocess_response(&bad_request, &wrong_kind),
        Err(FiniteSubprocessWireError::UnsupportedVersion { actual: 2, .. })
    ));
    let mut bad_response = Response::new(7, Reply::Close { result: None });
    bad_response.version = 3;
    assert!(matches!(
        validate_finite_subprocess_response(&request, &bad_response),
        Err(FiniteSubprocessWireError::UnsupportedVersion { actual: 3, .. })
    ));
}

#[test]
fn request_codec_rejects_encode_decode_size_version_and_trailing_failures() {
    let request = Request::new(1, FiniteSubprocessCommand::Close);
    let request_frame = encode_finite_subprocess_request(&request, usize::MAX).unwrap();
    assert_eq!(
        encode_finite_subprocess_request(&request, request_frame.len()).unwrap(),
        request_frame
    );
    assert!(matches!(
        encode_finite_subprocess_request(&request, request_frame.len() - 1),
        Err(FiniteSubprocessWireError::FrameTooLarge { .. })
    ));
    assert!(matches!(
        decode_finite_subprocess_request::<Vec<i32>, Payload>(
            &request_frame,
            request_frame.len() - 1
        ),
        Err(FiniteSubprocessWireError::FrameTooLarge { .. })
    ));
    assert!(matches!(
        decode_finite_subprocess_request::<Vec<i32>, Payload>(&[], usize::MAX),
        Err(FiniteSubprocessWireError::Empty)
    ));
    assert!(matches!(
        decode_finite_subprocess_request::<Vec<i32>, Payload>(&[1], usize::MAX),
        Err(FiniteSubprocessWireError::Decode(_))
    ));
    let mut trailing_request = request_frame.clone();
    trailing_request.push(0);
    assert!(matches!(
        decode_finite_subprocess_request::<Vec<i32>, Payload>(&trailing_request, usize::MAX),
        Err(FiniteSubprocessWireError::Decode(_))
    ));
    let mut version_request = request_frame.clone();
    version_request[0] = 2;
    assert!(matches!(
        decode_finite_subprocess_request::<Vec<i32>, Payload>(&version_request, usize::MAX),
        Err(FiniteSubprocessWireError::UnsupportedVersion { actual: 2, .. })
    ));
    let mut invalid_request = request.clone();
    invalid_request.version = 2;
    assert!(matches!(
        encode_finite_subprocess_request(&invalid_request, usize::MAX),
        Err(FiniteSubprocessWireError::UnsupportedVersion { actual: 2, .. })
    ));
    let refusing_request = Request::new(
        1,
        FiniteSubprocessCommand::SetAttribute {
            key: "refuse".to_owned(),
            value: Payload::Refuse,
        },
    );
    assert!(matches!(
        encode_finite_subprocess_request(&refusing_request, usize::MAX),
        Err(FiniteSubprocessWireError::Encode(_))
    ));
}

#[test]
fn response_codec_rejects_encode_decode_size_version_and_trailing_failures() {
    let response = Response::new(1, Reply::Close { result: None });
    let response_frame = encode_finite_subprocess_response(&response, usize::MAX).unwrap();
    assert_eq!(
        encode_finite_subprocess_response(&response, response_frame.len()).unwrap(),
        response_frame
    );
    assert!(matches!(
        encode_finite_subprocess_response(&response, response_frame.len() - 1),
        Err(FiniteSubprocessWireError::FrameTooLarge { .. })
    ));
    assert!(matches!(
        decode_finite_subprocess_response::<i64, f64, Info, Payload>(
            &response_frame,
            response_frame.len() - 1
        ),
        Err(FiniteSubprocessWireError::FrameTooLarge { .. })
    ));
    assert!(matches!(
        decode_finite_subprocess_response::<i64, f64, Info, Payload>(&[], usize::MAX),
        Err(FiniteSubprocessWireError::Empty)
    ));
    assert!(matches!(
        decode_finite_subprocess_response::<i64, f64, Info, Payload>(&[1], usize::MAX),
        Err(FiniteSubprocessWireError::Decode(_))
    ));
    let mut trailing_response = response_frame.clone();
    trailing_response.push(0);
    assert!(matches!(
        decode_finite_subprocess_response::<i64, f64, Info, Payload>(
            &trailing_response,
            usize::MAX,
        ),
        Err(FiniteSubprocessWireError::Decode(_))
    ));
    let mut version_response = response_frame.clone();
    version_response[0] = 4;
    assert!(matches!(
        decode_finite_subprocess_response::<i64, f64, Info, Payload>(&version_response, usize::MAX,),
        Err(FiniteSubprocessWireError::UnsupportedVersion { actual: 4, .. })
    ));
    let mut invalid_response = response.clone();
    invalid_response.version = 5;
    assert!(matches!(
        encode_finite_subprocess_response(&invalid_response, usize::MAX),
        Err(FiniteSubprocessWireError::UnsupportedVersion { actual: 5, .. })
    ));
    let refusing_response = Response::new(
        1,
        Reply::Close {
            result: Some(Payload::Refuse),
        },
    );
    assert!(matches!(
        encode_finite_subprocess_response(&refusing_response, usize::MAX),
        Err(FiniteSubprocessWireError::Encode(_))
    ));
}
