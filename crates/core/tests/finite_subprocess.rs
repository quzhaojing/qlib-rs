use std::{
    collections::VecDeque,
    fmt::Write as _,
    future::{Future, pending},
    io,
    panic::panic_any,
    path::PathBuf,
    pin::Pin,
    time::Duration,
};

use bytes::Bytes;
use domain_core::{
    FiniteBackendStep, FiniteSubprocessCommand, FiniteSubprocessCommandHandler,
    FiniteSubprocessExitStatus, FiniteSubprocessFailure, FiniteSubprocessFailureKind,
    FiniteSubprocessLifecycle, FiniteSubprocessOperation, FiniteSubprocessProgram,
    FiniteSubprocessProgramFactory, FiniteSubprocessReply, FiniteSubprocessRequest,
    FiniteSubprocessResponse, FiniteSubprocessRuntimeError, FiniteSubprocessTransport,
    FiniteSubprocessWireError, FiniteSubprocessWorkerExit, FiniteSubprocessWorkerFactory,
    decode_finite_subprocess_response, encode_finite_subprocess_request,
    encode_finite_subprocess_response, serve_finite_subprocess, terminate_finite_subprocess,
    wait_for_finite_subprocess_exit,
};
use futures_util::{SinkExt, StreamExt};
use tokio::io::{AsyncWriteExt, DuplexStream, ReadHalf, WriteHalf, duplex, split};
use tokio_util::codec::{FramedRead, FramedWrite, LengthDelimitedCodec};

#[path = "finite_subprocess/backend.rs"]
mod backend;

type Command = FiniteSubprocessCommand<i32, String>;
type Request = FiniteSubprocessRequest<i32, String>;
type Reply = FiniteSubprocessReply<i32, f64, i32, String>;
type Response = FiniteSubprocessResponse<i32, f64, i32, String>;

enum LifecycleOutcome<T> {
    Ready(io::Result<T>),
    Pending,
}

struct FakeLifecycle {
    kill: Option<LifecycleOutcome<()>>,
    waits: VecDeque<LifecycleOutcome<FiniteSubprocessExitStatus>>,
}

impl FakeLifecycle {
    fn new(
        kill: LifecycleOutcome<()>,
        waits: impl IntoIterator<Item = LifecycleOutcome<FiniteSubprocessExitStatus>>,
    ) -> Self {
        Self {
            kill: Some(kill),
            waits: waits.into_iter().collect(),
        }
    }
}

impl FiniteSubprocessLifecycle for FakeLifecycle {
    fn kill(&mut self) -> Pin<Box<dyn Future<Output = io::Result<()>> + Send + '_>> {
        let outcome = self.kill.take().expect("one kill outcome");
        Box::pin(async move {
            match outcome {
                LifecycleOutcome::Ready(result) => result,
                LifecycleOutcome::Pending => pending().await,
            }
        })
    }

    fn wait(
        &mut self,
    ) -> Pin<Box<dyn Future<Output = io::Result<FiniteSubprocessExitStatus>> + Send + '_>> {
        let outcome = self.waits.pop_front().expect("one wait outcome");
        Box::pin(async move {
            match outcome {
                LifecycleOutcome::Ready(result) => result,
                LifecycleOutcome::Pending => pending().await,
            }
        })
    }
}

const TEST_TIMEOUT: Duration = Duration::from_secs(2);

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/finite_subprocess_transport.py")
}

fn python_program(mode: &str) -> FiniteSubprocessProgram {
    FiniteSubprocessProgram::new("python")
        .args([fixture().into_os_string(), mode.into()])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut output, byte| {
        write!(output, "{byte:02x}").unwrap();
        output
    })
}

fn codec(limit: usize) -> LengthDelimitedCodec {
    LengthDelimitedCodec::builder()
        .max_frame_length(limit)
        .new_codec()
}

#[tokio::test(flavor = "current_thread")]
async fn program_factory_and_worker_plugin_drive_real_processes() {
    let reset_response = Response::new(
        0,
        Reply::Reset {
            observation: Some(41),
            info: None,
        },
    );
    let close_response = Response::new(1, Reply::Close { result: None });
    let reset_frame = encode_finite_subprocess_response(&reset_response, 4096).unwrap();
    let close_frame = encode_finite_subprocess_response(&close_response, 4096).unwrap();
    let program = FiniteSubprocessProgram::new("python")
        .args([
            fixture().into_os_string(),
            "replies".into(),
            hex(&reset_frame).into(),
            hex(&close_frame).into(),
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"));
    let mut factory = FiniteSubprocessProgramFactory::new(vec![program], 4096, TEST_TIMEOUT);
    let mut worker = FiniteSubprocessWorkerFactory::create(&mut factory, 0).unwrap();
    assert_eq!(worker.frame_limit(), 4096);

    let reset_request = Request::new(
        0,
        Command::Reset {
            options: String::new(),
        },
    );
    worker
        .send_frame(encode_finite_subprocess_request(&reset_request, 4096).unwrap())
        .await
        .unwrap();
    let response: Response =
        decode_finite_subprocess_response(&worker.receive_frame().await.unwrap(), 4096).unwrap();
    assert_eq!(response, reset_response);

    let close_request = Request::new(1, Command::Close);
    worker
        .send_frame(encode_finite_subprocess_request(&close_request, 4096).unwrap())
        .await
        .unwrap();
    let response: Response =
        decode_finite_subprocess_response(&worker.receive_frame().await.unwrap(), 4096).unwrap();
    assert_eq!(response, close_response);
    worker.close_input().await.unwrap();
    assert!(worker.wait_for_exit().await.unwrap().success());

    let Err(missing) = FiniteSubprocessWorkerFactory::create(&mut factory, 1) else {
        panic!("missing program must fail");
    };
    assert!(matches!(
        missing,
        FiniteSubprocessRuntimeError::MissingProgram {
            id: 1,
            program_count: 1
        }
    ));

    let invalid = FiniteSubprocessProgram::new("definitely-not-a-real-qlib-worker");
    let mut invalid_factory =
        FiniteSubprocessProgramFactory::new(vec![invalid], 4096, TEST_TIMEOUT);
    let Err(error) = FiniteSubprocessWorkerFactory::create(&mut invalid_factory, 0) else {
        panic!("invalid program must fail");
    };
    assert!(matches!(error, FiniteSubprocessRuntimeError::Spawn(_)));

    let hanging = python_program("hang").arg("5");
    let mut hanging_factory =
        FiniteSubprocessProgramFactory::new(vec![hanging], 4096, Duration::from_millis(100));
    let mut worker = FiniteSubprocessWorkerFactory::create(&mut hanging_factory, 0).unwrap();
    assert!(!worker.terminate().await.unwrap().success());
}

struct Client {
    requests: FramedWrite<WriteHalf<DuplexStream>, LengthDelimitedCodec>,
    responses: FramedRead<ReadHalf<DuplexStream>, LengthDelimitedCodec>,
    limit: usize,
}

impl Client {
    fn pair(limit: usize) -> (Self, ReadHalf<DuplexStream>, WriteHalf<DuplexStream>) {
        let (client, worker) = duplex(limit * 4);
        let (client_read, client_write) = split(client);
        let (worker_read, worker_write) = split(worker);
        (
            Self {
                requests: FramedWrite::new(client_write, codec(limit)),
                responses: FramedRead::new(client_read, codec(limit)),
                limit,
            },
            worker_read,
            worker_write,
        )
    }

    async fn send(&mut self, request: &Request) {
        let frame = encode_finite_subprocess_request(request, self.limit).unwrap();
        self.requests.send(Bytes::from(frame)).await.unwrap();
    }

    async fn receive(&mut self) -> Response {
        let frame = self.responses.next().await.unwrap().unwrap();
        decode_finite_subprocess_response(&frame, self.limit).unwrap()
    }
}

#[derive(Default)]
struct HappyHandler {
    kinds: Vec<&'static str>,
}

impl FiniteSubprocessCommandHandler<i32, i32, f64, i32, String> for HappyHandler {
    fn handle(&mut self, command: Command) -> Result<Option<Reply>, FiniteSubprocessFailure> {
        let reply = match command {
            Command::Reset { options } => {
                self.kinds.push("reset");
                Some(Reply::Reset {
                    observation: Some(1),
                    info: Some(options),
                })
            }
            Command::Step { action } => {
                self.kinds.push("step");
                Some(Reply::Step {
                    transition: FiniteBackendStep {
                        observation: Some(action + 1),
                        reward: Some(2.5),
                        done: false,
                        info: Some(7),
                    },
                })
            }
            Command::Close => {
                self.kinds.push("close");
                Some(Reply::Close { result: None })
            }
            Command::Render { options } => {
                self.kinds.push("render");
                Some(Reply::Render {
                    result: Some(options),
                })
            }
            Command::Seed { seed } => {
                self.kinds.push("seed");
                Some(Reply::Seed { result: seed })
            }
            Command::GetAttribute { key } => {
                self.kinds.push("getattr");
                Some(Reply::Attribute { value: Some(key) })
            }
            Command::SetAttribute { .. } => {
                self.kinds.push("setattr");
                None
            }
        };
        Ok(reply)
    }
}

#[tokio::test]
async fn program_builder_and_real_process_request_paths_are_typed() {
    let response = Response::new(8, Reply::Close { result: None });
    let encoded = encode_finite_subprocess_response(&response, 1024).unwrap();
    let program = python_program("env_reply")
        .arg("QLIB_TEST_REPLY")
        .env("QLIB_TEST_REPLY", hex(&encoded))
        .envs([("QLIB_TEST_EXTRA", "present")]);
    assert_eq!(program.executable(), PathBuf::from("python"));
    assert_eq!(program.arguments().len(), 3);
    assert_eq!(program.environment().len(), 2);
    assert_eq!(
        program.current_directory(),
        Some(PathBuf::from(env!("CARGO_MANIFEST_DIR")).as_path())
    );

    let mut transport = FiniteSubprocessTransport::spawn(&program, 1024, TEST_TIMEOUT).unwrap();
    assert!(transport.process_id().is_some());
    assert_eq!(transport.frame_limit(), 1024);
    assert_eq!(transport.operation_timeout(), TEST_TIMEOUT);
    let request = Request::new(8, Command::Close);
    assert_eq!(
        transport
            .request::<_, i32, f64, i32, _>(&request)
            .await
            .unwrap(),
        Some(response)
    );
    transport.close_input().await.unwrap();
    assert!(transport.wait_for_exit().await.unwrap().success());

    let one_way = Request::new(
        9,
        Command::SetAttribute {
            key: "value".to_owned(),
            value: "3".to_owned(),
        },
    );
    let mut transport =
        FiniteSubprocessTransport::spawn(&python_program("eof"), 1024, TEST_TIMEOUT).unwrap();
    let reply: Option<Response> = transport.request(&one_way).await.unwrap();
    assert_eq!(reply, None);
    assert!(transport.wait_for_exit().await.unwrap().success());
}

#[tokio::test]
async fn string_action_request_covers_reply_and_send_failure_paths() {
    let response = Response::new(
        31,
        Reply::Step {
            transition: FiniteBackendStep {
                observation: Some(9),
                reward: Some(1.5),
                done: false,
                info: Some(4),
            },
        },
    );
    let encoded = encode_finite_subprocess_response(&response, 1024).unwrap();
    let request = FiniteSubprocessRequest::<String, String>::new(
        31,
        FiniteSubprocessCommand::Step {
            action: "action".to_owned(),
        },
    );
    let mut transport = FiniteSubprocessTransport::spawn(
        &python_program("reply").arg(hex(&encoded)),
        1024,
        TEST_TIMEOUT,
    )
    .unwrap();
    let reply: Option<Response> = transport.request(&request).await.unwrap();
    assert_eq!(reply, Some(response));
    assert!(transport.wait_for_exit().await.unwrap().success());

    let mut tiny =
        FiniteSubprocessTransport::spawn(&python_program("eof"), 1, TEST_TIMEOUT).unwrap();
    assert!(matches!(
        tiny.request::<_, i32, f64, i32, _>(&request).await,
        Err(FiniteSubprocessRuntimeError::Wire(
            FiniteSubprocessWireError::FrameTooLarge { .. }
        ))
    ));
    tiny.terminate().await.unwrap();

    let mut closed =
        FiniteSubprocessTransport::spawn(&python_program("exit_before_read"), 1024, TEST_TIMEOUT)
            .unwrap();
    assert!(closed.wait_for_exit().await.unwrap().success());
    assert!(matches!(
        closed.request::<_, i32, f64, i32, _>(&request).await,
        Err(FiniteSubprocessRuntimeError::Write(_))
    ));

    let one_way = FiniteSubprocessRequest::<String, String>::new(
        32,
        FiniteSubprocessCommand::SetAttribute {
            key: "value".to_owned(),
            value: "3".to_owned(),
        },
    );
    let mut transport =
        FiniteSubprocessTransport::spawn(&python_program("eof"), 1024, TEST_TIMEOUT).unwrap();
    let reply: Option<Response> = transport.request(&one_way).await.unwrap();
    assert_eq!(reply, None);
    assert!(transport.wait_for_exit().await.unwrap().success());
}

#[tokio::test]
async fn string_action_receive_maps_every_transport_and_wire_failure() {
    let request = FiniteSubprocessRequest::<String, String>::new(
        41,
        FiniteSubprocessCommand::Step {
            action: "action".to_owned(),
        },
    );
    // The same timeout also bounds real OS child termination. Allow Windows to
    // reap the process under instrumented build load; the worker still hangs 5s.
    let short = Duration::from_secs(1);
    let mut hanging =
        FiniteSubprocessTransport::spawn(&python_program("hang").arg("5"), 1024, short).unwrap();
    hanging.send(&request).await.unwrap();
    assert!(matches!(
        hanging.receive::<_, i32, f64, i32, _>(&request).await,
        Err(FiniteSubprocessRuntimeError::Timeout {
            operation: FiniteSubprocessOperation::Read,
            ..
        })
    ));
    hanging.terminate().await.unwrap();

    for mode in ["eof", "partial"] {
        let mut transport =
            FiniteSubprocessTransport::spawn(&python_program(mode), 1024, TEST_TIMEOUT).unwrap();
        transport.send(&request).await.unwrap();
        let result = transport.receive::<_, i32, f64, i32, _>(&request).await;
        if mode == "eof" {
            assert!(matches!(result, Err(FiniteSubprocessRuntimeError::Eof)));
        } else {
            assert!(matches!(result, Err(FiniteSubprocessRuntimeError::Read(_))));
        }
        assert!(transport.wait_for_exit().await.unwrap().success());
    }

    let cases = [
        ("00".to_owned(), false),
        (
            hex(&encode_finite_subprocess_response(
                &Response::new(42, Reply::Close { result: None }),
                1024,
            )
            .unwrap()),
            true,
        ),
    ];
    for (payload, mismatched) in cases {
        let mut transport = FiniteSubprocessTransport::spawn(
            &python_program("reply").arg(payload),
            1024,
            TEST_TIMEOUT,
        )
        .unwrap();
        transport.send(&request).await.unwrap();
        let result = transport.receive::<_, i32, f64, i32, _>(&request).await;
        if mismatched {
            assert!(matches!(
                result,
                Err(FiniteSubprocessRuntimeError::Wire(
                    FiniteSubprocessWireError::RequestIdMismatch { .. }
                ))
            ));
        } else {
            assert!(matches!(
                result,
                Err(FiniteSubprocessRuntimeError::Wire(
                    FiniteSubprocessWireError::Decode(_)
                ))
            ));
        }
        assert!(transport.wait_for_exit().await.unwrap().success());
    }
}

#[tokio::test]
async fn integer_action_send_timeout_is_typed() {
    let timeout = Duration::from_millis(100);
    let mut transport = FiniteSubprocessTransport::spawn(
        &python_program("hang_before_read").arg("5"),
        8 * 1024 * 1024,
        timeout,
    )
    .unwrap();
    let request = Request::new(
        51,
        Command::SetAttribute {
            key: "value".to_owned(),
            value: "x".repeat(4 * 1024 * 1024),
        },
    );
    assert!(matches!(
        transport.send(&request).await,
        Err(FiniteSubprocessRuntimeError::Timeout {
            operation: FiniteSubprocessOperation::Write,
            ..
        })
    ));
    transport.terminate().await.unwrap();
}

#[tokio::test]
async fn real_process_maps_spawn_wire_read_and_eof_failures() {
    let spawn = FiniteSubprocessTransport::spawn(
        &FiniteSubprocessProgram::new("definitely-not-a-real-qlib-worker"),
        64,
        TEST_TIMEOUT,
    );
    assert!(matches!(spawn, Err(FiniteSubprocessRuntimeError::Spawn(_))));

    let request = Request::new(1, Command::Close);
    let mut tiny =
        FiniteSubprocessTransport::spawn(&python_program("eof"), 1, TEST_TIMEOUT).unwrap();
    assert!(matches!(
        tiny.request::<_, i32, f64, i32, _>(&request).await,
        Err(FiniteSubprocessRuntimeError::Wire(
            FiniteSubprocessWireError::FrameTooLarge { .. }
        ))
    ));
    tiny.terminate().await.unwrap();

    let mut eof =
        FiniteSubprocessTransport::spawn(&python_program("eof"), 1024, TEST_TIMEOUT).unwrap();
    eof.send(&request).await.unwrap();
    assert!(matches!(
        eof.receive::<_, i32, f64, i32, _>(&request).await,
        Err(FiniteSubprocessRuntimeError::Eof)
    ));
    assert!(eof.wait_for_exit().await.unwrap().success());

    let mut malformed =
        FiniteSubprocessTransport::spawn(&python_program("reply").arg("00"), 1024, TEST_TIMEOUT)
            .unwrap();
    malformed.send(&request).await.unwrap();
    assert!(matches!(
        malformed.receive::<_, i32, f64, i32, _>(&request).await,
        Err(FiniteSubprocessRuntimeError::Wire(
            FiniteSubprocessWireError::Decode(_)
        ))
    ));
    malformed.wait_for_exit().await.unwrap();

    let wrong_id = Response::new(2, Reply::Close { result: None });
    let wrong_id_frame = encode_finite_subprocess_response(&wrong_id, 1024).unwrap();
    let mut mismatched = FiniteSubprocessTransport::spawn(
        &python_program("reply").arg(hex(&wrong_id_frame)),
        1024,
        TEST_TIMEOUT,
    )
    .unwrap();
    mismatched.send(&request).await.unwrap();
    assert!(matches!(
        mismatched.receive::<_, i32, f64, i32, _>(&request).await,
        Err(FiniteSubprocessRuntimeError::Wire(
            FiniteSubprocessWireError::RequestIdMismatch { .. }
        ))
    ));
    mismatched.wait_for_exit().await.unwrap();

    for mode in ["partial", "oversized"] {
        let program = if mode == "oversized" {
            python_program(mode).arg("2048")
        } else {
            python_program(mode)
        };
        let mut transport = FiniteSubprocessTransport::spawn(&program, 1024, TEST_TIMEOUT).unwrap();
        transport.send(&request).await.unwrap();
        assert!(matches!(
            transport.receive::<_, i32, f64, i32, _>(&request).await,
            Err(FiniteSubprocessRuntimeError::Read(_))
        ));
        transport.wait_for_exit().await.unwrap();
    }
}

#[tokio::test]
async fn real_process_maps_read_write_close_and_wait_timeouts() {
    let request = Request::new(1, Command::Close);
    // Cleanup shares the operation deadline. Tokio's kill also waits for exit,
    // so an OS exit notification delayed past that deadline is a valid Kill
    // timeout. Test exact error classification with FakeLifecycle below, and
    // require actual reaping here even when cleanup initially times out.
    let short = Duration::from_secs(5);
    let mut hanging =
        FiniteSubprocessTransport::spawn(&python_program("hang").arg("60"), 1024, short).unwrap();
    hanging.send(&request).await.unwrap();
    assert!(matches!(
        hanging.receive::<_, i32, f64, i32, _>(&request).await,
        Err(FiniteSubprocessRuntimeError::Timeout {
            operation: FiniteSubprocessOperation::Read,
            duration,
        }) if duration == short
    ));
    let wait_result = hanging.wait_for_exit().await;
    assert!(
        matches!(
            wait_result,
            Err(FiniteSubprocessRuntimeError::Timeout {
                operation: FiniteSubprocessOperation::Wait | FiniteSubprocessOperation::Kill,
                ..
            })
        ),
        "unexpected wait/cleanup result: {wait_result:?}"
    );
    assert_child_reaped(&mut hanging).await;

    let write_timeout = Duration::from_secs(5);
    let mut blocked_writer = FiniteSubprocessTransport::spawn(
        &python_program("hang_before_read").arg("60"),
        8 * 1024 * 1024,
        write_timeout,
    )
    .unwrap();
    let huge_request = FiniteSubprocessRequest::new(
        3,
        FiniteSubprocessCommand::<String, String>::Step {
            action: "x".repeat(4 * 1024 * 1024),
        },
    );
    assert!(matches!(
        blocked_writer
            .request::<_, i32, f64, i32, _>(&huge_request)
            .await,
        Err(FiniteSubprocessRuntimeError::Timeout {
            operation: FiniteSubprocessOperation::Write,
            ..
        })
    ));
    assert!(matches!(
        blocked_writer.close_input().await,
        Err(FiniteSubprocessRuntimeError::Timeout {
            operation: FiniteSubprocessOperation::CloseInput,
            ..
        })
    ));
    let termination = blocked_writer.terminate().await;
    assert!(
        matches!(
            termination,
            Ok(_)
                | Err(FiniteSubprocessRuntimeError::Timeout {
                    operation: FiniteSubprocessOperation::Kill,
                    ..
                })
        ),
        "unexpected forced cleanup result: {termination:?}"
    );
    assert_child_reaped(&mut blocked_writer).await;
}

async fn assert_child_reaped(transport: &mut FiniteSubprocessTransport) {
    // A cooperative wait timeout remains a Wait error even when its subsequent
    // kill/reap succeeds. Prove terminal state before inspecting Tokio's cached
    // exit status; never retry a still-live child or hide kill/reap failure.
    let observed = transport.wait_for_exit().await;
    assert!(
        matches!(
            observed,
            Ok(_)
                | Err(FiniteSubprocessRuntimeError::Timeout {
                    operation: FiniteSubprocessOperation::Wait,
                    ..
                })
        ),
        "unexpected cleanup result: {observed:?}"
    );
    assert_eq!(
        transport.process_id(),
        None,
        "child not reaped: {observed:?}"
    );
    let status = transport
        .wait_for_exit()
        .await
        .expect("completed child has a cached exit status");
    if let Ok(previous) = observed {
        assert_eq!(status, previous);
    }
    assert!(
        !status.success(),
        "stalled child should have been terminated"
    );
    assert_eq!(transport.process_id(), None);
}

#[tokio::test]
async fn worker_runs_every_command_and_preserves_one_way_and_close_order() {
    let (mut client, worker_read, worker_write) = Client::pair(4096);
    let task = tokio::spawn(async move {
        let mut handler = HappyHandler::default();
        let exit = serve_finite_subprocess(worker_read, worker_write, &mut handler, 4096).await;
        (exit, handler.kinds)
    });
    let commands = [
        Command::Reset {
            options: "reset-info".to_owned(),
        },
        Command::Step { action: 4 },
        Command::Render {
            options: "rgb".to_owned(),
        },
        Command::Seed {
            seed: Some("seed".to_owned()),
        },
        Command::GetAttribute {
            key: "space".to_owned(),
        },
    ];
    for (index, command) in commands.into_iter().enumerate() {
        let request = Request::new(index as u64, command);
        client.send(&request).await;
        assert_eq!(client.receive().await.request_id, index as u64);
    }
    client
        .send(&Request::new(
            5,
            Command::SetAttribute {
                key: "value".to_owned(),
                value: "3".to_owned(),
            },
        ))
        .await;
    client.send(&Request::new(6, Command::Close)).await;
    assert_eq!(client.receive().await.reply, Reply::Close { result: None });
    let (exit, kinds) = task.await.unwrap();
    assert_eq!(exit.unwrap(), FiniteSubprocessWorkerExit::Closed);
    assert_eq!(
        kinds,
        [
            "reset", "step", "render", "seed", "getattr", "setattr", "close"
        ]
    );
}

#[tokio::test]
async fn closure_handler_adapter_serves_a_complete_close_exchange() {
    let (mut client, worker_read, worker_write) = Client::pair(1024);
    let task = tokio::spawn(async move {
        let mut handler = |command: Command| -> Result<Option<Reply>, FiniteSubprocessFailure> {
            assert_eq!(command, Command::Close);
            Ok(Some(Reply::Close {
                result: Some("closed".to_owned()),
            }))
        };
        serve_finite_subprocess(worker_read, worker_write, &mut handler, 1024).await
    });
    client.send(&Request::new(12, Command::Close)).await;
    assert_eq!(
        client.receive().await.reply,
        Reply::Close {
            result: Some("closed".to_owned())
        }
    );
    assert_eq!(
        task.await.unwrap().unwrap(),
        FiniteSubprocessWorkerExit::Closed
    );
}

#[derive(Clone, Copy)]
enum FailureMode {
    Environment,
    Missing,
    Wrong,
    PanicStr,
    PanicString,
    PanicOpaque,
}

struct FailureHandler(FailureMode);

impl FiniteSubprocessCommandHandler<i32, i32, f64, i32, String> for FailureHandler {
    fn handle(&mut self, _command: Command) -> Result<Option<Reply>, FiniteSubprocessFailure> {
        match self.0 {
            FailureMode::Environment => Err(FiniteSubprocessFailure {
                kind: FiniteSubprocessFailureKind::Environment,
                message: "environment".to_owned(),
            }),
            FailureMode::Missing => Ok(None),
            FailureMode::Wrong => Ok(Some(Reply::Close { result: None })),
            FailureMode::PanicStr => panic!("str panic"),
            FailureMode::PanicString => panic_any("string panic".to_owned()),
            FailureMode::PanicOpaque => panic_any(17_u8),
        }
    }
}

#[tokio::test]
async fn worker_converts_handler_failures_panics_and_protocol_mistakes() {
    let cases = [
        (
            FailureMode::Environment,
            FiniteSubprocessFailureKind::Environment,
            "environment",
        ),
        (
            FailureMode::Missing,
            FiniteSubprocessFailureKind::Protocol,
            "produced no reply",
        ),
        (
            FailureMode::Wrong,
            FiniteSubprocessFailureKind::Protocol,
            "expected reply",
        ),
        (
            FailureMode::PanicStr,
            FiniteSubprocessFailureKind::Panic,
            "str panic",
        ),
        (
            FailureMode::PanicString,
            FiniteSubprocessFailureKind::Panic,
            "string panic",
        ),
        (
            FailureMode::PanicOpaque,
            FiniteSubprocessFailureKind::Panic,
            "non-string payload",
        ),
    ];
    for (mode, expected_kind, expected_message) in cases {
        let (mut client, worker_read, worker_write) = Client::pair(4096);
        let task = tokio::spawn(async move {
            let mut handler = FailureHandler(mode);
            serve_finite_subprocess(worker_read, worker_write, &mut handler, 4096).await
        });
        client
            .send(&Request::new(
                11,
                Command::Reset {
                    options: String::new(),
                },
            ))
            .await;
        let response = client.receive().await;
        let Reply::Failure(failure) = response.reply else {
            panic!("worker must return a structured failure");
        };
        assert_eq!(failure.kind, expected_kind);
        assert!(failure.message.contains(expected_message));
        assert_eq!(
            task.await.unwrap().unwrap(),
            FiniteSubprocessWorkerExit::Failure(expected_kind)
        );
    }
}

#[tokio::test]
async fn worker_maps_eof_malformed_oversized_and_broken_response_streams() {
    let (client, worker_read, worker_write) = Client::pair(64);
    let eof = tokio::spawn(async move {
        let mut handler = HappyHandler::default();
        serve_finite_subprocess(worker_read, worker_write, &mut handler, 64).await
    });
    drop(client);
    assert_eq!(eof.await.unwrap().unwrap(), FiniteSubprocessWorkerExit::Eof);

    let (client, worker) = duplex(64);
    let (worker_read, worker_write) = split(worker);
    let malformed = tokio::spawn(async move {
        let mut handler = HappyHandler::default();
        serve_finite_subprocess(worker_read, worker_write, &mut handler, 64).await
    });
    let (_, client_write) = split(client);
    let mut requests = FramedWrite::new(client_write, codec(64));
    requests.send(Bytes::from_static(&[1])).await.unwrap();
    assert!(matches!(
        malformed.await.unwrap(),
        Err(FiniteSubprocessRuntimeError::Wire(
            FiniteSubprocessWireError::Decode(_)
        ))
    ));

    let (client, worker) = duplex(64);
    let (worker_read, worker_write) = split(worker);
    let oversized = tokio::spawn(async move {
        let mut handler = HappyHandler::default();
        serve_finite_subprocess(worker_read, worker_write, &mut handler, 8).await
    });
    let (_, mut client_write) = split(client);
    client_write.write_all(&9_u32.to_be_bytes()).await.unwrap();
    assert!(matches!(
        oversized.await.unwrap(),
        Err(FiniteSubprocessRuntimeError::Read(_))
    ));

    let (client, worker) = duplex(4096);
    let (worker_read, worker_write) = split(worker);
    let broken = tokio::spawn(async move {
        let mut handler = HappyHandler::default();
        serve_finite_subprocess(worker_read, worker_write, &mut handler, 4096).await
    });
    let (client_read, client_write) = split(client);
    drop(client_read);
    let request = Request::new(
        1,
        Command::Reset {
            options: String::new(),
        },
    );
    let frame = encode_finite_subprocess_request(&request, 4096).unwrap();
    let mut requests = FramedWrite::new(client_write, codec(4096));
    requests.send(Bytes::from(frame.clone())).await.unwrap();
    drop(requests);
    assert!(matches!(
        broken.await.unwrap(),
        Err(FiniteSubprocessRuntimeError::Write(_))
    ));
}

#[tokio::test]
async fn worker_maps_protocol_response_write_and_encode_failures() {
    for mode in [FailureMode::Missing, FailureMode::Wrong] {
        let (client, worker) = duplex(4096);
        let (worker_read, worker_write) = split(worker);
        let task = tokio::spawn(async move {
            let mut handler = FailureHandler(mode);
            serve_finite_subprocess(worker_read, worker_write, &mut handler, 4096).await
        });
        let (client_read, client_write) = split(client);
        let mut requests = FramedWrite::new(client_write, codec(4096));
        let request = Request::new(
            2,
            Command::Reset {
                options: String::new(),
            },
        );
        let frame = encode_finite_subprocess_request(&request, 4096).unwrap();
        requests.send(Bytes::from(frame)).await.unwrap();
        drop(requests);
        drop(client_read);
        assert!(matches!(
            task.await.unwrap(),
            Err(FiniteSubprocessRuntimeError::Write(_))
        ));
    }

    let (client, worker) = duplex(4096);
    let (worker_read, worker_write) = split(worker);
    let encoding = tokio::spawn(async move {
        let mut handler = |_command: Command| {
            Ok::<_, FiniteSubprocessFailure>(Some(Reply::Close {
                result: Some("x".repeat(4096)),
            }))
        };
        serve_finite_subprocess(worker_read, worker_write, &mut handler, 64).await
    });
    let (_, client_write) = split(client);
    let request = Request::new(3, Command::Close);
    let frame = encode_finite_subprocess_request(&request, 4096).unwrap();
    let mut requests = FramedWrite::new(client_write, codec(4096));
    requests.send(Bytes::from(frame)).await.unwrap();
    assert!(matches!(
        encoding.await.unwrap(),
        Err(FiniteSubprocessRuntimeError::Wire(
            FiniteSubprocessWireError::FrameTooLarge { .. }
        ))
    ));
}

fn assert_runtime_error<T: std::fmt::Debug>(
    result: Result<T, FiniteSubprocessRuntimeError>,
    expected: &str,
) {
    assert!(result.unwrap_err().to_string().contains(expected));
}

#[tokio::test]
async fn lifecycle_plugin_maps_every_wait_kill_error_and_timeout() {
    let duration = Duration::from_millis(1);
    let failure = |message| io::Error::other(message);

    let mut child = FakeLifecycle::new(
        LifecycleOutcome::Ready(Ok(())),
        [LifecycleOutcome::Ready(Ok(
            FiniteSubprocessExitStatus::new(true, Some(7)),
        ))],
    );
    let status = wait_for_finite_subprocess_exit(&mut child, duration)
        .await
        .unwrap();
    assert!(status.success());
    assert_eq!(status.code(), Some(7));

    let mut child = FakeLifecycle::new(
        LifecycleOutcome::Ready(Ok(())),
        [LifecycleOutcome::Ready(Err(failure("wait")))],
    );
    assert_runtime_error(
        wait_for_finite_subprocess_exit(&mut child, duration).await,
        "failed to wait",
    );

    let mut child = FakeLifecycle::new(
        LifecycleOutcome::Ready(Ok(())),
        [
            LifecycleOutcome::Pending,
            LifecycleOutcome::Ready(Ok(FiniteSubprocessExitStatus::new(false, Some(8)))),
        ],
    );
    assert_runtime_error(
        wait_for_finite_subprocess_exit(&mut child, duration).await,
        "wait timed out",
    );

    let mut child = FakeLifecycle::new(
        LifecycleOutcome::Ready(Err(failure("cleanup"))),
        [LifecycleOutcome::Pending],
    );
    assert_runtime_error(
        wait_for_finite_subprocess_exit(&mut child, duration).await,
        "failed to kill",
    );

    let cases = [
        (
            FakeLifecycle::new(LifecycleOutcome::Pending, []),
            "kill timed out",
        ),
        (
            FakeLifecycle::new(LifecycleOutcome::Ready(Err(failure("kill"))), []),
            "failed to kill",
        ),
        (
            FakeLifecycle::new(LifecycleOutcome::Ready(Ok(())), [LifecycleOutcome::Pending]),
            "wait timed out",
        ),
        (
            FakeLifecycle::new(
                LifecycleOutcome::Ready(Ok(())),
                [LifecycleOutcome::Ready(Err(failure("reap")))],
            ),
            "failed to wait",
        ),
    ];
    for (mut child, expected) in cases {
        assert_runtime_error(
            terminate_finite_subprocess(&mut child, duration).await,
            expected,
        );
    }
}

#[test]
fn operation_and_runtime_errors_have_stable_diagnostics() {
    let operations = [
        (FiniteSubprocessOperation::Write, "write"),
        (FiniteSubprocessOperation::Read, "read"),
        (FiniteSubprocessOperation::CloseInput, "close-input"),
        (FiniteSubprocessOperation::Wait, "wait"),
        (FiniteSubprocessOperation::Kill, "kill"),
    ];
    for (operation, expected) in operations {
        assert_eq!(operation.to_string(), expected);
        assert!(
            FiniteSubprocessRuntimeError::Timeout {
                operation,
                duration: Duration::from_millis(1),
            }
            .to_string()
            .contains(expected)
        );
    }
    let errors = [
        FiniteSubprocessRuntimeError::Write(io::Error::other("write")),
        FiniteSubprocessRuntimeError::Read(io::Error::other("read")),
        FiniteSubprocessRuntimeError::CloseInput(io::Error::other("close")),
        FiniteSubprocessRuntimeError::Wait(io::Error::other("wait")),
        FiniteSubprocessRuntimeError::Kill(io::Error::other("kill")),
    ];
    for error in errors {
        assert!(!error.to_string().is_empty());
    }
}
