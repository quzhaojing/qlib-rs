use std::{
    collections::VecDeque,
    future::Future,
    io,
    path::PathBuf,
    pin::Pin,
    process::Command,
    sync::{Arc, Mutex},
};

use domain_core::{
    BoxedFiniteSubprocessWorker, EnvironmentPluginError, FiniteBackendStep,
    FiniteSubprocessBackend, FiniteSubprocessBackendBuildError, FiniteSubprocessBackendError,
    FiniteSubprocessBackendOperation, FiniteSubprocessFailure, FiniteSubprocessFailureKind,
    FiniteSubprocessReply, FiniteSubprocessRequest, FiniteSubprocessResponse,
    FiniteSubprocessRuntime, FiniteSubprocessRuntimeError, FiniteSubprocessWorker,
    FiniteSubprocessWorkerFactory, FiniteVectorBackend, decode_finite_subprocess_request,
    encode_finite_subprocess_response,
};

type Request = FiniteSubprocessRequest<i32, String>;
type Response = FiniteSubprocessResponse<i32, f64, i32, String>;
type Worker = BoxedFiniteSubprocessWorker;
type Backend = FiniteSubprocessBackend<i32, i32, f64, i32, String>;
type WorkerFuture<'a, T> =
    Pin<Box<dyn Future<Output = Result<T, FiniteSubprocessRuntimeError>> + Send + 'a>>;

struct FakeWorker {
    id: usize,
    events: Arc<Mutex<Vec<String>>>,
    frame_limit: usize,
    responses: VecDeque<Response>,
    raw_responses: VecDeque<Vec<u8>>,
    send_errors: VecDeque<Option<&'static str>>,
    receive_errors: VecDeque<Option<&'static str>>,
    close_input_error: Option<&'static str>,
    wait_error: Option<&'static str>,
    terminate_error: Option<&'static str>,
    exit_code: i32,
}

impl Default for FakeWorker {
    fn default() -> Self {
        Self {
            id: 0,
            events: Arc::default(),
            frame_limit: 4096,
            responses: VecDeque::new(),
            raw_responses: VecDeque::new(),
            send_errors: VecDeque::new(),
            receive_errors: VecDeque::new(),
            close_input_error: None,
            wait_error: None,
            terminate_error: None,
            exit_code: 0,
        }
    }
}

impl FakeWorker {
    fn boxed(id: usize, events: Arc<Mutex<Vec<String>>>, responses: Vec<Response>) -> Worker {
        Box::new(Self {
            id,
            events,
            responses: responses.into(),
            exit_code: i32::try_from(id).expect("fake worker id fits i32"),
            ..Self::default()
        })
    }

    fn event(&self, event: &str) {
        self.events
            .lock()
            .unwrap()
            .push(format!("{event}:{}", self.id));
    }

    fn runtime_error(message: &'static str) -> FiniteSubprocessRuntimeError {
        FiniteSubprocessRuntimeError::Read(io::Error::other(message))
    }
}

impl FiniteSubprocessWorker for FakeWorker {
    fn frame_limit(&self) -> usize {
        self.frame_limit
    }

    fn send_frame(&mut self, frame: Vec<u8>) -> WorkerFuture<'_, ()> {
        let request: Request =
            decode_finite_subprocess_request(&frame, self.frame_limit()).unwrap();
        self.event(&format!("send-{}", request.request_id));
        let error = self.send_errors.pop_front().flatten();
        Box::pin(async move { error.map_or(Ok(()), |message| Err(Self::runtime_error(message))) })
    }

    fn receive_frame(&mut self) -> WorkerFuture<'_, Vec<u8>> {
        let error = self.receive_errors.pop_front().flatten();
        let raw_response = self.raw_responses.pop_front();
        let response = self.responses.pop_front();
        if raw_response.is_some() {
            self.event("recv-raw");
        } else if let Some(response) = &response {
            self.event(&format!("recv-{}", response.request_id));
        } else {
            self.event("recv-missing");
        }
        let frame_limit = self.frame_limit;
        Box::pin(async move {
            if let Some(message) = error {
                Err(Self::runtime_error(message))
            } else if let Some(raw_response) = raw_response {
                Ok(raw_response)
            } else {
                Ok(encode_finite_subprocess_response(
                    &response.expect("configured fake response"),
                    frame_limit,
                )
                .unwrap())
            }
        })
    }

    fn close_input(&mut self) -> WorkerFuture<'_, ()> {
        self.event("close-input");
        let error = self.close_input_error.take();
        Box::pin(async move { error.map_or(Ok(()), |message| Err(Self::runtime_error(message))) })
    }

    fn wait_for_exit(&mut self) -> WorkerFuture<'_, domain_core::FiniteSubprocessExitStatus> {
        self.event("wait");
        let error = self.wait_error.take();
        let code = self.exit_code;
        Box::pin(async move {
            error.map_or_else(
                || {
                    Ok(domain_core::FiniteSubprocessExitStatus::new(
                        true,
                        Some(code),
                    ))
                },
                |message| Err(Self::runtime_error(message)),
            )
        })
    }

    fn terminate(&mut self) -> WorkerFuture<'_, domain_core::FiniteSubprocessExitStatus> {
        self.event("terminate");
        let error = self.terminate_error.take();
        let code = self.exit_code;
        Box::pin(async move {
            error.map_or_else(
                || {
                    Ok(domain_core::FiniteSubprocessExitStatus::new(
                        false,
                        Some(code),
                    ))
                },
                |message| Err(Self::runtime_error(message)),
            )
        })
    }
}

fn reset_response(request_id: u64, observation: Option<i32>) -> Response {
    Response::new(
        request_id,
        FiniteSubprocessReply::Reset {
            observation,
            info: None,
        },
    )
}

fn step_response(request_id: u64, observation: i32) -> Response {
    Response::new(
        request_id,
        FiniteSubprocessReply::Step {
            transition: FiniteBackendStep {
                observation: Some(observation),
                reward: Some(f64::from(observation)),
                done: observation < 0,
                info: Some(observation + 1),
            },
        },
    )
}

fn close_response(request_id: u64) -> Response {
    Response::new(request_id, FiniteSubprocessReply::Close { result: None })
}

fn failure_response(request_id: u64, message: &str) -> Response {
    Response::new(
        request_id,
        FiniteSubprocessReply::Failure(FiniteSubprocessFailure {
            kind: FiniteSubprocessFailureKind::Environment,
            message: message.to_owned(),
        }),
    )
}

#[test]
fn batch_order_duplicates_and_request_ids_match_tianshou_sync_semantics() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let workers = vec![
        FakeWorker::boxed(
            0,
            Arc::clone(&events),
            vec![reset_response(1, Some(10)), reset_response(2, Some(11))],
        ),
        FakeWorker::boxed(1, Arc::clone(&events), vec![reset_response(0, Some(20))]),
    ];
    let mut backend: Backend = FiniteSubprocessBackend::new(workers).unwrap();
    assert_eq!(backend.environment_count(), 2);
    assert!(!backend.is_poisoned());
    assert!(!backend.is_closed());
    assert_eq!(
        backend.reset_environments(&[1, 0, 0]).unwrap(),
        [Some(20), Some(10), Some(11)]
    );
    assert_eq!(
        *events.lock().unwrap(),
        [
            "send-0:1", "send-1:0", "send-2:0", "recv-0:1", "recv-1:0", "recv-2:0"
        ]
    );

    events.lock().unwrap().clear();
    let workers = vec![
        FakeWorker::boxed(0, Arc::clone(&events), vec![step_response(1, 31)]),
        FakeWorker::boxed(
            1,
            Arc::clone(&events),
            vec![step_response(0, 30), step_response(2, -32)],
        ),
    ];
    let mut backend: Backend = FiniteSubprocessBackend::new(workers).unwrap();
    backend.reset_environments(&[]).unwrap();
    let transitions = backend.step_environments(&[7, 8, 9], &[1, 0, 1]).unwrap();
    assert_eq!(transitions[0].observation, Some(30));
    assert_eq!(transitions[1].reward, Some(31.0));
    assert!(transitions[2].done);
    assert_eq!(
        *events.lock().unwrap(),
        [
            "send-0:1", "send-1:0", "send-2:1", "recv-0:1", "recv-1:0", "recv-2:1"
        ]
    );
}

#[test]
fn construction_validation_and_factory_plugin_are_typed() {
    assert!(matches!(
        FiniteSubprocessBackend::<i32, i32, f64, i32, String>::new(Vec::new()),
        Err(FiniteSubprocessBackendBuildError::Empty)
    ));

    let events = Arc::new(Mutex::new(Vec::new()));
    let ids = Arc::new(Mutex::new(Vec::new()));
    let ids_for_factory = Arc::clone(&ids);
    let events_for_factory = Arc::clone(&events);
    let mut factory = move |id| {
        ids_for_factory.lock().unwrap().push(id);
        Ok(FakeWorker::boxed(
            id,
            Arc::clone(&events_for_factory),
            Vec::new(),
        ))
    };
    let backend: Backend = FiniteSubprocessBackend::from_factory(2, &mut factory).unwrap();
    assert_eq!(backend.environment_count(), 2);
    assert_eq!(*ids.lock().unwrap(), [0, 1]);

    let mut unused = |_id| -> Result<Worker, FiniteSubprocessRuntimeError> { unreachable!() };
    assert!(matches!(
        Backend::from_factory(0, &mut unused),
        Err(FiniteSubprocessBackendBuildError::Empty)
    ));

    let mut failing = |id| -> Result<Worker, FiniteSubprocessRuntimeError> {
        if id == 1 {
            Err(FiniteSubprocessRuntimeError::Read(io::Error::other(
                "factory",
            )))
        } else {
            Ok(FakeWorker::boxed(id, Arc::clone(&events), Vec::new()))
        }
    };
    let Err(error) = Backend::from_factory(3, &mut failing) else {
        panic!("factory must fail");
    };
    assert!(matches!(
        error,
        FiniteSubprocessBackendBuildError::Worker {
            environment_id: 1,
            ..
        }
    ));
    assert!(error.to_string().contains("factory"));

    let runtime_error =
        || -> Result<FiniteSubprocessRuntime, io::Error> { Err(io::Error::other("runtime")) };
    let mut runtime_factory = runtime_error;
    let runtime_worker = FakeWorker::boxed(0, Arc::default(), Vec::new());
    let Err(error) = Backend::new_with_runtime_factory(vec![runtime_worker], &mut runtime_factory)
    else {
        panic!("runtime must fail");
    };
    assert!(matches!(
        error,
        FiniteSubprocessBackendBuildError::Runtime(_)
    ));
    assert!(error.to_string().contains("runtime"));

    let mut runtime_factory = runtime_error;
    let mut unused_worker_factory = |_id| -> Result<Worker, FiniteSubprocessRuntimeError> {
        unreachable!("runtime failure precedes worker creation")
    };
    assert!(matches!(
        Backend::from_factories(1, &mut unused_worker_factory, &mut runtime_factory),
        Err(FiniteSubprocessBackendBuildError::Runtime(_))
    ));
}

#[test]
fn validation_happens_before_any_worker_mutation() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let worker = FakeWorker::boxed(0, Arc::clone(&events), Vec::new());
    let mut backend: Backend = FiniteSubprocessBackend::new(vec![worker]).unwrap();

    assert!(matches!(
        backend.reset_environments(&[1]),
        Err(FiniteSubprocessBackendError::InvalidEnvironmentId {
            id: 1,
            environment_count: 1
        })
    ));
    assert!(matches!(
        backend.step_environments(&[], &[0]),
        Err(FiniteSubprocessBackendError::ActionCount {
            required: 1,
            actual: 0
        })
    ));
    assert!(matches!(
        backend.step_environments(&[1], &[2]),
        Err(FiniteSubprocessBackendError::InvalidEnvironmentId { id: 2, .. })
    ));
    assert!(events.lock().unwrap().is_empty());
}

#[test]
fn transport_failures_poison_batches_and_force_cleanup() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut send_failure = FakeWorker {
        id: 0,
        events: Arc::clone(&events),
        send_errors: [Some("send")].into(),
        ..FakeWorker::default()
    };
    send_failure.exit_code = 4;
    let mut backend: Backend = FiniteSubprocessBackend::new(vec![Box::new(send_failure)]).unwrap();
    assert!(matches!(
        backend.reset_environments(&[0]),
        Err(FiniteSubprocessBackendError::Send {
            operation: FiniteSubprocessBackendOperation::Reset,
            environment_id: 0,
            ..
        })
    ));
    assert!(backend.is_poisoned());
    assert!(matches!(
        backend.reset_environments(&[]),
        Err(FiniteSubprocessBackendError::Poisoned)
    ));
    let statuses = backend.close().unwrap();
    assert_eq!(statuses[0].code(), Some(4));
    assert!(backend.is_closed());
    assert!(matches!(
        backend.close(),
        Err(FiniteSubprocessBackendError::Closed)
    ));

    let mut receive_failure = FakeWorker {
        id: 0,
        events,
        receive_errors: [Some("receive")].into(),
        ..FakeWorker::default()
    };
    receive_failure
        .responses
        .push_back(reset_response(0, Some(1)));
    let mut backend: Backend =
        FiniteSubprocessBackend::new(vec![Box::new(receive_failure)]).unwrap();
    assert!(matches!(
        backend.reset_environments(&[0]),
        Err(FiniteSubprocessBackendError::Receive {
            operation: FiniteSubprocessBackendOperation::Reset,
            ..
        })
    ));
}

#[test]
fn wire_failures_poison_batches_and_are_typed() {
    for worker in [
        FakeWorker {
            frame_limit: 1,
            ..FakeWorker::default()
        },
        FakeWorker {
            raw_responses: [vec![0]].into(),
            ..FakeWorker::default()
        },
        FakeWorker {
            responses: [reset_response(99, Some(1))].into(),
            ..FakeWorker::default()
        },
    ] {
        let mut backend: Backend = FiniteSubprocessBackend::new(vec![Box::new(worker)]).unwrap();
        let error = backend.reset_environments(&[0]).unwrap_err();
        assert!(matches!(
            error,
            FiniteSubprocessBackendError::Send { .. }
                | FiniteSubprocessBackendError::Receive { .. }
        ));
        assert!(backend.is_poisoned());
        backend.close().unwrap();
        assert!(backend.is_closed());
    }
}

#[test]
fn close_maps_wire_encoding_decoding_and_correlation_failures() {
    for worker in [
        FakeWorker {
            frame_limit: 1,
            ..FakeWorker::default()
        },
        FakeWorker {
            raw_responses: [vec![0]].into(),
            ..FakeWorker::default()
        },
        FakeWorker {
            responses: [close_response(99)].into(),
            ..FakeWorker::default()
        },
    ] {
        let mut backend: Backend = FiniteSubprocessBackend::new(vec![Box::new(worker)]).unwrap();
        let error = backend.close().unwrap_err();
        assert!(matches!(
            error,
            FiniteSubprocessBackendError::Send {
                operation: FiniteSubprocessBackendOperation::Close,
                ..
            } | FiniteSubprocessBackendError::Receive {
                operation: FiniteSubprocessBackendOperation::Close,
                ..
            }
        ));
        assert!(backend.is_closed());
    }
}

#[test]
fn structured_remote_failures_poison_reset_and_step() {
    for step in [false, true] {
        let response = if step {
            failure_response(0, "step")
        } else {
            failure_response(0, "reset")
        };
        let worker = FakeWorker::boxed(0, Arc::default(), vec![response]);
        let mut backend: Backend = FiniteSubprocessBackend::new(vec![worker]).unwrap();
        let error = if step {
            backend.step_environments(&[1], &[0]).unwrap_err()
        } else {
            backend.reset_environments(&[0]).unwrap_err()
        };
        assert!(matches!(
            error,
            FiniteSubprocessBackendError::Remote {
                environment_id: 0,
                ..
            }
        ));
        assert!(
            error
                .to_string()
                .contains(if step { "step" } else { "reset" })
        );
        assert!(backend.is_poisoned());
    }
}

#[test]
fn healthy_close_is_sequential_in_tianshou_worker_order() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let workers = vec![
        FakeWorker::boxed(0, Arc::clone(&events), vec![close_response(0)]),
        FakeWorker::boxed(1, Arc::clone(&events), vec![close_response(1)]),
    ];
    let mut backend: Backend = FiniteSubprocessBackend::new(workers).unwrap();
    let statuses = backend.close().unwrap();
    assert_eq!(
        statuses
            .iter()
            .map(|status| status.code())
            .collect::<Vec<_>>(),
        [Some(0), Some(1)]
    );
    assert_eq!(
        *events.lock().unwrap(),
        [
            "send-0:0",
            "recv-0:0",
            "close-input:0",
            "wait:0",
            "send-1:1",
            "recv-1:1",
            "close-input:1",
            "wait:1"
        ]
    );
}

#[derive(Clone, Copy)]
enum CloseFailure {
    Send,
    Receive,
    Remote,
    CloseInput,
    Wait,
}

#[test]
fn close_maps_each_failure_stage_and_always_attempts_forced_cleanup() {
    for mode in [
        CloseFailure::Send,
        CloseFailure::Receive,
        CloseFailure::Remote,
        CloseFailure::CloseInput,
        CloseFailure::Wait,
    ] {
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut worker = FakeWorker {
            id: 0,
            events: Arc::clone(&events),
            responses: [match mode {
                CloseFailure::Remote => failure_response(0, "close"),
                _ => close_response(0),
            }]
            .into(),
            ..FakeWorker::default()
        };
        match mode {
            CloseFailure::Send => worker.send_errors.push_back(Some("send")),
            CloseFailure::Receive => worker.receive_errors.push_back(Some("receive")),
            CloseFailure::CloseInput => worker.close_input_error = Some("close-input"),
            CloseFailure::Wait => worker.wait_error = Some("wait"),
            CloseFailure::Remote => {}
        }
        let mut backend: Backend = FiniteSubprocessBackend::new(vec![Box::new(worker)]).unwrap();
        let error = backend.close().unwrap_err();
        assert!(match mode {
            CloseFailure::Send => matches!(error, FiniteSubprocessBackendError::Send { .. }),
            CloseFailure::Receive => matches!(error, FiniteSubprocessBackendError::Receive { .. }),
            CloseFailure::Remote => matches!(error, FiniteSubprocessBackendError::Remote { .. }),
            CloseFailure::CloseInput => {
                matches!(error, FiniteSubprocessBackendError::CloseInput { .. })
            }
            CloseFailure::Wait => matches!(error, FiniteSubprocessBackendError::Wait { .. }),
        });
        assert!(backend.is_closed());
        assert!(events.lock().unwrap().contains(&"terminate:0".to_owned()));
    }

    let mut worker = FakeWorker {
        id: 0,
        events: Arc::new(Mutex::new(Vec::new())),
        send_errors: [Some("poison")].into(),
        terminate_error: Some("terminate"),
        ..FakeWorker::default()
    };
    worker.responses.push_back(close_response(0));
    let events = Arc::clone(&worker.events);
    let second = FakeWorker {
        id: 1,
        events: Arc::clone(&events),
        terminate_error: Some("second terminate"),
        ..FakeWorker::default()
    };
    let mut backend: Backend =
        FiniteSubprocessBackend::new(vec![Box::new(worker), Box::new(second)]).unwrap();
    backend.reset_environments(&[0]).unwrap_err();
    assert!(matches!(
        backend.close(),
        Err(FiniteSubprocessBackendError::Terminate {
            environment_id: 0,
            ..
        })
    ));
    assert!(events.lock().unwrap().contains(&"terminate:1".to_owned()));
}

#[test]
fn finite_vector_backend_adapter_preserves_diagnostics() {
    let worker = FakeWorker::boxed(0, Arc::default(), vec![step_response(0, 7)]);
    let mut backend: Box<dyn FiniteVectorBackend<i32, i32, f64, i32>> =
        Box::new(Backend::new(vec![worker]).unwrap());
    assert_eq!(backend.environment_count(), 1);
    let transition = backend.step(&[1], &[0]).unwrap().remove(0);
    assert_eq!(transition.observation, Some(7));
    assert!(
        backend
            .step(&[], &[0])
            .unwrap_err()
            .to_string()
            .contains("exactly 1")
    );
    let error: EnvironmentPluginError = backend.reset(&[2]).unwrap_err();
    assert!(error.to_string().contains("outside 0..1"));
}

#[test]
fn closure_factory_adapter_delegates() {
    let mut factory = |_id| -> Result<Worker, FiniteSubprocessRuntimeError> {
        Ok(FakeWorker::boxed(0, Arc::default(), Vec::new()))
    };
    let worker = FiniteSubprocessWorkerFactory::create(&mut factory, 7).unwrap();
    let backend: Backend = FiniteSubprocessBackend::new(vec![worker]).unwrap();
    assert_eq!(backend.environment_count(), 1);
}

#[test]
fn step_transport_failures_and_unexpected_replies_are_typed() {
    for receive in [false, true] {
        let mut worker = FakeWorker {
            id: 0,
            events: Arc::default(),
            responses: [step_response(0, 1)].into(),
            ..FakeWorker::default()
        };
        if receive {
            worker.receive_errors.push_back(Some("step receive"));
        } else {
            worker.send_errors.push_back(Some("step send"));
        }
        let mut backend: Backend = FiniteSubprocessBackend::new(vec![Box::new(worker)]).unwrap();
        let error = backend.step_environments(&[1], &[0]).unwrap_err();
        assert!(if receive {
            matches!(error, FiniteSubprocessBackendError::Receive { .. })
        } else {
            matches!(error, FiniteSubprocessBackendError::Send { .. })
        });
    }

    for reset in [false, true] {
        let response = if reset {
            step_response(0, 1)
        } else {
            reset_response(0, Some(1))
        };
        let worker = FakeWorker::boxed(0, Arc::default(), vec![response]);
        let mut backend: Backend = FiniteSubprocessBackend::new(vec![worker]).unwrap();
        let error = if reset {
            backend.reset_environments(&[0]).unwrap_err()
        } else {
            backend.step_environments(&[1], &[0]).unwrap_err()
        };
        assert!(matches!(
            error,
            FiniteSubprocessBackendError::UnexpectedReply {
                environment_id: 0,
                ..
            }
        ));
        assert!(error.to_string().contains("unexpected"));
    }

    let worker = FakeWorker::boxed(0, Arc::default(), vec![reset_response(0, Some(1))]);
    let mut backend: Backend = FiniteSubprocessBackend::new(vec![worker]).unwrap();
    assert!(matches!(
        backend.close(),
        Err(FiniteSubprocessBackendError::UnexpectedReply {
            operation: FiniteSubprocessBackendOperation::Close,
            ..
        })
    ));
}

#[test]
fn closed_state_rejects_vector_operations() {
    let worker = FakeWorker::boxed(0, Arc::default(), vec![close_response(0)]);
    let mut backend: Backend = FiniteSubprocessBackend::new(vec![worker]).unwrap();
    backend.close().unwrap();
    assert!(matches!(
        backend.reset_environments(&[]),
        Err(FiniteSubprocessBackendError::Closed)
    ));
    assert!(matches!(
        backend.step_environments(&[], &[]),
        Err(FiniteSubprocessBackendError::Closed)
    ));
}

#[test]
fn python_characterization_freezes_qlib_marker_and_tianshou_batch_contract() {
    let output = Command::new("python")
        .arg(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/finite_subprocess_backend_contract.py"),
        )
        .arg(r"D:\code\github\qlib\qlib\rl\utils\finite_env.py")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        value["bases"],
        serde_json::json!(["FiniteVectorEnv", "SubprocVectorEnv"])
    );
    assert_eq!(value["body"], serde_json::json!(["Pass"]));
    assert_eq!(
        value["reset_result"],
        serde_json::json!(["reset-1-0", "reset-0-0", "reset-0-1"])
    );
    assert_eq!(
        value["step_result"],
        serde_json::json!(["step-1-7", "step-0-8", "step-1-9"])
    );
    assert_eq!(value["action_mismatch"], "AssertionError");
    assert_eq!(
        value["close_events"],
        serde_json::json!([
            ["close-send", 0],
            ["close-recv", 0],
            ["join", 0],
            ["close-send", 1],
            ["close-recv", 1],
            ["join", 1]
        ])
    );
}
