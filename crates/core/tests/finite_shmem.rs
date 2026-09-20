use std::{
    collections::VecDeque,
    fs,
    future::Future,
    path::PathBuf,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use bincode::{DefaultOptions, Options};
use domain_core::{
    FiniteBackendStep, FiniteShmemBackend, FiniteShmemBuildError, FiniteShmemWorker,
    FiniteSubprocessBackendBuildError, FiniteSubprocessExitStatus, FiniteSubprocessFailure,
    FiniteSubprocessFailureKind, FiniteSubprocessProgram, FiniteSubprocessReply,
    FiniteSubprocessResponse, FiniteSubprocessRuntimeError, FiniteSubprocessWorker,
    FiniteVectorBackend, decode_finite_subprocess_response, encode_finite_subprocess_response,
};
use shmem::MappedObservationRegion;

type Response = FiniteSubprocessResponse<(), f64, i32, u32>;
type HydratedResponse = FiniteSubprocessResponse<i64, f64, i32, u32>;
type TestWorker = FiniteShmemWorker<i64, f64, i32, u32>;
type BuiltWorker = (
    TestWorker,
    MappedObservationRegion,
    Arc<Mutex<Vec<Event>>>,
    PathBuf,
);

static NEXT_PATH_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, PartialEq, Eq)]
enum Event {
    Send(Vec<u8>),
    Receive,
    CloseInput,
    Wait,
    Terminate,
}

struct Worker {
    limit: usize,
    responses: VecDeque<Result<Vec<u8>, FiniteSubprocessRuntimeError>>,
    events: Arc<Mutex<Vec<Event>>>,
}

impl FiniteSubprocessWorker for Worker {
    fn frame_limit(&self) -> usize {
        self.limit
    }

    fn send_frame(
        &mut self,
        frame: Vec<u8>,
    ) -> Pin<Box<dyn Future<Output = Result<(), FiniteSubprocessRuntimeError>> + Send + '_>> {
        self.events.lock().unwrap().push(Event::Send(frame));
        Box::pin(async { Ok(()) })
    }

    fn receive_frame(
        &mut self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<u8>, FiniteSubprocessRuntimeError>> + Send + '_>>
    {
        self.events.lock().unwrap().push(Event::Receive);
        let response = self.responses.pop_front().unwrap();
        Box::pin(async move { response })
    }

    fn close_input(
        &mut self,
    ) -> Pin<Box<dyn Future<Output = Result<(), FiniteSubprocessRuntimeError>> + Send + '_>> {
        self.events.lock().unwrap().push(Event::CloseInput);
        Box::pin(async { Ok(()) })
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
        self.events.lock().unwrap().push(Event::Wait);
        Box::pin(async { Ok(FiniteSubprocessExitStatus::new(true, Some(0))) })
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
        self.events.lock().unwrap().push(Event::Terminate);
        Box::pin(async { Ok(FiniteSubprocessExitStatus::new(false, None)) })
    }
}

fn path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "core-shmem-{name}-{}-{}",
        std::process::id(),
        NEXT_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ))
}

fn response(request_id: u64, reply: FiniteSubprocessReply<(), f64, i32, u32>) -> Vec<u8> {
    encode_finite_subprocess_response(&Response::new(request_id, reply), 4096).unwrap()
}

fn payload(observation: i64) -> Vec<u8> {
    DefaultOptions::new()
        .with_fixint_encoding()
        .with_little_endian()
        .reject_trailing_bytes()
        .serialize(&observation)
        .unwrap()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut output, byte| {
        use std::fmt::Write;
        write!(output, "{byte:02x}").unwrap();
        output
    })
}

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/finite_subprocess_transport.py")
}

fn build(
    name: &str,
    limit: usize,
    responses: Vec<Result<Vec<u8>, FiniteSubprocessRuntimeError>>,
) -> BuiltWorker {
    let path = path(name);
    let _ = fs::remove_file(&path);
    let writer = MappedObservationRegion::create(&path, 128).unwrap();
    let reader = MappedObservationRegion::open(&path, 128).unwrap();
    let events = Arc::new(Mutex::new(Vec::new()));
    let worker = Worker {
        limit,
        responses: responses.into(),
        events: Arc::clone(&events),
    };
    (
        FiniteShmemWorker::new(Box::new(worker), reader),
        writer,
        events,
        path,
    )
}

fn receive(
    runtime: &tokio::runtime::Runtime,
    worker: &mut TestWorker,
) -> Result<HydratedResponse, FiniteSubprocessRuntimeError> {
    let frame = runtime.block_on(worker.receive_frame())?;
    decode_finite_subprocess_response(&frame, worker.frame_limit()).map_err(Into::into)
}

#[test]
fn hydrates_reset_and_step_and_delegates_lifecycle() {
    let reset = FiniteSubprocessReply::Reset {
        observation: None,
        info: Some(4),
    };
    let step = FiniteSubprocessReply::Step {
        transition: FiniteBackendStep {
            observation: None,
            reward: Some(1.5),
            done: true,
            info: Some(8),
        },
    };
    let (mut worker, mut writer, events, path) = build(
        "happy",
        4096,
        vec![Ok(response(7, reset)), Ok(response(8, step))],
    );
    assert_eq!(worker.observation_path(), path);
    assert_eq!(worker.frame_limit(), 4096);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();

    writer.write_frame(7, &payload(11)).unwrap();
    assert_eq!(
        receive(&runtime, &mut worker).unwrap().reply,
        FiniteSubprocessReply::Reset {
            observation: Some(11),
            info: Some(4)
        }
    );
    writer.write_frame(8, &payload(12)).unwrap();
    assert_eq!(
        receive(&runtime, &mut worker).unwrap().reply,
        FiniteSubprocessReply::Step {
            transition: FiniteBackendStep {
                observation: Some(12),
                reward: Some(1.5),
                done: true,
                info: Some(8)
            }
        }
    );

    runtime.block_on(worker.send_frame(vec![1, 2])).unwrap();
    runtime.block_on(worker.close_input()).unwrap();
    assert_eq!(
        runtime.block_on(worker.wait_for_exit()).unwrap(),
        FiniteSubprocessExitStatus::new(true, Some(0))
    );
    assert_eq!(
        runtime.block_on(worker.terminate()).unwrap(),
        FiniteSubprocessExitStatus::new(false, None)
    );
    assert_eq!(
        *events.lock().unwrap(),
        vec![
            Event::Receive,
            Event::Receive,
            Event::Send(vec![1, 2]),
            Event::CloseInput,
            Event::Wait,
            Event::Terminate
        ]
    );
    drop(worker);
    drop(writer);
    fs::remove_file(path).unwrap();
}

#[test]
fn passes_non_observation_replies_without_touching_mapping() {
    let failure = FiniteSubprocessFailure {
        kind: FiniteSubprocessFailureKind::Environment,
        message: "failed".to_owned(),
    };
    let replies = [
        FiniteSubprocessReply::Close { result: Some(1) },
        FiniteSubprocessReply::Render { result: Some(2) },
        FiniteSubprocessReply::Seed { result: Some(3) },
        FiniteSubprocessReply::Attribute { value: Some(4) },
        FiniteSubprocessReply::Failure(failure.clone()),
    ];
    let expected: [FiniteSubprocessReply<i64, f64, i32, u32>; 5] = [
        FiniteSubprocessReply::Close { result: Some(1) },
        FiniteSubprocessReply::Render { result: Some(2) },
        FiniteSubprocessReply::Seed { result: Some(3) },
        FiniteSubprocessReply::Attribute { value: Some(4) },
        FiniteSubprocessReply::Failure(failure),
    ];
    let frames = replies
        .iter()
        .cloned()
        .enumerate()
        .map(|(id, reply)| Ok(response(id as u64, reply)))
        .collect();
    let (mut worker, writer, _, path) = build("passthrough", 4096, frames);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    for expected in expected {
        assert_eq!(receive(&runtime, &mut worker).unwrap().reply, expected);
    }
    drop(worker);
    drop(writer);
    fs::remove_file(path).unwrap();
}

#[test]
fn rejects_inline_stale_missing_and_malformed_observations() {
    let inline_reset = FiniteSubprocessReply::Reset {
        observation: Some(()),
        info: None,
    };
    let inline_step = FiniteSubprocessReply::Step {
        transition: FiniteBackendStep {
            observation: Some(()),
            reward: None,
            done: false,
            info: None,
        },
    };
    let reset = FiniteSubprocessReply::Reset {
        observation: None,
        info: None,
    };
    let shared_step = FiniteSubprocessReply::Step {
        transition: FiniteBackendStep {
            observation: None,
            reward: None,
            done: false,
            info: None,
        },
    };
    let frames = vec![
        Ok(vec![1]),
        Ok(response(1, inline_reset)),
        Ok(response(2, inline_step)),
        Ok(response(3, shared_step)),
        Ok(response(4, reset.clone())),
        Ok(response(5, reset.clone())),
        Ok(response(6, reset)),
    ];
    let (mut worker, mut writer, _, path) = build("invalid", 4096, frames);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    assert!(matches!(
        runtime.block_on(worker.receive_frame()),
        Err(FiniteSubprocessRuntimeError::Wire(_))
    ));
    assert!(matches!(
        runtime.block_on(worker.receive_frame()),
        Err(FiniteSubprocessRuntimeError::InlineSharedObservation)
    ));
    assert!(matches!(
        runtime.block_on(worker.receive_frame()),
        Err(FiniteSubprocessRuntimeError::InlineSharedObservation)
    ));
    assert!(matches!(
        runtime.block_on(worker.receive_frame()),
        Err(FiniteSubprocessRuntimeError::SharedMemory(_))
    ));
    assert!(matches!(
        runtime.block_on(worker.receive_frame()),
        Err(FiniteSubprocessRuntimeError::SharedMemory(_))
    ));
    writer.write_frame(99, &payload(1)).unwrap();
    assert!(matches!(
        runtime.block_on(worker.receive_frame()),
        Err(FiniteSubprocessRuntimeError::SharedObservationIdMismatch {
            expected: 5,
            actual: 99
        })
    ));
    writer.write_frame(6, b"bad").unwrap();
    assert!(matches!(
        runtime.block_on(worker.receive_frame()),
        Err(FiniteSubprocessRuntimeError::SharedObservationDecode(_))
    ));
    drop(worker);
    drop(writer);
    fs::remove_file(path).unwrap();
}

#[test]
fn propagates_control_and_reencoded_frame_failures() {
    let (mut failed, writer, _, path) = build(
        "control-error",
        4096,
        vec![Err(FiniteSubprocessRuntimeError::Eof)],
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    assert!(matches!(
        runtime.block_on(failed.receive_frame()),
        Err(FiniteSubprocessRuntimeError::Eof)
    ));
    drop(failed);
    drop(writer);
    fs::remove_file(path).unwrap();

    let reset = FiniteSubprocessReply::Reset {
        observation: None,
        info: None,
    };
    let control = response(6, reset);
    let (mut oversized, mut writer, _, path) = build("oversized", control.len(), vec![Ok(control)]);
    writer.write_frame(6, &payload(i64::MAX)).unwrap();
    assert!(matches!(
        runtime.block_on(oversized.receive_frame()),
        Err(FiniteSubprocessRuntimeError::Wire(_))
    ));
    drop(oversized);
    drop(writer);
    fs::remove_file(path).unwrap();
}

#[test]
fn production_backend_runs_real_shared_memory_process_and_cleans_owner() {
    let reset = response(
        0,
        FiniteSubprocessReply::Reset {
            observation: None,
            info: Some(4),
        },
    );
    let step = response(
        1,
        FiniteSubprocessReply::Step {
            transition: FiniteBackendStep {
                observation: None,
                reward: Some(2.5),
                done: true,
                info: Some(8),
            },
        },
    );
    let close = response(2, FiniteSubprocessReply::Close { result: None });
    let program = FiniteSubprocessProgram::new("python")
        .args([
            fixture().into_os_string(),
            "shmem_replies".into(),
            format!("41:{}", hex(&reset)).into(),
            format!("42:{}", hex(&step)).into(),
            format!("0:{}", hex(&close)).into(),
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"));
    let mut backend = FiniteShmemBackend::<i32, i64, f64, i32, u32>::from_programs(
        vec![program],
        128,
        4096,
        std::time::Duration::from_secs(2),
    )
    .unwrap();
    let owner = backend.owner_directory().to_path_buf();
    assert!(owner.exists());
    assert_eq!(backend.environment_count(), 1);
    assert_eq!(
        FiniteVectorBackend::reset(&mut backend, &[0]).unwrap(),
        vec![Some(41)]
    );
    assert_eq!(
        FiniteVectorBackend::step(&mut backend, &[5], &[0]).unwrap(),
        vec![FiniteBackendStep {
            observation: Some(42),
            reward: Some(2.5),
            done: true,
            info: Some(8)
        }]
    );
    assert!(backend.close().unwrap()[0].success());
    drop(backend);
    assert!(!owner.exists());
}

#[test]
fn production_backend_maps_every_construction_failure() {
    let timeout = std::time::Duration::from_secs(1);
    let empty =
        FiniteShmemBackend::<i32, i64, f64, i32, u32>::from_programs(Vec::new(), 8, 1024, timeout);
    assert!(matches!(
        empty,
        Err(FiniteShmemBuildError::Backend(
            FiniteSubprocessBackendBuildError::Empty
        ))
    ));

    let eof =
        FiniteSubprocessProgram::new("python").args([fixture().into_os_string(), "eof".into()]);
    let zero =
        FiniteShmemBackend::<i32, i64, f64, i32, u32>::from_programs(vec![eof], 0, 1024, timeout);
    assert!(matches!(
        zero,
        Err(FiniteShmemBuildError::Region {
            environment_id: 0,
            ..
        })
    ));

    let invalid = FiniteShmemBackend::<i32, i64, f64, i32, u32>::from_programs(
        vec![FiniteSubprocessProgram::new(
            "definitely-not-a-real-qlib-worker",
        )],
        8,
        1024,
        timeout,
    );
    assert!(matches!(
        invalid,
        Err(FiniteShmemBuildError::Worker {
            environment_id: 0,
            ..
        })
    ));

    let partial = FiniteShmemBackend::<i32, i64, f64, i32, u32>::from_programs(
        vec![
            FiniteSubprocessProgram::new("python").args([fixture().into_os_string(), "eof".into()]),
            FiniteSubprocessProgram::new("definitely-not-a-real-qlib-worker"),
        ],
        8,
        1024,
        timeout,
    );
    assert!(matches!(
        partial,
        Err(FiniteShmemBuildError::Worker {
            environment_id: 1,
            ..
        })
    ));

    let root_file = path("not-directory");
    let _ = fs::remove_file(&root_file);
    fs::write(&root_file, b"file").unwrap();
    let directory = FiniteShmemBackend::<i32, i64, f64, i32, u32>::from_programs_in(
        root_file.clone(),
        Vec::new(),
        8,
        1024,
        timeout,
    );
    assert!(matches!(
        directory,
        Err(FiniteShmemBuildError::Directory(_))
    ));
    fs::remove_file(root_file).unwrap();
}
