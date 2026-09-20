use std::{
    fmt::Write as _,
    path::PathBuf,
    process::Command,
    str::FromStr,
    sync::{Arc, Mutex},
    time::Duration,
};

use domain_core::{
    BoxedFiniteEnvironment, EnvironmentPluginError, FINITE_ENVIRONMENT_DESCRIPTOR_VERSION,
    FiniteBackendStep, FiniteDummyBackendPlugin, FiniteEnvironment, FiniteEnvironmentDescriptor,
    FiniteEnvironmentKind, FiniteObservationPredicate, FiniteShmemBackendPlugin,
    FiniteSubprocessBackendPlugin, FiniteSubprocessProgram, FiniteSubprocessReply,
    FiniteSubprocessResponse, FiniteVectorBackend, FiniteVectorBackendRegistry,
    FiniteVectorFactoryError, FiniteVectorLogger, encode_finite_subprocess_response, vectorize_env,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

type Observation = i64;
type Action = i32;
type Reward = f64;
type Info = i32;
type Backend = dyn FiniteVectorBackend<Observation, Action, Reward, Info>;
type Registry = FiniteVectorBackendRegistry<String, Observation, Action, Reward, Info>;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Config {
    seed: u64,
}

struct Predicate;

impl FiniteObservationPredicate<Observation> for Predicate {
    fn is_invalid(&mut self, _observation: &Observation) -> Result<bool, EnvironmentPluginError> {
        Ok(false)
    }
}

struct Worker {
    observation: Observation,
}

impl FiniteEnvironment<Observation, Action, Reward, Info> for Worker {
    fn reset(&mut self) -> Result<Observation, EnvironmentPluginError> {
        Ok(self.observation)
    }

    fn step(
        &mut self,
        action: &Action,
    ) -> Result<FiniteBackendStep<Observation, Reward, Info>, EnvironmentPluginError> {
        self.observation += i64::from(*action);
        Ok(FiniteBackendStep {
            observation: Some(self.observation),
            reward: Some(f64::from(*action)),
            done: false,
            info: Some(*action),
        })
    }
}

struct StaticBackend {
    environment_count: usize,
    base: Observation,
}

impl FiniteVectorBackend<Observation, Action, Reward, Info> for StaticBackend {
    fn environment_count(&self) -> usize {
        self.environment_count
    }

    fn reset(
        &mut self,
        environment_ids: &[usize],
    ) -> Result<Vec<Option<Observation>>, EnvironmentPluginError> {
        Ok(environment_ids
            .iter()
            .map(|id| Some(self.base + i64::try_from(*id).unwrap()))
            .collect())
    }

    fn step(
        &mut self,
        actions: &[Action],
        _environment_ids: &[usize],
    ) -> Result<Vec<FiniteBackendStep<Observation, Reward, Info>>, EnvironmentPluginError> {
        Ok(actions
            .iter()
            .map(|action| FiniteBackendStep {
                observation: Some(self.base + i64::from(*action)),
                reward: Some(f64::from(*action)),
                done: false,
                info: Some(*action),
            })
            .collect())
    }
}

struct Logger(Arc<Mutex<Vec<usize>>>);

impl FiniteVectorLogger<Observation, Reward, Info> for Logger {
    fn on_reset(
        &mut self,
        environment_id: usize,
        _all_observations: &[Option<Observation>],
    ) -> Result<(), EnvironmentPluginError> {
        self.0.lock().unwrap().push(environment_id);
        Ok(())
    }
}

fn boxed_worker(
    observation: Observation,
) -> BoxedFiniteEnvironment<Observation, Action, Reward, Info> {
    Box::new(Worker { observation })
}

fn plugin_error(message: &str) -> EnvironmentPluginError {
    EnvironmentPluginError::new(message)
}

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/finite_subprocess_transport.py")
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut output, byte| {
        write!(output, "{byte:02x}").unwrap();
        output
    })
}

fn python_contract() -> Value {
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/finite_dummy_contract.py");
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

#[test]
fn kinds_descriptors_and_live_python_selection_contract_are_stable() {
    let python = python_contract();
    assert_eq!(python["selected"], json!(["dummy", "subproc", "shmem"]));
    assert_eq!(python["factory_calls"], json!([3, 3, 3]));
    assert_eq!(python["same_factory_reference"], true);
    assert_eq!(python["same_logger_reference"], true);
    assert_eq!(python["invalid_key"], "invalid");

    for (text, kind) in [
        ("dummy", FiniteEnvironmentKind::Dummy),
        ("subproc", FiniteEnvironmentKind::Subproc),
        ("shmem", FiniteEnvironmentKind::Shmem),
    ] {
        assert_eq!(FiniteEnvironmentKind::from_str(text).unwrap(), kind);
        assert_eq!(kind.to_string(), text);
        assert_eq!(serde_json::to_string(&kind).unwrap(), format!("\"{text}\""));
        assert_eq!(
            serde_json::from_str::<FiniteEnvironmentKind>(&format!("\"{text}\"")).unwrap(),
            kind
        );
    }

    let descriptor = FiniteEnvironmentDescriptor::new("registered", Config { seed: 7 });
    assert_eq!(descriptor.version, FINITE_ENVIRONMENT_DESCRIPTOR_VERSION);
    assert_eq!(descriptor.factory, "registered");
    let encoded = serde_json::to_string(&descriptor).unwrap();
    assert_eq!(
        serde_json::from_str::<FiniteEnvironmentDescriptor<Config>>(&encoded).unwrap(),
        descriptor
    );
}

#[test]
fn registry_selects_plugins_in_stable_order_and_forwards_loggers() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let plugin_calls = Arc::clone(&calls);
    let mut registry = Registry::default();
    assert!(!registry.register(
        FiniteEnvironmentKind::Shmem,
        move |descriptor: &FiniteEnvironmentDescriptor<String>, concurrency| {
            plugin_calls.lock().unwrap().push((
                descriptor.factory.clone(),
                descriptor.config.clone(),
                concurrency,
            ));
            Ok(Box::new(StaticBackend {
                environment_count: concurrency,
                base: 70,
            }) as Box<Backend>)
        },
    ));
    assert!(!registry.register(
        FiniteEnvironmentKind::Dummy,
        |_descriptor: &FiniteEnvironmentDescriptor<String>, concurrency| {
            Ok(Box::new(StaticBackend {
                environment_count: concurrency,
                base: 0,
            }) as Box<Backend>)
        },
    ));
    assert_eq!(
        registry.registered_kinds().collect::<Vec<_>>(),
        [FiniteEnvironmentKind::Shmem, FiniteEnvironmentKind::Dummy]
    );
    assert!(registry.register(
        FiniteEnvironmentKind::Dummy,
        |_descriptor: &FiniteEnvironmentDescriptor<String>, concurrency| {
            Ok(Box::new(StaticBackend {
                environment_count: concurrency,
                base: 1,
            }) as Box<Backend>)
        },
    ));

    let logger_events = Arc::new(Mutex::new(Vec::new()));
    let descriptor = FiniteEnvironmentDescriptor::new("worker", "config".to_owned());
    let mut environment = vectorize_env(
        &descriptor,
        "shmem",
        2,
        &mut registry,
        Box::new(Predicate),
        vec![Box::new(Logger(Arc::clone(&logger_events)))],
    )
    .unwrap();
    assert_eq!(environment.environment_count(), 2);
    assert_eq!(
        environment.reset(None).unwrap().observations,
        [Some(70), Some(71)]
    );
    assert_eq!(*logger_events.lock().unwrap(), [0, 1]);
    assert_eq!(
        *calls.lock().unwrap(),
        [("worker".to_owned(), "config".to_owned(), 2)]
    );
}

#[test]
fn selector_validation_is_ordered_and_typed() {
    let mut registry = Registry::default();
    let descriptor = FiniteEnvironmentDescriptor::new("worker", String::new());
    let Err(error) = vectorize_env(
        &descriptor,
        "invalid",
        1,
        &mut registry,
        Box::new(Predicate),
        Vec::new(),
    ) else {
        panic!("invalid kind must fail");
    };
    assert!(matches!(
        error,
        FiniteVectorFactoryError::UnknownEnvironmentKind { .. }
    ));
    assert!(error.to_string().contains("invalid"));

    let mut version = descriptor.clone();
    version.version += 1;
    let Err(error) = vectorize_env(
        &version,
        "dummy",
        1,
        &mut registry,
        Box::new(Predicate),
        Vec::new(),
    ) else {
        panic!("unsupported version must fail");
    };
    assert!(matches!(
        error,
        FiniteVectorFactoryError::UnsupportedDescriptorVersion { .. }
    ));

    let empty = FiniteEnvironmentDescriptor::new("", String::new());
    let Err(error) = vectorize_env(
        &empty,
        "dummy",
        1,
        &mut registry,
        Box::new(Predicate),
        Vec::new(),
    ) else {
        panic!("empty factory must fail");
    };
    assert!(matches!(error, FiniteVectorFactoryError::EmptyFactoryName));

    let Err(error) = vectorize_env(
        &descriptor,
        "dummy",
        1,
        &mut registry,
        Box::new(Predicate),
        Vec::new(),
    ) else {
        panic!("missing backend must fail");
    };
    assert!(matches!(
        error,
        FiniteVectorFactoryError::MissingBackend {
            kind: FiniteEnvironmentKind::Dummy
        }
    ));
}

#[test]
fn dummy_plugin_builds_named_independent_workers() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let resolver_calls = Arc::clone(&calls);
    let mut plugin = FiniteDummyBackendPlugin::default();
    assert!(!plugin.register("counter", move |id, config: &String| {
        resolver_calls.lock().unwrap().push((id, config.clone()));
        Ok(boxed_worker(
            config.parse::<i64>().unwrap() + i64::try_from(id).unwrap(),
        ))
    }));
    assert!(plugin.register("counter", |id, config: &String| {
        Ok(boxed_worker(
            config.parse::<i64>().unwrap() + i64::try_from(id).unwrap(),
        ))
    }));
    let mut registry = Registry::default();
    assert!(!registry.register(FiniteEnvironmentKind::Dummy, plugin));
    let descriptor = FiniteEnvironmentDescriptor::new("counter", "10".to_owned());
    let mut environment = vectorize_env(
        &descriptor,
        "dummy",
        2,
        &mut registry,
        Box::new(Predicate),
        Vec::new(),
    )
    .unwrap();
    assert_eq!(
        environment.reset(None).unwrap().observations,
        [Some(10), Some(11)]
    );
    let step = environment.step(&[2, 3], None).unwrap();
    assert_eq!(step.transitions[0].observation, Some(12));
    assert_eq!(step.transitions[1].observation, Some(14));
}

#[test]
fn dummy_plugin_maps_missing_factory_worker_and_empty_pool_failures() {
    let descriptor = FiniteEnvironmentDescriptor::new("missing", String::new());
    let mut registry = Registry::default();
    assert!(!registry.register(
        FiniteEnvironmentKind::Dummy,
        FiniteDummyBackendPlugin::default(),
    ));
    let Err(error) = vectorize_env(
        &descriptor,
        "dummy",
        1,
        &mut registry,
        Box::new(Predicate),
        Vec::new(),
    ) else {
        panic!("missing factory must fail");
    };
    assert!(matches!(
        error,
        FiniteVectorFactoryError::MissingFactory { .. }
    ));

    let mut plugin = FiniteDummyBackendPlugin::default();
    assert!(!plugin.register("fail", |id, _config: &String| {
        if id == 1 {
            Err(plugin_error("worker"))
        } else {
            Ok(boxed_worker(0))
        }
    }));
    let mut registry = Registry::default();
    assert!(!registry.register(FiniteEnvironmentKind::Dummy, plugin));
    let descriptor = FiniteEnvironmentDescriptor::new("fail", String::new());
    let Err(error) = vectorize_env(
        &descriptor,
        "dummy",
        3,
        &mut registry,
        Box::new(Predicate),
        Vec::new(),
    ) else {
        panic!("worker failure must stop construction");
    };
    assert!(matches!(
        error,
        FiniteVectorFactoryError::WorkerFactory {
            kind: FiniteEnvironmentKind::Dummy,
            environment_id: 1,
            ..
        }
    ));
    assert!(error.to_string().contains("worker"));

    let mut plugin = FiniteDummyBackendPlugin::default();
    assert!(!plugin.register("empty", |_id, _config: &String| Ok(boxed_worker(0))));
    let mut registry = Registry::default();
    assert!(!registry.register(FiniteEnvironmentKind::Dummy, plugin));
    let descriptor = FiniteEnvironmentDescriptor::new("empty", String::new());
    let Err(error) = vectorize_env(
        &descriptor,
        "dummy",
        0,
        &mut registry,
        Box::new(Predicate),
        Vec::new(),
    ) else {
        panic!("empty pool must fail");
    };
    assert!(matches!(error, FiniteVectorFactoryError::DummyBuild(_)));
}

#[test]
fn subprocess_plugin_resolves_named_programs_and_runs_reset() {
    type Response = FiniteSubprocessResponse<Observation, Reward, Info, String>;
    let mut plugin =
        FiniteSubprocessBackendPlugin::<String, Action, Observation, Reward, Info, String>::new(
            4096,
            Duration::from_secs(2),
        );
    assert!(!plugin.register("python", |id, config: &String| {
        let response = Response::new(
            u64::try_from(id).unwrap(),
            FiniteSubprocessReply::Reset {
                observation: Some(config.parse::<i64>().unwrap() + i64::try_from(id).unwrap()),
                info: None,
            },
        );
        let frame = encode_finite_subprocess_response(&response, 4096).unwrap();
        Ok(FiniteSubprocessProgram::new("python")
            .args([
                fixture().into_os_string(),
                "replies".into(),
                hex(&frame).into(),
            ])
            .current_dir(env!("CARGO_MANIFEST_DIR")))
    }));
    assert!(plugin.register("python", |id, config: &String| {
        let response = Response::new(
            u64::try_from(id).unwrap(),
            FiniteSubprocessReply::Reset {
                observation: Some(config.parse::<i64>().unwrap() + i64::try_from(id).unwrap()),
                info: None,
            },
        );
        let frame = encode_finite_subprocess_response(&response, 4096).unwrap();
        Ok(FiniteSubprocessProgram::new("python")
            .args([
                fixture().into_os_string(),
                "replies".into(),
                hex(&frame).into(),
            ])
            .current_dir(env!("CARGO_MANIFEST_DIR")))
    }));
    let mut registry = Registry::default();
    assert!(!registry.register(FiniteEnvironmentKind::Subproc, plugin));
    let descriptor = FiniteEnvironmentDescriptor::new("python", "20".to_owned());
    let mut environment = vectorize_env(
        &descriptor,
        "subproc",
        2,
        &mut registry,
        Box::new(Predicate),
        Vec::new(),
    )
    .unwrap();
    assert_eq!(
        environment.reset(None).unwrap().observations,
        [Some(20), Some(21)]
    );
}

#[test]
fn subprocess_plugin_maps_registry_resolution_and_spawn_failures() {
    type Plugin = FiniteSubprocessBackendPlugin<String, Action, Observation, Reward, Info, String>;
    let descriptor = FiniteEnvironmentDescriptor::new("missing", String::new());
    let mut registry = Registry::default();
    assert!(!registry.register(
        FiniteEnvironmentKind::Subproc,
        Plugin::new(4096, Duration::from_secs(1)),
    ));
    let Err(error) = vectorize_env(
        &descriptor,
        "subproc",
        1,
        &mut registry,
        Box::new(Predicate),
        Vec::new(),
    ) else {
        panic!("missing resolver must fail");
    };
    assert!(matches!(
        error,
        FiniteVectorFactoryError::MissingFactory { .. }
    ));

    let mut plugin = Plugin::new(4096, Duration::from_secs(1));
    assert!(!plugin.register("fail", |id, _config: &String| {
        if id == 1 {
            Err(plugin_error("resolver"))
        } else {
            Ok(FiniteSubprocessProgram::new("python"))
        }
    }));
    let mut registry = Registry::default();
    assert!(!registry.register(FiniteEnvironmentKind::Subproc, plugin));
    let descriptor = FiniteEnvironmentDescriptor::new("fail", String::new());
    let Err(error) = vectorize_env(
        &descriptor,
        "subproc",
        2,
        &mut registry,
        Box::new(Predicate),
        Vec::new(),
    ) else {
        panic!("resolver failure must fail");
    };
    assert!(matches!(
        error,
        FiniteVectorFactoryError::WorkerFactory {
            kind: FiniteEnvironmentKind::Subproc,
            environment_id: 1,
            ..
        }
    ));

    let mut plugin = Plugin::new(4096, Duration::from_secs(1));
    assert!(!plugin.register("invalid", |_id, _config: &String| {
        Ok(FiniteSubprocessProgram::new(
            "definitely-not-a-real-qlib-worker",
        ))
    }));
    let mut registry = Registry::default();
    assert!(!registry.register(FiniteEnvironmentKind::Subproc, plugin));
    let descriptor = FiniteEnvironmentDescriptor::new("invalid", String::new());
    let Err(error) = vectorize_env(
        &descriptor,
        "subproc",
        1,
        &mut registry,
        Box::new(Predicate),
        Vec::new(),
    ) else {
        panic!("spawn failure must fail");
    };
    assert!(matches!(
        error,
        FiniteVectorFactoryError::SubprocessBuild(_)
    ));

    let mut plugin = Plugin::new(4096, Duration::from_secs(1));
    assert!(!plugin.register("empty", |_id, _config: &String| {
        Ok(FiniteSubprocessProgram::new("python"))
    }));
    let mut registry = Registry::default();
    assert!(!registry.register(FiniteEnvironmentKind::Subproc, plugin));
    let descriptor = FiniteEnvironmentDescriptor::new("empty", String::new());
    let Err(error) = vectorize_env(
        &descriptor,
        "subproc",
        0,
        &mut registry,
        Box::new(Predicate),
        Vec::new(),
    ) else {
        panic!("empty subprocess pool must fail");
    };
    assert!(matches!(
        error,
        FiniteVectorFactoryError::SubprocessBuild(_)
    ));
}

#[test]
fn shared_memory_plugin_registers_resolves_runs_and_maps_failures() {
    type Plugin = FiniteShmemBackendPlugin<String, Action, Observation, Reward, Info, u32>;
    let timeout = Duration::from_secs(2);

    let mut missing_registry = Registry::default();
    assert!(!missing_registry.register(
        FiniteEnvironmentKind::Shmem,
        Plugin::new(128, 4096, timeout),
    ));
    let descriptor = FiniteEnvironmentDescriptor::new("missing", String::new());
    let missing = vectorize_env(
        &descriptor,
        "shmem",
        1,
        &mut missing_registry,
        Box::new(Predicate),
        Vec::new(),
    );
    assert!(matches!(
        missing,
        Err(FiniteVectorFactoryError::MissingFactory { .. })
    ));

    let mut resolver_failure = Plugin::new(128, 4096, timeout);
    assert!(
        !resolver_failure.register("failed", |_id, _config: &String| {
            Err(plugin_error("shmem resolver"))
        })
    );
    assert!(resolver_failure.register("failed", |id, _config: &String| {
        if id == 1 {
            Err(plugin_error("shmem resolver"))
        } else {
            Ok(FiniteSubprocessProgram::new("python"))
        }
    }));
    let mut registry = Registry::default();
    assert!(!registry.register(FiniteEnvironmentKind::Shmem, resolver_failure));
    let descriptor = FiniteEnvironmentDescriptor::new("failed", String::new());
    let failure = vectorize_env(
        &descriptor,
        "shmem",
        2,
        &mut registry,
        Box::new(Predicate),
        Vec::new(),
    );
    assert!(matches!(
        failure,
        Err(FiniteVectorFactoryError::WorkerFactory {
            kind: FiniteEnvironmentKind::Shmem,
            environment_id: 1,
            ..
        })
    ));

    let mut invalid = Plugin::new(0, 4096, timeout);
    assert!(!invalid.register("invalid", |_id, _config: &String| {
        Ok(FiniteSubprocessProgram::new("python"))
    }));
    let mut registry = Registry::default();
    assert!(!registry.register(FiniteEnvironmentKind::Shmem, invalid));
    let descriptor = FiniteEnvironmentDescriptor::new("invalid", String::new());
    let build = vectorize_env(
        &descriptor,
        "shmem",
        1,
        &mut registry,
        Box::new(Predicate),
        Vec::new(),
    );
    assert!(matches!(
        build,
        Err(FiniteVectorFactoryError::SharedMemoryBuild(_))
    ));
}

#[test]
fn shared_memory_plugin_runs_real_registered_process() {
    type Plugin = FiniteShmemBackendPlugin<String, Action, Observation, Reward, Info, u32>;
    type ControlResponse = FiniteSubprocessResponse<(), Reward, Info, u32>;
    let timeout = Duration::from_secs(2);
    let control = ControlResponse::new(
        0,
        FiniteSubprocessReply::Reset {
            observation: None,
            info: None,
        },
    );
    let frame = encode_finite_subprocess_response(&control, 4096).unwrap();
    let mut plugin = Plugin::new(128, 4096, timeout);
    assert!(!plugin.register("python", move |_id, _config: &String| {
        Ok(FiniteSubprocessProgram::new("python")
            .args([
                fixture().into_os_string(),
                "shmem_replies".into(),
                format!("33:{}", hex(&frame)).into(),
            ])
            .current_dir(env!("CARGO_MANIFEST_DIR")))
    }));
    let mut registry = Registry::default();
    assert!(!registry.register(FiniteEnvironmentKind::Shmem, plugin));
    let descriptor = FiniteEnvironmentDescriptor::new("python", String::new());
    let mut environment = vectorize_env(
        &descriptor,
        "shmem",
        1,
        &mut registry,
        Box::new(Predicate),
        Vec::new(),
    )
    .unwrap();
    assert_eq!(environment.reset(None).unwrap().observations, [Some(33)]);
}
