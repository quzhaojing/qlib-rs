#![allow(clippy::float_cmp)]

use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    path::PathBuf,
    process::Command,
    sync::{Mutex, Weak},
};

use indexmap::IndexMap;
use serde_json::{Value, json};

use super::*;

type Runtime = RlTrainerRuntime<f64>;

#[path = "rl_trainer_driver_integration.rs"]
mod integration;

#[test]
fn native_seed_queue_context_preserves_entry_failures_and_non_suppressing_cleanup() {
    use crate::{DataQueue, DataQueueConfig, DataQueueError};

    let phase_error = RlTrainerDriverError::Plugin {
        stage: "train".into(),
        message: "phase failed".into(),
    };
    for error in [None, Some(&phase_error)] {
        let mut queue = DataQueue::new(
            Arc::new(vec![17_u32]),
            DataQueueConfig {
                repeat: -1,
                queue_maxsize: 1,
                shuffle: false,
                ..DataQueueConfig::default()
            },
        );
        assert!(!queue.is_activated());
        RlTrainerSeedContext::enter(&mut queue).unwrap();
        assert!(queue.is_activated());
        assert_eq!(queue.get().unwrap(), 17);
        assert!(!RlTrainerSeedContext::exit(&mut queue, error).unwrap());
        assert!(queue.done());
        assert!(matches!(queue.get(), Err(DataQueueError::Exhausted)));
        assert!(matches!(
            queue.put(18),
            Err(DataQueueError::Producer(
                crate::DataQueueProducerError::ConsumerDisconnected
            ))
        ));
        let failure = RlTrainerSeedContext::enter(&mut queue).unwrap_err();
        assert_eq!(
            failure,
            RlTrainerDriverError::Plugin {
                stage: "seed_enter".into(),
                message: DataQueueError::AlreadyActivated.to_string(),
            }
        );
    }
    let mut closed = DataQueue::new(Arc::new(vec![19_u32]), DataQueueConfig::default());
    closed.cleanup();
    assert_eq!(
        RlTrainerSeedContext::enter(&mut closed).unwrap_err(),
        RlTrainerDriverError::Plugin {
            stage: "seed_enter".into(),
            message: DataQueueError::Closed.to_string(),
        }
    );
}

#[test]
fn source_default_queue_context_enters_once_and_never_suppresses_phase_errors() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .args([
            root.join("tests/fixtures/rl_trainer_queue_context.py"),
            root.join("../../../qlib/qlib/rl/utils/data_queue.py"),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        cases,
        json!([
            {"events": ["activate", "body", "cleanup"], "identity": true, "failure": null},
            {"events": ["activate", "body", "cleanup"], "identity": true, "failure": "phase"},
            {"events": ["activate"], "identity": null, "failure": "entry"},
        ])
    );
}

#[test]
fn shared_seed_context_preserves_owner_cleanup_and_reports_poisoned_locks() {
    use crate::{DataQueue, DataQueueConfig, DataQueueError};
    let mut owner = Arc::new(Mutex::new(DataQueue::new(
        Arc::new(vec![7_i64]),
        DataQueueConfig {
            repeat: -1,
            queue_maxsize: 1,
            ..DataQueueConfig::default()
        },
    )));
    let consumer = owner.clone();
    RlTrainerSeedContext::enter(&mut owner).unwrap();
    assert_eq!(consumer.lock().unwrap().get().unwrap(), 7);
    assert!(matches!(RlTrainerSeedContext::enter(&mut owner),
        Err(RlTrainerDriverError::Plugin { stage, message })
            if stage == "seed_enter" && message == DataQueueError::AlreadyActivated.to_string()));
    let phase_error = RlTrainerDriverError::ZeroValidationInterval;
    assert!(!RlTrainerSeedContext::exit(&mut owner, Some(&phase_error)).unwrap());
    assert!(consumer.lock().unwrap().done());
    assert!(matches!(
        consumer.lock().unwrap().get(),
        Err(DataQueueError::Exhausted)
    ));

    let mut poisoned = Arc::new(Mutex::new(DataQueue::new(
        Arc::new(vec![1_u32]),
        DataQueueConfig::default(),
    )));
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _guard = poisoned.lock().unwrap();
            panic!("poison seed queue");
        }))
        .is_err()
    );
    let poison_message = poisoned.lock().err().unwrap().to_string();
    assert_eq!(
        RlTrainerSeedContext::enter(&mut poisoned).unwrap_err(),
        RlTrainerDriverError::Plugin {
            stage: "seed_enter".into(),
            message: poison_message.clone()
        }
    );
    assert_eq!(
        RlTrainerSeedContext::exit(&mut poisoned, None).unwrap_err(),
        RlTrainerDriverError::Plugin {
            stage: "seed_exit".into(),
            message: poison_message
        }
    );
}

struct Audit {
    spec: Value,
    runtime: Arc<Runtime>,
    events: Mutex<Vec<Value>>,
    environments: Mutex<Vec<Weak<()>>>,
}

fn snapshot(runtime: &Runtime) -> Value {
    runtime.read(|state| json!({
        "stage":state.current_stage,
        "iteration":state.current_iter.as_ref().map(ToString::to_string),
        "stop":state.should_stop,
        "metrics":state.metrics.as_ref().unwrap().iter().map(|(k,v)|json!([k,v])).collect::<Vec<_>>()
    })).unwrap()
}

impl Audit {
    fn flag(&self, name: &str) -> bool {
        self.spec[name] == true
    }
    fn live(&self) -> usize {
        self.environments
            .lock()
            .unwrap()
            .iter()
            .filter(|e| e.strong_count() > 0)
            .count()
    }
    fn record(&self, name: &str, extra: Value) -> Result<(), RlTrainerDriverError> {
        self.events.lock().unwrap().push(Value::Array(vec![
            json!(name),
            snapshot(&self.runtime),
            extra,
        ]));
        if self.spec["fail"] == name || self.spec["fail_exit"] == name {
            Err(RlTrainerDriverError::Plugin {
                stage: name.into(),
                message: "injected".into(),
            })
        } else {
            Ok(())
        }
    }
}

fn phase_name(phase: RlTrainerPhase) -> &'static str {
    match phase {
        RlTrainerPhase::Train => "train",
        RlTrainerPhase::Validation => "val",
        RlTrainerPhase::Test => "test",
    }
}
fn hook_name(hook: RlTrainerHook) -> &'static str {
    match hook {
        RlTrainerHook::FitStart => "on_fit_start",
        RlTrainerHook::FitEnd => "on_fit_end",
        RlTrainerHook::IterStart => "on_iter_start",
        RlTrainerHook::IterEnd => "on_iter_end",
        RlTrainerHook::TrainStart => "on_train_start",
        RlTrainerHook::TrainEnd => "on_train_end",
        RlTrainerHook::ValidateStart => "on_validate_start",
        RlTrainerHook::ValidateEnd => "on_validate_end",
        RlTrainerHook::TestStart => "on_test_start",
        RlTrainerHook::TestEnd => "on_test_end",
    }
}
fn error_name(error: &RlTrainerDriverError) -> String {
    match error {
        RlTrainerDriverError::Plugin { stage, .. } => stage.clone(),
        RlTrainerDriverError::ZeroValidationInterval => "zero_validation_interval".into(),
        RlTrainerDriverError::MissingState(field) => format!("missing:{field}"),
        other @ RlTrainerDriverError::State(_) => other.to_string(),
    }
}

struct Seed {
    audit: Arc<Audit>,
    phase: RlTrainerPhase,
}
impl RlTrainerSeedContext for Seed {
    fn enter(&mut self) -> Result<(), RlTrainerDriverError> {
        if self.audit.flag("plain") {
            Ok(())
        } else {
            self.audit
                .record(&format!("{}.enter", phase_name(self.phase)), Value::Null)
        }
    }
    fn exit(&mut self, error: Option<&RlTrainerDriverError>) -> Result<bool, RlTrainerDriverError> {
        if self.audit.flag("plain") {
            return Ok(false);
        }
        self.audit.record(
            &format!("{}.exit", phase_name(self.phase)),
            json!({"error":error.map(error_name),"live":self.audit.live()}),
        )?;
        Ok(self.audit.flag("suppress"))
    }
}
struct Vessel(Arc<Audit>);
impl RlTrainerVessel for Vessel {
    type Seed = Seed;
    type Environment = Arc<()>;
    fn assign_trainer(&mut self, runtime: &Arc<Runtime>) -> Result<(), RlTrainerDriverError> {
        assert!(Arc::ptr_eq(runtime, &self.0.runtime));
        self.0.record("assign", Value::Null)?;
        if self.0.flag("poison_assign") {
            poison(runtime);
        }
        Ok(())
    }
    fn seeds(&mut self, phase: RlTrainerPhase) -> Result<Seed, RlTrainerDriverError> {
        self.0
            .record(&format!("{}.seeds", phase_name(phase)), Value::Null)?;
        Ok(Seed {
            audit: self.0.clone(),
            phase,
        })
    }
    fn environment(
        &mut self,
        seed: &mut Seed,
        _control: &mut RlTrainerControl,
    ) -> Result<Arc<()>, RlTrainerDriverError> {
        self.0.record(
            &format!("{}.env", phase_name(seed.phase)),
            json!({"live":self.0.live()}),
        )?;
        let environment = Arc::new(());
        self.0
            .environments
            .lock()
            .unwrap()
            .push(Arc::downgrade(&environment));
        Ok(environment)
    }
    fn run(
        &mut self,
        phase: RlTrainerPhase,
        environment: &mut Arc<()>,
        control: &mut RlTrainerControl,
    ) -> Result<(), RlTrainerDriverError> {
        assert_eq!(Arc::strong_count(environment), 1);
        self.0
            .record(&format!("{}.run", phase_name(phase)), Value::Null)?;
        control.runtime.update(|state| {
            let prefix = if state.current_stage == "val" {
                "val/"
            } else {
                ""
            };
            state
                .metrics
                .as_mut()
                .unwrap()
                .insert(format!("{prefix}reward"), 3.0);
        })?;
        Ok(())
    }
}
struct Callback {
    audit: Arc<Audit>,
    index: usize,
}
impl RlTrainerCallback<Vessel> for Callback {
    fn call(
        &mut self,
        hook: RlTrainerHook,
        control: &mut RlTrainerControl,
        vessel: &mut Vessel,
    ) -> Result<(), RlTrainerDriverError> {
        assert!(Arc::ptr_eq(&self.audit, &vessel.0));
        let name = hook_name(hook);
        self.audit
            .record(&format!("{name}.{}", self.index), Value::Null)?;
        if self.index != 0 {
            return Ok(());
        }
        if self.audit.spec["clear"] == name {
            control
                .runtime
                .update(|state| match self.audit.spec["field"].as_str().unwrap() {
                    "current_iter" => state.current_iter = None,
                    "should_stop" => state.should_stop = None,
                    field => panic!("unexpected {field}"),
                })?;
        }
        if self.audit.spec["stop"] == name {
            control.runtime.update(|s| s.should_stop = Some(true))?;
        }
        if self.audit.flag("extend")
            && hook == RlTrainerHook::IterEnd
            && control.iteration()? == BigInt::from(1)
        {
            control.runtime.update(|s| s.should_stop = Some(false))?;
            control.config.max_iters = Some(2.into());
        }
        if self.audit.flag("change_val") && hook == RlTrainerHook::TrainEnd {
            control.config.val_every_n_iters = Some(2.into());
        }
        if self.audit.spec.get("max") == Some(&Value::Null)
            && hook == RlTrainerHook::IterEnd
            && control.iteration()? >= BigInt::from(2)
        {
            control.runtime.update(|s| s.should_stop = Some(true))?;
        }
        Ok(())
    }
}
struct Progress(Arc<Audit>);
impl RlTrainerProgress for Progress {
    fn iteration(
        &mut self,
        next: &BigInt,
        _maximum: Option<&BigInt>,
    ) -> Result<(), RlTrainerDriverError> {
        assert_eq!(
            *next,
            self.0
                .runtime
                .read(|s| s.current_iter.clone().unwrap() + 1)
                .unwrap()
        );
        self.0.record("progress", Value::Null)?;
        if self.0.flag("poison_progress") {
            poison(&self.0.runtime);
        }
        Ok(())
    }
}
struct Restore(Arc<Audit>);
impl RlTrainerRestore<Vessel> for Restore {
    fn restore(
        &mut self,
        control: &mut RlTrainerControl,
        _vessel: &mut Vessel,
    ) -> Result<(), RlTrainerDriverError> {
        self.0.record("restore", Value::Null)?;
        control.runtime.update(|state| {
            state.initialize();
            state.current_iter = Some(
                self.0.spec["resume_iter"]
                    .as_str()
                    .unwrap_or("1")
                    .parse()
                    .unwrap(),
            );
            state.current_stage = "val".into();
            state.should_stop = Some(self.0.flag("resume_stop"));
        })?;
        Ok(())
    }
}

fn setup(spec: Value) -> (RlTrainerDriver<Vessel>, Arc<Audit>) {
    let runtime = Arc::new(Runtime::new(None));
    runtime
        .update(|state| {
            state.current_stage = "old".into();
            state.metrics = Some(IndexMap::from([("old".into(), 9.0)]));
        })
        .unwrap();
    let audit = Arc::new(Audit {
        spec,
        runtime: runtime.clone(),
        events: Mutex::new(vec![]),
        environments: Mutex::new(vec![]),
    });
    let mut driver = RlTrainerDriver::new(
        Vessel(audit.clone()),
        runtime,
        RlTrainerConfig {
            max_iters: audit
                .spec
                .get("max")
                .unwrap_or(&json!(2))
                .as_i64()
                .map(BigInt::from),
            val_every_n_iters: audit
                .spec
                .get("val")
                .unwrap_or(&json!(1))
                .as_i64()
                .map(BigInt::from),
        },
    );
    for index in 0..audit.spec["callbacks"].as_u64().unwrap_or(2) {
        driver.callbacks.push(Box::new(Callback {
            audit: audit.clone(),
            index: usize::try_from(index).unwrap(),
        }));
    }
    driver.progress = Box::new(Progress(audit.clone()));
    (driver, audit)
}

#[test]
fn fit_test_callbacks_context_suppression_and_errors_match_live_python() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .args([
            root.join("tests/fixtures/rl_trainer_driver_contract.py"),
            root.join("../../../qlib/qlib/rl/trainer/trainer.py"),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
    assert!(cases.len() > 50);
    for case in cases {
        let (mut driver, audit) = setup(case["spec"].clone());
        let result = if audit.flag("test") {
            driver.test()
        } else if audit.flag("resume") {
            driver.fit(Some(&mut Restore(audit.clone())))
        } else {
            driver.fit(None)
        };
        let error = result.err().map(|e| {
            assert_eq!(e, e.clone());
            assert!(!format!("{e:?}").is_empty());
            assert!(!e.to_string().is_empty());
            error_name(&e)
        });
        assert_eq!(json!(error), case["error"], "{}", case["spec"]);
        assert_eq!(
            json!(*audit.events.lock().unwrap()),
            case["events"],
            "{}",
            case["spec"]
        );
        assert_eq!(snapshot(&audit.runtime), case["state"], "{}", case["spec"]);
        assert_eq!(audit.live(), 0);
    }
}

#[test]
fn default_plain_context_control_errors_and_tracing_are_explicit() {
    struct Plain;
    impl RlTrainerSeedContext for Plain {}
    Plain.enter().unwrap();
    assert!(!Plain.exit(None).unwrap());
    let error = RlTrainerDriverError::MissingState("current_iter");
    assert!(!Plain.exit(Some(&error)).unwrap());
    assert!(!error.to_string().is_empty());
    let runtime = Arc::new(Runtime::new(None));
    let mut control = RlTrainerControl {
        runtime: runtime.clone(),
        config: RlTrainerConfig::default(),
    };
    assert_eq!(control.config, control.config.clone());
    assert!(format!("{:?}", control.config).contains("max_iters"));
    assert_eq!(control.iteration(), Err(error.clone()));
    assert_eq!(control.finish_iteration(), Err(error.clone()));
    assert_eq!(
        control.stopped(),
        Err(RlTrainerDriverError::MissingState("should_stop"))
    );
    control.config.val_every_n_iters = Some(1.into());
    assert_eq!(control.begin_validation(), Err(error));
    runtime.update(crate::RlTrainerState::initialize).unwrap();
    TracingRlTrainerProgress.iteration(&1.into(), None).unwrap();
    let (mut driver, audit) = setup(json!({"max":1,"val":null}));
    driver.progress = Box::new(TracingRlTrainerProgress);
    driver.fit(None).unwrap();
    assert_eq!(driver.control.iteration().unwrap(), BigInt::from(1));
    driver.test().unwrap();
    assert_eq!(snapshot(&audit.runtime)["iteration"], "1");
    assert_eq!(snapshot(&audit.runtime)["stop"], true);
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _ = runtime.update::<()>(|_| panic!("poison"));
        }))
        .is_err()
    );
    let error = RlTrainerDriverError::State(RlTrainerStateError::Poisoned);
    assert_eq!(control.iteration(), Err(error.clone()));
    assert_eq!(control.stopped(), Err(error.clone()));
    assert_eq!(control.finish_iteration(), Err(error.clone()));
    assert_eq!(control.begin_validation(), Err(error.clone()));
    assert!(!error.to_string().is_empty());
}

fn poison(runtime: &Runtime) {
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _ = runtime.update::<()>(|_| panic!("injected runtime poison"));
        }))
        .is_err()
    );
}

struct PoisonCallback(RlTrainerHook);
impl RlTrainerCallback<Vessel> for PoisonCallback {
    fn call(
        &mut self,
        hook: RlTrainerHook,
        control: &mut RlTrainerControl,
        _vessel: &mut Vessel,
    ) -> Result<(), RlTrainerDriverError> {
        if hook == self.0 {
            poison(&control.runtime);
        }
        Ok(())
    }
}

#[test]
fn poisoned_runtime_aborts_each_lifecycle_transition_without_running_later_hooks() {
    let error = RlTrainerDriverError::State(RlTrainerStateError::Poisoned);
    for test in [false, true] {
        let (mut driver, audit) = setup(json!({"callbacks":0,"poison_assign":true}));
        let result = if test {
            driver.test()
        } else {
            driver.fit(None)
        };
        assert_eq!(result, Err(error.clone()));
        assert_eq!(audit.events.lock().unwrap().len(), 1);
    }
    let (mut driver, audit) = setup(json!({"callbacks":0,"poison_progress":true}));
    assert_eq!(driver.fit(None), Err(error.clone()));
    assert_eq!(audit.events.lock().unwrap().last().unwrap()[0], "progress");
    for hook in [RlTrainerHook::IterStart, RlTrainerHook::TrainEnd] {
        let (mut driver, audit) = setup(json!({"callbacks":0}));
        driver.callbacks.push(Box::new(PoisonCallback(hook)));
        assert_eq!(driver.fit(None), Err(error.clone()));
        let expected = if hook == RlTrainerHook::IterStart {
            "progress"
        } else {
            "train.exit"
        };
        assert_eq!(audit.events.lock().unwrap().last().unwrap()[0], expected);
        assert_eq!(audit.live(), 0);
    }
}

struct Capture(Arc<Mutex<Vec<(String, String)>>>);
impl tracing::Subscriber for Capture {
    fn enabled(&self, _metadata: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        event.record(
            &mut |field: &tracing::field::Field, value: &dyn std::fmt::Debug| {
                self.0
                    .lock()
                    .unwrap()
                    .push((field.name().into(), format!("{value:?}")));
            },
        );
    }
    fn enter(&self, _span: &tracing::span::Id) {}
    fn exit(&self, _span: &tracing::span::Id) {}
}

#[test]
fn default_progress_delivers_iteration_metadata_without_configuring_global_logging() {
    let fields = Arc::new(Mutex::new(vec![]));
    let _guard = tracing::subscriber::set_default(Capture(fields.clone()));
    TracingRlTrainerProgress
        .iteration(&3.into(), Some(&10.into()))
        .unwrap();
    assert_eq!(
        *fields.lock().unwrap(),
        vec![
            ("message".into(), "Train iteration".into()),
            ("iteration".into(), "3".into()),
            ("max_iters".into(), "Some(10)".into())
        ]
    );
}
