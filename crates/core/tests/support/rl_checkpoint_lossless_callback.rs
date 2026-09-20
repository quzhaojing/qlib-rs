use super::*;
use crate::{PythonRlLosslessCheckpointName, RlLosslessCheckpointValue, RlTrainerConfig};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

struct Clock;
impl RlCheckpointClock for Clock {
    fn timestamp(&mut self) -> Result<f64, String> {
        Ok(123.0)
    }
    fn local_time(&mut self) -> Result<String, String> {
        Ok("20260903123456".into())
    }
}

#[derive(Clone)]
struct Graph(Arc<Mutex<Vec<&'static str>>>);
impl RlCheckpointGraph<(), RlLosslessCheckpointValue> for Graph {
    type State = Vec<u8>;
    fn collect(
        &mut self,
        _control: &mut RlTrainerControl<RlLosslessCheckpointValue>,
        _vessel: &mut (),
    ) -> Result<Self::State, String> {
        self.0.lock().unwrap().push("graph");
        Ok(b"checkpoint-payload".to_vec())
    }
}

struct Files;
impl RlCheckpointStorage<Vec<u8>> for Files {
    fn create_directory(&mut self, path: &Path) -> Result<(), String> {
        fs::create_dir_all(path).map_err(|error| error.to_string())
    }
    fn save(&mut self, state: &Vec<u8>, path: &Path) -> Result<(), String> {
        fs::write(path, state).map_err(|error| error.to_string())
    }
    fn exists(&mut self, path: &Path) -> Result<bool, String> {
        Ok(path.exists())
    }
    fn is_link(&mut self, path: &Path) -> Result<bool, String> {
        Ok(path.is_symlink())
    }
    fn remove(&mut self, path: &Path) -> Result<(), String> {
        fs::remove_file(path).map_err(|error| error.to_string())
    }
    fn link(&mut self, target: &Path, link: &Path) -> Result<(), String> {
        std::os::windows::fs::symlink_file(target, link).map_err(|error| error.to_string())
    }
    fn copy(&mut self, source: &Path, destination: &Path) -> Result<(), String> {
        fs::copy(source, destination)
            .map(|_| ())
            .map_err(|error| error.to_string())
    }
}

fn raw(points: &serde_json::Value) -> RlCheckpointText {
    RlCheckpointText::try_from_code_points(
        points
            .as_array()
            .unwrap()
            .iter()
            .map(|point| u32::try_from(point.as_u64().unwrap()).unwrap()),
    )
    .unwrap()
}

#[test]
fn all_twenty_four_unchanged_qlib_save_cases_preserve_state_and_files() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = std::process::Command::new("python")
        .args([
            root.join("../../scripts/checkpoint-format-probe/surrogate_probe.py"),
            root.join("../../../qlib/qlib/rl/trainer/callbacks.py"),
        ])
        .arg("--save-oracle-only")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Vec<serde_json::Value> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(cases.len(), 24);
    for case in cases {
        let name = raw(&case["name"]);
        let mode = (!case["latest_mode"].is_null()).then(|| raw(&case["latest_mode"]));
        let directory = tempfile::tempdir().unwrap();
        let latest = directory.path().join("latest.pth");
        fs::write(&latest, b"old").unwrap();
        let runtime = Arc::new(RlTrainerRuntime::new(None));
        runtime
            .update(|state| {
                state.current_iter = Some(7.into());
                state.metrics = Some(IndexMap::new());
            })
            .unwrap();
        let mut control = RlTrainerControl {
            runtime,
            config: RlTrainerConfig::default(),
        };
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut config = RlLosslessCheckpointConfig::new(directory.path());
        config.filename = name.clone();
        config.save_latest = mode.clone();
        let mut callback = RlLosslessCheckpointCallback::new(
            config,
            Clock,
            PythonRlLosslessCheckpointName,
            Graph(events.clone()),
            Files,
        );
        let result = callback.save(&mut control, &mut ());
        assert_eq!(result.is_err(), !case["error"].is_null(), "{case}");
        assert_eq!(callback.state.last_name, Some(raw(&case["last_name"])));
        assert_eq!(
            callback.state.last_iter,
            Some(case["last_iter"].as_i64().unwrap().into())
        );
        assert_eq!(callback.state.last_time, case["last_time"].as_f64());
        assert_eq!(*events.lock().unwrap(), vec!["graph"]);
        let latest_bytes = latest.exists().then(|| fs::read(&latest).unwrap());
        let expected = case["latest"]
            .as_str()
            .map(|value| value.as_bytes().to_vec());
        assert_eq!(latest_bytes, expected);
        if result.is_ok() {
            assert_eq!(
                fs::read(name.join_to(directory.path()).unwrap()).unwrap(),
                b"checkpoint-payload"
            );
        }
    }
}

#[test]
fn established_utf8_configuration_remains_unchanged() {
    let old = crate::RlCheckpointConfig::new("root");
    assert_eq!(old.filename, "{iter:03d}.pth");
    assert_eq!(old.save_latest.as_deref(), Some("link"));
    let new = RlLosslessCheckpointConfig::new("root");
    assert_eq!(new.filename.to_utf8().unwrap(), old.filename);
    assert_eq!(new.save_latest.unwrap().to_utf8().unwrap(), "link");
    let _ = RlLosslessCheckpointConfig::new(PathBuf::from("root"));
}

fn callback_with(
    directory: &Path,
) -> (
    RlLosslessCheckpointCallback<Clock, PythonRlLosslessCheckpointName, Graph, Files>,
    RlTrainerControl<RlLosslessCheckpointValue>,
) {
    let runtime = Arc::new(RlTrainerRuntime::new(None));
    runtime
        .update(|state| {
            state.current_iter = Some(0.into());
            state.metrics = Some(IndexMap::new());
        })
        .unwrap();
    let control = RlTrainerControl {
        runtime,
        config: RlTrainerConfig::default(),
    };
    let callback = RlLosslessCheckpointCallback::new(
        RlLosslessCheckpointConfig::new(directory),
        Clock,
        PythonRlLosslessCheckpointName,
        Graph(Arc::new(Mutex::new(Vec::new()))),
        Files,
    );
    (callback, control)
}

#[test]
fn scheduling_trait_dispatch_and_link_cover_the_complete_additive_callback() {
    let directory = tempfile::tempdir().unwrap();
    let (mut callback, mut control) = callback_with(directory.path());
    callback
        .on_hook(RlTrainerHook::FitStart, &mut control, &mut ())
        .unwrap();
    callback
        .on_hook(RlTrainerHook::IterEnd, &mut control, &mut ())
        .unwrap();
    callback.config.every_n_iters = Some(0.into());
    assert_eq!(
        callback
            .on_hook(RlTrainerHook::IterEnd, &mut control, &mut ())
            .unwrap_err(),
        RlCheckpointCallbackError::ZeroIterationInterval
    );
    callback.config.every_n_iters = Some(1.into());
    callback.config.save_latest = Some("link".into());
    callback
        .on_hook(RlTrainerHook::IterEnd, &mut control, &mut ())
        .unwrap();
    assert!(directory.path().join("latest.pth").is_symlink());
    callback
        .on_hook(RlTrainerHook::FitEnd, &mut control, &mut ())
        .unwrap();
    control
        .runtime
        .update(|state| state.current_iter = Some(1.into()))
        .unwrap();
    callback.config.save_latest = None;
    RlTrainerCallback::call(&mut callback, RlTrainerHook::FitEnd, &mut control, &mut ()).unwrap();
    RlCheckpointState::save_checkpoint(&mut callback).unwrap();
    RlCheckpointState::load_checkpoint(&mut callback, &()).unwrap();

    let timed = tempfile::tempdir().unwrap();
    let (mut callback, mut control) = callback_with(timed.path());
    callback.config.time_interval = Some(RlCheckpointTimeInterval::Float(10.0));
    callback
        .on_hook(RlTrainerHook::IterEnd, &mut control, &mut ())
        .unwrap();
    callback.state.last_time = Some(120.0);
    callback
        .on_hook(RlTrainerHook::IterEnd, &mut control, &mut ())
        .unwrap();
    callback.state.last_time = Some(100.0);
    callback
        .on_hook(RlTrainerHook::IterEnd, &mut control, &mut ())
        .unwrap();
}

#[test]
fn missing_runtime_fields_and_reserved_names_fail_before_formatting() {
    let directory = tempfile::tempdir().unwrap();
    let (mut callback, control) = callback_with(directory.path());
    control
        .runtime
        .update(|state| state.current_iter = None)
        .unwrap();
    assert!(matches!(
        callback.new_name(&control.runtime),
        Err(RlCheckpointCallbackError::MissingTrainerField(
            "current_iter"
        ))
    ));
    control
        .runtime
        .update(|state| {
            state.current_iter = Some(0.into());
            state.metrics = None;
        })
        .unwrap();
    assert!(matches!(
        callback.new_name(&control.runtime),
        Err(RlCheckpointCallbackError::MissingTrainerField("metrics"))
    ));
    for reserved in ["iter", "time"] {
        control
            .runtime
            .update(|state| {
                let mut metrics = IndexMap::new();
                metrics.insert(
                    reserved.into(),
                    RlLosslessCheckpointValue::Integer(1.into()),
                );
                state.metrics = Some(metrics);
            })
            .unwrap();
        assert_eq!(
            callback.new_name(&control.runtime).unwrap_err(),
            RlCheckpointCallbackError::DuplicateKeyword(reserved)
        );
    }
}
