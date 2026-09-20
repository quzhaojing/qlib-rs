use super::*;
use serde_json::{Value, json};
use std::{cell::RefCell, process::Command, rc::Rc, sync::Arc};

#[path = "rl_checkpoint_lossless_contract.rs"]
mod lossless_contract;

#[derive(Default)]
struct World {
    events: Vec<Value>,
    files: serde_json::Map<String, Value>,
    directories: Vec<String>,
    fail: Option<String>,
    now: f64,
    clear_iteration_after_format: Option<Arc<RlTrainerRuntime>>,
    poison_after_local_time: Option<Arc<RlTrainerRuntime>>,
}
impl World {
    fn event(&mut self, stage: &str, args: &[Value]) -> Result<(), String> {
        let mut event = vec![json!(stage)];
        event.extend_from_slice(args);
        self.events.push(json!(event));
        if self.fail.as_deref() == Some(stage) {
            Err(stage.into())
        } else {
            Ok(())
        }
    }
}
#[derive(Clone)]
struct Spy(Rc<RefCell<World>>);
impl RlCheckpointClock for Spy {
    fn timestamp(&mut self) -> Result<f64, String> {
        self.0.borrow_mut().event("time", &[])?;
        Ok(self.0.borrow().now)
    }
    fn local_time(&mut self) -> Result<String, String> {
        self.0.borrow_mut().event("local_time", &[])?;
        if let Some(runtime) = self.0.borrow_mut().poison_after_local_time.take() {
            poison(&runtime);
        }
        Ok("20260902123456".into())
    }
}
// A deterministic test formatter, not the production Python-format adapter. It deliberately
// supports only these fixture templates; complete filename compatibility remains pending.
impl RlCheckpointName<f64> for Spy {
    fn render(
        &mut self,
        template: &str,
        iteration: &BigInt,
        time: &str,
        _metrics: &IndexMap<String, f64>,
    ) -> Result<String, String> {
        self.0
            .borrow_mut()
            .event("format", &[json!(iteration.to_string()), json!(time)])?;
        if let Some(runtime) = self.0.borrow_mut().clear_iteration_after_format.take() {
            runtime.update(|s| s.current_iter = None).unwrap();
        }
        if template == "{" || template == "{missing}" {
            return Err("invalid format".into());
        }
        Ok(template
            .replace("{iter:03d}", &format!("{iteration:03}"))
            .replace("{iter}", &iteration.to_string()))
    }
}
impl RlCheckpointGraph<(), f64> for Spy {
    type State = String;
    fn collect(
        &mut self,
        control: &mut RlTrainerControl,
        _vessel: &mut (),
    ) -> Result<String, String> {
        let iteration = control
            .runtime
            .read(|s| s.current_iter.as_ref().unwrap().to_string())
            .unwrap();
        self.0.borrow_mut().event("graph", &[json!(iteration)])?;
        Ok(format!("graph-{iteration}"))
    }
}
fn path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}
impl RlCheckpointStorage<String> for Spy {
    fn create_directory(&mut self, directory: &Path) -> Result<(), String> {
        let path = path(directory);
        let mut world = self.0.borrow_mut();
        world.event("mkdir", &[json!(path)])?;
        if !world.directories.contains(&path) {
            world.directories.push(path);
        }
        Ok(())
    }
    fn save(&mut self, state: &String, destination: &Path) -> Result<(), String> {
        let path = path(destination);
        let mut world = self.0.borrow_mut();
        world.files.insert(path.clone(), json!(["file", "partial"]));
        world.event("save", &[json!(path), json!(state)])?;
        world.files.insert(path, json!(["file", state]));
        Ok(())
    }
    fn exists(&mut self, file: &Path) -> Result<bool, String> {
        let path = path(file);
        let mut world = self.0.borrow_mut();
        world.event("exists", &[json!(path)])?;
        Ok(match world.files.get(&path) {
            None => false,
            Some(value) if value[0] == "link" => world
                .files
                .get(value[1].as_str().unwrap())
                .is_some_and(|v| v[0] == "file"),
            Some(_) => true,
        })
    }
    fn is_link(&mut self, file: &Path) -> Result<bool, String> {
        let path = path(file);
        let mut world = self.0.borrow_mut();
        world.event("is_link", &[json!(path)])?;
        Ok(world.files.get(&path).is_some_and(|v| v[0] == "link"))
    }
    fn remove(&mut self, file: &Path) -> Result<(), String> {
        let path = path(file);
        let mut world = self.0.borrow_mut();
        world.event("remove", &[json!(path)])?;
        if world.files.get(&path).is_some_and(|v| v[0] == "directory") {
            return Err("is a directory".into());
        }
        world.files.remove(&path).unwrap();
        Ok(())
    }
    fn link(&mut self, target: &Path, link: &Path) -> Result<(), String> {
        let (target, link) = (path(target), path(link));
        let mut world = self.0.borrow_mut();
        world.event("link", &[json!(target), json!(link)])?;
        world.files.insert(link, json!(["link", target]));
        Ok(())
    }
    fn copy(&mut self, source: &Path, destination: &Path) -> Result<(), String> {
        let (source, destination) = (path(source), path(destination));
        let mut world = self.0.borrow_mut();
        world.event("copy", &[json!(source), json!(destination)])?;
        let value = world.files.get(&source).ok_or("missing source")?.clone();
        world.files.insert(destination, value);
        Ok(())
    }
}

fn hook(step: &Value) -> RlTrainerHook {
    match step["hook"].as_str().unwrap_or("iter_end") {
        "iter_end" => RlTrainerHook::IterEnd,
        "fit_end" => RlTrainerHook::FitEnd,
        "fit_start" => RlTrainerHook::FitStart,
        "test_end" => RlTrainerHook::TestEnd,
        "save" => unreachable!(),
        other => panic!("unknown hook {other}"),
    }
}

fn config(spec: &Value) -> RlCheckpointConfig {
    let mut config = RlCheckpointConfig::new("checkpoints");
    if let Some(value) = spec.get("save_latest") {
        config.save_latest = value.as_str().map(str::to_owned);
    }
    if let Some(value) = spec["filename"].as_str() {
        config.filename = value.into();
    }
    config.every_n_iters = spec["every"].as_str().map(|v| v.parse().unwrap());
    config.save_on_fit_end = spec["save_on_fit_end"].as_bool().unwrap_or(true);
    config.time_interval = spec.get("interval").map(|v| {
        if v[0] == "int" {
            RlCheckpointTimeInterval::Integer(v[1].as_str().unwrap().parse().unwrap())
        } else {
            RlCheckpointTimeInterval::Float(v[1].as_str().unwrap().parse().unwrap())
        }
    });
    config
}

fn exercise(spec: &Value) -> Value {
    let world = Rc::new(RefCell::new(World::default()));
    let spy = Spy(world.clone());
    let mut callback =
        RlCheckpointCallback::new(config(spec), spy.clone(), spy.clone(), spy.clone(), spy);
    assert!(world.borrow().events.is_empty());
    if let Some(latest) = spec.get("latest") {
        world
            .borrow_mut()
            .files
            .insert("checkpoints/latest.pth".into(), latest.clone());
    }
    let runtime = Arc::new(RlTrainerRuntime::new(None));
    let mut control = RlTrainerControl {
        runtime: runtime.clone(),
        config: crate::RlTrainerConfig::default(),
    };
    let mut results = vec![];
    for step in spec["steps"].as_array().unwrap() {
        {
            let mut world = world.borrow_mut();
            world.events.clear();
            world.fail = step["fail"].as_str().map(str::to_owned);
            world.now = step["now"].as_str().unwrap_or("100").parse().unwrap();
        }
        runtime
            .update(|s| {
                s.current_iter = if step["missing_iter"] == true {
                    None
                } else {
                    Some(step["iter"].as_str().unwrap().parse().unwrap())
                };
                s.metrics = if step["missing_metrics"] == true {
                    None
                } else {
                    Some(
                        step["metrics"]
                            .as_object()
                            .into_iter()
                            .flatten()
                            .map(|(k, v)| (k.clone(), v.as_f64().unwrap()))
                            .collect(),
                    )
                };
            })
            .unwrap();
        let result = if step["hook"] == "save" {
            callback.save(&mut control, &mut ())
        } else {
            callback.on_hook(hook(step), &mut control, &mut ())
        };
        let world = world.borrow();
        results.push(json!({"error":result.is_err(),"events":world.events,"files":world.files,"directories":world.directories,
            "name":callback.state.last_name,"iteration":callback.state.last_iter.as_ref().map(ToString::to_string),
            "time":callback.state.last_time.map(crate::training_vessel_log::python_float)}));
    }
    let before = callback.state.clone();
    callback.save_checkpoint().unwrap();
    callback.load_checkpoint(&()).unwrap();
    assert_eq!(format!("{before:?}"), format!("{:?}", callback.state));
    json!({"spec":spec,"results":results})
}

#[test]
fn scheduling_and_partial_effects_match_live_python_checkpoint() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .args([
            root.join("tests/fixtures/rl_checkpoint_callback_contract.py"),
            root.join("../../../qlib/qlib/rl/trainer/callbacks.py"),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(cases.len(), 95);
    for expected in cases {
        assert_eq!(
            exercise(&expected["spec"]),
            expected,
            "spec: {}",
            expected["spec"]
        );
    }
}

fn poison(runtime: &RlTrainerRuntime) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = runtime.update(|_| panic!("intentional poison"));
    }));
}

struct MutatingName(Arc<RlTrainerRuntime>);
impl RlCheckpointName<f64> for MutatingName {
    fn render(
        &mut self,
        _template: &str,
        _iteration: &BigInt,
        _time: &str,
        _metrics: &IndexMap<String, f64>,
    ) -> Result<String, String> {
        self.0.update(|s| s.current_iter = None).unwrap();
        Ok("named.pth".into())
    }
}
struct PoisoningClock(Arc<RlTrainerRuntime>);
impl RlCheckpointClock for PoisoningClock {
    fn timestamp(&mut self) -> Result<f64, String> {
        unreachable!()
    }
    fn local_time(&mut self) -> Result<String, String> {
        poison(&self.0);
        Ok("time".into())
    }
}

#[test]
fn plugin_runtime_reentry_retains_order_and_reports_poisoning() {
    let runtime = Arc::new(RlTrainerRuntime::new(None));
    runtime
        .update(|s| {
            s.current_iter = Some(1.into());
            s.metrics = Some(IndexMap::new());
        })
        .unwrap();
    let world = Rc::new(RefCell::new(World::default()));
    let spy = Spy(world.clone());
    let mut callback = RlCheckpointCallback::new(
        RlCheckpointConfig::new("checkpoints"),
        spy.clone(),
        MutatingName(runtime.clone()),
        spy.clone(),
        spy.clone(),
    );
    let mut control = RlTrainerControl {
        runtime: runtime.clone(),
        config: crate::RlTrainerConfig::default(),
    };
    let error = callback.save(&mut control, &mut ()).unwrap_err();
    assert_eq!(
        error,
        RlCheckpointCallbackError::MissingTrainerField("current_iter")
    );
    assert_eq!(callback.state.last_name.as_deref(), Some("named.pth"));
    assert_eq!(callback.state.last_iter, None);
    assert!(world.borrow().files.is_empty());
    runtime.update(|s| s.current_iter = Some(1.into())).unwrap();
    let mut callback = RlCheckpointCallback::new(
        RlCheckpointConfig::new("checkpoints"),
        PoisoningClock(runtime.clone()),
        spy.clone(),
        spy.clone(),
        spy,
    );
    assert!(matches!(
        callback.new_name(&runtime),
        Err(RlCheckpointCallbackError::Runtime(_))
    ));
    let error = callback
        .call(RlTrainerHook::FitEnd, &mut control, &mut ())
        .unwrap_err();
    assert!(
        matches!(error, RlTrainerDriverError::Plugin { ref stage, .. } if stage == "checkpoint")
    );
    assert!(error.to_string().contains("poison"));
    assert!(matches!(
        callback.new_name(&runtime),
        Err(RlCheckpointCallbackError::Runtime(_))
    ));
}

#[test]
fn integer_time_intervals_compare_without_float_narrowing() {
    let interval = RlCheckpointTimeInterval::Integer((BigInt::from(1) << 53) + 1);
    assert!(!interval.elapsed(9_007_199_254_740_992.0));
    assert!(interval.elapsed(9_007_199_254_740_994.0));
    let negative = RlCheckpointTimeInterval::Integer((-1).into());
    assert!(!negative.elapsed(-1.5));
    assert!(negative.elapsed(-1.0));
    assert!(negative.elapsed(-0.5));
    assert!(!negative.elapsed(f64::NAN));
    assert!(!negative.elapsed(f64::NEG_INFINITY));
    assert!(negative.elapsed(f64::INFINITY));
    assert_eq!(negative.clone(), negative);
    let config = RlCheckpointConfig::new("checkpoints");
    assert_eq!(config.clone(), config);
    assert_eq!(
        RlCheckpointCallbackState::default(),
        RlCheckpointCallbackState::default()
    );
}

#[test]
fn configured_boundary_plugins_can_invalidate_runtime_between_successful_calls() {
    let runtime = Arc::new(RlTrainerRuntime::new(None));
    runtime
        .update(|s| {
            s.current_iter = Some(1.into());
            s.metrics = Some(IndexMap::new());
        })
        .unwrap();
    let world = Rc::new(RefCell::new(World::default()));
    let spy = Spy(world.clone());
    let mut callback = RlCheckpointCallback::new(
        RlCheckpointConfig::new("checkpoints"),
        spy.clone(),
        spy.clone(),
        spy.clone(),
        spy,
    );
    let mut control = RlTrainerControl {
        runtime: runtime.clone(),
        config: crate::RlTrainerConfig::default(),
    };
    world.borrow_mut().clear_iteration_after_format = Some(runtime.clone());
    assert_eq!(
        callback.save(&mut control, &mut ()).unwrap_err(),
        RlCheckpointCallbackError::MissingTrainerField("current_iter")
    );
    assert_eq!(callback.state.last_name.as_deref(), Some("001.pth"));
    assert_eq!(callback.state.last_iter, None);
    assert_eq!(
        world.borrow().events,
        vec![
            json!(["mkdir", "checkpoints"]),
            json!(["local_time"]),
            json!(["format", "1", "20260902123456"])
        ]
    );
    assert!(world.borrow().files.is_empty());
    runtime.update(|s| s.current_iter = Some(1.into())).unwrap();
    world.borrow_mut().poison_after_local_time = Some(runtime.clone());
    assert!(matches!(
        callback.new_name(&runtime),
        Err(RlCheckpointCallbackError::Runtime(_))
    ));
    assert_eq!(callback.state.last_name.as_deref(), Some("001.pth"));
    assert_eq!(world.borrow().events.last(), Some(&json!(["local_time"])));
    assert!(world.borrow().files.is_empty());
}

#[test]
fn live_python_filename_oracle_freezes_remaining_adapter_requirements() {
    // This is source characterization, not evidence that the absent Rust filename adapter
    // implements these cases. It prevents narrowing the contract to simple substitutions.
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .args([
            root.join("tests/fixtures/rl_checkpoint_filename_contract.py"),
            root.join("../../../qlib/qlib/rl/trainer/callbacks.py"),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(cases.len(), 547);
    assert_eq!(cases.iter().filter(|c| !c["error"].is_null()).count(), 116);
    let case = |template: &str| {
        cases
            .iter()
            .find(|c| c["spec"]["probe"] == false && c["spec"]["template"] == template)
            .unwrap()
    };
    assert_eq!(
        case("{iter:03d}-{reward:.2f}-{time}.pth")["output"],
        "007-2.67-20260902123456.pth"
    );
    assert_eq!(
        case("{data[:]}-{data[!]}-{data[name]}")["output"],
        "11-13-中文"
    );
    assert_eq!(
        case("{custom.real:03d}")["events"],
        json!([["clock"], ["attribute", "real"]])
    );
    assert_eq!(case("{custom!a}")["output"], "Custom\\u4e2d\\u6587");
    assert_eq!(case("{custom} {")["error"], "ValueError");
    assert_eq!(
        case("{custom} {")["events"],
        json!([["clock"], ["format", ""]])
    );
    assert_eq!(case("{custom} {missing}")["error"], "KeyError");
    assert_eq!(
        case("{custom} {missing}")["events"],
        json!([["clock"], ["format", ""]])
    );
    for reserved in cases.iter().filter(|c| c["spec"].get("reserved").is_some()) {
        assert_eq!(reserved["error"], "TypeError");
        assert_eq!(reserved["events"], json!([["clock"]]));
    }
}
