//! Reuse the deterministic boundary spies against the additive lossless callback.
//! These ASCII scheduling fixtures complement the separate raw-surrogate OS tests.
use super::*;
use crate::{
    RlCheckpointText, RlLosslessCheckpointCallback, RlLosslessCheckpointConfig,
    RlLosslessCheckpointName,
};

impl RlLosslessCheckpointName<f64> for Spy {
    fn render(
        &mut self,
        template: &RlCheckpointText,
        iteration: &BigInt,
        time: &RlCheckpointText,
        metrics: &IndexMap<RlCheckpointText, f64>,
    ) -> Result<RlCheckpointText, String> {
        let metrics = metrics
            .iter()
            .map(|(key, value)| Ok((key.to_utf8()?, *value)))
            .collect::<Result<IndexMap<_, _>, String>>()?;
        RlCheckpointName::render(
            self,
            &template.to_utf8()?,
            iteration,
            &time.to_utf8()?,
            &metrics,
        )
        .map(RlCheckpointText::from)
    }
}

fn lossless_config(spec: &Value) -> RlLosslessCheckpointConfig {
    let config = config(spec);
    RlLosslessCheckpointConfig {
        dirpath: config.dirpath,
        filename: config.filename.into(),
        save_latest: config.save_latest.map(RlCheckpointText::from),
        every_n_iters: config.every_n_iters,
        time_interval: config.time_interval,
        save_on_fit_end: config.save_on_fit_end,
    }
}

fn exercise_lossless(spec: &Value) -> Value {
    let world = Rc::new(RefCell::new(World::default()));
    let spy = Spy(world.clone());
    let mut callback = RlLosslessCheckpointCallback::new(
        lossless_config(spec),
        spy.clone(),
        spy.clone(),
        spy.clone(),
        spy,
    );
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
            .update(|state| {
                state.current_iter = if step["missing_iter"] == true {
                    None
                } else {
                    Some(step["iter"].as_str().unwrap().parse().unwrap())
                };
                state.metrics = if step["missing_metrics"] == true {
                    None
                } else {
                    Some(
                        step["metrics"]
                            .as_object()
                            .into_iter()
                            .flatten()
                            .map(|(key, value)| (key.clone(), value.as_f64().unwrap()))
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
        results.push(json!({
            "error": result.is_err(), "events": world.events,
            "files": world.files, "directories": world.directories,
            "name": callback.state.last_name.as_ref().map(|text| text.to_utf8().unwrap()),
            "iteration": callback.state.last_iter.as_ref().map(ToString::to_string),
            "time": callback.state.last_time.map(crate::training_vessel_log::python_float)
        }));
    }
    let before = callback.state.clone();
    callback.save_checkpoint().unwrap();
    callback.load_checkpoint(&()).unwrap();
    assert_eq!(before, callback.state);
    json!({"spec": spec, "results": results})
}

#[test]
fn lossless_scheduling_and_failures_match_all_live_source_cases() {
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
        assert_eq!(exercise_lossless(&expected["spec"]), expected);
    }
}

#[test]
fn lossless_plugin_reentry_preserves_partial_state_and_reports_poisoning() {
    let runtime = Arc::new(RlTrainerRuntime::new(None));
    runtime
        .update(|state| {
            state.current_iter = Some(1.into());
            state.metrics = Some(IndexMap::from([("reward".into(), 0.5)]));
        })
        .unwrap();
    let world = Rc::new(RefCell::new(World::default()));
    let spy = Spy(world.clone());
    let mut callback = RlLosslessCheckpointCallback::new(
        RlLosslessCheckpointConfig::new("checkpoints"),
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
    assert_eq!(callback.state.last_name, Some("001.pth".into()));
    assert_eq!(callback.state.last_iter, None);
    assert_eq!(callback.state.last_time, None);
    assert!(world.borrow().files.is_empty());
    assert_eq!(
        world.borrow().events,
        vec![
            json!(["mkdir", "checkpoints"]),
            json!(["local_time"]),
            json!(["format", "1", "20260902123456"])
        ]
    );

    runtime
        .update(|state| state.current_iter = Some(1.into()))
        .unwrap();
    world.borrow_mut().events.clear();
    world.borrow_mut().poison_after_local_time = Some(runtime.clone());
    assert!(matches!(
        callback.new_name(&runtime),
        Err(RlCheckpointCallbackError::Runtime(_))
    ));
    assert_eq!(world.borrow().events, vec![json!(["local_time"])]);
    world.borrow_mut().events.clear();
    let error = callback
        .call(RlTrainerHook::FitEnd, &mut control, &mut ())
        .unwrap_err();
    assert!(matches!(error, RlTrainerDriverError::Plugin {ref stage, ..} if stage == "checkpoint"));
    assert!(error.to_string().contains("poison"));
    assert!(world.borrow().events.is_empty());
    callback.config.every_n_iters = Some(1.into());
    assert!(matches!(
        callback.on_hook(RlTrainerHook::IterEnd, &mut control, &mut ()),
        Err(RlCheckpointCallbackError::Runtime(_))
    ));
    assert!(world.borrow().events.is_empty());
}
