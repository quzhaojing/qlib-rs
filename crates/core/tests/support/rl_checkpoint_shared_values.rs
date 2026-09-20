use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

// Deliberately not Clone: the callback must clone shared handles, not the model.
struct Model {
    calls: AtomicUsize,
    events: Events,
    leaf: Arc<dyn RlCheckpointFormatValue>,
}

impl RlCheckpointFormatValue for Model {
    fn format(&self, spec: &str) -> Result<String, String> {
        self.events.lock().unwrap().push(json!(["format", spec]));
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        if spec == "fail" {
            Err("model format failed".into())
        } else {
            Ok(format!("model-{call}"))
        }
    }

    fn representation(&self, repr: bool) -> Result<String, String> {
        self.events
            .lock()
            .unwrap()
            .push(json!([if repr { "repr" } else { "str" }]));
        Ok(if repr { "Model中文" } else { "model中文" }.into())
    }

    fn attribute(&self, name: &str) -> Result<Arc<dyn RlCheckpointFormatValue>, String> {
        self.events.lock().unwrap().push(json!(["attribute", name]));
        if name == "leaf" {
            Ok(self.leaf.clone())
        } else {
            Err("model attribute failed".into())
        }
    }

    fn item(
        &self,
        key: RlCheckpointFieldIndex<'_>,
    ) -> Result<Arc<dyn RlCheckpointFormatValue>, String> {
        let key = match key {
            RlCheckpointFieldIndex::Integer(n) => json!(n),
            RlCheckpointFieldIndex::Key(key) => json!(key),
        };
        self.events.lock().unwrap().push(json!(["item", key]));
        if key == json!(0) || key == json!("leaf") {
            Ok(self.leaf.clone())
        } else {
            Err("model item failed".into())
        }
    }
}

#[test]
fn shared_heterogeneous_metrics_preserve_model_identity_in_real_callback() {
    let events = Events::default();
    let leaf: Arc<dyn RlCheckpointFormatValue> = Arc::new(1234.5_f64);
    let model = Arc::new(Model {
        calls: AtomicUsize::new(0),
        events: events.clone(),
        leaf: leaf.clone(),
    });
    let erased: Arc<dyn RlCheckpointFormatValue> = model.clone();
    let metrics: IndexMap<String, Arc<dyn RlCheckpointFormatValue>> = IndexMap::from([
        ("model".into(), erased.clone()),
        ("alias".into(), erased.clone()),
        ("number".into(), leaf.clone()),
        (
            "text".into(),
            Arc::new(TrainingMetricScalar::Text("中文".into())),
        ),
    ]);
    let runtime = RlTrainerRuntime::new(None);
    runtime
        .update(|state| {
            state.current_iter = Some(7.into());
            state.metrics = Some(metrics);
        })
        .unwrap();
    let mut config = RlCheckpointConfig::new("unused");
    config.filename =
        "{model}-{alias}-{model.leaf:.1f}-{model[0]:.1f}-{model[leaf]:.1f}-{number:.2f}-{text!r}"
            .into();
    let mut callback = RlCheckpointCallback::new(
        config,
        Clock(events.clone()),
        PythonRlCheckpointName,
        (),
        (),
    );
    let references = Arc::strong_count(&model);
    assert_eq!(
        callback.new_name(&runtime).unwrap(),
        "model-1-model-2-1234.5-1234.5-1234.5-1234.50-'中文'"
    );
    assert_eq!(model.calls.load(Ordering::SeqCst), 2);
    assert_eq!(Arc::strong_count(&model), references);
    assert!(Arc::ptr_eq(
        &RlCheckpointFormatValue::attribute(&erased, "leaf").unwrap(),
        &leaf
    ));
    assert!(Arc::ptr_eq(
        &RlCheckpointFormatValue::item(&erased, RlCheckpointFieldIndex::Integer(0)).unwrap(),
        &leaf
    ));
    // No save or bookkeeping operation belongs to new_name.
    assert!(callback.state.last_name.is_none());
    assert_eq!(
        events.lock().unwrap()[..6],
        [
            json!(["clock"]),
            json!(["format", ""]),
            json!(["format", ""]),
            json!(["attribute", "leaf"]),
            json!(["item", 0]),
            json!(["item", "leaf"])
        ]
    );
}

#[test]
fn arc_forwarding_keeps_locale_conversions_and_failures_at_their_original_positions() {
    let number = Arc::new(1234.5_f64);
    assert_eq!(
        RlCheckpointFormatValue::format(&number, ".2f").unwrap(),
        "1234.50"
    );
    let erased: Arc<dyn RlCheckpointFormatValue> = number;
    let locale = RlCheckpointNumericLocale::new(",".into(), ".".into(), &[3, 0]).unwrap();
    assert_eq!(
        RlCheckpointFormatValue::format_with_locale(&erased, ".10n", &locale).unwrap(),
        "1.234,5"
    );
    assert_eq!(
        RlCheckpointFormatValue::representation(&erased, false).unwrap(),
        "1234.5"
    );
    for (template, stage) in [
        ("{x}", "format"),
        ("{x!s:{missing}}", "str"),
        ("{x!r}", "repr"),
        ("{x.a}", "attribute"),
        ("{x[0]}", "item"),
    ] {
        let events = Events::default();
        let failure: Arc<dyn RlCheckpointFormatValue> = Arc::new(Failure(stage, events.clone()));
        let metrics = IndexMap::from([("x".into(), failure)]);
        assert_eq!(
            PythonRlCheckpointName.render(template, &1.into(), "", &metrics),
            Err(format!("failed {stage}"))
        );
        assert_eq!(*events.lock().unwrap(), [json!(stage)]);
    }
    let events = Events::default();
    let failed: Arc<dyn RlCheckpointFormatValue> = Arc::new(Failure("format", events.clone()));
    assert_eq!(
        RlCheckpointFormatValue::format(&failed, ""),
        Err("failed format".into())
    );
    assert_eq!(*events.lock().unwrap(), [json!("format")]);
}

#[test]
fn shared_model_calls_match_live_qlib_across_numeric_locales() {
    let output = Command::new("python")
        .arg(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/rl_checkpoint_shared_contract.py"),
        )
        .arg("D:/code/github/qlib/qlib/rl/trainer/callbacks.py")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Vec<Json> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(cases.len(), 63);
    for case in cases {
        let events = Events::default();
        let leaf: Arc<dyn RlCheckpointFormatValue> = Arc::new(1234.5_f64);
        let model = Arc::new(Model {
            calls: AtomicUsize::new(0),
            events: events.clone(),
            leaf: leaf.clone(),
        });
        let erased: Arc<dyn RlCheckpointFormatValue> = model.clone();
        let metrics: IndexMap<String, Arc<dyn RlCheckpointFormatValue>> = IndexMap::from([
            ("model".into(), erased.clone()),
            ("alias".into(), erased),
            ("number".into(), leaf),
            (
                "text".into(),
                Arc::new(TrainingMetricScalar::Text("中文".into())),
            ),
        ]);
        let runtime = RlTrainerRuntime::new(None);
        runtime
            .update(|state| {
                state.current_iter = Some(7.into());
                state.metrics = Some(metrics);
            })
            .unwrap();
        let mut config = RlCheckpointConfig::new("unused");
        config.filename = case["template"].as_str().unwrap().into();
        let data = &case["locale"];
        let locale = RlCheckpointNumericLocale::new(
            data["decimal"].as_str().unwrap().into(),
            data["separator"].as_str().unwrap().into(),
            &data["grouping"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| u8::try_from(v.as_u64().unwrap()).unwrap())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let mut callback = RlCheckpointCallback::new(
            config,
            Clock(events.clone()),
            LocalizedPythonRlCheckpointName::new(Arc::new(locale)),
            (),
            (),
        );
        let references = Arc::strong_count(&model);
        let result = callback.new_name(&runtime);
        if case["error"].is_null() {
            assert_eq!(result.unwrap(), case["output"].as_str().unwrap(), "{case}");
        } else {
            assert!(result.is_err(), "{case}: {result:?}");
        }
        assert_eq!(Arc::strong_count(&model), references);
        assert_eq!(
            model.calls.load(Ordering::SeqCst),
            usize::try_from(case["calls"].as_u64().unwrap()).unwrap()
        );
        assert_eq!(
            *events.lock().unwrap(),
            case["events"].as_array().unwrap().clone(),
            "{case}"
        );
    }
}
