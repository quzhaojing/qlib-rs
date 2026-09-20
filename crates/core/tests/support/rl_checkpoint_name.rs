use super::*;
use crate::{RlCheckpointCallback, RlCheckpointClock, RlCheckpointConfig, RlTrainerRuntime};
use serde_json::{Value as Json, json};
use std::{path::PathBuf, process::Command, sync::Mutex};

#[path = "rl_checkpoint_shared_values.rs"]
mod shared_values;

type Events = Arc<Mutex<Vec<Json>>>;

#[derive(Clone)]
enum Metric {
    Scalar(TrainingMetricScalar),
    Map(IndexMap<String, Metric>),
    Custom(Events),
    Failure(Failure),
}

impl RlCheckpointFormatValue for Metric {
    fn format_with_locale(
        &self,
        spec: &str,
        locale: &dyn RlCheckpointLocaleProvider,
    ) -> Result<String, String> {
        match self {
            Self::Scalar(value) => value.format_with_locale(spec, locale),
            _ => self.format(spec),
        }
    }
    fn format(&self, spec: &str) -> Result<String, String> {
        match self {
            Self::Scalar(value) => value.format(spec),
            Self::Failure(value) => value.format(spec),
            Self::Custom(events) => {
                events.lock().unwrap().push(json!(["format", spec]));
                if spec == "fail" {
                    Err("requested format failure".into())
                } else {
                    Ok(format!("custom<{spec}>"))
                }
            }
            Self::Map(_) => Err("map formatting is not used in this fixture".into()),
        }
    }
    fn representation(&self, repr: bool) -> Result<String, String> {
        match self {
            Self::Scalar(value) => value.representation(repr),
            Self::Failure(value) => value.representation(repr),
            Self::Custom(events) => {
                events
                    .lock()
                    .unwrap()
                    .push(json!([if repr { "repr" } else { "str" }]));
                Ok(if repr { "Custom中文" } else { "value中文" }.into())
            }
            Self::Map(_) => Err("map representation not used in this fixture".into()),
        }
    }
    fn attribute(&self, name: &str) -> Result<Arc<dyn RlCheckpointFormatValue>, String> {
        match self {
            Self::Scalar(value) => value.attribute(name),
            Self::Failure(value) => value.attribute(name),
            Self::Custom(events) => {
                if name != "real" {
                    return Err("unknown custom attribute".into());
                }
                events.lock().unwrap().push(json!(["attribute", name]));
                Ok(Arc::new(TrainingMetricScalar::Integer(12.into())))
            }
            Self::Map(_) => Err("unknown map attribute".into()),
        }
    }
    fn item(
        &self,
        key: RlCheckpointFieldIndex<'_>,
    ) -> Result<Arc<dyn RlCheckpointFormatValue>, String> {
        match self {
            Self::Scalar(value) => value.item(key),
            Self::Failure(value) => value.item(key),
            Self::Custom(events) => {
                let event_key = match key {
                    RlCheckpointFieldIndex::Key(key) => json!(key),
                    RlCheckpointFieldIndex::Integer(key) => json!(key),
                };
                events.lock().unwrap().push(json!(["item", event_key]));
                let number = match key {
                    RlCheckpointFieldIndex::Key("x") => 7,
                    RlCheckpointFieldIndex::Key(":") => 8,
                    RlCheckpointFieldIndex::Key("!") => 9,
                    _ => return Err("missing item".into()),
                };
                Ok(Arc::new(TrainingMetricScalar::Integer(number.into())))
            }
            Self::Map(values) => {
                let RlCheckpointFieldIndex::Key(key) = key else {
                    return Err("missing integer key".into());
                };
                values
                    .get(key)
                    .cloned()
                    .map(|value| Arc::new(value) as Arc<dyn RlCheckpointFormatValue>)
                    .ok_or_else(|| "missing map key".into())
            }
        }
    }
}

struct Clock(Events);
impl RlCheckpointClock for Clock {
    fn timestamp(&mut self) -> Result<f64, String> {
        panic!("filename must not read numeric clock")
    }
    fn local_time(&mut self) -> Result<String, String> {
        self.0.lock().unwrap().push(json!(["clock"]));
        Ok("20260902123456".into())
    }
}

fn scalar(tag: &Json) -> Metric {
    let text = tag[1].as_str().unwrap();
    Metric::Scalar(match tag[0].as_str().unwrap() {
        "int" => TrainingMetricScalar::Integer(text.parse().unwrap()),
        "float" => TrainingMetricScalar::Float(text.parse().unwrap()),
        "text" => TrainingMetricScalar::Text(text.into()),
        "bool" => TrainingMetricScalar::Boolean(text == "true"),
        "null" => TrainingMetricScalar::Null,
        _ => panic!("invalid fixture scalar"),
    })
}

#[test]
fn complete_filename_calls_match_live_qlib_outputs_errors_and_protocol_order() {
    compare_source(false);
}

#[test]
fn extended_filename_grammar_and_conversions_match_live_qlib() {
    compare_source(true);
}

fn compare_source(extended: bool) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut command = Command::new("python");
    command.arg(root.join("tests/fixtures/rl_checkpoint_filename_contract.py"));
    if extended {
        command.arg("--extended-template-probes");
    }
    let output = command
        .arg(root.join("../../../qlib/qlib/rl/trainer/callbacks.py"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Vec<Json> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(cases.len(), if extended { 153 } else { 547 });
    compare_cases(cases, || PythonRlCheckpointName);
}

#[test]
fn localized_callback_names_match_live_qlib_in_five_numeric_locales() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/rl_checkpoint_locale_contract.py"))
        .arg("--filenames")
        .arg(root.join("../../../qlib/qlib/rl/trainer/callbacks.py"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let locales: Vec<Json> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(locales.len(), 5);
    for locale in locales {
        let data = &locale["metadata"];
        let provider = Arc::new(
            RlCheckpointNumericLocale::new(
                data["decimal"].as_str().unwrap().into(),
                data["separator"].as_str().unwrap().into(),
                &data["grouping"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| u8::try_from(v.as_u64().unwrap()).unwrap())
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
        );
        let cases = locale["cases"].as_array().unwrap().clone();
        assert_eq!(cases.len(), 700);
        compare_cases(cases, || {
            LocalizedPythonRlCheckpointName::new(provider.clone())
        });
    }
}

#[test]
fn locale_context_is_queried_per_primitive_field_after_custom_nested_effects() {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    struct ChangingLocale {
        changed: Arc<AtomicBool>,
        calls: AtomicUsize,
        fail: AtomicBool,
    }
    impl RlCheckpointLocaleProvider for ChangingLocale {
        fn numeric_locale(&self) -> Result<RlCheckpointNumericLocale, String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.fail.load(Ordering::SeqCst) {
                return Err("locale failed".into());
            }
            RlCheckpointNumericLocale::new(
                ".".into(),
                ",".into(),
                if self.changed.load(Ordering::SeqCst) {
                    &[3, 2, 0]
                } else {
                    &[3, 0]
                },
            )
        }
    }
    struct Change(Arc<AtomicBool>);
    impl RlCheckpointFormatValue for Change {
        fn format(&self, spec: &str) -> Result<String, String> {
            self.0.store(true, Ordering::SeqCst);
            Ok(if spec == "spec" { "n" } else { "changed" }.into())
        }
        fn representation(&self, _: bool) -> Result<String, String> {
            Err("not used".into())
        }
        fn attribute(&self, _: &str) -> Result<Arc<dyn RlCheckpointFormatValue>, String> {
            Err("not used".into())
        }
        fn item(
            &self,
            _: RlCheckpointFieldIndex<'_>,
        ) -> Result<Arc<dyn RlCheckpointFormatValue>, String> {
            Err("not used".into())
        }
    }
    let changed = Arc::new(AtomicBool::new(false));
    let provider = Arc::new(ChangingLocale {
        changed: changed.clone(),
        calls: AtomicUsize::new(0),
        fail: AtomicBool::new(false),
    });
    let mut name = LocalizedPythonRlCheckpointName::new(provider.clone());
    let metrics = IndexMap::from([("change".into(), Change(changed.clone()))]);
    assert_eq!(
        name.render(
            "{iter:n}-{change}-{iter:n}-{change:n}",
            &123_456_789.into(),
            "",
            &metrics
        )
        .unwrap(),
        "123,456,789-changed-12,34,56,789-changed"
    );
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    changed.store(false, Ordering::SeqCst);
    assert_eq!(
        name.render("{iter:{change:spec}}", &123_456_789.into(), "", &metrics)
            .unwrap(),
        "12,34,56,789"
    );
    assert_eq!(provider.calls.load(Ordering::SeqCst), 3);
    let floats = IndexMap::from([("x".into(), 12345.6789_f64)]);
    assert_eq!(
        name.render("{x:n}-{x.real:.10n}", &1.into(), "", &floats)
            .unwrap(),
        "12,345.7-12,345.6789"
    );
    assert_eq!(provider.calls.load(Ordering::SeqCst), 5);
    provider.fail.store(true, Ordering::SeqCst);
    changed.store(false, Ordering::SeqCst);
    assert!(
        name.render("{change}-{iter:n}-{missing}", &1.into(), "", &metrics)
            .unwrap_err()
            .contains("locale failed")
    );
    assert!(changed.load(Ordering::SeqCst));
    assert_eq!(provider.calls.load(Ordering::SeqCst), 6);
}

fn compare_cases<N: RlCheckpointName<Metric>>(cases: Vec<Json>, make_name: impl Fn() -> N) {
    for case in cases {
        let events = Events::default();
        let mut config = RlCheckpointConfig::new("unused");
        config.filename = case["spec"]["template"].as_str().unwrap().into();
        let mut callback =
            RlCheckpointCallback::new(config, Clock(events.clone()), make_name(), (), ());
        let runtime = RlTrainerRuntime::new(None);
        runtime
            .update(|state| {
                state.current_iter =
                    Some(case["spec"]["iteration"].as_str().unwrap().parse().unwrap());
                let mut metrics: IndexMap<String, Metric> = [
                    ("reward", scalar(&case["spec"]["value"])),
                    ("width", scalar(&json!(["int", "8"]))),
                    ("precision", scalar(&json!(["int", "2"]))),
                    (
                        "data",
                        Metric::Map(
                            [
                                (":".into(), scalar(&json!(["int", "11"]))),
                                ("!".into(), scalar(&json!(["int", "13"]))),
                                ("name".into(), scalar(&json!(["text", "中文"]))),
                            ]
                            .into(),
                        ),
                    ),
                    ("custom", Metric::Custom(events.clone())),
                    ("val/reward", scalar(&json!(["float", "1.25"]))),
                    ("中文", scalar(&json!(["text", "value"]))),
                ]
                .into_iter()
                .map(|(key, value)| (key.into(), value))
                .collect();
                if let Some(key) = case["spec"]["reserved"].as_str() {
                    metrics.insert(key.into(), scalar(&json!(["int", "999"])));
                }
                state.metrics = Some(metrics);
            })
            .unwrap();
        let result = callback.new_name(&runtime);
        if case["error"].is_null() {
            assert_eq!(
                result.unwrap_or_else(|error| panic!("{case}: {error}")),
                case["output"].as_str().unwrap(),
                "{case}"
            );
        } else {
            assert!(result.is_err(), "{case}: {result:?}");
        }
        assert_eq!(json!(*events.lock().unwrap()), case["events"], "{case}");
    }
}

#[test]
fn decimal_field_indices_match_the_full_python_unicode_database() {
    let output = Command::new("python")
        .arg(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/rl_checkpoint_filename_contract.py"),
        )
        .arg("--decimal-probes")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let expected: Vec<(u32, usize)> = serde_json::from_slice(&output.stdout).unwrap();
    let actual: Vec<_> = (0..=0x0010_ffff)
        .filter_map(|cp| {
            char::from_u32(cp)
                .and_then(decimal_digit)
                .map(|digit| (cp, digit))
        })
        .collect();
    assert_eq!(actual, expected);
    assert_eq!(integer_index(""), Ok(None));
    assert_eq!(integer_index("１２٣"), Ok(Some(123)));
    assert_eq!(integer_index("-1"), Ok(None));
    assert_eq!(integer_index("²"), Ok(None));
    assert_eq!(
        integer_index("9223372036854775807"),
        Ok(Some(usize::try_from(isize::MAX).unwrap()))
    );
    assert!(integer_index("9223372036854775808").is_err());
    assert!(integer_index("9999999999999999999999x").is_err());
    assert!(integer_index("92233720368547758070").is_err());
    assert!(integer_index("18446744073709551619").is_err());
}

#[test]
fn unicode_representations_match_every_python_scalar_and_mixed_quote_context() {
    let output = Command::new("python")
        .arg(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/rl_checkpoint_filename_contract.py"),
        )
        .arg("--repr-probes")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let oracle: Json = serde_json::from_slice(&output.stdout).unwrap();
    let single = oracle["single"].as_array().unwrap();
    assert_eq!(single.len(), 0x0011_0000 - 0x800);
    for (ch, expected) in (0..=0x0010_ffff).filter_map(char::from_u32).zip(single) {
        let value = TrainingMetricScalar::Text(ch.to_string());
        assert_eq!(
            value.representation(true).unwrap(),
            expected.as_str().unwrap(),
            "U+{:04X}",
            u32::from(ch)
        );
    }
    let mixed = oracle["mixed"].as_array().unwrap();
    assert_eq!(mixed.len(), 1136);
    for case in mixed {
        let text = case[0].as_str().unwrap();
        let metrics: IndexMap<String, Metric> = [(
            "x".into(),
            Metric::Scalar(TrainingMetricScalar::Text(text.into())),
        )]
        .into();
        let actual = PythonRlCheckpointName
            .render("{x!s}|{x!r}|{x!a}", &1.into(), "", &metrics)
            .unwrap();
        assert_eq!(
            actual,
            format!(
                "{text}|{}|{}",
                case[1].as_str().unwrap(),
                case[2].as_str().unwrap()
            )
        );
    }
}

#[derive(Clone)]
struct Failure(&'static str, Events);
impl RlCheckpointFormatValue for Failure {
    fn format(&self, _: &str) -> Result<String, String> {
        self.call("format")
    }
    fn representation(&self, repr: bool) -> Result<String, String> {
        self.call(if repr { "repr" } else { "str" })
    }
    fn attribute(&self, _: &str) -> Result<Arc<dyn RlCheckpointFormatValue>, String> {
        self.call("attribute")
    }
    fn item(
        &self,
        _: RlCheckpointFieldIndex<'_>,
    ) -> Result<Arc<dyn RlCheckpointFormatValue>, String> {
        self.call("item")
    }
}
impl Failure {
    fn call<T>(&self, stage: &str) -> Result<T, String> {
        assert_eq!(self.0, stage);
        self.1.lock().unwrap().push(json!(stage));
        Err(format!("failed {stage}"))
    }
}

#[test]
fn model_failures_and_builtin_protocols_are_not_hidden_or_reordered() {
    assert_eq!(1.5_f64.format(".2f").unwrap(), "1.50");
    assert!(1.5_f64.format("badn").is_err());
    let mut formatter = PythonRlCheckpointName;
    for (template, stage) in [
        ("{x}", "format"),
        ("{x!s:{missing}}", "str"),
        ("{x!r}", "repr"),
        ("{x!a}", "repr"),
        ("{x.a}", "attribute"),
        ("{x[0]}", "item"),
    ] {
        let events = Events::default();
        let metrics = [("x".into(), Metric::Failure(Failure(stage, events.clone())))].into();
        assert_eq!(
            formatter
                .render(template, &1.into(), "", &metrics)
                .unwrap_err(),
            format!("failed {stage}")
        );
        assert_eq!(*events.lock().unwrap(), [json!(stage)]);
    }
    for reserved in ["iter", "time"] {
        let metrics: IndexMap<_, f64> = [(reserved.into(), 2.0)].into();
        assert!(
            formatter
                .render("literal", &1.into(), "", &metrics)
                .unwrap_err()
                .contains("duplicate")
        );
    }
    assert_eq!(
        formatter
            .render(
                "{x:.2f}-{x!s}-{x!r}-{x.real}-{x.imag}",
                &1.into(),
                "",
                &[("x".into(), 1.5)].into()
            )
            .unwrap(),
        "1.50-1.5-1.5-1.5-0.0"
    );
    assert!(1.0_f64.item(RlCheckpointFieldIndex::Integer(0)).is_err());
    assert!(1.0_f64.attribute("missing").is_err());
    assert_eq!(
        ascii_repr("'é中文😀\\n'"),
        "'\\xe9\\u4e2d\\u6587\\U0001f600\\n'"
    );
    let mut malformed = "[";
    assert!(part(&mut malformed).is_err());
}
