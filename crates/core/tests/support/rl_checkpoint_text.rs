use super::*;
use serde_json::{Value, json};
use std::{
    path::PathBuf,
    process::{Command, Stdio},
    sync::Mutex,
};

fn text(value: &Value) -> RlCheckpointText {
    RlCheckpointText::try_from_code_points(
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|point| u32::try_from(point.as_u64().unwrap()).unwrap()),
    )
    .unwrap()
}

#[derive(Default)]
struct Model {
    events: Arc<Mutex<Vec<Value>>>,
    fail: Option<&'static str>,
}
impl Model {
    fn record(&self, event: Value) -> Result<(), String> {
        let failed = self.fail.is_some_and(|stage| event[0] == stage);
        self.events.lock().unwrap().push(event);
        if failed {
            Err(format!("injected {} failure", self.fail.unwrap()))
        } else {
            Ok(())
        }
    }
}
impl RlLosslessCheckpointFormatValue for Model {
    fn format(&self, spec: &RlCheckpointText) -> Result<RlCheckpointText, String> {
        self.record(json!(["format", spec.as_code_points()]))?;
        RlCheckpointText::try_from_code_points(
            [0xd800]
                .into_iter()
                .chain("<".chars().map(u32::from))
                .chain(spec.as_code_points().iter().copied())
                .chain(">".chars().map(u32::from)),
        )
    }
    fn representation(&self, repr: bool) -> Result<RlCheckpointText, String> {
        self.record(json!([if repr { "repr" } else { "str" }]))?;
        RlCheckpointText::try_from_code_points([if repr { 0xd800 } else { 0xdfff }])
    }
    fn attribute(
        &self,
        name: &RlCheckpointText,
    ) -> Result<Arc<dyn RlLosslessCheckpointFormatValue>, String> {
        self.record(json!(["attribute", name.as_code_points()]))?;
        Ok(Arc::new(RlLosslessCheckpointValue::Text(
            RlCheckpointText::try_from_code_points([0xdfff])?,
        )))
    }
    fn item(
        &self,
        key: RlLosslessCheckpointFieldIndex,
    ) -> Result<Arc<dyn RlLosslessCheckpointFormatValue>, String> {
        let event = match key {
            RlLosslessCheckpointFieldIndex::Integer(value) => json!(["item", value]),
            RlLosslessCheckpointFieldIndex::Key(value) => json!(["item", value.as_code_points()]),
        };
        self.record(event)?;
        Ok(Arc::new(RlLosslessCheckpointValue::Text(
            RlCheckpointText::try_from_code_points([0xd800, 0xdc00])?,
        )))
    }
}

fn oracle() -> Value {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts/checkpoint-format-probe/surrogate_probe.py");
    let source = PathBuf::from("D:/code/github/qlib/qlib/rl/trainer/callbacks.py");
    let output = Command::new("python")
        .args([fixture.as_os_str(), source.as_os_str()])
        .arg("--oracle-only")
        .stdin(Stdio::null())
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
fn validated_text_and_windows_paths_preserve_every_characterization_string() {
    let data = oracle();
    assert!(
        RlCheckpointText::try_from_code_points([0x11_0000])
            .unwrap_err()
            .contains("out of range")
    );
    assert_eq!(RlCheckpointText::from_utf8("é😀").to_utf8().unwrap(), "é😀");
    assert_eq!(RlCheckpointText::default().len(), 0);
    assert!(RlCheckpointText::default().is_empty());
    for case in data["strings"].as_array().unwrap() {
        let value = text(&case["points"]);
        assert_eq!(
            value.len(),
            usize::try_from(case["length"].as_u64().unwrap()).unwrap()
        );
        assert_eq!(value.to_utf8().is_ok(), case["scalar"].as_bool().unwrap());
        assert_eq!(
            RlLosslessCheckpointValue::Text(value.clone())
                .representation(true)
                .unwrap(),
            text(&case["repr"]),
            "repr mismatch for {:?}",
            value.as_code_points(),
        );
        #[cfg(windows)]
        {
            use std::os::windows::ffi::OsStrExt;
            let actual: Vec<_> = value
                .to_os_string()
                .unwrap()
                .encode_wide()
                .map(u64::from)
                .collect();
            assert_eq!(
                actual,
                case["utf16"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(Value::as_u64)
                    .collect::<Option<Vec<_>>>()
                    .unwrap()
            );
        }
    }
}

#[test]
fn unchanged_qlib_lossless_filename_contract_matches_all_cases() {
    let data = oracle();
    assert_eq!(data["filenames"].as_array().unwrap().len(), 405);
    for case in data["filenames"].as_array().unwrap() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut metrics: IndexMap<RlCheckpointText, Arc<dyn RlLosslessCheckpointFormatValue>> =
            IndexMap::new();
        metrics.insert(
            "text".into(),
            Arc::new(RlLosslessCheckpointValue::Text(text(&case["text"]))),
        );
        metrics.insert(
            "number".into(),
            Arc::new(RlLosslessCheckpointValue::Integer(
                case["number"].as_i64().unwrap().into(),
            )),
        );
        metrics.insert(
            "model".into(),
            Arc::new(Model {
                events: events.clone(),
                fail: None,
            }),
        );
        metrics.insert(
            "spec".into(),
            Arc::new(RlLosslessCheckpointValue::Text(
                RlCheckpointText::try_from_code_points([
                    0xd800,
                    u32::from('>'),
                    u32::from('8'),
                    u32::from('c'),
                ])
                .unwrap(),
            )),
        );
        metrics.insert(
            RlCheckpointText::try_from_code_points([0xd800]).unwrap(),
            Arc::new(RlLosslessCheckpointValue::Text(
                RlCheckpointText::try_from_code_points(
                    "key".chars().map(u32::from).chain([0xdfff]),
                )
                .unwrap(),
            )),
        );
        let actual = PythonRlLosslessCheckpointName.render(
            &text(&case["template"]),
            &7.into(),
            &"20260903123456".into(),
            &metrics,
        );
        if case["error"].is_null() {
            assert_eq!(
                actual.unwrap().as_code_points(),
                text(&case["output"]).as_code_points()
            );
        } else {
            assert!(
                actual.is_err(),
                "expected {}, got {actual:?}",
                case["error"]
            );
        }
        assert_eq!(
            events.lock().unwrap().as_slice(),
            &case["events"].as_array().unwrap()[1..]
        );
    }
}

#[test]
fn lossless_protocol_forwards_arcs_locale_errors_and_lookup_boundaries() {
    let text_value = RlLosslessCheckpointValue::Text(
        RlCheckpointText::try_from_code_points([0xd800, u32::from('x')]).unwrap(),
    );
    assert_eq!(
        text_value.representation(false).unwrap(),
        RlCheckpointText::try_from_code_points([0xd800, u32::from('x')]).unwrap()
    );
    assert_eq!(
        text_value.representation(true).unwrap().to_utf8().unwrap(),
        "'\\ud800x'"
    );
    assert!(
        text_value
            .item(RlLosslessCheckpointFieldIndex::Integer(99))
            .is_err()
    );
    assert!(
        text_value
            .item(RlLosslessCheckpointFieldIndex::Key("x".into()))
            .is_err()
    );
    assert!(
        text_value
            .attribute(&RlCheckpointText::try_from_code_points([0xd800]).unwrap())
            .is_err()
    );
    let shared: Arc<dyn RlLosslessCheckpointFormatValue> =
        Arc::new(RlLosslessCheckpointValue::Integer(1.into()));
    assert_eq!(
        shared.representation(false).unwrap().to_utf8().unwrap(),
        "1"
    );
    assert!(
        shared
            .item(RlLosslessCheckpointFieldIndex::Key("x".into()))
            .is_err()
    );
    assert!(shared.attribute(&"missing".into()).is_err());
    assert_eq!(
        shared
            .attribute(&"real".into())
            .unwrap()
            .format(&"03d".into())
            .unwrap()
            .to_utf8()
            .unwrap(),
        "001"
    );
    let mut metrics = IndexMap::new();
    metrics.insert("iter".into(), RlLosslessCheckpointValue::Integer(1.into()));
    assert!(
        PythonRlLosslessCheckpointName
            .render(&"{iter}".into(), &1.into(), &"".into(), &metrics)
            .unwrap_err()
            .contains("duplicate")
    );
    let empty = IndexMap::<RlCheckpointText, RlLosslessCheckpointValue>::new();
    for template in [
        "{0}",
        "{x",
        "}",
        "{x!}",
        "{x!r?}",
        "{x:{x:{x}}}",
        "{x[]}",
        "{x[missing}",
    ] {
        let mut values = empty.clone();
        values.insert("x".into(), RlLosslessCheckpointValue::Text("x".into()));
        assert!(
            PythonRlLosslessCheckpointName
                .render(&template.into(), &1.into(), &"".into(), &values)
                .is_err(),
            "{template}"
        );
    }
    let mut localized = LocalizedPythonRlLosslessCheckpointName::new(Arc::new(
        RlCheckpointNumericLocale::new(",".into(), ".".into(), &[3, 0]).unwrap(),
    ));
    let mut values = IndexMap::new();
    values.insert("x".into(), RlLosslessCheckpointValue::Float(1234.5));
    assert_eq!(
        localized
            .render(&"{x:n}".into(), &1.into(), &"".into(), &values)
            .unwrap()
            .to_utf8()
            .unwrap(),
        "1.234,5"
    );
}

#[test]
fn primitive_variants_and_parser_failures_are_explicit() {
    for value in [
        RlLosslessCheckpointValue::Boolean(true),
        RlLosslessCheckpointValue::None,
    ] {
        assert!(
            !value
                .format(&RlCheckpointText::default())
                .unwrap()
                .is_empty()
        );
    }
    for (value, attribute, expected) in [
        (RlLosslessCheckpointValue::Integer(2.into()), "imag", "0"),
        (RlLosslessCheckpointValue::Float(2.0), "imag", "0.0"),
        (
            RlLosslessCheckpointValue::Integer(2.into()),
            "denominator",
            "1",
        ),
        (RlLosslessCheckpointValue::Boolean(true), "real", "1"),
    ] {
        assert_eq!(
            value
                .attribute(&attribute.into())
                .unwrap()
                .format(&RlCheckpointText::default())
                .unwrap()
                .to_utf8()
                .unwrap(),
            expected
        );
    }
    let mut plain = IndexMap::new();
    plain.insert("x".into(), RlLosslessCheckpointValue::Text("value".into()));
    assert_eq!(
        PythonRlLosslessCheckpointName
            .render(
                &"{{{x!s}}}-{x!r}-{x!a}-{iter}-{time}".into(),
                &2.into(),
                &"T".into(),
                &plain
            )
            .unwrap()
            .to_utf8()
            .unwrap(),
        "{value}-'value'-'value'-2-T"
    );
    for template in ["{x{}}", "{x[missing", "{x:{x}"] {
        assert!(
            PythonRlLosslessCheckpointName
                .render(&template.into(), &1.into(), &"".into(), &plain)
                .is_err()
        );
    }
    let huge = format!("{{x[{}]}}", "9".repeat(usize::BITS as usize));
    assert!(
        PythonRlLosslessCheckpointName
            .render(&huge.into(), &1.into(), &"".into(), &plain)
            .is_err()
    );
}

#[cfg(windows)]
#[test]
fn real_windows_files_retain_lone_surrogates_and_scalar_aliasing() {
    let directory = tempfile::tempdir().unwrap();
    let high =
        RlCheckpointText::try_from_code_points([0xd800, u32::from('.'), u32::from('p')]).unwrap();
    let path = high.join_to(directory.path()).unwrap();
    std::fs::write(&path, b"payload").unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), b"payload");
    let pair = RlCheckpointText::try_from_code_points([0xd800, 0xdc00]).unwrap();
    let scalar = RlCheckpointText::try_from_code_points([0x10000]).unwrap();
    assert_ne!(pair, scalar);
    std::fs::write(pair.join_to(directory.path()).unwrap(), b"alias").unwrap();
    assert_eq!(
        std::fs::read(scalar.join_to(directory.path()).unwrap()).unwrap(),
        b"alias"
    );
}

#[test]
fn surrogate_repr_preserves_literal_escape_sequences() {
    let value = RlCheckpointText::try_from_code_points(
        [0xd800].into_iter().chain(r"\ue000".chars().map(u32::from)),
    )
    .unwrap();
    assert_eq!(
        RlLosslessCheckpointValue::Text(value)
            .representation(true)
            .unwrap()
            .to_utf8()
            .unwrap(),
        r"'\ud800\\ue000'",
    );
}

#[test]
fn text_format_does_not_require_unused_private_use_characters() {
    let value = RlCheckpointText::try_from_code_points(
        (0xe000..=0xf8ff).chain(0xf_0000..=0xf_fffd).chain([0xd800]),
    )
    .unwrap();
    assert_eq!(
        RlLosslessCheckpointValue::Text(value.clone())
            .format(&"".into())
            .unwrap(),
        value
    );
}

#[test]
fn surrogate_fill_never_rewrites_literal_locale_separators() {
    let locale = RlCheckpointNumericLocale::new(".".into(), "\u{e000}".into(), &[3, 0]).unwrap();
    let spec = RlCheckpointText::try_from_code_points(
        [0xd800].into_iter().chain(">8n".chars().map(u32::from)),
    )
    .unwrap();
    let expected =
        RlCheckpointText::try_from_code_points([0xd800, 0xd800, 0xd800, 49, 0xe000, 50, 51, 52])
            .unwrap();
    assert_eq!(
        RlLosslessCheckpointValue::Integer(1234.into())
            .format_with_locale(&spec, &locale)
            .unwrap(),
        expected
    );
}

#[test]
fn every_python_code_point_survives_string_layout() {
    let all_points = RlCheckpointText::try_from_code_points(0..=0x10_ffff).unwrap();
    assert_eq!(
        RlLosslessCheckpointValue::Text(all_points.clone())
            .format(&"".into())
            .unwrap(),
        all_points
    );
}

#[test]
fn lossless_primitive_layouts_match_live_python_across_tags_and_every_surrogate() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/rl_checkpoint_lossless_format_contract.py");
    let output = Command::new("python").arg(fixture).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(cases.len(), 9427);
    for case in cases {
        let value = match case["kind"].as_str().unwrap() {
            "text" => RlLosslessCheckpointValue::Text(text(&case["value"])),
            "int" => {
                RlLosslessCheckpointValue::Integer(case["value"].as_str().unwrap().parse().unwrap())
            }
            "float" => {
                RlLosslessCheckpointValue::Float(case["value"].as_str().unwrap().parse().unwrap())
            }
            "bool" => RlLosslessCheckpointValue::Boolean(case["value"].as_bool().unwrap()),
            "none" => RlLosslessCheckpointValue::None,
            other => panic!("unexpected fixture value: {other}"),
        };
        let actual = value.format(&text(&case["spec"]));
        if case["error"].is_null() {
            assert_eq!(actual.unwrap(), text(&case["output"]), "{case}");
        } else {
            assert!(actual.is_err(), "{case}: {actual:?}");
        }
    }
}

#[test]
fn model_errors_stop_before_later_fields_and_preserve_exact_events() {
    for (stage, template, event) in [
        ("str", "{model!s:{later}}{later}", json!(["str"])),
        ("repr", "{model!r:{later}}{later}", json!(["repr"])),
        ("repr", "{model!a:{later}}{later}", json!(["repr"])),
        ("format", "{model}{later}", json!(["format", []])),
        ("attribute", "{model.a}{later}", json!(["attribute", [97]])),
        ("item", "{model[0]}{later}", json!(["item", 0])),
    ] {
        let events = Arc::new(Mutex::new(Vec::new()));
        let value: Arc<dyn RlLosslessCheckpointFormatValue> = Arc::new(Model {
            events: events.clone(),
            fail: Some(stage),
        });
        let metrics = IndexMap::from([("model".into(), value.clone()), ("later".into(), value)]);
        assert_eq!(
            PythonRlLosslessCheckpointName
                .render(&template.into(), &0.into(), &"T".into(), &metrics)
                .unwrap_err(),
            format!("injected {stage} failure")
        );
        assert_eq!(*events.lock().unwrap(), vec![event]);
    }
}

#[test]
fn empty_positional_and_chained_field_boundaries_are_explicit() {
    assert_eq!(integer_index(&RlCheckpointText::default()).unwrap(), None);
    let events = Arc::new(Mutex::new(Vec::new()));
    let value = Model {
        events: events.clone(),
        fail: None,
    };
    let metrics = IndexMap::from([("model".into(), value)]);
    for template in [
        "{}",
        "{.real}",
        "{[0]}",
        "{model!",
        "{model[0]z}",
        "{model.a.b}",
    ] {
        assert!(
            PythonRlLosslessCheckpointName
                .render(&template.into(), &0.into(), &"T".into(), &metrics)
                .is_err(),
            "{template}"
        );
    }
    let huge = format!("{{{}}}", "9".repeat(100));
    assert!(
        PythonRlLosslessCheckpointName
            .render(&huge.into(), &0.into(), &"T".into(), &metrics)
            .unwrap_err()
            .contains("too many")
    );
    assert_eq!(
        PythonRlLosslessCheckpointName
            .render(&"{model.a[0]}".into(), &0.into(), &"T".into(), &metrics)
            .unwrap()
            .as_code_points(),
        &[0xdfff]
    );
    let value: &dyn RlLosslessCheckpointFormatValue =
        metrics.get(&RlCheckpointText::from("model")).unwrap();
    for name in ["model[", "model[unfinished"] {
        assert!(
            matches!(resolve(&name.into(), &|_| Ok(value)), Err(error) if error.contains("missing ']'"))
        );
    }
    assert_eq!(
        RlLosslessCheckpointValue::Boolean(false)
            .attribute(&"numerator".into())
            .unwrap()
            .representation(false)
            .unwrap()
            .to_utf8()
            .unwrap(),
        "0"
    );
}

struct LocaleProbe {
    calls: std::sync::atomic::AtomicUsize,
    result: Result<RlCheckpointNumericLocale, String>,
}
impl RlCheckpointLocaleProvider for LocaleProbe {
    fn numeric_locale(&self) -> Result<RlCheckpointNumericLocale, String> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.result.clone()
    }
}

#[test]
fn positional_numeric_tags_preserve_all_literal_markers_and_locale_call_order() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let provider = LocaleProbe {
        calls: AtomicUsize::new(0),
        result: Ok(RlCheckpointNumericLocale::new(
            "\u{e001}\u{e003}".into(),
            "\u{e000}\u{e002}".into(),
            &[3, 0],
        )
        .unwrap()),
    };
    let value = RlLosslessCheckpointValue::Float(1234.5);
    let body = [49, 0xe000, 0xe002, 50, 51, 52, 0xe001, 0xe003, 53];
    for (index, (align, before, after)) in [('>', 4, 0), ('<', 0, 4), ('^', 2, 2), ('=', 4, 0)]
        .into_iter()
        .enumerate()
    {
        let spec = RlCheckpointText::try_from_code_points(
            [0xd800]
                .into_iter()
                .chain(format!("{align}13n").chars().map(u32::from)),
        )
        .unwrap();
        let expected: Vec<_> = std::iter::repeat_n(0xd800, before)
            .chain(body)
            .chain(std::iter::repeat_n(0xd800, after))
            .collect();
        assert_eq!(
            value
                .format_with_locale(&spec, &provider)
                .unwrap()
                .as_code_points(),
            expected
        );
        assert_eq!(provider.calls.load(Ordering::SeqCst), index + 1);
    }
    let failure = LocaleProbe {
        calls: AtomicUsize::new(0),
        result: Err("locale failed".into()),
    };
    let spec = RlCheckpointText::try_from_code_points(
        [0xd800].into_iter().chain(">13n".chars().map(u32::from)),
    )
    .unwrap();
    assert_eq!(
        value.format_with_locale(&spec, &failure).unwrap_err(),
        "locale failed"
    );
    assert_eq!(failure.calls.load(Ordering::SeqCst), 1);
    let invalid = RlCheckpointText::try_from_code_points(
        [0xd800].into_iter().chain(">13,n".chars().map(u32::from)),
    )
    .unwrap();
    assert!(value.format_with_locale(&invalid, &failure).is_err());
    assert_eq!(failure.calls.load(Ordering::SeqCst), 1);
}
