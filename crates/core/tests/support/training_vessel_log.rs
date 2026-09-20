use std::{
    path::PathBuf,
    process::Command,
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use arrow_array::{BooleanArray, StringArray};
use serde_json::{Value, json};

use super::*;
use crate::{TrainingTrainerField, TrainingTrainerView};

type Messages = Arc<Mutex<Vec<String>>>;

struct Trainer {
    iteration: Mutex<BigInt>,
    events: Messages,
    fail: AtomicBool,
}

impl TrainingTrainerView for Trainer {
    fn current_iteration(&self) -> Result<BigInt, String> {
        self.events.lock().unwrap().push("iteration".into());
        if self.fail.load(Ordering::SeqCst) {
            return Err("trainer failure".into());
        }
        let mut iteration = self.iteration.lock().unwrap();
        let result = iteration.clone();
        *iteration += 1;
        Ok(result)
    }
    fn fast_dev_run(&self) -> Result<Option<i64>, String> {
        Ok(None)
    }
}

fn bind(start: i64, events: &Messages) -> (TrainingVesselBinding, Arc<Trainer>) {
    let trainer = Arc::new(Trainer {
        iteration: Mutex::new(start.into()),
        events: Arc::clone(events),
        fail: AtomicBool::new(false),
    });
    let view: Arc<dyn TrainingTrainerView> = trainer.clone();
    let mut binding = TrainingVesselBinding::default();
    binding.assign_trainer(&view);
    (binding, trainer)
}

struct Sink {
    messages: Messages,
    events: Messages,
    fail: bool,
}
impl TrainingMetricSink for Sink {
    fn info(&mut self, message: &str) -> Result<(), String> {
        self.messages.lock().unwrap().push(message.into());
        self.events.lock().unwrap().push(message.into());
        if self.fail {
            Err("sink failure".into())
        } else {
            Ok(())
        }
    }
}

struct Custom {
    events: Messages,
    fail: bool,
}
impl TrainingMetricDisplay for Custom {
    fn render(&self) -> Result<String, String> {
        self.events.lock().unwrap().push("format".into());
        if self.fail {
            Err("format failed".into())
        } else {
            Ok("(1, 2)".into())
        }
    }
}

struct Reducer {
    events: Messages,
    fail: bool,
}
impl TrainingMetricReducer for Reducer {
    fn mean(&mut self, _values: &dyn Array) -> Result<f64, TrainingMetricReductionError> {
        self.events.lock().unwrap().push("reduce".into());
        if self.fail {
            Err(TrainingMetricReductionError::Plugin("reduce failed".into()))
        } else {
            Ok(42.0)
        }
    }
}

fn logger(events: &Messages, messages: &Messages, fail: bool) -> TrainingVesselLog {
    TrainingVesselLog::with_plugins(
        Box::new(ArrowTrainingMetricReducer),
        Box::new(Sink {
            messages: Arc::clone(messages),
            events: Arc::clone(events),
            fail,
        }),
    )
}

fn scalar(value: TrainingMetricScalar) -> TrainingMetricValue {
    TrainingMetricValue::Scalar(value)
}
fn numeric(values: Vec<f64>) -> TrainingMetricValue {
    TrainingMetricValue::Numeric(Arc::new(Float64Array::from(values)))
}

fn dtype(name: &str) -> DataType {
    match name {
        "int8" => DataType::Int8,
        "int16" => DataType::Int16,
        "int32" => DataType::Int32,
        "int64" => DataType::Int64,
        "uint8" => DataType::UInt8,
        "uint16" => DataType::UInt16,
        "uint32" => DataType::UInt32,
        "uint64" => DataType::UInt64,
        "float16" => DataType::Float16,
        "float32" => DataType::Float32,
        "float64" => DataType::Float64,
        "bool" => DataType::Boolean,
        _ => panic!("unknown fixture dtype"),
    }
}

fn python_fixture() -> Value {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .args([
            root.join("tests/fixtures/training_vessel_log_contract.py"),
            root.join("../../../qlib/qlib/rl/trainer/vessel.py"),
        ])
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
fn all_numeric_dtypes_and_scalar_messages_match_live_qlib() {
    let python = python_fixture();
    let events = Arc::new(Mutex::new(vec![]));
    let messages = Arc::new(Mutex::new(vec![]));
    let (binding, trainer) = bind(0, &events);
    let mut values = IndexMap::new();
    for input in python["inputs"].as_array().unwrap() {
        let name = input["name"].as_str().unwrap().to_owned();
        let data_type = dtype(input["dtype"].as_str().unwrap());
        let raw = input["values"].as_array().unwrap();
        let array: ArrayRef = if data_type == DataType::Boolean {
            Arc::new(BooleanArray::from(
                raw.iter().map(|v| v.as_bool().unwrap()).collect::<Vec<_>>(),
            ))
        } else {
            let numbers = raw
                .iter()
                .map(|v| {
                    v.as_f64()
                        .unwrap_or_else(|| v.as_str().unwrap().parse().unwrap())
                })
                .collect::<Vec<_>>();
            cast(&Float64Array::from(numbers), &data_type).unwrap()
        };
        values.insert(name, TrainingMetricValue::Numeric(array));
    }
    values.insert("list_value".into(), numeric(vec![1.0, 2.5]));
    values.insert("nested_list".into(), numeric(vec![1.0, 2.0, 3.0, 4.0]));
    values.insert(
        "tuple_value".into(),
        scalar(TrainingMetricScalar::Custom(Arc::new(Custom {
            events: Arc::clone(&events),
            fail: false,
        }))),
    );
    values.insert(
        "text".into(),
        scalar(TrainingMetricScalar::Text("raw text".into())),
    );
    values.insert(
        "boolean".into(),
        scalar(TrainingMetricScalar::Boolean(true)),
    );
    values.insert("null".into(), scalar(TrainingMetricScalar::Null));
    values.insert(
        "integer".into(),
        scalar(TrainingMetricScalar::Integer(BigInt::from(10).pow(30))),
    );
    values.insert(
        "signed_zero".into(),
        scalar(TrainingMetricScalar::Float(-0.0)),
    );
    values.insert(
        "scientific".into(),
        scalar(TrainingMetricScalar::Float(1e16)),
    );
    values.insert("tiny".into(), scalar(TrainingMetricScalar::Float(1e-5)));
    logger(&events, &messages, false)
        .log_dict(&binding, &values)
        .unwrap();
    assert_eq!(json!(*messages.lock().unwrap()), python["messages"]);
    assert_eq!(
        trainer.iteration.lock().unwrap().to_string(),
        python["reads"].as_u64().unwrap().to_string()
    );
    for case in python["float_texts"].as_array().unwrap() {
        let bits: u64 = case["bits"].as_str().unwrap().parse().unwrap();
        assert_eq!(
            python_float(f64::from_bits(bits)),
            case["text"].as_str().unwrap()
        );
    }
}

#[test]
fn reduction_special_values_and_bounded_rounding_are_explicit() {
    let mut reducer = ArrowTrainingMetricReducer;
    for data_type in [DataType::Float16, DataType::Float32, DataType::Float64] {
        let empty = arrow_array::new_empty_array(&data_type);
        assert!(reducer.mean(empty.as_ref()).unwrap().is_nan());
    }
    assert_eq!(
        reducer
            .mean(&Float32Array::from(vec![f32::MAX, f32::MAX]))
            .unwrap()
            .to_bits(),
        f64::INFINITY.to_bits()
    );
    assert_eq!(
        reducer
            .mean(&Float64Array::from(vec![f64::NEG_INFINITY]))
            .unwrap()
            .to_bits(),
        f64::NEG_INFINITY.to_bits()
    );
    assert!(
        reducer
            .mean(&Float64Array::from(vec![f64::NAN, 2.0]))
            .unwrap()
            .is_nan()
    );
    let half_values = cast(
        &Float64Array::from(vec![65504.0, 65504.0]),
        &DataType::Float16,
    )
    .unwrap();
    assert_eq!(
        reducer.mean(half_values.as_ref()).unwrap().to_bits(),
        65504_f64.to_bits()
    );
    let python = python_fixture();
    let numbers = python["reduction_inputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect::<Vec<_>>();
    let max_input = numbers.iter().copied().fold(0.0, f64::max);
    for (name, epsilon) in [
        ("float16", 2_f64.powi(-10)),
        ("float32", f64::from(f32::EPSILON)),
        ("float64", f64::EPSILON),
    ] {
        let input = cast(&Float64Array::from(numbers.clone()), &dtype(name)).unwrap();
        let actual = reducer.mean(input.as_ref()).unwrap();
        let expected = python["means"][name].as_f64().unwrap();
        // Forward-error bound for finite, positive, non-cancelling inputs: n*eps*max.
        // Half inputs accumulate in f32; allow one final half rounding step.
        let tolerance = if name == "float16" {
            epsilon * max_input
        } else {
            1000.0 * epsilon * max_input
        };
        assert!(
            (actual - expected).abs() <= tolerance,
            "{name}: {actual} != {expected}"
        );
    }
    assert_eq!(
        reducer.mean(&Float64Array::from(vec![None, Some(1.0)])),
        Err(TrainingMetricReductionError::NullValues)
    );
    assert_eq!(
        reducer.mean(&StringArray::from(vec!["a", "b"])),
        Err(TrainingMetricReductionError::UnsupportedDtype(
            DataType::Utf8
        ))
    );
    assert_eq!(
        render_scalar(&TrainingMetricScalar::Boolean(false)).unwrap(),
        "False"
    );
    for (value, expected) in [
        (f64::NEG_INFINITY, "-inf"),
        (1e-4, "0.0001"),
        (1e-5, "1e-05"),
        (1e15, "1000000000000000.0"),
        (1e16, "1e+16"),
        (f64::MAX, "1.7976931348623157e+308"),
        (f64::MIN_POSITIVE, "2.2250738585072014e-308"),
    ] {
        assert_eq!(python_float(value), expected);
    }
}

#[test]
fn failures_preserve_python_reduction_iteration_format_sink_order() {
    let python = python_fixture();
    let events = Arc::new(Mutex::new(vec![]));
    let messages = Arc::new(Mutex::new(vec![]));
    let (binding, trainer) = bind(5, &events);
    let mut log = logger(&events, &messages, false);
    let unsupported = TrainingMetricValue::Numeric(Arc::new(StringArray::from(vec!["a", "b"])));
    assert!(matches!(
        log.log(&binding, "bad_array", &unsupported),
        Err(TrainingVesselLogError::Reduction(_))
    ));
    assert_eq!(json!(*events.lock().unwrap()), python["reduction_events"]);
    let custom = scalar(TrainingMetricScalar::Custom(Arc::new(Custom {
        events: Arc::clone(&events),
        fail: true,
    })));
    assert_eq!(
        log.log(&binding, "custom", &custom),
        Err(TrainingVesselLogError::Format("format failed".into()))
    );
    assert_eq!(json!(*events.lock().unwrap()), python["format_events"]);
    events.lock().unwrap().clear();
    *trainer.iteration.lock().unwrap() = 5.into();
    let values = IndexMap::from([
        (
            "first".into(),
            scalar(TrainingMetricScalar::Integer(1.into())),
        ),
        (
            "second".into(),
            scalar(TrainingMetricScalar::Integer(2.into())),
        ),
    ]);
    assert_eq!(
        logger(&events, &messages, true).log_dict(&binding, &values),
        Err(TrainingVesselLogError::Sink("sink failure".into()))
    );
    assert_eq!(json!(*events.lock().unwrap()), python["sink_events"]);
    assert_eq!(messages.lock().unwrap().len(), 1);
    events.lock().unwrap().clear();
    trainer.fail.store(true, Ordering::SeqCst);
    assert_eq!(
        log.log(&binding, "custom", &custom),
        Err(TrainingVesselLogError::Binding(
            TrainingVesselBindingError::Access {
                field: TrainingTrainerField::CurrentIteration,
                message: "trainer failure".into()
            }
        ))
    );
    assert_eq!(*events.lock().unwrap(), vec!["iteration"]);
    events.lock().unwrap().clear();
    drop(trainer);
    assert_eq!(
        log.log(&binding, "expired", &custom),
        Err(TrainingVesselLogError::Binding(
            TrainingVesselBindingError::Expired
        ))
    );
    assert!(events.lock().unwrap().is_empty());
    let unassigned = TrainingVesselBinding::default();
    assert_eq!(
        log.log(&unassigned, "missing", &custom),
        Err(TrainingVesselLogError::Binding(
            TrainingVesselBindingError::Unassigned
        ))
    );
    assert!(log.log_dict(&unassigned, &IndexMap::new()).is_ok());
}

#[test]
fn reduction_plugins_empty_maps_and_duplicate_names_are_ordered() {
    let events = Arc::new(Mutex::new(vec![]));
    let messages = Arc::new(Mutex::new(vec![]));
    let (binding, _trainer) = bind(-2, &events);
    let mut log = TrainingVesselLog::with_plugins(
        Box::new(Reducer {
            events: Arc::clone(&events),
            fail: false,
        }),
        Box::new(Sink {
            events: Arc::clone(&events),
            messages: Arc::clone(&messages),
            fail: false,
        }),
    );
    log.log(&binding, "value", &numeric(vec![])).unwrap();
    assert_eq!(
        *events.lock().unwrap(),
        vec!["reduce", "iteration", "[Iter -1] value = 42.0"]
    );
    events.lock().unwrap().clear();
    let mut failed = TrainingVesselLog::with_plugins(
        Box::new(Reducer {
            events: Arc::clone(&events),
            fail: true,
        }),
        Box::new(Sink {
            events: Arc::clone(&events),
            messages: Arc::clone(&messages),
            fail: false,
        }),
    );
    let result = failed.log(&binding, "value", &numeric(vec![]));
    assert_eq!(
        result,
        Err(TrainingVesselLogError::Reduction(
            TrainingMetricReductionError::Plugin("reduce failed".into())
        ))
    );
    assert!(result.unwrap_err().to_string().contains("reduce failed"));
    assert_eq!(*events.lock().unwrap(), vec!["reduce"]);
    let mut values = IndexMap::from([
        ("first".into(), scalar(TrainingMetricScalar::Null)),
        (
            "second".into(),
            scalar(TrainingMetricScalar::Boolean(false)),
        ),
    ]);
    values.insert(
        "first".into(),
        scalar(TrainingMetricScalar::Text("replacement".into())),
    );
    logger(&events, &messages, false)
        .log_dict(&binding, &values)
        .unwrap();
    assert_eq!(
        &messages.lock().unwrap()[1..],
        &["[Iter 0] first = replacement", "[Iter 1] second = False"]
    );
}

struct Capture(Messages);
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
                if field.name() == "message" {
                    self.0.lock().unwrap().push(format!("{value:?}"));
                }
            },
        );
    }
    fn enter(&self, _span: &tracing::span::Id) {}
    fn exit(&self, _span: &tracing::span::Id) {}
}

#[test]
fn default_sink_delivers_the_formatted_message_to_tracing() {
    let events = Arc::new(Mutex::new(vec![]));
    let (binding, _trainer) = bind(0, &events);
    let mut log = TrainingVesselLog::default();
    log.log(&binding, "disabled", &scalar(TrainingMetricScalar::Null))
        .unwrap();
    let messages = Arc::new(Mutex::new(vec![]));
    let _guard = tracing::subscriber::set_default(Capture(Arc::clone(&messages)));
    log.log(&binding, "answer", &numeric(vec![40.0, 44.0]))
        .unwrap();
    assert_eq!(*messages.lock().unwrap(), vec!["[Iter 2] answer = 42.0"]);
}
