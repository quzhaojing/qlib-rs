use std::{
    collections::VecDeque,
    path::PathBuf,
    process::Command,
    sync::{Arc, Mutex},
};

use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use chrono::NaiveDateTime;
use domain_core::{
    FeatureProvider, FeatureProviderError, FeatureQuery, FeatureResolutionError, FrequencyError,
    get_higher_eq_frequency_features,
};
use num_bigint::BigInt;
use serde_json::{Value, json};

#[derive(Debug, Clone)]
enum Outcome {
    Data(i64),
    Error(FeatureProviderError),
}

#[derive(Debug)]
struct MockProvider {
    outcomes: Mutex<VecDeque<Outcome>>,
    calls: Mutex<Vec<FeatureQuery>>,
}

impl MockProvider {
    fn new(outcomes: impl IntoIterator<Item = Outcome>) -> Self {
        Self {
            outcomes: Mutex::new(outcomes.into_iter().collect()),
            calls: Mutex::new(Vec::new()),
        }
    }

    fn frequencies(&self) -> Vec<String> {
        self.calls
            .lock()
            .expect("call log lock is available")
            .iter()
            .map(|query| query.frequency.clone())
            .collect()
    }
}

impl FeatureProvider for MockProvider {
    fn features(&self, query: &FeatureQuery) -> Result<Vec<RecordBatch>, FeatureProviderError> {
        self.calls
            .lock()
            .expect("call log lock is available")
            .push(query.clone());
        match self
            .outcomes
            .lock()
            .expect("outcome lock is available")
            .pop_front()
            .expect("fixture provides one outcome per call")
        {
            Outcome::Data(value) => {
                let array = Arc::new(Int64Array::from(vec![value])) as ArrayRef;
                Ok(vec![
                    RecordBatch::try_from_iter([("value", array)]).expect("fixture batch is valid"),
                ])
            }
            Outcome::Error(error) => Err(error),
        }
    }
}

fn value_error(message: &str) -> Outcome {
    Outcome::Error(FeatureProviderError::Value {
        message: message.to_owned(),
    })
}

fn key_error(message: &str) -> Outcome {
    Outcome::Error(FeatureProviderError::Key {
        message: message.to_owned(),
    })
}

fn other_error(message: &str) -> Outcome {
    Outcome::Error(FeatureProviderError::Other {
        message: message.to_owned(),
    })
}

fn query(frequency: &str) -> FeatureQuery {
    let mut query = FeatureQuery::new(vec!["A".to_owned()], vec!["$x".to_owned()]);
    query.start_time = Some(
        NaiveDateTime::parse_from_str("2024-01-02 09:30:00", "%Y-%m-%d %H:%M:%S")
            .expect("fixture time is valid"),
    );
    query.end_time = Some(
        NaiveDateTime::parse_from_str("2024-01-03 15:00:00", "%Y-%m-%d %H:%M:%S")
            .expect("fixture time is valid"),
    );
    frequency.clone_into(&mut query.frequency);
    query.disk_cache = BigInt::from(2);
    query
}

fn resolved_value(result: &domain_core::ResolvedFeatures) -> i64 {
    result.batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("fixture batch contains Int64")
        .value(0)
}

#[test]
fn defaults_and_direct_success_preserve_the_exact_unparsed_request() {
    let defaults = FeatureQuery::new(vec!["A".to_owned()], vec!["$x".to_owned()]);
    assert_eq!(defaults.start_time, None);
    assert_eq!(defaults.end_time, None);
    assert_eq!(defaults.frequency, "day");
    assert_eq!(defaults.disk_cache, BigInt::from(1));

    let provider = MockProvider::new([Outcome::Data(7)]);
    let request = query("hour");
    let result = get_higher_eq_frequency_features(&provider, &request)
        .expect("provider may accept a frequency outside the Qlib parser");
    assert_eq!(result.frequency, "hour");
    assert_eq!(resolved_value(&result), 7);
    assert_eq!(provider.frequencies(), ["hour"]);
    assert_eq!(
        provider.calls.lock().unwrap().as_slice(),
        std::slice::from_ref(&request)
    );
}

#[test]
fn calendar_frequencies_retry_day_then_minute_in_python_order() {
    for (requested, outcomes, expected_calls, resolved, value) in [
        (
            "2month",
            vec![value_error("missing"), Outcome::Data(10)],
            vec!["2month", "day"],
            "day",
            10,
        ),
        (
            "W",
            vec![key_error("missing"), value_error("day"), Outcome::Data(11)],
            vec!["W", "day", "1min"],
            "1min",
            11,
        ),
        (
            "day",
            vec![value_error("missing"), Outcome::Data(12)],
            vec!["day", "day"],
            "day",
            12,
        ),
    ] {
        let provider = MockProvider::new(outcomes);
        let result = get_higher_eq_frequency_features(&provider, &query(requested)).unwrap();
        assert_eq!(provider.frequencies(), expected_calls);
        assert_eq!(result.frequency, resolved);
        assert_eq!(resolved_value(&result), value);
    }
}

#[test]
fn minute_fallback_and_every_terminal_error_are_typed() {
    let provider = MockProvider::new([value_error("missing"), Outcome::Data(20)]);
    let result = get_higher_eq_frequency_features(&provider, &query("5MIN")).unwrap();
    assert_eq!(provider.frequencies(), ["5MIN", "1min"]);
    assert_eq!(result.frequency, "1min");
    assert_eq!(resolved_value(&result), 20);

    let initial = FeatureProviderError::Other {
        message: "initial".to_owned(),
    };
    let provider = MockProvider::new([other_error("initial")]);
    assert_eq!(
        get_higher_eq_frequency_features(&provider, &query("week")),
        Err(FeatureResolutionError::Provider(initial))
    );
    assert_eq!(provider.frequencies(), ["week"]);

    let day = FeatureProviderError::Other {
        message: "day".to_owned(),
    };
    let provider = MockProvider::new([value_error("missing"), other_error("day")]);
    assert_eq!(
        get_higher_eq_frequency_features(&provider, &query("week")),
        Err(FeatureResolutionError::Provider(day))
    );
    assert_eq!(provider.frequencies(), ["week", "day"]);

    let final_error = FeatureProviderError::Key {
        message: "minute".to_owned(),
    };
    let provider = MockProvider::new([value_error("missing"), Outcome::Error(final_error.clone())]);
    assert_eq!(
        get_higher_eq_frequency_features(&provider, &query("5min")),
        Err(FeatureResolutionError::Provider(final_error))
    );
    assert_eq!(provider.frequencies(), ["5min", "1min"]);

    let provider = MockProvider::new([value_error("missing")]);
    assert_eq!(
        get_higher_eq_frequency_features(&provider, &query("hour")),
        Err(FeatureResolutionError::Frequency(
            FrequencyError::UnsupportedFormat {
                input: "hour".to_owned()
            }
        ))
    );
    assert_eq!(provider.frequencies(), ["hour"]);
}

#[test]
fn fallback_call_order_matches_live_python_source() {
    let time_source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/utils/time.py");
    let resam_source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/utils/resam.py");
    assert!(time_source.is_file() && resam_source.is_file());
    let script = r#"
import ast, json, re, sys
from typing import Tuple, Union

time_tree = ast.parse(open(sys.argv[1], encoding="utf-8").read(), filename=sys.argv[1])
freq = next(node for node in time_tree.body if isinstance(node, ast.ClassDef) and node.name == "Freq")
namespace = {"re": re, "Tuple": Tuple, "Union": Union}
exec(compile(ast.Module(body=[freq], type_ignores=[]), sys.argv[1], "exec"), namespace)

resam_tree = ast.parse(open(sys.argv[2], encoding="utf-8").read(), filename=sys.argv[2])
function = next(node for node in resam_tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_higher_eq_freq_feature")
function.body = [node for node in function.body if not isinstance(node, ast.ImportFrom)]
ast.fix_missing_locations(function)

class Provider:
    def __init__(self, outcomes): self.outcomes, self.calls = list(outcomes), []
    def features(self, *args, **kwargs):
        self.calls.append(kwargs["freq"])
        outcome = self.outcomes.pop(0)
        if outcome == "value": raise ValueError("value")
        if outcome == "key": raise KeyError("key")
        if outcome == "type": raise TypeError("type")
        return outcome

result = {}
for name, requested, outcomes in [
    ("direct", "hour", ["direct"]),
    ("month_day", "2month", ["value", "day"]),
    ("week_min", "W", ["key", "value", "minute"]),
    ("day_twice", "day", ["value", "day"]),
    ("minute", "5MIN", ["value", "minute"]),
    ("initial_type", "week", ["type"]),
    ("day_type", "week", ["value", "type"]),
    ("final_key", "5min", ["value", "key"]),
    ("invalid", "hour", ["value"]),
]:
    provider = Provider(outcomes)
    scope = {"Freq": namespace["Freq"], "D": provider}
    exec(compile(ast.Module(body=[function], type_ignores=[]), sys.argv[2], "exec"), scope)
    try:
        _, resolved = scope["get_higher_eq_freq_feature"](["A"], ["$x"], freq=requested)
        result[name] = {"calls": provider.calls, "resolved": resolved}
    except Exception as error:
        result[name] = {"calls": provider.calls, "error": type(error).__name__}
print(json.dumps(result, sort_keys=True))
"#;
    let output = Command::new(std::env::var_os("PYTHON").unwrap_or_else(|| "python".into()))
        .arg("-c")
        .arg(script)
        .arg(&time_source)
        .arg(&resam_source)
        .output()
        .expect("Python interpreter starts");
    assert!(
        output.status.success(),
        "Python snapshot failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let actual: Value = serde_json::from_slice(&output.stdout).expect("valid Python JSON");
    let expected = json!({
        "direct": {"calls": ["hour"], "resolved": "hour"},
        "month_day": {"calls": ["2month", "day"], "resolved": "day"},
        "week_min": {"calls": ["W", "day", "1min"], "resolved": "1min"},
        "day_twice": {"calls": ["day", "day"], "resolved": "day"},
        "minute": {"calls": ["5MIN", "1min"], "resolved": "1min"},
        "initial_type": {"calls": ["week"], "error": "TypeError"},
        "day_type": {"calls": ["week", "day"], "error": "TypeError"},
        "final_key": {"calls": ["5min", "1min"], "error": "KeyError"},
        "invalid": {"calls": ["hour"], "error": "ValueError"},
    });
    assert_eq!(actual, expected);
}
