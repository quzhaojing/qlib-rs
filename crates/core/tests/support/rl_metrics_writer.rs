use super::*;
use serde_json::{Value, json};
use std::{io, process::Command, sync::Arc};

#[path = "rl_metrics_writer_integration.rs"]
mod integration;

struct Custom(String);
impl crate::TrainingMetricDisplay for Custom {
    fn render(&self) -> Result<String, String> {
        if self.0 == "FAIL" {
            Err("custom format failure".into())
        } else {
            Ok(self.0.clone())
        }
    }
}
fn scalar(value: &Value) -> TrainingMetricScalar {
    match value[0].as_str().unwrap() {
        "int" => TrainingMetricScalar::Integer(value[1].as_str().unwrap().parse().unwrap()),
        "float" => TrainingMetricScalar::Float(value[1].as_str().unwrap().parse().unwrap()),
        "text" => TrainingMetricScalar::Text(value[1].as_str().unwrap().into()),
        "bool" => TrainingMetricScalar::Boolean(value[1].as_bool().unwrap()),
        "custom" => {
            TrainingMetricScalar::Custom(Arc::new(Custom(value[1].as_str().unwrap().into())))
        }
        "null" => TrainingMetricScalar::Null,
        other => panic!("unknown value: {other}"),
    }
}
fn record(step: &Value) -> RlMetricRecord<TrainingMetricScalar> {
    step["metrics"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|pair| (pair[0].as_str().unwrap().into(), scalar(&pair[1])))
        .collect()
}
fn phase(step: &Value) -> RlTrainerHook {
    match step["phase"].as_str().unwrap() {
        "train" => RlTrainerHook::TrainEnd,
        "val" => RlTrainerHook::ValidateEnd,
        "fit_start" => RlTrainerHook::FitStart,
        "test_end" => RlTrainerHook::TestEnd,
        other => panic!("unknown phase: {other}"),
    }
}
fn files(directory: &Path) -> Value {
    let mut files = serde_json::Map::new();
    for name in ["train_result.csv", "validation_result.csv"] {
        let file = directory.join(name);
        if file.is_file() {
            files.insert(name.into(), json!(fs::read_to_string(file).unwrap()));
        }
    }
    Value::Object(files)
}
fn exercise(spec: &Value) -> Value {
    let temp = tempfile::tempdir().unwrap();
    let directory = temp.path().join("metrics");
    if spec["directory_file"] == true {
        fs::write(&directory, "existing").unwrap();
    }
    let mut writer = match RlMetricsWriter::<TrainingMetricScalar>::new(
        directory.clone(),
        CsvRlMetricsTableSink::default(),
    ) {
        Ok(writer) => writer,
        Err(error) => {
            assert!(matches!(error, RlMetricsWriterError::Directory(_)));
            assert!(!error.to_string().is_empty());
            return json!({"spec":spec, "init_error":true, "outputs":[]});
        }
    };
    let runtime = Arc::new(RlTrainerRuntime::new(None));
    let mut control = RlTrainerControl {
        runtime: runtime.clone(),
        config: crate::RlTrainerConfig::default(),
    };
    let mut outputs = vec![];
    for step in spec["steps"].as_array().unwrap() {
        if step["blocked"] == true {
            fs::create_dir(directory.join(if phase(step) == RlTrainerHook::TrainEnd {
                "train_result.csv"
            } else {
                "validation_result.csv"
            }))
            .unwrap();
        }
        runtime
            .update(|s| {
                s.metrics = if step["absent"] == true {
                    None
                } else {
                    Some(record(step))
                }
            })
            .unwrap();
        let result = writer.call(phase(step), &mut control, &mut ());
        outputs.push(json!({"error":result.is_err(), "train":writer.train_records.len(), "val":writer.valid_records.len(), "files":files(&directory)}));
    }
    let before = (writer.train_records.len(), writer.valid_records.len());
    writer.save_checkpoint().unwrap();
    writer.load_checkpoint(&()).unwrap();
    assert_eq!(
        (writer.train_records.len(), writer.valid_records.len()),
        before
    );
    json!({"spec":spec,"init_error":false,"outputs":outputs})
}

#[test]
fn real_python_callback_and_pandas_csv_match_after_every_hook() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .args([
            root.join("tests/fixtures/rl_metrics_writer_contract.py"),
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
    assert_eq!(cases.len(), 138);
    for expected in cases {
        assert_eq!(
            exercise(&expected["spec"]),
            expected,
            "spec: {}",
            expected["spec"]
        );
    }
}

struct BrokenWriter {
    remaining: usize,
    flush_error: bool,
    written: Vec<u8>,
}
impl Write for BrokenWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.remaining == 0 {
            return Err(io::Error::other("write failed"));
        }
        let count = self.remaining.min(bytes.len());
        self.written.extend_from_slice(&bytes[..count]);
        self.remaining -= count;
        Ok(count)
    }
    fn flush(&mut self) -> io::Result<()> {
        if self.flush_error {
            Err(io::Error::other("flush failed"))
        } else {
            Ok(())
        }
    }
}

#[test]
fn csv_writer_reports_header_row_and_flush_failures_and_supports_explicit_newlines() {
    for (header, text) in [
        ("h".repeat(50_000), "small".into()),
        ("header".into(), "x".repeat(50_000)),
    ] {
        let mut writer = BrokenWriter {
            remaining: 5,
            flush_error: false,
            written: vec![],
        };
        let records = [IndexMap::from([(header, TrainingMetricScalar::Text(text))])];
        let error =
            write_rl_metrics_csv(&records, &mut writer, csv::Terminator::Any(b'\n')).unwrap_err();
        assert!(matches!(error, RlMetricsCsvError::Csv(_)), "{error:?}");
        assert!(error.to_string().contains("write failed"));
        assert_eq!(writer.written.len(), 5);
    }
    let records = [
        IndexMap::from([("x".into(), Some(1.0))]),
        IndexMap::from([("x".into(), None)]),
    ];
    let mut writer = BrokenWriter {
        remaining: usize::MAX,
        flush_error: true,
        written: vec![],
    };
    let error =
        write_rl_metrics_csv(&records, &mut writer, csv::Terminator::Any(b'\n')).unwrap_err();
    assert!(matches!(error, RlMetricsCsvError::Io(_)));
    assert!(error.to_string().contains("flush failed"));
    assert_eq!(
        String::from_utf8(writer.written).unwrap(),
        ",x\n0,1.0\n1,\n"
    );
    let mut output = vec![];
    write_rl_metrics_csv(
        &[IndexMap::from([("x".into(), 1.5)])],
        &mut output,
        csv::Terminator::CRLF,
    )
    .unwrap();
    assert_eq!(output, b",x\r\n0,1.5\r\n");
}

#[test]
fn inference_failures_precede_output_and_format_failures_retain_written_history() {
    let huge = TrainingMetricScalar::Integer(num_bigint::BigInt::from(10_u32).pow(1000));
    let mut bytes = b"existing".to_vec();
    let error = write_rl_metrics_csv(
        &[IndexMap::from([("x".into(), huge)])],
        &mut bytes,
        csv::Terminator::CRLF,
    )
    .unwrap_err();
    assert!(matches!(error, RlMetricsCsvError::IntegerOverflow));
    assert_eq!(bytes, b"existing");
    assert_eq!(error.to_string(), "int too large to convert to float");
    let custom = TrainingMetricScalar::Custom(Arc::new(Custom("FAIL".into())));
    let error = write_rl_metrics_csv(
        &[IndexMap::from([("x".into(), custom)])],
        &mut bytes,
        csv::Terminator::CRLF,
    )
    .unwrap_err();
    assert!(matches!(error, RlMetricsCsvError::Format(_)));
    assert!(error.to_string().contains("custom format failure"));
    assert_eq!(bytes, b"existing,x\r\n");
}

#[derive(Default)]
struct RecordingSink {
    fail_directory: bool,
    fail_table: bool,
    writes: Vec<Vec<Vec<i64>>>,
    runtime: Option<Arc<RlTrainerRuntime<Arc<std::sync::Mutex<i64>>>>>,
}
impl RlMetricsTableSink<Arc<std::sync::Mutex<i64>>> for RecordingSink {
    fn create_directory(&mut self, _: &Path) -> Result<(), String> {
        if self.fail_directory {
            Err("directory plugin".into())
        } else {
            Ok(())
        }
    }
    fn write_records(
        &mut self,
        _: &Path,
        records: &[RlMetricRecord<Arc<std::sync::Mutex<i64>>>],
    ) -> Result<(), String> {
        self.runtime
            .as_ref()
            .unwrap()
            .update(|s| s.current_stage = "plugin".into())
            .unwrap();
        self.writes.push(
            records
                .iter()
                .map(|row| row.values().map(|v| *v.lock().unwrap()).collect())
                .collect(),
        );
        if self.fail_table {
            Err("table plugin".into())
        } else {
            Ok(())
        }
    }
}

#[test]
fn whole_table_plugins_keep_shallow_handles_failures_and_runtime_lock_boundaries() {
    assert!(
        matches!(RlMetricsWriter::new(PathBuf::from("unused"), RecordingSink { fail_directory: true, ..RecordingSink::default() }), Err(RlMetricsWriterError::Directory(message)) if message == "directory plugin")
    );
    let runtime = Arc::new(RlTrainerRuntime::new(None));
    let value = Arc::new(std::sync::Mutex::new(1));
    runtime
        .update(|s| s.metrics = Some(IndexMap::from([("x".into(), value.clone())])))
        .unwrap();
    let mut writer = RlMetricsWriter::new(
        PathBuf::from("unused"),
        RecordingSink {
            runtime: Some(runtime.clone()),
            fail_table: true,
            ..RecordingSink::default()
        },
    )
    .unwrap();
    assert_eq!(
        writer.on_train_end(&runtime),
        Err(RlMetricsWriterError::Table("table plugin".into()))
    );
    assert_eq!(writer.train_records.len(), 1);
    assert!(Arc::ptr_eq(&writer.train_records[0]["x"], &value));
    *value.lock().unwrap() = 7;
    writer.sink.fail_table = false;
    writer.on_train_end(&runtime).unwrap();
    assert_eq!(
        writer.sink.writes,
        vec![vec![vec![1]], vec![vec![7], vec![7]]]
    );
    assert_eq!(runtime.read(|s| s.current_stage.clone()).unwrap(), "plugin");
    let error = RlMetricsWriterError::Table("table plugin".into());
    assert_eq!(error, error.clone());
    assert!(format!("{error:?}: {error}").contains("table plugin"));
}

#[test]
fn poisoned_runtime_and_missing_metrics_never_append_or_write() {
    let directory = tempfile::tempdir().unwrap();
    let runtime = RlTrainerRuntime::<f64>::new(None);
    let mut writer =
        RlMetricsWriter::new(directory.path().into(), CsvRlMetricsTableSink::default()).unwrap();
    assert_eq!(
        writer.on_train_end(&runtime),
        Err(RlMetricsWriterError::MissingMetrics)
    );
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = runtime.update::<()>(|_| panic!("runtime poison"));
        }))
        .is_err()
    );
    let error = RlMetricsWriterError::Runtime(RlTrainerStateError::Poisoned);
    assert_eq!(writer.on_train_end(&runtime), Err(error.clone()));
    assert_eq!(writer.on_validate_end(&runtime), Err(error));
    assert!(writer.train_records.is_empty());
    assert!(writer.valid_records.is_empty());
    assert_eq!(files(directory.path()), json!({}));
}
