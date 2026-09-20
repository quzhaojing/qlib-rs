//! Qlib `MetricsWriter`: append ordered histories, then rewrite the selected CSV file.

use crate::{
    RlCheckpointState, RlTrainerCallback, RlTrainerControl, RlTrainerDriverError, RlTrainerHook,
    RlTrainerRuntime, RlTrainerStateError, TrainingMetricScalar,
};
use indexmap::IndexMap;
use num_traits::ToPrimitive;
use std::{
    borrow::Cow,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};
use thiserror::Error;

pub type RlMetricRecord<M = f64> = IndexMap<String, M>;

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum RlMetricsWriterError {
    #[error(transparent)]
    Runtime(#[from] RlTrainerStateError),
    #[error("trainer metrics have not been initialized")]
    MissingMetrics,
    #[error("metrics directory creation failed: {0}")]
    Directory(String),
    #[error("metrics table persistence failed: {0}")]
    Table(String),
}

/// A whole-history sink can use an alternate dataframe/storage backend without coercing M.
pub trait RlMetricsTableSink<M> {
    /// # Errors
    /// Returns directory initialization failures before a callback is constructed.
    fn create_directory(&mut self, path: &Path) -> Result<(), String>;
    /// # Errors
    /// Returns persistence failures; the callback retains the newly appended record.
    fn write_records(&mut self, path: &Path, records: &[RlMetricRecord<M>]) -> Result<(), String>;
}

pub struct RlMetricsWriter<M = f64, S = CsvRlMetricsTableSink> {
    pub dirpath: PathBuf,
    pub train_records: Vec<RlMetricRecord<M>>,
    pub valid_records: Vec<RlMetricRecord<M>>,
    pub sink: S,
}

impl<M, S: RlMetricsTableSink<M>> RlMetricsWriter<M, S> {
    /// # Errors
    /// Returns the sink's directory creation error before initializing histories.
    pub fn new(dirpath: PathBuf, mut sink: S) -> Result<Self, RlMetricsWriterError> {
        sink.create_directory(&dirpath)
            .map_err(RlMetricsWriterError::Directory)?;
        Ok(Self {
            dirpath,
            train_records: vec![],
            valid_records: vec![],
            sink,
        })
    }
    fn record(
        &mut self,
        runtime: &RlTrainerRuntime<M>,
        validation: bool,
    ) -> Result<(), RlMetricsWriterError>
    where
        M: Clone,
    {
        let record = runtime.read(|state| {
            let metrics = state
                .metrics
                .as_ref()
                .ok_or(RlMetricsWriterError::MissingMetrics)?;
            Ok::<_, RlMetricsWriterError>(
                metrics
                    .iter()
                    .filter(|(key, _)| key.starts_with("val/") == validation)
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect(),
            )
        })??;
        let (records, filename) = if validation {
            (&mut self.valid_records, "validation_result.csv")
        } else {
            (&mut self.train_records, "train_result.csv")
        };
        records.push(record);
        self.sink
            .write_records(&self.dirpath.join(filename), records)
            .map_err(RlMetricsWriterError::Table)
    }
    /// # Errors
    /// Reports metric access or table-write errors; append precedes persistence.
    pub fn on_train_end(
        &mut self,
        runtime: &RlTrainerRuntime<M>,
    ) -> Result<(), RlMetricsWriterError>
    where
        M: Clone,
    {
        self.record(runtime, false)
    }
    /// # Errors
    /// Reports metric access or table-write errors; validation prefixes are retained.
    pub fn on_validate_end(
        &mut self,
        runtime: &RlTrainerRuntime<M>,
    ) -> Result<(), RlMetricsWriterError>
    where
        M: Clone,
    {
        self.record(runtime, true)
    }
}

impl<M: Clone, S: RlMetricsTableSink<M>, V> RlTrainerCallback<V, M> for RlMetricsWriter<M, S> {
    fn call(
        &mut self,
        hook: RlTrainerHook,
        control: &mut RlTrainerControl<M>,
        _vessel: &mut V,
    ) -> Result<(), RlTrainerDriverError> {
        match hook {
            RlTrainerHook::TrainEnd => self.on_train_end(&control.runtime),
            RlTrainerHook::ValidateEnd => self.on_validate_end(&control.runtime),
            _ => Ok(()),
        }
        .map_err(|error| RlTrainerDriverError::Plugin {
            stage: "metrics_writer".into(),
            message: error.to_string(),
        })
    }
}

/// Qlib inherits Callback's None checkpoint and no-op restore; histories are not checkpointed.
impl<M, S> RlCheckpointState<()> for RlMetricsWriter<M, S> {
    fn save_checkpoint(&mut self) -> Result<(), String> {
        Ok(())
    }
    fn load_checkpoint(&mut self, (): &()) -> Result<(), String> {
        Ok(())
    }
}

/// Adapt scalar metric values for the bundled pandas-compatible scalar CSV sink.
/// Custom/object formatting remains deferred until rows are emitted.
pub trait RlMetricsCsvValue {
    fn csv_scalar(&self) -> Cow<'_, TrainingMetricScalar>;
}
impl RlMetricsCsvValue for TrainingMetricScalar {
    fn csv_scalar(&self) -> Cow<'_, TrainingMetricScalar> {
        Cow::Borrowed(self)
    }
}
impl RlMetricsCsvValue for f64 {
    fn csv_scalar(&self) -> Cow<'_, TrainingMetricScalar> {
        Cow::Owned(TrainingMetricScalar::Float(*self))
    }
}
impl RlMetricsCsvValue for Option<f64> {
    fn csv_scalar(&self) -> Cow<'_, TrainingMetricScalar> {
        Cow::Owned(self.map_or(TrainingMetricScalar::Null, TrainingMetricScalar::Float))
    }
}

#[derive(Debug, Error)]
pub enum RlMetricsCsvError {
    #[error("int too large to convert to float")]
    IntegerOverflow,
    #[error(transparent)]
    Csv(#[from] csv::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("metric formatting failed: {0}")]
    Format(String),
}

// pandas defaults to os.linesep. The explicit terminator can also be selected by an adapter.
const PLATFORM_TERMINATOR: csv::Terminator = if cfg!(windows) {
    csv::Terminator::CRLF
} else {
    csv::Terminator::Any(b'\n')
};
pub struct CsvRlMetricsTableSink {
    pub terminator: csv::Terminator,
}
impl Default for CsvRlMetricsTableSink {
    fn default() -> Self {
        Self {
            terminator: PLATFORM_TERMINATOR,
        }
    }
}
impl<M: RlMetricsCsvValue> RlMetricsTableSink<M> for CsvRlMetricsTableSink {
    fn create_directory(&mut self, path: &Path) -> Result<(), String> {
        fs::create_dir_all(path).map_err(|error| error.to_string())
    }
    fn write_records(&mut self, path: &Path, records: &[RlMetricRecord<M>]) -> Result<(), String> {
        let table = ScalarCsvTable::new(records).map_err(|error| error.to_string())?;
        let mut file = File::create(path).map_err(|error| error.to_string())?;
        table
            .write(&mut file, self.terminator)
            .map_err(|error| error.to_string())
    }
}

struct ScalarCsvTable {
    rows: Vec<RlMetricRecord<TrainingMetricScalar>>,
    columns: IndexMap<String, bool>,
}
impl ScalarCsvTable {
    fn new<M: RlMetricsCsvValue>(records: &[RlMetricRecord<M>]) -> Result<Self, RlMetricsCsvError> {
        let rows: Vec<RlMetricRecord<_>> = records
            .iter()
            .map(|row| {
                row.iter()
                    .map(|(key, value)| (key.clone(), value.csv_scalar().into_owned()))
                    .collect()
            })
            .collect();
        let mut columns = IndexMap::new();
        for row in &rows {
            for name in row.keys() {
                columns.entry(name.clone()).or_insert(false);
            }
        }
        for (name, floating) in &mut columns {
            *floating = float_column(rows.iter().map(|row| row.get(name)))?;
        }
        Ok(Self { rows, columns })
    }
    fn write(
        &self,
        writer: &mut dyn Write,
        terminator: csv::Terminator,
    ) -> Result<(), RlMetricsCsvError> {
        let mut writer = csv::WriterBuilder::new()
            .terminator(terminator)
            .from_writer(writer);
        writer.write_record(std::iter::once("").chain(self.columns.keys().map(String::as_str)))?;
        // pandas constructs a zero-row dataframe when every record has zero columns.
        if !self.columns.is_empty() {
            for (index, row) in self.rows.iter().enumerate() {
                let mut fields = vec![index.to_string()];
                for (name, floating) in &self.columns {
                    fields.push(csv_cell(row.get(name), *floating)?);
                }
                writer.write_record(fields)?;
            }
        }
        writer.flush()?;
        Ok(())
    }
}

/// Write a complete metric table to an arbitrary writer using the maintained csv crate.
/// # Errors
/// Reports deferred custom formatting, CSV write and final flush errors.
pub fn write_rl_metrics_csv<M: RlMetricsCsvValue>(
    records: &[RlMetricRecord<M>],
    writer: &mut dyn Write,
    terminator: csv::Terminator,
) -> Result<(), RlMetricsCsvError> {
    ScalarCsvTable::new(records)?.write(writer, terminator)
}

fn float_column<'a>(
    values: impl Iterator<Item = Option<&'a TrainingMetricScalar>>,
) -> Result<bool, RlMetricsCsvError> {
    let (mut floating, mut seen_null, mut boolean, mut negative, mut unsigned_large) =
        (false, false, false, false, false);
    for value in values {
        match value {
            None | Some(TrainingMetricScalar::Float(_)) => floating = true,
            Some(TrainingMetricScalar::Null) => {
                floating = true;
                seen_null = true;
            }
            Some(TrainingMetricScalar::Integer(value)) => {
                // pandas attempts Float64 conversion before checking integer ranges. A
                // preceding explicit None suppresses range checks; a missing key/NaN does not.
                value
                    .to_f64()
                    .filter(|value| value.is_finite())
                    .ok_or(RlMetricsCsvError::IntegerOverflow)?;
                if !seen_null {
                    if let Some(value) = value.to_i64() {
                        negative |= value < 0;
                    } else if value.to_u64().is_some() {
                        unsigned_large = true;
                    } else {
                        return Ok(false);
                    }
                    if negative && unsigned_large {
                        return Ok(false);
                    }
                }
            }
            Some(TrainingMetricScalar::Boolean(_)) => boolean = true,
            Some(TrainingMetricScalar::Text(_) | TrainingMetricScalar::Custom(_)) => {
                return Ok(false);
            }
        }
    }
    Ok(floating && !boolean)
}

fn csv_cell(
    value: Option<&TrainingMetricScalar>,
    floating: bool,
) -> Result<String, RlMetricsCsvError> {
    let Some(value) = value else {
        return Ok(String::new());
    };
    match value {
        TrainingMetricScalar::Null => Ok(String::new()),
        TrainingMetricScalar::Float(value) if value.is_nan() => Ok(String::new()),
        TrainingMetricScalar::Integer(value) if floating => {
            let value = value
                .to_f64()
                .expect("inferred floating-column integers were checked as finite-convertible");
            Ok(crate::training_vessel_log::python_float(value))
        }
        _ => crate::training_vessel_log::render_scalar(value)
            .map_err(|error| RlMetricsCsvError::Format(error.to_string())),
    }
}

#[cfg(test)]
#[path = "../tests/support/rl_metrics_writer.rs"]
mod tests;
