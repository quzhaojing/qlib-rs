//! Ordered portfolio metric records, benchmark sampling, Arrow export, and CSV persistence.

use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
    sync::Arc,
};

use arrow_array::{ArrayRef, Float64Array, RecordBatch, TimestampMicrosecondArray};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use chrono::{NaiveDate, NaiveDateTime};
use indexmap::IndexMap;
use thiserror::Error;

const COLUMNS: [&str; 10] = [
    "datetime",
    "account",
    "return",
    "total_turnover",
    "turnover",
    "total_cost",
    "cost",
    "value",
    "cash",
    "bench",
];

/// Failure returned by a replaceable benchmark-return sampler.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("benchmark return sampler error: {message}")]
pub struct BenchmarkReturnSamplerError {
    pub message: String,
}

/// Supplies the benchmark return for one closed trading interval.
pub trait BenchmarkReturnSampler: Send + Sync {
    /// Return a sampled return, or `None` when the interval contains no benchmark observations.
    ///
    /// # Errors
    ///
    /// Returns a local provider or plugin transport failure.
    fn sample_return(
        &self,
        trade_start_time: NaiveDateTime,
        trade_end_time: NaiveDateTime,
    ) -> Result<Option<f64>, BenchmarkReturnSamplerError>;
}

/// In-memory equivalent of a datetime-indexed Pandas benchmark-return series.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BenchmarkReturnSeries {
    values: Vec<(NaiveDateTime, f64)>,
}

impl BenchmarkReturnSeries {
    /// Build a stable timestamp-sorted series. Duplicate timestamps retain their input order.
    #[must_use]
    pub fn new(mut values: Vec<(NaiveDateTime, f64)>) -> Self {
        values.sort_by_key(|(timestamp, _)| *timestamp);
        Self { values }
    }

    #[must_use]
    pub fn values(&self) -> &[(NaiveDateTime, f64)] {
        &self.values
    }
}

impl BenchmarkReturnSampler for BenchmarkReturnSeries {
    fn sample_return(
        &self,
        trade_start_time: NaiveDateTime,
        trade_end_time: NaiveDateTime,
    ) -> Result<Option<f64>, BenchmarkReturnSamplerError> {
        let mut found = false;
        let mut compounded = 1.0;
        for (_, value) in self
            .values
            .iter()
            .filter(|(timestamp, _)| *timestamp >= trade_start_time && *timestamp <= trade_end_time)
        {
            found = true;
            if !value.is_nan() {
                compounded *= value + 1.0;
            }
        }
        Ok(found.then_some(compounded - 1.0))
    }
}

/// One aligned row in Qlib's portfolio-metrics table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PortfolioMetricRecord {
    pub trade_start_time: NaiveDateTime,
    pub account_value: f64,
    pub return_rate: f64,
    pub total_turnover: f64,
    pub turnover_rate: f64,
    pub total_cost: f64,
    pub cost_rate: f64,
    pub stock_value: f64,
    pub cash: f64,
    pub bench_value: Option<f64>,
}

/// Python-compatible optional argument surface for updating one portfolio row.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PortfolioMetricUpdate {
    pub trade_start_time: Option<NaiveDateTime>,
    pub trade_end_time: Option<NaiveDateTime>,
    pub account_value: Option<f64>,
    pub cash: Option<f64>,
    pub return_rate: Option<f64>,
    pub total_turnover: Option<f64>,
    pub turnover_rate: Option<f64>,
    pub total_cost: Option<f64>,
    pub cost_rate: Option<f64>,
    pub stock_value: Option<f64>,
    pub bench_value: Option<f64>,
}

/// Typed failures from portfolio storage, sampling, and CSV persistence.
#[derive(Debug, Error)]
pub enum PortfolioMetricsError {
    #[error(
        "None in [trade_start_time, account_value, cash, return_rate, total_turnover, turnover_rate, total_cost, cost_rate, stock_value]"
    )]
    MissingRequiredValue,
    #[error("Both trade_end_time and bench_value is None, benchmark is not usable.")]
    MissingBenchmarkInputs,
    #[error("portfolio metrics are empty")]
    Empty,
    #[error(transparent)]
    Benchmark(#[from] BenchmarkReturnSamplerError),
    #[error(transparent)]
    Csv(#[from] csv::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("portfolio metrics CSV is missing column {column}")]
    MissingColumn { column: &'static str },
    #[error("invalid portfolio metrics datetime {value}")]
    InvalidDateTime { value: String },
    #[error("invalid portfolio metrics number in column {column}: {value}")]
    InvalidNumber { column: &'static str, value: String },
}

/// Ordered portfolio history with an optional replaceable benchmark sampler.
pub struct PortfolioMetrics {
    frequency: String,
    records: IndexMap<NaiveDateTime, PortfolioMetricRecord>,
    latest_pm_time: Option<NaiveDateTime>,
    benchmark: Option<Arc<dyn BenchmarkReturnSampler>>,
}

impl PortfolioMetrics {
    #[must_use]
    pub fn new(
        frequency: impl Into<String>,
        benchmark: Option<Arc<dyn BenchmarkReturnSampler>>,
    ) -> Self {
        Self {
            frequency: frequency.into(),
            records: IndexMap::new(),
            latest_pm_time: None,
            benchmark,
        }
    }

    #[must_use]
    pub fn without_benchmark(frequency: impl Into<String>) -> Self {
        Self::new(frequency, None)
    }

    #[must_use]
    pub fn frequency(&self) -> &str {
        &self.frequency
    }

    #[must_use]
    pub fn records(&self) -> &IndexMap<NaiveDateTime, PortfolioMetricRecord> {
        &self.records
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    #[must_use]
    pub const fn latest_date(&self) -> Option<NaiveDateTime> {
        self.latest_pm_time
    }

    /// Return the latest record selected by the last update call, not maximum timestamp.
    ///
    /// # Errors
    ///
    /// Returns `Empty` before the first successful update.
    pub fn latest_record(&self) -> Result<&PortfolioMetricRecord, PortfolioMetricsError> {
        self.latest_pm_time
            .and_then(|timestamp| self.records.get(&timestamp))
            .ok_or(PortfolioMetricsError::Empty)
    }

    /// Reset only metric rows, matching Python's `init_vars`.
    pub fn clear(&mut self) {
        self.records.clear();
        self.latest_pm_time = None;
    }

    /// Replace benchmark configuration and optionally update the frequency.
    pub fn init_benchmark(
        &mut self,
        frequency: Option<&str>,
        benchmark: Option<Arc<dyn BenchmarkReturnSampler>>,
    ) {
        if let Some(frequency) = frequency {
            frequency.clone_into(&mut self.frequency);
        }
        self.benchmark = benchmark;
    }

    /// Insert or overwrite one aligned metric row.
    ///
    /// # Errors
    ///
    /// Returns before mutation for missing required values, unusable benchmark inputs, or a
    /// benchmark plugin failure.
    pub fn update_record(
        &mut self,
        update: PortfolioMetricUpdate,
    ) -> Result<(), PortfolioMetricsError> {
        let (
            Some(trade_start_time),
            Some(account_value),
            Some(cash),
            Some(return_rate),
            Some(total_turnover),
            Some(turnover_rate),
            Some(total_cost),
            Some(cost_rate),
            Some(stock_value),
        ) = (
            update.trade_start_time,
            update.account_value,
            update.cash,
            update.return_rate,
            update.total_turnover,
            update.turnover_rate,
            update.total_cost,
            update.cost_rate,
            update.stock_value,
        )
        else {
            return Err(PortfolioMetricsError::MissingRequiredValue);
        };

        let bench_value = match (update.bench_value, update.trade_end_time) {
            (Some(value), _) => Some(value),
            (None, None) => return Err(PortfolioMetricsError::MissingBenchmarkInputs),
            (None, Some(trade_end_time)) => match &self.benchmark {
                Some(benchmark) => benchmark.sample_return(trade_start_time, trade_end_time)?,
                None => None,
            },
        };
        let record = PortfolioMetricRecord {
            trade_start_time,
            account_value,
            return_rate,
            total_turnover,
            turnover_rate,
            total_cost,
            cost_rate,
            stock_value,
            cash,
            bench_value,
        };
        self.records.insert(trade_start_time, record);
        self.latest_pm_time = Some(trade_start_time);
        Ok(())
    }

    /// Export the Pandas-compatible column order as one Arrow record batch.
    ///
    /// # Panics
    ///
    /// Panics only if the module's invariant that every column is built from the same ordered
    /// record collection is violated.
    #[must_use]
    pub fn to_record_batch(&self) -> RecordBatch {
        let timestamps = self
            .records
            .keys()
            .map(|timestamp| timestamp.and_utc().timestamp_micros())
            .collect::<Vec<_>>();
        let fields = vec![
            Field::new(
                "datetime",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new("account", DataType::Float64, false),
            Field::new("return", DataType::Float64, false),
            Field::new("total_turnover", DataType::Float64, false),
            Field::new("turnover", DataType::Float64, false),
            Field::new("total_cost", DataType::Float64, false),
            Field::new("cost", DataType::Float64, false),
            Field::new("value", DataType::Float64, false),
            Field::new("cash", DataType::Float64, false),
            Field::new("bench", DataType::Float64, true),
        ];
        let values = self.records.values().collect::<Vec<_>>();
        let float_column = |select: fn(&PortfolioMetricRecord) -> f64| -> ArrayRef {
            Arc::new(Float64Array::from_iter_values(
                values.iter().map(|record| select(record)),
            ))
        };
        let columns: Vec<ArrayRef> = vec![
            Arc::new(TimestampMicrosecondArray::from(timestamps)),
            float_column(|record| record.account_value),
            float_column(|record| record.return_rate),
            float_column(|record| record.total_turnover),
            float_column(|record| record.turnover_rate),
            float_column(|record| record.total_cost),
            float_column(|record| record.cost_rate),
            float_column(|record| record.stock_value),
            float_column(|record| record.cash),
            Arc::new(Float64Array::from_iter(
                values.iter().map(|record| record.bench_value),
            )),
        ];
        RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
            .expect("portfolio metric columns always have one value per ordered record")
    }

    /// Save the table using the maintained `csv` crate.
    ///
    /// # Errors
    ///
    /// Returns file creation, serialization, or flush failures.
    pub fn save_csv(&self, path: &Path) -> Result<(), PortfolioMetricsError> {
        let mut file = File::create(path)?;
        self.write_csv(&mut file)
    }

    /// Write the table to an arbitrary file, memory, network, or plugin writer.
    ///
    /// # Errors
    ///
    /// Returns serialization or writer failures.
    pub fn write_csv(&self, writer: &mut dyn Write) -> Result<(), PortfolioMetricsError> {
        let mut writer = csv::WriterBuilder::new()
            .buffer_capacity(1)
            .from_writer(writer);
        writer.write_record(COLUMNS)?;
        for record in self.records.values() {
            writer.write_record([
                record.trade_start_time.to_string(),
                record.account_value.to_string(),
                record.return_rate.to_string(),
                record.total_turnover.to_string(),
                record.turnover_rate.to_string(),
                record.total_cost.to_string(),
                record.cost_rate.to_string(),
                record.stock_value.to_string(),
                record.cash.to_string(),
                record
                    .bench_value
                    .map_or_else(String::new, |value| value.to_string()),
            ])?;
        }
        writer.flush()?;
        Ok(())
    }

    /// Load a CSV table, replacing existing rows after its header is validated.
    ///
    /// # Errors
    ///
    /// Returns file, CSV, required-column, timestamp, or numeric parsing failures. Rows parsed
    /// before a later row failure remain loaded, matching Python's iterative reconstruction.
    pub fn load_csv(&mut self, path: &Path) -> Result<(), PortfolioMetricsError> {
        let mut file = File::open(path)?;
        self.read_csv(&mut file)
    }

    /// Load a table from an arbitrary file, memory, network, or plugin reader.
    ///
    /// # Errors
    ///
    /// Returns CSV, required-column, timestamp, or numeric parsing failures. Rows parsed before a
    /// later row failure remain loaded.
    pub fn read_csv(&mut self, reader: &mut dyn Read) -> Result<(), PortfolioMetricsError> {
        let mut reader = csv::ReaderBuilder::new().flexible(true).from_reader(reader);
        let headers = reader.headers()?.clone();
        let mut indices = [0_usize; COLUMNS.len()];
        for (index, column) in indices.iter_mut().zip(COLUMNS) {
            *index = headers
                .iter()
                .position(|candidate| candidate == column)
                .ok_or(PortfolioMetricsError::MissingColumn { column })?;
        }
        let [
            datetime,
            account,
            return_rate,
            total_turnover,
            turnover,
            total_cost,
            cost,
            value,
            cash,
            bench,
        ] = indices;

        self.clear();
        for row in reader.records() {
            let row = row?;
            let read = |index: usize, column| {
                row.get(index)
                    .ok_or(PortfolioMetricsError::MissingColumn { column })
            };
            let timestamp_text = read(datetime, "datetime")?;
            let timestamp = parse_datetime(timestamp_text).ok_or_else(|| {
                PortfolioMetricsError::InvalidDateTime {
                    value: timestamp_text.to_owned(),
                }
            })?;
            let numeric_columns = [
                (account, "account"),
                (return_rate, "return"),
                (total_turnover, "total_turnover"),
                (turnover, "turnover"),
                (total_cost, "total_cost"),
                (cost, "cost"),
                (value, "value"),
                (cash, "cash"),
            ];
            let mut numbers = [0.0; 8];
            for (number, (index, column)) in numbers.iter_mut().zip(numeric_columns) {
                let text = read(index, column)?;
                *number = text
                    .parse()
                    .map_err(|_| PortfolioMetricsError::InvalidNumber {
                        column,
                        value: text.to_owned(),
                    })?;
            }
            let [
                account,
                return_rate,
                total_turnover,
                turnover,
                total_cost,
                cost,
                value,
                cash,
            ] = numbers;
            let bench_text = read(bench, "bench")?;
            let bench_value = if bench_text.is_empty() {
                f64::NAN
            } else {
                bench_text
                    .parse()
                    .map_err(|_| PortfolioMetricsError::InvalidNumber {
                        column: "bench",
                        value: bench_text.to_owned(),
                    })?
            };
            let record = PortfolioMetricRecord {
                trade_start_time: timestamp,
                account_value: account,
                return_rate,
                total_turnover,
                turnover_rate: turnover,
                total_cost,
                cost_rate: cost,
                stock_value: value,
                cash,
                bench_value: Some(bench_value),
            };
            self.records.insert(timestamp, record);
            self.latest_pm_time = Some(timestamp);
        }
        Ok(())
    }
}

fn parse_datetime(value: &str) -> Option<NaiveDateTime> {
    NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S%.f")
        .ok()
        .or_else(|| {
            NaiveDate::parse_from_str(value, "%Y-%m-%d")
                .ok()
                .and_then(|date| date.and_hms_opt(0, 0, 0))
        })
}
