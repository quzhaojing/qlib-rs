use std::{
    fs,
    io::{self, Write},
    path::PathBuf,
    process::Command,
    sync::{Arc, Mutex},
};

use arrow_array::{Array, Float64Array, TimestampMicrosecondArray};
use arrow_schema::{DataType, TimeUnit};
use chrono::NaiveDateTime;
use domain_core::{
    BenchmarkReturnSampler, BenchmarkReturnSamplerError, BenchmarkReturnSeries,
    PortfolioMetricUpdate, PortfolioMetrics, PortfolioMetricsError,
};
use serde_json::{Value, json};

fn datetime(text: &str) -> NaiveDateTime {
    NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S%.f").unwrap()
}

fn update(timestamp: NaiveDateTime) -> PortfolioMetricUpdate {
    PortfolioMetricUpdate {
        trade_start_time: Some(timestamp),
        trade_end_time: None,
        account_value: Some(100.0),
        cash: Some(40.0),
        return_rate: Some(0.1),
        total_turnover: Some(20.0),
        turnover_rate: Some(0.2),
        total_cost: Some(2.0),
        cost_rate: Some(0.02),
        stock_value: Some(60.0),
        bench_value: Some(0.03),
    }
}

fn assert_same(actual: f64, expected: f64) {
    assert_eq!(actual.to_bits(), expected.to_bits());
}

#[test]
fn benchmark_series_sorts_samples_closed_ranges_and_skips_nan() {
    let first = datetime("2024-01-02 09:30:00");
    let duplicate = datetime("2024-01-02 09:31:00");
    let last = datetime("2024-01-02 09:32:00");
    let series = BenchmarkReturnSeries::new(vec![
        (last, -0.5),
        (duplicate, f64::NAN),
        (first, 0.1),
        (duplicate, 0.2),
    ]);
    assert_eq!(series.values()[0].0, first);
    assert_eq!(series.values()[1].0, duplicate);
    assert!(series.values()[1].1.is_nan());
    assert_eq!(series.values()[2], (duplicate, 0.2));
    assert_same(
        series.sample_return(first, duplicate).unwrap().unwrap(),
        (1.0 + 0.1) * (1.0 + 0.2) - 1.0,
    );
    assert_same(
        series.sample_return(duplicate, duplicate).unwrap().unwrap(),
        1.0 + 0.2 - 1.0,
    );
    assert_eq!(
        series.sample_return(last, first).unwrap(),
        None,
        "an empty selected interval maps to Python's None"
    );

    let only_nan = BenchmarkReturnSeries::new(vec![(first, f64::NAN)]);
    assert_same(only_nan.sample_return(first, first).unwrap().unwrap(), 0.0);
    assert!(BenchmarkReturnSeries::default().values().is_empty());
}

#[derive(Clone)]
struct ProbeSampler {
    events: Arc<Mutex<Vec<(NaiveDateTime, NaiveDateTime)>>>,
    result: Result<Option<f64>, BenchmarkReturnSamplerError>,
}

impl BenchmarkReturnSampler for ProbeSampler {
    fn sample_return(
        &self,
        trade_start_time: NaiveDateTime,
        trade_end_time: NaiveDateTime,
    ) -> Result<Option<f64>, BenchmarkReturnSamplerError> {
        self.events
            .lock()
            .unwrap()
            .push((trade_start_time, trade_end_time));
        self.result.clone()
    }
}

#[test]
fn record_updates_validate_sample_overwrite_and_track_last_call() {
    let first = datetime("2024-01-02 09:30:00");
    let second = datetime("2024-01-02 09:31:00");
    let end = datetime("2024-01-02 09:32:00");
    let events = Arc::new(Mutex::new(Vec::new()));
    let sampler = ProbeSampler {
        events: Arc::clone(&events),
        result: Ok(Some(0.25)),
    };
    let mut metrics = PortfolioMetrics::new("1min", Some(Arc::new(sampler)));
    assert_eq!(metrics.frequency(), "1min");
    assert!(metrics.is_empty());
    assert_eq!(metrics.latest_date(), None);
    assert!(matches!(
        metrics.latest_record(),
        Err(PortfolioMetricsError::Empty)
    ));

    let mut sampled_update = update(second);
    sampled_update.trade_end_time = Some(end);
    sampled_update.bench_value = None;
    metrics.update_record(sampled_update).unwrap();
    assert_eq!(events.lock().unwrap().as_slice(), [(second, end)]);
    assert_same(metrics.latest_record().unwrap().bench_value.unwrap(), 0.25);

    let mut earlier = update(first);
    earlier.account_value = Some(80.0);
    metrics.update_record(earlier).unwrap();
    assert_eq!(
        metrics.records().keys().copied().collect::<Vec<_>>(),
        [second, first]
    );
    assert_eq!(metrics.latest_date(), Some(first));
    assert_same(metrics.latest_record().unwrap().account_value, 80.0);

    let mut overwrite = update(second);
    overwrite.account_value = Some(120.0);
    overwrite.bench_value = Some(f64::NAN);
    metrics.update_record(overwrite).unwrap();
    assert_eq!(
        metrics.records().keys().copied().collect::<Vec<_>>(),
        [second, first]
    );
    assert_eq!(metrics.latest_date(), Some(second));
    assert_same(metrics.latest_record().unwrap().account_value, 120.0);
    assert!(
        metrics
            .latest_record()
            .unwrap()
            .bench_value
            .unwrap()
            .is_nan()
    );

    metrics.clear();
    assert!(metrics.is_empty());
    metrics.init_benchmark(None, None);
    let mut no_benchmark = update(first);
    no_benchmark.trade_end_time = Some(end);
    no_benchmark.bench_value = None;
    metrics.update_record(no_benchmark).unwrap();
    assert_eq!(metrics.latest_record().unwrap().bench_value, None);
    metrics.init_benchmark(Some("week"), None);
    assert_eq!(metrics.frequency(), "week");
}

#[test]
fn every_required_value_and_benchmark_failure_is_pre_mutation() {
    let timestamp = datetime("2024-01-02 09:30:00");
    let mut missing_cases = Vec::new();
    let mut case = update(timestamp);
    case.trade_start_time = None;
    missing_cases.push(case);
    let mut case = update(timestamp);
    case.account_value = None;
    missing_cases.push(case);
    let mut case = update(timestamp);
    case.cash = None;
    missing_cases.push(case);
    let mut case = update(timestamp);
    case.return_rate = None;
    missing_cases.push(case);
    let mut case = update(timestamp);
    case.total_turnover = None;
    missing_cases.push(case);
    let mut case = update(timestamp);
    case.turnover_rate = None;
    missing_cases.push(case);
    let mut case = update(timestamp);
    case.total_cost = None;
    missing_cases.push(case);
    let mut case = update(timestamp);
    case.cost_rate = None;
    missing_cases.push(case);
    let mut case = update(timestamp);
    case.stock_value = None;
    missing_cases.push(case);

    for case in missing_cases {
        let mut metrics = PortfolioMetrics::without_benchmark("day");
        assert!(matches!(
            metrics.update_record(case),
            Err(PortfolioMetricsError::MissingRequiredValue)
        ));
        assert!(metrics.is_empty());
    }

    let mut metrics = PortfolioMetrics::without_benchmark("day");
    let mut missing_benchmark = update(timestamp);
    missing_benchmark.bench_value = None;
    assert!(matches!(
        metrics.update_record(missing_benchmark),
        Err(PortfolioMetricsError::MissingBenchmarkInputs)
    ));
    assert!(metrics.is_empty());

    let sampler = ProbeSampler {
        events: Arc::new(Mutex::new(Vec::new())),
        result: Err(BenchmarkReturnSamplerError {
            message: "offline".to_owned(),
        }),
    };
    let mut metrics = PortfolioMetrics::new("day", Some(Arc::new(sampler)));
    let mut failed = update(timestamp);
    failed.trade_end_time = Some(datetime("2024-01-02 16:00:00"));
    failed.bench_value = None;
    assert!(matches!(
        metrics.update_record(failed),
        Err(PortfolioMetricsError::Benchmark(_))
    ));
    assert!(metrics.is_empty());
}

#[test]
fn arrow_export_preserves_schema_order_values_nulls_and_special_values() {
    let mut metrics = PortfolioMetrics::without_benchmark("day");
    let empty = metrics.to_record_batch();
    assert_eq!(empty.num_rows(), 0);
    assert_eq!(empty.num_columns(), 10);

    let timestamp = datetime("2024-01-02 09:30:00.123456");
    let mut row = update(timestamp);
    row.account_value = Some(f64::INFINITY);
    row.return_rate = Some(f64::NEG_INFINITY);
    row.bench_value = None;
    row.trade_end_time = Some(timestamp);
    metrics.update_record(row).unwrap();
    let batch = metrics.to_record_batch();
    assert_eq!(batch.schema().field(0).name(), "datetime");
    assert_eq!(
        batch.schema().field(0).data_type(),
        &DataType::Timestamp(TimeUnit::Microsecond, None)
    );
    assert_eq!(
        batch
            .schema()
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .collect::<Vec<_>>(),
        [
            "datetime",
            "account",
            "return",
            "total_turnover",
            "turnover",
            "total_cost",
            "cost",
            "value",
            "cash",
            "bench"
        ]
    );
    let timestamps = batch
        .column(0)
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .unwrap();
    assert_eq!(timestamps.value(0), timestamp.and_utc().timestamp_micros());
    let accounts = batch
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_same(accounts.value(0), f64::INFINITY);
    let returns = batch
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_same(returns.value(0), f64::NEG_INFINITY);
    let benches = batch
        .column(9)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert!(benches.is_null(0));
}

fn temp_csv(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "qlib-rs-portfolio-{name}-{}.csv",
        std::process::id()
    ))
}

#[test]
fn csv_save_load_supports_schema_reordering_dates_nulls_and_partial_failure() {
    let first = datetime("2024-01-02 09:30:00");
    let second = datetime("2024-01-03 00:00:00");
    let path = temp_csv("roundtrip");
    let mut source = PortfolioMetrics::without_benchmark("day");
    source.update_record(update(first)).unwrap();
    let mut null_bench = update(second);
    null_bench.bench_value = None;
    null_bench.trade_end_time = Some(second);
    source.update_record(null_bench).unwrap();
    source.save_csv(&path).unwrap();

    let mut loaded = PortfolioMetrics::without_benchmark("week");
    loaded.update_record(update(first)).unwrap();
    loaded.load_csv(&path).unwrap();
    assert_eq!(loaded.records().len(), 2);
    assert_eq!(loaded.latest_date(), Some(second));
    assert!(
        loaded
            .latest_record()
            .unwrap()
            .bench_value
            .unwrap()
            .is_nan()
    );
    fs::remove_file(&path).unwrap();

    let reordered = temp_csv("reordered");
    fs::write(
        &reordered,
        "extra,cash,datetime,value,cost,total_cost,turnover,total_turnover,return,account,bench\nignored,4,2024-01-04,6,.01,1,.02,2,.03,10,.04\n",
    )
    .unwrap();
    loaded.load_csv(&reordered).unwrap();
    assert_eq!(loaded.latest_date(), Some(datetime("2024-01-04 00:00:00")));
    assert_same(loaded.latest_record().unwrap().cash, 4.0);
    fs::remove_file(&reordered).unwrap();

    let partial = temp_csv("partial");
    fs::write(
        &partial,
        "datetime,account,return,total_turnover,turnover,total_cost,cost,value,cash,bench\n2024-01-05,10,.1,2,.2,1,.01,6,4,.03\nbad,10,.1,2,.2,1,.01,6,4,.03\n",
    )
    .unwrap();
    assert!(matches!(
        loaded.load_csv(&partial),
        Err(PortfolioMetricsError::InvalidDateTime { .. })
    ));
    assert_eq!(loaded.records().len(), 1);
    fs::remove_file(&partial).unwrap();
}

#[test]
fn csv_reports_file_header_row_and_numeric_failures() {
    let mut metrics = PortfolioMetrics::without_benchmark("day");
    assert!(matches!(
        metrics.load_csv(&temp_csv("absent")),
        Err(PortfolioMetricsError::Io(_))
    ));
    assert!(matches!(
        metrics.save_csv(&std::env::temp_dir()),
        Err(PortfolioMetricsError::Io(_))
    ));

    let missing = temp_csv("missing-column");
    fs::write(&missing, "datetime,account\n2024-01-02,1\n").unwrap();
    assert!(matches!(
        metrics.load_csv(&missing),
        Err(PortfolioMetricsError::MissingColumn { .. })
    ));
    fs::remove_file(&missing).unwrap();

    let invalid = temp_csv("invalid-number");
    fs::write(
        &invalid,
        "datetime,account,return,total_turnover,turnover,total_cost,cost,value,cash,bench\n2024-01-02,nope,.1,2,.2,1,.01,6,4,.03\n",
    )
    .unwrap();
    assert!(matches!(
        metrics.load_csv(&invalid),
        Err(PortfolioMetricsError::InvalidNumber {
            column: "account",
            ..
        })
    ));
    fs::remove_file(&invalid).unwrap();

    let invalid_bench = temp_csv("invalid-bench");
    fs::write(
        &invalid_bench,
        "datetime,account,return,total_turnover,turnover,total_cost,cost,value,cash,bench\n2024-01-02,10,.1,2,.2,1,.01,6,4,nope\n",
    )
    .unwrap();
    assert!(matches!(
        metrics.load_csv(&invalid_bench),
        Err(PortfolioMetricsError::InvalidNumber {
            column: "bench",
            ..
        })
    ));
    fs::remove_file(&invalid_bench).unwrap();
}

struct LimitedWriter {
    remaining: usize,
    fail_flush: bool,
}

impl Write for LimitedWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        if self.remaining == 0 {
            return Err(io::Error::other("write failed"));
        }
        let written = buffer.len().min(self.remaining);
        self.remaining -= written;
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.fail_flush {
            Err(io::Error::other("flush failed"))
        } else {
            Ok(())
        }
    }
}

#[test]
fn generic_csv_boundaries_report_header_row_flush_and_short_input_failures() {
    let mut metrics = PortfolioMetrics::without_benchmark("day");
    metrics
        .update_record(update(datetime("2024-01-02 00:00:00")))
        .unwrap();
    let mut writer = LimitedWriter {
        remaining: 0,
        fail_flush: false,
    };
    assert!(matches!(
        metrics.write_csv(&mut writer),
        Err(PortfolioMetricsError::Csv(_))
    ));
    let header_length =
        "datetime,account,return,total_turnover,turnover,total_cost,cost,value,cash,bench\n".len();
    let mut writer = LimitedWriter {
        remaining: header_length,
        fail_flush: false,
    };
    assert!(matches!(
        metrics.write_csv(&mut writer),
        Err(PortfolioMetricsError::Csv(_))
    ));
    let mut writer = LimitedWriter {
        remaining: usize::MAX,
        fail_flush: true,
    };
    assert!(matches!(
        metrics.write_csv(&mut writer),
        Err(PortfolioMetricsError::Io(_))
    ));

    let mut invalid_header = &[0xff][..];
    assert!(matches!(
        metrics.read_csv(&mut invalid_header),
        Err(PortfolioMetricsError::Csv(_))
    ));
    let header =
        "datetime,account,return,total_turnover,turnover,total_cost,cost,value,cash,bench\n";
    let mut invalid_row = header.as_bytes().to_vec();
    invalid_row.push(0xff);
    let mut invalid_row = invalid_row.as_slice();
    assert!(matches!(
        metrics.read_csv(&mut invalid_row),
        Err(PortfolioMetricsError::Csv(_))
    ));

    let datetime_last =
        "account,return,total_turnover,turnover,total_cost,cost,value,cash,bench,datetime\n1\n";
    let mut datetime_last = datetime_last.as_bytes();
    assert!(matches!(
        metrics.read_csv(&mut datetime_last),
        Err(PortfolioMetricsError::MissingColumn { column: "datetime" })
    ));
    let missing_numeric = format!("{header}2024-01-02\n");
    let mut missing_numeric = missing_numeric.as_bytes();
    assert!(matches!(
        metrics.read_csv(&mut missing_numeric),
        Err(PortfolioMetricsError::MissingColumn { column: "account" })
    ));
    let missing_bench = format!("{header}2024-01-02,1,.1,2,.2,1,.01,6,4\n");
    let mut missing_bench = missing_bench.as_bytes();
    assert!(matches!(
        metrics.read_csv(&mut missing_bench),
        Err(PortfolioMetricsError::MissingColumn { column: "bench" })
    ));
}

fn special(value: Option<f64>) -> Value {
    match value {
        None => Value::Null,
        Some(value) if value.is_nan() => json!("NaN"),
        Some(value) if value == f64::INFINITY => json!("Infinity"),
        Some(value) if value == f64::NEG_INFINITY => json!("-Infinity"),
        Some(value) => json!(value),
    }
}

fn live_python_snapshot() -> Value {
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/backtest/report.py");
    let script = r"
import ast,json,math,sys,pandas as pd
p=sys.argv[1];t=ast.parse(open(p,encoding='utf-8').read());c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='PortfolioMetrics');names={'init_vars','is_empty','get_latest_date','get_latest_account_value','get_latest_total_cost','get_latest_total_turnover','update_portfolio_metrics_record','generate_portfolio_metrics_dataframe'};body=[]
for n in c.body:
 if isinstance(n,ast.FunctionDef) and n.name in names:
  n.returns=None
  for a in n.args.args:a.annotation=None
  body.append(n)
K=ast.fix_missing_locations(ast.ClassDef('PortfolioMetrics',[],[],body,[]));ns={'OrderedDict':__import__('collections').OrderedDict,'pd':pd};exec(compile(ast.Module([K],[]),p,'exec'),ns);P=ns['PortfolioMetrics']
def obj():x=P();x.init_vars();x.bench=None;x._sample_benchmark=lambda b,s,e:None;return x
def val(x):
 if x is None:return None
 if isinstance(x,float) and math.isnan(x):return 'NaN'
 if x==float('inf'):return 'Infinity'
 if x==float('-inf'):return '-Infinity'
 return x
def add(x,t,a=100.,b=.03,end=None):x.update_portfolio_metrics_record(t,end,a,40.,.1,20.,.2,2.,.02,60.,b)
x=obj();empty=[x.is_empty(),x.get_latest_date()];errors=[]
try:x.update_portfolio_metrics_record()
except Exception as e:errors.append(type(e).__name__+':'+str(e))
try:add(x,pd.Timestamp('2024-01-02'),b=None)
except Exception as e:errors.append(type(e).__name__+':'+str(e))
add(x,pd.Timestamp('2024-01-03'),120.,float('nan'));add(x,pd.Timestamp('2024-01-02'),80.,float('inf'));add(x,pd.Timestamp('2024-01-03'),130.,float('-inf'));d=x.generate_portfolio_metrics_dataframe();rows=[[str(i),*[val(v) for v in row]] for i,row in d.iterrows()];print(json.dumps([empty,errors,str(x.get_latest_date()),val(x.get_latest_account_value()),val(x.get_latest_total_cost()),val(x.get_latest_total_turnover()),list(d.columns),rows],separators=(',',':')))
";
    let output = Command::new("python")
        .arg("-c")
        .arg(script)
        .arg(source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

fn rust_snapshot() -> Value {
    let first = datetime("2024-01-02 00:00:00");
    let second = datetime("2024-01-03 00:00:00");
    let mut metrics = PortfolioMetrics::without_benchmark("day");
    let empty = json!([metrics.is_empty(), Value::Null]);
    let mut errors = Vec::new();
    let missing = metrics
        .update_record(PortfolioMetricUpdate::default())
        .unwrap_err();
    errors.push(format!("ValueError:{missing}"));
    let mut missing_benchmark = update(first);
    missing_benchmark.bench_value = None;
    let missing = metrics.update_record(missing_benchmark).unwrap_err();
    errors.push(format!("ValueError:{missing}"));
    let mut row = update(second);
    row.account_value = Some(120.0);
    row.bench_value = Some(f64::NAN);
    metrics.update_record(row).unwrap();
    let mut row = update(first);
    row.account_value = Some(80.0);
    row.bench_value = Some(f64::INFINITY);
    metrics.update_record(row).unwrap();
    let mut row = update(second);
    row.account_value = Some(130.0);
    row.bench_value = Some(f64::NEG_INFINITY);
    metrics.update_record(row).unwrap();
    let columns = [
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
    let rows = metrics
        .records()
        .values()
        .map(|row| {
            json!([
                row.trade_start_time.to_string(),
                special(Some(row.account_value)),
                special(Some(row.return_rate)),
                special(Some(row.total_turnover)),
                special(Some(row.turnover_rate)),
                special(Some(row.total_cost)),
                special(Some(row.cost_rate)),
                special(Some(row.stock_value)),
                special(Some(row.cash)),
                special(row.bench_value),
            ])
        })
        .collect::<Vec<_>>();
    let latest = metrics.latest_record().unwrap();
    json!([
        empty,
        errors,
        metrics.latest_date().unwrap().to_string(),
        special(Some(latest.account_value)),
        special(Some(latest.total_cost)),
        special(Some(latest.total_turnover)),
        columns,
        rows,
    ])
}

#[test]
fn portfolio_metric_records_match_live_python_source() {
    assert_eq!(rust_snapshot(), live_python_snapshot());
}
