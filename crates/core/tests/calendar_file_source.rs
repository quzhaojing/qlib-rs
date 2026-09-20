#![cfg(windows)]

use arrow_array::{ArrayRef, RecordBatch, StringArray, TimestampNanosecondArray};
use domain_core::{
    ArrowQuote, CachedCalendarProvider, CalendarBackendSource, CalendarCache, CalendarFileSource,
    CalendarLoadError, CalendarProviderInput, CalendarRuntimeConfiguration, CalendarRuntimeValues,
    CalendarStorageFactory, CalendarTextEncoding, CalendarWarningSink, DataPathManager,
    DealPriceFields, ExchangeQuoteProvider, ExecutionCalendar, ExecutionCommonBindings,
    ExecutionExchange, ExecutionLevelBindings, IsoCalendarTimestampDecoder, LocalCalendarLoader,
    NestedCalendar, Order, OrderDir, ResettableNestedCalendar, SaoeCalendar, SaoeOrderFactory,
    SaoePluginError, SharedExecutionCalendar, SingleOrderStrategy,
    path_initialization::ConfigPathValue, windows_config_paths::WindowsConfigPathOperations,
};
use indexmap::IndexMap;
use num_bigint::BigInt;
use serde_json::{Value, json};
use std::{
    path::Path,
    process::Command,
    sync::{
        Arc, Mutex, RwLock,
        atomic::{AtomicUsize, Ordering},
    },
};

struct Runtime {
    values: RwLock<CalendarRuntimeValues>,
    reads: AtomicUsize,
}
impl CalendarRuntimeConfiguration for Runtime {
    fn region(&self) -> Result<Option<String>, CalendarLoadError> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        self.values.region()
    }
    fn minute_shift(&self) -> Result<BigInt, CalendarLoadError> {
        self.values.minute_shift()
    }
}

struct Warnings {
    runtime: Arc<Runtime>,
    change_region: bool,
    count: AtomicUsize,
}
impl CalendarWarningSink for Warnings {
    fn warning(&self, _message: &str) -> Result<(), CalendarLoadError> {
        self.count.fetch_add(1, Ordering::SeqCst);
        if self.change_region {
            self.runtime.values.write().unwrap().region = Some(Some("us".into()));
        }
        Ok(())
    }
}

fn source(root: &Path, mode: &str) -> (Arc<CalendarFileSource>, Arc<Runtime>, Arc<Warnings>) {
    let runtime = Arc::new(Runtime {
        values: RwLock::new(CalendarRuntimeValues {
            region: if mode == "missing_region" {
                None
            } else {
                Some(Some("cn".into()))
            },
            minute_shift: Some(0.into()),
        }),
        reads: AtomicUsize::new(0),
    });
    let warnings = Arc::new(Warnings {
        runtime: runtime.clone(),
        change_region: mode == "fallback",
        count: AtomicUsize::new(0),
    });
    let provider_template = match mode {
        "mapping" => CalendarProviderInput::Mapping(Arc::new(RwLock::new(IndexMap::from([(
            "day".into(),
            ConfigPathValue::Text(root.as_os_str().to_owned()),
        )])))),
        "partial_override" => {
            CalendarProviderInput::Mapping(Arc::new(RwLock::new(IndexMap::from([
                ("day".into(), ConfigPathValue::Text(".".into())),
                ("bad".into(), ConfigPathValue::Null),
            ]))))
        }
        "invalid_override" => {
            CalendarProviderInput::Scalar(ConfigPathValue::Unsupported("int".into()))
        }
        _ => CalendarProviderInput::Scalar(ConfigPathValue::Null),
    };
    let frequency = if mode == "fallback" { "1min" } else { "day" };
    let source = Arc::new(CalendarFileSource {
        provider_template,
        factory: Arc::new(CalendarStorageFactory {
            global_paths: Arc::new(RwLock::new(DataPathManager::from_native(
                IndexMap::from([(frequency.into(), root.as_os_str().to_owned())]),
                IndexMap::new(),
            ))),
            runtime: runtime.clone(),
            operations: Arc::new(WindowsConfigPathOperations),
            system: "Windows".into(),
            text_decoder: Arc::new(CalendarTextEncoding::Utf8),
            timestamp_decoder: Arc::new(IsoCalendarTimestampDecoder),
            cache: Arc::new(CalendarCache::new(None)),
        }),
    });
    (source, runtime, warnings)
}

#[test]
fn fresh_construction_and_fallback_match_actual_provider_storage_and_loader() {
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/calendar_file_source_contract.py"
        ))
        .arg("D:/code/github/qlib/qlib")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(cases.len(), 10);
    for case in cases {
        check_case(&case);
    }
}

fn check_case(case: &Value) {
    let mode = case["mode"].as_str().unwrap();
    let temp = tempfile::tempdir().unwrap();
    std::fs::create_dir(temp.path().join("calendars")).unwrap();
    let raw = if mode == "fallback" { "1min" } else { "day" };
    if mode != "missing_current" {
        std::fs::write(
            temp.path().join(format!("calendars/{raw}.txt")),
            case["contents"].as_str().unwrap(),
        )
        .unwrap();
    }
    if mode == "invalid_row" {
        std::fs::write(
            temp.path().join("calendars/day_future.txt"),
            case["contents"].as_str().unwrap(),
        )
        .unwrap();
    }
    let (source, runtime, warnings) = source(temp.path(), mode);
    let original = match &source.provider_template {
        CalendarProviderInput::Mapping(values) => Some(values.read().unwrap().clone()),
        CalendarProviderInput::Scalar(_) => None,
    };
    let loader = LocalCalendarLoader::new(
        source.clone(),
        Arc::new(IsoCalendarTimestampDecoder),
        warnings.clone(),
    );
    for expected in case["outputs"].as_array().unwrap() {
        let actual = match loader.load(
            case["frequency"].as_str().unwrap(),
            case["future"].as_bool().unwrap(),
        ) {
            Ok(values) => {
                json!({"values": values.iter().map(|x| x.format("%Y-%m-%dT%H:%M:%S").to_string()).collect::<Vec<_>>()})
            }
            Err(CalendarLoadError::Value(_)) => json!({"error":"Value"}),
            Err(CalendarLoadError::Other(_)) => json!({"error":"Other"}),
        };
        assert_eq!(&actual, expected, "{case}");
    }
    assert_eq!(
        json!(runtime.reads.load(Ordering::SeqCst)),
        case["region_reads"],
        "{case}"
    );
    assert_eq!(
        json!(warnings.count.load(Ordering::SeqCst)),
        case["warnings"],
        "{case}"
    );
    if let CalendarProviderInput::Mapping(values) = &source.provider_template {
        assert_eq!(&*values.read().unwrap(), original.as_ref().unwrap());
    }
}

#[test]
fn template_poison_precedes_construction_and_never_reaches_global_configuration() {
    let (source, runtime, _) = source(Path::new("."), "mapping");
    let CalendarProviderInput::Mapping(values) = &source.provider_template else {
        panic!("mapping expected")
    };
    let values = values.clone();
    assert!(
        std::thread::spawn(move || {
            let _guard = values.write().unwrap();
            panic!("intentional poison");
        })
        .join()
        .is_err()
    );
    let error = source.data("bad-frequency", true).err().unwrap();
    assert!(matches!(error, CalendarLoadError::Other(_)));
    assert!(error.to_string().contains("template lock poisoned"));
    assert_eq!(runtime.reads.load(Ordering::SeqCst), 0);
}

// The order factory remains a test collaborator; calendar infrastructure is native.
struct Orders;
impl SaoeOrderFactory for Orders {
    fn create(
        &mut self,
        stock_id: &str,
        amount: Option<f64>,
        direction: OrderDir,
    ) -> Result<Order, SaoePluginError> {
        Ok(Order::new(stock_id, amount.unwrap(), direction, None, None))
    }
}

#[test]
fn file_provider_execution_windows_and_strategy_share_the_real_calendar_chain() {
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/calendar_execution_chain_contract.py"
        ))
        .arg("D:/code/github/qlib/qlib")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(cases.len(), 2);
    for case in cases {
        check_execution_chain(&case);
    }
}

fn check_execution_chain(case: &Value) {
    let temp = tempfile::tempdir().unwrap();
    let directory = temp.path().join("calendars");
    std::fs::create_dir(&directory).unwrap();
    std::fs::write(
        directory.join("1min.txt"),
        case["contents"].as_str().unwrap(),
    )
    .unwrap();
    if case["future"].as_bool().unwrap() {
        std::fs::write(
            directory.join("1min_future.txt"),
            case["contents"].as_str().unwrap(),
        )
        .unwrap();
    }
    let (source, runtime, _) = source(temp.path(), "fallback");
    let warnings = Arc::new(Warnings {
        runtime,
        change_region: false,
        count: AtomicUsize::new(0),
    });
    let cache = source.factory.cache.clone();
    let loader = Arc::new(LocalCalendarLoader::new(
        source,
        Arc::new(IsoCalendarTimestampDecoder),
        warnings.clone(),
    ));
    let provider = Arc::new(CachedCalendarProvider::new(loader, cache.clone()));
    let start =
        chrono::NaiveDateTime::parse_from_str("2024-01-02T09:30:00", "%Y-%m-%dT%H:%M:%S").unwrap();
    let end = start + chrono::TimeDelta::minutes(5);
    let cursor = Arc::new(Mutex::new(
        ExecutionCalendar::new(provider, "2min".into(), Some(start), Some(end)).unwrap(),
    ));
    let batch = RecordBatch::try_from_iter(vec![
        (
            "instrument",
            Arc::new(StringArray::from(vec!["A"])) as ArrayRef,
        ),
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(vec![0])) as ArrayRef,
        ),
    ])
    .unwrap();
    let exchange = Arc::new(RwLock::new(ExecutionExchange {
        frequency: "1min".into(),
        quotes: ExchangeQuoteProvider::new(
            Arc::new(ArrowQuote::try_new(&batch, "instrument", "datetime").unwrap()),
            DealPriceFields::shared("close").unwrap(),
        ),
    }));
    let level = Arc::new(RwLock::new(ExecutionLevelBindings {
        common: Some(Arc::new(RwLock::new(ExecutionCommonBindings {
            exchange: Some(exchange.clone()),
        }))),
    }));
    let calendar = SharedExecutionCalendar::new(cursor, level);
    let strategy = SingleOrderStrategy::new(Order::new("A", 7.0, OrderDir::Sell, None, None), None);
    let mut steps = Vec::new();
    while !calendar.finished().unwrap() {
        let decision = strategy
            .generate_trade_decision(None, &mut Orders, &calendar)
            .unwrap();
        let order = &decision.orders()[0];
        assert_eq!(order.start_time(), Some(decision.start_time()));
        assert_eq!(order.end_time(), Some(decision.end_time()));
        assert_eq!(order.stock_id(), "A");
        assert_eq!(order.amount().to_bits(), 7.0_f64.to_bits());
        assert_eq!(order.direction(), OrderDir::Sell);
        assert_eq!(order.deal_amount().to_bits(), 0.0_f64.to_bits());
        assert_eq!(order.factor(), None);
        let minute = calendar.available_step_range().unwrap();
        exchange.write().unwrap().frequency = "2min".into();
        let coarse = calendar.available_step_range().unwrap();
        exchange.write().unwrap().frequency = "1min".into();
        steps.push(
            json!({"start":decision.start_time().format("%Y-%m-%dT%H:%M:%S").to_string(),
            "end":decision.end_time().format("%Y-%m-%dT%H:%M:%S").to_string(),
            "minute":[minute.0,minute.1], "coarse":[coarse.0,coarse.1]}),
        );
        calendar.step().unwrap();
    }
    assert_eq!(json!(steps), case["steps"]);
    assert_eq!(
        json!(warnings.count.load(Ordering::SeqCst)),
        case["warnings"]
    );
    assert!(calendar.step().is_err());
    // Delete only the two files created inside this test's unique temporary directory.
    std::fs::remove_file(directory.join("1min.txt")).unwrap();
    if case["future"].as_bool().unwrap() {
        std::fs::remove_file(directory.join("1min_future.txt")).unwrap();
    }
    calendar.reset_window(start, end).unwrap();
    assert_eq!(json!(calendar.trade_len().unwrap()), case["cached_length"]);
    cache.clear().unwrap();
    assert!(calendar.reset_window(start, end).is_err());
    assert_eq!(case["reload_error"], "Value");
}
