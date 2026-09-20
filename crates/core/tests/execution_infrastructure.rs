use arrow_array::{ArrayRef, Float64Array, RecordBatch, StringArray, TimestampNanosecondArray};
use domain_core::{
    ArrowQuote, DealPriceFields, ExchangeQuoteProvider, ExecutionCalendarContext,
    ExecutionCommonBindings, ExecutionExchange, ExecutionLevelBindings, TimeRange,
};
use serde_json::{Value, json};
use std::{
    process::Command,
    sync::{Arc, RwLock},
};

fn exchange(frequency: &str) -> Arc<RwLock<ExecutionExchange>> {
    let batch = RecordBatch::try_from_iter(vec![
        (
            "instrument",
            Arc::new(StringArray::from(vec!["A"])) as ArrayRef,
        ),
        (
            "datetime",
            Arc::new(TimestampNanosecondArray::from(vec![0])) as ArrayRef,
        ),
        (
            "$factor",
            Arc::new(Float64Array::from(vec![2.0])) as ArrayRef,
        ),
    ])
    .unwrap();
    Arc::new(RwLock::new(ExecutionExchange {
        frequency: frequency.into(),
        quotes: ExchangeQuoteProvider::new(
            Arc::new(ArrowQuote::try_new(&batch, "instrument", "datetime").unwrap()),
            DealPriceFields::shared("close").unwrap(),
        ),
    }))
}

fn read(level: &RwLock<ExecutionLevelBindings>) -> Value {
    match level.data_frequency() {
        Ok(value) => json!({"frequency":value}),
        Err(error) => {
            let message = error.to_string();
            let binding = message
                .strip_prefix(
                    "execution calendar provider error: missing execution infrastructure: ",
                )
                .unwrap();
            json!({"missing":format!("infra {binding} is not found!")})
        }
    }
}

#[test]
fn live_bindings_match_source_replacement_and_preserve_quote_identity() {
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/execution_infrastructure_contract.py"
        ))
        .arg("D:/code/github/qlib/qlib")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let expected: Value = serde_json::from_slice(&output.stdout).unwrap();
    let level = RwLock::new(ExecutionLevelBindings::default());
    let mut rows = vec![read(&level)];
    let common = Arc::new(RwLock::new(ExecutionCommonBindings::default()));
    level.write().unwrap().common = Some(common.clone());
    rows.push(read(&level));
    let old = exchange("1min");
    common.write().unwrap().exchange = Some(old.clone());
    rows.push(read(&level));
    let bound = ExecutionLevelBindings::exchange(&level).unwrap();
    assert!(Arc::ptr_eq(&bound, &old));
    assert_eq!(
        bound
            .read()
            .unwrap()
            .quotes
            .get_factor("A", TimeRange::default())
            .unwrap()
            .unwrap()
            .to_bits(),
        2.0_f64.to_bits()
    );
    old.write().unwrap().frequency = "bad-frequency".into();
    rows.push(read(&level));
    common.write().unwrap().exchange = Some(exchange("2min"));
    rows.push(read(&level));
    old.write().unwrap().frequency = "stale".into();
    rows.push(read(&level));
    level.write().unwrap().common = Some(Arc::new(RwLock::new(ExecutionCommonBindings {
        exchange: Some(exchange("day")),
    })));
    rows.push(read(&level));
    common
        .read()
        .unwrap()
        .exchange
        .as_ref()
        .unwrap()
        .write()
        .unwrap()
        .frequency = "stale-common".into();
    rows.push(read(&level));
    assert_eq!(json!(rows), expected);
}

fn poison<T: Send + Sync + 'static>(lock: Arc<RwLock<T>>) {
    assert!(
        std::thread::spawn(move || {
            let _guard = lock.write().unwrap();
            panic!("intentional poisoning");
        })
        .join()
        .is_err()
    );
}

#[test]
fn poisoned_binding_at_each_level_is_not_recovered() {
    for part in ["level", "common", "exchange"] {
        let exchange = exchange("1min");
        let common = Arc::new(RwLock::new(ExecutionCommonBindings {
            exchange: Some(exchange.clone()),
        }));
        let level = Arc::new(RwLock::new(ExecutionLevelBindings {
            common: Some(common.clone()),
        }));
        match part {
            "level" => poison(level.clone()),
            "common" => poison(common),
            _ => poison(exchange),
        }
        assert_eq!(
            level.data_frequency().unwrap_err().to_string(),
            format!("execution calendar provider error: execution {part} binding lock poisoned")
        );
    }
}
