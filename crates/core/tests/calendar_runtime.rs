use std::{
    process::Command,
    sync::{Arc, Mutex},
};

use chrono::NaiveDateTime;
use domain_core::{
    CalendarLoadError, CalendarResampling, CalendarRuntimeConfiguration, LiveCalendarResampling,
};
use num_bigint::BigInt;
use serde_json::{Value, json};

struct TracedConfig {
    region: Option<String>,
    mode: String,
    events: Mutex<Vec<String>>,
}

impl CalendarRuntimeConfiguration for TracedConfig {
    fn region(&self) -> Result<Option<String>, CalendarLoadError> {
        self.events.lock().unwrap().push("region".into());
        if self.region.as_deref() == Some("missing") {
            return Err(CalendarLoadError::Other("region".into()));
        }
        Ok(self.region.clone())
    }

    fn minute_shift(&self) -> Result<BigInt, CalendarLoadError> {
        let mut events = self.events.lock().unwrap();
        events.push("shift".into());
        let count = events.iter().filter(|event| *event == "shift").count();
        if self.mode == "fail_first" || self.mode == "fail_second" && count == 2 {
            return Err(CalendarLoadError::Other("min_data_shift".into()));
        }
        Ok(BigInt::from(if self.mode == "changing" {
            count - 1
        } else {
            0
        }))
    }
}

fn timestamp(text: &str) -> NaiveDateTime {
    NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%M:%S").unwrap()
}

#[test]
fn raw_region_and_live_shift_order_match_actual_source() {
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/calendar_runtime_contract.py"
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
    assert_eq!(cases.len(), 720);
    for case in cases {
        let configuration = Arc::new(TracedConfig {
            region: serde_json::from_value(case["global_region"].clone()).unwrap(),
            mode: case["mode"].as_str().unwrap().into(),
            events: Mutex::new(vec![]),
        });
        let policy = LiveCalendarResampling {
            captured_region: serde_json::from_value(case["captured"].clone()).unwrap(),
            configuration: configuration.clone(),
        };
        let raw: Vec<_> = case["raw"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| timestamp(v.as_str().unwrap()))
            .collect();
        let result = policy.resample(
            &raw,
            &case["source"].as_str().unwrap().parse().unwrap(),
            &case["requested"].as_str().unwrap().parse().unwrap(),
            &999.into(),
        );
        let actual = match result {
            Ok(values) => {
                json!({"values": values.iter().map(|v| v.format("%Y-%m-%dT%H:%M:%S").to_string()).collect::<Vec<_>>()})
            }
            Err(CalendarLoadError::Value(_)) => json!({"error": "Value"}),
            Err(CalendarLoadError::Other(_)) => json!({"error": "Other"}),
        };
        assert_eq!(actual, case["result"], "{case}");
        assert_eq!(
            json!(*configuration.events.lock().unwrap()),
            case["events"],
            "{case}"
        );
    }
}

#[test]
fn live_policy_is_used_by_persistent_storage_and_fresh_backend() {
    use domain_core::{
        CalendarBackendSource, CalendarCache, CalendarTextEncoding, CalendarValue, DataPathManager,
        FileCalendarBackend, FileCalendarStorage, IsoCalendarTimestampDecoder,
    };
    use indexmap::IndexMap;

    let directory = tempfile::tempdir().unwrap();
    std::fs::create_dir(directory.path().join("calendars")).unwrap();
    std::fs::write(
        directory.path().join("calendars/1min.txt"),
        "2024-01-02T09:37:00\n",
    )
    .unwrap();
    let configuration = Arc::new(TracedConfig {
        region: Some("cn".into()),
        mode: "changing".into(),
        events: Mutex::new(vec![]),
    });
    let backend = FileCalendarBackend {
        paths: DataPathManager::from_native(
            IndexMap::from([("1min".into(), directory.path().as_os_str().to_owned())]),
            IndexMap::new(),
        ),
        system: "Windows".into(),
        region: LiveCalendarResampling {
            captured_region: None,
            configuration: configuration.clone(),
        },
        minute_shift: 999.into(),
        text_decoder: Arc::new(CalendarTextEncoding::Utf8),
        timestamp_decoder: Arc::new(IsoCalendarTimestampDecoder),
        cache: Arc::new(CalendarCache::new(None)),
        enable_read_cache: true,
    };
    let mut storage = FileCalendarStorage::new(backend, "5min".into(), false);
    assert_eq!(
        storage.data().unwrap(),
        vec![CalendarValue::Timestamp(timestamp("2024-01-02T09:35:00"))]
    );
    assert_eq!(
        storage
            .backend
            .data("5min", false)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap(),
        vec![CalendarValue::Timestamp(timestamp("2024-01-02T09:34:00"))]
    );
    assert_eq!(
        *configuration.events.lock().unwrap(),
        ["region", "shift", "region", "shift"]
    );
    storage.frequency = "1min".into();
    assert_eq!(
        storage.data().unwrap(),
        vec![CalendarValue::Text("2024-01-02T09:37:00".into())]
    );
    assert_eq!(configuration.events.lock().unwrap().len(), 4);
}
