use domain_core::{
    CachedCalendarProvider, CalendarBackendSource, CalendarCache, CalendarLoadError, CalendarRows,
    CalendarTextDecoder, CalendarTextEncoding, CalendarValue, ExecutionCalendarProvider,
    IsoCalendarTimestampDecoder, LocalCalendarLoader, TracingCalendarWarnings,
};
use serde_json::{Value, json};
use std::{
    process::Command,
    sync::{Arc, Mutex},
};

fn decoded(bytes: &[u8], encoding: CalendarTextEncoding) -> Value {
    match encoding.decode(bytes) {
        Ok(text) => json!([text, null]),
        Err(error) => json!([null, [error.start, error.end, error.reason]]),
    }
}

#[test]
fn strict_gbk_matches_all_single_bytes_pairs_and_mixed_sequences() {
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/calendar_text_contract.py"
        ))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let expected: Value = serde_json::from_slice(&output.stdout).unwrap();
    for value in 0_u8..=u8::MAX {
        assert_eq!(
            decoded(&[value], CalendarTextEncoding::PythonGbk),
            expected["single"][usize::from(value)],
            "single {value:02x}"
        );
    }
    for value in 0_u16..=u16::MAX {
        assert_eq!(
            decoded(&value.to_be_bytes(), CalendarTextEncoding::PythonGbk),
            expected["pairs"][usize::from(value)],
            "pair {value:04x}"
        );
    }
    for case in expected["mixed"].as_array().unwrap() {
        let bytes: Vec<u8> = serde_json::from_value(case[0].clone()).unwrap();
        assert_eq!(
            decoded(&bytes, CalendarTextEncoding::PythonGbk),
            case[1],
            "{bytes:?}"
        );
    }
    for case in expected["utf8"].as_array().unwrap() {
        let bytes: Vec<u8> = serde_json::from_value(case[0].clone()).unwrap();
        match CalendarTextEncoding::Utf8.decode(&bytes) {
            Ok(text) => assert_eq!(json!([text, null]), case[1]),
            Err(error) => {
                assert_eq!(case[1][0], Value::Null);
                assert_eq!(
                    json!([error.start, error.end]),
                    json!([case[1][1][0], case[1][1][1]])
                );
                assert_eq!(error.encoding, CalendarTextEncoding::Utf8);
                assert!(error.reason.ends_with("utf-8 sequence"));
            }
        }
    }
}

struct BytesBackend {
    encoding: CalendarTextEncoding,
    cache: Arc<CalendarCache>,
    reads: Mutex<Vec<bool>>,
}

impl CalendarBackendSource for BytesBackend {
    fn data(&self, _: &str, future: bool) -> Result<CalendarRows, CalendarLoadError> {
        let lines =
            self.cache
                .get_or_load_raw(if future { "future" } else { "current" }, &mut || {
                    self.reads.lock().unwrap().push(future);
                    let bytes = if future {
                        b"\xff".as_slice()
                    } else {
                        b"2024-01-02".as_slice()
                    };
                    Ok(vec![self.encoding.decode_text(bytes)?].into())
                })?;
        let rows: Vec<_> = lines
            .iter()
            .cloned()
            .map(|line| Ok(CalendarValue::Text(line)))
            .collect();
        Ok(Box::new(rows.into_iter()))
    }
}

#[test]
fn real_decoders_preserve_value_errors_through_raw_cache_and_future_fallback() {
    for encoding in [CalendarTextEncoding::Utf8, CalendarTextEncoding::PythonGbk] {
        let decoder: &dyn CalendarTextDecoder = &encoding;
        assert_eq!(decoder.decode_text(b"date").unwrap(), "date");
        let error = encoding.decode(b"\xff").unwrap_err();
        assert_eq!(error.start, 0);
        assert_eq!(error.end, 1);
        assert_eq!(
            decoder.decode_text(b"\xff").unwrap_err(),
            CalendarLoadError::Value(error.to_string())
        );
        let cache = Arc::new(CalendarCache::new(None));
        let backend = Arc::new(BytesBackend {
            encoding,
            cache: Arc::clone(&cache),
            reads: Mutex::new(Vec::new()),
        });
        let loader = LocalCalendarLoader::new(
            backend.clone(),
            Arc::new(IsoCalendarTimestampDecoder),
            Arc::new(TracingCalendarWarnings),
        );
        let provider = CachedCalendarProvider::new(Arc::new(loader), cache);
        let result = provider.calendar("day", true).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].to_string(), "2024-01-02 00:00:00");
        assert_eq!(*backend.reads.lock().unwrap(), [true, false]);
        assert!(Arc::ptr_eq(
            &result,
            &provider.calendar("day", true).unwrap()
        ));
        assert_eq!(*backend.reads.lock().unwrap(), [true, false]);
    }
}
