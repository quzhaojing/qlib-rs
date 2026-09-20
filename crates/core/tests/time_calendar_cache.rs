use std::{
    path::PathBuf,
    process::Command,
    sync::{Arc, Barrier},
    thread,
};

use domain_core::{
    MarketCalendarError,
    time_calendar_cache::{
        TIME_CALENDAR_CACHE_MAXSIZE, TimeCalendarCache, TimeCalendarCacheError,
        TimeCalendarCacheInfo, TimeCalendarCall, TimeCalendarKeyword, TimeCalendarValue,
    },
};
use num_bigint::BigInt;
use serde_json::{Value, json};

const UPSTREAM_TIME_SHA256: &str =
    "af7ac3709cac0d2a11a15aac478c7ceb68579d492aed322695f5f25261a69699";

fn positional(values: Vec<TimeCalendarValue>) -> TimeCalendarCall {
    TimeCalendarCall::new(values, Vec::new())
}

fn keyword(values: Vec<TimeCalendarKeyword>) -> TimeCalendarCall {
    TimeCalendarCall::new(Vec::new(), values)
}

fn calendar_length(cache: &TimeCalendarCache, call: &TimeCalendarCall) -> usize {
    cache
        .get(call)
        .expect("fixture call succeeds")
        .lock()
        .expect("fixture calendar lock is healthy")
        .len()
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "the pinned whole-source cache snapshot is clearest beside its exact assertions"
)]
fn pinned_python_source_characterizes_raw_cache_semantics() {
    let source = std::env::var_os("QLIB_PYTHON_TIME").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/utils/time.py"),
        PathBuf::from,
    );
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/time_calendar_cache_probe.py");
    let python = std::env::var_os("PYTHON").unwrap_or_else(|| "python".into());
    let output = Command::new(python)
        .arg(fixture)
        .arg(source)
        .output()
        .expect("Python interpreter starts");
    assert!(
        output.status.success(),
        "source characterization failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let actual: Value = serde_json::from_slice(&output.stdout).expect("fixture returns JSON");
    assert_eq!(actual["source_sha256"], UPSTREAM_TIME_SHA256);
    assert_eq!(
        actual["call_shapes"],
        json!({
            "distinct": 8,
            "lengths": [241, 240, 240, 240, 240, 240, 240, 240],
            "info_after_misses": [0, 8, 240, 8],
            "repeat_identity": [true, true, true, true, true, true, true, true],
            "info_after_hits": [8, 8, 240, 8]
        })
    );
    assert_eq!(
        actual["eviction"],
        json!({
            "before": [0, 241, 240, 240], "after": [0, 242, 240, 240],
            "identity_changed": true, "retained_old_length": 240
        })
    );
    assert_eq!(
        actual["concurrency"],
        json!({
            "errors": [], "threads_alive": [false, false], "distinct_results": 2,
            "cached_is_one_result": true, "info": [1, 2, 240, 1]
        })
    );
    assert_eq!(
        actual["failures"]["unsupported_region_first"]["info"],
        json!([0, 1, 240, 0])
    );
    assert_eq!(
        actual["failures"]["unsupported_region_second"]["info"],
        json!([0, 2, 240, 0])
    );
    assert_eq!(
        actual["failures"]["duplicate"]["message"],
        "get_min_cal() got multiple values for argument 'shift'"
    );
    assert_eq!(
        actual["failures"]["duplicate_region"]["message"],
        "get_min_cal() got multiple values for argument 'region'"
    );
    assert_eq!(
        actual["failures"]["too_many"]["message"],
        "get_min_cal() takes from 0 to 2 positional arguments but 3 were given"
    );
    assert_eq!(
        actual["failures"]["unexpected"]["message"],
        "get_min_cal() got an unexpected keyword argument 'other'"
    );
    assert_eq!(actual["failures"]["shift_text"]["error"], "TypeError");
    assert_eq!(actual["failures"]["region_integer"]["error"], "ValueError");
    assert_eq!(
        actual["dynamic_inputs"],
        json!({
            "boolean_cold": {
                "error": "TypeError", "message": "Invalid type <class 'bool'>. Must be int or float.",
                "info": [0, 1, 240, 0]
            },
            "zero_identities": [false, false, true],
            "zero_info": [1, 2, 240, 2],
            "half_minute_first": "09:29:30",
            "none_region": {
                "error": "ValueError", "message": "None is not supported", "info": [1, 4, 240, 3]
            },
            "unhashable_before": [1, 4, 240, 3],
            "unhashable": {
                "error": "TypeError", "message": "unhashable type: 'list'", "info": [1, 4, 240, 3]
            },
            "unhashable_after": [1, 4, 240, 3]
        })
    );
    assert_eq!(actual["boundaries"]["cn_positive_max"]["ok"], 240);
    assert_eq!(
        actual["boundaries"]["cn_positive_over"]["error"],
        "OverflowError"
    );
    assert_eq!(actual["boundaries"]["tw_positive_max"]["ok"], 270);
    assert_eq!(
        actual["boundaries"]["tw_positive_over"]["error"],
        "OverflowError"
    );
    assert_eq!(actual["boundaries"]["negative_min"]["ok"], 390);
    assert_eq!(
        actual["boundaries"]["negative_under"]["error"],
        "OutOfBoundsTimedelta"
    );
}

#[test]
fn native_cache_preserves_call_shape_identity_mutation_and_clear() {
    let cache = TimeCalendarCache::default();
    assert_eq!(TIME_CALENDAR_CACHE_MAXSIZE, 240);
    let calls = [
        TimeCalendarCall::default(),
        positional(vec![0_i64.into()]),
        positional(vec![0_i64.into(), "cn".into()]),
        keyword(vec![TimeCalendarKeyword::new("shift", 0_i64)]),
        keyword(vec![TimeCalendarKeyword::new("region", "cn")]),
        keyword(vec![
            TimeCalendarKeyword::new("shift", 0_i64),
            TimeCalendarKeyword::new("region", "cn"),
        ]),
        keyword(vec![
            TimeCalendarKeyword::new("region", "cn"),
            TimeCalendarKeyword::new("shift", 0_i64),
        ]),
        TimeCalendarCall::new(
            vec![0_i64.into()],
            vec![TimeCalendarKeyword::new("region", "cn")],
        ),
    ];
    let values: Vec<_> = calls
        .iter()
        .map(|call| cache.get(call).expect("supported call shape succeeds"))
        .collect();
    assert!(values.iter().enumerate().all(|(index, value)| {
        values
            .iter()
            .skip(index + 1)
            .all(|other| !Arc::ptr_eq(value, other))
    }));
    values[0]
        .lock()
        .expect("calendar lock is healthy")
        .push(chrono::NaiveTime::MIN);
    assert_eq!(
        values
            .iter()
            .map(|value| value.lock().expect("calendar lock is healthy").len())
            .collect::<Vec<_>>(),
        [241, 240, 240, 240, 240, 240, 240, 240]
    );
    assert_eq!(
        cache.cache_info(),
        TimeCalendarCacheInfo {
            hits: 0,
            misses: 8,
            maxsize: 240,
            currsize: 8
        }
    );
    for (call, original) in calls.iter().zip(&values) {
        let repeated = cache.get(call).expect("repeat call succeeds");
        assert!(Arc::ptr_eq(original, &repeated));
    }
    assert_eq!(cache.cache_info().hits, 8);
    cache.cache_clear();
    assert_eq!(
        cache.cache_info(),
        TimeCalendarCacheInfo {
            hits: 0,
            misses: 0,
            maxsize: 240,
            currsize: 0
        }
    );
    assert_eq!(
        values[0]
            .lock()
            .expect("retained value remains usable")
            .len(),
        241
    );
    assert!(!Arc::ptr_eq(
        &values[0],
        &cache.get(&calls[0]).expect("cleared call rebuilds")
    ));
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "the eviction, failure, and numeric-boundary matrix is one cache-state sequence"
)]
fn native_cache_eviction_failures_and_boundaries_match_source_contract() {
    let cache = TimeCalendarCache::new();
    let first_call = positional(vec![10_000_i64.into()]);
    let first = cache.get(&first_call).expect("first call succeeds");
    for shift in 10_001_i64..10_241_i64 {
        cache
            .get(&positional(vec![shift.into()]))
            .expect("eviction fixture call succeeds");
    }
    assert_eq!(
        cache.cache_info(),
        TimeCalendarCacheInfo {
            hits: 0,
            misses: 241,
            maxsize: 240,
            currsize: 240
        }
    );
    let rebuilt = cache.get(&first_call).expect("evicted call rebuilds");
    assert!(!Arc::ptr_eq(&first, &rebuilt));
    assert_eq!(
        first
            .lock()
            .expect("retained evicted value is usable")
            .len(),
        240
    );

    cache.cache_clear();
    let unsupported = keyword(vec![TimeCalendarKeyword::new("region", "xx")]);
    for misses in 1..=2 {
        assert_eq!(
            cache
                .get(&unsupported)
                .expect_err("unsupported region is rejected"),
            TimeCalendarCacheError::UnsupportedRegion {
                region: "xx".to_owned()
            }
        );
        assert_eq!(cache.cache_info().misses, misses);
        assert_eq!(cache.cache_info().currsize, 0);
    }
    let binding_cases = [
        (
            TimeCalendarCall::new(
                vec![0_i64.into()],
                vec![TimeCalendarKeyword::new("shift", 0_i64)],
            ),
            "get_min_cal() got multiple values for argument 'shift'",
        ),
        (
            TimeCalendarCall::new(
                vec![0_i64.into(), "cn".into()],
                vec![TimeCalendarKeyword::new("region", "cn")],
            ),
            "get_min_cal() got multiple values for argument 'region'",
        ),
        (
            positional(vec![0_i64.into(), "cn".into(), 1_i64.into()]),
            "get_min_cal() takes from 0 to 2 positional arguments but 3 were given",
        ),
        (
            keyword(vec![TimeCalendarKeyword::new("other", 0_i64)]),
            "get_min_cal() got an unexpected keyword argument 'other'",
        ),
    ];
    for (call, message) in binding_cases {
        assert_eq!(
            cache.get(&call).expect_err("invalid binding is rejected"),
            TimeCalendarCacheError::Binding {
                message: message.to_owned()
            }
        );
    }
    assert_eq!(
        cache
            .get(&positional(vec!["0".into()]))
            .expect_err("text shift is outside the native boundary"),
        TimeCalendarCacheError::UnsupportedValueType {
            parameter: "shift",
            expected: "an integer"
        }
    );
    assert_eq!(
        cache
            .get(&keyword(vec![TimeCalendarKeyword::new("region", 1_i64)]))
            .expect_err("integer region is outside the native boundary"),
        TimeCalendarCacheError::UnsupportedValueType {
            parameter: "region",
            expected: "a string"
        }
    );
    assert_eq!(cache.cache_info().currsize, 0);
    assert_eq!(cache.cache_info().misses, 8);

    for (shift, region, expected_length) in [
        (116_906_957_i64, "cn", 240),
        (116_906_927_i64, "tw", 270),
        (-153_722_867_i64, "us", 390),
    ] {
        assert_eq!(
            calendar_length(&cache, &positional(vec![shift.into(), region.into()])),
            expected_length
        );
    }
    for (shift, region) in [
        (116_906_958_i64, "cn"),
        (116_906_928_i64, "tw"),
        (-153_722_868_i64, "us"),
    ] {
        let shift = BigInt::from(shift);
        assert_eq!(
            cache
                .get(&positional(vec![shift.clone().into(), region.into()]))
                .expect_err("out-of-range shift is rejected"),
            TimeCalendarCacheError::Calendar(MarketCalendarError::ShiftOutOfRange { shift })
        );
    }
    assert_eq!(
        TimeCalendarValue::from(String::from("cn")),
        TimeCalendarValue::Text("cn".to_owned())
    );
}

#[test]
fn simultaneous_misses_return_independent_values_and_cache_one() {
    const THREADS: usize = 32;
    let cache = Arc::new(TimeCalendarCache::new());
    let barrier = Arc::new(Barrier::new(THREADS));
    let call = positional(vec![42_424_i64.into(), "us".into()]);
    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            let cache = Arc::clone(&cache);
            let barrier = Arc::clone(&barrier);
            let call = call.clone();
            thread::spawn(move || {
                barrier.wait();
                cache.get(&call).expect("concurrent call succeeds")
            })
        })
        .collect();
    let values: Vec<_> = handles
        .into_iter()
        .map(|handle| handle.join().expect("worker thread succeeds"))
        .collect();
    let distinct = values.iter().enumerate().any(|(index, value)| {
        values
            .iter()
            .skip(index + 1)
            .any(|other| !Arc::ptr_eq(value, other))
    });
    assert!(
        distinct,
        "the synchronized cold calls must overlap at least one miss"
    );
    let cached = cache.get(&call).expect("cached call succeeds");
    assert!(values.iter().any(|value| Arc::ptr_eq(value, &cached)));
    let info = cache.cache_info();
    assert!(info.misses >= 2);
    assert_eq!(
        info.hits + info.misses,
        u64::try_from(THREADS).expect("fixture thread count fits u64") + 1
    );
    assert_eq!(info.currsize, 1);
}
