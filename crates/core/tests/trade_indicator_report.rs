use std::process::Command;

use arrow_array::{Array, Float64Array};
use chrono::NaiveDateTime;
use domain_core::{Indicator, NumpyOrderIndicator, TradeIndicatorReport};
use indexmap::IndexMap;
use serde_json::{Value, json};

#[test]
fn reports_match_source_sparse_index_union_empty_rows_and_replacement() {
    let a = "2024-01-03 09:30:00.123456789";
    let b = "2024-01-01 09:30:00";
    let c = "2024-01-02 09:30:00";
    let cases = json!([
        [],
        [[a, []]],
        [[a, []], [b, [["x", 1.0]]], [c, [["y", 2.0]]]],
        [[a, [["z", 1.0]]], [b, [["x", 2.0]]], [c, [["z", 3.0]]]],
        [
            [a, [["datetime", -0.0], ["x", 2.0]]],
            [b, [["y", 3.0]]],
            [a, [["y", 4.0]]]
        ]
    ]);
    let output = Command::new("python")
        .args([
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/trade_indicator_report.py"
            ),
            r"D:\code\github\qlib\qlib\backtest\report.py",
            &cases.to_string(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let expected: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
    for (case, expected) in cases.as_array().unwrap().iter().zip(expected) {
        let history: IndexMap<_, IndexMap<String, f64>> = case
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                let time =
                    NaiveDateTime::parse_from_str(row[0].as_str().unwrap(), "%Y-%m-%d %H:%M:%S%.f")
                        .unwrap();
                let values = row[1]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| (v[0].as_str().unwrap().to_owned(), v[1].as_f64().unwrap()))
                    .collect();
                (time, values)
            })
            .collect();
        let report = TradeIndicatorReport::from_history(&history);
        let arrays: Vec<_> = report
            .metrics
            .columns()
            .iter()
            .map(|a| {
                assert_eq!(a.null_count(), 0);
                a.as_any().downcast_ref::<Float64Array>().unwrap()
            })
            .collect();
        let rows: Vec<Vec<Value>> = (0..report.metrics.num_rows())
            .map(|i| arrays.iter().map(|a| json!(a.value(i))).collect())
            .collect();
        assert_eq!(
            json!({
                "columns": report.metrics.schema().fields().iter().map(|f| f.name()).collect::<Vec<_>>(),
                "index": report.timestamps.iter().map(ToString::to_string).collect::<Vec<_>>(),
                "data": rows, "dtypes": vec!["float64"; arrays.len()], "index_name": null,
            }),
            expected
        );
    }
}

#[test]
fn report_preserves_nan_signed_zero_and_does_not_change_recorded_history() {
    let time = NaiveDateTime::parse_from_str("2024-01-01 09:30:00", "%Y-%m-%d %H:%M:%S").unwrap();
    let values = IndexMap::from([("datetime".into(), -0.0), ("nan".into(), f64::NAN)]);
    let history = IndexMap::from([(time, values)]);
    let report = TradeIndicatorReport::from_history(&history);
    let zero = report
        .metrics
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(zero.value(0).to_bits(), (-0.0_f64).to_bits());
    let nan = report
        .metrics
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert!(nan.value(0).is_nan());
    let mut indicator = Indicator::<NumpyOrderIndicator>::default();
    indicator.record(time);
    let plugin: &dyn domain_core::AccountIndicator = &indicator;
    let exported = plugin.trade_indicator_report().unwrap();
    assert_eq!(exported.metrics.num_rows(), 0);
    assert!(exported.timestamps.is_empty());
    assert_eq!(
        indicator
            .trade_indicator_report()
            .unwrap()
            .metrics
            .num_rows(),
        0
    );
    assert!(
        indicator
            .trade_indicator_report()
            .unwrap()
            .timestamps
            .is_empty()
    );
    assert_eq!(indicator.trade_indicator_history().len(), 1);
}
