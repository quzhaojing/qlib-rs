use super::block_tests::batch;
use super::{IndexedFrame, dataframe_append_with_warnings};
use arrow_array::{ArrayRef, Float64Array, Int64Array};
use arrow_schema::{DataType, TimeUnit};
use serde_json::{Value, json};
use std::{path::PathBuf, process::Command, sync::Arc};

fn temporal(unit: &str, zone: Option<&str>, values: &Value) -> ArrayRef {
    let unit = match unit {
        "s" => TimeUnit::Second,
        "ms" => TimeUnit::Millisecond,
        "us" => TimeUnit::Microsecond,
        "ns" => TimeUnit::Nanosecond,
        _ => panic!("unknown timestamp unit"),
    };
    let values = Arc::new(Int64Array::from(
        values
            .as_array()
            .unwrap()
            .iter()
            .map(Value::as_i64)
            .collect::<Vec<_>>(),
    )) as ArrayRef;
    super::objects::cast(&values, &DataType::Timestamp(unit, zone.map(Into::into))).unwrap()
}

fn ordinal(rows: usize) -> ArrayRef {
    Arc::new(Int64Array::from(
        (0..rows)
            .map(|i| i64::try_from(i).unwrap())
            .collect::<Vec<_>>(),
    ))
}

#[test]
fn mixed_timezone_columns_and_indexes_preserve_boxed_units_and_zones() {
    for (left_zone, right_zone) in [(None, Some("UTC")), (Some("UTC"), Some("Asia/Shanghai"))] {
        let left = temporal("ns", left_zone, &json!([1]));
        let right = temporal("ns", right_zone, &json!([2]));
        for is_index in [false, true] {
            let (li, ri, ld, rd) = if is_index {
                (left.clone(), right.clone(), ordinal(1), ordinal(1))
            } else {
                (ordinal(1), ordinal(1), left.clone(), right.clone())
            };
            let frame = IndexedFrame::new(li, None, batch(vec![("x", ld)], 1)).unwrap();
            let other = batch(vec![("datetime", ri), ("x", rd)], 1);
            let result = super::dataframe_append(&frame, &other).unwrap();
            let expected = [(1, left_zone), (2, right_zone)]
                .into_iter()
                .map(|(ticks, zone)| super::TemporalFrameValue::Timestamp {
                    ticks,
                    unit: TimeUnit::Nanosecond,
                    timezone: zone.map(Into::into),
                })
                .collect::<Vec<_>>();
            assert_eq!(
                super::temporal_frame_values(if is_index {
                    result.index()
                } else {
                    result.data().column(0)
                })
                .unwrap(),
                expected
            );
        }
    }
}

#[test]
fn timestamp_units_nat_empty_and_overflow_match_actual_source() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/dataframe_temporal_contract.py"))
        .arg(root.join("../../../qlib/qlib/rl/order_execution/utils.py"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let contract: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(contract["pandas"], "2.3.3");
    assert_eq!(contract["numpy"], "2.4.0");
    assert_eq!(
        contract["digest"],
        "9c5cfb2c8d7e8d2d23474be65835077d245a767f1d7d80f1a5c9070ff5296acd"
    );
    let cases = contract["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 1024);
    for case in cases {
        check_case(case);
    }
    check_text_indexes(&contract);
}

fn check_text_indexes(contract: &Value) {
    let cases = contract["text_indexes"].as_array().unwrap();
    assert_eq!(cases.len(), 4);
    let array = |values: &Value| -> ArrayRef {
        Arc::new(arrow_array::StringArray::from(
            values
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap())
                .collect::<Vec<_>>(),
        ))
    };
    for case in cases {
        let left = array(&case["left"]);
        let right = array(&case["right"]);
        let frame = IndexedFrame::new(
            left.clone(),
            None,
            batch(
                vec![("x", Arc::new(Float64Array::from(vec![0.; left.len()])))],
                left.len(),
            ),
        )
        .unwrap();
        let other = batch(
            vec![
                ("datetime", right.clone()),
                ("x", Arc::new(Float64Array::from(vec![1.; right.len()]))),
            ],
            right.len(),
        );
        let mut warnings = vec![];
        let result = dataframe_append_with_warnings(&frame, &other, &mut |message| {
            warnings.push(json!(["FutureWarning", message]));
        })
        .unwrap();
        assert_eq!(result.index().to_data(), array(&case["output"]).to_data());
        assert_eq!(result.index_name(), case["name"].as_str());
        assert_eq!(case["dtype"], "object");
        assert_eq!(json!(warnings), case["warnings"]);
    }
}

fn check_case(case: &Value) {
    let left = temporal(
        case["lu"].as_str().unwrap(),
        case["zone"].as_str(),
        &case["left"],
    );
    let right = temporal(
        case["ru"].as_str().unwrap(),
        case["zone"].as_str(),
        &case["right"],
    );
    let is_index = case["mode"] == "index";
    let (li, ri, ld, rd) = if is_index {
        (
            left.clone(),
            right.clone(),
            Arc::new(Float64Array::from(vec![0.; left.len()])) as ArrayRef,
            Arc::new(Float64Array::from(vec![1.; right.len()])) as ArrayRef,
        )
    } else {
        (
            ordinal(left.len()),
            ordinal(right.len()),
            left.clone(),
            right.clone(),
        )
    };
    let frame =
        IndexedFrame::new(li.clone(), None, batch(vec![("x", ld.clone())], left.len())).unwrap();
    let other = batch(
        vec![("datetime", ri.clone()), ("x", rd.clone())],
        right.len(),
    );
    let mut warnings = vec![];
    let result = dataframe_append_with_warnings(&frame, &other, &mut |message| {
        warnings.push(json!(["FutureWarning", message]));
    });
    assert_eq!(json!(warnings), case["warnings"], "{case}");
    if case.get("error").is_some() {
        assert_eq!(case["error"], "OutOfBoundsDatetime");
        let error = result.unwrap_err();
        assert!(
            matches!(error, super::DataframeAppendError::Arrow(_)),
            "{error}"
        );
        assert!(
            error.to_string().to_lowercase().contains("overflow"),
            "{error}"
        );
    } else {
        let result = result.unwrap_or_else(|error| panic!("{case}: {error}"));
        let values = if is_index {
            result.index()
        } else {
            result.data().column(0)
        };
        let expected = temporal(
            case["output"]["unit"].as_str().unwrap(),
            case["zone"].as_str(),
            &case["output"]["values"],
        );
        assert_eq!(values.to_data(), expected.to_data(), "{case}");
        assert_eq!(result.index_name(), case["index_name"].as_str());
        assert_eq!(result.blocks(), &[vec![0]]);
        let ints = super::objects::cast(values, &DataType::Int64).unwrap();
        assert_eq!(ints.len(), left.len() + right.len());
    }
    assert!(Arc::ptr_eq(frame.index(), &li));
    assert!(Arc::ptr_eq(frame.data().column(0), &ld));
    assert!(Arc::ptr_eq(other.column(0), &ri));
    assert!(Arc::ptr_eq(other.column(1), &rd));
}
