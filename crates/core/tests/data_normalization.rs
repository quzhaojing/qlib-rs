use std::{path::PathBuf, process::Command, sync::Arc};

use arrow_array::{
    Array, ArrayRef, Float16Array, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array,
    Int64Array, StringArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow_schema::DataType;
use domain_core::{
    NormalizationError, NormalizationWarning, NumericColumn, NumericFrame, NumericSeries,
    PandasNumericDtype, robust_zscore, robust_zscore_with_warnings, zscore, zscore_with_warnings,
};
use half::f16;
use serde_json::Value;

fn native(values: ArrayRef) -> NumericColumn {
    let dtype = PandasNumericDtype::native(values.data_type().clone()).unwrap();
    NumericColumn::try_new(values, dtype).unwrap()
}

fn nullable(values: ArrayRef) -> NumericColumn {
    let dtype = PandasNumericDtype::nullable(values.data_type().clone()).unwrap();
    NumericColumn::try_new(values, dtype).unwrap()
}

fn series(column: NumericColumn) -> NumericSeries {
    let index = Arc::new(StringArray::from_iter_values(
        (0..column.values().len()).map(|position| format!("row-{position}")),
    ));
    NumericSeries::try_new(
        index,
        Some("rows".to_owned()),
        Some("signal".to_owned()),
        column,
    )
    .unwrap()
}

fn python_snapshot() -> Value {
    let source = std::env::var_os("QLIB_PYTHON_DATA").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/utils/data.py"),
        PathBuf::from,
    );
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/data_normalization.py");
    let python = std::env::var_os("QLIB_PYTHON").unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../qlib/.venv/Scripts/python.exe")
            .into_os_string()
    });
    let output = Command::new(python)
        .arg(fixture)
        .arg(source)
        .output()
        .expect("source-pinned Python fixture starts");
    assert!(
        output.status.success(),
        "Python fixture failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("fixture returns JSON")
}

fn encoded_values(column: &NumericColumn) -> Vec<Value> {
    fn encode(value: Option<f64>) -> Value {
        match value {
            None => Value::Null,
            Some(value) if value.is_nan() => Value::String("nan".to_owned()),
            Some(value) if value == f64::INFINITY => Value::String("inf".to_owned()),
            Some(value) if value == f64::NEG_INFINITY => Value::String("-inf".to_owned()),
            Some(value) => Value::from(value),
        }
    }
    match column.values().data_type() {
        DataType::Float16 => column
            .values()
            .as_any()
            .downcast_ref::<Float16Array>()
            .unwrap()
            .iter()
            .map(|value| encode(value.map(f16::to_f64)))
            .collect(),
        DataType::Float32 => column
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .iter()
            .map(|value| encode(value.map(f64::from)))
            .collect(),
        DataType::Float64 => column
            .values()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .iter()
            .map(encode)
            .collect(),
        dtype => panic!("normalization returned unexpected dtype {dtype:?}"),
    }
}

fn assert_values(actual: &[Value], expected: &Value, tolerance: f64) {
    let expected = expected.as_array().unwrap();
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        match (actual.as_f64(), expected.as_f64()) {
            (Some(actual), Some(expected)) => {
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "{actual} != {expected}"
                );
            }
            _ => assert_eq!(actual, expected),
        }
    }
}

fn assert_series_case(snapshot: &Value, case_id: &str, actual: &NumericSeries, tolerance: f64) {
    let expected = &snapshot["cases"][case_id];
    assert_values(
        &encoded_values(actual.column()),
        &expected["values"],
        tolerance,
    );
    assert_eq!(
        actual.column().dtype().is_nullable(),
        expected["dtype"].as_str().unwrap().starts_with("Float")
    );
    let expected_physical = match expected["dtype"]
        .as_str()
        .unwrap()
        .to_ascii_lowercase()
        .as_str()
    {
        "float16" => DataType::Float16,
        "float32" => DataType::Float32,
        "float64" => DataType::Float64,
        dtype => panic!("unexpected fixture dtype {dtype}"),
    };
    assert_eq!(actual.column().dtype().physical(), &expected_physical);
    assert_eq!(actual.index_name(), expected["index_name"].as_str());
    assert_eq!(actual.name(), expected["name"].as_str());
    assert_eq!(
        actual
            .index()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|value| value.unwrap())
            .collect::<Vec<_>>(),
        expected["index"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap())
            .collect::<Vec<_>>()
    );
    assert_eq!(expected["input_preserved"], true);
}

fn warning_strings(warnings: &[NormalizationWarning]) -> Vec<String> {
    warnings
        .iter()
        .map(|warning| format!("{}:{}", warning.category(), warning.message()))
        .collect()
}

fn frame(columns: Vec<(String, NumericColumn)>) -> NumericFrame {
    let rows = columns
        .first()
        .map_or(0, |(_, column)| column.values().len());
    NumericFrame::try_new(
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|position| format!("row-{position}")),
        )),
        Some("rows".to_owned()),
        columns,
    )
    .unwrap()
}

#[test]
fn series_dtypes_and_precision_match_live_source() {
    let snapshot = python_snapshot();
    assert_eq!(snapshot["source_sha256"].as_str().unwrap().len(), 64);

    let input = series(native(Arc::new(Int8Array::from(vec![1, 2, 3]))));
    let before = input.column().values().to_data();
    assert_series_case(&snapshot, "int8_zscore", &zscore(&input).unwrap(), 1e-14);
    assert_series_case(
        &snapshot,
        "int8_robust",
        &robust_zscore(&input, false).unwrap(),
        1e-14,
    );
    assert_eq!(input.column().values().to_data(), before);

    let input = series(native(Arc::new(UInt64Array::from(vec![
        1,
        2,
        2_u64.pow(63),
    ]))));
    assert_series_case(
        &snapshot,
        "uint64_large_robust_post",
        &robust_zscore(&input, true).unwrap(),
        1e-14,
    );

    let input = series(native(Arc::new(Float16Array::from(vec![
        f16::ONE,
        f16::from_f32(2.0),
        f16::from_f32(3.0),
    ]))));
    assert_series_case(
        &snapshot,
        "float16_robust",
        &robust_zscore(&input, false).unwrap(),
        0.0,
    );

    let input = series(native(Arc::new(Float32Array::from(vec![1.0, 2.0, 3.0]))));
    assert_series_case(
        &snapshot,
        "float32_robust",
        &robust_zscore(&input, false).unwrap(),
        0.0,
    );

    let input = series(nullable(Arc::new(Int8Array::from(vec![
        Some(1),
        None,
        Some(3),
    ]))));
    assert_series_case(
        &snapshot,
        "nullable_int8_zscore",
        &zscore(&input).unwrap(),
        1e-14,
    );

    let input = series(nullable(Arc::new(Float32Array::from(vec![
        Some(1.0),
        None,
        Some(3.0),
    ]))));
    assert_series_case(
        &snapshot,
        "nullable_float32_zscore",
        &zscore(&input).unwrap(),
        0.0,
    );
}

#[test]
fn missing_empty_constant_nan_and_infinity_match_live_source() {
    let snapshot = python_snapshot();
    let cases = [
        ("empty", vec![]),
        ("single", vec![5.0]),
        ("constant", vec![5.0, 5.0]),
        ("nan", vec![1.0, f64::NAN, 3.0]),
        ("positive_inf", vec![1.0, f64::INFINITY, 3.0]),
        ("negative_inf", vec![1.0, f64::NEG_INFINITY, 3.0]),
        ("both_inf", vec![f64::NEG_INFINITY, f64::INFINITY]),
        ("all_nan", vec![f64::NAN, f64::NAN]),
    ];
    for (name, values) in cases {
        let input = series(native(Arc::new(Float64Array::from(values))));
        assert_series_case(
            &snapshot,
            &format!("{name}_zscore"),
            &zscore(&input).unwrap(),
            1e-14,
        );
        assert_series_case(
            &snapshot,
            &format!("{name}_robust"),
            &robust_zscore(&input, false).unwrap(),
            1e-14,
        );
        assert_series_case(
            &snapshot,
            &format!("{name}_robust_post"),
            &robust_zscore(&input, true).unwrap(),
            1e-14,
        );
    }
    assert_eq!(
        snapshot["cases"]["positive_inf_zscore"]["warnings"][0],
        "RuntimeWarning:invalid value encountered in subtract"
    );
    assert_eq!(
        snapshot["cases"]["both_inf_robust"]["warnings"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn pairwise_reductions_match_long_cancellation_and_scale_boundaries() {
    let snapshot = python_snapshot();
    let cancellation = [1e20, 1.0, -1e20].repeat(100);
    let cancellation = cancellation.into_iter().chain([3.0]).collect::<Vec<_>>();
    for (case_id, input, tolerance) in [
        (
            "float32_cancellation_zscore",
            series(native(Arc::new(Float32Array::from(
                cancellation
                    .iter()
                    .map(|value| *value as f32)
                    .collect::<Vec<_>>(),
            )))),
            0.0,
        ),
        (
            "float64_cancellation_zscore",
            series(native(Arc::new(Float64Array::from(cancellation)))),
            1e-14,
        ),
    ] {
        assert_series_case(&snapshot, case_id, &zscore(&input).unwrap(), tolerance);
    }

    let nan_positions = [1e20, f64::NAN, 1.0, -1e20].repeat(40);
    let nan_positions = nan_positions.into_iter().chain([3.0]).collect::<Vec<_>>();
    let input = series(native(Arc::new(Float64Array::from(nan_positions))));
    assert_series_case(
        &snapshot,
        "float64_nan_position_zscore",
        &zscore(&input).unwrap(),
        1e-14,
    );

    let input = series(native(Arc::new(Float32Array::from(
        [1e-20_f32, 2e-20, 3e-20].repeat(50),
    ))));
    assert_series_case(
        &snapshot,
        "float32_small_robust_post",
        &robust_zscore(&input, true).unwrap(),
        0.0,
    );

    let mut large = [3.4e38_f32, -3.4e38].repeat(100);
    large.push(1.0);
    let input = series(native(Arc::new(Float32Array::from(large))));
    assert_series_case(
        &snapshot,
        "float32_large_zscore",
        &zscore(&input).unwrap(),
        0.0,
    );

    let input = series(nullable(Arc::new(Float64Array::from(vec![
        Some(1.0),
        None,
        None,
        Some(3.0),
    ]))));
    assert_series_case(
        &snapshot,
        "nullable_float64_nan_zscore",
        &zscore(&input).unwrap(),
        1e-14,
    );
}

#[test]
fn ordered_runtime_warnings_match_series_and_dataframe_source_blocks() {
    let snapshot = python_snapshot();
    let cancellation = [1e20_f32, 1.0, -1e20]
        .repeat(100)
        .into_iter()
        .chain([3.0])
        .collect::<Vec<_>>();
    let large = [3.4e38_f32, -3.4e38]
        .repeat(100)
        .into_iter()
        .chain([1.0])
        .collect::<Vec<_>>();
    let series_cases = vec![
        (
            "positive_inf_zscore",
            series(native(Arc::new(Float64Array::from(vec![
                1.0,
                f64::INFINITY,
                3.0,
            ])))),
            None,
        ),
        (
            "both_inf_zscore",
            series(native(Arc::new(Float64Array::from(vec![
                f64::NEG_INFINITY,
                f64::INFINITY,
            ])))),
            None,
        ),
        (
            "both_inf_robust",
            series(native(Arc::new(Float64Array::from(vec![
                f64::NEG_INFINITY,
                f64::INFINITY,
            ])))),
            Some(false),
        ),
        (
            "float32_cancellation_zscore",
            series(native(Arc::new(Float32Array::from(cancellation)))),
            None,
        ),
        (
            "float32_large_zscore",
            series(native(Arc::new(Float32Array::from(large)))),
            None,
        ),
        (
            "empty_zscore",
            series(native(Arc::new(Float64Array::from(Vec::<f64>::new())))),
            None,
        ),
        (
            "all_nan_robust_post",
            series(native(Arc::new(Float64Array::from(vec![
                f64::NAN,
                f64::NAN,
            ])))),
            Some(true),
        ),
        (
            "both_inf_robust_post",
            series(native(Arc::new(Float64Array::from(vec![
                f64::NEG_INFINITY,
                f64::INFINITY,
            ])))),
            Some(true),
        ),
        (
            "native_extreme_opposite_zscore",
            series(native(Arc::new(Float64Array::from(vec![1e308, -1e308])))),
            None,
        ),
        (
            "native_extreme_opposite_robust",
            series(native(Arc::new(Float64Array::from(vec![1e308, -1e308])))),
            Some(false),
        ),
        (
            "nullable_extreme_opposite_zscore",
            series(nullable(Arc::new(Float64Array::from(vec![1e308, -1e308])))),
            None,
        ),
        (
            "nullable_extreme_opposite_robust_post",
            series(nullable(Arc::new(Float64Array::from(vec![1e308, -1e308])))),
            Some(true),
        ),
        (
            "native_extreme_same_zscore",
            series(native(Arc::new(Float64Array::from(vec![1.7e308, 1.7e308])))),
            None,
        ),
        (
            "native_extreme_same_robust_post",
            series(native(Arc::new(Float64Array::from(vec![1.7e308, 1.7e308])))),
            Some(true),
        ),
        (
            "nullable_extreme_same_zscore",
            series(nullable(Arc::new(Float64Array::from(vec![
                1.7e308, 1.7e308,
            ])))),
            None,
        ),
        (
            "nullable_extreme_same_robust",
            series(nullable(Arc::new(Float64Array::from(vec![
                1.7e308, 1.7e308,
            ])))),
            Some(false),
        ),
    ];
    for (case_id, input, robust) in series_cases {
        let report = if let Some(post) = robust {
            robust_zscore_with_warnings(&input, post).unwrap()
        } else {
            zscore_with_warnings(&input).unwrap()
        };
        assert_eq!(report.output().name(), Some("signal"));
        assert_series_case(&snapshot, case_id, report.output(), 2e-7);
        assert_eq!(
            warning_strings(report.warnings()),
            snapshot["cases"][case_id]["warnings"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_str().unwrap())
                .collect::<Vec<_>>()
        );
    }

    let both_inf = || {
        native(Arc::new(Float64Array::from(vec![
            f64::NEG_INFINITY,
            f64::INFINITY,
        ])))
    };
    let frame_cases = vec![
        (
            "native_positive_inf_zscore_false",
            frame(vec![
                (
                    "a".to_owned(),
                    native(Arc::new(Float64Array::from(vec![1.0, f64::INFINITY, 3.0]))),
                ),
                (
                    "b".to_owned(),
                    native(Arc::new(Float64Array::from(vec![2.0, 4.0, 6.0]))),
                ),
            ]),
            None,
        ),
        (
            "native_both_inf_zscore_false",
            frame(vec![
                ("a".to_owned(), both_inf()),
                ("b".to_owned(), both_inf()),
            ]),
            None,
        ),
        (
            "two_native_blocks_zscore_false",
            frame(vec![
                (
                    "a".to_owned(),
                    native(Arc::new(Float32Array::from(vec![1.0, f32::INFINITY, 3.0]))),
                ),
                (
                    "b".to_owned(),
                    native(Arc::new(Float64Array::from(vec![1.0, f64::INFINITY, 3.0]))),
                ),
            ]),
            None,
        ),
        (
            "nullable_positive_inf_zscore_false",
            frame(vec![(
                "a".to_owned(),
                nullable(Arc::new(Float32Array::from(vec![1.0, f32::INFINITY, 3.0]))),
            )]),
            None,
        ),
        (
            "native_both_inf_robust_zscore_false",
            frame(vec![
                ("a".to_owned(), both_inf()),
                ("b".to_owned(), both_inf()),
            ]),
            Some(false),
        ),
        (
            "native_both_inf_robust_zscore_true",
            frame(vec![
                ("a".to_owned(), both_inf()),
                ("b".to_owned(), both_inf()),
            ]),
            Some(true),
        ),
        (
            "native_extreme_opposite_zscore_false",
            frame(vec![(
                "a".to_owned(),
                native(Arc::new(Float64Array::from(vec![1e308, -1e308]))),
            )]),
            None,
        ),
        (
            "nullable_extreme_opposite_robust_zscore_true",
            frame(vec![(
                "a".to_owned(),
                nullable(Arc::new(Float64Array::from(vec![1e308, -1e308]))),
            )]),
            Some(true),
        ),
        (
            "native_extreme_same_zscore_false",
            frame(vec![(
                "a".to_owned(),
                native(Arc::new(Float64Array::from(vec![1.7e308, 1.7e308]))),
            )]),
            None,
        ),
        (
            "nullable_extreme_same_robust_zscore_false",
            frame(vec![(
                "a".to_owned(),
                nullable(Arc::new(Float64Array::from(vec![1.7e308, 1.7e308]))),
            )]),
            Some(false),
        ),
    ];
    for (case_id, input, robust) in frame_cases {
        let report = if let Some(post) = robust {
            robust_zscore_with_warnings(&input, post).unwrap()
        } else {
            zscore_with_warnings(&input).unwrap()
        };
        assert_eq!(
            warning_strings(report.warnings()),
            snapshot["warning_frames"][case_id]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_str().unwrap())
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn mixed_dataframe_is_columnwise_and_preserves_labels_order_and_input() {
    let snapshot = python_snapshot();
    let index: ArrayRef = Arc::new(StringArray::from(vec!["r2", "r1", "r3"]));
    let input = NumericFrame::try_new(
        Arc::clone(&index),
        Some("rows".to_owned()),
        vec![
            (
                "integer".to_owned(),
                native(Arc::new(Int32Array::from(vec![1, 2, 3]))),
            ),
            (
                "float".to_owned(),
                native(Arc::new(Float32Array::from(vec![1.0, f32::NAN, 3.0]))),
            ),
            (
                "masked".to_owned(),
                nullable(Arc::new(Int64Array::from(vec![Some(1), None, Some(3)]))),
            ),
        ],
    )
    .unwrap();
    let before = input
        .columns()
        .iter()
        .map(|(_, column)| column.values().to_data())
        .collect::<Vec<_>>();

    for (case_id, output) in [
        ("mixed_frame_zscore_false", zscore(&input).unwrap()),
        (
            "mixed_frame_robust_zscore_false",
            robust_zscore(&input, false).unwrap(),
        ),
        (
            "mixed_frame_robust_zscore_true",
            robust_zscore(&input, true).unwrap(),
        ),
    ] {
        let expected = &snapshot["cases"][case_id];
        assert_eq!(output.index().to_data(), input.index().to_data());
        assert_eq!(output.index_name(), Some("rows"));
        assert_eq!(
            output
                .columns()
                .iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>(),
            ["integer", "float", "masked"]
        );
        for ((_, column), values) in output
            .columns()
            .iter()
            .zip(expected["values"].as_array().unwrap())
        {
            assert_values(&encoded_values(column), values, 2e-7);
        }
        assert_eq!(
            output
                .columns()
                .iter()
                .map(|(_, column)| {
                    let prefix = if column.dtype().is_nullable() {
                        "Float"
                    } else {
                        "float"
                    };
                    let width = match column.dtype().physical() {
                        DataType::Float32 => "32",
                        DataType::Float64 => "64",
                        dtype => panic!("unexpected frame output dtype {dtype:?}"),
                    };
                    format!("{prefix}{width}")
                })
                .collect::<Vec<_>>(),
            expected["dtypes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_str().unwrap())
                .collect::<Vec<_>>()
        );
        assert_eq!(expected["input_preserved"], true);
    }
    assert_eq!(
        input
            .columns()
            .iter()
            .map(|(_, column)| column.values().to_data())
            .collect::<Vec<_>>(),
        before
    );
}

#[test]
fn every_supported_integer_width_and_nullable_only_or_empty_frames_work() {
    let columns = vec![
        native(Arc::new(Int16Array::from(vec![1, 2, 3]))),
        native(Arc::new(Int32Array::from(vec![1, 2, 3]))),
        native(Arc::new(Int64Array::from(vec![1, 2, 3]))),
        native(Arc::new(UInt8Array::from(vec![1, 2, 3]))),
        native(Arc::new(UInt16Array::from(vec![1, 2, 3]))),
        native(Arc::new(UInt32Array::from(vec![1, 2, 3]))),
    ];
    for column in columns {
        assert_values(
            &encoded_values(zscore(&series(column)).unwrap().column()),
            &serde_json::json!([-1.0, 0.0, 1.0]),
            0.0,
        );
    }

    let index: ArrayRef = Arc::new(StringArray::from(vec!["a", "b"]));
    let nullable_only = NumericFrame::try_new(
        Arc::clone(&index),
        None,
        vec![(
            "x".to_owned(),
            nullable(Arc::new(Float64Array::from(vec![Some(1.0), None]))),
        )],
    )
    .unwrap();
    assert_eq!(zscore(&nullable_only).unwrap().columns().len(), 1);

    let empty = NumericFrame::try_new(index, None, Vec::new()).unwrap();
    assert!(robust_zscore(&empty, false).unwrap().columns().is_empty());
}

#[test]
fn invalid_dtype_storage_null_and_shape_boundaries_are_typed_failures() {
    assert_eq!(
        PandasNumericDtype::native(DataType::Utf8),
        Err(NormalizationError::UnsupportedDtype(DataType::Utf8))
    );
    assert_eq!(
        PandasNumericDtype::nullable(DataType::Float16),
        Err(NormalizationError::UnsupportedDtype(DataType::Float16))
    );

    let dtype = PandasNumericDtype::native(DataType::Int64).unwrap();
    assert_eq!(dtype.physical(), &DataType::Int64);
    assert!(!dtype.is_nullable());
    assert!(matches!(
        NumericColumn::try_new(Arc::new(Int32Array::from(vec![1])), dtype),
        Err(NormalizationError::DtypeMismatch {
            declared: DataType::Int64,
            actual: DataType::Int32
        })
    ));
    assert!(matches!(
        NumericColumn::try_new(
            Arc::new(Float64Array::from(vec![None, Some(1.0)])),
            PandasNumericDtype::native(DataType::Float64).unwrap()
        ),
        Err(NormalizationError::NativeNulls)
    ));

    let one_index: ArrayRef = Arc::new(StringArray::from(vec!["a"]));
    let two_values = native(Arc::new(Int8Array::from(vec![1, 2])));
    assert!(matches!(
        NumericSeries::try_new(Arc::clone(&one_index), None, None, two_values.clone()),
        Err(NormalizationError::IndexLength {
            index: 1,
            values: 2
        })
    ));
    assert!(matches!(
        NumericFrame::try_new(one_index, None, vec![("x".to_owned(), two_values)]),
        Err(NormalizationError::IndexLength {
            index: 1,
            values: 2
        })
    ));

    assert_eq!(snapshot_error_classes(), ("TypeError", "TypeError"));
}

fn snapshot_error_classes() -> (&'static str, &'static str) {
    let snapshot = python_snapshot();
    let text = snapshot["errors"]["text_zscore"]["class"].as_str().unwrap();
    let robust_text = snapshot["errors"]["text_robust"]["class"].as_str().unwrap();
    assert_eq!(text, "TypeError");
    assert_eq!(robust_text, "TypeError");
    ("TypeError", "TypeError")
}
