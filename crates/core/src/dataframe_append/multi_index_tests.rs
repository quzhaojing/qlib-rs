use super::*;
use crate::dataframe_append::{
    IndexMetadata, IndexedFrame, block_tests, dataframe_append_with_warnings,
    temporal_append_tests, tuple_frame_array,
    tuple_objects::tests::{cell, compare},
};
use arrow_array::{Int64Array, new_null_array};
use serde_json::{Value, json};
use std::{path::PathBuf, process::Command, sync::Arc};

fn level(value: &Value) -> ArrayRef {
    if value.get("string").is_some() {
        super::super::string_append::tests::descriptor(value).storage()
    } else if let Some(masked) = value.get("masked") {
        super::super::nullable_index::tests::array(masked)
    } else if value["kind"] == "CategoricalIndex" {
        super::super::categorical_append_tests::descriptor(value).storage()
    } else if value["dtype"] == "object" {
        tuple_frame_array(
            &value["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(cell)
                .collect::<Vec<_>>(),
        )
        .unwrap()
    } else {
        temporal_append_tests::column(value["dtype"].as_str().unwrap(), &value["values"], 0)
    }
}

fn typed_level(value: &Value) -> MultiIndexLevel {
    if value.get("string").is_some() {
        MultiIndexLevel::String(super::super::string_append::tests::descriptor(value))
    } else if value.get("masked").is_some() {
        MultiIndexLevel::Nullable(NullableIndexDescriptor::new(level(value)).unwrap())
    } else if value["kind"] == "CategoricalIndex" {
        MultiIndexLevel::Categorical(super::super::categorical_append_tests::descriptor(value))
    } else {
        MultiIndexLevel::Array(level(value))
    }
}

fn imported(value: &Value) -> MultiIndexDescriptor {
    MultiIndexDescriptor::from_levels(
        value["levels"]
            .as_array()
            .unwrap()
            .iter()
            .map(typed_level)
            .collect(),
        value["codes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                v.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_i64().unwrap())
                    .collect()
            })
            .collect(),
        value["names"]
            .as_array()
            .unwrap()
            .iter()
            .map(cell)
            .collect(),
        value["sortorder"].as_i64(),
    )
    .unwrap()
}

#[test]
fn multi_index_import_and_ignored_append_preserve_source_identity() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/dataframe_multi_import.py"))
        .arg(root.join("../../../qlib/qlib/rl/order_execution/utils.py"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let contract: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        contract["digest"],
        "14abd38cb90ae7935b1cdee7f1c33a7a8245141e5028f47ef11543c38610f186"
    );
    let cases = contract["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 1539);
    let valid = cases
        .iter()
        .filter(|c| c.get("output").is_some())
        .collect::<Vec<_>>();
    assert_eq!(valid.len(), 729);
    let right = block_tests::batch(vec![("datetime", tuple_frame_array(&[]).unwrap())], 0);
    for case in valid {
        // The explicit constructor checks sortorder before missing-code
        // normalization; reconstruct from its inputs, not its normalized snapshot.
        let descriptor = imported(case);
        let rows = descriptor.values().len();
        let data = block_tests::batch(
            vec![(
                "x",
                Arc::new(arrow_array::Float64Array::from_iter_values(
                    (0..rows).map(|i| f64::from(u32::try_from(i).unwrap())),
                )),
            )],
            rows,
        );
        let initial = IndexedFrame::from_multi_index(descriptor.clone(), data.clone()).unwrap();
        assert_eq!(
            initial.index_metadata(),
            &IndexMetadata::Multi(Arc::new(descriptor.clone()))
        );
        let attached = IndexedFrame::new(initial.index().clone(), Some("old".into()), data)
            .unwrap()
            .with_index_metadata(IndexMetadata::Multi(Arc::new(descriptor)))
            .unwrap();
        assert_eq!(attached.index_name(), None);
        assert_eq!(attached.index_metadata(), initial.index_metadata());
        let mut current = initial.clone();
        for step in ["empty_append", "second_empty_append"] {
            let mut warnings = vec![];
            current = dataframe_append_with_warnings(&current, &right, &mut |v| {
                warnings.push(v.to_owned());
            })
            .unwrap();
            assert!(warnings.is_empty());
            assert_eq!(case[step]["warnings"], json!([]));
            assert_eq!(case[step]["output"]["index"]["kind"], "MultiIndex");
            assert_eq!(
                current.index_metadata(),
                &IndexMetadata::Multi(Arc::new(imported(&case[step]["output"]["index"])))
            );
            compare(
                &tuple_index::values(current.index()).unwrap(),
                case[step]["output"]["index"]["values"].as_array().unwrap(),
            );
            assert_eq!(current.index_name(), None);
            assert_eq!(current.data(), initial.data());
            assert_eq!(current.blocks(), initial.blocks());
        }
        assert_eq!(initial.index_metadata(), attached.index_metadata());
    }
}

#[test]
fn multi_index_attachment_rejects_mismatches_and_never_silently_demotes() {
    let name = V::Tuple(vec![V::Scalar(T::Builtin(B::Float(f64::NAN)))]);
    let descriptor = MultiIndexDescriptor::new(
        vec![Arc::new(Int64Array::from(vec![1]))],
        vec![vec![0]],
        vec![name],
        Some(1),
    )
    .unwrap();
    assert_eq!(descriptor, descriptor.clone());
    assert_ne!(descriptor, descriptor.appended_without_other());
    assert!(!same_name(
        &V::Tuple(vec![]),
        &V::Tuple(vec![V::Tuple(vec![])])
    ));
    assert!(!same_name(
        &V::Scalar(T::Builtin(B::Float(-0.))),
        &V::Scalar(T::Builtin(B::Float(0.)))
    ));
    assert!(
        IndexedFrame::from_multi_index(descriptor.clone(), block_tests::batch(vec![], 0)).is_err()
    );
    let wrong = IndexedFrame::new(
        Arc::new(Int64Array::from(vec![1])),
        None,
        block_tests::batch(vec![], 1),
    )
    .unwrap();
    assert!(
        wrong
            .with_index_metadata(IndexMetadata::Multi(Arc::new(descriptor.clone())))
            .is_err()
    );
    let frame = IndexedFrame::from_multi_index(descriptor, block_tests::batch(vec![], 1)).unwrap();
    let before = frame.index_metadata().clone();
    let right = block_tests::batch(vec![("datetime", frame.index().clone())], 1);
    let mut warnings = vec![];
    let result =
        dataframe_append_with_warnings(&frame, &right, &mut |v| warnings.push(v.to_owned()))
            .unwrap();
    let IndexMetadata::Multi(output) = result.index_metadata() else {
        panic!("lost multi-level identity")
    };
    assert_eq!(output.codes(), &[vec![Some(0), Some(0)]]);
    assert_eq!(output.sortorder(), None);
    assert!(warnings.is_empty());
    assert_eq!(frame.index_metadata(), &before);
}

fn check(case: &Value) {
    let levels = case["levels"]
        .as_array()
        .unwrap()
        .iter()
        .map(level)
        .collect::<Vec<_>>();
    let before = levels.iter().map(|a| a.to_data()).collect::<Vec<_>>();
    let codes = case["codes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| {
            v.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect()
        })
        .collect();
    let names = case["names"].as_array().unwrap().iter().map(cell).collect();
    let result = MultiIndexDescriptor::from_levels(
        case["levels"]
            .as_array()
            .unwrap()
            .iter()
            .map(typed_level)
            .collect(),
        codes,
        names,
        case["sortorder"].as_i64(),
    );
    assert_eq!(case["warnings"], json!([]));
    if let Some(message) = case.get("message") {
        assert_eq!(case["error"], "ValueError");
        let error = result.unwrap_err();
        if let MultiIndexError::DuplicateLevel(number) = error {
            assert!(
                message
                    .as_str()
                    .unwrap()
                    .starts_with("Level values must be unique:")
            );
            assert!(
                message
                    .as_str()
                    .unwrap()
                    .ends_with(&format!("on level {number}"))
            );
        } else {
            assert_eq!(error.to_string(), message.as_str().unwrap(), "{case}");
        }
    } else {
        let result = result.unwrap_or_else(|e| panic!("{case}: {e}"));
        let expected = &case["output"];
        let codes = result
            .codes()
            .iter()
            .map(|row| {
                row.iter()
                    .map(|code| code.map_or(-1, |code| i64::try_from(code).unwrap()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(json!(codes), expected["codes"], "{case}");
        assert_eq!(json!(result.sortorder()), expected["sortorder"]);
        compare(result.names(), expected["names"].as_array().unwrap());
        compare(&result.values(), expected["values"].as_array().unwrap());
        compare(
            &result.clone().values(),
            expected["values"].as_array().unwrap(),
        );
        compare_levels(&result, expected);
    }
    assert_eq!(
        levels.iter().map(|a| a.to_data()).collect::<Vec<_>>(),
        before
    );
}

#[test]
fn string_multi_levels_preserve_source_missing_identity_and_ignored_appends() {
    check_string_multi_contract(
        "dataframe_multi_string_levels.py",
        "7cf65158f53bf9ed5086c0bb472408d8af97bef6c80bea0e6d2299c4c24c6911",
        1170,
        444,
    );
}

#[test]
fn string_multi_levels_two_level_combinations_match_source() {
    check_string_multi_contract(
        "dataframe_multi_string_pairs.py",
        "787437f50ded4733c2c5b3c6cdaac01da5c2718ad38345f3a3cebd847cce52de",
        3024,
        1412,
    );
}

fn check_string_multi_contract(fixture: &str, digest: &str, total: usize, valid: usize) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new(root.join("../../../qlib/.venv/Scripts/python.exe"))
        .arg(root.join("tests/fixtures").join(fixture))
        .arg(root.join("../../../qlib/qlib/rl/order_execution/utils.py"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let contract: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(contract["digest"], digest);
    let cases = contract["cases"].as_array().unwrap();
    assert_eq!(cases.len(), total);
    assert_eq!(
        cases.iter().filter(|c| c.get("output").is_some()).count(),
        valid
    );
    let right = block_tests::batch(vec![("datetime", tuple_frame_array(&[]).unwrap())], 0);
    for case in cases {
        check(case);
        if case.get("output").is_none() {
            continue;
        }
        let descriptor = imported(case);
        let before = descriptor.clone();
        let rows = descriptor.values().len();
        let data = block_tests::batch(
            vec![(
                "x",
                Arc::new(arrow_array::Float64Array::from_iter_values(
                    (0..rows).map(|i| f64::from(u32::try_from(i).unwrap())),
                )),
            )],
            rows,
        );
        let initial = IndexedFrame::from_multi_index(descriptor.clone(), data).unwrap();
        let mut current = initial.clone();
        for step in ["empty_append", "second_empty_append"] {
            let mut warnings = Vec::new();
            current = dataframe_append_with_warnings(&current, &right, &mut |w| {
                warnings.push(w.to_owned())
            })
            .unwrap();
            assert!(warnings.is_empty());
            assert_eq!(case[step]["warnings"], json!([]));
            let expected = &case[step]["output"]["index"];
            let IndexMetadata::Multi(actual) = current.index_metadata() else {
                panic!("lost multi identity")
            };
            assert_eq!(actual.as_ref(), &imported(expected));
            compare_levels(actual, expected);
            compare(&actual.values(), expected["values"].as_array().unwrap());
            assert_eq!(current.data(), initial.data());
            assert_eq!(current.blocks(), initial.blocks());
        }
        assert_eq!(descriptor, before);
        assert_eq!(
            initial.index_metadata(),
            &IndexMetadata::Multi(Arc::new(before))
        );
    }
}

fn compare_levels(result: &MultiIndexDescriptor, expected: &Value) {
    assert_eq!(
        result.levels().len(),
        expected["levels"].as_array().unwrap().len()
    );
    for ((((array, category), nullable), string), expected) in result
        .levels()
        .iter()
        .zip(result.categorical_levels())
        .zip(result.nullable_levels())
        .zip(result.string_levels())
        .zip(expected["levels"].as_array().unwrap())
    {
        assert_eq!(array.data_type(), level(expected).data_type());
        let values = if expected.get("string").is_some() {
            let wanted = super::super::string_append::tests::descriptor(expected);
            assert_eq!(string.as_ref().unwrap(), &wanted);
            assert!(category.is_none());
            assert!(nullable.is_none());
            wanted
                .values()
                .iter()
                .cloned()
                .map(|v| V::Scalar(T::Builtin(v)))
                .collect()
        } else if expected["kind"] == "CategoricalIndex" {
            assert!(nullable.is_none());
            let wanted = super::super::categorical_append_tests::descriptor(expected);
            assert_eq!(category.as_ref().unwrap(), &wanted);
            category.as_ref().unwrap().values()
        } else if expected.get("masked").is_some() {
            assert!(category.is_none());
            let wanted = NullableIndexDescriptor::new(level(expected)).unwrap();
            assert_eq!(nullable.as_ref().unwrap(), &wanted);
            assert_eq!(wanted.dtype_name(), expected["dtype"].as_str().unwrap());
            wanted.values().iter().cloned().map(V::Scalar).collect()
        } else {
            assert!(category.is_none());
            assert!(nullable.is_none());
            tuple_index::values(array).unwrap()
        };
        compare(&values, expected["values"].as_array().unwrap());
    }
}

#[test]
fn nullable_multi_level_append_failures_preserve_inputs() {
    use super::super::{append_with_operations, categorical_append_tests::Failure};
    let descriptor = MultiIndexDescriptor::from_levels(
        vec![MultiIndexLevel::Nullable(
            NullableIndexDescriptor::new(Arc::new(Int64Array::from(vec![Some(i64::MAX), None])))
                .unwrap(),
        )],
        vec![vec![0, 1, -1]],
        vec![V::Scalar(T::Builtin(B::None))],
        None,
    )
    .unwrap();
    let left = IndexedFrame::from_multi_index(descriptor, block_tests::batch(vec![], 3)).unwrap();
    let right = block_tests::batch(
        vec![(
            "datetime",
            tuple_frame_array(&[V::Tuple(vec![V::Scalar(T::Builtin(B::Int(7)))])]).unwrap(),
        )],
        1,
    );
    let before = (
        left.index().to_data(),
        left.index_metadata().clone(),
        left.data().clone(),
        right.clone(),
    );
    let mut baseline = Failure {
        at: usize::MAX,
        calls: 0,
    };
    let result = append_with_operations(&left, &right, &mut baseline, &mut |_| {
        panic!("unexpected warning")
    })
    .unwrap();
    assert_eq!(result.index().len(), 4);
    assert!(baseline.calls > 0);
    for at in 1..=baseline.calls {
        let mut backend = Failure { at, calls: 0 };
        let error = append_with_operations(&left, &right, &mut backend, &mut |_| {
            panic!("unexpected warning")
        })
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "Compute error: categorical backend failed"
        );
        assert_eq!(backend.calls, at);
        assert_eq!(
            (
                left.index().to_data(),
                left.index_metadata().clone(),
                left.data().clone(),
                right.clone()
            ),
            before
        );
    }
}

#[test]
fn categorical_multi_levels_match_source_and_survive_ignored_appends() {
    source_multi_levels(
        "dataframe_multi_category_levels.py",
        "71c30ab64504117be7aabe054b8a99db98f311be690194b2ef1bd787206c31c9",
        5112,
        1486,
    );
}

#[test]
fn categorical_level_integer_conversion_failures_are_atomic() {
    for masked in [false, true] {
        let array: ArrayRef = Arc::new(Int64Array::from(vec![9_007_199_254_740_993]));
        let category = if masked {
            CategoricalIndexDescriptor::from_nullable_categories(
                NullableIndexDescriptor::new(array).unwrap(),
                vec![0],
                false,
            )
        } else {
            CategoricalIndexDescriptor::new(array, vec![0], false)
        }
        .unwrap();
        let before = category.clone();
        for bad_type in [false, true] {
            let error = MultiIndexDescriptor::build_typed(
                vec![MultiIndexLevel::Categorical(category.clone())],
                vec![vec![0, -1]],
                vec![V::Scalar(T::Builtin(B::None))],
                None,
                &mut |input| {
                    assert_eq!(input.to_data(), category.storage().to_data());
                    if bad_type {
                        Ok(new_null_array(&DataType::Binary, 1))
                    } else {
                        Err(ArrowError::ComputeError(
                            "category level conversion failed".into(),
                        ))
                    }
                },
            )
            .unwrap_err();
            if bad_type {
                assert!(error.to_string().contains("Binary"));
            } else {
                assert_eq!(
                    error.to_string(),
                    "Compute error: category level conversion failed"
                );
            }
            assert_eq!(category, before);
        }
    }
}

#[test]
fn categorical_multi_level_boundaries_match_source() {
    source_multi_levels(
        "dataframe_multi_category_boundaries.py",
        "69f3be01f0a94a5a21037b1fd2b3bcc487d4857ca59e0bd32f7bbe9333effd82",
        11520,
        3888,
    );
}

#[test]
fn nullable_multi_levels_match_source_and_survive_ignored_appends() {
    source_multi_levels(
        "dataframe_multi_nullable_levels.py",
        "8911b6e86bf0a01e12a2ac585c931d353be671fa48b77d35c2769c774d65797c",
        3048,
        868,
    );
}

#[test]
fn nullable_level_identity_is_distinct_from_ordinary_buffers() {
    let array: ArrayRef = Arc::new(Int64Array::from(vec![i64::MAX]));
    let nullable = NullableIndexDescriptor::new(array.clone()).unwrap();
    let masked = MultiIndexDescriptor::from_levels(
        vec![MultiIndexLevel::Nullable(nullable.clone())],
        vec![vec![0]],
        vec![V::Scalar(T::Builtin(B::None))],
        None,
    )
    .unwrap();
    let ordinary = MultiIndexDescriptor::new(
        vec![array.clone()],
        vec![vec![0]],
        vec![V::Scalar(T::Builtin(B::None))],
        None,
    )
    .unwrap();
    assert_eq!(masked.codes(), ordinary.codes());
    assert_eq!(masked.levels()[0].to_data(), ordinary.levels()[0].to_data());
    assert_ne!(masked, ordinary);
    assert_eq!(masked.nullable_levels(), &[Some(nullable)]);
    let missing = MultiIndexDescriptor::build_typed(
        vec![MultiIndexLevel::Nullable(
            NullableIndexDescriptor::new(array.clone()).unwrap(),
        )],
        vec![vec![0, -1]],
        vec![V::Scalar(T::Builtin(B::None))],
        None,
        &mut |_| panic!("masked integers must not be converted to float"),
    )
    .unwrap();
    assert_eq!(
        missing.values(),
        vec![
            V::Tuple(vec![V::Scalar(T::Builtin(B::Int(i64::MAX)))]),
            V::Tuple(vec![V::Scalar(T::Builtin(B::PandasNa))]),
        ]
    );
    assert_eq!(missing.levels()[0].to_data(), array.to_data());
}

fn source_multi_levels(fixture: &str, digest: &str, total: usize, valid: usize) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures").join(fixture))
        .arg(root.join("../../../qlib/qlib/rl/order_execution/utils.py"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let contract: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(contract["digest"], digest);
    let cases = contract["cases"].as_array().unwrap();
    assert_eq!(cases.len(), total);
    assert_eq!(
        cases.iter().filter(|c| c.get("output").is_some()).count(),
        valid
    );
    let right = block_tests::batch(vec![("datetime", tuple_frame_array(&[]).unwrap())], 0);
    for case in cases {
        check(case);
        if case.get("output").is_none() {
            continue;
        }
        let descriptor = imported(case);
        let rows = descriptor.values().len();
        let initial = IndexedFrame::from_multi_index(
            descriptor,
            block_tests::batch(
                vec![(
                    "x",
                    Arc::new(arrow_array::Float64Array::from_iter_values(
                        (0..rows).map(|i| f64::from(u32::try_from(i).unwrap())),
                    )),
                )],
                rows,
            ),
        )
        .unwrap();
        let before = initial.index_metadata().clone();
        let mut current = initial.clone();
        for step in ["empty_append", "second_empty_append"] {
            current = dataframe_append_with_warnings(&current, &right, &mut |_| {
                panic!("unexpected warning")
            })
            .unwrap();
            let expected = &case[step]["output"]["index"];
            assert_eq!(case[step]["warnings"], json!([]));
            assert_eq!(
                current.index_metadata(),
                &IndexMetadata::Multi(Arc::new(imported(expected)))
            );
            compare(
                &tuple_index::values(current.index()).unwrap(),
                expected["values"].as_array().unwrap(),
            );
            assert_eq!(current.data(), initial.data());
            assert_eq!(current.index_name(), None);
        }
        assert_eq!(initial.index_metadata(), &before);
    }
}

#[test]
fn explicit_multi_index_levels_codes_sortorder_and_rows_match_source() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/dataframe_multi_levels.py"))
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
        "0503002cc9d640a66115c080c4e12d7ca8997be49ba2c64dc1266b3723e82a07"
    );
    let cases = contract["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 1539);
    assert_eq!(
        cases.iter().filter(|v| v.get("error").is_some()).count(),
        810
    );
    for case in cases {
        check(case);
    }
}

#[test]
fn malformed_levels_names_and_conversion_failures_never_publish_a_descriptor() {
    let none = V::Scalar(T::Builtin(B::None));
    let native: ArrayRef = Arc::new(Int64Array::from(vec![1]));
    let before = native.to_data();
    let invalid_name = V::Scalar(T::Timestamp {
        ticks: i64::MIN,
        unit: arrow_schema::TimeUnit::Second,
        timezone: None,
    });
    let malformed_cells = vec![invalid_name.clone()];
    let error = unique_level(0, malformed_cells.clone()).unwrap_err();
    assert_eq!(
        error.to_string(),
        "Invalid argument error: temporal object uses reserved NaT ticks"
    );
    assert_eq!(malformed_cells, vec![invalid_name.clone()]);
    assert!(matches!(
        MultiIndexDescriptor::new(
            vec![native.clone()],
            vec![vec![0]],
            vec![invalid_name],
            None
        ),
        Err(MultiIndexError::Arrow(_))
    ));
    for array in [
        new_null_array(&super::super::tuple_frame_dtype(), 1),
        new_null_array(&DataType::Float16, 1),
    ] {
        assert!(
            MultiIndexDescriptor::new(vec![array], vec![vec![0]], vec![none.clone()], None)
                .is_err()
        );
    }
    let error = MultiIndexDescriptor::build(
        vec![native.clone()],
        vec![vec![-1, 0]],
        vec![none.clone()],
        None,
        &mut |_| Err(ArrowError::ComputeError("level conversion failed".into())),
    )
    .unwrap_err();
    assert_eq!(error.to_string(), "Compute error: level conversion failed");
    let error = MultiIndexDescriptor::build(
        vec![native.clone()],
        vec![vec![-1, 0]],
        vec![none],
        None,
        &mut |_| Ok(new_null_array(&DataType::Binary, 1)),
    )
    .unwrap_err();
    assert!(error.to_string().contains("Binary"));
    assert_eq!(native.to_data(), before);
}

#[test]
fn descriptor_structural_equality_and_corrupt_materialization_are_explicit() {
    let descriptor = MultiIndexDescriptor::new(
        vec![Arc::new(Int64Array::from(vec![1]))],
        vec![vec![0]],
        vec![V::Scalar(T::Builtin(B::None))],
        None,
    )
    .unwrap();
    let categories = Arc::new(Int64Array::from(vec![1])) as ArrayRef;
    let category =
        |ordered| CategoricalIndexDescriptor::new(categories.clone(), vec![0], ordered).unwrap();
    let categorical = |ordered| {
        MultiIndexDescriptor::from_levels(
            vec![MultiIndexLevel::Categorical(category(ordered))],
            vec![vec![0]],
            vec![V::Scalar(T::Builtin(B::None))],
            None,
        )
        .unwrap()
    };
    assert_ne!(categorical(false), categorical(true));
    assert_eq!(categorical(true).clone(), categorical(true));
    assert!(
        MultiIndexDescriptor::new(
            vec![category(true).storage()],
            vec![vec![0]],
            vec![V::Scalar(T::Builtin(B::None))],
            None
        )
        .is_err()
    );
    // Exercise each structural field independently, including private malformed
    // states. This is invariant defense, not public-constructor reachability.
    let mut changed = descriptor.clone();
    changed.codes[0][0] = None;
    assert_ne!(descriptor, changed);
    changed = descriptor.clone();
    changed.sortorder = Some(0);
    assert_ne!(descriptor, changed);
    changed = descriptor.clone();
    changed.names.clear();
    assert_ne!(descriptor, changed);
    changed = descriptor.clone();
    changed.names[0] = V::Tuple(vec![]);
    assert_ne!(descriptor, changed);
    changed = descriptor.clone();
    changed.levels.clear();
    assert_ne!(descriptor, changed);
    changed = descriptor.clone();
    changed.levels[0] = Arc::new(Int64Array::from(vec![2]));
    assert_ne!(descriptor, changed);
    changed = descriptor.clone();
    changed.materialized[0][0] = V::Scalar(T::Duration {
        ticks: i64::MIN,
        unit: arrow_schema::TimeUnit::Second,
    });
    let error = super::super::IndexedFrame::from_multi_index(
        changed.clone(),
        super::super::block_tests::batch(vec![], 1),
    )
    .unwrap_err();
    assert!(error.to_string().contains("reserved NaT ticks"));
    let metadata = super::super::IndexMetadata::Multi(Arc::new(changed));
    assert!(
        metadata
            .validate(&super::super::tuple_frame_array(&descriptor.values()).unwrap())
            .unwrap_err()
            .to_string()
            .contains("reserved NaT ticks")
    );
    assert!(
        MultiIndexDescriptor::empty_tuple_rows(&[], new_null_array(&DataType::Binary, 1)).is_err()
    );
}
