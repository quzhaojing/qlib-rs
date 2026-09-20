use super::super::{builtin_frame_array, builtin_object_tests::compare};
use super::*;
use crate::RlCheckpointText;
use arrow_array::{Int64Array, RecordBatch, new_null_array};
use arrow_ipc::{reader::StreamReader, writer::StreamWriter};
use arrow_schema::{DataType, Schema};
use serde_json::{Value, json};
use std::{io::Cursor, path::PathBuf, process::Command, sync::Arc};

fn array(values: &Value) -> ArrayRef {
    builtin_frame_array(
        &values
            .as_array()
            .unwrap()
            .iter()
            .map(|value| {
                if value[0] == "none" {
                    V::None
                } else {
                    assert_eq!(value[0], "str");
                    V::Text(
                        RlCheckpointText::try_from_code_points(
                            value[1]
                                .as_array()
                                .unwrap()
                                .iter()
                                .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap()),
                        )
                        .unwrap(),
                    )
                }
            })
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

#[test]
fn string_storage_matches_source_and_survives_arrow_interchange() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/dataframe_string_storage.py"))
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
        "823538a19ac7e71214c7cb3b50cddb38b9b2b56350e61111b69bc084a45c132b"
    );
    let cases = contract["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 48);
    assert_eq!(cases.iter().filter(|c| c.get("error").is_some()).count(), 6);
    for case in cases {
        let array = array(&case["input"]);
        let before = array.to_data();
        let storage = if case["storage"] == "python" {
            StringStorage::Python
        } else {
            StringStorage::PyArrow
        };
        let missing = if case["missing"] == "NA" {
            StringMissing::PandasNa
        } else {
            StringMissing::NaN
        };
        let result = StringIndexDescriptor::new(array.clone(), storage, missing);
        assert_eq!(case["warnings"], json!([]));
        if case.get("error").is_some() {
            assert_eq!(case["error"], "UnicodeEncodeError");
            let error = result.unwrap_err();
            assert_eq!(error.to_string(), case["message"].as_str().unwrap());
            assert!(matches!(error, StringIndexError::Unicode { row: 0, .. }));
            assert_eq!(array.to_data(), before);
            continue;
        }
        let descriptor = result.unwrap();
        assert_eq!(descriptor.dtype_name(), case["output"]["dtype"]);
        assert_eq!(descriptor.storage_kind(), storage);
        assert_eq!(descriptor.missing_kind(), missing);
        compare(descriptor.values(), &case["output"]["values"]);
        for (value, masked) in descriptor
            .values()
            .iter()
            .zip(case["mask"].as_array().unwrap())
        {
            assert_eq!(!matches!(value, V::Text(_)), masked.as_bool().unwrap());
        }
        assert_eq!(descriptor, descriptor.clone());
        assert_eq!(descriptor.storage().to_data(), before);
        let field = descriptor.field("datetime");
        assert_eq!(
            StringIndexDescriptor::from_storage(&array, &field).unwrap(),
            descriptor
        );
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![field.clone()])),
            vec![array.clone()],
        )
        .unwrap();
        let mut bytes = vec![];
        {
            let mut writer = StreamWriter::try_new(&mut bytes, &batch.schema()).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
        }
        let read = StreamReader::try_new(Cursor::new(bytes), None)
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        let imported =
            StringIndexDescriptor::from_storage(read.column(0), read.schema().field(0)).unwrap();
        assert_eq!(imported, descriptor);
        compare(imported.values(), &case["output"]["values"]);
        for offset in 0..=array.len() {
            let sliced = array.slice(offset, array.len() - offset);
            let imported = StringIndexDescriptor::from_storage(&sliced, &field).unwrap();
            compare(imported.values(), &case["slices"][offset]["values"]);
            assert_eq!(imported.storage().to_data(), sliced.to_data());
        }
        let order = Int64Array::from(
            (0..array.len())
                .rev()
                .map(|v| i64::try_from(v).unwrap())
                .collect::<Vec<_>>(),
        );
        let reordered = arrow_select::take::take(array.as_ref(), &order, None).unwrap();
        let imported = StringIndexDescriptor::from_storage(&reordered, &field).unwrap();
        compare(imported.values(), &case["reversed"]["values"]);
        assert_eq!(array.to_data(), before);
    }
}

#[test]
fn string_storage_identity_and_failures_are_explicit_and_atomic() {
    let array = builtin_frame_array(&[V::Text(RlCheckpointText::from_utf8("a")), V::None]).unwrap();
    let before = array.to_data();
    let descriptor = StringIndexDescriptor::new(
        array.clone(),
        StringStorage::Python,
        StringMissing::PandasNa,
    )
    .unwrap();
    let field = descriptor.field("datetime");
    for (candidate, field, expected) in [
        (
            array.clone(),
            Field::new("datetime", DataType::Int64, true),
            "field dtype does not match array",
        ),
        (
            array.clone(),
            Field::new("datetime", array.data_type().clone(), true),
            "missing string dtype metadata",
        ),
        (
            array.clone(),
            field
                .clone()
                .with_metadata([(DTYPE_KEY.into(), "python".into())].into()),
            "unknown string dtype metadata",
        ),
        (
            array.clone(),
            field.clone().with_nullable(false),
            "non-nullable field contains missing cells",
        ),
    ] {
        assert_eq!(
            StringIndexDescriptor::from_storage(&candidate, &field)
                .unwrap_err()
                .to_string(),
            format!("invalid string index storage: {expected}")
        );
    }
    for missing in [StringMissing::PandasNa, StringMissing::NaN] {
        let descriptor =
            StringIndexDescriptor::new(array.clone(), StringStorage::Python, missing).unwrap();
        assert!(
            StringIndexDescriptor::from_storage(
                &array,
                &descriptor.field("x").with_nullable(false)
            )
            .is_err()
        );
        let nonnull = array.slice(0, 1);
        assert!(
            StringIndexDescriptor::from_storage(
                &nonnull,
                &descriptor.field("x").with_nullable(false)
            )
            .is_ok()
        );
    }
    for value in [V::Int(1), V::PandasNa, V::Float(f64::NAN), V::NotATime] {
        let invalid = builtin_frame_array(&[value]).unwrap();
        assert_eq!(
            StringIndexDescriptor::from_storage(&invalid, &field)
                .unwrap_err()
                .to_string(),
            "invalid string index storage: expected text or None cell"
        );
    }
    assert_eq!(array.to_data(), before);
}

#[test]
fn string_storage_decoder_failures_and_distinct_identities_are_explicit() {
    let array = builtin_frame_array(&[V::Text(RlCheckpointText::from_utf8("a")), V::None]).unwrap();
    let before = array.to_data();
    let descriptor = StringIndexDescriptor::new(
        array.clone(),
        StringStorage::Python,
        StringMissing::PandasNa,
    )
    .unwrap();
    let unsupported = new_null_array(&DataType::Int64, 1);
    assert!(matches!(
        StringIndexDescriptor::new(unsupported, StringStorage::Python, StringMissing::NaN),
        Err(StringIndexError::Arrow(_))
    ));
    assert_eq!(
        StringIndexDescriptor::build(
            array.clone(),
            StringStorage::Python,
            StringMissing::NaN,
            &mut |_| Err(ArrowError::ComputeError("string decode failed".into()))
        )
        .unwrap_err()
        .to_string(),
        "Compute error: string decode failed"
    );
    assert_eq!(
        StringIndexDescriptor::build(
            array.clone(),
            StringStorage::Python,
            StringMissing::NaN,
            &mut |_| Ok(vec![])
        )
        .unwrap_err()
        .to_string(),
        "invalid string index storage: decoder changed row count"
    );
    assert_ne!(
        descriptor,
        StringIndexDescriptor::new(
            array.clone(),
            StringStorage::PyArrow,
            StringMissing::PandasNa
        )
        .unwrap()
    );
    assert_ne!(
        descriptor,
        StringIndexDescriptor::new(array.clone(), StringStorage::Python, StringMissing::NaN)
            .unwrap()
    );
    assert_ne!(
        descriptor,
        StringIndexDescriptor::new(
            array.slice(0, 1),
            StringStorage::Python,
            StringMissing::PandasNa
        )
        .unwrap()
    );
    assert_eq!(array.to_data(), before);
}
