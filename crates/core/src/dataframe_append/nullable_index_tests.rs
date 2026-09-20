use super::super::{temporal_append_tests, temporal_objects::tests::compare_cells};
use super::*;
use arrow_array::{
    Array, BooleanArray, Float64Array, Int64Array, RecordBatch, UInt64Array, new_null_array,
};
use arrow_ipc::{reader::StreamReader, writer::StreamWriter};
use arrow_schema::Schema;
use serde_json::Value;
use std::{io::Cursor, path::PathBuf, process::Command, sync::Arc};

pub(in crate::dataframe_append) fn array(case: &Value) -> ArrayRef {
    let name = case["dtype"].as_str().unwrap();
    let cells = case["physical"].as_array().unwrap();
    let wide: ArrayRef = if name.starts_with("uint") {
        Arc::new(UInt64Array::from(
            cells
                .iter()
                .map(|v| v[1].as_str().unwrap().parse::<u64>().unwrap())
                .collect::<Vec<_>>(),
        ))
    } else if name.starts_with("int") {
        Arc::new(Int64Array::from(
            cells
                .iter()
                .map(|v| v[1].as_str().unwrap().parse::<i64>().unwrap())
                .collect::<Vec<_>>(),
        ))
    } else {
        temporal_append_tests::column(name, &case["physical"], 0)
    };
    let dtype = match name {
        "int8" => DataType::Int8,
        "int16" => DataType::Int16,
        "int32" => DataType::Int32,
        "int64" => DataType::Int64,
        "uint8" => DataType::UInt8,
        "uint16" => DataType::UInt16,
        "uint32" => DataType::UInt32,
        "uint64" => DataType::UInt64,
        "float32" => DataType::Float32,
        "float64" => DataType::Float64,
        "bool" => DataType::Boolean,
        _ => panic!("unknown fixture dtype"),
    };
    let physical = arrow_cast::cast(&wide, &dtype).unwrap();
    let validity = BooleanArray::from(
        case["mask"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| !v.as_bool().unwrap())
            .collect::<Vec<_>>(),
    );
    arrow_array::make_array(
        physical
            .to_data()
            .into_builder()
            .nulls(Some(validity.values().clone().into()))
            .build()
            .unwrap(),
    )
}

#[test]
fn masked_numeric_storage_matches_source_and_arrow_roundtrips() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/dataframe_nullable_storage.py"))
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
        "0bcfdbf82799531a7d8a6ce50af03b4ee795f84962628c33f2b74a612307586c"
    );
    let cases = contract["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 271);
    for case in cases {
        let array = array(case);
        let before = array.to_data();
        let descriptor = NullableIndexDescriptor::new(array.clone()).unwrap();
        assert_eq!(
            descriptor.dtype_name(),
            case["output"]["dtype"].as_str().unwrap()
        );
        compare_cells(
            descriptor.values(),
            case["output"]["values"].as_array().unwrap(),
        );
        for (i, missing) in case["missing"].as_array().unwrap().iter().enumerate() {
            assert_eq!(descriptor.storage().is_null(i), missing.as_bool().unwrap());
        }
        assert_eq!(descriptor, descriptor.clone());
        assert_eq!(descriptor.storage().to_data(), before);
        let field = descriptor.field("datetime");
        assert_eq!(
            NullableIndexDescriptor::from_storage(&array, &field).unwrap(),
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
            NullableIndexDescriptor::from_storage(read.column(0), read.schema().field(0)).unwrap();
        compare_cells(
            imported.values(),
            case["output"]["values"].as_array().unwrap(),
        );
        assert_eq!(imported.dtype_name(), descriptor.dtype_name());
        for offset in 0..=array.len() {
            let sliced = array.slice(offset, array.len() - offset);
            let imported = NullableIndexDescriptor::from_storage(&sliced, &field).unwrap();
            compare_cells(
                imported.values(),
                &case["output"]["values"].as_array().unwrap()[offset..],
            );
            assert_eq!(imported.storage().to_data(), sliced.to_data());
        }
        assert_eq!(array.to_data(), before);
        let order = (0..array.len())
            .rev()
            .map(|i| i64::try_from(i).unwrap())
            .collect::<Vec<_>>();
        let reordered =
            arrow_select::take::take(array.as_ref(), &Int64Array::from(order), None).unwrap();
        let reordered = NullableIndexDescriptor::from_storage(&reordered, &field).unwrap();
        let expected = case["output"]["values"]
            .as_array()
            .unwrap()
            .iter()
            .rev()
            .cloned()
            .collect::<Vec<_>>();
        compare_cells(reordered.values(), &expected);
    }
}

#[test]
fn masked_float_storage_keeps_nan_payloads_and_signed_zero() {
    for bits in [
        0_u64,
        1_u64 << 63,
        0x7ff8_0000_0000_0001,
        0x7ff8_0000_0000_0002,
    ] {
        let array: ArrayRef = Arc::new(Float64Array::from(vec![f64::from_bits(bits)]));
        let descriptor = NullableIndexDescriptor::new(array.clone()).unwrap();
        assert_eq!(descriptor.storage().to_data(), array.to_data());
        let T::Builtin(B::Float(value)) = descriptor.values()[0] else {
            panic!("float identity lost")
        };
        assert_eq!(value.to_bits(), bits);
        let other: ArrayRef = Arc::new(Float64Array::from(vec![f64::from_bits(bits ^ 1)]));
        assert_ne!(descriptor, NullableIndexDescriptor::new(other).unwrap());
    }
}

#[test]
fn masked_storage_validation_and_decoder_failures_are_atomic() {
    let valid: ArrayRef = Arc::new(Float64Array::from(vec![Some(f64::NAN), None, Some(-0.)]));
    let before = valid.to_data();
    let descriptor = NullableIndexDescriptor::new(valid.clone()).unwrap();
    let field = descriptor.field("datetime");
    for (array, field, message) in [
        (
            valid.clone(),
            Field::new("datetime", DataType::Int64, true),
            "field dtype does not match array",
        ),
        (
            valid.clone(),
            field.clone().with_nullable(false),
            "non-nullable field contains nulls",
        ),
        (
            valid.clone(),
            Field::new("datetime", DataType::Float64, true),
            "missing masked dtype metadata",
        ),
        (
            valid.clone(),
            field
                .clone()
                .with_metadata([(DTYPE_KEY.into(), "Int64".into())].into()),
            "masked dtype metadata does not match array",
        ),
    ] {
        assert_eq!(
            NullableIndexDescriptor::from_storage(&array, &field)
                .unwrap_err()
                .to_string(),
            format!("invalid masked index storage: {message}")
        );
    }
    let unsupported = new_null_array(&DataType::Utf8, 0);
    assert!(matches!(
        NullableIndexDescriptor::new(unsupported.clone()),
        Err(NullableIndexError::Unsupported(DataType::Utf8))
    ));
    let field = Field::new("datetime", DataType::Utf8, true)
        .with_metadata([(DTYPE_KEY.into(), "string".into())].into());
    assert!(matches!(
        NullableIndexDescriptor::from_storage(&unsupported, &field),
        Err(NullableIndexError::Unsupported(_))
    ));
    assert_eq!(
        NullableIndexDescriptor::build(valid.clone(), &mut |_| Err(ArrowError::ComputeError(
            "decode failed".into()
        )))
        .unwrap_err()
        .to_string(),
        "Compute error: decode failed"
    );
    assert_eq!(
        NullableIndexDescriptor::build(valid.clone(), &mut |_| Ok(vec![]))
            .unwrap_err()
            .to_string(),
        "invalid masked index storage: decoder changed row count"
    );
    assert_eq!(valid.to_data(), before);
    let nonnull: ArrayRef = Arc::new(Float64Array::from(vec![f64::NAN]));
    let typed = NullableIndexDescriptor::new(nonnull.clone()).unwrap();
    assert!(
        NullableIndexDescriptor::from_storage(&nonnull, &typed.field("x").with_nullable(false))
            .is_ok()
    );
    assert_ne!(descriptor, typed);
}

#[test]
fn corrupt_masked_scalar_cache_cannot_publish_category_codes() {
    use super::super::{ArrowOperations, CategoricalIndexDescriptor, IndexMetadata};
    let storage: ArrayRef = Arc::new(arrow_array::Int64Array::from(vec![1]));
    let nullable = NullableIndexDescriptor::build(storage.clone(), &mut |_| {
        Ok(vec![T::Timestamp {
            ticks: i64::MIN,
            unit: arrow_schema::TimeUnit::Second,
            timezone: None,
        }])
    })
    .unwrap();
    let category = CategoricalIndexDescriptor::new(storage.clone(), vec![0], false).unwrap();
    assert_eq!(
        CategoricalIndexDescriptor::from_nullable_categories(nullable.clone(), vec![0], false)
            .unwrap_err()
            .to_string(),
        "Invalid argument error: temporal object uses reserved NaT ticks"
    );
    let left = category.storage();
    let before = (left.to_data(), storage.to_data());
    let result = super::super::nullable_append::join(
        &left,
        &IndexMetadata::Categorical(Arc::new(category)),
        &storage,
        &IndexMetadata::Nullable(Arc::new(nullable)),
        &mut ArrowOperations,
        &mut |_| panic!("invalid cached values must not warn"),
    );
    assert_eq!(
        result.unwrap_err().to_string(),
        "Invalid argument error: temporal object uses reserved NaT ticks"
    );
    assert_eq!((left.to_data(), storage.to_data()), before);
}
