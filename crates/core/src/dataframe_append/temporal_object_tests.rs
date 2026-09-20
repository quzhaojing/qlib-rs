use super::*;
use crate::dataframe_append::{builtin_object_tests::compare, inference::tests::atom};
use arrow_array::{RecordBatch, new_null_array};
use arrow_ipc::{reader::StreamReader, writer::StreamWriter};
use arrow_schema::Schema;
use serde_json::{Value, json};
use std::{io::Cursor, path::PathBuf, process::Command};

pub(crate) fn decode(value: &Value) -> TemporalFrameValue {
    if !matches!(value[0].as_str().unwrap(), "Timestamp" | "Timedelta") {
        return TemporalFrameValue::Builtin(atom(value));
    }
    let unit = match value[1].as_str().unwrap().split('[').nth(1).unwrap() {
        "s]" => TimeUnit::Second,
        "ms]" => TimeUnit::Millisecond,
        "us]" => TimeUnit::Microsecond,
        "ns]" => TimeUnit::Nanosecond,
        _ => panic!("unknown unit"),
    };
    let ticks = value[2].as_str().unwrap().parse().unwrap();
    if value[0] == "Timestamp" {
        TemporalFrameValue::Timestamp {
            ticks,
            unit,
            timezone: value[3]
                .as_str()
                .filter(|zone| *zone != "None")
                .map(Into::into),
        }
    } else {
        TemporalFrameValue::Duration { ticks, unit }
    }
}

pub(crate) fn compare_cells(values: &[TemporalFrameValue], expected: &[Value]) {
    assert_eq!(values.len(), expected.len());
    for (value, expected) in values.iter().zip(expected) {
        if let TemporalFrameValue::Builtin(value) = value {
            compare(std::slice::from_ref(value), &json!([expected]));
        } else {
            assert_eq!(value, &decode(expected));
        }
    }
}

#[test]
fn temporal_objects_preserve_source_units_zones_boundaries_and_arrow_roundtrips() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/dataframe_temporal_objects.py"))
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
        "c97370b457cbef062dac735906592af4281236ed655c667481803f7f02c9f526"
    );
    let expected = contract["values"].as_array().unwrap();
    assert_eq!(expected.len(), 61);
    let values = expected.iter().map(decode).collect::<Vec<_>>();
    let built = temporal_frame_array(&values).unwrap();
    compare_cells(&temporal_frame_values(&built).unwrap(), expected);
    check_empty_append(&built, expected);
    compare_cells(
        &temporal_frame_values(&built.slice(12, 30)).unwrap(),
        &expected[12..42],
    );
    let joined =
        arrow_select::concat::concat(&[built.slice(0, 17).as_ref(), built.slice(17, 44).as_ref()])
            .unwrap();
    compare_cells(&temporal_frame_values(&joined).unwrap(), expected);
    let indices = arrow_array::UInt32Array::from(vec![60, 1, 60, 13, 12]);
    let taken = arrow_select::take::take(built.as_ref(), &indices, None).unwrap();
    compare_cells(
        &temporal_frame_values(&taken).unwrap(),
        &[
            expected[60].clone(),
            expected[1].clone(),
            expected[60].clone(),
            expected[13].clone(),
            expected[12].clone(),
        ],
    );
    let schema = Arc::new(Schema::new(vec![Field::new(
        "x",
        temporal_frame_dtype(),
        true,
    )]));
    let batch = RecordBatch::try_new(schema.clone(), vec![built]).unwrap();
    let mut bytes = vec![];
    let mut writer = StreamWriter::try_new(&mut bytes, &schema).unwrap();
    writer.write(&batch).unwrap();
    writer.finish().unwrap();
    let decoded = StreamReader::try_new(Cursor::new(bytes), None)
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    assert_eq!(decoded.schema(), schema);
    compare_cells(&temporal_frame_values(decoded.column(0)).unwrap(), expected);
    let empty = temporal_frame_array(&[]).unwrap();
    assert!(temporal_frame_values(&empty).unwrap().is_empty());
    assert_eq!(empty.data_type(), &temporal_frame_dtype());
    assert_ne!(temporal_frame_dtype(), builtin_frame_dtype());
}

fn check_empty_append(array: &ArrayRef, expected: &[Value]) {
    use crate::dataframe_append::{
        FrameColumnInput, IndexedFrame, dataframe_append_with_warnings, frame_from_columns,
    };
    use indexmap::IndexMap;
    let empty = IndexedFrame::new(
        Arc::new(Int64Array::from(Vec::<i64>::new())),
        None,
        frame_from_columns(&IndexMap::new()).unwrap(),
    )
    .unwrap();
    let index = Arc::new(Int64Array::from_iter_values(
        (0..array.len()).map(|i| i64::try_from(i).unwrap()),
    )) as ArrayRef;
    let other = frame_from_columns(&IndexMap::from([
        ("datetime".into(), FrameColumnInput::Array(index.clone())),
        ("x".into(), FrameColumnInput::Array(array.clone())),
    ]))
    .unwrap();
    let mut warnings = vec![];
    let result = dataframe_append_with_warnings(&empty, &other, &mut |message| {
        warnings.push(message.to_owned());
    })
    .unwrap();
    compare_cells(
        &temporal_frame_values(result.data().column(0)).unwrap(),
        expected,
    );
    assert_eq!(result.index().to_data(), index.to_data());
    assert_eq!(result.index_name(), Some("datetime"));
    assert_eq!(result.blocks(), &[vec![0]]);
    assert!(warnings.is_empty());
}

#[test]
fn temporal_wrapper_preserves_builtin_float_payload_bits() {
    for bits in [
        0_u64,
        1_u64 << 63,
        0x7ff8_0000_0000_1234,
        0xfff8_0000_0000_4321,
    ] {
        let value = TemporalFrameValue::Builtin(BuiltinFrameValue::Float(f64::from_bits(bits)));
        let built = temporal_frame_array(&[value]).unwrap();
        let decoded = temporal_frame_values(&built).unwrap();
        let TemporalFrameValue::Builtin(BuiltinFrameValue::Float(value)) = decoded[0] else {
            panic!("not a float")
        };
        assert_eq!(value.to_bits(), bits);
    }
}

#[test]
fn nested_builtin_encoding_failure_prevents_partial_publication() {
    let values = [
        TemporalFrameValue::Builtin(BuiltinFrameValue::Int(7)),
        TemporalFrameValue::Timestamp {
            ticks: 42,
            unit: TimeUnit::Second,
            timezone: Some("UTC".into()),
        },
    ];
    let before = values.clone();
    let mut calls = 0;
    let error = encode_with(&values, &mut |builtin| {
        calls += 1;
        assert_eq!(
            builtin,
            &[BuiltinFrameValue::Int(7), BuiltinFrameValue::None]
        );
        Err(ArrowError::ComputeError("nested encoding failed".into()))
    })
    .unwrap_err();
    assert_eq!(error.to_string(), "Compute error: nested encoding failed");
    assert_eq!(calls, 1);
    assert_eq!(values, before);
    let error = encode_with(&values, &mut |_| builtin_frame_array(&[])).unwrap_err();
    assert_eq!(
        error.to_string(),
        "Invalid argument error: Sparse union child arrays must be equal in length to the length of the union"
    );
    assert_eq!(values, before);
}

fn wrap(id: i8, child: ArrayRef) -> ArrayRef {
    let mut children = fields()
        .iter()
        .map(|(_, field)| new_null_array(field.data_type(), 1))
        .collect::<Vec<_>>();
    children[usize::try_from(id).unwrap()] = child;
    Arc::new(UnionArray::try_new(fields(), vec![id].into(), None, children).unwrap())
}

fn bad_temporal(ticks: Option<i64>, unit: Option<u8>, timestamp: bool) -> ArrayRef {
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(vec![ticks])),
        Arc::new(UInt8Array::from(vec![unit])),
    ];
    if timestamp {
        columns.push(Arc::new(StringArray::from(vec![Some("UTC")])));
    }
    Arc::new(StructArray::new(temporal_fields(timestamp), columns, None))
}

#[test]
fn temporal_objects_reject_ambiguous_payloads_without_changing_existing_schemas() {
    for value in [
        TemporalFrameValue::Timestamp {
            ticks: i64::MIN,
            unit: TimeUnit::Second,
            timezone: None,
        },
        TemporalFrameValue::Duration {
            ticks: i64::MIN,
            unit: TimeUnit::Second,
        },
    ] {
        assert_eq!(
            temporal_frame_array(&[value]).unwrap_err().to_string(),
            "Invalid argument error: temporal object uses reserved NaT ticks"
        );
    }
    let foreign = Arc::new(Int64Array::from(vec![1])) as ArrayRef;
    assert_eq!(
        temporal_frame_values(&foreign).unwrap_err().to_string(),
        "Invalid argument error: expected temporal object schema"
    );
    for timestamp in [false, true] {
        let id = if timestamp { 1 } else { 2 };
        let missing_parent = new_null_array(&DataType::Struct(temporal_fields(timestamp)), 1);
        for child in [
            missing_parent,
            bad_temporal(None, Some(0), timestamp),
            bad_temporal(Some(1), None, timestamp),
        ] {
            assert_eq!(
                temporal_frame_values(&wrap(id, child))
                    .unwrap_err()
                    .to_string(),
                "Invalid argument error: temporal object has null active fields"
            );
        }
        assert_eq!(
            temporal_frame_values(&wrap(id, bad_temporal(Some(i64::MIN), Some(0), timestamp)))
                .unwrap_err()
                .to_string(),
            "Invalid argument error: temporal object uses reserved NaT ticks"
        );
        assert_eq!(
            temporal_frame_values(&wrap(id, bad_temporal(Some(1), Some(4), timestamp)))
                .unwrap_err()
                .to_string(),
            "Invalid argument error: temporal object has unknown unit code"
        );
    }
    let DataType::Union(builtin_fields, _) = builtin_frame_dtype() else {
        unreachable!()
    };
    let children = builtin_fields
        .iter()
        .map(|(_, field)| new_null_array(field.data_type(), 1))
        .collect();
    let invalid_builtin =
        Arc::new(UnionArray::try_new(builtin_fields, vec![0].into(), None, children).unwrap());
    assert_eq!(
        temporal_frame_values(&wrap(0, invalid_builtin))
            .unwrap_err()
            .to_string(),
        "Invalid argument error: typed null is not an explicit object sentinel"
    );
}
