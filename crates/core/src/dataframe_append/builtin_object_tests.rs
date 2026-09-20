use super::{
    BuiltinFrameValue as V, builtin_frame_array, builtin_frame_dtype, builtin_frame_values,
};
use crate::RlCheckpointText;
use arrow_array::builder::{ListBuilder, UInt32Builder};
use arrow_array::{Array, ArrayRef, RecordBatch, UnionArray, new_null_array};
use arrow_ipc::{reader::StreamReader, writer::StreamWriter};
use arrow_schema::{DataType, Field, Schema};
use serde_json::{Value, json};
use std::{io::Cursor, path::PathBuf, process::Command, sync::Arc};

pub(super) fn compare(actual: &[V], expected: &Value) {
    assert_eq!(actual.len(), expected.as_array().unwrap().len());
    for (value, expected) in actual.iter().zip(expected.as_array().unwrap()) {
        let encoded = match value {
            V::None => json!(["none"]),
            V::PandasNa => json!(["pd.NA"]),
            V::NotATime => json!(["NaT"]),
            V::Bool(value) => json!(["bool", value]),
            V::Int(value) => json!(["int", value.to_string()]),
            V::UInt(value) => json!(["int", value.to_string()]),
            V::Text(value) => json!(["str", value.as_code_points()]),
            V::Float(value) => {
                assert_eq!(expected[0], "float");
                let wanted = super::block_tests::expected_float(expected[1].as_str().unwrap());
                if wanted.is_nan() {
                    assert!(value.is_nan());
                } else {
                    assert_eq!(value.to_bits(), wanted.to_bits());
                }
                continue;
            }
        };
        assert_eq!(encoded, *expected);
    }
}

#[test]
fn builtin_payloads_match_source_and_survive_arrow_operations() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/dataframe_missing_contract.py"))
        .arg(root.join("../../../qlib/qlib/rl/order_execution/utils.py"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let contract: Value = serde_json::from_slice(&output.stdout).unwrap();
    let nan = V::Float(f64::NAN);
    let text = |value| V::Text(RlCheckpointText::from_utf8(value));
    let samples = [
        ("empty", vec![]),
        ("none", vec![V::None, V::None]),
        ("nan", vec![nan.clone(), nan.clone()]),
        ("none_nan", vec![V::None, nan.clone()]),
        ("nan_none", vec![nan.clone(), V::None]),
        ("na", vec![V::PandasNa, V::PandasNa]),
        ("nat", vec![V::NotATime, V::NotATime]),
        ("none_na", vec![V::None, V::PandasNa]),
        ("na_none", vec![V::PandasNa, V::None]),
        ("none_text", vec![V::None, text("x")]),
        (
            "text",
            vec![
                text(""),
                V::Text(RlCheckpointText::try_from_code_points([20013, 25991, 55296]).unwrap()),
            ],
        ),
        ("finite_float", vec![V::Float(0.0), V::Float(-0.0)]),
        ("integer", vec![V::Int(i64::MIN), V::UInt(u64::MAX)]),
        ("boolean", vec![V::Bool(false), V::Bool(true)]),
    ];
    let mut arrays = vec![];
    let mut expected = vec![];
    for (name, values) in samples {
        let array = builtin_frame_array(&values).unwrap();
        assert_eq!(array.data_type(), &builtin_frame_dtype());
        let source = &contract["inputs"][format!("object_{name}")]["values"][0];
        compare(&builtin_frame_values(&array).unwrap(), source);
        expected.extend(source.as_array().unwrap().iter().cloned());
        arrays.push(array);
    }
    let joined =
        arrow_select::concat::concat(&arrays.iter().map(AsRef::as_ref).collect::<Vec<_>>())
            .unwrap();
    compare(&builtin_frame_values(&joined).unwrap(), &json!(expected));
    compare(
        &builtin_frame_values(&joined.slice(1, 7)).unwrap(),
        &json!(&expected[1..8]),
    );
    let schema = Arc::new(Schema::new(vec![Field::new(
        "object",
        builtin_frame_dtype(),
        true,
    )]));
    let batch = RecordBatch::try_new(schema.clone(), vec![joined]).unwrap();
    let mut bytes = vec![];
    let mut writer = StreamWriter::try_new(&mut bytes, &schema).unwrap();
    writer.write(&batch).unwrap();
    writer.finish().unwrap();
    let decoded = StreamReader::try_new(Cursor::new(bytes), None)
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    compare(
        &builtin_frame_values(decoded.column(0)).unwrap(),
        &json!(expected),
    );
    assert_eq!(decoded.schema(), schema);
    assert_ne!(super::numeric_object_dtype(), builtin_frame_dtype());
}

fn malformed(id: i8, text: Option<ArrayRef>) -> ArrayRef {
    let DataType::Union(fields, _) = builtin_frame_dtype() else {
        unreachable!()
    };
    let mut children = fields
        .iter()
        .map(|(_, f)| new_null_array(f.data_type(), 1))
        .collect::<Vec<_>>();
    if let Some(text) = text {
        children[7] = text;
    }
    Arc::new(UnionArray::try_new(fields, vec![id].into(), None, children).unwrap())
}

#[test]
fn floating_object_bits_are_not_canonicalized() {
    let bits = [
        0,
        (-0.0_f64).to_bits(),
        f64::INFINITY.to_bits(),
        f64::NEG_INFINITY.to_bits(),
        0x7ff8_0000_0000_1234,
        0xfff8_0000_0000_5678,
    ];
    let values = bits.map(|value| V::Float(f64::from_bits(value)));
    let array = builtin_frame_array(&values).unwrap();
    let decoded = builtin_frame_values(&array).unwrap();
    for (value, expected) in decoded.iter().zip(bits) {
        let V::Float(value) = value else {
            panic!("float tag must be retained")
        };
        assert_eq!(value.to_bits(), expected);
    }
}

#[test]
fn builtin_payload_decoder_rejects_ambiguous_or_invalid_inputs() {
    let ordinary = Arc::new(arrow_array::Int64Array::from(vec![1])) as ArrayRef;
    assert!(
        builtin_frame_values(&ordinary)
            .unwrap_err()
            .to_string()
            .contains("expected extended built-in object schema")
    );
    for id in [0, 1, 2, 3, 7] {
        assert!(
            builtin_frame_values(&malformed(id, None))
                .unwrap_err()
                .to_string()
                .contains("typed null is not an explicit object sentinel")
        );
    }
    for point in [None, Some(0x11_0000)] {
        let mut builder = ListBuilder::new(UInt32Builder::new());
        builder.values().append_option(point);
        builder.append(true);
        let error = builtin_frame_values(&malformed(7, Some(Arc::new(builder.finish()))))
            .unwrap_err()
            .to_string();
        assert!(error.contains(if point.is_none() {
            "null code point"
        } else {
            "U+110000 is out of range"
        }));
    }
    for id in [4, 5, 6] {
        assert_eq!(builtin_frame_values(&malformed(id, None)).unwrap().len(), 1);
    }
}
