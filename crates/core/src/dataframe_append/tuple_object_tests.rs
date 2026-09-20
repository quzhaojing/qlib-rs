use super::*;
use crate::dataframe_append::temporal_objects::tests::{compare_cells, decode};
use crate::dataframe_append::{PythonTemporalValue, factorize_index_objects};
use arrow_array::{RecordBatch, UInt32Array, new_null_array};
use arrow_ipc::{reader::StreamReader, writer::StreamWriter};
use arrow_schema::{Schema, TimeUnit};
use serde_json::Value;
use std::{io::Cursor, path::PathBuf, process::Command};

pub(crate) fn cell(value: &Value) -> TupleFrameValue {
    if value[0] == "tuple" {
        TupleFrameValue::Tuple(value[1].as_array().unwrap().iter().map(cell).collect())
    } else if value[0] == "datetime" {
        assert_eq!(value[8], 0);
        assert_eq!(value[9], "None");
        let date = chrono::NaiveDate::from_ymd_opt(
            i32::try_from(value[1].as_i64().unwrap()).unwrap(),
            u32::try_from(value[2].as_u64().unwrap()).unwrap(),
            u32::try_from(value[3].as_u64().unwrap()).unwrap(),
        )
        .unwrap();
        let part = |i: usize| u32::try_from(value[i].as_u64().unwrap()).unwrap();
        let microseconds = date
            .and_hms_micro_opt(part(4), part(5), part(6), part(7))
            .unwrap()
            .and_utc()
            .timestamp_micros();
        TupleFrameValue::PythonTemporal(PythonTemporalValue::Datetime { microseconds })
    } else if value[0] == "timedelta" {
        let part = |i: usize| i128::from(value[i].as_i64().unwrap());
        TupleFrameValue::PythonTemporal(PythonTemporalValue::Timedelta {
            microseconds: part(1) * 86_400_000_000 + part(2) * 1_000_000 + part(3),
        })
    } else {
        TupleFrameValue::Scalar(decode(value))
    }
}

pub(crate) fn compare(values: &[TupleFrameValue], expected: &[Value]) {
    assert_eq!(values.len(), expected.len());
    for (value, expected) in values.iter().zip(expected) {
        match value {
            TupleFrameValue::PythonTemporal(_) => assert_eq!(value, &cell(expected)),
            TupleFrameValue::Scalar(value) => {
                assert_ne!(expected[0], "tuple");
                compare_cells(std::slice::from_ref(value), std::slice::from_ref(expected));
            }
            TupleFrameValue::Tuple(values) => {
                assert_eq!(expected[0], "tuple");
                compare(values, expected[1].as_array().unwrap());
            }
        }
    }
}

fn check_index(index: &Value) {
    for key in ["values", "names"] {
        let expected = index[key].as_array().unwrap();
        let values = expected.iter().map(cell).collect::<Vec<_>>();
        let array = tuple_frame_array(&values).unwrap();
        compare(&tuple_frame_values(&array).unwrap(), expected);
    }
    if let Some(levels) = index.get("levels") {
        for level in levels.as_array().unwrap() {
            check_index(level);
        }
    }
    if let Some(categories) = index.get("categories") {
        check_index(categories);
    }
}

#[test]
fn source_nested_tuple_cells_names_and_levels_round_trip_without_factoring() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/dataframe_multi_index_contract.py"))
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
        "6cc7f3f1e47d0f6e40f1ecbc2a7b6ba52d5fa7ede33bb23e665e440ff9ec7ecd"
    );
    let inputs = contract["inputs"].as_object().unwrap();
    assert_eq!(inputs.len(), 28);
    for index in inputs.values() {
        check_index(index);
    }
    let pairs = contract["pairs"].as_array().unwrap();
    assert_eq!(pairs.len(), 3136);
    for case in pairs {
        check_index(&case["output"]["index"]);
    }
    let chains = contract["chains"].as_array().unwrap();
    assert_eq!(chains.len(), 112);
    for case in chains {
        check_index(&case["initial"]);
        check_index(&case["first_output"]["output"]["index"]);
        check_index(&case["second_output"]["output"]["index"]);
    }
    let expected = inputs
        .values()
        .flat_map(|i| i["values"].as_array().unwrap().iter().cloned())
        .collect::<Vec<_>>();
    check_arrow_operations(&expected);
}

fn check_arrow_operations(expected: &[Value]) {
    let values = expected.iter().map(cell).collect::<Vec<_>>();
    let built = tuple_frame_array(&values).unwrap();
    assert_eq!(built.data_type(), &tuple_frame_dtype());
    compare(
        &tuple_frame_values(&built.slice(1, 3)).unwrap(),
        &expected[1..4],
    );
    let joined = arrow_select::concat::concat(&[
        built.slice(0, 3).as_ref(),
        built.slice(3, built.len() - 3).as_ref(),
    ])
    .unwrap();
    compare(&tuple_frame_values(&joined).unwrap(), expected);
    let indices = UInt32Array::from(vec![3, 1, 3, 0]);
    let taken = arrow_select::take::take(built.as_ref(), &indices, None).unwrap();
    compare(
        &tuple_frame_values(&taken).unwrap(),
        &[
            expected[3].clone(),
            expected[1].clone(),
            expected[3].clone(),
            expected[0].clone(),
        ],
    );
    let schema = Arc::new(Schema::new(vec![Field::new(
        "x",
        tuple_frame_dtype(),
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
    compare(&tuple_frame_values(decoded.column(0)).unwrap(), expected);
    let empty = tuple_frame_array(&[]).unwrap();
    assert!(tuple_frame_values(&empty).unwrap().is_empty());
    assert_eq!(empty.data_type(), &tuple_frame_dtype());
}

#[test]
fn python_temporal_identity_and_full_range_survive_arrow_operations() {
    let expected = serde_json::json!([
        ["datetime", 1, 1, 1, 0, 0, 0, 0, 0, "None"],
        ["datetime", 9999, 12, 31, 23, 59, 59, 999_999, 0, "None"],
        ["timedelta", -999_999_999, 0, 0],
        ["timedelta", 999_999_999, 86399, 999_999],
        [
            "tuple",
            [
                ["datetime", 1970, 1, 1, 0, 0, 0, 0, 0, "None"],
                ["timedelta", -1, 86399, 999_999]
            ]
        ]
    ]);
    check_arrow_operations(expected.as_array().unwrap());
    for (tag, payload) in [
        (3, BuiltinFrameValue::None),
        (3, BuiltinFrameValue::Int(i64::MAX)),
        (4, BuiltinFrameValue::Int(0)),
        (
            4,
            BuiltinFrameValue::Text(crate::RlCheckpointText::from_utf8("+0")),
        ),
        (
            4,
            BuiltinFrameValue::Text(crate::RlCheckpointText::from_utf8("-0")),
        ),
        (
            4,
            BuiltinFrameValue::Text(crate::RlCheckpointText::from_utf8("invalid")),
        ),
        (
            4,
            BuiltinFrameValue::Text(crate::RlCheckpointText::from_utf8("86400000000000000000")),
        ),
        (
            4,
            BuiltinFrameValue::Text(
                crate::RlCheckpointText::try_from_code_points(vec![0xd800]).unwrap(),
            ),
        ),
    ] {
        let stream = malformed(
            &[Some(tag)],
            false,
            temporal_frame_array(&[TemporalFrameValue::Builtin(payload)]).unwrap(),
        );
        assert_eq!(
            tuple_frame_values(&stream).unwrap_err().to_string(),
            "Invalid argument error: invalid Python temporal object payload"
        );
    }
    for value in [
        PythonTemporalValue::Datetime {
            microseconds: -62_135_596_800_000_001,
        },
        PythonTemporalValue::Datetime {
            microseconds: 253_402_300_800_000_000,
        },
        PythonTemporalValue::Timedelta {
            microseconds: -999_999_999 * 86_400_000_000 - 1,
        },
        PythonTemporalValue::Timedelta {
            microseconds: 86_400_000_000_000_000_000,
        },
    ] {
        let value = TupleFrameValue::PythonTemporal(value);
        assert!(tuple_frame_array(std::slice::from_ref(&value)).is_err());
        assert!(factorize_index_objects(&[value], true).is_err());
    }
}

#[test]
fn python_temporal_factorization_preserves_source_first_representative() {
    let output = Command::new("python").args(["-c", "import pandas as p,datetime as d,numpy as n,json; v=n.empty(4,dtype=object); v[:]=[d.datetime(1970,1,1),p.Timestamp(0,unit='us'),d.timedelta(0),p.Timedelta(0,unit='us')]; c,u=p.factorize(v,sort=False); print(json.dumps([c.tolist(),[type(x).__name__ for x in u]]))"]).output().unwrap();
    assert!(output.status.success());
    let source: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        source,
        serde_json::json!([[0, 0, 1, 1], ["datetime", "timedelta"]])
    );
    let values = vec![
        TupleFrameValue::PythonTemporal(PythonTemporalValue::Datetime { microseconds: 0 }),
        TupleFrameValue::Scalar(TemporalFrameValue::Timestamp {
            ticks: 0,
            unit: TimeUnit::Microsecond,
            timezone: None,
        }),
        TupleFrameValue::PythonTemporal(PythonTemporalValue::Timedelta { microseconds: 0 }),
        TupleFrameValue::Scalar(TemporalFrameValue::Duration {
            ticks: 0,
            unit: TimeUnit::Microsecond,
        }),
    ];
    let result = factorize_index_objects(&values, true).unwrap();
    assert_eq!(result.codes, vec![Some(0), Some(0), Some(1), Some(1)]);
    assert_eq!(result.uniques, vec![values[0].clone(), values[2].clone()]);
    for tag in [3, 4] {
        let invalid = malformed(
            &[Some(tag)],
            false,
            new_null_array(&temporal_frame_dtype(), 1),
        );
        assert!(tuple_frame_values(&invalid).is_err());
    }
}

fn malformed(tags: &[Option<u8>], null_tokens: bool, scalars: ArrayRef) -> ArrayRef {
    let mut builder = LargeListBuilder::new(UInt8Builder::new());
    for tag in tags {
        builder.values().append_option(*tag);
    }
    builder.append(true);
    let tags = builder.finish();
    let tokens = StructArray::new(
        fields(),
        vec![tags.values().clone(), scalars],
        null_tokens.then(|| vec![false; tags.values().len()].into()),
    );
    Arc::new(LargeListArray::new(
        item(),
        tags.offsets().clone(),
        Arc::new(tokens),
        None,
    ))
}

#[test]
fn malformed_tuple_streams_and_invalid_active_scalars_fail_explicitly() {
    let scalar = TemporalFrameValue::Builtin(BuiltinFrameValue::None);
    for (tags, null_tokens, message) in [
        (vec![], false, "no root"),
        (vec![Some(2)], false, "unmatched close"),
        (vec![Some(1)], false, "unclosed tuple"),
        (vec![Some(0), Some(0)], false, "multiple roots"),
        (
            vec![Some(1), Some(2), Some(1), Some(2)],
            false,
            "multiple roots",
        ),
        (vec![Some(0), Some(1)], false, "unclosed tuple"),
        (vec![Some(255)], false, "unknown token"),
        (vec![None], false, "null token"),
        (vec![Some(0)], true, "null token"),
    ] {
        let scalars = temporal_frame_array(&vec![scalar.clone(); tags.len()]).unwrap();
        let array = malformed(&tags, null_tokens, scalars);
        assert_eq!(
            tuple_frame_values(&array).unwrap_err().to_string(),
            format!("Invalid argument error: tuple object has {message}")
        );
    }
    let foreign: ArrayRef = Arc::new(UInt8Array::from(vec![1]));
    assert!(
        tuple_frame_values(&foreign)
            .unwrap_err()
            .to_string()
            .contains("expected tuple object schema")
    );
    assert!(
        tuple_frame_values(&new_null_array(&tuple_frame_dtype(), 1))
            .unwrap_err()
            .to_string()
            .contains("null row")
    );
    let invalid_scalars = new_null_array(&temporal_frame_dtype(), 2);
    let markers = malformed(&[Some(1), Some(2)], false, invalid_scalars.clone());
    assert_eq!(
        tuple_frame_values(&markers).unwrap(),
        vec![TupleFrameValue::Tuple(vec![])]
    );
    let active = malformed(&[Some(0), Some(0)], false, invalid_scalars);
    assert!(tuple_frame_values(&active).is_err());
    for value in [
        TemporalFrameValue::Duration {
            ticks: i64::MIN,
            unit: TimeUnit::Second,
        },
        TemporalFrameValue::Timestamp {
            ticks: i64::MIN,
            unit: TimeUnit::Nanosecond,
            timezone: None,
        },
    ] {
        assert!(
            tuple_frame_array(&[TupleFrameValue::Tuple(vec![TupleFrameValue::Scalar(value)])])
                .unwrap_err()
                .to_string()
                .contains("reserved NaT ticks")
        );
    }
}

#[test]
fn tuple_scalar_identity_float_bits_and_nested_boundaries_are_preserved() {
    let bits = [
        0,
        1_u64 << 63,
        0x7ff8_0000_0000_0042,
        0x7ff0_0000_0000_0001,
        u64::MAX,
    ];
    let values = bits
        .iter()
        .map(|bits| {
            TupleFrameValue::Tuple(vec![TupleFrameValue::Scalar(TemporalFrameValue::Builtin(
                BuiltinFrameValue::Float(f64::from_bits(*bits)),
            ))])
        })
        .collect::<Vec<_>>();
    let decoded = tuple_frame_values(&tuple_frame_array(&values).unwrap()).unwrap();
    for (value, bits) in decoded.iter().zip(bits) {
        let TupleFrameValue::Tuple(values) = value else {
            panic!("lost tuple")
        };
        let TupleFrameValue::Scalar(TemporalFrameValue::Builtin(BuiltinFrameValue::Float(value))) =
            values[0]
        else {
            panic!("lost float")
        };
        assert_eq!(value.to_bits(), bits);
    }
    let mut nested = TupleFrameValue::Tuple(vec![]);
    for _ in 0..256 {
        nested = TupleFrameValue::Tuple(vec![nested]);
    }
    assert_eq!(
        tuple_frame_values(&tuple_frame_array(&[nested.clone()]).unwrap()).unwrap(),
        vec![nested]
    );
}
