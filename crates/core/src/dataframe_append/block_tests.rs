use std::{path::PathBuf, process::Command, sync::Arc};

use arrow_array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, RecordBatchOptions,
    StringArray, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use num_traits::ToPrimitive;
use serde_json::{Value, json};

use super::{DataframeAppendError, IndexedFrame, dataframe_append_with_warnings};

pub(super) fn batch(columns: Vec<(&str, ArrayRef)>, rows: usize) -> RecordBatch {
    let fields: Vec<_> = columns
        .iter()
        .map(|(name, a)| Field::new(*name, a.data_type().clone(), true))
        .collect();
    RecordBatch::try_new_with_options(
        Arc::new(Schema::new(fields)),
        columns.into_iter().map(|(_, a)| a).collect(),
        &RecordBatchOptions::new().with_row_count(Some(rows)),
    )
    .unwrap()
}

fn floats(kind: &str, values: Vec<f64>) -> ArrayRef {
    let array = Arc::new(Float64Array::from(values)) as ArrayRef;
    if kind == "float16" {
        arrow_cast::cast(&array, &DataType::Float16).unwrap()
    } else if kind == "float32" {
        arrow_cast::cast(&array, &DataType::Float32).unwrap()
    } else {
        array
    }
}

fn index(rows: usize) -> ArrayRef {
    Arc::new(Int64Array::from(
        (0..rows)
            .map(|i| i64::try_from(i).unwrap())
            .collect::<Vec<_>>(),
    ))
}

pub(super) fn expected_float(text: &str) -> f64 {
    match text {
        "nan" => f64::NAN,
        "inf" => f64::INFINITY,
        "-inf" => f64::NEG_INFINITY,
        hex => {
            let negative = hex.starts_with('-');
            let hex = hex.trim_start_matches('-').trim_start_matches("0x");
            let (mantissa, exponent) = hex.split_once('p').unwrap();
            let (_, fraction) = mantissa.split_once('.').unwrap();
            let integer = u64::from_str_radix(&mantissa.replace('.', ""), 16)
                .unwrap()
                .to_f64()
                .unwrap();
            let value = integer
                * 2.0_f64.powi(
                    exponent.parse::<i32>().unwrap() - 4 * i32::try_from(fraction.len()).unwrap(),
                );
            if negative { -value } else { value }
        }
    }
}

pub(super) fn compare(result: &IndexedFrame, warnings: &[Value], expected: &Value) {
    let dtypes: Vec<_> = result
        .data()
        .columns()
        .iter()
        .map(|a| {
            if super::objects::is_object(a.data_type())
                || super::extended_plan::is_extended(a.data_type())
            {
                "object".to_owned()
            } else if a.data_type() == &DataType::Boolean {
                "bool".to_owned()
            } else {
                a.data_type().to_string().to_lowercase()
            }
        })
        .collect();
    assert_eq!(json!(dtypes), expected["output"]["dtypes"]);
    let groups: Vec<_> = result
        .blocks()
        .iter()
        .map(|group| json!([dtypes[group[0]], group]))
        .collect();
    assert_eq!(json!(groups), expected["output"]["blocks"]);
    assert_eq!(json!(warnings), expected["warnings"]);
    for (column, wanted) in result
        .data()
        .columns()
        .iter()
        .zip(expected["output"]["values"].as_array().unwrap())
    {
        compare_values(column, wanted);
    }
}

pub(super) fn compare_values(column: &ArrayRef, wanted: &Value) {
    if super::extended_plan::is_extended(column.data_type()) {
        super::builtin_object_tests::compare(&super::builtin_frame_values(column).unwrap(), wanted);
        return;
    }
    if super::objects::is_object(column.data_type()) {
        let union = column
            .as_any()
            .downcast_ref::<arrow_array::UnionArray>()
            .unwrap();
        assert_eq!(union.len(), wanted.as_array().unwrap().len());
        for (i, value) in wanted.as_array().unwrap().iter().enumerate() {
            compare_values(&union.value(i), &json!([value]));
        }
        return;
    }
    if column.data_type() == &DataType::Boolean {
        let array = column.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(array.len(), wanted.as_array().unwrap().len());
        for (position, value) in wanted.as_array().unwrap().iter().enumerate() {
            assert_eq!(*value, json!(["bool", array.value(position)]));
        }
        return;
    }
    if column.data_type().is_integer() {
        let text = arrow_cast::cast(column, &DataType::Utf8).unwrap();
        let text = text.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(text.len(), wanted.as_array().unwrap().len());
        for (position, value) in wanted.as_array().unwrap().iter().enumerate() {
            assert_eq!(value[0], "int");
            assert_eq!(text.value(position), value[1].as_str().unwrap());
        }
        return;
    }
    let wide = arrow_cast::cast(column, &DataType::Float64).unwrap();
    let wide = wide.as_any().downcast_ref::<Float64Array>().unwrap();
    assert_eq!(wide.len(), wanted.as_array().unwrap().len());
    for (position, value) in wanted.as_array().unwrap().iter().enumerate() {
        let expected = expected_float(value[1].as_str().unwrap());
        if wide.is_null(position) {
            assert!(
                expected.is_nan(),
                "native missing value must correspond to source NaN"
            );
            continue;
        }
        let actual = wide.value(position);
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }
}

fn sample(kind: &str, state: &str) -> ArrayRef {
    if kind == "bool" {
        return Arc::new(BooleanArray::from(match state {
            "empty" => vec![],
            "finite" => vec![false, true],
            "extreme" => vec![true, false],
            _ => panic!("unknown bool state"),
        }));
    }
    if kind.starts_with("int") || kind.starts_with("uint") {
        return integer_sample(kind, state);
    }
    let max = if kind == "float16" {
        half::f16::MAX.to_f64()
    } else if kind == "float32" {
        f64::from(f32::MAX)
    } else {
        f64::MAX
    };
    let values = match state {
        "empty" => vec![],
        "finite" => vec![0.0, 1.0],
        "all_na" => vec![f64::NAN, f64::NAN],
        "extreme" => vec![-max, -0.0, max, f64::INFINITY, f64::NEG_INFINITY, f64::NAN],
        _ => panic!("unknown state"),
    };
    floats(kind, values)
}

fn integer_sample(kind: &str, state: &str) -> ArrayRef {
    let dtype = match kind {
        "int8" => DataType::Int8,
        "int16" => DataType::Int16,
        "int32" => DataType::Int32,
        "int64" => DataType::Int64,
        "uint8" => DataType::UInt8,
        "uint16" => DataType::UInt16,
        "uint32" => DataType::UInt32,
        "uint64" => DataType::UInt64,
        _ => panic!("unknown integer type"),
    };
    let bits = kind
        .trim_start_matches('u')
        .trim_start_matches("int")
        .parse::<u32>()
        .unwrap();
    let array: ArrayRef = if kind.starts_with('u') {
        let max = if bits == 64 {
            u64::MAX
        } else {
            (1_u64 << bits) - 1
        };
        let values = match state {
            "empty" => vec![],
            "finite" => vec![0, 1],
            "extreme" => vec![0, max],
            _ => panic!("unknown integer state"),
        };
        Arc::new(UInt64Array::from(values))
    } else {
        let (min, max) = if bits == 64 {
            (i64::MIN, i64::MAX)
        } else {
            (-(1_i64 << (bits - 1)), (1_i64 << (bits - 1)) - 1)
        };
        let values = match state {
            "empty" => vec![],
            "finite" => vec![0, 1],
            "extreme" => vec![min, max],
            _ => panic!("unknown integer state"),
        };
        Arc::new(Int64Array::from(values))
    };
    arrow_cast::cast(&array, &dtype).unwrap()
}

#[test]
fn actual_source_float_pairs_and_block_partitions() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/dataframe_numeric_contract.py"))
        .arg(root.join("../../../qlib/qlib/rl/order_execution/utils.py"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let contract: Value = serde_json::from_slice(&output.stdout).unwrap();
    let mut count = 0;
    for case in contract["numeric"].as_array().unwrap() {
        let left_kind = case["left"][0].as_str().unwrap();
        let right_kind = case["right"][0].as_str().unwrap();
        let left = sample(left_kind, case["left"][1].as_str().unwrap());
        let right = sample(right_kind, case["right"][1].as_str().unwrap());
        let frame = IndexedFrame::new(
            index(left.len()),
            None,
            batch(vec![("x", left.clone())], left.len()),
        )
        .unwrap();
        let other = batch(
            vec![("datetime", index(right.len())), ("x", right.clone())],
            right.len(),
        );
        let mut warnings = vec![];
        let result = dataframe_append_with_warnings(&frame, &other, &mut |message| {
            warnings.push(json!(["FutureWarning", message]));
        })
        .unwrap();
        compare(&result, &warnings, case);
        count += 1;
    }
    assert_eq!(count, 1521);
    compare_object_chains(&contract);
    compare_object_blocks(&contract);
    compare_numeric_objects(&contract, "numeric_object_inputs", "x", "x");
    compare_numeric_objects(&contract, "numeric_object_alignment", "a", "b");
    compare_integer_alignment(&contract);
    compare_numeric_indexes(&contract);
    assert_eq!(contract["floating_blocks"].as_array().unwrap().len(), 144);
    for case in contract["floating_blocks"].as_array().unwrap() {
        let names = if case["reverse"] == true {
            ["y", "x"]
        } else {
            ["x", "y"]
        };
        let columns = names
            .iter()
            .map(|&name| {
                (
                    name,
                    floats(
                        case["left_dtype"].as_str().unwrap(),
                        vec![if name == "x" || case["all_na"] == true {
                            f64::NAN
                        } else {
                            2.0
                        }],
                    ),
                )
            })
            .collect();
        let mut frame = IndexedFrame::new(index(1), None, batch(columns, 1)).unwrap();
        if case["left_fragmented"] == true {
            frame = frame.with_blocks(vec![vec![0], vec![1]]).unwrap();
        }
        let mut other = vec![("datetime", index(1))];
        other.extend(names.iter().map(|&name| {
            (
                name,
                floats(case["right_dtype"].as_str().unwrap(), vec![1.0]),
            )
        }));
        let mut warnings = vec![];
        let result = dataframe_append_with_warnings(&frame, &batch(other, 1), &mut |message| {
            warnings.push(json!(["FutureWarning", message]));
        })
        .unwrap();
        compare(&result, &warnings, case);
    }
}

fn compare_integer_alignment(contract: &Value) {
    let cases = contract["integer_alignment"].as_array().unwrap();
    assert_eq!(cases.len(), 256);
    for case in cases {
        let left = sample(
            case["left"][0].as_str().unwrap(),
            case["left"][1].as_str().unwrap(),
        );
        let right = sample(
            case["right"][0].as_str().unwrap(),
            case["right"][1].as_str().unwrap(),
        );
        let frame = IndexedFrame::new(
            index(left.len()),
            None,
            batch(vec![("a", left.clone())], left.len()),
        )
        .unwrap();
        let other = batch(
            vec![("datetime", index(right.len())), ("b", right.clone())],
            right.len(),
        );
        let mut warnings = vec![];
        let result = dataframe_append_with_warnings(&frame, &other, &mut |message| {
            warnings.push(json!(["FutureWarning", message]));
        })
        .unwrap();
        compare(&result, &warnings, case);
    }
}

fn compare_object_chains(contract: &Value) {
    let cases = contract["object_contract"]["chains"].as_array().unwrap();
    assert_eq!(cases.len(), 2808);
    for case in cases {
        let read = |name: &str| {
            sample(
                case[name][0].as_str().unwrap(),
                case[name][1].as_str().unwrap(),
            )
        };
        let left = read("left");
        let right = read("right");
        let third = read("third");
        let frame = IndexedFrame::new(
            index(left.len()),
            None,
            batch(vec![("x", left.clone())], left.len()),
        )
        .unwrap();
        let other = batch(
            vec![("datetime", index(right.len())), ("x", right.clone())],
            right.len(),
        );
        let first = super::dataframe_append(&frame, &other).unwrap();
        compare(
            &first,
            &[],
            &json!({"output":case["intermediate"],"warnings":[]}),
        );
        let other = batch(
            vec![("datetime", index(third.len())), ("x", third.clone())],
            third.len(),
        );
        let mut warnings = vec![];
        let result = dataframe_append_with_warnings(&first, &other, &mut |message| {
            warnings.push(json!(["FutureWarning", message]));
        })
        .unwrap();
        compare(&result, &warnings, case);
    }
}

fn compare_numeric_objects(contract: &Value, key: &str, left_name: &str, right_name: &str) {
    let cases = contract[key].as_array().unwrap();
    assert_eq!(cases.len(), 192);
    for case in cases {
        let read = |name: &str, boxed: &str| {
            let array = sample(
                case[name][0].as_str().unwrap(),
                case[name][1].as_str().unwrap(),
            );
            if case[boxed] == true {
                super::numeric_object_array(&array).unwrap()
            } else {
                array
            }
        };
        let left = read("left", "left_object");
        let right = read("right", "right_object");
        let frame = IndexedFrame::new(
            index(left.len()),
            None,
            batch(vec![(left_name, left.clone())], left.len()),
        )
        .unwrap();
        let other = batch(
            vec![
                ("datetime", index(right.len())),
                (right_name, right.clone()),
            ],
            right.len(),
        );
        let mut warnings = vec![];
        let result = dataframe_append_with_warnings(&frame, &other, &mut |message| {
            warnings.push(json!(["FutureWarning", message]));
        })
        .unwrap();
        compare(&result, &warnings, case);
    }
}

#[test]
fn object_buffers_round_trip_and_reject_unsupported_conversion() {
    use arrow_ipc::{reader::StreamReader, writer::StreamWriter};
    use std::io::Cursor;
    let inputs = [
        sample("bool", "finite"),
        sample("int64", "extreme"),
        sample("uint64", "extreme"),
        sample("float64", "extreme"),
    ];
    let boxed = inputs
        .iter()
        .map(|a| super::numeric_object_array(a).unwrap())
        .collect::<Vec<_>>();
    let joined =
        arrow_select::concat::concat(&boxed.iter().map(AsRef::as_ref).collect::<Vec<_>>()).unwrap();
    let same = super::numeric_object_array(&joined).unwrap();
    assert!(Arc::ptr_eq(&same, &joined));
    let data = batch(vec![("objects", joined.clone())], joined.len());
    let mut bytes = vec![];
    let mut writer = StreamWriter::try_new(&mut bytes, data.schema_ref()).unwrap();
    writer.write(&data).unwrap();
    writer.finish().unwrap();
    let decoded = StreamReader::try_new(Cursor::new(bytes), None)
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    assert_eq!(decoded.schema(), data.schema());
    assert_eq!(decoded.column(0).to_data(), joined.to_data());
    let slice = joined.slice(1, 3);
    assert!(!super::objects::all_na(&slice));
    assert!(
        super::objects::cast(&joined, &DataType::Float64)
            .unwrap_err()
            .to_string()
            .contains("cannot unbox numeric objects as Float64")
    );
    assert!(
        super::objects::cast(&joined, &DataType::Int64)
            .unwrap_err()
            .to_string()
            .contains("cannot unbox numeric objects as Int64")
    );
    let missing = super::numeric_object_array(
        &(Arc::new(Float64Array::from(vec![None, Some(f64::NAN)])) as ArrayRef),
    )
    .unwrap();
    assert!(super::objects::all_na(&missing));
    assert_eq!(
        super::objects::cast(&missing, &DataType::Float32)
            .unwrap()
            .null_count(),
        2
    );
    let unsupported = Arc::new(StringArray::from(vec!["x"])) as ArrayRef;
    assert!(
        super::numeric_object_array(&unsupported)
            .unwrap_err()
            .to_string()
            .contains("numeric object adapter does not support Utf8")
    );
    for (left, right) in [(joined.clone(), unsupported.clone()), (unsupported, joined)] {
        let frame = IndexedFrame::new(
            index(left.len()),
            None,
            batch(vec![("x", left.clone())], left.len()),
        )
        .unwrap();
        let other = batch(
            vec![("datetime", index(right.len())), ("x", right.clone())],
            right.len(),
        );
        assert!(matches!(
            super::dataframe_append(&frame, &other),
            Err(DataframeAppendError::DtypeAdapter(_, _))
        ));
    }
}

fn compare_object_blocks(contract: &Value) {
    let cases = contract["object_contract"]["blocks"].as_array().unwrap();
    assert_eq!(cases.len(), 96);
    for case in cases {
        let names = if case["reverse"] == true {
            ["y", "x"]
        } else {
            ["x", "y"]
        };
        let make = |boolean: bool| {
            names
                .iter()
                .map(|&name| {
                    let value: ArrayRef = if boolean {
                        Arc::new(BooleanArray::from(vec![name == "x"]))
                    } else {
                        floats(
                            case["dtype"].as_str().unwrap(),
                            vec![if name == "x" || case["all_na"] == true {
                                f64::NAN
                            } else {
                                2.0
                            }],
                        )
                    };
                    (name, value)
                })
                .collect::<Vec<_>>()
        };
        let mut frame =
            IndexedFrame::new(index(1), None, batch(make(case["bool_first"] == true), 1)).unwrap();
        if case["left_fragmented"] == true {
            frame = frame.with_blocks(vec![vec![0], vec![1]]).unwrap();
        }
        let mut columns = vec![("datetime", index(1))];
        columns.extend(make(case["bool_first"] != true));
        let mut warnings = vec![];
        let result = dataframe_append_with_warnings(&frame, &batch(columns, 1), &mut |message| {
            warnings.push(json!(["FutureWarning", message]));
        })
        .unwrap();
        compare(&result, &warnings, case);
    }
}

fn compare_numeric_indexes(contract: &Value) {
    let frame = IndexedFrame::new(
        index(0),
        None,
        batch(vec![("x", floats("float64", vec![]))], 0),
    )
    .unwrap();
    let other = batch(
        vec![
            ("datetime", floats("float32", vec![1.0])),
            ("x", floats("float32", vec![1.0])),
        ],
        1,
    );
    let mut warnings = vec![];
    let result = dataframe_append_with_warnings(&frame, &other, &mut |message| {
        warnings.push(json!(["FutureWarning", message]));
    })
    .unwrap();
    assert_eq!(warnings.len(), 2);
    compare(&result, &warnings, &contract["warning_order"]);
    let cases = contract["numeric_indexes"].as_array().unwrap();
    assert_eq!(cases.len(), 81);
    for case in cases {
        let left = sample(
            case["left"][0].as_str().unwrap(),
            case["left"][1].as_str().unwrap(),
        );
        let right = sample(
            case["right"][0].as_str().unwrap(),
            case["right"][1].as_str().unwrap(),
        );
        let frame = IndexedFrame::new(
            left.clone(),
            None,
            batch(vec![("x", index(left.len()))], left.len()),
        )
        .unwrap();
        let other = batch(
            vec![("datetime", right.clone()), ("x", index(right.len()))],
            right.len(),
        );
        let mut warnings = vec![];
        let result = dataframe_append_with_warnings(&frame, &other, &mut |message| {
            warnings.push(json!(["FutureWarning", message]));
        })
        .unwrap();
        compare(&result, &warnings, case);
        // Reuse the exact array comparison for the separately stored index.
        let n = result.index().len();
        let index_view = IndexedFrame::new(
            index(n),
            None,
            batch(vec![("index", result.index().clone())], n),
        )
        .unwrap();
        let dtype = &case["index"]["dtype"];
        compare(
            &index_view,
            &[],
            &json!({"output":{"dtypes":[dtype],"blocks":[[dtype,[0]]],"values":[case["index"]["values"]]},"warnings":[]}),
        );
    }
}

#[test]
fn block_partition_validation() {
    let frame = IndexedFrame::new(
        index(1),
        None,
        batch(
            vec![
                ("a", floats("float64", vec![1.0])),
                ("b", floats("float32", vec![2.0])),
            ],
            1,
        ),
    )
    .unwrap();
    for (groups, reason) in [
        (vec![vec![], vec![0, 1]], "empty block"),
        (vec![vec![2]], "column out of bounds"),
        (vec![vec![0, 2]], "column out of bounds"),
        (vec![vec![0], vec![0]], "column occurs more than once"),
        (vec![vec![0, 1]], "mixed dtypes in one block"),
        (vec![vec![0]], "column omitted from blocks"),
    ] {
        let error = frame.clone().with_blocks(groups).unwrap_err();
        assert!(matches!(error, DataframeAppendError::BlockLayout(_)));
        assert_eq!(
            error.to_string(),
            format!("invalid frame block layout: {reason}")
        );
    }
    assert_eq!(
        frame
            .clone()
            .with_blocks(vec![vec![1], vec![0]])
            .unwrap()
            .blocks(),
        &[vec![1], vec![0]]
    );
}

#[test]
fn block_metadata_survives_a_second_append() {
    let data = batch(
        vec![
            ("x", floats("float64", vec![f64::NAN])),
            ("y", floats("float64", vec![2.0])),
        ],
        1,
    );
    let frame = IndexedFrame::new(index(1), None, data)
        .unwrap()
        .with_blocks(vec![vec![0], vec![1]])
        .unwrap();
    let ignored = batch(vec![("datetime", index(0))], 0);
    let retained = super::dataframe_append(&frame, &ignored).unwrap();
    assert_eq!(retained.blocks(), frame.blocks());
    let other = batch(
        vec![
            ("datetime", index(1)),
            ("x", floats("float32", vec![1.0])),
            ("y", floats("float32", vec![1.0])),
        ],
        1,
    );
    let mut warnings = vec![];
    let result = dataframe_append_with_warnings(&retained, &other, &mut |message| {
        warnings.push(message.to_owned());
    })
    .unwrap();
    assert_eq!(result.data().column(0).data_type(), &DataType::Float32);
    assert_eq!(result.data().column(1).data_type(), &DataType::Float64);
    assert_eq!(warnings.len(), 1);
    assert_eq!(result.blocks(), &[vec![0], vec![1]]);
    assert!(!super::blocks::float_all_na(&index(0)));
    assert!(super::blocks::float_all_na(
        &(Arc::new(Float64Array::from(vec![None])) as ArrayRef)
    ));
}
