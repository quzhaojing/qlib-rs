use super::block_tests::{batch, compare, expected_float};
use super::{
    BuiltinFrameValue as V, IndexedFrame, builtin_frame_array, dataframe_append_with_warnings,
};
use arrow_array::{ArrayRef, BooleanArray, Float64Array, Int64Array, UInt64Array};
use arrow_schema::DataType;
use serde_json::{Value, json};
use std::{collections::HashMap, path::PathBuf, process::Command, sync::Arc};

fn object(value: &Value) -> V {
    match value[0].as_str().unwrap() {
        "none" => V::None,
        "pd.NA" => V::PandasNa,
        "NaT" => V::NotATime,
        "bool" => V::Bool(value[1].as_bool().unwrap()),
        "int" => {
            let text = value[1].as_str().unwrap();
            if text.starts_with('-') {
                V::Int(text.parse().unwrap())
            } else {
                V::UInt(text.parse().unwrap())
            }
        }
        "float" => V::Float(expected_float(value[1].as_str().unwrap())),
        "str" => V::Text(
            crate::RlCheckpointText::try_from_code_points(
                value[1]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap()),
            )
            .unwrap(),
        ),
        _ => panic!("unknown fixture tag"),
    }
}

fn column(dtype: &str, values: &Value) -> ArrayRef {
    let narrow = match dtype {
        "int8" => Some(("int64", DataType::Int8)),
        "int16" => Some(("int64", DataType::Int16)),
        "int32" => Some(("int64", DataType::Int32)),
        "uint8" => Some(("uint64", DataType::UInt8)),
        "uint16" => Some(("uint64", DataType::UInt16)),
        "uint32" => Some(("uint64", DataType::UInt32)),
        _ => None,
    };
    if let Some((wide, target)) = narrow {
        return arrow_cast::cast(&column(wide, values), &target).unwrap();
    }
    let values = values.as_array().unwrap();
    match dtype {
        "object" => builtin_frame_array(&values.iter().map(object).collect::<Vec<_>>()).unwrap(),
        "bool" => Arc::new(BooleanArray::from(
            values
                .iter()
                .map(|v| v[1].as_bool().unwrap())
                .collect::<Vec<_>>(),
        )),
        "int64" => Arc::new(Int64Array::from(
            values
                .iter()
                .map(|v| v[1].as_str().unwrap().parse::<i64>().unwrap())
                .collect::<Vec<_>>(),
        )),
        "uint64" => Arc::new(UInt64Array::from(
            values
                .iter()
                .map(|v| v[1].as_str().unwrap().parse::<u64>().unwrap())
                .collect::<Vec<_>>(),
        )),
        name => {
            let values = Arc::new(Float64Array::from(
                values
                    .iter()
                    .map(|v| expected_float(v[1].as_str().unwrap()))
                    .collect::<Vec<_>>(),
            )) as ArrayRef;
            let dtype = match name {
                "float16" => DataType::Float16,
                "float32" => DataType::Float32,
                "float64" => DataType::Float64,
                _ => panic!("unknown fixture dtype"),
            };
            arrow_cast::cast(&values, &dtype).unwrap()
        }
    }
}

fn index(rows: usize) -> ArrayRef {
    Arc::new(Int64Array::from(
        (0..rows)
            .map(|v| i64::try_from(v).unwrap())
            .collect::<Vec<_>>(),
    ))
}

pub(super) fn source_column(dtype: &str, values: &Value, legacy: bool) -> ArrayRef {
    let items = values.as_array().unwrap();
    if dtype != "object"
        || !legacy
        || !items
            .iter()
            .all(|v| matches!(v[0].as_str().unwrap(), "bool" | "int" | "float"))
    {
        return column(dtype, values);
    }
    if items.is_empty() {
        return super::numeric_object_array(
            &(Arc::new(Float64Array::from(Vec::<f64>::new())) as ArrayRef),
        )
        .unwrap();
    }
    let arrays = items
        .iter()
        .map(|value| {
            let dtype = match value[0].as_str().unwrap() {
                "bool" => "bool",
                "float" => "float64",
                _ => {
                    if value[1].as_str().unwrap().starts_with('-') {
                        "int64"
                    } else {
                        "uint64"
                    }
                }
            };
            super::numeric_object_array(&column(dtype, &json!([value]))).unwrap()
        })
        .collect::<Vec<_>>();
    arrow_select::concat::concat(&arrays.iter().map(AsRef::as_ref).collect::<Vec<_>>()).unwrap()
}
fn frame(array: &ArrayRef) -> IndexedFrame {
    IndexedFrame::new(
        index(array.len()),
        None,
        batch(vec![("x", array.clone())], array.len()),
    )
    .unwrap()
}
fn append(frame: &IndexedFrame, right: &ArrayRef) -> (IndexedFrame, Vec<Value>) {
    let other = batch(
        vec![("datetime", index(right.len())), ("x", right.clone())],
        right.len(),
    );
    let mut warnings = vec![];
    let result = dataframe_append_with_warnings(frame, &other, &mut |message| {
        warnings.push(json!(["FutureWarning", message]));
    })
    .unwrap();
    (result, warnings)
}
fn check(result: &IndexedFrame, warnings: &[Value], case: &Value) {
    compare(result, warnings, case);
    assert_eq!(result.index_name(), case["output"]["index_name"].as_str());
    let actual = result
        .index()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(json!(actual.values().as_ref()), case["output"]["index"]);
    assert_eq!(
        json!(
            result
                .data()
                .schema_ref()
                .fields()
                .iter()
                .map(|f| f.name())
                .collect::<Vec<_>>()
        ),
        case["output"]["columns"]
    );
}

#[test]
fn extended_objects_match_actual_append_pairs_chains_and_blocks() {
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
    for legacy in [false, true] {
        run_contract(&contract, legacy);
    }
}

fn run_contract(contract: &Value, legacy: bool) {
    let input = |name: &Value| {
        let sample = &contract["inputs"][name.as_str().unwrap()];
        source_column(
            sample["dtypes"][0].as_str().unwrap(),
            &sample["values"][0],
            legacy,
        )
    };
    let mut intermediates = HashMap::new();
    let pairs = contract["pairs"].as_array().unwrap();
    assert_eq!(pairs.len(), 841);
    for case in pairs {
        let (result, warnings) = append(&frame(&input(&case["left"])), &input(&case["right"]));
        check(&result, &warnings, case);
        intermediates.insert(
            (
                case["left"].as_str().unwrap(),
                case["right"].as_str().unwrap(),
            ),
            result,
        );
    }
    let chains = contract["chains"].as_array().unwrap();
    assert_eq!(chains.len(), 1682);
    for case in chains {
        let first = &intermediates[&(
            case["left"].as_str().unwrap(),
            case["right"].as_str().unwrap(),
        )];
        let (result, warnings) = append(first, &input(&case["third"]));
        check(&result, &warnings, case);
    }
    check_blocks(contract, legacy);
    let alignment = contract["alignment"].as_array().unwrap();
    assert_eq!(alignment.len(), 841);
    for case in alignment {
        let first = frame(&input(&case["left"]));
        let right = input(&case["right"]);
        let other = batch(
            vec![("datetime", index(right.len())), ("y", right)],
            input(&case["right"]).len(),
        );
        let mut warnings = vec![];
        let result = dataframe_append_with_warnings(&first, &other, &mut |message| {
            warnings.push(json!(["FutureWarning", message]));
        })
        .unwrap();
        check(&result, &warnings, case);
    }
}

fn check_blocks(contract: &Value, legacy: bool) {
    let cases = contract["blocks"].as_array().unwrap();
    assert_eq!(cases.len(), 384);
    let layouts = contract["layouts"].as_array().unwrap();
    assert_eq!(layouts.len(), 2);
    for case in cases.iter().chain(layouts) {
        let table = |value: &Value| {
            batch(
                value["columns"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .map(|(i, name)| {
                        (
                            name.as_str().unwrap(),
                            source_column(
                                value["dtypes"][i].as_str().unwrap(),
                                &value["values"][i],
                                legacy,
                            ),
                        )
                    })
                    .collect(),
                2,
            )
        };
        let left = &case["left_input"];
        let groups = left["blocks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| {
                b[1].as_array()
                    .unwrap()
                    .iter()
                    .map(|v| usize::try_from(v.as_u64().unwrap()).unwrap())
                    .collect()
            })
            .collect();
        let frame = IndexedFrame::new(index(2), None, table(left))
            .unwrap()
            .with_blocks(groups)
            .unwrap();
        let mut warnings = vec![];
        let result =
            dataframe_append_with_warnings(&frame, &table(&case["right_input"]), &mut |message| {
                warnings.push(json!(["FutureWarning", message]));
            })
            .unwrap();
        assert_eq!(
            result.blocks().len(),
            case["output"]["blocks"].as_array().unwrap().len(),
            "{case}"
        );
        check(&result, &warnings, case);
    }
}

#[test]
fn extended_policy_propagates_invalid_inputs_and_preserves_missing_identity() {
    let unsupported = Arc::new(arrow_array::StringArray::from(vec!["x"])) as ArrayRef;
    assert!(
        super::extended_plan::promote(&unsupported)
            .unwrap_err()
            .to_string()
            .contains("numeric object adapter does not support")
    );
    let malformed = arrow_array::new_null_array(&super::builtin_frame_dtype(), 1);
    let malformed_frame = frame(&malformed);
    let aligned_other = batch(vec![("datetime", index(1)), ("y", index(1))], 1);
    assert!(
        super::dataframe_append(&malformed_frame, &aligned_other)
            .unwrap_err()
            .to_string()
            .contains("typed null")
    );
    assert!(
        super::extended_plan::unbox_missing(&malformed, &DataType::Float32)
            .unwrap_err()
            .to_string()
            .contains("typed null")
    );
    let none = builtin_frame_array(&[V::None]).unwrap();
    let mut plan = super::blocks::Plan {
        positions: 0..1,
        dtype: None,
        warn: false,
        box_floats: false,
        fill: [None, None],
    };
    assert!(
        super::extended_plan::configure(
            &mut plan,
            &[none.clone(), unsupported.clone()],
            std::slice::from_ref(&none),
        )
        .unwrap_err()
        .to_string()
        .contains("numeric object adapter does not support")
    );
    for (value, dtype) in [
        (V::NotATime, DataType::Float64),
        (V::Int(1), DataType::Float32),
        (V::None, DataType::Int64),
    ] {
        let array = builtin_frame_array(&[value]).unwrap();
        assert!(
            super::extended_plan::unbox_missing(&array, &dtype)
                .unwrap_err()
                .to_string()
                .contains("cannot unbox extended objects")
        );
    }
    let legacy_null = arrow_array::new_null_array(&super::numeric_object_dtype(), 1);
    let promoted = super::extended_plan::promote(&legacy_null).unwrap();
    let decoded = super::builtin_frame_values(&promoted).unwrap();
    assert!(matches!(decoded.as_slice(), [V::Float(value)] if value.is_nan()));
    for (left, right) in [
        (malformed.clone(), none.clone()),
        (none.clone(), malformed),
        (unsupported.clone(), none.clone()),
        (none, unsupported),
    ] {
        let first = frame(&left);
        let other = batch(
            vec![("datetime", index(right.len())), ("x", right.clone())],
            right.len(),
        );
        let error = super::dataframe_append(&first, &other).unwrap_err();
        assert!(!error.to_string().is_empty());
        assert!(Arc::ptr_eq(first.data().column(0), &left));
        assert!(Arc::ptr_eq(other.column(1), &right));
    }
}

struct FillFailure {
    fail: usize,
    calls: usize,
}

impl super::FrameOperations for FillFailure {
    fn fill(
        &mut self,
        array: &ArrayRef,
        value: Option<&V>,
    ) -> Result<ArrayRef, arrow_schema::ArrowError> {
        let call = self.calls;
        self.calls += 1;
        if call == self.fail {
            return Err(arrow_schema::ArrowError::ComputeError(
                "object fill failed".into(),
            ));
        }
        super::extended_plan::fill(array, value)
    }
    fn cast(
        &mut self,
        array: &ArrayRef,
        dtype: &DataType,
    ) -> Result<ArrayRef, arrow_schema::ArrowError> {
        super::ArrowOperations.cast(array, dtype)
    }
    fn concat(
        &mut self,
        left: &ArrayRef,
        right: &ArrayRef,
    ) -> Result<ArrayRef, arrow_schema::ArrowError> {
        super::ArrowOperations.concat(left, right)
    }
    fn batch(
        &mut self,
        fields: Vec<arrow_schema::Field>,
        columns: Vec<ArrayRef>,
        rows: usize,
    ) -> Result<arrow_array::RecordBatch, arrow_schema::ArrowError> {
        super::ArrowOperations.batch(fields, columns, rows)
    }
}

#[test]
fn missing_object_fill_failure_is_atomic_on_either_side() {
    let values = builtin_frame_array(&[V::None, V::PandasNa]).unwrap();
    let first = frame(&values);
    let other = batch(vec![("datetime", index(2)), ("x", values.clone())], 2);
    for fail in 0..2 {
        let mut operations = FillFailure { fail, calls: 0 };
        let mut warnings = vec![];
        let error =
            super::append_with_operations(&first, &other, &mut operations, &mut |message| {
                warnings.push(message.to_owned());
            })
            .unwrap_err();
        assert!(error.to_string().ends_with("object fill failed"));
        assert_eq!(operations.calls, fail + 1);
        assert!(warnings.is_empty());
        assert!(Arc::ptr_eq(first.data().column(0), &values));
        assert!(Arc::ptr_eq(other.column(1), &values));
        let decoded = super::builtin_frame_values(&values).unwrap();
        assert!(matches!(decoded.as_slice(), [V::None, V::PandasNa]));
    }
}
