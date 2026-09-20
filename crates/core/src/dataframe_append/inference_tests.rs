use super::*;
use crate::dataframe_append::{self as append, FrameColumnInput, IndexedFrame};
use indexmap::IndexMap;
use serde_json::{Value, json};
use std::{path::PathBuf, process::Command};

pub(crate) fn atom(value: &Value) -> V {
    match value[0].as_str().unwrap() {
        "none" => V::None,
        "pd.NA" => V::PandasNa,
        "NaT" => V::NotATime,
        "bool" => V::Bool(value[1].as_bool().unwrap()),
        "int" => {
            let value = value[1].as_str().unwrap();
            if value.starts_with('-') {
                V::Int(value.parse().unwrap())
            } else {
                V::UInt(value.parse().unwrap())
            }
        }
        "float" => V::Float(append::block_tests::expected_float(
            value[1].as_str().unwrap(),
        )),
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
        _ => panic!("unknown atom"),
    }
}

pub(crate) fn check(array: &ArrayRef, case: &Value) {
    let dtype = match case["dtype"].as_str().unwrap() {
        "object" => append::builtin_frame_dtype(),
        "datetime64[ns]" => DataType::Timestamp(TimeUnit::Nanosecond, None),
        "int64" => DataType::Int64,
        "uint64" => DataType::UInt64,
        "float64" => DataType::Float64,
        "bool" => DataType::Boolean,
        _ => panic!("unknown inferred dtype"),
    };
    assert_eq!(array.data_type(), &dtype, "{case}");
    if matches!(dtype, DataType::Timestamp(_, _)) {
        assert_eq!(array.null_count(), case["values"].as_array().unwrap().len());
        assert!(
            case["values"]
                .as_array()
                .unwrap()
                .iter()
                .all(|v| *v == json!(["NaT"]))
        );
    } else {
        append::block_tests::compare_values(array, &case["values"]);
    }
}

#[test]
fn inferred_lists_match_actual_source_values_types_and_append() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/dataframe_inference_contract.py"))
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
    assert_eq!(contract["ignored_right_verified"], true);
    assert_eq!(
        contract["digest"],
        "4ccd46f6b405c1f3232b3055ac91b6069d9a0c7e2ac6bbf76aea1d8b45d255d6"
    );
    let atoms = contract["atoms"]
        .as_array()
        .unwrap()
        .iter()
        .map(atom)
        .collect::<Vec<_>>();
    let cases = contract["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 6175);
    let empty = IndexedFrame::new(
        Arc::new(Int64Array::from(Vec::<i64>::new())),
        None,
        append::frame_from_columns(&IndexMap::new()).unwrap(),
    )
    .unwrap();
    for case in cases {
        let values = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| atoms[usize::try_from(v.as_u64().unwrap()).unwrap()].clone())
            .collect::<Vec<_>>();
        let array = infer_frame_values(&values).unwrap();
        check(&array, case);
        let index = Arc::new(Int64Array::from(
            (0..values.len())
                .map(|i| i64::try_from(i).unwrap())
                .collect::<Vec<_>>(),
        )) as ArrayRef;
        let columns = IndexMap::from([
            ("datetime".into(), FrameColumnInput::Array(index.clone())),
            ("x".into(), FrameColumnInput::Untyped(values)),
        ]);
        let other = append::frame_from_columns(&columns).unwrap();
        check(other.column(1), case);
        let mut warnings = vec![];
        let result = append::dataframe_append_with_warnings(&empty, &other, &mut |message| {
            warnings.push(json!(["FutureWarning", message]));
        })
        .unwrap();
        check(result.data().column(0), case);
        assert_eq!(result.index().to_data(), index.to_data());
        assert_eq!(result.index_name(), Some("datetime"));
        assert_eq!(result.blocks(), &[vec![0]]);
        assert_eq!(json!(warnings), case["warnings"]);
        let right_empty = append::frame_from_columns(&IndexMap::from([(
            "datetime".into(),
            FrameColumnInput::Array(index.slice(0, 0)),
        )]))
        .unwrap();
        let mut ignored_warnings = vec![];
        let unchanged =
            append::dataframe_append_with_warnings(&result, &right_empty, &mut |message| {
                ignored_warnings.push(message.to_owned());
            })
            .unwrap();
        check(unchanged.data().column(0), case);
        assert_eq!(unchanged.index().to_data(), result.index().to_data());
        assert!(ignored_warnings.is_empty());
    }
}
