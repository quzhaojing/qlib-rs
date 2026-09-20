use std::{path::PathBuf, process::Command, sync::Arc};

use arrow_array::{Array, ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use chrono::{NaiveDateTime, TimeDelta};
use serde::Deserialize;

use domain_core::{IndicatorAnalysisError, TradingIndicatorTable, indicator_analysis};

#[derive(Debug, Deserialize)]
struct Fixture {
    ast: String,
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    name: String,
    method: String,
    index: Vec<String>,
    columns: Vec<(String, Vec<String>)>,
    #[serde(default)]
    integer_columns: Vec<String>,
    #[serde(default)]
    native_integer_columns: Vec<String>,
    #[serde(default)]
    nullable_columns: Vec<String>,
    result: ResultSnapshot,
}

#[derive(Debug, Deserialize)]
struct ResultSnapshot {
    index: Option<Vec<String>>,
    columns: Option<Vec<String>>,
    value_bits: Option<Vec<String>>,
    value_dtype: Option<String>,
    error_type: Option<String>,
    error: Option<String>,
}

fn float(value: &str) -> f64 {
    match value {
        "nan" => f64::NAN,
        "inf" => f64::INFINITY,
        "-inf" => f64::NEG_INFINITY,
        value => value.parse().unwrap(),
    }
}

fn live_fixture() -> Fixture {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = std::env::var_os("QLIB_PYTHON_EVALUATE").map_or_else(
        || root.join("../../../qlib/qlib/contrib/evaluate.py"),
        PathBuf::from,
    );
    let output = Command::new(std::env::var_os("PYTHON").unwrap_or_else(|| "python".into()))
        .arg(root.join("tests/fixtures/trading_indicator_analysis_contract.py"))
        .arg(source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

fn table(case: &Case) -> TradingIndicatorTable {
    let columns = case
        .columns
        .iter()
        .map(|(name, values)| {
            let array: ArrayRef = if case.integer_columns.contains(name)
                || case.native_integer_columns.contains(name)
            {
                Arc::new(Int64Array::from(
                    values
                        .iter()
                        .map(|value| (value != "null").then(|| value.parse().unwrap()))
                        .collect::<Vec<_>>(),
                ))
            } else if case.nullable_columns.contains(name) {
                Arc::new(Float64Array::from(
                    values
                        .iter()
                        .map(|value| (value != "null").then(|| float(value)))
                        .collect::<Vec<_>>(),
                ))
            } else {
                Arc::new(Float64Array::from(
                    values.iter().map(|value| float(value)).collect::<Vec<_>>(),
                ))
            };
            (name.as_str(), array)
        })
        .collect::<Vec<_>>();
    let batch = RecordBatch::try_from_iter(columns).unwrap();
    let index = case
        .index
        .iter()
        .enumerate()
        .map(|(row, _)| {
            NaiveDateTime::default() + TimeDelta::microseconds(i64::try_from(row).unwrap())
        })
        .collect();
    TradingIndicatorTable::try_new(index, batch).unwrap()
}

fn bits(values: &Float64Array) -> Vec<String> {
    values
        .iter()
        .map(|value| format!("{:016x}", value.unwrap().to_bits()))
        .collect()
}

#[test]
fn values_schema_order_and_failures_match_live_source() {
    let fixture = live_fixture();
    assert!(
        fixture
            .ast
            .starts_with("FunctionDef(name='indicator_analysis'")
    );
    assert!(fixture.ast.contains("attr='abs'"));
    assert_eq!(fixture.cases.len(), 21);
    for case in fixture.cases {
        let result = indicator_analysis(&table(&case), &case.method);
        if let Some(expected) = case.result.value_bits {
            let result = result.unwrap_or_else(|error| panic!("{}: {error}", case.name));
            assert_eq!(
                result.index().as_slice(),
                case.result.index.unwrap(),
                "{}",
                case.name
            );
            assert_eq!(result.values().schema().fields().len(), 1);
            assert_eq!(
                result.values().schema().field(0).name(),
                &case.result.columns.unwrap()[0]
            );
            let values = result
                .values()
                .column(0)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            assert_eq!(bits(values), expected, "{}", case.name);
            assert!(
                matches!(
                    case.result.value_dtype.as_deref(),
                    Some("float64" | "Float64")
                ),
                "{}",
                case.name
            );
        } else {
            let error = result.unwrap_err();
            let expected_column = match case.name.as_str() {
                "missing_count_precedes_invalid_method" => Some("count"),
                "missing_deal_amount_precedes_invalid_method" => Some("deal_amount"),
                "missing_value_precedes_invalid_method" => Some("value"),
                "missing_ffr"
                | "missing_pa"
                | "missing_ffr_and_pa"
                | "invalid_method_precedes_missing_indicators" => None,
                "missing_pos_last" => Some("pos"),
                other => panic!("unexpected failure case {other}"),
            };
            if let Some(column) = expected_column {
                assert_eq!(error, IndicatorAnalysisError::MissingColumn { column });
                assert_eq!(case.result.error_type.as_deref(), Some("KeyError"));
            } else if case.name == "invalid_method_precedes_missing_indicators" {
                assert_eq!(
                    error,
                    IndicatorAnalysisError::UnsupportedMethod {
                        method: "bogus".to_owned()
                    }
                );
                assert_eq!(case.result.error_type.as_deref(), Some("ValueError"));
            } else {
                let columns = match case.name.as_str() {
                    "missing_ffr" => vec!["ffr"],
                    "missing_pa" => vec!["pa"],
                    "missing_ffr_and_pa" => vec!["ffr", "pa"],
                    other => panic!("unexpected indicator-column failure {other}"),
                };
                assert_eq!(
                    error,
                    IndicatorAnalysisError::MissingIndicatorColumns {
                        columns,
                        message: case.result.error.clone().unwrap(),
                    }
                );
                assert_eq!(case.result.error_type.as_deref(), Some("KeyError"));
            }
            assert!(case.result.error.is_some());
        }
    }
}

#[test]
fn typed_boundary_rejects_misalignment_and_non_numeric_columns_in_access_order() {
    let empty = RecordBatch::new_empty(Arc::new(Schema::empty()));
    assert_eq!(
        TradingIndicatorTable::try_new(vec![NaiveDateTime::default()], empty).unwrap_err(),
        IndicatorAnalysisError::LengthMismatch { index: 1, rows: 0 }
    );

    let fields = vec![
        Field::new("count", DataType::Utf8, false),
        Field::new("deal_amount", DataType::Float64, false),
        Field::new("value", DataType::Float64, false),
    ];
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(vec!["1"])),
        Arc::new(Float64Array::from(vec![1.0])),
        Arc::new(Float64Array::from(vec![1.0])),
    ];
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
    let table = TradingIndicatorTable::try_new(vec![NaiveDateTime::default()], batch).unwrap();
    assert_eq!(
        indicator_analysis(&table, "bogus").unwrap_err(),
        IndicatorAnalysisError::InvalidColumnType {
            column: "count",
            actual: DataType::Utf8
        }
    );

    for invalid in ["ffr", "pa"] {
        let columns = ["count", "deal_amount", "value", "ffr", "pa", "pos"]
            .into_iter()
            .map(|name| {
                let array: ArrayRef = if name == invalid {
                    Arc::new(StringArray::from(vec!["1"]))
                } else {
                    Arc::new(Float64Array::from(vec![1.0]))
                };
                (name, array)
            })
            .collect::<Vec<_>>();
        let table = TradingIndicatorTable::try_new(
            vec![NaiveDateTime::default()],
            RecordBatch::try_from_iter(columns).unwrap(),
        )
        .unwrap();
        assert_eq!(
            indicator_analysis(&table, "mean").unwrap_err(),
            IndicatorAnalysisError::InvalidColumnType {
                column: invalid,
                actual: DataType::Utf8,
            }
        );
    }
}

#[test]
fn input_index_and_columns_are_borrowed_without_mutation() {
    let fixture = live_fixture();
    let table = table(&fixture.cases[0]);
    let index = table.index().to_vec();
    let columns = table.columns().clone();
    indicator_analysis(&table, "mean").unwrap();
    assert_eq!(table.index(), index);
    assert_eq!(table.columns(), &columns);
}
