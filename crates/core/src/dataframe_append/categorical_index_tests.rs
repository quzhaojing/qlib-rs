use super::super::{tuple_index_tests::array, tuple_objects::tests::compare};
use super::*;
use arrow_array::{RecordBatch, UInt32Array};
use arrow_ipc::{reader::StreamReader, writer::StreamWriter};
use arrow_schema::Schema;
use serde_json::{Value, json};
use std::{io::Cursor, path::PathBuf, process::Command};

#[test]
fn masked_category_precision_validates_original_cached_values() {
    use super::super::{ArrowOperations, IndexMetadata, NullableIndexDescriptor, nullable_append};
    let mut category = CategoricalIndexDescriptor::from_nullable_categories(
        NullableIndexDescriptor::new(Arc::new(Int64Array::from(vec![9_007_199_254_740_993])))
            .unwrap(),
        vec![0, -1],
        false,
    )
    .unwrap();
    category.materialized_categories[0] = V::Scalar(T::Duration {
        ticks: i64::MIN,
        unit: arrow_schema::TimeUnit::Second,
    });
    let right = NullableIndexDescriptor::new(Arc::new(Int64Array::from(vec![1]))).unwrap();
    let before = (category.storage().to_data(), right.storage().to_data());
    let error = super::super::MultiIndexDescriptor::from_levels(
        vec![super::super::MultiIndexLevel::Categorical(category.clone())],
        vec![vec![0, -1]],
        vec![V::Scalar(T::Builtin(B::None))],
        None,
    )
    .unwrap_err();
    assert_eq!(
        error.to_string(),
        "Invalid argument error: temporal object uses reserved NaT ticks"
    );
    let error = nullable_append::fallback(
        &category.storage(),
        &IndexMetadata::Categorical(Arc::new(category.clone())),
        &right.storage(),
        &IndexMetadata::Nullable(Arc::new(right.clone())),
        &mut ArrowOperations,
        &mut |_| panic!("nonempty append cannot emit empty warnings"),
    )
    .unwrap_err();
    assert_eq!(
        error.to_string(),
        "Invalid argument error: temporal object uses reserved NaT ticks"
    );
    assert_eq!(
        (category.storage().to_data(), right.storage().to_data()),
        before
    );
}

#[test]
fn invalid_cached_category_rows_are_not_masked_by_valid_dictionary_keys() {
    use super::super::{ArrowOperations, IndexMetadata, categorical_append};
    let left = CategoricalIndexDescriptor::new(Arc::new(Int64Array::from(vec![1])), vec![0], false)
        .unwrap();
    let mut right = left.clone();
    right.materialized_categories[0] = V::Scalar(T::Duration {
        ticks: i64::MIN,
        unit: arrow_schema::TimeUnit::Second,
    });
    let before = (left.storage().to_data(), right.storage().to_data());
    let error = categorical_append::join(
        &left.storage(),
        &IndexMetadata::Categorical(Arc::new(left.clone())),
        &right.storage(),
        &IndexMetadata::Categorical(Arc::new(right.clone())),
        &mut ArrowOperations,
        &mut |_| panic!("invalid rows cannot emit warnings"),
    )
    .unwrap_err();
    assert_eq!(
        error.to_string(),
        "Invalid argument error: temporal object uses reserved NaT ticks"
    );
    assert_eq!(
        (left.storage().to_data(), right.storage().to_data()),
        before
    );
}

#[test]
fn categorical_frame_metadata_validates_storage_and_row_counts() {
    use super::super::{IndexMetadata, IndexedFrame, block_tests::batch};
    let categories: ArrayRef = Arc::new(Int64Array::from(vec![2, 1, 99]));
    let descriptor =
        CategoricalIndexDescriptor::new(categories.clone(), vec![1, -1], true).unwrap();
    let storage = descriptor.storage();
    let before = storage.to_data();
    let metadata = IndexMetadata::Categorical(Arc::new(descriptor.clone()));
    let frame = IndexedFrame::new(storage.clone(), Some("history".into()), batch(vec![], 2))
        .unwrap()
        .with_index_metadata(metadata.clone())
        .unwrap();
    assert_eq!(frame.index_metadata(), &metadata);
    assert_eq!(frame.index_name(), Some("history"));
    for index in [
        categories.clone(),
        CategoricalIndexDescriptor::new(categories, vec![0, -1], true)
            .unwrap()
            .storage(),
    ] {
        let len = index.len();
        let error = IndexedFrame::new(index, None, batch(vec![], len))
            .unwrap()
            .with_index_metadata(metadata.clone())
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "invalid index metadata: CategoricalIndex descriptor does not match dictionary storage"
        );
    }
    assert!(matches!(
        IndexedFrame::from_categorical_index(descriptor, None, batch(vec![], 1)),
        Err(DataframeAppendError::IndexLength)
    ));
    assert_eq!(storage.to_data(), before);
}

#[test]
fn categorical_frame_import_and_ignored_empty_appends_match_actual_source() {
    use super::super::{
        IndexMetadata, IndexedFrame, block_tests::batch, dataframe_append_with_warnings,
        tuple_index_tests::ordinal,
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/dataframe_categorical_import.py"))
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
        "36e63d8fc1e23037579ab443c400e624d6ca959f7ac97b05861a70c6f77f7143"
    );
    let cases = contract["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 556);
    let mut imported = 0;
    for case in cases {
        let initial = &case["initial"]["index"];
        let codes: Vec<i64> = initial["codes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let rows = codes.len();
        let descriptor = CategoricalIndexDescriptor::new(
            array(&initial["categories"], false),
            codes,
            initial["ordered"] == true,
        )
        .unwrap();
        for outcome in case["right_import"].as_array().unwrap() {
            imported += 1;
            check_right_import(&descriptor, outcome);
        }
        let name = case["name"].as_str().map(str::to_owned);
        let mut current = IndexedFrame::from_categorical_index(
            descriptor,
            name.clone(),
            batch(vec![("x", ordinal(rows))], rows),
        )
        .unwrap();
        let empty = super::super::tuple_frame_array(&[]).unwrap();
        let other = batch(vec![("datetime", empty)], 0);
        for phase in ["first", "second"] {
            let before = (
                current.index().to_data(),
                current.data().clone(),
                current.index_metadata().clone(),
            );
            let mut warnings = vec![];
            let next = dataframe_append_with_warnings(&current, &other, &mut |w| {
                warnings.push(w.to_owned());
            })
            .unwrap();
            assert_eq!(json!(warnings), case[phase]["warnings"]);
            let expected = &case[phase]["output"];
            assert_eq!(expected["index"]["kind"], "CategoricalIndex");
            let IndexMetadata::Categorical(descriptor) = next.index_metadata() else {
                panic!("categorical identity lost: {case}")
            };
            check(descriptor, &expected["index"]);
            assert_eq!(next.index_name(), name.as_deref());
            assert_eq!(next.index().to_data(), descriptor.storage().to_data());
            assert_eq!(next.data().num_rows(), rows);
            assert_eq!(next.data().num_columns(), 1);
            assert_eq!(next.data().schema().field(0).name(), "x");
            let expected_data = super::super::temporal_append_tests::column(
                expected["frame"]["dtypes"][0].as_str().unwrap(),
                &expected["frame"]["values"][0],
                0,
            );
            assert_eq!(next.data().column(0).to_data(), expected_data.to_data());
            assert_eq!(
                next.column_axis(),
                super::super::EmptyColumnAxis::ObjectIndex
            );
            assert_eq!(next.blocks(), &[vec![0]]);
            assert_eq!(
                (
                    current.index().to_data(),
                    current.data().clone(),
                    current.index_metadata().clone()
                ),
                before
            );
            current = next;
        }
    }
    assert_eq!(imported, 1028);
}

fn check_right_import(descriptor: &CategoricalIndexDescriptor, outcome: &Value) {
    use super::super::{
        IndexMetadata, IndexedFrame, block_tests::batch, dataframe_append_with_warnings,
        tuple_index_tests::ordinal,
    };
    let rows = descriptor.codes().len();
    let mut fields = vec![descriptor.field("datetime")];
    let mut arrays = vec![descriptor.storage()];
    if outcome["columns"] == true {
        let values = ordinal(rows);
        fields.push(arrow_schema::Field::new(
            "x",
            values.data_type().clone(),
            true,
        ));
        arrays.push(values);
    }
    let other = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap();
    let before = other.clone();
    let left = IndexedFrame::new(
        Arc::new(Int64Array::from(Vec::<i64>::new())),
        None,
        batch(vec![], 0),
    )
    .unwrap();
    let mut warnings = vec![];
    let result = dataframe_append_with_warnings(&left, &other, &mut |w| {
        warnings.push(w.to_owned());
    })
    .unwrap();
    assert_eq!(json!(warnings), outcome["warnings"]);
    let IndexMetadata::Categorical(restored) = result.index_metadata() else {
        panic!("right categorical identity lost: {outcome}")
    };
    check(restored, &outcome["output"]["index"]);
    assert_eq!(result.index_name(), Some("datetime"));
    assert_eq!(result.index().to_data(), descriptor.storage().to_data());
    assert_eq!(result.data().num_rows(), rows);
    assert_eq!(
        result.data().num_columns(),
        usize::from(outcome["columns"] == true)
    );
    if outcome["columns"] == true {
        assert_eq!(result.data().column(0).to_data(), ordinal(rows).to_data());
        assert_eq!(result.data().schema().field(0).name(), "x");
    }
    assert_eq!(
        result.column_axis(),
        super::super::EmptyColumnAxis::ObjectIndex
    );
    assert_eq!(other, before);
}

#[test]
fn invalid_categorical_right_column_fails_before_publication() {
    use super::super::{IndexedFrame, block_tests::batch, dataframe_append_with_warnings};
    let categories: ArrayRef = Arc::new(Int64Array::from(vec![1, 1]));
    let invalid: ArrayRef = Arc::new(
        DictionaryArray::<Int64Type>::try_new(Int64Array::from(vec![0]), categories).unwrap(),
    );
    let other = batch(vec![("datetime", invalid)], 1);
    let before = other.clone();
    let left = IndexedFrame::new(
        Arc::new(Int64Array::from(Vec::<i64>::new())),
        None,
        batch(vec![], 0),
    )
    .unwrap();
    let mut warnings = vec![];
    let error = dataframe_append_with_warnings(&left, &other, &mut |w| {
        warnings.push(w.to_owned());
    })
    .unwrap_err();
    assert_eq!(error.to_string(), "Categorical categories must be unique");
    assert!(warnings.is_empty());
    assert_eq!(other, before);
    assert_eq!(left.data().num_rows(), 0);
}

#[test]
fn categorical_storage_rejects_foreign_or_invalid_imports_without_mutation() {
    use arrow_array::{Int8Array, new_null_array, types::Int8Type};
    use arrow_schema::{DataType, TimeUnit};
    let invalid_cell = V::Scalar(T::Duration {
        ticks: i64::MIN,
        unit: TimeUnit::Second,
    });
    assert_eq!(
        validate_values(&[invalid_cell]).unwrap_err().to_string(),
        "Invalid argument error: temporal object uses reserved NaT ticks"
    );
    let categories: ArrayRef = Arc::new(Int64Array::from(vec![2, 1, 99]));
    let descriptor =
        CategoricalIndexDescriptor::new(categories.clone(), vec![1, -1, 0], true).unwrap();
    assert_eq!(descriptor, descriptor.clone());
    assert_ne!(
        descriptor,
        CategoricalIndexDescriptor::new(categories.clone(), vec![1, -1, 0], false).unwrap()
    );
    assert_ne!(
        descriptor,
        CategoricalIndexDescriptor::new(categories.clone(), vec![0, -1, 1], true).unwrap()
    );
    let storage = descriptor.storage();
    let before = storage.to_data();
    assert_eq!(
        CategoricalIndexDescriptor::from_storage(&storage, &Field::new("x", DataType::Int64, true))
            .unwrap_err()
            .to_string(),
        "invalid categorical storage: field dtype does not match array"
    );
    let foreign: ArrayRef = Arc::new(
        DictionaryArray::<Int8Type>::try_new(Int8Array::from(vec![0]), categories).unwrap(),
    );
    assert_eq!(
        CategoricalIndexDescriptor::from_storage(
            &foreign,
            &Field::new("x", foreign.data_type().clone(), true)
        )
        .unwrap_err()
        .to_string(),
        "invalid categorical storage: expected canonical Int64 dictionary"
    );
    let invalid: ArrayRef = Arc::new(
        DictionaryArray::<Int64Type>::try_new(
            Int64Array::from(vec![0]),
            new_null_array(&DataType::Int64, 1),
        )
        .unwrap(),
    );
    assert!(matches!(
        CategoricalIndexDescriptor::from_storage(
            &invalid,
            &Field::new("x", invalid.data_type().clone(), true)
        ),
        Err(CategoricalIndexError::NullCategories)
    ));
    for invalid in [
        new_null_array(&DataType::Float16, 0),
        new_null_array(&DataType::Binary, 0),
        new_null_array(&super::super::tuple_frame_dtype(), 1),
    ] {
        assert!(CategoricalIndexDescriptor::new(invalid, vec![], false).is_err());
    }
    assert_eq!(storage.to_data(), before);
    let joined = arrow_select::concat::concat(&[storage.as_ref(), storage.as_ref()]).unwrap();
    let joined = CategoricalIndexDescriptor::from_storage(&joined, &descriptor.field("x")).unwrap();
    assert_eq!(joined.codes(), vec![1, -1, 0, 1, -1, 0]);
    assert_eq!(
        joined.categories().to_data(),
        descriptor.categories().to_data()
    );
    assert!(joined.ordered());
    let datetime: ArrayRef =
        Arc::new(arrow_array::TimestampSecondArray::from(vec![0]).with_timezone("UTC"));
    let temporal = CategoricalIndexDescriptor::new(datetime, vec![-1, 0], false).unwrap();
    assert_eq!(temporal.values()[0], V::Scalar(T::Builtin(B::NotATime)));
    assert_eq!(
        temporal.values()[1],
        V::Scalar(T::Timestamp {
            ticks: 0,
            unit: TimeUnit::Second,
            timezone: Some("UTC".into())
        })
    );
}

fn check(descriptor: &CategoricalIndexDescriptor, expected: &Value) {
    assert_eq!(json!(descriptor.codes()), expected["codes"]);
    assert_eq!(json!(descriptor.ordered()), expected["ordered"]);
    let wanted = if let Some(masked) = expected["categories"].get("masked") {
        let nullable =
            NullableIndexDescriptor::new(super::super::nullable_index::tests::array(masked))
                .unwrap();
        assert_eq!(descriptor.nullable_categories(), Some(&nullable));
        nullable.storage()
    } else {
        assert!(descriptor.nullable_categories().is_none());
        array(&expected["categories"], false)
    };
    assert_eq!(descriptor.categories().data_type(), wanted.data_type());
    compare(
        &tuple_index::values(descriptor.categories()).unwrap(),
        expected["categories"]["values"].as_array().unwrap(),
    );
    compare(&descriptor.values(), expected["values"].as_array().unwrap());
}

#[test]
fn nullable_categorical_storage_and_import_match_actual_source() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/dataframe_categorical_nullable_storage.py"))
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
        "7eb9bc100d0b40a664b3195169c2d53bf4b626010f83af8653d46cfed34040c0"
    );
    let cases = contract["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 1220);
    let mut valid = 0;
    for case in cases {
        let categories = NullableIndexDescriptor::new(super::super::nullable_index::tests::array(
            &case["categories"]["masked"],
        ))
        .unwrap();
        let before = categories.clone();
        let codes = case["codes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let result = CategoricalIndexDescriptor::from_nullable_categories(
            categories.clone(),
            codes,
            case["ordered"] == true,
        );
        assert_eq!(categories, before);
        assert_eq!(case["warnings"], json!([]));
        if let Some(message) = case.get("message") {
            assert_eq!(case["error"], "ValueError");
            assert_eq!(
                result.unwrap_err().to_string(),
                message.as_str().unwrap(),
                "{case}"
            );
            continue;
        }
        valid += 1;
        let descriptor = result.unwrap();
        check(&descriptor, &case["output"]);
        nullable_category_roundtrip(&descriptor, &case["output"]);
        nullable_category_ignored_append(&descriptor, case);
        for outcome in case["right_import"].as_array().unwrap() {
            check_right_import(&descriptor, outcome);
        }
    }
    assert_eq!(valid, 220);
}

fn nullable_category_roundtrip(descriptor: &CategoricalIndexDescriptor, expected: &Value) {
    let schema = Arc::new(Schema::new(vec![descriptor.field("datetime")]));
    let batch = RecordBatch::try_new(schema.clone(), vec![descriptor.storage()]).unwrap();
    let mut bytes = vec![];
    let mut writer = StreamWriter::try_new(&mut bytes, &schema).unwrap();
    writer.write(&batch).unwrap();
    writer.finish().unwrap();
    let decoded = StreamReader::try_new(Cursor::new(bytes), None)
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    let restored =
        CategoricalIndexDescriptor::from_storage(decoded.column(0), decoded.schema().field(0))
            .unwrap();
    assert_eq!(descriptor, &restored);
    check(&restored, expected);
    let storage = descriptor.storage();
    for slice in [
        storage.clone(),
        storage.slice(storage.len().min(1), storage.len().saturating_sub(1)),
    ] {
        let restored =
            CategoricalIndexDescriptor::from_storage(&slice, &descriptor.field("datetime"))
                .unwrap();
        assert_eq!(restored.storage().to_data(), slice.to_data());
        assert_eq!(
            restored.nullable_categories(),
            descriptor.nullable_categories()
        );
    }
}

#[test]
fn nullable_category_metadata_rejects_malformed_imports_and_distinguishes_identity() {
    use arrow_schema::DataType;
    let categories: ArrayRef = Arc::new(Int64Array::from(vec![1]));
    let ordinary = CategoricalIndexDescriptor::new(categories.clone(), vec![0, -1], false).unwrap();
    let masked = CategoricalIndexDescriptor::from_nullable_categories(
        NullableIndexDescriptor::new(categories).unwrap(),
        vec![0, -1],
        false,
    )
    .unwrap();
    assert_eq!(ordinary.storage().to_data(), masked.storage().to_data());
    assert_ne!(ordinary, masked);
    let storage = masked.storage();
    let before = storage.to_data();
    let field = masked
        .field("x")
        .with_metadata([(MASKED_CATEGORIES.into(), "UInt64".into())].into());
    assert_eq!(
        CategoricalIndexDescriptor::from_storage(&storage, &field)
            .unwrap_err()
            .to_string(),
        "invalid categorical storage: masked category dtype does not match array"
    );
    let invalid: ArrayRef = Arc::new(
        DictionaryArray::<Int64Type>::try_new(
            Int64Array::from(vec![0]),
            arrow_array::new_null_array(&DataType::Int64, 1),
        )
        .unwrap(),
    );
    assert_eq!(
        CategoricalIndexDescriptor::from_storage(&invalid, &masked.field("x"))
            .unwrap_err()
            .to_string(),
        "Categorical categories cannot be null"
    );
    let foreign: ArrayRef = Arc::new(
        DictionaryArray::<Int64Type>::try_new(
            Int64Array::from(vec![0]),
            Arc::new(arrow_array::StringArray::from(vec!["x"])),
        )
        .unwrap(),
    );
    let field = Field::new("x", foreign.data_type().clone(), true)
        .with_metadata([(MASKED_CATEGORIES.into(), "string".into())].into());
    assert_eq!(
        CategoricalIndexDescriptor::from_storage(&foreign, &field)
            .unwrap_err()
            .to_string(),
        "unsupported masked index storage: Utf8"
    );
    assert_eq!(storage.to_data(), before);
}

fn nullable_category_ignored_append(descriptor: &CategoricalIndexDescriptor, case: &Value) {
    use super::super::{
        IndexedFrame, block_tests, temporal_append_tests, tuple_frame_array, tuple_index_tests,
    };
    let source = &case["first"]["output"]["frame"];
    let data = temporal_append_tests::column(
        source["dtypes"][0].as_str().unwrap(),
        &source["values"][0],
        0,
    );
    let initial = IndexedFrame::from_categorical_index(
        descriptor.clone(),
        Some("datetime".into()),
        block_tests::batch(vec![("x", data)], descriptor.codes().len()),
    )
    .unwrap();
    let right = block_tests::batch(vec![("datetime", tuple_frame_array(&[]).unwrap())], 0);
    let mut current = initial.clone();
    for step in ["first", "second"] {
        current = tuple_index_tests::append_batch(&current, &right, &case[step]);
        assert_eq!(current.index_metadata(), initial.index_metadata());
        assert_eq!(current.data(), initial.data());
    }
}

#[test]
fn categorical_codes_and_categories_match_actual_source() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .arg(root.join("tests/fixtures/dataframe_categorical_codes.py"))
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
        "e68d833926ac25d113e3498042e693af7d56314972532ba5bf3e1b14a8614782"
    );
    let cases = contract["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 620);
    let mut successful = 0;
    for case in cases {
        assert_eq!(case["warnings"], json!([]));
        let categories = array(&case["categories"], false);
        let before = categories.to_data();
        let codes = case["codes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let result =
            CategoricalIndexDescriptor::new(categories.clone(), codes, case["ordered"] == true);
        assert_eq!(categories.to_data(), before);
        if let Some(message) = case.get("message") {
            assert_eq!(case["error"], "ValueError");
            assert_eq!(
                result.unwrap_err().to_string(),
                message.as_str().unwrap(),
                "{case}"
            );
        } else {
            successful += 1;
            let descriptor = result.unwrap();
            check(&descriptor, &case["output"]);
            let schema = Arc::new(Schema::new(vec![descriptor.field("datetime")]));
            let batch = RecordBatch::try_new(schema.clone(), vec![descriptor.storage()]).unwrap();
            let mut bytes = vec![];
            let mut writer = StreamWriter::try_new(&mut bytes, &schema).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
            let decoded = StreamReader::try_new(Cursor::new(bytes), None)
                .unwrap()
                .next()
                .unwrap()
                .unwrap();
            let restored = CategoricalIndexDescriptor::from_storage(
                decoded.column(0),
                decoded.schema().field(0),
            )
            .unwrap();
            check(&restored, &case["output"]);
            assert_eq!(descriptor, restored);
            assert_eq!(
                restored.field("datetime").dict_is_ordered(),
                Some(case["ordered"] == true)
            );
            if !descriptor.codes().is_empty() {
                let sliced = descriptor.storage().slice(1, descriptor.codes().len() - 1);
                let sliced = CategoricalIndexDescriptor::from_storage(
                    &sliced,
                    &descriptor.field("datetime"),
                )
                .unwrap();
                assert_eq!(sliced.codes(), descriptor.codes()[1..]);
                compare(
                    &sliced.values(),
                    &case["output"]["values"].as_array().unwrap()[1..],
                );
                let taken = arrow_select::take::take(
                    descriptor.storage().as_ref(),
                    &UInt32Array::from(vec![0, 0]),
                    None,
                )
                .unwrap();
                let taken =
                    CategoricalIndexDescriptor::from_storage(&taken, &descriptor.field("datetime"))
                        .unwrap();
                compare(
                    &taken.values(),
                    &[
                        case["output"]["values"][0].clone(),
                        case["output"]["values"][0].clone(),
                    ],
                );
            }
        }
    }
    assert_eq!(successful, 278);
}
