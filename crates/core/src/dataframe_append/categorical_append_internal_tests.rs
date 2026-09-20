use super::super::ArrowOperations;
use super::*;
use arrow_array::{BinaryArray, RecordBatch, new_null_array};
use arrow_schema::{ArrowError, Field, TimeUnit};

#[test]
fn masked_category_native_float_lookup_uses_common_numeric_dtype() {
    use super::super::NullableIndexDescriptor;
    for (categories, value) in [
        (
            Arc::new(Int64Array::from(vec![9_007_199_254_740_993])) as ArrayRef,
            9_007_199_254_740_992.0,
        ),
        (
            Arc::new(arrow_array::UInt64Array::from(vec![u64::MAX])) as ArrayRef,
            18_446_744_073_709_551_616.0,
        ),
    ] {
        let category = Category::from_nullable_categories(
            NullableIndexDescriptor::new(categories).unwrap(),
            vec![0, -1],
            false,
        )
        .unwrap();
        let right: ArrayRef = Arc::new(arrow_array::Float64Array::from(vec![value]));
        let before = (category.storage().to_data(), right.to_data());
        let mut baseline = super::super::categorical_append_tests::Failure {
            at: usize::MAX,
            calls: 0,
        };
        let (_, metadata) = join(
            &category.storage(),
            &IndexMetadata::Categorical(Arc::new(category.clone())),
            &right,
            &IndexMetadata::Array,
            &mut baseline,
            &mut |_| panic!("compatible categories do not emit warnings"),
        )
        .unwrap();
        let IndexMetadata::Categorical(result) = metadata else {
            panic!("numeric lookup must retain compatible categories");
        };
        assert_eq!(result.codes(), vec![0, -1, 0]);
        assert_eq!(result.nullable_categories(), category.nullable_categories());
        assert_eq!((category.storage().to_data(), right.to_data()), before);
        for at in 1..=baseline.calls {
            let mut backend = super::super::categorical_append_tests::Failure { at, calls: 0 };
            assert_eq!(
                compatible_codes(&category, &right, None, &mut backend)
                    .unwrap_err()
                    .to_string(),
                "Compute error: categorical backend failed"
            );
            assert_eq!(backend.calls, at);
            assert_eq!((category.storage().to_data(), right.to_data()), before);
        }
        for normal in [0, 1] {
            assert_eq!(
                compatible_codes(&category, &right, None, &mut TruncatedCast(normal))
                    .unwrap_err()
                    .to_string(),
                "numeric lookup cast changed array length"
            );
            assert_eq!((category.storage().to_data(), right.to_data()), before);
        }
    }
}

#[test]
fn masked_category_finite_object_floats_preserve_category_identity() {
    use super::super::{NullableIndexDescriptor, tuple_frame_array};
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(vec![0, 1])),
        Arc::new(arrow_array::Float64Array::from(vec![0.0, 1.0])),
        Arc::new(arrow_array::BooleanArray::from(vec![false, true])),
    ];
    let right = tuple_frame_array(&[
        V::Scalar(T::Builtin(B::Float(1.0))),
        V::Scalar(T::Builtin(B::Float(0.0))),
        V::Scalar(T::Builtin(B::None)),
    ])
    .unwrap();
    for array in arrays {
        let category = Category::from_nullable_categories(
            NullableIndexDescriptor::new(array).unwrap(),
            vec![0, -1],
            false,
        )
        .unwrap();
        let before = (category.storage().to_data(), right.to_data());
        let (_, metadata) = join(
            &category.storage(),
            &IndexMetadata::Categorical(Arc::new(category.clone())),
            &right,
            &IndexMetadata::Array,
            &mut ArrowOperations,
            &mut |_| panic!("compatible categories do not emit warnings"),
        )
        .unwrap();
        let IndexMetadata::Categorical(result) = metadata else {
            panic!("finite objects must retain compatible categories");
        };
        assert_eq!(result.codes(), vec![0, -1, 1, 0, -1]);
        assert_eq!(result.nullable_categories(), category.nullable_categories());
        assert_eq!((category.storage().to_data(), right.to_data()), before);
    }
}

#[test]
fn masked_ordered_category_equality_preserves_dtype_identity_and_order() {
    use super::super::NullableIndexDescriptor;
    let masked = |values: ArrayRef| {
        Category::from_nullable_categories(
            NullableIndexDescriptor::new(values).unwrap(),
            vec![0, -1],
            true,
        )
        .unwrap()
    };
    let categories = [
        masked(Arc::new(Int64Array::from(vec![1, 2]))),
        masked(Arc::new(Int64Array::from(vec![1, 2]))),
        masked(Arc::new(Int64Array::from(vec![2, 1]))),
        masked(Arc::new(arrow_array::Int32Array::from(vec![1, 2]))),
        Category::new(Arc::new(Int64Array::from(vec![1, 2])), vec![0, -1], true).unwrap(),
    ];
    for (i, left) in categories.iter().enumerate() {
        for (j, right) in categories.iter().enumerate() {
            let before = (left.storage().to_data(), right.storage().to_data());
            assert_eq!(
                ordered_categories_equal(left, right, &mut ArrowOperations).unwrap(),
                i == j || (i < 2 && j < 2),
                "categories {i}, {j}"
            );
            assert_eq!(
                (left.storage().to_data(), right.storage().to_data()),
                before
            );
        }
    }
}

#[test]
fn empty_boolean_category_cast_failure_preserves_storage() {
    let descriptor = Category::new(new_null_array(&DataType::Boolean, 0), vec![-1], false).unwrap();
    let before = descriptor.storage().to_data();
    let mut backend = super::super::categorical_append_tests::Failure { at: 1, calls: 0 };
    assert_eq!(
        materialize(
            &descriptor.storage(),
            Some(&descriptor),
            &DataType::Boolean,
            &mut backend
        )
        .unwrap_err()
        .to_string(),
        "Compute error: categorical backend failed"
    );
    assert_eq!(backend.calls, 1);
    assert_eq!(descriptor.storage().to_data(), before);
}

#[test]
fn invalid_recode_and_unsupported_arrays_never_publish_partial_categories() {
    let categories: ArrayRef = Arc::new(Int64Array::from(vec![1, 2]));
    let descriptor = Category::new(categories, vec![0, -1], false).unwrap();
    let before = descriptor.storage().to_data();
    for codes in [vec![-2], vec![2], vec![0, 99]] {
        assert_eq!(
            merged(&descriptor, codes).unwrap_err().to_string(),
            "codes need to be between -1 and len(categories)-1"
        );
        assert_eq!(descriptor.storage().to_data(), before);
    }
    let foreign: ArrayRef = Arc::new(BinaryArray::from(vec![b"x".as_slice()]));
    let invalid = new_null_array(&tuple_frame_dtype(), 1);
    for array in [foreign, invalid] {
        let before = array.to_data();
        assert!(compatible_codes(&descriptor, &array, None, &mut ArrowOperations).is_err());
        assert!(materialize(&array, None, &tuple_frame_dtype(), &mut ArrowOperations).is_err());
        assert_eq!(array.to_data(), before);
    }
    assert!(matches!(
        common(&DataType::Binary, &DataType::Int64),
        Err(Error::DtypeAdapter(_, _))
    ));
    assert_eq!(
        common(
            &DataType::Duration(TimeUnit::Second),
            &DataType::Duration(TimeUnit::Nanosecond)
        )
        .unwrap(),
        DataType::Duration(TimeUnit::Nanosecond)
    );
    let foreign: ArrayRef = Arc::new(BinaryArray::from(vec![b"x".as_slice()]));
    assert!(matches!(
        join(
            &foreign,
            &IndexMetadata::Array,
            &descriptor.storage(),
            &IndexMetadata::Categorical(Arc::new(descriptor)),
            &mut ArrowOperations,
            &mut |_| {}
        ),
        Err(Error::DtypeAdapter(_, _))
    ));
}

struct TruncatedCast(usize);

impl FrameOperations for TruncatedCast {
    fn cast(&mut self, array: &ArrayRef, dtype: &DataType) -> Result<ArrayRef, ArrowError> {
        if self.0 > 0 {
            self.0 -= 1;
            return ArrowOperations.cast(array, dtype);
        }
        Ok(new_null_array(dtype, 0))
    }
    fn concat(&mut self, _: &ArrayRef, _: &ArrayRef) -> Result<ArrayRef, ArrowError> {
        panic!("invalid take must fail before concat")
    }
    fn batch(
        &mut self,
        _: Vec<Field>,
        _: Vec<ArrayRef>,
        _: usize,
    ) -> Result<RecordBatch, ArrowError> {
        panic!("invalid take must not publish a batch")
    }
}

#[test]
fn category_take_rejects_a_backend_that_truncates_the_category_array() {
    let descriptor =
        Category::new(Arc::new(Int64Array::from(vec![1])), vec![0, -1], false).unwrap();
    let storage = descriptor.storage();
    let before = storage.to_data();
    let error = materialize(
        &storage,
        Some(&descriptor),
        &DataType::Int64,
        &mut TruncatedCast(0),
    )
    .unwrap_err();
    assert!(
        error.to_string().to_lowercase().contains("out of bounds"),
        "{error}"
    );
    assert_eq!(storage.to_data(), before);
}

#[test]
fn temporal_category_equality_and_fallback_propagate_backend_errors() {
    use super::super::categorical_append_tests::Failure;
    let native: ArrayRef = Arc::new(arrow_array::TimestampNanosecondArray::from(vec![0]));
    let boxed = super::super::tuple_frame_array(&tuple_index::values(&native).unwrap()).unwrap();
    let left = Category::new(native, vec![0], true).unwrap();
    let right = Category::new(boxed, vec![0], true).unwrap();
    for (left, right) in [(&left, &right), (&right, &left)] {
        let error =
            ordered_categories_equal(left, right, &mut Failure { at: 1, calls: 0 }).unwrap_err();
        assert_eq!(
            error.to_string(),
            "Compute error: categorical backend failed"
        );
        let error = fallback(
            &left.storage(),
            &IndexMetadata::Categorical(Arc::new(left.clone())),
            &right.storage(),
            &IndexMetadata::Categorical(Arc::new(right.clone())),
            &mut Failure { at: 1, calls: 0 },
            &mut |_| {},
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "Compute error: categorical backend failed"
        );
    }
    let foreign: ArrayRef = Arc::new(BinaryArray::from(vec![b"x".as_slice()]));
    assert!(matches!(
        join(
            &foreign,
            &IndexMetadata::Array,
            &left.storage(),
            &IndexMetadata::Categorical(Arc::new(left)),
            &mut ArrowOperations,
            &mut |_| {}
        ),
        Err(Error::Arrow(_))
    ));
    let empty_int =
        Category::new(Arc::new(Int64Array::from(Vec::<i64>::new())), vec![], true).unwrap();
    let empty_time = Category::new(
        Arc::new(arrow_array::TimestampNanosecondArray::from(
            Vec::<i64>::new(),
        )),
        vec![],
        true,
    )
    .unwrap();
    assert!(!ordered_categories_equal(&empty_int, &empty_time, &mut ArrowOperations).unwrap());
    assert!(!ordered_categories_equal(&empty_time, &empty_int, &mut ArrowOperations).unwrap());
}
