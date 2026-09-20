use super::*;
use arrow_array::{Array, DurationSecondArray, StringArray, TimestampSecondArray};
use arrow_schema::TimeUnit;
use std::sync::Arc;

#[test]
fn native_boxing_retains_ticks_zones_and_missing_sentinels() {
    for array in [
        Arc::new(
            TimestampSecondArray::from(vec![Some(1), None, Some(i64::MIN)]).with_timezone("UTC"),
        ) as ArrayRef,
        Arc::new(DurationSecondArray::from(vec![
            Some(1),
            None,
            Some(i64::MIN),
        ])),
    ] {
        let boxed = temporal_object_array(&array).unwrap();
        let values = temporal_frame_values(&boxed).unwrap();
        let first = if matches!(array.data_type(), DataType::Timestamp(_, _)) {
            V::Timestamp {
                ticks: 1,
                unit: TimeUnit::Second,
                timezone: Some("UTC".into()),
            }
        } else {
            V::Duration {
                ticks: 1,
                unit: TimeUnit::Second,
            }
        };
        assert_eq!(
            values,
            vec![first, V::Builtin(B::NotATime), V::Builtin(B::NotATime)]
        );
        assert!(Arc::ptr_eq(&boxed, &temporal_object_array(&boxed).unwrap()));
        let mut calls = 0;
        let error = promote_with(&array, &mut |_| {
            calls += 1;
            Err(ArrowError::ComputeError("ticks failed".into()))
        })
        .unwrap_err();
        assert_eq!(calls, 1);
        assert_eq!(error.to_string(), "Compute error: ticks failed");
        assert_eq!(array.len(), 3);
    }
    let text = Arc::new(StringArray::from(vec!["x"])) as ArrayRef;
    assert!(
        values(&text)
            .unwrap_err()
            .to_string()
            .contains("does not support Utf8")
    );
    assert!(
        temporal_object_array(&text)
            .unwrap_err()
            .to_string()
            .contains("does not support Utf8")
    );
    let bad = new_null_array(&super::super::builtin_frame_dtype(), 1);
    assert_eq!(
        temporal_object_array(&bad).unwrap_err().to_string(),
        "Invalid argument error: typed null is not an explicit object sentinel"
    );
    let malformed = new_null_array(&temporal_frame_dtype(), 1);
    assert_eq!(
        values(&malformed).unwrap_err().to_string(),
        "Invalid argument error: typed null is not an explicit object sentinel"
    );
}

#[test]
fn object_unboxing_requires_compatible_all_missing_values() {
    let missing = temporal_frame_array(&[
        V::Builtin(B::None),
        V::Builtin(B::PandasNa),
        V::Builtin(B::Float(f64::NAN)),
    ])
    .unwrap();
    for dtype in [
        DataType::Float32,
        DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
        DataType::Duration(TimeUnit::Microsecond),
    ] {
        let array = unbox_missing(&missing, &dtype).unwrap();
        assert_eq!(array.data_type(), &dtype);
        assert_eq!(array.null_count(), 3);
        let routed = super::super::objects::cast(&missing, &dtype).unwrap();
        assert_eq!(routed.to_data(), array.to_data());
    }
    let nat = temporal_frame_array(&[V::Builtin(B::NotATime)]).unwrap();
    assert!(unbox_missing(&nat, &DataType::Float64).is_err());
    assert!(unbox_missing(&missing, &DataType::Int64).is_err());
    let present = temporal_frame_array(&[V::Builtin(B::Int(1))]).unwrap();
    assert!(unbox_missing(&present, &DataType::Float64).is_err());
    assert!(unbox_missing(&present, &DataType::Duration(TimeUnit::Nanosecond)).is_err());
    let bad = new_null_array(&temporal_frame_dtype(), 1);
    assert!(unbox_missing(&bad, &DataType::Float64).is_err());
}
