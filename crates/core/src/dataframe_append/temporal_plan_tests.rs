use super::*;
use arrow_array::{StringArray, new_null_array};
use arrow_schema::TimeUnit;
use std::sync::Arc;

fn plan() -> Plan {
    Plan {
        positions: 0..1,
        dtype: None,
        warn: false,
        box_floats: false,
        fill: [None, None],
    }
}

#[test]
fn malformed_objects_and_unsupported_types_do_not_acquire_a_plan() {
    let bad = new_null_array(&temporal_frame_dtype(), 1);
    let time = new_null_array(&DataType::Duration(TimeUnit::Second), 1);
    let text = Arc::new(StringArray::from(vec!["x"])) as ArrayRef;
    for (left, right) in [(&bad, &time), (&time, &bad)] {
        let mut result = plan();
        assert!(
            configure(
                &mut result,
                std::slice::from_ref(left),
                std::slice::from_ref(right)
            )
            .is_err()
        );
        assert!(result.dtype.is_none());
        assert!(result.fill.iter().all(Option::is_none));
    }
    assert!(alignment_fill(&[bad]).is_err());
    for (left, right) in [(&text, &time), (&time, &text)] {
        let mut result = plan();
        assert!(
            !configure(
                &mut result,
                std::slice::from_ref(left),
                std::slice::from_ref(right)
            )
            .unwrap()
        );
        assert!(result.dtype.is_none());
    }
}

#[test]
fn numeric_missing_validity_and_public_planning_errors_are_explicit() {
    let missing = State {
        all_na: true,
        float_valid: true,
        first_none: false,
    };
    assert!(valid_na(&missing, &DataType::Float32, &DataType::Float64));
    assert!(!valid_na(&missing, &DataType::Float32, &DataType::Boolean));
    let bad = new_null_array(&temporal_frame_dtype(), 1);
    let index = Arc::new(arrow_array::Int64Array::from(vec![0])) as ArrayRef;
    let frame = super::super::IndexedFrame::new(
        index.clone(),
        None,
        super::super::block_tests::batch(vec![("x", bad.clone())], 1),
    )
    .unwrap();
    let right = super::super::block_tests::batch(
        vec![
            ("datetime", index),
            (
                "x",
                new_null_array(&DataType::Duration(TimeUnit::Second), 1),
            ),
        ],
        1,
    );
    let before = bad.to_data();
    let mut warnings = Vec::new();
    let error = super::super::dataframe_append_with_warnings(&frame, &right, &mut |message| {
        warnings.push(message.to_owned());
    })
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("typed null is not an explicit object sentinel")
    );
    assert!(warnings.is_empty());
    assert_eq!(frame.data().column(0).to_data(), before);
}
