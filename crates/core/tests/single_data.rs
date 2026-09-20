use domain_core::{DenseMetric, MetricBinaryOp, SingleData, SingleDataError};
use ndarray::Array1;

fn data(values: &[(&str, Option<f64>)]) -> SingleData {
    SingleData::from_f64(values.iter().copied()).unwrap()
}

fn values(metric: &dyn DenseMetric) -> Vec<Option<f64>> {
    metric
        .values()
        .iter()
        .map(|value| (!value.is_nan()).then_some(*value))
        .collect()
}

#[test]
fn construction_storage_and_reductions_are_numpy_compatible() {
    let metric = data(&[("b", Some(1.0)), ("a", None), ("c", Some(-3.0))]);
    let storage: &dyn DenseMetric = &metric;
    assert_eq!(storage.index(), ["b", "a", "c"]);
    assert_eq!(values(storage), [Some(1.0), None, Some(-3.0)]);
    assert_eq!(metric.len(), 3);
    assert!(!metric.is_empty());
    assert_eq!(metric.get("b"), Some(1.0));
    assert_eq!(metric.get("missing"), None);
    assert_eq!(metric.count(), 2);
    assert!((metric.sum() + 2.0).abs() < f64::EPSILON);
    assert!((metric.mean() + 1.0).abs() < f64::EPSILON);
    assert!(metric.all());
    assert!(!data(&[("x", Some(0.0))]).all());

    let empty = SingleData::default();
    assert!(empty.is_empty());
    assert!(empty.mean().is_nan());
    assert!(empty.all());
    let broadcast = SingleData::broadcast(2.0, &["x", "y"]).unwrap();
    assert_eq!(values(&broadcast), [Some(2.0), Some(2.0)]);

    assert!(matches!(
        SingleData::try_new(vec!["x".to_owned()], Array1::from_vec(vec![1.0, 2.0])),
        Err(SingleDataError::LengthMismatch {
            index: 1,
            values: 2
        })
    ));
    assert!(matches!(
        SingleData::from_f64([("x", Some(1.0)), ("x", Some(2.0))]),
        Err(SingleDataError::DuplicateIndex(key)) if key == "x"
    ));
    assert!(matches!(
        SingleData::broadcast(1.0, &["x", "x"]),
        Err(SingleDataError::DuplicateIndex(key)) if key == "x"
    ));
}

#[test]
fn scalar_binary_alignment_and_comparisons_cover_every_operation() {
    let lhs = data(&[("b", Some(1.0)), ("a", None)]);
    let scalar_cases = [
        (MetricBinaryOp::Add, vec![Some(3.0), None]),
        (MetricBinaryOp::Subtract, vec![Some(-1.0), None]),
        (MetricBinaryOp::Multiply, vec![Some(2.0), None]),
        (MetricBinaryOp::Divide, vec![Some(0.5), None]),
        (MetricBinaryOp::Equal, vec![Some(0.0), Some(0.0)]),
        (MetricBinaryOp::Greater, vec![Some(0.0), Some(0.0)]),
        (MetricBinaryOp::Less, vec![Some(1.0), Some(0.0)]),
    ];
    for (operation, expected) in scalar_cases {
        assert_eq!(values(&lhs.scalar(operation, 2.0)), expected);
    }
    assert_eq!(
        values(&lhs.reverse_scalar(MetricBinaryOp::Subtract, 2.0)),
        [Some(1.0), None]
    );
    assert_eq!(
        values(&lhs.reverse_scalar(MetricBinaryOp::Divide, 2.0)),
        [Some(2.0), None]
    );

    let same = data(&[("b", Some(3.0)), ("a", Some(2.0))]);
    let reordered = data(&[("a", Some(2.0)), ("b", Some(3.0))]);
    assert_eq!(
        values(&lhs.binary(MetricBinaryOp::Add, &same).unwrap()),
        [Some(4.0), None]
    );
    assert_eq!(
        values(&lhs.binary(MetricBinaryOp::Add, &reordered).unwrap()),
        [Some(4.0), None]
    );
    assert_eq!(
        values(&lhs.binary(MetricBinaryOp::Equal, &same).unwrap()),
        [Some(0.0), Some(0.0)]
    );
    assert_eq!(
        values(&lhs.binary(MetricBinaryOp::Greater, &same).unwrap()),
        [Some(0.0), Some(0.0)]
    );
    assert_eq!(
        values(&lhs.binary(MetricBinaryOp::Less, &same).unwrap()),
        [Some(1.0), Some(0.0)]
    );
    assert!(matches!(
        lhs.binary(MetricBinaryOp::Add, &data(&[("b", Some(1.0))])),
        Err(SingleDataError::IndexMismatch)
    ));
    assert!(matches!(
        lhs.binary(
            MetricBinaryOp::Add,
            &data(&[("b", Some(1.0)), ("z", Some(2.0))])
        ),
        Err(SingleDataError::IndexMismatch)
    ));
}

#[test]
fn alignment_transforms_and_missing_helpers_match_single_data() {
    let lhs = data(&[("b", Some(1.0)), ("a", None)]);
    let rhs = data(&[("a", Some(2.0)), ("b", Some(3.0)), ("c", None)]);
    let added = lhs.add(&rhs, 0.0).unwrap();
    assert_eq!(added.index(), ["a", "b", "c"]);
    assert_eq!(values(&added), [Some(2.0), Some(4.0), Some(0.0)]);

    let unchanged = lhs.reindex(&["b", "a"], 8.0).unwrap();
    assert_eq!(values(&unchanged), [Some(1.0), None]);
    let reindexed = lhs.reindex(&["a", "z", "b"], 8.0).unwrap();
    assert_eq!(values(&reindexed), [None, Some(8.0), Some(1.0)]);
    assert!(matches!(
        lhs.reindex(&["a", "a"], 0.0),
        Err(SingleDataError::DuplicateIndex(key)) if key == "a"
    ));

    assert_eq!(values(&lhs.abs()), [Some(1.0), None]);
    assert_eq!(
        values(&lhs.replace(&[(1.0, 7.0), (7.0, 9.0)])),
        [Some(7.0), None]
    );
    assert_eq!(
        values(
            &lhs.apply(&|items| Ok(items.iter().map(|item| item + 1.0).collect()))
                .unwrap()
        ),
        [Some(2.0), None]
    );
    assert!(matches!(
        lhs.apply(&|_| Err("boom".to_owned())),
        Err(SingleDataError::Transform(message)) if message == "boom"
    ));
    assert!(matches!(
        lhs.apply(&|_| Ok(vec![1.0])),
        Err(SingleDataError::TransformLengthMismatch {
            expected: 2,
            actual: 1
        })
    ));
    assert_eq!(values(&lhs.isna()), [Some(0.0), Some(1.0)]);
    assert_eq!(values(&lhs.fillna(5.0)), [Some(1.0), Some(5.0)]);
}
