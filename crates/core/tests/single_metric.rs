use std::{path::PathBuf, process::Command, sync::Arc};

use arrow_array::{Array, BooleanArray, Float64Array, Int32Array, StringArray};
use domain_core::{
    MetricBinaryOp, MetricReplacement, MetricValue, PandasSingleMetric, SingleMetric,
    SingleMetricError,
};
use serde_json::{Value, json};

fn numbers(metric: &PandasSingleMetric) -> Vec<Option<f64>> {
    metric
        .values_ref()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .iter()
        .collect()
}

fn booleans(metric: &PandasSingleMetric) -> Vec<Option<bool>> {
    metric
        .values_ref()
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap()
        .iter()
        .collect()
}

fn keys(metric: &dyn SingleMetric) -> Vec<&str> {
    metric.index().iter().flatten().collect()
}

fn sample() -> PandasSingleMetric {
    PandasSingleMetric::from_f64([("b", Some(1.0)), ("a", None), ("c", Some(-3.0))]).unwrap()
}

#[test]
fn construction_reductions_and_storage_boundary_are_typed() {
    let metric = sample();
    let plugin: &dyn SingleMetric = &metric;
    assert_eq!(keys(plugin), ["b", "a", "c"]);
    assert_eq!(metric.len(), 3);
    assert!(!metric.is_empty());
    assert_eq!(metric.count(), 2);
    assert!((metric.sum() + 2.0).abs() < f64::EPSILON);
    assert!((metric.mean() + 1.0).abs() < f64::EPSILON);

    let empty = PandasSingleMetric::try_new(
        StringArray::from(Vec::<&str>::new()),
        Arc::new(Float64Array::from(Vec::<f64>::new())),
    )
    .unwrap();
    assert!(empty.is_empty());
    assert!(empty.sum().abs() < f64::EPSILON);
    assert!(empty.mean().is_nan());

    let integers = PandasSingleMetric::try_new(
        StringArray::from(vec!["x", "y", "z"]),
        Arc::new(Int32Array::from(vec![Some(2), None, Some(4)])),
    )
    .unwrap();
    assert_eq!(numbers(&integers), [Some(2.0), None, Some(4.0)]);

    let nan = PandasSingleMetric::try_new(
        StringArray::from(vec!["x"]),
        Arc::new(Float64Array::from(vec![f64::NAN])),
    )
    .unwrap();
    assert_eq!(nan.count(), 0);

    let flags = PandasSingleMetric::try_new(
        StringArray::from(vec!["x", "y", "z"]),
        Arc::new(BooleanArray::from(vec![Some(true), None, Some(false)])),
    )
    .unwrap();
    assert_eq!(flags.count(), 2);
    assert!((flags.sum() - 1.0).abs() < f64::EPSILON);
    assert!((flags.mean() - 0.5).abs() < f64::EPSILON);
    assert_eq!(
        booleans(&flags.abs().unwrap()),
        [Some(true), None, Some(false)]
    );

    assert!(matches!(
        PandasSingleMetric::try_new(
            StringArray::from(vec!["x"]),
            Arc::new(Float64Array::from(vec![1.0, 2.0]))
        ),
        Err(SingleMetricError::LengthMismatch {
            index: 1,
            values: 2
        })
    ));
    assert!(matches!(
        PandasSingleMetric::try_new(
            StringArray::from(vec![Some("x"), None]),
            Arc::new(Float64Array::from(vec![1.0, 2.0]))
        ),
        Err(SingleMetricError::NullIndex)
    ));
    assert!(matches!(
        PandasSingleMetric::try_new(
            StringArray::from(vec!["x", "x"]),
            Arc::new(Float64Array::from(vec![1.0, 2.0]))
        ),
        Err(SingleMetricError::DuplicateIndex(key)) if key == "x"
    ));
    assert!(matches!(
        PandasSingleMetric::try_new(
            StringArray::from(vec!["x"]),
            Arc::new(StringArray::from(vec!["text"]))
        ),
        Err(SingleMetricError::UnsupportedDataType(_))
    ));
}

#[test]
fn scalar_arithmetic_and_comparisons_match_pandas() {
    let metric = sample();
    let cases = [
        (MetricBinaryOp::Add, vec![Some(3.0), None, Some(-1.0)]),
        (MetricBinaryOp::Subtract, vec![Some(-1.0), None, Some(-5.0)]),
        (MetricBinaryOp::Multiply, vec![Some(2.0), None, Some(-6.0)]),
    ];
    for (operation, expected) in cases {
        assert_eq!(numbers(&metric.scalar(operation, 2.0).unwrap()), expected);
    }
    let divided = numbers(&metric.scalar(MetricBinaryOp::Divide, 0.0).unwrap());
    assert!(divided[0].unwrap().is_infinite());
    assert_eq!(divided[1], None);
    assert!(divided[2].unwrap().is_infinite());

    assert_eq!(
        booleans(&metric.scalar(MetricBinaryOp::Equal, 1.0).unwrap()),
        [Some(true), Some(false), Some(false)]
    );
    assert_eq!(
        booleans(&metric.scalar(MetricBinaryOp::Greater, 0.0).unwrap()),
        [Some(true), Some(false), Some(false)]
    );
    assert_eq!(
        booleans(&metric.scalar(MetricBinaryOp::Less, 0.0).unwrap()),
        [Some(false), Some(false), Some(true)]
    );

    assert_eq!(
        numbers(
            &metric
                .reverse_scalar(MetricBinaryOp::Subtract, 2.0)
                .unwrap()
        ),
        [Some(1.0), None, Some(5.0)]
    );
    assert_eq!(
        numbers(&metric.reverse_scalar(MetricBinaryOp::Add, 2.0).unwrap()),
        [Some(3.0), None, Some(-1.0)]
    );
    assert_eq!(
        numbers(
            &metric
                .reverse_scalar(MetricBinaryOp::Multiply, 2.0)
                .unwrap()
        ),
        [Some(2.0), None, Some(-6.0)]
    );
    assert_eq!(
        numbers(&metric.reverse_scalar(MetricBinaryOp::Divide, 3.0).unwrap()),
        [Some(3.0), None, Some(-1.0)]
    );
    assert_eq!(
        booleans(&metric.reverse_scalar(MetricBinaryOp::Greater, 0.0).unwrap()),
        [Some(false), Some(false), Some(true)]
    );
    assert_eq!(
        booleans(&metric.reverse_scalar(MetricBinaryOp::Less, 0.0).unwrap()),
        [Some(true), Some(false), Some(false)]
    );
    assert_eq!(
        booleans(&metric.reverse_scalar(MetricBinaryOp::Equal, 1.0).unwrap()),
        [Some(true), Some(false), Some(false)]
    );
}

#[test]
fn metric_alignment_filled_addition_and_comparisons_match_pandas() {
    let lhs = sample();
    let rhs =
        PandasSingleMetric::from_f64([("a", Some(10.0)), ("b", Some(20.0)), ("d", None)]).unwrap();
    let arithmetic = [
        (MetricBinaryOp::Add, vec![None, Some(21.0), None, None]),
        (
            MetricBinaryOp::Subtract,
            vec![None, Some(-19.0), None, None],
        ),
        (MetricBinaryOp::Multiply, vec![None, Some(20.0), None, None]),
        (MetricBinaryOp::Divide, vec![None, Some(0.05), None, None]),
    ];
    for (operation, expected) in arithmetic {
        let output = lhs.binary(operation, &rhs).unwrap();
        assert_eq!(keys(&output), ["a", "b", "c", "d"]);
        assert_eq!(numbers(&output), expected);
    }
    assert_eq!(
        numbers(&lhs.add_filled(&rhs, None).unwrap()),
        [None, Some(21.0), None, None]
    );
    assert_eq!(
        numbers(&lhs.add_filled(&rhs, Some(0.0)).unwrap()),
        [Some(10.0), Some(21.0), Some(-3.0), None]
    );

    let same =
        PandasSingleMetric::from_f64([("b", Some(1.0)), ("a", Some(2.0)), ("c", Some(-4.0))])
            .unwrap();
    assert_eq!(
        booleans(&lhs.binary(MetricBinaryOp::Equal, &same).unwrap()),
        [Some(true), Some(false), Some(false)]
    );
    assert_eq!(
        booleans(&lhs.binary(MetricBinaryOp::Greater, &same).unwrap()),
        [Some(false), Some(false), Some(true)]
    );
    assert_eq!(
        booleans(&lhs.binary(MetricBinaryOp::Less, &same).unwrap()),
        [Some(false), Some(false), Some(false)]
    );
    assert!(matches!(
        lhs.binary(MetricBinaryOp::Equal, &rhs),
        Err(SingleMetricError::ComparisonIndexMismatch)
    ));

    let same_order =
        PandasSingleMetric::from_f64([("b", Some(2.0)), ("a", Some(3.0)), ("c", Some(4.0))])
            .unwrap();
    let preserved = lhs.binary(MetricBinaryOp::Add, &same_order).unwrap();
    assert_eq!(keys(&preserved), ["b", "a", "c"]);

    let shorter = PandasSingleMetric::from_f64([("b", Some(2.0)), ("a", Some(3.0))]).unwrap();
    let unequal_lengths = lhs.binary(MetricBinaryOp::Add, &shorter).unwrap();
    assert_eq!(keys(&unequal_lengths), ["a", "b", "c"]);
}

#[test]
fn transformations_and_reindex_preserve_missing_value_rules() {
    let metric = sample();
    assert_eq!(
        numbers(&metric.abs().unwrap()),
        [Some(1.0), None, Some(3.0)]
    );
    let flags = metric.scalar(MetricBinaryOp::Greater, 0.0).unwrap();
    assert_eq!(
        booleans(&flags.abs().unwrap()),
        [Some(true), Some(false), Some(false)]
    );

    let replaced = metric
        .replace(&[
            MetricReplacement {
                from: MetricValue::Number(1.0),
                to: MetricValue::Number(7.0),
            },
            MetricReplacement {
                from: MetricValue::Missing,
                to: MetricValue::Number(9.0),
            },
        ])
        .unwrap();
    assert_eq!(numbers(&replaced), [Some(7.0), Some(9.0), Some(-3.0)]);

    let replaced_flags = flags
        .replace(&[MetricReplacement {
            from: MetricValue::Boolean(true),
            to: MetricValue::Boolean(false),
        }])
        .unwrap();
    assert_eq!(booleans(&replaced_flags), [Some(false); 3]);

    let applied = metric
        .apply(&|value| {
            Ok(match value {
                MetricValue::Missing => MetricValue::Number(100.0),
                MetricValue::Number(value) => MetricValue::Number(value * 2.0),
                MetricValue::Boolean(value) => MetricValue::Boolean(value),
            })
        })
        .unwrap();
    assert_eq!(numbers(&applied), [Some(2.0), Some(100.0), Some(-6.0)]);
    assert!(matches!(
        metric.apply(&|_| Err("boom".to_owned())),
        Err(SingleMetricError::Transform(message)) if message == "boom"
    ));
    assert!(matches!(
        metric.apply(&|value| Ok(if matches!(value, MetricValue::Missing) {
            MetricValue::Boolean(true)
        } else {
            value
        })),
        Err(SingleMetricError::MixedTransformTypes)
    ));

    let reindexed = metric.reindex(&["c", "x", "b"], Some(8.0)).unwrap();
    assert_eq!(keys(&reindexed), ["c", "x", "b"]);
    assert_eq!(numbers(&reindexed), [Some(-3.0), Some(8.0), Some(1.0)]);
    assert_eq!(numbers(&metric.reindex(&["x"], None).unwrap()), [None]);
}

#[test]
fn contract_matches_live_python_classes() {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../qlib/qlib/backtest/high_performance_ds.py");
    let script = r#"
import ast,json,math,sys
from typing import Union,Callable,Any,cast
import numpy as np,pandas as pd
t=ast.parse(open(sys.argv[1],encoding="utf-8").read(),filename=sys.argv[1])
nodes=[next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name==name) for name in ["BaseSingleMetric","SingleMetric","PandasSingleMetric"]]
m=ast.Module(body=nodes,type_ignores=[]);ast.fix_missing_locations(m);exec(compile(m,sys.argv[1],"exec"),globals())
def vals(metric):
 s=metric.metric
 return {"index":list(s.index),"values":[None if pd.isna(v) else (bool(v) if isinstance(v,(bool,np.bool_)) else float(v)) for v in s.tolist()]}
a=PandasSingleMetric(pd.Series([1.0,np.nan,-3.0],index=["b","a","c"]))
b=PandasSingleMetric(pd.Series([10.0,20.0,np.nan],index=["a","b","d"]))
out={
 "base":[len(a),float(a.sum()),float(a.mean()),int(a.count()),a.empty,a.index],
 "scalar":vals(a+2),"reverse":vals(2-a),"equal":vals(a==1),
 "binary":vals(a+b),"filled":vals(a.add(b,fill_value=0)),"absolute":vals(a.abs()),
 "replace":vals(a.replace({1.0:7.0,np.nan:9.0})),
 "apply":vals(a.apply(lambda v:100.0 if pd.isna(v) else v*2)),
 "reindex":vals(a.reindex(["c","x","b"],fill_value=8.0)),
 "empty":[float(PandasSingleMetric({}).sum()),math.isnan(PandasSingleMetric({}).mean())],
}
try: a==b
except Exception as error: out["comparison_error"]=type(error).__name__
print(json.dumps(out,sort_keys=True))
"#;
    let output = Command::new(std::env::var_os("PYTHON").unwrap_or_else(|| "python".into()))
        .arg("-c")
        .arg(script)
        .arg(source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let actual: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        actual,
        json!({
            "absolute":{"index":["b","a","c"],"values":[1.0,null,3.0]},
            "apply":{"index":["b","a","c"],"values":[2.0,100.0,-6.0]},
            "base":[3,-2.0,-1.0,2,false,["b","a","c"]],
            "binary":{"index":["a","b","c","d"],"values":[null,21.0,null,null]},
            "comparison_error":"ValueError",
            "empty":[0.0,true],
            "equal":{"index":["b","a","c"],"values":[true,false,false]},
            "filled":{"index":["a","b","c","d"],"values":[10.0,21.0,-3.0,null]},
            "reindex":{"index":["c","x","b"],"values":[-3.0,8.0,1.0]},
            "replace":{"index":["b","a","c"],"values":[7.0,9.0,-3.0]},
            "reverse":{"index":["b","a","c"],"values":[1.0,null,5.0]},
            "scalar":{"index":["b","a","c"],"values":[3.0,null,-1.0]}
        })
    );
}
