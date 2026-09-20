use std::{path::PathBuf, process::Command};

use domain_core::{
    DenseIndicatorTransform, DenseIndicatorValue, DenseMetric, DenseOrderIndicator,
    NumpyOrderIndicator, NumpyOrderIndicatorError, SingleData, SingleDataError, sum_by_index,
    transfer_dense,
};
use serde_json::{Value, json};

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

#[derive(Clone)]
struct FixedTransform {
    inputs: Vec<String>,
    output: Result<DenseIndicatorValue, String>,
}

impl DenseIndicatorTransform for FixedTransform {
    fn input_names(&self) -> &[String] {
        &self.inputs
    }

    fn apply(&self, inputs: &[&dyn DenseMetric]) -> Result<DenseIndicatorValue, String> {
        assert_eq!(inputs.len(), self.inputs.len());
        self.output.clone()
    }
}

#[test]
fn dense_storage_order_overwrite_and_snapshots_are_stable() {
    let mut indicator = NumpyOrderIndicator::new();
    indicator.assign("amount", data(&[("b", Some(1.0)), ("a", None)]));
    indicator.assign("flag", data(&[("a", Some(1.0)), ("b", Some(0.0))]));
    indicator.assign("amount", data(&[("c", Some(3.0))]));

    let storage: &dyn DenseOrderIndicator = &indicator;
    assert_eq!(
        storage.metric_names().collect::<Vec<_>>(),
        ["amount", "flag"]
    );
    assert_eq!(storage.metric("amount").unwrap().index(), ["c"]);
    assert!(storage.metric("missing").is_none());
    assert_eq!(values(&indicator.get_index_data("amount")), [Some(3.0)]);
    assert!(indicator.get_index_data("missing").is_empty());
    assert_eq!(
        values(&indicator.get_metric_series("flag").unwrap()),
        [Some(1.0), Some(0.0)]
    );
    assert!(matches!(
        indicator.get_metric_series("missing"),
        Err(NumpyOrderIndicatorError::MissingMetric(name)) if name == "missing"
    ));
    let snapshot = indicator.to_series();
    assert_eq!(
        snapshot.keys().map(String::as_str).collect::<Vec<_>>(),
        ["amount", "flag"]
    );
}

#[test]
fn dense_transfer_resolves_plugins_and_rejects_invalid_assignment() {
    let mut indicator = NumpyOrderIndicator::new();
    indicator.assign("amount", data(&[("a", Some(2.0))]));
    let transform = FixedTransform {
        inputs: vec!["amount".to_owned()],
        output: Ok(DenseIndicatorValue::Metric(data(&[("a", Some(4.0))]))),
    };
    assert!(
        transfer_dense(&mut indicator, &transform, Some("double"))
            .unwrap()
            .is_none()
    );
    assert_eq!(values(&indicator.get_index_data("double")), [Some(4.0)]);

    for value in [
        DenseIndicatorValue::Number(2.5),
        DenseIndicatorValue::Integer(2),
        DenseIndicatorValue::Boolean(true),
        DenseIndicatorValue::Missing,
    ] {
        let transform = FixedTransform {
            inputs: vec![],
            output: Ok(value),
        };
        assert!(
            transfer_dense(&mut indicator, &transform, None)
                .unwrap()
                .is_some()
        );
    }
    let missing = FixedTransform {
        inputs: vec!["unknown".to_owned()],
        output: Ok(DenseIndicatorValue::Missing),
    };
    assert!(matches!(
        transfer_dense(&mut indicator, &missing, None),
        Err(NumpyOrderIndicatorError::MissingMetric(name)) if name == "unknown"
    ));
    let failed = FixedTransform {
        inputs: vec![],
        output: Err("plugin failure".to_owned()),
    };
    assert!(matches!(
        transfer_dense(&mut indicator, &failed, None),
        Err(NumpyOrderIndicatorError::Transform(message)) if message == "plugin failure"
    ));
    let scalar = FixedTransform {
        inputs: vec![],
        output: Ok(DenseIndicatorValue::Number(1.0)),
    };
    assert!(matches!(
        transfer_dense(&mut indicator, &scalar, Some("bad")),
        Err(NumpyOrderIndicatorError::AssignedNonMetric)
    ));
}

#[test]
fn dense_sum_uses_first_metric_stock_union_and_numpy_fill_rules() {
    let mut first = NumpyOrderIndicator::new();
    first.assign("value", data(&[("a", Some(2.0)), ("c", None)]));
    first.assign("other", data(&[("a", None)]));
    let mut second = NumpyOrderIndicator::new();
    second.assign("value", data(&[("b", Some(1.0)), ("a", None)]));
    second.assign("other", data(&[("a", Some(2.0))]));

    let mut output = NumpyOrderIndicator::new();
    NumpyOrderIndicator::sum_all_indicators(
        &mut output,
        &[&first, &second],
        &["value", "other"],
        0.0,
    )
    .unwrap();
    let value = output.get_index_data("value");
    assert_eq!(value.index(), ["a", "b", "c"]);
    assert_eq!(values(&value), [Some(2.0), Some(1.0), Some(0.0)]);
    assert_eq!(
        values(&output.get_index_data("other")),
        [Some(2.0), Some(0.0), Some(0.0)]
    );

    NumpyOrderIndicator::sum_all_indicators(&mut output, &[], &["empty"], 0.0).unwrap();
    assert!(output.get_index_data("empty").is_empty());
    NumpyOrderIndicator::sum_all_indicators(&mut output, &[], &[], 0.0).unwrap();
    assert!(matches!(
        NumpyOrderIndicator::sum_all_indicators(&mut output, &[&first], &[], 0.0),
        Err(NumpyOrderIndicatorError::EmptyMetrics)
    ));

    let no_first = NumpyOrderIndicator::new();
    assert!(matches!(
        NumpyOrderIndicator::sum_all_indicators(
            &mut output,
            &[&no_first],
            &["value"],
            0.0
        ),
        Err(NumpyOrderIndicatorError::MissingMetric(name)) if name == "value"
    ));
    assert!(matches!(
        NumpyOrderIndicator::sum_all_indicators(
            &mut output,
            &[&first],
            &["value", "missing"],
            0.0
        ),
        Err(NumpyOrderIndicatorError::MissingMetric(name)) if name == "missing"
    ));

    let duplicate = vec!["x".to_owned(), "x".to_owned()];
    assert!(matches!(
        sum_by_index(&[&data(&[("x", Some(1.0))])], &duplicate, 0.0),
        Err(SingleDataError::DuplicateIndex(key)) if key == "x"
    ));
}

#[test]
fn contract_matches_live_python_numpy_indicator_and_single_data() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib");
    let indicator_source = root.join("qlib/backtest/high_performance_ds.py");
    let index_source = root.join("qlib/utils/index_data.py");
    let script = r"
import ast,importlib.util,inspect,json,sys
from collections import OrderedDict
from typing import *
import numpy as np,pandas as pd
spec=importlib.util.spec_from_file_location('index_data_live',sys.argv[2]);idd=importlib.util.module_from_spec(spec);spec.loader.exec_module(idd)
SingleData=idd.SingleData
class Logger: pass
def get_module_logger(_): return Logger()
t=ast.parse(open(sys.argv[1],encoding='utf-8').read(),filename=sys.argv[1])
nodes=[next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name==name) for name in ['BaseOrderIndicator','NumpyOrderIndicator']]
m=ast.Module(body=nodes,type_ignores=[]);ast.fix_missing_locations(m);exec(compile(m,sys.argv[1],'exec'),globals())
def snap(sd):return {'index':sd.index.tolist(),'values':[None if np.isnan(v) else float(v) for v in sd.data.tolist()]}
a=NumpyOrderIndicator();a.assign('x',{'b':1,'a':None});a.assign('flag',{'a':True,'b':False});a.assign('x',{'c':3})
b=NumpyOrderIndicator();b.assign('v',{'a':2,'c':None});b.assign('w',{'a':None})
c=NumpyOrderIndicator();c.assign('v',{'b':1,'a':None});c.assign('w',{'a':2})
out=NumpyOrderIndicator();NumpyOrderIndicator.sum_all_indicators(out,[b,c],['v','w'],0)
empty=NumpyOrderIndicator();NumpyOrderIndicator.sum_all_indicators(empty,[],['v'],0)
s1=idd.SingleData({'b':1,'a':None});s2=idd.SingleData({'a':2,'b':3})
result={'order':list(a.data),'x':snap(a.get_index_data('x')),'flag':snap(a.get_index_data('flag')),'missing':snap(a.get_index_data('z')),'sum_v':snap(out.get_index_data('v')),'sum_w':snap(out.get_index_data('w')),'empty':snap(empty.get_index_data('v')),'aligned':snap(s1+s2),'add':snap(s1.add(s2,0)),'reindex':snap(s1.reindex(idd.Index(['a','z','b']),8)),'cmp':snap(s1>0),'scalar':float(a.transfer(lambda x:x.sum()))}
a.transfer(lambda x:x+1,'next');result['next']=snap(a.get_index_data('next'))
for key,call in [('series_present',lambda:a.get_metric_series('x')),('series_missing',lambda:a.get_metric_series('z')),('to_series',a.to_series),('missing_transfer',lambda:a.transfer(lambda z:z)),('string_metric',lambda:NumpyOrderIndicator.sum_all_indicators(NumpyOrderIndicator(),[b,c],'value',0)),('fill_none',lambda:idd.sum_by_index([s1],['z'],None))]:
 try:call()
 except Exception as error:result[key]=type(error).__name__
print(json.dumps(result,sort_keys=True))
";
    let output = Command::new(std::env::var_os("PYTHON").unwrap_or_else(|| "python".into()))
        .arg("-c")
        .arg(script)
        .arg(indicator_source)
        .arg(index_source)
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
            "add":{"index":["a","b"],"values":[2.0,4.0]},
            "aligned":{"index":["b","a"],"values":[4.0,null]},
            "cmp":{"index":["b","a"],"values":[1.0,0.0]},
            "empty":{"index":[],"values":[]},"fill_none":"TypeError",
            "flag":{"index":["a","b"],"values":[1.0,0.0]},
            "missing":{"index":[],"values":[]},"missing_transfer":"KeyError",
            "next":{"index":["c"],"values":[4.0]},"order":["x","flag"],
            "reindex":{"index":["a","z","b"],"values":[null,8.0,1.0]},
            "scalar":3.0,"series_missing":"KeyError","series_present":"TypeError",
            "string_metric":"KeyError",
            "sum_v":{"index":["a","b","c"],"values":[2.0,1.0,0.0]},
            "sum_w":{"index":["a","b","c"],"values":[2.0,0.0,0.0]},
            "to_series":"TypeError","x":{"index":["c"],"values":[3.0]}
        })
    );
}
