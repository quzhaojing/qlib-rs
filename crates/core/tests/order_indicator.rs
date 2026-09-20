use std::{path::PathBuf, process::Command, sync::Arc};

use arrow_array::{Array, BooleanArray, Float64Array, StringArray};
use domain_core::{
    IndicatorValue, OrderIndicator, OrderIndicatorError, OrderIndicatorTransform,
    PandasOrderIndicator, PandasSingleMetric, SingleMetric, transfer,
};
use serde_json::{Value, json};

fn metric(values: &[(&str, Option<f64>)]) -> PandasSingleMetric {
    PandasSingleMetric::from_f64(values.iter().copied()).unwrap()
}

fn keys(metric: &dyn SingleMetric) -> Vec<&str> {
    metric.index().iter().flatten().collect()
}

fn numbers(metric: &PandasSingleMetric) -> Vec<Option<f64>> {
    metric
        .values_ref()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .iter()
        .collect()
}

#[derive(Clone)]
struct FixedTransform {
    inputs: Vec<String>,
    output: Result<IndicatorValue, String>,
}

impl OrderIndicatorTransform for FixedTransform {
    fn input_names(&self) -> &[String] {
        &self.inputs
    }

    fn apply(&self, inputs: &[&dyn SingleMetric]) -> Result<IndicatorValue, String> {
        assert_eq!(inputs.len(), self.inputs.len());
        self.output.clone()
    }
}

#[test]
fn ordered_storage_overwrite_and_access_match_pandas() {
    let mut indicator = PandasOrderIndicator::new();
    indicator.assign("amount", metric(&[("b", Some(2.0)), ("a", None)]));
    indicator.assign("price", metric(&[("a", Some(3.0))]));
    indicator.assign("amount", metric(&[("c", Some(4.0))]));

    let storage: &dyn OrderIndicator = &indicator;
    assert_eq!(
        storage.metric_names().collect::<Vec<_>>(),
        ["amount", "price"]
    );
    assert_eq!(keys(storage.metric("amount").unwrap()), ["c"]);
    assert!(storage.metric("missing").is_none());

    let raw = indicator.get_metric_series("amount");
    assert_eq!(numbers(&raw), [Some(4.0)]);
    assert!(indicator.get_metric_series("missing").is_empty());
    assert!(indicator.get_index_data("missing").is_empty());

    let flags = PandasSingleMetric::try_new(
        StringArray::from(vec!["x", "y", "z"]),
        Arc::new(BooleanArray::from(vec![Some(true), None, Some(false)])),
    )
    .unwrap();
    indicator.assign("flags", flags.clone());
    assert_eq!(
        indicator.get_metric_series("flags").values().data_type(),
        flags.values().data_type()
    );
    assert_eq!(
        numbers(&indicator.get_index_data("flags")),
        [Some(1.0), None, Some(0.0)]
    );

    let snapshot = indicator.to_series();
    assert_eq!(
        snapshot.keys().map(String::as_str).collect::<Vec<_>>(),
        ["amount", "price", "flags"]
    );
    assert_eq!(numbers(snapshot.get("price").unwrap()), [Some(3.0)]);
}

#[test]
fn transfer_resolves_declared_inputs_and_has_typed_results() {
    let mut indicator = PandasOrderIndicator::new();
    indicator.assign("amount", metric(&[("a", Some(2.0))]));
    let derived = metric(&[("a", Some(4.0))]);
    let transform = FixedTransform {
        inputs: vec!["amount".to_owned()],
        output: Ok(IndicatorValue::Metric(derived)),
    };
    assert!(
        transfer(&mut indicator, &transform, Some("double"))
            .unwrap()
            .is_none()
    );
    assert_eq!(numbers(&indicator.get_metric_series("double")), [Some(4.0)]);

    for value in [
        IndicatorValue::Number(2.5),
        IndicatorValue::Integer(2),
        IndicatorValue::Boolean(true),
        IndicatorValue::Missing,
    ] {
        let transform = FixedTransform {
            inputs: vec![],
            output: Ok(value),
        };
        assert!(
            transfer(&mut indicator, &transform, None)
                .unwrap()
                .is_some()
        );
    }

    let missing = FixedTransform {
        inputs: vec!["unknown".to_owned()],
        output: Ok(IndicatorValue::Missing),
    };
    assert!(matches!(
        transfer(&mut indicator, &missing, None),
        Err(OrderIndicatorError::MissingMetric(name)) if name == "unknown"
    ));
    let failed = FixedTransform {
        inputs: vec![],
        output: Err("plugin failure".to_owned()),
    };
    assert!(matches!(
        transfer(&mut indicator, &failed, None),
        Err(OrderIndicatorError::Transform(message)) if message == "plugin failure"
    ));
    let scalar = FixedTransform {
        inputs: vec![],
        output: Ok(IndicatorValue::Number(1.0)),
    };
    assert!(matches!(
        transfer(&mut indicator, &scalar, Some("bad")),
        Err(OrderIndicatorError::AssignedNonMetric)
    ));
}

#[test]
fn summing_indicators_preserves_alignment_fill_and_empty_rules() {
    let mut first = PandasOrderIndicator::new();
    first.assign("value", metric(&[("b", Some(1.0)), ("a", None)]));
    first.assign("strict", metric(&[("a", None)]));
    let mut second = PandasOrderIndicator::new();
    second.assign("value", metric(&[("a", Some(2.0)), ("c", None)]));
    second.assign("strict", metric(&[("a", Some(2.0))]));

    let mut output = PandasOrderIndicator::new();
    PandasOrderIndicator::sum_all_indicators(
        &mut output,
        &[&first, &second],
        &["value"],
        Some(0.0),
    )
    .unwrap();
    let summed = output.get_metric_series("value");
    assert_eq!(keys(&summed), ["a", "b", "c"]);
    assert_eq!(numbers(&summed), [Some(2.0), Some(1.0), None]);

    PandasOrderIndicator::sum_all_indicators(&mut output, &[&first, &second], &["strict"], None)
        .unwrap();
    assert_eq!(numbers(&output.get_metric_series("strict")), [None]);

    PandasOrderIndicator::sum_all_indicators(&mut output, &[], &["empty"], Some(0.0)).unwrap();
    assert!(output.get_metric_series("empty").is_empty());
    assert!(matches!(
        PandasOrderIndicator::sum_all_indicators(
            &mut output,
            &[&first],
            &["missing"],
            Some(0.0)
        ),
        Err(OrderIndicatorError::MissingMetric(name)) if name == "missing"
    ));
}

#[test]
fn contract_matches_live_python_order_indicator() {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../qlib/qlib/backtest/high_performance_ds.py");
    let script = r"
import ast,inspect,json,sys
from collections import OrderedDict
from typing import Any,Callable,Dict,List,Optional,Text,Union,cast
import numpy as np,pandas as pd
class Logger: pass
def get_module_logger(_): return Logger()
class SD:
 def __init__(self,value=None):
  s=pd.Series(dtype=float) if value is None else pd.Series(value,dtype=float)
  self.index=list(s.index);self.data=s.tolist()
class IDD: SingleData=SD
idd=IDD();SingleData=SD
t=ast.parse(open(sys.argv[1],encoding='utf-8').read(),filename=sys.argv[1])
names=['BaseSingleMetric','BaseOrderIndicator','SingleMetric','PandasSingleMetric','PandasOrderIndicator']
nodes=[next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name==name) for name in names]
m=ast.Module(body=nodes,type_ignores=[]);ast.fix_missing_locations(m);exec(compile(m,sys.argv[1],'exec'),globals())
def vals(s): return {'index':list(s.index),'values':[None if pd.isna(v) else float(v) for v in s.tolist()]}
a=PandasOrderIndicator();a.assign('amount',{'b':1.0,'a':np.nan});a.assign('price',{'a':3.0});a.assign('amount',{'c':4.0})
raw=a.to_series();idx=a.get_index_data('amount')
b=PandasOrderIndicator();b.assign('value',{'a':2.0,'c':np.nan})
c=PandasOrderIndicator();c.assign('value',{'b':1.0,'a':np.nan})
out=PandasOrderIndicator();PandasOrderIndicator.sum_all_indicators(out,[b,c],'value',fill_value=0)
empty=PandasOrderIndicator();PandasOrderIndicator.sum_all_indicators(empty,[],['x'],fill_value=0)
scalar=a.transfer(lambda amount: amount.sum())
a.transfer(lambda amount: amount+1,'next')
result={
 'order':list(a.data.keys()),'raw_order':list(raw.keys()),'raw_amount':vals(raw['amount']),
 'index':{'index':idx.index,'values':idx.data},'missing_series':len(a.get_metric_series('none')),
 'missing_index':len(a.get_index_data('none').data),'sum':vals(out.get_metric_series('value')),
 'empty':vals(empty.get_metric_series('x')),'scalar':float(scalar),'next':vals(a.get_metric_series('next')),
}
try: a.transfer(lambda unknown: unknown)
except Exception as error: result['missing_error']=type(error).__name__
print(json.dumps(result,sort_keys=True))
";
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
            "empty":{"index":[],"values":[]},
            "index":{"index":["c"],"values":[4.0]},
            "missing_error":"KeyError","missing_index":0,"missing_series":0,
            "next":{"index":["c"],"values":[5.0]},
            "order":["amount","price","next"],"raw_amount":{"index":["c"],"values":[4.0]},
            "raw_order":["amount","price"],"scalar":4.0,
            "sum":{"index":["a","b","c"],"values":[2.0,1.0,null]}
        })
    );
}
