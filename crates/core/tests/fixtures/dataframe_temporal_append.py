"""Actual-source native/temporal-object append, missing alignment and block contracts."""
import ast
import hashlib
import itertools
import json
import math
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

source = Path(sys.argv[1]).read_bytes()
assert hashlib.sha256(source).hexdigest() == "89267f5cfc9e38751cb2c3a37c74ca712e8c395f492d53ed02f029a66272074f"
node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "dataframe_append")
future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
exec(compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])), "source", "exec"))
helpers = ast.parse(Path(__file__).with_name("dataframe_constructor_contract.py").read_bytes())
helpers = [n for n in helpers.body if isinstance(n, ast.FunctionDef) and n.name in {"scalar", "snapshot"}]
assert len(helpers) == 2
exec(compile(ast.Module(body=helpers, type_ignores=[]), "snapshot_helpers", "exec"))

objects = dict(empty=[], none=[None,None], nan=[np.nan,np.nan], none_nan=[None,np.nan],
               nan_none=[np.nan,None], na=[pd.NA,pd.NA], nat=[pd.NaT,pd.NaT],
               none_na=[None,pd.NA], na_none=[pd.NA,None], none_text=[None,"x"],
               text=["", "中\ud800"], finite_float=[0.,-0.], integer=[-(2**63),2**64-1], boolean=[False,True],
               timestamp=[pd.Timestamp("2024-01-02"),None], utc=[pd.Timestamp("2024-01-02",tz="UTC"),None],
               duration=[pd.Timedelta("1h").as_unit("s"),None],
               zones=[pd.Timestamp("2024-01-02",tz="UTC"),pd.Timestamp("2024-01-02",tz="Asia/Shanghai")],
               kinds=[pd.Timestamp("2024-01-02"),pd.Timedelta("1h")],
               outside=[pd.Timestamp("2500-01-01"),pd.Timedelta(np.timedelta64(10**12,"s"))])
samples = {"object_"+k: pd.Series(v,dtype=object) for k,v in objects.items()}
for dtype in ["bool","int64","uint64","float16","float32","float64"]:
    for state, values in [("empty",[]),("finite",[0,1])]:
        samples[dtype+"_"+state] = pd.Series(values,dtype=dtype)
    if dtype.startswith("float"):
        samples[dtype+"_na"] = pd.Series([np.nan,np.nan],dtype=dtype)
for unit in ["s","ms","us","ns"]:
    for zone in [None,"UTC","Asia/Shanghai"]:
        dtype = f"datetime64[{unit}]" if zone is None else f"datetime64[{unit}, {zone}]"
        for state, values in [("empty",[]),("finite",[pd.Timestamp("2024-01-02",tz=zone),pd.NaT]),("na",[pd.NaT,pd.NaT])]:
            samples[f"timestamp_{unit}_{zone}_{state}"] = pd.Series(values,dtype=dtype)
    for state, values in [("empty",[]),("finite",[pd.Timedelta("1h"),pd.NaT]),("na",[pd.NaT,pd.NaT])]:
        samples[f"duration_{unit}_{state}"] = pd.Series(values,dtype=f"timedelta64[{unit}]")


def execute(left, right):
    before = [snapshot(left),snapshot(right)]
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            result = dataframe_append(left,right)
            outcome = dict(output=snapshot(result))
        except Exception as error:
            outcome = dict(error=type(error).__name__,message=str(error))
    assert before == [snapshot(left),snapshot(right)]
    return dict(warnings=[[type(w.message).__name__,str(w.message)] for w in captured],**outcome)


pairs, alignment = [], []
for (ln,left),(rn,right) in itertools.product(samples.items(),repeat=2):
    a = pd.DataFrame(dict(x=left))
    for key, target in [("x",pairs),("y",alignment)]:
        b = pd.DataFrame({"datetime":pd.Series(range(len(right)),dtype="int64"),key:right})
        target.append(dict(left=ln,right=rn,**execute(a,b)))

blocks = []
for missing, other, companion, reverse, fragmented in itertools.product(
        ["object_none_nan","object_na","object_nat"],
        ["timestamp_s_None_finite","timestamp_ns_UTC_finite","duration_us_finite"],
        ["object_timestamp","object_none"],[False,True],[False,True]):
    data = dict(x=samples[missing], y=samples[companion])
    a = pd.DataFrame(data)
    if fragmented:
        a = pd.concat([pd.DataFrame({k:v}) for k,v in data.items()],axis=1)
    b = pd.DataFrame(dict(x=samples[other],y=samples[other]))
    if reverse:
        a,b = b,a
    b.insert(0,"datetime",[0,1])
    blocks.append(dict(left_input=snapshot(a),right_input=snapshot(b),**execute(a,b)))

contract = dict(inputs={name:snapshot(pd.DataFrame(dict(x=value))) for name,value in samples.items()},
                pairs=pairs,alignment=alignment,blocks=blocks)
digest = hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__,numpy=np.__version__,digest=digest,**contract)))
