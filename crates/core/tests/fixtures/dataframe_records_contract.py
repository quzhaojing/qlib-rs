"""Freeze typed record inference against the unchanged Qlib append function.

This is source characterization, not evidence of native record-constructor parity.
"""
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
source_hash = hashlib.sha256(source).hexdigest()
assert source_hash == "89267f5cfc9e38751cb2c3a37c74ca712e8c395f492d53ed02f029a66272074f"
node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "dataframe_append")
future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
exec(compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])), "source", "exec"))

# Reuse only the explicit lossless snapshot helpers, without executing the other matrix.
helpers = ast.parse(Path(__file__).with_name("dataframe_constructor_contract.py").read_bytes())
helpers = [n for n in helpers.body if isinstance(n, ast.FunctionDef) and n.name in {"scalar", "snapshot"}]
assert len(helpers) == 2
exec(compile(ast.Module(body=helpers, type_ignores=[]), "snapshot_helpers", "exec"))

samples = {"none": None, "na": pd.NA, "nat": pd.NaT, "bool": True,
           "integer": -3, "unsigned": 2**64-1, "bigint": 2**80,
           "float": 1.25, "negative_zero": -0.0, "nan": np.nan,
           "text": "中\ud800", "complex": 1+2j}
for dtype in ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
              "float16", "float32", "float64", "complex64", "complex128", "bool"]:
    samples["numpy_" + dtype] = np.array([1], dtype=dtype)[0]
for unit in ["s", "ms", "us", "ns"]:
    for zone in [None, "UTC", "Asia/Shanghai"]:
        samples[f"timestamp_{unit}_{zone}"] = pd.Timestamp("2024-01-02", tz=zone).as_unit(unit)
    samples["duration_" + unit] = pd.Timedelta("1h").as_unit(unit)
samples["timestamp_outside_ns"] = pd.Timestamp("2500-01-01").as_unit("s")
samples["duration_outside_ns"] = pd.Timedelta(np.timedelta64(10**12, "s"))

lefts = {"empty": pd.DataFrame(),
         "initialized": pd.DataFrame(columns=["datetime", "x", "stock_id"]).set_index("datetime"),
         "populated": pd.DataFrame({"x": [0.5]}, index=pd.DatetimeIndex(["2024-01-01"]))}
cases = []
for (name, value), missing, order in itertools.product(samples.items(),
                                                     ["single", "omitted", "none", "nan", "nat", "na"],
                                                     ["forward", "reverse"]):
    first = {"x": value, "datetime": pd.Timestamp("2024-01-02")}
    second = {"stock_id": "SH600000", "datetime": pd.Timestamp("2024-01-03")}
    if missing not in {"single", "omitted"}:
        second["x"] = {"none": None, "nan": np.nan, "nat": pd.NaT, "na": pd.NA}[missing]
    records = [first] if missing == "single" else [first, second]
    if order == "reverse":
        records.reverse()
    before = [[(key, scalar(item)) for key, item in record.items()] for record in records]
    for left_name, left in lefts.items():
        left_before = snapshot(left)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            constructed = snapshot(pd.DataFrame(records))
        constructor_warnings = [[type(w.message).__name__, str(w.message)] for w in captured]
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            try:
                outcome = {"output": snapshot(dataframe_append(left, records))}
            except Exception as error:
                outcome = {"error": type(error).__name__, "message": str(error)}
        assert snapshot(left) == left_before
        assert [[(key, scalar(item)) for key, item in record.items()] for record in records] == before
        cases.append(dict(name=name, missing=missing, order=order, left=left_name,
                          construction=constructed, construction_warnings=constructor_warnings,
                          warnings=[[type(w.message).__name__, str(w.message)] for w in captured], **outcome))

digest = hashlib.sha256(json.dumps(cases, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(dict(source_sha256=source_hash, pandas=pd.__version__, numpy=np.__version__,
                      samples=len(samples), digest=digest, cases=cases)))
