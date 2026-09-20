"""Lossless temporal object payloads passed through the unchanged source append."""
import ast
import hashlib
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

values = [None, pd.NA, pd.NaT, False, True, -(2**63), 2**64-1, 0.5, -0., np.nan, np.inf, -np.inf, "中\ud800"]
for unit in ["s", "ms", "us", "ns"]:
    for zone in [None, "UTC", "Asia/Shanghai", "Etc/GMT+5"]:
        values.append(pd.Timestamp("2024-01-02", tz=zone).as_unit(unit))
    for ticks in [-(2**63)+1, -1, 0, 2**63-1]:
        values.append(pd.Timestamp(np.datetime64(ticks, unit)))
        values.append(pd.Timedelta(np.timedelta64(ticks, unit)))

before = [scalar(value) for value in values]
other = dict(datetime=list(range(len(values))), x=pd.Series(values, dtype=object))
with warnings.catch_warnings(record=True) as captured:
    warnings.simplefilter("always")
    result = dataframe_append(pd.DataFrame(), other)
assert str(result.x.dtype) == "object"
assert [scalar(value) for value in values] == before
assert [scalar(value) for value in result.x.tolist()] == before
assert not captured
digest = hashlib.sha256(json.dumps(before, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__, numpy=np.__version__, values=before, digest=digest)))
