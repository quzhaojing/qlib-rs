"""Actual append/list-column inference for built-in scalar mixtures."""
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


def scalar(v):
    if v is None:
        return ["none"]
    if v is pd.NA:
        return ["pd.NA"]
    if v is pd.NaT:
        return ["NaT"]
    if isinstance(v, bool):
        return ["bool", v]
    if isinstance(v, int):
        return ["int", str(v)]
    if isinstance(v, float):
        return ["float", "nan" if math.isnan(v) else v.hex()]
    if isinstance(v, str):
        return ["str", [ord(c) for c in v]]
    raise TypeError(type(v))


atoms = [None, pd.NA, pd.NaT, False, True, -(2**63), -1, 0, 2**63-1, 2**63, 2**64-1,
         0.5, -0., np.nan, np.inf, -np.inf, "", "中\ud800"]
cases = []
for size in range(4):
    for ids in itertools.product(range(len(atoms)), repeat=size):
        values = [atoms[i] for i in ids]
        before = [scalar(v) for v in values]
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            constructed = pd.DataFrame(dict(datetime=np.arange(size), x=values))
            result = dataframe_append(pd.DataFrame(), dict(datetime=np.arange(size), x=values))
        assert [scalar(v) for v in values] == before
        assert str(result["x"].dtype) == str(constructed["x"].dtype)
        encoded = [scalar(v) for v in result["x"].tolist()]
        assert encoded == [scalar(v) for v in constructed["x"].tolist()]
        with warnings.catch_warnings(record=True) as ignored_warnings:
            warnings.simplefilter("always")
            ignored = dataframe_append(result, dict(datetime=np.array([], dtype="int64")))
        pd.testing.assert_frame_equal(result, ignored)
        assert not ignored_warnings
        cases.append(dict(ids=ids, dtype=str(result["x"].dtype), values=encoded,
                          warnings=[[type(w.message).__name__, str(w.message)] for w in recorded]))
digest = hashlib.sha256(json.dumps(cases, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(dict(source_sha256=source_hash, pandas=pd.__version__, numpy=np.__version__,
                      atoms=[scalar(v) for v in atoms], cases=cases, digest=digest, ignored_right_verified=True)))
