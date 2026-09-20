"""Hash-pinned Qlib append with mixed timestamp resolutions, NaT and overflow."""
import ast
import hashlib
import itertools
import json
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


def temporal(unit, zone, values):
    raw = np.array([-(2**63) if v is None else v for v in values], dtype="int64")
    index = pd.DatetimeIndex(raw.view("datetime64[" + unit + "]"))
    return index if zone is None else index.tz_localize(zone)


def snapshot(array):
    return dict(unit=array.dtype.unit if isinstance(array.dtype, pd.DatetimeTZDtype)
                else np.datetime_data(array.dtype)[0],
                values=[None if v == -(2**63) else int(v) for v in array.asi8])


states = [[], [None], [-1, 0, 1, None], [2**63 - 1, -(2**63) + 1]]
cases = []
for mode, zone, lu, ru, left, right in itertools.product(
        ["index", "data"], [None, "UTC"], ["s", "ms", "us", "ns"],
        ["s", "ms", "us", "ns"], states, states):
    la, ra = temporal(lu, zone, left), temporal(ru, zone, right)
    if mode == "index":
        a = pd.DataFrame({"x": np.zeros(len(left))}, index=la)
        b = pd.DataFrame({"datetime": ra, "x": np.ones(len(right))})
    else:
        a = pd.DataFrame({"x": la})
        b = pd.DataFrame({"datetime": np.arange(len(right)), "x": ra})
    before = (a.copy(deep=True), b.copy(deep=True))
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        try:
            result = dataframe_append(a, b)
            values = result.index if mode == "index" else pd.DatetimeIndex(result["x"])
            outcome = dict(output=snapshot(values), dtype=str(values.dtype), index_name=result.index.name)
        except Exception as error:
            outcome = dict(error=type(error).__name__, message=str(error))
    pd.testing.assert_frame_equal(a, before[0])
    pd.testing.assert_frame_equal(b, before[1])
    cases.append(dict(mode=mode, zone=zone, lu=lu, ru=ru, left=left, right=right,
                      warnings=[[type(w.message).__name__, str(w.message)] for w in recorded], **outcome))
digest = hashlib.sha256(json.dumps(cases, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
text_indexes = []
for left, right in itertools.product([[], ["x", "中文"]], repeat=2):
    a = pd.DataFrame({"x": np.zeros(len(left))}, index=pd.Index(left, dtype=object))
    b = pd.DataFrame({"datetime": pd.Series(right, dtype=object), "x": np.ones(len(right))})
    before = (a.copy(deep=True), b.copy(deep=True))
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        result = dataframe_append(a, b)
    pd.testing.assert_frame_equal(a, before[0])
    pd.testing.assert_frame_equal(b, before[1])
    text_indexes.append(dict(left=left, right=right, output=result.index.tolist(),
                             name=result.index.name, dtype=str(result.index.dtype),
                             warnings=[[type(w.message).__name__, str(w.message)] for w in recorded]))
print(json.dumps(dict(source_sha256=source_hash, pandas=pd.__version__, numpy=np.__version__,
                      digest=digest, cases=cases, text_indexes=text_indexes)))
