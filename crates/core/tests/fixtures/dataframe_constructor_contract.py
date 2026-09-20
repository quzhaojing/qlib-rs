"""Actual Qlib constructor/append contracts, before designing native input adapters."""
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


def scalar(value):
    if value is None:
        return ["none"]
    if value is pd.NA:
        return ["pd.NA"]
    if value is pd.NaT:
        return ["NaT"]
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return [type(value).__name__, str(value.asm8.dtype), str(value.asm8.view("i8")),
                str(value.tz) if isinstance(value, pd.Timestamp) else None]
    if isinstance(value, (bool, np.bool_)):
        return ["bool", bool(value)]
    if isinstance(value, (int, np.integer)):
        return ["int", str(value)]
    if isinstance(value, (float, np.floating)):
        return ["float", "nan" if math.isnan(value) else float(value).hex()]
    if isinstance(value, (complex, np.complexfloating)):
        return ["complex", scalar(value.real), scalar(value.imag)]
    if isinstance(value, str):
        return ["str", [ord(c) for c in value]]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [scalar(v) for v in value]]
    raise TypeError(type(value))


def snapshot(frame):
    return dict(columns=[scalar(v) for v in frame.columns],
                dtypes=[str(v) for v in frame.dtypes],
                index=[scalar(v) for v in frame.index], index_dtype=str(frame.index.dtype),
                index_name=scalar(frame.index.name),
                column_axis=[type(frame.columns).__name__, str(frame.columns.dtype)],
                values=[[scalar(v) for v in frame.iloc[:, i].tolist()] for i in range(len(frame.columns))],
                blocks=[[str(b.dtype), b.mgr_locs.as_array.tolist()] for b in frame._mgr.blocks])


lefts = dict(empty=pd.DataFrame(), initialized=pd.DataFrame(columns=["datetime", "x", "stock_id"]).set_index("datetime"),
             populated=pd.DataFrame({"x": [0.5]}, index=pd.DatetimeIndex(["2024-01-01"])))
cases = []


def execute(name, other):
    for left_name, left in lefts.items():
        before = snapshot(left)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            try:
                constructed = pd.DataFrame(other)
                construction = snapshot(constructed)
            except Exception as error:
                construction = dict(error=type(error).__name__, message=str(error))
        construction_warnings = [[type(w.message).__name__, str(w.message)] for w in recorded]
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            try:
                result = dataframe_append(left, other)
                outcome = dict(output=snapshot(result))
            except Exception as error:
                outcome = dict(error=type(error).__name__, message=str(error))
        assert snapshot(left) == before, "source mutated left input"
        # Reconstruct to verify the caller-owned input has not been altered by append.
        if "error" not in construction:
            assert snapshot(pd.DataFrame(other)) == construction, "source mutated other input"
        cases.append(dict(name=name, left=left_name, construction=construction, construction_warnings=construction_warnings,
                          warnings=[[type(w.message).__name__, str(w.message)] for w in recorded], **outcome))


samples = dict(none=None, na=pd.NA, nat=pd.NaT, false=False, integer=-3, unsigned=2**64-1,
               bigint=2**80, float=1.25, negative_zero=-0.0, nan=np.nan, text="中文",
               surrogate="x\ud800", timestamp=pd.Timestamp("2024-01-02"),
               utc=pd.Timestamp("2024-01-02", tz="UTC"), timedelta=pd.Timedelta("1h"), complex=1+2j)
for (name, value), rows in itertools.product(samples.items(), [0, 1, 3]):
    execute(f"scalar/{name}/{rows}", dict(datetime=pd.date_range("2024-01-02", periods=rows), x=value, stock_id="SH600000"))
vectors = dict(empty=[], ints=[1, 2], missing=[None, 1], bools=[True, False], bool_none=[True, None],
               uint=[0, 2**64-1], signed_uint=[-1, 2**64-1], bigint=[1, 2**80],
               text=["x", "y"], nested=[[1], [2]], float32=np.array([1, 2], dtype="float32"),
               timestamp=pd.date_range("2024-01-02", periods=2))
for (name, value), rows in itertools.product(vectors.items(), [0, 1, 2, 3]):
    execute(f"vector/{name}/{rows}", dict(datetime=pd.date_range("2024-01-02", periods=rows), x=value))
for (ln, left), (rn, right), omit in itertools.product(samples.items(), samples.items(), [False, True]):
    if omit and rn != "none":
        continue  # The omitted right value is not an input; avoid repeating identical cases.
    first = dict(datetime=pd.Timestamp("2024-01-02"), x=left)
    second = dict(stock_id="SH600000", datetime=pd.Timestamp("2024-01-03"))
    if not omit:
        second["x"] = right
    execute(f"records/{ln}/{rn}/{omit}", [first, second])
for labels in [[0, 1], [1, 0], [2, 3], [0, 0]]:
    execute(f"series/{labels}", dict(datetime=pd.Series(pd.date_range("2024-01-02", periods=2), index=[0, 1]),
                                    x=pd.Series([1, 2], index=labels)))
special = dict(none=None, empty_dict={}, empty_records=[], empty_record=[{}],
               scalar_only=dict(datetime=pd.Timestamp("2024-01-02"), x=1),
               missing_datetime=dict(x=[1, 2]), matrix=[[1, 2], [3, 4]],
               ragged=[[1], [2, 3]], rank3=np.zeros((1, 1, 1)),
               unequal=dict(datetime=[1], x=[1, 2]),
               duplicate_datetime=pd.DataFrame([[1, 2, 3]], columns=["datetime", "datetime", "x"]),
               duplicate_data=pd.DataFrame([[1, 2, 3]], columns=["datetime", "x", "x"]),
               structured=np.array([(1, 2.)], dtype=[("datetime", "i8"), ("x", "f8")]),
               record_order=[dict(x=1, datetime=1), dict(datetime=2, y=3)])
for name, value in special.items():
    execute("special/" + name, value)
digest = hashlib.sha256(json.dumps(cases, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(dict(source_sha256=source_hash, pandas=pd.__version__, numpy=np.__version__,
                      digest=digest, cases=cases)))
