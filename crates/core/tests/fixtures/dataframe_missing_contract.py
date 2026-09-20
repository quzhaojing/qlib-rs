"""Execute unchanged Qlib for object missing-value identity and block conversion."""
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
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        return ["float", "nan" if math.isnan(value) else value.hex()]
    if isinstance(value, str):
        return ["str", [ord(char) for char in value]]
    raise TypeError(type(value))


def snapshot(frame):
    return dict(columns=list(frame.columns), dtypes=[str(dtype) for dtype in frame.dtypes],
                index=frame.index.tolist(), index_name=frame.index.name,
                values=[[scalar(v) for v in frame.iloc[:, i].tolist()] for i in range(len(frame.columns))],
                blocks=[[str(b.dtype), b.mgr_locs.as_array.tolist()] for b in frame._mgr.blocks])


def execute(left, right):
    before = [snapshot(left), snapshot(right)]
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        result = dataframe_append(left, right)
    assert before == [snapshot(left), snapshot(right)], "upstream mutated inputs"
    return result, dict(output=snapshot(result), warnings=[[type(w.message).__name__, str(w.message)] for w in recorded])


objects = dict(empty=[], none=[None, None], nan=[np.nan, np.nan], none_nan=[None, np.nan],
               nan_none=[np.nan, None], na=[pd.NA, pd.NA], nat=[pd.NaT, pd.NaT],
               none_na=[None, pd.NA], na_none=[pd.NA, None], none_text=[None, "x"],
               text=["", "中文\ud800"], finite_float=[0.0, -0.0],
               integer=[-(2**63), 2**64 - 1], boolean=[False, True])
samples = {"object_" + key: pd.Series(values, dtype=object) for key, values in objects.items()}
for dtype in ["bool", "int64", "uint64", "float16", "float32", "float64"]:
    for state, values in [("empty", []), ("finite", [0, 1])]:
        samples[dtype + "_" + state] = pd.Series(values, dtype=dtype)
    if dtype.startswith("float"):
        samples[dtype + "_all_na"] = pd.Series([np.nan, np.nan], dtype=dtype)

pairs = []
chains = []
for (ln, left), (rn, right) in itertools.product(samples.items(), repeat=2):
    a = pd.DataFrame({"x": left})
    b = pd.DataFrame({"datetime": pd.Series(range(len(right)), dtype="int64"), "x": right})
    result, case = execute(a, b)
    pairs.append(dict(left=ln, right=rn, **case))
    for tn in ["float32_finite", "object_text"]:
        third = samples[tn]
        c = pd.DataFrame({"datetime": pd.Series(range(len(third)), dtype="int64"), "x": third})
        _, chained = execute(result, c)
        chains.append(dict(left=ln, right=rn, third=tn, **chained))


def build(data, fragmented):
    if not fragmented:
        return pd.DataFrame(data)
    frame = pd.DataFrame()
    for key, value in data.items():
        frame[key] = value
    return frame


blocks = []
for state, partner, sibling, swapped, lf, rf, reverse in itertools.product(
        ["object_none_nan", "object_nan_none", "object_na", "object_nat"],
        ["float32_finite", "object_finite_float", "int64_finite"],
        ["object_integer", "object_nan_none"], *([[False, True]] * 4)):
    names = ["y", "x"] if reverse else ["x", "y"]
    obj = {k: samples[state] if k == "x" else samples[sibling] for k in names}
    other = {k: samples[partner] for k in names}
    a = build(other if swapped else obj, lf)
    b = build({"datetime": pd.Series([0, 1], dtype="int64"), **(obj if swapped else other)}, rf)
    _, case = execute(a, b)
    blocks.append(dict(state=state, partner=partner, sibling=sibling, swapped=swapped, left_fragmented=lf,
                       right_fragmented=rf, reverse=reverse, left_input=snapshot(a), right_input=snapshot(b), **case))

contract = dict(pairs=pairs, chains=chains, blocks=blocks)
digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
alignment = []
for (ln, left), (rn, right) in itertools.product(samples.items(), repeat=2):
    a = pd.DataFrame({"x": left})
    b = pd.DataFrame({"datetime": pd.Series(range(len(right)), dtype="int64"), "y": right})
    _, case = execute(a, b)
    alignment.append(dict(left=ln, right=rn, **case))
layouts = []
from pandas.core.internals.managers import BlockManager
from pandas._libs.internals import BlockPlacement
for dtype in ["float32", "float64"]:
    a = pd.DataFrame({"x": pd.Series([1, 2], dtype=dtype), "z": pd.Series([3, 4], dtype=dtype)})
    block = a._mgr.blocks[0].take_nd(np.array([1, 0]), axis=0, new_mgr_locs=BlockPlacement([1, 0]))
    a._mgr = BlockManager((block,), a._mgr.axes)
    b = pd.DataFrame({"datetime": [0, 1], "y": pd.Series([5, 6], dtype=dtype)})
    _, case = execute(a, b)
    layouts.append(dict(left_input=snapshot(a), right_input=snapshot(b), **case))
print(json.dumps(dict(source_sha256=source_hash, pandas=pd.__version__, numpy=np.__version__,
                      alignment=alignment, layouts=layouts,
                      inputs={name: snapshot(pd.DataFrame({"x": value})) for name, value in samples.items()},
                      digest=digest, **contract), allow_nan=False))
