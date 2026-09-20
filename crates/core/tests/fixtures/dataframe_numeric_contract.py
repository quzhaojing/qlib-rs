"""Freeze actual Qlib/Pandas numeric concat, including block-sensitive outcomes.

This is a source characterization oracle, not an assertion of Rust dtype parity.
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


def scalar(value):
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        return ["float", "nan" if math.isnan(value) else value.hex()]
    raise TypeError(type(value))


def snapshot(frame):
    return dict(columns=[str(c) for c in frame.columns],
                dtypes=[str(t) for t in frame.dtypes],
                values=[[scalar(v) for v in frame.iloc[:, i].tolist()] for i in range(len(frame.columns))],
                blocks=[[str(b.dtype), b.mgr_locs.as_array.tolist()] for b in frame._mgr.blocks])


def execute(left, right, include_index=False):
    before = [snapshot(left), snapshot(right)]
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        result = dataframe_append(left, right)
    assert before == [snapshot(left), snapshot(right)], "upstream mutated input data"
    output = dict(output=snapshot(result), warnings=[[type(w.message).__name__, str(w.message)] for w in recorded])
    if include_index:
        output["index"] = dict(dtype=str(result.index.dtype), values=[scalar(v) for v in result.index.tolist()])
    return output


samples = []
for dtype in ["bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "float32", "float64"]:
    values = {"empty": [], "finite": [0, 1]}
    if dtype == "bool":
        values["extreme"] = [True, False]
    elif dtype.startswith("float"):
        limit = np.finfo(dtype).max
        values["extreme"] = [-limit, -0.0, limit, np.inf, -np.inf, np.nan]
        values["all_na"] = [np.nan, np.nan]
    else:
        info = np.iinfo(dtype)
        values["extreme"] = [info.min, info.max]
    for state, items in values.items():
        samples.append((dtype, state, pd.Series(items, dtype=dtype)))

numeric = []
for (ld, ls, left), (rd, rs, right) in itertools.product(samples, repeat=2):
    a = pd.DataFrame({"x": left})
    b = pd.DataFrame({"datetime": pd.Series(range(len(right)), dtype="int64"), "x": right})
    numeric.append(dict(left=[ld, ls], right=[rd, rs], **execute(a, b)))


def build(data, fragmented):
    if not fragmented:
        return pd.DataFrame(data)
    frame = pd.DataFrame()
    for key, value in data.items():
        frame[key] = value
    return frame


blocks = []
for left_fragmented, right_fragmented, reverse, all_na in itertools.product([False, True], repeat=4):
    names = ["y", "x"] if reverse else ["x", "y"]
    left = build({k: pd.Series([np.nan if k == "x" or all_na else 2], dtype="float64") for k in names}, left_fragmented)
    right = build({"datetime": pd.Series([1], dtype="int64"), **{k: pd.Series([1], dtype="float32") for k in names}}, right_fragmented)
    blocks.append(dict(left_fragmented=left_fragmented, right_fragmented=right_fragmented, reverse=reverse, all_na=all_na,
                       left_input=snapshot(left), right_input=snapshot(right), **execute(left, right)))

contract = dict(numeric=numeric, blocks=blocks)
digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
floating_blocks = []
for ld, rd, left_fragmented, right_fragmented, reverse, all_na in itertools.product(
        ["float16", "float32", "float64"], ["float16", "float32", "float64"], *([[False, True]] * 4)):
    names = ["y", "x"] if reverse else ["x", "y"]
    left = build({k: pd.Series([np.nan if k == "x" or all_na else 2], dtype=ld) for k in names}, left_fragmented)
    right = build({"datetime": pd.Series([1], dtype="int64"), **{k: pd.Series([1], dtype=rd) for k in names}}, right_fragmented)
    floating_blocks.append(dict(left_dtype=ld, right_dtype=rd, left_fragmented=left_fragmented,
                                right_fragmented=right_fragmented, reverse=reverse, all_na=all_na, **execute(left, right)))
integer_samples = [(dtype, state, array) for dtype, state, array in samples
                   if dtype.startswith(("int", "uint")) and state in ("empty", "extreme")]
integer_alignment = []
for (ld, ls, left), (rd, rs, right) in itertools.product(integer_samples, repeat=2):
    a = pd.DataFrame({"a": left})
    b = pd.DataFrame({"datetime": pd.Series(range(len(right)), dtype="int64"), "b": right})
    integer_alignment.append(dict(left=[ld, ls], right=[rd, rs], **execute(a, b)))
index_samples = [(dtype, state, array) for dtype, state, array in samples
                 if dtype in ("int64", "float32", "float64") and state != "all_na"]
numeric_indexes = []
for (ld, ls, left), (rd, rs, right) in itertools.product(index_samples, repeat=2):
    a = pd.DataFrame({"x": pd.Series(range(len(left)), dtype="int64")})
    a.index = pd.Index(left)
    b = pd.DataFrame({"datetime": right, "x": pd.Series(range(len(right)), dtype="int64")})
    numeric_indexes.append(dict(left=[ld, ls], right=[rd, rs], **execute(a, b, include_index=True)))
ordered_left = pd.DataFrame({"x": pd.Series([], dtype="float64")}, index=pd.Index([], dtype="int64"))
ordered_right = pd.DataFrame({"datetime": pd.Series([1], dtype="float32"), "x": pd.Series([1], dtype="float32")})
warning_order = execute(ordered_left, ordered_right)
bool_float_chains = []
for (ld, ls, left), (rd, rs, right) in itertools.product(samples, repeat=2):
    if not ((ld == "bool" and rd.startswith("float")) or (rd == "bool" and ld.startswith("float"))):
        continue
    a = pd.DataFrame({"x": left})
    b = pd.DataFrame({"datetime": pd.Series(range(len(right)), dtype="int64"), "x": right})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        first = dataframe_append(a, b)
    for td, ts, third in samples:
        c = pd.DataFrame({"datetime": pd.Series(range(len(third)), dtype="int64"), "x": third})
        bool_float_chains.append(dict(left=[ld, ls], right=[rd, rs], third=[td, ts],
                                      intermediate=snapshot(first), **execute(first, c)))
bool_float_blocks = []
for dtype, bool_first, left_fragmented, right_fragmented, reverse, all_na in itertools.product(
        ["float16", "float32", "float64"], *([[False, True]] * 5)):
    names = ["y", "x"] if reverse else ["x", "y"]
    floats = {k: pd.Series([np.nan if k == "x" or all_na else 2], dtype=dtype) for k in names}
    bools = {k: pd.Series([k == "x"], dtype="bool") for k in names}
    left = build(bools if bool_first else floats, left_fragmented)
    right = build({"datetime": pd.Series([1], dtype="int64"),
                   **(floats if bool_first else bools)}, right_fragmented)
    bool_float_blocks.append(dict(dtype=dtype, bool_first=bool_first, left_fragmented=left_fragmented,
                                  right_fragmented=right_fragmented, reverse=reverse, all_na=all_na,
                                  left_input=snapshot(left), right_input=snapshot(right), **execute(left, right)))
object_contract = dict(chains=bool_float_chains, blocks=bool_float_blocks)
object_digest = hashlib.sha256(json.dumps(object_contract, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode()).hexdigest()
object_samples = [(dtype, state, array) for dtype, state, array in samples if
                  (dtype == "bool" and state == "finite") or
                  (dtype in ("int64", "uint64", "float16", "float32", "float64") and state == "extreme") or
                  (dtype == "float64" and state in ("all_na", "empty"))]
numeric_object_inputs = []
numeric_object_alignment = []
for (ld, ls, left), (rd, rs, right), (lo, ro) in itertools.product(object_samples, object_samples,
                                                                 [(True, False), (False, True), (True, True)]):
    a = pd.DataFrame({"x": left.astype(object) if lo else left})
    b = pd.DataFrame({"datetime": pd.Series(range(len(right)), dtype="int64"),
                      "x": right.astype(object) if ro else right})
    numeric_object_inputs.append(dict(left=[ld, ls], right=[rd, rs], left_object=lo, right_object=ro,
                                      **execute(a, b)))
    numeric_object_alignment.append(dict(left=[ld, ls], right=[rd, rs], left_object=lo, right_object=ro,
                                         **execute(a.rename(columns={"x": "a"}), b.rename(columns={"x": "b"}))))
print(json.dumps(dict(source_sha256=source_hash, pandas=pd.__version__, numpy=np.__version__, digest=digest,
                      warning_order=warning_order,
                      object_contract=object_contract, object_digest=object_digest,
                      numeric_object_inputs=numeric_object_inputs,
                      numeric_object_alignment=numeric_object_alignment,
                      floating_blocks=floating_blocks, integer_alignment=integer_alignment,
                      numeric_indexes=numeric_indexes, **contract), allow_nan=False))
