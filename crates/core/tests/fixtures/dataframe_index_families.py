"""Actual Qlib index-family and metadata contracts; no native parity claim."""
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
exec(compile(ast.Module(body=[n for n in helpers.body if isinstance(n, ast.FunctionDef)
                             and n.name in {"scalar", "snapshot"}], type_ignores=[]), "helpers", "exec"))
basic_scalar = scalar


def scalar(value):
    if isinstance(value, pd.Period):
        return ["Period", str(value.ordinal), value.freqstr]
    if isinstance(value, pd.Interval):
        return ["Interval", scalar(value.left), scalar(value.right), value.closed]
    return basic_scalar(value)


def index_snapshot(index):
    result = dict(kind=type(index).__name__, dtype=str(index.dtype), nlevels=index.nlevels,
                  names=[scalar(v) for v in index.names], values=[scalar(v) for v in index])
    if isinstance(index, pd.RangeIndex):
        result["range"] = [index.start, index.stop, index.step]
    if isinstance(index, pd.MultiIndex):
        result.update(levels=[index_snapshot(v) for v in index.levels],
                      codes=[v.tolist() for v in index.codes], sortorder=index.sortorder)
    if isinstance(index, pd.CategoricalIndex):
        result.update(categories=index_snapshot(index.categories), ordered=index.ordered,
                      codes=index.codes.tolist())
    if isinstance(index, (pd.DatetimeIndex, pd.TimedeltaIndex, pd.PeriodIndex)):
        result["freq"] = index.freqstr
    if isinstance(index.dtype, pd.StringDtype):
        result["storage"] = index.dtype.storage
    return result


def frame_snapshot(frame):
    return dict(frame=snapshot(frame), index=index_snapshot(frame.index))


samples = dict(
    range_empty=pd.RangeIndex(0, name="datetime"),
    range_empty_offset=pd.RangeIndex(9, 9, 3, name="history"),
    range_start=pd.RangeIndex(0, 2, name="datetime"),
    range_next=pd.RangeIndex(2, 4, name="datetime"),
    range_step=pd.RangeIndex(0, 4, 2, name="datetime"),
    range_reverse=pd.RangeIndex(2, 0, -1),
    range_negative=pd.RangeIndex(-2, 0, name="datetime"),
    int_empty=pd.Index([], dtype="int64", name="datetime"),
    int_values=pd.Index([0, 1], dtype="int64", name="datetime"),
    object_empty=pd.Index([], dtype=object, name="datetime"),
    tuple_values=pd.Index([("a", 1), ("b", 2)], tupleize_cols=False, name="datetime"),
    multi_values=pd.MultiIndex.from_tuples([("a", 1), ("b", 2)], names=["asset", "time"]),
    multi_missing=pd.MultiIndex(levels=[["a", "b", "unused"], [1, 2, 99]], codes=[[0, -1], [1, 0]], names=["asset", "time"]),
    multi_empty=pd.MultiIndex(levels=[["unused"], [99]], codes=[[], []], names=["asset", "time"]),
    multi_three=pd.MultiIndex.from_tuples([("a", 1, True), ("b", 2, False)], names=["asset", "time", "flag"]),
    category=pd.CategoricalIndex(["a", "b"], categories=["a", "b", "unused"], name="datetime"),
    category_reordered=pd.CategoricalIndex(["b", "a"], categories=["unused", "b", "a"], name="datetime"),
    category_ordered=pd.CategoricalIndex(["a", "b"], categories=["a", "b"], ordered=True, name="datetime"),
    category_missing=pd.CategoricalIndex(["a", None], categories=["a", "b"], name="datetime"),
    category_empty=pd.CategoricalIndex([], categories=["a", "b"], name="datetime"),
    period_month=pd.period_range("2024-01", periods=2, freq="M", name="datetime"),
    period_day=pd.period_range("2024-01-01", periods=2, freq="D", name="datetime"),
    period_empty=pd.PeriodIndex([], freq="M", name="datetime"),
    period_missing=pd.PeriodIndex(["2024-01", None], freq="M", name="datetime"),
    interval_right=pd.IntervalIndex.from_breaks([0, 1, 2], closed="right", name="datetime"),
    interval_left=pd.IntervalIndex.from_breaks([0, 1, 2], closed="left", name="datetime"),
    interval_empty=pd.IntervalIndex([], closed="right", dtype="interval[int64, right]", name="datetime"),
    interval_missing=pd.IntervalIndex.from_tuples([(0., 1.), None], name="datetime"),
    datetime_regular=pd.date_range("2024-01-01", periods=2, freq="D", name="datetime"),
    datetime_next=pd.date_range("2024-01-03", periods=2, freq="D", name="datetime"),
    datetime_utc=pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC", name="datetime"),
    datetime_empty=pd.date_range("2024-01-01", periods=0, freq="D", name="datetime"),
    timedelta_regular=pd.timedelta_range("0D", periods=2, freq="D", name="datetime"),
    timedelta_next=pd.timedelta_range("2D", periods=2, freq="D", name="datetime"),
)
for dtype in ["Int64", "UInt64", "Float64", "boolean", "string"]:
    selected = pd.StringDtype(storage="python") if dtype == "string" else dtype
    values = ["a", None] if dtype == "string" else [True, None] if dtype == "boolean" else [1, None]
    for state, values in [("missing", values), ("empty", [])]:
        samples[f"nullable_{dtype}_{state}"] = pd.Index(pd.array(values, dtype=selected), name="datetime")


def execute(left, index, right_columns):
    before = [frame_snapshot(left), index_snapshot(index)]
    other = dict(datetime=index)
    if right_columns:
        other["x"] = list(range(len(index)))
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            result = dataframe_append(left, other)
        except Exception as error:
            result = None
            outcome = dict(error=type(error).__name__, message=str(error))
        else:
            outcome = dict(output=frame_snapshot(result))
    assert before == [frame_snapshot(left), index_snapshot(index)]
    return result, dict(warnings=[[type(w.message).__name__, str(w.message)] for w in captured], **outcome)


pairs = []
for (ln, left), (rn, right), lc, rc in itertools.product(samples.items(), samples.items(), [False, True], [False, True]):
    frame = pd.DataFrame(dict(x=list(range(len(left)))) if lc else {}, index=left)
    _, outcome = execute(frame, right, rc)
    pairs.append(dict(left=ln, right=rn, left_columns=lc, right_columns=rc, **outcome))

chains = []
for (name, index), next_name in itertools.product(samples.items(),
        ["range_next", "tuple_values", "category", "period_month", "nullable_Int64_missing"]):
    initial = pd.DataFrame(columns=["x"], index=index[:0])
    current, first = execute(initial, index, True)
    second = None
    if current is not None:
        _, second = execute(current, samples[next_name], True)
    chains.append(dict(first=name, second=next_name, initial=index_snapshot(initial.index), first_output=first, second_output=second))

contract = dict(inputs={name:index_snapshot(value) for name,value in samples.items()}, pairs=pairs, chains=chains)
digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__, numpy=np.__version__, digest=digest, **contract)))
