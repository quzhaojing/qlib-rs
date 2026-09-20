"""Source-pinned differential cases for contrib/eva/alpha.py."""

import ast
import json
import math
import struct
import sys
from typing import Tuple

import pandas as pd


def load_function(path):
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "calc_long_short_return"
    )
    namespace = {"pd": pd, "Tuple": Tuple}
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    return namespace["calc_long_short_return"], ast.dump(node, include_attributes=False)


def bits(value):
    return struct.pack(">d", float(value)).hex()


def encoded(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    if value == float("inf"):
        return "inf"
    if value == float("-inf"):
        return "-inf"
    return repr(float(value))


def series(spec):
    levels = []
    for kind, values in zip(spec["level_types"], spec["levels"]):
        if kind == "utf8":
            levels.append(values)
        elif kind == "timestamp_ns":
            levels.append(pd.to_datetime(values, unit="ns"))
        elif kind == "timestamp_ns_utc":
            levels.append(pd.to_datetime(values, unit="ns", utc=True))
        elif kind == "int64":
            levels.append(pd.Index(values, dtype="int64"))
        elif kind == "uint64":
            levels.append(pd.Index(values, dtype="uint64"))
        else:
            raise AssertionError(kind)
    index = pd.MultiIndex.from_arrays(levels, names=spec["level_names"])
    return pd.Series(spec["values"], index=index, dtype="float64")


def encoded_index(index):
    dtype = str(index.dtype)
    if isinstance(index, pd.DatetimeIndex):
        kind = "timestamp_ns_utc" if index.tz is not None else "timestamp_ns"
        values = [None if pd.isna(value) else str(value.value) for value in index]
    elif dtype == "int64":
        kind = "int64"
        values = [str(value) for value in index]
    elif dtype == "uint64":
        kind = "uint64"
        values = [str(value) for value in index]
    else:
        kind = "utf8"
        values = [None if pd.isna(value) else str(value) for value in index]
    return kind, values


def run(function, case):
    pred = series(case["pred"])
    label = series(case["label"])
    pred_before = pred.copy(deep=True)
    label_before = label.copy(deep=True)
    try:
        long_short, average = function(
            pred,
            label,
            date_col=case["date_col"],
            quantile=float(case["quantile"]),
            dropna=case["dropna"],
        )
        long_short_kind = "frame" if isinstance(long_short, pd.DataFrame) else "series"
        date_type, dates = encoded_index(long_short.index)
        result = {
            "long_short_kind": long_short_kind,
            "long_short_columns": list(long_short.columns) if long_short_kind == "frame" else None,
            "long_short_column_types": (
                [str(dtype) for dtype in long_short.dtypes]
                if long_short_kind == "frame"
                else None
            ),
            "long_short_index_class": type(long_short.index).__name__,
            "long_short_index_names": list(long_short.index.names),
            "date_type": date_type,
            "dates": dates,
            "index_name": getattr(long_short.index, "name", None),
            "long_short_name": getattr(long_short, "name", None),
            "average_name": average.name,
            "long_short_bits": (
                None if long_short_kind == "frame" else [bits(value) for value in long_short]
            ),
            "average_bits": [bits(value) for value in average],
        }
    except Exception as error:  # the exact live-source failure is evidence
        result = {"error_type": type(error).__name__, "error": str(error)}
    assert pred.equals(pred_before)
    assert label.equals(label_before)
    case["pred"]["values"] = [encoded(value) for value in case["pred"]["values"]]
    case["label"]["values"] = [encoded(value) for value in case["label"]["values"]]
    case["result"] = result
    return case


def spec(levels, values, names=("datetime", "instrument"), types=("utf8", "utf8")):
    return {
        "level_names": list(names),
        "level_types": list(types),
        "levels": levels,
        "values": values,
    }


nan = float("nan")
cases = [
    {
        "name": "stable_ties_nan_and_unsorted_exact_index",
        "pred": spec(
            [["d2", "d1", "d1", "d1", "d2", "d1"], ["a", "b", "a", "c", "b", "z"]],
            [1.0, 2.0, 2.0, -1.0, nan, 2.0],
        ),
        "label": spec(
            [["d2", "d1", "d1", "d1", "d2", "d1"], ["a", "b", "a", "c", "b", "z"]],
            [10.0, 20.0, 30.0, -5.0, 90.0, nan],
        ),
        "date_col": "datetime",
        "quantile": "0.5",
        "dropna": False,
    },
    {
        "name": "dropna_changes_group_size",
        "pred": spec([["d", "d", "d", "d"], ["a", "b", "c", "d"]], [4.0, nan, 2.0, 1.0]),
        "label": spec([["d", "d", "d", "d"], ["a", "b", "c", "d"]], [40.0, 30.0, nan, 10.0]),
        "date_col": "datetime",
        "quantile": "0.5",
        "dropna": True,
    },
    {
        "name": "zero_selection_is_nan",
        "pred": spec([["d"], ["a"]], [1.0]),
        "label": spec([["d"], ["a"]], [2.0]),
        "date_col": "datetime",
        "quantile": "0.2",
        "dropna": False,
    },
    {
        "name": "unique_outer_alignment_sorts_before_ties",
        "pred": spec([["d2", "d1"], ["b", "a"]], [1.0, 2.0]),
        "label": spec([["d1", "d3"], ["a", "c"]], [3.0, 4.0]),
        "date_col": "datetime",
        "quantile": "1.0",
        "dropna": False,
    },
    {
        "name": "unique_series_broadcasts_to_duplicate_index",
        "pred": spec([["d", "d"], ["a", "a"]], [2.0, 1.0]),
        "label": spec([["d"], ["a"]], [8.0]),
        "date_col": "datetime",
        "quantile": "0.5",
        "dropna": False,
    },
    {
        "name": "missing_date_dropped_missing_instrument_retained",
        "pred": spec([[None, "d", "d"], ["a", None, "b"]], [9.0, 2.0, 1.0]),
        "label": spec([[None, "d", "d"], ["a", None, "b"]], [99.0, 20.0, 10.0]),
        "date_col": "datetime",
        "quantile": "0.5",
        "dropna": False,
    },
    {
        "name": "empty_input",
        "pred": spec([[], []], []),
        "label": spec([[], []], []),
        "date_col": "datetime",
        "quantile": "0.2",
        "dropna": False,
    },
    {
        "name": "all_missing_date_keys_return_empty_frame",
        "pred": spec([[None, None], ["a", "b"]], [2.0, 1.0]),
        "label": spec([[None, None], ["a", "b"]], [20.0, 10.0]),
        "date_col": "datetime",
        "quantile": "0.5",
        "dropna": False,
    },
    {
        "name": "dropna_all_rows_return_empty_frame",
        "pred": spec([["d", "d"], ["a", "b"]], [float("nan"), 1.0]),
        "label": spec([["d", "d"], ["a", "b"]], [20.0, float("nan")]),
        "date_col": "datetime",
        "quantile": "0.5",
        "dropna": True,
    },
    {
        "name": "selection_larger_than_group_includes_nan_prediction_last",
        "pred": spec([["d", "d", "d"], ["a", "b", "c"]], [2.0, nan, 1.0]),
        "label": spec([["d", "d", "d"], ["a", "b", "c"]], [20.0, 99.0, 10.0]),
        "date_col": "datetime",
        "quantile": "2.0",
        "dropna": False,
    },
    {
        "name": "timestamp_dates_and_integer_instruments",
        "pred": spec(
            [[1704240000000000000, 1704153600000000000, 1704153600000000000], [9, 2, 1]],
            [1.0, 2.0, -1.0],
            types=("timestamp_ns", "int64"),
        ),
        "label": spec(
            [[1704240000000000000, 1704153600000000000, 1704153600000000000], [9, 2, 1]],
            [10.0, 20.0, -5.0],
            types=("timestamp_ns", "int64"),
        ),
        "date_col": "datetime",
        "quantile": "0.5",
        "dropna": False,
    },
    {
        "name": "timezone_schema_and_unsigned_instruments",
        "pred": spec(
            [[1704153600000000000, 1704153600000000000], [1, 2]],
            [2.0, 1.0],
            types=("timestamp_ns_utc", "uint64"),
        ),
        "label": spec(
            [[1704153600000000000, 1704153600000000000], [1, 2]],
            [20.0, 10.0],
            types=("timestamp_ns_utc", "uint64"),
        ),
        "date_col": "datetime",
        "quantile": "0.5",
        "dropna": False,
    },
    {
        "name": "integer_date_groups_preserve_dtype",
        "pred": spec([[2, 1, 1], ["a", "b", "a"]], [1.0, 2.0, -1.0], types=("int64", "utf8")),
        "label": spec([[2, 1, 1], ["a", "b", "a"]], [10.0, 20.0, -5.0], types=("int64", "utf8")),
        "date_col": "datetime",
        "quantile": "0.5",
        "dropna": False,
    },
    {
        "name": "cancellation_and_infinity_aggregation",
        "pred": spec([["cancel", "cancel", "cancel", "infinite", "infinite"], ["a", "b", "c", "a", "b"]], [3.0, 2.0, 1.0, 2.0, 1.0]),
        "label": spec([["cancel", "cancel", "cancel", "infinite", "infinite"], ["a", "b", "c", "a", "b"]], [1e16, 1.0, -1e16, float("inf"), float("-inf")]),
        "date_col": "datetime",
        "quantile": "1.0",
        "dropna": False,
    },
    {
        "name": "negative_quantile_selects_nothing",
        "pred": spec([["d"], ["a"]], [1.0]),
        "label": spec([["d"], ["a"]], [2.0]),
        "date_col": "datetime",
        "quantile": "-1.1",
        "dropna": False,
    },
    {
        "name": "nan_quantile_error",
        "pred": spec([["d"], ["a"]], [1.0]),
        "label": spec([["d"], ["a"]], [2.0]),
        "date_col": "datetime",
        "quantile": "nan",
        "dropna": False,
    },
    {
        "name": "missing_date_level_error",
        "pred": spec([["d"], ["a"]], [1.0]),
        "label": spec([["d"], ["a"]], [2.0]),
        "date_col": "missing",
        "quantile": "0.2",
        "dropna": False,
    },
    {
        "name": "incompatible_duplicate_alignment_error",
        "pred": spec([["d", "d"], ["a", "a"]], [1.0, 2.0]),
        "label": spec([["d", "d"], ["a", "b"]], [3.0, 4.0]),
        "date_col": "datetime",
        "quantile": "0.5",
        "dropna": False,
    },
]

function, source_ast = load_function(sys.argv[1])
print(
    json.dumps(
        {"ast": source_ast, "cases": [run(function, case) for case in cases]},
        separators=(",", ":"),
        allow_nan=False,
    )
)
