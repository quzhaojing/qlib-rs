"""Characterize qlib.contrib.evaluate.risk_analysis from the checked-out source."""

import ast
import hashlib
import json
import math
import pathlib
import re
import sys
import warnings
from typing import Literal, Tuple, Union

import numpy as np
import pandas as pd


def selected(path, name, kind):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    node = next(item for item in tree.body if isinstance(item, kind) and item.name == name)
    return compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec")


def selected_digest(path, name, kind):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    node = next(item for item in tree.body if isinstance(item, kind) and item.name == name)
    selected_source = ast.get_source_segment(source, node).encode("utf-8")
    return hashlib.sha256(selected_source).hexdigest()


root = pathlib.Path(sys.argv[1])
evaluate_path = root / "qlib/contrib/evaluate.py"
time_path = root / "qlib/utils/time.py"
namespace = {
    "Literal": Literal,
    "Tuple": Tuple,
    "Union": Union,
    "np": np,
    "pd": pd,
    "re": re,
    "warnings": warnings,
}
exec(selected(time_path, "Freq", ast.ClassDef), namespace)
exec(selected(evaluate_path, "risk_analysis", ast.FunctionDef), namespace)
risk_analysis = namespace["risk_analysis"]


def encoded(value):
    value = float(value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    # repr preserves enough precision for a round trip and distinguishes -0.0.
    return repr(value)


def run(case):
    index = case.get("index")
    series = pd.Series(case["returns"], index=index, dtype="float64")
    kwargs = {"N": case.get("N"), "freq": case.get("freq"), "mode": case["mode"]}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = risk_analysis(series, **kwargs)
            return {
                "name": case["name"],
                "schema": list(result.index),
                "column": list(result.columns),
                "values": [encoded(value) for value in result["risk"]],
                "warnings": [
                    {"category": item.category.__name__, "message": str(item.message)} for item in caught
                ],
            }
        except Exception as error:  # characterization intentionally records source failures
            return {
                "name": case["name"],
                "error": {"type": type(error).__name__, "message": str(error)},
                "warnings": [
                    {"category": item.category.__name__, "message": str(item.message)} for item in caught
                ],
            }


cases = [
    {"name": "sum_day", "returns": [0.1, -0.2, 0.3, 0.0], "N": None, "freq": "day", "mode": "sum"},
    {"name": "sum_missing", "returns": [0.1, None, -0.2, 0.3], "N": None, "freq": "2week", "mode": "sum"},
    {"name": "sum_empty", "returns": [], "N": None, "freq": "month", "mode": "sum"},
    {"name": "sum_singleton", "returns": [0.25], "N": 4, "freq": None, "mode": "sum"},
    {"name": "sum_all_nan", "returns": [None, None], "N": 4, "freq": None, "mode": "sum"},
    {"name": "sum_positive_inf", "returns": [0.1, float("inf"), -0.1], "N": 4, "freq": None, "mode": "sum"},
    {"name": "sum_mixed_inf", "returns": [float("inf"), float("-inf")], "N": 4, "freq": None, "mode": "sum"},
    {"name": "precedence", "returns": [0.1, 0.2], "N": 7, "freq": "not-a-freq", "mode": "sum"},
    {"name": "missing_scaler", "returns": [0.1], "N": None, "freq": None, "mode": "sum"},
    {"name": "invalid_freq", "returns": [0.1], "N": None, "freq": "hour", "mode": "sum"},
    {"name": "zero_freq", "returns": [0.1], "N": None, "freq": "0day", "mode": "sum"},
    {"name": "invalid_mode", "returns": [0.1], "N": 4, "freq": None, "mode": "median"},
    {"name": "product_string_index", "returns": [0.1, -0.2, 0.3], "index": ["a", "b", "c"], "N": 12, "freq": None, "mode": "product"},
    {"name": "product_missing_middle", "returns": [0.1, None, 0.2], "N": 12, "freq": None, "mode": "product"},
    {"name": "product_missing_last", "returns": [0.1, 0.2, None], "N": 12, "freq": None, "mode": "product"},
    {"name": "product_empty", "returns": [], "N": 12, "freq": None, "mode": "product"},
    {"name": "product_minus_one", "returns": [0.1, -1.0, 0.2], "N": 12, "freq": None, "mode": "product"},
    {"name": "product_below_minus_one", "returns": [0.1, -1.5, 0.2], "N": 12, "freq": None, "mode": "product"},
    {"name": "product_positive_inf", "returns": [0.1, float("inf"), 0.2], "N": 12, "freq": None, "mode": "product"},
    {"name": "sum_zero_std", "returns": [0.25, 0.25], "N": 4, "freq": None, "mode": "sum"},
    {"name": "sum_zero_over_zero", "returns": [0.0, 0.0], "N": 4, "freq": None, "mode": "sum"},
    {"name": "sum_negative_scaler", "returns": [0.1, 0.2], "N": -4, "freq": None, "mode": "sum"},
    {"name": "product_cancellation", "returns": [-0.9999999999999999, -0.9999999999999999], "N": 1, "freq": None, "mode": "product"},
    {"name": "product_overflow", "returns": [1e308, 1e308], "N": 4, "freq": None, "mode": "product"},
    {"name": "product_invalid_accumulate", "returns": [float("inf"), -1.0], "N": 4, "freq": None, "mode": "product"},
    {"name": "product_negative_fractional_annual", "returns": [-1.5, 0.0], "N": 3, "freq": None, "mode": "product"},
    {"name": "product_negative_singleton", "returns": [-1.5], "N": 2, "freq": None, "mode": "product"},
    {"name": "product_negative_infinite_scaler", "returns": [-1.5, 0.0], "N": float("inf"), "freq": None, "mode": "product"},
    {"name": "sum_pairwise_tail", "returns": [1e16, 1.0, -1e16, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], "N": 4, "freq": None, "mode": "sum"},
    {"name": "sum_mean_overflow", "returns": [1e308, 1e308], "N": 1, "freq": None, "mode": "sum"},
    {"name": "sum_long_cancellation", "returns": [1e16, 1.0, -1e16, 1.0] * 1024, "N": 4, "freq": None, "mode": "sum"},
]

payload = {
    "source": {
        "risk_analysis_sha256": selected_digest(evaluate_path, "risk_analysis", ast.FunctionDef),
        "freq_sha256": selected_digest(time_path, "Freq", ast.ClassDef),
    },
    "cases": [run(case) for case in cases],
}
print(json.dumps(payload, separators=(",", ":"), allow_nan=False))
