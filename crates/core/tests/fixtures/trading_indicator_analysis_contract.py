"""Characterize qlib.contrib.evaluate.indicator_analysis from its actual source."""

import ast
import json
import sys

import numpy as np
import pandas as pd


def load_function(path):
    source = open(path, encoding="utf-8").read()
    tree = ast.parse(source, filename=path)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "indicator_analysis"
    )
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {"np": np, "pd": pd}
    exec(compile(module, path, "exec"), namespace)
    return namespace["indicator_analysis"], ast.dump(node, include_attributes=False)


def number(value):
    if value is None or value == "null":
        return None
    if value == "nan":
        return np.nan
    if value == "inf":
        return np.inf
    if value == "-inf":
        return -np.inf
    return float(value)


def snapshot(case, function):
    integer_columns = set(case.get("integer_columns", []))
    native_integer_columns = set(case.get("native_integer_columns", []))
    nullable_columns = set(case.get("nullable_columns", []))
    columns = {}
    for name, values in case["columns"]:
        values = [number(value) for value in values]
        if name in integer_columns:
            columns[name] = pd.array(values, dtype="Int64")
        elif name in native_integer_columns:
            columns[name] = np.asarray(values, dtype=np.int64)
        elif name in nullable_columns:
            columns[name] = pd.array(values, dtype="Float64")
        else:
            columns[name] = values
    frame = pd.DataFrame(columns, index=pd.Index(case["index"], name="datetime"))
    try:
        result = function(frame, method=case["method"])
        return {
            "index": list(result.index),
            "columns": list(result.columns),
            "value_bits": [format(np.float64(value).view(np.uint64), "016x") for value in result["value"]],
            "value_dtype": str(result["value"].dtype),
            "error_type": None,
            "error": None,
        }
    except Exception as error:  # contract includes source failure order
        return {
            "index": None,
            "columns": None,
            "value_bits": None,
            "value_dtype": None,
            "error_type": type(error).__name__,
            "error": str(error),
        }


BASE = [
    ["ffr", ["0.5", "0.25", "nan"]],
    ["pa", ["1", "-2", "4"]],
    ["pos", ["1", "0", "1"]],
    ["count", ["2", "-1", "3"]],
    ["deal_amount", ["10", "-20", "nan"]],
    ["value", ["-100", "50", "25"]],
]


CASES = [
    {"name": "mean_negative_count_and_nan", "method": "mean", "index": ["t2", "t1", "t2"], "columns": BASE},
    {"name": "amount_weighted_absolute_and_nan", "method": "amount_weighted", "index": ["t2", "t1", "t2"], "columns": BASE},
    {"name": "value_weighted_absolute", "method": "value_weighted", "index": ["t2", "t1", "t2"], "columns": BASE},
    {"name": "zero_denominators", "method": "mean", "index": ["a", "b"], "columns": [["ffr", ["1", "2"]], ["pa", ["3", "4"]], ["pos", ["1", "0"]], ["count", ["1", "-1"]], ["deal_amount", ["0", "0"]], ["value", ["0", "0"]]]},
    {"name": "infinities", "method": "value_weighted", "index": ["a", "b", "c"], "columns": [["ffr", ["1", "2", "3"]], ["pa", ["inf", "1", "-inf"]], ["pos", ["1", "2", "3"]], ["count", ["inf", "1", "-inf"]], ["deal_amount", ["1", "2", "3"]], ["value", ["1", "inf", "2"]]]},
    {"name": "integer_columns", "method": "amount_weighted", "index": ["c", "a", "b"], "integer_columns": ["ffr", "pa", "pos", "count", "deal_amount", "value"], "columns": [["ffr", ["1", "2", "3"]], ["pa", ["-2", "4", "8"]], ["pos", ["0", "1", "1"]], ["count", ["1", "2", "3"]], ["deal_amount", ["-1", "2", "-3"]], ["value", ["3", "2", "1"]]]},
    {"name": "native_integer_columns", "method": "amount_weighted", "index": ["c", "a", "b"], "native_integer_columns": ["ffr", "pa", "pos", "count", "deal_amount", "value"], "columns": [["ffr", ["1", "2", "3"]], ["pa", ["-2", "4", "8"]], ["pos", ["0", "1", "1"]], ["count", ["1", "2", "3"]], ["deal_amount", ["-1", "2", "-3"]], ["value", ["3", "2", "1"]]]},
    {"name": "nullable_integer_and_float_columns", "method": "value_weighted", "index": ["a", "b", "c"], "integer_columns": ["ffr", "pos", "count"], "nullable_columns": ["pa", "deal_amount", "value"], "columns": [["ffr", ["1", "null", "3"]], ["pa", ["null", "2", "4"]], ["pos", ["1", "0", "null"]], ["count", ["2", "null", "1"]], ["deal_amount", ["null", "2", "3"]], ["value", ["-2", "null", "1"]]]},
    {"name": "cancellation_preserves_row_order", "method": "mean", "index": ["first", "middle", "last"], "columns": [["ffr", ["1", "1", "1"]], ["pa", ["1", "2", "3"]], ["pos", ["1", "1", "1"]], ["count", ["10000000000000000", "1", "-10000000000000000"]], ["deal_amount", ["1", "1", "1"]], ["value", ["1", "1", "1"]]]},
    {"name": "integer_cancellation_is_exact_before_division", "method": "mean", "index": ["first", "middle", "last"], "integer_columns": ["ffr", "pa", "pos", "count", "deal_amount", "value"], "columns": [["ffr", ["1", "1", "1"]], ["pa", ["1", "2", "3"]], ["pos", ["1", "1", "1"]], ["count", ["10000000000000000", "1", "-10000000000000000"]], ["deal_amount", ["1", "1", "1"]], ["value", ["1", "1", "1"]]]},
    {"name": "empty", "method": "amount_weighted", "index": [], "columns": [[name, []] for name, _ in BASE]},
    {"name": "missing_count_precedes_invalid_method", "method": "bogus", "index": ["a", "b", "c"], "columns": [item for item in BASE if item[0] != "count"]},
    {"name": "missing_deal_amount_precedes_invalid_method", "method": "bogus", "index": ["a", "b", "c"], "columns": [item for item in BASE if item[0] != "deal_amount"]},
    {"name": "missing_value_precedes_invalid_method", "method": "bogus", "index": ["a", "b", "c"], "columns": [item for item in BASE if item[0] != "value"]},
    {"name": "invalid_method_precedes_missing_indicators", "method": "bogus", "index": ["a", "b", "c"], "columns": [item for item in BASE if item[0] not in ("ffr", "pa", "pos")]},
    {"name": "missing_ffr", "method": "mean", "index": ["a", "b", "c"], "columns": [item for item in BASE if item[0] != "ffr"]},
    {"name": "missing_pa", "method": "mean", "index": ["a", "b", "c"], "columns": [item for item in BASE if item[0] != "pa"]},
    {"name": "missing_ffr_and_pa", "method": "mean", "index": ["a", "b", "c"], "columns": [item for item in BASE if item[0] not in ("ffr", "pa")]},
    {"name": "missing_pos_last", "method": "mean", "index": ["a", "b", "c"], "columns": [item for item in BASE if item[0] != "pos"]},
]

long_count = ["10000000000000000"] + ["1"] * 255 + ["-10000000000000000"]
CASES.insert(9, {"name": "long_pairwise_cancellation", "method": "mean", "index": [f"r{i}" for i in range(257)], "columns": [["ffr", ["1"] * 257], ["pa", [str((i % 5) - 2) for i in range(257)]], ["pos", ["1"] * 257], ["count", long_count], ["deal_amount", ["1"] * 257], ["value", ["1"] * 257]]})
long_special_weights = ["nan" if i % 31 == 0 else ("inf" if i == 128 else ("-inf" if i == 192 else "1")) for i in range(257)]
CASES.insert(10, {"name": "long_pairwise_missing_and_infinite_weights", "method": "value_weighted", "index": [f"r{i}" for i in range(257)], "columns": [["ffr", ["nan" if i % 29 == 0 else "1" for i in range(257)]], ["pa", [str((i % 7) - 3) for i in range(257)]], ["pos", ["nan" if i % 37 == 0 else "1" for i in range(257)]], ["count", long_special_weights], ["deal_amount", ["1"] * 257], ["value", long_special_weights]]})


function, function_ast = load_function(sys.argv[1])
print(json.dumps({"ast": function_ast, "cases": [{**case, "result": snapshot(case, function)} for case in CASES]}, allow_nan=False))
