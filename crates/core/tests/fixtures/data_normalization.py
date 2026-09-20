"""Execute the two normalization functions directly from the pinned Qlib source."""

import ast
import hashlib
import json
import sys
import warnings

import numpy as np
import pandas as pd


source_path = sys.argv[1]
raw = open(source_path, "rb").read()
tree = ast.parse(raw, filename=source_path)
functions = [
    node
    for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name in {"robust_zscore", "zscore"}
]
namespace = {"np": np, "pd": pd, "Union": __import__("typing").Union}
exec(compile(ast.Module(body=functions, type_ignores=[]), source_path, "exec"), namespace)


def scalar(value):
    if value is pd.NA:
        return None
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "nan"
        if np.isposinf(value):
            return "inf"
        if np.isneginf(value):
            return "-inf"
        return float(value)
    return int(value)


def series_case(case_id, dtype, values, operation, post=False):
    source = pd.Series(
        values,
        dtype=dtype,
        index=pd.Index([f"row-{index}" for index in range(len(values))], name="rows"),
        name="signal",
    )
    original = source.copy(deep=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if operation == "zscore":
            output = namespace[operation](source)
        else:
            output = namespace[operation](source, post)
    cases[case_id] = {
        "dtype": str(output.dtype),
        "values": [scalar(value) for value in output.array],
        "index": list(output.index),
        "index_name": output.index.name,
        "name": output.name,
        "input_preserved": source.equals(original),
        "warnings": [f"{type(item.message).__name__}:{item.message}" for item in caught],
    }


cases = {}
series_case("int8_zscore", "int8", [1, 2, 3], "zscore")
series_case("int8_robust", "int8", [1, 2, 3], "robust_zscore")
series_case("uint64_large_robust_post", "uint64", [1, 2, 2**63], "robust_zscore", True)
series_case("float16_robust", "float16", [1, 2, 3], "robust_zscore")
series_case("float32_robust", "float32", [1, 2, 3], "robust_zscore")
series_case("nullable_int8_zscore", "Int8", [1, None, 3], "zscore")
series_case("nullable_float32_zscore", "Float32", [1, None, 3], "zscore")
series_case(
    "float32_cancellation_zscore",
    "float32",
    [1e20, 1.0, -1e20] * 100 + [3.0],
    "zscore",
)
series_case(
    "float64_cancellation_zscore",
    "float64",
    [1e20, 1.0, -1e20] * 100 + [3.0],
    "zscore",
)
series_case(
    "float64_nan_position_zscore",
    "float64",
    [1e20, np.nan, 1.0, -1e20] * 40 + [3.0],
    "zscore",
)
series_case(
    "float32_small_robust_post",
    "float32",
    [1e-20, 2e-20, 3e-20] * 50,
    "robust_zscore",
    True,
)
series_case(
    "float32_large_zscore",
    "float32",
    [3.4e38, -3.4e38] * 100 + [1.0],
    "zscore",
)
series_case(
    "nullable_float64_nan_zscore",
    "Float64",
    [1.0, np.nan, pd.NA, 3.0],
    "zscore",
)
for prefix, dtype, values in [
    ("native_extreme_opposite", "float64", [1e308, -1e308]),
    ("nullable_extreme_opposite", "Float64", [1e308, -1e308]),
    ("native_extreme_same", "float64", [1.7e308, 1.7e308]),
    ("nullable_extreme_same", "Float64", [1.7e308, 1.7e308]),
]:
    series_case(f"{prefix}_zscore", dtype, values, "zscore")
    series_case(f"{prefix}_robust", dtype, values, "robust_zscore")
    series_case(f"{prefix}_robust_post", dtype, values, "robust_zscore", True)

for label, values in {
    "empty": [],
    "single": [5.0],
    "constant": [5.0, 5.0],
    "nan": [1.0, np.nan, 3.0],
    "positive_inf": [1.0, np.inf, 3.0],
    "negative_inf": [1.0, -np.inf, 3.0],
    "both_inf": [-np.inf, np.inf],
    "all_nan": [np.nan, np.nan],
}.items():
    series_case(f"{label}_zscore", "float64", values, "zscore")
    series_case(f"{label}_robust", "float64", values, "robust_zscore")
    series_case(f"{label}_robust_post", "float64", values, "robust_zscore", True)

index = pd.Index(["r2", "r1", "r3"], name="rows")
frame = pd.DataFrame(
    {
        "integer": pd.Series([1, 2, 3], dtype="int32", index=index),
        "float": pd.Series([1.0, np.nan, 3.0], dtype="float32", index=index),
        "masked": pd.Series([1, None, 3], dtype="Int64", index=index),
    },
    index=index,
)
frame_original = frame.copy(deep=True)
for operation, post in [("zscore", False), ("robust_zscore", False), ("robust_zscore", True)]:
    output = namespace[operation](frame, post) if operation == "robust_zscore" else namespace[operation](frame)
    cases[f"mixed_frame_{operation}_{str(post).lower()}"] = {
        "dtypes": [str(dtype) for dtype in output.dtypes],
        "values": [[scalar(value) for value in output[column].array] for column in output.columns],
        "index": list(output.index),
        "index_name": output.index.name,
        "columns": list(output.columns),
        "input_preserved": frame.equals(frame_original),
    }

errors = {}
for case_id, value, operation in [
    ("text_zscore", pd.Series(["a", "b"]), "zscore"),
    ("text_robust", pd.Series(["a", "b"]), "robust_zscore"),
]:
    try:
        namespace[operation](value)
    except Exception as error:  # noqa: BLE001 - the exact source exception is the fixture.
        errors[case_id] = {"class": type(error).__name__, "message": str(error)}

warning_frames = {}
frame_inputs = {
    "native_positive_inf": pd.DataFrame({"a": [1.0, np.inf, 3.0], "b": [2.0, 4.0, 6.0]}),
    "native_both_inf": pd.DataFrame({"a": [-np.inf, np.inf], "b": [-np.inf, np.inf]}),
    "two_native_blocks": pd.DataFrame(
        {
            "a": pd.Series([1.0, np.inf, 3.0], dtype="float32"),
            "b": pd.Series([1.0, np.inf, 3.0], dtype="float64"),
        }
    ),
    "nullable_positive_inf": pd.DataFrame(
        {"a": pd.Series([1.0, np.inf, 3.0], dtype="Float32")}
    ),
    "native_extreme_opposite": pd.DataFrame(
        {"a": pd.Series([1e308, -1e308], dtype="float64")}
    ),
    "nullable_extreme_opposite": pd.DataFrame(
        {"a": pd.Series([1e308, -1e308], dtype="Float64")}
    ),
    "native_extreme_same": pd.DataFrame(
        {"a": pd.Series([1.7e308, 1.7e308], dtype="float64")}
    ),
    "nullable_extreme_same": pd.DataFrame(
        {"a": pd.Series([1.7e308, 1.7e308], dtype="Float64")}
    ),
}
for frame_id, value in frame_inputs.items():
    for operation, post in [("zscore", False), ("robust_zscore", False), ("robust_zscore", True)]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if operation == "zscore":
                namespace[operation](value)
            else:
                namespace[operation](value, post)
        warning_frames[f"{frame_id}_{operation}_{str(post).lower()}"] = [
            f"{type(item.message).__name__}:{item.message}" for item in caught
        ]

print(
    json.dumps(
        {
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "cases": cases,
            "errors": errors,
            "warning_frames": warning_frames,
        },
        sort_keys=True,
    )
)
