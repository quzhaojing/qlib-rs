"""Real upstream object-array arithmetic over Python built-in payloads."""
import ast
import enum
import json
import math
import struct
import sys
from pathlib import Path
from typing import cast

import numpy as np

float_or_ndarray = object


class OrderDir(enum.IntEnum):
    SELL = 0
    BUY = 1


tree = ast.parse(Path(sys.argv[1]).read_bytes())
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "price_advantage")
future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
exec(compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])), "source", "exec"))


def bits(value):
    return "nan" if math.isnan(value) else struct.pack(">d", value).hex()


def snapshot(value):
    if isinstance(value, complex):
        return dict(kind="complex", real=bits(value.real), imag=bits(value.imag))
    if isinstance(value, float):
        return dict(kind="float", real=bits(value))
    return dict(kind="integer", value=str(value))


samples = []
for value in (False, True):
    samples.append((dict(kind="bool", value=str(value).lower()), value))
for value in (0, 1, -2, 2**53 + 1, 2**53 + 3, 2**1023, -(2**1023), 2**1024, -(2**1024)):
    samples.append((dict(kind="integer", value=str(value)), value))
for value in (0.0, -0.0, 1.25, -2.5, float("nan"), float("inf"), -float("inf")):
    samples.append((dict(kind="float", value=str(value)), value))
for value in (complex(1.25, -2.5), complex(-0.0, 0.0), complex(-0.0, -0.0), complex(float("inf"), 2), complex(1, float("inf")), complex(float("nan"), float("inf"))):
    samples.append((dict(kind="complex", value=str(value.real), imag=str(value.imag)), value))
for kind, value in (("NoneType", None), ("str", "3"), ("bytes", b"3"), ("list", [3]), ("tuple", (3,)), ("dict", {}), ("set", {3}), ("frozenset", frozenset({3})), ("range", range(3)), ("ellipsis", Ellipsis), ("NotImplementedType", NotImplemented)):
    samples.append((dict(kind=kind), value))

rows = []
for descriptor, value in samples:
    for shape in ([], [0], [1], [2, 2]):
        for baseline in (0.0, -0.0, 3.0, -3.0, float("nan"), float("inf"), -float("inf"), 1e-320):
            for direction in (-1, 0, 1, 2):
                prices = np.empty(shape, dtype=object)
                for i in range(prices.size):
                    prices.flat[i] = value
                row = dict(input=descriptor, shape=shape, baseline=str(baseline), direction=direction)
                try:
                    result = price_advantage(prices, baseline, direction)
                    row["output"] = [snapshot(v) for v in np.asarray(result, dtype=object).flat]
                    row["scalar"] = not isinstance(result, np.ndarray)
                except Exception as error:
                    row["error"] = [type(error).__name__, str(error)]
                rows.append(row)
for indices in ((0, 5, 13, 18), (15, 21, 16, 1), (12, 24, 25, 2), (3, 9, 20, 11)):
    for baseline in (0.0, -0.0, 3.0, -3.0, float("nan"), float("inf"), -float("inf"), 1e-320):
        for direction in (-1, 0, 1, 2):
            prices = np.empty([4], dtype=object)
            descriptors = []
            for i, index in enumerate(indices):
                descriptor, value = samples[index]
                prices[i] = value
                descriptors.append(descriptor)
            row = dict(input=descriptors, shape=[4], baseline=str(baseline), direction=direction)
            try:
                result = price_advantage(prices, baseline, direction)
                row["output"] = [snapshot(v) for v in np.asarray(result, dtype=object).flat]
                row["scalar"] = not isinstance(result, np.ndarray)
            except Exception as error:
                row["error"] = [type(error).__name__, str(error)]
            rows.append(row)
print(json.dumps(rows))
