"""Characterize object ufunc traversal and stage-wise failures in real NumPy."""
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

events = []
failure = None


class Item:
    def __init__(self, name, stage="input"):
        self.name, self.stage = name, stage

    def run(self, stage, operand):
        events.append([stage, self.name, str(operand)])
        if failure == (stage, self.name):
            raise RuntimeError("sentinel")
        return Item(self.name, stage)

    def __truediv__(self, value):
        return self.run("divide", value)

    def __rsub__(self, value):
        return self.run("one_minus", value)

    def __sub__(self, value):
        return self.run("minus_one", value)

    def __mul__(self, value):
        return self.run("multiply", value)


base = np.array([Item(i) for i in range(6)], dtype=object).reshape(2, 3)
views = {"c": base, "transpose": base.T, "reverse": base[:, ::-1], "both_reverse": base[::-1, ::-1], "slice": base[:, ::2], "empty": base[:0], "singleton": base[:1, :1], "zero_dim": np.array(Item(0), dtype=object)}
views.update(fortran=np.asfortranarray(base), transpose_reverse=base[:, ::-1].T,
             broadcast=np.broadcast_to(base[:1, :], (2, 3)),
             three=base.reshape(1, 2, 3).transpose(2, 0, 1))
rows = []
for view_name, prices in views.items():
    for baseline, direction in ((3.0, 1), (3.0, 0), (0.0, 2), (3.0, 2)):
        for failure in (None, ("divide", 1), ("one_minus", 1), ("minus_one", 1), ("multiply", 1)):
            events.clear()
            row = dict(view=view_name, baseline=baseline, direction=direction,
                       failure=failure, shape=list(prices.shape),
                       names=[v.name for v in prices.flat],
                       strides=[v // prices.itemsize for v in prices.strides])
            try:
                result = price_advantage(prices, baseline, direction)
                values = np.asarray(result, dtype=object)
                row["output"] = [v.name if isinstance(v, Item) else v for v in values.flat]
                row["scalar"] = isinstance(result, Item)
                row["same_last_object"] = isinstance(result, Item) and result.stage == "multiply"
            except Exception as error:
                row["error"] = [type(error).__name__, str(error)]
            row["events"] = list(events)
            rows.append(row)
float_rows = []
for shape in ([], [0], [1], [2, 3]):
    for value in (float("nan"), float("inf"), -float("inf"), -0.0, 0.0, 2.5):
        for baseline in (0.0, -0.0, 3.0, -3.0, float("nan"), float("inf"), -float("inf"), 1e-320):
            for direction in (-1, 0, 1, 2):
                prices = np.full(shape, value, dtype=object)
                row = dict(shape=shape, value=str(value), baseline=str(baseline), direction=direction)
                try:
                    result = price_advantage(prices, baseline, direction)
                    row["output"] = ["nan" if math.isnan(float(v)) else struct.pack(">d", float(v)).hex() for v in np.asarray(result).flat]
                    row["scalar"] = not isinstance(result, np.ndarray)
                except Exception as error:
                    row["error"] = [type(error).__name__, str(error)]
                float_rows.append(row)
print(json.dumps(dict(objects=rows, floats=float_rows)))
