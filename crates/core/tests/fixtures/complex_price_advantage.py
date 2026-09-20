"""Exercise the actual upstream function with both NumPy complex dtypes."""
import ast
import enum
import itertools
import json
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


def snapshot(value):
    array = np.asarray(value)
    return dict(dtype="scalar" if isinstance(value, complex) else str(array.dtype),
                shape=list(array.shape),
                bits=[[struct.pack(">d", float(v.real)).hex(), struct.pack(">d", float(v.imag)).hex()] for v in array.flat])


components = [-float("inf"), -1e308, -1e30, -2.5, -0.0, 0.0, 1.25, 1e-30, 1e30, 1e308, float("inf"), float("nan")]
rows = []
with np.errstate(all="ignore"):
    for dtype, shape, pair in itertools.product(("complex64", "complex128"), ([], [0], [1], [2, 2]), itertools.product(components, repeat=2)):
        count = int(np.prod(shape))
        data = [complex(*pair)] * count
        prices = np.array(data, dtype=dtype).reshape(shape)
        for baseline in (0.0, -0.0, 3.0, -3.0, float("nan"), float("inf"), -float("inf"), 1e-320, 1e300, 1e-40, 1e-6):
            for direction in (-1, 0, 1, 2):
                row = dict(dtype=dtype, shape=shape, values=[[str(v.real), str(v.imag)] for v in prices.flat], baseline=str(baseline), direction=direction)
                try:
                    row["output"] = snapshot(price_advantage(prices, baseline, direction))
                except ValueError as error:
                    row["error"] = str(error)
                rows.append(row)
print(json.dumps(rows))
