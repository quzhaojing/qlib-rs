"""Real NumPy oracle for the unmodified upstream function body."""
import ast
import enum
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


def token(value):
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value)).lower()
    return str(value)


def snapshot(value):
    array = np.asarray(value)
    return dict(dtype="scalar" if isinstance(value, float) else str(array.dtype),
                shape=list(array.shape),
                bits=[struct.pack(">d", float(v)).hex() for v in array.flat])


rows = []
with np.errstate(all="ignore"):
    for dtype in ("scalar", "float16", "float32", "float64", "bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"):
        for shape in ([], [0], [1], [2, 2]):
            if dtype == "scalar" and shape:
                continue
            for sample in ("regular", "special"):
                if dtype.startswith("float") or dtype == "scalar":
                    data = [1.25, -2.5, 0.0, -0.0] if sample == "regular" else [float("nan"), float("inf"), -float("inf"), 1e-30]
                elif dtype.startswith("int"):
                    data = [-2, 0, 1, 3] if sample == "regular" else [np.iinfo(dtype).min, np.iinfo(dtype).max, 0, 1]
                elif dtype.startswith("uint"):
                    data = [0, 1, 2, 3] if sample == "regular" else [np.iinfo(dtype).max, 0, 1, 2]
                else:
                    data = [True, False, True, False]
                count = int(np.prod(shape))
                prices = float(data[0]) if dtype == "scalar" else np.array(data[:count], dtype=dtype).reshape(shape)
                for baseline in (0.0, -0.0, 3.0, -3.0, float("nan"), float("inf"), -float("inf"), 1e-320, 1e300):
                    for direction in (-1, 0, 1, 2):
                        row = dict(dtype=dtype, shape=shape, values=[token(v) for v in np.asarray(prices).flat], baseline=str(baseline), direction=direction)
                        try:
                            row["output"] = snapshot(price_advantage(prices, baseline, direction))
                        except ValueError as error:
                            row["error"] = str(error)
                        rows.append(row)
print(json.dumps(rows))
