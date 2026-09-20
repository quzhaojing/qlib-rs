"""Pinned characterization for remaining public qlib.utils.time boundaries."""

import ast
import bisect
from datetime import date, datetime, time, timedelta
import functools
import hashlib
import json
import re
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd


source_path = sys.argv[1]
source_bytes = open(source_path, "rb").read()
tree = ast.parse(source_bytes.decode("utf-8"), filename=source_path)
names = {
    "CN_TIME", "US_TIME", "TW_TIME", "get_min_cal", "Freq", "concat_date_time",
    "cal_sam_minute", "epsilon_change",
}
body = []
for node in tree.body:
    if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id in names for target in node.targets
    ):
        body.append(node)
    elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
        body.append(node)
module = ast.Module(body=body, type_ignores=[])
ast.fix_missing_locations(module)
namespace = {
    "bisect": bisect, "date": date, "datetime": datetime, "time": time,
    "timedelta": timedelta, "functools": functools, "pd": pd, "re": re,
    "C": SimpleNamespace(min_data_shift=0), "REG_CN": "cn", "REG_US": "us",
    "REG_TW": "tw",
}
exec(compile(module, source_path, "exec"), namespace)
Freq = namespace["Freq"]


def captured(call):
    try:
        return {"ok": describe(call())}
    except Exception as error:
        return {"error": type(error).__name__, "message": str(error)}


def describe(value):
    if value is None:
        return None
    if isinstance(value, Freq):
        return {"kind": "Freq", "text": str(value), "count": value.count, "base": value.base}
    if isinstance(value, str):
        return {"kind": "str", "text": value}
    if value is pd.NaT:
        return {"kind": "NaT", "text": "NaT"}
    if isinstance(value, pd.Timestamp):
        return {
            "kind": "Timestamp", "ticks": int(value.asm8.view("i8")), "unit": value.unit,
            "timezone": None if value.tz is None else str(value.tz), "text": str(value),
        }
    raise TypeError(type(value).__name__)


frequency_cases = {
    "first_text": ("day", ["01MIN"]),
    "first_freq": ("day", [Freq("01MIN")]),
    "later_text": ("day", ["1min", "02MIN"]),
    "later_freq": ("day", ["1min", Freq("02MIN")]),
    "tie": ("day", ["2min", "02MIN"]),
    "none": ("1min", ["day", "week"]),
    "huge": ("9" * 80 + "min", ["1min", "2min"]),
    "invalid_after_eligible": ("day", ["1min", "bad"]),
}
frequency = {
    name: captured(lambda base=base, values=values: Freq.get_recent_freq(base, values))
    for name, (base, values) in frequency_cases.items()
}

concat = {
    "minimum": describe(namespace["concat_date_time"](date(1, 1, 1), time(1, 2, 3, 456789))),
    "ordinary": describe(namespace["concat_date_time"](date(2020, 2, 29), time(1, 2, 3, 456789))),
    "maximum": describe(namespace["concat_date_time"](date(9999, 12, 31), time(23, 59, 59, 999999))),
}

epsilon_inputs = {
    "seconds": pd.Timestamp(np.datetime64("2021-01-01T00:00:00", "s")),
    "microseconds": pd.Timestamp(np.datetime64("2021-01-01T00:00:00.123456", "us")),
    "nanoseconds": pd.Timestamp("2021-01-01 00:00:00.123456789"),
    "iana": pd.Timestamp("2021-01-01 09:30:45.123456789", tz="Asia/Shanghai"),
    "nat": pd.NaT,
}
epsilon = {}
for name, value in epsilon_inputs.items():
    for direction in ["backward", "forward"]:
        epsilon[f"{name}_{direction}"] = captured(
            lambda value=value, direction=direction: namespace["epsilon_change"](value, direction)
        )
epsilon["nat_invalid_direction"] = captured(
    lambda: namespace["epsilon_change"](pd.NaT, "Backward")
)
epsilon["minimum_backward"] = captured(
    lambda: namespace["epsilon_change"](pd.Timestamp.min, "backward")
)
epsilon["maximum_forward"] = captured(
    lambda: namespace["epsilon_change"](pd.Timestamp.max, "forward")
)

alignment_inputs = {
    "naive": pd.Timestamp("2021-01-01 10:38:45.123456789"),
    "utc": pd.Timestamp("2021-01-01 10:38:45.123456789", tz="UTC"),
    "fixed": pd.Timestamp("2021-01-01 10:38:45.123456789+08:00"),
    "new_york_summer": pd.Timestamp("2021-07-01 14:38:45.123456789Z").tz_convert("America/New_York"),
    "new_york_winter": pd.Timestamp("2021-01-01 15:38:45.123456789Z").tz_convert("America/New_York"),
}
alignment = {
    name: captured(lambda value=value: namespace["cal_sam_minute"](value, 5))
    for name, value in alignment_inputs.items()
}
alignment["nat"] = captured(lambda: namespace["cal_sam_minute"](pd.NaT, 5))
alignment["zero_step"] = captured(
    lambda: namespace["cal_sam_minute"](pd.Timestamp("2021-01-01 10:38"), 0)
)

print(json.dumps({
    "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
    "frequency": frequency,
    "concat": concat,
    "epsilon": epsilon,
    "alignment": alignment,
}, sort_keys=True))
