"""Pinned source characterization for get_min_cal's observable cache contract."""

import ast
from datetime import datetime, time, timedelta
import functools
import hashlib
import json
import sys
import threading

import pandas as pd


source_path = sys.argv[1]
source_bytes = open(source_path, "rb").read()
tree = ast.parse(source_bytes.decode("utf-8"), filename=source_path)
names = {"CN_TIME", "US_TIME", "TW_TIME", "get_min_cal"}
body = []
for node in tree.body:
    if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id in names for target in node.targets
    ):
        body.append(node)
    elif isinstance(node, ast.FunctionDef) and node.name == "get_min_cal":
        body.append(node)
module = ast.Module(body=body, type_ignores=[])
ast.fix_missing_locations(module)
namespace = {
    "datetime": datetime, "time": time, "timedelta": timedelta,
    "functools": functools, "pd": pd, "REG_CN": "cn", "REG_US": "us",
    "REG_TW": "tw",
}
exec(compile(module, source_path, "exec"), namespace)
get_min_cal = namespace["get_min_cal"]


def info():
    value = get_min_cal.cache_info()
    return [value.hits, value.misses, value.maxsize, value.currsize]


def capture(call):
    try:
        value = call()
        return {"ok": len(value), "info": info()}
    except Exception as error:
        return {"error": type(error).__name__, "message": str(error), "info": info()}


get_min_cal.cache_clear()
shape_values = [
    get_min_cal(),
    get_min_cal(0),
    get_min_cal(0, "cn"),
    get_min_cal(shift=0),
    get_min_cal(region="cn"),
    get_min_cal(shift=0, region="cn"),
    get_min_cal(region="cn", shift=0),
    get_min_cal(0, region="cn"),
]
shape_values[0].append(time(0, 0))
call_shapes = {
    "distinct": len({id(value) for value in shape_values}),
    "lengths": [len(value) for value in shape_values],
    "info_after_misses": info(),
}
repeat_values = [
    get_min_cal(),
    get_min_cal(0),
    get_min_cal(0, "cn"),
    get_min_cal(shift=0),
    get_min_cal(region="cn"),
    get_min_cal(shift=0, region="cn"),
    get_min_cal(region="cn", shift=0),
    get_min_cal(0, region="cn"),
]
call_shapes["repeat_identity"] = [
    left is right for left, right in zip(shape_values, repeat_values)
]
call_shapes["info_after_hits"] = info()

get_min_cal.cache_clear()
first = get_min_cal(10_000)
for shift in range(10_001, 10_241):
    get_min_cal(shift)
eviction_before = info()
rebuilt = get_min_cal(10_000)
eviction = {
    "before": eviction_before,
    "after": info(),
    "identity_changed": first is not rebuilt,
    "retained_old_length": len(first),
}

get_min_cal.cache_clear()
failures = {
    "unsupported_region_first": capture(lambda: get_min_cal(region="xx")),
    "unsupported_region_second": capture(lambda: get_min_cal(region="xx")),
    "duplicate": capture(lambda: get_min_cal(0, shift=0)),
    "duplicate_region": capture(lambda: get_min_cal(0, "cn", region="cn")),
    "too_many": capture(lambda: get_min_cal(0, "cn", 1)),
    "unexpected": capture(lambda: get_min_cal(other=0)),
    "shift_text": capture(lambda: get_min_cal("0")),
    "region_integer": capture(lambda: get_min_cal(region=1)),
}

get_min_cal.cache_clear()
boolean_cold = capture(lambda: get_min_cal(False))
get_min_cal.cache_clear()
integer_zero = get_min_cal(0)
floating_zero = get_min_cal(0.0)
boolean_false_after_float = get_min_cal(False)
dynamic_inputs = {
    "boolean_cold": boolean_cold,
    "zero_identities": [
        integer_zero is floating_zero,
        integer_zero is boolean_false_after_float,
        floating_zero is boolean_false_after_float,
    ],
    "zero_info": info(),
    "half_minute_first": str(get_min_cal(0.5)[0]),
    "none_region": capture(lambda: get_min_cal(region=None)),
    "unhashable_before": info(),
    "unhashable": capture(lambda: get_min_cal([])),
    "unhashable_after": info(),
}

get_min_cal.cache_clear()
boundaries = {}
for name, shift, region in [
    ("cn_positive_max", 116_906_957, "cn"),
    ("cn_positive_over", 116_906_958, "cn"),
    ("tw_positive_max", 116_906_927, "tw"),
    ("tw_positive_over", 116_906_928, "tw"),
    ("negative_min", -153_722_867, "us"),
    ("negative_under", -153_722_868, "us"),
]:
    boundaries[name] = capture(lambda shift=shift, region=region: get_min_cal(shift, region))


class SlowPandas:
    def __init__(self, delegate, barrier):
        self._delegate = delegate
        self._barrier = barrier

    def date_range(self, *args, **kwargs):
        self._barrier.wait(timeout=10)
        return self._delegate.date_range(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._delegate, name)


get_min_cal.cache_clear()
barrier = threading.Barrier(2)
namespace["pd"] = SlowPandas(pd, barrier)
concurrent_values = []
concurrent_errors = []


def worker():
    try:
        concurrent_values.append(get_min_cal(7, "us"))
    except Exception as error:
        concurrent_errors.append(type(error).__name__ + ": " + str(error))


threads = [threading.Thread(target=worker) for _ in range(2)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join(timeout=15)
namespace["pd"] = pd
cached = get_min_cal(7, "us")
concurrency = {
    "errors": concurrent_errors,
    "threads_alive": [thread.is_alive() for thread in threads],
    "distinct_results": len({id(value) for value in concurrent_values}),
    "cached_is_one_result": any(cached is value for value in concurrent_values),
    "info": info(),
}

print(json.dumps({
    "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
    "call_shapes": call_shapes,
    "eviction": eviction,
    "failures": failures,
    "dynamic_inputs": dynamic_inputs,
    "boundaries": boundaries,
    "concurrency": concurrency,
}, sort_keys=True))
