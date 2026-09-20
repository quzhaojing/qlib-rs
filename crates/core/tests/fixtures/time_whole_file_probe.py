"""Live, source-pinned characterization for qlib/utils/time.py."""

import ast
import bisect
from datetime import date, datetime, time, timedelta
import functools
import hashlib
import inspect
import json
import re
import sys
from types import SimpleNamespace

import pandas as pd


source_path = sys.argv[1]
source_bytes = open(source_path, "rb").read()
source_text = source_bytes.decode("utf-8")
tree = ast.parse(source_text, filename=source_path)
public_names = {
    "CN_TIME", "US_TIME", "TW_TIME", "get_min_cal", "is_single_value", "Freq",
    "time_to_day_index", "get_day_min_idx_range", "concat_date_time", "cal_sam_minute",
    "epsilon_change",
}
body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in public_names:
        body.append(node)
    elif isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id in public_names for target in node.targets
    ):
        body.append(node)
module = ast.Module(body=body, type_ignores=[])
ast.fix_missing_locations(module)
namespace = {
    "bisect": bisect, "date": date, "datetime": datetime, "time": time,
    "timedelta": timedelta, "functools": functools, "pd": pd, "re": re,
    "C": SimpleNamespace(min_data_shift=0), "REG_CN": "cn", "REG_US": "us", "REG_TW": "tw",
}
exec(compile(module, source_path, "exec"), namespace)


def captured(call):
    try:
        value = call()
        return {"ok": str(value), "type": type(value).__name__}
    except Exception as error:
        return {"error": type(error).__name__, "message": str(error)}


Freq = namespace["Freq"]
get_min_cal = namespace["get_min_cal"]
is_single_value = namespace["is_single_value"]
time_to_day_index = namespace["time_to_day_index"]
get_day_min_idx_range = namespace["get_day_min_idx_range"]
concat_date_time = namespace["concat_date_time"]
cal_sam_minute = namespace["cal_sam_minute"]
epsilon_change = namespace["epsilon_change"]

calendars = {}
for region in ["cn", "us", "tw"]:
    calendar = get_min_cal(region=region)
    calendars[region] = {
        "length": len(calendar), "first": str(calendar[0]), "last": str(calendar[-1]),
        "type": type(calendar).__name__, "element_type": type(calendar[0]).__name__,
    }

cached = get_min_cal()
cache_identity = cached is get_min_cal()
cached.append(time(0, 0))
mutated_cache = {"same": cached is get_min_cal(), "length": len(get_min_cal())}
get_min_cal.cache_clear()

default_call = get_min_cal()
positional_call = get_min_cal(0, "cn")
region_keyword_call = get_min_cal(region="cn")
all_keyword_call = get_min_cal(shift=0, region="cn")
default_call.append(time(0, 1))
cache_call_shapes = {
    "identities": [
        default_call is positional_call,
        default_call is region_keyword_call,
        default_call is all_keyword_call,
        positional_call is region_keyword_call,
        positional_call is all_keyword_call,
        region_keyword_call is all_keyword_call,
    ],
    "lengths_after_default_mutation": [
        len(default_call), len(positional_call), len(region_keyword_call), len(all_keyword_call)
    ],
    "info": list(get_min_cal.cache_info()),
}
get_min_cal.cache_clear()

freq_copy = Freq(Freq("5MIN"))
recent_strings = Freq.get_recent_freq("day", ["1min", "2min"])
recent_objects = Freq.get_recent_freq("day", [Freq("1min"), Freq("2min")])

namespace["C"].min_data_shift = 1
aligned = cal_sam_minute(pd.Timestamp("2021-03-03 09:30:45+08:00"), 5)
namespace["C"].min_data_shift = 0

class_symbols = []
for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name == "Freq":
        for member in node.body:
            if isinstance(member, ast.FunctionDef):
                class_symbols.append(member.name)
            elif isinstance(member, ast.Assign):
                class_symbols.extend(
                    target.id for target in member.targets if isinstance(target, ast.Name)
                )

result = {
    "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
    "module_symbols": sorted(public_names),
    "freq_symbols": sorted(class_symbols),
    "defaults": {
        name: str(inspect.signature(namespace[name]))
        for name in ["get_min_cal", "is_single_value", "time_to_day_index",
                     "get_day_min_idx_range", "concat_date_time", "cal_sam_minute",
                     "epsilon_change"]
    },
    "constants": {
        name: {"type": type(namespace[name]).__name__,
               "values": [value.strftime("%H:%M:%S") for value in namespace[name]],
               "element_type": type(namespace[name][0]).__name__}
        for name in ["CN_TIME", "US_TIME", "TW_TIME"]
    },
    "calendar": {
        "regions": calendars, "cache_identity": cache_identity, "mutated_cache": mutated_cache,
        "call_shapes": cache_call_shapes, "post_clear_length": len(get_min_cal()),
        "unsupported": captured(lambda: get_min_cal(region="xx")),
    },
    "frequency": {
        "attributes": [Freq.NORM_FREQ_MONTH, Freq.NORM_FREQ_WEEK, Freq.NORM_FREQ_DAY,
                       Freq.NORM_FREQ_MINUTE, Freq.SUPPORT_CAL_LIST],
        "copy": {"count": freq_copy.count, "base": freq_copy.base, "text": str(freq_copy)},
        "repr": repr(Freq("day")), "equal_string": Freq("day") == "D",
        "invalid_init": captured(lambda: Freq(1)),
        "invalid_equal": captured(lambda: Freq("day") == 1),
        "parse_type": type(Freq.parse("2W")).__name__,
        "timedelta": {"type": type(Freq.get_timedelta(2, "min")).__name__,
                      "nanoseconds": Freq.get_timedelta(2, "min").value,
                      "invalid": captured(lambda: Freq.get_timedelta(1, "week"))},
        "delta": {"type": type(Freq.get_min_delta("day", "1min")).__name__,
                  "value": Freq.get_min_delta("day", "1min")},
        "recent_strings": {"type": type(recent_strings).__name__, "value": str(recent_strings)},
        "recent_objects": {"type": type(recent_objects).__name__, "value": str(recent_objects)},
    },
    "single_value": {
        "default": is_single_value(pd.Timestamp("2021-01-01 11:29"),
                                   pd.Timestamp("2021-01-01 12:29"), pd.Timedelta("1min")),
        "unsupported": captured(lambda: is_single_value(pd.Timestamp("2021-01-01"),
                                pd.Timestamp("2021-01-02"), pd.Timedelta("1min"), "xx")),
    },
    "index": {
        "string": captured(lambda: time_to_day_index("9:30")),
        "datetime": captured(lambda: time_to_day_index(datetime(2021, 1, 1, 13, 0))),
        "closed": get_day_min_idx_range("8:30", "14:59", "10min", "cn"),
        "ignored_unit": get_day_min_idx_range("9:30", "9:40", "2day", "cn"),
        "zero_step": captured(lambda: get_day_min_idx_range("9:30", "9:40", "0min", "cn")),
        "outside": captured(lambda: time_to_day_index("11:30")),
    },
    "concat": {"type": type(concat_date_time(date(2020, 2, 29),
                                               time(1, 2, 3, 456789))).__name__,
               "value": str(concat_date_time(date(2020, 2, 29), time(1, 2, 3, 456789)))},
    "alignment": {"type": type(aligned).__name__, "value": str(aligned),
                  "timezone": str(aligned.tz),
                  "zero_step": captured(lambda: cal_sam_minute(
                      pd.Timestamp("2021-01-01 10:00"), 0))},
    "epsilon": {
        "default": captured(lambda: epsilon_change(
            pd.Timestamp("2021-01-01 00:00:00.123456789"))),
        "forward": captured(lambda: epsilon_change(pd.Timestamp("2021-01-01"), "forward")),
        "nat": captured(lambda: epsilon_change(pd.NaT)),
        "invalid": captured(lambda: epsilon_change(pd.Timestamp("2021-01-01"), "Backward")),
    },
}
print(json.dumps(result, sort_keys=True))
