"""Actual CalendarProvider and length-LRU methods with deterministic native-like loaders."""
import abc
import ast
import bisect
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

root = Path(sys.argv[1])
def extract(path, name):
    return next(node for node in ast.parse((root / path).read_text(encoding="utf-8")).body if isinstance(node, ast.ClassDef) and node.name == name)
namespace = {"abc": abc, "np": np, "pd": pd, "bisect": bisect, "OrderedDict": __import__("collections").OrderedDict}
body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        extract("data/data.py", "CalendarProvider"), extract("data/cache.py", "MemCacheUnit"), extract("data/cache.py", "MemCacheLengthUnit")]
exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), "actual-calendar-provider", "exec"), namespace)
base = pd.Timestamp("2024-01-02 09:30")
def timestamp(seconds):
    return None if seconds is None else base + pd.Timedelta(seconds=seconds)

class Provider(namespace["CalendarProvider"]):
    def __init__(self, seconds):
        self.values = [timestamp(value) for value in seconds]
        self.reads = []
        self.fail = False
    def load_calendar(self, freq, future):
        self.reads.append([freq, future])
        if self.fail:
            self.fail = False
            raise ValueError("load")
        return self.values

rows = []
for seconds in [[], [0, 60, 120], [0, 0, 60, 120, 120]]:
    namespace["H"] = {"c": namespace["MemCacheLengthUnit"]()}
    provider = Provider(seconds)
    for start, end in [(None, None), (-60, -30), (0, 0), (30, 90), (120, 60), (180, 240), (None, 0), (0, None)]:
        row = {"values": seconds, "start": start, "end": end}
        for method in ["calendar", "locate_index"]:
            try:
                result = getattr(provider, method)(timestamp(start), timestamp(end), "day", True)
                row[method] = [int((value - base).total_seconds()) for value in result] if method == "calendar" else list(result[2:])
            except IndexError:
                row[method] = "IndexError"
        rows.append(row)

namespace["H"] = {"c": namespace["MemCacheLengthUnit"](2)}
provider = Provider([0, 60, 120])
for freq, future in [("day", False), ("1day", False), ("day", False), ("day", True), ("1day", False)]:
    provider.calendar(freq=freq, future=future)
lru_reads = provider.reads.copy()
namespace["H"]["c"].clear()
provider.fail = True
try:
    provider.calendar(freq="day")
except ValueError:
    pass
provider.calendar(freq="day")
provider.calendar(freq="day")
retry_reads = provider.reads[len(lru_reads):]

# Execute the actual file-storage data property; frequency selection and filesystem
# access are controlled here. Equal frequencies deliberately bypass resampling.
file_class = extract("data/storage/file_storage.py", "FileCalendarStorage")
data_method = next(node for node in file_class.body if isinstance(node, ast.FunctionDef) and node.name == "data")
namespace["Freq"] = lambda frequency: frequency
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), data_method], type_ignores=[])), "actual-file-calendar-data", "exec"), namespace)
mixed = []
for capacity in [1, 2, 3]:
    namespace["H"] = {"c": namespace["MemCacheLengthUnit"](capacity)}
    events = []
    class File:
        data = namespace["data"]
        enable_read_cache = True
        freq = _freq_file = "day"
        def __init__(self, uri):
            self.uri = uri
        def check(self):
            events.append(["check", self.uri])
        def _read_calendar(self):
            events.append(["read", self.uri])
            return ["2024-01-02 09:30:00", "2024-01-02 09:31:00"]
    class MixedProvider(Provider):
        def load_calendar(self, freq, future):
            events.append(["parse", freq, future])
            return list(map(pd.Timestamp, File(freq).data))
    mixed_provider = MixedProvider([])
    for operation, key in [("parsed", "day"), ("raw", "day"), ("parsed", "day"), ("raw", "other"), ("parsed", "day"), ("parsed", "1day"), ("raw", "day"), ("parsed", "day")]:
        if operation == "parsed":
            assert list(mixed_provider.calendar(freq=key)) == [base, base + pd.Timedelta(minutes=1)]
        else:
            assert len(File(key).data) == 2
    mixed.append(events)
print(json.dumps({"rows": rows, "lru_reads": lru_reads, "retry_reads": retry_reads, "mixed": mixed}))
