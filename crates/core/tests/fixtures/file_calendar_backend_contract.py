"""Actual Qlib storage/property/resampling methods with isolated configuration."""
import ast
import bisect
from datetime import date, datetime, time, timedelta
import functools
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd

root = Path(sys.argv[1])

def selected(path, names):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = [n for n in tree.body if
            isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names or
            isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    return compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec")

class Conf(dict):
    min_data_shift = 0
    DEFAULT_FREQ = "__DEFAULT_FREQ"

ns = dict(globals(), REG_CN="cn", REG_US="us", REG_TW="tw", C=Conf(region="cn"), CalVT=str)
exec(selected(root / "utils/time.py", {"CN_TIME", "US_TIME", "TW_TIME", "get_min_cal", "Freq", "concat_date_time", "cal_sam_minute"}), ns)
exec(selected(root / "utils/resam.py", {"resam_calendar"}), ns)
path = root / "data/storage/file_storage.py"
tree = ast.parse(path.read_text(encoding="utf-8"))
methods = []
for cls in tree.body:
    if isinstance(cls, ast.ClassDef) and cls.name in ("FileStorageMixin", "FileCalendarStorage"):
        methods += [m for m in cls.body if isinstance(m, ast.FunctionDef) and m.name in
                    {"support_freq", "check", "_freq_file", "file_name", "uri", "data", "_read_calendar"}]
definition = ast.ClassDef(name="Storage", bases=[], keywords=[], body=methods, decorator_list=[])
exec(compile(ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[])), str(path), "exec"), ns)

def snapshot(storage):
    try:
        values = storage.data
        return {"values": [["text", v] if isinstance(v, str) else ["timestamp", pd.Timestamp(v).isoformat()] for v in values],
                "selected": str(storage._freq_file), "file": storage.file_name}
    except Exception as error:
        return {"error": "Value" if isinstance(error, ValueError) else "Other"}

result = []
with tempfile.TemporaryDirectory() as directory:
    base = Path(directory)
    (base / "calendars").mkdir()
    raw = "2024-01-02 09:31:00\n2024-01-02 09:30:00\n2024-01-02 09:31:00\n2024-01-03 09:30:00\n"
    (base / "calendars/1min.txt").write_bytes(raw.encode())
    (base / "calendars/1min_future.txt").write_bytes(b"bad-date\n")
    (base / "calendars/ignore.bin").write_bytes(b"")
    ns["H"] = {"c": {}}
    for frequency, future in [("1min", False), ("5min", False), ("day", False), ("2day", False),
                              ("1min", True), ("5min", True), ("bad", False), ("0min", False)]:
        storage = ns["Storage"]()
        storage.provider_uri = {Conf.DEFAULT_FREQ: str(base)}
        storage.dpm = SimpleNamespace(get_data_uri=lambda freq: base)
        storage.freq, storage.future = frequency, future
        storage.region, storage.enable_read_cache, storage.storage_name = "cn", True, "calendar"
        result.append([frequency, future, snapshot(storage)])
print(json.dumps(result))
