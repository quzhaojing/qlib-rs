"""Unchanged upstream execution manager + locator, with deterministic source data."""
import ast
import bisect
import json
from pathlib import Path
import sys
from datetime import datetime
from abc import abstractmethod
import numpy as np
import pandas as pd

root = Path(sys.argv[1])
def extract(path, cls, method=None):
    tree = ast.parse((root / path).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef if cls else ast.FunctionDef) and n.name == (cls or method))
    return next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == method) if cls and method else node

body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        extract("utils/time.py", None, "epsilon_change"),
        extract("data/data.py", "CalendarProvider", "locate_index"),
        extract("backtest/utils.py", "TradeCalendarManager"),
        extract("utils/time.py", None, "concat_date_time"),
        extract("backtest/decision.py", "TradeRange"),
        extract("backtest/decision.py", "TradeRangeByTime")]
namespace = {"pd": pd, "np": np, "bisect": bisect, "datetime": datetime, "abstractmethod": abstractmethod}
exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), "upstream-calendar", "exec"), namespace)

class CalendarSource:
    def __init__(self):
        self.values = np.array([pd.Timestamp("2024-01-02 09:30") + pd.Timedelta(minutes=i) for i in range(5)], dtype=object)
    def calendar(self, **kwargs):
        return self.values
    def _get_calendar(self, **kwargs):
        return self.values, {value: index for index, value in enumerate(self.values)}
    locate_index = namespace["locate_index"]

namespace["Cal"] = CalendarSource()
manager = namespace["TradeCalendarManager"]
rows = []
for start, end in [(0, 120), (30, 150), (180, 60), (-60, -60), (240, 240)]:
    base = pd.Timestamp("2024-01-02 09:30")
    calendar = manager("1min", base + pd.Timedelta(seconds=start), base + pd.Timedelta(seconds=end))
    intervals = []
    for shift in [0, 1, -1, 6, -6]:
        try:
            intervals.append([str(value) for value in calendar.get_step_time(shift=shift)])
        except IndexError:
            intervals.append("IndexError")
    steps = 0
    while not calendar.finished():
        calendar.step()
        steps += 1
    try:
        calendar.step()
    except RuntimeError:
        pass
    else:
        raise AssertionError("exhausted step succeeded")
    rows.append({"indices": [calendar.start_index, calendar.end_index], "length": calendar.trade_len,
                 "range": calendar.get_range_idx(base - pd.Timedelta(minutes=1), base + pd.Timedelta(minutes=10)),
                 "intervals": intervals, "steps": steps, "finished": calendar.finished(),
                 "time_range": namespace["TradeRangeByTime"]("09:31", "09:33")(calendar)})
print(json.dumps(rows))
