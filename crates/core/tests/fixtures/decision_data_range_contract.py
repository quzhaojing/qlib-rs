import ast
import json
from pathlib import Path

import pandas as pd

source = Path(r"D:\code\github\qlib\qlib\backtest\decision.py")
tree = ast.parse(source.read_text(encoding="utf-8"))
base = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BaseTradeDecision")
method = next(node for node in base.body if isinstance(node, ast.FunctionDef) and node.name == "get_data_cal_range_limit")
method.decorator_list = []


def epsilon_change(value):
    return value - pd.Timedelta(seconds=1)


class Harness:
    get_data_cal_range_limit = None


namespace = {"pd": pd, "epsilon_change": epsilon_change}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
Harness.get_data_cal_range_limit = namespace["get_data_cal_range_limit"]


class Exchange:
    freq = "5min"


class Strategy:
    trade_exchange = Exchange()


class Range:
    def __init__(self, events, fail=False):
        self.events = events
        self.fail = fail

    def clip_time_range(self, start, end):
        self.events.append(["clip", start.isoformat(), end.isoformat()])
        if self.fail:
            raise RuntimeError("clip")
        return start + pd.Timedelta(minutes=10), end - pd.Timedelta(minutes=20)


def run(range_type="full", raise_error=False, has_range=True, fail_at=0, clip_fail=False, indexes=None):
    events = []
    calls = 0

    class Cal:
        @staticmethod
        def locate_index(start, end, freq):
            nonlocal calls
            calls += 1
            events.append(["locate", start.isoformat(), end.isoformat(), freq])
            if calls == fail_at:
                raise RuntimeError(f"locate-{calls}")
            values = indexes or [(100, 387), (110, 367)]
            start_idx, end_idx = values[calls - 1]
            return start, end, start_idx, end_idx

    namespace["Cal"] = Cal
    target = Harness()
    target.start_time = pd.Timestamp("2024-01-02 09:30:00")
    target.end_time = pd.Timestamp("2024-01-02 10:00:00")
    target.strategy = Strategy()
    target.trade_range = Range(events, clip_fail) if has_range else None
    try:
        result = list(target.get_data_cal_range_limit(range_type, raise_error))
        error = None
    except Exception as failure:
        result = None
        error = f"{type(failure).__name__}:{failure}"
    return {"events": events, "result": result, "error": error}


print(json.dumps([
    run(has_range=False),
    run(has_range=False, raise_error=True),
    run("full"),
    run("step"),
    run("bad"),
    run(fail_at=1),
    run(fail_at=2),
    run(clip_fail=True),
], separators=(",", ":")))
