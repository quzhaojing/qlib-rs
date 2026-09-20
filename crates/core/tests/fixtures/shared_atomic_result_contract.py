"""Freeze actual BaseExecutor result-list aliases through reporting and return-value update."""
import ast
import hashlib
import json
from pathlib import Path
from types import GeneratorType, SimpleNamespace

source = Path(r"D:\code\github\qlib\qlib\backtest\executor.py")
raw = source.read_bytes()
assert hashlib.sha256(raw).hexdigest() == "76ab94ce77691487da6cd41bcfe3fe14b5149e917835bcaaba0ed174afd0fa88"
base = next(node for node in ast.parse(raw).body if isinstance(node, ast.ClassDef) and node.name == "BaseExecutor")
method = next(node for node in base.body if isinstance(node, ast.FunctionDef) and node.name == "collect_data")


class NestedExecutor:
    pass


class BasePosition:
    ST_NO = "None"


exec(compile(ast.Module(body=ast.parse("from __future__ import annotations").body + [method], type_ignores=[]), str(source), "exec"))


def run(fail):
    decision = SimpleNamespace(marker=None, get_range_limit=lambda **kwargs: None)
    retained = []

    class Account:
        def update_bar_end(self, *args, **kwargs):
            assert kwargs["outer_trade_decision"] is decision
            decision.marker = 9
            retained.append(kwargs["trade_info"])

    class Sink(dict):
        def update(self, values):
            values["execute_result"].clear()
            super().update(values)
            if fail:
                raise RuntimeError("sink")

    class Executor:
        track_data = False
        _settle_type = "None"
        indicator_config = {}
        trade_exchange = object()
        trade_account = Account()
        trade_calendar = SimpleNamespace(get_step_time=lambda: (1, 2), step=lambda: None)

        def _collect_data(self, **kwargs):
            rows = [object()]
            return rows, {"trade_info": rows}

    Executor.collect_data = collect_data
    sink = Sink()
    generator = Executor().collect_data(decision, return_value=sink)
    try:
        next(generator)
    except StopIteration as complete:
        ok = True
        assert complete.value is retained[0]
    except RuntimeError:
        ok = False
    return {"ok": ok, "bar_len": len(retained[0]), "sink_len": len(sink["execute_result"]),
            "same_list": retained[0] is sink["execute_result"], "decision_marker": decision.marker}


print(json.dumps({str(fail): run(fail) for fail in (False, True)}))
