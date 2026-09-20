"""Characterize the unchanged upstream qlib.backtest.format_decisions function."""

from __future__ import annotations

import ast
import json
from pathlib import Path


source_path = Path(r"D:\code\github\qlib\qlib\backtest\__init__.py")
module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "format_decisions")
namespace = {}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), namespace)
format_decisions = namespace["format_decisions"]


class Calendar:
    def __init__(self, name, frequency, events, fail_on=None):
        self.name = name
        self.frequency = frequency
        self.events = events
        self.fail_on = fail_on
        self.calls = 0

    def get_freq(self):
        self.calls += 1
        self.events.append(["frequency", self.name, self.calls])
        if self.calls == self.fail_on:
            raise RuntimeError(f"frequency:{self.name}:{self.calls}")
        return self.frequency


class Strategy:
    def __init__(self, calendar):
        self.trade_calendar = calendar


class Decision:
    def __init__(self, name, frequency, events, fail_on=None):
        self.name = name
        self.strategy = Strategy(Calendar(name, frequency, events, fail_on))


def serialize(tree):
    if tree is None:
        return None
    frequency, items = tree
    return {
        "frequency": frequency,
        "items": [[decision.name, serialize(nested)] for decision, nested in items],
    }


def run(name, specifications):
    events = []
    decisions = [
        Decision(
            specification[0],
            specification[1],
            events,
            specification[2] if len(specification) == 3 else None,
        )
        for specification in specifications
    ]
    try:
        output = serialize(format_decisions(decisions))
        error = None
    except RuntimeError as exception:
        output = None
        error = str(exception)
    return {"name": name, "output": output, "events": events, "error": error}


cases = [
    run("empty", []),
    run("single", [("d0", "day")]),
    run("flat", [("d0", "day"), ("d1", "day"), ("d2", "day")]),
    run(
        "nested",
        [("d0", "day"), ("m0", "1min"), ("m1", "1min"), ("d1", "day"), ("m2", "1min"), ("d2", "day")],
    ),
    run("irregular", [("d0", "day"), ("h0", "hour"), ("m0", "1min"), ("h1", "hour"), ("d1", "day")]),
    run("root_failure", [("d0", "day", 1), ("tail", "tick")]),
    run("scan_failure", [("d0", "day"), ("m0", "1min", 1), ("tail", "tick")]),
    run("nested_failure", [("d0", "day"), ("m0", "1min", 2), ("d1", "day"), ("tail", "tick")]),
    run("final_nested_failure", [("d0", "day"), ("m0", "1min", 2)]),
]
print(json.dumps(cases, separators=(",", ":")))
