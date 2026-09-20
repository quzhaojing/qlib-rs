import ast
import json
from pathlib import Path

source = Path(r"D:\code\github\qlib\qlib\backtest\decision.py")
tree = ast.parse(source.read_text(encoding="utf-8"))
base = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BaseTradeDecision")
method = next(node for node in base.body if isinstance(node, ast.FunctionDef) and node.name == "update")
method.decorator_list = []
method.returns = None
for argument in [*method.args.posonlyargs, *method.args.args, *method.args.kwonlyargs]:
    argument.annotation = None


class Harness:
    update = None


namespace = {}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
Harness.update = namespace["update"]


class Calendar:
    def __init__(self, events, value, fail=False):
        self.events = events
        self.value = value
        self.fail = fail

    def get_trade_len(self):
        self.events.append(["calendar"])
        if self.fail:
            raise RuntimeError("calendar")
        return self.value


class Strategy:
    def __init__(self, events, action):
        self.events = events
        self.action = action

    def update_trade_decision(self, decision, calendar):
        self.events.append(["strategy", decision.total_step, calendar is decision.calendar])
        decision.marker = "mutated"
        if self.action == "fail":
            raise RuntimeError("strategy")
        if self.action == "self":
            return decision
        if self.action == "replacement":
            replacement = Harness()
            replacement.marker = "replacement"
            return replacement
        return None


def run(value, action, calendar_fail=False, initial=99):
    events = []
    target = Harness()
    target.total_step = initial
    target.marker = "original"
    target.calendar = Calendar(events, value, calendar_fail)
    target.strategy = Strategy(events, action)
    try:
        result = target.update(target.calendar)
        error = None
        returned = "none" if result is None else ("self" if result is target else result.marker)
    except Exception as failure:
        returned = None
        error = f"{type(failure).__name__}:{failure}"
    return {
        "events": events,
        "total_step": target.total_step,
        "marker": target.marker,
        "returned": returned,
        "error": error,
    }


print(json.dumps([
    run(3, "none"),
    run(0, "self"),
    run(-2, "replacement"),
    run(7, "fail"),
    run(11, "none", calendar_fail=True),
], separators=(",", ":")))
