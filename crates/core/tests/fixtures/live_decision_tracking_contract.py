"""Actual source identity through tracking/update; characterization, not Rust parity."""
import ast
import hashlib
import json
import logging
from pathlib import Path
from types import GeneratorType, SimpleNamespace

get_module_logger = logging.getLogger


def source_method(filename, digest, cls, method):
    path = Path(r"D:\code\github\qlib\qlib\backtest") / filename
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest
    node = next(n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef) and n.name == cls)
    selected = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == method)
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + [selected], type_ignores=[])
    exec(compile(module, str(path), "exec"), globals())


class NestedExecutor:
    pass


class BasePosition:
    ST_NO = "None"


source_method("executor.py", "76ab94ce77691487da6cd41bcfe3fe14b5149e917835bcaaba0ed174afd0fa88",
              "BaseExecutor", "collect_data")
source_method("decision.py", "a6866d15bc8f3ad1c75bfc3856ccde5245d43f0a2de07b1ad565d6e7e20d8251",
              "BaseTradeDecision", "update")
source_method("decision.py", "a6866d15bc8f3ad1c75bfc3856ccde5245d43f0a2de07b1ad565d6e7e20d8251",
              "BaseTradeDecision", "mod_inner_decision")
source_method("decision.py", "a6866d15bc8f3ad1c75bfc3856ccde5245d43f0a2de07b1ad565d6e7e20d8251",
              "BaseTradeDecision", "get_range_limit")


def live_range():
    decision = SimpleNamespace(total_step=99)
    def rule(**kwargs):
        decision.total_step = 3
        return -2, 10
    decision._get_range_limit = rule
    resolved = get_range_limit(decision)
    assert resolved == (0, 2) and decision.total_step == 3
    return resolved


def propagation():
    rule = object()
    outer = SimpleNamespace(trade_range=rule)
    inner = SimpleNamespace(trade_range=None)
    mod_inner_decision(outer, inner)
    retained = inner.trade_range is rule
    # Missing outer metadata must not be read when the inner already has a rule.
    mod_inner_decision(SimpleNamespace(), inner)
    preserved = inner.trade_range is rule
    alias = SimpleNamespace(trade_range=None)
    mod_inner_decision(alias, alias)
    failures = []
    for left, right in [(outer, SimpleNamespace()), (SimpleNamespace(), alias)]:
        try:
            mod_inner_decision(left, right)
        except AttributeError:
            failures.append(True)
        else:
            failures.append(False)
    return [retained, preserved, alias.trade_range is None, *failures]


def tracking(action):
    events = []
    decision = SimpleNamespace(amount=1, range=None)

    def get_range_limit(**kwargs):
        events.append("range")
        return decision.range

    decision.get_range_limit = get_range_limit

    class Executor:
        track_data = True
        _settle_type = "None"
        indicator_config = {}
        trade_exchange = None
        trade_calendar = SimpleNamespace(get_step_time=lambda: (0, 1), step=lambda: events.append("step"))
        trade_account = SimpleNamespace(update_bar_end=lambda *args, **kw:
            events.append(["account", kw["outer_trade_decision"] is decision, decision.amount]))

        def _collect_data(self, trade_decision, level):
            events.append(["collect", trade_decision is decision, trade_decision.amount])
            return [], {}

    driver = collect_data(Executor(), decision)
    yielded = next(driver)
    assert yielded is decision and events == []
    yielded.amount = 9
    if action == "close":
        driver.close()
        assert events == []
        return events
    if action == "range":
        yielded.range = (0, 0)
    try:
        next(driver)
    except StopIteration as complete:
        assert action == "mutate" and complete.value == []
    except ValueError:
        assert action == "range"
    assert decision.amount == 9
    assert events == (["range"] if action == "range" else
                      ["range", ["collect", True, 9], ["account", True, 9], "step"])
    return events


def updating(calendar_fails, strategy_fails):
    events = []
    decision = SimpleNamespace()  # Deliberately not initialized: total_step is absent.

    def length():
        events.append("calendar")
        if calendar_fails:
            raise RuntimeError("calendar")
        return 7

    calendar = SimpleNamespace(get_trade_len=length)

    class Strategy:
        def update_trade_decision(self, current, inner_calendar):
            assert current is decision and current.strategy is self and inner_calendar is calendar
            events.append(["strategy", current.total_step])
            current.marker = "changed"
            if strategy_fails:
                raise RuntimeError("strategy")
            return current

    decision.strategy = Strategy()
    try:
        assert update(decision, calendar) is decision
    except RuntimeError:
        assert calendar_fails or strategy_fails
    assert hasattr(decision, "total_step") is not calendar_fails
    assert hasattr(decision, "marker") is not calendar_fails
    return {"events": events, "total_step": getattr(decision, "total_step", None),
            "marker": getattr(decision, "marker", None)}


print(json.dumps({
    "live_range": live_range(),
    "propagation": propagation(),
    "tracking": {action: tracking(action) for action in ("mutate", "range", "close")},
    "update": [updating(True, False), updating(False, True), updating(False, False)],
}))
