"""Characterize open decision migration gaps; this is not a Rust parity gate."""

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


source = Path(r"D:\code\github\qlib\qlib\backtest\decision.py")
raw = source.read_bytes()
assert hashlib.sha256(raw).hexdigest() == "a6866d15bc8f3ad1c75bfc3856ccde5245d43f0a2de07b1ad565d6e7e20d8251"
tree = ast.parse(raw, filename=str(source))
names = {
    "OrderDir", "Order", "TradeRange", "IdxTradeRange", "BaseTradeDecision",
    "EmptyTradeDecision", "TradeDecisionWO", "TradeDecisionWithDetails",
}
# Execute unchanged classes together, including the actual dataclass constructor.
# Only imports unrelated to these probes are omitted; no parent class is stubbed.
prelude = ast.parse(
    "from __future__ import annotations\n"
    "from dataclasses import dataclass\n"
    "from enum import IntEnum\n"
    "from abc import abstractmethod\n"
    "from typing import *\n"
    "DecisionType = TypeVar('DecisionType')\n"
)
selected = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
assert {node.name for node in selected} == names
exec(compile(ast.Module(body=prelude.body + selected, type_ignores=[]), str(source), "exec"))


class Calendar:
    def __init__(self, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at

    def get_step_time(self):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError(f"calendar-{self.calls}")
        return f"start-{self.calls}", f"end-{self.calls}"


def construct(fail_at=None, invalid_order=False):
    calendar = Calendar(fail_at)
    strategy = SimpleNamespace(trade_calendar=calendar)
    first = Order("A", 1.0, Order.BUY, None, "explicit-end")
    orders = [first, object()] if invalid_order else [first]
    details = object()
    decision = TradeDecisionWithDetails.__new__(TradeDecisionWithDetails)
    try:
        decision.__init__(orders, strategy, (2, 5), details)
        error = None
    except (RuntimeError, AssertionError) as failure:
        error = type(failure).__name__ + ":" + str(failure)
    return {
        "calls": calendar.calls,
        "decision_start": getattr(decision, "start_time", None),
        "order_start": first.start_time,
        "order_end": first.end_time,
        "strategy_identity": decision.strategy is strategy,
        "order_list_identity": getattr(decision, "order_list", None) is orders,
        "has_total_step": hasattr(decision, "total_step"),
        "has_details": hasattr(decision, "details"),
        "error": error,
    }


success = construct()
assert success == {
    "calls": 2, "decision_start": "start-1", "order_start": "start-2",
    "order_end": "explicit-end", "strategy_identity": True,
    "order_list_identity": True, "has_total_step": True, "has_details": True,
    "error": None,
}
first_failure = construct(fail_at=1)
assert first_failure["calls"] == 1
assert first_failure["strategy_identity"] and not first_failure["has_total_step"]
assert not first_failure["order_list_identity"] and not first_failure["has_details"]
assert first_failure["error"] == "RuntimeError:calendar-1"
second_failure = construct(fail_at=2)
assert second_failure["decision_start"] == "start-1"
assert second_failure["order_list_identity"] and second_failure["order_start"] is None
assert not second_failure["has_details"]
assert second_failure["error"] == "RuntimeError:calendar-2"
partial_failure = construct(invalid_order=True)
assert partial_failure["order_start"] == "start-2"
assert partial_failure["order_list_identity"] and not partial_failure["has_details"]
assert partial_failure["error"] == "AssertionError:"


class MixedDecision(BaseTradeDecision):
    def get_decision(self):
        return self.values


mixed = MixedDecision(SimpleNamespace(trade_calendar=Calendar()))
positive = Order("B", 1.0, Order.BUY, None, None)
zero = Order("C", 0.0, Order.BUY, None, None)
mixed_results = []
for values in ([object(), positive], [positive, object()], [zero, object(), positive], []):
    mixed.values = values
    mixed_results.append(mixed.empty())
assert mixed_results == [True, False, True, True]

live_orders = []
live = TradeDecisionWO(live_orders, SimpleNamespace(trade_calendar=Calendar()))
live_results = []
for amounts in ([], [0.0], [1e-6], [1.000001e-6], [float('nan')],
                [float('-inf')], [float('inf')], [None, 1.0], [1.0, None],
                [0.0, None, 1.0]):
    live_orders[:] = [object() if amount is None else Order('A', amount, Order.BUY, None, None)
                      for amount in amounts]
    assert live.get_decision() is live_orders
    live_results.append(live.empty())
replacement = [Order('A', 1.0, Order.BUY, None, None)]
live.order_list = replacement
assert live.get_decision() is replacement and live_orders is not replacement
replacement[0].amount = 0.0
assert live.empty()

reinit_steps = []
for fail_at in (1, 2, None):
    partial = TradeDecisionWO.__new__(TradeDecisionWO)
    partial.total_step = 7
    try:
        partial.__init__([], SimpleNamespace(trade_calendar=Calendar(fail_at)))
    except RuntimeError:
        assert fail_at is not None
    reinit_steps.append(partial.total_step)
assert reinit_steps == [7, None, None]

print(json.dumps({
    "success": success, "first_failure": first_failure,
    "second_failure": second_failure, "partial_failure": partial_failure,
    "mixed_empty": mixed_results,
    "live_empty": live_results,
    "reinit_total_step": reinit_steps,
}, sort_keys=True))
