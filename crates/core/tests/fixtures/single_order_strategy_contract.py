"""Run unchanged strategy, order helper, order and decision classes with controlled infrastructure."""
import ast
import json
import sys
from abc import abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar, Generic, List, TypeVar, cast

import pandas as pd

DecisionType = TypeVar("DecisionType")
root = Path(sys.argv[1])

class BaseStrategy:
    def __init__(self):
        pass

nodes = []
for path, names in [
    ("backtest/decision.py", ["OrderDir", "Order", "OrderHelper", "BaseTradeDecision", "TradeDecisionWO"]),
    ("rl/strategy/single_order.py", ["SingleOrderStrategy"]),
]:
    tree = ast.parse((root / path).read_text(encoding="utf-8"))
    nodes.extend(next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
                 for name in names)
module = ast.Module(body=[ast.ImportFrom(module="__future__",
    names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
exec(compile(ast.fix_missing_locations(module), "unchanged-single-order-source", "exec"), globals())

def run(failure=None, custom=False, ranged=True):
    events = []
    original = Order("A", 7.0, OrderDir.SELL, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"))
    original.deal_amount, original.factor = 4.0, 2.0
    trade_range = object() if ranged else None
    strategy = SingleOrderStrategy(original, trade_range)
    created = []
    class Helper:
        def create(self, **kwargs):
            events.append(["create", kwargs["code"], kwargs["amount"], int(kwargs["direction"])])
            if failure == "create":
                raise ValueError("create")
            order = OrderHelper.create(**kwargs)
            if custom:
                order.start_time = pd.Timestamp("2022-01-01")
            created.append(order)
            return order
    class Calendar:
        count = 0
        def get_step_time(self):
            self.count += 1
            events.append(["calendar", self.count])
            if failure == "calendar" + str(self.count):
                raise ValueError(failure)
            start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=self.count)
            return start, start + pd.Timedelta(hours=1)
    helper = Helper()
    strategy.common_infra = {"trade_exchange": SimpleNamespace(get_order_helper=lambda: helper)}
    strategy.trade_calendar = Calendar()
    decisions = []
    try:
        for previous in [None, ["ignored"]]:
            decision = strategy.generate_trade_decision(previous)
            order = decision.order_list[0]
            assert decision.strategy is strategy and decision.trade_range is trade_range
            assert order is created[-1] and order is not original
            decisions.append({"decision": [str(decision.start_time), str(decision.end_time)],
                "order": [str(order.start_time), str(order.end_time)],
                "amount": order.amount, "deal_amount": order.deal_amount, "factor": order.factor})
            order.deal_amount = 99.0
        error = None
    except ValueError as caught:
        error = str(caught)
    assert original.deal_amount == 4.0 and original.factor == 2.0
    return {"events": events, "decisions": decisions, "error": error}

print(json.dumps({"normal": run(), "custom": run(custom=True, ranged=False),
    **{failure: run(failure) for failure in ["create", "calendar1", "calendar2"]}}))
