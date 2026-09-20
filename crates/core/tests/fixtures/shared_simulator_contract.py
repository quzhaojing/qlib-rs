"""Actual source extraction/collection with duplicated live orders and list mutation."""
import ast
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

root = Path(r"D:\code\github\qlib\qlib\backtest")
decision = (root / "decision.py").read_bytes()
executor = (root / "executor.py").read_bytes()
assert hashlib.sha256(decision).hexdigest() == "a6866d15bc8f3ad1c75bfc3856ccde5245d43f0a2de07b1ad565d6e7e20d8251"
assert hashlib.sha256(executor).hexdigest() == "76ab94ce77691487da6cd41bcfe3fe14b5149e917835bcaaba0ed174afd0fa88"
body = ast.parse("from __future__ import annotations").body
body += [node for node in ast.parse(decision).body if isinstance(node, ast.ClassDef) and node.name in {"OrderDir", "Order"}]
tree = ast.parse(executor)
body += [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_retrieve_orders_from_decision"]
simulator = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SimulatorExecutor")
simulator.bases = []
simulator.body = [node for node in simulator.body if
    isinstance(node, ast.FunctionDef) and node.name in {"_get_order_iterator", "_collect_data"}
    or isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in {"TT_SERIAL", "TT_PARAL"} for target in node.targets)]
body.append(simulator)
exec(compile(ast.Module(body=body, type_ignores=[]), str(root / "executor.py"), "exec"))


def run(mode, fail):
    buy = Order("B", 1.0, Order.BUY, None, None)
    sell = Order("S", 1.0, Order.SELL, None, None)
    original = [sell, buy, buy]
    events = []

    class Calendar:
        def get_step_time(self):
            events.append("calendar")
            value = pd.Timestamp("2024-01-02 09:30:00")
            return value, value

    class Exchange:
        calls = 0

        def deal_order(self, order, trade_account, dealt_order_amount):
            self.calls += 1
            original.clear()
            order.deal_amount = float(self.calls)
            order.factor = 2.0
            events.append([order.stock_id, dict(dealt_order_amount)])
            if self.calls == fail:
                raise RuntimeError("deal failed")
            return self.calls * 10.0, 1.0, 10.0

    instance = SimulatorExecutor.__new__(SimulatorExecutor)
    instance.trade_calendar = Calendar()
    instance.trade_exchange = Exchange()
    instance.trade_account = object()
    instance.trade_type = mode
    instance.verbose = False
    instance.deal_day = None
    instance.dealt_order_amount = defaultdict(float)
    try:
        rows, info = instance._collect_data(SimpleNamespace(get_decision=lambda: original))
        result = [[o.stock_id, o.deal_amount, value, cost, price] for o, value, cost, price in rows]
        same_list = rows is info["trade_info"]
        error = None
    except RuntimeError as failure:
        result, same_list, error = None, None, str(failure)
    return {"rows": result, "same_list": same_list, "error": error,
            "buy_amount": buy.deal_amount, "sell_amount": sell.deal_amount,
            "original_length": len(original), "fills": dict(instance.dealt_order_amount), "events": events}


print(json.dumps({f"{mode}-{fail}": run(mode, fail) for mode in ("serial", "parallel") for fail in (0, 2)}))
