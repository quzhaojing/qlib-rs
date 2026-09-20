"""Exercise unchanged SAOE methods with mutations at real callback boundaries.

The constructor is an observation boundary, not a substitute for its separate
source-backed initialization tests. No Torch inference is required here.
"""
import ast
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd

source = Path(sys.argv[1]).read_bytes()
assert hashlib.sha256(source).hexdigest() == "dc4a4e8cb0577c197547ff2c2e3195a3588861b9176d166960c68f1d5d13b49f"
tree = ast.parse(source)
original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SAOEIntStrategy")
methods = [n for n in original.body if isinstance(n, ast.FunctionDef)
           and n.name in ("_generate_trade_details", "_generate_trade_decision")]
assert len(methods) == 2
body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        ast.ClassDef(name="Strategy", bases=[], keywords=[], body=methods, decorator_list=[])]


def run(mode):
    events = []
    def order(name):
        return SimpleNamespace(stock_id=name, direction=1)

    class Outer:
        def __init__(self):
            self.order_list = [order("A"), order("B")]

        def get_decision(self):
            events.append("read")
            return self.order_list

    outer = Outer()
    if mode == "delete":
        outer.order_list.append(order("C"))

    class Calendar:
        def get_step_time(self):
            events.append("time")
            if mode == "details" and events.count("time") == 1:
                outer.order_list[1] = order("E")
            return pd.Timestamp("2024-01-02 09:30"), pd.Timestamp("2024-01-02 09:31")

        def get_freq(self):
            events.append("freq")
            return "1min"

    class Helper:
        def create(self, name, volume, direction):
            events.append(f"create:{name}")
            if mode == "factory" and name == "A":
                outer.order_list = [order("D"), order("E")]
            return [name, volume, direction]

    def state(item):
        events.append(f"state:{item.stock_id}")
        if mode == "append" and item.stock_id == "A":
            outer.order_list.append(order("C"))
        return item

    def observe(item):
        events.append(f"observe:{item.stock_id}")
        if mode == "delete" and item.stock_id == "A":
            del outer.order_list[1]
        if mode == "failure" and item.stock_id == "B":
            raise RuntimeError("observation failed")
        return item.stock_id

    def policy(batch):
        names = [row["obs"] for row in batch]
        events.append("policy:" + ",".join(names))
        if mode == "policy":
            outer.order_list = [order("X"), order("Y")]
        count = len(names) + (1 if mode == "long" else -1 if mode == "short" else 0)
        return SimpleNamespace(act=list(range(1, count + 1)))

    def action(item, value):
        events.append(f"action:{item.stock_id}:{value}")
        return value

    def construct(order_list, strategy, details):
        events.append("construct")
        return SimpleNamespace(order_list=order_list, strategy=strategy, details=details)

    namespace = {"pd": pd, "Batch": lambda rows: rows,
                 "torch": SimpleNamespace(no_grad=nullcontext, is_tensor=lambda _: False),
                 "cast": lambda _, value: value, "Order": object,
                 "TradeDecisionWithDetails": construct}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                 "upstream-live-saoe", "exec"), namespace)
    strategy = namespace["Strategy"]()
    strategy.outer_trade_decision = outer
    strategy.get_saoe_state_by_order = state
    strategy._state_interpreter = SimpleNamespace(interpret=observe)
    strategy._policy = policy
    strategy._action_interpreter = SimpleNamespace(interpret=action)
    strategy.trade_calendar = Calendar()
    strategy.trade_exchange = SimpleNamespace(get_order_helper=lambda: Helper())
    try:
        result = strategy._generate_trade_decision()
        assert result.strategy is strategy
        return {"events": events, "children": result.order_list,
                "details": result.details["instrument"].tolist(), "error": None}
    except RuntimeError as error:
        return {"events": events, "children": None, "details": None, "error": str(error)}


print(json.dumps({mode: run(mode) for mode in
                  ["normal", "append", "delete", "policy", "factory", "details", "short", "long", "failure"]}))
