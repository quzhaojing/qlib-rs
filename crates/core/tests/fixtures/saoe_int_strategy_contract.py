import ast
import collections
import json
import sys
from contextlib import nullcontext
from types import GeneratorType, SimpleNamespace

import numpy as np
import pandas as pd


class RLStrategy:
    pass


class Batch:
    def __init__(self, values):
        self.values = values


class TradeDecisionWithDetails:
    def __init__(self, order_list, strategy, trade_range=None, details=None):
        self.order_list = order_list
        self.details = details
        strategy.trade_calendar.get_step_time()


tree = ast.parse(open(sys.argv[1], encoding="utf-8").read(), sys.argv[1])
body = [
    ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
    next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SAOEStrategy"),
    next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SAOEIntStrategy"),
]
namespace = {
    "RLStrategy": RLStrategy,
    "Batch": Batch,
    "TradeDecisionWithDetails": TradeDecisionWithDetails,
    "pd": pd,
    "np": np,
    "torch": SimpleNamespace(no_grad=nullcontext, is_tensor=lambda value: False),
    "cast": lambda _type, value: value,
    "Order": object,
    "collections": collections,
    "GeneratorType": GeneratorType,
}
exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), sys.argv[1], "exec"), namespace)

events = []
orders = [
    SimpleNamespace(stock_id="A", direction=1),
    SimpleNamespace(stock_id="B", direction=0),
]


class Calendar:
    def get_step_time(self):
        events.append("time")
        return pd.Timestamp("2024-01-02 09:30:00"), pd.Timestamp("2024-01-02 09:31:00")

    def get_freq(self):
        events.append("freq")
        return "1min"


class OrderHelper:
    def create(self, stock_id, amount, direction):
        events.append(f"order:{stock_id}:{amount:g}:{direction}")
        return SimpleNamespace(stock_id=stock_id, amount=amount, direction=direction)


strategy = namespace["SAOEIntStrategy"].__new__(namespace["SAOEIntStrategy"])
strategy.outer_trade_decision = SimpleNamespace(order_list=orders, get_decision=lambda: orders)
strategy.get_saoe_state_by_order = lambda order: SimpleNamespace(order=order)
strategy._state_interpreter = SimpleNamespace(interpret=lambda state: state.order.stock_id)
strategy._policy = lambda batch: SimpleNamespace(act=np.array([0.0, 2.0]))
strategy._action_interpreter = SimpleNamespace(interpret=lambda state, action: float(action))
strategy.trade_calendar = Calendar()
strategy.trade_exchange = SimpleNamespace(get_order_helper=lambda: OrderHelper())

decision = strategy._generate_trade_decision()


class Adapter:
    def __init__(self, name):
        self.name = name

    def update(self, executions, step_range):
        events.append(f"update:{self.name}:{len(executions)}:{step_range[0]}-{step_range[1]}")

    def generate_metrics_after_done(self):
        events.append(f"finalize:{self.name}")


lifecycle = namespace["SAOEStrategy"].__new__(namespace["SAOEStrategy"])
lifecycle.adapter_dict = {"A": Adapter("A"), "B": Adapter("B")}
lifecycle._last_step_range = (1, 3)
lifecycle.post_exe_step(
    [
        (SimpleNamespace(key_by_day="B"), "fill-b"),
        (SimpleNamespace(key_by_day="X"), "ignored"),
        (SimpleNamespace(key_by_day="A"), "fill-a"),
    ]
)
lifecycle.post_upper_level_exe_step()
lifecycle._last_step_range = (0, 0)
lifecycle.post_exe_step([])
lifecycle.get_data_cal_avail_range = lambda rtype: (2, 4)
lifecycle._generate_trade_decision = lambda execute_result: "inner-decision"
generator = lifecycle.generate_trade_decision(None)
try:
    next(generator)
except StopIteration as stop:
    generated = stop.value

print(
    json.dumps(
        {
            "orders": [[order.stock_id, order.amount, order.direction] for order in decision.order_list],
            "details": decision.details.to_dict(orient="records"),
            "events": events,
            "lifecycle_generated": generated,
            "lifecycle_range": lifecycle._last_step_range,
        },
        default=str,
        sort_keys=True,
    )
)
