"""Execute unchanged, hash-pinned SAOE lifecycle and BaseStrategy reset methods.

Collaborators expose callback ordering and aliases, not adapter numerical parity.
No production Python bridge or dependency on a Torch runtime is introduced.
"""
import ast
import collections
import hashlib
import json
from pathlib import Path
import sys


def methods(path, digest, class_name, names):
    source = Path(path).read_bytes()
    assert hashlib.sha256(source).hexdigest() == digest
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    result = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(result) == len(names)
    return result


base = methods(sys.argv[2], "8d8045892d6b52fb0740025a2fe446ee43310ac1c0cc4384cd146beb7cfd3f08",
               "BaseStrategy", {"reset", "_reset"})
saoe = methods(sys.argv[1], "dc4a4e8cb0577c197547ff2c2e3195a3588861b9176d166960c68f1d5d13b49f",
               "SAOEStrategy", {"reset", "get_saoe_state_by_order", "post_exe_step",
                                "post_upper_level_exe_step"})
body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        ast.ClassDef(name="Base", bases=[], keywords=[], body=base, decorator_list=[]),
        ast.ClassDef(name="SAOEStrategy", bases=[ast.Name(id="Base", ctx=ast.Load())],
                     keywords=[], body=saoe, decorator_list=[])]
namespace = {"collections": collections, "cast": lambda _, value: value, "Order": object}
exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
             "upstream-saoe-lifecycle", "exec"), namespace)
Strategy = namespace["SAOEStrategy"]


def reset_case(mode):
    events = []

    class Order:
        def __init__(self, name):
            self.name = name

        @property
        def key_by_day(self):
            events.append("key:" + self.name)
            if mode == "key-failure" and self.name == "B":
                raise RuntimeError("key failed")
            return self.name

    class Outer:
        def __init__(self):
            self.orders = [Order("A"), Order("B")]
            self.rule = object()

        def empty(self):
            events.append("empty")
            return mode == "empty"

        @property
        def trade_range(self):
            events.append("range")
            return None if mode == "no-range" else self.rule

        def get_decision(self):
            events.append("orders")
            return self.orders

    outer = Outer()
    original_rule = outer.rule
    original_orders = outer.orders
    if mode == "duplicate":
        original_orders.append(Order("A"))
    strategy = Strategy()
    previous_outer = object()
    strategy.outer_trade_decision = previous_outer
    strategy.adapter_dict = {"old": "old"}
    previous_registry = strategy.adapter_dict
    strategy._last_step_range = (7, 9)
    created = []

    def level(_):
        events.append("level")
        if mode == "base-failure":
            raise RuntimeError("base failed")

    def create(order, decision, rule):
        assert decision is outer and rule is original_rule
        assert any(order is item for item in original_orders)
        label = f"{order.name}:{len(created)}"
        events.append("create:" + label)
        if mode == "factory-failure" and order.name == "B":
            raise RuntimeError("factory failed")
        created.append(label)
        if len(created) == 1:
            if mode == "late-key":
                order.name = "X"
            elif mode == "append":
                original_orders.append(Order("C"))
            elif mode == "delete":
                del original_orders[1]
            elif mode == "replace":
                outer.orders = [Order("D")]
            outer.rule = object()  # The one captured range is retained across factories.
        return label

    strategy.reset_level_infra = level
    strategy._create_qlib_backtest_adapter = create
    error = None
    try:
        strategy.reset(None if mode == "none" else outer, level_infra=object())
    except (RuntimeError, AssertionError) as exc:
        error = type(exc).__name__ + ":" + str(exc)
    return {"events": events, "entries": list(strategy.adapter_dict.items()),
            "last": strategy._last_step_range, "error": error,
            "outer_retained": strategy.outer_trade_decision is previous_outer,
            "registry_retained": strategy.adapter_dict is previous_registry}


def post_case(mode):
    events = []
    strategy = Strategy()
    strategy._last_step_range = (4, 4) if mode.startswith("zero") else (2, 5)

    class Order:
        def __init__(self, name):
            self.name = name

        @property
        def key_by_day(self):
            events.append("key:" + self.name)
            return self.name

    rows = [(Order("B"), 11), (Order("A"), 12), (Order("B"), 13), (Order("X"), 14)]

    class Adapter:
        def __init__(self, name):
            self.name = name

        def update(self, executions, step_range):
            assert step_range is strategy._last_step_range
            assert all(any(row is original for original in rows) for row in executions)
            events.append(["update", self.name, [[row[0].name, row[1]] for row in executions]])
            if mode == "mutate" and self.name == "A":
                rows[0][0].name = "Z"
            if mode == "update-failure" and self.name == "B":
                raise RuntimeError("update failed")

        def generate_metrics_after_done(self):
            events.append("final:" + self.name)
            if mode == "final-failure" and self.name == "B":
                raise RuntimeError("final failed")

    strategy.adapter_dict = {name: Adapter(name) for name in ["A", "B", "C"]}
    executions = None if mode == "none" else [] if mode == "zero-empty" else rows
    error = None
    try:
        strategy.post_exe_step(executions)
        strategy.post_upper_level_exe_step()
    except (RuntimeError, AssertionError) as exc:
        error = type(exc).__name__ + ":" + str(exc)
    return {"events": events, "error": error}


print(json.dumps({
    "reset": {mode: reset_case(mode) for mode in [
        "normal", "late-key", "append", "delete", "replace", "duplicate",
        "factory-failure", "key-failure", "no-range", "empty", "none", "base-failure"]},
    "post": {mode: post_case(mode) for mode in [
        "normal", "none", "zero-empty", "zero-rows", "mutate", "update-failure", "final-failure"]}
}))
