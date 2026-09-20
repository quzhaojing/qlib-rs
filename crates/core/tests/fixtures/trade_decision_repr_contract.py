import ast
import hashlib
import json
from pathlib import Path

source = Path(r"D:\code\github\qlib\qlib\backtest\decision.py")
source_bytes = source.read_bytes()
assert hashlib.sha256(source_bytes).hexdigest() == "a6866d15bc8f3ad1c75bfc3856ccde5245d43f0a2de07b1ad565d6e7e20d8251"
tree = ast.parse(source_bytes.decode("utf-8"), filename=str(source))
target_class = next(
    node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TradeDecisionWO"
)
target = next(
    node for node in target_class.body if isinstance(node, ast.FunctionDef) and node.name == "__repr__"
)
target.decorator_list = []
target.returns = None
for argument in [*target.args.posonlyargs, *target.args.args, *target.args.kwonlyargs]:
    argument.annotation = None

namespace = {}
exec(compile(ast.Module(body=[target], type_ignores=[]), str(source), "exec"), namespace)
source_repr = namespace["__repr__"]


class Rendered:
    def __init__(self, label, text, events, fail=False):
        self.label = label
        self.text = text
        self.events = events
        self.fail = fail

    def __format__(self, spec):
        self.events.append(f"format_{self.label}:{spec}")
        if self.fail:
            raise RuntimeError(f"{self.label}-failed")
        return self.text


class Orders:
    def __init__(self, count, events, fail=False):
        self.count = count
        self.events = events
        self.fail = fail

    def __len__(self):
        self.events.append("len_orders")
        if self.fail:
            raise RuntimeError("orders-failed")
        return self.count


class Base:
    def __init__(self, strategy, trade_range, orders, events):
        self._strategy = strategy
        self._trade_range = trade_range
        self._orders = orders
        self.events = events

    @property
    def strategy(self):
        self.events.append("get_strategy")
        return self._strategy

    @property
    def trade_range(self):
        self.events.append("get_range")
        return self._trade_range

    @property
    def order_list(self):
        self.events.append("get_orders")
        return self._orders


TradeDecisionWO = type("TradeDecisionWO", (Base,), {"__repr__": source_repr})
TradeDecisionWithDetails = type("TradeDecisionWithDetails", (TradeDecisionWO,), {})


def run(kind, strategy_text="strategy", range_text=None, count=0, failure=None):
    events = []
    strategy = Rendered("strategy", strategy_text, events, failure == "strategy")
    trade_range = (
        None
        if range_text is None
        else Rendered("range", range_text, events, failure == "range")
    )
    orders = Orders(count, events, failure == "orders")
    cls = TradeDecisionWO if kind == "TradeDecisionWO" else TradeDecisionWithDetails
    instance = cls(strategy, trade_range, orders, events)
    try:
        value = repr(instance)
        error = None
    except Exception as exc:
        value = None
        error = f"{type(exc).__name__}:{exc}"
    return {"value": value, "error": error, "events": events}


print(
    json.dumps(
        [
            run("TradeDecisionWO", strategy_text="策略\nalpha", count=0),
            run("TradeDecisionWithDetails", range_text="(2, 5)", count=3),
            run("TradeDecisionWO", range_text="unused", count=1, failure="strategy"),
            run("TradeDecisionWO", range_text="custom-range", count=2, failure="range"),
            run("TradeDecisionWO", range_text="custom-range", count=4, failure="orders"),
        ],
        separators=(",", ":"),
        ensure_ascii=True,
    )
)
