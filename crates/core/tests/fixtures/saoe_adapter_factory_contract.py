import ast
import json
import sys
from types import SimpleNamespace

import pandas as pd


tree = ast.parse(open(sys.argv[1], encoding="utf-8").read(), sys.argv[1])
strategy = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SAOEStrategy")
method = next(
    node for node in strategy.body if isinstance(node, ast.FunctionDef) and node.name == "_create_qlib_backtest_adapter"
)
events = []


def load_backtest_data(order, exchange, trade_range):
    events.append(["load", order.stock_id, exchange.name, trade_range.name])
    return "backtest"


class Adapter:
    def __init__(self, **kwargs):
        events.append(["start", kwargs["trade_decision"].start_idx])
        events.append(
            [
                "adapter",
                kwargs["order"].stock_id,
                kwargs["trade_decision"].name,
                kwargs["executor"].name,
                kwargs["exchange"].name,
                kwargs["ticks_per_step"],
                kwargs["backtest_data"],
                kwargs["data_granularity"],
            ]
        )


namespace = {
    "load_backtest_data": load_backtest_data,
    "SAOEStateAdapter": Adapter,
    "pd": pd,
    "ONE_MIN": pd.Timedelta("1min"),
}
exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), sys.argv[1], "exec"), namespace)


class Calendar:
    def get_freq(self):
        events.append(["frequency", "30min"])
        return "30min"


instance = SimpleNamespace(
    trade_exchange=SimpleNamespace(name="exchange"),
    executor=SimpleNamespace(name="executor"),
    trade_calendar=Calendar(),
    _data_granularity=5,
)
order = SimpleNamespace(stock_id="A")
decision = SimpleNamespace(name="decision", start_idx=4)
trade_range = SimpleNamespace(name="range")
result = namespace["_create_qlib_backtest_adapter"](instance, order, decision, trade_range)
print(json.dumps({"events": events, "type": type(result).__name__}, sort_keys=True))
