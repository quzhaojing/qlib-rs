import ast
import json
import math
import sys
import warnings
from types import SimpleNamespace
from typing import TypedDict, cast

import numpy as np
import pandas as pd

strategy_path, utils_path = sys.argv[1:]


def functions_from(path, class_name, names):
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    selected = [next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == name) for name in names]
    for function in selected:
        function.returns = None
        for argument in function.args.args:
            argument.annotation = None
    return selected


strategy_tree = ast.parse(open(strategy_path, encoding="utf-8").read(), filename=strategy_path)
top_level = [
    next(node for node in strategy_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    for name in ["_get_all_timestamps", "fill_missing_data"]
]
methods = functions_from(
    strategy_path,
    "SAOEStateAdapter",
    [
        "__init__",
        "_next_time",
        "update",
        "generate_metrics_after_done",
        "_collect_multi_order_metric",
        "_collect_single_order_metric",
    ],
)
utils_tree = ast.parse(open(utils_path, encoding="utf-8").read(), filename=utils_path)
utility_functions = [
    next(node for node in utils_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    for name in ["dataframe_append", "price_advantage"]
]
for function in top_level + utility_functions:
    function.returns = None
    for argument in function.args.args:
        argument.annotation = None


def get_start_end_idx(calendar, decision):
    return 0, 5


def get_day_min_idx_range(start, end, frequency, region):
    base = pd.Timestamp("2024-01-02 09:30:00")
    return int((pd.Timestamp(start) - base).total_seconds() // 60), int((pd.Timestamp(end) - base).total_seconds() // 60)


namespace = {
    "np": np,
    "pd": pd,
    "warnings": warnings,
    "cast": cast,
    "Callable": object,
    "Tuple": tuple,
    "Optional": object,
    "Order": object,
    "OrderDir": SimpleNamespace(BUY=1, SELL=0),
    "BaseTradeDecision": object,
    "BaseExecutor": object,
    "Exchange": object,
    "IntradayBacktestData": object,
    "IndexData": object,
    "float_or_ndarray": object,
    "SAOEMetrics": TypedDict(
        "SAOEMetrics",
        {
            key: object
            for key in [
                "stock_id",
                "datetime",
                "direction",
                "market_volume",
                "market_price",
                "amount",
                "inner_amount",
                "deal_amount",
                "trade_price",
                "trade_value",
                "position",
                "ffr",
                "pa",
            ]
        },
    ),
    "SAOEState": tuple,
    "EPS": 1.0e-8,
    "ONE_MIN": pd.Timedelta(minutes=1),
    "REG_CN": "cn",
    "get_start_end_idx": get_start_end_idx,
    "get_day_min_idx_range": get_day_min_idx_range,
}
exec(
    compile(ast.fix_missing_locations(ast.Module(body=top_level + utility_functions + methods, type_ignores=[])), strategy_path, "exec"),
    namespace,
)


class Order:
    def __init__(self, amount=10.0, start=None, end=None, direction=1, deal_amount=0.0):
        self.stock_id = "A"
        self.amount = amount
        self.start_time = pd.Timestamp(start or "2024-01-02 09:30:00")
        self.end_time = pd.Timestamp(end or "2024-01-02 09:35:00")
        self.direction = direction
        self.deal_amount = deal_amount


ticks = pd.date_range("2024-01-02 09:30:00", periods=6, freq="1min")


class Backtest:
    ticks_index = ticks
    ticks_for_order = ticks

    def get_deal_price(self):
        return pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], index=ticks)


class Exchange:
    volume = np.array([100.0, np.nan, 120.0, 140.0, 160.0, 180.0])
    price = np.array([10.0, np.nan, 12.0, 14.0, 16.0, 18.0])

    def _slice(self, values, start, end):
        left = ticks.get_loc(start)
        right = ticks.get_loc(end)
        return values[left : right + 1]

    def get_volume(self, stock, start, end, method=None):
        return self._slice(self.volume, start, end)

    def get_deal_price(self, stock, start, end, method=None, direction=None):
        return self._slice(self.price, start, end)


class Indicator:
    def generate_trade_indicators_dataframe(self):
        return pd.DataFrame([{"pa": 7.5}])


class Account:
    def get_trade_indicator(self):
        return Indicator()


class Calendar:
    def get_trade_step(self):
        return 3


class Executor:
    trade_calendar = Calendar()
    trade_account = Account()


class Adapter:
    pass


for method in methods:
    setattr(Adapter, method.name, namespace[method.name])

adapter = Adapter(Order(), object(), Executor(), Exchange(), 2, Backtest(), 1)
snapshots = []


def record(label):
    snapshots.append(
        {
            "label": label,
            "position": adapter.position,
            "cur_time": adapter.cur_time,
            "history_exec": adapter.history_exec.reset_index().to_dict("records"),
            "history_steps": adapter.history_steps.reset_index().to_dict("records"),
            "metrics": adapter.metrics,
        }
    )


adapter.update(
    [
        (Order(start=ticks[0], end=ticks[0], deal_amount=2.0), 0, 0, 0),
        (Order(start=ticks[1], end=ticks[1], deal_amount=3.0), 0, 0, 0),
    ],
    (0, 1),
)
record("normal")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    adapter.update(
        [
            (Order(start=ticks[2], end=ticks[2], deal_amount=4.0), 0, 0, 0),
            (Order(start=ticks[3], end=ticks[3], deal_amount=4.0), 0, 0, 0),
        ],
        (2, 3),
    )
    warning_count = len(caught)
record("scaled")
adapter.update([], (4, 4))
record("empty")
adapter.generate_metrics_after_done()
record("done")

price_advantage = namespace["price_advantage"]
edge_values = {
    "sell": price_advantage(15.0, 12.5, 0),
    "zero_buy": price_advantage(15.0, 0.0, 1),
    "array": price_advantage(np.array([10.0, np.nan, 15.0]), 12.5, 1),
    "timestamps_exclude_end": namespace["_get_all_timestamps"](ticks[0], ticks[2], include_end=False),
    "filled_all_nan": namespace["fill_missing_data"](np.array([np.nan, np.nan])),
}


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray, pd.Series, pd.Index)):
        return [clean(item) for item in list(value)]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return "nan"
        if math.isinf(float(value)):
            return "inf" if float(value) > 0 else "-inf"
        return float(value)
    return value


print(
    json.dumps(
        clean(
            {
                "snapshots": snapshots,
                "warning_count": warning_count,
                "cur_step": Executor.trade_calendar.get_trade_step() - adapter.start_idx,
                "twap_price": adapter.twap_price,
                "edges": edge_values,
            }
        ),
        separators=(",", ":"),
    )
)
