import ast
import json
import sys

import cachetools
import pandas as pd


class BaseIntradayBacktestData:
    pass


class TradeRangeByTime:
    def __init__(self, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time


tree = ast.parse(open(sys.argv[1], encoding="utf-8").read(), sys.argv[1])
nodes = [
    ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
    next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_ticks_slice"),
    next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "IntradayBacktestData"),
    next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "load_backtest_data"),
]
namespace = {
    "pd": pd,
    "cachetools": cachetools,
    "EPS_T": pd.Timedelta("1ns"),
    "cast": lambda _type, value: value,
    "BaseIntradayBacktestData": BaseIntradayBacktestData,
    "TradeRangeByTime": TradeRangeByTime,
}
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), sys.argv[1], "exec"), namespace)


class Order:
    def __init__(self, stock_id, direction, start_time, end_time):
        self.stock_id = stock_id
        self.direction = direction
        self.start_time = pd.Timestamp(start_time)
        self.end_time = pd.Timestamp(end_time)

    @property
    def key_by_day(self):
        return self.stock_id, self.start_time.replace(hour=0, minute=0, second=0), self.direction


class Exchange:
    def __init__(self, name, timestamps):
        self.name = name
        self.events = []
        index = pd.MultiIndex.from_arrays(
            [["A"] * len(timestamps), pd.DatetimeIndex(timestamps)], names=["instrument", "datetime"]
        )
        self.quote_df = pd.DataFrame({"value": range(len(timestamps))}, index=index)

    def get_deal_price(self, stock_id, start, end, direction, method):
        self.events.append(["deal", stock_id, str(start), str(end), direction, method])
        return pd.Series([10.0, 11.0])

    def get_volume(self, stock_id, start, end, method):
        self.events.append(["volume", stock_id, str(start), str(end), method])
        return pd.Series([100.0, 110.0])


timestamps = [
    "2024-01-02 09:31:00",
    "2024-01-02 09:29:00",
    "2024-01-02 09:30:00",
    "2024-01-02 09:30:00",
    "2024-01-02 09:32:00",
]
first_exchange = Exchange("first", timestamps)
second_exchange = Exchange("second", ["2024-01-02 09:30:00"])
first_order = Order("A", 1, "2024-01-02 09:29:00", "2024-01-02 09:32:00")
same_key_order = Order("A", 1, "2024-01-02 09:30:00", "2024-01-02 09:30:00")
first = namespace["load_backtest_data"](
    first_order, first_exchange, TradeRangeByTime(pd.Timestamp("09:30").time(), pd.Timestamp("09:31").time())
)
cached = namespace["load_backtest_data"](
    same_key_order, second_exchange, TradeRangeByTime(pd.Timestamp("09:30").time(), pd.Timestamp("09:32").time())
)
first._deal_price.iloc[0] = 77.0


def failure(range_value, start="2024-01-03 09:30", end="2024-01-03 09:31"):
    try:
        namespace["load_backtest_data"](Order("Z", 0, start, end), first_exchange, range_value)
    except Exception as error:
        return type(error).__name__
    return "ok"


print(
    json.dumps(
        {
            "ticks_index": [str(value) for value in first.ticks_index],
            "ticks_for_order": [str(value) for value in first.ticks_for_order],
            "events": first_exchange.events,
            "cache_identity": first is cached,
            "cached_mutation": float(cached.get_deal_price().iloc[0]),
            "cached_order_is_first": cached._order is first_order,
            "cached_exchange_is_first": cached._exchange is first_exchange,
            "second_events": second_exchange.events,
            "unsupported": failure(object()),
            "empty": failure(TradeRangeByTime(pd.Timestamp("10:00").time(), pd.Timestamp("10:01").time()), "2024-01-02 09:29", "2024-01-02 09:32"),
        },
        sort_keys=True,
    )
)
