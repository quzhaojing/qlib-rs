"""Hash-pinned numerical adapter methods with mutations at real callback boundaries."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import TypedDict, cast
import warnings

import numpy as np
import pandas as pd


def tree(path, digest):
    source = Path(path).read_bytes()
    assert hashlib.sha256(source).hexdigest() == digest
    return ast.parse(source)


strategy = tree(sys.argv[1], "dc4a4e8cb0577c197547ff2c2e3195a3588861b9176d166960c68f1d5d13b49f")
utils = tree(sys.argv[2], "89267f5cfc9e38751cb2c3a37c74ca712e8c395f492d53ed02f029a66272074f")
adapter_class = next(n for n in strategy.body if isinstance(n, ast.ClassDef) and n.name == "SAOEStateAdapter")
functions = [n for n in strategy.body if isinstance(n, ast.FunctionDef) and n.name in {"_get_all_timestamps", "fill_missing_data"}]
functions += [n for n in utils.body if isinstance(n, ast.FunctionDef) and n.name in {"dataframe_append", "price_advantage"}]
assert len(functions) == 4
body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + functions + [adapter_class]
compiled = compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), "upstream-live-adapter", "exec")
metric_keys = ["stock_id", "datetime", "direction", "market_volume", "market_price", "amount", "inner_amount",
               "deal_amount", "trade_price", "trade_value", "position", "ffr", "pa"]
ticks = pd.date_range("2024-01-02 09:30:00", periods=6, freq="1min")


def run(mode):
    events = []
    order = SimpleNamespace(stock_id="A", amount=10.0, start_time=ticks[0], end_time=ticks[5], direction=1)
    decision = object()

    def mutate(name, amount):
        order.stock_id, order.amount, order.direction, order.end_time = name, amount, 0, ticks[1]

    def start_end(calendar, actual_decision):
        assert actual_decision is decision
        events.append("start")
        if mode == "init":
            mutate("I", 20.0)
            order.start_time = ticks[1]
        return 1, 5

    class Backtest:
        def __init__(self):
            self.ticks_index = ticks
            self.ticks_for_order = ticks
            self._deal_price = np.array([10.0, 12.0])

        def get_deal_price(self):
            events.append("baseline")
            return self._deal_price

    class Market:
        def get_volume(self, stock, start, end, method=None):
            ranges = {
                "aliases": [(ticks[0], ticks[1]), (ticks[2], ticks[3])],
                "backtest-aliases": [(ticks[0], ticks[1]), (ticks[2], ticks[3])],
            }.get(mode, [(ticks[0], ticks[1])])
            assert (start, end) in ranges
            assert method is None
            events.append("volume:" + stock)
            if mode == "volume":
                mutate("V", 20.0)
            if mode == "volume-failure":
                raise RuntimeError("volume failed")
            return np.array([100.0, 200.0])

        def get_deal_price(self, stock, start, end, method=None, direction=None):
            ranges = {
                "aliases": [(ticks[0], ticks[1]), (ticks[2], ticks[3])],
                "backtest-aliases": [(ticks[0], ticks[1]), (ticks[2], ticks[3])],
            }.get(mode, [(ticks[0], ticks[1])])
            assert (start, end) in ranges
            assert method is None
            events.append(f"price:{stock}:{direction}")
            if mode == "price":
                mutate("P", 40.0)
            if mode == "price-failure":
                raise RuntimeError("price failed")
            return np.array([10.0, 12.0])

    class Indicator:
        def generate_trade_indicators_dataframe(self):
            events.append("indicator")
            if mode == "indicator":
                mutate("Q", 50.0)
            if mode == "next-failure":
                order.end_time = None
            if mode == "indicator-failure":
                raise RuntimeError("indicator failed")
            return pd.DataFrame([{"pa": 7.5}])

    class Calendar:
        def get_trade_step(self):
            events.append("step")
            if mode == "state":
                mutate("S", 80.0)
            return 3

    namespace = {"np": np, "pd": pd, "warnings": warnings, "cast": cast, "IndexData": object,
                 "OrderDir": SimpleNamespace(BUY=1, SELL=0), "EPS": 1e-8,
                 "ONE_MIN": pd.Timedelta(minutes=1), "REG_CN": "cn",
                 "SAOEMetrics": TypedDict("SAOEMetrics", {key: object for key in metric_keys}),
                 "SAOEState": SimpleNamespace, "get_start_end_idx": start_end,
                 "get_day_min_idx_range": lambda start, end, freq, region: (ticks.get_loc(start), ticks.get_loc(end))}
    exec(compiled, namespace)
    executor = SimpleNamespace(trade_calendar=Calendar(), trade_account=SimpleNamespace(get_trade_indicator=Indicator))
    adapter = namespace["SAOEStateAdapter"](order, decision, executor, Market(), 2, Backtest(), 1)
    assert adapter.order is order
    initial = {"position": adapter.position, "time": str(adapter.cur_time), "amount": adapter.order.amount}
    rows = [(SimpleNamespace(start_time=ticks[i], end_time=ticks[i], deal_amount=amount), 0, 0, 0)
            for i, amount in enumerate([2.0, 3.0])]
    error = None
    try:
        adapter.update(rows, (0, 1))
    except (RuntimeError, TypeError) as exc:
        error = type(exc).__name__
    state = adapter.saoe_state
    assert state.order is order and state.history_exec is adapter.history_exec and state.history_steps is adapter.history_steps
    if mode == "aliases":
        state.order.stock_id = "Z"
        state.history_exec.iloc[0, state.history_exec.columns.get_loc("stock_id")] = "M"
        state.history_steps.iloc[0, state.history_steps.columns.get_loc("amount")] = 42.0
        old_exec, old_steps = state.history_exec, state.history_steps
        later = [(SimpleNamespace(start_time=ticks[i], end_time=ticks[i], deal_amount=amount), 0, 0, 0)
                 for i, amount in enumerate([1.0, 1.0], start=2)]
        adapter.update(later, (2, 3))
        current = adapter.saoe_state
        return {
            "order_alias": state.order is adapter.order and current.order is state.order,
            "initial_history_alias": old_exec is not adapter.history_exec and old_steps is not adapter.history_steps,
            "old_lengths": [len(old_exec), len(old_steps)],
            "new_lengths": [len(adapter.history_exec), len(adapter.history_steps)],
            "mutated_exec_stock": adapter.history_exec.iloc[0]["stock_id"],
            "mutated_step_amount": adapter.history_steps.iloc[0]["amount"],
            "new_exec_stock": adapter.history_exec.iloc[-1]["stock_id"],
            "state_stock": state.order.stock_id,
        }
    if mode == "backtest-aliases":
        old_ticks, old_order_ticks = state.ticks_index, state.ticks_for_order
        state.backtest_data._deal_price[0] = 99.0
        state.backtest_data.ticks_index = pd.DatetimeIndex([ticks[1], ticks[2], ticks[3], ticks[4], ticks[5]])
        state.backtest_data.ticks_for_order = pd.DatetimeIndex([ticks[1], ticks[2]])
        later = [(SimpleNamespace(start_time=ticks[i], end_time=ticks[i], deal_amount=amount), 0, 0, 0)
                 for i, amount in zip([1, 2], [1.0, 1.0])]
        adapter.update(later, (1, 2))
        adapter.generate_metrics_after_done()
        current = adapter.saoe_state
        return {
            "same_backtest_object": state.backtest_data is adapter.backtest_data is current.backtest_data,
            "old_ticks_retained": state.ticks_index is old_ticks and state.ticks_index[0] == ticks[0],
            "old_order_ticks_retained": state.ticks_for_order is old_order_ticks and len(state.ticks_for_order) == 6,
            "current_ticks_rebound": current.ticks_index is state.backtest_data.ticks_index and current.ticks_index[0] == ticks[1],
            "current_order_ticks_rebound": current.ticks_for_order is state.backtest_data.ticks_for_order and len(current.ticks_for_order) == 2,
            "deal_mutation_visible": current.backtest_data.get_deal_price()[0],
            "adapter_time": str(adapter.cur_time),
            "metric_time": str(adapter.metrics["datetime"]),
        }
    adapter.generate_metrics_after_done()
    if mode == "metric-aliases":
        first = adapter.saoe_state
        assert first.metrics is adapter.metrics
        first.metrics["stock_id"] = "M"
        second = adapter.saoe_state
        old_metrics = first.metrics
        adapter.generate_metrics_after_done()
        current = adapter.saoe_state
        return {
            "before_finalize_none": state.metrics is None,
            "same_before_replacement": first.metrics is second.metrics and second.metrics is old_metrics,
            "mutation_visible": second.metrics["stock_id"],
            "replaced_after_finalize": current.metrics is not old_metrics and current.metrics is adapter.metrics,
            "old_stock": old_metrics["stock_id"],
            "new_stock": current.metrics["stock_id"],
        }
    metric = adapter.metrics
    return {"events": events, "initial": initial, "error": error, "position": adapter.position,
            "time": str(adapter.cur_time), "state_stock": state.order.stock_id, "state_amount": state.order.amount,
            "state_step": state.cur_step, "state_metrics_none": state.metrics is None,
            "exec_stock": adapter.history_exec["stock_id"].tolist(),
            "exec_ffr": adapter.history_exec["ffr"].tolist(), "step_count": len(adapter.history_steps),
            "metric_stock": metric["stock_id"], "metric_ffr": metric["ffr"], "metric_position": metric["position"]}


print(json.dumps({mode: run(mode) for mode in ["normal", "init", "volume", "price", "indicator", "state",
    "volume-failure", "price-failure", "indicator-failure", "next-failure", "aliases", "metric-aliases",
    "backtest-aliases"]}))
