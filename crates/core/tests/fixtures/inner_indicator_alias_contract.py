"""Execute unchanged Qlib stores/Indicator: raw aliases and aggregation failure order.

Only import wiring and unused logger construction are substituted. Actual numerical
engines, transfers, get_order_indicator, record, reset and aggregation execute here.
"""
import ast
from collections import OrderedDict
from dataclasses import dataclass
from enum import IntEnum
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from typing import *

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("alias_index_data", root / "utils/index_data.py")
idd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(idd)
SingleData = idd.SingleData

def get_module_logger(_):
    return object()

def load(relative, names):
    path = root / relative
    nodes = [node for node in ast.parse(path.read_text(encoding="utf-8")).body
             if isinstance(node, ast.ClassDef) and node.name in names]
    assert len(nodes) == len(names)
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), globals())

load("backtest/decision.py", {"OrderDir", "Order"})
load("backtest/high_performance_ds.py", {"BaseSingleMetric", "BaseOrderIndicator",
    "SingleMetric", "PandasSingleMetric", "PandasOrderIndicator", "NumpyOrderIndicator"})
load("backtest/report.py", {"Indicator"})

METRICS = ["inner_amount", "deal_amount", "trade_price", "trade_value", "trade_cost", "trade_dir"]

def make(cls, omit=()):
    store = cls()
    for name, value in zip(METRICS, [4.0, 2.0, 3.0, 6.0, 1.0, 1.0]):
        if name not in omit:
            store.assign(name, {"A": value})
    return store

def snap(store):
    return {name: {"index": store.get_index_data(name).index.tolist(),
                   "values": store.get_index_data(name).data.tolist()} for name in store.data}

def failure(cls, missing):
    inner = [make(cls, omit) for omit in missing]
    output = Indicator(cls)
    for name in METRICS:
        output.order_indicator.assign(name, {"OLD": 9.0})
    try:
        output._agg_order_trade_info(inner)
    except KeyError as error:
        return {"error": error.args[0], "out": snap(output.order_indicator),
                "prices": [snap(store).get("trade_price") for store in inner]}
    raise AssertionError("missing metric must fail")

def shared_pipeline(cls):
    class Decision:
        trade_range = None
        def get_decision(self):
            return [type("Target", (), {"stock_id": "A", "amount_delta": 8.0})()]

    class Exchange:
        def __init__(self, target):
            self.target, self.calls = target, 0
        def get_deal_price(self, *args, **kwargs):
            self.calls += 1
            self.target.assign("base_price", {"A": 200.0})
            self.target.assign("base_volume", {"A": 3.0})
            return 100.0

    stamp = pd.Timestamp("2024-01-02 09:30")
    steps = [(Decision(), stamp, stamp + pd.Timedelta(hours=1))] * 2
    results = {}
    for self_alias in [False, True]:
        output = Indicator(cls)
        raw = make(cls)
        raw.assign("base_price", {"A": 100.0})
        raw.assign("base_volume", {"A": 2.0})
        if self_alias:
            output.order_indicator = raw
        exchange = Exchange(raw)
        output.agg_order_indicators([raw, raw], steps, Decision(), exchange)
        assert exchange.calls == 0
        results["self" if self_alias else "duplicate"] = snap(output.order_indicator)

    for self_alias in [False, True]:
        output = Indicator(cls)
        output.order_indicator.assign("trade_dir", {"A": 1.0})
        raw = output.order_indicator if self_alias else cls()
        raw.assign("base_price", {"A": float("nan")})
        raw.assign("base_volume", {"A": 9.0})
        exchange = Exchange(raw)
        output._agg_base_price([raw, raw], steps, exchange)
        results["callback_self" if self_alias else "callback"] = {
            "price": snap(output.order_indicator)["base_price"],
            "volume": snap(output.order_indicator)["base_volume"],
            "calls": exchange.calls,
        }
    return results

def run(cls):
    child = Indicator(cls)
    child.order_indicator = make(cls)
    stamp = pd.Timestamp("2024-01-02 09:30")
    child.record(stamp)
    raw = child.get_order_indicator(raw=True)
    frozen = snap(raw)
    child.reset()
    parent = Indicator(cls)
    parent._agg_order_trade_info([raw, raw])
    assert raw is child.order_indicator_his[stamp]
    assert raw is not child.order_indicator
    duplicate = {"old": snap(raw), "out": snap(parent.order_indicator), "frozen": frozen,
                 "new_empty": not child.order_indicator.data}

    self_output = Indicator(cls)
    self_output.order_indicator = make(cls)
    self_raw = self_output.get_order_indicator(raw=True)
    self_output._agg_order_trade_info([self_raw, self_raw])
    assert self_raw is self_output.order_indicator
    return {"duplicate": duplicate, "self_alias": snap(self_raw),
            "shared_pipeline": shared_pipeline(cls),
            "late_missing": failure(cls, [("trade_cost",)]),
            "metric_order": failure(cls, [("trade_cost",), ("trade_value",)]),
            "early_missing": failure(cls, [(), ("deal_amount",)])}

print(json.dumps({"numpy": run(NumpyOrderIndicator), "pandas": run(PandasOrderIndicator)},
                 allow_nan=False, sort_keys=True))
