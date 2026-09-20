"""Unchanged Indicator and NumpyOrderIndicator: row aliases, reset and recalculation.

Loads the real SingleData implementation. Only unused module wiring/logger construction
is replaced; indicator mutation, metric calculation and table export are actual source.
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
spec = importlib.util.spec_from_file_location("identity_index_data", root / "utils/index_data.py")
index_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(index_data)
idd = index_data
SingleData = index_data.SingleData

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
load("backtest/high_performance_ds.py", {"BaseOrderIndicator", "NumpyOrderIndicator"})
load("backtest/report.py", {"Indicator"})

first, second = pd.Timestamp("2024-01-02 09:30"), pd.Timestamp("2024-01-02 09:31")
indicator = Indicator()
order = Order("A", 10.0, OrderDir.BUY, None, None)
order.deal_amount = 4.0
indicator._update_order_trade_info([(order, 40.0, 1.0, 10.0)])
indicator._update_order_fulfill_rate()
indicator.order_indicator.assign("pa", {"A": 0.25})
indicator.trade_indicator["custom"] = 7.0
indicator.cal_trade_indicators(first, "1min")
indicator.record(first)
indicator.record(second)
old_order, old_trade = indicator.order_indicator, indicator.trade_indicator
frozen = indicator.generate_trade_indicators_dataframe()
assert indicator.order_indicator_his[first] is indicator.order_indicator_his[second] is old_order
assert indicator.trade_indicator_his[first] is indicator.trade_indicator_his[second] is old_trade
old_order.assign("pa", {"A": 0.5})
indicator.cal_trade_indicators(second, "1min")
assert old_trade is indicator.trade_indicator
assert old_trade["custom"] == 7.0
assert indicator.trade_indicator_his[first]["pa"] == 0.5
assert frozen.loc[first, "pa"] == 0.25
before_failure = dict(old_trade)
try:
    indicator.cal_trade_indicators(second, "1min", {"pa_config": {"weight_method": "bad"}})
except ValueError:
    pass
else:
    raise AssertionError("invalid weight method must fail")
assert dict(old_trade) == before_failure and indicator.trade_indicator is old_trade
keys = list(old_trade)
indicator.reset()
assert indicator.order_indicator is not old_order and indicator.trade_indicator is not old_trade
assert not indicator.order_indicator.data and not indicator.trade_indicator
old_order.assign("pa", {"A": 0.75})
old_trade["custom"] = 9.0
assert indicator.order_indicator_his[second].get_index_data("pa").data[0] == 0.75
assert indicator.trade_indicator_his[second]["custom"] == 9.0
indicator.record(first)
assert list(indicator.trade_indicator_his) == [first, second]
assert indicator.trade_indicator_his[first] is indicator.trade_indicator
assert indicator.order_indicator_his[first] is indicator.order_indicator
assert indicator.trade_indicator_his[second] is old_trade
print(json.dumps({"same_generation_aliases": True, "recalculation_preserves_identity": True,
    "keys": keys, "frozen_pa": float(frozen.loc[first, "pa"]), "live_pa": old_trade["pa"],
    "failed_calculation_retains_map": True, "reset_detaches": True,
    "old_order_pa": float(old_order.get_index_data("pa").data[0]), "old_custom": old_trade["custom"],
    "replacement_keeps_key_order": True}))
