"""Characterize the unchanged Qlib simulator wrapper, not numerical execution."""

import ast
import json
import sys
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


class Simulator:
    @classmethod
    def __class_getitem__(cls, _):
        return cls

    def __init__(self, initial):
        self.initial = initial


class NestedExecutor:
    def __init__(self):
        self.terminal = False

    def finished(self):
        return self.terminal


class BaseTradeDecision:
    pass


class SAOEStrategy:
    pass


tree = ast.parse(Path(sys.argv[1]).read_text(encoding="utf-8"))
source_class = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == "SingleAssetOrderExecution")
module = ast.fix_missing_locations(ast.Module(body=[
    ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
    source_class,
], type_ignores=[]))


def run(cash, config, failure=None):
    events = []
    executor = NestedExecutor()
    adapter = SimpleNamespace(saoe_state={"position": 10}, twap_price=12.5)
    strategy = SAOEStrategy()
    strategy.adapter_dict = {"order-key": adapter}
    order = SimpleNamespace(
        date=pd.Timestamp("2024-01-02"), stock_id="A", key_by_day="order-key",
        start_time=pd.Timestamp("2024-01-02 09:30"),
        end_time=pd.Timestamp("2024-01-02 15:00"),
    )

    def initialize(value):
        events.append(["init", value])
        if failure == "init":
            raise ValueError("init failed")

    def build(**kwargs):
        events.append(["build", kwargs["account"], kwargs["pos_type"],
                       str(kwargs["start_time"]), str(kwargs["end_time"])])
        if failure == "build":
            raise ValueError("build failed")
        return object(), executor

    def collect(**kwargs):
        events.append(["collect", str(kwargs["end_time"])])
        yield BaseTradeDecision()
        action = yield strategy
        events.append(["action", action])
        # Source sends the same action through every intervening decision yield.
        first = yield BaseTradeDecision()
        events.append(["forwarded", first])
        if failure == "advance":
            raise ValueError("advance failed")
        adapter.saoe_state = {"position": 0}
        kwargs["return_value"]["complete"] = True
        executor.terminal = True

    ns = dict(pd=pd, Simulator=Simulator, NestedExecutor=NestedExecutor,
              Order=SimpleNamespace, SAOEState=dict,
              Generator=Generator, SAOEStrategy=SAOEStrategy,
              BaseTradeDecision=BaseTradeDecision, init_qlib=initialize,
              get_strategy_executor=build, collect_data_loop=collect,
              TradeRangeByTime=lambda start, end: (start, end))
    exec(compile(module, sys.argv[1], "exec"), ns)
    instance = ns["SingleAssetOrderExecution"].__new__(ns["SingleAssetOrderExecution"])
    try:
        instance.__init__(order, {}, {}, config, cash)
    except ValueError as error:
        assert not hasattr(instance, "_order")
        return {"failure": str(error), "events": events}

    assert instance.initial is order and instance._order is order
    assert len(instance.decisions) == 1 and not instance.done()
    assert instance.get_state() is adapter.saoe_state
    assert instance.twap_price == 12.5
    initial_state = instance.get_state()
    try:
        instance.step(7.0)
    except ValueError as error:
        assert failure == "advance" and str(error) == "advance failed"
        assert instance.get_state() is initial_state and not instance.done()
        assert len(instance.decisions) == 2
        return {"failure": str(error), "events": events, "decisions": 2}

    assert instance.done() and instance.report_dict == {"complete": True}
    assert instance.get_state() == {"position": 0}
    assert instance.get_state() is not initial_state
    assert len(instance.decisions) == 2
    try:
        instance.step(1.0)
    except AssertionError as error:
        assert str(error) == "Simulator has already done!"
    else:
        raise AssertionError("terminal step was accepted")
    return {"events": events, "decisions": 2, "final_position": 0,
            "terminal_step_rejected": True}


print(json.dumps({
    "unlimited": run(None, None),
    "zero": run(0, {}),
    "positive": run(25, {"region": "cn"}),
    "init_failure": run(None, {}, "init"),
    "build_failure": run(None, None, "build"),
    "advance_failure": run(None, None, "advance"),
}))
