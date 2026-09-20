"""Characterize the unchanged qlib.backtest.get_strategy_executor body."""

from __future__ import annotations

import ast
import copy
import json
import sys
import types
from pathlib import Path


source_path = Path(r"D:\code\github\qlib\qlib\backtest\__init__.py")
module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
function = next(
    node
    for node in module.body
    if isinstance(node, ast.FunctionDef) and node.name == "get_strategy_executor"
)


class BaseStrategy:
    pass


class BaseExecutor:
    pass


qlib = types.ModuleType("qlib")
qlib.__path__ = []
strategy_package = types.ModuleType("qlib.strategy")
strategy_package.__path__ = []
strategy_module = types.ModuleType("qlib.strategy.base")
strategy_module.BaseStrategy = BaseStrategy
backtest_package = types.ModuleType("qlib.backtest")
backtest_package.__path__ = []
executor_module = types.ModuleType("qlib.backtest.executor")
executor_module.BaseExecutor = BaseExecutor
sys.modules.update(
    {
        "qlib": qlib,
        "qlib.strategy": strategy_package,
        "qlib.strategy.base": strategy_module,
        "qlib.backtest": backtest_package,
        "qlib.backtest.executor": executor_module,
    }
)

namespace = {"__name__": "qlib.backtest.contract", "__package__": "qlib.backtest"}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), namespace)
get_strategy_executor = namespace["get_strategy_executor"]


class TracingKwargs(dict):
    def __init__(self, values, events):
        super().__init__(values)
        self.events = events

    def __copy__(self):
        self.events.append(["copy_exchange_kwargs"])
        return type(self)(self, self.events)


def run(name, failure=None, provided_times=False):
    events = []
    nested = object()
    account_marker = object()
    exchange_marker = object()
    strategy_config = object()
    executor_config = object()
    account = {"cash": 10.0, "A": 2.0}
    values = {"freq": "1min", "nested": nested}
    if provided_times:
        values.update(start_time=None, end_time="exchange-end")
    exchange_arguments = TracingKwargs(values, events)
    infrastructures = []

    def create_account_instance(**kwargs):
        events.append(
            [
                "account",
                kwargs["start_time"],
                kwargs["end_time"],
                kwargs["benchmark"],
                kwargs["pos_type"],
                kwargs["account"] is account,
            ]
        )
        kwargs["account"].pop("cash")
        if failure == "account":
            raise RuntimeError("account")
        return account_marker

    def get_exchange(**kwargs):
        events.append(
            [
                "exchange",
                kwargs["start_time"],
                kwargs["end_time"],
                kwargs["freq"],
                kwargs["nested"] is nested,
            ]
        )
        if failure == "exchange":
            raise RuntimeError("exchange")
        return exchange_marker

    class CommonInfrastructure:
        def __init__(self, trade_account, trade_exchange):
            events.append(
                [
                    "infra",
                    trade_account is account_marker,
                    trade_exchange is exchange_marker,
                ]
            )
            self.trade_account = trade_account
            self.trade_exchange = trade_exchange
            infrastructures.append(self)

    class Strategy(BaseStrategy):
        def reset_common_infra(self, common_infra):
            events.append(["strategy_reset", common_infra is infrastructures[0]])
            if failure == "strategy_reset":
                raise RuntimeError("strategy_reset")

    class Executor(BaseExecutor):
        def reset_common_infra(self, common_infra):
            events.append(["executor_reset", common_infra is infrastructures[0]])
            if failure == "executor_reset":
                raise RuntimeError("executor_reset")

    strategy = Strategy()
    executor = Executor()

    def init_instance_by_config(config, accept_types):
        if config is strategy_config:
            events.append(["strategy_resolve", accept_types is BaseStrategy])
            if failure == "strategy_resolve":
                raise RuntimeError("strategy_resolve")
            return strategy
        events.append(["executor_resolve", config is executor_config, accept_types is BaseExecutor])
        if failure == "executor_resolve":
            raise RuntimeError("executor_resolve")
        return executor

    namespace.update(
        copy=copy,
        create_account_instance=create_account_instance,
        get_exchange=get_exchange,
        CommonInfrastructure=CommonInfrastructure,
        init_instance_by_config=init_instance_by_config,
    )
    try:
        result = get_strategy_executor(
            start_time="outer-start",
            end_time="outer-end",
            strategy=strategy_config,
            executor=executor_config,
            benchmark="BENCH",
            account=account,
            exchange_kwargs=exchange_arguments,
            pos_type="PositionX",
        )
        returned = [result[0] is strategy, result[1] is executor]
        error = None
    except Exception as exception:
        returned = None
        error = f"{type(exception).__name__}:{exception}"
    return {
        "name": name,
        "events": events,
        "returned": returned,
        "error": error,
        "account_keys": list(account),
        "original_exchange_keys": list(exchange_arguments),
    }


cases = [
    run("success_missing_times"),
    run("success_present_times", provided_times=True),
    run("account_failure", failure="account"),
    run("exchange_failure", failure="exchange"),
    run("strategy_resolution_failure", failure="strategy_resolve"),
    run("strategy_reset_failure", failure="strategy_reset"),
    run("executor_resolution_failure", failure="executor_resolve"),
    run("executor_reset_failure", failure="executor_reset"),
]
print(json.dumps(cases, separators=(",", ":")))
