"""Characterize the unchanged public qlib.backtest.backtest wrapper."""

from __future__ import annotations

import ast
import json
from pathlib import Path


source_path = Path(r"D:\code\github\qlib\qlib\backtest\__init__.py")
module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
function = next(
    node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "backtest"
)
namespace = {}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), namespace)
backtest = namespace["backtest"]


def run(name, failure=None, explicit=True):
    events = []
    strategy_config = object()
    executor_config = object()
    strategy = object()
    executor = object()
    report = object()
    exchange_arguments = {"freq": "1min"}
    account = {"cash": 10.0}

    def get_strategy_executor(*args, **kwargs):
        events.append(
            [
                "assemble",
                args[0],
                args[1],
                args[2] is strategy_config,
                args[3] is executor_config,
                args[4],
                args[5] is account if explicit else args[5],
                args[6] is exchange_arguments if explicit else len(args[6]),
                kwargs["pos_type"],
            ]
        )
        if failure == "assembly":
            raise RuntimeError("assembly")
        return strategy, executor

    def backtest_loop(start_time, end_time, actual_strategy, actual_executor):
        events.append(
            [
                "loop",
                start_time,
                end_time,
                actual_strategy is strategy,
                actual_executor is executor,
            ]
        )
        if failure == "loop":
            raise RuntimeError("loop")
        return report

    namespace.update(
        get_strategy_executor=get_strategy_executor,
        backtest_loop=backtest_loop,
    )
    try:
        if explicit:
            result = backtest(
                "start",
                "end",
                strategy_config,
                executor_config,
                "BENCH",
                account,
                exchange_arguments,
                pos_type="PositionX",
            )
        else:
            result = backtest("start", "end", strategy_config, executor_config)
        returned = result is report
        error = None
    except Exception as exception:
        returned = None
        error = f"{type(exception).__name__}:{exception}"
    return {"name": name, "events": events, "returned": returned, "error": error}


cases = [
    run("explicit_success"),
    run("default_success", explicit=False),
    run("assembly_failure", failure="assembly"),
    run("loop_failure", failure="loop"),
]
print(json.dumps(cases, separators=(",", ":")))
