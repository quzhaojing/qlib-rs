"""Characterize the unchanged qlib.backtest.get_exchange dispatcher."""

from __future__ import annotations

import ast
import json
from pathlib import Path


source_path = Path(r"D:\code\github\qlib\qlib\backtest\__init__.py")
module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "get_exchange")
namespace = {}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), namespace)
get_exchange = namespace["get_exchange"]


def run(name, source=None, threshold="omitted", fail_default=False, **overrides):
    events = []
    marker = object()

    class Config:
        @property
        def limit_threshold(self):
            events.append(["default"])
            if fail_default:
                raise RuntimeError("default")
            return 0.095

    class Logger:
        def info(self, message):
            events.append(["log", message])

    class FakeExchange:
        def __init__(self, **kwargs):
            events.append(["create", kwargs])
            if kwargs.get("fail"):
                raise RuntimeError("create")
            self.created = True

    existing = object.__new__(FakeExchange)
    existing.created = False
    configured = existing if source == "existing" else {"class": "Configured"} if source == "mapping" else "fail"

    def init_instance_by_config(config, accept_types):
        events.append([
            "resolve",
            "existing" if config is existing else config,
            accept_types is FakeExchange,
        ])
        if config == "fail":
            raise RuntimeError("resolve")
        return existing if config is existing else marker

    namespace.update(C=Config(), logger=Logger(), Exchange=FakeExchange, init_instance_by_config=init_instance_by_config)
    arguments = dict(
        exchange=None if source is None else configured,
        freq="1min",
        start_time="2024-01-02",
        end_time="2024-01-31",
        codes=["A", "B"],
        subscribe_fields=["$vwap"],
        open_cost=0.1,
        close_cost=0.2,
        min_cost=3.0,
        deal_price=["$ask", "$bid"],
        extra=7,
    )
    arguments.update(overrides)
    if threshold != "omitted":
        arguments["limit_threshold"] = threshold
    try:
        result = get_exchange(**arguments)
        returned = "existing" if result is existing else "marker" if result is marker else "new"
        error = None
    except Exception as exception:
        returned = None
        error = f"{type(exception).__name__}:{exception}"
    return {"name": name, "events": events, "returned": returned, "error": error}


cases = [
    run("new_default"),
    run("new_explicit", threshold=("up", "down"), deal_price=None, codes="all"),
    run("configured", source="mapping", threshold=0.2),
    run("existing_default", source="existing"),
    run("default_failure", source="mapping", fail_default=True),
    run("create_failure", threshold=0.1, fail=True),
    run("resolve_failure", source="failure", threshold=0.1),
]
print(json.dumps(cases, separators=(",", ":")))
