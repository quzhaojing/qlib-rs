"""Characterize the unchanged create_account_instance wrapper in isolation."""

from __future__ import annotations

import ast
import json
from pathlib import Path


source_path = Path(r"D:\code\github\qlib\qlib\backtest\__init__.py")
module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
function = next(
    node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "create_account_instance"
)
namespace = {}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), namespace)
create_account_instance = namespace["create_account_instance"]


def run(name, account, benchmark=None, pos_type="Position"):
    events = []
    original = account
    marker = object()

    def fake_account(**kwargs):
        events.append(
            [
                "account",
                kwargs["init_cash"],
                kwargs["position_dict"],
                kwargs["pos_type"],
                kwargs["benchmark_config"],
                kwargs["position_dict"] is original,
            ]
        )
        if pos_type == "Fail":
            raise RuntimeError("account-construction")
        return marker

    namespace["Account"] = fake_account
    try:
        result = create_account_instance("2024-01-02", "2024-01-31", benchmark, account, pos_type)
        returned_marker = result is marker
        error = None
    except Exception as exception:
        returned_marker = False
        error = f"{type(exception).__name__}:{exception}"
    return {
        "name": name,
        "events": events,
        "remaining": account if isinstance(account, dict) else account,
        "returned_marker": returned_marker,
        "error": error,
    }


cases = [
    run("integer", 7),
    run("float", 2.5, "CUSTOM"),
    run("dictionary", {"cash": 11.0, "A": 2, "B": {"amount": 3.0, "price": 4.0}}, "BM", "InfPosition"),
    run("construction_failure", {"cash": 13.0, "A": 1}, "BM", "Fail"),
    run("missing_cash", {"A": 1}),
    run("unsupported", "7"),
]
print(json.dumps(cases, separators=(",", ":")))
