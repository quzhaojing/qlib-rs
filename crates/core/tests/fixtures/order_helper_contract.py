import ast
import json
from pathlib import Path

source = Path(r"D:\code\github\qlib\qlib\backtest\decision.py")
tree = ast.parse(source.read_text(encoding="utf-8"))
helper = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OrderHelper")
methods = {
    node.name: node
    for node in helper.body
    if isinstance(node, ast.FunctionDef) and node.name in {"__init__", "create"}
}
for method in methods.values():
    method.decorator_list = []
    method.returns = None
    for argument in [*method.args.posonlyargs, *method.args.args, *method.args.kwonlyargs]:
        argument.annotation = None

namespace = {}
exec(compile(ast.Module(body=list(methods.values()), type_ignores=[]), str(source), "exec"), namespace)


class Harness:
    __init__ = namespace["__init__"]
    create = staticmethod(namespace["create"])


class Stamp:
    def __init__(self, label):
        self.label = label


events = []


class Pandas:
    @staticmethod
    def Timestamp(value):
        label = value.label if isinstance(value, Stamp) else value
        events.append(["timestamp", label])
        if label in {"bad-start", "bad-end"}:
            raise ValueError(label)
        return value if isinstance(value, Stamp) else Stamp(f"parsed:{value}")


class Order:
    def __init__(self, **kwargs):
        encoded = dict(kwargs)
        for key in ("start_time", "end_time"):
            value = encoded[key]
            encoded[key] = None if value is None else value.label
        encoded["direction"] = kwargs["direction"]
        events.append(["order", encoded])
        self.values = encoded


namespace["pd"] = Pandas
namespace["Order"] = Order


def run(start=None, end=None):
    events.clear()
    exchange = object()
    target = Harness(exchange)
    try:
        order = target.create("股票/α", -0.0, 1, start, end)
        result = order.values
        error = None
    except Exception as failure:
        result = None
        error = f"{type(failure).__name__}:{failure}"
    return {
        "exchange_identity": target.exchange is exchange,
        "events": list(events),
        "result": result,
        "error": error,
    }


print(json.dumps([
    run(),
    run("2024-01-02", "2024-01-03 09:30"),
    run(Stamp("existing-start"), Stamp("existing-end")),
    run("bad-start", "unreached"),
    run("ok", "bad-end"),
], separators=(",", ":"), ensure_ascii=True))
