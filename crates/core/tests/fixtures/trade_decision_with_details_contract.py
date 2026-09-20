import ast
import hashlib
import json
from pathlib import Path

source = Path(r"D:\code\github\qlib\qlib\backtest\decision.py")
source_bytes = source.read_bytes()
assert hashlib.sha256(source_bytes).hexdigest() == "a6866d15bc8f3ad1c75bfc3856ccde5245d43f0a2de07b1ad565d6e7e20d8251"
tree = ast.parse(source_bytes.decode("utf-8"), filename=str(source))
target = next(
    node
    for node in tree.body
    if isinstance(node, ast.ClassDef) and node.name == "TradeDecisionWithDetails"
)
target.decorator_list = []
target.bases = [
    ast.copy_location(ast.Name(id="TradeDecisionWO", ctx=ast.Load()), target.bases[0])
]
target.keywords = []
for node in ast.walk(target):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        node.returns = None
        for argument in [
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ]:
            argument.annotation = None

events = []


class TradeDecisionWO:
    def __init__(self, order_list, strategy, trade_range=None):
        events.append(["parent", len(order_list), trade_range])
        self.order_list = order_list
        self.strategy = strategy
        self.trade_range = trade_range
        order_list.append("parent-mutated")
        if strategy["fail"]:
            raise RuntimeError("parent-failed")


namespace = {"TradeDecisionWO": TradeDecisionWO}
exec(compile(ast.Module(body=[target], type_ignores=[]), str(source), "exec"), namespace)
Decision = namespace["TradeDecisionWithDetails"]


def run(details, trade_range=None, fail=False):
    events.clear()
    orders = ["original"]
    strategy = {"fail": fail}
    instance = Decision.__new__(Decision)
    try:
        instance.__init__(orders, strategy, trade_range, details)
        error = None
    except Exception as failure:
        error = f"{type(failure).__name__}:{failure}"
    return {
        "events": list(events),
        "orders": orders,
        "order_identity": getattr(instance, "order_list", None) is orders,
        "strategy_identity": getattr(instance, "strategy", None) is strategy,
        "details_present": hasattr(instance, "details"),
        "details_identity": getattr(instance, "details", None) is details,
        "details_is_none": getattr(instance, "details", object()) is None,
        "error": error,
    }


opaque = object()
print(
    json.dumps(
        [run(None), run(opaque, [2, 5]), run(opaque, [8, 9], fail=True)],
        separators=(",", ":"),
        ensure_ascii=True,
    )
)
