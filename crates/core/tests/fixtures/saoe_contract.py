import ast
import collections
import json
import sys
import typing
from types import GeneratorType, SimpleNamespace

state_path, strategy_path = sys.argv[1:]
state_tree = ast.parse(open(state_path, encoding="utf-8").read())
state_class = next(node for node in state_tree.body if isinstance(node, ast.ClassDef) and node.name == "SAOEState")
state_namespace = {
    "NamedTuple": typing.NamedTuple,
    "Optional": typing.Optional,
    "Order": object,
    "SAOEMetrics": dict,
    "BaseIntradayBacktestData": object,
    "pd": SimpleNamespace(Timestamp=object, DataFrame=object, DatetimeIndex=object),
}
exec(compile(ast.fix_missing_locations(ast.Module(body=[state_class], type_ignores=[])), state_path, "exec"), state_namespace)
state_type = state_namespace["SAOEState"]
sentinels = [object() for _ in state_type._fields]
state = state_type(*sentinels)

strategy_tree = ast.parse(open(strategy_path, encoding="utf-8").read())
saoe = next(node for node in strategy_tree.body if isinstance(node, ast.ClassDef) and node.name == "SAOEStrategy")
generate = next(node for node in saoe.body if isinstance(node, ast.FunctionDef) and node.name == "generate_trade_decision")
post = next(node for node in saoe.body if isinstance(node, ast.FunctionDef) and node.name == "post_exe_step")
proxy = next(node for node in strategy_tree.body if isinstance(node, ast.ClassDef) and node.name == "ProxySAOEStrategy")
proxy_generate = next(node for node in proxy.body if isinstance(node, ast.FunctionDef) and node.name == "_generate_trade_decision")
proxy_reset = next(node for node in proxy.body if isinstance(node, ast.FunctionDef) and node.name == "reset")
for function in (generate, post, proxy_generate, proxy_reset):
    function.returns = None
    for argument in function.args.args:
        argument.annotation = None
proxy_reset.body = proxy_reset.body[1:]


class TradeDecision:
    def __init__(self, orders, strategy):
        self.order_list = orders
        self.strategy = strategy


events = []


class OrderHelper:
    def create(self, *args):
        events.append(["create", *args])
        return ["order", *args]


class Exchange:
    def get_order_helper(self):
        events.append(["helper"])
        return OrderHelper()


namespace = {
    "GeneratorType": GeneratorType,
    "TradeDecisionWO": TradeDecision,
    "collections": collections,
    "Optional": object,
}
functions = ast.Module(body=[generate, post, proxy_generate, proxy_reset], type_ignores=[])
exec(compile(ast.fix_missing_locations(functions), strategy_path, "exec"), namespace)


class Order:
    def __init__(self, key):
        self.key_by_day = key
        self.stock_id = "A"
        self.direction = 1


class Adapter:
    def __init__(self, key):
        self.key = key

    def update(self, results, step_range):
        events.append(["update", self.key, len(results), list(step_range)])


class Proxy:
    generate_trade_decision = namespace["generate_trade_decision"]
    _generate_trade_decision = namespace["_generate_trade_decision"]
    reset = namespace["reset"]
    post_exe_step = namespace["post_exe_step"]

    def __init__(self):
        self.trade_exchange = Exchange()
        self.adapter_dict = {"K": Adapter("K"), "Z": Adapter("Z")}
        self._last_step_range = (0, 0)
        self._order = Order("old")

    def get_data_cal_avail_range(self, rtype):
        events.append(["range", rtype])
        return 3, 5


strategy = Proxy()
strategy.reset(TradeDecision([Order("K")], None))
events.append(["reset", strategy._order.key_by_day])
generator = strategy.generate_trade_decision([])
yielded = next(generator)
events.append(["yield", yielded is strategy, list(strategy._last_step_range)])
try:
    generator.send(7.5)
except StopIteration as result:
    events.append(["return", result.value.order_list, result.value.strategy is strategy])
strategy.post_exe_step([(Order("K"), 0, 0, 0)])
strategy._last_step_range = (2, 2)
strategy.post_exe_step(None)
events.append(["zero-range"])

print(
    json.dumps(
        {
            "fields": list(state_type._fields),
            "tuple_identity": [state[index] is value for index, value in enumerate(sentinels)],
            "events": events,
        },
        separators=(",", ":"),
    )
)
