import ast
import json
import sys
from types import GeneratorType

executor_path, simulator_path, *action_arg = sys.argv[1:]
action = float(action_arg[0]) if action_arg else 7.0
executor_tree = ast.parse(open(executor_path, encoding="utf-8").read())
base = next(node for node in executor_tree.body if isinstance(node, ast.ClassDef) and node.name == "BaseExecutor")
collect = next(node for node in base.body if isinstance(node, ast.FunctionDef) and node.name == "collect_data")
nested = next(node for node in executor_tree.body if isinstance(node, ast.ClassDef) and node.name == "NestedExecutor")
inner_collect = next(node for node in nested.body if isinstance(node, ast.FunctionDef) and node.name == "_collect_data")
simulator_tree = ast.parse(open(simulator_path, encoding="utf-8").read())
simulator = next(node for node in simulator_tree.body if isinstance(node, ast.ClassDef) and node.name == "SingleAssetOrderExecution")
iterate = next(node for node in simulator.body if isinstance(node, ast.FunctionDef) and node.name == "_iter_strategy")
for function in (collect, inner_collect, iterate):
    function.returns = None
    for argument in function.args.args:
        argument.annotation = None


class NestedExecutor:
    pass


class BasePosition:
    ST_NO = "None"


class BaseTradeDecision:
    pass


class SAOEStrategy:
    pass


namespace = {
    "GeneratorType": GeneratorType,
    "NestedExecutor": NestedExecutor,
    "BasePosition": BasePosition,
    "BaseTradeDecision": BaseTradeDecision,
    "SAOEStrategy": SAOEStrategy,
    "Optional": object,
    "get_start_end_idx": lambda calendar, decision: (0, 0),
}
module = ast.Module(body=[collect, inner_collect, iterate], type_ignores=[])
exec(compile(ast.fix_missing_locations(module), executor_path, "exec"), namespace)
received = []
steps = []


class Decision(BaseTradeDecision):
    def __init__(self, name):
        self.name = name

    def get_range_limit(self, default_value=None):
        return None

    def update(self, calendar):
        return None

    def empty(self):
        return False

    def mod_inner_decision(self, decision):
        pass


class Position:
    def settle_start(self, settle_type):
        pass

    def settle_commit(self):
        pass


class Indicator:
    def get_order_indicator(self, raw=True):
        return "indicator"


class Account:
    current_position = Position()

    def update_bar_end(self, *args, **kwargs):
        pass

    def get_trade_indicator(self):
        return Indicator()


class Calendar:
    def __init__(self, name):
        self.name = name
        self.index = 0

    def get_step_time(self):
        return self.name, self.index

    def get_trade_step(self):
        return self.index

    def step(self):
        steps.append(self.name)
        self.index += 1


class Infrastructure:
    def set_sub_level_infra(self, infrastructure):
        pass


class Immediate:
    def reset(self, *args, **kwargs):
        pass

    def alter_outer_trade_decision(self, decision):
        return decision

    def generate_trade_decision(self, previous):
        return Decision("middle")

    def post_exe_step(self, result):
        pass

    def post_upper_level_exe_step(self):
        pass


class Proxy(SAOEStrategy):
    def reset(self, *args, **kwargs):
        pass

    def alter_outer_trade_decision(self, decision):
        return decision

    def generate_trade_decision(self, previous):
        action = yield self
        received.append(action)
        return Decision("atomic")

    def post_exe_step(self, result):
        pass

    def post_upper_level_exe_step(self):
        pass


class Atomic:
    collect_data = namespace["collect_data"]
    track_data = True
    _settle_type = "None"
    trade_exchange = object()
    indicator_config = {}

    def __init__(self):
        self.trade_calendar = Calendar("atomic")
        self.trade_account = Account()

    def reset(self, **kwargs):
        self.trade_calendar.index = 0

    def get_level_infra(self):
        return 0

    def finished(self):
        return self.trade_calendar.index >= 1

    def _collect_data(self, trade_decision, level=0):
        return [("fill", trade_decision.name)], {"trade_info": []}


class Recursive(NestedExecutor):
    collect_data = namespace["collect_data"]
    _collect_data = namespace["_collect_data"]
    track_data = True
    _settle_type = "None"
    trade_exchange = object()
    indicator_config = {}
    _skip_empty_decision = True
    _align_range_limit = True

    def __init__(self, name, inner, strategy):
        self.name = name
        self.inner_executor = inner
        self.inner_strategy = strategy
        self.trade_calendar = Calendar(name)
        self.trade_account = Account()
        self.level_infra = Infrastructure()

    def reset(self, **kwargs):
        self.trade_calendar.index = 0

    def get_level_infra(self):
        return 0

    def finished(self):
        return self.trade_calendar.index >= 1

    def _init_sub_trading(self, decision):
        self.inner_executor.reset()
        self.level_infra.set_sub_level_infra(self.inner_executor.get_level_infra())
        self.inner_strategy.reset(0, decision)

    def _update_trade_decision(self, decision):
        return decision

    def post_inner_exe_step(self, result):
        self.inner_strategy.post_exe_step(result)


class Driver:
    _iter_strategy = namespace["_iter_strategy"]

    def __init__(self, generator):
        self._collect_data_loop = generator
        self.decisions = []


atomic = Atomic()
child = Recursive("child", atomic, Proxy())
top = Recursive("top", child, Immediate())
driver = Driver(top.collect_data(Decision("outer")))
prompt = driver._iter_strategy(None)
first = [decision.name for decision in driver.decisions] + [type(prompt).__name__]
try:
    driver._iter_strategy(action)
except StopIteration:
    pass
second = [decision.name for decision in driver.decisions[2:]] + ["Complete"]
print(json.dumps({"first": first, "second": second, "received": received, "steps": steps}, separators=(",", ":")))
