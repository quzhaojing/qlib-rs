"""Execute pinned upstream base/nested generators together, including local rebinding."""
import ast
import hashlib
import json
from pathlib import Path
from types import GeneratorType, SimpleNamespace

path = Path(r"D:\code\github\qlib\qlib\backtest\executor.py")
raw = path.read_bytes()
assert hashlib.sha256(raw).hexdigest() == "76ab94ce77691487da6cd41bcfe3fe14b5149e917835bcaaba0ed174afd0fa88"
tree = ast.parse(raw)
methods = []
for cls, names in [("BaseExecutor", {"collect_data"}),
                   ("NestedExecutor", {"_collect_data", "_update_trade_decision", "post_inner_exe_step"})]:
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    methods.extend(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in names)
assert len(methods) == 4
BasePosition = SimpleNamespace(ST_NO="None")
get_start_end_idx = lambda calendar, decision: (0, 0)
exec(compile(ast.Module(body=ast.parse("from __future__ import annotations").body + methods,
                        type_ignores=[]), str(path), "exec"))

events = []
empty_calls = []
class Calendar:
    index = 0
    def get_step_time(self): return (10, 20)
    def get_trade_step(self): raise AssertionError("alignment disabled must skip step lookup")
    def step(self): self.index += 1

class Decision:
    def __init__(self, name): self.name = name
    def update(self, calendar): return updated
    def empty(self): empty_calls.append(self.name); return False
    def mod_inner_decision(self, value): assert self is altered and value is child

original, updated, altered, child = [Decision(name) for name in ("original", "updated", "altered", "child")]
class Strategy:
    def alter_outer_trade_decision(self, value):
        assert value is updated
        events.append("alter")
        return altered
    def generate_trade_decision(self, previous):
        assert previous is None
        events.append("generate")
        value = yield "prompt"
        assert value == 42
        events.append("action")
        return child
    def post_exe_step(self, rows):
        assert rows is child_rows
        events.append("post")
    def post_upper_level_exe_step(self): events.append("upper")

child_rows = [object()]
inner_calendar, outer_calendar = Calendar(), Calendar()
class Inner:
    trade_calendar = inner_calendar
    trade_account = SimpleNamespace(get_trade_indicator=lambda: SimpleNamespace(get_order_indicator=lambda raw: {}))
    def finished(self): return inner_calendar.index == 1
    def collect_data(self, trade_decision, level):
        assert trade_decision is child and level == 1
        yield child
        inner_calendar.step()
        return child_rows

def account(**kwargs):
    assert kwargs["outer_trade_decision"] is original
    assert kwargs["decision_list"][0] == (child, 10, 20)
    events.append("account")

class NestedExecutor:
    track_data = True
    _settle_type = "None"
    _skip_empty_decision = False
    _align_range_limit = False
    trade_calendar = outer_calendar
    trade_account = SimpleNamespace(update_bar_end=lambda *args, **kwargs: account(**kwargs))
    trade_exchange = None
    indicator_config = {}
    inner_executor = Inner()
    inner_strategy = Strategy()
    _collect_data = _collect_data
    _update_trade_decision = _update_trade_decision
    post_inner_exe_step = post_inner_exe_step
    def _init_sub_trading(self, value):
        assert value is original
        events.append("reset")

generator = collect_data(NestedExecutor(), original)
assert next(generator) is original
assert events == []
assert generator.send(99) == "prompt"
assert generator.send(42) is child
try:
    generator.send(999)
    raise AssertionError("expected completion")
except StopIteration as done:
    assert done.value is not child_rows and done.value[0] is child_rows[0]
assert empty_calls == ["altered"]
assert outer_calendar.index == 1
print(json.dumps({"events": events, "empty_calls": empty_calls, "outer_steps": outer_calendar.index}))
