"""Exercise the unchanged outer generator, including multi-round sends and cleanup."""
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

path = Path(sys.argv[1])
tree = ast.parse(path.read_text(encoding="utf-8"))
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"backtest_loop", "collect_data_loop"}]
module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
ns = {"cast": lambda _kind, value: value, "PORT_METRIC": object(), "INDICATOR_METRIC": object()}
exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)

def run(failures=(), steps=2, reports=True, cancel=False):
    events = []
    def stage(name, *values):
        events.append([name, *values])
        if name in failures:
            raise ValueError(name)
    class Executor:
        step = 0
        def reset(self, **_): stage("reset")
        def get_level_infra(self): stage("level"); return self
        def finished(self): stage("finished"); return self.step == steps
        def get_all_executors(self): stage("reports"); return []
        def collect_data(self, decision, level):
            assert level == 0 and decision == "decision"
            stage("begin")
            try:
                yield "decision"
                action = yield "prompt"
            except GeneratorExit:
                stage("child_close")
                raise
            stage("resume", action)
            self.step += 1
            return [action]
    class Strategy:
        def reset(self, **_): stage("strategy_reset")
        def generate_trade_decision(self, previous): stage("generate", previous); return "decision"
        def post_exe_step(self, result): stage("post", result)
        def post_upper_level_exe_step(self): stage("finalize")
    class Bar:
        def __enter__(self): stage("progress_enter"); return self
        def __exit__(self, *_): stage("progress_close")
        def update(self, count): assert count == 1; stage("progress_update")
    executor = Executor()
    executor.trade_calendar = SimpleNamespace(get_trade_len=lambda: (stage("trade_len"), steps)[1])
    ns["tqdm"] = lambda **_: Bar()
    result = {}
    generator = ns["collect_data_loop"](None, None, Strategy(), executor, result if reports else None)
    yielded = []
    error = None
    context = None
    try:
        current = next(generator)
        if cancel:
            yielded.append(current)
            generator.close()
        else:
            while True:
                yielded.append(current)
                # Decision-yield sends are ignored by the delegated generator.
                current = generator.send(999.0 if current == "decision" else 3.0 + executor.step)
    except StopIteration:
        pass
    except ValueError as caught:
        error = str(caught)
        if isinstance(caught.__context__, ValueError): context = str(caught.__context__)
    return {"events": events, "yielded": yielded, "published": bool(result), "error": error, "context": context}

names = ["reset", "level", "strategy_reset", "trade_len", "progress_enter", "finished",
         "generate", "begin", "resume", "post", "progress_update", "finalize", "progress_close", "reports"]
out = {"normal": run(), "empty": run(steps=0), "no_reports": run(reports=False), "cancel": run(cancel=True)}
out.update({name: run((name,)) for name in names})
out["double_failure"] = run(("post", "progress_close"))
out["cancel_failure"] = run(("child_close", "progress_close"), cancel=True)

wrapper_events = []
def wrapper_collect(_start, _end, _strategy, _executor, return_value):
    wrapper_events.append("start")
    yield "ignored"
    wrapper_events.append("resume")
    return_value.update({"portfolio_dict": {"day": 1}, "indicator_dict": {"day": 2}})
saved_collect = ns["collect_data_loop"]
ns["collect_data_loop"] = wrapper_collect
try:
    portfolio, indicator = ns["backtest_loop"](None, None, object(), object())
finally:
    ns["collect_data_loop"] = saved_collect
out["wrapper"] = {"events": wrapper_events, "portfolio": portfolio, "indicator": indicator}
print(json.dumps(out))
