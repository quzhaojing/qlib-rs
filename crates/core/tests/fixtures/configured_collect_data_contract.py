import ast
import json
from pathlib import Path

source = Path(r"D:\code\github\qlib\qlib\backtest\__init__.py")
tree = ast.parse(source.read_text(encoding="utf-8"))
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "collect_data")
function.returns = None
for argument in (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs):
    argument.annotation = None
namespace = {}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
collect_data = namespace["collect_data"]


def run(mode, defaults=False, invalid_first=False):
    events = []
    strategy = object()
    executor = object()
    account = {"cash": 7}
    exchange = {"nested": object()}
    result = {"keep": object()}
    token = object()
    first = object()
    second = object()

    def assemble(*args, **kwargs):
        events.append(["assemble", len(args), kwargs.get("pos_type"), args[4], args[5], args[6] is exchange])
        if mode == "assembly":
            raise RuntimeError("assembly")
        return strategy, executor

    def loop(start, end, actual_strategy, actual_executor, return_value=None):
        events.append(["loop", start, end, actual_strategy is strategy, actual_executor is executor, return_value is result])
        if mode == "loop":
            raise RuntimeError("loop")
        received = yield first
        events.append(["received", received is token])
        yield second
        if return_value is not None:
            return_value.update({"published": first})

    collect_data.__globals__.update(get_strategy_executor=assemble, collect_data_loop=loop)
    if defaults:
        generator = collect_data("start", "end", object(), object())
    else:
        generator = collect_data("start", "end", object(), object(), "BENCH", account, exchange, "PositionX", result)
    before = list(events)
    invalid = None
    if invalid_first:
        try:
            generator.send(token)
        except Exception as error:
            invalid = type(error).__name__
    yielded = []
    error = None
    try:
        yielded.append(next(generator) is first)
        yielded.append(generator.send(token) is second)
        next(generator)
    except StopIteration:
        pass
    except Exception as failure:
        error = f"{type(failure).__name__}:{failure}"
    return {
        "before": before,
        "events": events,
        "yielded": yielded,
        "invalid": invalid,
        "error": error,
        "kept": "keep" in result,
        "published": result.get("published") is first,
    }


print(json.dumps([
    run("success"),
    run("success", defaults=True),
    run("success", invalid_first=True),
    run("assembly"),
    run("loop"),
], separators=(",", ":")))
