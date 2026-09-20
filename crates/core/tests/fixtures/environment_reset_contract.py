import ast
import copy
import json
import sys
import typing


MISSING = object()


def load_reset(path):
    tree = ast.parse(open(path, encoding="utf-8").read(), path)
    wrapper = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "EnvWrapper")
    method = copy.deepcopy(next(node for node in wrapper.body if isinstance(node, ast.FunctionDef) and node.name == "reset"))
    method.returns = None
    for argument in [*method.args.posonlyargs, *method.args.args, *method.args.kwonlyargs]:
        argument.annotation = None
    if method.args.kwarg is not None:
        method.args.kwarg.annotation = None
    shell = ast.ClassDef(name="ExtractedEnv", bases=[], keywords=[], body=[method], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[shell], type_ignores=[]))
    namespace = {
        "SEED_INTERATOR_MISSING": MISSING,
        "EnvWrapperStatus": lambda **values: values,
        "generate_nan_observation": lambda _space: "invalid",
        "weakref": type("WeakRef", (), {"proxy": staticmethod(lambda value: value)}),
        "cast": lambda _kind, value: value,
        "Callable": typing.Callable,
        "Iterator": typing.Iterator,
        "Simulator": object,
        "InitialStateType": object,
        "EnvWrapper": object,
        "RuntimeError": RuntimeError,
    }
    exec(compile(module, path, "exec"), namespace)
    return namespace["ExtractedEnv"]


ExtractedEnv = load_reset(sys.argv[1])


class Logger:
    def __init__(self, events):
        self.events = events

    def reset(self):
        self.events.append("logger_reset")


class Simulator:
    def __init__(self, value, events, state_failure=None):
        self.value = value
        self.events = events
        self.state_failure = state_failure
        self.env = None

    def get_state(self):
        self.events.append("state")
        if self.state_failure is not None:
            raise self.state_failure()
        return self.value


class Interpreter:
    observation_space = object()

    def __init__(self, events, failure=None):
        self.events = events
        self.failure = failure

    def __call__(self, value):
        self.events.append("interpret")
        if self.failure is not None:
            raise self.failure()
        return value * 10


def run(seeds, factory_failure=None, state_failure=None, interpreter_failure=None, count=1):
    env = ExtractedEnv()
    events = []
    env.logger = Logger(events)
    env.seed_iterator = MISSING if seeds == "missing" else iter(seeds)
    env.state_interpreter = Interpreter(events, interpreter_failure)
    env.observation_space = object()

    def factory(*args):
        events.append("factory:" + ("none" if not args else str(args[0])))
        if factory_failure is not None:
            raise factory_failure()
        return Simulator(7 if not args else args[0], events, state_failure)

    env.simulator_fn = factory
    outputs = []
    for _ in range(count):
        try:
            outputs.append(["ok", env.reset()])
        except Exception as error:
            outputs.append(["error", type(error).__name__])
    return {
        "outputs": outputs,
        "events": events,
        "dead": env.seed_iterator is None,
        "status": getattr(env, "status", None),
    }


class FallibleSeeds:
    def __init__(self, values):
        self.values = iter(values)

    def __iter__(self):
        return self

    def __next__(self):
        value = next(self.values)
        if isinstance(value, BaseException):
            raise value
        return value


print(json.dumps({
    "seeded": run([2], count=3),
    "unseeded": run("missing", count=2),
    "factory_stop": run([1], factory_failure=StopIteration),
    "state_stop": run([1], state_failure=StopIteration),
    "interpreter_stop": run([1], interpreter_failure=StopIteration),
    "factory_error": run([1], factory_failure=ValueError),
    "state_error": run([1], state_failure=ValueError),
    "iterator_error_then_retry": run(FallibleSeeds([2, ValueError("seed"), 3, StopIteration(), 4]), count=5),
    "iterator_error_state": run(FallibleSeeds([2, ValueError("seed")]), count=2),
    "iterator_end": run(FallibleSeeds([2]), count=3),
}))
