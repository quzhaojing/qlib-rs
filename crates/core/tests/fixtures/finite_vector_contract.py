import ast
import contextlib
import copy
import json
import sys
import warnings

import numpy as np


class MockBase:
    def __init__(self, reset_batches, step_batches, loggers):
        self.env_num = 2
        self.reset_batches = iter(reset_batches)
        self.step_batches = iter(step_batches)
        self.backend_events = []
        self._logger = loggers
        self._alive_env_ids = set()
        self._reset_alive_envs()
        self._default_obs = self._default_info = self._default_rew = None
        self._zombie = False
        self._collector_guarded = False

    def _wrap_id(self, ids):
        if ids is None:
            return list(range(self.env_num))
        if isinstance(ids, int):
            return [ids]
        return list(ids)

    def reset(self, ids):
        batch = next(self.reset_batches)
        self.backend_events.append(["reset", list(ids)])
        return [batch[i] for i in ids if i in batch]

    def step(self, actions, ids):
        batch = next(self.step_batches)
        self.backend_events.append(["step", list(ids), np.asarray(actions).tolist()])
        rows = [batch[i] for i in ids if i in batch]
        return tuple(np.asarray([row[column] for row in rows], dtype=object) for column in range(4))


def load_class(path):
    tree = ast.parse(open(path, encoding="utf-8").read(), path)
    source = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FiniteVectorEnv")
    names = {
        "_reset_alive_envs", "_set_default_obs", "_set_default_info", "_set_default_rew",
        "_get_default_obs", "_get_default_info", "_get_default_rew", "_postproc_env_obs",
        "collector_guard", "reset", "step",
    }
    methods = []
    for node in source.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node = copy.deepcopy(node)
            node.returns = None
            for argument in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
                argument.annotation = None
            methods.append(node)
    shell = ast.ClassDef(
        name="ExtractedFiniteVectorEnv",
        bases=[ast.Name(id="MockBase", ctx=ast.Load())],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[shell], type_ignores=[]))
    namespace = {
        "MockBase": MockBase,
        "copy": copy,
        "warnings": warnings,
        "np": np,
        "contextmanager": contextlib.contextmanager,
        "check_nan_observation": lambda value: value == "invalid",
        "StopIteration": StopIteration,
        "RuntimeWarning": RuntimeWarning,
        "cast": lambda _kind, value: value,
        "Tuple": tuple,
    }
    exec(compile(module, path, "exec"), namespace)
    return namespace["ExtractedFiniteVectorEnv"]


Extracted = load_class(sys.argv[1])


class Logger:
    def __init__(self, events, fail=None):
        self.events = events
        self.fail = fail

    def _event(self, name, *values):
        self.events.append([name, *values])
        if self.fail == name:
            raise ValueError(name)

    def on_env_all_ready(self):
        self._event("ready")

    def on_env_all_done(self):
        self._event("done")

    def on_env_reset(self, env_id, observations):
        self._event("reset", env_id, copy.deepcopy(observations))

    def on_env_step(self, env_id, observation, reward, done, info):
        self._event("step", env_id, observation, reward, bool(done), info)


events = []
env = Extracted(
    [{0: 10, 1: "invalid"}, {0: "invalid"}],
    [{0: (20, 1.0, True, {"source": 0})}],
    [Logger(events)],
)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    first_reset = env.reset()
first_alive = sorted(env._alive_env_ids)
step = env.step(np.asarray([100, 200]), [1, 0])
try:
    env.reset([1, 0])
    exhausted = False
except StopIteration:
    exhausted = True

guard_events = []
guarded = Extracted([{0: "invalid", 1: "invalid"}], [], [Logger(guard_events)])
with guarded.collector_guard():
    guarded.reset()

error_events = []
errored = Extracted([], [], [Logger(error_events)])
try:
    with errored.collector_guard():
        raise ValueError("collector")
except ValueError:
    pass


def native(value):
    if isinstance(value, np.ndarray):
        return [native(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {key: native(item) for key, item in value.items()}
    if isinstance(value, list):
        return [native(item) for item in value]
    return value


print(json.dumps({
    "first_reset": native(first_reset),
    "warning_count": len(caught),
    "first_alive": first_alive,
    "step": native(step),
    "backend_events": env.backend_events,
    "logger_events": events,
    "exhausted": exhausted,
    "zombie": env._zombie,
    "guard_events": guard_events,
    "guarded_flag": guarded._collector_guarded,
    "error_events": error_events,
    "error_guarded_flag": errored._collector_guarded,
}))
