import ast
import json
import math
import sys
from types import SimpleNamespace
from typing import Any, Dict, Generic, Optional, Tuple, TypeVar


SimulatorState = TypeVar("SimulatorState")


def load_classes(path):
    tree = ast.parse(open(path, encoding="utf-8").read(), path)
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in {"Reward", "RewardCombination"}
    ]
    module = ast.fix_missing_locations(ast.Module(selected, type_ignores=[]))
    namespace = {
        "Any": Any,
        "Dict": Dict,
        "EnvWrapper": object,
        "Generic": Generic,
        "Optional": Optional,
        "SimulatorState": SimulatorState,
        "Tuple": Tuple,
        "final": lambda function: function,
    }
    exec(compile(module, path, "exec"), namespace)
    return namespace["Reward"], namespace["RewardCombination"]


Reward, RewardCombination = load_classes(sys.argv[1])


class FixedReward(Reward):
    def __init__(self, name, value, calls, failure=False, self_log=False):
        self.name = name
        self.value = value
        self.calls = calls
        self.failure = failure
        self.self_log = self_log

    def reward(self, _state):
        self.calls.append(self.name)
        if self.self_log:
            self.log("child", self.value)
        if self.failure:
            raise RuntimeError(self.name)
        return self.value


class Logger:
    def __init__(self, fail_name=None):
        self.values = []
        self.fail_name = fail_name

    def add_scalar(self, name, value):
        self.values.append([name, float(value)])
        if name == self.fail_name:
            raise RuntimeError(name)


def attached(rewards, logger):
    combination = RewardCombination(rewards)
    combination.env = SimpleNamespace(logger=logger)
    return combination


calls = []
logger = Logger()
success = attached(
    {
        "alpha": (FixedReward("alpha", 1.5, calls), 2.0),
        "beta": (FixedReward("beta", 4.0, calls), -0.5),
        "gamma": (FixedReward("gamma", -7.0, calls), 0.0),
    },
    logger,
)
success_value = success(None)

child_calls = []
child_logger = Logger()
child_failure = attached(
    {
        "first": (FixedReward("first", 2.0, child_calls), 3.0),
        "broken": (FixedReward("broken", 4.0, child_calls, failure=True), 1.0),
        "never": (FixedReward("never", 8.0, child_calls), 1.0),
    },
    child_logger,
)
try:
    child_failure(None)
    child_failed = False
except RuntimeError:
    child_failed = True

log_calls = []
log_logger = Logger(fail_name="first")
log_failure = attached(
    {
        "first": (FixedReward("first", 2.0, log_calls), 3.0),
        "never": (FixedReward("never", 8.0, log_calls), 1.0),
    },
    log_logger,
)
try:
    log_failure(None)
    log_failed = False
except RuntimeError:
    log_failed = True

missing_calls = []
missing = RewardCombination({"first": (FixedReward("first", 2.0, missing_calls), 3.0)})
try:
    missing(None)
    missing_failed = False
except AssertionError:
    missing_failed = True

self_log_calls = []
self_log_logger = Logger()
self_logging_child = attached(
    {"outer": (FixedReward("inner", 2.0, self_log_calls, self_log=True), 3.0)},
    self_log_logger,
)
try:
    self_logging_child(None)
    self_log_failed = False
except AssertionError:
    self_log_failed = True

empty = RewardCombination({})(None)

ieee_logger = Logger()
ieee = attached(
    {
        "positive": (FixedReward("positive", 1.0, [],), math.inf),
        "negative": (FixedReward("negative", -1.0, []), math.inf),
    },
    ieee_logger,
)(None)

print(
    json.dumps(
        {
            "success": success_value,
            "calls": calls,
            "logs": logger.values,
            "child_failure": [child_failed, child_calls, child_logger.values],
            "log_failure": [log_failed, log_calls, log_logger.values],
            "missing": [missing_failed, missing_calls],
            "self_log": [self_log_failed, self_log_calls, self_log_logger.values],
            "empty": empty,
            "ieee": [math.isnan(ieee), math.isinf(ieee_logger.values[0][1]), math.isinf(ieee_logger.values[1][1])],
        },
        allow_nan=True,
    )
)
