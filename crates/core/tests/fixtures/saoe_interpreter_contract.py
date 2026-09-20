import ast
import json
import math
import sys
from types import SimpleNamespace
from typing import TypedDict, cast

import numpy as np
import pandas as pd


class GenericBase:
    @classmethod
    def __class_getitem__(cls, _item):
        return cls

    def __init__(self, *args, **kwargs):
        pass


class Batch:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def load(path, names, namespace):
    tree = ast.parse(open(path, encoding="utf-8").read(), path)
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    body.extend(node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names)
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), path, "exec"), namespace)


ns = {
    "np": np,
    "math": math,
    "TypedDict": TypedDict,
    "cast": cast,
    "StateInterpreter": GenericBase,
    "ActionInterpreter": GenericBase,
    "SAOEState": object,
    "ProcessedDataProvider": GenericBase,
    "BasePolicy": GenericBase,
    "Batch": Batch,
    "pd": pd,
    "init_instance_by_config": lambda value, **kwargs: value,
}
load(
    sys.argv[1],
    {
        "CurrentStateObs",
        "FullHistoryObs",
        "DummyStateInterpreter",
        "FullHistoryStateInterpreter",
        "CurrentStepStateInterpreter",
        "CategoricalActionInterpreter",
        "TwapRelativeActionInterpreter",
        "_to_int32",
        "_to_float32",
        "canonicalize",
    },
    ns,
)
load(sys.argv[2], {"NonLearnablePolicy", "AllOne"}, ns)


def state(direction=1, cur_step=0, position=7.0, ticks=5, ticks_per_step=2):
    index = pd.date_range("2024-01-02 09:00:00", periods=ticks, freq="min")
    order = SimpleNamespace(
        direction=direction,
        BUY=1,
        amount=10.0,
        stock_id="A",
        start_time=pd.Timestamp("2024-01-02 09:00:00"),
    )
    return SimpleNamespace(
        order=order,
        cur_step=cur_step,
        position=position,
        ticks_for_order=list(range(ticks)),
        ticks_per_step=ticks_per_step,
        ticks_index=index,
        cur_time=pd.Timestamp("2024-01-02 09:02:00"),
        history_steps=pd.DataFrame({"position": [7.0, 4.0]}),
    )


categorical = ns["CategoricalActionInterpreter"](4)
categorical_final = ns["CategoricalActionInterpreter"]([0.0, 0.25, 0.5, 1.0], 3)
twap = ns["TwapRelativeActionInterpreter"]()
current = ns["CurrentStepStateInterpreter"](3).interpret(state(direction=0, cur_step=2))
dummy = ns["DummyStateInterpreter"]().interpret(state())
policy_default = ns["AllOne"](None, None).forward([0, 0, 0])
policy_discrete = ns["AllOne"](None, None, 2).forward([0, 0])
pipeline_states = [state(position=7.0), state(position=4.0)]
pipeline = [categorical.interpret(item, action) for item, action in zip(pipeline_states, policy_discrete.act)]


class Provider:
    def get_data(self, **kwargs):
        index = pd.date_range("2024-01-02 09:00:00", periods=5, freq="min")
        today = pd.DataFrame(np.arange(10, dtype=float).reshape(5, 2), index=index)
        yesterday = pd.DataFrame(np.arange(10, 20, dtype=float).reshape(5, 2), index=index)
        return SimpleNamespace(today=today, yesterday=yesterday)


full = ns["FullHistoryStateInterpreter"](3, 5, 2, Provider()).interpret(state(direction=0, cur_step=4, position=4.0))

try:
    categorical.interpret(state(), 99)
    categorical_error = None
except AssertionError:
    categorical_error = "AssertionError"

print(
    json.dumps(
        {
            "categorical": [
                categorical.interpret(state(), 2),
                categorical.interpret(state(), 4),
                categorical_final.interpret(state(cur_step=2), 0),
            ],
            "categorical_values": categorical.action_values,
            "categorical_error": categorical_error,
            "twap": twap.interpret(state(cur_step=1, position=8.0), 1.5),
            "current": current,
            "dummy": int(dummy["DUMMY"]),
            "policy_default": policy_default.act.tolist(),
            "policy_discrete": policy_discrete.act.tolist(),
            "pipeline": pipeline,
            "full": {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in full.items()},
            "full_dtypes": {key: str(value.dtype) for key, value in full.items() if isinstance(value, np.ndarray)},
        },
        sort_keys=True,
    )
)
