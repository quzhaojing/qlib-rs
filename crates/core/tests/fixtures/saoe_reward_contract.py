import ast
import json
import sys
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd


class Logger:
    def __init__(self):
        self.values = []

    def add_scalar(self, name, value):
        self.values.append([name, float(value)])


class Reward:
    env = None

    @classmethod
    def __class_getitem__(cls, _item):
        return cls

    def log(self, name, value):
        assert self.env is not None
        self.env.logger.add_scalar(name, value)


class OrderDir:
    SELL = 0
    BUY = 1


def load_classes(path):
    tree = ast.parse(open(path, encoding="utf-8").read(), path)
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in {"PAPenaltyReward", "PPOReward"}
    ]
    module = ast.fix_missing_locations(ast.Module(selected, type_ignores=[]))
    namespace = {
        "Reward": Reward,
        "SAOEState": object,
        "SAOEMetrics": object,
        "OrderDir": OrderDir,
        "cast": cast,
        "np": np,
    }
    exec(compile(module, path, "exec"), namespace)
    return namespace["PAPenaltyReward"], namespace["PPOReward"]


def state(direction=OrderDir.BUY, cur_step=0, position=10.0, prices=(12.0,), exec_rows=None):
    if exec_rows is None:
        exec_rows = [
            ("2024-01-02 09:30", 9.0, 0.0, 9.0),
            ("2024-01-02 09:31", 10.0, 1.0, 2.0),
            ("2024-01-02 09:32", 20.0, 3.0, 1.0),
        ]
    history_exec = pd.DataFrame(
        exec_rows,
        columns=["datetime", "market_price", "deal_amount", "amount"],
    ).set_index("datetime")
    history_exec.index = pd.to_datetime(history_exec.index)
    history_steps = pd.DataFrame(
        [("2024-01-02 09:31", 4.0, 2.0)], columns=["datetime", "amount", "pa"]
    ).set_index("datetime")
    history_steps.index = pd.to_datetime(history_steps.index)
    return SimpleNamespace(
        order=SimpleNamespace(amount=10.0, direction=direction),
        cur_step=cur_step,
        position=position,
        history_exec=history_exec,
        history_steps=history_steps,
        backtest_data=SimpleNamespace(get_deal_price=lambda: pd.Series(prices)),
    )


PAPenaltyReward, PPOReward = load_classes(sys.argv[1])
pa_logger = Logger()
pa = PAPenaltyReward(penalty=100.0, scale=2.0)
pa.env = SimpleNamespace(logger=pa_logger)

ppo_cases = []
ppo_cases.append(PPOReward(4).reward(state(cur_step=1)))
ppo_cases.append(PPOReward(4).reward(state(cur_step=3, prices=(24.0,))))
ppo_cases.append(PPOReward(4).reward(state(direction=OrderDir.SELL, position=0.0, prices=(20.0,))))
ppo_cases.append(
    PPOReward(4).reward(
        state(cur_step=3, prices=(10.0,), exec_rows=[("2024-01-02 09:30", 10.0, 0.0, 1.0)])
    )
)
ppo_cases.append(
    PPOReward(4).reward(
        state(cur_step=3, prices=(11.0,), exec_rows=[("2024-01-02 09:30", 10.0, 0.0, 1.0)])
    )
)
ppo_cases.append(
    PPOReward(4).reward(
        state(cur_step=3, prices=(0.0,), exec_rows=[("2024-01-02 09:30", 0.0, 0.0, 1.0)])
    )
)
ppo_cases.append(
    PPOReward(4).reward(
        state(cur_step=3, prices=(np.nan,), exec_rows=[("2024-01-02 09:30", np.nan, 0.0, 1.0)])
    )
)

assertions = {}
for name, value in [
    ("amount", state()),
    ("finite", state()),
    ("logger", state()),
]:
    reward = PAPenaltyReward()
    if name == "amount":
        value.order.amount = 0.0
    elif name == "finite":
        value.history_steps.iloc[-1, value.history_steps.columns.get_loc("pa")] = np.nan
        reward.env = SimpleNamespace(logger=Logger())
    try:
        reward.reward(value)
        assertions[name] = False
    except AssertionError:
        assertions[name] = True

print(
    json.dumps(
        {
            "pa": pa.reward(state()),
            "logs": pa_logger.values,
            "ppo": ppo_cases,
            "assertions": assertions,
        },
        allow_nan=True,
    )
)
