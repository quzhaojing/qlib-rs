"""Real PPO.update keyword behavior and NumPy permutation side effects."""
import hashlib
import json
from pathlib import Path

import gym
import numpy as np
import torch
from tianshou.data import Batch, ReplayBuffer
from tianshou.policy import base
from tianshou.data import batch as batch_module
from ppo_policy_contract import Features
from network_contract import source_module


def main():
    module = source_module(Path('D:/code/github/qlib/qlib/rl/order_execution/policy.py'), {})
    torch.set_num_threads(1)
    options = [{}, {'repeat': 1}, {'batch_size': 5}]
    options += [dict(batch_size=size, repeat=repeat) for size, repeat in [
        (0, 0), (None, 0), ('bad', 0), (1.5, 0), (-1, -2), (2, None),
        (2, 1.0), (2, 0.0), (2, True), (2, False), (False, True),
        (True, 1), (None, 1), (0, 1), (-1, 1), (.5, 1),
        (1.0, 1), (2.5, 1), ('bad', 1), ([], 1), ({}, 1), (5, 2)]]
    options += [dict(batch_size=5, repeat=1, extra='ignored')]
    options += [dict(sample_size=9), dict(buffer=None)]
    cases = []
    for missing_buffer in (False, True):
        for kwargs in options:
            policy = module.PPO(Features(torch.float32), gym.spaces.Box(-1., 1., (2,)),
                                gym.spaces.Discrete(3), lr=.003)
            policy._norm_adv = False
            buffer = ReplayBuffer(size=8)
            for index in range(5):
                obs = Batch(data_processed=torch.tensor([.1 * index, .2]))
                buffer.add(Batch(obs=obs, obs_next=obs, act=index % 3, rew=.1 * index,
                                 terminated=index in (1, 4), truncated=False))
            np.random.seed(81)
            before = np.random.get_state()
            try:
                result = policy.update(0, None if missing_buffer else buffer, **kwargs)
                lengths = {name: len(values) for name, values in result.items()}
                error = None
            except Exception as exception:
                lengths, error = None, type(exception).__name__
            after = np.random.get_state()
            cases.append(dict(options=kwargs, missing_buffer=missing_buffer, error=error,
                              lengths=lengths, state=dict(updating=policy.updating, count=policy.ret_rms.count,
                                                        learned=bool(policy.optim.state)),
                              permutation_consumed=before[2] != after[2] or not np.array_equal(before[1], after[1])))
    print(json.dumps(dict(sources={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in [Path(base.__file__), Path(batch_module.__file__)]}, cases=cases), indent=2))


if __name__ == '__main__':
    main()
