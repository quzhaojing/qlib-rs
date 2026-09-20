"""Actual Tianshou DQN inference, dense Adam learning and exploration contracts."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from tianshou.data import Batch
from tianshou.policy.modelfree import dqn
from network_contract import record


class Logits(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.values = torch.nn.Parameter(torch.tensor(
            [[-1., 0., 1.], [2., -2., .5], [.7, .7, -.3]], dtype=dtype))

    def forward(self, obs, state=None, info=None):
        return self.values, state


def policy(dtype, huber=False, target=False):
    model = Logits(dtype)
    return dqn.DQNPolicy(model, torch.optim.Adam(model.parameters(), lr=.003, weight_decay=.1),
                         clip_loss_grad=huber, target_update_freq=2 if target else 0)


def main():
    torch.set_num_threads(1)
    inference, learning, exploration = [], [], []
    masks = [None, [[0., 1., 0.], [0., 0., 0.], [1., 1., 0.]],
             [[.5, 2., -1.]]]
    for dtype, mask, target, double in itertools.product(
            [torch.float32, torch.float64], masks, [False, True], [False, True]):
        p = policy(dtype, target=target)
        p._is_double = double
        if target:
            with torch.no_grad():
                p.model_old.values.copy_(p.model.values.flip(1) + .25)
        obs = Batch(obs=np.zeros(3), **({} if mask is None else {'mask': np.array(mask)}))
        batch = Batch(obs=obs, obs_next=obs, info={})
        class Buffer:
            def __getitem__(self, indices):
                return batch
        out = p(batch, state='kept')
        inference.append(dict(logits=record(p.model.values), mask=mask, double=double,
                              old=None if not target else record(p.model_old.values),
                              q=record(p.compute_q_value(out.logits, None if mask is None else np.array(mask))),
                              actions=out.act.tolist(), state=out.state,
                              target=record(p._target_q(Buffer(), np.arange(3)))))
    for dtype, huber, kind in itertools.product(
            [torch.float32, torch.float64], [False, True],
            ['none', 'vector', 'column', 'scalar', 'integer', 'float32', 'float16']):
        p = policy(dtype, huber)
        initial = record(p.model.values)
        steps = []
        for step in range(3):
            weights = {'none': None, 'vector': torch.tensor([.2, -1., 2.], dtype=dtype),
                       'column': torch.tensor([[.2], [-1.], [2.]], dtype=torch.float64),
                       'scalar': torch.tensor(0., dtype=torch.float64),
                       'integer': torch.tensor([0, -1, 2], dtype=torch.int64),
                       'float32': torch.tensor([.2, -1., 2.], dtype=torch.float32),
                       'float16': torch.tensor([.2, -1., 2.], dtype=torch.float16)}
            weight = weights[kind]
            returns = torch.tensor([[-2.], [2.5], [.7]], dtype=torch.float64) + step * .125
            batch = Batch(obs=np.zeros(3), info={}, act=np.array([0, -1, 1]), returns=returns)
            if weight is not None:
                batch.weight = weight
            result = p.learn(batch)
            steps.append(dict(returns=record(returns), weight=None if weight is None else record(weight),
                              loss=result['loss'], td=record(batch.weight),
                              gradient=record(p.model.values.grad), final=record(p.model.values)))
        learning.append(dict(initial=initial, huber=huber, steps=steps))
    for eps, masked in itertools.product([0., 1e-8, 1.1e-8, .5, 1., -1., 2.], [False, True]):
        p = policy(torch.float32)
        p.max_action_num = 3
        p.set_eps(eps)
        mask = np.array([[0., 1., 0.], [0., 0., 0.], [1., 1., 0.]]) if masked else None
        obs = Batch(**({} if mask is None else {'mask': mask}))
        draws = []
        original = np.random.rand
        def draw(*shape):
            value = original(*shape)
            draws.extend(value.reshape(-1).tolist())
            return value
        np.random.seed(73)
        np.random.rand = draw
        try:
            out = p.exploration_noise(np.array([2, 0, 1]), Batch(obs=obs))
        finally:
            np.random.rand = original
        exploration.append(dict(eps=eps, mask=None if mask is None else mask.tolist(),
                                draws=draws, actions=out.tolist()))
    print(json.dumps(dict(source_sha256=hashlib.sha256(Path(dqn.__file__).read_bytes()).hexdigest(),
                          inference=inference, learning=learning, exploration=exploration), separators=(',', ':')))


if __name__ == '__main__':
    main()
