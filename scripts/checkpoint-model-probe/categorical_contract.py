"""Actual Tianshou forward + Torch Categorical contracts and gradients."""
import hashlib
import itertools
import json
from pathlib import Path

import gym
import numpy as np
import torch
import tianshou
from tianshou.data import Batch
from tianshou.policy import PGPolicy
from tianshou.policy.modelfree import pg
from torch.distributions import categorical


def record(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                values=[str(float(x)) for x in value.detach().to(torch.float64).flatten().tolist()])


class Actor(torch.nn.Module):
    def __init__(self, probabilities):
        super().__init__()
        self.raw = torch.nn.Parameter(probabilities)

    def forward(self, obs, state=None, info=None):
        return self.raw, state


def run(name, values, dtype, actions):
    actor = Actor(torch.tensor(values, dtype=dtype))
    policy = PGPolicy(actor, torch.optim.Adam(actor.parameters()), torch.distributions.Categorical,
                      deterministic_eval=True, action_space=gym.spaces.Discrete(3))
    policy.eval()
    state = object()
    result = policy(Batch(obs=np.arange(2)), state=state)
    assert result.state is state
    actions = torch.tensor(actions)
    log_prob = result.dist.log_prob(actions)
    entropy = result.dist.entropy()
    (log_prob.sum() + entropy.sum()).backward()
    return dict(name=f'{name}-{dtype}', raw=record(actor.raw), probabilities=record(result.dist.probs),
                logits=record(result.dist.logits), actions=record(actions),
                deterministic_actions=record(result.act), log_prob=record(log_prob),
                entropy=record(entropy), gradients=record(actor.raw.grad))


def main():
    torch.set_num_threads(1)
    layouts = [
        ('vector', [.2,.3,.5], 1),
        ('batch', [[.2,.3,.5],[0.,1.,0.]], [2,1]),
        ('negative', [[-1.,-2.,-3.],[-2.,-2.,-1.]], [0,2]),
        ('sample-broadcast', [[2.,3.,5.],[4.,4.,2.]], [[0],[1],[2]]),
        ('nested-batch', [[[1.,0.,0.],[1.,1.,1.]],[[.5,.25,.25],[0.,0.,1.]]], 2.),
    ]
    cases = [run(name, values, dtype, actions)
             for dtype,(name,values,actions) in itertools.product(
                 (torch.float16, torch.bfloat16, torch.float32, torch.float64),layouts)]
    errors = []
    for name,values in [('scalar',1.),('zero',[0.,0.,0.]),('mixed-sign',[-1.,1.,1.]),
                        ('nan',[float('nan'),1.,1.]),('inf',[float('inf'),1.,1.]),
                        ('empty-events',[])]:
        raw = torch.tensor(values, dtype=torch.float32)
        try:
            torch.distributions.Categorical(raw)
        except Exception as error:
            errors.append(dict(name=name, raw=record(raw), error=type(error).__name__))
        else:
            raise AssertionError(name)
    print(json.dumps(dict(torch=torch.__version__,tianshou=tianshou.__version__,
                          sources={Path(module.__file__).name:hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                                   for module in (pg,categorical)},cases=cases,errors=errors),allow_nan=False))


if __name__ == '__main__':
    main()
