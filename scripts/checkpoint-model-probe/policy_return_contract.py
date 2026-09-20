"""Actual A2C/PPO return preprocessing and running variance, not a copied formula."""
import hashlib
import itertools
import json
from pathlib import Path

import gym
import numpy as np
import torch
import tianshou
from tianshou.data import Batch, ReplayBuffer
from tianshou.policy import PPOPolicy
from tianshou.policy.modelfree import a2c
from tianshou.utils import statistics


def numbers(values):
    return [str(float(value)) for value in np.asarray(values).reshape(-1)]


def stats(value):
    return dict(mean=str(float(value.mean)), variance=str(float(value.var)), count=value.count)


class Actor(torch.nn.Module):
    def forward(self, obs, state=None, info=None):
        return torch.ones((len(obs), 2)), state


class Critic(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones((), dtype=dtype))

    def forward(self, obs):
        return torch.as_tensor(obs, dtype=self.weight.dtype).reshape(-1, 1) * self.weight


def run(dtype, normalize, capacity, special=None, initial=None):
    critic = Critic(dtype)
    policy = PPOPolicy(Actor(), critic, torch.optim.Adam(critic.parameters()),
                       torch.distributions.Categorical, reward_normalization=normalize,
                       value_clip=normalize, discount_factor=.9, gae_lambda=.95,
                       max_batchsize=2, action_space=gym.spaces.Discrete(2))
    if initial is not None:
        policy.ret_rms.mean, policy.ret_rms.var, policy.ret_rms.count = initial
    steps = []
    for step in range(3):
        buffer = ReplayBuffer(size=capacity)
        for index in range(7):
            reward = (-1.)**index * (.1 + index * .07) + step * .3
            current, following = -.1 + .13 * index, .2 + .07 * index
            if special == 'constant':
                reward, current, following = 0., 0., 0.
            elif special is not None and index == 6:
                reward = special
            buffer.add(Batch(obs=current, obs_next=following, act=0, rew=reward,
                             terminated=index in (1, 5), truncated=index == 3))
        indices = buffer.sample_indices(0)
        batch = buffer[indices]
        values = critic(batch.obs).detach().numpy().flatten()
        next_values = critic(batch.obs_next).detach().numpy().flatten()
        before = stats(policy.ret_rms)
        output = policy._compute_returns(batch, buffer, indices)
        assert not output.v_s.requires_grad
        assert not output.returns.requires_grad and not output.adv.requires_grad
        steps.append(dict(rewards=numbers(batch.rew), values=numbers(values),
                          next_values=numbers(next_values), terminated=batch.terminated.tolist(),
                          truncated=batch.truncated.tolist(), indices=indices.tolist(),
                          bootstrap_valid=(~buffer.terminated[indices]).tolist(),
                          unfinished_indices=buffer.unfinished_index().tolist(),
                          before=before, after=stats(policy.ret_rms),
                          old_values=numbers(output.v_s.numpy()),
                          returns=numbers(output.returns.numpy()), advantages=numbers(output.adv.numpy())))
    return dict(name=f'{dtype}-{normalize}-{capacity}-{special}-{initial}', dtype=str(dtype),
                normalize=normalize, gamma=.9, gae_lambda=.95, steps=steps)


def main():
    torch.set_num_threads(1)
    cases = [run(dtype, normalize, capacity)
             for dtype, normalize, capacity in itertools.product(
                 (torch.float32, torch.float64), (False, True), (4, 16))]
    for dtype, special in itertools.product((torch.float32, torch.float64),
                                           ('constant', float('nan'), float('inf'), -float('inf'))):
        cases.append(run(dtype, True, 16, special))
    for dtype, initial in itertools.product((torch.float32, torch.float64),
                                            ((100., 9., 5), (3., 1e80, 9), (0., float('inf'), 2))):
        cases.append(run(dtype, True, 16, initial=initial))
    print(json.dumps(dict(tianshou=tianshou.__version__, torch=torch.__version__,
                          sources={Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                                   for module in (a2c, statistics)}, cases=cases), allow_nan=False))


if __name__ == '__main__':
    main()
