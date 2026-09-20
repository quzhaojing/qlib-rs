"""Actual Qlib PPO constructor + inherited process/learn with live dense Adam."""
import hashlib
import itertools
import json
from pathlib import Path

import gym
import numpy as np
import torch
import tianshou
from tianshou.data import Batch, ReplayBuffer
from tianshou.policy.modelfree import a2c, ppo
from network_contract import source_module, record


class Features(torch.nn.Module):
    output_dim = 2

    def __init__(self, dtype):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor([.8, 1.2], dtype=dtype))

    def forward(self, obs):
        return obs.data_processed * self.scale


def prepared(batch):
    return {key: record(getattr(batch, field)) for key, field in
            [('old_values', 'v_s'), ('returns', 'returns'),
             ('advantages', 'adv'), ('actions', 'act'), ('old_log_prob', 'logp_old')]}


def statistics(policy):
    rms = policy.ret_rms
    return dict(mean=float(rms.mean), variance=float(rms.var), count=rms.count)


def main():
    path = Path('D:/code/github/qlib/qlib/rl/order_execution/policy.py')
    module = source_module(path, {})
    torch.set_num_threads(1)
    cases = []
    for dtype, normalize, recompute in itertools.product(
            (torch.float32, torch.float64), (False, True), (False, True)):
        torch.manual_seed(91)
        np.random.seed(37)
        policy = module.PPO(Features(dtype), gym.spaces.Box(-10., 10., (2,)),
                            gym.spaces.Discrete(3), lr=.003, weight_decay=.1,
                            discount_factor=.9, max_grad_norm=.2 if normalize else 0.,
                            reward_normalization=normalize, value_clip=normalize,
                            gae_lambda=.95, max_batch_size=2).to(dtype)
        policy._recompute_adv = recompute
        with torch.no_grad():
            policy.actor.layer_out[0].weight.copy_(torch.tensor([[.1,.2],[-.3,.4],[.5,-.2]], dtype=dtype))
            policy.actor.layer_out[0].bias.copy_(torch.tensor([.1,-.2,.3], dtype=dtype))
            policy.critic.value_out.weight.copy_(torch.tensor([[.3,-.1]], dtype=dtype))
            policy.critic.value_out.bias.fill_(.05)
        initial = {name: record(value) for name, value in policy.named_parameters()}
        observations = np.array([[.2,-.4],[.5,.1],[-.3,.8],[.7,-.2],[-.1,.3]])
        following = observations + .1
        buffer = ReplayBuffer(size=8)
        for index in range(5):
            buffer.add(Batch(obs=Batch(data_processed=observations[index]),
                             obs_next=Batch(data_processed=following[index]), act=index % 3,
                             rew=[.2,-.4,.8,.1,-.3][index], terminated=index == 1, truncated=index == 3))
        indices = buffer.sample_indices(0)
        batch = buffer[indices]
        # Qlib to_torch preserves incoming observation dtype; use the model dtype.
        batch.obs.data_processed = torch.as_tensor(batch.obs.data_processed, dtype=dtype)
        batch.obs_next.data_processed = torch.as_tensor(batch.obs_next.data_processed, dtype=dtype)
        output = policy.process_fn(batch, buffer, indices)
        before = prepared(output)
        stats_before = statistics(policy)
        metrics = policy.learn(output, batch_size=5, repeat=3)
        cases.append(dict(dtype=str(dtype).removeprefix('torch.'), normalize=normalize, recompute=recompute,
                          initial=initial, observations=record(batch.obs.data_processed),
                          next_observations=record(batch.obs_next.data_processed),
                          indices=indices.tolist(), unfinished_indices=buffer.unfinished_index().tolist(),
                          prepared=before, stats_before=stats_before, metrics=metrics,
                          final_prepared=prepared(output), stats_after=statistics(policy),
                          final_weights={name: record(value) for name,value in policy.named_parameters()},
                          initialized=len(policy.optim.state)))
    paths = [path, Path(a2c.__file__), Path(ppo.__file__)]
    print(json.dumps(dict(torch=torch.__version__, tianshou=tianshou.__version__,
                          sources={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
                          cases=cases), allow_nan=False))


if __name__ == '__main__':
    main()
