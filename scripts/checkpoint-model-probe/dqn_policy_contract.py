"""Actual Qlib DQN construction, target cadence, n-step and policy updates."""
import hashlib
import itertools
import json
from collections import OrderedDict
from pathlib import Path

import gym
import numpy as np
import torch
from tianshou.data import Batch, ReplayBuffer
from network_contract import record, source_module
from ppo_policy_contract import Features


def source_policy(dtype, frequency, double, huber):
    path = Path('D:/code/github/qlib/qlib/rl/order_execution/policy.py')
    module = source_module(path, {})
    p = module.DQN(Features(dtype), gym.spaces.Box(-10., 10., (2,)), gym.spaces.Discrete(3),
                   lr=.003, weight_decay=.1, discount_factor=.9, estimation_step=3,
                   target_update_freq=frequency, is_double=double, clip_loss_grad=huber).to(dtype)
    independent = not frequency or all(a.data_ptr() != b.data_ptr()
                                       for a, b in zip(p.model.parameters(), p.model_old.parameters()))
    with torch.no_grad():
        for prefix in ['model'] + (['model_old'] if frequency else []):
            m = getattr(p, prefix)
            m.extractor.scale.copy_(torch.tensor([.8, 1.2], dtype=dtype))
            m.layer_out[0].weight.copy_(torch.tensor([[.1,.2],[-.3,.4],[.5,-.2]], dtype=dtype))
            m.layer_out[0].bias.copy_(torch.tensor([.1,-.2,.3], dtype=dtype))
        if frequency:
            p.model_old.layer_out[0].bias.copy_(torch.tensor([.5,.1,-.3], dtype=dtype))
    return p, independent


def replay(dtype):
    observations = torch.tensor([[.2,-.4],[.5,.1],[-.3,.8],[.7,-.2],[-.1,.3]], dtype=dtype)
    buffer = ReplayBuffer(size=8)
    for i in range(5):
        buffer.add(Batch(obs=Batch(data_processed=observations[i].numpy()),
                         obs_next=Batch(data_processed=(observations[i] + .1).numpy()), act=i % 3,
                         rew=[.2,-.4,.8,.1,-.3][i], terminated=i == 1, truncated=i == 3))
    return buffer, observations


def weights(p):
    return {name: record(value) for name, value in p.state_dict().items()}


def main():
    torch.set_num_threads(1)
    cases = []
    for dtype, frequency, double, huber in itertools.product(
            [torch.float32, torch.float64], [0, 2], [False, True], [False, True]):
        p, independent = source_policy(dtype, frequency, double, huber)
        buffer, observations = replay(dtype)
        initial = weights(p)
        steps = []
        for step in range(4):
            ids = np.array([4, 2, 0, 1, 3])
            batch = buffer[ids]
            batch.weight = np.array([.2, .5, 1., 1.5, 2.])
            p.process_fn(batch, buffer, ids)
            returns = record(batch.returns)
            result = p.learn(batch)
            steps.append(dict(returns=returns, loss=result['loss'], td=record(batch.weight),
                              weights=weights(p), iteration=p._iter,
                              target_grad_absent=not frequency or all(v.grad is None for v in p.model_old.parameters())))
        p.train(False)
        p.train(True)
        cases.append(dict(dtype=str(dtype).removeprefix('torch.'), frequency=frequency,
                          double=double, huber=huber, independent=independent, initial=initial,
                          observations=record(observations), next_observations=record(observations + .1),
                          steps=steps, target_training=False if frequency else None,
                          mode=[p.training, p.model.training], max_actions=p.max_action_num))
    lifecycle = []
    for failure in [None, 'sample', 'process', 'learn', 'priority', 'scheduler']:
        p, _ = source_policy(torch.float32, 2, True, False)
        buffer, _ = replay(torch.float32)
        events = []
        def enter(stage):
            events.append(stage)
            if failure == stage:
                raise RuntimeError(stage)
        sample = buffer.sample
        def sampled(size):
            enter('sample')
            return sample(0)
        buffer.sample = sampled
        process = p.process_fn
        def processed(*args):
            enter('process')
            return process(*args)
        p.process_fn = processed
        learn = p.learn
        original_forward = p.model.forward
        online_calls = [0]
        def model_forward(*args, **kwargs):
            online_calls[0] += 1
            if failure == 'learn' and online_calls[0] == 2:
                raise RuntimeError('learn')
            return original_forward(*args, **kwargs)
        p.model.forward = model_forward
        def learned(*args, **kwargs):
            events.append('learn')
            return learn(*args, **kwargs)
        p.learn = learned
        def priority(indices, value):
            enter('priority')
            assert list(value.shape) == [5]
        buffer.update_weight = priority
        class Scheduler:
            def step(self):
                enter('scheduler')
        p.lr_scheduler = Scheduler()
        error = None
        try:
            p.update(0, buffer)
        except RuntimeError as exc:
            error = str(exc)
        lifecycle.append(dict(failure=failure, events=events, error=error,
                              updating=p.updating, iteration=p._iter))
    path = Path('D:/code/github/qlib/qlib/rl/order_execution/policy.py')
    module = source_module(path, {})
    weight_cases = []
    for frequency, kind in itertools.product([0, 2], ['full', 'missing', 'shape']):
        p, _ = source_policy(torch.float32, frequency, True, False)
        loaded = OrderedDict((key, torch.full_like(value, .6)) for key, value in p.state_dict().items())
        if kind == 'missing':
            del loaded['model.layer_out.0.bias']
        if kind == 'shape':
            loaded['model.layer_out.0.bias'] = torch.tensor([.8, .9])
        initial = {key: record(value) for key, value in loaded.items()}
        error = None
        try:
            module.set_weight(p, loaded)
        except RuntimeError as exc:
            error = type(exc).__name__
        weight_cases.append(dict(frequency=frequency, kind=kind, initial=initial,
                                 error=error, input_keys=list(loaded), final=weights(p)))
    print(json.dumps(dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          cases=cases, lifecycle=lifecycle, weight_cases=weight_cases), separators=(',', ':')))


if __name__ == '__main__':
    main()
