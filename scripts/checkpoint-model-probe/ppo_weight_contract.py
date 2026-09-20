"""Actual Qlib PPO state registration and constructor/set_weight load behavior."""
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import gym
import torch
from network_contract import record, source_module
from ppo_policy_contract import Features


def main():
    path = Path('D:/code/github/qlib/qlib/rl/order_execution/policy.py')
    module = source_module(path, {})
    torch.set_num_threads(1)
    results = []
    for kind in ['full', 'legacy', 'missing', 'shape', 'conflicting_aliases']:
        network = Features(torch.float32)
        policy = module.PPO(network, gym.spaces.Box(-10., 10., (2,)),
                            gym.spaces.Discrete(3), lr=.003)
        with torch.no_grad():
            for parameter in policy.parameters():
                parameter.fill_(.1)
        state = policy.state_dict()
        inputs = OrderedDict((key, torch.full_like(value, (index + 2) / 10.))
                             for index, (key, value) in enumerate(state.items())
                             if kind != 'legacy' or not key.startswith('_actor_critic.'))
        if kind == 'missing':
            del inputs['actor.layer_out.0.bias']
        if kind == 'shape':
            inputs['actor.layer_out.0.bias'] = torch.tensor([.8, .9])
        # Full, internally consistent snapshots must restore without conversion.
        if kind == 'full':
            inputs = OrderedDict((key, torch.full_like(value, .7)) for key, value in state.items())
        original = {key: record(value) for key, value in inputs.items()}
        error = None
        try:
            module.set_weight(policy, inputs)
        except RuntimeError as exc:
            error = type(exc).__name__
        final = policy.state_dict()
        values = list(final.values())
        results.append(dict(kind=kind, initial=original, error=error,
                            input_keys=list(inputs), final={key: record(value) for key, value in final.items()},
                            aliases=[next(i for i, value in enumerate(values)
                                          if value.data_ptr() == tensor.data_ptr()) for tensor in values]))

    # The file-reading boundary is substituted; constructor + set_weight are actual source.
    events = []
    original_adam = torch.optim.Adam

    def adam(*args, **kwargs):
        events.append('optimizer')
        return original_adam(*args, **kwargs)

    class Trainer:
        @staticmethod
        def get_policy_state_dict(weight_file):
            events.append(str(weight_file))
            return inputs_for_constructor

    module.Trainer = Trainer
    inputs_for_constructor = OrderedDict((key, torch.full_like(value, .6))
                                         for key, value in state.items()
                                         if not key.startswith('_actor_critic.'))
    torch.optim.Adam = adam
    try:
        constructed = module.PPO(Features(torch.float32), gym.spaces.Box(-10., 10., (2,)),
                                 gym.spaces.Discrete(3), lr=.003, weight_file=Path('weights.native'))
    finally:
        torch.optim.Adam = original_adam
    print(json.dumps(dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          cases=results, constructor=dict(events=events, keys=list(inputs_for_constructor),
                          initialized=len(constructed.optim.state),
                          final={key: record(value) for key, value in constructed.state_dict().items()})),
                     separators=(',', ':')))


if __name__ == '__main__':
    main()
