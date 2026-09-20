"""Verify real Qlib policy checkpoint scope after an actual optimizer step."""
import copy
import hashlib
import json
from pathlib import Path

import gym
import torch

from network_contract import source_module
from ppo_policy_contract import Features


def replacement_contract(module):
    results = []
    for component in ['actor', 'critic']:
        policy = module.PPO(Features(torch.float32), gym.spaces.Box(-10., 10., (2,)),
                            gym.spaces.Discrete(3), lr=.003)
        saved = copy.deepcopy(policy.state_dict())
        original = getattr(policy, component)
        optimizer_ids = [id(value) for group in policy.optim.param_groups for value in group['params']]
        replacement = (module.PPOActor(Features(torch.float32), 3) if component == 'actor'
                       else module.PPOCritic(Features(torch.float32)))
        setattr(policy, component, replacement)
        assert getattr(policy._actor_critic, component) is original
        head = 'layer_out.0.weight' if component == 'actor' else 'value_out.weight'
        with torch.no_grad():
            original.get_parameter(head).fill_(7.)
            replacement.get_parameter(head).fill_(9.)
        state = policy.state_dict()
        assert state[f'{component}.{head}'].data_ptr() == replacement.get_parameter(head).data_ptr()
        assert state[f'_actor_critic.{component}.{head}'].data_ptr() == original.get_parameter(head).data_ptr()
        policy.load_state_dict(saved)
        assert torch.equal(replacement.get_parameter(head), saved[f'{component}.{head}'])
        assert torch.equal(original.get_parameter(head), saved[f'_actor_critic.{component}.{head}'])
        assert optimizer_ids == [id(value) for group in policy.optim.param_groups for value in group['params']]
        results.append(dict(component=component, container_keeps_original=True,
                            both_heads_restored=True, optimizer_rebound=False))
    return results


def main():
    torch.set_num_threads(1)
    path = Path('D:/code/github/qlib/qlib/rl/order_execution/policy.py')
    module = source_module(path, {})
    result = []
    for kind in ['PPO', 'DQN']:
        options = {'target_update_freq': 2} if kind == 'DQN' else {}
        policy = getattr(module, kind)(Features(torch.float32),
            gym.spaces.Box(-10., 10., (2,)), gym.spaces.Discrete(3), lr=.003, **options)
        saved = copy.deepcopy(policy.state_dict())
        sum(parameter.sum() for group in policy.optim.param_groups
            for parameter in group['params']).backward()
        policy.optim.step()
        optimizer = copy.deepcopy(policy.optim.state_dict())
        assert any(not torch.equal(value, saved[name])
                   for name, value in policy.state_dict().items())
        policy.updating = True
        if kind == 'DQN':
            policy._iter, policy.eps = 17, .7
        else:
            policy.ret_rms.mean, policy.ret_rms.var, policy.ret_rms.count = 4., 9., 23
        policy.load_state_dict(saved)
        assert all(torch.equal(value, saved[name])
                   for name, value in policy.state_dict().items())
        assert policy.updating
        assert len(policy.optim.state) == len(optimizer['state'])
        for key, state in policy.optim.state_dict()['state'].items():
            assert all(torch.equal(value, optimizer['state'][key][name])
                       for name, value in state.items())
        if kind == 'DQN':
            assert (policy._iter, policy.eps) == (17, .7)
        else:
            assert (policy.ret_rms.mean, policy.ret_rms.var, policy.ret_rms.count) == (4., 9., 23)
        result.append(dict(kind=kind, tensor_count=len(saved),
                           initialized_optimizer_count=len(optimizer['state']),
                           restored_parameters=True, retained_runtime=True))
    print(json.dumps(dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), cases=result,
                          replacements=replacement_contract(module))))


if __name__ == '__main__':
    main()
