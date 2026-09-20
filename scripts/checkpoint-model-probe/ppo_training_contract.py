"""Actual Qlib PPO.learn updates from prepared batches, including all live weights."""
import json
from pathlib import Path
import typing
from unittest.mock import patch

import gym
import numpy as np
import torch
from tianshou.data import Batch
from network_contract import source_module, record


def main():
    source = Path('D:/code/github/qlib/qlib/rl/order_execution')
    network = source_module(source / 'network.py', {'Literal':typing.Literal,'FullHistoryObs':dict})
    policy_module = source_module(source / 'policy.py', {})
    fixture = json.loads(Path('crates/core/tests/fixtures/rl_candle_network.json').read_text())
    torch.set_num_threads(1)
    trajectories = []
    for case in fixture['cases']:
        if case['kind'] != 'recurrent':
            continue
        config = case['config']
        space = {'data_processed':gym.spaces.Box(-1.,1.,(3,2))}
        extractor = network.Recurrent(space,hidden_dim=config['hidden_dim'],output_dim=config['output_dim'],
                                      rnn_type=config['kind'],rnn_num_layers=config['layers'])
        policy = policy_module.PPO(extractor,gym.spaces.Dict(space),gym.spaces.Discrete(3),
                                   lr=.003,weight_decay=.1,max_grad_norm=.1)
        initial = {name:torch.tensor(value['values'],dtype=torch.float32).reshape(value['shape'])
                   for name,value in case['weights'].items()}
        # Source registers the same ActorCritic again. Load every alias explicitly.
        policy.load_state_dict({name:initial[name.removeprefix('_actor_critic.')]
                                for name in policy.state_dict()})
        obs = Batch(**{name:torch.tensor(value['values'],dtype=getattr(torch,value['dtype'])).reshape(value['shape'])
                       for name,value in case['inputs'].items()})
        actions = torch.tensor([0,2])
        old_values = policy.critic(obs).detach()
        old_log_prob = torch.distributions.Categorical(policy.actor(obs)[0]).log_prob(actions).detach()
        steps = []
        for lr in (.003,0.,.002,.004):
            policy.optim.param_groups[0]['lr'] = lr
            prepared = dict(actions=actions,old_values=old_values,old_log_prob=old_log_prob,
                            advantages=torch.tensor([-1.,.7]),returns=torch.tensor([.3,-.2]))
            batch = Batch(obs=obs,act=actions,logp_old=old_log_prob,v_s=old_values,
                          adv=prepared['advantages'],returns=prepared['returns'])
            norms = []
            original_clip = torch.nn.utils.clip_grad_norm_
            def observe_clip(*args, **kwargs):
                norm = original_clip(*args,**kwargs)
                norms.append(record(norm))
                return norm
            np.random.seed(0)
            with patch('torch.nn.utils.clip_grad_norm_',side_effect=observe_clip):
                metrics = policy.learn(batch,batch_size=2,repeat=1)
            assert len(norms) == 1
            steps.append(dict(learning_rate=lr,initialized=len(policy.optim.state),max_grad_norm=.1,
                gradient_norm=norms[0],ppo=dict(inputs={name:record(value) for name,value in prepared.items()},
                                              metrics={name:value[0] for name,value in metrics.items()}),
                weights={name:record(value) for name,value in policy.state_dict().items()},
                actor=record(policy.actor(obs)[0]),critic=record(policy.critic(obs))))
        trajectories.append(dict(name=case['name'],parameter_count=len(policy.optim.param_groups[0]['params']),steps=steps))
    print(json.dumps(dict(torch=torch.__version__,trajectories=trajectories),allow_nan=False))


if __name__ == '__main__':
    main()
