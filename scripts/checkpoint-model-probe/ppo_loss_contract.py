"""Prepared PPO loss and real gradients from installed Tianshou's actual learn()."""
import itertools
import json
import hashlib
from pathlib import Path

import gym
import numpy as np
import torch
import tianshou
from tianshou.data import Batch
from tianshou.policy import PPOPolicy
from tianshou.policy.modelfree import ppo
from adam_contract import record


class Actor(torch.nn.Module):
    def __init__(self, probabilities):
        super().__init__()
        self.probabilities = torch.nn.Parameter(probabilities)

    def forward(self, obs, state=None, info=None):
        return self.probabilities[obs], state


class Critic(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = torch.nn.Parameter(values)

    def forward(self, obs):
        return self.values[obs]


def main():
    cases = []
    torch.set_num_threads(1)
    configurations = [(dtype, norm, value_clip, dual, .25, 1., .01, 'valid')
                      for dtype, norm, value_clip, dual in itertools.product(
                          (torch.float32, torch.float64), (False, True), (False, True), (None, 2.))]
    configurations += [(torch.float32, False, True, None, eps, vf, ent, 'valid')
                       for eps, vf, ent in ((0.,1.,.01),(-.25,1.,.01),(.25,0.,.01),(.25,1.,0.))]
    configurations += [(torch.float32,False,False,None,.25,1.,.01,mode)
                       for mode in ('missing','integer')]
    configurations += [(dtype,True,True,None,.25,1.,.01,'float-actions')
                       for dtype in (torch.float32,torch.float64)]
    for dtype, norm, value_clip, dual, eps, vf, ent, old_mode in configurations:
        actor = Actor(torch.tensor([[.2,.3,.5],[1.,0.,0.],[0.,.25,.75],[.6,.3,.1]],dtype=dtype))
        critic = Critic(torch.tensor([-.4,.5,1.,-.3],dtype=dtype))
        config = dict(eps_clip=eps, value_clip=value_clip, normalize_advantage=norm,
                      value_weight=vf, entropy_weight=ent, dual_clip=dual)
        actions = torch.tensor([2,0,1,2])
        old_log_prob = torch.distributions.Categorical(actor.probabilities).log_prob(actions).detach() \
            - torch.tensor([1.,1.25,.75,1.6],dtype=dtype).log()
        inputs = dict(probabilities=actor.probabilities,values=critic.values,actions=actions,
                      old_log_prob=old_log_prob,advantages=torch.tensor([-2.,-.5,.75,1.75],dtype=dtype),
                      returns=torch.tensor([-.4,.1,1.2,2.],dtype=dtype),
                      old_values=torch.tensor([-.2,.25,1.4,-.7],dtype=dtype))
        if old_mode == 'missing':
            del inputs['old_values']
        elif old_mode == 'integer':
            inputs['old_values'] = torch.full((4,),99,dtype=torch.int64)
        elif old_mode == 'float-actions':
            actions = actions.to(dtype)
            inputs['actions'] = actions
        frozen = {key:record(value) for key,value in inputs.items()}
        optimizer = torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),lr=0.)
        policy = PPOPolicy(actor,critic,optimizer,torch.distributions.Categorical,
            eps_clip=config['eps_clip'], value_clip=value_clip, advantage_normalization=norm,
            dual_clip=dual, vf_coef=vf, ent_coef=ent, reward_normalization=True,
            max_grad_norm=None, action_space=gym.spaces.Discrete(3))
        batch = Batch(obs=np.arange(4),act=actions,logp_old=old_log_prob,
                      adv=inputs['advantages'],returns=inputs['returns'])
        if 'old_values' in inputs:
            batch.v_s = inputs['old_values']
        np.random.seed(0)
        metrics = policy.learn(batch,batch_size=4,repeat=1)
        cases.append(dict(name=f'{dtype}-{norm}-{value_clip}-{dual}-{eps}-{vf}-{ent}-{old_mode}',config=config,inputs=frozen,
                          metrics={key:value[0] for key,value in metrics.items()},
                          gradients=dict(probabilities=record(actor.probabilities.grad),values=record(critic.values.grad))))
    source = Path(ppo.__file__)
    print(json.dumps(dict(torch=torch.__version__,tianshou=tianshou.__version__,
                          source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),cases=cases),allow_nan=False))


if __name__ == '__main__':
    main()
