"""Real ReplayBuffer/BasePolicy episodic-return contracts, including IEEE values."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import tianshou
from tianshou.data import Batch, ReplayBuffer
from tianshou.policy import BasePolicy
from tianshou.policy import base


def numbers(values):
    return [str(float(value)) for value in values]


def run(name, capacity, dtype, gamma, lam, next_present=True, values_present=True, special=None):
    rewards = [.1,-.3,1.2,0.,.6,-.7,.2]
    if special is not None:
        rewards[-1] = special
    buffer = ReplayBuffer(size=capacity)
    for index,reward in enumerate(rewards):
        buffer.add(Batch(obs=index,act=0,rew=reward,terminated=index in (1,5),
                         truncated=index == 3,obs_next=index+1))
    indices = buffer.sample_indices(0)
    if name.startswith('empty'):
        indices = indices[:0]
    if name.startswith('duplicate'):
        indices = np.array([indices[0],indices[0],indices[-1]])
    batch = buffer[indices]
    count = len(indices)
    next_values = np.asarray([.2+i*.13 for i in range(count)],dtype=dtype) if next_present else None
    values = np.asarray([-.1+i*.07 for i in range(count)],dtype=dtype) if values_present else None
    result,adv = BasePolicy.compute_episodic_return(batch,buffer,indices,next_values,values,gamma,lam)
    return dict(name=name,gamma=gamma,gae_lambda=lam,rewards=numbers(batch.rew),
                terminated=batch.terminated.tolist(),truncated=batch.truncated.tolist(),
                bootstrap_valid=(~buffer.terminated[indices]).tolist(),indices=indices.tolist(),
                unfinished_indices=buffer.unfinished_index().tolist(),
                next_values=None if next_values is None else numbers(next_values),
                values=None if values is None else numbers(values),returns=numbers(result),advantages=numbers(adv))


def main():
    cases = []
    for dtype,capacity,next_present,values_present in itertools.product(
            (np.float32,np.float64),(16,4),(False,True),(False,True)):
        configurations = [(1.,1.),(0.,1.)]
        if next_present:
            configurations.append((.9,.95))
        for gamma,lam in configurations:
            name = f'{dtype}-{capacity}-{next_present}-{values_present}-{gamma}-{lam}'
            cases.append(run(name,capacity,dtype,gamma,lam,next_present,values_present))
    cases += [run('empty-next',16,np.float64,.9,.95),run('empty-none',16,np.float64,1.,1.,False,False),
              run('duplicate-indices',4,np.float64,.9,.95),
              run('lambda-close-one',16,np.float64,1.,1.+1e-6,False,False),
              run('negative-discount',16,np.float64,-.5,1.2)]
    for special in (float('nan'),float('inf'),-float('inf')):
        cases.append(run(f'nonfinite-{special}',16,np.float64,0.,1.,special=special))
    print(json.dumps(dict(tianshou=tianshou.__version__,source_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
                          cases=cases),allow_nan=False))


if __name__ == '__main__':
    main()
