"""Real ReplayBuffer state, episode and frame-availability behavior."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from tianshou.data import Batch, ReplayBuffer
from tianshou.data.buffer import base


def main():
    cases = []
    for capacity, stack, available in itertools.product((1, 3, 5), (1, 2, 3, 5), (False, True)):
        buffer = ReplayBuffer(capacity, stack_num=stack, sample_avail=available)
        steps = []
        for index in range(9):
            reward = .1 * index - .2
            terminated, truncated = index in (2, 5), index == 4
            result = buffer.add(Batch(obs=index, obs_next=index+1, act=index % 3,
                                      rew=reward, terminated=terminated, truncated=truncated))
            ptr, rew, length, start = [value.tolist()[0] for value in result]
            steps.append(dict(reward=reward, done=bool(terminated or truncated),
                              added=dict(index=ptr, reward=rew, length=length, start=start),
                              size=len(buffer), next_write=buffer._index, last=buffer.last_index.tolist()[0],
                              unfinished=buffer.unfinished_index().tolist(),
                              previous=buffer.prev(np.arange(capacity)).tolist(),
                              following=buffer.next(np.arange(capacity)).tolist(),
                              available=buffer.sample_indices(0).tolist()))
        cases.append(dict(capacity=capacity, stack=stack, available=available, steps=steps))
    path = Path(base.__file__)
    print(json.dumps(dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), cases=cases), indent=2))


if __name__ == '__main__':
    main()
