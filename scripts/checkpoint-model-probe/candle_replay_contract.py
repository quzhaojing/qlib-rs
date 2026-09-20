"""Single-row Torch-backed ReplayBuffer assignment and physical slot reads."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from tianshou.data import Batch, ReplayBuffer
from tianshou.data.buffer import base
from network_contract import record


def observation(index, dtype):
    return Batch(data_processed=torch.tensor([[[index + .13, .27], [.37, index + .47]]], dtype=dtype),
                 cur_tick=torch.tensor([index % 3]), cur_step=torch.tensor([index % 2]),
                 position_history=torch.tensor([[index + 1., .3]], dtype=dtype),
                 target=torch.tensor([index + 2.], dtype=dtype), num_step=torch.tensor([2]),
                 acquiring=torch.tensor([index % 2]))


def main():
    cases = []
    for capacity, first_dtype in ((1, torch.float32), (3, torch.float32), (3, torch.float64)):
        buffer = ReplayBuffer(capacity)
        steps = []
        for index in range(7):
            dtype = first_dtype
            obs, following = observation(index, dtype), observation(index + 1, dtype)
            actions = torch.tensor([index % 3], dtype=torch.int64)
            reward, terminated, truncated = .1 * index - .2, index == 2, index == 4
            added = buffer.add(Batch(obs=obs, obs_next=following, act=actions, rew=np.array([reward]),
                                     terminated=np.array([terminated]), truncated=np.array([truncated])), buffer_ids=[0])
            stored = buffer[np.arange(capacity)]
            steps.append(dict(observation={key:record(value) for key,value in obs.items()},
                              next_observation={key:record(value) for key,value in following.items()},
                              action=record(actions), reward=reward, terminated=terminated, truncated=truncated,
                              added=[value.tolist()[0] for value in added], indices=buffer.sample_indices(0).tolist(),
                              stored_observation={key:record(value) for key,value in stored.obs.items()},
                              stored_next={key:record(value) for key,value in stored.obs_next.items()},
                              stored_actions=record(stored.act), stored_rewards=stored.rew.tolist(),
                              stored_terminated=stored.terminated.tolist(), stored_truncated=stored.truncated.tolist(),
                              unfinished=buffer.unfinished_index().tolist()))
        cases.append(dict(capacity=capacity, steps=steps))
    source = Path(base.__file__)
    errors = []
    for field in ('observation', 'action'):
        buffer = ReplayBuffer(3)
        for index in range(2):
            dtype = torch.float64 if index == 1 and field == 'observation' else torch.float32
            action_dtype = torch.float32 if index == 1 and field == 'action' else torch.int64
            try:
                buffer.add(Batch(obs=observation(index, dtype), obs_next=observation(index + 1, torch.float32),
                                 act=torch.tensor([index], dtype=action_dtype), rew=np.array([1.]),
                                 terminated=np.array([False]), truncated=np.array([False])), buffer_ids=[0])
            except RuntimeError as error:
                errors.append(dict(field=field, error=str(error), size=len(buffer), index=buffer._index,
                                   last=buffer.last_index.tolist()[0], episode_length=buffer._ep_len))
    print(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), cases=cases, errors=errors), separators=(',', ':')))


if __name__ == '__main__':
    main()
