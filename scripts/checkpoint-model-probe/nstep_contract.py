"""Actual Tianshou n-step returns over single/vector physical replay rings."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from tianshou.data import Batch, ReplayBuffer, VectorReplayBuffer
from tianshou.policy import BasePolicy
from tianshou.policy import base
from network_contract import record


def main():
    torch.set_num_threads(1)
    buffers = []
    for count, capacity, additions in [(1, 5, 8), (2, 7, 11)]:
        buffer = ReplayBuffer(capacity) if count == 1 else VectorReplayBuffer(capacity, count)
        inserted = []
        for index in range(additions):
            item = dict(id=index % count, reward=(index - 3) / 4.,
                        terminated=index in [1, 6], truncated=index in [3, 9])
            transition = Batch(obs=np.array([index], dtype=np.float32),
                               obs_next=np.array([index + 1], dtype=np.float32),
                               act=np.int64(0), rew=item['reward'],
                               terminated=item['terminated'], truncated=item['truncated'])
            if count == 1:
                buffer.add(transition)
            else:
                buffer.add(Batch.stack([transition]), buffer_ids=np.array([item['id']]))
            inserted.append(item)
        indices = buffer.sample_indices(0)[::-1].copy()
        cases = []
        for dtype, n, gamma, tail in itertools.product(
                [torch.float32, torch.float64, torch.int64], [1, 3, 5], [0., .7, 1.], [[], [2, 2]]):
            trace = []
            target_record = []

            def target(replay, terminal):
                trace.append(terminal.tolist())
                width = int(np.prod(tail)) if tail else 1
                values = terminal[:, None] + np.arange(width)[None, :] / 4. + .125
                result = torch.tensor(values, dtype=dtype).reshape([len(terminal)] + tail)
                target_record.append(record(result))
                return result

            batch = buffer[indices]
            batch.weight = np.arange(len(indices), dtype=np.float64) + .25
            result = BasePolicy.compute_nstep_return(batch, buffer, indices, target, gamma, n)
            cases.append(dict(dtype=str(dtype).removeprefix('torch.'), steps=n, gamma=gamma,
                              terminal=trace[0], target=target_record[0], returns=record(result.returns),
                              weight=record(result.weight)))
        buffers.append(dict(count=count, capacity=capacity, additions=inserted, indices=indices.tolist(),
                            rewards=buffer.rew.tolist(), done=buffer.done.tolist(),
                            bootstrap=(~buffer.terminated).tolist(), unfinished=buffer.unfinished_index().tolist(),
                            cases=cases))
    print(json.dumps(dict(source_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
                          buffers=buffers), separators=(',', ':')))


if __name__ == '__main__':
    main()
