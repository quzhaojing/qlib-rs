"""Actual Tianshou vector replay: shared schema, ring state, failures and sampling."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from tianshou.data import Batch, VectorReplayBuffer
from tianshou.data.buffer import manager, vecbuf
from candle_replay_contract import observation
from network_contract import record


def batch(values, dtype):
    obs = Batch.cat([observation(value, dtype) for value in values]) if values else observation(0, dtype)[:0]
    following = Batch.cat([observation(value + 1, dtype) for value in values]) if values else observation(1, dtype)[:0]
    return Batch(obs=obs, obs_next=following, act=torch.tensor([value % 3 for value in values], dtype=torch.int64),
                 rew=np.array([value * .1 - .2 for value in values]),
                 terminated=np.array([value % 4 == 2 for value in values], dtype=bool),
                 truncated=np.array([value % 5 == 3 for value in values], dtype=bool))


def data(value):
    return dict(observation={key: record(tensor) for key, tensor in value.obs.items()},
                next_observation={key: record(tensor) for key, tensor in value.obs_next.items()},
                actions=record(value.act), rewards=value.rew.tolist(),
                terminated=value.terminated.tolist(), truncated=value.truncated.tolist())


def state(buffer):
    try:
        unfinished = buffer.unfinished_index().tolist()
    except AttributeError:
        unfinished = None
    return dict(lengths=buffer._lengths.tolist(), last=buffer.last_index.tolist(),
                next_write=[child._index for child in buffer.buffers],
                unfinished=unfinished, indices=buffer.sample_indices(0).tolist())


def main():
    cases = []
    for total, count, dtype in ((5, 2, torch.float32), (2, 3, torch.float64), (6, 3, torch.float32)):
        buffer = VectorReplayBuffer(total, count)
        steps = []
        schedule = [[0], [count - 1, 0], [0, 0], list(range(count)), [1], [0, 1, 0, 1], [count - 1]]
        for step, ids in enumerate(schedule):
            reset = {3: True, 6: False}.get(step)
            if reset is not None:
                buffer.reset(keep_statistics=reset)
            value = batch([step * 10 + index for index in range(len(ids))], dtype)
            added = buffer.add(value, buffer_ids=ids)
            query = np.arange(buffer.maxsize * 2 + 2)
            steps.append(dict(ids=ids, reset=reset, input=data(value), added=[part.tolist() for part in added],
                              state=state(buffer), stored=data(buffer[np.arange(buffer.maxsize)]),
                              query=query.tolist(), previous=buffer.prev(query).tolist(), following=buffer.next(query).tolist()))
        cases.append(dict(total=total, environments=count, capacity=buffer.maxsize, steps=steps))
    failures = []
    for field in ('observation', 'action', 'invalid_id', 'empty'):
        buffer = VectorReplayBuffer(5, 2)
        if field not in ('invalid_id', 'empty'):
            buffer.add(batch([0], torch.float32), buffer_ids=[0])
        value = batch([] if field == 'empty' else [1, 2], torch.float32)
        value.terminated[:] = False
        value.truncated[:] = False
        ids = [] if field == 'empty' else [0, 9] if field == 'invalid_id' else [0, 1]
        if field == 'observation':
            value.obs.data_processed = value.obs.data_processed.double()
        if field == 'action':
            value.act = value.act.float()
        try:
            buffer.add(value, buffer_ids=ids)
            raise AssertionError('expected source failure')
        except (RuntimeError, IndexError) as error:
            failures.append(dict(field=field, error=type(error).__name__, state=state(buffer)))
    buffer = VectorReplayBuffer(9, 3)
    buffer.add(batch([1, 2, 3], torch.float32), buffer_ids=[0, 0, 2])
    original = np.random.choice
    calls = []
    def choice(population, size, **kwargs):
        result = original(population, size, **kwargs)
        calls.append(dict(population=int(population), size=int(size),
                          probabilities=kwargs.get('p', np.array([])).tolist()))
        return result
    np.random.seed(41)
    np.random.choice = choice
    try:
        draws = buffer.sample_indices(40).tolist()
    finally:
        np.random.choice = original
    print(json.dumps(dict(source_sha256={name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                                        for name, module in [('manager', manager), ('vecbuf', vecbuf)]},
                          cases=cases, failures=failures, sampling=dict(calls=calls, draws=draws)), separators=(',', ':')))


if __name__ == '__main__':
    main()
