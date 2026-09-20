"""Exercise actual Tianshou synchronous collection, action hooks and failures."""
import hashlib
import json
from pathlib import Path
import gym
import numpy as np
from tianshou.data import Batch, Collector, VectorReplayBuffer
from tianshou.data import collector as source
from tianshou.env import DummyVectorEnv
from tianshou.policy import BasePolicy


def observation(index, tick):
    return dict(data_processed=np.array([[index, tick], [1., 2.]], dtype=np.float32),
                cur_tick=np.int64(tick % 3), cur_step=np.int64(tick % 2),
                position_history=np.ones(2, dtype=np.float32), target=np.float32(1),
                num_step=np.int64(2), acquiring=np.int64(index % 2))


class Policy(BasePolicy):
    def __init__(self, events, fail):
        super().__init__(action_space=gym.spaces.Discrete(32))
        self.events, self.fail = events, fail

    def forward(self, batch, state=None, **kwargs):
        self.events.append(['act', batch.obs.data_processed[:, 0].tolist()])
        if self.fail == 'forward':
            raise RuntimeError('forward')
        return Batch(act=np.zeros(len(batch.obs), dtype=np.int64), state=state)

    def exploration_noise(self, actions, batch):
        self.events.append(['noise'])
        if self.fail == 'noise':
            raise RuntimeError('noise')
        return actions + 1

    def map_action(self, actions):
        self.events.append(['map', actions.tolist()])
        if self.fail == 'map':
            raise RuntimeError('map')
        return actions + 10

    def learn(self, batch, **kwargs):
        raise AssertionError('collect must not learn')


class Env(gym.Env):
    def __init__(self, index, events, fail):
        self.index, self.events, self.fail = index, events, fail
        self.tick, self.resets = 0, 0
        self.action_space = gym.spaces.Discrete(32)
        self.observation_space = gym.spaces.Dict({})

    def reset(self, **kwargs):
        self.resets += 1
        self.events.append(['reset', self.index])
        if self.index == 0 and ((self.fail == 'reset_finished' and self.resets == 2)
                                or (self.fail == 'reset_final' and self.resets == 3)):
            raise RuntimeError(self.fail)
        self.tick = 0
        return observation(self.index, self.tick)

    def step(self, action):
        self.events.append(['step', self.index, int(action)])
        if self.fail == 'step':
            raise RuntimeError('step')
        self.tick += 1
        done = self.tick == (1 if self.index == 0 else 3)
        truncated = 'invalid' if self.fail == 'truncation' else self.index == 1 and done
        return observation(self.index, self.tick), float(self.index + 1), done, {'TimeLimit.truncated': truncated}


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def run(limits, noise, capacity, fail=None):
    events = []
    env = DummyVectorEnv([lambda: Env(0, events, fail), lambda: Env(1, events, fail)])
    buffer = None if capacity is None else VectorReplayBuffer(capacity, 2)
    collector = Collector(Policy(events, fail), env, buffer, exploration_noise=noise)
    output = []
    original_time = source.time.time
    source.time.time = lambda: 100.
    try:
        for kind, count in limits:
            try:
                result = collector.collect(**{'n_episode' if kind == 'episodes' else 'n_step': count})
                error = None
                result = {key: plain(value) for key, value in result.items()}
            except (RuntimeError, TypeError, ValueError, AssertionError) as exc:
                result, error = None, type(exc).__name__
            replay = collector.buffer
            snapshot = dict(metrics=result, error=error, steps=collector.collect_step,
                            episodes=collector.collect_episode, seconds=collector.collect_time,
                            lengths=replay._lengths.tolist(), last=replay.last_index.tolist(),
                            next_write=[child._index for child in replay.buffers],
                            events=list(events))
            if result is not None:
                indices = replay.sample_indices(0)
                stored = replay[indices]
                snapshot['stored'] = dict(indices=indices.tolist(), actions=stored.act.tolist(),
                                          rewards=stored.rew.tolist(), terminated=stored.terminated.tolist(),
                                          truncated=stored.truncated.tolist(), obs=stored.obs.data_processed[:, 0].tolist(),
                                          following=stored.obs_next.data_processed[:, 0].tolist(),
                                          unfinished=replay.unfinished_index().tolist())
            output.append(snapshot)
            if error:
                break
    finally:
        source.time.time = original_time
        env.close()
    return dict(limits=limits, noise=noise, capacity=capacity, fail=fail, output=output)


def main():
    cases = []
    for noise in (False, True):
        for limits in ([('episodes', 3)], [('episodes', 1), ('episodes', 3)],
                       [('steps', 1), ('steps', 3)], [('steps', 6)]):
            cases.append(run(limits, noise, None if not noise else 7))
    for fail in ('forward', 'noise', 'map', 'step', 'reset_finished', 'reset_final', 'truncation'):
        cases.append(run([('episodes', 1)], True, 7, fail))
    for kind, count in (('episodes', 0), ('steps', 0), ('episodes', -1)):
        cases.append(run([(kind, count)], False, None))
    print(json.dumps(dict(source_sha256=hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest(),
                          cases=cases), separators=(',', ':')))


if __name__ == '__main__':
    main()
