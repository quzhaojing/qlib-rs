"""Execute actual BasePolicy.update on observable failure-injection adapters."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from tianshou.policy import BasePolicy
from tianshou.policy import base


def run(failure, initial=False):
    events = []
    policy = SimpleNamespace(updating=initial)

    def event(stage, value):
        events.append([stage, policy.updating])
        if failure == stage:
            raise RuntimeError(stage)
        return value

    buffer = SimpleNamespace(sample=lambda size: event('sample', ('batch', [4, 7])))
    policy.process_fn = lambda batch, buffer, indices: event('process', batch)
    policy.learn = lambda batch, **kwargs: event('learn', {'loss': [1.]})
    policy.post_process_fn = lambda batch, buffer, indices: event('post-process', None)
    policy.lr_scheduler = SimpleNamespace(step=lambda: event('scheduler', None))
    try:
        result = BasePolicy.update(policy, 0, None if failure == 'no-buffer' else buffer)
        error = None
    except RuntimeError as exception:
        result, error = None, str(exception)
    return dict(failure=failure, initial=initial, events=events, updating=policy.updating,
                result=result, error=error)


if __name__ == '__main__':
    source = Path(base.__file__)
    print(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                          cases=[run(failure, initial) for initial in (False, True) for failure in
                                 (None, 'sample', 'process', 'learn', 'post-process', 'scheduler', 'no-buffer')])) )
