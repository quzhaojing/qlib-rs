"""Actual Tianshou Batch.split boundaries used by PPO preprocessing and learn."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import tianshou
from tianshou.data import Batch, batch


def main():
    cases = []
    for length,size,merge in itertools.product((0,1,2,3,4,5,7,8,9),(1,2,3,8),(False,True)):
        source = Batch(position=np.arange(length))
        groups = [part.position.tolist() for part in source.split(size,shuffle=False,merge_last=merge)]
        cases.append(dict(length=length,size=size,merge_last=merge,groups=groups))
    for length in (0,5):
        try:
            list(Batch(position=np.arange(length)).split(0))
        except AssertionError:
            pass
        else:
            raise AssertionError('zero batch size did not fail')
    print(json.dumps(dict(tianshou=tianshou.__version__,source_sha256=hashlib.sha256(Path(batch.__file__).read_bytes()).hexdigest(),cases=cases)))


if __name__ == '__main__':
    main()
