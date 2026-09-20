"""Characterize actual Tianshou mask dtype arithmetic, without rewriting source."""
import hashlib
import json
from pathlib import Path

import torch
from tianshou.policy.modelfree import dqn

from dqn_contract import policy


def record(tensor):
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype).removeprefix("torch."),
                values=[str(value) for value in tensor.reshape(-1).tolist()])


def main():
    torch.set_num_threads(1)
    cases = []
    masks = {
        "uint8": [0, 1, 2, 127, 128, 255],
        "uint32": [0, 1, 2, 2**31, 2**32 - 2, 2**32 - 1],
        "int64": [0, 1, 2, -(2**63), -(2**63) + 1, 2**63 - 1],
        "float16": [0.0003, 0.9995, 1.001, -0.0, -65504, 65504],
        "bfloat16": [0.003, 0.996, 1.008, -0.0, -1000, 1000],
        "float32": [0.00000003, 0.99999994, 1.0000001, -0.0, -1e10, 1e10],
        "float64": [0.0000000000000001, 0.9999999999999999, 1.0000000000000002, -0.0, -1e20, 1e20],
    }
    for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
        model = policy(dtype)
        for name, values in masks.items():
            for layout in ["row", "column", "scalar"]:
                tensor = torch.tensor(values, dtype=getattr(torch, name))
                if layout == "row":
                    tensor = tensor.reshape(1, 6)
                elif layout == "column":
                    tensor = tensor.reshape(6, 1)
                else:
                    tensor = tensor[2]
                # NumPy owns subtraction for normal replay masks. BF16 has no
                # NumPy representation; Tianshou also accepts a Torch mask.
                mask = tensor if name == "bfloat16" else tensor.numpy()
                logits = (torch.arange(36, dtype=dtype).reshape(6, 6) / 4)
                q = model.compute_q_value(logits, mask)
                cases.append(dict(layout=layout, mask=record(tensor), logits=record(logits),
                                  q=record(q), actions=q.argmax(1).tolist()))
        # Exhaust every unsigned byte, including values whose 1-mask wraps.
        tensor = torch.arange(256, dtype=torch.uint8).reshape(1, 256)
        logits = torch.arange(256, dtype=dtype).reshape(1, 256) / 4
        q = model.compute_q_value(logits, tensor.numpy())
        cases.append(dict(layout="all_bytes", mask=record(tensor), logits=record(logits),
                          q=record(q), actions=q.argmax(1).tolist()))
    print(json.dumps(dict(source_sha256=hashlib.sha256(Path(dqn.__file__).read_bytes()).hexdigest(),
                          cases=cases), allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
