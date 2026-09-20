"""Emit real Torch contracts for the default L2 clipping used by Qlib PPO."""
import json
import math

import torch

from adam_contract import record


def case(name, dtypes, values, limit):
    parameters = [torch.nn.Parameter(torch.zeros(len(value or []), dtype=dtype))
                  for dtype, value in zip(dtypes, values)]
    for parameter, value in zip(parameters, values):
        if value is not None:
            parameter.grad = torch.tensor(value, dtype=parameter.dtype)
    before = [None if p.grad is None else record(p.grad) for p in parameters]
    norm = torch.nn.utils.clip_grad_norm_(parameters, limit)
    return dict(name=name, limit=limit, parameters=[record(p) for p in parameters],
                before=before, norm=record(norm),
                after=[None if p.grad is None else record(p.grad) for p in parameters])


def encoded(value):
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return str(value)


def main():
    cases = []
    for dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
        for limit in (1., 5., 100., 0., -2.):
            cases.append(case(f"{dtype}-{limit}", [dtype] * 4,
                              [[3., -4.], [0.25, -0.125, 0.5], [], None], limit))
        cases.append(case(f"{dtype}-zeros", [dtype], [[0., 0.]], 1.))
        cases.append(case(f"{dtype}-absent", [dtype], [None], 1.))
        cases.append(case(f"{dtype}-empty", [dtype], [[]], 1.))
    for dtypes in ((torch.float32, torch.float64), (torch.float64, torch.float32),
                   (torch.float16, torch.bfloat16), (torch.bfloat16, torch.float32),
                   (torch.float16, torch.float64)):
        cases.append(case(str(dtypes), dtypes, [[3., -4.], [0.25, -0.125, 0.5]], 1.))
    special = []
    for value, limit in ((float("nan"), 1.), (float("inf"), 1.),
                         (-float("inf"), 1.), (3., float("nan")),
                         (3., float("inf")), (3., -float("inf"))):
        parameter = torch.nn.Parameter(torch.zeros(2))
        parameter.grad = torch.tensor([value, 4.])
        norm = torch.nn.utils.clip_grad_norm_([parameter], limit)
        special.append(dict(value=encoded(value), limit=encoded(limit), norm=encoded(norm.item()),
                            after=[encoded(v) for v in parameter.grad.tolist()]))
    print(json.dumps(dict(torch=torch.__version__, cases=cases, special=special), allow_nan=False))


if __name__ == "__main__":
    main()
