"""Emit dense Adam update contracts from the installed real Torch CPU oracle."""
import json
import torch


def record(tensor):
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype).removeprefix("torch."),
                values=tensor.detach().to(torch.float64).reshape(-1).tolist())


def main():
    cases = []
    for dtype in (torch.float32, torch.float64):
        for decay in (0., 0.2):
            for shape in ((3,), (), (0,)):
                parameters = [torch.nn.Parameter(torch.full(shape,value,dtype=dtype)) for value in (1.5,-0.7)]
                optimizer = torch.optim.Adam(parameters,lr=0.003,weight_decay=decay)
                initial = [record(parameter) for parameter in parameters]
                steps = []
                for index, values in enumerate(((1.,None),(None,None),(None,-2.),(0.,0.),(3.,None),(-1.,4.))):
                    if index == 2:
                        optimizer.param_groups[0]["lr"] = 0.009
                    if index == 4:
                        optimizer.param_groups[0]["lr"] = 0.
                    if index == 5:
                        optimizer.param_groups[0]["lr"] = 0.002
                    gradients = []
                    for parameter, value in zip(parameters,values):
                        parameter.grad = None if value is None else torch.full_like(parameter,value)
                        gradients.append(None if parameter.grad is None else record(parameter.grad))
                    optimizer.step()
                    steps.append(dict(learning_rate=optimizer.param_groups[0]["lr"], gradients=gradients,
                                      parameters=[record(parameter) for parameter in parameters],
                                      initialized=len(optimizer.state)))
                cases.append(dict(name=f"{dtype}-{decay}-{shape}",decay=decay,initial=initial,steps=steps))
    print(json.dumps(dict(torch=torch.__version__,cases=cases)))


if __name__ == "__main__":
    main()
