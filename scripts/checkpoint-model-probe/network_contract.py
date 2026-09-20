"""Generate reproducible forward/gradient fixtures from unchanged Qlib model bodies.

Uses the existing isolated Torch oracle. Output is JSON on stdout; does not edit
Qlib or load pickle files. Rust tests consume this generated numerical contract.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys
import types
import typing

import gym
import torch
from tianshou.data import Batch


def source_module(path, replacements):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom)
        and (node.level or (node.module or "").startswith("qlib")))]
    module = types.ModuleType(path.stem)
    module.__dict__.update(replacements)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


def record(value):
    return {"shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch."),
            "values": value.detach().to(torch.float64).reshape(-1).tolist()}


def weights_and_aliases(model):
    state = model.state_dict()
    groups = {}
    for name, tensor in state.items():
        key = (tensor.untyped_storage()._cdata, tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()))
        groups.setdefault(key, []).append(name)
    return {name: record(value) for name, value in state.items()}, [v for v in groups.values() if len(v) > 1]


def collect(name, kind, config, model, inputs, outputs, loss):
    loss.backward()
    weights, aliases = weights_and_aliases(model)
    gradients = {name: None if value.grad is None else record(value.grad)
                 for name, value in model.named_parameters(remove_duplicate=False)}
    input_gradients = {name: None if value.grad is None else record(value.grad)
                       for name, value in inputs.items()}
    return {"name": name, "kind": kind, "config": config, "weights": weights,
            "aliases": aliases, "inputs": {k: record(v) for k, v in inputs.items()},
            "outputs": {k: record(v) for k, v in outputs.items()},
            "gradients": gradients, "input_gradients": input_gradients}


def initialize(model):
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.uniform_(-0.25, 0.25)
            if "bias" in name:
                parameter.add_(0.15)


def main():
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("D:/code/github/qlib")
    network_path = source / "qlib/rl/order_execution/network.py"
    policy_path = source / "qlib/rl/order_execution/policy.py"
    network = source_module(network_path, {"Literal": typing.Literal, "FullHistoryObs": dict})
    policy = source_module(policy_path, {})
    torch.manual_seed(97)
    torch.set_num_threads(1)
    cases = []
    for kind in ("rnn", "lstm", "gru"):
        for layers in (1, 2):
            config = dict(data_dim=2, hidden_dim=2, output_dim=3, kind=kind, layers=layers)
            extractor = network.Recurrent({"data_processed": gym.spaces.Box(-1., 1., (3, 2))},
                hidden_dim=2, output_dim=3, rnn_type=kind, rnn_num_layers=layers)
            model = torch.nn.ModuleDict({"actor": policy.PPOActor(extractor, 3), "critic": policy.PPOCritic(extractor)})
            initialize(model)
            inputs = dict(data_processed=(torch.arange(12, dtype=torch.float32).reshape(2, 3, 2) / 10).requires_grad_(),
                cur_tick=torch.tensor([0, -1]), cur_step=torch.tensor([0, -1]),
                position_history=torch.tensor([[8., 4., 0.], [6., 3., 1.]], requires_grad=True),
                target=torch.tensor([8., 6.], requires_grad=True), num_step=torch.tensor([3, 3]),
                acquiring=torch.tensor([0, 1]))
            obs = Batch(**inputs)
            state = object()
            actor, returned = model["actor"](obs, state)
            assert returned is state
            critic = model["critic"](obs)
            features = extractor(obs)
            sources, public = extractor._source_features(obs, torch.device("cpu"))
            outputs = dict(actor=actor, critic=critic, features=features, public=public,
                           public_slice=sources[0], private=sources[1], direction=sources[2])
            loss = actor.square().sum() + critic.square().sum() + features.sum() * 0.2 + public.sum() * 0.3
            cases.append(collect(f"{kind}-{layers}", "recurrent", config, model, inputs, outputs, loss))
    for name, dtype, shapes, output_dim in [
        ("normal", torch.float32, ((2, 3, 3), (2, 4, 3), (2, 4, 3)), 2),
        ("double", torch.float64, ((2, 3, 3), (2, 4, 3), (2, 4, 3)), 2),
        ("half", torch.float16, ((2, 3, 3), (2, 4, 3), (2, 4, 3)), 2),
        ("bfloat", torch.bfloat16, ((2, 3, 3), (2, 4, 3), (2, 4, 3)), 2),
        ("batch-broadcast", torch.float32, ((1, 3, 3), (2, 4, 3), (1, 4, 3)), 2),
        ("contract-left-singleton", torch.float32, ((2, 3, 3), (2, 1, 3), (2, 4, 3)), 2),
        ("contract-right-singleton", torch.float32, ((2, 3, 3), (2, 4, 3), (2, 1, 3)), 2),
        ("zero-query", torch.float32, ((2, 0, 3), (2, 4, 3), (2, 4, 3)), 2),
        ("zero-key", torch.float32, ((2, 3, 3), (2, 0, 3), (2, 0, 3)), 2),
        ("zero-batch", torch.float32, ((0, 3, 3), (1, 4, 3), (1, 4, 3)), 2),
        ("zero-output", torch.float32, ((2, 3, 3), (2, 4, 3), (2, 4, 3)), 0),
    ]:
        model = network.Attention(3, output_dim).to(dtype)
        initialize(model)
        inputs = {key: torch.randn(shape, dtype=dtype).requires_grad_() for key, shape in zip(("q", "k", "v"), shapes)}
        output = model(inputs["q"], inputs["k"], inputs["v"])
        projected_q, projected_k, projected_v = model.q_net(inputs["q"]), model.k_net(inputs["k"]), model.v_net(inputs["v"])
        scores = torch.einsum("ijk,ilk->ijl", projected_q, projected_k)
        probabilities = torch.softmax(scores, dim=-1)
        intermediate = dict(q_projected=projected_q, k_projected=projected_k, v_projected=projected_v, scores=scores, probabilities=probabilities)
        staged_output = torch.einsum("ijk,ikl->ijl", probabilities, projected_v)
        stage_gradients = torch.autograd.grad(staged_output.square().sum(), tuple(intermediate.values()))
        cases.append(collect(name, "attention", dict(input_dim=3, output_dim=output_dim), model, inputs,
                             {"attention": output, "q_projected": projected_q, "k_projected": projected_k,
                              "v_projected": projected_v, "scores": scores, "probabilities": probabilities}, output.square().sum()))
        cases[-1]["intermediate_gradients"] = {key: record(value) for key, value in zip(intermediate, stage_gradients)}
    print(json.dumps({"torch": torch.__version__, "source_sha256": {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (network_path, policy_path)}, "cases": cases}))


if __name__ == "__main__":
    main()
