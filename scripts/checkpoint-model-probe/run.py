"""Compare unchanged Qlib model states with a CPU Candle candidate.

Requires isolated test-only torch, tianshou, gym, numpy and safetensors.
No source edits, production runtime bridge, or legacy pickle loading occurs.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import types

import gym
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tianshou.data import Batch

NAME_ENCODING_KEY = "core.candle_policy.tensor_names"
NAME_ENCODING_VERSION = "prefix-p-v1"


def read_tensors(path):
    with safe_open(str(path), framework="pt") as reader:
        version = (reader.metadata() or {}).get(NAME_ENCODING_KEY)
    tensors = load_file(str(path))
    if version is None:
        return tensors
    assert version == NAME_ENCODING_VERSION, version
    assert all(name.startswith("p") for name in tensors)
    return {name[1:]: tensor for name, tensor in tensors.items()}


def write_tensors(state, path, encoded=False, include_layout=False):
    tensors = {("p" + key if encoded else key): value.contiguous().clone()
               for key, value in state.items()}
    metadata = {NAME_ENCODING_KEY: NAME_ENCODING_VERSION} if encoded else None
    if include_layout:
        sources = {}
        entries = []
        for index, (name, value) in enumerate(state.items()):
            identity = (value.untyped_storage()._cdata, value.storage_offset(),
                        tuple(value.shape), tuple(value.stride()), value.dtype)
            source = sources.setdefault(identity, index)
            entries.append([name, source])
        metadata = metadata or {}
        metadata["core.candle_policy.layout"] = json.dumps({"version": 1, "entries": entries})
    save_file(tensors, str(path), metadata=metadata)


def load_source(path, replacements):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    # Only replace imports into Qlib's large application initialization graph.
    # Every class/function body and all actual ML dependencies remain unchanged.
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom)
        and (node.level or (node.module or "").startswith("qlib"))
    )]
    module = types.ModuleType(path.stem)
    module.__dict__.update(replacements)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


def raw(tensor):
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def aliases(state):
    groups = {}
    for name, value in state.items():
        key = (value.untyped_storage()._cdata, value.storage_offset(), tuple(value.shape), tuple(value.stride()), value.dtype)
        groups.setdefault(key, []).append(name)
    return [group for group in groups.values() if len(group) > 1]


def check_case(name, model, forward, binary, root, conflict=False, encoded=False):
    model.eval()
    state = model.state_dict()
    initial = copy.deepcopy(state)
    names = list(state)
    groups = aliases(state)
    payload = copy.deepcopy(state)
    if conflict:
        assert groups
        # Deliberately disagree at shared names. PyTorch registration order decides
        # the final value, even when the incoming mapping order is reversed.
        payload = type(state)((key, value.clone()) for key, value in reversed(list(state.items())))
        for index, key in enumerate(groups[0]):
            payload[key].fill_(index + 1)
        payload._metadata = copy.deepcopy(state._metadata)
    model.load_state_dict(payload)
    expected = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        before = forward() if forward else None
    folder = root / name
    folder.mkdir()
    write_tensors(initial, folder / "initial.safetensors", encoded)
    write_tensors(payload, folder / "input.safetensors", encoded)
    (folder / "manifest.json").write_text(json.dumps({"names": names, "alias_groups": groups,
                                                      "encoded_names": encoded}), encoding="utf-8")
    result = subprocess.run([str(binary), str(folder)], capture_output=True, text=True)
    assert result.returncode == 0, (name, result.stdout, result.stderr)
    assert json.loads((folder / "load-errors.json").read_text()) == []
    for filename in ("snapshot.safetensors", "restored.safetensors"):
        actual = read_tensors(folder / filename)
        assert set(actual) == set(expected), name
        for key in names:
            assert actual[key].dtype == expected[key].dtype, (name, key, "dtype")
            assert actual[key].shape == expected[key].shape, (name, key, "shape")
            assert raw(actual[key]) == raw(expected[key]), (name, key, "bits")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    # Metadata belongs to the model state envelope; SafeTensors doesn't supply it.
    restored = type(state)((key, actual[key]) for key in names)
    restored._metadata = copy.deepcopy(state._metadata)
    model.load_state_dict(restored)
    assert aliases(model.state_dict()) == groups
    if forward:
        with torch.no_grad():
            after = forward()
        assert raw(before) == raw(after), (name, "forward")
    return {"name": name, "tensors": len(names), "alias_groups": groups,
            "metadata": state._metadata, "forward_checked": forward is not None,
            "dtypes": sorted({str(value.dtype) for value in state.values()}),
            "optimizer_entries": len(model.optim.state) if hasattr(model, "optim") else None,
            "tensor_digest": hashlib.sha256(b"".join(raw(expected[key]) for key in names)).hexdigest()}


def load_error_contract(network, binary, root):
    rows = []
    for scenario in ("shape", "missing", "unexpected", "dtype_cast", "noncontiguous"):
        model = network.Attention(3, 2)
        original = copy.deepcopy(model.state_dict())
        names = list(original)
        payload = type(original)((key, torch.full_like(value, 9)) for key, value in reversed(list(original.items())))
        first = names[0]
        if scenario == "shape":
            payload[first] = torch.zeros(1)
        elif scenario == "missing":
            del payload[first]
        elif scenario == "unexpected":
            payload["extra"] = torch.zeros(1)
        elif scenario == "dtype_cast":
            payload[first] = payload[first].double()
        else:
            payload[first] = payload[first].t().contiguous().t()
            assert not payload[first].is_contiguous()
        error = None
        try:
            model.load_state_dict(payload)
        except RuntimeError as failure:
            error = str(failure)
        current = model.state_dict()
        assert bool(error) == (scenario in ("shape", "missing", "unexpected"))
        for key in names:
            expected = original[key] if key == first and scenario in ("shape", "missing") else torch.full_like(current[key], 9)
            assert raw(current[key]) == raw(expected), (scenario, key)
        folder = root / f"load-{scenario}"
        folder.mkdir()
        save_file({key: value.contiguous().clone() for key, value in original.items()}, str(folder / "initial.safetensors"))
        save_file({key: value.contiguous().clone() for key, value in payload.items()}, str(folder / "input.safetensors"))
        (folder / "manifest.json").write_text(json.dumps({"names": names, "alias_groups": []}), encoding="utf-8")
        result = subprocess.run([str(binary), str(folder)], capture_output=True, text=True)
        assert result.returncode == 0, (scenario, result.stderr)
        rust_errors = json.loads((folder / "load-errors.json").read_text())
        assert bool(rust_errors) == bool(error), (scenario, rust_errors, error)
        actual = read_tensors(folder / "snapshot.safetensors")
        for key in names:
            assert actual[key].dtype == current[key].dtype
            assert raw(actual[key]) == raw(current[key]), (scenario, key)
        rows.append({"scenario": scenario, "error": error,
                     "rust_errors": rust_errors,
                     "updated_after_first_invalid_field": names[1:],
                     "target_dtype": str(current[first].dtype)})
    return rows


def dtype_limits(binary, root, runtime):
    rows = []
    for dtype in (torch.bool, torch.int8, torch.uint8, torch.int16, torch.int32,
                  torch.int64, torch.uint16):
        tensor = torch.tensor([0, 1], dtype=dtype)
        folder = root / f"buffer-{dtype}"
        folder.mkdir()
        save_file({"buffer": tensor}, str(folder / "input.safetensors"))
        save_file({"buffer": tensor}, str(folder / "initial.safetensors"))
        (folder / "manifest.json").write_text(json.dumps({"names": ["buffer"], "alias_groups": []}), encoding="utf-8")
        result = subprocess.run([str(binary), str(folder)], capture_output=True, text=True)
        restore_errors = json.loads((folder / "load-errors.json").read_text()) if (folder / "load-errors.json").exists() else []
        if (dtype in (torch.bool, torch.int8)
                or (runtime in ("0.9.1", "production-0.9.1") and dtype == torch.int16)
                or (runtime == "production-0.9.1" and dtype in (torch.int32, torch.uint16))):
            assert result.returncode != 0 or restore_errors, dtype
            rows.append({"dtype": str(dtype), "outcome": "rejected", "error": result.stderr, "restore_errors": restore_errors})
        else:
            assert result.returncode == 0, (dtype, result.stderr)
            actual = read_tensors(folder / "restored.safetensors")["buffer"]
            if dtype == torch.uint16 or (runtime == "0.9.1" and dtype == torch.int32):
                target = torch.uint32 if dtype == torch.uint16 else torch.int64
                assert actual.dtype == target
                assert actual.tolist() == tensor.tolist()
                rows.append({"dtype": str(dtype), "outcome": f"widened_to_{target}_not_lossless_dtype"})
            else:
                assert actual.dtype == dtype and raw(actual) == raw(tensor)
                rows.append({"dtype": str(dtype), "outcome": "exact"})
    return rows


def incoming_dtype_contract(binary, root):
    """Different source types may copy into an existing native destination type."""
    sources = {
        "bool": torch.frombuffer(bytearray([0, 1, 2, 255]), dtype=torch.bool),
        "i8": torch.tensor([-128, -1, 0, 127], dtype=torch.int8),
        "i16": torch.tensor([-32768, -1, 0, 32767], dtype=torch.int16),
        "i32": torch.tensor([-(2**31), -1, 0, 2**31 - 1], dtype=torch.int32),
        "u16": torch.tensor([0, 255, 32768, 65535], dtype=torch.uint16),
        "u8": torch.tensor([0, 1, 128, 255], dtype=torch.uint8),
        "i64": torch.tensor([-(2**63), -1, 0, 2**63 - 1], dtype=torch.int64),
    }
    targets = (torch.uint8, torch.uint32, torch.int64, torch.float16,
               torch.bfloat16, torch.float32, torch.float64)
    rows = []
    for source_name, values in sources.items():
        for dtype in targets:
            for shape in ("vector", "scalar", "empty"):
                incoming = values if shape == "vector" else values[0] if shape == "scalar" else values[:0]
                model = torch.nn.Module()
                model.register_buffer("value", torch.zeros(incoming.shape, dtype=dtype))
                initial = copy.deepcopy(model.state_dict())
                model.load_state_dict({"value": incoming})
                expected = model.state_dict()["value"]
                folder = root / f"cast-{source_name}-{dtype}-{shape}"
                folder.mkdir()
                write_tensors(initial, folder / "initial.safetensors")
                write_tensors({"value": incoming}, folder / "input.safetensors")
                (folder / "manifest.json").write_text(json.dumps({"names": ["value"], "alias_groups": []}), encoding="utf-8")
                result = subprocess.run([str(binary), str(folder)], capture_output=True, text=True)
                assert result.returncode == 0, (folder.name, result.stderr)
                assert json.loads((folder / "load-errors.json").read_text()) == [], folder.name
                for filename in ("snapshot.safetensors", "restored.safetensors"):
                    actual = read_tensors(folder / filename)["value"]
                    assert actual.dtype == expected.dtype and actual.shape == expected.shape, folder.name
                    assert raw(actual) == raw(expected), (folder.name, "bits", actual, expected)
                rows.append({"source": source_name, "target": str(dtype), "shape": shape,
                             "outcome": "exact_target_dtype_and_bits"})
    return rows


def scalar_shape_contract(binary, root):
    """Compare scalar normalization, fatal indexing and ordered partial effects."""
    targets = (torch.uint8, torch.uint32, torch.int64, torch.float16,
               torch.bfloat16, torch.float32, torch.float64)
    scenarios = (
        ("singleton", (1,), ()),
        ("long-vector", (2,), ()),
        ("empty", (0,), ()),
        ("scalar", (), ()),
        ("matrix", (1, 1), ()),
        ("scalar-to-vector", (), (1,)),
        ("matrix-to-vector", (1, 1), (1,)),
        ("vector-to-matrix", (1,), (1, 1)),
        ("missing-before-empty", (0,), ()),
        ("shape-before-empty", (0,), ()),
    )
    rows = []
    for source_dtype in (torch.bool, torch.int16, torch.float64, torch.uint64):
        for target_dtype in targets:
            for scenario, source_shape, target_shape in scenarios:
                # Unsupported U64 values remain a separate limitation; empty
                # indexing must still happen before any source dtype conversion.
                if source_dtype == torch.uint64 and source_shape != (0,):
                    continue
                model = torch.nn.Module()
                model.register_buffer("before", torch.tensor([0.]))
                model.register_buffer("value", torch.full(target_shape, 9, dtype=target_dtype))
                model.register_buffer("after", torch.tensor([0.]))
                initial = copy.deepcopy(model.state_dict())
                count = int(np.prod(source_shape))
                incoming = torch.tensor([3., 8.][:count], dtype=source_dtype).reshape(source_shape)
                payload = {"after": torch.tensor([7.]), "value": incoming,
                           "before": torch.tensor([6.])}
                if scenario == "missing-before-empty":
                    del payload["before"]
                elif scenario == "shape-before-empty":
                    payload["before"] = torch.tensor([[6.]])
                error = None
                try:
                    model.load_state_dict(payload)
                except (RuntimeError, IndexError) as failure:
                    error = {"type": type(failure).__name__, "message": str(failure)}
                fatal = source_shape == (0,)
                succeeds = scenario in ("singleton", "long-vector", "scalar")
                assert (error is None) == succeeds, (scenario, error)
                if error:
                    assert error["type"] == ("IndexError" if fatal else "RuntimeError")
                expected = model.state_dict()
                assert expected["before"].item() == (0 if "before-empty" in scenario else 6)
                assert expected["after"].item() == (0 if fatal else 7)
                folder = root / f"shape-{source_dtype}-{target_dtype}-{scenario}"
                folder.mkdir()
                write_tensors(initial, folder / "initial.safetensors")
                write_tensors(payload, folder / "input.safetensors")
                (folder / "manifest.json").write_text(json.dumps({
                    "names": list(initial), "alias_groups": []}), encoding="utf-8")
                result = subprocess.run([str(binary), str(folder)], capture_output=True, text=True)
                assert result.returncode == 0, (folder.name, result.stderr)
                rust_errors = json.loads((folder / "load-errors.json").read_text())
                assert bool(rust_errors) == bool(error), (folder.name, rust_errors, error)
                if fatal:
                    assert rust_errors == ["index:value:index 0 is out of bounds for dimension 0 with size 0"]
                elif error:
                    assert "copy:value" in rust_errors[0], (folder.name, rust_errors)
                for filename in ("snapshot.safetensors", "restored.safetensors"):
                    actual = read_tensors(folder / filename)
                    assert set(actual) == set(expected), folder.name
                    for name, value in expected.items():
                        assert actual[name].dtype == value.dtype and actual[name].shape == value.shape, (folder.name, name)
                        assert raw(actual[name]) == raw(value), (folder.name, name, "bits")
                rows.append({"scenario": scenario, "source": str(source_dtype),
                             "target": str(target_dtype), "source_shape": source_shape,
                             "target_shape": target_shape, "python_error": error,
                             "rust_errors": rust_errors, "outcome": "exact_partial_state_and_bits"})
    return rows


def check_policy_weights(name, model, payload, set_weight, binary, root):
    """Use Qlib's real retry wrapper and model loader, then compare both mutations."""
    initial = copy.deepcopy(model.state_dict())
    groups = aliases(model.state_dict())
    folder = root / f"weights-{name}"
    folder.mkdir()
    write_tensors(initial, folder / "initial.safetensors", encoded=True)
    write_tensors(payload, folder / "input.safetensors", encoded=True, include_layout=True)
    (folder / "manifest.json").write_text(json.dumps({
        "names": list(initial), "alias_groups": groups, "encoded_names": True,
        "weight_names": list(payload)}), encoding="utf-8")
    metadata = getattr(payload, "_metadata", None)
    error = None
    try:
        set_weight(model, payload)
    except (RuntimeError, IndexError) as failure:
        error = {"type": "runtime" if isinstance(failure, RuntimeError) else "other",
                 "message": str(failure)}
    assert getattr(payload, "_metadata", None) is metadata
    expected = model.state_dict()
    result = subprocess.run([str(binary), str(folder)], capture_output=True, text=True)
    assert result.returncode == 0, (name, result.stderr)
    kind = json.loads((folder / "weight-error-kind.json").read_text())
    assert kind == (error["type"] if error else None), (name, kind, error)
    assert json.loads((folder / "weight-names.json").read_text()) == list(payload), name
    assert json.loads((folder / "weight-aliases.json").read_text()) == aliases(payload), name
    for filename, values in (("converted.safetensors", payload),
                             ("snapshot.safetensors", expected),
                             ("restored.safetensors", expected)):
        actual = read_tensors(folder / filename)
        assert set(actual) == set(values), (name, filename)
        for key, value in values.items():
            assert actual[key].dtype == value.dtype and actual[key].shape == value.shape, (name, filename, key)
            assert raw(actual[key]) == raw(value), (name, filename, key, "bits")
    return {"name": name, "error": error, "converted_keys": list(payload),
            "model_tensors": len(expected), "alias_groups": groups,
            "input_alias_groups": aliases(payload),
            "outcome": "exact_materialized_input_order_aliases_and_model_state"}


def policy_weight_edge_cases(set_weight, binary, root):
    rows = []
    for reverse in (False, True):
        model = torch.nn.Module()
        model.register_parameter("a", torch.nn.Parameter(torch.tensor([0.])))
        model.register_buffer("trigger", torch.tensor([0.]))
        model.add_module("_actor_critic", torch.nn.Module())
        model._actor_critic.register_parameter("a", torch.nn.Parameter(torch.tensor([0.])))
        payload = {"a": torch.tensor([3.]), "_actor_critic.a": torch.tensor([7.])}
        if reverse:
            payload = dict(reversed(list(payload.items())))
        rows.append(check_policy_weights(f"collision-{reverse}", model, payload, set_weight, binary, root))
    for prefix in ("", "_actor_critic."):
        model = torch.nn.Module()
        target = model
        if prefix:
            model.add_module("_actor_critic", torch.nn.Module())
            target = model._actor_critic
        target.register_buffer("value", torch.tensor(9.))
        target.register_buffer("later", torch.tensor([0.]))
        payload = {"value": torch.empty(0), "later": torch.tensor([7.])}
        rows.append(check_policy_weights(f"index-{bool(prefix)}", model, payload, set_weight, binary, root))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("D:/code/github/qlib"))
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--runtime", choices=("0.11.0", "0.9.1", "production-0.9.1"), default="0.11.0")
    args = parser.parse_args()
    import typing
    network_path = args.source / "qlib/rl/order_execution/network.py"
    policy_path = args.source / "qlib/rl/order_execution/policy.py"
    network = load_source(network_path, {"Literal": typing.Literal, "FullHistoryObs": dict})
    policy = load_source(policy_path, {})  # weight_file is None; Trainer is never called.
    torch.manual_seed(17)
    torch.set_num_threads(1)
    obs_space = gym.spaces.Dict({"data_processed": gym.spaces.Box(-np.inf, np.inf, (3, 2))})
    action_space = gym.spaces.Discrete(3)
    obs = Batch(data_processed=torch.arange(12, dtype=torch.float32).reshape(2, 3, 2) / 10,
                cur_step=torch.tensor([0, 2]), cur_tick=torch.tensor([1, 3]),
                position_history=torch.tensor([[8., 5., 1.], [6., 4., 0.]]),
                target=torch.tensor([8., 6.]), num_step=torch.tensor([3, 3]),
                acquiring=torch.tensor([0, 1]))
    rows = []
    weight_rows = []
    with tempfile.TemporaryDirectory(prefix="model-state-probe-") as temporary:
        root = Path(temporary)
        for rnn_type in ("rnn", "lstm", "gru"):
            for layers in (1, 2):
                for family in ("PPO", "DQN", "DQN-target"):
                    extractor = network.Recurrent(obs_space, hidden_dim=4, output_dim=3,
                                                  rnn_type=rnn_type, rnn_num_layers=layers)
                    if family == "PPO":
                        model = policy.PPO(extractor, obs_space, action_space, lr=0.001)
                        forward = lambda: torch.cat((model.actor(obs)[0], model.critic(obs).unsqueeze(1)), 1)
                    else:
                        model = policy.DQN(extractor, obs_space, action_space, lr=0.001,
                                           target_update_freq=2 if family == "DQN-target" else 0)
                        forward = lambda: model.model(obs)[0]
                    # Materialize real Adam moments before collecting the policy
                    # state; do not assume a fresh empty optimizer proves omission.
                    for group in model.optim.param_groups:
                        for parameter in group["params"]:
                            parameter.grad = torch.ones_like(parameter)
                    model.optim.step()
                    model.optim.zero_grad(set_to_none=True)
                    name = f"{family}-{rnn_type}-{layers}"
                    rows.append(check_case(name, model, forward, args.binary, root))
                    if family == "PPO":
                        rows.append(check_case(name + "-alias-conflict", model, forward, args.binary, root, True))
                    if args.runtime == "production-0.9.1":
                        for scenario in ("current", "legacy", "shape-error"):
                            payload = copy.deepcopy(model.state_dict())
                            for value in payload.values():
                                value.fill_(9)
                            if scenario == "legacy":
                                for key in list(payload):
                                    if key.startswith("_actor_critic."):
                                        del payload[key]
                            elif scenario == "shape-error":
                                payload[next(iter(payload))] = torch.zeros(1)
                            weight_rows.append(check_policy_weights(name + "-" + scenario,
                                model, payload, policy.set_weight, args.binary, root))
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            model = network.Attention(3, 2).to(dtype=dtype)
            rows.append(check_case(str(dtype), model, None, args.binary, root))
        if args.runtime == "production-0.9.1":
            model = torch.nn.Module()
            for index, name in enumerate(("__metadata__", "p__metadata__", "p", "模型\0weight")):
                model.register_parameter(name, torch.nn.Parameter(torch.tensor([2. + index, 3. + index])))
            forward = lambda: sum(parameter.square().sum() for parameter in model.parameters())
            rows.append(check_case("reserved-names", model, forward, args.binary, root, encoded=True))
        dtype_results = dtype_limits(args.binary, root, args.runtime)
        error_results = load_error_contract(network, args.binary, root)
        cast_results = incoming_dtype_contract(args.binary, root) if args.runtime == "production-0.9.1" else []
        shape_results = scalar_shape_contract(args.binary, root) if args.runtime == "production-0.9.1" else []
        if args.runtime == "production-0.9.1":
            weight_rows.extend(policy_weight_edge_cases(policy.set_weight, args.binary, root))
    report = {"torch": torch.__version__, "candidate": args.runtime, "cases": rows,
              "load_error_contract": error_results,
              "dtype_limits": dtype_results,
              "incoming_dtype_contract": cast_results,
              "scalar_shape_contract": shape_results,
              "policy_weight_contract": weight_rows,
              "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (network_path, policy_path)}}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Passed {len(rows)} model-state cases; report: {args.report}")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
