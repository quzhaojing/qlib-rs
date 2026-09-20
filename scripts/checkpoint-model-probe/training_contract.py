"""Real Qlib network forward/backward/Adam trajectories; not a full PPO trainer."""
import argparse
import json
from pathlib import Path
import typing

import gym
import torch
from tianshou.data import Batch
from network_contract import source_module, record


def main(clip_gradients=False):
    source = Path("D:/code/github/qlib/qlib/rl/order_execution")
    network = source_module(source / "network.py", {"Literal": typing.Literal, "FullHistoryObs": dict})
    policy = source_module(source / "policy.py", {})
    fixture = json.loads(Path("crates/core/tests/fixtures/rl_candle_network.json").read_text())
    torch.set_num_threads(1)
    trajectories = []
    for case in fixture["cases"]:
        if case["kind"] != "recurrent":
            continue
        config = case["config"]
        extractor = network.Recurrent({"data_processed": gym.spaces.Box(-1.,1.,(3,2))},
            hidden_dim=config["hidden_dim"],output_dim=config["output_dim"],
            rnn_type=config["kind"],rnn_num_layers=config["layers"])
        model = torch.nn.ModuleDict({"actor":policy.PPOActor(extractor,3),"critic":policy.PPOCritic(extractor)})
        model.load_state_dict({name:torch.tensor(item["values"],dtype=torch.float32).reshape(item["shape"])
                               for name,item in case["weights"].items()})
        obs = Batch(**{name:torch.tensor(item["values"],dtype=getattr(torch,item["dtype"])).reshape(item["shape"])
                       for name,item in case["inputs"].items()})
        # Invoke Qlib's real helper rather than assuming duplicated parameters are harmless.
        parameters = list(policy.chain_dedup(model["actor"].parameters(),model["critic"].parameters()))
        optimizer = torch.optim.Adam(parameters,lr=0.003,weight_decay=0.1)
        steps = []
        for index,lr in enumerate((0.003,0.,0.002,0.004)):
            optimizer.param_groups[0]["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            actor = model["actor"](obs)[0]
            critic = model["critic"](obs)
            if index == 1:
                loss = actor.square().sum()
            elif index == 2:
                loss = critic.square().sum()
            else:
                loss = actor.square().sum() + critic.square().sum() + extractor(obs).sum()*0.2
            loss.backward()
            limit = (0.1, None, 100., 0.2)[index] if clip_gradients else None
            norm = None if limit is None else record(torch.nn.utils.clip_grad_norm_(parameters, limit))
            optimizer.step()
            steps.append(dict(learning_rate=lr,initialized=len(optimizer.state),
                max_grad_norm=limit,gradient_norm=norm,
                weights={name:record(tensor) for name,tensor in model.state_dict().items()},
                actor=record(model["actor"](obs)[0]),critic=record(model["critic"](obs))))
        trajectories.append(dict(name=case["name"],parameter_count=len(parameters),steps=steps))
    print(json.dumps(dict(torch=torch.__version__,trajectories=trajectories)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-gradients", action="store_true")
    main(parser.parse_args().clip_gradients)
