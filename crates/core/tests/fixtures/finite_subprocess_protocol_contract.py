import ast
import json
import sys


# Command behavior frozen from the official Tianshou v0.4.10 tagged source:
# https://github.com/thu-ml/tianshou/blob/v0.4.10/tianshou/env/worker/subproc.py
source_path = sys.argv[1]
tree = ast.parse(open(source_path, encoding="utf-8").read(), filename=source_path)


def class_contract(name):
    node = next(
        item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == name
    )
    return [base.id for base in node.bases], [type(item).__name__ for item in node.body]


class Environment:
    def __init__(self):
        self.unwrapped = self
        self.value = 1
        self.reset_seed = None

    def step(self, action):
        return action, 1.0, False, {"action": action}

    def reset(self, **options):
        self.reset_seed = options.get("seed")
        return self.value

    def close(self):
        return "closed"

    def seed(self, seed):
        return [seed]


def dispatch(environment, command, data):
    if command == "step":
        return True, environment.step(data)
    if command == "reset":
        return True, environment.reset(**data)
    if command == "close":
        return True, environment.close()
    if command == "render":
        return True, environment.render(**data) if hasattr(environment, "render") else None
    if command == "seed":
        if hasattr(environment, "seed"):
            return True, environment.seed(data)
        environment.reset(seed=data)
        return True, None
    if command == "getattr":
        return True, getattr(environment, data) if hasattr(environment, data) else None
    if command == "setattr":
        setattr(environment.unwrapped, data["key"], data["value"])
        return False, None
    raise NotImplementedError


subproc_bases, subproc_body = class_contract("FiniteSubprocVectorEnv")
shmem_bases, shmem_body = class_contract("FiniteShmemVectorEnv")
commands = ["step", "reset", "close", "render", "seed", "getattr", "setattr"]
no_reply = []
for command in commands:
    replied, _ = dispatch(
        Environment(),
        command,
        {
            "step": 2,
            "reset": {},
            "close": None,
            "render": {},
            "seed": 3,
            "getattr": "missing",
            "setattr": {"key": "value", "value": 4},
        }[command],
    )
    if not replied:
        no_reply.append(command)

missing_attribute = dispatch(Environment(), "getattr", "missing")[1]


class SeedFallback:
    def __init__(self):
        self.reset_seed = None

    def reset(self, **options):
        self.reset_seed = options.get("seed")
        return 1


fallback = SeedFallback()
seed_fallback = dispatch(fallback, "seed", 5)[1]
try:
    dispatch(Environment(), "unknown", None)
except Exception as error:
    unknown = type(error).__name__

print(
    json.dumps(
        {
            "subproc_bases": subproc_bases,
            "subproc_body": subproc_body,
            "shmem_bases": shmem_bases,
            "shmem_body": shmem_body,
            "commands": commands,
            "no_reply": no_reply,
            "missing_attribute": missing_attribute,
            "seed_fallback": seed_fallback,
            "unknown": unknown,
        }
    )
)
