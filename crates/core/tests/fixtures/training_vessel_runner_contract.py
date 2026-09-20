"""Execute live Qlib training, evaluation, logging, and collector-guard orchestration."""
import ast
from contextlib import contextmanager
import json
import sys
from types import SimpleNamespace

import numpy as np

def extract(path, names):
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)
    result = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in names:
            node.bases = [ast.Name(id="TrainingVesselBase", ctx=ast.Load())] if node.name == "TrainingVessel" else []
            node.body = [method for method in node.body if isinstance(method, ast.FunctionDef) and method.name in names[node.name]]
            result.append(node)
    return result

body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
body += extract(sys.argv[1], {"TrainingVesselBase": {"log", "log_dict"}, "TrainingVessel": {"train", "validate", "test"}})
body += extract(sys.argv[2], {"FiniteVectorEnv": {"collector_guard"}})
namespace = {"contextmanager": contextmanager, "np": np, "INF": 10**18}
exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), sys.argv[1], "exec"), namespace)

def run(phase="train", fast=None, fail=None, kwargs=None, count=2, second="unchanged"):
    events = []
    def record(stage, *values):
        events.append([stage, *values])
        if fail == stage:
            raise RuntimeError(stage)
        if fail == "stop:" + stage:
            raise StopIteration(stage)
    class Env(namespace["FiniteVectorEnv"]):
        def __init__(self):
            self._collector_guarded = False
            self._logger = [SimpleNamespace(on_env_all_ready=lambda: record("ready"), on_env_all_done=lambda: record("done"))]
        def __len__(self):
            record("len", count)
            return count
    class Trainer:
        current_iter = 0
        reads = 0
        @property
        def fast_dev_run(self):
            self.reads += 1
            value = fast if self.reads == 1 or second == "unchanged" else second
            record("fast_dev", value)
            if fail == "fast_dev_second" and self.reads == 2:
                raise RuntimeError("second trainer read")
            return value
    class Policy:
        def train(self):
            record("mode", "train")
        def eval(self):
            record("mode", "evaluation")
        def update(self, sample_size, buffer, **options):
            assert buffer is buffer_identity[0]
            record("update", sample_size, options)
            return {"shared": 20, "loss": 3}
    buffer_identity = []
    def buffer_factory(size, environments):
        record("buffer", size, environments)
        value = object()
        buffer_identity.append(value)
        return value
    class Collector:
        def __init__(self, policy, env, buffer=None, exploration_noise=False):
            record("collector", buffer is not None, exploration_noise)
            self._buffer = buffer
        @property
        def buffer(self):
            record("buffer_access")
            return self._buffer
        def collect(self, **options):
            record("collect", {key: str(value) if key == "n_step" else value for key, value in options.items()})
            return {"reward": 1, "shared": 2}
    namespace.update(Collector=Collector, VectorReplayBuffer=buffer_factory, _logger=SimpleNamespace(info=lambda message: record("log", message)))
    vessel = namespace["TrainingVessel"]()
    vessel.policy = Policy()
    vessel.trainer = Trainer()
    vessel.buffer_size = 20000
    vessel.episode_per_iter = 1000
    vessel.update_kwargs = kwargs or {}
    env = Env()
    result = None
    error = None
    try:
        result = getattr(vessel, phase)(env)
    except BaseException as caught:
        error = type(caught).__name__
    return {"input": {"phase":phase,"fast":fast,"fail":fail,"kwargs":kwargs or {},"count":count,"second":second}, "events":events,"result":None if result is None else list(result.items()),"error":error,"guarded":env._collector_guarded}

cases = [run(), run(fast=3), run(fast=0), run(fast=-2), run(fast=3,second=None), run(fast=3,fail="fast_dev_second"), run(phase="validate"), run(phase="test"), run(phase="test",count=0)]
for stage in ["mode","ready","buffer","collector","fast_dev","collect","buffer_access","update","log","done"]:
    cases.append(run(fail=stage))
for stage in ["mode","buffer","collector","collect","buffer_access","update"]:
    cases.append(run(fail="stop:"+stage))
for phase in ["validate","test"]:
    for stage in ["mode","ready","collector","collect","log","done"]:
        cases.append(run(phase=phase,fail=stage))
    cases.append(run(phase=phase,fail="stop:collect"))
cases.extend([run(kwargs={"repeat":10,"batch_size":64}),run(kwargs={"sample_size":1}),run(kwargs={"buffer":1})])
print(json.dumps(cases))
