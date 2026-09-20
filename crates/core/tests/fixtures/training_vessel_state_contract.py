"""Execute live Qlib base methods without importing the optional ML runtime."""
import ast
import gc
import json
import sys
import weakref

source = sys.argv[1]
with open(source, encoding="utf-8") as handle:
    tree = ast.parse(handle.read(), filename=source)
base = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TrainingVesselBase")
base.bases = []
base.body = [node for node in base.body if isinstance(node, ast.FunctionDef) and node.name in {"assign_trainer", "state_dict", "load_state_dict"}]
namespace = {"weakref": weakref}
module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), base], type_ignores=[])
exec(compile(ast.fix_missing_locations(module), source, "exec"), namespace)
vessel = namespace["TrainingVesselBase"]()
events = []
payload = {"weights": [1, 2], "optimizer": None}

class Policy:
    fail = None
    def state_dict(self):
        events.append("save")
        if self.fail == "save":
            raise RuntimeError("save failed")
        return payload
    def load_state_dict(self, value):
        events.append(["load", value])
        if self.fail == "load":
            raise RuntimeError("load failed")
        self.loaded = value
        return "ignored policy return value"

vessel.policy = Policy()
checkpoint = vessel.state_dict()
snapshot = {"keys": list(checkpoint), "same_payload": checkpoint["policy"] is payload}
load_result = vessel.load_state_dict({"policy": payload, "extra": 3})
snapshot.update(load_returns_none=load_result is None, same_loaded_payload=vessel.policy.loaded is payload)
errors = []
for operation in ["missing", "save", "load"]:
    vessel.policy.fail = operation
    before = len(events)
    try:
        if operation == "missing":
            vessel.load_state_dict({})
        elif operation == "save":
            vessel.state_dict()
        else:
            vessel.load_state_dict({"policy": payload})
    except Exception as error:
        errors.append([operation, type(error).__name__, len(events) - before])

class Trainer:
    current_iter = -1
    fast_dev_run = None

try:
    vessel.trainer.current_iter
except AttributeError:
    unassigned = True
trainer = Trainer()
reference = weakref.ref(trainer)
vessel.assign_trainer(trainer)
initial_iter = vessel.trainer.current_iter
trainer.current_iter = 10**30
trainer.fast_dev_run = -2
updated = [str(vessel.trainer.current_iter), vessel.trainer.fast_dev_run]
del trainer
gc.collect()
try:
    vessel.trainer.current_iter
except ReferenceError:
    expired = True
replacement = Trainer()
vessel.assign_trainer(replacement)
print(json.dumps({"checkpoint": snapshot, "events": events, "errors": errors, "binding": {"unassigned": unassigned, "initial_iter": initial_iter, "updated": updated, "not_owned": reference() is None, "expired": expired, "replacement_iter": vessel.trainer.current_iter}}))
