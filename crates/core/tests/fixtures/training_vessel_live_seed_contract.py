"""Execute unchanged seed methods with observable boundary collaborators."""
import ast
import copy
import json
import sys
from types import SimpleNamespace

path = sys.argv[1]
tree = ast.parse(open(path, encoding="utf-8").read(), path)
names = ("train_seed_iterator", "val_seed_iterator", "test_seed_iterator")
classes = []
for name in ("TrainingVesselBase", "TrainingVessel"):
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    body = [copy.deepcopy(n) for n in original.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for method in body:
        method.returns = None
        for argument in method.args.args:
            argument.annotation = None
    bases = [] if name == "TrainingVesselBase" else [ast.Name(id="TrainingVesselBase", ctx=ast.Load())]
    classes.append(ast.ClassDef(name=name, bases=bases, keywords=[], body=body, decorator_list=[]))
module = compile(ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[])), path, "exec")


class SeedIteratorNotAvailable(BaseException):
    pass


def run(method, phase, field, failure):
    events = []

    def log(*args):
        events.append("log")
        if failure == "logger":
            raise RuntimeError("logger")

    class Trainer:
        @property
        def fast_dev_run(self):
            events.append("trainer")
            if failure == "trainer":
                raise RuntimeError("trainer")
            return 2

    def subset(name, items, size):
        events.append("subset")
        if failure == "subset":
            raise RuntimeError("subset")
        return items[:size]

    namespace = dict(SeedIteratorNotAvailable=SeedIteratorNotAvailable,
                     _logger=SimpleNamespace(info=log), DataQueue=lambda items, **kwargs: items)
    exec(module, namespace)
    vessel = namespace["TrainingVessel"]()
    setattr(vessel, field, None if failure == "missing" else [1, 2, 3])
    vessel.trainer = Trainer()
    vessel._random_subset = subset
    try:
        items = getattr(vessel, method)()
        ok = True
    except (SeedIteratorNotAvailable, RuntimeError):
        items, ok = None, False
    return dict(phase=phase, case=failure, events=events, ok=ok, items=items)


print(json.dumps([
    run(method, phase, field, failure)
    for method, phase, field in zip(names, ("train", "val", "test"),
                                   ("train_initial_states", "val_initial_states", "test_initial_states"))
    for failure in ("missing", "none", "logger", "trainer", "subset")
]))
