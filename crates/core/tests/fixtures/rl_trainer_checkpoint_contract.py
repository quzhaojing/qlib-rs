"""Qlib's real Trainer checkpoint traversal and naming without Torch dependencies."""
import ast
import collections
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    tree = ast.parse(source.read())
trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Trainer")
methods = {"state_dict", "load_state_dict", "named_callbacks", "named_loggers"}
trainer.body = [n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name in methods]
named = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_named_collection")
ns = dict(collections=collections)
body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), trainer, named]
exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), sys.argv[1], "exec"), ns)


def names(items):
    objects = [type(name, (), {})() for name in items]
    return [[key, objects.index(value)] for key, value in ns["_named_collection"](objects).items()]


name_cases = [[], ["Callback"], ["Foo", "Foo", "Foo"], ["Foo", "Foo1", "Foo", "FOO1", "Foo"],
              ["", "", "1", ""], ["ΟΣ", "ΟΣ", "İ", "AΣ", "Straße", "ẞ", "Callback"],
              ["A", "a", "A1", "A", "a1", "a2"]]


def run(spec):
    events = []
    components = []
    t = ns["Trainer"]()
    t.should_stop = False
    t.current_iter = 1
    t.current_episode = 2
    t.current_stage = "train"
    t.metrics = {"old": 9.0}

    def create(name, identity):
        def save(self):
            events.append("save:" + identity)
            self.value += 1
            if spec.get("fail") == events[-1]:
                raise RuntimeError(events[-1])
            return self.value

        def load(self, value):
            events.append("load:" + identity)
            self.value = value
            if spec.get("fail") == events[-1]:
                raise RuntimeError(events[-1])
        obj = type(name, (), {"state_dict": save, "load_state_dict": load})()
        obj.value = 0
        components.append((identity, obj))
        return obj

    t.vessel = create("Vessel", "vessel")
    t.callbacks = [] if spec.get("empty") else [create(n, f"c{i}") for i, n in enumerate(["Foo", "Foo1", "Foo", "Foo"])]
    t.loggers = [] if spec.get("empty") else [create(n, f"l{i}") for i, n in enumerate(["Log", "Log"])]
    checkpoint = dict(vessel=10, callbacks=dict(foo=20, foo1=22, foo2=23, unused=29),
                      loggers=dict(log=30, log1=31, unused=39), should_stop=True,
                      current_iter=12, current_episode=34, current_stage="val", metrics={"score": 5.0})
    if spec.get("missing"):
        path = spec["missing"].split(".")
        target = checkpoint
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]
    if spec.get("absent"):
        delattr(t, spec["absent"])
    if spec.get("empty"):
        checkpoint.pop("callbacks", None)
        checkpoint.pop("loggers", None)
    output = error = None
    try:
        if spec.get("load"):
            t.load_state_dict(checkpoint)
        else:
            output = t.state_dict()
    except Exception as exc:
        if isinstance(exc, KeyError):
            error = "missing:" + str(exc.args[0])
        elif isinstance(exc, AttributeError):
            error = "uninitialized:" + exc.name
        else:
            error = str(exc)
    state = {name: getattr(t, name, None) for name in ["should_stop", "current_iter", "current_episode", "current_stage", "metrics"]}
    return dict(spec=spec, events=events, components=[[key, obj.value] for key, obj in components], output=output, error=error, state=state)


specs = [{}, {"load": True}, {"empty": True}, {"load": True, "empty": True}]
specs += [{"fail": f"save:{name}"} for name in ["vessel", "c0", "c2", "c3", "l0", "l1"]]
specs += [{"load": True, "fail": f"load:{name}"} for name in ["vessel", "c0", "c2", "c3", "l0", "l1"]]
specs += [{"load": True, "missing": name} for name in ["vessel", "callbacks", "callbacks.foo", "callbacks.foo1", "callbacks.foo2",
          "loggers", "loggers.log", "loggers.log1", "should_stop", "current_iter", "current_episode", "current_stage", "metrics"]]
specs += [{"absent": name} for name in ["should_stop", "current_iter", "current_episode", "metrics"]]
print(json.dumps(dict(names=[dict(input=case, output=names(case)) for case in name_cases], cases=[run(spec) for spec in specs])))
