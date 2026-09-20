"""Execute unchanged Qlib Checkpoint methods against deterministic boundary spies.

This freezes callback control flow and partial effects, not Torch file bytes or OS links.
Production filename/codec/filesystem adapters require additional integration tests.
"""
import ast
import json
import sys
from pathlib import PurePosixPath
from types import SimpleNamespace

with open(sys.argv[1], encoding="utf-8") as source:
    tree = ast.parse(source.read())
classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in {"Callback", "Checkpoint"}]
module = ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *classes], type_ignores=[]))


class World:
    def __init__(self):
        self.events = []
        self.files = {}
        self.fail = None
        self.now = 100.0
        self.directories = []

    def event(self, stage, *args):
        self.events.append([stage, *args])
        if self.fail == stage:
            raise RuntimeError(stage)

    def timestamp(self):
        self.event("time")
        return self.now

    def local(self):
        self.event("local_time")
        return SimpleNamespace(strftime=lambda pattern: "20260902123456")

    def save(self, state, path):
        self.files[str(path)] = ["file", "partial"]
        self.event("save", str(path), state)
        self.files[str(path)] = ["file", state]

    def copy(self, source, destination):
        self.event("copy", str(source), str(destination))
        if str(source) not in self.files:
            raise FileNotFoundError(str(source))
        self.files[str(destination)] = self.files[str(source)].copy()


def run(spec):
    world = World()

    class Path:
        def __init__(self, path):
            self.path = str(PurePosixPath(str(path)))

        def __str__(self):
            return self.path

        def __truediv__(self, child):
            return Path(PurePosixPath(self.path) / child)

        def mkdir(self, **kwargs):
            assert kwargs == dict(exist_ok=True, parents=True)
            world.event("mkdir", self.path)
            if self.path not in world.directories:
                world.directories.append(self.path)

        def exists(self):
            world.event("exists", self.path)
            value = world.files.get(self.path)
            if value is None:
                return False
            if value[0] == "link":
                return world.files.get(value[1], [None])[0] == "file"
            return True

        def unlink(self):
            world.event("remove", self.path)
            if world.files.get(self.path, [None])[0] == "directory":
                raise IsADirectoryError(self.path)
            del world.files[self.path]

        def symlink_to(self, target):
            world.event("link", str(target), self.path)
            world.files[self.path] = ["link", str(target)]

    def islink(path):
        world.event("is_link", str(path))
        return world.files.get(str(path), [None])[0] == "link"

    class Filename:
        def format(self, **kwargs):
            world.event("format", str(kwargs["iter"]), kwargs["time"])
            return spec.get("filename", "{iter:03d}.pth").format(**kwargs)

    ns = dict(Path=Path, time=SimpleNamespace(time=world.timestamp), datetime=SimpleNamespace(now=world.local),
              torch=SimpleNamespace(save=world.save), shutil=SimpleNamespace(copyfile=world.copy),
              os=SimpleNamespace(path=SimpleNamespace(islink=islink)))
    exec(compile(module, sys.argv[1], "exec"), ns)
    kwargs = {name: spec[name] for name in ("save_latest", "save_on_fit_end") if name in spec}
    if "every" in spec:
        kwargs["every_n_iters"] = None if spec["every"] is None else int(spec["every"])
    if "interval" in spec:
        kind, value = spec["interval"]
        kwargs["time_interval"] = int(value) if kind == "int" else float(value)
    callback = ns["Checkpoint"]("checkpoints", filename=Filename(), **kwargs)
    assert world.events == [] and world.directories == []
    if "latest" in spec:
        world.files["checkpoints/latest.pth"] = spec["latest"].copy()
    results = []
    for step in spec["steps"]:
        world.events = []
        world.fail = step.get("fail")
        world.now = float(step.get("now", "100"))
        trainer = SimpleNamespace(current_iter=int(step.get("iter", "1")), metrics=step.get("metrics", {}))
        if step.get("missing_iter"):
            del trainer.current_iter
        if step.get("missing_metrics"):
            del trainer.metrics

        def graph():
            world.event("graph", str(trainer.current_iter))
            return "graph-" + str(trainer.current_iter)

        trainer.state_dict = graph
        error = False
        try:
            hook = step.get("hook", "iter_end")
            if hook == "save":
                callback._save_checkpoint(trainer)
            else:
                getattr(callback, "on_" + hook)(trainer, None)
        except Exception:
            error = True
        results.append(dict(error=error, events=world.events.copy(), files=world.files.copy(),
                            directories=world.directories.copy(), name=callback._last_checkpoint_name,
                            iteration=None if callback._last_checkpoint_iter is None else str(callback._last_checkpoint_iter),
                            time=None if callback._last_checkpoint_time is None else repr(callback._last_checkpoint_time)))
    state_before = vars(callback).copy()
    assert callback.state_dict() is None
    callback.load_state_dict({"last_iter": 999})
    assert vars(callback) == state_before
    return dict(spec=spec, results=results)


def step(iteration, **extra):
    return dict(iter=str(iteration), **extra)


specs = [
    dict(steps=[]),
    dict(steps=[step(0), step(1), step(2, hook="fit_end"), step(2, hook="fit_end"), step(0, hook="fit_start"), step(0, hook="fit_end")]),
]
for every in [None, "0", "1", "2", "3", "-2", str(10**100)]:
    for latest in [None, "", "link", "copy", "unknown"]:
        specs.append(dict(every=every, save_latest=latest, steps=[step(i) for i in [0, 1, 2, -1, 2]] + [step(2, hook="fit_end")]))
for interval in [["int", "10"], ["int", "0"], ["int", "-1"], ["int", str(2**53+1)], ["int", str(10**1000)], ["float", "nan"], ["float", "inf"], ["float", "-inf"], ["float", "1.5"]]:
    specs.append(dict(interval=interval, save_latest=None, steps=[step(i, now=now) for i, now in enumerate(["100", "109", "110", "99", "111.5", "9007199254741102", "inf", "-inf", "nan"])]))
for failure in ["mkdir", "local_time", "format", "time", "graph", "save", "exists", "is_link", "remove", "link", "copy"]:
    specs.append(dict(every="1", save_latest="copy" if failure == "copy" else "link", latest=["link", "absent"],
                      steps=[step(1, fail=failure), step(1, hook="fit_end"), step(2)]))
for latest in [["file", "old"], ["link", "absent"], ["directory", ""]]:
    for mode in [None, "", "unknown", "copy", "link"]:
        specs.append(dict(every="1", save_latest=mode, latest=latest, steps=[step(1)]))
for missing in ["missing_iter", "missing_metrics"]:
    for hook in ["fit_start", "test_end", "iter_end", "fit_end", "save"]:
        specs.append(dict(every="1", steps=[step(1, hook=hook, **{missing: True})]))
for reserved in ["iter", "time"]:
    specs.append(dict(every="1", steps=[step(1, metrics={reserved: 99})]))
for filename in ["latest.pth", "fixed.pth", "{iter}.pth", "{missing}", "{", "/absolute/{iter:03d}.pth", "sub/{iter:03d}.pth"]:
    specs.append(dict(every="1", save_latest="copy", filename=filename, steps=[step(1), step(2)]))
specs.extend([
    dict(every="1", interval=["int", "10"], steps=[step(1), step(2, fail="time"), step(3, now="99")]),
    dict(every="0", interval=["int", "0"], steps=[step(1)]),
    dict(save_on_fit_end=False, steps=[step(1, hook="fit_end", missing_iter=True)]),
    dict(every="1", steps=[step(10**100), step(-10**100)]),
])
print(json.dumps([run(spec) for spec in specs], ensure_ascii=True, allow_nan=False))
