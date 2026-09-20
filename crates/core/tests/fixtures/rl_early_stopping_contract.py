"""Execute Qlib's actual EarlyStopping, with observable snapshot/log failure boundaries."""
import ast
import copy
import json
import math
import sys
from types import SimpleNamespace
import numpy as np

with open(sys.argv[1], encoding="utf-8") as source:
    tree = ast.parse(source.read())
classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in {"Callback", "EarlyStopping"}]
ns = dict(copy=copy, np=np)
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *classes], type_ignores=[])), sys.argv[1], "exec"), ns)


def clean(value):
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return value


def number(value):
    return float(value) if isinstance(value, str) else value


def run(spec):
    events = []
    fail = spec.get("fail")
    class Logger:
        def emit(self, level, msg, *args):
            events.append([level, msg % args if args else msg])
            if fail == level + str(sum(e[0] == level for e in events)):
                raise RuntimeError(fail)
        def info(self, msg, *args): self.emit("info", msg, *args)
        def warning(self, msg, *args): self.emit("warning", msg, *args)
    ns["_logger"] = Logger()
    class Weights(dict):
        def __deepcopy__(self, memo):
            events.append(["copy", self["weights"][:]])
            if fail == "copy" + str(sum(e[0] == "copy" for e in events)):
                raise RuntimeError(fail)
            if spec.get("null_snapshot"): return None
            return {"weights": self["weights"][:]}
    class Vessel:
        weights = [1]
        def state_dict(self):
            events.append(["save", self.weights[:]])
            if fail == "save" + str(sum(e[0] == "save" for e in events)):
                raise RuntimeError(fail)
            return Weights(weights=self.weights)
        def load_state_dict(self, state):
            self.weights[:] = state["weights"]
            events.append(["load", self.weights[:]])
            if fail == "load": raise RuntimeError(fail)
    vessel = Vessel()
    trainer = SimpleNamespace(current_iter=0, should_stop=False, metrics={})
    config = {k: spec[k] for k in ("mode", "min_delta", "patience", "baseline", "restore_best_weights", "monitor") if k in spec}
    for key in ("baseline", "min_delta"):
        if key in config: config[key] = number(config[key])
    if "patience" in config: config["patience"] = int(config["patience"])
    error = None
    cb = None
    snapshots = []
    try:
        cb = ns["EarlyStopping"](**config)
        if not spec.get("fresh"): cb.on_fit_start(trainer, vessel)
        if spec.get("absent"): delattr(cb, spec["absent"])
        if "load_missing" in spec:
            state = dict(wait=8, best=9., best_weights={"weights":[10]}, best_iter=11)
            del state[spec["load_missing"]]
            cb.load_state_dict(state)
        for i, value in enumerate(spec.get("values", [])):
            trainer.current_iter = spec.get("start", 0) + i
            vessel.weights[:] = [100 + i]
            trainer.metrics = {"other": 5., "reward": None} if value is None else {"reward": number(value)}
            if spec.get("missing_metrics"): del trainer.metrics
            if spec.get("missing_iter"): del trainer.current_iter
            cb.on_validate_end(trainer, vessel)
            snapshots.append(dict(state=clean(cb.state_dict()), stopped=trainer.should_stop, weights=vessel.weights[:]))
        if spec.get("reuse"):
            cb.on_fit_start(trainer, vessel)
    except Exception as exc:
        if isinstance(exc, AttributeError): error = "missing:" + exc.name
        elif isinstance(exc, KeyError): error = "missing:" + exc.args[0]
        else: error = str(exc)
    state = None if cb is None else {k: clean(getattr(cb, k, "MISSING")) for k in ("wait", "best", "best_weights", "best_iter")}
    return dict(spec=spec, events=events, state=state, snapshots=snapshots, stopped=trainer.should_stop, weights=vessel.weights, error=error)


specs = [dict(null_snapshot=True, restore_best_weights=True, values=[1., 1.]), dict(values=[1., 2., 2.]), dict(mode="min", values=[3., 2., 2.]),
         dict(mode="bad"), dict(fresh=True), dict(values=[None]), dict(monitor="val/reward", values=[1.]),
         dict(values=[1., 2.], reuse=True), dict(values=[None], fresh=True),
         dict(values=[1.], fresh=True), dict(values=[1.], fresh=True, restore_best_weights=True),
         dict(values=[1.], missing_metrics=True), dict(values=[1.], missing_iter=True)]
for mode in ("min", "max"):
    for patience in (-1, 0, 1, 3, str(10**30)):
        for baseline in (None, 0., 10., "nan"):
            specs.append(dict(mode=mode, patience=patience, baseline=baseline, min_delta=-.5, restore_best_weights=True, values=[1., 1.5, 2., 1., None, "nan", "inf", "-inf"], start=-1))
for delta in ("nan", "inf", "-inf", -0.):
    specs.append(dict(min_delta=delta, restore_best_weights=True, values=[1., 1.]))
for fail in ("save1", "save2", "copy1", "copy2", "load", "info1", "info2", "info3", "warning1"):
    specs.append(dict(restore_best_weights=True, start=1, values=[None] if fail == "warning1" else [1.], fail=fail))
for field in ("wait", "best", "best_weights", "best_iter"):
    specs.append(dict(load_missing=field))
    specs.append(dict(absent=field, restore_best_weights=True, values=["nan"]))
print(json.dumps([run(spec) for spec in specs], allow_nan=False))
