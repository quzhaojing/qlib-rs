"""Execute Qlib's actual fit/test driver and context manager, without importing Torch."""
import ast
import gc
import json
import sys
import weakref
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime
from types import SimpleNamespace

with open(sys.argv[1], encoding="utf-8") as source:
    tree = ast.parse(source.read())
trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Trainer")
methods = {"fit", "test", "initialize", "initialize_iter", "_call_callback_hooks"}
trainer.body = [n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name in methods]
wrap = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_wrap_context")
ns = dict(AbstractContextManager=AbstractContextManager, contextmanager=contextmanager, datetime=datetime)
body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), trainer, wrap]
exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), sys.argv[1], "exec"), ns)


def run(spec):
    t = ns["Trainer"]()
    t.current_stage = "old"
    t.max_iters = spec.get("max", 2)
    t.val_every_n_iters = spec.get("val", 1)
    t.metrics = {"old": 9.0}
    events = []
    envs = []

    def live():
        # Exception traceback cycles have GC-dependent lifetime. Collect unreachable cycles
        # before observing owners; do not conflate them with the live vector_env binding.
        gc.collect()
        return sum(e() is not None for e in envs)

    def snapshot():
        return dict(stage=t.current_stage, iteration=str(t.current_iter) if hasattr(t, "current_iter") else None,
                    stop=getattr(t, "should_stop", None), metrics=list(t.metrics.items()))

    def record(name, extra=None):
        events.append([name, snapshot(), extra])
        if spec.get("fail") == name or spec.get("fail_exit") == name:
            raise RuntimeError(name)

    class Env:
        pass

    class Seeds(AbstractContextManager):
        def __init__(self, phase):
            self.phase = phase

        def __enter__(self):
            record(self.phase + ".enter")
            return self.phase

        def __exit__(self, kind, value, tb):
            record(self.phase + ".exit", dict(error=str(value) if value else None,
                                               live=live()))
            return spec.get("suppress", False)

    class Vessel:
        def assign_trainer(self, trainer):
            assert trainer is t and trainer.vessel is self
            record("assign")

        def seeds(self, phase):
            record(phase + ".seeds")
            return phase if spec.get("plain", False) else Seeds(phase)

        def train_seed_iterator(self):
            return self.seeds("train")

        def val_seed_iterator(self):
            return self.seeds("val")

        def test_seed_iterator(self):
            return self.seeds("test")

        def execute(self, phase, env):
            assert isinstance(env, Env)
            record(phase + ".run")
            t.metrics[("val/" if t.current_stage == "val" else "") + "reward"] = 3.0

        def train(self, env):
            self.execute("train", env)

        def validate(self, env):
            self.execute("val", env)

        def test(self, env):
            self.execute("test", env)

    def environment(iterator):
        record(iterator + ".env", dict(live=live()))
        env = Env()
        envs.append(weakref.ref(env))
        return env

    class Callback:
        def __init__(self, index):
            self.index = index

        def __getattr__(self, hook):
            def call(trainer, vessel):
                assert trainer is t and vessel is t.vessel
                record(f"{hook}.{self.index}")
                if self.index == 0:
                    if spec.get("clear") == hook:
                        delattr(t, spec["field"])
                    if spec.get("stop") == hook:
                        t.should_stop = True
                    if spec.get("extend") and hook == "on_iter_end" and t.current_iter == 1:
                        t.should_stop = False
                        t.max_iters = 2
                    if spec.get("change_val") and hook == "on_train_end":
                        t.val_every_n_iters = 2
                    if spec.get("max") is None and hook == "on_iter_end" and t.current_iter >= 2:
                        t.should_stop = True
            return call

    t.callbacks = [Callback(i) for i in range(spec.get("callbacks", 2))]
    t.venv_from_iterator = environment

    def load(_):
        record("restore")
        t.initialize()
        t.current_iter = int(spec.get("resume_iter", 1))
        t.current_stage = "val"
        t.should_stop = spec.get("resume_stop", False)

    t.load_state_dict = load

    def log(message, *args):
        if "Resuming" not in message:
            record("progress")

    ns["_logger"] = SimpleNamespace(info=log)
    ns["torch"] = SimpleNamespace(load=lambda *args, **kwargs: {})
    error = None
    try:
        if spec.get("test"):
            t.test(Vessel())
        else:
            t.fit(Vessel(), "resume" if spec.get("resume") else None)
    except Exception as exc:
        if isinstance(exc, AttributeError):
            error = "missing:" + exc.name
        else:
            error = "zero_validation_interval" if isinstance(exc, ZeroDivisionError) else str(exc)
    gc.collect()
    return dict(spec=spec, events=events, state=snapshot(), error=error)


specs = [{}, {"max": 0}, {"max": -1}, {"max": 1, "val": None}, {"max": 3, "val": 2},
         {"max": 2, "val": -2}, {"val": 0}, {"max": None}, {"callbacks": 0},
         {"plain": True}, {"test": True}, {"test": True, "plain": True},
         {"resume": True}, {"resume": True, "resume_stop": True},
         {"resume": True, "resume_iter": "1000000000000000000000000000000"},
         {"max": 1, "extend": True}, {"change_val": True}]
specs += [{"stop": h} for h in ["on_fit_start", "on_iter_start", "on_train_start", "on_train_end", "on_validate_end", "on_iter_end"]]
specs += [{"fail": f} for f in ["assign", "progress", "on_fit_start.0", "on_fit_start.1", "on_iter_start.0",
          "on_train_start.0", "train.seeds", "train.enter", "train.env", "train.run", "train.exit",
          "on_train_end.0", "on_validate_start.0", "val.seeds", "val.enter", "val.env", "val.run", "val.exit",
          "on_validate_end.0", "on_iter_end.0", "on_fit_end.0"]]
specs += [{"test": True, "fail": f} for f in ["on_test_start.0", "test.seeds", "test.enter", "test.env", "test.run", "test.exit", "on_test_end.0"]]
specs += [{"resume": True, "fail": "restore"}]
specs += [{"fail": f, "suppress": True} for f in ["train.env", "train.run", "val.env", "val.run"]]
specs += [{"test": True, "fail": "test.run", "suppress": True}, {"suppress": True}, {"fail": "train.run", "plain": True}]
specs += [{"fail": "train.run", "fail_exit": "train.exit"}, {"fail": "val.env", "fail_exit": "val.exit"}]
specs += [{"clear": "on_fit_start", "field": "current_iter"}, {"clear": "on_fit_start", "field": "should_stop"},
          {"clear": "on_train_end", "field": "current_iter"}, {"clear": "on_train_end", "field": "current_iter", "val": None}]
specs += [{"test": True, "fail": "assign"}, {"fail": "train.run", "fail_exit": "val.env", "suppress": True}]
print(json.dumps([run(s) for s in specs]))
