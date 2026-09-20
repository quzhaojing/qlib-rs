"""Live Trainer initialization and metric-update contracts, independent of Torch imports."""
import ast
import gc
import json
import sys
import weakref
from types import SimpleNamespace

with open(sys.argv[1],encoding="utf-8") as handle:
    tree=ast.parse(handle.read())
trainer=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=="Trainer")
trainer.body=[node for node in trainer.body if isinstance(node,ast.FunctionDef) and node.name in {"__init__","initialize","initialize_iter","_metrics_callback","_min_loglevel"}]
class LogWriter:
    def __init__(self,loglevel=20): self.loglevel=loglevel
class LogBuffer(LogWriter):
    def __init__(self,callback,loglevel=20): super().__init__(loglevel);self.callback=callback
ns={"LogWriter":LogWriter,"LogBuffer":LogBuffer,"LogLevel":SimpleNamespace(PERIODIC=20),"cast":lambda _,v:v,"TrainingVesselBase":object}
body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),trainer]
exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),sys.argv[1],"exec"),ns)

def snapshot(t):
    return {"should_stop":getattr(t,"should_stop",None),"current_iter":None if not hasattr(t,"current_iter") else str(t.current_iter),"current_episode":None if not hasattr(t,"current_episode") else str(t.current_episode),"current_stage":t.current_stage,"metrics":None if not hasattr(t,"metrics") else list(t.metrics.items())}

t=ns["Trainer"]()
initial=snapshot(t)
t.current_iter=12;t.current_episode=8;t.current_stage="val";t.should_stop=True;t.metrics={"old":99.0}
t.initialize();initialized=snapshot(t)
t.current_stage="test";t.current_iter=7;t.should_stop=True;t.initialize_iter();iteration=snapshot(t)

def run(stage="train",episode=True,collect=False,fail=None,have_metrics=True,empty=False):
    t=ns["Trainer"]()
    t.current_stage=stage
    if have_metrics: t.metrics={"old":99.0,"reward":-1.0,"val/reward":-2.0}
    events=[]
    def record(name):
        events.append(name)
        if name==fail: raise RuntimeError(name)
    class Buffer:
        @property
        def global_episode(self): record("global_episode");return 77
        def episode_metrics(self): record("episode_metrics");return {} if empty else {"reward":2.5,"val/nested":4.0}
        def collect_metrics(self): record("collect_metrics");return {} if empty else {"reward":5.0,"val/nested":8.0}
    error=None
    try: t._metrics_callback(episode,collect,Buffer())
    except Exception as exc: error=type(exc).__name__
    return {"input":{"stage":stage,"episode":episode,"collect":collect,"fail":fail,"have_metrics":have_metrics,"empty":empty},"events":events,"state":snapshot(t),"error":error}

cases=[run(stage=s,episode=e,collect=c) for s in ["train","val","test","custom"] for e,c in [(True,False),(False,True),(True,True),(False,False)]]
cases += [run(fail=f) for f in ["global_episode","episode_metrics"]]
cases += [run(episode=False,collect=True,fail="collect_metrics")]
cases += [run(stage=s,episode=e,collect=c,have_metrics=False) for s in ["train","val"] for e,c in [(True,False),(False,True),(False,False)]]
cases += [run(empty=True),run(stage="val",episode=False,collect=True,empty=True)]
levels=[]
for values in [[],[30,40],[-3,0],[20,20],[40,10,30]]:
    t=ns["Trainer"]();t.loggers=[LogWriter(v) for v in values]
    levels.append([values,t._min_loglevel()])
owner=ns["Trainer"]();owner.initialize();owner.initialize_iter()
retained_buffer=owner.loggers[-1];owner_ref=weakref.ref(owner)
del owner
lifetime={"retained":owner_ref() is not None}
retained_buffer.callback(True,False,SimpleNamespace(global_episode=1,episode_metrics=lambda:{"reward":2.0}))
lifetime["after_callback"]=snapshot(owner_ref())
del retained_buffer
gc.collect()
lifetime["released"]=owner_ref() is None
print(json.dumps({"initial":initial,"initialized":initialized,"iteration":iteration,"cases":cases,"levels":levels,"lifetime":lifetime}))
