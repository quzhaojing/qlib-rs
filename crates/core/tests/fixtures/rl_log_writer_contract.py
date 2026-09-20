"""Execute the actual Qlib log writer/buffer, without importing Torch or Tianshou."""
import ast
from collections import defaultdict
from enum import IntEnum
import json
import sys
import numpy as np

with open(sys.argv[1], encoding="utf-8") as handle:
    tree = ast.parse(handle.read())
classes = []
for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name in {"LogLevel", "LogWriter", "LogBuffer"}:
        if node.name == "LogWriter":
            node.bases = []
        classes.append(node)
body = [ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0), *classes]
ns = {"np":np,"defaultdict":defaultdict,"IntEnum":IntEnum}
exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),sys.argv[1],"exec"),ns)

def clean(value):
    if isinstance(value, (float,np.floating)):
        if np.isnan(value): return "nan"
        if np.isposinf(value): return "inf"
        if np.isneginf(value): return "-inf"
        return float(value)
    if isinstance(value, set): return sorted(value)
    if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean(v) for v in value]
    return value

def snapshot(writer):
    return clean(writer.state_dict())

steps = [
    [0, 5.0, False, {"reward":[20,1.0],"x":[20,4.0],"debug":[10,9.0],"int":[20,7],"text":[20,"a"]}],
    [1, 6.0, True, {"reward":[20,10.0],"y":[30,8.0],"text":[20,"b"]}],
    [0, 7.0, True, {"reward":[20,2.0],"x":[20,6.0],"y":[20,2.0],"bool":[20,True]}],
    [0, 8.0, True, {"reward":[20,3.0],"x":[20,8.0]}],
]

def run(fail=None, missing=None):
    events=[]
    class Hooks(ns["LogWriter"]):
        def log_step(self,reward,contents):
            events.append(["step",reward,clean(contents),self.step_count,self.episode_count])
            if fail == "step": raise RuntimeError("step")
        def log_episode(self,length,rewards,contents):
            events.append(["episode",length,clean(rewards),clean(contents),self.step_count,self.episode_count])
            if fail == "episode": raise RuntimeError("episode")
    writer=Hooks()
    writer.on_env_all_ready()
    if missing != "unreset":
        writer.on_env_reset(0,None)
        writer.on_env_reset(1,None)
    if missing in {"episode_rewards","episode_logs"}:
        del getattr(writer,missing)[0]
    error=None
    states=[]
    try:
        for env,reward,done,logs in steps:
            writer.on_env_step(env,None,reward,done,{} if missing == "log" else {"log":logs})
            states.append(snapshot(writer))
    except Exception as exc:
        error=type(exc).__name__
    before=snapshot(writer)
    writer.on_env_all_done()
    writer.clear()
    return {"fail":fail,"missing":missing,"events":events,"states":states,"before_clear":before,"after_clear":snapshot(writer),"error":error}

def buffer_run(fail=None):
    events=[]
    def callback(episode,collect,writer):
        events.append([episode,collect,snapshot(writer),clean(writer.collect_metrics())])
        if fail == ("episode" if episode else "collect"): raise RuntimeError("callback")
    writer=ns["LogBuffer"](callback)
    initial=snapshot(writer)
    try: writer.episode_metrics()
    except Exception as exc: initial_error=type(exc).__name__
    writer.on_env_reset(0,None)
    writer.on_env_reset(1,None)
    error=None
    try:
        for env,reward,done,logs in steps:
            writer.on_env_step(env,None,reward,done,{"log":logs})
        writer.on_env_all_done()
    except Exception as exc: error=type(exc).__name__
    before=snapshot(writer)
    episode=clean(writer.episode_metrics())
    collect=clean(writer.collect_metrics())
    restored=ns["LogBuffer"](lambda *args:None)
    restored.load_state_dict(writer.state_dict())
    writer.on_env_all_ready()
    return {"fail":fail,"initial":initial,"initial_error":initial_error,"events":events,"before_clear":before,"after_clear":snapshot(writer),"episode":episode,"collect":collect,"latest_order":list(episode),"aggregated_order":list(collect),"restored":snapshot(restored),"error":error}

aggregations=[]
for values,name in [([],None),([1.0,3.0],None),([1.0,3.0],"reward"),([1,3.0],"reward"),(["x",2.0],None),([True,False],None),([float("nan"),1.0],None),([float("inf"),-float("inf")],"reward"),([np.float32(1),np.float32(3)],None),([np.float64(1),np.float64(3)],None)]:
    try:
        with np.errstate(invalid="ignore"):
            result=ns["LogWriter"].aggregation(values,name)
        error=None
    except Exception as exc: result=None;error=type(exc).__name__
    aggregations.append({"values":clean(values),"float_mask":[isinstance(v,float) for v in values],"name":name,"result":clean(result),"error":error})
count_edges=[]
for count in [0,-2,10**1000]:
    buffer=ns["LogBuffer"](lambda *args:None)
    buffer.episode_count=count
    buffer._aggregated_metrics={"value":np.float64(1.0)}
    try:
        with np.errstate(divide="ignore",invalid="ignore"):
            result=buffer.collect_metrics()
        error=None
    except Exception as exc: result=None;error=type(exc).__name__
    count_edges.append({"count":str(count),"result":clean(result),"error":error})
print(json.dumps({"steps":[[env,reward,done,list(logs.items())] for env,reward,done,logs in steps],"writer":[run(),run("step"),run("episode"),run(missing="unreset"),run(missing="episode_rewards"),run(missing="episode_logs"),run(missing="log")],"buffer":[buffer_run(),buffer_run("episode"),buffer_run("collect")],"aggregation":aggregations,"count_edges":count_edges}))
