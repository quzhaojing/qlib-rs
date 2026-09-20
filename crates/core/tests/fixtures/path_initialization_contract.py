"""Unchanged Qlib normalization methods with observable injected path operations."""
import ast
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

source = Path(sys.argv[1])
tree = ast.parse(source.read_text(encoding="utf-8"))
outer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "QlibConfig")
manager = next(n for n in outer.body if isinstance(n, ast.ClassDef) and n.name == "DataPathManager")
method = next(n for n in outer.body if isinstance(n, ast.FunctionDef) and n.name == "resolve_path")
events, fail_at = [], 0

def operation(name, value):
    events.append([name, value])
    if len(events) == fail_at:
        raise RuntimeError("injected")
    return ("E[" if name == "expand" else "R[") + value + "]"

class TracePath:
    def __init__(self, value):
        if isinstance(value, TracePath):
            self.value = value.value
        elif isinstance(value, str):
            self.value = value
        else:
            raise TypeError("not path-like")
    def __str__(self):
        return self.value
    def expanduser(self):
        return TracePath(operation("expand", self.value))
    def resolve(self):
        return TracePath(operation("resolve", self.value))

config = SimpleNamespace(DEFAULT_FREQ="__DEFAULT_FREQ", LOCAL_URI="local", NFS_URI="nfs")
namespace = {"QlibConfig": config, "Path": TracePath, "re": re}
exec(compile(ast.fix_missing_locations(ast.Module(body=[
    ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), manager, method
], type_ignores=[])), str(source), "exec"), namespace)
config.DataPathManager = namespace["DataPathManager"]
class Configuration(dict):
    DataPathManager = namespace["DataPathManager"]
    resolve_path = namespace["resolve_path"]

def value(descriptor):
    kind, text = descriptor
    return {"text": lambda: text, "path": lambda: TracePath(text),
            "null": lambda: None, "bad": lambda: 3}[kind]()
def setting(descriptor):
    if "scalar" in descriptor:
        return value(descriptor["scalar"])
    return {key: value(item) for key, item in descriptor["map"]}
def encode_value(item):
    if isinstance(item, TracePath): return ["path", str(item)]
    if isinstance(item, str): return ["text", item]
    if item is None: return ["null", None]
    return ["bad", "int"]
def encode(item):
    if isinstance(item, dict): return {"map": [[key, encode_value(v)] for key, v in item.items()]}
    return {"scalar": encode_value(item)}

providers = [
    {"scalar": ["text", "local"]}, {"scalar": ["path", "native"]},
    {"scalar": ["text", "host:/remote"]}, {"scalar": ["null", None]},
    {"scalar": ["bad", "int"]}, {"map": []},
    {"map": [["day", ["text", "first"]], ["1min", ["path", "second"]], ["week", ["text", "host:/remote"]]]},
    {"map": [["day", ["text", "first"]], ["1min", ["null", None]], ["week", ["text", "last"]]]},
    {"map": [["day", ["text", "first"]], ["1min", ["bad", "int"]]]},
    {"map": [["day", ["path", "host:/remote"]]]},
]
mounts = [
    {"scalar": ["null", None]}, {"scalar": ["text", "mount"]},
    {"scalar": ["path", "mountpath"]}, {"scalar": ["bad", "int"]},
    {"map": []}, {"map": [["day", ["text", "m1"]]]},
    {"map": [["week", ["null", None]], ["1min", ["path", "m2"]], ["day", ["text", "m1"]], ["extra", ["bad", "int"]]]},
]
cases = []
for provider in providers:
    for mount in mounts:
        for fail_at in range(10):
            events = []
            instance = Configuration(provider_uri=setting(provider), mount_path=setting(mount))
            try:
                instance.resolve_path()
                error = None
            except Exception as failure:
                error = type(failure).__name__
            cases.append({"provider": provider, "mount": mount, "fail_at": fail_at,
                          "events": events, "error": error,
                          "after_provider": encode(instance["provider_uri"]),
                          "after_mount": encode(instance["mount_path"])})
print(json.dumps(cases))
