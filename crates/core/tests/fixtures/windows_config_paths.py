"""Execute original Qlib configuration methods with real pathlib and filesystem."""
import ast
import contextlib
import io
import json
from pathlib import Path
import re
import runpy
import sys
from types import SimpleNamespace

source = Path(sys.argv[1])
base = Path(sys.argv[2])
saved_argv = sys.argv
sys.argv = ["read_link.py", str(base)]
with contextlib.redirect_stdout(io.StringIO()):
    fixture = runpy.run_path(str(Path(__file__).parents[3] / "path/tests/fixtures/read_link.py"))
sys.argv = saved_argv
units = fixture["units"]
tree = ast.parse(source.read_text(encoding="utf-8"))
outer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "QlibConfig")
manager = next(n for n in outer.body if isinstance(n, ast.ClassDef) and n.name == "DataPathManager")
method = next(n for n in outer.body if isinstance(n, ast.FunctionDef) and n.name == "resolve_path")
config = SimpleNamespace(DEFAULT_FREQ="__DEFAULT_FREQ", LOCAL_URI="local", NFS_URI="nfs")
namespace = {"QlibConfig":config, "Path":Path, "re":re}
exec(compile(ast.fix_missing_locations(ast.Module(body=[
    ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), manager, method
], type_ignores=[])), str(source), "exec"), namespace)
config.DataPathManager = namespace["DataPathManager"]
class Configuration(dict):
    DataPathManager = namespace["DataPathManager"]
    resolve_path = namespace["resolve_path"]

def encode_value(value):
    if isinstance(value, Path): return ["path", units(str(value))]
    if isinstance(value, str): return ["text", units(value)]
    if value is None: return ["null", None]
    return ["bad", "int"]

def encode(value):
    if isinstance(value, dict):
        return {"map":[[key, encode_value(item)] for key, item in value.items()]}
    return {"scalar":encode_value(value)}

inputs = fixture["inputs"] + ["nul", "./nul", "~/cache", "host:/remote", str(base)+"/junction/absent/child"]
providers = [value for text in inputs for value in [text, Path(text)]]
providers += [None, 3, {},
              {"day":str(base / "relative"), "1min":Path(base / "missing"), "week":"host:/remote"},
              {"day":str(base / "relative"), "1min":None},
              {"day":str(base / "relative"), "1min":3}]
mounts = [None, "~/cache", Path(base / "junction"), 3, {}, {"day":"~"},
          {"week":None, "day":str(base / "dir-link"), "1min":Path(base / "missing"), "extra":3}]
cases = []
for provider in providers:
    for mount in mounts:
        instance = Configuration(provider_uri=provider.copy() if isinstance(provider,dict) else provider,
                                 mount_path=mount.copy() if isinstance(mount,dict) else mount)
        before_provider, before_mount = encode(instance["provider_uri"]), encode(instance["mount_path"])
        try:
            instance.resolve_path()
            error = None
        except Exception as failure:
            error = {"class":type(failure).__name__, "code":getattr(failure,"winerror",None)}
        cases.append({"provider":before_provider, "mount":before_mount, "error":error,
                      "after_provider":encode(instance["provider_uri"]), "after_mount":encode(instance["mount_path"])})
print(json.dumps(cases))
