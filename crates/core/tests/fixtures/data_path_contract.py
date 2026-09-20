"""Execute unchanged Qlib DataPathManager lookup with explicit platform labels."""
import ast
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

tree = ast.parse(Path(sys.argv[1]).read_text(encoding="utf-8"))
outer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "QlibConfig")
node = next(node for node in outer.body if isinstance(node, ast.ClassDef) and node.name == "DataPathManager")
config = SimpleNamespace(DEFAULT_FREQ="__DEFAULT_FREQ", LOCAL_URI="local", NFS_URI="nfs")
namespace = {"QlibConfig": config, "Path": Path, "re": re}
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])), "actual-data-path-manager", "exec"), namespace)
manager = namespace["DataPathManager"]
config.DataPathManager = manager
kinds = [[uri, manager.get_uri_type(uri)] for uri in ["", "D:", "D:/data", "host:/data", "https://example/data", "//server/share", "relative/data", "host:", "x:/data", "host:\nline", "host:\nnext:/ok"]]
cases = []

def check(provider, mount, freq, system):
    namespace["platform"] = SimpleNamespace(system=lambda: system)
    try:
        result = str(manager(provider, mount).get_data_uri(freq))
        error = None
    except Exception as failure:
        result = None
        error = type(failure).__name__
    cases.append(dict(provider=provider, mount=mount, freq=freq, system=system, result=result, error=error))

check({}, {}, None, "Windows")
for freq in [None, "day", "1day"]:
    check({"day": "D:/daily"}, {}, freq, "Windows")
    check({"day": "D:/daily", "__DEFAULT_FREQ": "D:/fallback"}, {}, freq, "Windows")
for value in ["", ".", "./a//b/", "C:", "C:/", "//server/share//a/", "a/../b", "\\rooted", "a/./b/", "./"]:
    check({"__DEFAULT_FREQ": value}, {}, "day", "Windows")
for uri in ["host:/data", "https://example/data"]:
    for system in ["Windows", "Linux", "Darwin", "CYGWIN_NT", "win32"]:
        check({"day": uri}, {}, "day", system)
        for mount in [None, "Z", "Z:", "Z:\\", "/mnt/data"]:
            check({"day": uri}, {"day": mount}, "day", system)
print(json.dumps(dict(kinds=kinds, paths=cases)))
