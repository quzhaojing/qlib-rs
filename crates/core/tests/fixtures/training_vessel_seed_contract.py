import ast
import json
import sys
from types import SimpleNamespace

source = sys.argv[1]
tree = ast.parse(open(source, encoding="utf-8").read(), filename=source)
classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
methods = {node.name: node for node in classes["TrainingVessel"].body if isinstance(node, ast.FunctionDef)}
subset = methods["_random_subset"]
subset.decorator_list = []
logs = []
calls = []

def permutation(length):
    calls.append(length)
    return list(reversed(range(length)))

namespace = {
    "np": SimpleNamespace(random=SimpleNamespace(permutation=permutation)),
    "_logger": SimpleNamespace(info=lambda *args: logs.append(args)),
}
module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), subset], type_ignores=[])
exec(compile(ast.fix_missing_locations(module), source, "exec"), namespace)
collection = [10, 20, 30, 40]
results = []
for size in [None, 0, 2, -1, 10, -10]:
    result = namespace["_random_subset"]("val", collection, size)
    results.append({"size": size, "values": result, "same_object": result is collection})
queue_kwargs = {}
for name in ["train_seed_iterator", "val_seed_iterator", "test_seed_iterator"]:
    call = next(node for node in ast.walk(methods[name]) if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "DataQueue")
    queue_kwargs[name] = {item.arg: ast.literal_eval(item.value) for item in call.keywords}
base_methods = {node.name: node for node in classes["TrainingVesselBase"].body if isinstance(node, ast.FunctionDef)}
missing = []
for name in queue_kwargs:
    raised = next(node for node in ast.walk(base_methods[name]) if isinstance(node, ast.Raise))
    missing.append(ast.literal_eval(raised.exc.args[0]))
print(json.dumps({"subsets": results, "permutation_calls": calls, "log_count": len(logs), "queues": queue_kwargs, "missing": missing}))
