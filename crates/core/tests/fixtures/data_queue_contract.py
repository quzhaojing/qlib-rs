import ast
import json
import sys


source_path = sys.argv[1]
tree = ast.parse(open(source_path, encoding="utf-8").read(), filename=source_path)
queue = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DataQueue")
methods = {node.name: node for node in queue.body if isinstance(node, ast.FunctionDef)}
init = methods["__init__"]
defaults = [ast.literal_eval(node) for node in init.args.defaults]
get_constants = [node.value for node in ast.walk(methods["get"]) if isinstance(node, ast.Constant)]
producer = methods["_producer"]
loader = next(
    node
    for node in ast.walk(producer)
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "DataLoader"
)
loader_keywords = {
    keyword.arg: ast.literal_eval(keyword.value)
    for keyword in loader.keywords
    if keyword.arg in {"batch_size", "shuffle", "num_workers"}
    and isinstance(keyword.value, ast.Constant)
}
print(
    json.dumps(
        {
            "defaults": defaults,
            "methods": sorted(methods),
            "timeouts": [value for value in get_constants if value in (5.0, 0.5)],
            "infinite_repeat_power": any(
                isinstance(node, ast.BinOp)
                and isinstance(node.op, ast.Pow)
                and ast.literal_eval(node.left) == 10
                and ast.literal_eval(node.right) == 18
                for node in ast.walk(producer)
            ),
            "loader_keywords": loader_keywords,
            "daemon_thread": any(
                isinstance(node, ast.keyword)
                and node.arg == "daemon"
                and isinstance(node.value, ast.Constant)
                and node.value.value is True
                for node in ast.walk(methods["activate"])
            ),
        }
    )
)
