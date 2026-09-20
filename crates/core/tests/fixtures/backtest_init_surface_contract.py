import ast
import json
from pathlib import Path

source = Path(r"D:\code\github\qlib\qlib\backtest\__init__.py")
tree = ast.parse(source.read_text(encoding="utf-8"))

runtime_imports = []
type_checking_imports = []
for node in tree.body:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        runtime_imports.extend(alias.asname or alias.name for alias in node.names)
    elif isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
        for child in node.body:
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                type_checking_imports.extend(alias.asname or alias.name for alias in child.names)

functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
assignments = {
    target.id: ast.literal_eval(node.value)
    for node in tree.body
    if isinstance(node, ast.Assign)
    for target in node.targets
    if isinstance(target, ast.Name) and target.id == "__all__"
}
logger_calls = [
    ast.literal_eval(node.value.args[0])
    for node in tree.body
    if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "logger" for target in node.targets)
    and isinstance(node.value, ast.Call)
    and isinstance(node.value.func, ast.Name)
    and node.value.func.id == "get_module_logger"
]

print(json.dumps({
    "functions": functions,
    "all": assignments["__all__"],
    "logger_calls": logger_calls,
    "runtime_imports": runtime_imports,
    "type_checking_imports": type_checking_imports,
}, separators=(",", ":")))
