"""Execute the unchanged upstream lookup body with observable executor doubles."""

import ast
import json
import sys
from pathlib import Path

tree = ast.parse(Path(sys.argv[1]).read_bytes())
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_simulator_executor")
events = []


class BaseExecutor:
    pass


class SimulatorExecutor(BaseExecutor):
    pass


class NestedExecutor(BaseExecutor):
    def __init__(self, child, name, failure=None):
        self.child, self.name, self.failure = child, name, failure

    @property
    def inner_executor(self):
        events.append(self.name)
        if self.failure is not None:
            raise self.failure
        return self.child


class DerivedSimulator(SimulatorExecutor):
    @property
    def inner_executor(self):
        raise RuntimeError("must not inspect a non-nested child")


class Both(NestedExecutor, SimulatorExecutor):
    pass


exec(compile(ast.Module(body=[function], type_ignores=[]), "lookup_source", "exec"))
rows = []
for depth in (0, 1, 7, 1024):
    for kind in ("simulator", "derived", "invalid", "failure", "both"):
        events.clear()
        sentinel = RuntimeError("child unavailable")
        terminal = DerivedSimulator() if kind == "derived" else SimulatorExecutor()
        root = terminal
        if kind == "invalid":
            root = BaseExecutor()
        elif kind == "failure":
            root = NestedExecutor(terminal, "failure", sentinel)
        elif kind == "both":
            root = Both(terminal, "both")
        for i in range(depth):
            root = NestedExecutor(root, str(i))
        try:
            result = get_simulator_executor(root)
            status = "same" if result is terminal else "wrong identity"
            message = ""
        except Exception as error:
            status = "access" if error is sentinel else type(error).__name__
            message = str(error)
        rows.append(dict(depth=depth, kind=kind, events=list(events), status=status, message=message))
print(json.dumps(rows))
