"""Run the unchanged local loader, including lazy rows and logging failures."""
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pandas as pd

path = Path(sys.argv[1])
tree = ast.parse(path.read_text(encoding="utf-8"))
cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LocalCalendarProvider")
method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "load_calendar")
cases = []
for mode, future in [("ok", True), ("value", False), ("value", True), ("other", True), ("parse", True), ("retry", True), ("construct", True), ("lazy", True), ("empty", True), ("warning1", True), ("warning2", True), ("unicode", True), ("missing", True)]:
    events, warnings = [], []
    def warning(message):
        warnings.append(message)
        events.append(["warning", len(warnings)])
        if mode == "warning" + str(len(warnings)):
            raise RuntimeError("warning")
    def timestamp(value):
        events.append(["decode", value])
        return pd.Timestamp(value)
    namespace = {"pd": SimpleNamespace(Timestamp=timestamp), "get_module_logger": lambda name: SimpleNamespace(warning=warning)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"), namespace)
    def rows():
        if mode == "empty":
            return
        for index in range(2):
            events.append(["next", index])
            if mode == "lazy" and index == 1:
                raise ValueError("lazy")
            yield "bad-date" if mode == "parse" else "2024-01-02"
    class Backend:
        def __init__(self, future):
            self.future = future
        @property
        def data(self):
            events.append(["data", self.future])
            if self.future and mode == "unicode":
                raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid byte")
            if self.future and mode == "missing":
                raise FileNotFoundError("gone")
            if mode == "retry" or (mode in ("value", "warning1", "warning2") and self.future == future):
                raise ValueError("backend")
            if mode == "other":
                raise OSError("backend")
            return rows()
    class Provider:
        load_calendar = namespace["load_calendar"]
        def backend_obj(self, freq, future):
            events.append(["backend", freq, future])
            if mode == "construct" and future:
                raise ValueError("construct")
            return Backend(future)
    try:
        result = [str(value) for value in Provider().load_calendar("1min", future)]
        error = None
    except Exception as failure:
        result = None
        error = "Value" if isinstance(failure, ValueError) else "Other"
    cases.append({"mode": mode, "future": future, "events": events, "warnings": warnings, "result": result, "error": error})
print(json.dumps(cases))
