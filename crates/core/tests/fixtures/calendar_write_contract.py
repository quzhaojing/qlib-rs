"""Actual Qlib binary calendar writer: bytes, conversion failure and truncation."""
import ast
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import numpy as np

source = Path(sys.argv[1])
tree = ast.parse(source.read_text(encoding="utf-8"))
cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FileCalendarStorage")
method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_write_calendar")
ns = {"np": np, "Iterable": list, "CalVT": str}
exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source), "exec"), ns)
arrays = [
    ([], ["scalar"]),
    ([0], []),
    ([1], [""]),
    ([4], ["2024-01-02", "  spaced  ", "a\r\nb", "\ufeff中😀\x00"]),
    ([2, 2], ["a", "b", "c d", "e\nf"]),
    ([0, 2], []),
    ([2, 0], []),
    ([0, 0], []),
    ([1, 1, 1], ["x"]),
    ([0, 1, 2], []),
    ([4], ["a\x00b", "\x00\x00", "a\x00\x00", "\x00a"]),
    ([1, 2], ["", ""]),
]
cases = []
with tempfile.TemporaryDirectory() as folder:
    directory = Path(folder)
    for shape, values in arrays:
        array = np.array(values, dtype=str).reshape(shape)
        for mode in ("wb", "ab"):
            for state in ("existing", "missing", "missing_parent", "directory"):
                path = directory / "file.txt"
                if path.exists():
                    path.unlink()
                if state == "existing":
                    path.write_bytes(b"original\n")
                elif state == "missing_parent":
                    path = directory / "absent" / "file.txt"
                elif state == "directory":
                    path = directory
                try:
                    ns["_write_calendar"](SimpleNamespace(uri=path), array, mode)
                    error = None
                except Exception as failure:
                    error = "Value" if isinstance(failure, ValueError) else "Other"
                data = list(path.read_bytes()) if path.is_file() else None
                cases.append(dict(shape=shape, values=values, mode=mode, state=state, error=error, data=data))
print(json.dumps(cases))
