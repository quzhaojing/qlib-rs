"""Execute unchanged upstream raw reader against real UTF-8 files."""
import ast
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

source = Path(sys.argv[1])
tree = ast.parse(source.read_text(encoding="utf-8"))
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FileCalendarStorage")
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_read_calendar")
namespace = {"List": list, "CalVT": str}
exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source), "exec"), namespace)
whitespace = "".join(chr(i) for i in range(0x110000) if chr(i).isspace())
inputs = ["", "\r\n\r\n", " a\rb\r\nc\n d ", "\ufeff2024-01-02\n",
          "a\u0085b\u2028c\u2029d\v\fend", whitespace + "日期" + whitespace,
          "\x00 x\x00\r\n", "last", "  \n\t\r", "\x1cfirst\x1f\n"]
cases = []
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "calendar.txt"
    for text in inputs:
        path.write_bytes(text.encode("utf-8"))
        cases.append([text, namespace["_read_calendar"](SimpleNamespace(uri=path))])
print(json.dumps({"cases": cases, "whitespace": whitespace}))
