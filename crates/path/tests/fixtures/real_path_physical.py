"""Measure actual outer realpath using test-owned links and native errors."""
import contextlib
import io
import json
import ntpath
import runpy
from pathlib import Path

with contextlib.redirect_stdout(io.StringIO()):
    fixture = runpy.run_path(str(Path(__file__).with_name("non_strict_physical.py")))
units = fixture["units"]
base = fixture["base"]
inputs = fixture["inputs"] + ["nul", "NUL", "./nul", "a/../nul", ".", "..", "relative/../absent",
    "\\\\?\\" + base + "\\missing", "\\\\?\\" + base + "\\file.txt", base + "/folder/../missing"]
results = []
for value in inputs:
    try:
        result = {"value": units(ntpath.realpath(value))}
    except ValueError:
        result = {"error":"nul"}
    except OSError as error:
        result = {"error":error.winerror}
    results.append([units(value), result])
print(json.dumps(results))
