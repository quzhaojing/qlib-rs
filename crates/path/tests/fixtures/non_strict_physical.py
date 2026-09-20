"""Measure actual native fallback, retaining raw UTF-16 spellings."""
import contextlib
import io
import json
import ntpath
import runpy
from pathlib import Path

with contextlib.redirect_stdout(io.StringIO()):
    fixture = runpy.run_path(str(Path(__file__).with_name("read_link.py")))
units = fixture["units"]
base = str(fixture["base"])
inputs = fixture["inputs"] + [base + suffix for suffix in [
    "/dir-link/absent/child", "/junction/absent/child", "/missing/child",
    "/file.txt/child", "/folder/./missing", "/folder///", "/surrogate/child",
    "/locked.txt/child", "/cycle-a/child"]]
results = []
for value in inputs:
    try:
        result = {"value": units(ntpath._getfinalpathname_nonstrict(value))}
    except ValueError:
        result = {"error": "nul"}
    except OSError as error:
        result = {"error": error.winerror}
    results.append([units(value), result])
print(json.dumps(results))
