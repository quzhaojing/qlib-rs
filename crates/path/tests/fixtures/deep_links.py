"""Reuse physical test links, then measure the unchanged source chain resolver."""
import contextlib
import io
import json
import ntpath
import runpy
from pathlib import Path

with contextlib.redirect_stdout(io.StringIO()):
    fixture = runpy.run_path(str(Path(__file__).with_name("read_link.py")))
units = fixture["units"]
results = []
for value in fixture["inputs"]:
    try:
        result = {"value": units(ntpath._readlink_deep(value))}
    except OSError as error:
        result = {"error": error.winerror}
    results.append([units(value), result])
print(json.dumps(results))
