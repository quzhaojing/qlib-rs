"""Actual Python lstat and islink results, not a rewritten fallback algorithm."""
import contextlib
import io
import json
import ntpath
import os
from pathlib import Path
import runpy

with contextlib.redirect_stdout(io.StringIO()):
    fixture=runpy.run_path(str(Path(__file__).with_name("read_link.py")))
base=fixture["base"]
inputs=fixture["inputs"] + [str(base)+"\\"+part for part in ["*.txt","?elative","junction///","dir-link/\\","absent///"]] + ["nul", "\\\\.\\NUL", "C:/", "a/", "a\\", "C://"]
results=[]
for value in inputs:
    try:
        info=os.lstat(value)
        stat={"attributes":info.st_file_attributes,"tag":info.st_reparse_tag}
    except ValueError:
        stat={"error":"nul"}
    except OSError as error:
        stat={"error":error.winerror}
    results.append([fixture["units"](value),stat,ntpath.islink(value)])
print(json.dumps(results))
