"""Build links inside a test-owned directory, report actual nt.readlink results."""
import json
import nt
import os
import sys
import _winapi
from pathlib import Path


def units(text):
    data = text.encode("utf-16-le", "surrogatepass")
    return [int.from_bytes(data[i:i+2], "little") for i in range(0, len(data), 2)]


base = Path(sys.argv[1])
(base / "folder").mkdir()
(base / "file.txt").write_text("contents")
links = {
    "relative": "file.txt",
    "absolute": str(base / "file.txt"),
    "missing": "missing-target",
    "surrogate": "a\ud800z",
    "dots": "folder/../file.txt",
    "chain": "relative",
    "cycle-a": "cycle-b",
    "cycle-b": "cycle-a",
}
for name, destination in links.items():
    os.symlink(destination, base / name)
os.symlink("folder", base / "dir-link", target_is_directory=True)
_winapi.CreateJunction(str(base / "folder"), str(base / "junction"))
inputs = [str(base / name) for name in [*links, "dir-link", "junction", "file.txt", "folder", "absent", "no-parent/child", "locked.txt"]]
inputs += ["", str(base / "relative") + "\0", str(base / "dir-link") + "\\"]
results = []
for value in inputs:
    try:
        result = {"value": units(nt.readlink(value))}
    except ValueError:
        result = {"error": "ValueError"}
    except OSError as error:
        result = {"error": error.winerror}
    results.append([units(value), result])
print(json.dumps(results))
