"""Exercise unchanged upstream sequence methods, including file/cache effects."""
import contextlib
import io
import itertools
import json
from pathlib import Path
import runpy
import tempfile

with contextlib.redirect_stdout(io.StringIO()):
    setup = runpy.run_path(str(Path(__file__).with_name("file_calendar_storage_contract.py")))
ns, Conf = setup["ns"], setup["Conf"]

def outcome(call):
    try:
        return {"value": call()}
    except Exception as failure:
        return {"error": "Value" if isinstance(failure, ValueError) else "Other"}

huge = 10 ** 100
indices = [str(i) for i in [-huge, -5, -3, -1, 0, 1, 3, 5, huge]]
bounds = [None, str(-huge), "-1", "0", "2", str(huge)]
slices = list(itertools.product(bounds, bounds, [None, "0", "1", "-1", "2", "-2", str(huge), str(-huge)]))
assignments = ["", "XY😀", [], ["x"], ["x", "yy"], ["x", "y", "z"]]
cases = []
with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    (root / "calendars").mkdir()
    path = root / "calendars/day.txt"
    for initial in [None, [], ["a"], ["a", "bb", "a"], ["中", "😀", "é", "\0"]]:
        operations = [(op, selection, None) for op in ["get", "delete"] for selection in indices + slices]
        operations += [("set", selection, value) for selection in indices + slices for value in assignments]
        operations += [(op, value, None) for op in ["index", "remove"] for value in ["a", "bb", "missing", "😀", "\0", ""]]
        for op, selection, assignment in operations:
            if path.exists():
                path.unlink()
            if initial is not None:
                path.write_bytes(("" if not initial else "\n".join(initial) + "\n").encode())
            ns["C"] = Conf(region="cn", provider_uri={"day": str(root)}, mount_path={"day": None})
            ns["H"] = {"c": {}}
            storage = ns["FileCalendarStorage"]("day", False)
            if initial is not None:
                storage.data
            if op in ["index", "remove"]:
                key = selection
            elif isinstance(selection, str):
                key = int(selection)
            else:
                key = slice(*(None if v is None else int(v) for v in selection))
            def invoke():
                if op == "get": return storage[key]
                if op == "delete": del storage[key]
                elif op == "set": storage[key] = assignment
                elif op == "index": return storage.index(key)
                elif op == "remove": storage.remove(key)
                return None
            result = outcome(invoke)
            result["bytes"] = list(path.read_bytes()) if path.exists() else None
            result["cached"] = outcome(lambda: list(storage.data))
            cases.append(dict(initial=initial, op=op, selection=selection, assignment=assignment, result=result))
print(json.dumps(cases))
