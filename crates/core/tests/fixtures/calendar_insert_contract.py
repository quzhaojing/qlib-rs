"""Real Qlib insert conversion and persistence, without replacing NumPy."""
import contextlib
import io
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

values = ["x", "LONG😀", "", "\0", "A\0B\0", "😀中é", " 2 ", "1", "-0", "+0.0", "0.1",
          "1e-5", "1e-4", "1e15", "1e16", "1e20", "1e309", "-1e309", "1e-9999", "-1e-9999",
          "NaN", "+nan", "-NaN", "INF", "+Infinity", "-inf", " １_２.３e２\u2003", "١.٢",
          "1_2.3_4", "_1", "1_", "1__2", "1_e2", "1e_2", "nan(foo)", "0x1p2", "1\0", "1\n2",
          "\x1c1\x1c", "\u00851\u0085", "−1", "²", "1e", "1.2.3", ".5", "1."]
initial_states = [None, [], ["a"], ["a", "bb"], ["😀", "é"], ["a\0", "中\0b"]]
indices = [-10**100, -4, -2, -1, 0, 1, 2, 4, 10**100]
cases = []
with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    (root / "calendars").mkdir()
    path = root / "calendars/day.txt"
    for initial in initial_states:
        for index in indices:
            for value in values:
                if path.exists(): path.unlink()
                if initial is not None:
                    path.write_bytes(("" if not initial else "\n".join(initial) + "\n").encode())
                ns["C"] = Conf(region="cn", provider_uri={"day": str(root)}, mount_path={"day": None})
                ns["H"] = {"c": {}}
                storage = ns["FileCalendarStorage"]("day", False)
                if initial is not None: storage.data
                result = outcome(lambda: storage.insert(index, value))
                result["bytes"] = list(path.read_bytes()) if path.exists() else None
                result["cached"] = outcome(lambda: list(storage.data))
                cases.append(dict(initial=initial, index=str(index), value=value, result=result))
print(json.dumps(cases))
