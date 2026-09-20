"""Actual Qlib frequency enumeration over Windows-native filenames."""
import contextlib
import io
import json
from pathlib import Path
import runpy
import tempfile

with contextlib.redirect_stdout(io.StringIO()):
    setup = runpy.run_path(str(Path(__file__).with_name("file_calendar_storage_contract.py")))
ns, Conf = setup["ns"], setup["Conf"]

def units(text):
    raw = text.encode("utf-16-le", "surrogatepass")
    return [int.from_bytes(raw[i:i+2], "little") for i in range(0, len(raw), 2)]

names = [".txt", "day_future.txt", "DAY_other.TXT", "weird.more.txt", "_future.txt", "name", "x",
         ".hidden.txt", "é.txt", "é.txt", "\ud800.txt", "\udc00.txt", "\ud800A.txt", "\ue000.txt",
         "😀.txt", "𐀀.txt", "not.txt.bin", "...txt", "..txt", "....txt", "a..txt", "中", "old.TXT"]
cases = []
with tempfile.TemporaryDirectory() as folder:
    for mode in ["missing", "not_directory", "empty", "files", "directories"]:
        for frequency in ["day", "bad"]:
            for named in [True, False]:
                root = Path(folder) / str(len(cases))
                root.mkdir()
                directory = root / "calendars"
                if mode == "not_directory": directory.write_bytes(b"not a directory")
                elif mode != "missing": directory.mkdir()
                if mode in ["files", "directories"]:
                    for name in names:
                        if mode == "files": (directory / name).write_bytes(b"unchanged")
                        else: (directory / name).mkdir()
                key = frequency if named else Conf.DEFAULT_FREQ
                ns["C"] = Conf(region="cn", provider_uri={key: str(root)}, mount_path={key: None})
                ns["H"] = {"c": {}}
                storage = ns["FileCalendarStorage"](frequency, False)
                results = []
                for attempt in range(2):
                    if attempt and directory.is_dir(): (directory / "added_future.txt").write_bytes(b"new")
                    try:
                        results.append({"value": [units(name) for name in storage._get_storage_freq()]})
                    except Exception as failure:
                        results.append({"error": "Value" if isinstance(failure, ValueError) else "Other"})
                cases.append(dict(mode=mode, frequency=frequency, named=named,
                                  names=[units(name) for name in names], results=results))
print(json.dumps(cases))
