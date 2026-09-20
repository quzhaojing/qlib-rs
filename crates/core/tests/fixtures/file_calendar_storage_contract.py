"""Persistent instances of unchanged Qlib storage classes over real files."""
import ast
import contextlib
import io
import json
from pathlib import Path
import platform
import re
import runpy
import sys
import tempfile
from types import SimpleNamespace
from typing import Iterable, overload

root = Path(sys.argv[1])
# Reuse loading of actual Freq/resampling definitions, not a substitute algorithm.
with contextlib.redirect_stdout(io.StringIO()):
    setup = runpy.run_path(str(Path(__file__).with_name("file_calendar_backend_contract.py")))
ns = setup["ns"]
ns.update(Iterable=Iterable, overload=overload)
selected = setup["selected"]
exec(selected(root / "data/storage/storage.py", {"BaseStorage", "CalendarStorage"}), ns)
exec(selected(root / "data/storage/file_storage.py", {"FileStorageMixin", "FileCalendarStorage"}), ns)
tree = ast.parse((root / "config.py").read_text(encoding="utf-8"))
outer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "QlibConfig")
manager_node = next(n for n in outer.body if isinstance(n, ast.ClassDef) and n.name == "DataPathManager")
constants = SimpleNamespace(DEFAULT_FREQ="__DEFAULT_FREQ", LOCAL_URI="local", NFS_URI="nfs")
manager_ns = dict(QlibConfig=constants, Path=Path, re=re, platform=platform)
body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), manager_node]
exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), "actual-data-path-manager", "exec"), manager_ns)
manager = manager_ns["DataPathManager"]
constants.DataPathManager = manager

class Conf(dict):
    DEFAULT_FREQ = "__DEFAULT_FREQ"
    min_data_shift = 0
    DataPathManager = manager

    @property
    def dpm(self):
        return manager(self["provider_uri"], self["mount_path"])

    @property
    def mount_path(self):
        return self["mount_path"]

minute = "2024-01-02 09:30:00\n2024-01-02 09:31:00\n"
day = "2024-01-02\n2024-01-03\n"
cases = [
    dict(name="cached_discovery", frequency="5min", initial={"1min.txt": minute}, actions=[
        ["support"], ["put", 0, "5min.txt", minute], ["uri"], ["file_frequency"], ["data"],
        ["set_frequency", "1min"], ["data"], ["length"], ["empty"],
        ["set_future", True], ["uri"], ["data"], ["put", 0, "1min_future.txt", minute], ["data"]]),
    dict(name="empty_discovery_cached", frequency="1min", initial={}, actions=[
        ["uri"], ["put", 0, "1min.txt", minute], ["uri"], ["support"], ["clear"], ["read"], ["length"], ["empty"]]),
    dict(name="discovery_failure_retry", frequency="1min", initial={"bad.txt": "x\n"}, actions=[
        ["support"], ["remove_file", 0, "bad.txt"], ["put", 0, "1min.txt", minute], ["support"], ["uri"], ["data"]]),
    dict(name="parse_before_discovery", frequency="bad", initial={}, actions=[
        ["uri"], ["put", 0, "1min.txt", minute], ["set_frequency", "1min"], ["uri"], ["data"]]),
    dict(name="selection_failure_retains_discovery", frequency="1min", initial={"day.txt": day}, actions=[
        ["uri"], ["put", 0, "1min.txt", minute], ["uri"], ["set_frequency", "day"], ["uri"], ["data"]]),
    dict(name="live_roots_and_flags", frequency="day", initial={"day.txt": day}, actions=[
        ["uri"], ["data"], ["put", 1, "day.txt", "2024-02-01\n"], ["set_root", 1], ["uri"], ["data"],
        ["set_future", True], ["uri"], ["read"], ["empty"], ["extend", ["2024-02-02"]],
        ["read"], ["length"], ["cache_clear"], ["length"]]),
    dict(name="read_before_changed_request_parse", frequency="day", initial={"day.txt": day}, actions=[
        ["uri"], ["set_frequency", "bad"], ["data"], ["put", 0, "day.txt", "2025-01-01\n"],
        ["set_frequency", "day"], ["data"], ["set_cache", False], ["data"], ["clear"], ["empty"]]),
    dict(name="named_missing_file_mutations", frequency="day", named=True, initial={}, actions=[
        ["data"], ["read"], ["empty"], ["extend", ["2024-01-02", "2024-01-03"]], ["read"],
        ["data"], ["cache_clear"], ["data"], ["clear"], ["data"], ["read"], ["length"],
        ["cache_clear"], ["empty"], ["overwrite", ["2024-01-04"]], ["read"]]),
    dict(name="existing_writes_keep_cache", frequency="day", initial={"day.txt": day}, actions=[
        ["data"], ["extend", ["2024-01-04"]], ["data"], ["read"], ["clear"], ["data"],
        ["read"], ["length"], ["cache_clear"], ["empty"]]),
    dict(name="root_failure_before_discovery_retries", frequency="day", initial={"day.txt": day}, actions=[
        ["set_provider", "host:/data"], ["support"], ["uri"], ["set_root", 0], ["support"], ["data"]]),
    dict(name="root_failure_after_selection_is_live", frequency="day", initial={"day.txt": day}, actions=[
        ["uri"], ["clear_providers"], ["uri"], ["data"], ["read"], ["extend", ["x"]],
        ["set_root", 0], ["uri"], ["data"]]),
    dict(name="read_write_errors_and_retry", frequency="day", initial={"day.txt": day}, actions=[
        ["uri"], ["put_bytes", 0, "day.txt", [255]], ["read"], ["data"],
        ["put", 0, "day.txt", day], ["data"], ["overwrite_scalar", "bad"], ["read"],
        ["data"], ["cache_clear"], ["empty"], ["set_frequency", "1min"], ["data"]]),
]

def normalize(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return [["text", v] if isinstance(v, str) else ["timestamp", v.isoformat()] for v in value]
    return value

with tempfile.TemporaryDirectory() as folder:
    base = Path(folder)
    for index, case in enumerate(cases):
        roots = [base / str(index) / str(i) for i in range(2)]
        for path in roots:
            (path / "calendars").mkdir(parents=True)
        for name, text in case["initial"].items():
            (roots[0] / "calendars" / name).write_bytes(text.encode())
        key = case["frequency"] if case.get("named") else Conf.DEFAULT_FREQ
        conf = Conf(region="cn", provider_uri={key: str(roots[0])}, mount_path={key: None})
        ns["C"], ns["H"] = conf, {"c": {}}
        storage = ns["FileCalendarStorage"](case["frequency"], False)
        results = []
        for action in case["actions"]:
            name, *args = action
            try:
                value = None
                if name == "support": value = list(map(str, storage.support_freq))
                elif name == "file_frequency": value = str(storage._freq_file)
                elif name == "uri":
                    path = storage.uri
                    value = next([i, path.relative_to(r).as_posix()] for i, r in enumerate(roots) if path.is_relative_to(r))
                elif name == "data": value = normalize(storage.data)
                elif name == "read": value = storage._read_calendar()
                elif name == "length": value = len(storage)
                elif name == "empty": value = len(storage) == 0
                elif name == "clear": storage.clear()
                elif name == "extend": storage.extend(args[0])
                elif name == "overwrite": storage._write_calendar(args[0])
                elif name == "overwrite_scalar": storage._write_calendar(args[0])
                elif name == "cache_clear": ns["H"]["c"].clear()
                elif name == "set_cache": storage.enable_read_cache = args[0]
                elif name == "set_frequency": storage.freq = args[0]
                elif name == "set_future": storage.future = args[0]
                elif name == "set_root": conf["provider_uri"][key] = str(roots[args[0]])
                elif name == "set_provider":
                    conf["provider_uri"][key] = args[0]
                    conf["mount_path"].clear()
                elif name == "clear_providers": conf["provider_uri"].clear()
                elif name == "put": (roots[args[0]] / "calendars" / args[1]).write_bytes(args[2].encode())
                elif name == "put_bytes": (roots[args[0]] / "calendars" / args[1]).write_bytes(bytes(args[2]))
                elif name == "remove_file": (roots[args[0]] / "calendars" / args[1]).unlink()
                else: raise AssertionError(name)
                result = dict(value=value)
            except Exception as failure:
                result = dict(error="Value" if isinstance(failure, ValueError) else "Other")
            result["files"] = [{p.name: list(p.read_bytes()) for p in sorted((r / "calendars").iterdir())} for r in roots]
            results.append(result)
        case["results"] = results
print(json.dumps(cases))
