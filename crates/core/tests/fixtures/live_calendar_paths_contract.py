"""Unchanged Qlib storage with live globals, shared overrides and mount changes."""
import contextlib
import io
import json
from pathlib import Path
import runpy
import tempfile

with contextlib.redirect_stdout(io.StringIO()):
    setup = runpy.run_path(str(Path(__file__).with_name("file_calendar_storage_contract.py")))
ns, Conf = setup["ns"], setup["Conf"]
actions = [["data"], ["uri"], ["support"], ["provider", 1], ["data"], ["mount", 0], ["data"],
           ["override", 2], ["data"], ["override_root", 1], ["data"], ["override_nfs"], ["data"],
           ["mount", 2], ["data"], ["provider", 0], ["data"], ["mount", 1], ["data"],
           ["clear"], ["data"], ["read"], ["extend"], ["read"], ["cache_clear"], ["data"],
           ["override_empty"], ["uri"], ["drop_override"], ["data"], ["provider_empty"],
           ["uri"], ["data"], ["provider", 0], ["data"], ["support"]]
cases = []
with tempfile.TemporaryDirectory() as folder:
    for mode in ["global", "local_override", "nfs_override"]:
        roots = [Path(folder) / mode / str(i) for i in range(3)]
        for i, root in enumerate(roots):
            (root / "calendars").mkdir(parents=True)
            (root / "calendars/day.txt").write_bytes(f"row{i}\n".encode())
        conf = Conf(region="cn", provider_uri={"day": str(roots[0])}, mount_path={"day": str(roots[1])})
        ns["C"], ns["H"] = conf, {"c": {}}
        override = None if mode == "global" else {"day": str(roots[2]) if mode == "local_override" else "host:/data"}
        storage = ns["FileCalendarStorage"]("day", False, provider_uri=override)
        results = []
        for action in actions:
            name, *args = action
            try:
                value = None
                if name == "data": value = list(storage.data)
                elif name == "uri": value = next(i for i, root in enumerate(roots) if storage.uri == root / "calendars/day.txt")
                elif name == "support": value = list(map(str, storage.support_freq))
                elif name == "provider": conf["provider_uri"] = {"day": str(roots[args[0]])}
                elif name == "provider_empty": conf["provider_uri"] = {}
                elif name == "mount": conf["mount_path"]["day"] = str(roots[args[0]])
                elif name == "override":
                    override = {"day": str(roots[args[0]])}
                    storage._provider_uri = override
                elif name == "override_root": override["day"] = str(roots[args[0]])
                elif name == "override_nfs": override["day"] = "host:/data"
                elif name == "override_empty": override.clear()
                elif name == "drop_override": storage._provider_uri = None
                elif name == "clear": storage.clear()
                elif name == "read": value = storage._read_calendar()
                elif name == "extend": storage.extend(["new"])
                elif name == "cache_clear": ns["H"]["c"].clear()
                else: raise AssertionError(name)
                result = {"value": value}
            except Exception as failure:
                result = {"error": "Value" if isinstance(failure, ValueError) else "Other"}
            result["files"] = [list((root / "calendars/day.txt").read_bytes()) for root in roots]
            results.append(result)
        cases.append(dict(mode=mode, actions=actions, results=results))
print(json.dumps(cases))
