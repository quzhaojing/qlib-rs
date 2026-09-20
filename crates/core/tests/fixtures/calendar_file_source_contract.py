"""Actual mixin, local loader and file constructor with a file dispatch boundary."""
import ast
import contextlib
import copy
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace

with contextlib.redirect_stdout(io.StringIO()):
    setup = runpy.run_path(str(Path(__file__).with_name("file_calendar_storage_contract.py")))
ns = setup["ns"]
ns["copy"] = copy
ns["init_instance_by_config"] = lambda spec: ns["FileCalendarStorage"](**spec["kwargs"])
source = Path(sys.argv[1]) / "data/data.py"
exec(setup["selected"](source, {"ProviderBackendMixin"}), ns)
node = next(n for n in ast.parse(source.read_text(encoding="utf-8")).body if isinstance(n, ast.ClassDef) and n.name == "LocalCalendarProvider")
method = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "load_calendar")
exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source), "exec"), ns)
Provider = type("LocalCalendarProvider", (ns["ProviderBackendMixin"],), {"load_calendar": ns["load_calendar"]})
results = []
for mode in ("inherited", "mapping", "fallback", "missing_region", "invalid_override",
             "partial_override", "invalid_frequency", "invalid_frequency_future", "missing_current", "invalid_row"):
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        (root / "calendars").mkdir()
        raw_frequency = "1min" if mode == "fallback" else "day"
        contents = "2024-01-02T14:59:00\n" if mode == "fallback" else "2024-01-02\n"
        if mode == "invalid_row": contents = "invalid-timestamp\n"
        if mode != "missing_current":
            (root / f"calendars/{raw_frequency}.txt").write_text(contents, encoding="utf-8")
        if mode == "invalid_row":
            (root / "calendars/day_future.txt").write_text(contents, encoding="utf-8")
        reads, warnings = [], []
        class Config(setup["Conf"]):
            def __getitem__(self, key):
                if key == "region": reads.append(key)
                return super().__getitem__(key)
        conf = Config(provider_uri={raw_frequency: str(root)}, mount_path={})
        if mode != "missing_region": conf["region"] = "cn"
        ns["C"], ns["H"] = conf, {"c": {}}
        def warning(message):
            warnings.append(message)
            if mode == "fallback": conf["region"] = "us"
        ns["get_module_logger"] = lambda name: SimpleNamespace(warning=warning)
        provider = Provider()
        kwargs = {"freq": "template-default", "future": False}
        if mode == "mapping": kwargs["provider_uri"] = {"day": str(root)}
        if mode == "invalid_override": kwargs["provider_uri"] = 123
        if mode == "partial_override": kwargs["provider_uri"] = {"day": ".", "bad": None}
        provider.backend = {"class": "FileCalendarStorage", "kwargs": kwargs}
        before = copy.deepcopy(provider.backend)
        frequency = "bad-frequency" if mode in ("missing_region", "invalid_frequency", "invalid_frequency_future") else "60min" if mode == "fallback" else "day"
        future = mode in ("fallback", "missing_region", "invalid_override", "partial_override", "invalid_frequency_future", "invalid_row")
        outputs = []
        for _ in range(2 if mode == "mapping" else 1):
            try:
                outputs.append({"values": [x.isoformat() for x in provider.load_calendar(frequency, future)]})
            except Exception as error:
                outputs.append({"error": "Value" if isinstance(error, ValueError) else "Other"})
        assert provider.backend == before
        results.append(dict(mode=mode, frequency=frequency, future=future, contents=contents,
                            outputs=outputs, region_reads=len(reads), warnings=len(warnings)))
print(json.dumps(results))
