"""Real file storage, local provider/cache and execution calendar in one source graph."""
import abc
import bisect
import contextlib
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace

with contextlib.redirect_stdout(io.StringIO()):
    setup = runpy.run_path(str(Path(__file__).with_name("calendar_file_source_contract.py")))
ns = setup["ns"]
select = setup["setup"]["selected"]
root = Path(sys.argv[1])
ns.update(abc=abc, bisect=bisect)
exec(select(root / "data/data.py", {"CalendarProvider"}), ns)
exec(select(root / "utils/time.py", {"epsilon_change"}), ns)
exec(select(root / "backtest/utils.py", {"TradeCalendarManager"}), ns)
Provider = type("LocalCalendarProvider", (ns["CalendarProvider"], ns["ProviderBackendMixin"]),
                {"load_calendar": ns["load_calendar"]})
pd = ns["pd"]
base = pd.Timestamp("2024-01-02 09:30:00")
contents = "".join((base + pd.Timedelta(minutes=i)).isoformat() + "\n" for i in range(10))
rows = []
for future in (False, True):
    with tempfile.TemporaryDirectory() as folder:
        directory = Path(folder) / "calendars"
        directory.mkdir()
        (directory / "1min.txt").write_text(contents, encoding="utf-8")
        if future:
            (directory / "1min_future.txt").write_text(contents, encoding="utf-8")
        conf = setup["setup"]["Conf"](provider_uri={"1min": folder}, mount_path={})
        conf["region"] = "cn"
        ns["C"], ns["H"] = conf, {"c": {}}
        warnings = []
        ns["get_module_logger"] = lambda name: SimpleNamespace(warning=warnings.append)
        provider = Provider()
        provider.backend = {"class": "FileCalendarStorage", "kwargs": {}}
        ns["Cal"] = provider
        exchange = SimpleNamespace(freq="1min")
        calendar = ns["TradeCalendarManager"]("2min", base, base + pd.Timedelta(minutes=5),
            level_infra={"common_infra": {"trade_exchange": exchange}})
        steps = []
        while not calendar.finished():
            start, end = calendar.get_step_time()
            minute_range = calendar.get_data_cal_range("step")
            exchange.freq = "2min"
            coarse_range = calendar.get_data_cal_range("step")
            exchange.freq = "1min"
            steps.append(dict(start=start.isoformat(), end=end.isoformat(),
                              minute=list(minute_range), coarse=list(coarse_range)))
            calendar.step()
        initial_warnings = len(warnings)
        # Cached future calendar must survive deletion until the shared cache is cleared.
        for path in directory.iterdir():
            path.unlink()
        calendar.reset("2min", base, base + pd.Timedelta(minutes=5))
        cached_length = calendar.get_trade_len()
        ns["H"]["c"].clear()
        try:
            calendar.reset("2min", base, base + pd.Timedelta(minutes=5))
        except ValueError:
            reload_error = "Value"
        else:
            raise AssertionError("cleared cache unexpectedly loaded deleted files")
        rows.append(dict(future=future, contents=contents, steps=steps,
                         warnings=initial_warnings, cached_length=cached_length,
                         reload_error=reload_error))
print(json.dumps(rows))
