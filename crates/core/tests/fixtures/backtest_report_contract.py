"""Characterize report publication using the unchanged outer loop and frequency parser."""
import ast
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

root = Path(sys.argv[1])
def extract(file, name, kind):
    tree = ast.parse((root / file).read_text(encoding="utf-8"))
    return next(n for n in tree.body if isinstance(n, kind) and n.name == name)

module = ast.Module(body=[
    ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
    extract("utils/time.py", "Freq", ast.ClassDef),
    extract("backtest/backtest.py", "collect_data_loop", ast.FunctionDef),
], type_ignores=[])
namespace = {"re": re}
exec(compile(ast.fix_missing_locations(module), "unchanged-backtest-source", "exec"), namespace)

def run(failure=None, last_enabled=False, publish=True):
    events = []
    class Bar:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            events.append("close")
    class Indicator:
        def __init__(self, name):
            self.name = name
        def generate_trade_indicators_dataframe(self):
            events.append("export:" + self.name)
            if failure == "export" and self.name == "last":
                raise ValueError("export failure")
            return self.name
    class Account:
        def __init__(self, name, enabled):
            self.name, self.enabled = name, enabled
            self.indicator = Indicator(name)
        def is_port_metr_enabled(self):
            events.append("enabled:" + self.name)
            return self.enabled
        def get_portfolio_metrics(self):
            events.append("portfolio:" + self.name)
            return self.name
        def get_trade_indicator(self):
            events.append("indicator:" + self.name)
            return self.indicator
    levels = [SimpleNamespace(time_per_step=freq, trade_account=Account(name, enabled))
              for freq, name, enabled in [("day", "first", True), ("min", "middle", True),
                  ("bad" if failure == "frequency" else "1D", "last", last_enabled)]]
    executor = SimpleNamespace(reset=lambda **_: None, get_level_infra=lambda: None,
        trade_calendar=SimpleNamespace(get_trade_len=lambda: 0), finished=lambda: True,
        get_all_executors=lambda: levels)
    strategy = SimpleNamespace(reset=lambda **_: None,
        post_upper_level_exe_step=lambda: events.append("finalize"))
    namespace["tqdm"] = lambda **_: Bar()
    result = {"prior": 42}
    error = None
    try:
        list(namespace["collect_data_loop"](None, None, strategy, executor,
                                             result if publish else None))
    except ValueError as caught:
        error = str(caught)
    if "indicator_dict" in result:
        assert result["indicator_dict"]["1day"][1] is levels[-1].trade_account.indicator
        result["indicator_dict"] = {key: (frame, obj.name)
                                    for key, (frame, obj) in result["indicator_dict"].items()}
    return {"events": events, "result": result, "error": error}

print(json.dumps({"disabled": run(), "enabled": run(last_enabled=True),
                  "export_failure": run("export"), "frequency_failure": run("frequency"),
                  "no_reports": run(publish=False)}))
