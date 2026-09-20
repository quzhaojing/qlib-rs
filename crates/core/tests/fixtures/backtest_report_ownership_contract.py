"""Actual account reset + report publication: frozen tables and retained object identities.

Position construction, raw order-indicator creation, progress UI and benchmark acquisition
are explicit boundary collaborators. Account/report/outer-loop methods are unchanged.
"""
import ast
from collections import OrderedDict
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import pandas as pd

root = Path(sys.argv[1])
ns = {"pd": pd, "OrderedDict": OrderedDict, "re": re,
      "NumpyOrderIndicator": type("RawOrderIndicator", (), {})}
def load(relative, names):
    path = root / relative
    nodes = [n for n in ast.parse(path.read_text(encoding="utf-8")).body
             if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert len(nodes) == len(names)
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)

load("utils/time.py", {"Freq"})
load("backtest/report.py", {"PortfolioMetrics", "Indicator"})
load("backtest/account.py", {"Account"})
load("backtest/backtest.py", {"collect_data_loop"})
PortfolioMetrics, Indicator, Account = (ns[name] for name in ("PortfolioMetrics", "Indicator", "Account"))

class Bar:
    def __enter__(self): return self
    def __exit__(self, *_): return False
ns["tqdm"] = lambda **_: Bar()
start = pd.Timestamp("2024-01-02")
later = pd.Timestamp("2024-01-03")

def run(mode):
    events = []
    class Position:
        skip = False
        def skip_update(self): return self.skip
        def fill_stock_value(self, time, freq):
            events.append(["fill", time.isoformat(), freq])
            if mode == "fill_failure": raise ValueError("fill")
    account = Account.__new__(Account)
    account.current_position, account.accum_info = Position(), object()
    account._port_metr_enabled, account.freq, account.benchmark_config = True, "day", None
    account.hist_positions = {start: {"amount": 1.0}}
    account.portfolio_metrics = PortfolioMetrics("day", None)
    account.portfolio_metrics.accounts[start] = 100.0
    account.indicator = Indicator()
    account.indicator.trade_indicator["value"] = 1.0
    account.indicator.record(start)
    executor = SimpleNamespace(reset=lambda **_: None, get_level_infra=lambda: None,
        finished=lambda: True, trade_calendar=SimpleNamespace(get_trade_len=lambda: 0),
        get_all_executors=lambda: [SimpleNamespace(time_per_step="day", trade_account=account)])
    strategy = SimpleNamespace(reset=lambda **_: None, post_upper_level_exe_step=lambda: None)
    published = {}
    list(ns["collect_data_loop"](start, later, strategy, executor, published))
    portfolio_table, positions = published["portfolio_dict"]["1day"]
    indicator_table, indicator = published["indicator_dict"]["1day"]
    old_metrics = account.portfolio_metrics
    old_position, old_accum = account.current_position, account.accum_info
    assert positions is account.hist_positions and indicator is account.indicator

    # Same-object changes remain visible through retained references, never through old tables.
    account.hist_positions[later] = {"amount": 2.0}
    account.hist_positions[start]["amount"] = 3.0
    account.indicator.trade_indicator["value"] = 2.0
    account.indicator.record(later)
    account.portfolio_metrics.accounts[start] = 200.0
    assert positions[start]["amount"] == 3.0 and len(positions) == 2
    assert len(indicator.trade_indicator_his) == 2
    assert indicator.trade_indicator_his[start]["value"] == 2.0  # record stores row identity
    assert portfolio_table.loc[start, "account"] == 100.0
    assert indicator_table.loc[start, "value"] == 1.0 and len(indicator_table) == 1

    def metrics_factory(freq, benchmark_config):
        events.append(["metrics", freq])
        if mode == "portfolio_failure": raise ValueError("metrics")
        return PortfolioMetrics(freq, None)  # benchmark acquisition is outside this contract
    def indicator_factory():
        events.append(["indicator"])
        if mode == "indicator_failure": raise ValueError("indicator")
        return Indicator()
    ns["PortfolioMetrics"], ns["Indicator"] = metrics_factory, indicator_factory
    account.current_position.skip = mode == "skip"
    error = None
    try:
        account.reset(freq="2min", benchmark_config={"start_time": start},
                      port_metr_enabled=mode != "disabled")
    except ValueError as caught:
        error = str(caught)
    finally:
        ns["PortfolioMetrics"], ns["Indicator"] = PortfolioMetrics, Indicator
    result = dict(mode=mode, events=events, error=error, frequency=account.freq,
        same_metrics=account.portfolio_metrics is old_metrics,
        same_positions=account.hist_positions is positions,
        same_indicator=account.indicator is indicator,
        same_position=account.current_position is old_position,
        same_accumulated=account.accum_info is old_accum,
        old_positions=len(positions), old_indicator_rows=len(indicator.trade_indicator_his),
        portfolio_snapshot=float(portfolio_table.loc[start, "account"]),
        indicator_snapshot=float(indicator_table.loc[start, "value"]))
    # Dropping the account/executor never invalidates a published report's retained objects.
    del account, executor
    assert len(positions) == 2 and len(indicator.trade_indicator_his) == 2
    return result

print(json.dumps([run(mode) for mode in ["enabled", "disabled", "skip", "portfolio_failure",
                                       "fill_failure", "indicator_failure"]]))
