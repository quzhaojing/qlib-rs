"""Unchanged infrastructure lookup and replacement semantics, without a copied frequency."""
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from abc import abstractmethod
import warnings

path = Path(sys.argv[1]) / "backtest/utils.py"
tree = ast.parse(path.read_text(encoding="utf-8"))
names = {"BaseInfrastructure", "CommonInfrastructure", "LevelInfrastructure"}
nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
exec(compile(ast.fix_missing_locations(module), str(path), "exec"))

level = LevelInfrastructure()
rows = []
def read():
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always")
        try:
            value = level.get("common_infra").get("trade_exchange").freq
            result = {"frequency": value}
        except AttributeError:
            result = {"missing": str(messages[0].message)}
        rows.append(result)

read()
common = CommonInfrastructure()
level.reset_infra(common_infra=common)
read()
old = SimpleNamespace(freq="1min")
common.reset_infra(trade_exchange=old)
read()
old.freq = "bad-frequency"
read()
common.reset_infra(trade_exchange=SimpleNamespace(freq="2min"))
read()
old.freq = "stale"
read()
level.reset_infra(common_infra=CommonInfrastructure(trade_exchange=SimpleNamespace(freq="day")))
read()
common.trade_exchange.freq = "stale-common"
read()
print(json.dumps(rows))
