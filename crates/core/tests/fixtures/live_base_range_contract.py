"""Actual Indicator baseline method: range is reread for every reached quote."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

path = Path(r"D:\code\github\qlib\qlib\backtest\report.py")
raw = path.read_bytes()
assert hashlib.sha256(raw).hexdigest() == "78105d68730fe19925c0d17e53b019433373b9231ff0ff1197e64f8234475816"
cls = next(n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef) and n.name == "Indicator")
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_get_base_vol_pri")
module = ast.Module(body=ast.parse("from __future__ import annotations").body + [method], type_ignores=[])
exec(compile(module, str(path), "exec"))

calls = []
decision = SimpleNamespace(trade_range=None)
class Rule:
    def clip_time_range(self, start_time, end_time):
        return "2024-01-02 10:00:00", end_time

class Exchange:
    def get_deal_price(self, inst, start, end, **kwargs):
        calls.append([inst, start, end])
        decision.trade_range = Rule()
        return None

for stock in ("A", "B"):
    assert _get_base_vol_pri(None, stock, "2024-01-02 09:00:00", "2024-01-02 11:00:00",
                             1, decision, Exchange()) == (None, None)
print(json.dumps(calls))
