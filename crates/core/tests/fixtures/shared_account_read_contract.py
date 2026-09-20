"""Source account/indicator methods prove live order read timing across callbacks."""
import ast
from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json
from pathlib import Path

root = Path(r"D:\code\github\qlib\qlib\backtest")


def tree(name, expected):
    raw = (root / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected
    return ast.parse(raw)


def methods(module, owner, names):
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == owner)
    return [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]


decision = tree("decision.py", "a6866d15bc8f3ad1c75bfc3856ccde5245d43f0a2de07b1ad565d6e7e20d8251")
account = tree("account.py", "701886f32a36becd9c5d78b28e4b5a751ede01becc12146a9ccf99400235538b")
report = tree("report.py", "78105d68730fe19925c0d17e53b019433373b9231ff0ff1197e64f8234475816")
body = ast.parse("from __future__ import annotations").body
body += [node for node in decision.body if isinstance(node, ast.ClassDef) and node.name in {"Order", "OrderDir"}]
body += methods(account, "Account", {"update_bar_end", "update_indicator"})
body += methods(report, "Indicator", {"_update_order_trade_info"})
exec(compile(ast.Module(body=body, type_ignores=[]), "shared-account-source", "exec"))

order = Order("B", 2.0, Order.BUY, None, None)
order.deal_amount = 1.0
events, columns = [], {}


class Store:
    def assign(self, name, values):
        columns[name] = dict(values)


class Indicator:
    order_indicator = Store()
    _update_order_trade_info = _update_order_trade_info

    def reset(self):
        events.append("reset")
        assert order.deal_amount == 2.0
        order.deal_amount = 3.0

    def update_order_indicators(self, rows):
        events.append("read")
        self._update_order_trade_info(rows)

    def cal_trade_indicators(self, *args):
        events.append("calculate")

    def record(self, *args):
        events.append("record")


class Account:
    update_bar_end = update_bar_end
    update_indicator = update_indicator
    indicator = Indicator()
    freq = "day"

    def update_current_position(self, *args):
        events.append("market")
        order.deal_amount = 2.0

    def is_port_metr_enabled(self):
        return False


Account().update_bar_end(1, 2, object(), True, object(), trade_info=[(order, 5.0, 0.0, 5.0)])
print(json.dumps({"events": events, "deal_amount": columns["deal_amount"]["B"], "amount": columns["amount"]["B"]}))
