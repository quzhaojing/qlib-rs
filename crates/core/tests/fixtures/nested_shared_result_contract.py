"""Actual NestedExecutor list/tuple/order aliases; source evidence, not Rust parity."""
import ast
import hashlib
import json
from pathlib import Path
from types import GeneratorType, SimpleNamespace


source = Path(r"D:\code\github\qlib\qlib\backtest\executor.py")
raw = source.read_bytes()
assert hashlib.sha256(raw).hexdigest() == "76ab94ce77691487da6cd41bcfe3fe14b5149e917835bcaaba0ed174afd0fa88"
node = next(n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef) and n.name == "NestedExecutor")
methods = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in ("_collect_data", "post_inner_exe_step")]
assert len(methods) == 2
get_start_end_idx = lambda calendar, decision: (0, 1)
exec(compile(ast.Module(body=ast.parse("from __future__ import annotations").body + methods, type_ignores=[]), str(source), "exec"))


def run(suspended=False, fail=None):
    events = []
    order = SimpleNamespace(amount=1)
    retained = []
    indicators = []
    seen_previous = []

    class Calendar:
        index = 0

        def get_step_time(self):
            events.append("time")
            return self.index, self.index + 1

    calendar = Calendar()

    class Decision:
        def empty(self):
            return False

        def mod_inner_decision(self, decision):
            events.append("propagate")

    class Strategy:
        def generate_trade_decision(self, previous):
            seen_previous.append(previous)
            events.append("generate")
            if previous is not None:
                assert previous is retained[0]
                # This is the real prior list, but extend has already copied its element references.
                previous.clear()
                previous.append((order, 99, 0, 99))
                order.amount = 7
            if not suspended:
                return Decision()

            def resume():
                action = yield "action"
                assert action == 42
                return Decision()

            return resume()

        def post_exe_step(self, rows):
            events.append("post")
            assert rows is retained[-1]
            if len(retained) == 1:
                rows.append(rows[0])
            order.amount += 1
            if fail == "post":
                raise RuntimeError("post")

        def post_upper_level_exe_step(self):
            events.append("upper")

    class Inner:
        trade_calendar = calendar

        def finished(self):
            return calendar.index == 2

        def collect_data(self, trade_decision, level):
            assert level == 4
            events.append("collect")
            calendar.index += 1
            rows = [(order, calendar.index * 10, 0, 5)]
            retained.append(rows)
            yield from ()
            return rows

        def get_order_indicator(self, raw):
            assert raw is True
            events.append("indicator")
            if fail == "indicator":
                raise RuntimeError("indicator")
            result = {}
            indicators.append(result)
            return result

    inner = Inner()
    inner.trade_account = SimpleNamespace(get_trade_indicator=lambda: inner)
    executor = SimpleNamespace(
        inner_executor=inner, inner_strategy=Strategy(),
        _skip_empty_decision=False, _align_range_limit=False,
        _init_sub_trading=lambda decision: events.append("init"),
        _update_trade_decision=lambda decision: decision,
    )
    executor.post_inner_exe_step = lambda rows: post_inner_exe_step(executor, rows)
    generator = _collect_data(executor, Decision(), level=3)
    pauses = 0
    try:
        request = next(generator)
        while True:
            assert request == "action"
            pauses += 1
            request = generator.send(42)
    except StopIteration as complete:
        flat, metadata = complete.value
        assert seen_previous[0] is None
        assert flat is not retained[0] and flat is not retained[1]
        assert flat[0] is flat[1]
        assert flat[2] is retained[1][0]
        assert all(row[0] is order for row in flat)
        assert metadata["inner_order_indicators"] == indicators
        assert all(a is b for a, b in zip(metadata["inner_order_indicators"], indicators))
        assert len(metadata["decision_list"]) == 2
        outcome = {"flat_values": [row[1] for row in flat], "flat_amounts": [row[0].amount for row in flat]}
        assert outcome == {"flat_values": [10, 10, 20], "flat_amounts": [8, 8, 8]}
    except RuntimeError as error:
        assert str(error) == fail
        assert len(retained) == 1 and len(retained[0]) == 2
        assert order.amount == 2 and "upper" not in events
        assert ("indicator" in events) == (fail == "indicator")
        outcome = {"error": str(error)}
    outcome.update(events=events, pauses=pauses, retained_lengths=[len(rows) for rows in retained])
    return outcome


if __name__ == "__main__":
    result = {"plain": run(), "suspended": run(True), "post_failure": run(fail="post"), "indicator_failure": run(fail="indicator")}
    assert result["plain"]["retained_lengths"] == [1, 1]
    assert result["suspended"]["pauses"] == 2
    print(json.dumps(result))
