"""Execute upstream generator close while retaining references to active delegates."""
import contextlib
import io
import json
from pathlib import Path
import runpy

with contextlib.redirect_stdout(io.StringIO()):
    source = runpy.run_path(str(Path(__file__).with_name("recursive_nested_protocol.py")))

cases = []
for mode in ("outer_track", "strategy", "inner"):
    for fail in (False, True):
        events = []

        class Proxy(source["Proxy"]):
            def generate_trade_decision(self, previous):
                def active():
                    try:
                        yield self
                        return source["Decision"]("atomic")
                    finally:
                        events.append("strategy_close")
                        if fail and mode == "strategy":
                            raise RuntimeError("strategy_close")
                self.active = active()
                return self.active

            def post_exe_step(self, result):
                events.append("post")

            def post_upper_level_exe_step(self):
                events.append("finalize")

        class Atomic(source["Atomic"]):
            def _collect_data(self, trade_decision, level=0):
                def active():
                    try:
                        yield "inner-prompt"
                        return [], {"trade_info": []}
                    finally:
                        events.append("inner_close")
                        if fail:
                            raise RuntimeError("inner_close")
                self.active = active()
                return self.active

        proxy, atomic = Proxy(), Atomic()
        top = source["Recursive"]("top", atomic, proxy)
        generator = top.collect_data(source["Decision"]("outer"))
        next(generator)
        if mode != "outer_track":
            assert next(generator) is proxy
        if mode == "inner":
            assert isinstance(generator.send(1), source["Decision"])
            assert next(generator) == "inner-prompt"
        events.clear()
        error = None
        try:
            generator.close()
        except RuntimeError as failure:
            error = str(failure)
        generator.close()
        assert generator.gi_frame is None
        assert top.trade_calendar.index == atomic.trade_calendar.index == 0
        assert "post" not in events and "finalize" not in events
        if mode != "outer_track":
            assert proxy.active.gi_frame is None
        if mode == "inner":
            assert atomic.active.gi_frame is None
        cases.append({"mode": mode, "fail": fail, "events": events, "error": error})
print(json.dumps(cases))
