"""Actual resam_calendar with traced raw region and per-timestamp shift lookups."""
import contextlib
import io
import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd

with contextlib.redirect_stdout(io.StringIO()):
    setup = runpy.run_path(str(Path(__file__).with_name("file_calendar_backend_contract.py")))
ns = setup["ns"]
cases = []
for captured in ("cn", "us", "tw", "unknown", None):
    for global_region in ("cn", "unknown", None, "missing"):
        for source, requested, empty in (
            ("1min", "5min", False), ("1min", "0min", False),
            ("0min", "0min", False),
            ("day", "5min", False), ("10min", "5min", False),
            ("1min", "2day", False), ("day", "week", False),
            ("day", "month", False), ("day", "0min", True),
        ):
            for mode in ("fixed", "changing", "fail_first", "fail_second"):
                events = []

                class Config:
                    def __getitem__(self, key):
                        assert key == "region"
                        events.append("region")
                        if global_region == "missing":
                            raise KeyError("region")
                        return global_region

                    @property
                    def min_data_shift(self):
                        events.append("shift")
                        index = events.count("shift")
                        if mode == "fail_first" or mode == "fail_second" and index == 2:
                            raise KeyError("min_data_shift")
                        return index - 1 if mode == "changing" else 0

                ns["C"] = Config()
                raw = [] if empty else ["2024-01-02T09:37:00", "2024-01-02T09:36:00", "2024-01-02T09:37:00"]
                try:
                    output = ns["resam_calendar"](np.array([pd.Timestamp(x) for x in raw], dtype=object), source, requested, captured)
                    result = dict(values=[pd.Timestamp(x).isoformat() for x in output])
                except Exception as error:
                    result = dict(error="Value" if isinstance(error, ValueError) else "Other")
                cases.append(dict(captured=captured, global_region=global_region, source=source,
                                  requested=requested, raw=raw, mode=mode, events=events, result=result))
assert len(cases) == 720
print(json.dumps(cases))
