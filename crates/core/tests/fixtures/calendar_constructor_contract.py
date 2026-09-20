"""Assert actual Qlib constructor ordering, aliasing and deferred validation.

Run with the upstream qlib package directory as argv[1]. This is source
characterization, not evidence that the Rust constructor is implemented.
"""
import contextlib
import io
import json
from pathlib import Path
import runpy

with contextlib.redirect_stdout(io.StringIO()):
    setup = runpy.run_path(str(Path(__file__).with_name("file_calendar_storage_contract.py")))

ns, manager = setup["ns"], setup["manager"]
results = []

for mode in ("global", "scalar", "mapping", "nfs", "empty", "invalid", "partial"):
    for region in ("cn", "unknown", None, "missing"):
        for frequency in ("day", "bad-frequency"):
            events = []

            class TracedManager:
                @staticmethod
                def format_provider_uri(value):
                    events.append("normalize")
                    return manager.format_provider_uri(value)

            class TracedConfig(dict):
                DataPathManager = TracedManager

                def __getitem__(self, key):
                    events.append("get:" + key)
                    return super().__getitem__(key)

            config = TracedConfig() if region == "missing" else TracedConfig(region=region)
            ns["C"] = config
            provider = {
                "global": None, "scalar": ".", "mapping": {"day": "."},
                "nfs": {"day": "host:/data"}, "empty": {}, "invalid": 123,
                "partial": {"day": ".", "bad": None, "week": "untouched"},
            }[mode]
            storage = ns["FileCalendarStorage"].__new__(ns["FileCalendarStorage"])
            error = None
            kwargs = dict(enable_read_cache=False, region="us", custom=[1, 2])
            try:
                storage.__init__(frequency, True, provider_uri=provider, **kwargs)
            except Exception as failure:
                error = type(failure).__name__

            expected_error = (
                "TypeError" if mode == "invalid" else
                "AttributeError" if mode == "partial" else
                "KeyError" if region == "missing" else None
            )
            assert error == expected_error, (mode, region, frequency, error)
            expected_events = [] if mode == "global" else ["normalize"]
            if mode not in ("invalid", "partial"):
                expected_events.append("get:region")
            assert events == expected_events, (mode, region, events)
            assert storage.freq == frequency and storage.future is True
            assert storage.kwargs == kwargs
            assert storage.kwargs["custom"] is kwargs["custom"]
            if mode in ("mapping", "partial"):
                assert provider["day"] == str(Path(".").resolve())
            if mode == "partial":
                assert provider["bad"] is None and provider["week"] == "untouched"
            if mode in ("invalid", "partial"):
                assert "_provider_uri" not in vars(storage)
                assert "enable_read_cache" not in vars(storage)
            else:
                assert storage.enable_read_cache is True
                if isinstance(provider, dict):
                    assert storage._provider_uri is provider
                    provider["external"] = "host:/changed"
                    assert storage._provider_uri["external"] == "host:/changed"
                elif mode == "global":
                    assert storage._provider_uri is None
                else:
                    assert storage._provider_uri == {"__DEFAULT_FREQ": str(Path(".").resolve())}
                if region != "missing":
                    assert storage.region == region
                    config["region"] = "tw"
                    assert storage.region == region  # Constructor captures, not a live property.
            assert ("region" in vars(storage)) == (error is None)
            results.append(dict(mode=mode, region=region, frequency=frequency,
                                events=events, error=error, provider=provider,
                                attributes=sorted(vars(storage))))

assert len(results) == 56
print(json.dumps(results))
