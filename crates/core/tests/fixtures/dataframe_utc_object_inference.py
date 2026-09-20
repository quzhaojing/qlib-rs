"""Source object-inference boundaries, including Timestamp's surrogate year.

Keep actual Qlib outputs, not normalized or calculated expected dates.
"""
import ast
import random
from pathlib import Path

fixture = Path(__file__).with_name("dataframe_multi_numpy_temporal.py")
tree = ast.parse(fixture.read_bytes())
stop = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "left_indexes" for t in node.targets))
exec(compile(ast.Module(body=tree.body[:stop], type_ignores=[]), str(fixture), "exec"))

cases = []
rng = random.Random(61924)
for unit in ["s", "ms", "us", "ns"]:
    scale = {"s": 10**9, "ms": 10**6, "us": 1000, "ns": 1}[unit]
    ticks = [-(2**63)+1, 2**63-1, -1, 0, 1]
    ticks += [rng.randrange(-(2**63)+1, 2**63) for _ in range(40)]
    for year in [-10000, -400, -100, -1, 0, 1, 1600, 1677, 1678,
                 2262, 2263, 9999, 10000, 10100, 10400, 100000]:
        # Calculate in seconds first; exclude casts that would wrap the input.
        value = int(np.datetime64(f"{year:04d}-03-01T12:34:56", "s").view("i8")) * 10**9 // scale
        if -(2**63) < value < 2**63:
            ticks += [value-1, value, value+1]
    for tick in dict.fromkeys(ticks):
        for pattern in [[tick], [None, tick], [tick, 0], [tick, -(2**63)+1]]:
            raw = np.asarray([-(2**63) if v is None else v for v in pattern], dtype="int64")
            for zone in [None, "UTC"]:
                index = pd.DatetimeIndex(raw.view(f"datetime64[{unit}]"))
                if zone:
                    index = index.tz_localize(zone)
                index.name = "datetime"
                left_index = pd.MultiIndex(levels=[["unused"], [99]], codes=[[], []])
                frame = pd.DataFrame(dict(x=[]), index=left_index)
                _, outcome = execute(frame, index, True)
                cases.append(dict(unit=unit, timezone=zone, input=index_snapshot(index),
                                  left=frame_snapshot(frame), right_columns=True, **outcome))
contract = dict(cases=cases)
digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__, numpy=np.__version__, digest=digest, **contract)))
