"""Actual MultiIndex append's NumPy boxing, including Python temporal scalars.

Preserve builtin datetime/timedelta identity explicitly, including UTC extrema.
"""
import ast
import datetime
from pathlib import Path

boxing_fixture = Path(__file__).with_name("dataframe_index_families.py")
boxing_tree = ast.parse(boxing_fixture.read_bytes())
boxing_stop = next(i for i, node in enumerate(boxing_tree.body) if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "pairs" for t in node.targets))
exec(compile(ast.Module(body=boxing_tree.body[:boxing_stop], type_ignores=[]), str(boxing_fixture), "exec"))
boxing_scalar = scalar


def scalar(value):
    if type(value) is datetime.datetime:
        return ["datetime", value.year, value.month, value.day, value.hour, value.minute,
                value.second, value.microsecond, value.fold, str(value.tzinfo)]
    if type(value) is datetime.timedelta:
        return ["timedelta", value.days, value.seconds, value.microseconds]
    return boxing_scalar(value)


left_indexes = [pd.MultiIndex.from_tuples([("a", 1)]),
                pd.MultiIndex(levels=[["unused"], [99]], codes=[[], []])]
patterns = [[], [0], [None], [1, None, -1], [-(2**63)+1, 2**63-1], [1, 2000, 1]]
cases = []
for unit in ["s", "ms", "us", "ns"]:
    for kind in ["datetime", "utc", "duration"]:
        for pattern in patterns:
            ticks = np.asarray([-(2**63) if v is None else v for v in pattern], dtype="int64")
            dtype = f"{'timedelta64' if kind == 'duration' else 'datetime64'}[{unit}]"
            index = (pd.TimedeltaIndex(ticks.view(dtype)) if kind == "duration"
                     else pd.DatetimeIndex(ticks.view(dtype)))
            if kind == "utc":
                index = index.tz_localize("UTC")
            index.name = "datetime"
            for left_index in left_indexes:
                for lc in [False, True]:
                    frame = pd.DataFrame(dict(x=list(range(len(left_index)))) if lc else {}, index=left_index)
                    for rc in [False, True]:
                        _, outcome = execute(frame, index, rc)
                        cases.append(dict(unit=unit, kind=kind, input=index_snapshot(index),
                                          left=frame_snapshot(frame), right_columns=rc, **outcome))
contract = dict(cases=cases)
digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__, numpy=np.__version__, digest=digest, **contract)))
