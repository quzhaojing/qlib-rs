"""Actual ordinary object-index append, including tuples and scalar inference."""
import ast
import hashlib
import itertools
import json
import sys
from pathlib import Path

fixture = Path(__file__).with_name("dataframe_multi_index_contract.py")
tree = ast.parse(fixture.read_bytes())
stop = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "descriptors" for t in node.targets))
exec(compile(ast.Module(body=tree.body[:stop], type_ignores=[]), str(fixture), "exec"))
samples = {name:index for name,index in samples.items() if not isinstance(index, pd.MultiIndex)}
samples.update(
    bool_values=pd.Index([True, False], name="datetime"),
    float_values=pd.Index([1.5, np.nan], name="datetime"),
    uint_values=pd.Index([0, 2**64-1], dtype="uint64", name="datetime"),
    int_empty=pd.Index([], dtype="int64", name="datetime"),
    bool_empty=pd.Index([], dtype="bool", name="datetime"),
    datetime_values=pd.DatetimeIndex(["2024-01-01", None], name="datetime"),
    datetime_empty=pd.DatetimeIndex([], name="datetime"),
    utc_values=pd.DatetimeIndex(["2024-01-01", None], tz="UTC", name="datetime"),
    timedelta_values=pd.TimedeltaIndex(["1h", None], name="datetime"),
    object_int=object_index([1, 2]),
    object_bool=object_index([True, False]),
    object_time=object_index([pd.Timestamp("2024-01-01"), None]),
    object_utc=object_index([pd.Timestamp("2024-01-01", tz="UTC"), pd.NaT]),
    object_delta=object_index([pd.Timedelta("1h"), np.nan]),
    object_missing=object_index([None, pd.NA, pd.NaT, np.nan]),
)
pairs = []
for (ln, left), (rn, right), lc, rc in itertools.product(samples.items(), samples.items(), [False, True], [False, True]):
    frame = pd.DataFrame(dict(x=list(range(len(left)))) if lc else {}, index=left)
    _, outcome = execute(frame, right, rc)
    pairs.append(dict(left=ln, right=rn, left_columns=lc, right_columns=rc, **outcome))
chains = []
for (name, index), next_name in itertools.product(samples.items(), ["tuple_two", "object_empty", "scalar_int", "datetime_values"]):
    initial = pd.DataFrame(columns=["x"], index=index[:0])
    current, first = execute(initial, index, True)
    second = None
    if current is not None:
        _, second = execute(current, samples[next_name], True)
    chains.append(dict(first=name, second=next_name, initial=index_snapshot(initial.index), first_output=first, second_output=second))
contract = dict(inputs={name:index_snapshot(value) for name,value in samples.items()}, pairs=pairs, chains=chains)
digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__, numpy=np.__version__, digest=digest, **contract)))
