"""Actual-source index append rules, including initialized and chained histories."""
import ast
import hashlib
import itertools
import json
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

# Reuse the exact 83-array temporal/numeric/object boundary inventory, not its
# append cases. Its prelude also loads the hash-pinned actual Qlib function and
# lossless scalar snapshots. Stop structurally before its execution helper.
fixture = Path(__file__).with_name("dataframe_temporal_append.py")
prelude = []
for node in ast.parse(fixture.read_bytes()).body:
    if isinstance(node, ast.FunctionDef) and node.name == "execute":
        break
    prelude.append(node)
else:
    raise AssertionError("missing fixture prelude boundary")
exec(compile(ast.Module(body=prelude, type_ignores=[]), str(fixture), "exec"))
assert len(samples) == 83


def index_snapshot(index):
    return dict(kind=type(index).__name__, dtype=str(index.dtype),
                name=scalar(index.name), values=[scalar(v) for v in index])


def execute(left, right):
    before = [snapshot(left), snapshot(right)]
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            result = dataframe_append(left, right)
            outcome = dict(output=index_snapshot(result.index))
        except Exception as error:
            result = None
            outcome = dict(error=type(error).__name__, message=str(error))
    if result is not None:
        assert result.columns.tolist() == ["x"]
        assert result.x.tolist() == left.x.tolist() + right.x.tolist()
    assert before == [snapshot(left), snapshot(right)]
    return result, dict(warnings=[[type(w.message).__name__, str(w.message)] for w in captured], **outcome)


def right_frame(values):
    return pd.DataFrame(dict(datetime=values, x=pd.Series(range(len(values)), dtype="int64")))


indexes, inputs = {}, {}
for name, value in samples.items():
    try:
        indexes[name] = pd.Index(value, dtype=value.dtype)
        inputs[name] = index_snapshot(indexes[name])
    except Exception as error:
        inputs[name] = dict(error=type(error).__name__, message=str(error))

pairs = []
for (ln, left), (rn, right), name in itertools.product(indexes.items(), samples.items(), [None, "datetime", "history"]):
    frame = pd.DataFrame(dict(x=pd.Series(range(len(left)), dtype="int64").to_numpy()),
                         index=left.rename(name))
    _, outcome = execute(frame, right_frame(right))
    pairs.append(dict(left=ln, right=rn, name=name, **outcome))

# A real initialized history has an empty object index, not an empty datetime
# index. Include second appends to detect lost dtype/identity metadata.
chains = []
for (first_name, first), second_name in itertools.product(samples.items(),
        ["object_none", "object_utc", "object_empty", "timestamp_s_None_finite",
         "timestamp_ns_UTC_finite", "duration_us_finite", "int64_finite"]):
    initial = pd.DataFrame(columns=["datetime", "x"]).set_index("datetime")
    current, first_outcome = execute(initial, right_frame(first))
    if current is not None:
        _, second_outcome = execute(current, right_frame(samples[second_name]))
    else:
        second_outcome = None
    chains.append(dict(first=first_name, second=second_name, initial=index_snapshot(initial.index),
                       first_output=first_outcome, second_output=second_outcome))

contract = dict(inputs=inputs, pairs=pairs, chains=chains)
digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__, numpy=np.__version__, digest=digest, **contract)))
