"""Actual MultiIndex explicit-level/code validation and row materialization."""
import ast
import hashlib
import itertools
import json
import sys
from pathlib import Path

fixture = Path(__file__).with_name("dataframe_index_families.py")
tree = ast.parse(fixture.read_bytes())
stop = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "pairs" for t in node.targets))
exec(compile(ast.Module(body=tree.body[:stop], type_ignores=[]), str(fixture), "exec"))

levels = dict(
    integer=pd.Index([2, 1, 99], dtype="int64"),
    unsigned=pd.Index([0, 2**64-1, 2**63], dtype="uint64"),
    boolean=pd.Index([False, True], dtype="bool"),
    floating=pd.Index([1.5, np.nan, -0.], dtype="float64"),
    float32=pd.Index(np.array([1.25, np.nan, -0.], dtype="float32")),
    object=pd.Index([None, "a", "b"], dtype=object),
    mixed_missing=pd.Index([None, np.nan, pd.NA, pd.NaT], dtype=object),
    tuple=pd.Index([("b", 2), ("a", 1), (None, 0)], dtype=object, tupleize_cols=False),
    duplicates=pd.Index([True, 1, 1.], dtype=object),
    empty=pd.Index([], dtype=object),
    int_empty=pd.Index([], dtype="int64"),
)
for unit in ["s", "ms", "us", "ns"]:
    levels[f"datetime_{unit}"] = pd.DatetimeIndex(["2024-01-01", None, "2024-01-02"]).as_unit(unit)
    levels[f"utc_{unit}"] = pd.DatetimeIndex(["2024-01-01", None, "2024-01-02"], tz="UTC").as_unit(unit)
    levels[f"duration_{unit}"] = pd.TimedeltaIndex(["1h", None, "2h"]).as_unit(unit)


def run(selected, codes, names, sortorder):
    before = [index_snapshot(v) for v in selected]
    key = dict(levels=[index_snapshot(v) for v in selected], codes=codes,
               names=[scalar(v) for v in names], sortorder=sortorder)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            result = pd.MultiIndex(levels=selected, codes=codes, names=names, sortorder=sortorder)
        except Exception as error:
            outcome = dict(error=type(error).__name__, message=str(error))
        else:
            outcome = dict(output=index_snapshot(result))
    assert before == [index_snapshot(v) for v in selected]
    return dict(**key, **outcome, warnings=[[type(w.message).__name__, str(w.message)] for w in captured])


patterns = [[], [-1], [0], [1], [0, 1, -1], [1, 0, -1], [-1, 0, 1], [0, 0, 1], [-2], [99]]
cases = []
for (name, level), codes, order in itertools.product(levels.items(), patterns, [None, -1, 0, 1, 2]):
    cases.append(dict(name=name, **run([level], [codes], [("level", 7)], order)))
for left, right, order in itertools.product(patterns[:8], patterns[:8], [None, -1, 0, 1, 2, 3]):
    cases.append(dict(name="two_levels", **run([levels["object"], levels["integer"]], [left, right], [None, 7], order)))
for selected, codes, names in [([], [], []), ([levels["integer"]], [], [None]),
                                ([levels["integer"], levels["object"]], [[0]], [None, None]),
                                ([levels["integer"]], [[0]], []),
                                ([levels["integer"]], [[0]], [None, None])]:
    cases.append(dict(name="shape", **run(selected, codes, names, None)))

contract = dict(cases=cases)
digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__, numpy=np.__version__, digest=digest, **contract)))
