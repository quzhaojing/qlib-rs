"""Actual object Index uniqueness/factorization used by MultiIndex level construction."""
import ast
import hashlib
import itertools
import json
import sys
from pathlib import Path

# Reuse the source-pinned temporal boundary inventory and snapshot helpers.
fixture = Path(__file__).with_name("dataframe_temporal_objects.py")
tree = ast.parse(fixture.read_bytes())
stop = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "before" for t in node.targets))
exec(compile(ast.Module(body=tree.body[:stop], type_ignores=[]), str(fixture), "exec"))
values += [0, 1, 1., 2**53, 2**53+1, float(2**53), float(2**53+1),
           float(2**64-1), float(2**63-1), 5e-324, -5e-324, float('nan'), "", "1"]
values += [(), (None,), (pd.NA,), (pd.NaT,), (float('nan'),), (float('nan'),),
           (False,), (0,), (-0.,), (True,), (1,), (1.,), ((1, None),), ((True, None),),
           (1, 2), (1,), (pd.Timestamp("2024-01-02", tz="UTC"),),
           (pd.Timestamp("2024-01-02", tz="UTC").tz_convert("Asia/Shanghai"),)]


def describe(items):
    index = pd.Index(items, dtype=object, tupleize_cols=False, name="level")
    before = [scalar(v) for v in index]
    result = dict(unique=bool(index.is_unique))
    for sentinel in [False, True]:
        codes, uniques = pd.factorize(index, sort=False, use_na_sentinel=sentinel)
        result[str(sentinel)] = dict(codes=codes.tolist(), uniques=[scalar(v) for v in uniques])
    # This is the actual consumer of level uniqueness and null code normalization.
    try:
        multi = pd.MultiIndex(levels=[index], codes=[list(range(len(index))) + [-1]], names=["level"])
    except Exception as error:
        result["multi"] = dict(error=type(error).__name__, message=str(error))
    else:
        frame = pd.DataFrame(dict(x=list(range(len(multi)))), index=multi)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            appended = dataframe_append(frame, dict(datetime=pd.Index([], dtype=object)))
        result["multi"] = dict(codes=multi.codes[0].tolist(), values=[scalar(v) for v in multi],
                               appended=snapshot(appended), warnings=[[type(w.message).__name__, str(w.message)] for w in captured])
    assert before == [scalar(v) for v in index]
    return result


pairs = [dict(left=i, right=j, **describe([left, right]))
         for (i, left), (j, right) in itertools.product(enumerate(values), repeat=2)]
histories = [[], values, list(reversed(values)), values + values,
             [True, 1, 1., False, 0, -0., None, pd.NA, pd.NaT, np.nan],
             [(1, None), (True, None), (1., None), (1, np.nan)]]
contract = dict(values=[scalar(v) for v in values], pairs=pairs,
                histories=[dict(values=[scalar(v) for v in row], **describe(row)) for row in histories])
digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__, numpy=np.__version__, digest=digest, **contract)))
