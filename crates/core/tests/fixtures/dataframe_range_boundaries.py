"""Actual-source RangeIndex metadata and materializable wide Python bounds."""
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

bounds = [(0, 0, 1), (9, 9, 3), (0, 4, 2), (5, -2, -2), (3, 0, 1), (0, 3, -1),
          (2**63-1, 2**63, 1), (-2**63, -2**63-1, -1),
          (0, 2**80, 2**80), (0, -2**80, -2**80), (2**80, 2**80, 1),
          (-2**80, -2**80, -1), (0, 2, 0)]
cases, constructors = [], []
for start, end, step in bounds:
    key = [str(start), str(end), str(step)]
    try:
        index = pd.RangeIndex(start, end, step, name="history")
    except Exception as error:
        constructors.append(dict(bounds=key, error=type(error).__name__, message=str(error)))
        continue
    # The full range descriptor remains separate from its materialized i64 values.
    constructors.append(dict(bounds=key, values=[str(v) for v in index.to_numpy()]))
    for lc, rc, right in itertools.product([False, True], [False, True], ["int_empty", "int_values"]):
        frame = pd.DataFrame(dict(x=list(range(len(index)))) if lc else {}, index=index)
        _, outcome = execute(frame, samples[right], rc)
        if "range" in outcome["output"]["index"]:
            outcome["output"]["index"]["range"] = [str(v) for v in outcome["output"]["index"]["range"]]
        cases.append(dict(bounds=key, left_columns=lc, right_columns=rc, right=right, **outcome))

contract = dict(constructors=constructors, cases=cases)
digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__, numpy=np.__version__, digest=digest, **contract)))
