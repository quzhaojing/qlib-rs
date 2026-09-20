"""Actual returned temporal/tuple category frames through two string appends."""
import ast
from pathlib import Path

fixture = Path(__file__).with_name("dataframe_string_temporal_categories.py")
tree = ast.parse(fixture.read_bytes())
stop = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "pairs" for t in node.targets))
exec(compile(ast.Module(body=tree.body[:stop], type_ignores=[]), str(fixture), "exec"))
firsts = ["python_NA_missing", "python_nan_missing", "pyarrow_NA_missing", "pyarrow_nan_missing"]
seconds = ["python_NA_unicode", "pyarrow_NA_empty", "pyarrow_nan_all_missing"]
cases = []
for (name, index), first_name, second_name, columns in itertools.product(
        categories.items(), firsts, seconds, [False, True]):
    frame = pd.DataFrame(dict(x=list(range(len(index)))) if columns else {}, index=index)
    current, first = execute(frame, samples[first_name], columns)
    second = None
    if current is not None:
        _, second = execute(current, samples[second_name], True)
    cases.append(dict(left=name, first_name=first_name, second_name=second_name,
                      columns=columns, first=first, second=second))
contract = dict(inputs={k:index_snapshot(v) for k,v in samples.items()}, cases=cases)
digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__, numpy=np.__version__, digest=digest, **contract)))
